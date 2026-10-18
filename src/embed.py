from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

if __package__:
    from .runtime_device import clear_cuda_cache, device_preference, is_cuda_resource_error, select_device
else:
    from runtime_device import clear_cuda_cache, device_preference, is_cuda_resource_error, select_device

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "config" / "settings.yaml"
CORPUS_REPORT = ROOT / "evaluation" / "corpus_preparation_report.json"
INDEX_DIR = ROOT / "artifacts" / "index"
PASSAGES = ROOT / "data" / "passages.jsonl"
CHUNK_REPORT = ROOT / "evaluation" / "chunking_report.json"


def resolve_config(args: argparse.Namespace) -> dict:
    settings = yaml.safe_load(SETTINGS.read_text(encoding="utf-8")) or {}
    section = settings.get("embedding") or {}
    if args.experiment:
        config = {"model_name": args.model, "revision": args.revision,
                  "document_prefix": args.document_prefix, "query_prefix": args.query_prefix}
        if not config["model_name"]:
            raise SystemExit("--experiment needs --model <huggingface model name>.")
    else:
        config = {"model_name": section.get("model_name"), "revision": section.get("revision"),
                  "document_prefix": section.get("document_prefix") or "",
                  "query_prefix": section.get("query_prefix") or ""}
        if not config["model_name"] or not config["revision"]:
            raise SystemExit(
                "The official index needs the project-approved model and a pinned revision.\n"
                "Set embedding.model_name and embedding.revision in config/settings.yaml once the coordinator "
                "announces them. Until then, use --experiment (see the top of this file).")
    config["include_section"] = bool(section.get("include_section", True))
    config["batch_size"] = int(section.get("batch_size", 32))
    config["device"] = device_preference(settings, "embedding")
    config["normalize"] = True
    return config


def load_model(config: dict):
    from sentence_transformers import SentenceTransformer
    kwargs = {"revision": config["revision"]} if config.get("revision") else {}
    if config.get("local_files_only"):
        kwargs["local_files_only"] = True
    config["requested_device"] = config.get("device") or "auto"
    config["device"] = select_device(config["requested_device"])
    try:
        return SentenceTransformer(config["model_name"], device=config["device"], **kwargs)
    except RuntimeError as error:
        if not config["device"].startswith("cuda") or not is_cuda_resource_error(error):
            raise
        reason = str(error)
    # Leave the exception handler first, releasing failed GPU tensors/tracebacks.
    print(f"GPU model loading failed ({reason}); retrying on CPU.", flush=True)
    config["device"] = "cpu"
    clear_cuda_cache()
    return SentenceTransformer(config["model_name"], device="cpu", **kwargs)


def passage_text(chunk: dict, config: dict) -> str:
    """The exact string that gets embedded for a chunk (retrieve.py must not change this)."""
    text = chunk["text"]
    if config["include_section"] and chunk.get("section"):
        text = f"{chunk['section']}\n{text}"
    return config["document_prefix"] + text


def encode_texts(model, texts: list[str], config: dict, *, show_progress: bool = False) -> np.ndarray:
    """Use the selected model device, with one CPU retry for CUDA resource failures."""
    kwargs = {
        "batch_size": config["batch_size"], "normalize_embeddings": True,
        "convert_to_numpy": True, "show_progress_bar": show_progress,
    }
    try:
        return model.encode(texts, device=config["device"], **kwargs).astype("float32")
    except RuntimeError as error:
        if not config["device"].startswith("cuda") or not is_cuda_resource_error(error):
            raise
        reason = str(error)
    print(f"GPU encoding failed ({reason}); retrying this book/query batch on CPU.", flush=True)
    model.to("cpu")
    config["device"] = "cpu"
    clear_cuda_cache()
    return model.encode(texts, device="cpu", **kwargs).astype("float32")


def embed_queries(model, questions: list[str], config: dict) -> np.ndarray:
    prefixed = [config["query_prefix"] + q for q in questions]
    return encode_texts(model, prefixed, config)


def atomic_bytes(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        writer(stream)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def embed_book(model, config: dict, source: Path, vector_dir: Path, force: bool) -> tuple[list[dict], np.ndarray, bool]:
    raw = source.read_bytes()
    chunks = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    book_id = chunks[0]["book_id"]
    key = {"chunks_sha256": hashlib.sha256(raw).hexdigest(), "model_name": config["model_name"],
           "revision": config["revision"], "document_prefix": config["document_prefix"],
           "include_section": config["include_section"]}
    vector_path, key_path = vector_dir / f"{book_id}.npy", vector_dir / f"{book_id}.json"
    if not force and vector_path.exists() and key_path.exists():
        if json.loads(key_path.read_text(encoding="utf-8")) == key:
            vectors = np.load(vector_path)
            if vectors.shape[0] == len(chunks):
                return chunks, vectors, True
    texts = [passage_text(chunk, config) for chunk in chunks]
    vectors = encode_texts(model, texts, config, show_progress=True)
    atomic_bytes(vector_path, lambda stream: np.save(stream, vectors))
    atomic_bytes(key_path, lambda stream: stream.write(json.dumps(key, indent=2).encode("utf-8")))
    return chunks, vectors, False


def truncation_estimate(model, config: dict, chunks: list[dict], sample: int = 400) -> dict:
    """Share of sampled passages longer than the model's maximum sequence length."""
    limit = getattr(model, "max_seq_length", None)
    try:
        picked = random.Random(0).sample(chunks, min(sample, len(chunks)))
        encoded = model.tokenizer([passage_text(c, config) for c in picked],
                                  add_special_tokens=True, truncation=False)["input_ids"]
        lengths = [len(ids) for ids in encoded]
        over = sum(length > limit for length in lengths) if limit else None
        return {"max_seq_length": limit, "sampled": len(lengths), "max_tokens_seen": max(lengths),
                "mean_tokens": round(sum(lengths) / len(lengths)),
                "share_truncated": None if over is None else round(over / len(lengths), 3)}
    except Exception as error:  # tokenizer API differs between models
        return {"max_seq_length": limit, "error": f"{type(error).__name__}: {error}"}


def library_versions() -> dict:
    import faiss
    versions = {"numpy": np.__version__, "faiss": faiss.__version__}
    for name in ("sentence_transformers", "torch", "transformers"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book-id", action="append", dest="book_ids", help="Re-embed only these books (others use the cache).")
    parser.add_argument("--rebuild", action="store_true", help="Ignore the vector cache and re-embed everything.")
    parser.add_argument("--experiment", metavar="LABEL", help="Build a separate, labelled experiment index.")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--query-prefix", default="")
    args = parser.parse_args()

    import faiss
    config = resolve_config(args)
    out_dir = INDEX_DIR / "experiments" / args.experiment if args.experiment else INDEX_DIR
    passages_path = out_dir / "passages.jsonl" if args.experiment else PASSAGES
    if __package__:
        from .prepare_corpus import load_prepared_sources
    else:
        from prepare_corpus import load_prepared_sources
    try:
        files, corpus_report = load_prepared_sources(ROOT, SETTINGS)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Embedding inputs are not ready: {error}") from error
    if not config["revision"]:
        print("Warning: no revision given. This index is not reproducible and is labelled experimental.")

    started = time.monotonic()
    print(f"Loading {config['model_name']} (revision {config['revision'] or 'unpinned'})", flush=True)
    model = load_model(config)
    force_books = set(args.book_ids or [])
    all_chunks: list[dict] = []
    all_vectors: list[np.ndarray] = []
    books: dict[str, int] = {}
    book_devices: dict[str, str] = {}
    for source in files:
        book_id = source.name.removesuffix(".chunks.jsonl")
        chunks, vectors, cached = embed_book(model, config, source, out_dir / "vectors",
                                             args.rebuild or book_id in force_books)
        print(f"{book_id}: {len(chunks)} chunks ({'cached' if cached else 'embedded'})", flush=True)
        all_chunks.extend(chunks)
        all_vectors.append(vectors)
        books[book_id] = len(chunks)
        book_devices[book_id] = "cached (original device not recorded)" if cached else config["device"]

    matrix = np.vstack(all_vectors).astype("float32")
    ids = [c["chunk_id"] for c in all_chunks]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate chunk_ids across books; fix chunking before indexing.")
    if matrix.shape[0] != len(all_chunks):
        raise SystemExit("Vector count does not match passage count.")

    index = faiss.IndexFlatIP(matrix.shape[1])  # exact search; vectors are normalized, so inner product = cosine
    index.add(matrix)

    # Sanity check: a passage must retrieve itself with a score of about 1.0.
    rows = random.Random(1).sample(range(len(all_chunks)), min(5, len(all_chunks)))
    scores, _ = index.search(matrix[rows], 1)
    if float(scores.min()) < 0.99:
        raise SystemExit(f"Self-retrieval check failed (min score {scores.min():.3f}); index not written.")

    def write_passages(stream):
        for chunk in all_chunks:
            stream.write((json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8"))

    atomic_bytes(passages_path, write_passages)
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / ".index.faiss.tmp"
    faiss.write_index(index, str(temporary))
    os.replace(temporary, out_dir / "index.faiss")

    chunking = {}
    if CHUNK_REPORT.exists():
        chunking = json.loads(CHUNK_REPORT.read_text(encoding="utf-8")).get("settings", {})
    index_config = {
        "built_at": datetime.now(ZoneInfo("Africa/Lusaka")).isoformat(),
        "experiment": args.experiment,
        "corpus_preparation": {
            "report": CORPUS_REPORT.relative_to(ROOT).as_posix(),
            "report_sha256": hashlib.sha256(CORPUS_REPORT.read_bytes()).hexdigest(),
            "unique_passages": corpus_report["unique_passages"],
            "source_chunks": corpus_report["retained_source_chunks"],
            "books_represented": corpus_report["books_represented"],
        },
        "embedding": {**{k: config[k] for k in ("model_name", "revision", "document_prefix", "query_prefix",
                                                "include_section", "normalize")},
                      "dimension": int(matrix.shape[1]),
                      "requested_device": config["requested_device"],
                      "final_device": config["device"],
                      "book_devices": book_devices,
                      "truncation": truncation_estimate(model, config, all_chunks)},
        "index": {"type": "IndexFlatIP", "device": "cpu",
                  "metric": "inner product on normalized vectors (cosine)",
                  "passages": len(all_chunks), "file": "index.faiss",
                  "passages_file": passages_path.relative_to(ROOT).as_posix(),
                  "sha256": file_sha256(out_dir / "index.faiss"),
                  "passages_sha256": file_sha256(passages_path)},
        "chunking": chunking,
        "books": books,
        "libraries": library_versions(),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    (out_dir / "index_config.json").write_text(json.dumps(index_config, indent=2) + "\n", encoding="utf-8")
    truncation = index_config["embedding"]["truncation"]
    print(f"Index: {len(all_chunks)} passages, dimension {matrix.shape[1]}, {index_config['elapsed_seconds']}s")
    print(f"Truncation estimate: {truncation}")
    print(f"Written to {out_dir.relative_to(ROOT).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
