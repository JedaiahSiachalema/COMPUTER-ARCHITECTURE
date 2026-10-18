"""Billy's retrieval interface: return ranked book passages, without generating answers.

Run: python src/retrieve.py "What is a data hazard?" --top-k 5
Python: from src.retrieve import retrieve
Scores are cosine similarities, not answer-confidence probabilities.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from threading import RLock

import faiss
import numpy as np
import yaml

if __package__:
    from .embed import embed_queries, load_model
    from .runtime_device import device_preference
else:
    from embed import embed_queries, load_model
    from runtime_device import device_preference

ROOT = Path(__file__).resolve().parents[1]
CACHE_LOCK = RLock()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = (root / path).resolve()
    if ".." in path.parts or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Retrieval artifact must be inside the project: {value}")
    return resolved


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_request(question: str, top_k: int) -> str:
    if not isinstance(question, str):
        raise TypeError("question must be a string.")
    if not question.strip():
        raise ValueError("Enter a nonempty question.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    return question.strip()


def artifact_paths(root: Path, settings: dict, index_directory=None):
    directory = project_path(root, index_directory or
                             (settings.get("retrieval") or {}).get("index_directory", "artifacts/index"))
    metadata_path = directory / "index_config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("Index metadata is missing. Finish src/embed.py first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    index_path = project_path(root, directory / metadata["index"]["file"])
    passages_path = project_path(root, metadata["index"]["passages_file"])
    return directory, metadata_path, index_path, passages_path, metadata


def validate_bundle(index, passages: list[dict], metadata: dict) -> None:
    """Check the vector/passage contract before loading the question encoder."""
    embedding = metadata["embedding"]
    if not embedding.get("model_name") or not embedding.get("revision"):
        raise ValueError("Index metadata must identify the model and its pinned revision.")
    if embedding.get("normalize") is not True or not isinstance(embedding.get("query_prefix"), str):
        raise ValueError("Index is missing normalized-embedding/query-prefix settings.")
    if not isinstance(index, faiss.IndexFlatIP) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("This retriever requires the saved IndexFlatIP cosine-search index.")
    if not passages or index.ntotal != len(passages) or len(passages) != metadata["index"]["passages"]:
        raise ValueError("Index vector count and passage count do not match.")
    if index.d != embedding["dimension"]:
        raise ValueError("Index dimension does not match its embedding metadata.")
    ids = set()
    for passage in passages:
        for key in ("text", "book_id", "title", "chunk_id"):
            if not isinstance(passage.get(key), str) or not passage[key].strip():
                raise ValueError(f"A stored passage is missing {key}.")
        if (not isinstance(passage.get("page"), int) or isinstance(passage["page"], bool)
                or passage["page"] < 1 or passage["page"] != passage.get("page_end")):
            raise ValueError("A stored passage does not have a valid single-page citation.")
        if passage["chunk_id"] in ids:
            raise ValueError("Duplicate canonical chunk IDs in the passage store.")
        ids.add(passage["chunk_id"])
        if not passage.get("sources") or not any(
            all(source.get(key) == passage[key] for key in ("book_id", "page", "chunk_id"))
            for source in passage["sources"]
        ):
            raise ValueError("A stored passage has lost its primary source reference.")
    vectors = index.reconstruct_n(0, index.ntotal)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        raise ValueError("Index contains nonfinite or nonnormalized vectors.")


def verify_legacy_order(root: Path, directory: Path, index, passages: list[dict], metadata: dict) -> None:
    """Verify existing pre-fingerprint indexes against their prepared corpus and vector caches."""
    provenance = metadata["corpus_preparation"]
    report_path = project_path(root, provenance["report"])
    if file_hash(report_path) != provenance["report_sha256"]:
        raise ValueError("Corpus preparation changed after indexing. Rerun src/embed.py.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    position = 0
    for output in report["outputs"]:
        source = project_path(root, output["path"])
        if file_hash(source) != output["sha256"]:
            raise ValueError("Prepared passages changed after indexing. Rebuild the corpus/index.")
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        count = len(rows)
        if passages[position:position + count] != rows:
            raise ValueError("Passage order/content does not match the indexed prepared corpus.")
        vector_path = project_path(root, directory / "vectors" / f"{output['book_id']}.npy")
        cache_path = vector_path.with_suffix(".json")
        expected_key = {
            "chunks_sha256": output["sha256"],
            **{key: metadata["embedding"][key] for key in (
                "model_name", "revision", "document_prefix", "include_section",
            )},
        }
        if json.loads(cache_path.read_text(encoding="utf-8")) != expected_key:
            raise ValueError("Vector cache model/source settings differ from the index.")
        cached = np.load(vector_path, allow_pickle=False)
        if cached.shape != (count, index.d) or not np.array_equal(
            cached, index.reconstruct_n(position, count),
        ):
            raise ValueError("Vector order/content differs from passage order. Rerun src/embed.py.")
        position += count
    if position != len(passages):
        raise ValueError("The prepared corpus does not account for every indexed passage.")


class Retriever:
    """Load an index once; lazily load its pinned encoder on the preferred device."""

    def __init__(self, root: Path = ROOT, index_directory=None):
        self.root = Path(root).resolve()
        settings = yaml.safe_load((self.root / "config/settings.yaml").read_text(encoding="utf-8")) or {}
        directory, _, index_path, passages_path, self.metadata = artifact_paths(
            self.root, settings, index_directory,
        )
        stored = self.metadata["index"]
        fingerprints = (stored.get("sha256"), stored.get("passages_sha256"))
        if any(fingerprints) and not all(fingerprints):
            raise ValueError("Index metadata has incomplete artifact fingerprints.")
        if all(fingerprints) and (file_hash(index_path), file_hash(passages_path)) != fingerprints:
            raise ValueError("Index/passages changed after embedding. Rerun src/embed.py.")
        self.index = faiss.read_index(str(index_path))
        self.passages = [json.loads(line) for line in passages_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        validate_bundle(self.index, self.passages, self.metadata)
        if not all(fingerprints):
            verify_legacy_order(self.root, directory, self.index, self.passages, self.metadata)
        # Model identity and prefix always come from the saved index, not editable model settings.
        self.config = {
            **self.metadata["embedding"], "batch_size": 1,
            "device": device_preference(settings, "retrieval"), "local_files_only": True,
        }
        self._model = None
        self._lock = RLock()

    def search(self, question: str, top_k: int = 5) -> list:
        question = validate_request(question, top_k)
        with self._lock:
            if self._model is None:
                self._model = load_model(self.config)
            prefixed = self.config["query_prefix"] + question
            tokens = self._model.tokenizer(prefixed, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
            if len(tokens) > self._model.max_seq_length:
                raise ValueError(f"Question exceeds the model's {self._model.max_seq_length}-token limit; shorten it.")
            query = np.ascontiguousarray(embed_queries(self._model, [question], self.config), dtype="float32")
            if query.shape != (1, self.index.d) or not np.isfinite(query).all():
                raise ValueError("Question encoder returned invalid vectors or the wrong dimension.")
            if not np.allclose(np.linalg.norm(query, axis=1), 1.0, atol=1e-5):
                raise ValueError("Question embeddings must be normalized for cosine search.")
            scores, indices = self.index.search(query, min(top_k, len(self.passages)))
            results = []
            for rank, (score, row) in enumerate(zip(scores[0], indices[0]), 1):
                if not 0 <= row < len(self.passages) or not np.isfinite(score):
                    raise ValueError("Index returned an invalid passage row or score.")
                # Copies prevent callers from changing the cached passage/citation data.
                result = deepcopy(self.passages[int(row)])
                result.update({"score": float(score), "rank": rank})
                results.append(result)
            return results


@lru_cache(maxsize=1)
def _cached_retriever(signature) -> Retriever:
    return Retriever()


def retrieve(question: str, top_k: int = 5) -> list:
    """Return top passages with text, score, book_id, title, page, chunk_id and sources.

    No similarity threshold or answer/abstention decision is applied here.
    """
    question = validate_request(question, top_k)
    settings_path = ROOT / "config/settings.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    directory, metadata_path, index_path, passages_path, _ = artifact_paths(ROOT, settings)
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                      for path in (settings_path, metadata_path, index_path, passages_path))
    with CACHE_LOCK:
        engine = _cached_retriever(signature)
    return engine.search(question, top_k)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print full records as JSON; loading logs go to stderr.")
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            results = retrieve(args.question, args.top_k)
    except (ValueError, TypeError, OSError, KeyError, RuntimeError) as error:
        parser.exit(2, f"Retrieval failed: {error}\n")
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(f"\n{result['rank']}. {result['title']} | {result['book_id']} | "
                  f"PDF page {result['page']} | similarity {result['score']:.4f}")
            print(f"   Chunk: {result['chunk_id']} | source references: {len(result['sources'])}")
            text = result["text"]
            print(text[:700] + ("..." if len(text) > 700 else ""))
        print("\nThese are retrieved source passages. Full text and metadata are available with --json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
