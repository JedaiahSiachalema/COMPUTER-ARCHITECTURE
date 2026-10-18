from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import yaml

if __package__:
    from .chunk import BLOCKING_FLAGS, atomic_text, embedding_text
else:
    from chunk import BLOCKING_FLAGS, atomic_text, embedding_text

ROOT = Path(__file__).resolve().parents[1]
POLICY = "config/corpus_filters.json"
REPORT = "evaluation/corpus_preparation_report.json"
EXCLUSIONS = "evaluation/corpus_exclusions.jsonl"
OUTPUT_DIR = "data/embedding"
SOURCE_FIELDS = (
    "group_id", "subject", "book_id", "title", "author", "edition", "page", "page_end",
    "chunk_id", "page_id", "page_reference_type", "pdf_page_label", "printed_page",
    "section", "source_file", "source_sha256", "source_spans", "start_char", "end_char",
)


def inside(root: Path, value: str) -> Path:
    relative = Path(value)
    resolved = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Expected a path inside the project: {value}")
    return resolved


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def embedding_settings(settings: dict) -> dict:
    section = settings.get("embedding") or {}
    return {
        "model_name": section.get("model_name"), "revision": section.get("revision"),
        "document_prefix": section.get("document_prefix") or "",
        "include_section": bool(section.get("include_section", True)),
    }


def validate_policy(policy: dict, books: list[dict]) -> None:
    if policy.get("version") != 1 or policy.get("page_reference_type") != "pdf_position":
        raise ValueError("Unsupported corpus policy version or page reference type.")
    ids = [book["book_id"] for book in books]
    if len(ids) != len(set(ids)) or set(policy.get("books", {})) != set(ids):
        raise ValueError("The filter policy must cover each manifest book exactly once.")
    for book in books:
        bid = book["book_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", bid):
            raise ValueError(f"Invalid book ID: {bid}")
        entry = policy["books"][bid]
        if entry.get("source_sha256") != book["sha256"] or entry.get("pdf_page_count") != book["pdf_page_count"]:
            raise ValueError(f"{bid}: source changed; review its page exclusions again.")
        previous_end = 0
        for span in entry["excluded_ranges"]:
            if not (isinstance(span["start"], int) and isinstance(span["end"], int)
                    and previous_end < span["start"] <= span["end"] <= book["pdf_page_count"]
                    and isinstance(span.get("reason"), str) and span["reason"].strip()):
                raise ValueError(f"{bid}: invalid, overlapping, or unexplained exclusion range.")
            previous_end = span["end"]


def exclusion_reason(passage: dict, policy: dict) -> str | None:
    for span in policy["books"][passage["book_id"]]["excluded_ranges"]:
        if span["start"] <= passage["page"] <= span["end"]:
            return span["reason"]
    if (passage.get("usable") is not True or passage.get("review_required")
            or passage.get("exclusion_reasons")
            or set(passage.get("quality_flags", [])) & BLOCKING_FLAGS):
        return "Unresolved cleaning quality flags"
    return None


def build_corpus(chunks: list[dict], policy: dict, options: dict):
    """Pure filtering/deduplication; retain every eligible original citation."""
    passages, excluded, by_input, seen_ids = [], [], {}, set()
    for original in chunks:
        if original["chunk_id"] in seen_ids:
            raise ValueError(f"Repeated source chunk ID: {original['chunk_id']}")
        seen_ids.add(original["chunk_id"])
        if original["page"] != original["page_end"]:
            raise ValueError("Corpus preparation requires the new single-page chunks.")
        if original.get("corpus_prepared"):
            raise ValueError("Read original data/chunks files, not already prepared passages.")
        reference = {key: original.get(key) for key in SOURCE_FIELDS}
        reason = exclusion_reason(original, policy)
        if reason:
            excluded.append({**reference, "reason": reason})
            continue
        text_input = embedding_text(original["text"], original.get("section"), options)
        # Equality is exact: no whitespace, punctuation, case or Unicode normalization.
        if text_input in by_input:
            canonical = passages[by_input[text_input]]
            canonical["sources"].append(reference)
            canonical["source_count"] += 1
        else:
            canonical = dict(original)
            canonical.update({
                "corpus_prepared": True, "sources": [reference], "source_count": 1,
                "text_sha256": hashlib.sha256(original["text"].encode("utf-8")).hexdigest(),
                "embedding_input_sha256": hashlib.sha256(text_input.encode("utf-8")).hexdigest(),
            })
            by_input[text_input] = len(passages)
            passages.append(canonical)
    kept = sum(p["source_count"] for p in passages)
    if len(chunks) != kept + len(excluded):
        raise ValueError("Corpus accounting mismatch; some source references were lost.")
    return passages, excluded


def read_inputs(root: Path, settings_path: Path):
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    manifest_path = inside(root, settings["sources"]["book_manifest"])
    books = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy_path = inside(root, POLICY)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    validate_policy(policy, books)
    options = embedding_settings(settings)
    chunk_report_path = inside(root, "evaluation/chunking_report.json")
    chunk_report = json.loads(chunk_report_path.read_text(encoding="utf-8"))
    if chunk_report.get("scope") != "all_manifest_books":
        raise ValueError("Run chunk.py for all manifest books before preparing the corpus.")
    if any(chunk_report["settings"].get(key) != value for key, value in options.items()):
        raise ValueError("Embedding input settings changed after chunking; rerun chunk.py.")
    chunking = settings.get("chunking") or {}
    for key, default in (("size_tokens", 400), ("overlap_tokens", 50)):
        value = chunking.get(key)
        if chunk_report["settings"].get(key) != (default if value is None else int(value)):
            raise ValueError("Chunk size/overlap settings changed after chunking; rerun chunk.py.")
    book_reports = {item["book_id"]: item for item in chunk_report["books"]}
    if set(book_reports) != {book["book_id"] for book in books}:
        raise ValueError("The chunking report does not cover all manifest books.")
    chunks, snapshots = [], []
    for book in books:
        bid = book["book_id"]
        cleaned_relative = f"data/cleaned/{bid}.pages.jsonl"
        cleaned_hash = sha256(inside(root, cleaned_relative))
        if cleaned_hash != book_reports[bid]["source_cleaned_sha256"]:
            raise ValueError(f"{bid}: cleaning changed after chunking; rerun chunk.py.")
        relative = f"data/chunks/{bid}.chunks.jsonl"
        path = inside(root, relative)
        raw = path.read_bytes()
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if not records or len(records) != book_reports[bid]["chunks"]:
            raise ValueError(f"{bid}: missing chunks or count differs from the chunking report.")
        for record in records:
            if (record["book_id"] != bid or record.get("source_sha256") != book["sha256"]
                    or record.get("tokenizer_model") != options["model_name"]
                    or record.get("tokenizer_revision") != options["revision"]):
                raise ValueError(f"{bid}: source/model provenance mismatch.")
            if not 0 < record["embedding_token_count"] <= chunk_report["settings"]["size_tokens"]:
                raise ValueError(f"{bid}: invalid embedding input length.")
        chunks.extend(records)
        snapshots.append({
            "book_id": bid, "path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
            "cleaned_path": cleaned_relative, "cleaned_sha256": cleaned_hash,
        })
    return chunks, policy, options, snapshots, {
        "manifest_path": manifest_path.relative_to(root).as_posix(), "manifest_sha256": sha256(manifest_path),
        "policy_path": POLICY, "policy_sha256": sha256(policy_path),
        "chunk_report_path": chunk_report_path.relative_to(root).as_posix(),
        "chunk_report_sha256": sha256(chunk_report_path),
    }


def prepare(root: Path = ROOT) -> dict:
    root = root.resolve()
    chunks, policy, options, snapshots, provenance = read_inputs(root, root / "config/settings.yaml")
    passages, excluded = build_corpus(chunks, policy, options)
    represented = {source["book_id"] for passage in passages for source in passage["sources"]}
    if represented != set(policy["books"]):
        raise ValueError("Filtering leaves a book without any retained evidence; review exclusions.")
    grouped = defaultdict(list)
    for passage in passages:
        grouped[passage["book_id"]].append(passage)
    outputs = []
    for bid, records in grouped.items():
        relative = f"{OUTPUT_DIR}/{bid}.chunks.jsonl"
        path = inside(root, relative)
        atomic_text(path, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))
        outputs.append({"book_id": bid, "path": relative, "sha256": sha256(path), "passages": len(records)})
    atomic_text(inside(root, EXCLUSIONS), "".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in excluded
    ))
    counts = Counter(item["book_id"] for item in excluded)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(ZoneInfo("Africa/Lusaka")).isoformat(),
        "embedding_input": options, **provenance, "inputs": snapshots, "outputs": outputs,
        "input_chunks": len(chunks), "excluded_chunks": len(excluded),
        "retained_source_chunks": len(chunks) - len(excluded), "unique_passages": len(passages),
        "duplicate_copies_merged": len(chunks) - len(excluded) - len(passages),
        "books_represented": sorted(represented), "exclusion_log": EXCLUSIONS,
        "excluded_by_book": {bid: counts[bid] for bid in policy["books"]},
        "deduplication": "Exact embedding-input equality, retaining all original citations in sources",
    }
    # Publish the report last; embedding verifies every listed input/output hash.
    atomic_text(inside(root, REPORT), json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def load_prepared_sources(root: Path, settings_path: Path):
    """Fail before embedding if prepared inputs are absent, stale or edited."""
    root = root.resolve()
    report_path = inside(root, REPORT)
    if not report_path.is_file():
        raise ValueError("Run src/prepare_corpus.py before embedding.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    if report.get("schema_version") != 1 or report["embedding_input"] != embedding_settings(settings):
        raise ValueError("Prepared corpus settings changed; rerun prepare_corpus.py.")
    for key in ("manifest", "policy", "chunk_report"):
        if sha256(inside(root, report[f"{key}_path"])) != report[f"{key}_sha256"]:
            raise ValueError(f"The {key} changed; rerun prepare_corpus.py.")
    for item in report["inputs"]:
        if (sha256(inside(root, item["path"])) != item["sha256"]
                or sha256(inside(root, item["cleaned_path"])) != item["cleaned_sha256"]):
            raise ValueError("Source chunks/cleaning changed; rerun prepare_corpus.py.")
    files = []
    for output in report["outputs"]:
        path = inside(root, output["path"])
        if sha256(path) != output["sha256"]:
            raise ValueError("Prepared passage file changed; rerun prepare_corpus.py.")
        files.append(path)
    if not files:
        raise ValueError("Prepared corpus is empty.")
    return files, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        report = prepare()
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Corpus preparation failed: {error}\n")
    for field in ("input_chunks", "excluded_chunks", "duplicate_copies_merged", "unique_passages"):
        print(f"{field}: {report[field]}")
    print(f"Books represented: {len(report['books_represented'])}")
    print(f"Prepared passages: {OUTPUT_DIR}/")
    print(f"Report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
