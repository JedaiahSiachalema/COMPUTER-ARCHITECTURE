from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "config" / "settings.yaml"
IN_DIR = ROOT / "data" / "cleaned"
OUT_DIR = ROOT / "data" / "chunks"
REPORT = ROOT / "evaluation" / "chunking_report.json"

BLOCKING_FLAGS = {
    "unicode_review", "fragmented_word_spacing", "image_without_substantive_text",
}
PARAGRAPH = re.compile(r"\S[^\n]*(?:\n(?![ \t]*\n)[^\n]*)*")
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){1,3})\s+([A-Z][^\n]{1,75})$")
CHAPTER_HEADING = re.compile(r"^(?:Chapter|CHAPTER|chapter)\s+(?:\d+|[IVXLC]+)$")
QUESTION_WORDS = {
    "Assume", "Calculate", "Consider", "Determine", "Explain", "Find", "For",
    "Given", "How", "If", "Prove", "Show", "Suppose", "What", "When", "Which",
    "Why", "Write",
}


def load_runtime(settings_path: Path = SETTINGS):
    """Load configuration and cached tokenizer, without loading model weights."""
    from transformers import AutoConfig, AutoTokenizer
    from transformers.utils.hub import cached_file

    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    embedding = settings.get("embedding") or {}
    chunking = settings.get("chunking") or {}
    model, revision = embedding.get("model_name"), embedding.get("revision")
    if not model or not revision:
        raise ValueError("Set embedding.model_name and its exact revision before chunking.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, use_fast=True, local_files_only=True,
        )
        model_config = AutoConfig.from_pretrained(model, revision=revision, local_files_only=True)
        sentence_config_path = cached_file(
            model, "sentence_bert_config.json", revision=revision, local_files_only=True,
            _raise_exceptions_for_missing_entries=False,
        )
    except OSError as error:
        raise ValueError("The pinned tokenizer/config is not cached. Download that revision first.") from error
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for source character offsets.")
    limits = [
        int(limit) for limit in (
            tokenizer.model_max_length, getattr(model_config, "max_position_embeddings", None),
        ) if limit is not None and 0 < int(limit) < 1_000_000
    ]
    if sentence_config_path:
        sentence_config = json.loads(Path(sentence_config_path).read_text(encoding="utf-8"))
        if sentence_config.get("max_seq_length"):
            limits.append(int(sentence_config["max_seq_length"]))
    if not limits:
        raise ValueError("Could not establish the embedding model's input token limit.")
    size = chunking.get("size_tokens")
    overlap = chunking.get("overlap_tokens")
    options = {
        "model_name": model, "revision": revision, "max_input_tokens": min(limits),
        "size_tokens": int(400 if size is None else size),
        "overlap_tokens": int(50 if overlap is None else overlap),
        "document_prefix": embedding.get("document_prefix") or "",
        "include_section": bool(embedding.get("include_section", True)),
    }
    validate_options(options)
    return tokenizer, options


def validate_options(options: dict) -> None:
    if not 1 <= options["size_tokens"] <= options["max_input_tokens"]:
        raise ValueError("chunking.size_tokens must fit within the embedding model's token limit.")
    if not 0 <= options["overlap_tokens"] < options["size_tokens"]:
        raise ValueError("overlap_tokens must be nonnegative and smaller than size_tokens.")


def embedding_text(text: str, section: str | None, options: dict) -> str:
    """Match src/embed.py's passage_text(), including document/section prefixes."""
    if options["include_section"] and section:
        text = f"{section}\n{text}"
    return options["document_prefix"] + text


def token_count(tokenizer, text: str, *, special: bool = True) -> int:
    return len(tokenizer(
        text, add_special_tokens=special, truncation=False, verbose=False,
    )["input_ids"])


def heading_label(paragraph: str) -> str | None:
    """Recognize short standalone numbered headings; leave uncertain labels empty."""
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not 1 <= len(lines) <= 2:
        return None
    text = " ".join(lines)
    if CHAPTER_HEADING.fullmatch(text):
        return text
    match = NUMBERED_HEADING.fullmatch(text)
    if not match:
        return None
    title = match.group(2)
    if (title.endswith((".", "?", "!", ";")) or title[-1].isdigit()
            or len(title.split()) > 10 or title.split()[0] in QUESTION_WORDS):
        return None
    return text


def page_segments(text: str) -> list[tuple[int, int, str | None]]:
    """Return exact source spans, splitting at recognized headings within this page."""
    segments = []
    start, section = 0, None
    for paragraph in PARAGRAPH.finditer(text):
        label = heading_label(paragraph.group())
        if label is not None:
            if text[start:paragraph.start()].strip():
                segments.append((start, paragraph.start(), section))
            start, section = paragraph.start(), label
    if text[start:].strip():
        segments.append((start, len(text), section))
    return segments


def segment_windows(text: str, section: str | None, tokenizer, options: dict):
    """Yield source spans/counts; retain original characters instead of decoding tokens."""
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True,
        truncation=False, verbose=False,
    )
    offsets, words = encoded["offset_mapping"], encoded.word_ids()
    total = len(offsets)
    if not total:
        return
    overhead = token_count(tokenizer, embedding_text("", section, options))
    budget = options["size_tokens"] - overhead
    if budget <= options["overlap_tokens"]:
        raise ValueError("Prefixes leave too little room for text/overlap; shorten prefixes or reduce overlap.")
    start, previous_end = 0, None
    while start < total:
        end = min(start + budget, total)
        while True:
            # Keep all WordPiece pieces of a word on the same side of a boundary.
            while end > start and end < total and words[end - 1] == words[end]:
                end -= 1
            if end <= start:
                raise ValueError("A single word exceeds the chunk token budget; review this page.")
            char_start = 0 if start == 0 else offsets[start][0]
            # Include intervening characters that normalization may omit (e.g. accents).
            char_end = len(text) if end == total else offsets[end][0]
            while char_start < char_end and text[char_start].isspace():
                char_start += 1
            while char_end > char_start and text[char_end - 1].isspace():
                char_end -= 1
            passage = text[char_start:char_end]
            count = token_count(tokenizer, embedding_text(passage, section, options))
            if count <= options["size_tokens"]:
                break
            end -= 1
        overlap = 0
        if previous_end is not None and char_start < previous_end:
            overlap = token_count(tokenizer, text[char_start:previous_end], special=False)
        yield char_start, char_end, count, overlap
        if end == total:
            break
        previous_end = char_end
        next_start = end - options["overlap_tokens"]
        if options["overlap_tokens"]:
            while next_start > start and words[next_start] == words[next_start - 1]:
                next_start -= 1
        # Very long words can consume the overlap; always advance.
        start = next_start if next_start > start else end


def build_chunks(records: list[dict], tokenizer, options: dict) -> tuple[list[dict], list[dict]]:
    """Pure in-memory construction; publishing is handled separately."""
    validate_options(options)
    chunks, skipped = [], []
    seen_pages = set()
    book_id = records[0]["book_id"] if records else None
    for record in records:
        page = record["page"]
        if record["book_id"] != book_id or page in seen_pages:
            raise ValueError("Cleaned book contains mixed book IDs or duplicate page positions.")
        seen_pages.add(page)
        if not record.get("group_id"):
            raise ValueError(f"{book_id}: missing group_id; correct ingestion metadata first.")
        reasons = set(record.get("exclusion_reasons", []))
        reasons.update(set(record.get("quality_flags", [])) & BLOCKING_FLAGS)
        if record.get("usable") is not True:
            reasons.add("not_marked_usable")
        if record.get("review_required"):
            reasons.add("review_required")
        if not record["text"].strip():
            reasons.add("empty_text")
        if reasons:
            skipped.append({"page": page, "reasons": sorted(reasons)})
            continue
        sequence = 0
        for segment_number, (begin, end, section) in enumerate(page_segments(record["text"]), 1):
            segment = record["text"][begin:end]
            for start, stop, count, overlap in segment_windows(segment, section, tokenizer, options):
                sequence += 1
                char_start, char_end = begin + start, begin + stop
                text = record["text"][char_start:char_end]
                chunk = {key: record.get(key) for key in (
                    "group_id", "subject", "book_id", "title", "author", "edition",
                    "source_file", "source_sha256", "page_id", "pdf_page_label", "printed_page",
                )}
                chunk.update({
                    "page": page, "page_end": page, "page_reference_type": "pdf_position",
                    "chunk_id": f"{book_id}_P{page:04d}_C{sequence:02d}",
                    "section": section, "segment": segment_number, "text": text,
                    "start_char": char_start, "end_char": char_end,
                    "source_spans": [{"page": page, "start_char": char_start, "end_char": char_end}],
                    "character_count": len(text),
                    "token_count": token_count(tokenizer, text, special=False),
                    "embedding_token_count": count, "overlap_tokens": overlap,
                    "tokenizer_model": options["model_name"], "tokenizer_revision": options["revision"],
                    "quality_flags": list(record.get("quality_flags", [])),
                    "needs_respacing": False, "usable": True,
                    "review_required": False, "exclusion_reasons": [],
                })
                chunks.append(chunk)
    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{book_id}: duplicate chunk IDs")
    return chunks, skipped


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def chunk_book(source: Path, tokenizer, options: dict) -> dict:
    raw = source.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Empty cleaned book: {source.name}")
    book_id = records[0]["book_id"]
    if source.name != f"{book_id}.pages.jsonl":
        raise ValueError(f"Filename does not match book_id: {source.name}")
    chunks, skipped = build_chunks(records, tokenizer, options)
    if not chunks:
        raise ValueError(f"{book_id}: no eligible chunks; inspect cleaning before embedding.")
    atomic_text(OUT_DIR / f"{book_id}.chunks.jsonl", "".join(
        json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks
    ))
    sizes = [chunk["character_count"] for chunk in chunks]
    tokens = [chunk["embedding_token_count"] for chunk in chunks]
    overlaps = [chunk["overlap_tokens"] for chunk in chunks if chunk["overlap_tokens"]]
    return {
        "book_id": book_id, "chunks": len(chunks), "source_pages": len(records),
        "source_cleaned_sha256": hashlib.sha256(raw).hexdigest(),
        "skipped_unusable_pdf_pages": [item["page"] for item in skipped],
        "skipped_page_details": skipped,
        "chunks_with_section": sum(chunk["section"] is not None for chunk in chunks),
        "chunks_from_respacing_pages": 0, "dropped_small_chunks": 0,
        "characters_min": min(sizes), "characters_mean": round(mean(sizes)), "characters_max": max(sizes),
        "embedding_tokens_min": min(tokens), "embedding_tokens_mean": round(mean(tokens)),
        "embedding_tokens_max": max(tokens), "over_limit_chunks": 0,
        "continuation_chunks_with_overlap": len(overlaps),
        "overlap_tokens_min": min(overlaps) if overlaps else 0,
        "overlap_tokens_max": max(overlaps) if overlaps else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-id", action="append", dest="book_ids")
    arguments = parser.parse_args()
    try:
        tokenizer, options = load_runtime()
        settings = yaml.safe_load(SETTINGS.read_text(encoding="utf-8")) or {}
        manifest = json.loads((ROOT / settings["sources"]["book_manifest"]).read_text(encoding="utf-8"))
        known = [book["book_id"] for book in manifest]
        if len(known) != len(set(known)):
            raise ValueError("Duplicate book IDs in the manifest.")
        selected = set(arguments.book_ids or known)
        if selected - set(known):
            raise ValueError(f"Unknown book IDs: {sorted(selected - set(known))}")
        files = [IN_DIR / f"{book}.pages.jsonl" for book in known if book in selected]
        if not files:
            raise ValueError("No books selected for chunking.")
        missing = [path.name for path in files if not path.is_file()]
        if missing:
            raise ValueError(f"Missing cleaned books: {missing}. Run clean.py first.")
        print(f"Tokenizer: {options['model_name']} @ {options['revision']}", flush=True)
        print(f"Limit: {options['size_tokens']} tokens including prefixes/special tokens; "
              f"overlap target: {options['overlap_tokens']} tokens; single-page chunks.", flush=True)
        results = []
        for source in files:
            result = chunk_book(source, tokenizer, options)
            results.append(result)
            print(f"{result['book_id']}: {result['chunks']} chunks, "
                  f"max {result['embedding_tokens_max']} input tokens, "
                  f"{len(result['skipped_unusable_pdf_pages'])} excluded pages", flush=True)
        report = {
            "settings": {**options, "method": "single_page_token_windows", "single_page": True,
                         "overlap_scope": "within_page_and_detected_section",
                         "offset_reference": "cleaned_page_text", "word_boundary_alignment": True},
            "scope": "selected_books" if arguments.book_ids else "all_manifest_books",
            "total_chunks": sum(result["chunks"] for result in results),
            "over_limit_chunks": 0, "books": results,
        }
        atomic_text(REPORT, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"Total chunks: {report['total_chunks']}. Report: {REPORT.relative_to(ROOT).as_posix()}")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Chunking failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
