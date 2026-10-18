"""Split cleaned pages into passages using the project's passage schema.

Reads  data/cleaned/<book_id>.pages.jsonl
Writes data/chunks/<book_id>.chunks.jsonl and evaluation/chunking_report.json

Chunk IDs: <book_id>_P<start page, 4 digits>_C<n, 2 digits>, e.g. G090_BOOK_003_P0143_C02.
Pages are PDF positions. Run:  python src/chunk.py [--book-id G090_BOOK_001 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "cleaned"
OUT_DIR = ROOT / "data" / "chunks"
REPORT = ROOT / "evaluation" / "chunking_report.json"

# About 4 characters per token, so 1500 chars is roughly 375 tokens.
# Check this against the approved embedding model's maximum sequence length.
TARGET_CHARS = 1500
MAX_CHARS = 2200
OVERLAP_CHARS = 200
MIN_ALNUM = 60  # chunks with fewer alphanumeric characters are dropped

HEADING = re.compile(r"^(?:chapter\s+\d+\b.*|\d+(?:\.\d+){0,3}\s+[A-Z].{2,80})$", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def is_heading(line: str) -> bool:
    line = line.strip()
    return 3 < len(line) <= 90 and not line.endswith((".", ",", ";")) and bool(HEADING.match(line))


def split_long(text: str) -> list[str]:
    """Split a paragraph longer than MAX_CHARS at sentence boundaries."""
    if len(text) <= MAX_CHARS:
        return [text]
    parts, current = [], ""
    for sentence in SENTENCE.split(text):
        while len(sentence) > MAX_CHARS:  # no sentence boundary: cut at whitespace
            cut = sentence.rfind(" ", 0, MAX_CHARS)
            cut = cut if cut > MAX_CHARS // 2 else MAX_CHARS
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > TARGET_CHARS:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def build_units(records: list[dict]) -> tuple[list[dict], list[int]]:
    units, skipped, section = [], [], None
    for record in records:
        if not record.get("usable", True):
            skipped.append(record["page"])
            continue
        for paragraph in re.split(r"\n\s*\n", record["text"]):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            headings = [line for line in paragraph.splitlines() if is_heading(line)]
            if headings:
                section = headings[0].strip()
            for piece in split_long(paragraph):
                units.append({"text": piece, "page": record["page"], "section": section,
                              "respace": bool(record.get("needs_respacing"))})
            if headings:
                section = headings[-1].strip()
    return units, skipped


def overlap_tail(units: list[dict]) -> list[dict]:
    tail, total = [], 0
    for unit in reversed(units):
        if total + len(unit["text"]) > OVERLAP_CHARS:
            break
        tail.insert(0, unit)
        total += len(unit["text"])
    return tail


def group_units(units: list[dict]) -> list[list[dict]]:
    groups, current, size, fresh = [], [], 0, 0
    for unit in units:
        add = len(unit["text"]) + 2
        if current and fresh and size + add > TARGET_CHARS:
            groups.append(current)
            current = overlap_tail(current)
            size = sum(len(u["text"]) + 2 for u in current)
            fresh = 0
        if current and size + add > MAX_CHARS:  # drop the overlap rather than exceed the cap
            current, size = [], 0
        current.append(unit)
        size += add
        fresh += 1
    if current and fresh:
        groups.append(current)
    return groups


def chunk_book(source: Path) -> dict:
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    first = records[0]
    book_id, group_id = first["book_id"], first.get("group_id")
    if not group_id:
        raise SystemExit(f"{book_id}: group_id is empty. Set project.group_id in config/settings.yaml, "
                         "then re-run ingest.py and clean.py.")
    units, skipped = build_units(records)
    per_page: Counter = Counter()
    chunks, dropped = [], 0
    for group in group_units(units):
        text = "\n\n".join(u["text"] for u in group)
        if sum(c.isalnum() for c in text) < MIN_ALNUM:
            dropped += 1
            continue
        page = group[0]["page"]
        per_page[page] += 1
        chunks.append({
            "group_id": group_id,
            "subject": first.get("subject"),
            "book_id": book_id,
            "title": first["title"],
            "author": first.get("author", []),
            "edition": first.get("edition"),
            "page": page,
            "page_end": group[-1]["page"],
            "page_reference_type": "pdf_position",
            "chunk_id": f"{book_id}_P{page:04d}_C{per_page[page]:02d}",
            "section": group[0]["section"],
            "text": text,
            "character_count": len(text),
            "needs_respacing": any(u["respace"] for u in group),
        })
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), f"{book_id}: duplicate chunk IDs"
    path = OUT_DIR / f"{book_id}.chunks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    sizes = [c["character_count"] for c in chunks] or [0]
    return {
        "book_id": book_id,
        "chunks": len(chunks),
        "dropped_small_chunks": dropped,
        "skipped_unusable_pdf_pages": skipped,
        "chunks_with_section": sum(c["section"] is not None for c in chunks),
        "chunks_from_respacing_pages": sum(c["needs_respacing"] for c in chunks),
        "characters_min": min(sizes), "characters_mean": round(mean(sizes)), "characters_max": max(sizes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-id", action="append", dest="book_ids")
    arguments = parser.parse_args()
    files = sorted(IN_DIR.glob("*.pages.jsonl"))
    if arguments.book_ids:
        wanted = {f"{b}.pages.jsonl" for b in arguments.book_ids}
        files = [f for f in files if f.name in wanted]
    if not files:
        raise SystemExit(f"No cleaned files found in {IN_DIR}. Run clean.py first.")
    results = []
    for source in files:
        result = chunk_book(source)
        results.append(result)
        print(f"{result['book_id']}: {result['chunks']} chunks, mean {result['characters_mean']} chars, "
              f"max {result['characters_max']}, dropped {result['dropped_small_chunks']}", flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "settings": {"target_chars": TARGET_CHARS, "max_chars": MAX_CHARS,
                     "overlap_chars": OVERLAP_CHARS, "min_alnum": MIN_ALNUM},
        "total_chunks": sum(r["chunks"] for r in results), "books": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Total chunks: {sum(r['chunks'] for r in results)}. Report: {REPORT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())