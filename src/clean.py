"""Clean extracted page text.

Reads  data/extracted/<book_id>.pages.jsonl
Writes data/cleaned/<book_id>.pages.jsonl and evaluation/cleaning_report.json

Raw extraction is never modified. Run:  python src/clean.py [--book-id CA_BOOK_001 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "extracted"
OUT_DIR = ROOT / "data" / "cleaned"
REPORT = ROOT / "evaluation" / "cleaning_report.json"

MIN_ALNUM = 40        # pages below this are marked usable=false
EDGE_LINES = 2        # first/last N non-empty lines are header/footer candidates
REPEAT_MIN = 8        # an edge line seen on this many pages is a running header/footer
MAX_HEADER_LEN = 80

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

# Removed wherever they appear inside text (add more as you find them).
WATERMARKS = [re.compile(p, re.I) for p in (
    r"team\s*ling",
    r"ptg\d{5,}",
    r"pdf\s*drive",
    r"www\.pdfdrive\.\w+",
)]

# A line matching any of these is dropped entirely (typesetting/production stamps).
JUNK_LINES = [re.compile(p) for p in (
    rf"\b(?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}\s+\d{{1,2}}:\d{{2}}",
    r"\bV\d+\s*-\s*\d{1,2}/\d{1,2}/\d{4}",
)]

PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?(?:\d{1,4}|[ivxlc]{1,6})\s*$", re.I)

LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
             "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
# Remove invisible formatting characters.
STRIP_CHARS = re.compile("[\u00ad\u200b\u200c\u200d\ufeff]")

# Preserve suspicious symbols so we can identify and review them.
REVIEW_CHARS = re.compile("[\ue000-\uf8ff\ufffd]")

# Temporarily exclude pages with these unresolved extraction problems.
BLOCKING_QUALITY_FLAGS = {
    "fragmented_word_spacing",
    "unicode_review",
    "image_without_substantive_text",
}


def normalize(text: str) -> str:
    for ligature, plain in LIGATURES.items():
        text = text.replace(ligature, plain)
    text = text.replace("\u00a0", " ")
    text = STRIP_CHARS.sub("", text)
    for pattern in WATERMARKS:
        text = pattern.sub("", text)
    return text


def line_key(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().casefold()))


def edge_indexes(lines: list[str]) -> set[int]:
    filled = [i for i, line in enumerate(lines) if line.strip()]
    return set(filled[:EDGE_LINES] + filled[-EDGE_LINES:])


def find_running_headers(texts: list[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        lines = text.splitlines()
        keys = {line_key(lines[i]) for i in edge_indexes(lines) if len(lines[i].strip()) <= MAX_HEADER_LEN}
        counts.update(keys)
    return Counter({key: n for key, n in counts.items() if n >= REPEAT_MIN})


def clean_page(text: str, headers: Counter, stats: Counter) -> str:
    lines = text.splitlines()
    edge = edge_indexes(lines)
    kept = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and any(p.search(stripped) for p in JUNK_LINES):
            stats["junk_lines"] += 1
            continue
        if stripped and i in edge:
            if PAGE_NUMBER.match(stripped):
                stats["page_numbers"] += 1
                continue
            if len(stripped) <= MAX_HEADER_LEN and line_key(stripped) in headers:
                stats["running_headers"] += 1
                continue
        kept.append(line.rstrip())
    text = "\n".join(kept)
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", text)     # rejoin hyphenated line breaks
    text = re.sub(r"(?<=[A-Za-z,;])\n(?=[a-z])", " ", text)   # join soft-wrapped lines
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_lines(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def clean_book(source: Path) -> dict:
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = [normalize(r["text"]) for r in records]
    headers = find_running_headers(normalized)
    stats: Counter = Counter()
    unusable = []
    for record, text in zip(records, normalized):
        cleaned = clean_page(text, headers, stats)
        alnum = sum(c.isalnum() for c in cleaned)
        flags = set(record.get("quality_flags", []))

        # Check original text for suspicious symbols.
        if REVIEW_CHARS.search(record["text"]):
            flags.add("unicode_review")

        blocked_flags = flags & BLOCKING_QUALITY_FLAGS
        exclusion_reasons = sorted(blocked_flags)

        if alnum < MIN_ALNUM:
            exclusion_reasons.append("insufficient_clean_text")

        record["raw_character_count"] = record["character_count"]
        record["text"] = cleaned
        record["character_count"] = len(cleaned)
        record["alphanumeric_characters"] = alnum
        record["quality_flags"] = sorted(flags)

        record["needs_respacing"] = "fragmented_word_spacing" in flags
        record["review_required"] = bool(blocked_flags)
        record["exclusion_reasons"] = exclusion_reasons
        record["usable"] = not exclusion_reasons
        if not record["usable"]:
            unusable.append(record["pdf_page"])
    book_id = records[0]["book_id"]
    write_lines(OUT_DIR / f"{book_id}.pages.jsonl", records)
    return {
        "book_id": book_id,
        "pages": len(records),
        "usable_pages": len(records) - len(unusable),
        "unusable_pdf_pages": unusable,
        "needs_respacing_pages": [r["pdf_page"] for r in records if r["needs_respacing"]],
        "removed": dict(stats),
        "running_headers_removed": [{"pattern": k, "pages": n} for k, n in headers.most_common(25)],
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
        raise SystemExit(f"No extracted files found in {IN_DIR}")
    results = []
    for source in files:
        result = clean_book(source)
        results.append(result)
        print(f"{result['book_id']}: {result['usable_pages']}/{result['pages']} usable, "
              f"removed {result['removed']}, respacing needed on {len(result['needs_respacing_pages'])} pages", flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"books": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {REPORT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())