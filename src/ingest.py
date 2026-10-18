from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pymupdf
import yaml


DEFAULT_EXCLUDED = {"ignored", "output", "tmp"}


def _project_path(root: Path, value: str, *, source: bool = False,
                  excluded: set[str] | None = None) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a project-relative path without traversal: {value}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path escapes the project or resolves to its root: {value}")
    if source:
        directory_names = {part.casefold() for part in path.relative_to(root).parts[:-1]}
        if directory_names & (excluded or DEFAULT_EXCLUDED):
            raise ValueError(f"Source is inside an excluded directory: {value}")
        if path.suffix.casefold() != ".pdf":
            raise ValueError(f"Manifest source must be a PDF: {value}")
    return path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_manifest(books: object, root: Path, excluded: set[str]) -> list[dict]:
    if not isinstance(books, list) or not books:
        raise ValueError("The book manifest must be a non-empty JSON list.")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for book in books:
        if not isinstance(book, dict):
            raise ValueError("Every manifest entry must be a JSON object.")
        book_id = book.get("book_id", "")
        if not isinstance(book_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", book_id):
            raise ValueError(f"Invalid book_id: {book_id!r}")
        if book_id.casefold() in seen_ids:
            raise ValueError(f"Duplicate book_id: {book_id}")
        seen_ids.add(book_id.casefold())
        if not book.get("title") or not isinstance(book.get("file_path"), str):
            raise ValueError(f"Missing title or file_path for {book_id}")
        path = _project_path(root, book["file_path"], source=True, excluded=excluded)
        path_key = str(path).casefold()
        if path_key in seen_paths:
            raise ValueError(f"Duplicate source file: {book['file_path']}")
        seen_paths.add(path_key)
    return books


def _text_flags(text: str, image_count: int, minimum: int) -> list[str]:
    flags = []
    alphanumeric = sum(character.isalnum() for character in text)
    if alphanumeric == 0:
        flags.append("no_text")
    if alphanumeric < minimum:
        flags.append("low_text")
        if image_count:
            flags.append("image_without_substantive_text")
    if "\ufffd" in text or sum("\ue000" <= character <= "\uf8ff" for character in text) >= 3:
        flags.append("unicode_review")
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) >= 80 and sum(len(word) == 1 for word in words) / len(words) > 0.35:
        flags.append("fragmented_word_spacing")
    return flags


def extract_book(book: dict, project_root: Path, output_dir: Path, *,
                 group_id: str | None = None, minimum_text_characters: int = 40) -> dict:
    """Publish a complete page JSONL or return a failed/OCR-needed report."""
    started = time.monotonic()
    book_id = book["book_id"]
    source = (project_root / book["file_path"]).resolve()
    destination = output_dir / f"{book_id}.pages.jsonl"
    result = {
        "book_id": book_id,
        "title": book["title"],
        "source_file": book["file_path"],
        "processing_status": "failed",
        "pdf_page_count": None,
        "pages_written": 0,
        "pages_read": 0,
        "output_file": None,
        "sha256": None,
        "characters": 0,
        "alphanumeric_characters": 0,
        "empty_text_pdf_pages": [],
        "low_text_pdf_pages": [],
        "image_without_text_pdf_pages": [],
        "unicode_review_pdf_pages": [],
        "fragmented_spacing_pdf_pages": [],
        "error": None,
        "failed_pdf_page": None,
        "warnings": [],
    }
    temporary = None
    current_page = None
    try:
        with source.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        result["sha256"] = checksum
        expected_checksum = book.get("sha256")
        if expected_checksum and checksum.casefold() != expected_checksum.casefold():
            raise ValueError("Source SHA256 differs from the manifest; verify the file before ingestion.")
        with pymupdf.open(source) as document:
            if not document.is_pdf:
                raise ValueError("The source does not contain a PDF document.")
            if document.needs_pass:
                raise ValueError("PDF requires a password; no password is configured.")
            if not document.page_count:
                raise ValueError("PDF contains no pages.")
            result["pdf_page_count"] = document.page_count
            if book.get("pdf_page_count") not in (None, document.page_count):
                raise ValueError("PDF page count differs from the manifest.")
            if document.is_repaired:
                result["warnings"].append("PyMuPDF repaired the document in memory; inspect source integrity.")
            output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=output_dir,
                prefix=f".{book_id}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                for page_index in range(document.page_count):
                    current_page = page_index + 1
                    page = document.load_page(page_index)
                    text = page.get_text("text", sort=True)
                    image_count = len(page.get_image_info())
                    flags = _text_flags(text, image_count, minimum_text_characters)
                    alphanumeric = sum(character.isalnum() for character in text)
                    record = {
                        "group_id": book.get("group_id") or group_id,
                        "subject": book.get("subject", "Computer Architecture"),
                        "book_id": book_id,
                        "title": book["title"],
                        "author": book.get("author", []),
                        "edition": book.get("edition"),
                        "source_file": book["file_path"],
                        "source_sha256": checksum,
                        "page_id": f"{book_id}_P{current_page:04d}",
                        "page": current_page,
                        "pdf_page": current_page,
                        "page_reference_type": "pdf_position",
                        "pdf_page_label": page.get_label() or None,
                        "printed_page": None,
                        "text": text,
                        "character_count": len(text),
                        "alphanumeric_characters": alphanumeric,
                        "image_count": image_count,
                        "quality_flags": flags,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    result["pages_read"] += 1
                    result["characters"] += len(text)
                    result["alphanumeric_characters"] += alphanumeric
                    for flag, key in (
                        ("no_text", "empty_text_pdf_pages"),
                        ("low_text", "low_text_pdf_pages"),
                        ("image_without_substantive_text", "image_without_text_pdf_pages"),
                        ("unicode_review", "unicode_review_pdf_pages"),
                        ("fragmented_word_spacing", "fragmented_spacing_pdf_pages"),
                    ):
                        if flag in flags:
                            result[key].append(current_page)
            current_page = None
            if result["alphanumeric_characters"] == 0:
                result["processing_status"] = "needs OCR"
                result["warnings"].append("No searchable text found; excluded from extracted page outputs. No OCR performed.")
            else:
                os.replace(temporary, destination)
                result["processing_status"] = "processed"
                result["pages_written"] = document.page_count
                result["output_file"] = destination.relative_to(project_root).as_posix()
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["failed_pdf_page"] = current_page
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def run_ingestion(project_root: Path, config_path: Path | None = None,
                  book_ids: list[str] | None = None) -> dict:
    """Extract only manifest sources and update processing status, not approvals."""
    root = Path(project_root).resolve()
    configuration_path = Path(config_path) if config_path is not None else root / "config/settings.yaml"
    if not configuration_path.is_absolute():
        configuration_path = root / configuration_path
    configuration_path = configuration_path.resolve()
    if not configuration_path.is_relative_to(root):
        raise ValueError("Configuration must be inside the project.")
    settings = yaml.safe_load(configuration_path.read_text(encoding="utf-8")) or {}
    sources = settings.get("sources", {})
    if sources.get("ocr_enabled", False) is not False:
        raise ValueError("OCR is disabled for this project; sources.ocr_enabled must be false.")
    paths = settings.get("paths", {})
    excluded = DEFAULT_EXCLUDED | {str(name).casefold() for name in sources.get("excluded_directories", [])}
    manifest_path = _project_path(root, sources.get("book_manifest", "data/book_manifest.json"))
    output_dir = _project_path(root, paths.get("extracted_directory", "data/extracted"))
    report_path = _project_path(root, paths.get("ingestion_report", "evaluation/ingestion_report.json"))
    if report_path == manifest_path:
        raise ValueError("The ingestion report cannot overwrite the book manifest.")
    books = _validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), root, excluded)
    selected_ids = set(book_ids) if book_ids else {book["book_id"] for book in books}
    unknown_ids = selected_ids - {book["book_id"] for book in books}
    if unknown_ids:
        raise ValueError(f"Unknown book IDs: {sorted(unknown_ids)}")
    minimum = int(settings.get("ingestion", {}).get("minimum_text_characters", 40))
    if minimum < 1:
        raise ValueError("minimum_text_characters must be positive.")
    started = time.monotonic()
    report = {
        "started_at": datetime.now(ZoneInfo("Africa/Lusaka")).isoformat(),
        "stage": "page_text_extraction",
        "extractor": {"library": "PyMuPDF", "version": pymupdf.VersionBind, "sort": True, "ocr": False},
        "manifest_file": manifest_path.relative_to(root).as_posix(),
        "page_reference_type": "pdf_position",
        "printed_page_numbers_verified": False,
        "minimum_text_characters": minimum,
        "books": [],
    }
    for book in books:
        if book["book_id"] not in selected_ids:
            continue
        print(f"Extracting {book['book_id']}: {book['title']}", flush=True)
        result = extract_book(
            book, root, output_dir, group_id=settings.get("project", {}).get("group_id"),
            minimum_text_characters=minimum,
        )
        report["books"].append(result)
        book["processing_status"] = result["processing_status"]
        book["ingestion"] = {
            "extracted_file": result["output_file"],
            "pages_written": result["pages_written"],
            "empty_text_page_count": len(result["empty_text_pdf_pages"]),
            "low_text_page_count": len(result["low_text_pdf_pages"]),
            "image_without_text_page_count": len(result["image_without_text_pdf_pages"]),
            "quality_review_status": "pending",
            "error": result["error"],
        }
        print(f"  {result['processing_status']}: {result['pages_written']} page records", flush=True)
    results = report["books"]
    report["summary"] = {
        "books_attempted": len(results),
        "books_processed": sum(item["processing_status"] == "processed" for item in results),
        "books_failed": sum(item["processing_status"] == "failed" for item in results),
        "books_needing_ocr": sum(item["processing_status"] == "needs OCR" for item in results),
        "pages_written": sum(item["pages_written"] for item in results),
        "characters": sum(item["characters"] for item in results if item["processing_status"] == "processed"),
        "empty_text_pages": sum(len(item["empty_text_pdf_pages"]) for item in results),
        "low_text_pages": sum(len(item["low_text_pdf_pages"]) for item in results),
        "image_without_text_pages": sum(len(item["image_without_text_pdf_pages"]) for item in results),
        "unicode_review_pages": sum(len(item["unicode_review_pdf_pages"]) for item in results),
        "fragmented_spacing_pages": sum(len(item["fragmented_spacing_pdf_pages"]) for item in results),
    }
    report["limitations"] = [
        "Extraction reads existing text layers only; images, figures and unencoded equations are not transcribed.",
        "PDF positions and labels are retained; printed book page numbers have not been verified.",
        "Quality flags are review heuristics and do not prove that every formula or reading order is correct.",
        "Processing status records extraction only; permission and coordinator approval flags are unchanged.",
        "Chunking, embeddings, model downloads and training have not been performed.",
    ]
    report["completed_at"] = datetime.now(ZoneInfo("Africa/Lusaka")).isoformat()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    _write_json(report_path, report)
    _write_json(manifest_path, books)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--book-id", action="append", dest="book_ids", help="Run only selected IDs; repeat for more than one.")
    arguments = parser.parse_args()
    try:
        report = run_ingestion(arguments.project_root, arguments.config, arguments.book_ids)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(2, f"Ingestion configuration error: {error}\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 1 if report["summary"]["books_failed"] or report["summary"]["books_needing_ocr"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
