"""Exercise ingestion with tiny generated PDFs, without reading the book collection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import fitz
import pytest
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "ingest.py"
MODULE_SPEC = importlib.util.spec_from_file_location("ingest_under_test", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
ingest = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = ingest
MODULE_SPEC.loader.exec_module(ingest)


PAGE_TEXT = "\n".join(
    [
        "Computer Architecture Test",
        "A processor executes machine instructions.",
        "A cache keeps frequently used memory blocks.",
        "Pipeline stages can overlap instruction work.",
        "A dependency may cause a pipeline hazard.",
        "The control unit coordinates the datapath.",
    ]
    * 3
)


def make_pdf(path: Path, kinds: tuple[str, ...] = ("text",), *, labels=False, encrypted=False):
    """Make text/blank/image fixtures; image fixtures contain no text layer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as document:
        for kind in kinds:
            page = document.new_page(width=300, height=300)
            if kind == "text":
                page.insert_text((20, 25), PAGE_TEXT, fontsize=8, lineheight=1.1)
            elif kind == "image":
                pixels = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)
                pixels.clear_with(0)
                page.insert_image(fitz.Rect(30, 30, 200, 200), stream=pixels.tobytes("png"))
            elif kind != "blank":
                raise AssertionError(f"Unknown fixture page kind: {kind}")
        if labels:
            document.set_page_labels(
                [{"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1}]
            )
        save_options = {}
        if encrypted:
            save_options.update(
                encryption=fitz.PDF_ENCRYPT_AES_256,
                owner_pw="fixture-owner-password",
                user_pw="fixture-reader-password",
            )
        document.save(path, **save_options)


def manifest_entry(root: Path, source: Path, book_id="CA_TEST_001"):
    return {
        "book_id": book_id,
        "group_id": None,
        "subject": "Computer Architecture",
        "title": "Computer Architecture — Synthetic Test",
        "author": ["Fixture Author One", "Fixture Author Two"],
        "edition": "First edition",
        "publisher": "Synthetic fixtures only",
        "year": 2026,
        "isbn": None,
        "file_path": str(source.relative_to(root)),
        "source_type": "PDF text",
        "permission_status": "unverified",
        "permission_basis": None,
        "approval_status": "pending",
        "processing_status": "pending",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else "0" * 64,
        "notes": ["Retain this pre-existing manifest note."],
    }


def write_project(root: Path, entries: list[dict], *, custom_paths=False):
    manifest_name = "data/custom_manifest.json" if custom_paths else "data/book_manifest.json"
    config_name = "config/custom.yaml" if custom_paths else "config/settings.yaml"
    config = {
        "sources": {
            "book_manifest": manifest_name,
            "excluded_directories": ["ignored", "output", "tmp"],
            "ocr_enabled": False,
        },
        "paths": {
            "extracted_directory": "data/custom_pages" if custom_paths else "data/extracted",
            "ingestion_report": "evaluation/custom.json" if custom_paths else "evaluation/ingestion_report.json",
        },
    }
    manifest_path = root / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path = root / config_name
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, manifest_path


def read_records(root: Path, book_report: dict):
    output = Path(book_report["output_file"])
    assert not output.is_absolute(), "Report output paths should be relative to the project."
    path = root / output
    assert path.resolve().is_relative_to(root.resolve())
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(autouse=True)
def forbid_ocr(monkeypatch):
    def unexpected_ocr(*args, **kwargs):
        pytest.fail("Ingestion must never invoke OCR, including for image-only PDFs.")

    monkeypatch.setattr(fitz.Page, "get_textpage_ocr", unexpected_ocr)


def test_extracts_text_blank_and_image_pages_with_traceable_metadata(tmp_path):
    source = tmp_path / "selected.pdf"
    make_pdf(source, ("text", "blank", "image"), labels=True)
    entry = manifest_entry(tmp_path, source)
    write_project(tmp_path, [entry])
    with fitz.open(source) as document:
        expected_text = document[0].get_text("text")

    report = ingest.run_ingestion(tmp_path)

    assert len(report["books"]) == 1
    book = report["books"][0]
    assert book["processing_status"] == "processed"
    assert book["pdf_page_count"] == book["pages_written"] == 3
    assert book["error"] is None
    assert book["empty_text_pdf_pages"] == [2, 3]
    assert book["image_without_text_pdf_pages"] == [3]
    records = read_records(tmp_path, book)
    assert len(records) == 3
    assert records[0]["text"].splitlines() == expected_text.splitlines()
    assert records[1]["text"].strip() == records[2]["text"].strip() == ""
    assert [record["pdf_page_label"] for record in records] == ["i", "ii", "iii"]
    for position, record in enumerate(records, start=1):
        for name in ("group_id", "subject", "book_id", "title", "author", "edition"):
            assert record[name] == entry[name]
        assert record["page_id"] == f"CA_TEST_001_P{position:04d}"
        assert record["page"] == record["pdf_page"] == position
        assert record["page_reference_type"] == "pdf_position"
        assert record["printed_page"] is None
        assert isinstance(record["quality_flags"], list)
    assert records[1]["quality_flags"]
    assert records[2]["quality_flags"]
    assert report["summary"]["books_processed"] == 1
    assert report["summary"]["pages_written"] == 3
    saved = json.loads((tmp_path / "evaluation/ingestion_report.json").read_text(encoding="utf-8"))
    assert saved["summary"] == report["summary"]


def test_only_manifest_sources_are_ingested_and_approvals_are_preserved(tmp_path):
    source = tmp_path / "selected.pdf"
    make_pdf(source)
    make_pdf(tmp_path / "unlisted.pdf")
    make_pdf(tmp_path / "ignored" / "excluded.pdf")
    original = manifest_entry(tmp_path, source)
    _, manifest_path = write_project(tmp_path, [original])
    original_pdf_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    report = ingest.run_ingestion(tmp_path)

    assert [book["book_id"] for book in report["books"]] == [original["book_id"]]
    assert len(list((tmp_path / "data/extracted").glob("*.pages.jsonl"))) == 1
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))[0]
    for name, value in original.items():
        if name != "processing_status":
            assert updated[name] == value, f"Ingestion changed unrelated manifest field {name}."
    assert updated["processing_status"] == "processed"
    assert isinstance(updated["ingestion"], dict)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_pdf_hash


def test_image_only_book_is_flagged_for_ocr_without_moving_or_publishing_it(tmp_path):
    source = tmp_path / "image_only.pdf"
    make_pdf(source, ("image", "image"))
    write_project(tmp_path, [manifest_entry(tmp_path, source)])

    report = ingest.run_ingestion(tmp_path)

    book = report["books"][0]
    assert book["processing_status"] == "needs OCR"
    assert book["pdf_page_count"] == 2
    assert book["pages_written"] == 0
    assert book["output_file"] is None
    assert book["empty_text_pdf_pages"] == [1, 2]
    assert book["image_without_text_pdf_pages"] == [1, 2]
    assert report["summary"]["books_needing_ocr"] == 1
    assert source.exists()
    assert not (tmp_path / "data/extracted/CA_TEST_001.pages.jsonl").exists()


@pytest.mark.parametrize("directory", ["ignored", "IGNORED", "output", "tmp"])
def test_preflight_rejects_manifest_sources_in_excluded_directories(tmp_path, directory):
    source = tmp_path / directory / "excluded.pdf"
    make_pdf(source)
    write_project(tmp_path, [manifest_entry(tmp_path, source)])

    with pytest.raises(ValueError):
        ingest.run_ingestion(tmp_path)

    assert not list(tmp_path.glob("data/extracted/*.pages.jsonl"))


def test_preflight_rejects_source_path_traversal(tmp_path):
    root = tmp_path / "project"
    outside = tmp_path / "outside.pdf"
    make_pdf(outside)
    entry = manifest_entry(tmp_path, outside)
    entry["file_path"] = "../outside.pdf"
    write_project(root, [entry])

    with pytest.raises(ValueError):
        ingest.run_ingestion(root)

    assert not list(root.glob("data/extracted/*.pages.jsonl"))


def test_preflight_rejects_a_symlink_escaping_the_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.pdf"
    make_pdf(outside)
    link = root / "linked.pdf"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("This platform/account cannot create a test symlink.")
    entry = manifest_entry(root, link)
    write_project(root, [entry])

    with pytest.raises(ValueError):
        ingest.run_ingestion(root)


@pytest.mark.parametrize("duplicate", ["book_id", "source_path"])
def test_preflight_rejects_duplicate_ids_and_normalized_source_paths(tmp_path, duplicate):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    make_pdf(first)
    make_pdf(second)
    entries = [manifest_entry(tmp_path, first), manifest_entry(tmp_path, second, "CA_TEST_002")]
    if duplicate == "book_id":
        entries[1]["book_id"] = entries[0]["book_id"]
    else:
        entries[1]["file_path"] = "./first.pdf"
        entries[1]["sha256"] = entries[0]["sha256"]
    write_project(tmp_path, entries)

    with pytest.raises(ValueError):
        ingest.run_ingestion(tmp_path)

    assert not list(tmp_path.glob("data/extracted/*.pages.jsonl"))


@pytest.mark.parametrize("failure", ["missing", "corrupt", "sha_mismatch", "encrypted"])
def test_bad_source_is_reported_and_other_books_continue(tmp_path, failure):
    bad = tmp_path / "bad.pdf"
    if failure == "corrupt":
        bad.write_bytes(b"This is deliberately not a PDF.")
    elif failure != "missing":
        make_pdf(bad, encrypted=failure == "encrypted")
    good = tmp_path / "good.pdf"
    make_pdf(good)
    entries = [manifest_entry(tmp_path, bad), manifest_entry(tmp_path, good, "CA_TEST_002")]
    if failure == "sha_mismatch":
        entries[0]["sha256"] = "0" * 64
    write_project(tmp_path, entries)

    report = ingest.run_ingestion(tmp_path)

    books = {book["book_id"]: book for book in report["books"]}
    failed = books["CA_TEST_001"]
    assert failed["processing_status"] == "failed"
    assert failed["pages_written"] == 0
    assert failed["output_file"] is None
    assert isinstance(failed["error"], str) and failed["error"]
    assert not (tmp_path / "data/extracted/CA_TEST_001.pages.jsonl").exists()
    assert books["CA_TEST_002"]["processing_status"] == "processed"
    assert len(read_records(tmp_path, books["CA_TEST_002"])) == 1
    assert report["summary"]["books_failed"] == 1
    assert report["summary"]["books_processed"] == 1
    assert report["summary"]["pages_written"] == 1


def test_mid_book_failure_does_not_publish_a_partial_output(tmp_path, monkeypatch):
    source = tmp_path / "three_pages.pdf"
    make_pdf(source, ("text", "text", "text"))
    write_project(tmp_path, [manifest_entry(tmp_path, source)])
    original_get_text = fitz.Page.get_text
    failures = []

    def fail_on_second_page(page, *args, **kwargs):
        if page.number == 1:
            failures.append(page.number)
            raise RuntimeError("Synthetic failure on PDF page 2")
        return original_get_text(page, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_text", fail_on_second_page)

    report = ingest.run_ingestion(tmp_path)

    assert failures, "The synthetic page extraction failure must actually be exercised."
    book = report["books"][0]
    assert book["processing_status"] == "failed"
    assert book["output_file"] is None
    assert book["pages_written"] == 0
    assert "Synthetic failure" in book["error"]
    assert not (tmp_path / "data/extracted/CA_TEST_001.pages.jsonl").exists()


def test_repeated_ingestion_has_identical_page_records(tmp_path):
    source = tmp_path / "selected.pdf"
    make_pdf(source, ("text", "blank"), labels=True)
    write_project(tmp_path, [manifest_entry(tmp_path, source)])

    first = ingest.run_ingestion(tmp_path)
    first_path = tmp_path / first["books"][0]["output_file"]
    first_bytes = first_path.read_bytes()
    second = ingest.run_ingestion(tmp_path)
    second_path = tmp_path / second["books"][0]["output_file"]

    assert second_path == first_path
    assert second_path.read_bytes() == first_bytes
    assert len(second["books"]) == 1
    assert second["summary"]["pages_written"] == 2


def test_explicit_config_controls_manifest_and_output_locations(tmp_path):
    source = tmp_path / "selected.pdf"
    make_pdf(source)
    config_path, _ = write_project(tmp_path, [manifest_entry(tmp_path, source)], custom_paths=True)

    report = ingest.run_ingestion(tmp_path, config_path=config_path)

    output = tmp_path / report["books"][0]["output_file"]
    assert output.parent == tmp_path / "data/custom_pages"
    assert len(read_records(tmp_path, report["books"][0])) == 1
    assert (tmp_path / "evaluation/custom.json").is_file()
    assert not (tmp_path / "evaluation/ingestion_report.json").exists()


def test_image_pages_with_only_watermark_text_still_require_ocr(tmp_path):
    source = tmp_path / "watermarked_images.pdf"
    make_pdf(source, ("image", "image"))
    with fitz.open(source) as document:
        for position, page in enumerate(document, start=1):
            page.insert_text((20, 270), f"Sample {position}", fontsize=8)
        document.saveIncr()
    write_project(tmp_path, [manifest_entry(tmp_path, source)])

    report = ingest.run_ingestion(tmp_path)

    book = report["books"][0]
    assert book["processing_status"] == "needs OCR"
    assert book["pdf_page_count"] == 2
    assert book["empty_text_pdf_pages"] == [], "These pages have text, but it is only a watermark."
    assert book["low_text_pdf_pages"] == [1, 2]
    assert book["image_without_text_pdf_pages"] == [1, 2]
    assert book["output_file"] is None
    assert book["pages_written"] == 0
    assert report["summary"]["books_needing_ocr"] == 1
    assert not (tmp_path / "data/extracted/CA_TEST_001.pages.jsonl").exists()


def test_preflight_rejects_lexically_excluded_symlink_to_an_allowed_source(tmp_path):
    source = tmp_path / "selected.pdf"
    make_pdf(source)
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    link = ignored / "link.pdf"
    try:
        link.symlink_to(source)
    except (OSError, NotImplementedError):
        pytest.skip("This platform/account cannot create a test symlink.")
    write_project(tmp_path, [manifest_entry(tmp_path, link)])

    with pytest.raises(ValueError):
        ingest.run_ingestion(tmp_path)

    assert not list(tmp_path.glob("data/extracted/*.pages.jsonl"))


def test_report_destination_cannot_overwrite_a_source_pdf(tmp_path):
    source = tmp_path / "selected.pdf"
    make_pdf(source)
    config_path, _ = write_project(tmp_path, [manifest_entry(tmp_path, source)])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paths"]["ingestion_report"] = "selected.pdf"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(ValueError):
        ingest.run_ingestion(tmp_path)

    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
