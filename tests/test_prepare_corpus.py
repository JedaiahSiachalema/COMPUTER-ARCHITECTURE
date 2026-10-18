"""Corpus filtering/deduplication regressions; no model or dataset writes required."""
from __future__ import annotations

from copy import deepcopy
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare_corpus as prep

OPTIONS = {"include_section": False, "document_prefix": ""}


def policy():
    return {
        "books": {
            "BOOK_A": {"excluded_ranges": [{"start": 1, "end": 3, "reason": "Publisher pages"}]},
            "BOOK_B": {"excluded_ranges": []},
        },
    }


def passage(book="BOOK_A", number=4, text="A cache stores recently used data.", **changes):
    record = {
        "book_id": book, "page": number, "page_end": number,
        "chunk_id": f"{book}_P{number:04d}_C01", "text": text,
        "usable": True, "quality_flags": [], "review_required": False,
        "exclusion_reasons": [], "section": None,
        "start_char": 5, "end_char": 5 + len(text),
        "source_spans": [{"page": number, "start_char": 5, "end_char": 5 + len(text)}],
    }
    record.update(changes)
    return record


def test_explicit_pages_excluded_but_technical_preface_kept():
    records = [passage(number=2, text="Copyright and publisher information."),
               passage(number=4, text="A technical preface can mention copyright without being excluded.")]
    result, excluded = prep.build_corpus(records, policy(), OPTIONS)
    assert len(result) == len(excluded) == 1
    assert result[0]["page"] == 4
    assert excluded[0]["reason"] == "Publisher pages"


def test_duplicates_keep_both_citations_and_original_records():
    records = [passage(), passage("BOOK_B", 18)]
    before = deepcopy(records)
    result, excluded = prep.build_corpus(records, policy(), OPTIONS)
    assert not excluded and len(result) == 1
    assert result[0]["source_count"] == 2
    assert {source["chunk_id"] for source in result[0]["sources"]} == {r["chunk_id"] for r in records}
    assert {source["page"] for source in result[0]["sources"]} == {4, 18}
    assert all(source["source_spans"] for source in result[0]["sources"])
    assert records == before


def test_case_punctuation_and_whitespace_differences_stay_separate():
    records = [passage(number=i, text=text) for i, text in enumerate([
        "CPU reads memory.", "cpu reads memory.", "CPU  reads memory.", "CPU reads memory!",
    ], 4)]
    result, _ = prep.build_corpus(records, policy(), OPTIONS)
    assert len(result) == 4


def test_section_prefix_differences_are_not_merged():
    records = [passage(section="Cache"), passage("BOOK_B", 8, section="Memory")]
    result, _ = prep.build_corpus(records, policy(), {**OPTIONS, "include_section": True})
    assert len(result) == 2


def test_unresolved_quality_is_excluded_and_sources_balance():
    records = [passage(number=4, quality_flags=["unicode_review"]),
               passage(number=5, review_required=True), passage(number=6)]
    result, excluded = prep.build_corpus(records, policy(), OPTIONS)
    assert len(result) == 1 and len(excluded) == 2
    assert len(records) == sum(r["source_count"] for r in result) + len(excluded)


def test_duplicate_ids_and_old_multipage_chunks_are_rejected():
    with pytest.raises(ValueError):
        prep.build_corpus([passage(), passage()], policy(), OPTIONS)
    with pytest.raises(ValueError):
        prep.build_corpus([passage(page_end=5)], policy(), OPTIONS)


def test_policy_is_bound_to_source_and_page_positions():
    books = [{"book_id": "BOOK_A", "sha256": "abc", "pdf_page_count": 30}]
    filters = {"version": 1, "page_reference_type": "pdf_position", "books": {
        "BOOK_A": {"source_sha256": "abc", "pdf_page_count": 30,
                   "excluded_ranges": [{"start": 1, "end": 3, "reason": "Title pages"}]},
    }}
    prep.validate_policy(filters, books)
    wrong_source = deepcopy(filters)
    wrong_source["books"]["BOOK_A"]["source_sha256"] = "changed"
    with pytest.raises(ValueError):
        prep.validate_policy(wrong_source, books)
    overlapping = deepcopy(filters)
    overlapping["books"]["BOOK_A"]["excluded_ranges"].append({"start": 3, "end": 5, "reason": "Overlap"})
    with pytest.raises(ValueError):
        prep.validate_policy(overlapping, books)


def test_embedding_rejects_changed_preparation_inputs_and_outputs():
    root = Path(__file__).resolve().parents[1]
    settings_path = root / "config/settings.yaml"
    settings = {"embedding": {"model_name": "test-model", "revision": "pinned"}}
    report = {
        "schema_version": 1, "embedding_input": prep.embedding_settings(settings),
        "manifest_path": "data/book_manifest.json", "manifest_sha256": "manifest",
        "policy_path": prep.POLICY, "policy_sha256": "policy",
        "chunk_report_path": "evaluation/chunking_report.json", "chunk_report_sha256": "chunk_report",
        "inputs": [{"path": "data/chunks/BOOK_A.chunks.jsonl", "sha256": "chunks",
                    "cleaned_path": "data/cleaned/BOOK_A.pages.jsonl", "cleaned_sha256": "cleaned"}],
        "outputs": [{"path": "data/embedding/BOOK_A.chunks.jsonl", "sha256": "prepared"}],
    }
    hashes = {
        report["manifest_path"]: "manifest", prep.POLICY: "policy",
        report["chunk_report_path"]: "chunk_report",
        report["inputs"][0]["path"]: "chunks", report["inputs"][0]["cleaned_path"]: "cleaned",
        report["outputs"][0]["path"]: "prepared",
    }

    def read_text(path, *args, **kwargs):
        return json.dumps(settings if path == settings_path else report)

    def checksum(path):
        return hashes[path.relative_to(root).as_posix()]

    with patch.object(Path, "is_file", return_value=True), patch.object(Path, "read_text", read_text), \
            patch.object(prep, "sha256", checksum):
        files, _ = prep.load_prepared_sources(root, settings_path)
        assert files == [root / report["outputs"][0]["path"]]
        for name in list(hashes):
            original = hashes[name]
            hashes[name] = "changed"
            with pytest.raises(ValueError):
                prep.load_prepared_sources(root, settings_path)
            hashes[name] = original
