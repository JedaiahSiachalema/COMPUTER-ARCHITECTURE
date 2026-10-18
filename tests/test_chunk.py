"""Regression checks using the cached tokenizer; no book outputs are written."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "chunk_under_test", Path(__file__).resolve().parents[1] / "src" / "chunk.py",
)
chunk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chunk)


@pytest.fixture(scope="module")
def runtime():
    return chunk.load_runtime()


def page(text: str, number: int = 1, **changes) -> dict:
    record = {
        "group_id": "G090", "subject": "Computer Architecture", "book_id": "G090_TEST",
        "title": "Synthetic architecture text", "author": ["Test Author"], "edition": None,
        "page": number, "page_id": f"G090_TEST_P{number:04d}",
        "source_file": "books/test.pdf", "source_sha256": "a" * 64,
        "text": text, "usable": True, "quality_flags": [], "exclusion_reasons": [],
        "review_required": False,
    }
    record.update(changes)
    return record


def test_long_paragraph_fits_and_has_overlap_without_losing_text(runtime):
    tokenizer, options = runtime
    text = ("The CPU executes instructions and reads frequently used cache memory. " * 200).strip()
    passages, skipped = chunk.build_chunks([page(text)], tokenizer, options)
    assert not skipped and len(passages) > 1
    covered = set()
    for passage in passages:
        start, end = passage["start_char"], passage["end_char"]
        assert passage["text"] == text[start:end]
        assert passage["embedding_token_count"] <= options["size_tokens"]
        assert passage["embedding_token_count"] == chunk.token_count(
            tokenizer, chunk.embedding_text(passage["text"], passage["section"], options),
        )
        covered.update(range(start, end))
    assert all(index in covered or character.isspace() for index, character in enumerate(text))
    assert all(passage["overlap_tokens"] >= options["overlap_tokens"] for passage in passages[1:])


def test_page_boundaries_and_source_metadata_are_preserved(runtime):
    tokenizer, options = runtime
    passages, _ = chunk.build_chunks([
        page("The processor executes arithmetic instructions. " * 90),
        page("Cache coherence coordinates shared memory. " * 90, 2),
    ], tokenizer, options)
    assert len({p["chunk_id"] for p in passages}) == len(passages)
    for passage in passages:
        assert passage["page"] == passage["page_end"]
        assert passage["source_spans"] == [{
            "page": passage["page"], "start_char": passage["start_char"],
            "end_char": passage["end_char"],
        }]
        assert passage["source_sha256"] == "a" * 64
    assert next(p for p in passages if p["page"] == 2)["overlap_tokens"] == 0


def test_unresolved_quality_flags_are_excluded_even_if_marked_usable(runtime):
    tokenizer, options = runtime
    records = [page("Computer architecture describes processors and memory.", n, quality_flags=[flag])
               for n, flag in enumerate(sorted(chunk.BLOCKING_FLAGS), 1)]
    records.append(page("Text requiring manual review.", 4, review_required=True))
    records.append(page("A stale record without a usability decision.", 5))
    records[-1].pop("usable")
    passages, skipped = chunk.build_chunks(records, tokenizer, options)
    assert passages == [] and len(skipped) == len(records)


def test_headings_split_context_and_prose_is_not_mistaken_for_a_heading(runtime):
    tokenizer, options = runtime
    assert chunk.heading_label("Chapter 3 is about performance. How should we characterize it") is None
    assert chunk.heading_label("1.1 Four Decades of Computing 2") is None
    assert chunk.heading_label("12.6 For the following code") is None
    assert chunk.heading_label("5.4\nMeasuring Cache Performance") == "5.4 Measuring Cache Performance"
    text = ("8.4 Cache memory\n\n" + "A cache stores recently accessed data. " * 90
            + "\n\n8.5 Pipelining\n\n" + "Pipeline stages overlap instruction execution. " * 90)
    passages, _ = chunk.build_chunks([page(text)], tokenizer, options)
    assert {p["section"] for p in passages} == {"8.4 Cache memory", "8.5 Pipelining"}
    first_new_section = next(p for p in passages if p["section"] == "8.5 Pipelining")
    assert first_new_section["text"].startswith("8.5 Pipelining")
    assert first_new_section["overlap_tokens"] == 0


def test_exact_source_spelling_and_unicode_are_retained(runtime):
    tokenizer, options = runtime
    text = "MixedCase x86_64: naïve CPU design — compare café and CAFÉ; A != B.\n\n" * 80
    passages, _ = chunk.build_chunks([page(text)], tokenizer, options)
    assert passages[0]["text"].startswith("MixedCase x86_64: naïve")
    assert all(p["text"] == text[p["start_char"]:p["end_char"]] for p in passages)


def test_prefixes_and_special_tokens_are_included_in_the_limit(runtime):
    tokenizer, options = runtime
    options = {**options, "include_section": True, "document_prefix": "Technical passage: "}
    passages, _ = chunk.build_chunks([
        page("8.4 Cache memory\n\n" + "Cache replacement selects a block to evict. " * 150),
    ], tokenizer, options)
    assert all(p["embedding_token_count"] <= options["size_tokens"] for p in passages)
    assert passages[0]["embedding_token_count"] > passages[0]["token_count"]


def test_short_tail_and_zero_overlap_do_not_drop_content(runtime):
    tokenizer, options = runtime
    options = {**options, "overlap_tokens": 0}
    text = "processor cafe\u0301 " * 400 + "FINAL"
    passages, _ = chunk.build_chunks([page(text)], tokenizer, options)
    assert passages[-1]["text"].endswith("FINAL")
    assert all(p["overlap_tokens"] == 0 for p in passages)
    covered = set()
    for passage in passages:
        covered.update(range(passage["start_char"], passage["end_char"]))
    assert all(index in covered or character.isspace() for index, character in enumerate(text))


def test_invalid_limits_and_mixed_page_metadata_are_rejected(runtime):
    tokenizer, options = runtime
    with pytest.raises(ValueError):
        chunk.validate_options({**options, "size_tokens": options["max_input_tokens"] + 1})
    with pytest.raises(ValueError):
        chunk.validate_options({**options, "overlap_tokens": options["size_tokens"]})
    with pytest.raises(ValueError):
        chunk.build_chunks([page("Text"), page("Repeated page")], tokenizer, options)
    with pytest.raises(ValueError):
        chunk.build_chunks([page("Text", group_id=None)], tokenizer, options)
