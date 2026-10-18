"""Billy's retrieval-contract tests with synthetic vectors and no model/dataset writes."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock, patch

import faiss
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import retrieve as retrieval


class Encoder:
    max_seq_length = 512

    def __init__(self, vector=None):
        self.vector = np.array([[1.0, 0.0]] if vector is None else vector, dtype="float32")
        self.last_texts = None

    def tokenizer(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()) + 2))}

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        self.last_texts = texts
        return self.vector


def bundle():
    passages = []
    for n, text in enumerate(("A cache stores recently used data.", "Pipeline stages overlap instructions."), 1):
        reference = {"book_id": f"BOOK_{n}", "title": f"Book {n}", "page": n,
                     "chunk_id": f"BOOK_{n}_P{n:04d}_C01"}
        passages.append({**reference, "page_end": n, "text": text, "sources": [dict(reference)]})
    metadata = {
        "embedding": {"model_name": "saved-model", "revision": "saved-revision", "dimension": 2,
                      "normalize": True, "query_prefix": "saved-prefix: ",
                      "document_prefix": "", "include_section": False},
        "index": {"passages": 2},
    }
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    return index, passages, metadata


def engine():
    index, passages, metadata = bundle()
    value = retrieval.Retriever.__new__(retrieval.Retriever)
    value.index, value.passages, value.metadata = index, passages, metadata
    value.config = {**metadata["embedding"], "batch_size": 1, "device": "cpu"}
    value._model, value._lock = Encoder(), RLock()
    return value


def test_invalid_requests_fail_before_loading_any_model():
    for question in (None, 4, [], "", " \n\t"):
        with pytest.raises((TypeError, ValueError)):
            retrieval.retrieve(question)
    for count in (0, -1, True, 1.5, "5"):
        with pytest.raises(ValueError):
            retrieval.retrieve("What is a cache?", count)


def test_ranked_results_follow_the_required_interface_and_clamp_top_k():
    value = engine()
    results = value.search("What is a cache?", 20)
    assert len(results) == 2
    assert results[0]["book_id"] == "BOOK_1"
    assert [r["score"] for r in results] == [1.0, 0.0]
    for rank, result in enumerate(results, 1):
        assert {"text", "score", "book_id", "title", "page", "chunk_id"} <= result.keys()
        assert result["rank"] == rank and isinstance(result["score"], float)


def test_queries_use_the_saved_index_prefix():
    value = engine()
    value.search("  What is a cache?  ")
    assert value._model.last_texts == ["saved-prefix: What is a cache?"]


def test_all_duplicate_sources_are_returned_and_caller_mutation_is_isolated():
    value = engine()
    extra = {"book_id": "OTHER_BOOK", "title": "Other edition", "page": 55, "chunk_id": "OTHER_P0055_C01"}
    value.passages[0]["sources"].append(extra)
    original = deepcopy(value.passages)
    result = value.search("cache", 1)[0]
    assert len(result["sources"]) == 2
    result["text"] = "Modified by caller"
    result["sources"][1]["page"] = 999
    assert value.passages == original


def test_oversized_questions_are_rejected_without_silent_truncation():
    value = engine()
    value._model.max_seq_length = 5
    with pytest.raises(ValueError, match="token limit"):
        value.search("Please explain cache replacement policy in detail")
    assert value._model.last_texts is None


def test_bad_query_vectors_are_rejected():
    for vector in ([[1.0, 0.0, 0.0]], [[float("nan"), 0.0]], [[2.0, 0.0]]):
        value = engine()
        value._model = Encoder(vector)
        with pytest.raises(ValueError):
            value.search("cache")


def test_invalid_faiss_rows_cannot_resolve_to_the_last_passage():
    value = engine()
    value.index = SimpleNamespace(d=2, search=Mock(return_value=(
        np.array([[1.0]], dtype="float32"), np.array([[-1]], dtype="int64"),
    )))
    with pytest.raises(ValueError, match="invalid passage row"):
        value.search("cache", 1)


def test_bundle_count_dimension_and_source_integrity_checks():
    index, passages, metadata = bundle()
    retrieval.validate_bundle(index, passages, metadata)
    broken = deepcopy(metadata)
    broken["index"]["passages"] = 3
    with pytest.raises(ValueError):
        retrieval.validate_bundle(index, passages, broken)
    broken = deepcopy(metadata)
    broken["embedding"]["dimension"] = 3
    with pytest.raises(ValueError):
        retrieval.validate_bundle(index, passages, broken)
    broken_passages = deepcopy(passages)
    broken_passages[0]["sources"] = []
    with pytest.raises(ValueError):
        retrieval.validate_bundle(index, broken_passages, metadata)
    broken_passages = deepcopy(passages)
    broken_passages[1]["chunk_id"] = passages[0]["chunk_id"]
    with pytest.raises(ValueError):
        retrieval.validate_bundle(index, broken_passages, metadata)


def test_changed_artifact_hashes_fail_before_reading_index_or_loading_model():
    root = Path(__file__).resolve().parents[1]
    directory = root / "artifacts/index"
    _, _, metadata = bundle()
    metadata["index"].update({"sha256": "index-hash", "passages_sha256": "passage-hash"})
    paths = (directory, directory / "index_config.json", directory / "index.faiss", root / "data/passages.jsonl", metadata)
    with patch.object(Path, "read_text", return_value="{}"), \
            patch.object(retrieval, "artifact_paths", return_value=paths), \
            patch.object(retrieval, "file_hash", return_value="changed"), \
            patch.object(retrieval.faiss, "read_index") as read_index, \
            patch.object(retrieval, "load_model") as load_model:
        with pytest.raises(ValueError, match="changed after embedding"):
            retrieval.Retriever(root)
        read_index.assert_not_called()
        load_model.assert_not_called()


def test_legacy_passage_reordering_is_rejected_even_when_counts_match():
    root = Path(__file__).resolve().parents[1]
    index, passages, metadata = bundle()
    metadata["corpus_preparation"] = {"report": "evaluation/corpus_preparation_report.json", "report_sha256": "same"}
    report = {"outputs": [{"path": "data/embedding/BOOK_1.chunks.jsonl", "sha256": "same", "book_id": "BOOK_1"}]}

    def read_text(path, **kwargs):
        if path.name == "corpus_preparation_report.json":
            return json.dumps(report)
        return "\n".join(json.dumps(p) for p in passages)

    with patch.object(Path, "read_text", read_text), patch.object(retrieval, "file_hash", return_value="same"):
        with pytest.raises(ValueError, match="Passage order/content"):
            retrieval.verify_legacy_order(root, root / "artifacts/index", index, list(reversed(passages)), metadata)
