"""GPU preference and CPU fallback checks using mocked hardware/models."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import embed
import runtime_device as runtime


class FakeOOM(RuntimeError):
    pass


def torch_backend(available=True):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=Mock(return_value=available), device_count=Mock(return_value=1),
            get_device_name=Mock(return_value="Test GPU"), synchronize=Mock(),
            empty_cache=Mock(), OutOfMemoryError=FakeOOM,
        ),
        ones=Mock(return_value=Mock()),
    )


def configuration(device="auto"):
    return {"device": device, "model_name": "test-model", "revision": "pinned",
            "batch_size": 8, "query_prefix": "query: "}


def test_global_setting_and_component_override():
    settings = {"runtime": {"device": "auto"}, "embedding": {"device": "cpu"}}
    assert runtime.device_preference(settings, "embedding") == "cpu"
    assert runtime.device_preference(settings, "generation") == "auto"
    assert runtime.device_preference({}) == "auto"


def test_auto_prefers_working_gpu():
    backend = torch_backend()
    with patch.dict(sys.modules, {"torch": backend}):
        assert runtime.select_device("auto") == "cuda:0"
    backend.ones.assert_called_once_with(1, device="cuda:0")
    backend.cuda.synchronize.assert_called_once_with("cuda:0")


def test_unavailable_gpu_uses_cpu_and_cpu_override_avoids_gpu():
    backend = torch_backend(False)
    with patch.dict(sys.modules, {"torch": backend}):
        assert runtime.select_device("auto") == "cpu"
    backend.ones.assert_not_called()
    backend.cuda.is_available.reset_mock()
    with patch.dict(sys.modules, {"torch": backend}):
        assert runtime.select_device("cpu") == "cpu"
    backend.cuda.is_available.assert_not_called()


def test_detected_but_unusable_gpu_falls_back():
    backend = torch_backend()
    backend.ones.side_effect = RuntimeError("no kernel image is available for execution")
    with patch.dict(sys.modules, {"torch": backend}):
        assert runtime.select_device("auto") == "cpu"
    backend.cuda.empty_cache.assert_called_once()
    backend = torch_backend()
    backend.cuda.is_available.side_effect = RuntimeError("CUDA driver initialization failed")
    with patch.dict(sys.modules, {"torch": backend}):
        assert runtime.select_device("auto") == "cpu"


def test_invalid_setting_rejected_and_missing_gpu_index_falls_back():
    with pytest.raises(ValueError):
        runtime.select_device("invalid")
    with patch.dict(sys.modules, {"torch": torch_backend()}):
        assert runtime.select_device("cuda:9") == "cpu"


def test_loading_retries_once_on_cpu_after_gpu_oom():
    constructor = Mock(side_effect=[FakeOOM("CUDA out of memory"), "cpu-model"])
    config = configuration()
    with patch.dict(sys.modules, {
        "torch": torch_backend(),
        "sentence_transformers": SimpleNamespace(SentenceTransformer=constructor),
    }):
        assert embed.load_model(config) == "cpu-model"
    assert config["requested_device"] == "auto" and config["device"] == "cpu"
    assert [call.kwargs["device"] for call in constructor.call_args_list] == ["cuda:0", "cpu"]


def test_encoding_oom_moves_to_cpu_and_subsequent_queries_stay_there():
    vectors = np.ones((1, 384), dtype="float32")
    model = Mock()
    model.encode.side_effect = [FakeOOM("CUDA out of memory"), vectors, vectors]
    config = configuration("cuda:0")
    with patch.dict(sys.modules, {"torch": torch_backend()}):
        assert np.array_equal(embed.encode_texts(model, ["A passage"], config), vectors)
        assert np.array_equal(embed.embed_queries(model, ["What is a cache?"], config), vectors)
    model.to.assert_called_once_with("cpu")
    assert config["device"] == "cpu"
    assert [call.kwargs["device"] for call in model.encode.call_args_list] == ["cuda:0", "cpu", "cpu"]
    assert model.encode.call_args_list[-1].args[0] == ["query: What is a cache?"]


def test_unrelated_model_errors_and_device_assertions_are_not_hidden():
    with patch.dict(sys.modules, {"torch": torch_backend()}):
        assert not runtime.is_cuda_resource_error(RuntimeError("CUDA error: device-side assert triggered"))
        for device in ("cpu", "cuda:0"):
            model = Mock()
            model.encode.side_effect = RuntimeError("Invalid token shape")
            with pytest.raises(RuntimeError, match="Invalid token shape"):
                embed.encode_texts(model, ["Text"], configuration(device))
            model.to.assert_not_called()
