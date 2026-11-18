from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from oaa.backends.llama_cpp import LlamaCppBackend
from oaa.config import GenerationConfig


class FakeLlama:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.metadata = {
            "general.architecture": "qwen2",
            "general.file_type": 15,
        }
        self.requests = []

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs["stream"]:
            return iter([
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " from GGUF"}}]},
            ])
        return {"choices": [{"message": {"content": "Hello from a local checkpoint"}}]}


def load_fake(monkeypatch, tmp_path: Path):
    model = tmp_path / "local-model.gguf"
    model.write_bytes(b"test fixture - not a real GGUF checkpoint")
    fake_module = SimpleNamespace(Llama=FakeLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    backend = LlamaCppBackend(
        model, n_ctx=2048, n_gpu_layers=0, n_threads=2, chat_format="chatml"
    )
    return backend, model


def test_gguf_backend_loads_only_an_explicit_local_file(monkeypatch, tmp_path):
    backend, model = load_fake(monkeypatch, tmp_path)
    assert backend.capabilities.backend_name == "llama_cpp_python"
    assert backend.capabilities.checkpoint_loaded is True
    assert backend.capabilities.pretrained_weights_loaded is False
    assert backend.capabilities.true_streaming is True
    assert backend.capabilities.structured_tool_calls is False
    assert backend.model_info["model_format"] == "GGUF"
    assert backend.model_info["architecture"] == "qwen2"
    assert backend.model_info["model_path"] == str(model.resolve())


def test_gguf_backend_chat_maps_generation_parameters(monkeypatch, tmp_path):
    backend, _ = load_fake(monkeypatch, tmp_path)
    result = backend.chat(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Say hello."},
        ],
        GenerationConfig(
            max_tokens=24, temperature=0.4, top_k=20, top_p=0.9,
            repetition_penalty=1.1, seed=7,
        ),
    )
    assert result == "Hello from a local checkpoint"
    request = backend._model.requests[-1]
    assert request["max_tokens"] == 24
    assert request["temperature"] == 0.4
    assert request["top_k"] == 20
    assert request["top_p"] == 0.9
    assert request["repeat_penalty"] == 1.1
    assert request["seed"] == 7
    assert request["stream"] is False


def test_gguf_backend_streaming_skips_role_only_chunks(monkeypatch, tmp_path):
    backend, _ = load_fake(monkeypatch, tmp_path)
    chunks = list(backend.chat_stream(
        [{"role": "user", "content": "Say hello"}],
        GenerationConfig(max_tokens=12),
    ))
    assert chunks == ["Hello", " from GGUF"]
    assert backend._model.requests[-1]["stream"] is True


def test_gguf_backend_validates_model_path_and_settings(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LlamaCppBackend(tmp_path / "missing.gguf")
    not_gguf = tmp_path / "model.bin"
    not_gguf.write_bytes(b"not gguf")
    with pytest.raises(ValueError, match="local .gguf"):
        LlamaCppBackend(not_gguf)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    with pytest.raises(ValueError, match="n_ctx"):
        LlamaCppBackend(model, n_ctx=0)
    with pytest.raises(ValueError, match="n_gpu_layers"):
        LlamaCppBackend(model, n_gpu_layers=-2)
    with pytest.raises(ValueError, match="n_threads"):
        LlamaCppBackend(model, n_threads=0)


def test_gguf_backend_rejects_untrusted_malformed_messages(monkeypatch, tmp_path):
    backend, _ = load_fake(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="non-empty list"):
        backend.chat([])
    with pytest.raises(ValueError, match="unsupported role"):
        backend.chat([{"role": "tool", "content": "do this"}])
    with pytest.raises(ValueError, match="only role and content"):
        backend.chat([{"role": "user", "content": "hello", "tool_call": {"name": "shell"}}])
    with pytest.raises(ValueError, match="non-empty string"):
        backend.chat([{"role": "user", "content": "  "}])
