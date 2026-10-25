from __future__ import annotations

import pytest

from oaa.backend import BackendCapabilities, GenerationService
from oaa.config import GenerationConfig


class FakeBackend:
    capabilities = BackendCapabilities(
        backend_name="fake", model_family="test", modalities=("text",),
        pretrained_weights_loaded=True, true_streaming=True,
        structured_tool_calls=True, quantization="test", max_context_tokens=100,
    )

    def __init__(self, output="answer"):
        self.output = output
        self.calls = []

    def generate(self, prompt, config):
        self.calls.append((prompt, config))
        return self.output


def test_generation_service_returns_text_and_privacy_minimizing_metrics():
    backend = FakeBackend()
    service = GenerationService(backend, max_prompt_chars=20, max_output_tokens=10)
    result = service.generate("private prompt", GenerationConfig(max_tokens=3))
    assert result.text == "answer"
    assert result.output_characters == 6
    assert result.backend_name == "fake"
    assert result.elapsed_seconds >= 0
    assert "private prompt" not in str(result.metrics())
    assert "answer" not in str(result.metrics())


def test_generation_service_enforces_prompt_token_and_output_budgets():
    backend = FakeBackend()
    service = GenerationService(backend, max_prompt_chars=3, max_output_tokens=4)
    with pytest.raises(ValueError, match="max_prompt_chars"):
        service.generate("too long")
    service = GenerationService(backend, max_output_tokens=2)
    with pytest.raises(ValueError, match="service limit"):
        service.generate("ok", GenerationConfig(max_tokens=3))
    service = GenerationService(FakeBackend("toolong"), max_output_chars=3)
    with pytest.raises(ValueError, match="max_output_chars"):
        service.generate("ok")


def test_generation_service_validates_backend_and_capabilities():
    with pytest.raises(TypeError, match="BackendCapabilities"):
        GenerationService(object())
    with pytest.raises(ValueError, match="modalities"):
        BackendCapabilities("test", "test", modalities=("text", "text"))
    with pytest.raises(ValueError, match="max_context_tokens"):
        BackendCapabilities("test", "test", max_context_tokens=0)
