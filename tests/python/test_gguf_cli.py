from __future__ import annotations

from oaa.cli import build_parser, main


class FakeGGUFBackend:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.messages_seen = []

    def chat(self, messages, config):
        self.messages_seen.append(list(messages))
        return "local GGUF answer"

    def chat_stream(self, messages, config):
        self.messages_seen.append(list(messages))
        yield "streamed "
        yield "response"


def test_gguf_ask_cli_uses_local_model_backend(monkeypatch, capsys):
    monkeypatch.setattr("oaa.backends.llama_cpp.LlamaCppBackend", FakeGGUFBackend)
    assert main([
        "gguf-ask", "--model", "C:/models/local.gguf",
        "--system", "Be helpful", "--max-tokens", "10", "Hello",
    ]) == 0
    out = capsys.readouterr().out
    assert "local GGUF answer" in out


def test_gguf_chat_cli_streams_and_handles_reset_exit(monkeypatch, capsys):
    monkeypatch.setattr("oaa.backends.llama_cpp.LlamaCppBackend", FakeGGUFBackend)
    commands = iter(["hello", "/reset", "second prompt", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert main([
        "gguf-chat", "--model", "C:/models/local.gguf",
        "--max-tokens", "8", "--max-history-messages", "4",
    ]) == 0
    out = capsys.readouterr().out
    assert "streamed response" in out
    assert "session reset" in out


def test_gguf_info_cli_reports_backend_capabilities(monkeypatch, capsys):
    class InfoBackend(FakeGGUFBackend):
        @property
        def model_info(self):
            return {
                "model_format": "GGUF",
                "architecture": "qwen2",
                "capabilities": {
                    "checkpoint_loaded": True,
                    "true_streaming": True,
                },
            }
    monkeypatch.setattr("oaa.backends.llama_cpp.LlamaCppBackend", InfoBackend)
    assert main(["gguf-info", "--model", "C:/models/local.gguf"]) == 0
    assert '"model_format": "GGUF"' in capsys.readouterr().out
