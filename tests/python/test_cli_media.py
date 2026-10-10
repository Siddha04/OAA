from __future__ import annotations

import json
import wave
from pathlib import Path

from PIL import Image

from oaa.cli import build_parser, main


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 32)


def test_media_cli_parser_for_image_and_audio():
    args = build_parser().parse_args(["media", "image-info", "--path", "pic.png"])
    assert (args.command, args.media_action, args.path) == ("media", "image-info", "pic.png")
    args = build_parser().parse_args(["media", "audio-info", "--path", "voice.wav"])
    assert args.media_action == "audio-info"


def test_media_cli_image_info(tmp_path: Path, capsys):
    path = tmp_path / "picture.png"
    Image.new("RGB", (3, 2), (5, 10, 15)).save(path)
    assert main(["media", "image-info", "--path", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["filename"] == "picture.png"
    assert out["source_format"] == "PNG"
    assert out["width"] == 3
    assert out["height"] == 2


def test_media_cli_audio_info(tmp_path: Path, capsys):
    path = tmp_path / "voice.wav"
    _wav(path)
    assert main(["media", "audio-info", "--path", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["filename"] == "voice.wav"
    assert out["format"] == "WAV/PCM"
    assert out["sample_rate"] == 16000
    assert out["channels"] == 1


def test_media_cli_rejects_oversized_image_and_malformed_audio(tmp_path: Path, capsys):
    image = tmp_path / "large.png"
    Image.new("RGB", (10, 10)).save(image)
    assert main([
        "media", "image-info", "--path", str(image), "--max-pixels", "10",
    ]) == 2
    assert "max_pixels" in capsys.readouterr().err

    audio = tmp_path / "bad.wav"
    audio.write_bytes(b"not wav")
    assert main(["media", "audio-info", "--path", str(audio)]) == 2
    assert "valid" in capsys.readouterr().err
