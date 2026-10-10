from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest
from PIL import Image

from oaa.media import (
    AudioClip, SynthesizedAudio, load_image, load_wav, synthesize_speech,
    transcribe_audio, write_wav,
)


def test_load_image_decodes_rgb_and_downscales_within_limit(tmp_path: Path):
    path = tmp_path / "source.png"
    Image.new("RGB", (8, 4), (12, 34, 56)).save(path)
    frame = load_image(path, max_dimension=4)
    assert (frame.width, frame.height) == (4, 2)
    assert frame.source_format == "PNG"
    assert frame.resized
    assert frame.rgb_bytes[:3] == bytes((12, 34, 56))
    assert frame.summary()["pixel_bytes"] == 4 * 2 * 3


def test_image_decoder_rejects_oversized_pixels_and_unsupported_formats(tmp_path: Path):
    large = tmp_path / "large.png"
    Image.new("RGB", (8, 8)).save(large)
    with pytest.raises(ValueError, match="max_pixels"):
        load_image(large, max_pixels=16)
    gif = tmp_path / "image.gif"
    Image.new("RGB", (2, 2)).save(gif)
    with pytest.raises(ValueError, match="unsupported image format"):
        load_image(gif)


def test_image_decoder_rejects_bad_input_and_file_size(tmp_path: Path):
    bad = tmp_path / "not-an-image.png"
    bad.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="not a supported"):
        load_image(bad)
    real = tmp_path / "ok.png"
    Image.new("RGB", (2, 2)).save(real)
    with pytest.raises(ValueError, match="max_file_bytes"):
        load_image(real, max_file_bytes=1)


def write_wav_file(path: Path, values: list[int], *, sample_width=2, rate=8000, channels=1):
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(rate)
        if sample_width == 1:
            payload = bytes(values)
        else:
            payload = b"".join(
                value.to_bytes(sample_width, "little", signed=True) for value in values
            )
        writer.writeframes(payload)


def test_load_wav_parses_pcm_and_normalizes_samples(tmp_path: Path):
    path = tmp_path / "speech.wav"
    write_wav_file(path, [-32768, 0, 32767])
    audio = load_wav(path)
    assert audio.sample_rate == 8000
    assert audio.channels == 1
    assert audio.frame_count == 3
    assert audio.summary()["format"] == "WAV/PCM"
    samples = list(audio.iter_normalized_samples())
    assert samples[0] == -1.0
    assert samples[1] == 0.0
    assert math.isclose(samples[2], 32767 / 32768)


def test_load_wav_handles_24_bit_pcm_and_stereo(tmp_path: Path):
    path = tmp_path / "stereo24.wav"
    write_wav_file(path, [-8388608, 8388607, 0, 100], sample_width=3, channels=2)
    audio = load_wav(path)
    assert audio.channels == 2
    samples = list(audio.iter_normalized_samples())
    assert samples[0] == -1.0
    assert math.isclose(samples[1], 8388607 / 8388608)
    assert samples[2:] == [0.0, 100 / 8388608]


def test_load_wav_enforces_duration_size_and_validity_limits(tmp_path: Path):
    path = tmp_path / "long.wav"
    write_wav_file(path, [0] * 16000, rate=8000)
    with pytest.raises(ValueError, match="duration exceeds"):
        load_wav(path, max_duration_seconds=1.0)
    with pytest.raises(ValueError, match="max_file_bytes"):
        load_wav(path, max_file_bytes=1)
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"broken")
    with pytest.raises(ValueError, match="valid"):
        load_wav(bad)


def test_voice_adapters_are_explicit_and_return_bounded_types(tmp_path: Path):
    audio = AudioClip("clip.wav", 8000, 1, 2, 1, b"\x00\x00")
    with pytest.raises(RuntimeError, match="no speech-to-text backend"):
        transcribe_audio(audio)
    with pytest.raises(RuntimeError, match="no text-to-speech backend"):
        synthesize_speech("hello")

    class ASR:
        name = "test-local-asr"
        def transcribe(self, supplied):
            assert supplied is audio
            return "hello locally"

    class TTS:
        name = "test-local-tts"
        def synthesize(self, text):
            assert text == "Hello"
            return SynthesizedAudio(8000, 1, 2, b"\x01\x00\x02\x00")

    assert transcribe_audio(audio, ASR()) == "hello locally"
    generated = synthesize_speech("Hello", TTS())
    output = tmp_path / "generated.wav"
    assert write_wav(output, generated) == output
    assert load_wav(output).frame_count == 2
    with pytest.raises(FileExistsError, match="overwrite"):
        write_wav(output, generated)


def test_audio_and_tts_reject_invalid_bounds():
    with pytest.raises(ValueError, match="sample_rate"):
        SynthesizedAudio(100, 1, 2, b"\x00\x00")
    with pytest.raises(ValueError, match="frame-aligned"):
        SynthesizedAudio(8000, 2, 2, b"\x00")
    with pytest.raises(ValueError, match="non-empty"):
        synthesize_speech("   ", object())
