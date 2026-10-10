"""Local-only media decoding and stable adapter contracts.

This module prepares bounded image pixels and PCM WAV samples for future model
backends. It does not perform image understanding, OCR, ASR, or speech synthesis.
"""
from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

DEFAULT_MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 16_777_216
DEFAULT_IMAGE_MAX_DIMENSION = 1024
DEFAULT_MAX_AUDIO_BYTES = 40 * 1024 * 1024
DEFAULT_MAX_AUDIO_SECONDS = 60.0
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 192_000
_ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
_SUPPORTED_SAMPLE_WIDTHS = frozenset({1, 2, 3, 4})


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """An orientation-corrected RGB image in row-major byte layout."""
    filename: str
    width: int
    height: int
    rgb_bytes: bytes
    source_format: str
    original_width: int
    original_height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if len(self.rgb_bytes) != self.width * self.height * 3:
            raise ValueError("RGB byte count does not match the image dimensions")

    @property
    def resized(self) -> bool:
        return self.width != self.original_width or self.height != self.original_height

    def summary(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "source_format": self.source_format,
            "width": self.width,
            "height": self.height,
            "channels": 3,
            "mode": "RGB",
            "pixel_bytes": len(self.rgb_bytes),
            "original_width": self.original_width,
            "original_height": self.original_height,
            "resized": self.resized,
        }


def load_image(
    path: str | Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    max_dimension: int = DEFAULT_IMAGE_MAX_DIMENSION,
) -> ImageFrame:
    """Decode one selected local image into bounded RGB bytes.

    Install Pillow with: python -m pip install "oaa[vision]". URLs are never
    fetched. Animated images, if supported by Pillow, use only their first frame.
    """
    for name, value in (
        ("max_file_bytes", max_file_bytes),
        ("max_pixels", max_pixels),
        ("max_dimension", max_dimension),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"image file does not exist: {source}")
    size = source.stat().st_size
    if size <= 0:
        raise ValueError("image file is empty")
    if size > max_file_bytes:
        raise ValueError(f"image file exceeds max_file_bytes ({size} > {max_file_bytes})")

    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError(
            'image decoding requires Pillow; install with: python -m pip install "oaa[vision]"'
        ) from exc

    try:
        with Image.open(source) as decoded:
            source_format = str(decoded.format or "").upper()
            if source_format not in _ALLOWED_IMAGE_FORMATS:
                raise ValueError(f"unsupported image format: {source_format or 'unknown'}")
            original_width, original_height = decoded.size
            if original_width <= 0 or original_height <= 0:
                raise ValueError("image dimensions must be positive")
            total_pixels = original_width * original_height
            if total_pixels > max_pixels:
                raise ValueError(f"image exceeds max_pixels ({total_pixels} > {max_pixels})")
            if getattr(decoded, "is_animated", False):
                decoded.seek(0)
            rgb = ImageOps.exif_transpose(decoded).convert("RGB")
            rgb.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            width, height = rgb.size
            pixels = rgb.tobytes()
    except UnidentifiedImageError as exc:
        raise ValueError("file is not a supported, decodable image") from exc
    except Image.DecompressionBombError as exc:
        raise ValueError("image exceeds the decoder safety limit") from exc

    return ImageFrame(
        filename=source.name,
        width=width,
        height=height,
        rgb_bytes=pixels,
        source_format=source_format,
        original_width=original_width,
        original_height=original_height,
    )


@dataclass(frozen=True, slots=True)
class AudioClip:
    """Bounded uncompressed PCM samples from a local WAV file."""
    filename: str
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int
    pcm_bytes: bytes

    def __post_init__(self) -> None:
        if not MIN_SAMPLE_RATE <= self.sample_rate <= MAX_SAMPLE_RATE:
            raise ValueError("sample_rate is outside the supported range")
        if self.channels not in (1, 2):
            raise ValueError("only mono or stereo audio is supported")
        if self.sample_width not in _SUPPORTED_SAMPLE_WIDTHS:
            raise ValueError("sample_width must be 1, 2, 3, or 4 bytes")
        if self.frame_count <= 0:
            raise ValueError("audio must contain at least one frame")
        expected_bytes = self.frame_count * self.channels * self.sample_width
        if len(self.pcm_bytes) != expected_bytes:
            raise ValueError("PCM byte count does not match the declared audio format")

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate

    def summary(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "format": "WAV/PCM",
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width,
            "frames": self.frame_count,
            "duration_seconds": round(self.duration_seconds, 6),
            "pcm_bytes": len(self.pcm_bytes),
        }

    def iter_normalized_samples(self) -> Iterator[float]:
        """Yield interleaved samples normalized to [-1.0, 1.0]."""
        width = self.sample_width
        if width == 1:
            for value in self.pcm_bytes:
                yield (value - 128) / 128.0
            return
        scale = float(1 << (width * 8 - 1))
        for offset in range(0, len(self.pcm_bytes), width):
            value = int.from_bytes(
                self.pcm_bytes[offset:offset + width],
                byteorder="little",
                signed=True,
            )
            yield max(-1.0, min(1.0, value / scale))


def load_wav(
    path: str | Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
    max_duration_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
) -> AudioClip:
    """Load bounded local PCM WAV input; it does not access a microphone or network."""
    if not isinstance(max_file_bytes, int) or isinstance(max_file_bytes, bool) or max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be a positive integer")
    if (
        isinstance(max_duration_seconds, bool)
        or not isinstance(max_duration_seconds, (int, float))
        or not math.isfinite(max_duration_seconds)
        or max_duration_seconds <= 0
    ):
        raise ValueError("max_duration_seconds must be a positive finite number")
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"audio file does not exist: {source}")
    size = source.stat().st_size
    if size <= 0:
        raise ValueError("audio file is empty")
    if size > max_file_bytes:
        raise ValueError(f"audio file exceeds max_file_bytes ({size} > {max_file_bytes})")

    try:
        with wave.open(str(source), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frame_count = reader.getnframes()
            if reader.getcomptype() != "NONE":
                raise ValueError("only uncompressed PCM WAV audio is supported")
            if channels not in (1, 2):
                raise ValueError("only mono or stereo audio is supported")
            if sample_width not in _SUPPORTED_SAMPLE_WIDTHS:
                raise ValueError("WAV sample width must be 1, 2, 3, or 4 bytes")
            if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
                raise ValueError(
                    f"sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz"
                )
            if frame_count <= 0:
                raise ValueError("audio file contains no frames")
            duration = frame_count / sample_rate
            if duration > max_duration_seconds:
                raise ValueError(
                    f"audio duration exceeds max_duration_seconds "
                    f"({duration:.3f} > {max_duration_seconds})"
                )
            pcm_bytes = reader.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise ValueError("file is not a valid, uncompressed PCM WAV") from exc

    expected = frame_count * channels * sample_width
    if len(pcm_bytes) != expected:
        raise ValueError("WAV data ended before all declared PCM frames were read")
    return AudioClip(source.name, sample_rate, channels, sample_width, frame_count, pcm_bytes)


@runtime_checkable
class SpeechToTextBackend(Protocol):
    """Adapter contract for an explicitly installed local ASR model."""
    name: str

    def transcribe(self, audio: AudioClip) -> str: ...


@runtime_checkable
class TextToSpeechBackend(Protocol):
    """Adapter contract for an explicitly installed local TTS model."""
    name: str

    def synthesize(self, text: str) -> "SynthesizedAudio": ...


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    """PCM output returned by a TTS adapter, ready to write as WAV."""
    sample_rate: int
    channels: int
    sample_width: int
    pcm_bytes: bytes

    def __post_init__(self) -> None:
        if not MIN_SAMPLE_RATE <= self.sample_rate <= MAX_SAMPLE_RATE:
            raise ValueError("sample_rate is outside the supported range")
        if self.channels not in (1, 2):
            raise ValueError("only mono or stereo output is supported")
        if self.sample_width not in _SUPPORTED_SAMPLE_WIDTHS:
            raise ValueError("sample_width must be 1, 2, 3, or 4 bytes")
        if not self.pcm_bytes:
            raise ValueError("synthesized PCM must not be empty")
        if len(self.pcm_bytes) % (self.channels * self.sample_width):
            raise ValueError("synthesized PCM byte count is not frame-aligned")


def transcribe_audio(audio: AudioClip, backend: SpeechToTextBackend | None = None) -> str:
    """Use only a caller-provided ASR backend; no cloud/default backend is selected."""
    if backend is None:
        raise RuntimeError(
            "no speech-to-text backend is configured; install and explicitly pass a local ASR adapter"
        )
    result = backend.transcribe(audio)
    if not isinstance(result, str):
        raise TypeError("speech-to-text backend must return a string")
    if len(result) > 100_000:
        raise ValueError("transcript exceeds the 100000-character limit")
    return result


def synthesize_speech(
    text: str,
    backend: TextToSpeechBackend | None = None,
) -> SynthesizedAudio:
    """Use only a caller-provided TTS backend; does not discover services."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > 10_000:
        raise ValueError("text exceeds the 10000-character limit")
    if backend is None:
        raise RuntimeError(
            "no text-to-speech backend is configured; install and explicitly pass a local TTS adapter"
        )
    audio = backend.synthesize(text)
    if not isinstance(audio, SynthesizedAudio):
        raise TypeError("text-to-speech backend must return SynthesizedAudio")
    return audio


def write_wav(path: str | Path, audio: SynthesizedAudio, *, overwrite: bool = False) -> Path:
    """Write PCM from a supplied TTS adapter; refuse overwrite by default."""
    destination = Path(path).expanduser()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing audio file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(audio.channels)
        writer.setsampwidth(audio.sample_width)
        writer.setframerate(audio.sample_rate)
        writer.writeframes(audio.pcm_bytes)
    return destination
