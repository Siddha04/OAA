"""Python interface for the OAA runtime."""

from .assistant import PersonalAssistant
from .config import GenerationConfig
from .cuda import available as cuda_available
from .cuda import compiled as cuda_compiled
from .cuda import vector_add as cuda_vector_add
from .engine import Engine
from .media import AudioClip, ImageFrame, SynthesizedAudio, load_image, load_wav, synthesize_speech, transcribe_audio, write_wav
from .session import ChatMessage, ChatSession

__all__ = [
    "Engine",
    "GenerationConfig",
    "PersonalAssistant",
    "ChatMessage",
    "ChatSession",
    "cuda_available",
    "cuda_compiled",
    "cuda_vector_add",
    "AudioClip",
    "ImageFrame",
    "SynthesizedAudio",
    "load_image",
    "load_wav",
    "transcribe_audio",
    "synthesize_speech",
    "write_wav",
]
__version__ = "0.8.0"
