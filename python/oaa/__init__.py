"""Python interface for the OAA runtime."""

from .assistant import PersonalAssistant
from .config import GenerationConfig
from .cuda import available as cuda_available
from .cuda import compiled as cuda_compiled
from .cuda import vector_add as cuda_vector_add
from .engine import Engine
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
]
__version__ = "0.4.0"
