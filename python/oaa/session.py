from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


class ChatSession:
    def __init__(
        self,
        system_prompt: str = "You are OAA, a local personal AI assistant.",
        *,
        max_history_messages: int = 32,
        session_id: str | None = None,
    ) -> None:
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")
        if max_history_messages <= 0:
            raise ValueError("max_history_messages must be greater than zero")
        self._system_prompt = system_prompt
        self._max_history_messages = max_history_messages
        self._session_id = session_id or uuid4().hex
        self._messages: list[ChatMessage] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("system_prompt must be a string")
        self._system_prompt = value

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        return tuple(self._messages)

    def add(self, role: Role, content: str) -> None:
        if role not in {"system", "user", "assistant"}:
            raise ValueError("unsupported message role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be a non-empty string")
        if role == "system":
            self._system_prompt = content
            return
        self._messages.append(ChatMessage(role, content))
        overflow = len(self._messages) - self._max_history_messages
        if overflow > 0:
            del self._messages[:overflow]

    def clear(self) -> None:
        self._messages.clear()

    def build_prompt(self, current_user_message: str | None = None) -> str:
        if current_user_message is not None:
            if not isinstance(current_user_message, str) or not current_user_message.strip():
                raise ValueError("current user message must be non-empty")
        lines = [f"<system>{self._system_prompt}</system>"]
        for message in self._messages:
            lines.append(f"<{message.role}>{message.content}</{message.role}>")
        if current_user_message is not None:
            lines.append(f"<user>{current_user_message}</user>")
        lines.append("<assistant>")
        return "\n".join(lines)

    def fork(self, *, session_id: str | None = None) -> "ChatSession":
        clone = ChatSession(
            self._system_prompt,
            max_history_messages=self._max_history_messages,
            session_id=session_id,
        )
        clone._messages = list(self._messages)
        return clone
