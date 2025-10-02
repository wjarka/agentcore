from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any, Literal, Protocol, overload

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from agentcore.prompts.protocols import Prompt, SystemPrompt

from ..models import Document


class LLMService(Protocol):
    @overload
    async def completion(
        self,
        *,
        user_prompt: str | ChatCompletionMessageParam | Prompt | None = ...,
        system_prompt: str | ChatCompletionMessageParam | SystemPrompt | None = ...,
        history: list[ChatCompletionMessageParam] | None = ...,
        model: str = ...,
        json_mode: bool = ...,
        max_tokens: int | None = ...,
        cache_key: str | None = None,
        name: str | None = None,
    ) -> ChatCompletion: ...

    @overload
    async def completion(
        self,
        *,
        user_prompt: str | ChatCompletionMessageParam | Prompt | None = ...,
        system_prompt: str | ChatCompletionMessageParam | SystemPrompt | None = ...,
        history: list[ChatCompletionMessageParam] | None = ...,
        model: str = ...,
        stream: Literal[False] = False,
        json_mode: bool = ...,
        max_tokens: int | None = ...,
        cache_key: str | None = None,
        name: str | None = None,
    ) -> ChatCompletion: ...

    @overload
    async def completion(
        self,
        *,
        user_prompt: str | ChatCompletionMessageParam | Prompt | None = ...,
        system_prompt: str | ChatCompletionMessageParam | SystemPrompt | None = ...,
        history: list[ChatCompletionMessageParam] | None = ...,
        model: str = ...,
        stream: Literal[True] = True,
        json_mode: bool = ...,
        max_tokens: int | None = ...,
        cache_key: str | None = None,
        name: str | None = None,
    ) -> AsyncIterable[ChatCompletionChunk]: ...

    async def transcribe(
        self,
        audio_files: list[str],
        language: str = "en",
        prompt: str | None = None,
        file_name: str = "transcription.md",
    ) -> list[Document]: ...


class EmbeddingService(Protocol):
    async def get_openai_embedding(
        self, text: str, model: str = "text-embedding-3-large"
    ) -> list[float]: ...
    async def get_jina_embedding(self, text: str) -> list[float]: ...


class TextService(Protocol):
    def document(
        self,
        text: str,
        additional_metadata: dict[str, Any] | None = None,
        model: str | None = None,  # User can specify a model for this specific document
    ) -> Document: ...

    def restore_placeholders(self, doc: Document) -> Document: ...

    def split(
        self, text: str, limit: int, additional_metadata: dict[str, Any] | None = None
    ) -> list[Document]: ...
