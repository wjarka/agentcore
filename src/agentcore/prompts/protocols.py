import abc
from typing import Protocol, runtime_checkable

from openai.types.chat import ChatCompletionMessageParam


@runtime_checkable
class Prompt(Protocol):
    async def build_prompt(self) -> str: ...
    async def to_message(self) -> ChatCompletionMessageParam: ...

    @property
    def cache_key(self) -> str | None: ...


@runtime_checkable
class SystemPrompt(Prompt, Protocol):
    @property
    def suggested_model(self) -> str: ...

    @property
    def json_mode(self) -> bool: ...

    @property
    def max_tokens(self) -> int | None: ...


class ToolSelectorPrompt(abc.ABC, SystemPrompt): ...


class ToolBuilderPrompt(abc.ABC, SystemPrompt): ...


class AnswerGeneratorPrompt(abc.ABC, SystemPrompt): ...


class ThinkPrompt(abc.ABC, SystemPrompt): ...


class DataProcessPrompt(abc.ABC, SystemPrompt): ...
