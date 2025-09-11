import abc
from pathlib import PurePath
from typing import override

import jinja2
from openai.types.chat import ChatCompletionMessageParam

from .protocols import (
    Prompt,
    SystemPrompt,
)


class BasePrompt(abc.ABC, Prompt):
    def __init__(self, jinja: jinja2.Environment):
        self._jinja: jinja2.Environment = jinja

    @property
    @abc.abstractmethod
    def _template_path(self) -> PurePath:
        return PurePath("prompts")

    @abc.abstractmethod
    async def _prepare_vars(self) -> dict[str, str | None]:
        pass

    def _join(self, parts: list[str]) -> str:
        return "\n".join(parts)

    @override
    async def build_prompt(self) -> str:
        path = str(self._template_path)
        template = self._jinja.get_template(path)
        vars = await self._prepare_vars()
        return await template.render_async(**vars)

    @override
    async def to_message(self) -> ChatCompletionMessageParam:
        return {
            "role": "user",
            "content": await self.build_prompt(),
        }

    @property
    @override
    def cache_key(self) -> str | None:
        return self.__class__.__name__


class BaseSystemPrompt(BasePrompt, SystemPrompt, abc.ABC):
    def __init__(self, jinja: jinja2.Environment):
        super().__init__(jinja)

    @property
    @override
    def suggested_model(self) -> str:
        return "gpt-4.1"

    @property
    @override
    def json_mode(self) -> bool:
        return True

    @property
    @override
    def max_tokens(self) -> int | None:
        return None

    @override
    async def to_message(self) -> ChatCompletionMessageParam:
        return {
            "role": "system",
            "content": await self.build_prompt(),
        }
