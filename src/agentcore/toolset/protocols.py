from __future__ import annotations

import abc
import collections.abc
from types import UnionType
from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pydantic.fields import FieldInfo
from pydantic.types import JsonValue

from agentcore.models import ActionResult, ToolParam, Validator
from agentcore.protocols import Executable
from agentcore.structures.protocols import SupportsAdding

T = TypeVar("T")

FunctionToolCallable = collections.abc.Callable[
    ..., collections.abc.Awaitable[ActionResult] | ActionResult
]


@runtime_checkable
class Action(Executable[ActionResult], Protocol):
    @property
    def tool_name(self) -> str: ...

    @property
    def params(self) -> JsonValue: ...


@runtime_checkable
class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def required_parameters(self) -> dict[str, ToolParam]: ...

    @property
    def optional_parameters(self) -> dict[str, ToolParam]: ...

    async def prepare_action(
        self, params: dict[str, JsonValue], **kwargs: Any
    ) -> Action: ...


@runtime_checkable
class AdaptableTool(Tool, Protocol):
    def with_name(self, name: str) -> AdaptableTool: ...
    def with_parameter(
        self, name: str, param_type: type | UnionType, field_info: FieldInfo
    ) -> AdaptableTool: ...
    def with_validators(self, **validators: Validator) -> AdaptableTool: ...


ToolTypes = Tool | AdaptableTool


class ToolRegistry(
    abc.ABC,
    collections.abc.MutableMapping[str, ToolTypes],
    collections.abc.Iterable[str],
    SupportsAdding[str, ToolTypes],
):
    @abc.abstractmethod
    def adaptable(self, name: str) -> AdaptableTool: ...
