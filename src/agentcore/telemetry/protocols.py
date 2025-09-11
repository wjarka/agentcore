from __future__ import annotations

import datetime
from typing import Any, ContextManager, ParamSpec, Protocol, TypeVar, runtime_checkable

P = ParamSpec("P")
R = TypeVar("R")


class SpanView(Protocol):
    @property
    def name(self) -> str | None: ...
    @property
    def input(self) -> Any | None: ...
    @property
    def output(self) -> Any | None: ...
    @property
    def metadata(self) -> dict[str, Any] | None: ...
    @property
    def status_message(self) -> str | None: ...


@runtime_checkable
class Span(SpanView, Protocol):
    def set_name(self, name: str) -> None: ...
    def set_input(self, input: Any) -> None: ...
    def set_output(self, output: Any) -> None: ...
    def append_output(self, chunk: Any) -> None: ...
    def add_metadata(self, metadata: dict[str, Any]) -> None: ...
    def set_status_message(self, message: str) -> None: ...


class GenerationSpanView(SpanView, Protocol):
    @property
    def completion_start_time(self) -> datetime.datetime | None: ...
    @property
    def model(self) -> str | None: ...
    @property
    def model_parameters(self) -> dict[str, Any] | None: ...
    @property
    def usage(self) -> dict[str, Any] | None: ...
    @property
    def cost(self) -> dict[str, float] | None: ...


@runtime_checkable
class GenerationSpan(GenerationSpanView, Span, Protocol):
    def set_completion_start_time(self, time: datetime.datetime) -> None: ...
    def set_model(self, name: str) -> None: ...
    def set_model_parameters(self, parameters: dict[str, Any]) -> None: ...
    def set_usage(self, usage: dict[str, Any]) -> None: ...
    def set_cost(self, cost: dict[str, float]) -> None: ...
    def add_usage(self, usage: dict[str, Any]) -> None: ...


@runtime_checkable
class ToolSpan(Span, Protocol): ...


@runtime_checkable
class Provider(Protocol):
    def span(
        self,
        *,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        status_message: str | None = None,
    ) -> ContextManager[Span]: ...

    def generation(
        self,
        *,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        status_message: str | None = None,
        completion_start_time: datetime.datetime | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        cost: dict[str, float] | None = None,
    ) -> ContextManager[GenerationSpan]: ...

    def tool(
        self,
        *,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        status_message: str | None = None,
    ) -> ContextManager[ToolSpan]: ...


SpanTypes = Span | GenerationSpan | ToolSpan
