import contextvars
import datetime
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any, ContextManager, Literal, TypeVar, overload, override

from .protocols import (
    GenerationSpan,
    GenerationSpanView,
    Provider,
    Span,
    SpanView,
    ToolSpan,
)
from .utils import merge_usage

T = TypeVar("T", bound=Span)


class SpanBackend(ABC):
    def on_set_name(self, span: SpanView, name: str) -> None:
        _ = span, name

    def on_set_input(self, span: SpanView, input: Any) -> None:
        _ = span, input

    def on_set_output(self, span: SpanView, output: Any) -> None:
        _ = span, output

    def on_append_output(self, span: SpanView, chunk: Any) -> None:
        _ = span, chunk

    def on_add_metadata(self, span: SpanView, metadata: dict[str, Any]) -> None:
        _ = span, metadata

    def on_set_status_message(self, span: SpanView, message: str) -> None:
        _ = span, message

    # Generation-specific
    def on_set_completion_start_time(
        self, span: GenerationSpanView, time: datetime.datetime
    ) -> None:
        _ = span, time

    def on_set_model(self, span: GenerationSpanView, name: str) -> None:
        _ = span, name

    def on_set_model_parameters(
        self, span: GenerationSpanView, params: dict[str, Any]
    ) -> None:
        _ = span, params

    def on_add_usage(self, span: GenerationSpanView, usage: dict[str, Any]) -> None:
        _ = span, usage

    def on_set_usage(self, span: GenerationSpanView, usage: dict[str, Any]) -> None:
        _ = span, usage

    def on_set_cost(self, span: GenerationSpanView, cost: dict[str, float]) -> None:
        _ = span, cost


class NoopSpanBackend(SpanBackend): ...


class BaseSpan(Span):
    def __init__(
        self,
        *,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        status_message: str | None = None,
        _span_backend: SpanBackend | None = None,
    ):
        self._name: str = name
        self._input: Any | None = input
        self._output: Any | None = output
        self._metadata: dict[str, Any] = metadata or {}
        self._status_message: str | None = status_message
        self._span_backend: SpanBackend = _span_backend or NoopSpanBackend()

    @override
    def set_name(self, name: str) -> None:
        self._name = name
        self._span_backend.on_set_name(self, name)

    @override
    def set_input(self, input: Any) -> None:
        _ = self._input = input
        self._span_backend.on_set_input(self, input)

    @override
    def set_output(self, output: Any) -> None:
        _ = self._output = output
        self._span_backend.on_set_output(self, output)

    @override
    def append_output(self, chunk: Any) -> None:
        if self._output is None:
            self._output = chunk
        elif isinstance(self._output, str) and isinstance(chunk, str):
            self._output = self._output + chunk
        elif isinstance(self._output, bytes) and isinstance(chunk, (bytes, bytearray)):
            self._output = self._output + bytes(chunk)
        elif isinstance(self._output, list):
            if isinstance(chunk, list):
                self._output.extend(chunk)  # pyright: ignore[reportUnknownMemberType]
            else:
                self._output.append(chunk)  # pyright: ignore[reportUnknownMemberType]
        else:
            try:
                # Fallback: attempt concatenation; may raise TypeError
                self._output = self._output + chunk  # type: ignore[operator]
            except TypeError:
                pass  # Goes silently as instrumentation should not break the application
        self._span_backend.on_append_output(self, chunk)

    @override
    def add_metadata(self, metadata: dict[str, Any]) -> None:
        _ = self._metadata.update(metadata)
        self._span_backend.on_add_metadata(self, metadata)

    @override
    def set_status_message(self, message: str) -> None:
        _ = self._status_message = message
        self._span_backend.on_set_status_message(self, message)

    @property
    @override
    def name(self) -> str | None:
        return self._name

    @property
    @override
    def input(self) -> Any | None:
        return self._input

    @property
    @override
    def output(self) -> Any | None:
        return self._output

    @property
    @override
    def metadata(self) -> dict[str, Any] | None:
        return self._metadata

    @property
    @override
    def status_message(self) -> str | None:
        return self._status_message


class BaseGenerationSpan(BaseSpan, GenerationSpan):
    def __init__(
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
        _span_backend: SpanBackend | None = None,
    ):
        super().__init__(
            name=name,
            input=input,
            output=output,
            metadata=metadata,
            status_message=status_message,
            _span_backend=_span_backend,
        )
        self._completion_start_time: datetime.datetime | None = completion_start_time
        self._model: str | None = model
        self._model_parameters: dict[str, Any] = model_parameters or {}
        self._usage: dict[str, Any] = usage or {}
        self._cost: dict[str, float] = cost or {}

    @override
    def set_completion_start_time(self, time: datetime.datetime) -> None:
        self._completion_start_time = time
        self._span_backend.on_set_completion_start_time(self, time)

    @override
    def set_model(self, name: str) -> None:
        self._model = name
        self._span_backend.on_set_model(self, name)

    @override
    def set_model_parameters(self, parameters: dict[str, Any]) -> None:
        self._model_parameters = parameters
        self._span_backend.on_set_model_parameters(self, parameters)

    @override
    def set_usage(self, usage: dict[str, Any]) -> None:
        self._usage = usage
        self._span_backend.on_set_usage(self, usage)

    @override
    def set_cost(self, cost: dict[str, float]) -> None:
        self._cost = cost
        self._span_backend.on_set_cost(self, cost)

    @override
    def add_usage(self, usage: dict[str, Any]) -> None:
        self._usage = merge_usage(self._usage, usage)
        self.set_usage(self._usage)
        self._span_backend.on_add_usage(self, usage)

    @property
    @override
    def model(self) -> str | None:
        return self._model

    @property
    @override
    def model_parameters(self) -> dict[str, Any] | None:
        return self._model_parameters

    @property
    @override
    def completion_start_time(self) -> datetime.datetime | None:
        return self._completion_start_time

    @property
    @override
    def usage(self) -> dict[str, Any] | None:
        return self._usage

    @property
    @override
    def cost(self) -> dict[str, float] | None:
        return self._cost


class BaseToolSpan(BaseSpan, ToolSpan):
    pass


class SpanKind(Enum):
    SPAN = "span"
    GENERATION = "generation"
    TOOL = "tool"


BaseSpanTypes = BaseSpan | BaseGenerationSpan | BaseToolSpan

BS = TypeVar("BS", bound=BaseSpan)

span_stack: contextvars.ContextVar[list[Any]] = contextvars.ContextVar(
    "span_stack", default=[]
)


class SpanBehavior(ABC):
    """Defines how to create spans and wrap external systems."""

    @abstractmethod
    @contextmanager
    def make_span(
        self,
        kind: SpanKind,
        base_cls: type[BS],
        **kwargs: Any,
    ) -> Iterator[BS]: ...


class ProviderBehavior(ABC):
    """Defines what happens on enter/exit."""

    def on_enter(self, kind: SpanKind, span: BaseSpanTypes, stack: list[Any]) -> None:
        _ = (kind, span, stack)

    def on_exit(
        self,
        kind: SpanKind,
        span: BaseSpanTypes,
        stack: list[Any],
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> bool | None:
        _ = (kind, span, stack, exc_type, exc_val, exc_tb)
        return None


class DefaultSpanBehavior(SpanBehavior):
    """Just instantiates the base span with kwargs."""

    @contextmanager
    def make_span(
        self,
        kind: SpanKind,
        base_cls: type[BS],
        **kwargs: Any,
    ) -> Iterator[BS]:
        yield base_cls(**kwargs)


class NoopProviderBehavior(ProviderBehavior):
    pass


SPAN_KIND_TO_BASE: dict[SpanKind, type[BaseSpanTypes]] = {
    SpanKind.SPAN: BaseSpan,
    SpanKind.GENERATION: BaseGenerationSpan,
    SpanKind.TOOL: BaseToolSpan,
}


class BaseProvider(Provider):
    """Composes SpanBehavior and SinkBehavior."""

    def __init__(
        self,
        *,
        span_behavior: SpanBehavior | None = None,
        provider_behavior: ProviderBehavior | None = None,
    ):
        self._span_behavior: SpanBehavior = span_behavior or DefaultSpanBehavior()
        self._provider_behavior: ProviderBehavior = (
            provider_behavior or NoopProviderBehavior()
        )

    @overload
    @contextmanager
    def _context(
        self, kind: Literal[SpanKind.SPAN], **kwargs: Any
    ) -> Iterator[BaseSpan]: ...

    @overload
    @contextmanager
    def _context(
        self, kind: Literal[SpanKind.GENERATION], **kwargs: Any
    ) -> Iterator[BaseGenerationSpan]: ...

    @overload
    @contextmanager
    def _context(
        self, kind: Literal[SpanKind.TOOL], **kwargs: Any
    ) -> Iterator[BaseToolSpan]: ...

    @contextmanager
    def _context(self, kind: SpanKind, **kwargs: Any) -> Iterator[BaseSpanTypes]:
        base_cls = SPAN_KIND_TO_BASE[kind]
        with self._span_behavior.make_span(kind, base_cls, **kwargs) as span:
            current_stack = span_stack.get() + [span]
            _ = span_stack.set(current_stack)
            self._provider_behavior.on_enter(kind, span, current_stack)
            try:
                yield span
            except BaseException as e:
                suppress = self._provider_behavior.on_exit(
                    kind, span, current_stack, type(e), e, e.__traceback__
                )
                if not suppress:
                    raise
            else:
                _ = self._provider_behavior.on_exit(
                    kind, span, current_stack, None, None, None
                )
            finally:
                _ = span_stack.set(current_stack[:-1])

    @override
    def span(self, **kwargs: Any) -> ContextManager[BaseSpan]:
        return self._context(SpanKind.SPAN, **kwargs)

    @override
    def generation(self, **kwargs: Any) -> ContextManager[BaseGenerationSpan]:
        return self._context(SpanKind.GENERATION, **kwargs)

    @override
    def tool(self, **kwargs: Any) -> ContextManager[BaseToolSpan]:
        return self._context(SpanKind.TOOL, **kwargs)

    # escape hatch
    def span_of_kind(
        self, kind: SpanKind, **kwargs: Any
    ) -> ContextManager[BaseSpanTypes]:
        return self._context(kind, **kwargs)
