import datetime
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, ContextManager, override

from agentcore.telemetry.protocols import (
    GenerationSpan,
    GenerationSpanView,
    Provider,
    Span,
    SpanView,
    ToolSpan,
)

from ..base import (
    BS,
    SpanBackend,
    SpanBehavior,
    SpanKind,
)


class MultiSpanBackend(SpanBackend):
    def __init__(self, spans: list[Span | GenerationSpan | ToolSpan]):
        self._spans: list[Span | GenerationSpan | ToolSpan] = spans

    @override
    def on_set_name(self, span: SpanView, name: str) -> None:
        for _span in self._spans:
            _span.set_name(name)

    @override
    def on_set_input(self, span: SpanView, input: Any) -> None:
        for _span in self._spans:
            _span.set_input(input)

    @override
    def on_set_output(self, span: SpanView, output: Any) -> None:
        for _span in self._spans:
            _span.set_output(output)

    @override
    def on_append_output(self, span: SpanView, chunk: Any) -> None:
        for _span in self._spans:
            _ = _span.append_output(chunk)

    @override
    def on_add_metadata(self, span: SpanView, metadata: dict[str, Any]) -> None:
        for _span in self._spans:
            _span.add_metadata(metadata)

    @override
    def on_set_status_message(self, span: SpanView, message: str) -> None:
        for _span in self._spans:
            _span.set_status_message(message)

    # Generation-specific
    @override
    def on_set_completion_start_time(
        self, span: GenerationSpanView, time: datetime.datetime
    ) -> None:
        for _span in self._spans:
            if isinstance(_span, GenerationSpan):
                _span.set_completion_start_time(time)

    @override
    def on_set_model(self, span: GenerationSpanView, name: str) -> None:
        for _span in self._spans:
            if isinstance(_span, GenerationSpan):
                _span.set_model(name)

    @override
    def on_set_model_parameters(
        self, span: GenerationSpanView, params: dict[str, Any]
    ) -> None:
        for _span in self._spans:
            if isinstance(_span, GenerationSpan):
                _span.set_model_parameters(params)

    @override
    def on_add_usage(self, span: GenerationSpanView, usage: dict[str, Any]) -> None:
        for _span in self._spans:
            if isinstance(_span, GenerationSpan):
                _span.add_usage(usage)

    @override
    def on_set_usage(self, span: GenerationSpanView, usage: dict[str, Any]) -> None:
        for _span in self._spans:
            if isinstance(_span, GenerationSpan):
                _span.set_usage(usage)

    @override
    def on_set_cost(self, span: GenerationSpanView, cost: dict[str, float]) -> None:
        for _span in self._spans:
            if isinstance(_span, GenerationSpan):
                _span.set_cost(cost)


class MultiProviderSpanBehavior(SpanBehavior):
    def __init__(self, sinks: list[Provider]):
        self._sinks: list[Provider] = sinks

    @override
    @contextmanager
    def make_span(
        self, kind: SpanKind, base_cls: type[BS], **kwargs: Any
    ) -> Iterator[BS]:
        cms: list[ContextManager[BS]] = [
            getattr(s, kind.value)(**kwargs) for s in self._sinks
        ]

        with ExitStack() as st:
            spans = [st.enter_context(cm) for cm in cms]
            yield base_cls(**kwargs, _span_backend=MultiSpanBackend(spans))
