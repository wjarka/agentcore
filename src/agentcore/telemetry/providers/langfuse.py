import datetime
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Generic, TypeVar, cast, override

import langfuse

from agentcore.telemetry.base import (
    BS,
    SpanBackend,
    SpanBehavior,
    SpanKind,
)
from agentcore.telemetry.protocols import (
    GenerationSpanView,
    SpanView,
)

T = TypeVar(
    "T", langfuse.LangfuseSpan, langfuse.LangfuseTool, langfuse.LangfuseGeneration
)


class LangfuseSpanBackend(SpanBackend, Generic[T]):
    def __init__(self, langfuse_span: T):
        self._langfuse_span: T = langfuse_span

    @override
    def on_set_name(self, span: SpanView, name: str) -> None:
        _ = self._langfuse_span.update(name=span.name)

    @override
    def on_set_input(self, span: SpanView, input: Any) -> None:
        _ = self._langfuse_span.update(input=span.input)

    @override
    def on_set_output(self, span: SpanView, output: Any) -> None:
        _ = self._langfuse_span.update(output=span.output)

    @override
    def on_append_output(self, span: SpanView, chunk: Any) -> None:
        _ = self._langfuse_span.update(output=span.output)

    @override
    def on_add_metadata(self, span: SpanView, metadata: dict[str, Any]) -> None:
        _ = self._langfuse_span.update(metadata=span.metadata)

    @override
    def on_set_status_message(self, span: SpanView, message: str) -> None:
        _ = self._langfuse_span.update(status_message=span.status_message)

    @override
    def on_set_completion_start_time(
        self, span: GenerationSpanView, time: datetime.datetime
    ) -> None:
        _ = self._langfuse_span.update(completion_start_time=span.completion_start_time)

    @override
    def on_set_model(self, span: GenerationSpanView, name: str) -> None:
        _ = self._langfuse_span.update(model=span.model)

    @override
    def on_set_model_parameters(
        self, span: GenerationSpanView, params: dict[str, Any]
    ) -> None:
        _ = self._langfuse_span.update(model_parameters=span.model_parameters)

    @override
    def on_set_usage(self, span: GenerationSpanView, usage: dict[str, Any]) -> None:
        _ = self._langfuse_span.update(usage_details=span.usage)

    @override
    def on_set_cost(self, span: GenerationSpanView, cost: dict[str, float]) -> None:
        _ = self._langfuse_span.update(cost_details=span.cost)

    @override
    def on_add_usage(self, span: GenerationSpanView, usage: dict[str, Any]) -> None:
        _ = self._langfuse_span.update(usage_details=span.usage)


class LangfuseSpanBehavior(SpanBehavior):
    def __init__(self):
        self._client: langfuse.Langfuse = langfuse.get_client()

    @override
    @contextmanager
    def make_span(
        self,
        kind: SpanKind,
        base_cls: type[BS],
        **kwargs: Any,
    ) -> Iterator[BS]:
        langfuse_kwargs = {
            key: value for key, value in kwargs.items() if key not in ("usage", "cost")
        }
        langfuse_kwargs["usage_details"] = (
            langfuse_kwargs["usage"] if "usage" in langfuse_kwargs else None
        )
        langfuse_kwargs["cost_details"] = (
            langfuse_kwargs["cost"] if "cost" in langfuse_kwargs else None
        )
        with self._client.start_as_current_observation(
            as_type=kind.value, **langfuse_kwargs
        ) as lf_span:
            match kind:
                case SpanKind.GENERATION:
                    yield base_cls(
                        **kwargs,
                        _span_backend=LangfuseSpanBackend(
                            cast(langfuse.LangfuseGeneration, lf_span)
                        ),
                    )
                case SpanKind.TOOL:
                    yield base_cls(
                        **kwargs,
                        _span_backend=LangfuseSpanBackend(
                            cast(langfuse.LangfuseTool, lf_span)
                        ),
                    )
                case _:
                    yield base_cls(
                        **kwargs,
                        _span_backend=LangfuseSpanBackend(
                            cast(langfuse.LangfuseSpan, lf_span)
                        ),
                    )
