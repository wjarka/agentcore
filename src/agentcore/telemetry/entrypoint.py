import datetime
from typing import Any, ContextManager, override

from agentcore.di import global_injector as injector
from agentcore.telemetry.protocols import (
    GenerationSpan,
    Provider,
    Span,
    ToolSpan,
)
from agentcore.telemetry.providers.factory import ProviderFactory


class Telemetry(Provider):
    def __init__(self):
        self._provider: Provider | None = None
        self._factory: ProviderFactory = ProviderFactory()

    def _get_provider(self) -> Provider:
        if self._provider is None:
            self._provider = injector.resolve(Provider)
        return self._provider

    @override
    def span(
        self,
        *,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        status_message: str | None = None,
    ) -> ContextManager[Span]:
        return self._get_provider().span(
            name=name,
            input=input,
            output=output,
            metadata=metadata,
            status_message=status_message,
        )

    @override
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
    ) -> ContextManager[GenerationSpan]:
        return self._get_provider().generation(
            name=name,
            input=input,
            output=output,
            metadata=metadata,
            status_message=status_message,
            completion_start_time=completion_start_time,
            model=model,
            model_parameters=model_parameters,
            usage=usage,
            cost=cost,
        )

    @override
    def tool(
        self,
        *,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        status_message: str | None = None,
    ) -> ContextManager[ToolSpan]:
        return self._get_provider().tool(
            name=name,
            input=input,
            output=output,
            metadata=metadata,
            status_message=status_message,
        )

    @property
    def providers(self) -> ProviderFactory:
        return self._factory


def telemetry() -> Telemetry:
    return injector.resolve(Telemetry)
