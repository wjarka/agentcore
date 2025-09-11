import logging

from agentcore.telemetry.base import BaseProvider
from agentcore.telemetry.protocols import Provider
from agentcore.telemetry.providers.logger import IndentedLoggerBehavior
from agentcore.telemetry.providers.multi import MultiProviderSpanBehavior

from .langfuse import LangfuseSpanBehavior


class ProviderFactory:
    def langfuse(self) -> Provider:
        return BaseProvider(span_behavior=LangfuseSpanBehavior())

    def logger(
        self,
        logger: logging.Logger | None = None,
        max_text_length: int = 200,
        use_custom_formatting: bool = False,
        logger_name: str = "telemetry",
    ) -> Provider:
        return BaseProvider(
            provider_behavior=IndentedLoggerBehavior(
                logger=logger,
                max_text_length=max_text_length,
                use_custom_formatting=use_custom_formatting,
                logger_name=logger_name,
            )
        )

    def multiprovider(self, providers: list[Provider]) -> Provider:
        return BaseProvider(span_behavior=MultiProviderSpanBehavior(providers))

    def noop(self) -> Provider:
        return BaseProvider()
