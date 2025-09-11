from __future__ import annotations

from typing import override

from .contexts import (
    ActionContext,
    ConfigurationContext,
    DocumentContext,
    EnvironmentContext,
    MessageContext,
    ToolContext,
)
from .protocols import State


class DefaultState(State):
    def __init__(
        self,
        actions: ActionContext,
        messages: MessageContext,
        documents: DocumentContext,
        configuration: ConfigurationContext,
        tools: ToolContext,
        environment: EnvironmentContext,
    ):
        self.actions: ActionContext = actions
        self.messages: MessageContext = messages
        self.documents: DocumentContext = documents
        self.configuration: ConfigurationContext = configuration
        self.tools: ToolContext = tools
        self.environment: EnvironmentContext = environment
        self._current_step: int = 0

    @property
    @override
    def current_step(self) -> int:
        return self._current_step

    @override
    def increment_step(self) -> None:
        self._current_step += 1
