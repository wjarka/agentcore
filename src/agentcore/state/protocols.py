from __future__ import annotations

from typing import Protocol

from .contexts import (
    ActionContext,
    ConfigurationContext,
    DocumentContext,
    EnvironmentContext,
    MessageContext,
    ToolContext,
)


class State(Protocol):
    actions: ActionContext
    messages: MessageContext
    documents: DocumentContext
    configuration: ConfigurationContext
    tools: ToolContext
    environment: EnvironmentContext

    @property
    def current_step(self) -> int: ...

    def increment_step(self) -> None:
        """Increment the step number by one."""
        ...
