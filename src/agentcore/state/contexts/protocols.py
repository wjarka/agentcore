from __future__ import annotations

from datetime import datetime
from typing import (
    Hashable,
    ParamSpec,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from openai.types.chat import ChatCompletionMessageParam

from agentcore.models import ActionIntent, ActionTrace, Document
from agentcore.structures.protocols import Mapping, Sequence
from agentcore.toolset.protocols import Tool

from .documents.models import DocumentMatch, DocumentQuery
from .documents.protocols import DocumentStore

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
P = ParamSpec("P")


IndexT = TypeVar("IndexT", bound=Hashable)
IndexT_co = TypeVar("IndexT_co", covariant=True)
IndexT_contra = TypeVar("IndexT_contra", contravariant=True)
ValueT = TypeVar("ValueT")
ValueT_co = TypeVar("ValueT_co", covariant=True)
ValueT_contra = TypeVar("ValueT_contra", contravariant=True)


@runtime_checkable
class ActionContext(Protocol):
    @property
    def history(self) -> Sequence[ActionTrace]: ...
    def add_history_trace(self, trace: ActionTrace) -> None: ...
    @property
    def current_intent(self) -> ActionIntent | None: ...
    def set_current_intent(self, intent: ActionIntent) -> None: ...
    def clear_current_intent(self) -> None: ...
    @property
    def current_trace(self) -> ActionTrace | None: ...
    def set_current_trace(self, trace: ActionTrace) -> None: ...
    def clear_current_trace(self) -> None: ...


class MessageContext(Sequence[ChatCompletionMessageParam], Protocol):
    def add(self, message: ChatCompletionMessageParam) -> None: ...


class DocumentContext(Sequence[Document], Protocol):
    # Existing simple API (kept for compatibility in this step)
    def add(self, document: Document) -> None: ...

    # New memory-like capabilities (store management and search)
    def register_store(self, name: str, store: DocumentStore) -> None: ...
    def store(self, name: str) -> DocumentStore: ...
    def search(self, query: DocumentQuery) -> list[DocumentMatch]: ...


class ToolContext(Mapping[str, Tool], Protocol): ...


class EnvironmentContext(Protocol):
    @property
    def current_datetime(self) -> datetime: ...


class ConfigurationContext(Protocol):
    """Defines the readable properties for agent configuration."""

    @property
    def max_steps(self) -> int: ...
