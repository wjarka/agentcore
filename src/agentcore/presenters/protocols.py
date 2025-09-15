from typing import Protocol, overload

from ..models import ActionIntent, Document


class ToolPresenter(Protocol):
    """Defines the contract for presenting tools."""

    async def list(self) -> str:
        """Renders a simple list of tool names."""
        ...

    async def detailed(self, intent: ActionIntent) -> str:
        """Renders a detailed view of a single tool."""
        ...


class DocumentPresenter(Protocol):
    @overload
    async def full_metadata(
        self, documents: list[Document], doc_tag: str = "document"
    ) -> str: ...

    @overload
    async def full_metadata(
        self, documents: None = None, doc_tag: str = "document", *, store: str
    ) -> str: ...

    @overload
    async def basic_metadata(
        self, documents: list[Document], doc_tag: str = "document"
    ) -> str: ...

    @overload
    async def basic_metadata(
        self, documents: None = None, doc_tag: str = "document", *, store: str
    ) -> str: ...


class ActionPresenter(Protocol):
    async def history_detailed(self) -> str: ...
    async def history_brief(self) -> str: ...


class EnvironmentPresenter(Protocol):
    async def current_datetime(self) -> str: ...
    async def current_date(self) -> str: ...


class MessagePresenter(Protocol):
    async def last_message(self) -> str: ...
