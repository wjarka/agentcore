from pathlib import PurePath
from typing import override

import jinja2

from agentcore.models import Document
from agentcore.state.contexts import (
    DocumentContext,
)

from .base import BasePresenter
from .protocols import (
    DocumentPresenter,
)


class XmlDocumentPresenter(BasePresenter, DocumentPresenter):
    def __init__(self, jinja: jinja2.Environment, documents: DocumentContext):
        super().__init__(jinja)
        self._documents: DocumentContext = documents

    @property
    @override
    def _template_path(self) -> PurePath:
        return super()._template_path / "documents"

    @override
    async def full_metadata(
        self,
        documents: list[Document] | None = None,
        doc_tag: str = "document",
        *,
        store: str | None = None,
    ) -> str:
        return await self._list("full_metadata.jinja", documents, doc_tag, store)

    @override
    async def basic_metadata(
        self,
        documents: list[Document] | None = None,
        doc_tag: str = "document",
        *,
        store: str | None = None,
    ) -> str:
        return await self._list("basic_metadata.jinja", documents, doc_tag, store)

    async def _list(
        self,
        template_name: str,
        documents: list[Document] | None = None,
        doc_tag: str = "document",
        store: str | None = None,
    ):
        # Enforce either-or: caller must provide documents or a store, not both
        if documents is not None and store is not None:
            raise ValueError("Provide either documents or store, not both")
        if documents is None:
            if store is None:
                raise ValueError(
                    "store must be provided when documents is None"
                )
            docs_to_render = self._documents.store(store).all()
        else:
            docs_to_render = documents
        result = (
            await self._render(template_name, documents=docs_to_render, doc_tag=doc_tag)
        ).strip()
        return result
