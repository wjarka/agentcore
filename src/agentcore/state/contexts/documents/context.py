from __future__ import annotations

from collections.abc import Sequence
from typing import override

from agentcore.models import Document
from agentcore.structures import ItemSequence
from agentcore.structures.sequences import SequenceMixin

from ..protocols import DocumentContext
from .models import DocumentMatch, DocumentQuery
from .protocols import DocumentStore
from .stores import InMemoryListStore


class InMemoryDocumentContext(SequenceMixin[Document], DocumentContext):
    def __init__(self, documents: list[Document] | None = None):
        self._documents: ItemSequence[Document] = ItemSequence[Document](
            items=documents or []
        )
        self._stores: dict[str, DocumentStore] = {}

    @property
    @override
    def _datastore(self) -> Sequence[Document]:
        return self._documents

    @override
    def add(self, document: Document) -> None:
        self._documents.append(document)

    # New API stubs to satisfy the protocol; will be implemented in next steps
    @override
    def register_store(self, name: str, store: DocumentStore) -> None:
        self._stores[name] = store

    @override
    def store(self, name: str) -> DocumentStore:
        if name not in self._stores:
            # Lazily create simple in-memory store
            self._stores[name] = InMemoryListStore()
        return self._stores[name]

    @override
    def search(self, query: DocumentQuery) -> list[DocumentMatch]:
        if query.store:
            return self.store(query.store).search(query)
        # Cross-store: aggregate results, simple concat then trim
        matches: list[DocumentMatch] = []
        for store in self._stores.values():
            matches.extend(store.search(query))
            if len(matches) >= query.max_results:
                break
        return matches[: query.max_results]
