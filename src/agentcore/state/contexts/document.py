from __future__ import annotations

from collections.abc import Sequence
from typing import override

from agentcore.models import (
    Document,
)
from agentcore.structures import ItemSequence
from agentcore.structures.sequences import SequenceMixin

from .protocols import (
    DocumentContext,
)


class InMemoryDocumentContext(SequenceMixin[Document], DocumentContext):
    def __init__(self, documents: list[Document] | None = None):
        self._documents: ItemSequence[Document] = ItemSequence[Document](
            items=documents or []
        )

    @property
    @override
    def _datastore(self) -> Sequence[Document]:
        return self._documents

    @override
    def add(self, document: Document) -> None:
        self._documents.append(document)
