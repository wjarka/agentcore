from __future__ import annotations

from typing import override

from agentcore.models import Document

from .models import DocumentMatch, DocumentQuery
from .protocols import DocumentStore


class InMemoryListStore(DocumentStore):
    def __init__(self):
        self._docs: dict[str, Document] = {}

    @override
    def add(self, document: Document) -> str:
        doc_id = document.metadata.uuid or str(id(document))
        self._docs[doc_id] = document
        return doc_id

    @override
    def get(self, id: str) -> Document | None:
        return self._docs.get(id)

    @override
    def delete(self, id: str) -> bool:
        return self._docs.pop(id, None) is not None

    @override
    def search(self, query: DocumentQuery) -> list[DocumentMatch]:
        text = (query.text or "").lower()
        matches: list[DocumentMatch] = []
        if not text:
            # Return most recent up to max_results
            for doc in list(self._docs.values())[-query.max_results :]:
                matches.append(DocumentMatch(document=doc, score=1.0))
            return matches

        for doc in self._docs.values():
            hay = doc.text.lower()
            if text in hay:
                matches.append(DocumentMatch(document=doc, score=1.0))
                if len(matches) >= query.max_results:
                    break
        return matches
