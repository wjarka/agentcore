from __future__ import annotations

from pydantic import BaseModel, Field

from agentcore.models import Document


class DocumentQuery(BaseModel):
    text: str | None = None
    max_results: int = 10
    store: str | None = None


class DocumentMatch(BaseModel):
    document: Document
    score: float = Field(ge=0)

