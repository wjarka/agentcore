from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    id: str
    text: str


class MemoryQuery(BaseModel):
    text: str | None = None
    max_results: int = 10
    store: str | None = None


class MemoryMatch(BaseModel):
    record: MemoryRecord
    score: float = Field(ge=0)
