from __future__ import annotations

from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from .models import Document


class DocumentProcessor(Protocol):
    def __call__(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> Document: ...


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Executable(Protocol[T_co]):
    async def execute(self, **kwargs: Any) -> T_co: ...
