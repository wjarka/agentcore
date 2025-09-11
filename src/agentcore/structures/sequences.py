from __future__ import annotations

import abc
from collections.abc import (
    MutableSequence,
    Sequence,
)
from typing import (
    Any,
    Generic,
    TypeVar,
    override,
)

ValueT = TypeVar("ValueT")
ValueT_co = TypeVar("ValueT_co", covariant=True)


class SequenceMixin(abc.ABC, Sequence[ValueT_co]):
    @property
    @abc.abstractmethod
    def _datastore(self) -> Sequence[ValueT_co]:
        """Contract: The consuming class must provide a list-like object."""
        raise NotImplementedError

    @override
    def __len__(self) -> int:
        return self._datastore.__len__()

    @override
    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, int | slice):
            return self._datastore[index]


class MutableSequenceMixin(abc.ABC, MutableSequence[ValueT]):
    @property
    @abc.abstractmethod
    def _datastore(self) -> MutableSequence[ValueT]:
        """Contract: The consuming class must provide a list-like object."""
        raise NotImplementedError

    @override
    def __len__(self) -> int:
        return self._datastore.__len__()

    @override
    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, int | slice):
            return self._datastore[index]

    @override
    def __setitem__(self, index: Any, value: Any) -> None:
        self._datastore[index] = value

    @override
    def __delitem__(self, index: Any) -> None:
        del self._datastore[index]

    @override
    def insert(self, index: int, value: ValueT) -> None:
        self._datastore.insert(index, value)


class ItemSequence(MutableSequenceMixin[ValueT], Generic[ValueT]):
    def __init__(self, *, items: list[ValueT] | None = None):
        self._items: list[ValueT] = items or []

    @property
    @override
    def _datastore(self) -> list[ValueT]:
        return self._items
