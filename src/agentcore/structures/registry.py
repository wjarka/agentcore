from __future__ import annotations

import abc
from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    MutableMapping,
)
from typing import (
    Any,
    ClassVar,
    Generic,
    Hashable,
    TypeVar,
    cast,
    overload,
    override,
)

from .protocols import (
    Identifiable,
    SupportsAdding,
    SupportsSetting,
)

IndexT = TypeVar("IndexT", bound=Hashable)
ValueT = TypeVar("ValueT")


class MappingMixin(abc.ABC, Mapping[IndexT, ValueT]):
    @property
    @abc.abstractmethod
    def _datastore(self) -> Mapping[IndexT, ValueT]:
        """Contract: The consuming class must provide a dict-like object."""
        raise NotImplementedError

    @override
    def __getitem__(self, key: IndexT) -> ValueT:
        return self._datastore[key]

    @override
    def __iter__(self) -> Iterator[IndexT]:
        return iter(self._datastore)

    @override
    def __len__(self) -> int:
        return len(self._datastore)


class MutableMappingMixin(abc.ABC, MutableMapping[IndexT, ValueT]):
    """
    A mixin that provides a full MutableMapping implementation by delegating
    all operations to a '_datastore' property that must be implemented
    by the consuming class.
    """

    @property
    @abc.abstractmethod
    def _datastore(self) -> MutableMapping[IndexT, ValueT]:
        """Contract: The consuming class must provide a dict-like object."""
        raise NotImplementedError

    @override
    def __getitem__(self, key: IndexT) -> ValueT:
        return self._datastore[key]

    @override
    def __iter__(self) -> Iterator[IndexT]:
        return iter(self._datastore)

    @override
    def __len__(self) -> int:
        return len(self._datastore)

    @override
    def __setitem__(self, key: IndexT, value: ValueT) -> None:
        self._datastore[key] = value

    @override
    def __delitem__(self, key: IndexT) -> None:
        del self._datastore[key]


class Registry(
    MutableMappingMixin[IndexT, ValueT],
    Generic[IndexT, ValueT],
    SupportsAdding[IndexT, ValueT],
    SupportsSetting[IndexT, ValueT],
):
    key_retriever: ClassVar[Callable[[object, Any], Any] | None] = None

    def __init__(
        self,
        key_retriever: Callable[[ValueT], IndexT] | None = None,
        *,
        items: dict[IndexT, ValueT] | None = None,
    ):
        self._items: dict[IndexT, ValueT] = items or {}
        self._key_retriever: Callable[[ValueT], IndexT]
        if key_retriever is not None:
            self._key_retriever = key_retriever
        elif self.key_retriever is not None:
            self._key_retriever = cast(Callable[[ValueT], IndexT], self.key_retriever)
        else:
            self._key_retriever = self._default_key_retriever

    def _default_key_retriever(self, value: ValueT) -> IndexT:
        """
        The default strategy for finding a key.
        Checks if the object conforms to the Identifiable protocol.
        """
        if isinstance(value, Identifiable):
            return cast(Identifiable[IndexT], value).get_unique_identifier()
        raise TypeError(
            """Cannot infer index. Provide an explicit index, a key_retriever, or implement the Identifiable protocol."""
        )

    @property
    @override
    def _datastore(self) -> dict[IndexT, ValueT]:
        return self._items

    @overload
    def add(self, index: IndexT, value: ValueT, /) -> None:
        """
        Overload 1: Register a value with a specific index.
        """
        ...

    @overload
    def add(self, value: ValueT, /) -> None:
        """
        Overload 2: Register a value with an inferred index.
        """
        ...

    @override
    def add(self, *args: Any):
        """
        Registers a value. The implementation uses `Any` and `cast` to correctly
        handle the two distinct overload signatures.
        """
        self._set(*args, _warn_on_overwrite=True)

    @overload
    def set(self, index: IndexT, value: ValueT, /) -> None:
        """
        Overload 1: Register a value with a specific index.
        """
        ...

    @overload
    def set(self, value: ValueT, /) -> None:
        """
        Overload 2: Register a value with an inferred index.
        """
        ...

    @override
    def set(self, *args: Any):
        self._set(*args)

    def _set(self, *args: Any, _warn_on_overwrite: bool = False):
        """
        Registers a value. The implementation uses `Any` and `cast` to correctly
        handle the two distinct overload signatures.
        """
        index: IndexT
        value: ValueT

        if len(args) == 2:
            index = args[0]
            value = args[1]
        elif len(args) == 1:
            index = self._key_retriever(args[0])
            value = cast(ValueT, args[0])
        else:
            raise TypeError("register() takes 1 or 2 arguments")

        if _warn_on_overwrite and index in self._items:
            print(f"⚠️ Warning: Overwriting existing registration for index '{index}'")

        self._items[index] = value

    def get_or_fail(self, key: IndexT) -> ValueT:
        """
        Gets an item by its key, raising a KeyError if it's not found.
        """
        item = self.get(key)
        if item is None:
            raise KeyError(f"Item with key '{key}' not found in registry.")
        return item
