from __future__ import annotations

import asyncio
import functools
import inspect
import types
from collections.abc import Awaitable, Collection, Hashable, Mapping
from typing import (
    Any,
    Callable,
    ParamSpec,
    TypeVar,
    Union,  # pyright: ignore[reportDeprecated] # For backward compatibility when resolving union types
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from agentcore.log import logger
from agentcore.structures import Registry

T = TypeVar("T")
U = TypeVar("U")


class ResolutionError(TypeError):
    """Raised when the injector cannot resolve a dependency."""

    pass


class TypeMismatchError(TypeError):
    """Raised when a resolved value doesn't match the expected type."""

    def __init__(
        self,
        param_name: str,
        expected_type: Any,
        actual_value: Any,
    ):
        self.param_name: str = param_name
        self.expected_type: Any = expected_type
        self.actual_value: Any = actual_value

        message = (
            f"Parameter '{param_name}' expected type {expected_type}, "
            f"got {type(actual_value).__name__}: {actual_value}"
        )
        super().__init__(message)


P = ParamSpec("P")
R = TypeVar("R")


class Injector(Registry[Hashable, object | type]):
    def __init__(self, enable_type_checking: bool = True):
        super().__init__(key_retriever=lambda obj: type(obj))
        self._enable_type_checking: bool = enable_type_checking

    def create_child(self) -> Injector:
        """Creates a new child injector."""
        injector = Injector(self._enable_type_checking)
        for key, value in self.items():
            injector.set(key, value)
        return injector

    def _is_type_compatible(self, value: Any, expected_type: Any) -> bool:
        """Check if a value is compatible with the expected type."""
        if expected_type is None or expected_type is Any:
            return True

        # Handle None/NoneType
        if value is None:
            origin = get_origin(expected_type)
            if origin in (types.UnionType, Union):  # pyright: ignore[reportDeprecated]
                return types.NoneType in get_args(expected_type)
            return expected_type is types.NoneType

        # Handle Union types (including Optional)
        origin = get_origin(expected_type)
        if origin in (types.UnionType, Union):  # pyright: ignore[reportDeprecated]
            return any(
                self._is_type_compatible(value, arg) for arg in get_args(expected_type)
            )

        # Handle generic types by checking origin
        if origin:
            try:
                return isinstance(value, origin)
            except TypeError:
                # Handle protocols that aren't runtime_checkable
                logger().debug(
                    f"Cannot check isinstance for {origin} (likely non-runtime_checkable protocol)"
                )
                return True  # Skip validation for non-runtime_checkable protocols

        # Handle regular types
        if inspect.isclass(expected_type):
            try:
                return isinstance(value, expected_type)
            except TypeError:
                # Handle protocols that aren't runtime_checkable
                if hasattr(expected_type, "_is_protocol"):
                    logger().debug(
                        f"Cannot check isinstance for {expected_type} (likely non-runtime_checkable protocol)"
                    )
                    return True  # Skip validation for non-runtime_checkable protocols
                raise  # Re-raise if it's not a protocol issue

        return True  # Fallback for complex types we can't easily check

    def _validate_parameter_type(
        self,
        param_name: str,
        value: Any,
        expected_type: Any,
    ) -> None:
        """Validate that a parameter value matches its expected type."""
        if not self._enable_type_checking or expected_type is None:
            return

        if not self._is_type_compatible(value, expected_type):
            raise TypeMismatchError(param_name, expected_type, value)

    def bind(self, abstract: type, to: object | type) -> None:
        """Binds an abstract type to a concrete implementation or instance."""
        # This replaces the overloaded .add() method
        super().set(abstract, to)

    def bind_instance(self, instance: object) -> None:
        """Registers a specific instance."""
        self.bind(type(instance), instance)

    def bind_singleton(self, abstract: type[T], **kwargs: Any) -> None:
        instance = self.resolve(abstract, **kwargs)
        self.bind(abstract, instance)

    def bind_to_instance_of(
        self, abstract: type, concrete: type[T], **kwargs: Any
    ) -> None:
        instance = self.resolve(concrete, **kwargs)
        self.bind(abstract, instance)

    def resolve(self, abstract_type: type[T], **kwargs: Any) -> T:
        instance = self._resolve_type(abstract_type, [], **kwargs)
        if instance is None:
            raise ResolutionError(f"Could not resolve dependency for {abstract_type}")
        return instance

    def _resolve_type(
        self,
        abstract_type: type[T],
        _resolution_stack: list[type],
        **kwargs: Any,
    ) -> T | None:
        """The single, central method for resolving any type."""
        logger().debug(f"Trying to resolve {abstract_type}")

        origin_type = get_origin(abstract_type)
        # --- New: Handle Union types first ---
        if origin_type in (types.UnionType, Union):  # pyright: ignore[reportDeprecated]
            for arg_type in get_args(abstract_type):
                instance = self._resolve_type(arg_type, _resolution_stack, **kwargs)
                if instance is not None:
                    return instance
            return None
        # --- End Union handling ---

        if (
            not abstract_type
            or abstract_type in {str, int, bool, float, Any, types.NoneType}
            or not inspect.isclass(abstract_type)
            or (origin_type and issubclass(origin_type, (Collection, Mapping)))
        ):
            logger().debug(f"Skipping {abstract_type}")
            return None

        # 1. Check for circular dependencies.
        if abstract_type in _resolution_stack:
            logger().debug(f"Skipping {abstract_type} due to circular dependency")
            return None

        # 2. Check for a pre-registered instance or binding.
        result = self.get(abstract_type, None)
        if result is not None:
            if inspect.isclass(result):
                logger().debug(f"Found class {result}")
                # It's an alias; recursively resolve the concrete type.
                return cast(
                    T,
                    self._resolve_type(
                        result, [*_resolution_stack, abstract_type], **kwargs
                    ),
                )
            else:
                logger().debug(f"Found an instance {result}")
                # It's a ready-to-use instance.
                return cast(T, result)

        # 3. If it's a valid, non-registered class, create it.
        if inspect.isclass(abstract_type):
            logger().debug(f"Trying to create {abstract_type}")
            params = self._resolve_params(
                abstract_type.__init__, [*_resolution_stack, abstract_type], **kwargs
            )
            return abstract_type(*params.args, **params.kwargs)

    def resolve_params(
        self,
        target: Callable[..., Any],
        **provided_kwargs: Any,
    ) -> inspect.BoundArguments:
        return self._resolve_params(target, [], **provided_kwargs)

    def _resolve_params(
        self,
        target: Callable[..., Any],
        _resolution_stack: list[type],
        **provided_kwargs: Any,
    ) -> inspect.BoundArguments:
        """Resolves dependencies for any callable."""
        logger().debug(f"Resolving params for {target}. Stack: {_resolution_stack}")
        try:
            type_hints = get_type_hints(target)
        except (NameError, TypeError) as e:
            raise TypeError(f"Failed to get type hints for {target}") from e
        resolved: dict[str, object] = {}
        signature = inspect.signature(target)

        for param in signature.parameters.values():
            logger().debug(f"- Param: {param.name}")
            if param.name in ("self", "cls"):
                logger().debug("Skipped as self/cls")
                continue

            param_type = cast(type[object], type_hints.get(param.name))

            if param.name in provided_kwargs:
                value = provided_kwargs[param.name]
                # Type check provided kwargs
                self._validate_parameter_type(param.name, value, param_type)
                resolved[param.name] = value
                logger().debug("Skipped as provided in kwargs")
                continue

            if isinstance(target, functools.partial) and param.name in target.keywords:
                value = target.keywords[param.name]
                # Type check functools.partial values
                self._validate_parameter_type(param.name, value, param_type)
                logger().debug("Skipped as provided in functools.partial")
                continue

            resolved_type = self._resolve_type(param_type, _resolution_stack)
            if resolved_type is not None:
                # Type check injected dependencies
                self._validate_parameter_type(param.name, resolved_type, param_type)
                resolved[param.name] = resolved_type
                # Check for dependencies specified by the @inject decorator
        try:
            bound_args = signature.bind_partial(**resolved)
        except TypeError as e:
            raise TypeError(f"Failed to bind arguments for {target}: {e}") from e
        return bound_args

    def resolve_class(self, abstract: type[object]) -> type:
        return self._resolve_class_recursive(abstract)

    def _resolve_class_recursive(self, abstract_type: type) -> type:
        # Check for a binding
        concrete = self.get(abstract_type)

        if isinstance(concrete, type):
            # If bound to another class, follow the chain
            return self._resolve_class_recursive(concrete)
        else:
            # If no further binding, this is the final class
            return abstract_type


class AsyncCaller:
    def __init__(self, injector: Injector):
        self._injector: Injector = injector

    async def call(
        self,
        callable_: Callable[..., Awaitable[T] | T],
        **kwargs: object,
    ) -> T:
        """
        Calls any function, method, or constructor, automatically injecting
        dependencies and passing through all other arguments.
        """

        # --- Handle functions, methods, coroutine functions
        if inspect.isfunction(callable_) or inspect.ismethod(callable_):
            target = callable_

        # --- Handle functools.partial (still has __call__)
        elif isinstance(callable_, functools.partial):
            target = callable_.func

        # --- Fallback: callable object (instance with __call__)
        elif callable(callable_):
            target = callable_.__call__
        else:
            raise TypeError(f"Unsupported callable type: {type(callable_)}")  # pyright: ignore[reportUnreachable]

        bound_args = self._injector.resolve_params(target, **kwargs)
        return await self._call(target, bound_args)

    async def _call(
        self,
        method: Callable[..., Awaitable[T] | T],
        bound_args: inspect.BoundArguments,
    ) -> T:
        if inspect.iscoroutinefunction(method):
            method = cast(Callable[..., Awaitable[T]], method)
            return await method(*bound_args.args, **bound_args.kwargs)
        else:
            method = cast(Callable[..., T], method)
            return await asyncio.to_thread(
                method, *bound_args.args, **bound_args.kwargs
            )


global_injector: Injector = Injector()


def set_dependency(abstract: type, dependency: object | type) -> None:
    global_injector.bind(abstract, dependency)
