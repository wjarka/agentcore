import functools
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar, cast, override

from agentcore.protocols import Executable
from agentcore.telemetry import entrypoint as telemetry

S = TypeVar("S")


class ObservableExecutable(Executable[S], Protocol):
    observed_execute: Callable[..., Awaitable[S]]


def record_execution(name: str):
    def decorator[T](class_: type[T]) -> type[T]:
        if not issubclass(class_, Executable):
            return class_

        return _decorate_executable(class_, name)

    return decorator


def _decorate_executable[S](
    class_: type[Executable[S]], name: str
) -> type[Executable[S]]:
    _execute = class_.execute

    class Observable(class_):
        observed_execute: Callable[..., Awaitable[Any]] = _execute

        @override
        async def execute(self, **kwargs: Any):
            return await observe_executable(self, name=name)

    _ = functools.update_wrapper(Observable.execute, class_.execute)
    if hasattr(class_.execute, "_dependencies"):
        setattr(Observable, "_dependencies", getattr(class_.execute, "_dependencies"))
    return cast(type[Executable[S]], Observable)


async def observe_executable[T](executable: ObservableExecutable[T], name: str) -> T:
    with telemetry().span(name=name) as span:
        result = await executable.observed_execute()
        _ = span.set_output(output=result)
        return result
