from __future__ import annotations

import abc
from typing import Protocol, TypeVar, runtime_checkable

from openai.types.chat import ChatCompletion
from pydantic.types import JsonValue

from agentcore.models import IS_FINAL_STEP, ActionIntent
from agentcore.protocols import Executable
from agentcore.state.protocols import State
from agentcore.toolset.protocols import Action

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Strategy(Executable[T_co], Protocol): ...


class AnswerGenerator(Strategy[ChatCompletion], abc.ABC): ...


class ActionIntentBuilder(Strategy[ActionIntent], abc.ABC): ...


class ActionParamBuilder(Strategy[JsonValue], abc.ABC): ...


class ActionBuilder(Strategy[Action | None], abc.ABC): ...


class ExecutionStage(Strategy[IS_FINAL_STEP], abc.ABC): ...


class Agent(Executable[ChatCompletion], Protocol):
    _state: State
