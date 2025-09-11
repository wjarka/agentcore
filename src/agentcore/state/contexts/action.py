from __future__ import annotations

from collections.abc import Sequence
from typing import override

from agentcore.models import ActionIntent, ActionTrace
from agentcore.state.contexts.protocols import ActionContext
from agentcore.structures import ItemSequence


class InMemoryActionContext(ActionContext):
    def __init__(self):
        self._history: ItemSequence[ActionTrace] = ItemSequence[ActionTrace]()
        self._current_intent: ActionIntent | None = None
        self._current_trace: ActionTrace | None = None

    @property
    @override
    def history(self) -> Sequence[ActionTrace]:
        return self._history

    @override
    def add_history_trace(self, trace: ActionTrace) -> None:
        self._history.append(trace)

    @override
    def set_current_intent(self, intent: ActionIntent) -> None:
        self._current_intent = intent

    @property
    @override
    def current_intent(self) -> ActionIntent | None:
        return self._current_intent

    @override
    def clear_current_intent(self) -> None:
        self._current_intent = None

    @property
    @override
    def current_trace(self) -> ActionTrace | None:
        return self._current_trace

    @override
    def set_current_trace(self, trace: ActionTrace) -> None:
        self._current_trace = trace

    @override
    def clear_current_trace(self) -> None:
        self._current_trace = None
