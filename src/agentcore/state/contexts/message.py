from __future__ import annotations

from collections.abc import Sequence
from typing import override

from openai.types.chat import ChatCompletionMessageParam

from agentcore.structures import ItemSequence
from agentcore.structures.sequences import SequenceMixin

from .protocols import (
    MessageContext,
)


class InMemoryMessageContext(SequenceMixin[ChatCompletionMessageParam], MessageContext):
    def __init__(self, messages: list[ChatCompletionMessageParam]):
        self._messages: ItemSequence[ChatCompletionMessageParam] = ItemSequence[
            ChatCompletionMessageParam
        ](items=messages)

    @property
    @override
    def _datastore(self) -> Sequence[ChatCompletionMessageParam]:
        return self._messages

    @override
    def add(self, message: ChatCompletionMessageParam):
        self._messages.append(message)
