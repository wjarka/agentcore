from typing import override

from agentcore.presenters.protocols import (
    MessagePresenter,
)
from agentcore.state.contexts import (
    MessageContext,
)


class XmlMessagePresenter(MessagePresenter):
    def __init__(self, messages: MessageContext):
        self._messages: MessageContext = messages

    @override
    async def last_message(self) -> str:
        last_message = self._messages[-1]
        if last_message.get("role") == "system" or not last_message.get(
            "content", None
        ):
            return ""
        content = last_message.get("content")
        if isinstance(content, str):
            return content
        raise TypeError("Only text messages supported as of now.")
