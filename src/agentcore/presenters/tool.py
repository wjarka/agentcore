from pathlib import PurePath
from typing import override

import jinja2

from agentcore.models import ActionIntent
from agentcore.state.contexts import (
    ToolContext,
)

from .base import BasePresenter
from .protocols import (
    ToolPresenter,
)


class XmlToolPresenter(BasePresenter, ToolPresenter):
    def __init__(self, jinja: jinja2.Environment, tools: ToolContext):
        super().__init__(jinja=jinja)
        self._tools: ToolContext = tools

    @property
    @override
    def _template_path(self) -> PurePath:
        return super()._template_path / "tools"

    @override
    async def list(self) -> str:
        return await self._render("list.jinja", tools=self._tools.values())

    @override
    async def detailed(self, intent: ActionIntent) -> str:
        return await self._render(
            "detailed.jinja",
            tool=self._tools.get(intent.tool),
            query=intent.query,
        )
