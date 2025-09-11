from pathlib import PurePath
from typing import Any, override

import jinja2

from agentcore.state.contexts import (
    ActionContext,
)

from .base import BasePresenter
from .protocols import (
    ActionPresenter,
    DocumentPresenter,
)


class XmlActionPresenter(BasePresenter, ActionPresenter):
    @property
    @override
    def _template_path(self) -> PurePath:
        return super()._template_path / "actions"

    def __init__(
        self,
        jinja: jinja2.Environment,
        actions: ActionContext,
        doc_presenter: DocumentPresenter,
    ):
        super().__init__(jinja)
        self._actions: ActionContext = actions
        self._doc_presenter: DocumentPresenter = doc_presenter

    async def _prepare_action_history(self) -> list[dict[str, Any]]:
        result = [
            {
                "name": action.action_name,
                "description": action.action_description,
                "query": action.action_query,
                "params": action.action_params,
                "results": await self._doc_presenter.basic_metadata(
                    documents=action.result, doc_tag="result"
                ),
                "errors": "\n".join(
                    [
                        await self._jinja.get_template(
                            "presenters/simple.jinja"
                        ).render_async(tag="error", value=error)
                        for error in action.errors
                    ]
                ),
            }
            for action in self._actions.history
        ]
        return result

    @override
    async def history_detailed(self) -> str:
        return await self._render(
            template_name="detailed_history.jinja",
            history=await self._prepare_action_history(),
        )

    @override
    async def history_brief(self) -> str:
        return await self._render(
            template_name="brief_history.jinja",
            history=await self._prepare_action_history(),
        )
