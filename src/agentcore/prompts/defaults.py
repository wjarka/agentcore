from pathlib import PurePath
from typing import override

import jinja2

from agentcore.presenters import (
    ActionPresenter,
    DocumentPresenter,
    EnvironmentPresenter,
    MessagePresenter,
    ToolPresenter,
)
from agentcore.state.contexts import ActionContext

from .base import BaseSystemPrompt
from .protocols import (
    AnswerGeneratorPrompt,
    DataProcessPrompt,
    ThinkPrompt,
    ToolBuilderPrompt,
    ToolSelectorPrompt,
)


@ToolSelectorPrompt.register
class XmlToolSelectorPrompt(BaseSystemPrompt):
    def __init__(
        self,
        jinja: jinja2.Environment,
        tool_presenter: ToolPresenter,
        action_presenter: ActionPresenter,
        environment_presenter: EnvironmentPresenter,
        message_presenter: MessagePresenter,
    ):
        super().__init__(jinja)
        self._tool_presenter: ToolPresenter = tool_presenter
        self._action_presenter: ActionPresenter = action_presenter
        self._environment_presenter: EnvironmentPresenter = environment_presenter
        self._message_presenter: MessagePresenter = message_presenter

    @property
    @override
    def _template_path(self) -> PurePath:
        return super()._template_path / "toolselector.jinja"

    @override
    async def _prepare_vars(self) -> dict[str, str | None]:
        return {
            "current_date": await self._environment_presenter.current_date(),
            "last_message": await self._message_presenter.last_message() or "",
            "tools": await self._tool_presenter.list(),
            "action_history": await self._action_presenter.history_detailed(),
        }


@ToolBuilderPrompt.register
class XmlToolBuilderPrompt(BaseSystemPrompt):
    def __init__(
        self,
        jinja: jinja2.Environment,
        environment_presenter: EnvironmentPresenter,
        tool_presenter: ToolPresenter,
        message_presenter: MessagePresenter,
        action_presenter: ActionPresenter,
        actions: ActionContext,
    ):
        super().__init__(jinja)
        self._environment_presenter: EnvironmentPresenter = environment_presenter
        self._tool_presenter: ToolPresenter = tool_presenter
        self.actions: ActionContext = actions
        self._message_presenter: MessagePresenter = message_presenter
        self._action_presenter: ActionPresenter = action_presenter

    @property
    @override
    def _template_path(self) -> PurePath:
        return super()._template_path / "toolbuilder.jinja"

    @override
    async def _prepare_vars(self) -> dict[str, str | None]:
        if not self.actions.current_intent:
            raise Exception("No current action intention set in the action context.")
        return {
            "current_date": await self._environment_presenter.current_date(),
            "tool": await self._tool_presenter.detailed(self.actions.current_intent),
            "last_message": await self._message_presenter.last_message(),
            "actions": await self._action_presenter.history_brief(),
        }


@AnswerGeneratorPrompt.register
class DefaultAnswerGeneratorPrompt(BaseSystemPrompt):
    def __init__(
        self,
        jinja: jinja2.Environment,
        actions: ActionContext,
        environment: EnvironmentPresenter,
        document_presenter: DocumentPresenter,
    ):
        super().__init__(jinja)
        self._actions: ActionContext = actions
        self._environment_presenter: EnvironmentPresenter = environment
        self._document_presenter: DocumentPresenter = document_presenter

    @property
    @override
    def _template_path(self) -> PurePath:
        return super()._template_path / "answer.jinja"

    @override
    async def _prepare_vars(self) -> dict[str, str | None]:
        query = None
        if self._actions.current_intent:
            query = self._actions.current_intent.query

        return {
            "current_date": await self._environment_presenter.current_date(),
            "documents": await self._document_presenter.full_metadata(),
            "query": query,
        }

    @property
    @override
    def json_mode(self) -> bool:
        return False


@ThinkPrompt.register
class DefaultThinkPrompt(BaseSystemPrompt):
    def __init__(
        self,
        jinja: jinja2.Environment,
        message_presenter: MessagePresenter,
        action_presenter: ActionPresenter,
        actions: ActionContext,
    ):
        super().__init__(jinja)
        self._message_presenter: MessagePresenter = message_presenter
        self._action_presenter: ActionPresenter = action_presenter
        self._actions: ActionContext = actions

    @property
    @override
    def _template_path(self) -> PurePath:
        return super()._template_path / "think.jinja"

    @override
    async def _prepare_vars(self) -> dict[str, str | None]:
        assert self._actions.current_intent
        return {
            "last_message": await self._message_presenter.last_message(),
            "actions": await self._action_presenter.history_detailed(),
            "query": self._actions.current_intent.query,
        }


@DataProcessPrompt.register
class DefaultDataProcessPrompt(BaseSystemPrompt):
    def __init__(
        self, jinja: jinja2.Environment, document_presenter: DocumentPresenter
    ):
        super().__init__(jinja)
        self._document_presenter: DocumentPresenter = document_presenter

    @property
    @override
    def _template_path(self) -> PurePath:
        return super()._template_path / "process_data.jinja"

    @override
    async def _prepare_vars(self) -> dict[str, str | None]:
        return {"documents": await self._document_presenter.full_metadata()}
