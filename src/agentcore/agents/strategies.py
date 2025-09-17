from __future__ import annotations

from typing import Any, TypeVar, override

from openai.types.chat import ChatCompletion
from pydantic.types import JsonValue
from pydantic_core import from_json

from agentcore.di import AsyncCaller
from agentcore.exceptions import (
    ActionBuildingError,
    ActionIntentCreationError,
)
from agentcore.log import logger
from agentcore.models import IS_FINAL_STEP, ActionIntent, ActionTrace
from agentcore.prompts.protocols import (
    AnswerGeneratorPrompt,
    ToolBuilderPrompt,
    ToolSelectorPrompt,
)
from agentcore.services import LLMService
from agentcore.state.contexts import (
    ActionContext,
    DocumentContext,
    MessageContext,
    ToolContext,
)
from agentcore.telemetry.decorators import record_execution
from agentcore.toolset.protocols import Action
from agentcore.utils import completion_to_text

from .protocols import (
    ActionBuilder,
    ActionIntentBuilder,
    ActionParamBuilder,
    ExecutionStage,
    Strategy,
)

T_co = TypeVar("T_co", covariant=True)


@record_execution("Action Intent Builder")
class DefaultActionIntentBuilder(ActionIntentBuilder):
    def __init__(self, prompt: ToolSelectorPrompt, aiservice: LLMService):
        self._prompt: ToolSelectorPrompt = prompt
        self._aiservice: LLMService = aiservice

    @override
    async def execute(self, **kwargs: Any) -> ActionIntent:
        try:
            result: str = completion_to_text(
                await self._aiservice.completion(
                    system_prompt=self._prompt, name="Generating Action Intent"
                )
            )
            intent: ActionIntent = ActionIntent.model_validate_json(result)
            return intent
        except Exception as e:
            logger().exception(e)
            logger().error("Failed to create ToolIntent")
            raise ActionIntentCreationError(f"Failed to create ToolIntent {e}") from e


@record_execution("Action Param Builder")
class DefaultActionParamBuilder(
    ActionParamBuilder,
    Strategy[JsonValue],
):
    def __init__(
        self,
        prompt: ToolBuilderPrompt,
        actions: ActionContext,
        aiservice: LLMService,
    ):
        self._prompt: ToolBuilderPrompt = prompt
        self._actions: ActionContext = actions
        self._aiservice: LLMService = aiservice

    @override
    async def execute(self, **kwargs: Any) -> JsonValue:
        if self._actions.current_intent is None:
            raise ActionBuildingError(
                "Can't build an Action because there is no current ActionIntent available"
            )
        try:
            result = completion_to_text(
                await self._aiservice.completion(
                    system_prompt=self._prompt, name="Generating Action Parameters"
                )
            )
            return from_json(result, allow_partial=True)
        except Exception as e:
            logger().error("Error building Action Object")
            logger().exception(e)
            raise ActionBuildingError(
                f"Error building Action object: {e}", self._actions.current_intent
            ) from e


@record_execution("Action Builder")
class DefaultActionBuilder(ActionBuilder):
    def __init__(
        self,
        tools: ToolContext,
        actions: ActionContext,
        caller: AsyncCaller,
        intent_builder: ActionIntentBuilder,
        param_builder: ActionParamBuilder,
    ):
        self._tools: ToolContext = tools
        self._actions: ActionContext = actions
        self._caller: AsyncCaller = caller
        self._intent_builder: ActionIntentBuilder = intent_builder
        self._param_builder: ActionParamBuilder = param_builder

    async def _build_intent(self) -> ActionIntent:
        intent = await self._caller.call(self._intent_builder.execute)
        self._actions.set_current_intent(intent)
        if self._actions.current_trace is not None:
            self._actions.current_trace.set_intent(intent)
        return intent

    async def _build_params(self) -> JsonValue:
        params = await self._caller.call(self._param_builder.execute)
        if self._actions.current_trace is not None:
            self._actions.current_trace.action_params = params
        return params

    @override
    async def execute(self, **kwargs: Any):
        intent = await self._build_intent()
        if intent.is_final_answer:
            return None
        params = await self._build_params()
        tool = self._tools.get(intent.tool)
        if not tool:
            raise ActionBuildingError(
                "Chosen tool does not exist", self._actions.current_intent
            )
        return await self._caller.call(tool.prepare_action, params=params)


@record_execution("Execution Stage")
class DefaultExecutionStage(ExecutionStage):
    def __init__(
        self,
        caller: AsyncCaller,
        actions: ActionContext,
        action_builder: ActionBuilder,
        documents: DocumentContext,
    ):
        self._caller: AsyncCaller = caller
        self._actions: ActionContext = actions
        self._action_builder: ActionBuilder = action_builder
        self._documents: DocumentContext = documents

    @override
    async def execute(self, **kwargs: Any) -> IS_FINAL_STEP:
        trace = ActionTrace()
        self._actions.set_current_trace(trace)
        try:
            action = await self.build_action()
            if action is None:
                return True
            result = await self._caller.call(action.execute)
            trace.result.extend(result)
        except Exception as e:
            cause: str = f" (Cause: {str(e.__cause__)})" if e.__cause__ else ""
            trace.errors.append(f"{e}{cause}")
            logger().exception(e)
        finally:
            for document in trace.result:
                _ = self._documents.store("action_results").add(document)
            self._actions.clear_current_intent()
            self._actions.clear_current_trace()
            self._actions.add_history_trace(trace)
        return False

    async def build_action(self) -> Action | None:
        return await self._caller.call(self._action_builder.execute)


@record_execution("Answer Generator")
class DefaultAnswerGenerator(Strategy[ChatCompletion]):
    def __init__(
        self,
        prompt: AnswerGeneratorPrompt,
        aiservice: LLMService,
        messages: MessageContext,
    ):
        self._prompt: AnswerGeneratorPrompt = prompt
        self._aiservice: LLMService = aiservice
        self._messages: MessageContext = messages

    @override
    async def execute(self, **kwargs: Any) -> ChatCompletion:
        return await self._aiservice.completion(
            system_prompt=self._prompt, name="Generating Answer"
        )
