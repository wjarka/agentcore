from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self, TypeVar, override

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from agentcore.agents.protocols import Agent
from agentcore.di import AsyncCaller, Injector, global_injector
from agentcore.log import logger
from agentcore.models import IS_FINAL_STEP, Document
from agentcore.state.contexts import (
    ActionContext,
    ConfigurationContext,
    DocumentContext,
    EnvironmentContext,
    MessageContext,
    ToolContext,
)
from agentcore.state.contexts.documents import InMemoryListStore
from agentcore.state.protocols import State
from agentcore.telemetry.entrypoint import Telemetry
from agentcore.toolset.protocols import (
    Tool,
    ToolRegistry,
)

from .protocols import (
    AnswerGenerator,
    Strategy,
)

U = TypeVar("U")


class BaseAgent(ABC, Agent):
    def __init__(
        self,
        injector: Injector,
    ):
        self._injector: Injector = injector
        self._state: State = injector.resolve(State)
        self._tool_registry: ToolRegistry = injector.resolve(ToolRegistry)
        self._caller: AsyncCaller = injector.resolve(AsyncCaller)
        self._answer_generator: AnswerGenerator = self._injector.resolve(
            AnswerGenerator
        )
        self._telemetry: Telemetry = injector.resolve(Telemetry)

    async def _run_strategy(self, __strategy: Strategy[U]) -> U:
        return await self._caller.call(__strategy.execute)

    async def generate_answer(self) -> ChatCompletion:
        return await self._run_strategy(self._answer_generator)

    @override
    async def execute(self, **kwargs: Any) -> ChatCompletion:
        with self._telemetry.span(
            name=f"execute() [{self.__class__.__name__}]",
        ):
            await self._execute_loop()
            return await self.generate_answer()

    async def _execute_loop(self):
        for step in range(0, self._state.configuration.max_steps):
            try:
                with self._telemetry.span(name=f"execute_step() [{step}]"):
                    is_final_step = await self._execute_step()
                if is_final_step:
                    break
            except Exception as e:
                logger().exception(e)
            self._state.increment_step()

    @abstractmethod
    async def _execute_step(self) -> IS_FINAL_STEP: ...

    @classmethod
    def _prepare_blueprints(
        cls,
        base_injector: Injector,
        overrides: dict[type, type | object] | None = None,
    ) -> Injector:
        """Hook for registering stateless components (strategies, prompts)."""
        injector = base_injector
        if overrides:
            for abstract, concrete in overrides.items():
                injector.bind(abstract, concrete)
        return injector

    @classmethod
    def _create_singletons(
        cls,
        injector: Injector,
        messages: list[ChatCompletionMessageParam],
        tools: list[Tool],
        documents: list[Document],
        max_steps: int,
    ) -> None:
        """Hook for creating and registering stateful components."""
        injector.bind_singleton(
            ToolRegistry,
            items={tool.name: tool for tool in tools},
        )
        injector.bind(AsyncOpenAI, AsyncOpenAI())
        injector.bind_singleton(ActionContext)
        injector.bind_singleton(ToolContext)
        injector.bind_singleton(EnvironmentContext)
        injector.bind_singleton(MessageContext, messages=messages)
        injector.bind_singleton(ConfigurationContext, max_steps=max_steps)
        injector.bind_singleton(DocumentContext, documents=documents)
        # Wire default document stores
        docs_ctx: DocumentContext = injector.resolve(DocumentContext)
        try:
            docs_ctx.register_store("action_results", InMemoryListStore())
        except Exception:
            # Older implementations may not support register_store yet
            pass
        injector.bind(Injector, injector)
        injector.bind_singleton(AsyncCaller)

        injector.bind_singleton(State)

    @classmethod
    def create(
        cls,
        messages: list[ChatCompletionMessageParam],
        tools: list[Tool] | None = None,
        documents: list[Document] | None = None,
        max_steps: int = 10,
        overrides: dict[type, type | object] | None = None,
    ) -> Self:
        injector = global_injector.create_child()
        injector = cls._prepare_blueprints(injector, overrides)
        cls._create_singletons(
            injector=injector,
            messages=messages,
            tools=tools or [],
            documents=documents or [],
            max_steps=max_steps,
        )
        return cls(injector)
