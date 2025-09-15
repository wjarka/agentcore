import logging
import os
from collections.abc import Sequence

import jinja2
from openai import AsyncOpenAI

from agentcore.agents.protocols import (
    ActionBuilder,
    ActionIntentBuilder,
    ActionParamBuilder,
    AnswerGenerator,
    ExecutionStage,
)
from agentcore.agents.strategies import (
    DefaultActionBuilder,
    DefaultActionIntentBuilder,
    DefaultActionParamBuilder,
    DefaultAnswerGenerator,
    DefaultExecutionStage,
)
from agentcore.di import Injector, global_injector
from agentcore.log import set_logger
from agentcore.presenters.action import (
    XmlActionPresenter,
)
from agentcore.presenters.document import (
    XmlDocumentPresenter,
)
from agentcore.presenters.environment import (
    PlainEnvironmentPresenter,
)
from agentcore.presenters.message import (
    XmlMessagePresenter,
)
from agentcore.presenters.protocols import (
    ActionPresenter,
    DocumentPresenter,
    EnvironmentPresenter,
    MessagePresenter,
    ToolPresenter,
)
from agentcore.presenters.tool import (
    XmlToolPresenter,
)
from agentcore.prompts.defaults import (
    DefaultAnswerGeneratorPrompt,
    DefaultDataProcessPrompt,
    DefaultThinkPrompt,
    XmlToolBuilderPrompt,
    XmlToolSelectorPrompt,
)
from agentcore.prompts.protocols import (
    AnswerGeneratorPrompt,
    DataProcessPrompt,
    ThinkPrompt,
    ToolBuilderPrompt,
    ToolSelectorPrompt,
)
from agentcore.services import (
    EmbeddingService,
    LLMService,
    TextService,
)
from agentcore.services.embedding import DefaultEmbeddingService
from agentcore.services.openai import OpenAIService
from agentcore.services.text import DefaultTextService
from agentcore.state.contexts import (
    ActionContext,
    ConfigurationContext,
    DocumentContext,
    EnvironmentContext,
    MessageContext,
    ToolContext,
)
from agentcore.state.contexts.documents import InMemoryDocumentContext
from agentcore.state.contexts.environment import PydanticEnvironmentContext
from agentcore.state.contexts.message import InMemoryMessageContext
from agentcore.state.contexts.tool import InMemoryToolContext
from agentcore.state.default import DefaultState
from agentcore.state.protocols import State
from agentcore.telemetry import Telemetry
from agentcore.telemetry import entrypoint as _telemetry
from agentcore.telemetry.protocols import Provider
from agentcore.toolset.base import (
    DefaultAction,
    InMemoryToolRegistry,
)
from agentcore.toolset.protocols import (
    Action,
    ToolRegistry,
)


def bootstrap(
    *,
    telemetry: Sequence[type[Provider] | Provider] | None = None,
    jinja_templates_path: str | None = None,
    logger: logging.Logger | None = None,
):
    # --- Telemetry ---
    injector.bind_singleton(Telemetry)

    # --- Services ---
    injector.bind(AsyncOpenAI, AsyncOpenAI())
    injector.bind_to_instance_of(TextService, DefaultTextService)
    injector.bind_to_instance_of(EmbeddingService, DefaultEmbeddingService)
    injector.bind_to_instance_of(LLMService, OpenAIService)

    # --- Core  Components ---
    injector.bind(State, DefaultState)
    injector.bind(Action, DefaultAction)
    injector.bind(ToolRegistry, InMemoryToolRegistry)

    # --- Core  Prompts ---
    injector.bind(ToolSelectorPrompt, XmlToolSelectorPrompt)
    injector.bind(ToolBuilderPrompt, XmlToolBuilderPrompt)
    injector.bind(AnswerGeneratorPrompt, DefaultAnswerGeneratorPrompt)
    injector.bind(ThinkPrompt, DefaultThinkPrompt)
    injector.bind(DataProcessPrompt, DefaultDataProcessPrompt)

    # --- Core Contexts ---
    injector.bind(ActionContext, InMemoryActionContext)
    injector.bind(ToolContext, InMemoryToolContext)
    injector.bind(ConfigurationContext, InMemoryConfigurationContext)
    injector.bind(DocumentContext, InMemoryDocumentContext)
    injector.bind(EnvironmentContext, PydanticEnvironmentContext)
    injector.bind(MessageContext, InMemoryMessageContext)

    # --- Core Presenters ---
    injector.bind(ActionPresenter, XmlActionPresenter)
    injector.bind(ToolPresenter, XmlToolPresenter)
    injector.bind(DocumentPresenter, XmlDocumentPresenter)
    injector.bind(EnvironmentPresenter, PlainEnvironmentPresenter)
    injector.bind(MessagePresenter, XmlMessagePresenter)

    # --- Core Strategies ---
    injector.bind(AnswerGenerator, DefaultAnswerGenerator)
    injector.bind(ActionIntentBuilder, DefaultActionIntentBuilder)
    injector.bind(ActionParamBuilder, DefaultActionParamBuilder)
    injector.bind(ActionBuilder, DefaultActionBuilder)
    injector.bind(ExecutionStage, DefaultExecutionStage)

    if telemetry is not None:
        if isinstance(telemetry, list):
            provider = _telemetry().providers.multiprovider(telemetry)
        else:
            provider = telemetry

    else:
        provider = _telemetry().providers.noop()
    injector.bind(Provider, provider)
    bind_jinja_environment(jinja_templates_path)
    if logger is not None:
        set_logger(logger)


def bind_jinja_environment(path: str | None = None):
    if path is None:
        path = os.getenv("JINJA_TEMPLATES_PATH")
    loaders: list[jinja2.BaseLoader] = [jinja2.PackageLoader("agentcore", "templates")]
    if path:
        loaders.append(jinja2.FileSystemLoader(path))

    injector.bind(
        jinja2.Environment,
        jinja2.Environment(
            loader=jinja2.ChoiceLoader(loaders=loaders),
            enable_async=True,
        ),
    )


injector: Injector = global_injector
