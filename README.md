# AgentCore

AgentCore is a small, batteries-included framework for building LLM-powered agents without a Rube Goldberg machine of abstractions. It focuses on readable code, type-safety, and composability via dependency injection (DI).

- Swappable components everywhere (strategies, prompts, presenters, services, state contexts, telemetry, tools)
- Strongly-typed tools (Pydantic-powered schemas generated for you)
- Prompt building via Jinja templates (for both prompts and presenters)
- Useful telemetry out of the box (concise logger, Langfuse; or both)
- Sensible defaults; easy to override with DI

## Why AgentCore

- Explicit over magical
  - Most things are thin, swappable components wired with DI.
  - Replace just the piece you care about (a prompt, a presenter, a strategy) without forking the world.

- Tools with real schemas
  - Define a tool once; get runtime validation and structured execution.
  - Decorator-based registration and Pydantic type safety, or implement the Tool protocol directly.

- Prompts and presenters, not string soup
  - Prompts as controllers + Jinja templates + small presenter classes that assemble structured context (tools, history, docs, env, messages).
  - Override templates via filesystem or package loaders, or swap the classes via DI.

- Telemetry that helps during dev and scales to prod
  - Logger provider shows a concise, readable flow (short input/output).
  - Langfuse captures full detail. Use both with a simple list.

- Async-first, strongly typed, minimal surface
  - Python 3.13+, Pydantic 2.x, idiomatic protocols for crisp boundaries.

## Install

AgentCore targets Python 3.13+.

Using uv (recommended):

From a local path:

```bash
git clone https://github.com/wjarka/agentcore.git
cd your-project-name
uv add ../agentcore
```

Environment variables you may care about:

- OPENAI_API_KEY – required for the default OpenAI LLM service
- JINJA_TEMPLATES_PATH – optional; point to a folder with your custom templates
- JINA_API_KEY – optional; enables Jina embeddings for EmbeddingService
- Langfuse (optional): LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

## Quick start

This minimal example lets the LLM answer directly (no tools). Great for kicking the tires.

```python
import asyncio
from agentcore import bootstrap, agents
from agentcore.utils import completion_to_text

async def main():
    # Default bootstrap: OpenAI LLM + in-memory state + noop telemetry
    bootstrap()

    # Build an agent from messages (tools/documents are optional)
    agent = agents.defaults.QuickStart.create(
        messages=[{"role": "user", "content": "What's the capital of France?"}],
        tools=[],     # you pass a list of Tool instances here
        documents=[], # optional: pre-seeded documents
        max_steps=3,
    )

    completion = await agent.execute()
    print(completion_to_text(completion))

asyncio.run(main())
```

Output (example):

```
Paris.
```

## Add a tool (strongly typed, schema-validated)

AgentCore tools return documents (list[Document]). If your function returns anything else, the helper below converts outputs to Document(s).

```python
import asyncio
from agentcore import bootstrap, agents, tools
from agentcore.models import ToolParam
from agentcore.utils import convert_output_to_action_result, completion_to_text

# Define a tool using a decorator. The second decorator converts your
# return value into Document(s), so you can return text/dicts/lists.
@tools.wrap_and_register(
    name="echo",
    description="Returns the provided message. Demo tool.",
    parameters={
        "message": ToolParam(type=str, description="What to echo back")
    },
)
@convert_output_to_action_result
def echo(message: str):
    return f"Echo: {message}"

async def main():
    bootstrap()

    # The agent will decide the next step. With this tool available, it can use it.
    agent = agents.defaults.QuickStart.create(
        messages=[{"role": "user", "content": "Use the echo tool to echo 'hello world'"}],
        tools=list(tools.values()),  # pass a list of Tool instances
        max_steps=3,
    )

    completion = await agent.execute()
    print(completion_to_text(completion))

asyncio.run(main())
```

What’s going on:

- `@tools.wrap_and_register(...)` generates a Pydantic schema for params and registers the tool.
- `@convert_output_to_action_result` turns your return value into Document(s).
- The agent’s parent strategy (ExecutionStage) picks the next step (a tool or final answer) and the AnswerGenerator produces the final response.

## Bootstrap and telemetry

`bootstrap(...)` prepares the global injector with sensible defaults (OpenAI LLM, text/embedding services, in-memory state, prompts/presenters, Jinja env, telemetry).

Common setups:

```python
from agentcore import bootstrap
from agentcore.telemetry import entrypoint

# 1) Minimal (noop telemetry)
bootstrap()

# 2) Log telemetry to console (concise, readable)
bootstrap(
    telemetry=entrypoint().providers.logger(
        use_custom_formatting=True,
        logger_name="agentcore.telemetry",
    )
)

# 3) Langfuse telemetry (requires LANGFUSE_* env vars)
bootstrap(
    telemetry=entrypoint().providers.langfuse()
)

# 4) Combine providers by passing a list (auto-multiprovider)
bootstrap(
    telemetry=[
        entrypoint().providers.logger(use_custom_formatting=True),
        entrypoint().providers.langfuse(),
    ]
)

# 5) Load extra Jinja templates from your filesystem directory
bootstrap(jinja_templates_path="./my_templates")
```

Use your own app logger:

```python
import logging
from agentcore import bootstrap

my_logger = logging.getLogger("myapp")
my_logger.setLevel(logging.INFO)
my_logger.addHandler(logging.StreamHandler())

bootstrap(logger=my_logger)
```

## Mental model

The agent loop is intentionally simple:

1. ExecutionStage orchestrates a single step:
   - Builds an action via ActionBuilder, which itself uses:
     - ActionIntentBuilder to choose a tool or “final_answer”
     - ActionParamBuilder to build the parameters for that tool
   - Executes the action and captures results/errors
   - Stores documents and traces in state
   - Decides whether to continue (or stop on final answer)

2. AnswerGenerator crafts the final response using accumulated documents.

Everything is replaceable via DI if you want to change any piece of that flow.

## Key components (where to look)

List by protocols/ABCs (defaults exist but you can swap them):

- Agents
  - `agents.base.BaseAgent` – loop scaffold
  - `agents.defaults.QuickStart` – production-ready minimal agent

- Strategies (the “how” of each step)
  - `agents.protocols.ActionIntentBuilder`
  - `agents.protocols.ActionParamBuilder`
  - `agents.protocols.ActionBuilder`
  - `agents.protocols.ExecutionStage` ← orchestrates build + execute + decide
  - `agents.protocols.AnswerGenerator`

- Tools
  - `toolset.base.FunctionTool` – creates typed tools from callables
  - `toolset.library.tools` – global tool registry you can populate
  - Implement `toolset.protocols.Tool` directly if you prefer full control

- Prompts & Presenters (templated system prompts with structured context)
  - Prompts (Jinja): `templates/prompts/*.jinja`
  - Presenters: small helpers that turn state into strings for prompts
  - Both can be swapped with DI and both use Jinja templates

- Services
  - `services.openai.OpenAIService` – chat completions, vision, transcription
  - `services.embedding.DefaultEmbeddingService` – OpenAI + optional Jina embeddings
  - `services.text.DefaultTextService` – tokenization/splitting, URL/image placeholders

- State
  - In-memory contexts for actions, messages, tools, documents, env, config
  - Crisp protocols to swap out in tests or for persistence

- Telemetry
  - `telemetry.providers.logger` – concise logs (good for dev flow visibility)
  - `telemetry.providers.langfuse` – full detail (great even in dev)
  - Pass a list to `bootstrap(telemetry=[...])` to use both

## DI: what’s swappable and how injection works

DI is the backbone. Most of AgentCore’s classes are constructed via DI, and functions are called through an async injector wrapper (AsyncCaller) that injects by type hints.

What’s swappable via DI:

- Strategies:
  - ActionIntentBuilder, ActionParamBuilder, ActionBuilder, ExecutionStage, AnswerGenerator
- Prompts:
  - ToolSelectorPrompt, ToolBuilderPrompt, AnswerGeneratorPrompt, ThinkPrompt, DataProcessPrompt
- Presenters:
  - ToolPresenter, ActionPresenter, DocumentPresenter, EnvironmentPresenter, MessagePresenter
- Services:
  - LLMService, EmbeddingService, TextService
- State contexts:
  - ActionContext, MessageContext, DocumentContext, ToolContext, EnvironmentContext, ConfigurationContext
- Telemetry:
  - Provider (you can pass a single provider or a list of providers to bootstrap)
- Tooling:
  - ToolRegistry, Action implementation
- Infrastructure:
  - jinja2.Environment, AsyncCaller, app logger, etc.

Constructor injection:

- Most components are constructed via DI. If a constructor has type-annotated parameters, DI will try to resolve and inject them.
- This makes customizing behavior as simple as subclassing and adding dependencies to your constructor.

Function/method injection via AsyncCaller:

- AgentCore calls most “do work” functions through an `AsyncCaller` which resolves parameters by type hints.
- If your callable requires user-supplied parameters (like tool input) and you also want DI, prefer keyword-only args for injected stuff. Example:

```python
from agentcore.telemetry import Telemetry
from agentcore.utils import convert_output_to_action_result
from agentcore.models import ToolParam
from agentcore import tools

@tools.wrap_and_register(
    name="search_and_log",
    description="Demo showing DI-injected params alongside user input",
    parameters={
        "query": ToolParam(type=str, description="Search query")
    },
)
@convert_output_to_action_result
def search_and_log(
    query: str,  # user-provided
    *,           # everything after * is keyword-only (good for DI)
    telemetry: Telemetry,  # injected by type
):
    with telemetry.span(name="custom-tool"):
        # do something with query...
        return f"Searched for: {query}"
```

- Why keyword-only? AsyncCaller binds by parameter names and type hints. Mark DI-only params as keyword-only so the original Protocol is still satisfied (most methods use \*\*kwargs at the end to give you flexibility of adding necessary dependencies).
- You can inject into:
  - Tool callables (as above)
  - Strategy execute() methods
  - Any function/method AgentCore calls through `AsyncCaller`

Injecting via constructor is a preferred method unless you need late resolution.

Type checking:

- The injector validates types against annotations (where feasible). If something doesn’t match, you’ll get a clear error.

## Working with documents

Documents are first-class: tools return them, presenters render them, prompts consume them.

- Turn arbitrary data into Document(s):

  ```python
  from agentcore.utils import data_to_documents

  docs = data_to_documents({"status": "ok", "items": [1,2,3]})
  ```

- Restore placeholders (URLs/images) for final rendering:

  ```python
  from agentcore.services.text import DefaultTextService

  ts = DefaultTextService()
  doc = ts.document("Check [this](https://example.com)")
  restored = ts.restore_placeholders(doc)
  ```

- Split large text into token-bound chunks:
  ```python
  chunks = ts.split(long_text, limit=4000)
  ```

## Template customization

- All system prompts live under `agentcore/templates/prompts/` and presenters’ sub-templates live under `agentcore/templates/presenters/`.
- Provide your own folder via `bootstrap(jinja_templates_path="...")`, or set `JINJA_TEMPLATES_PATH`.
- Use the same filenames/paths to override built-ins, or subclass prompt/presenter classes and bind your versions via DI.

## Telemetry: logger vs Langfuse

- Logger provider logs a short, friendly summary of each span’s input/output. It’s designed to give you flow visibility without drowning you in text. Great for local dev.
- Langfuse captures full details: complete inputs/outputs, usage/cost, hierarchy, and more. Use it even in dev if you want all the detail all the time.
- You can pass both at once: `bootstrap(telemetry=[logger_provider, langfuse_provider])`.
- Future: the logger may gain a more verbose mode.

## LLM providers

- OpenAI is the default for now.
- You can use OpenRouter or LiteLLM to access other models, while still keeping the AgentCore structure.
- Additional native providers may come later, but not in the immediate roadmap.

## Custom tools without Pydantic

If you don’t want `FunctionTool`/Pydantic, just implement the `Tool` protocol directly (and return an `Action` that implements `execute()`). You still get DI and telemetry around execution.

## Overriding behavior (without forking)

Per-agent overrides at creation time:

```python
from agentcore import agents
from agentcore.agents.protocols import ExecutionStage

class MyExecutionStage:
    # implement the ExecutionStage protocol
    async def execute(self, **kwargs):
        # custom orchestration
        return True

agent = agents.defaults.QuickStart.create(
    messages=[{"role": "user", "content": "Hi!"}],
    overrides={ExecutionStage: MyExecutionStage},  # per-agent override via DI
)
```

Global dependency swap:

```python
from agentcore import set_dependency
from agentcore.services import LLMService
from agentcore.services.openai import OpenAIService

# Swap the LLM service globally (before creating agents)
set_dependency(LLMService, OpenAIService)
```

## Troubleshooting

- Auth errors calling LLM
  - Ensure OPENAI_API_KEY is set and accessible to your Python process.

- My tool returns a str/dict and the agent crashes
  - Wrap it with `@convert_output_to_action_result` (or return a list[Document]).

- The tool selector never chooses my tool
  - Make sure you pass the tool instance(s) to the agent (e.g., `tools=list(tools.values())`).
  - Inspect telemetry logs to see what context the selector saw.

- Prompts don’t change when I edit templates
  - If using a filesystem path, ensure `bootstrap(jinja_templates_path=...)` points to the right directory and filenames match built-ins.

## Roadmap-ish

- Better docs around building a custom Agent
- More built-in strategies and more advanced agents
- More built-in tools (web, files, data)
- More services (and adapters for multi-provider setups)
- CLI and examples
- Better docs around testing strategies/components

If you want specific usage examples (e.g., custom strategies, using embeddings, or complex tool schemas), say what you’re building and I’ll add focused snippets.

---

Made with a bias for simple code you can actually read and change.
