from __future__ import annotations

from agentcore.models import ToolParam
from agentcore.prompts.protocols import ThinkPrompt
from agentcore.services.protocols import LLMService
from agentcore.toolset.library import tools
from agentcore.utils import completion_to_text, convert_output_to_action_result


@tools.wrap_and_register(
    name="think",
    description="Use to take a short pause and think. You can also verify the previous steps or results.",
    parameters={
        "query": ToolParam(
            type=str, description="Describe what you want to think about"
        )
    },
)
@convert_output_to_action_result
async def think(prompt: ThinkPrompt, aiservice: LLMService):
    return completion_to_text(
        await aiservice.completion(system_prompt=prompt, name="Thinking")
    )
