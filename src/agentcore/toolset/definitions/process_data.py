from __future__ import annotations

from agentcore.models import ToolParam
from agentcore.prompts.protocols import DataProcessPrompt
from agentcore.services.protocols import LLMService
from agentcore.toolset.library import tools
from agentcore.utils import completion_to_json, convert_output_to_action_result


@tools.wrap_and_register(
    name="process_data",
    description="Use to process data or documents. For best results, provide a clear and concise description of the data processing task and focus on one task at the time.",
    parameters={
        "query": ToolParam(
            type=str,
            description="Describe the data processing task. Include any specific requirements or constraints. Provide any clues or hints that may help.",
        )
    },
)
@convert_output_to_action_result
async def process_data(query: str, prompt: DataProcessPrompt, aiservice: LLMService):
    result = completion_to_json(
        await aiservice.completion(
            system_prompt=prompt, user_prompt=query, name="Processing Data"
        )
    )
    if isinstance(result, dict):
        return result.get("result")
    raise ValueError("Completion is not a valid JSON dict")
