from typing import Literal

import requests
from pydantic import HttpUrl, JsonValue

from agentcore.models import ActionResult, Document, Metadata, ToolParam
from agentcore.toolset.library import tools


@tools.wrap_and_register(
    name="web_request",
    description="Send a web request or API request",
    parameters={
        "url": ToolParam(type=HttpUrl, description="The URL to send the request to"),
        "method": ToolParam(
            type=Literal["GET", "POST"],
            description="The HTTP method to use for the request. Only GET and POST are supported.",
        ),
        "payload": ToolParam(
            type=JsonValue,
            description="The data to send in the POST request. Should be a valid JSON object.",
        ),
    },
)
def web_request(
    url: HttpUrl, method: str, payload: JsonValue | None = None
) -> ActionResult:
    match method:
        case "GET":
            response = requests.get(str(url))
        case "POST":
            response = requests.post(str(url), json=payload)
        case _:
            raise ValueError(f"Unsupported method: {method}")
    return [Document(text=response.text, metadata=Metadata(source=str(url)))]
