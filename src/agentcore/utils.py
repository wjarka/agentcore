import json
from asyncio import iscoroutinefunction
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

import yaml
from openai.types.chat import ChatCompletion
from pydantic.types import JsonValue

from .models import ActionResult, Document, Metadata
from .protocols import DocumentProcessor
from .toolset.protocols import FunctionToolCallable


def completion_to_text(completion: ChatCompletion, default: str = "") -> str:
    return completion.choices[0].message.content or default


def completion_to_json(completion: ChatCompletion) -> JsonValue:
    return json.loads(completion_to_text(completion, "{}"))


def completion_to_documents(
    completion: ChatCompletion,
    document_processor: DocumentProcessor,
    additional_metadata: dict[str, Any] | None = None,
) -> list[Document]:
    try:
        json_data = completion_to_json(completion)
        return data_to_documents(json_data, document_processor, additional_metadata)
    except Exception:
        return data_to_documents(
            completion_to_text(completion, "No content."),
            document_processor,
            additional_metadata,
        )


def default_document_processor(
    text: str,
    metadata: dict[str, Any] | None = None,
) -> Document:
    metadata = metadata or {}
    return Document(
        text=text,
        metadata=Metadata(**metadata),
    )


def required[T](t: T | None) -> T:
    if t is None:
        raise ValueError("Required argument is missing")
    return t


def convert_output_to_action_result(
    func: Callable[..., Any],
) -> FunctionToolCallable:
    if iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> ActionResult:
            result = await func(*args, **kwargs)
            return data_to_documents(result)

        return async_wrapper
    else:

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> ActionResult:
            result = func(*args, **kwargs)
            return data_to_documents(result)

        return wrapper


def data_to_documents(
    data: Any,
    document_processor: DocumentProcessor | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    convert_dict_to: Literal["json", "yaml"] = "json",
) -> list[Document]:
    """
    Convert various data types to a list of Document objects.
    Preserves existing Documents as-is.

    Args:
        data: Input data to convert (list, dict, or any other type)
        document_processor: Function that converts text strings to Document objects

    Returns:
        list of Document objects
    """

    document_processor = document_processor or default_document_processor

    if isinstance(data, list):
        return [
            item
            if isinstance(item, Document)
            else document_processor(str(item), metadata=metadata)  # pyright: ignore[reportUnknownArgumentType]
            for item in data  # pyright: ignore[reportUnknownVariableType]
        ]
    elif isinstance(data, dict):
        match convert_dict_to:
            case "json":
                dict_as_str = json.dumps(data)
            case "yaml":
                dict_as_str = yaml.dump(data, default_flow_style=False)
        return [document_processor(dict_as_str, metadata=metadata)]
    else:
        return [document_processor(str(data), metadata=metadata)]
