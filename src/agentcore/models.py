from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from pydantic.types import JsonValue

InputModel = BaseModel
InputModelClass: TypeAlias = type[InputModel]

IS_FINAL_STEP = bool


class Headers(BaseModel):
    h1: list[str] | None = None
    h2: list[str] | None = None
    h3: list[str] | None = None
    h4: list[str] | None = None
    h5: list[str] | None = None
    h6: list[str] | None = None


class Metadata(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")
    tokens: int | None = None
    headers: Headers | None = None
    urls: list[str] | None = Field(default_factory=list)
    images: list[str] | None = Field(default_factory=list)
    source: str | None = None
    mime_type: str | None = None
    name: str | None = None
    source_uuid: str | None = None
    conversation_uuid: str | None = None
    uuid: str | None = None
    duration: float | None = None
    screenshots: list[str] | None = None
    chunk_index: int | None = None
    total_chunks: int | None = None


class Document(BaseModel):
    text: str
    metadata: Metadata


class ActionIntent(BaseModel):
    FINAL_ANSWER_TOOL_NAME: str = "final_answer"

    tool: str
    query: str
    reasoning: str | None = Field(alias="_reasoning", default=None)

    @computed_field
    @property
    def is_final_answer(self) -> bool:
        """Returns True if this intent represents a final answer."""
        return self.tool == self.FINAL_ANSWER_TOOL_NAME


class ActionTrace(BaseModel):
    action_name: str | None = None
    action_query: str | None = None
    action_params: JsonValue = Field(default_factory=dict)
    action_description: str | None = None
    result: list[Document] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def completed(self) -> bool:
        return bool(self.result or self.errors)

    @computed_field
    @property
    def success(self) -> bool:
        return bool(self.result and not self.errors)

    def set_intent(self, intent: ActionIntent):
        self.action_name = intent.tool
        self.action_query = intent.query


class AgentRuntimeConfig(BaseModel):
    max_steps: int


class Environment(BaseModel):
    @computed_field
    @property
    def current_datetime(self) -> datetime:
        return datetime.now()


ActionResult = list[Document]


@dataclass
class Validator:
    """A data structure for defining a Pydantic validator."""

    field: str
    func: Callable[..., Any]


class ToolParam(BaseModel):
    type: Any
    description: str
    default: Any | None = None
    alias: str | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("type")
    @classmethod
    def check_is_valid_type_hint(cls, v: Any) -> Any:
        """
        Checks if the value 'v' can be used as a Pydantic field type.
        It does this by attempting to create a temporary model.
        """
        try:
            # Create a dummy model to test if the type hint is valid
            class _TestModel(BaseModel):  # pyright: ignore[reportUnusedClass]
                field: v  # pyright: ignore[reportInvalidTypeForm]

            # If the model was created without error, the type hint is valid
            return v
        except TypeError:
            # Pydantic raises a TypeError if the hint is invalid
            raise ValueError(f"'{v!r}' is not a valid Pydantic type hint.")
