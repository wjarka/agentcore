from __future__ import annotations

import copy
from collections.abc import Callable
from types import UnionType
from typing import Any, overload, override

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator
from pydantic.fields import FieldInfo
from pydantic.types import JsonValue

from agentcore.structures import Registry
from agentcore.telemetry.entrypoint import Telemetry

from ..di import AsyncCaller
from ..models import ActionResult, InputModelClass, ToolParam, Validator
from .protocols import (
    Action,
    AdaptableTool,
    FunctionToolCallable,
    ToolRegistry,
    ToolTypes,
)


class FunctionTool(AdaptableTool):
    def __init__(
        self,
        callable_: FunctionToolCallable,
        name: str,
        description: str,
        parameters: dict[str, ToolParam],
        validators: dict[str, Validator] | None = None,
    ):
        self._model_class: InputModelClass = self._create_model_class(
            name, description, parameters, validators
        )
        self._schema: dict[str, Any] = self._model_class.model_json_schema()
        self._callable: FunctionToolCallable = callable_
        self._creation_params: dict[str, Any] = {
            "name": name,
            "description": description,
            "callable_": callable_,
            "parameters": parameters,
            "validators": validators or {},
        }

    @property
    def creation_params(self) -> dict[str, Any]:
        return self._creation_params

    def _create_model_class(
        self,
        name: str,
        description: str,
        parameters: dict[str, ToolParam],
        validators: dict[str, Validator] | None = None,
    ) -> InputModelClass:
        final_parameters: dict[str, tuple[type, FieldInfo]] = {}
        for key, tool_param in parameters.items():
            param = {"description": tool_param.description}
            default = []
            if tool_param.default is not None:
                default = [tool_param.default]
            if tool_param.alias is not None:
                param["alias"] = tool_param.alias
            final_parameters[key] = (tool_param.type, Field(*default, **param))
        args = [name]
        model_class = create_model(
            *args,
            __config__=ConfigDict(
                title=name,
                json_schema_extra={"description": description},
            ),
            __base__=BaseModel,
            __validators__={
                name: field_validator(validator.field)(validator.func)
                for name, validator in validators.items()
            }
            if validators
            else {},
            **(final_parameters),
        )
        return model_class

    @property
    @override
    def name(self) -> str:
        return self._schema.get("title", "")

    @property
    @override
    def description(self) -> str:
        return self._schema.get("description", "")

    def _get_parameters_by_requirement(self, required: bool) -> dict[str, ToolParam]:
        """Helper method to get parameters filtered by requirement status."""
        required_names = set(self._schema.get("required", []))

        params = {
            name: ToolParam(
                description=properties.get("description", ""),
                type=properties.get("type"),
                default=properties.get("default"),
                alias=properties.get("alias", None),
            )
            for name, properties in self._schema.get("properties", {}).items()
            if (name in required_names) == required
        }
        return params

    @property
    @override
    def required_parameters(self) -> dict[str, ToolParam]:
        return self._get_parameters_by_requirement(required=True)

    @property
    @override
    def optional_parameters(self) -> dict[str, ToolParam]:
        return self._get_parameters_by_requirement(required=False)

    @override
    async def prepare_action(
        self, params: JsonValue, *, caller: AsyncCaller, **kwargs: Any
    ) -> Action:
        """Factory method to create a runnable Action."""
        if isinstance(params, str):
            validated_params = self._model_class.model_validate_json(params)
        else:
            validated_params = self._model_class.model_validate(params)
        return DefaultAction(
            tool_name=self.name,
            callable_=self._callable,
            caller=caller,
            validated_params=validated_params,
        )

    @classmethod
    def create(
        cls,
        callable_: FunctionToolCallable,
        name: str,
        description: str,
        parameters: dict[str, ToolParam],
        validators: dict[str, Validator] | None = None,
    ) -> FunctionTool:
        return cls(
            callable_=callable_,
            name=name,
            description=description,
            parameters=parameters,
            validators=validators,
        )

    @override
    def with_name(self, name: str) -> AdaptableTool:
        new_params = copy.deepcopy(self._creation_params)
        new_params["name"] = name
        return FunctionTool.create(**new_params)

    @override
    def with_parameter(
        self, name: str, param_type: type | UnionType, field_info: FieldInfo
    ) -> AdaptableTool:
        new_params = copy.deepcopy(self._creation_params)
        new_params["parameters"][name] = (param_type, field_info)
        return FunctionTool.create(**new_params)

    @override
    def with_validators(self, **validators: Validator) -> AdaptableTool:
        new_params = copy.deepcopy(self._creation_params)
        new_params["validators"].update(validators)
        return FunctionTool.create(**new_params)


class DefaultAction(Action):
    """Represents a single, configured, ready-to-run tool invocation."""

    def __init__(
        self,
        tool_name: str,
        callable_: FunctionToolCallable,
        caller: AsyncCaller,
        validated_params: BaseModel,
    ):
        self._callable: FunctionToolCallable = callable_
        self._caller: AsyncCaller = caller
        self._validated_params: BaseModel = validated_params
        self._tool_name: str = tool_name

    @override
    async def execute(self, *, telemetry: Telemetry, **kwargs: Any) -> ActionResult:
        """Executes the command."""
        input = self._validated_params.model_dump()
        with telemetry.tool(name=self._tool_name, input=input) as tool:
            try:
                result = await self._execute()
                tool.set_output(result)
            except Exception as e:
                tool.set_status_message(str(e))
                raise e
            return result

    async def _execute(self) -> ActionResult:
        return await self._caller.call(
            self._callable, **self._validated_params.model_dump()
        )

    @property
    @override
    def tool_name(self) -> str:
        return self._tool_name

    @property
    @override
    def params(self) -> JsonValue:
        return self._validated_params.model_dump()


class InMemoryToolRegistry(ToolRegistry, Registry[str, ToolTypes]):
    @override
    def _default_key_retriever(self, value: ToolTypes) -> str:
        return value.name

    def register(self, tool: ToolTypes) -> ToolTypes:
        self.add(tool)
        return tool

    @override
    def adaptable(self, name: str) -> AdaptableTool:
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        if not isinstance(tool, AdaptableTool):
            raise ValueError(f"Tool '{name}' is not adaptable")
        return tool

    @overload
    def wrap_and_register(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, ToolParam],
        validators: dict[str, Validator] | None = None,
    ) -> Callable[[FunctionToolCallable], AdaptableTool]:
        """
        Decorator overload for wrapping and registering a function as a tool.

        Use this overload when you want to decorate a function with tool metadata.
        The function will be automatically wrapped into an AdaptableTool and
        registered in the registry.

        Args:
            name: Unique identifier for the tool within the registry.
            description: Human-readable description of what the tool does.
            parameters: Mapping of parameter names to their specifications.
                Use empty dict {} if the tool has no parameters.
            validators: Optional mapping of parameter names to validation functions that will be applied before tool execution.

        Returns:
            A decorator function that accepts a FunctionToolCallable and returns
            the wrapped AdaptableTool after registering it.

        Example:
            @tools.wrap_and_register(
                name="greeting_tool",
                description="Creates a personalized greeting",
                parameters={
                    "name": ToolParam(
                        type=str,
                        description="The name of the person to greet"
                    )
                }
            )
            def create_greeting(name: str) -> str:
                return f"Hello, {name}!"

            # For functions with no parameters, use an empty dict:
            # parameters={}
        """
        ...

    @overload
    def wrap_and_register(
        self,
        callable_: FunctionToolCallable,
        *,
        name: str,
        description: str,
        parameters: dict[str, ToolParam],
        validators: dict[str, Validator] | None = None,
    ) -> AdaptableTool:
        """
        Direct function call overload for wrapping and registering a function as a tool.

        Use this overload when you want to programmatically wrap an existing
        function without using decorator syntax.

        Args:
            callable_: The function to be wrapped as a tool.
            name: Unique identifier for the tool within the registry.
            description: Human-readable description of what the tool does.
            parameters: Mapping of parameter names to their specifications.
                Use empty dict {} if the tool has no parameters.
            validators: Optional mapping of parameter names to validation functions
                that will be applied before tool execution.

        Returns:
            The AdaptableTool instance that wraps the provided function,
            now registered in the registry.

        Example:
            def calculate_area(length: float, width: float) -> float:
                return length * width

            tool = tools.wrap_and_register(
                calculate_area,
                name="area_calculator",
                description="Calculates the area of a rectangle",
                parameters={
                    "length": ToolParam(
                        type=float,
                        description="Length of the rectangle"
                    ),
                    "width": ToolParam(
                        type=float,
                        description="Width of the rectangle"
                    )
                }
            )

            # For functions with no parameters, use an empty dict:
            # parameters={}
        """
        ...

    def wrap_and_register(
        self,
        callable_: FunctionToolCallable | None = None,
        *,
        name: str,
        description: str,
        parameters: dict[str, ToolParam],
        validators: dict[str, Validator] | None = None,
    ) -> AdaptableTool | Callable[[FunctionToolCallable], AdaptableTool]:
        def decorator(func: FunctionToolCallable):
            tool = FunctionTool(
                callable_=func,
                name=name,
                description=description,
                parameters=parameters,
                validators=validators,
            )
            self.add(tool)
            return tool

        if callable_ is not None:
            return decorator(callable_)

        return decorator


# tool_factory = DefaultToolFactory()
