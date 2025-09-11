from __future__ import annotations

from collections.abc import Mapping
from typing import override

from agentcore.structures.registry import MappingMixin
from agentcore.toolset.protocols import Tool, ToolRegistry

from .protocols import (
    ToolContext,
)


class InMemoryToolContext(MappingMixin[str, Tool], ToolContext):
    def __init__(self, registry: ToolRegistry):
        self._tools: ToolRegistry = registry

    @property
    @override
    def _datastore(self) -> Mapping[str, Tool]:
        return self._tools

    def add(self, tool: Tool):
        self._tools.add(tool)
