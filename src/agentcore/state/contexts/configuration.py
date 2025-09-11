from __future__ import annotations

from typing import override

from agentcore.models import (
    AgentRuntimeConfig,
)

from .protocols import (
    ConfigurationContext,
)


class InMemoryConfigurationContext(ConfigurationContext):
    def __init__(self, max_steps: int):
        self._config: AgentRuntimeConfig = AgentRuntimeConfig(max_steps=max_steps)

    @property
    @override
    def max_steps(self) -> int:
        return self._config.max_steps
