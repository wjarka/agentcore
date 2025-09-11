from __future__ import annotations

from datetime import datetime
from typing import override

from agentcore.models import (
    Environment,
)

from .protocols import (
    EnvironmentContext,
)


class PydanticEnvironmentContext(EnvironmentContext):
    def __init__(self, environment: Environment):
        self._environment: Environment = environment

    @property
    @override
    def current_datetime(self) -> datetime:
        return self._environment.current_datetime
