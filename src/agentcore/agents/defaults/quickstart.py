from typing import override

from agentcore.agents.base import BaseAgent
from agentcore.agents.protocols import (
    ExecutionStage,
)
from agentcore.di import Injector
from agentcore.models import IS_FINAL_STEP


class QuickStart(BaseAgent):
    def __init__(self, injector: Injector):
        super().__init__(injector)
        self._execution_strategy: ExecutionStage = injector.resolve(ExecutionStage)

    @override
    async def _execute_step(self) -> IS_FINAL_STEP:
        return await self._run_strategy(self._execution_strategy)
