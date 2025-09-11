from typing import override

from agentcore.presenters.protocols import (
    EnvironmentPresenter,
)
from agentcore.state.contexts import (
    EnvironmentContext,
)


class PlainEnvironmentPresenter(EnvironmentPresenter):
    def __init__(self, environment: EnvironmentContext):
        self._environment: EnvironmentContext = environment

    @override
    async def current_datetime(self) -> str:
        return self._environment.current_datetime.isoformat()

    @override
    async def current_date(self) -> str:
        return self._environment.current_datetime.date().isoformat()
