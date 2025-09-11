from agentcore.toolset.protocols import Action

from .models import ActionIntent


class AgentCoreException(BaseException):
    pass


class StrategyError(AgentCoreException):
    pass


class ExecutionStrategyError(StrategyError):
    """Raised when a strategy fails to create a ToolIntent."""

    pass


class ActionIntentCreationError(ExecutionStrategyError):
    pass


class ActionBuildingError(ExecutionStrategyError):
    def __init__(
        self,
        message: str,
        intent: ActionIntent | None = None,
    ):
        super().__init__(message)
        self.intent: ActionIntent | None = intent


class ActionExecutionError(ExecutionStrategyError):
    def __init__(self, message: str, action: Action):
        super().__init__(message)
        self.action: Action = action
