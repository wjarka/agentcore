__all__ = [
    "ActionContext",
    "ConfigurationContext",
    "DocumentContext",
    "MemoryContext",
    "MessageContext",
    "ToolContext",
    "EnvironmentContext",
    # Document memory types
    "DocumentStore",
    "DocumentQuery",
    "DocumentMatch",
    "InMemoryDocumentContext",
]
from .protocols import (
    ActionContext,
    ConfigurationContext,
    DocumentContext,
    EnvironmentContext,
    MessageContext,
    ToolContext,
)

# Backwards/UX alias: expose MemoryContext as an alias of DocumentContext
MemoryContext = DocumentContext

# Re-export document memory types and implementation for discoverability
from .documents import (
    DocumentStore,
    DocumentQuery,
    DocumentMatch,
    InMemoryDocumentContext,
)
