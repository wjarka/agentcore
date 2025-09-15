"""Shim module for backward compatibility.

InMemoryDocumentContext has moved to agentcore.state.contexts.documents.
This module re-exports the class to avoid breaking imports during transition.
"""

from .documents import InMemoryDocumentContext  # noqa: F401
