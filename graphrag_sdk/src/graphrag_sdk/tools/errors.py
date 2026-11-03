# GraphRAG SDK — Tools: errors
# A tool error carries a hint: what the caller's agent should try next.

from __future__ import annotations

from graphrag_sdk.core.exceptions import GraphRAGError


class ToolError(GraphRAGError):
    """A tool declined or could not run the call.

    The message says what went wrong; ``hint`` says what to try instead
    (an exact name to retry with, a tool to call first). MCP returns both
    to the client as an error result so its agent can correct the call.
    """

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} Hint: {self.hint}" if self.hint else base
