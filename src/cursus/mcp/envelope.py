"""
The result envelope every cursus MCP tool returns.

Tools never raise across the tool boundary and never print. They return a
:class:`ToolResult` — a small, JSON-serializable success/error envelope — so that any
agent framework can consume the outcome uniformly. The server adapter and the in-process
:func:`cursus.mcp.registry.call_tool` both rely on this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ToolError(Exception):
    """
    Raised inside a tool to signal a *handled*, user-facing failure.

    The registry's invoker catches this and converts it to ``ToolResult.failure(...)``,
    preserving ``code`` and ``details``. Use this for expected failures (bad input,
    not-found, validation) — let genuinely unexpected exceptions propagate so the
    invoker can wrap them as an ``internal_error``.
    """

    def __init__(
        self,
        message: str,
        code: str = "tool_error",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


@dataclass
class ToolResult:
    """
    Uniform tool outcome.

    Attributes:
        ok: Whether the tool succeeded.
        data: JSON-serializable payload on success (None on error).
        error: Human-readable error message on failure (None on success).
        code: Machine-readable error code on failure (e.g. ``"not_found"``,
            ``"invalid_input"``, ``"internal_error"``).
        warnings: Non-fatal messages an agent may surface or reason about.
        meta: Optional side-band info (tool name, timings, counts) — never required
            for correctness.
    """

    ok: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    code: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # --- constructors ---

    @classmethod
    def success(
        cls,
        data: Any = None,
        warnings: Optional[List[str]] = None,
        **meta: Any,
    ) -> "ToolResult":
        return cls(ok=True, data=data, warnings=list(warnings or []), meta=dict(meta))

    @classmethod
    def failure(
        cls,
        message: str,
        code: str = "tool_error",
        details: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
    ) -> "ToolResult":
        meta: Dict[str, Any] = {}
        if details:
            meta["details"] = details
        return cls(
            ok=False,
            error=message,
            code=code,
            warnings=list(warnings or []),
            meta=meta,
        )

    # --- serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Render to a plain JSON-serializable dict (drops empty optional fields)."""
        out: Dict[str, Any] = {"ok": self.ok}
        if self.ok:
            out["data"] = self.data
        else:
            out["error"] = self.error
            out["code"] = self.code
        if self.warnings:
            out["warnings"] = self.warnings
        if self.meta:
            out["meta"] = self.meta
        return out
