"""
cursus.mcp — Cursus as an agentic toolset.

This module exposes the cursus pipeline-engine capabilities as framework-neutral
**tools**: small, typed functions that take JSON-serializable arguments and return a
JSON-serializable :class:`~cursus.mcp.envelope.ToolResult`. The same tool functions
are consumable by any agent framework (MCP, OpenAI function-calling, Claude tool_use,
Bedrock), and a thin optional adapter (:mod:`cursus.mcp.server`) mounts them on an
actual MCP server.

Design (parallel to ``core`` / ``steps`` — it does not replace the human-facing
``cursus.cli``):

- ``envelope``  — the ``ToolResult`` success/error envelope every tool returns.
- ``tools/*``   — one file per namespace (catalog, dag, config, compile, validate,
                  execdoc, pipeline_catalog), each a set of pure tool functions.
- ``schemas``   — not a module; each tool carries its JSON schema via :class:`ToolDef`.
- ``registry``  — the canonical name → :class:`ToolDef` mapping (``catalog.list_steps`` …).
- ``server``    — optional MCP server adapter (imports the ``mcp`` SDK lazily).

Quick start::

    from cursus.mcp import get_registry, call_tool

    reg = get_registry()                       # name -> ToolDef
    reg["catalog.list_steps"].schema           # JSON schema for the tool
    result = call_tool("catalog.list_steps", {"framework": "xgboost"})
    result.ok                                  # True / False
    result.data                                # JSON-serializable payload

The registry is the single source of truth; the MCP server, an OpenAI tool list, and
the CLI (in future) can all be generated from it.
"""

from __future__ import annotations

from .envelope import ToolResult, ToolError
from .registry import (
    ToolDef,
    get_registry,
    get_namespaces,
    list_tools,
    get_tool,
    call_tool,
    render_description,
    export_openai_tools,
    export_mcp_tools,
)

__all__ = [
    "ToolResult",
    "ToolError",
    "ToolDef",
    "get_registry",
    "get_namespaces",
    "list_tools",
    "get_tool",
    "call_tool",
    "render_description",
    "export_openai_tools",
    "export_mcp_tools",
]
