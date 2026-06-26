"""
``tools.*`` — a small meta/discovery namespace so an agent can find the right cursus tool
without scanning all of them in-context.

- ``tools.by_phase(phase)`` returns the tools tagged for a lifecycle phase
  (``planner`` / ``validator`` / ``programmer``), turning tool selection from "read 39
  descriptions" into a single filtered query.
- ``tools.describe_tool(name)`` returns one tool's full descriptor (description, JSON
  schema, phase tags, destructive flag).

This namespace reads the live :func:`cursus.mcp.registry` — it has no engine dependencies,
so it never fails to import and is cheap to call.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..envelope import ToolResult
from ..registry import ToolDef

# The phase taxonomy already carried by every ToolDef's `tags`.
_KNOWN_PHASES = ("planner", "validator", "programmer")


def _summarize(td: ToolDef) -> Dict[str, Any]:
    """Compact, JSON-safe descriptor for a tool (used by both tools)."""
    return {
        "name": td.name,
        "namespace": td.namespace,
        "description": td.description,
        "tags": list(td.tags),
        "destructive": td.destructive,
    }


def _by_phase(args: Dict[str, Any]) -> ToolResult:
    """List the registered tools tagged for a given lifecycle phase.

    ``phase`` is constrained to ``_KNOWN_PHASES`` by the tool schema's enum (the registry
    validates it before this handler runs), so no further value-checking is needed here.
    """
    # Lazy import avoids a circular import at module load (registry imports the tool
    # modules, including this one, to assemble itself).
    from ..registry import list_tools

    phase = args["phase"]
    matches = [td for td in list_tools() if phase in td.tags]
    data = {
        "phase": phase,
        "count": len(matches),
        "tools": [_summarize(td) for td in matches],
    }
    return ToolResult.success(data)


def _describe_tool(args: Dict[str, Any]) -> ToolResult:
    """Return the full descriptor (schema + tags + flags) for one tool by name."""
    from ..registry import get_tool

    name = args["name"]
    if not isinstance(name, str) or not name.strip():
        return ToolResult.failure(
            "'name' must be a non-empty string", code="invalid_input"
        )

    td = get_tool(name)
    if td is None:
        from ..registry import get_registry

        return ToolResult.failure(
            f"unknown tool: '{name}'",
            code="not_found",
            details={"available": sorted(get_registry().keys())},
            remedy={
                "suggested_tools": ["tools.by_phase"],
                "fix_action": "List tools for a phase (planner/validator/programmer) to "
                "find the correct tool name.",
            },
        )

    descriptor = _summarize(td)
    descriptor["schema"] = td.schema
    return ToolResult.success(descriptor)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[ToolDef] = [
    ToolDef(
        name="tools.by_phase",
        description=(
            "List the cursus MCP tools tagged for a lifecycle phase — 'planner' "
            "(discover/select/assemble), 'validator' (check before building), or "
            "'programmer' (compile/generate). Use this to pick the right tool for the "
            "step you are on instead of scanning every tool."
        ),
        schema={
            "type": "object",
            "properties": {
                "phase": {
                    "type": "string",
                    "enum": list(_KNOWN_PHASES),
                    "description": "Lifecycle phase to filter by.",
                },
            },
            "required": ["phase"],
            "additionalProperties": False,
        },
        handler=_by_phase,
        tags=("planner",),
    ),
    ToolDef(
        name="tools.describe_tool",
        description=(
            "Return the full descriptor for one cursus MCP tool by name: its description, "
            "JSON input schema, phase tags, and whether it is destructive. Use to learn "
            "exactly how to call a tool you discovered via tools.by_phase."
        ),
        schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Dotted tool name, e.g. 'compile.dag'.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=_describe_tool,
        tags=("planner",),
    ),
]
