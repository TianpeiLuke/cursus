"""
The canonical tool registry for ``cursus.mcp``.

Every tool is declared as a :class:`ToolDef` (name, description, JSON schema, handler)
and collected into one registry keyed by dotted name (``"catalog.list_steps"``). The
registry is the single source of truth: the MCP server, an OpenAI/Claude tool list, and
(eventually) the CLI can all be generated from it.

``call_tool`` is the in-process invoker. It enforces the tool contract:
- light JSON-schema validation of arguments (required keys, no unknown keys),
- handlers may raise :class:`ToolError` for handled failures (-> ``ToolResult.error``),
- any other exception is caught and wrapped as an ``internal_error`` ``ToolResult`` so a
  tool call never crashes the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .envelope import ToolResult, ToolError

logger = logging.getLogger(__name__)

# A tool handler takes the validated argument dict and returns a ToolResult.
ToolHandler = Callable[[Dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class ToolDef:
    """
    Declarative definition of one agent-callable tool.

    Attributes:
        name: Dotted, namespaced, unique tool name (e.g. ``"compile.dag"``).
        description: One/two-sentence description for the agent (what + when to call).
        schema: JSON Schema (draft-07 style ``{"type": "object", "properties": {...},
            "required": [...]}``) describing the arguments.
        handler: Callable taking the validated arg dict, returning a :class:`ToolResult`.
        namespace: Leading segment of ``name`` (derived; ``"compile"``).
        destructive: True if the tool mutates external state (e.g. upserts/starts a
            SageMaker pipeline). Agents/servers may gate these behind confirmation.
        tags: Free-form labels (e.g. ``"planner"``, ``"validator"``) for grouping.
    """

    name: str
    description: str
    schema: Dict[str, Any]
    handler: ToolHandler
    destructive: bool = False
    tags: tuple = ()

    @property
    def namespace(self) -> str:
        return self.name.split(".", 1)[0]


# ---------------------------------------------------------------------------
# Registry assembly
# ---------------------------------------------------------------------------

_REGISTRY: Optional[Dict[str, ToolDef]] = None


def _collect_tooldefs() -> List[ToolDef]:
    """
    Import every namespace tool module and collect its ``TOOLS`` list.

    Each ``cursus.mcp.tools.<ns>`` module exposes a module-level ``TOOLS: List[ToolDef]``.
    A namespace that fails to import (e.g. an optional engine dependency missing) is
    logged and skipped rather than breaking the whole registry.
    """
    from importlib import import_module

    namespaces = [
        "catalog",
        "dag",
        "config",
        "compile",
        "validate",
        "execdoc",
        "pipeline_catalog",
        "info",
    ]
    defs: List[ToolDef] = []
    for ns in namespaces:
        try:
            mod = import_module(f"{__package__}.tools.{ns}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Skipping MCP tool namespace '%s': %s", ns, exc)
            continue
        ns_tools = getattr(mod, "TOOLS", None)
        if not ns_tools:
            logger.warning("MCP tool namespace '%s' exposes no TOOLS", ns)
            continue
        defs.extend(ns_tools)
    return defs


def get_registry(force_reload: bool = False) -> Dict[str, ToolDef]:
    """Return the canonical ``name -> ToolDef`` map, building it once and caching."""
    global _REGISTRY
    if _REGISTRY is None or force_reload:
        registry: Dict[str, ToolDef] = {}
        for td in _collect_tooldefs():
            if td.name in registry:
                logger.warning("Duplicate MCP tool name '%s' — keeping first", td.name)
                continue
            registry[td.name] = td
        _REGISTRY = registry
    return _REGISTRY


def list_tools(namespace: Optional[str] = None) -> List[ToolDef]:
    """List all registered tools, optionally filtered to one namespace."""
    tools = list(get_registry().values())
    if namespace:
        tools = [t for t in tools if t.namespace == namespace]
    return sorted(tools, key=lambda t: t.name)


def get_tool(name: str) -> Optional[ToolDef]:
    return get_registry().get(name)


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


def _validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
    """
    Light, dependency-free validation against a JSON-schema object.

    Checks required keys, unknown keys (when ``additionalProperties`` is False), and
    top-level ``enum`` membership. Deliberately not a full validator — the handler is
    the final authority; this just gives the agent fast, clear feedback on obvious
    mistakes.
    """
    errors: List[str] = []
    if schema.get("type") != "object":
        return errors
    props: Dict[str, Any] = schema.get("properties", {})
    required: List[str] = schema.get("required", [])

    for key in required:
        if key not in args or args[key] is None:
            errors.append(f"missing required argument: '{key}'")

    if schema.get("additionalProperties", True) is False:
        for key in args:
            if key not in props:
                errors.append(f"unknown argument: '{key}'")

    for key, value in args.items():
        spec = props.get(key)
        if not spec or value is None:
            continue
        enum = spec.get("enum")
        if enum is not None and value not in enum:
            errors.append(f"argument '{key}' must be one of {enum}, got {value!r}")
    return errors


def call_tool(name: str, args: Optional[Dict[str, Any]] = None) -> ToolResult:
    """
    Invoke a registered tool by name with an argument dict.

    Returns a :class:`ToolResult` in all cases — unknown tool, invalid args, handled
    :class:`ToolError`, or unexpected exception are all converted to error envelopes.
    """
    args = dict(args or {})
    td = get_tool(name)
    if td is None:
        return ToolResult.failure(
            f"unknown tool: '{name}'",
            code="unknown_tool",
            details={"available": sorted(get_registry().keys())},
        )

    schema_errors = _validate_args(td.schema, args)
    if schema_errors:
        return ToolResult.failure(
            f"invalid arguments for '{name}': {'; '.join(schema_errors)}",
            code="invalid_input",
            details={"errors": schema_errors},
        )

    try:
        result = td.handler(args)
        if not isinstance(result, ToolResult):  # defensive: normalize raw returns
            result = ToolResult.success(result)
        result.meta.setdefault("tool", name)
        return result
    except ToolError as te:
        return ToolResult.failure(te.message, code=te.code, details=te.details)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Tool '%s' raised an unexpected exception", name)
        return ToolResult.failure(
            f"tool '{name}' failed: {exc}",
            code="internal_error",
            details={"exception": type(exc).__name__},
        )


# ---------------------------------------------------------------------------
# Exporters — generate framework-specific tool descriptors from the registry
# ---------------------------------------------------------------------------


def export_openai_tools(namespace: Optional[str] = None) -> List[Dict[str, Any]]:
    """Export tools in OpenAI / Claude function-calling shape.

    The phase ``tags`` (planner/validator/programmer) are surfaced under the function's
    ``metadata`` so an agent can group/route tools by lifecycle phase rather than scanning
    every description.
    """
    out: List[Dict[str, Any]] = []
    for td in list_tools(namespace):
        fn: Dict[str, Any] = {
            "name": td.name,
            "description": td.description,
            "parameters": td.schema,
        }
        if td.tags:
            fn["metadata"] = {"tags": list(td.tags)}
        out.append({"type": "function", "function": fn})
    return out


def export_mcp_tools(namespace: Optional[str] = None) -> List[Dict[str, Any]]:
    """Export tools in MCP ``list_tools`` shape (name / description / inputSchema).

    Includes the phase ``tags`` so agents can filter by planner/validator/programmer.
    """
    out: List[Dict[str, Any]] = []
    for td in list_tools(namespace):
        entry: Dict[str, Any] = {
            "name": td.name,
            "description": td.description,
            "inputSchema": td.schema,
        }
        if td.tags:
            entry["tags"] = list(td.tags)
        out.append(entry)
    return out
