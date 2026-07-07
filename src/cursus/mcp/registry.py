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
        when: One-line "call this when …" cue — the trigger condition, complementing the
            *what* in ``description``. Optional; surfaced by help/describe and exporters.
        examples: Copy-paste invocation strings showing real calls, e.g.
            ``'catalog.list_steps {"framework": "xgboost"}  # every XGBoost step'``. Stored
            as a tuple so this frozen dataclass field is truly immutable (a list could be
            mutated in place); surfaced by help/describe and folded into exported tool
            descriptions so external MCP/OpenAI clients see them too.
    """

    name: str
    description: str
    schema: Dict[str, Any]
    handler: ToolHandler
    destructive: bool = False
    tags: tuple = ()
    when: str = ""
    examples: tuple = ()
    # --- Public-server safety markers (drive gating + MCP tool annotations) ---
    writes: bool = False  # writes to the local filesystem
    exec_code: bool = False  # runs arbitrary code / installs packages
    network: bool = False  # reaches the network / AWS (open-world)

    @property
    def namespace(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def wire_name(self) -> str:
        """The on-the-wire tool name for MCP hosts (dotted ``.`` -> ``__``).

        Host tool-calling APIs (Anthropic ``^[a-zA-Z0-9_-]{1,128}$``, OpenAI
        ``^[a-zA-Z0-9_-]+$``) reject the ``.`` in the internal dotted names, so every
        externally-exposed name uses ``__`` instead. It round-trips unambiguously — no
        tool or namespace name contains a literal ``__`` — via :func:`get_tool`.
        """
        return self.name.replace(".", "__")


# ---------------------------------------------------------------------------
# Registry assembly
# ---------------------------------------------------------------------------

_REGISTRY: Optional[Dict[str, ToolDef]] = None
# name -> one-line namespace purpose, collected from each module's ``NAMESPACE`` constant.
_NAMESPACES: Optional[Dict[str, str]] = None

# The tool modules to import, in registry order. Every ``cursus.mcp.tools.<ns>`` module
# exposes a module-level ``TOOLS: List[ToolDef]`` and (except the meta ``info`` module) a
# ``NAMESPACE: str`` one-liner describing what the namespace is for.
_TOOL_MODULES = [
    "catalog",
    "dag",
    "config",
    "compile",
    "validate",
    "execdoc",
    "pipeline_catalog",
    "project",
    "strategies",
    "steps",
    "author",
    "info",
]

# Namespaces that do NOT get an auto-generated ``<ns>.help`` tool. ``tools`` is the meta
# namespace: ``tools.help`` is already its (hand-written, global) front door, so a
# generated ``tools.help`` would collide with it.
_NO_AUTO_HELP = {"tools"}


def _collect_modules() -> List[Any]:
    """Import every namespace tool module, returning the imported module objects.

    A namespace that fails to import (e.g. an optional engine dependency missing) is
    logged and skipped rather than breaking the whole registry.
    """
    from importlib import import_module

    mods: List[Any] = []
    for ns in _TOOL_MODULES:
        try:
            mods.append(import_module(f"{__package__}.tools.{ns}"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Skipping MCP tool namespace '%s': %s", ns, exc)
            continue
    return mods


def _make_namespace_help_tool(namespace: str, description: str) -> ToolDef:
    """Build the auto-generated ``<namespace>.help`` ToolDef.

    Each namespace's help is the same shape as the global ``tools.help`` but pre-scoped
    to that namespace, so an agent working in (say) the compile space can call
    ``compile.help`` and get just the compile overview + its tools. The handler delegates
    to ``tools.help`` with ``namespace`` pinned, so there is exactly one rendering path.
    """

    def _handler(args: Dict[str, Any]) -> ToolResult:
        merged = dict(args or {})
        merged["namespace"] = namespace
        return call_tool("tools.help", merged)

    return ToolDef(
        name=f"{namespace}.help",
        description=(
            f"Overview of the '{namespace}' tools — {description} Returns each "
            f"{namespace}.* tool with its description, when to use it, and usage examples. "
            f"Set include_schema for full JSON input schemas. Start here before using "
            f"{namespace} tools."
        ),
        schema={
            "type": "object",
            "properties": {
                "phase": {
                    "type": "string",
                    "enum": ["planner", "validator", "programmer"],
                    "description": "Restrict to one lifecycle phase.",
                },
                "include_schema": {
                    "type": "boolean",
                    "description": "Attach each tool's JSON input schema (default false).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_handler,
        tags=("planner",),
        when=f"You are about to work with {namespace} tools and want that namespace's overview + examples.",
        examples=(
            f"{namespace}.help {{}}  # every {namespace}.* tool with when + examples",
            f'{namespace}.help {{"include_schema": true}}  # same, plus JSON input schemas',
        ),
    )


def _build_registry() -> Dict[str, ToolDef]:
    """Assemble the ``name -> ToolDef`` map from all modules, incl. auto ``<ns>.help``."""
    modules = _collect_modules()

    # 1) Collect declared tools and, per module, the tool namespace they belong to.
    #    The namespace key is derived from the tools themselves (the leading segment of
    #    ``td.name``), NOT the module filename — e.g. ``info.py`` defines ``tools.*``.
    registry: Dict[str, ToolDef] = {}
    namespaces: Dict[str, str] = {}
    for mod in modules:
        ns_tools = getattr(mod, "TOOLS", None)
        if not ns_tools:
            logger.warning("MCP tool namespace '%s' exposes no TOOLS", mod.__name__)
            continue
        for td in ns_tools:
            if td.name in registry:
                logger.warning("Duplicate MCP tool name '%s' — keeping first", td.name)
                continue
            registry[td.name] = td

        # Associate this module's NAMESPACE description with the namespace its tools use.
        desc = getattr(mod, "NAMESPACE", None)
        if isinstance(desc, str) and desc.strip():
            ns_key = ns_tools[0].namespace
            namespaces[ns_key] = desc.strip()

    # 3) Auto-generate a ``<ns>.help`` tool for every namespace that declared a
    #    description (except the meta ``tools`` namespace, whose help is hand-written).
    #    Skip if a hand-written ``<ns>.help`` already exists.
    for ns_key, desc in sorted(namespaces.items()):
        if ns_key in _NO_AUTO_HELP:
            continue
        help_name = f"{ns_key}.help"
        if help_name in registry:
            continue
        registry[help_name] = _make_namespace_help_tool(ns_key, desc)

    # Cache the namespace map alongside the registry (they are built together).
    global _NAMESPACES
    _NAMESPACES = namespaces
    return registry


def get_registry(force_reload: bool = False) -> Dict[str, ToolDef]:
    """Return the canonical ``name -> ToolDef`` map, building it once and caching."""
    global _REGISTRY
    if _REGISTRY is None or force_reload:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_namespaces() -> Dict[str, str]:
    """Return the ``namespace -> one-line description`` map (built with the registry)."""
    if _NAMESPACES is None:
        get_registry()
    return dict(_NAMESPACES or {})


def list_tools(namespace: Optional[str] = None) -> List[ToolDef]:
    """List all registered tools, optionally filtered to one namespace."""
    tools = list(get_registry().values())
    if namespace:
        tools = [t for t in tools if t.namespace == namespace]
    return sorted(tools, key=lambda t: t.name)


def get_tool(name: str) -> Optional[ToolDef]:
    """Look up a tool by its internal dotted name OR its on-the-wire ``__`` name."""
    reg = get_registry()
    td = reg.get(name)
    if td is None and "__" in name:
        td = reg.get(name.replace("__", "."))
    return td


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


def render_description(td: ToolDef) -> str:
    """Compose a tool's full-text description: description + 'When' + 'Examples'.

    External MCP/OpenAI clients only receive a single description string per tool, so the
    ``when`` cue and ``examples`` are folded in here (they are otherwise only reachable via
    the in-process help/describe tools). In-process agents get the fields structured; this
    keeps external agents from missing the usage guidance.
    """
    parts = [td.description]
    if td.when:
        parts.append(f"When: {td.when}")
    if td.examples:
        example_lines = "\n".join(f"  - {ex}" for ex in td.examples)
        parts.append(f"Examples:\n{example_lines}")
    return "\n\n".join(parts)


def export_openai_tools(namespace: Optional[str] = None) -> List[Dict[str, Any]]:
    """Export tools in OpenAI / Claude function-calling shape.

    The phase ``tags`` (planner/validator/programmer) are surfaced under the function's
    ``metadata`` so an agent can group/route tools by lifecycle phase rather than scanning
    every description. The ``when``/``examples`` guidance is folded into the description
    (see :func:`render_description`) so external clients see it too.
    """
    out: List[Dict[str, Any]] = []
    for td in list_tools(namespace):
        fn: Dict[str, Any] = {
            "name": td.wire_name,  # host APIs reject the dotted internal name
            "description": render_description(td),
            "parameters": td.schema,
        }
        if td.tags:
            fn["metadata"] = {"tags": list(td.tags)}
        out.append({"type": "function", "function": fn})
    return out


def export_mcp_tools(namespace: Optional[str] = None) -> List[Dict[str, Any]]:
    """Export tools in MCP ``list_tools`` shape (name / description / inputSchema).

    Includes the phase ``tags`` so agents can filter by planner/validator/programmer, and
    folds the ``when``/``examples`` guidance into the description (see
    :func:`render_description`) so external MCP clients see it too.
    """
    out: List[Dict[str, Any]] = []
    for td in list_tools(namespace):
        entry: Dict[str, Any] = {
            "name": td.wire_name,  # host APIs reject the dotted internal name
            "description": render_description(td),
            "inputSchema": td.schema,
        }
        if td.tags:
            entry["tags"] = list(td.tags)
        out.append(entry)
    return out
