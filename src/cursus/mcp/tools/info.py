"""
``tools.*`` — a small meta/discovery namespace so an agent can find the right cursus tool
without scanning all of them in-context.

- ``tools.help`` is the front door: one call returns the whole toolset — every namespace,
  its purpose, and each tool's one-line description — so an agent can orient itself before
  doing anything else. Optional ``namespace`` / ``phase`` filters and an ``include_schema``
  flag let it zoom from the full overview down to call-ready detail.
- ``tools.by_phase(phase)`` returns the tools tagged for a lifecycle phase
  (``planner`` / ``validator`` / ``programmer``), turning tool selection from "read 50+
  descriptions" into a single filtered query.
- ``tools.describe_tool(name)`` returns one tool's full descriptor (description, JSON
  schema, phase tags, destructive flag).

This namespace reads the live :func:`cursus.mcp.registry` — it has no engine dependencies,
so it never fails to import and is cheap to call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..envelope import ToolResult
from ..registry import ToolDef

# The meta/discovery namespace's own one-line purpose (collected by the registry like
# every other namespace's NAMESPACE constant).
NAMESPACE = (
    "Meta/discovery over the tool registry itself (help, by_phase, describe_tool)."
)

# The phase taxonomy already carried by every ToolDef's `tags`.
_KNOWN_PHASES = ("planner", "validator", "programmer")

# One-line intent for each lifecycle phase (mirrors the wording in tools.by_phase).
_PHASE_DESCRIPTIONS: Dict[str, str] = {
    "planner": "Discover, select, and assemble — explore steps/configs/DAGs and plan the pipeline.",
    "validator": "Check before building — alignment, dependencies, config/DAG integrity, scripts.",
    "programmer": "Compile and generate — turn a resolved DAG into a pipeline / execution doc.",
}

# Short prose introducing the whole surface — the first thing an agent should read.
_OVERVIEW = (
    "cursus exposes its ML-pipeline engine as a set of namespaced, JSON-in/JSON-out tools. "
    "Work flows through three phases: PLAN (discover steps, assemble a DAG, generate configs), "
    "VALIDATE (check alignment/dependencies/integrity before building), and PROGRAM "
    "(compile the DAG into a SageMaker pipeline and generate the execution document). "
    "Start here, then call <namespace>.help (e.g. compile.help) to drill into one namespace, "
    "tools.describe_tool(name) for a tool's full input schema, or tools.by_phase(phase) to "
    "list the tools for the phase you are on."
)


def _summarize(td: ToolDef, include_schema: bool = False) -> Dict[str, Any]:
    """Compact, JSON-safe descriptor for a tool (shared by every tool in this namespace).

    Always includes ``when`` (call-cue) and ``examples`` (copy-paste invocation strings)
    when present, so an agent reading help/describe knows both *when* and *how* to call
    the tool without a further round-trip.
    """
    out: Dict[str, Any] = {
        "name": td.name,
        "namespace": td.namespace,
        "description": td.description,
        "tags": list(td.tags),
        "destructive": td.destructive,
    }
    if td.when:
        out["when"] = td.when
    if td.examples:
        out["examples"] = list(td.examples)
    if include_schema:
        out["schema"] = td.schema
    return out


def _help(args: Dict[str, Any]) -> ToolResult:
    """Introduce the whole cursus toolset in one call: namespaces, phases, and every tool.

    Optional filters narrow the output without changing its shape:
    - ``namespace`` restricts the listing to one namespace (e.g. ``"compile"``),
    - ``phase`` restricts to one lifecycle phase (``planner``/``validator``/``programmer``),
    - ``include_schema`` (default False) attaches each tool's JSON input schema so the
      result is call-ready without a follow-up ``tools.describe_tool``.
    """
    from ..registry import list_tools, get_registry, get_namespaces

    namespace: Optional[str] = args.get("namespace")
    phase: Optional[str] = args.get("phase")
    include_schema: bool = bool(args.get("include_schema", False))

    all_tools = list_tools()
    ns_descriptions = get_namespaces()
    known_namespaces = sorted({td.namespace for td in all_tools})

    # Validate the namespace filter here (it is intentionally not enum-constrained in the
    # schema so new namespaces need no schema edit) and give an actionable error.
    if namespace is not None and namespace not in known_namespaces:
        return ToolResult.failure(
            f"unknown namespace: '{namespace}'",
            code="not_found",
            details={"available_namespaces": known_namespaces},
            remedy={
                "suggested_tools": ["tools.help"],
                "fix_action": "Call tools.help with no namespace to see the valid namespaces.",
            },
        )

    # Apply filters (phase is enum-validated by the schema before the handler runs).
    selected = all_tools
    if namespace is not None:
        selected = [td for td in selected if td.namespace == namespace]
    if phase is not None:
        selected = [td for td in selected if phase in td.tags]

    # Group the selection by namespace, preserving name order within each group.
    grouped: Dict[str, List[ToolDef]] = {}
    for td in selected:
        grouped.setdefault(td.namespace, []).append(td)

    namespaces_out: List[Dict[str, Any]] = []
    for ns in sorted(grouped):
        namespaces_out.append(
            {
                "namespace": ns,
                "description": ns_descriptions.get(ns, ""),
                "count": len(grouped[ns]),
                "tools": [
                    _summarize(td, include_schema=include_schema)
                    for td in sorted(grouped[ns], key=lambda t: t.name)
                ],
            }
        )

    # Phase counts reflect the current selection so a filtered view stays coherent.
    phases_out: Dict[str, Any] = {}
    for ph in _KNOWN_PHASES:
        phases_out[ph] = {
            "description": _PHASE_DESCRIPTIONS[ph],
            "count": sum(1 for td in selected if ph in td.tags),
        }

    data: Dict[str, Any] = {
        "overview": _OVERVIEW,
        "total_tools": len(get_registry()),
        "shown": len(selected),
        "phases": phases_out,
        "namespaces": namespaces_out,
    }
    if namespace is not None:
        data["filtered_namespace"] = namespace
    if phase is not None:
        data["filtered_phase"] = phase

    next_steps = [
        {
            "tool": "<namespace>.help",
            "when": "You know which namespace you need and want just its tools + examples.",
            "why": "Each namespace has its own help (e.g. compile.help, catalog.help).",
            "args_hint": {"include_schema": False},
        },
        {
            "tool": "tools.describe_tool",
            "when": "You have chosen a tool and need its exact input schema.",
            "why": "Returns the full JSON schema, when-cue, examples, tags, and flags.",
            "args_hint": {"name": "<dotted tool name, e.g. compile.dag>"},
        },
        {
            "tool": "tools.by_phase",
            "when": "You want just the tools for the phase you are currently on.",
            "why": "Filters to planner / validator / programmer tools.",
            "args_hint": {"phase": "planner"},
        },
    ]
    return ToolResult.success(data, next_steps=next_steps)


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
                "suggested_tools": ["tools.help", "tools.by_phase"],
                "fix_action": "Call tools.help for the full tool list, or tools.by_phase "
                "(planner/validator/programmer) to find the correct tool name.",
            },
        )

    return ToolResult.success(_summarize(td, include_schema=True))


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[ToolDef] = [
    ToolDef(
        name="tools.help",
        description=(
            "START HERE. Introduce the entire cursus toolset in one call: a short overview, "
            "the lifecycle phases (planner/validator/programmer), and every tool grouped by "
            "namespace with its one-line description. Optional 'namespace' or 'phase' filters "
            "narrow the listing; set 'include_schema' to also get each tool's JSON input schema. "
            "Use this to orient before picking a tool, then tools.describe_tool for exact args."
        ),
        schema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Restrict to one namespace (e.g. 'catalog', 'compile'). "
                    "Omit to list all namespaces.",
                },
                "phase": {
                    "type": "string",
                    "enum": list(_KNOWN_PHASES),
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
        handler=_help,
        tags=("planner",),
        when="At the start of any task, or whenever you are unsure which cursus tool to use.",
        examples=(
            "tools.help {}  # full overview of every namespace and tool",
            'tools.help {"namespace": "compile"}  # just the compile.* tools',
            'tools.help {"phase": "validator"}  # every validator-phase tool',
            'tools.help {"namespace": "dag", "include_schema": true}  # dag tools + input schemas',
        ),
    ),
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
        when="You know the lifecycle phase you are on and want its tools without the full overview.",
        examples=(
            'tools.by_phase {"phase": "planner"}  # discover/select/assemble tools',
            'tools.by_phase {"phase": "validator"}  # pre-build checks',
            'tools.by_phase {"phase": "programmer"}  # compile/generate tools',
        ),
    ),
    ToolDef(
        name="tools.describe_tool",
        description=(
            "Return the full descriptor for one cursus MCP tool by name: its description, "
            "when-to-use cue, usage examples, JSON input schema, phase tags, and whether it "
            "is destructive. Use to learn exactly how to call a tool you found via tools.help, "
            "a <namespace>.help, or tools.by_phase."
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
        when="You have a specific tool name and need its exact arguments before calling it.",
        examples=(
            'tools.describe_tool {"name": "compile.dag"}  # schema + examples for compile.dag',
            'tools.describe_tool {"name": "catalog.list_steps"}',
        ),
    ),
]
