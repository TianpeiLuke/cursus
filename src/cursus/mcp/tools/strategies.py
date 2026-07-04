"""
``strategies.*`` — the agent-facing twin of ``cursus strategies`` (FZ 31e1d3b1).

Under the Strategy + Facade design there is no per-step builder class for an agent to introspect —
a step is a *selection* of strategies + knobs the facade binds by ``sagemaker_step_type`` (+
``step_assembly``). These tools expose that selection space so an agent can author a step's registry
row + knobs correctly without reading source:

- ``strategies.list_axes()`` — the routing axes + strategy counts.
- ``strategies.list(axis?)`` — every registered strategy (filter by axis).
- ``strategies.show(name, axis?)`` — one strategy's full descriptor (verb, handler, knobs, presets).
- ``strategies.for_step_type(sagemaker_step_type, step_assembly?)`` — THE high-value call: the
  strategy the facade would bind for a step type, the analogue of "read the builder class".
- ``strategies.knobs(axis, name)`` — just the declarative knobs a strategy accepts.

Every tool reads ``cursus.registry.strategy_registry`` — the same single source the runtime router
reads — so its answers can never drift from what the builder actually does. All read-only.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..envelope import ToolResult
from ..registry import ToolDef

# One-line purpose of this namespace (collected by the registry for strategies.help).
NAMESPACE = "Inspect the builder strategy library — axes and knobs (registry.strategy_registry)."


def _list_axes(args: Dict[str, Any]) -> ToolResult:
    from ...registry import strategy_registry as sr

    rows = sr.list_strategies()
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.axis] = counts.get(r.axis, 0) + 1
    data = {
        "count": len(sr.axes()),
        "axes": [{"axis": a, "strategy_count": counts[a]} for a in sr.axes()],
    }
    return ToolResult.success(data)


def _list(args: Dict[str, Any]) -> ToolResult:
    from ...registry import strategy_registry as sr

    axis = args.get("axis")
    rows = sorted(sr.list_strategies(axis=axis), key=lambda i: (i.axis, i.name))
    data = {
        "axis": axis,
        "count": len(rows),
        "strategies": [sr.strategy_to_dict(i) for i in rows],
    }
    return ToolResult.success(data)


def _show(args: Dict[str, Any]) -> ToolResult:
    from ...registry import strategy_registry as sr

    name = args["name"]
    axis = args.get("axis")
    matches = sr.find_strategies(name, axis=axis)
    if not matches:
        return ToolResult.failure(
            f"no strategy named {name!r}" + (f" on axis {axis!r}" if axis else ""),
            code="not_found",
            details={"available": sorted({i.name for i in sr.list_strategies()})},
            remedy={
                "suggested_tools": ["strategies.list"],
                "fix_action": "Call strategies.list to see registered strategy names.",
            },
        )
    if len(matches) > 1:
        return ToolResult.failure(
            f"{name!r} is ambiguous across axes {[m.axis for m in matches]}; pass 'axis'.",
            code="invalid_input",
            details={"axes": [m.axis for m in matches]},
        )
    return ToolResult.success(sr.strategy_to_dict(matches[0]))


def _for_step_type(args: Dict[str, Any]) -> ToolResult:
    from ...registry import strategy_registry as sr

    step_type = args["sagemaker_step_type"]
    step_assembly = args.get("step_assembly")
    axis, name = sr.axis_name_for_step_type(step_type, step_assembly)
    try:
        info = sr.resolve_strategy(axis, name)
    except sr.NoBuilderError as e:
        return ToolResult.failure(
            str(e),
            code="not_found",
            details={"sagemaker_step_type": step_type, "step_assembly": step_assembly},
            remedy={
                "suggested_tools": ["strategies.list"],
                "fix_action": "This step type binds no construction handler (e.g. Base/Lambda are "
                "builder-less). Call strategies.list to see routable step types.",
            },
        )
    data = {
        "sagemaker_step_type": step_type,
        "step_assembly": step_assembly,
        "resolved_axis": axis,
        "resolved_name": name,
        "strategy": sr.strategy_to_dict(info),
    }
    return ToolResult.success(data)


def _knobs(args: Dict[str, Any]) -> ToolResult:
    from ...registry import strategy_registry as sr

    axis = args["axis"]
    name = args["name"]
    try:
        knobs = sr.knobs_for(axis, name)
    except sr.NoBuilderError as e:
        return ToolResult.failure(
            str(e),
            code="not_found",
            remedy={
                "suggested_tools": ["strategies.list"],
                "fix_action": "Call strategies.list to see valid (axis, name) pairs.",
            },
        )
    data = {
        "axis": axis,
        "name": name,
        "count": len(knobs),
        "knobs": [sr.knob_to_dict(k) for k in knobs],
    }
    return ToolResult.success(data)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[ToolDef] = [
    ToolDef(
        name="strategies.list_axes",
        description=(
            "List the builder strategy library's routing axes (sagemaker_step_type, step_assembly) "
            "and how many strategies each carries. Start here to understand the strategy space."
        ),
        schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=_list_axes,
        tags=("planner",),
        when="Call this first when you want an overview of the strategy space — which routing axes exist and how many strategies each carries.",
        examples=("strategies.list_axes {}  # the routing axes + strategy counts",),
    ),
    ToolDef(
        name="strategies.list",
        description=(
            "List the registered builder strategies (construction-verb handlers + their knobs), "
            "optionally filtered to one routing axis. Each entry is a full strategy descriptor."
        ),
        schema={
            "type": "object",
            "properties": {
                "axis": {
                    "type": "string",
                    "description": "Filter to one axis (e.g. 'sagemaker_step_type', 'step_assembly').",
                },
            },
            "additionalProperties": False,
        },
        handler=_list,
        tags=("planner",),
        when="Call this when you want to enumerate the registered strategies (all axes, or one axis) with their full descriptors.",
        examples=(
            "strategies.list {}  # every registered strategy across all axes",
            'strategies.list {"axis": "sagemaker_step_type"}  # only the step-type strategies (Training, Transform, ...)',
            'strategies.list {"axis": "step_assembly"}  # only the Processing assembly strategies (code, step_args, delegation)',
        ),
    ),
    ToolDef(
        name="strategies.show",
        description=(
            "Return one strategy's full descriptor: verb, handler class, every knob "
            "(name/type/default/required/doc), and preset knobs. Pass 'axis' to disambiguate a "
            "name that exists on more than one axis."
        ),
        schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Strategy name, e.g. 'Training', 'code'.",
                },
                "axis": {
                    "type": "string",
                    "description": "Disambiguating axis (optional).",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=_show,
        tags=("planner",),
        when="Call this when you know a strategy name and want its full descriptor — verb, handler class, and every knob (name/type/default/required/doc).",
        examples=(
            'strategies.show {"name": "Training"}  # full descriptor for the Training strategy',
            'strategies.show {"name": "code", "axis": "step_assembly"}  # disambiguate the Processing "code" assembly strategy by axis',
            'strategies.show {"name": "Transform"}  # the batch-Transform strategy descriptor',
        ),
    ),
    ToolDef(
        name="strategies.for_step_type",
        description=(
            "Given a sagemaker_step_type (and, for Processing, a step_assembly: code | step_args | "
            "delegation), return the strategy the builder facade would bind — handler, verb, preset "
            "knobs, and available knobs. This is the replacement for 'read the builder class': it "
            "tells an agent which strategy + knobs compose a step of this type."
        ),
        schema={
            "type": "object",
            "properties": {
                "sagemaker_step_type": {
                    "type": "string",
                    "description": "The step's SageMaker step type, e.g. 'Training', 'Processing'.",
                },
                "step_assembly": {
                    "type": "string",
                    "description": "Processing sub-discriminator (code | step_args | delegation); "
                    "ignored for non-Processing types. Defaults to 'code'.",
                },
            },
            "required": ["sagemaker_step_type"],
            "additionalProperties": False,
        },
        handler=_for_step_type,
        tags=("planner",),
        when="Call this when authoring a step and you need the strategy + knobs the facade would bind for a given sagemaker_step_type — the replacement for reading the builder class.",
        examples=(
            'strategies.for_step_type {"sagemaker_step_type": "Training"}  # strategy the facade binds for a Training step',
            'strategies.for_step_type {"sagemaker_step_type": "Processing", "step_assembly": "code"}  # Processing via the "code" assembly (script-driven)',
            'strategies.for_step_type {"sagemaker_step_type": "Processing", "step_assembly": "step_args"}  # Processing via the "step_args" assembly',
        ),
    ),
    ToolDef(
        name="strategies.knobs",
        description=(
            "List just the declarative knobs a strategy accepts, identified by (axis, name). Use to "
            "learn which knobs a step's .step.yaml registry block can set."
        ),
        schema={
            "type": "object",
            "properties": {
                "axis": {
                    "type": "string",
                    "description": "Routing axis of the strategy.",
                },
                "name": {
                    "type": "string",
                    "description": "Strategy name on that axis.",
                },
            },
            "required": ["axis", "name"],
            "additionalProperties": False,
        },
        handler=_knobs,
        tags=("planner",),
        when="Call this when you already know the (axis, name) of a strategy and just want the declarative knobs its .step.yaml registry block can set.",
        examples=(
            'strategies.knobs {"axis": "sagemaker_step_type", "name": "Training"}  # knobs the Training strategy accepts',
            'strategies.knobs {"axis": "step_assembly", "name": "code"}  # knobs for the Processing "code" assembly strategy',
        ),
    ),
]
