"""
cursus.mcp tools for the ``config.*`` namespace.

Schema-driven configuration *introspection* for a pipeline DAG. These tools wrap the
interactive ``cursus.api.factory.DAGConfigFactory`` and the field-extraction utilities in
``cursus.api.factory.field_extractor``, plus the config (de)serialization helpers in
``cursus.core.config_fields``. The factory is inherently stateful (it accumulates base +
per-step config across calls), so this surface deliberately exposes only the *stateless*,
JSON-clean operations: building a factory from a DAG to read out its field requirements,
extracting requirements for a single named config class, and summarizing a saved config
file. Operations that require live, in-process Pydantic config *objects*
(``merge_and_save_configs``) cannot be driven from a JSON tool boundary and return an
explanatory error instead of pretending to work.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..envelope import ToolResult, ToolError
from ..registry import ToolDef
from .shared import resolve_dag as _build_dag  # canonical DAG resolver


# ---------------------------------------------------------------------------
# Internal helpers (no engine imports at module scope — keep those lazy)
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """
    Best-effort coercion of an engine value into something JSON-serializable.

    Field requirement dicts can carry defaults that are enums, ``Path`` objects, or
    arbitrary objects; we never let those leak into ``ToolResult.data``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    # Enum -> its value; everything else -> str() (e.g. Path, factory sentinel objects)
    enum_value = getattr(value, "value", None)
    if enum_value is not None and not callable(enum_value):
        return _json_safe(enum_value)
    return str(value)


def _clean_requirements(reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coerce a list of field-requirement dicts into JSON-safe form."""
    cleaned: List[Dict[str, Any]] = []
    for req in reqs or []:
        cleaned.append(
            {
                "name": req.get("name"),
                "type": req.get("type"),
                "description": req.get("description"),
                "required": bool(req.get("required")),
                "default": _json_safe(req.get("default")),
            }
        )
    return cleaned


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _requirements(args: Dict[str, Any]) -> ToolResult:
    """
    Build a DAGConfigFactory for the given DAG and read out every field requirement.

    Returns base-pipeline-config requirements, base-processing-config requirements (if any
    step needs them), and per-step (non-inherited) requirements, plus the node->config-class
    map and the list of steps that still require user input.
    """
    dag = _build_dag(args)

    from ...api.factory.dag_config_factory import DAGConfigFactory

    try:
        factory = DAGConfigFactory(dag)
    except Exception as exc:  # construction analyzes the DAG; surface failures cleanly
        raise ToolError(
            f"failed to analyze DAG for configuration requirements: {exc}",
            code="invalid_input",
            details={"exception": type(exc).__name__},
        )

    config_class_map = {
        node: cls.__name__ for node, cls in factory.get_config_class_map().items()
    }

    # Per-step requirements (cached during factory init).
    step_requirements: Dict[str, List[Dict[str, Any]]] = {}
    warnings: List[str] = []
    for node in config_class_map:
        try:
            step_requirements[node] = _clean_requirements(
                factory.get_step_requirements(node)
            )
        except (
            Exception
        ) as exc:  # defensive: one bad step shouldn't kill the whole tool
            step_requirements[node] = []
            warnings.append(f"could not extract requirements for step '{node}': {exc}")

    unmapped = [n for n in dag.nodes if n not in config_class_map]
    if unmapped:
        warnings.append(
            "no config class resolved for DAG node(s): " + ", ".join(unmapped)
        )

    data = {
        "base_config_requirements": _clean_requirements(
            factory.get_base_config_requirements()
        ),
        "base_processing_config_requirements": _clean_requirements(
            factory.get_base_processing_config_requirements()
        ),
        "needs_processing_config": bool(
            factory.get_base_processing_config_requirements()
        ),
        "config_class_map": config_class_map,
        "step_requirements": step_requirements,
        "pending_steps": list(factory.get_pending_steps()),
    }
    return ToolResult.success(
        data,
        warnings=warnings,
        node_count=len(dag.nodes),
        mapped_count=len(config_class_map),
    )


def _field_info(args: Dict[str, Any]) -> ToolResult:
    """
    Extract field requirements for a single named config class.

    Resolves ``config_class`` (e.g. ``"TabularPreprocessingConfig"``) against the discovered
    config classes, then returns its field requirements. With ``categorized=true`` the result
    is split into required/optional groups.
    """
    class_name = args.get("config_class")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ToolError(
            "'config_class' must be a non-empty class name", code="invalid_input"
        )
    class_name = class_name.strip()
    categorized = bool(args.get("categorized", False))

    from ...core.config_fields.unified_config_manager import get_unified_config_manager

    try:
        manager = get_unified_config_manager()
        config_classes = manager.get_config_classes()
    except Exception as exc:
        raise ToolError(
            f"failed to discover config classes: {exc}",
            code="internal_error",
            details={"exception": type(exc).__name__},
        )

    config_class = config_classes.get(class_name)
    if config_class is None:
        return ToolResult.failure(
            f"config class '{class_name}' not found",
            code="not_found",
            details={"available": sorted(config_classes.keys())},
        )

    from ...api.factory.field_extractor import (
        extract_field_requirements,
        categorize_field_requirements,
    )

    requirements = _clean_requirements(extract_field_requirements(config_class))

    if categorized:
        # categorize_field_requirements groups the *cleaned* dicts by 'required'.
        grouped = categorize_field_requirements(requirements)
        data: Dict[str, Any] = {
            "config_class": class_name,
            "required": grouped.get("required", []),
            "optional": grouped.get("optional", []),
        }
    else:
        data = {"config_class": class_name, "requirements": requirements}

    return ToolResult.success(data, field_count=len(requirements))


def _load(args: Dict[str, Any]) -> ToolResult:
    """
    Load a saved merged-config JSON file and return a JSON-safe summary of its contents.

    Returns the shared field names and per-step (specific) field names/counts rather than the
    full values, which keeps the payload small and avoids leaking non-serializable objects.
    """
    path = args.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ToolError("'path' must be a non-empty file path", code="invalid_input")
    path = path.strip()

    from ...core.config_fields import load_configs

    try:
        loaded = load_configs(path)
    except FileNotFoundError:
        return ToolResult.failure(f"config file not found: {path}", code="not_found")
    except Exception as exc:
        raise ToolError(
            f"failed to load configs from '{path}': {exc}",
            code="invalid_input",
            details={"exception": type(exc).__name__},
        )

    shared = loaded.get("shared", {}) if isinstance(loaded, dict) else {}
    specific = loaded.get("specific", {}) if isinstance(loaded, dict) else {}

    specific_summary = {
        step: {"field_count": len(fields), "fields": sorted(map(str, fields.keys()))}
        for step, fields in specific.items()
        if isinstance(fields, dict)
    }

    data = {
        "path": path,
        "shared_field_count": len(shared) if isinstance(shared, dict) else 0,
        "shared_fields": sorted(map(str, shared.keys()))
        if isinstance(shared, dict)
        else [],
        "step_count": len(specific_summary),
        "steps": specific_summary,
    }
    return ToolResult.success(data, step_count=len(specific_summary))


def _merge_save(args: Dict[str, Any]) -> ToolResult:
    """
    Explain why merge_and_save_configs cannot be driven from a stateless JSON tool.

    ``cursus.core.config_fields.merge_and_save_configs`` requires a list of live, in-process
    Pydantic config *objects* (instances of BasePipelineConfig subclasses) — it relies on each
    object's type metadata and ``categorize_fields()`` methods for type-aware serialization.
    Those objects cannot be passed across a JSON tool boundary, so this tool returns a clear
    error directing the caller to the in-process API.
    """
    return ToolResult.failure(
        "config.merge_save is not available over the JSON tool boundary: "
        "merge_and_save_configs(config_list, output_file) requires live in-process Pydantic "
        "config objects, not JSON. Build the configs with DAGConfigFactory in-process and call "
        "cursus.core.config_fields.merge_and_save_configs directly.",
        code="unsupported",
        details={
            "engine_api": "cursus.core.config_fields.merge_and_save_configs",
            "reason": "requires live Pydantic config objects, not JSON-serializable input",
        },
        remedy={
            "suggested_tools": ["config.requirements"],
            "fix_action": (
                "Use config.requirements to learn the fields a DAG needs, then build the "
                "configs in-process with DAGConfigFactory and call "
                "cursus.core.config_fields.merge_and_save_configs directly — this step is "
                "not expressible over the stateless JSON tool boundary."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Tool registry for this namespace
# ---------------------------------------------------------------------------

_DAG_SCHEMA = {
    "type": "object",
    "description": (
        "Pipeline DAG topology. Either a flat object with 'nodes' and 'edges', or the "
        "serializer form {'dag': {'nodes': [...], 'edges': [...]}}."
    ),
    "properties": {
        "nodes": {
            "type": "array",
            "description": "Step/node names, e.g. 'TabularPreprocessing_training'.",
            "items": {"type": "string"},
        },
        "edges": {
            "type": "array",
            "description": "Directed dependency edges as [src, dst] pairs.",
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "dag": {
            "type": "object",
            "description": "Wrapped form: nested {'nodes': [...], 'edges': [...]}.",
        },
    },
    "additionalProperties": True,
}


TOOLS: List[ToolDef] = [
    ToolDef(
        name="config.requirements",
        description=(
            "Given a pipeline DAG, return the configuration field requirements for it: base "
            "pipeline config fields, base processing config fields (if any step needs them), "
            "and per-step (non-inherited) fields, plus the node->config-class map and which "
            "steps still need user input. Primary planning tool for building configs."
        ),
        schema={
            "type": "object",
            "properties": {"dag": _DAG_SCHEMA},
            "required": ["dag"],
            "additionalProperties": False,
        },
        handler=_requirements,
        tags=("planner",),
    ),
    ToolDef(
        name="config.field_info",
        description=(
            "Return the field requirements (name, type, description, required, default) for a "
            "single named config class, e.g. 'TabularPreprocessingConfig'. Pass categorized=true "
            "to split into required vs optional. Use to inspect one step's config in detail."
        ),
        schema={
            "type": "object",
            "properties": {
                "config_class": {
                    "type": "string",
                    "description": "Config class name to introspect (e.g. 'XGBoostTrainingConfig').",
                },
                "categorized": {
                    "type": "boolean",
                    "description": "If true, return separate 'required' and 'optional' field lists.",
                },
            },
            "required": ["config_class"],
            "additionalProperties": False,
        },
        handler=_field_info,
        tags=("planner",),
    ),
    ToolDef(
        name="config.load",
        description=(
            "Load a saved merged-config JSON file and return a JSON-safe summary: shared field "
            "names and per-step field names/counts. Use to inspect what a previously saved config "
            "file contains without loading full values."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filesystem path to a merged-config JSON file produced by merge_and_save_configs.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_load,
        tags=("planner",),
    ),
    ToolDef(
        name="config.merge_save",
        description=(
            "Reports that merging and saving configs is NOT available over the JSON tool boundary "
            "because merge_and_save_configs needs live in-process Pydantic config objects. Returns "
            "an explanatory error pointing at the in-process engine API."
        ),
        schema={
            "type": "object",
            "properties": {
                "output_file": {
                    "type": "string",
                    "description": "Intended output path (informational only; the operation is not supported here).",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_merge_save,
    ),
]
