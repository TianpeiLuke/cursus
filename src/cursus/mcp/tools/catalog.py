"""
``catalog.*`` MCP tools — discovery over the unified step catalog + registry.

These tools wrap :class:`cursus.step_catalog.step_catalog.StepCatalog` (step listing,
search, per-step info, config-class and builder discovery) and the canonical step-name
registry in :mod:`cursus.registry.step_names` (config/builder/spec/sagemaker resolution,
file-name -> canonical-name mapping). Engine imports are lazy so a missing optional
discovery dependency only fails the specific tool call, not module import. Tools are
read-only — none mutate external state.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..envelope import ToolResult, ToolError
from ..registry import ToolDef


# ---------------------------------------------------------------------------
# Internal helpers (not tools)
# ---------------------------------------------------------------------------


def _new_catalog() -> Any:
    """
    Construct a package-scoped :class:`StepCatalog`.

    ``StepCatalog.__init__`` takes an optional ``workspace_dirs`` argument; with no
    argument it discovers only package components, which is the correct default for an
    MCP discovery surface.
    """
    from ...step_catalog import StepCatalog

    return StepCatalog()


def _step_info_to_dict(step_info: Any) -> Dict[str, Any]:
    """Convert a ``StepInfo`` pydantic model into a JSON-serializable dict."""
    file_components = {
        component_type: (str(meta.path) if meta is not None else None)
        for component_type, meta in (step_info.file_components or {}).items()
    }
    return {
        "step_name": step_info.step_name,
        "workspace_id": step_info.workspace_id,
        "registry_data": dict(step_info.registry_data or {}),
        "config_class": step_info.config_class or None,
        "builder_step_name": step_info.builder_step_name or None,
        "sagemaker_step_type": step_info.sagemaker_step_type or None,
        "description": step_info.description or None,
        "components_available": sorted(file_components.keys()),
        "file_components": file_components,
    }


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _list_steps(args: Dict[str, Any]) -> ToolResult:
    """List concrete pipeline step names, optionally filtered by workspace/job_type."""
    workspace_id = args.get("workspace_id")
    job_type = args.get("job_type")

    catalog = _new_catalog()
    steps = catalog.list_available_steps(workspace_id=workspace_id, job_type=job_type)
    steps = sorted(steps)
    return ToolResult.success(
        {"steps": steps, "count": len(steps)},
        workspace_id=workspace_id,
        job_type=job_type,
    )


def _search(args: Dict[str, Any]) -> ToolResult:
    """Fuzzy-search steps by name; return scored matches with available components."""
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a non-empty string", code="invalid_input")
    job_type = args.get("job_type")

    catalog = _new_catalog()
    results = catalog.search_steps(query, job_type=job_type)
    matches = [
        {
            "step_name": r.step_name,
            "workspace_id": r.workspace_id,
            "score": r.match_score,
            "match_reason": r.match_reason,
            "components_available": list(r.components_available),
        }
        for r in results
    ]
    return ToolResult.success(
        {"matches": matches, "count": len(matches)},
        query=query,
    )


def _step_info(args: Dict[str, Any]) -> ToolResult:
    """Return full info for one step: registry data, components, framework, sagemaker type."""
    step_name = args.get("step_name")
    job_type = args.get("job_type")

    catalog = _new_catalog()
    info = catalog.get_step_info(step_name, job_type=job_type)
    if info is None:
        return ToolResult.failure(
            f"step not found: '{step_name}'",
            code="not_found",
            details={"step_name": step_name},
        )

    data = _step_info_to_dict(info)
    # Framework detection is best-effort and may be None.
    try:
        data["framework"] = catalog.detect_framework(step_name)
    except Exception:  # pragma: no cover - detection is non-essential
        data["framework"] = None
    return ToolResult.success(data, step_name=step_name)


def _config_fields(args: Dict[str, Any]) -> ToolResult:
    """List the fields (name/type/required/default) of a step's configuration class."""
    step_name = args.get("step_name")

    # Resolve the step's config class name via the registry.
    from ...registry.step_names import get_config_class_name

    try:
        config_class_name = get_config_class_name(step_name)
    except ValueError as exc:
        return ToolResult.failure(
            str(exc), code="not_found", details={"step_name": step_name}
        )

    # Discover the actual config class object via the catalog's config discovery.
    catalog = _new_catalog()
    config_classes = catalog.discover_config_classes()
    config_class = config_classes.get(config_class_name)
    if config_class is None:
        return ToolResult.failure(
            f"config class '{config_class_name}' for step '{step_name}' "
            f"could not be discovered",
            code="not_found",
            details={"step_name": step_name, "config_class": config_class_name},
        )

    fields: List[Dict[str, Any]] = []
    warnings: List[str] = []
    model_fields = getattr(config_class, "model_fields", None)
    if model_fields is None:
        warnings.append(
            f"config class '{config_class_name}' is not a Pydantic model; "
            "no field metadata available"
        )
    else:
        # Pydantic v2 FieldInfo: is_required(), default, annotation, description.
        from pydantic_core import PydanticUndefined  # type: ignore

        for field_name, field_info in model_fields.items():
            try:
                required = bool(field_info.is_required())
            except Exception:
                required = (
                    getattr(field_info, "default", PydanticUndefined)
                    is PydanticUndefined
                )

            default = getattr(field_info, "default", PydanticUndefined)
            if default is PydanticUndefined:
                default = None
            else:
                # Keep default JSON-serializable; fall back to str() for complex objects.
                try:
                    import json

                    json.dumps(default)
                except (TypeError, ValueError):
                    default = str(default)

            annotation = getattr(field_info, "annotation", None)
            type_name = getattr(annotation, "__name__", None) or (
                str(annotation) if annotation is not None else None
            )

            fields.append(
                {
                    "name": field_name,
                    "type": type_name,
                    "required": required,
                    "default": default,
                    "description": getattr(field_info, "description", None),
                }
            )

    return ToolResult.success(
        {
            "step_name": step_name,
            "config_class": config_class_name,
            "fields": fields,
            "count": len(fields),
        },
        warnings=warnings,
    )


def _resolve_step(args: Dict[str, Any]) -> ToolResult:
    """Resolve a step name (canonical or file-style) to its registered classes/types."""
    name = args.get("step_name")
    if not isinstance(name, str) or not name.strip():
        raise ToolError("step_name must be a non-empty string", code="invalid_input")

    from ...registry.step_names import (
        get_step_names,
        get_config_class_name,
        get_builder_step_name,
        get_spec_step_type,
        get_sagemaker_step_type,
        get_canonical_name_from_file_name,
    )

    registry = get_step_names()
    source = "canonical"
    canonical = name
    if name not in registry:
        # Try to map a file-style name (e.g. "model_evaluation_xgb") to canonical.
        try:
            canonical = get_canonical_name_from_file_name(name)
            source = "file_name"
        except ValueError:
            return ToolResult.failure(
                f"could not resolve step '{name}' to a canonical registry name",
                code="not_found",
                details={"step_name": name, "available": sorted(registry.keys())},
            )

    return ToolResult.success(
        {
            "input": name,
            "canonical_name": canonical,
            "config_class": get_config_class_name(canonical),
            "builder_class": get_builder_step_name(canonical),
            "spec_type": get_spec_step_type(canonical),
            "sagemaker_step_type": get_sagemaker_step_type(canonical),
            "source": source,
        }
    )


def _list_builders(args: Dict[str, Any]) -> ToolResult:
    """List builder class names, optionally filtered by SageMaker step type (no instantiation)."""
    step_type = args.get("sagemaker_step_type")

    catalog = _new_catalog()
    if step_type:
        builders = catalog.get_builders_by_step_type(step_type)
    else:
        builders = catalog.get_all_builders()

    # builders maps canonical step name -> builder class; summarize names only.
    summary = [
        {"step_name": step_name, "builder_class": getattr(cls, "__name__", str(cls))}
        for step_name, cls in sorted(builders.items())
    ]
    return ToolResult.success(
        {"builders": summary, "count": len(summary)},
        sagemaker_step_type=step_type,
    )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[ToolDef] = [
    ToolDef(
        name="catalog.list_steps",
        description=(
            "List the concrete pipeline step names known to cursus. Call this to "
            "discover what steps are available before planning or building a pipeline."
        ),
        schema={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Optional workspace filter; omit for package-scope steps.",
                },
                "job_type": {
                    "type": "string",
                    "description": "Optional job-type filter (e.g. 'training', 'validation').",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_list_steps,
        tags=("planner",),
    ),
    ToolDef(
        name="catalog.search",
        description=(
            "Fuzzy-search steps by name and return scored matches with the component "
            "types available for each. Use when you have a partial/approximate step name."
        ),
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search substring to match against step names.",
                },
                "job_type": {
                    "type": "string",
                    "description": "Optional job-type filter (e.g. 'training').",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=_search,
        tags=("planner",),
    ),
    ToolDef(
        name="catalog.step_info",
        description=(
            "Get full details for one step: registry data, config/builder names, "
            "SageMaker step type, detected framework, and which components exist."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical step name (PascalCase, e.g. 'XGBoostTraining').",
                },
                "job_type": {
                    "type": "string",
                    "description": "Optional job-type variant (e.g. 'training').",
                },
            },
            "required": ["step_name"],
            "additionalProperties": False,
        },
        handler=_step_info,
        tags=("planner",),
    ),
    ToolDef(
        name="catalog.config_fields",
        description=(
            "List the configuration fields (name, type, required, default, description) "
            "of a step's config class. Use to learn what config a step expects."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical step name whose config class fields to list.",
                },
            },
            "required": ["step_name"],
            "additionalProperties": False,
        },
        handler=_config_fields,
        tags=("planner", "programmer"),
    ),
    ToolDef(
        name="catalog.resolve_step",
        description=(
            "Resolve a step name (canonical or file-style like 'model_evaluation_xgb') "
            "to its config class, builder class, spec type, and SageMaker step type."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical or file-style step name to resolve.",
                },
            },
            "required": ["step_name"],
            "additionalProperties": False,
        },
        handler=_resolve_step,
        tags=("programmer",),
    ),
    ToolDef(
        name="catalog.list_builders",
        description=(
            "List builder class names (no instantiation), optionally filtered by "
            "SageMaker step type (e.g. 'Processing', 'Training', 'Transform')."
        ),
        schema={
            "type": "object",
            "properties": {
                "sagemaker_step_type": {
                    "type": "string",
                    "description": (
                        "Optional SageMaker step type filter "
                        "(Processing, Training, Transform, CreateModel, ...)."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_list_builders,
        tags=("programmer",),
    ),
]
