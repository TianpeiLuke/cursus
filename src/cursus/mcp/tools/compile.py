"""
cursus.mcp ``compile.*`` tools.

Wraps the DAG-to-SageMaker-pipeline compiler in ``cursus.core.compiler`` —
``PipelineDAGCompiler`` (validate / preview / compile / compile_with_report),
``compile_dag_to_pipeline``, ``compile_single_node_to_pipeline``, and the pure
``name_generator`` helpers. Tools take a DAG (an inline ``dag`` JSON object and/or a
``dag_file`` path) plus a ``config_file`` path, and turn it into a SageMaker pipeline
definition. Pipeline *building* is non-destructive; only ``upsert`` actually mutates
SageMaker state, so those code paths are gated behind a ``destructive=True`` flag.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..envelope import ToolResult, ToolError
from ..registry import ToolDef


# ---------------------------------------------------------------------------
# Helpers (not tools) — shared DAG-loading logic
# ---------------------------------------------------------------------------


def _resolve_dag(args: Dict[str, Any]) -> Any:
    """
    Build a ``PipelineDAG`` from either an inline ``dag`` JSON object or a ``dag_file``
    path. Exactly one source must be supplied. Raises :class:`ToolError` on bad input.

    Inline ``dag`` shape:  ``{"nodes": ["a", "b"], "edges": [["a", "b"]]}`` (edges
    optional). A ``dag_file`` is read via the engine's ``import_dag_from_json``.
    """
    dag_file = args.get("dag_file")
    inline = args.get("dag")

    if dag_file and inline:
        raise ToolError(
            "provide either 'dag' (inline JSON) or 'dag_file' (path), not both",
            code="invalid_input",
        )
    if not dag_file and not inline:
        raise ToolError(
            "must provide one of 'dag' (inline JSON) or 'dag_file' (path)",
            code="invalid_input",
        )

    if dag_file:
        import os

        if not os.path.exists(dag_file):
            raise ToolError(f"dag_file not found: {dag_file}", code="not_found")
        from ...api.dag import import_dag_from_json

        try:
            return import_dag_from_json(dag_file)
        except Exception as exc:  # noqa: BLE001 - surface as handled error
            raise ToolError(
                f"failed to load DAG from '{dag_file}': {exc}",
                code="invalid_input",
            )

    # Inline DAG object -> PipelineDAG
    if not isinstance(inline, dict):
        raise ToolError(
            "'dag' must be a JSON object with 'nodes' and optional 'edges'",
            code="invalid_input",
        )
    nodes = inline.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ToolError(
            "'dag.nodes' must be a non-empty list of step names",
            code="invalid_input",
        )
    raw_edges = inline.get("edges", []) or []
    edges: List[tuple] = []
    for edge in raw_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ToolError(
                f"invalid edge {edge!r}: each edge must be a [src, dst] pair",
                code="invalid_input",
            )
        edges.append((edge[0], edge[1]))

    from ...api.dag.base_dag import PipelineDAG

    return PipelineDAG(nodes=list(nodes), edges=edges)


def _require_config_exists(config_file: str) -> None:
    import os

    if not config_file or not os.path.exists(config_file):
        raise ToolError(f"config_file not found: {config_file}", code="not_found")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _validate(args: Dict[str, Any]) -> ToolResult:
    """compile.validate -> PipelineDAGCompiler.validate_dag_compatibility (non-destructive)."""
    config_file = args.get("config_file")
    _require_config_exists(config_file)
    dag = _resolve_dag(args)

    from ...core.compiler import PipelineDAGCompiler

    compiler = PipelineDAGCompiler(config_path=config_file, role=args.get("role"))
    result = compiler.validate_dag_compatibility(dag)

    # ValidationResult is a pydantic BaseModel -> model_dump() is JSON-safe.
    data = result.model_dump()
    data["summary"] = result.summary()
    return ToolResult.success(data, warnings=list(result.warnings or []))


def _preview(args: Dict[str, Any]) -> ToolResult:
    """compile.preview -> PipelineDAGCompiler.preview_resolution (non-destructive)."""
    config_file = args.get("config_file")
    _require_config_exists(config_file)
    dag = _resolve_dag(args)

    from ...core.compiler import PipelineDAGCompiler

    compiler = PipelineDAGCompiler(config_path=config_file, role=args.get("role"))
    preview = compiler.preview_resolution(dag)

    # ResolutionPreview is a pydantic BaseModel -> model_dump() is JSON-safe.
    return ToolResult.success(
        preview.model_dump(),
        warnings=list(preview.recommendations or []),
    )


def _dag(args: Dict[str, Any]) -> ToolResult:
    """
    compile.dag -> compile_dag_to_pipeline. Builds the SageMaker Pipeline definition.

    Non-destructive by default (build only). If ``upsert`` is true, calls
    ``pipeline.upsert()`` which mutates SageMaker state — this tool is marked
    destructive for that path.
    """
    config_file = args.get("config_file")
    _require_config_exists(config_file)
    dag = _resolve_dag(args)
    role = args.get("role")
    pipeline_name = args.get("pipeline_name")
    upsert = bool(args.get("upsert", False))

    from ...core.compiler import compile_dag_to_pipeline

    pipeline = compile_dag_to_pipeline(
        dag=dag,
        config_path=config_file,
        role=role,
        pipeline_name=pipeline_name,
    )

    # Pipeline is a SageMaker SDK object — extract only JSON-safe fields.
    step_count = len(getattr(pipeline, "steps", []) or [])
    data: Dict[str, Any] = {
        "pipeline_name": getattr(pipeline, "name", None),
        "step_count": step_count,
        "upserted": False,
    }
    warnings: List[str] = []

    if upsert:
        if not role:
            raise ToolError(
                "'role' is required to upsert a pipeline to SageMaker",
                code="invalid_input",
            )
        try:
            response = pipeline.upsert(role_arn=role)
        except Exception as exc:  # noqa: BLE001 - surface as handled error
            raise ToolError(f"pipeline upsert failed: {exc}", code="upsert_failed")
        data["upserted"] = True
        # response is typically a dict with PipelineArn; keep only the ARN if present.
        if isinstance(response, dict):
            data["pipeline_arn"] = response.get("PipelineArn")
    else:
        warnings.append(
            "build-only: pipeline definition compiled but not upserted to SageMaker"
        )

    return ToolResult.success(data, warnings=warnings, upserted=data["upserted"])


def _with_report(args: Dict[str, Any]) -> ToolResult:
    """
    compile.with_report -> PipelineDAGCompiler.compile_with_report. Builds the pipeline
    definition and returns the ConversionReport (non-destructive — no upsert/start).
    """
    config_file = args.get("config_file")
    _require_config_exists(config_file)
    dag = _resolve_dag(args)
    pipeline_name = args.get("pipeline_name")

    from ...core.compiler import PipelineDAGCompiler

    compiler = PipelineDAGCompiler(config_path=config_file, role=args.get("role"))
    pipeline, report = compiler.compile_with_report(dag, pipeline_name=pipeline_name)

    # ConversionReport is a pydantic BaseModel -> model_dump() is JSON-safe.
    data = report.model_dump()
    data["step_count"] = len(report.steps)
    data["summary"] = report.summary()
    return ToolResult.success(data, warnings=list(report.warnings or []))


def _single_node(args: Dict[str, Any]) -> ToolResult:
    """
    compile.single_node -> compile_single_node_to_pipeline. Re-runs one node in
    isolation with manual S3 inputs (build only — does not start/upsert).
    """
    config_file = args.get("config_file")
    _require_config_exists(config_file)
    dag = _resolve_dag(args)
    target_node = args.get("target_node")
    manual_inputs = args.get("manual_inputs") or {}
    role = args.get("role")
    pipeline_name = args.get("pipeline_name")
    validate_inputs = bool(args.get("validate_inputs", True))

    if not isinstance(manual_inputs, dict):
        raise ToolError(
            "'manual_inputs' must be an object of {logical_name: s3_uri}",
            code="invalid_input",
        )

    from ...core.compiler import compile_single_node_to_pipeline

    try:
        pipeline = compile_single_node_to_pipeline(
            dag=dag,
            config_path=config_file,
            target_node=target_node,
            manual_inputs=manual_inputs,
            role=role,
            pipeline_name=pipeline_name,
            validate_inputs=validate_inputs,
        )
    except ValueError as exc:
        # Engine raises ValueError for validation failures / missing node.
        raise ToolError(str(exc), code="invalid_input")
    except FileNotFoundError as exc:
        raise ToolError(str(exc), code="not_found")

    step_count = len(getattr(pipeline, "steps", []) or [])
    data = {
        "pipeline_name": getattr(pipeline, "name", None),
        "target_node": target_node,
        "step_count": step_count,
    }
    return ToolResult.success(
        data,
        warnings=["build-only: single-node pipeline compiled but not started/upserted"],
    )


def _name(args: Dict[str, Any]) -> ToolResult:
    """
    compile.name -> name_generator helpers. Pure string utilities: generate a
    SageMaker-valid pipeline name, sanitize an arbitrary name, and report validity.
    """
    base = args.get("base")
    version = args.get("version", "1.0")
    sanitize = args.get("sanitize")

    from ...core.compiler import (
        generate_pipeline_name,
        sanitize_pipeline_name,
        validate_pipeline_name,
    )

    data: Dict[str, Any] = {}
    if base is not None:
        generated = generate_pipeline_name(base, version)
        data["generated"] = generated
        data["generated_is_valid"] = validate_pipeline_name(generated)
    if sanitize is not None:
        sanitized = sanitize_pipeline_name(sanitize)
        data["sanitized"] = sanitized
        data["sanitized_is_valid"] = validate_pipeline_name(sanitized)

    if not data:
        raise ToolError(
            "provide 'base' (to generate a name) and/or 'sanitize' (to clean one)",
            code="invalid_input",
        )
    return ToolResult.success(data)


# ---------------------------------------------------------------------------
# Schema fragments shared across DAG-taking tools
# ---------------------------------------------------------------------------

_DAG_PROPS: Dict[str, Any] = {
    "dag": {
        "type": "object",
        "description": (
            'Inline DAG as JSON: {"nodes": ["step_a", "step_b"], '
            '"edges": [["step_a", "step_b"]]}. Provide this OR \'dag_file\'.'
        ),
        "properties": {
            "nodes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Step names (one per DAG node).",
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "description": "Directed dependency edges as [src, dst] pairs.",
            },
        },
    },
    "dag_file": {
        "type": "string",
        "description": (
            "Path to a serialized DAG JSON file (loaded via import_dag_from_json). "
            "Provide this OR inline 'dag'."
        ),
    },
    "config_file": {
        "type": "string",
        "description": "Path to the configuration JSON file containing step configs.",
    },
    "role": {
        "type": "string",
        "description": "Optional IAM role ARN for the SageMaker pipeline.",
    },
}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[ToolDef] = [
    ToolDef(
        name="compile.validate",
        description=(
            "Validate that a DAG's nodes can be resolved to configs and step builders "
            "for a given config file. Call before compiling to catch missing configs "
            "or unresolvable builders. Returns is_valid, missing_configs, "
            "unresolvable_builders, config_errors, dependency_issues, warnings."
        ),
        schema={
            "type": "object",
            "properties": dict(_DAG_PROPS),
            "required": ["config_file"],
            "additionalProperties": False,
        },
        handler=_validate,
        destructive=False,
        tags=("validator",),
    ),
    ToolDef(
        name="compile.preview",
        description=(
            "Preview how each DAG node will resolve to a config type and step builder, "
            "including per-node confidence scores, ambiguities, and recommendations. "
            "Non-destructive; use to inspect resolution before compiling."
        ),
        schema={
            "type": "object",
            "properties": dict(_DAG_PROPS),
            "required": ["config_file"],
            "additionalProperties": False,
        },
        handler=_preview,
        destructive=False,
        tags=("planner", "validator"),
    ),
    ToolDef(
        name="compile.dag",
        description=(
            "Compile a DAG + config file into a SageMaker pipeline definition and "
            "return its name and step count. Build-only by default; set upsert=true "
            "(requires role) to push the pipeline to SageMaker (mutates external state)."
        ),
        schema={
            "type": "object",
            "properties": {
                **_DAG_PROPS,
                "pipeline_name": {
                    "type": "string",
                    "description": "Optional pipeline name override.",
                },
                "upsert": {
                    "type": "boolean",
                    "description": (
                        "If true, upsert the compiled pipeline to SageMaker "
                        "(destructive). Requires 'role'. Default false (build only)."
                    ),
                    "default": False,
                },
            },
            "required": ["config_file"],
            "additionalProperties": False,
        },
        handler=_dag,
        destructive=True,  # the optional upsert path mutates SageMaker state
        tags=("programmer",),
    ),
    ToolDef(
        name="compile.with_report",
        description=(
            "Compile a DAG + config file into a pipeline and return a detailed "
            "ConversionReport: pipeline_name, steps, avg_confidence, resolution_details, "
            "and warnings. Non-destructive (builds the definition, never upserts)."
        ),
        schema={
            "type": "object",
            "properties": {
                **_DAG_PROPS,
                "pipeline_name": {
                    "type": "string",
                    "description": "Optional pipeline name override.",
                },
            },
            "required": ["config_file"],
            "additionalProperties": False,
        },
        handler=_with_report,
        destructive=False,
        tags=("programmer",),
    ),
    ToolDef(
        name="compile.single_node",
        description=(
            "Compile an isolated, single-node pipeline for one DAG node using manual "
            "S3 input overrides — useful to re-run one failed step without rerunning "
            "upstream steps. Build-only (does not start/upsert)."
        ),
        schema={
            "type": "object",
            "properties": {
                **_DAG_PROPS,
                "target_node": {
                    "type": "string",
                    "description": "Name of the DAG node to compile in isolation.",
                },
                "manual_inputs": {
                    "type": "object",
                    "description": (
                        "Map of logical input name -> S3 URI "
                        '(e.g. {"input_path": "s3://bucket/run-1/output/"}).'
                    ),
                    "additionalProperties": {"type": "string"},
                },
                "pipeline_name": {
                    "type": "string",
                    "description": "Optional pipeline name (default '<node>-isolated').",
                },
                "validate_inputs": {
                    "type": "boolean",
                    "description": "Validate node existence + S3 URIs first. Default true.",
                    "default": True,
                },
            },
            "required": ["config_file", "target_node", "manual_inputs"],
            "additionalProperties": False,
        },
        handler=_single_node,
        destructive=False,
        tags=("programmer",),
    ),
    ToolDef(
        name="compile.name",
        description=(
            "Pure string helper for SageMaker pipeline names. Pass 'base' to generate a "
            "valid name ('<base>-<version>-pipeline', sanitized) and/or 'sanitize' to "
            "clean an arbitrary string. Returns the result(s) plus validity flags."
        ),
        schema={
            "type": "object",
            "properties": {
                "base": {
                    "type": "string",
                    "description": "Base name to generate a pipeline name from.",
                },
                "version": {
                    "type": "string",
                    "description": "Version segment for the generated name (default '1.0').",
                    "default": "1.0",
                },
                "sanitize": {
                    "type": "string",
                    "description": "An arbitrary name to sanitize to SageMaker constraints.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_name,
        destructive=False,
        tags=("programmer",),
    ),
]
