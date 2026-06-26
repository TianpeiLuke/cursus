"""
``dag.*`` tools — author, validate, and serialize a pipeline DAG.

This namespace wraps :mod:`cursus.api.dag`: ``PipelineDAG`` (build a topology from
nodes + edges), ``PipelineDAGResolver`` (integrity validation, execution planning,
dependency lookup), and the serializer helpers ``PipelineDAGWriter`` /
``export_dag_to_json`` / ``import_dag_from_json``. DAGs cross the tool boundary as plain
JSON (``{"nodes": [...], "edges": [[src, dst], ...]}``) so an agent can round-trip them.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..envelope import ToolResult, ToolError
from ..registry import ToolDef


# ---------------------------------------------------------------------------
# Helpers (no engine imports at module scope — keep imports lazy in handlers)
# ---------------------------------------------------------------------------


def _coerce_nodes(raw: Any) -> List[str]:
    """Validate and normalize a ``nodes`` argument into a list of step-name strings."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ToolError(
            "'nodes' must be a list of step-name strings", code="invalid_input"
        )
    nodes: List[str] = []
    for n in raw:
        if not isinstance(n, str):
            raise ToolError(
                f"every node must be a string, got {type(n).__name__}: {n!r}",
                code="invalid_input",
            )
        nodes.append(n)
    return nodes


def _coerce_edges(raw: Any) -> List[tuple]:
    """Validate and normalize an ``edges`` argument into a list of (src, dst) tuples."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ToolError(
            "'edges' must be a list of [src, dst] pairs", code="invalid_input"
        )
    edges: List[tuple] = []
    for edge in raw:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ToolError(
                f"each edge must be a [src, dst] pair, got {edge!r}",
                code="invalid_input",
            )
        src, dst = edge[0], edge[1]
        if not isinstance(src, str) or not isinstance(dst, str):
            raise ToolError(
                f"edge endpoints must be strings, got {edge!r}", code="invalid_input"
            )
        edges.append((src, dst))
    return edges


def _build_dag(args: Dict[str, Any]):
    """Construct a ``PipelineDAG`` from ``nodes``/``edges`` args (engine import is lazy)."""
    from ...api.dag import PipelineDAG

    nodes = _coerce_nodes(args.get("nodes"))
    edges = _coerce_edges(args.get("edges"))

    # Edges may reference nodes not present in the explicit node list; PipelineDAG's
    # add_edge auto-creates them, but its constructor assumes nodes already exist. To
    # avoid a KeyError we union edge endpoints into the node list first.
    node_set = set(nodes)
    for src, dst in edges:
        if src not in node_set:
            nodes.append(src)
            node_set.add(src)
        if dst not in node_set:
            nodes.append(dst)
            node_set.add(dst)

    return PipelineDAG(nodes=nodes, edges=edges)


def _build_resolver(dag):
    """Build a ``PipelineDAGResolver`` for a DAG without firing init-time catalog warnings."""
    from ...api.dag import PipelineDAGResolver

    # validate_on_init=False: we expose validation explicitly via dag.validate_integrity;
    # we don't want the resolver to emit log warnings just for being constructed.
    return PipelineDAGResolver(dag, validate_on_init=False)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _construct(args: Dict[str, Any]) -> ToolResult:
    """Build a DAG from nodes/edges and return the serialized, round-trippable JSON."""
    from ...api.dag import PipelineDAGWriter

    dag = _build_dag(args)
    metadata = args.get("metadata") or {}
    if not isinstance(metadata, dict):
        return ToolResult.failure("'metadata' must be an object", code="invalid_input")

    writer = PipelineDAGWriter(dag, metadata=metadata)
    serialized = (
        writer.to_dict()
    )  # {created_at, metadata, dag:{nodes,edges}, statistics}

    warnings: List[str] = []
    if serialized.get("statistics", {}).get("has_cycles"):
        warnings.append("DAG contains a cycle — it is not a valid pipeline topology.")

    return ToolResult.success(
        serialized,
        warnings=warnings,
        node_count=len(dag.nodes),
        edge_count=len(dag.edges),
    )


def _validate_integrity(args: Dict[str, Any]) -> ToolResult:
    """Run full DAG integrity validation (cycles, dangling edges, isolated/missing steps)."""
    dag = _build_dag(args)
    resolver = _build_resolver(dag)

    issues = (
        resolver.validate_dag_integrity()
    )  # Dict[str, List[str]] (already json-safe)
    is_valid = len(issues) == 0

    return ToolResult.success(
        {
            "is_valid": is_valid,
            "issues": issues,
            "issue_categories": sorted(issues.keys()),
            "node_count": len(dag.nodes),
            "edge_count": len(dag.edges),
        }
    )


def _resolve_plan(args: Dict[str, Any]) -> ToolResult:
    """Produce a topologically sorted execution plan (order + per-step dependencies)."""
    dag = _build_dag(args)
    resolver = _build_resolver(dag)

    try:
        plan = resolver.create_execution_plan()  # PipelineExecutionPlan (pydantic)
    except ValueError as exc:
        # Raised when the graph contains cycles ("Pipeline contains cycles").
        return ToolResult.failure(str(exc), code="invalid_input")

    # PipelineExecutionPlan is a pydantic BaseModel -> json-safe via model_dump().
    plan_dict = (
        plan.model_dump()
        if hasattr(plan, "model_dump")
        else plan.dict()  # pydantic v1 fallback
    )

    return ToolResult.success(
        plan_dict,
        step_count=len(plan_dict.get("execution_order", [])),
    )


def _dependencies(args: Dict[str, Any]) -> ToolResult:
    """Return the upstream (dependencies) and downstream (dependents) steps for one step."""
    step = args.get("step")
    if not isinstance(step, str) or not step:
        return ToolResult.failure(
            "'step' must be a non-empty string", code="invalid_input"
        )

    dag = _build_dag(args)
    if step not in dag.nodes:
        return ToolResult.failure(
            f"step '{step}' is not a node in the DAG",
            code="not_found",
            details={"available_nodes": dag.nodes},
        )

    resolver = _build_resolver(dag)
    upstream = list(resolver.get_step_dependencies(step))
    downstream = list(resolver.get_dependent_steps(step))

    return ToolResult.success(
        {
            "step": step,
            "dependencies": upstream,  # immediate parents (must run before)
            "dependents": downstream,  # immediate children (run after)
        }
    )


def _serialize(args: Dict[str, Any]) -> ToolResult:
    """Serialize a DAG to JSON: write to ``path`` if given, else return the JSON string."""
    from ...api.dag import PipelineDAGWriter, export_dag_to_json

    dag = _build_dag(args)
    metadata = args.get("metadata") or {}
    if not isinstance(metadata, dict):
        return ToolResult.failure("'metadata' must be an object", code="invalid_input")
    pretty = bool(args.get("pretty", True))
    path = args.get("path")

    if path:
        if not isinstance(path, str):
            return ToolResult.failure("'path' must be a string", code="invalid_input")
        try:
            export_dag_to_json(dag, path, metadata=metadata, pretty=pretty)
        except ValueError as exc:
            # write_to_file validates the DAG (cycles / empty / dangling edges).
            return ToolResult.failure(
                f"cannot serialize DAG: {exc}", code="invalid_input"
            )
        return ToolResult.success(
            {"path": path, "node_count": len(dag.nodes), "edge_count": len(dag.edges)}
        )

    writer = PipelineDAGWriter(dag, metadata=metadata)
    json_str = writer.to_json(pretty=pretty)
    return ToolResult.success(
        {"json": json_str, "node_count": len(dag.nodes), "edge_count": len(dag.edges)}
    )


def _deserialize(args: Dict[str, Any]) -> ToolResult:
    """Load a DAG from a JSON file path; return its nodes, edges, and statistics."""
    from ...api.dag import PipelineDAGReader, import_dag_from_json

    path = args.get("path")
    if not isinstance(path, str) or not path:
        return ToolResult.failure(
            "'path' must be a non-empty string", code="invalid_input"
        )

    try:
        dag = import_dag_from_json(path)
    except FileNotFoundError as exc:
        return ToolResult.failure(str(exc), code="not_found")
    except ValueError as exc:
        # Malformed JSON / missing fields / bad edge shape.
        return ToolResult.failure(f"invalid DAG file: {exc}", code="invalid_input")

    # Pull persisted metadata/statistics (best-effort; never fatal). Surface a warning
    # rather than swallowing silently so an agent can tell "no metadata" from "read failed".
    file_meta: Dict[str, Any] = {}
    warnings: List[str] = []
    try:
        file_meta = PipelineDAGReader.extract_metadata(path)
    except Exception as exc:  # pragma: no cover - metadata is optional
        file_meta = {}
        warnings.append(f"could not read embedded DAG metadata: {exc}")

    return ToolResult.success(
        {
            "nodes": list(dag.nodes),
            "edges": [[src, dst] for src, dst in dag.edges],
            "stats": file_meta.get("statistics", {}),
            "metadata": file_meta.get("metadata", {}),
            "created_at": file_meta.get("created_at"),
        },
        warnings=warnings,
        node_count=len(dag.nodes),
        edge_count=len(dag.edges),
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_NODES_PROP = {
    "type": "array",
    "items": {"type": "string"},
    "description": "Step names (nodes) of the pipeline DAG, e.g. ['preprocess', 'train'].",
}

_EDGES_PROP = {
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 2,
        "maxItems": 2,
    },
    "description": (
        "Directed dependency edges as [src, dst] pairs; src runs before dst. "
        "Endpoints not listed in 'nodes' are added automatically."
    ),
}

_METADATA_PROP = {
    "type": "object",
    "description": "Optional free-form metadata embedded in the serialized DAG.",
    "additionalProperties": True,
}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[ToolDef] = [
    ToolDef(
        name="dag.construct",
        description=(
            "Build a pipeline DAG from nodes and [src, dst] edges and return its "
            "serialized JSON (nodes, edges, statistics). Use this to create a "
            "round-trippable DAG the other dag.* tools accept."
        ),
        schema={
            "type": "object",
            "properties": {
                "nodes": _NODES_PROP,
                "edges": _EDGES_PROP,
                "metadata": _METADATA_PROP,
            },
            "required": ["nodes"],
            "additionalProperties": False,
        },
        handler=_construct,
        tags=("planner",),
    ),
    ToolDef(
        name="dag.validate_integrity",
        description=(
            "Validate a DAG's integrity (cycles, dangling edges, isolated nodes, and — "
            "when the step catalog is available — missing steps/components). Returns "
            "{is_valid, issues}. Call before compiling a DAG into a pipeline."
        ),
        schema={
            "type": "object",
            "properties": {
                "nodes": _NODES_PROP,
                "edges": _EDGES_PROP,
            },
            "required": ["nodes"],
            "additionalProperties": False,
        },
        handler=_validate_integrity,
        tags=("validator",),
    ),
    ToolDef(
        name="dag.resolve_plan",
        description=(
            "Compute a topologically sorted execution plan for a DAG: the execution "
            "order plus per-step dependencies and data-flow map. Fails if the DAG has "
            "a cycle."
        ),
        schema={
            "type": "object",
            "properties": {
                "nodes": _NODES_PROP,
                "edges": _EDGES_PROP,
            },
            "required": ["nodes"],
            "additionalProperties": False,
        },
        handler=_resolve_plan,
        tags=("planner",),
    ),
    ToolDef(
        name="dag.dependencies",
        description=(
            "For a single step in the DAG, return its immediate upstream dependencies "
            "(steps that must run before) and downstream dependents (steps that run "
            "after)."
        ),
        schema={
            "type": "object",
            "properties": {
                "step": {
                    "type": "string",
                    "description": "Step name to inspect; must be a node in the DAG.",
                },
                "nodes": _NODES_PROP,
                "edges": _EDGES_PROP,
            },
            "required": ["step", "nodes"],
            "additionalProperties": False,
        },
        handler=_dependencies,
        tags=("planner",),
    ),
    ToolDef(
        name="dag.serialize",
        description=(
            "Serialize a DAG to JSON. If 'path' is given, write the JSON file and "
            "return the path; otherwise return the JSON string. Validates the DAG "
            "before writing to a file."
        ),
        schema={
            "type": "object",
            "properties": {
                "nodes": _NODES_PROP,
                "edges": _EDGES_PROP,
                "metadata": _METADATA_PROP,
                "path": {
                    "type": "string",
                    "description": "Optional output file path. If omitted, the JSON is returned inline.",
                },
                "pretty": {
                    "type": "boolean",
                    "description": "Pretty-print the JSON with indentation (default true).",
                },
            },
            "required": ["nodes"],
            "additionalProperties": False,
        },
        handler=_serialize,
        tags=("planner",),
    ),
    ToolDef(
        name="dag.deserialize",
        description=(
            "Load a DAG from a JSON file written by dag.serialize and return its "
            "nodes, edges, statistics, and metadata."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to a DAG JSON file to read.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_deserialize,
        tags=("planner",),
    ),
]
