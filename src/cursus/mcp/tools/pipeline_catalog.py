"""
``pipeline_catalog.*`` MCP tools — recommend, inspect, and load pre-built shared DAGs.

This namespace wraps the pipeline catalog engine: it delegates the recommendation,
DAG-detail, and config-guidance logic to
``cursus.pipeline_catalog.core.agent_tool.pipeline_catalog_tool`` (which already returns
clean, JSON-serializable dicts), uses
``cursus.pipeline_catalog.core.router.auto_select_dag`` for single-best selection, and
uses ``cursus.pipeline_catalog.shared_dags`` / ``core.pipeline_factory`` for listing and
loading shared DAGs. Engine modules are imported lazily inside each handler so a missing
optional dependency only fails that one tool call.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..envelope import ToolResult, ToolError
from ..registry import ToolDef


# One-line purpose of this namespace (collected by the registry for <ns>.help).
NAMESPACE = "Recommend/select/load pre-built shared DAGs (pipeline_catalog)."


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------


def _recommend(args: Dict[str, Any]) -> ToolResult:
    """Rank shared DAGs against semantic requirements via pipeline_catalog_tool."""
    from ...pipeline_catalog.core.agent_tool import pipeline_catalog_tool

    result = pipeline_catalog_tool(
        action="recommend",
        data_type=args.get("data_type"),
        has_labels=args.get("has_labels", True),
        needs_llm=args.get("needs_llm", False),
        multi_task=args.get("multi_task", False),
        incremental=args.get("incremental", False),
        framework=args.get("framework"),
        gpu_available=args.get("gpu_available", True),
    )
    if result.get("status") == "error":
        return ToolResult.failure(
            result.get("message", "recommendation failed"), code="invalid_input"
        )
    # Surface the catalog's own "next_step" guidance as a structured next step: once a
    # candidate is chosen, the agent fetches its config guidance then compiles.
    next_steps = [
        {
            "tool": "pipeline_catalog.config_guidance",
            "when": "after choosing a recommended dag_id",
            "why": "get prerequisites + the config fields that DAG needs",
            "args_hint": {"dag_id": "<chosen dag_id>"},
        },
        {
            "tool": "pipeline_catalog.get_dag",
            "when": "to inspect a candidate's nodes/edges before committing",
            "why": "see the topology you would compile",
            "args_hint": {"dag_id": "<chosen dag_id>"},
        },
    ]
    return ToolResult.success(
        result, total_matches=result.get("total_matches"), next_steps=next_steps
    )


# ---------------------------------------------------------------------------
# get_dag
# ---------------------------------------------------------------------------


def _get_dag(args: Dict[str, Any]) -> ToolResult:
    """Return nodes/edges/requirements for one shared DAG via pipeline_catalog_tool."""
    from ...pipeline_catalog.core.agent_tool import pipeline_catalog_tool

    dag_id = args["dag_id"]
    result = pipeline_catalog_tool(action="get_dag", dag_id=dag_id)
    if result.get("status") == "error":
        return ToolResult.failure(
            result.get("message", f"DAG '{dag_id}' not found"),
            code="not_found",
            details={"dag_id": dag_id},
        )
    return ToolResult.success(result, dag_id=dag_id)


# ---------------------------------------------------------------------------
# config_guidance
# ---------------------------------------------------------------------------


def _config_guidance(args: Dict[str, Any]) -> ToolResult:
    """Return prerequisites + config guidance for one DAG via pipeline_catalog_tool."""
    from ...pipeline_catalog.core.agent_tool import pipeline_catalog_tool

    dag_id = args["dag_id"]
    result = pipeline_catalog_tool(action="get_config_guidance", dag_id=dag_id)
    if result.get("status") == "error":
        return ToolResult.failure(
            result.get("message", f"DAG '{dag_id}' not found"),
            code="not_found",
            details={"dag_id": dag_id},
        )
    return ToolResult.success(result, dag_id=dag_id)


# ---------------------------------------------------------------------------
# auto_select
# ---------------------------------------------------------------------------


def _auto_select(args: Dict[str, Any]) -> ToolResult:
    """Pick the single best-matching DAG id (or null) via router.auto_select_dag."""
    from ...pipeline_catalog.core.router import auto_select_dag

    # auto_select_dag returns Optional[Tuple[dag_id, PipelineDAG, score]] — the
    # PipelineDAG object is not JSON-serializable, so we surface only id + score.
    selection = auto_select_dag(
        framework=args.get("framework"),
        features=args.get("features"),
        task_type=args.get("task_type"),
        min_score=args.get("min_score", 0.6),
    )
    if selection is None:
        return ToolResult.success(
            {"dag_id": None, "score": None, "matched": False},
            warnings=["No DAG met the minimum score threshold."],
        )
    dag_id, _dag, score = selection
    return ToolResult.success(
        {"dag_id": dag_id, "score": score, "matched": True},
        dag_id=dag_id,
    )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _list(args: Dict[str, Any]) -> ToolResult:
    """List all shared DAG ids with their catalog-index metadata."""
    from ...pipeline_catalog.shared_dags import get_all_shared_dags

    dags = get_all_shared_dags()  # Dict[dag_id, metadata-dict] — already JSON-safe
    entries = [
        {
            "dag_id": dag_id,
            "framework": meta.get("framework"),
            "description": meta.get("description"),
            "task_type": meta.get("task_type"),
            "complexity": meta.get("complexity"),
            "features": meta.get("features", []),
            "node_count": meta.get("node_count"),
        }
        for dag_id, meta in dags.items()
    ]
    entries.sort(key=lambda e: e["dag_id"])
    return ToolResult.success(
        {"dags": entries, "count": len(entries)}, count=len(entries)
    )


# ---------------------------------------------------------------------------
# load_dag
# ---------------------------------------------------------------------------


def _load_dag(args: Dict[str, Any]) -> ToolResult:
    """Load a shared DAG and return its nodes + edges in JSON-safe form."""
    from ...pipeline_catalog.shared_dags import load_shared_dag

    dag_id = args["dag_id"]
    try:
        dag = load_shared_dag(dag_id)
    except ValueError as exc:
        # load_shared_dag raises ValueError when the dag_id is unknown.
        raise ToolError(str(exc), code="not_found", details={"dag_id": dag_id})

    # PipelineDAG.nodes is a list of node names; PipelineDAG.edges is a list of
    # (src, dst) tuples — convert tuples to lists so the payload is JSON-serializable.
    nodes = list(getattr(dag, "nodes", []))
    edges = [list(edge) for edge in getattr(dag, "edges", [])]
    return ToolResult.success(
        {
            "dag_id": dag_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        dag_id=dag_id,
    )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_FRAMEWORK_ENUM = ["pytorch", "xgboost", "lightgbm", "lightgbmmt", "any"]
_DATA_TYPE_ENUM = ["text", "tabular", "mixed"]


TOOLS: List[ToolDef] = [
    ToolDef(
        name="pipeline_catalog.recommend",
        description=(
            "Recommend pre-built shared DAGs ranked by how well they match semantic "
            "requirements (data type, labels, LLM need, framework, GPU). Call when the "
            "user describes an ML problem and you need candidate pipeline DAGs."
        ),
        schema={
            "type": "object",
            "properties": {
                "data_type": {
                    "type": "string",
                    "enum": _DATA_TYPE_ENUM,
                    "description": "Primary data type to train on.",
                },
                "has_labels": {
                    "type": "boolean",
                    "description": "Whether labeled training data already exists (default true).",
                },
                "needs_llm": {
                    "type": "boolean",
                    "description": "Whether an LLM (Bedrock) is needed for labeling/enrichment (default false).",
                },
                "multi_task": {
                    "type": "boolean",
                    "description": "Whether multiple output tasks are required (default false).",
                },
                "incremental": {
                    "type": "boolean",
                    "description": "Whether this is incremental retraining rather than first-time (default false).",
                },
                "framework": {
                    "type": "string",
                    "enum": _FRAMEWORK_ENUM,
                    "description": "Preferred ML framework; 'any' or omit to not filter.",
                },
                "gpu_available": {
                    "type": "boolean",
                    "description": "Whether GPU instances are available (default true).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_recommend,
        tags=("planner",),
        when="Call when the user describes an ML problem and you need a ranked list of candidate shared-DAG pipelines to choose from.",
        examples=(
            "pipeline_catalog.recommend {}  # rank all DAGs against defaults (labeled tabular, GPU on)",
            'pipeline_catalog.recommend {"data_type": "tabular", "framework": "xgboost"}  # tabular XGBoost candidates',
            'pipeline_catalog.recommend {"data_type": "text", "needs_llm": true, "framework": "pytorch"}  # LLM-enriched PyTorch text pipelines',
        ),
    ),
    ToolDef(
        name="pipeline_catalog.get_dag",
        description=(
            "Get full details of one shared DAG — nodes, edges, input requirements, "
            "constraints, cost, and agent context. Call after choosing a dag_id from "
            "recommend/list to inspect its structure."
        ),
        schema={
            "type": "object",
            "properties": {
                "dag_id": {
                    "type": "string",
                    "description": "DAG identifier (e.g. 'bedrock_pytorch_incremental_edx').",
                },
            },
            "required": ["dag_id"],
            "additionalProperties": False,
        },
        handler=_get_dag,
        tags=("planner",),
        when="Call after picking a dag_id from recommend/list to inspect that DAG's nodes, edges, input requirements, and constraints.",
        examples=(
            'pipeline_catalog.get_dag {"dag_id": "bedrock_pytorch_incremental_edx"}  # inspect an incremental LLM-scoring DAG',
            'pipeline_catalog.get_dag {"dag_id": "lightgbm_complete_e2e"}  # inspect an end-to-end LightGBM DAG',
        ),
    ),
    ToolDef(
        name="pipeline_catalog.config_guidance",
        description=(
            "Get configuration guidance for one shared DAG — prerequisites, required vs "
            "default config values, common pitfalls, and a decision tree. Call before "
            "building a config for a chosen DAG."
        ),
        schema={
            "type": "object",
            "properties": {
                "dag_id": {
                    "type": "string",
                    "description": "DAG identifier to fetch configuration guidance for.",
                },
            },
            "required": ["dag_id"],
            "additionalProperties": False,
        },
        handler=_config_guidance,
        tags=("planner",),
        when="Call before authoring a config for a chosen DAG to learn its prerequisites, required vs default config fields, and common pitfalls.",
        examples=(
            'pipeline_catalog.config_guidance {"dag_id": "bedrock_pytorch_incremental_edx"}  # config prerequisites for the incremental EDX DAG',
            'pipeline_catalog.config_guidance {"dag_id": "lightgbmmt_complete_e2e"}  # config guidance for a multi-task LightGBM DAG',
        ),
    ),
    ToolDef(
        name="pipeline_catalog.auto_select",
        description=(
            "Auto-select the single best-matching shared DAG for a framework/features/"
            "task-type, or return null if no DAG meets the score threshold. Call when "
            "you want one decisive pick rather than a ranked list."
        ),
        schema={
            "type": "object",
            "properties": {
                "framework": {
                    "type": "string",
                    "enum": _FRAMEWORK_ENUM,
                    "description": "Required framework; omit to not filter.",
                },
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required features (e.g. ['training','bedrock','edx_uploading']).",
                },
                "task_type": {
                    "type": "string",
                    "description": "Task-type keyword (e.g. 'incremental', 'end_to_end').",
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum match score in 0-1 to accept a pick (default 0.6).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_auto_select,
        tags=("planner",),
        when="Call when you want one decisive best-match DAG (or null) instead of a ranked list — e.g. to auto-pick a pipeline for a known framework/task.",
        examples=(
            'pipeline_catalog.auto_select {"framework": "xgboost"}  # single best XGBoost DAG at default 0.6 threshold',
            'pipeline_catalog.auto_select {"framework": "pytorch", "features": ["training", "bedrock_realtime_processing", "edx_uploading"], "task_type": "incremental"}  # target the incremental EDX DAG',
            'pipeline_catalog.auto_select {"framework": "lightgbm", "min_score": 0.8}  # stricter threshold',
        ),
    ),
    ToolDef(
        name="pipeline_catalog.list",
        description=(
            "List every available shared DAG id with its metadata (framework, "
            "description, task type, complexity, features). Call to browse the catalog."
        ),
        schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=_list,
        tags=("planner",),
        when="Call to browse the whole catalog when you don't yet know which dag_id exists — returns every shared DAG id with its metadata.",
        examples=(
            "pipeline_catalog.list {}  # list every available shared DAG with metadata",
        ),
    ),
    ToolDef(
        name="pipeline_catalog.load_dag",
        description=(
            "Load a shared DAG by id and return its nodes and edges in JSON-safe form. "
            "Call to retrieve the concrete graph structure for compilation or display."
        ),
        schema={
            "type": "object",
            "properties": {
                "dag_id": {
                    "type": "string",
                    "description": "DAG identifier to load from the shared catalog.",
                },
            },
            "required": ["dag_id"],
            "additionalProperties": False,
        },
        handler=_load_dag,
        tags=("planner",),
        when="Call to retrieve the concrete node/edge graph of a chosen DAG for compilation or display (lighter than get_dag, no requirements/constraints).",
        examples=(
            'pipeline_catalog.load_dag {"dag_id": "bedrock_pytorch_incremental_edx"}  # load nodes+edges for compilation',
            'pipeline_catalog.load_dag {"dag_id": "lightgbm_complete_e2e"}  # load the end-to-end LightGBM graph',
        ),
    ),
]
