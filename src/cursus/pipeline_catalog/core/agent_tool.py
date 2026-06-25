"""
Agent Tool Interface for Pipeline Catalog

Exposes pipeline catalog as tool calls for LLM agents.
Compatible with MCP, OpenAI function calling, and Claude tool_use.

Usage:
    # Agent receives tool schema, calls with parameters:
    result = pipeline_catalog_tool(action="recommend", data_type="text", needs_llm=True)
    result = pipeline_catalog_tool(action="get_dag", dag_id="bedrock_pytorch_incremental_edx")
    result = pipeline_catalog_tool(action="get_config_guidance", dag_id="...")
"""

import json
import logging
from typing import Any, Dict, Optional

from .router import recommend_for_agent
from ..shared_dags import get_catalog_index, SHARED_DAGS_DIR

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Schema (for agent registration)
# ============================================================================

TOOL_SCHEMA = {
    "name": "pipeline_catalog",
    "description": (
        "Query the ML pipeline catalog to find, recommend, and configure SageMaker training pipelines. "
        "Supports: recommending DAGs based on requirements, getting DAG details with prerequisites "
        "and configuration guidance, and listing available frameworks/features."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "recommend",
                    "get_dag",
                    "get_config_guidance",
                    "list_frameworks",
                    "list_features",
                ],
                "description": "Action to perform",
            },
            "dag_id": {
                "type": "string",
                "description": "DAG identifier (for get_dag, get_config_guidance actions)",
            },
            "data_type": {
                "type": "string",
                "enum": ["text", "tabular", "mixed"],
                "description": "Primary data type for recommendation",
            },
            "has_labels": {
                "type": "boolean",
                "description": "Whether labeled training data already exists",
            },
            "needs_llm": {
                "type": "boolean",
                "description": "Whether LLM (Bedrock) is needed for labeling/enrichment",
            },
            "multi_task": {
                "type": "boolean",
                "description": "Whether multiple output tasks are needed",
            },
            "incremental": {
                "type": "boolean",
                "description": "Whether this is incremental retraining (not first-time)",
            },
            "framework": {
                "type": "string",
                "enum": ["pytorch", "xgboost", "lightgbm", "lightgbmmt", "any"],
                "description": "Preferred ML framework",
            },
            "gpu_available": {
                "type": "boolean",
                "description": "Whether GPU instances are available",
            },
        },
        "required": ["action"],
    },
}


# ============================================================================
# Tool Implementation
# ============================================================================


def pipeline_catalog_tool(
    action: str,
    dag_id: Optional[str] = None,
    data_type: Optional[str] = None,
    has_labels: bool = True,
    needs_llm: bool = False,
    multi_task: bool = False,
    incremental: bool = False,
    framework: Optional[str] = None,
    gpu_available: bool = True,
) -> Dict[str, Any]:
    """
    Execute a pipeline catalog tool action. Returns structured response for the agent.
    """
    if action == "recommend":
        fw = framework if framework and framework != "any" else None
        results = recommend_for_agent(
            data_type=data_type,
            has_labels=has_labels,
            needs_llm=needs_llm,
            multi_task=multi_task,
            incremental=incremental,
            gpu_available=gpu_available,
        )
        # Filter by framework if specified
        if fw:
            results = [r for r in results if r.get("framework") == fw] or results[:3]

        return {
            "status": "success",
            "action": "recommend",
            "total_matches": len(results),
            "recommendations": [
                {
                    "dag_id": r["id"],
                    "score": r["score"],
                    "framework": r.get("framework"),
                    "description": r.get("description", "")[:100],
                    "node_count": r.get("node_count"),
                    "cost": r.get("cost", {}),
                    "when_to_use": r.get("agent_context", {}).get("when_to_use", ""),
                    "differentiators": r.get("agent_context", {}).get(
                        "differentiators", []
                    ),
                }
                for r in results[:5]
            ],
            "next_step": "Call with action='get_config_guidance' and the chosen dag_id to get configuration requirements.",
        }

    elif action == "get_dag":
        index = get_catalog_index()
        entry = next((d for d in index["dags"] if d["id"] == dag_id), None)
        if not entry:
            return {"status": "error", "message": f"DAG '{dag_id}' not found"}

        # Load full DAG JSON for node/edge details
        dag_path = SHARED_DAGS_DIR / entry["path"]
        with open(dag_path) as f:
            full_data = json.load(f)

        return {
            "status": "success",
            "action": "get_dag",
            "dag_id": dag_id,
            "description": entry.get("description"),
            "framework": entry.get("framework"),
            "nodes": full_data["dag"]["nodes"],
            "edges": full_data["dag"]["edges"],
            "input_requirements": entry.get("input_requirements", {}),
            "constraints": entry.get("constraints", {}),
            "cost": entry.get("cost", {}),
            "agent_context": entry.get("agent_context", {}),
        }

    elif action == "get_config_guidance":
        index = get_catalog_index()
        entry = next((d for d in index["dags"] if d["id"] == dag_id), None)
        if not entry:
            return {"status": "error", "message": f"DAG '{dag_id}' not found"}

        ctx = entry.get("agent_context", {})
        return {
            "status": "success",
            "action": "get_config_guidance",
            "dag_id": dag_id,
            "prerequisites": ctx.get("prerequisites", []),
            "config_guidance": ctx.get("config_guidance", {}),
            "common_pitfalls": ctx.get("config_guidance", {}).get(
                "common_pitfalls", []
            ),
            "decision_tree": ctx.get("decision_tree", {}),
            "next_step": (
                "Verify prerequisites, then ask the user for values in 'user_must_provide'. "
                "Use 'safe_defaults' for unspecified parameters. "
                "Call build_and_compile(dag_path, config_path, session, role) to create the pipeline."
            ),
        }

    elif action == "list_frameworks":
        index = get_catalog_index()
        frameworks = {}
        for d in index["dags"]:
            fw = d.get("framework", "unknown")
            frameworks.setdefault(fw, 0)
            frameworks[fw] += 1
        return {
            "status": "success",
            "action": "list_frameworks",
            "frameworks": frameworks,
            "hint": "Use framework parameter in 'recommend' action to filter.",
        }

    elif action == "list_features":
        index = get_catalog_index()
        all_features = set()
        for d in index["dags"]:
            all_features.update(d.get("features", []))
        return {
            "status": "success",
            "action": "list_features",
            "features": sorted(all_features),
            "hint": "These are the features available across all DAGs. Use in recommendation queries.",
        }

    else:
        return {"status": "error", "message": f"Unknown action: {action}"}
