"""
DAG Router — recommends or auto-selects the best DAG for a user's requirements.

Usage:
    from cursus.pipeline_catalog import recommend_dag, auto_select_dag

    # Get ranked recommendations:
    results = recommend_dag(
        framework="pytorch",
        features=["bedrock", "training", "edx_uploading"],
        task_type="incremental",
    )

    # Auto-select best match:
    dag_id, dag, score = auto_select_dag(framework="pytorch", features=["training", "calibration"])
"""

import logging
from typing import List, Optional, Dict, Any, Tuple

from ..shared_dags import get_catalog_index, load_shared_dag, SHARED_DAGS_DIR
from ...api.dag.base_dag import PipelineDAG

logger = logging.getLogger(__name__)


def recommend_dag(
    framework: Optional[str] = None,
    features: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    complexity: Optional[str] = None,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Recommend DAGs based on requirements. Returns ranked list.

    Scoring (0-1):
      - Feature overlap: |required ∩ dag_features| / |required|  (weight: 0.5)
      - Framework match: 1.0 if exact match                      (weight: 0.25)
      - Task type match: 1.0 if substring match                  (weight: 0.15)
      - Complexity match: 1.0 if exact, 0.5 if adjacent          (weight: 0.1)

    Args:
        framework: Required framework (pytorch, xgboost, lightgbm, etc.)
        features: Required features (e.g., ["training", "bedrock", "edx_uploading"])
        task_type: Task type keyword (e.g., "incremental", "end_to_end")
        complexity: Desired complexity (simple, standard, advanced, comprehensive)
        max_results: Maximum number of results to return

    Returns:
        List of dicts with 'id', 'score', 'reasoning', and all DAG metadata
    """
    index = get_catalog_index()
    scored = []

    complexity_order = ["simple", "standard", "advanced", "comprehensive"]

    for dag in index["dags"]:
        score = 0.0
        reasons = []

        # Feature overlap (weight: 0.5)
        if features:
            dag_features = set(dag.get("features", []))
            overlap = len(set(features) & dag_features)
            feature_score = overlap / len(features) if features else 0
            score += 0.5 * feature_score
            if overlap > 0:
                matched = set(features) & dag_features
                reasons.append(f"features: {','.join(sorted(matched))}")
        else:
            score += 0.5  # no filter = full score

        # Framework match (weight: 0.25)
        if framework:
            if dag.get("framework") == framework:
                score += 0.25
                reasons.append(f"framework={framework}")
            elif framework in dag.get("id", ""):
                score += 0.1
                reasons.append(f"framework partial match")
        else:
            score += 0.25

        # Task type match (weight: 0.15)
        if task_type:
            dag_task = dag.get("task_type", "")
            if task_type in dag_task or dag_task in task_type:
                score += 0.15
                reasons.append(f"task_type={dag_task}")
            elif task_type in dag.get("id", ""):
                score += 0.07
                reasons.append("task_type partial")
        else:
            score += 0.15

        # Complexity match (weight: 0.1)
        if complexity:
            dag_complexity = dag.get("complexity", "standard")
            if dag_complexity == complexity:
                score += 0.1
                reasons.append(f"complexity={complexity}")
            elif dag_complexity in complexity_order and complexity in complexity_order:
                dist = abs(
                    complexity_order.index(dag_complexity)
                    - complexity_order.index(complexity)
                )
                score += 0.1 * max(0, 1 - dist * 0.33)
        else:
            score += 0.1

        if score > 0.2:  # minimum threshold
            result = dict(dag)
            result["score"] = round(score, 3)
            result["reasoning"] = "; ".join(reasons) if reasons else "baseline match"
            scored.append(result)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:max_results]


def auto_select_dag(
    framework: Optional[str] = None,
    features: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    min_score: float = 0.6,
) -> Optional[Tuple[str, PipelineDAG, float]]:
    """
    Auto-select the best matching DAG. Returns None if no good match.

    Args:
        framework: Required framework
        features: Required features
        task_type: Task type keyword
        min_score: Minimum score threshold (0-1)

    Returns:
        Tuple of (dag_id, PipelineDAG, score) or None
    """
    results = recommend_dag(
        framework=framework, features=features, task_type=task_type, max_results=1
    )
    if not results or results[0]["score"] < min_score:
        return None

    best = results[0]
    dag = load_shared_dag(best["id"])
    logger.info(
        f"Auto-selected DAG '{best['id']}' (score={best['score']:.2f}): {best['reasoning']}"
    )
    return best["id"], dag, best["score"]


def recommend_for_agent(
    data_type: Optional[str] = None,  # "text", "tabular", "mixed"
    has_labels: bool = True,
    needs_llm: bool = False,
    multi_task: bool = False,
    incremental: bool = False,
    data_volume: Optional[
        str
    ] = None,  # "small" (<100K), "medium" (100K-10M), "large" (>10M)
    gpu_available: bool = True,
) -> List[Dict[str, Any]]:
    """
    Agent-friendly recommendation using semantic constraints.

    Returns ranked DAGs with agent_context (when_to_use, prerequisites, config_guidance).
    Designed for LLM agents to make pipeline selection decisions.
    """
    index = get_catalog_index()
    scored = []

    for dag in index["dags"]:
        score = 1.0
        reasons = []
        req = dag.get("input_requirements", {})
        constraints = dag.get("constraints", {})

        # Data type filter
        if data_type == "text" and not req.get("text_support", True):
            continue  # XGBoost can't handle text
        if (
            data_type == "tabular"
            and dag.get("framework") == "pytorch"
            and not needs_llm
        ):
            score *= 0.5  # PyTorch overkill for pure tabular

        # Multi-task filter
        if multi_task and not req.get("multi_task", False):
            score *= 0.3
        if not multi_task and req.get("multi_task", False):
            score *= 0.5

        # LLM requirement
        if needs_llm and not req.get("requires_llm", False):
            score *= 0.3
        if not needs_llm and req.get("requires_llm", False):
            score *= 0.6
            reasons.append("has LLM (not required but available)")

        # GPU constraint
        if not gpu_available and constraints.get("requires_gpu", False):
            continue  # Can't run without GPU

        # Incremental
        if incremental:
            if "incremental" in dag.get("task_type", "") or "edx_uploading" in dag.get(
                "features", []
            ):
                score *= 1.5
                reasons.append("supports incremental")
            else:
                score *= 0.4

        # Labels
        if not has_labels and not req.get("requires_llm", False):
            score *= 0.3  # No labels and no LLM = can't train

        # Data volume + batch preference
        if data_volume == "large" and "bedrock_realtime" in " ".join(
            dag.get("features", [])
        ):
            score *= 0.6
            reasons.append("realtime LLM slow for large data")

        # Normalize
        score = min(score, 1.0)
        if score > 0.2:
            result = dict(dag)
            result["score"] = round(score, 3)
            result["reasoning"] = "; ".join(reasons) if reasons else "good fit"
            scored.append(result)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:5]
