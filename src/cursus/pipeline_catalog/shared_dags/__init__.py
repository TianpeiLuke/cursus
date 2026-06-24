"""
Shared DAG Definitions for Pipeline Catalog

JSON-based DAG store. Each DAG is a .dag.json file containing nodes, edges,
and metadata. The catalog_index.json provides a queryable index of all DAGs.

Usage:
    from cursus.api.dag import import_dag_from_json
    dag = import_dag_from_json("path/to/some.dag.json")

    # Or use the catalog:
    from cursus.pipeline_catalog.shared_dags import load_shared_dag, get_all_shared_dags
    dag = load_shared_dag("bedrock_pytorch_incremental_edx")
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from ...api.dag.base_dag import PipelineDAG

logger = logging.getLogger(__name__)

__all__ = ["load_shared_dag", "get_all_shared_dags", "get_catalog_index", "DAGMetadata"]

SHARED_DAGS_DIR = Path(__file__).parent
CATALOG_INDEX_PATH = SHARED_DAGS_DIR / "catalog_index.json"


def get_catalog_index() -> Dict[str, Any]:
    """Load the catalog index."""
    with open(CATALOG_INDEX_PATH) as f:
        return json.load(f)


def load_shared_dag(dag_id: str) -> PipelineDAG:
    """
    Load a shared DAG by ID from the JSON catalog.

    Args:
        dag_id: DAG identifier (e.g., "bedrock_pytorch_incremental_edx")

    Returns:
        PipelineDAG ready for compilation
    """
    from ...api.dag import import_dag_from_json

    index = get_catalog_index()
    entry = next((d for d in index["dags"] if d["id"] == dag_id), None)
    if entry is None:
        available = [d["id"] for d in index["dags"]]
        raise ValueError(f"DAG '{dag_id}' not found. Available: {available}")

    dag_path = str(SHARED_DAGS_DIR / entry["path"])
    return import_dag_from_json(dag_path)


def get_all_shared_dags() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all available shared DAGs from the catalog index.

    Returns:
        Dict mapping DAG id to metadata dict
    """
    index = get_catalog_index()
    return {d["id"]: d for d in index["dags"]}


def list_dags_by_framework(framework: str) -> List[Dict[str, Any]]:
    """List all DAGs for a given framework."""
    index = get_catalog_index()
    return [d for d in index["dags"] if d["framework"] == framework]


def search_dags(features: Optional[List[str]] = None, framework: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search DAGs by features and/or framework.

    Args:
        features: List of required features (e.g., ["training", "bedrock", "edx_uploading"])
        framework: Framework filter (e.g., "pytorch")

    Returns:
        List of matching DAG entries, sorted by feature overlap
    """
    index = get_catalog_index()
    results = []

    for dag in index["dags"]:
        if framework and dag["framework"] != framework:
            continue
        if features:
            overlap = len(set(features) & set(dag.get("features", [])))
            if overlap == 0:
                continue
            dag_copy = dict(dag)
            dag_copy["_score"] = overlap / len(features)
            results.append(dag_copy)
        else:
            results.append(dag)

    results.sort(key=lambda d: d.get("_score", 0), reverse=True)
    return results


# Backward compat: DAGMetadata kept as import target
class DAGMetadata:
    """Legacy metadata class. Use catalog_index.json instead."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.extra_metadata = kwargs.get("extra_metadata", {})


def validate_dag_metadata(metadata) -> bool:
    """Legacy validation. Always returns True (JSON schema handles validation)."""
    return True
