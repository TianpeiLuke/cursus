"""
Utility Functions for Pipeline Execution Document Generation

Maps pipeline names to configurations, DAGs, and execution document templates
using the catalog index.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ...api.dag.base_dag import PipelineDAG
from ...api.dag import import_dag_from_json
from ..shared_dags import get_catalog_index, SHARED_DAGS_DIR

logger = logging.getLogger(__name__)


def get_dag_for_pipeline(pipeline_name: str) -> Optional[PipelineDAG]:
    """Load the DAG for a pipeline by name/ID."""
    index = get_catalog_index()
    entry = next((d for d in index["dags"] if d["id"] == pipeline_name), None)
    if entry is None:
        logger.warning(f"Pipeline '{pipeline_name}' not found in catalog")
        return None
    return import_dag_from_json(str(SHARED_DAGS_DIR / entry["path"]))


def get_pipeline_metadata(pipeline_name: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a pipeline by name/ID."""
    index = get_catalog_index()
    return next((d for d in index["dags"] if d["id"] == pipeline_name), None)


def list_available_pipelines() -> List[Dict[str, Any]]:
    """List all available pipelines with metadata."""
    index = get_catalog_index()
    return index["dags"]


def validate_pipeline(pipeline_name: str) -> bool:
    """Check if a pipeline exists in the catalog."""
    index = get_catalog_index()
    return any(d["id"] == pipeline_name for d in index["dags"])
