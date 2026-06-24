"""
Pipeline Factory — simplified.

All pipelines follow: load DAG → compile with config → return Pipeline.
No dynamic class generation needed.
"""

import logging
from typing import Optional, Tuple
from pathlib import Path

from sagemaker import Session
from sagemaker.workflow.pipeline import Pipeline

from ...api.dag.base_dag import PipelineDAG
from ...api.dag import import_dag_from_json
from ...core.compiler.dag_compiler import PipelineDAGCompiler
from ..shared_dags import load_shared_dag

logger = logging.getLogger(__name__)


def create_pipeline(
    dag_id: Optional[str] = None,
    dag_path: Optional[str] = None,
    config_path: str = None,
    sagemaker_session: Optional[Session] = None,
    role: Optional[str] = None,
) -> Tuple[Pipeline, any]:
    """
    Create a pipeline from a DAG + config.

    Args:
        dag_id: Shared DAG ID from catalog (e.g., "bedrock_pytorch_incremental_edx")
        dag_path: Path to a .dag.json file (alternative to dag_id)
        config_path: Path to pipeline config JSON
        sagemaker_session: SageMaker session
        role: IAM role ARN

    Returns:
        Tuple of (Pipeline, CompilationReport)
    """
    if dag_id:
        dag = load_shared_dag(dag_id)
    elif dag_path:
        dag = import_dag_from_json(dag_path)
    else:
        raise ValueError("Either dag_id or dag_path is required")

    if not config_path:
        raise ValueError("config_path is required")

    compiler = PipelineDAGCompiler(
        config_path=config_path,
        sagemaker_session=sagemaker_session,
        role=role,
    )
    pipeline, report = compiler.compile_with_report(dag=dag)
    logger.info(f"Pipeline '{pipeline.name}' created: {len(pipeline.steps)} steps")
    return pipeline, report


def list_available_pipelines():
    """List all available shared DAG IDs."""
    from ..shared_dags import get_catalog_index
    index = get_catalog_index()
    return [d["id"] for d in index["dags"]]
