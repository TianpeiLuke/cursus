"""
Common Pipeline Builders

Generates MODS-compatible pipeline classes from declarative config.
Eliminates the per-project boilerplate (97-285 lines → 1 function call).

Usage:
    # Generate a MODS pipeline class:
    MungedAddressPipelineNA = build_mods_pipeline(
        author="bjjin",
        version="0.0.5",
        description="Munged Address Detection DistilBERT Training Pipeline",
        dag_path="pipeline_config/dag_NA.json",
        config_path="pipeline_config/config_NA.json",
    )

    # Or build and run directly (SAIS notebook):
    pipeline, report = build_and_compile(
        dag_path="pipeline_config/dag_training_NA.json",
        config_path="pipeline_config/config_training_NA.json",
        sagemaker_session=pipeline_session,
        role=role,
    )
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Type

from sagemaker import Session
from sagemaker.workflow.pipeline import Pipeline

from ...api.dag import import_dag_from_json
from ...api.dag.base_dag import PipelineDAG
from ...core.compiler.dag_compiler import PipelineDAGCompiler

logger = logging.getLogger(__name__)

try:
    from mods.mods_template import MODSTemplate

    MODS_AVAILABLE = True
except ImportError:
    MODS_AVAILABLE = False

    def MODSTemplate(author, description, version):
        def decorator(cls):
            cls._mods_author = author
            cls._mods_description = description
            cls._mods_version = version
            return cls

        return decorator


def build_and_compile(
    dag_path: str,
    config_path: str,
    sagemaker_session: Optional[Session] = None,
    role: Optional[str] = None,
) -> Tuple[Pipeline, any]:
    """
    Build and compile a pipeline from DAG + config paths. No class needed.

    Args:
        dag_path: Path to .dag.json file
        config_path: Path to config JSON
        sagemaker_session: SageMaker session
        role: IAM role ARN

    Returns:
        (Pipeline, CompilationReport)
    """
    dag = import_dag_from_json(dag_path)
    compiler = PipelineDAGCompiler(
        config_path=config_path,
        sagemaker_session=sagemaker_session,
        role=role,
    )
    pipeline, report = compiler.compile_with_report(dag=dag)
    logger.info(f"Pipeline '{pipeline.name}' compiled: {len(pipeline.steps)} steps")
    return pipeline, report


def build_mods_pipeline(
    author: str,
    version: str,
    description: str,
    dag_path: str,
    config_path: str,
    class_name: Optional[str] = None,
) -> Type:
    """
    Generate a @MODSTemplate-decorated pipeline class from declarative config.

    The generated class has the standard MODS interface:
    - __init__(sagemaker_session, execution_role, regional_alias)
    - generate_pipeline() -> Pipeline

    Args:
        author: Pipeline author alias
        version: Pipeline version string
        description: Pipeline description
        dag_path: Relative path to DAG JSON (relative to the calling module's directory)
        config_path: Relative path to config JSON
        class_name: Optional class name (default: derived from description)

    Returns:
        A @MODSTemplate decorated class ready for MODS Lambda
    """
    # Resolve paths relative to caller
    import inspect

    caller_frame = inspect.stack()[1]
    caller_dir = Path(caller_frame.filename).parent

    abs_dag_path = str(caller_dir / dag_path)
    abs_config_path = str(caller_dir / config_path)

    @MODSTemplate(author=author, description=description, version=version)
    class _GeneratedPipeline:
        def __init__(
            self,
            sagemaker_session: Optional[Session] = None,
            execution_role: Optional[str] = None,
            regional_alias: str = "NA",
        ):
            self.sagemaker_session = sagemaker_session or Session()
            self.execution_role = (
                execution_role or self.sagemaker_session.get_caller_identity_arn()
            )
            self.dag = import_dag_from_json(abs_dag_path)
            self.dag_compiler = PipelineDAGCompiler(
                config_path=abs_config_path,
                sagemaker_session=self.sagemaker_session,
                role=self.execution_role,
            )

        def generate_pipeline(self) -> Pipeline:
            pipeline, report = self.dag_compiler.compile_with_report(dag=self.dag)
            logger.info(
                f"Pipeline '{pipeline.name}' created: {len(pipeline.steps)} steps"
            )
            return pipeline

    # Set class name
    name = class_name or description.replace(" ", "").replace("-", "")[:40] + "Pipeline"
    _GeneratedPipeline.__name__ = name
    _GeneratedPipeline.__qualname__ = name

    return _GeneratedPipeline
