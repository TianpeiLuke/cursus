"""
XGBoost Training with Calibration Pipeline

This pipeline implements a workflow for training an XGBoost model with calibration:
1) Data Loading (training)
2) Preprocessing (training)
3) XGBoost Model Training
4) Model Calibration
5) Data Loading (calibration) 
6) Preprocessing (calibration)

This pipeline is useful when you need to calibrate your XGBoost model's 
probability outputs to improve reliability of predictions, especially for 
classification tasks where accurate probability estimates are important.

Example:
    ```python
    from cursus.pipeline_catalog.frameworks.xgboost.training.with_calibration import create_pipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create the pipeline
    pipeline, report, dag_compiler, pipeline_template = create_pipeline(
        config_path="path/to/config.json",
        session=pipeline_session,
        role=role
    )
    
    # Execute the pipeline
    pipeline.upsert()
    execution = pipeline.start()
    ```
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from .....api.dag.base_dag import PipelineDAG
from .....core.compiler.dag_compiler import PipelineDAGCompiler
from ....shared_dags.xgboost.training_with_calibration_dag import create_xgboost_training_with_calibration_dag

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dag() -> PipelineDAG:
    """
    Create a DAG for training and calibrating an XGBoost model.
    
    This function now uses the shared DAG definition to ensure consistency
    between regular and MODS pipeline variants.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = create_xgboost_training_with_calibration_dag()
    logger.info(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def create_pipeline(
    config_path: str,
    session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    pipeline_description: Optional[str] = None,
    validate: bool = True
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler, Any]:
    """
    Create a SageMaker Pipeline from the DAG for XGBoost training with calibration.
    
    Args:
        config_path: Path to the configuration file
        session: SageMaker pipeline session
        role: IAM role for pipeline execution
        pipeline_name: Custom name for the pipeline (optional)
        pipeline_description: Description for the pipeline (optional)
        validate: Whether to validate the DAG before compilation
        
    Returns:
        Tuple containing:
            - Pipeline: The created SageMaker pipeline
            - Dict: Conversion report with details about the compilation
            - PipelineDAGCompiler: The compiler instance for further operations
            - Any: Pipeline template instance for further operations
    """
    dag = create_dag()
    
    # Create compiler with the configuration
    dag_compiler = PipelineDAGCompiler(
        config_path=config_path,
        sagemaker_session=session,
        role=role
    )
    
    # Set optional pipeline properties
    if pipeline_name:
        dag_compiler.pipeline_name = pipeline_name
    if pipeline_description:
        dag_compiler.pipeline_description = pipeline_description
    
    # Validate the DAG if requested
    if validate:
        validation = dag_compiler.validate_dag_compatibility(dag)
        if not validation.is_valid:
            logger.warning(f"DAG validation failed: {validation.summary()}")
            if validation.missing_configs:
                logger.warning(f"Missing configs: {validation.missing_configs}")
            if validation.unresolvable_builders:
                logger.warning(f"Unresolvable builders: {validation.unresolvable_builders}")
            if validation.config_errors:
                logger.warning(f"Config errors: {validation.config_errors}")
            if validation.dependency_issues:
                logger.warning(f"Dependency issues: {validation.dependency_issues}")
    
    # Compile the DAG into a pipeline
    pipeline, report = dag_compiler.compile_with_report(dag=dag)
    
    # Get the pipeline template instance for further operations
    pipeline_template = dag_compiler.get_last_template()
    if pipeline_template is None:
        logger.warning("Pipeline template instance not found after compilation")
    else:
        logger.info("Pipeline template instance retrieved for further operations")
    
    logger.info(f"Pipeline '{pipeline.name}' created successfully")
    logger.info(f"Average resolution confidence: {report.avg_confidence:.2f}")
    
    return pipeline, report, dag_compiler, pipeline_template


def fill_execution_document(
    pipeline: Pipeline,
    document: Dict[str, Any],
    dag_compiler: PipelineDAGCompiler
) -> Dict[str, Any]:
    """
    Fill an execution document for the pipeline with all necessary parameters.
    
    Args:
        pipeline: The compiled SageMaker pipeline
        document: Initial parameter document with user-provided values
        dag_compiler: The DAG compiler used to create the pipeline
    
    Returns:
        Dict: Complete execution document ready for pipeline execution
    """
    # Create execution document with all required parameters
    execution_doc = dag_compiler.create_execution_document(document)
    return execution_doc


if __name__ == "__main__":
    # Example usage
    import os
    from sagemaker import Session
    
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Assuming config file is in a standard location
    config_dir = Path.cwd().parent / "pipeline_config"
    config_path = os.path.join(config_dir, "config.json")
    
    pipeline, report, dag_compiler, pipeline_template = create_pipeline(
        config_path=config_path,
        session=pipeline_session,
        role=role,
        pipeline_name="XGBoost-Training-With-Calibration",
        pipeline_description="XGBoost training pipeline with model calibration"
    )
    
    # You can now upsert and execute the pipeline
    # pipeline.upsert()
    # execution_doc = fill_execution_document(
    #     pipeline=pipeline, 
    #     document={"training_dataset": "my-dataset", "calibration_dataset": "my-calibration-dataset"}, 
    #     dag_compiler=dag_compiler
    # )
    # execution = pipeline.start(execution_input=execution_doc)
