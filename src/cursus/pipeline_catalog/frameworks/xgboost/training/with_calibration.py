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
    pipeline, report, dag_compiler = create_pipeline(
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

from src.cursus.api.dag.base_dag import PipelineDAG
from src.cursus.core.compiler.dag_compiler import PipelineDAGCompiler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dag() -> PipelineDAG:
    """
    Create a DAG for training and calibrating an XGBoost model.
    
    This DAG represents a workflow that includes training an XGBoost model
    and then calibrating it with a separate calibration dataset.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()
    
    # Add nodes
    dag.add_node("CradleDataLoading_training")       # Data load for training
    dag.add_node("TabularPreprocessing_training")    # Tabular preprocessing for training
    dag.add_node("XGBoostTraining")                  # XGBoost training step
    dag.add_node("ModelCalibration")                 # Model calibration step
    dag.add_node("CradleDataLoading_calibration")    # Data load for calibration
    dag.add_node("TabularPreprocessing_calibration") # Tabular preprocessing for calibration
    
    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")
    dag.add_edge("XGBoostTraining", "ModelCalibration")
    
    # Calibration flow
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")
    
    # Connect calibration data to model calibration
    dag.add_edge("TabularPreprocessing_calibration", "ModelCalibration")
    
    logger.info(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def create_pipeline(
    config_path: str,
    session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    pipeline_description: Optional[str] = None,
    validate: bool = True
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler]:
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
    
    logger.info(f"Pipeline '{pipeline.name}' created successfully")
    logger.info(f"Average resolution confidence: {report.avg_confidence:.2f}")
    
    return pipeline, report, dag_compiler


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
    
    pipeline, report, dag_compiler = create_pipeline(
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
