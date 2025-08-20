"""
Cradle Data Loading Pipeline

This simple pipeline focuses only on data loading operations:
1) Data Loading (training)
2) Data Loading (validation)
3) Data Loading (testing)

This pipeline is useful when you need to extract data from Cradle for use in 
separate downstream processes or when you want to verify data loading 
configurations before building a full training pipeline.

Example:
    ```python
    from cursus.pipeline_catalog.components.cradle_dataload import create_pipeline
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
import os
import json
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
    Create a simple data loading pipeline DAG.
    
    This DAG focuses only on data loading from Cradle for different purposes:
    training, validation, and testing.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()
    
    # Add nodes for different data loading purposes
    dag.add_node("CradleDataLoading_training")    # Data load for training
    dag.add_node("CradleDataLoading_validation")  # Data load for validation
    dag.add_node("CradleDataLoading_testing")     # Data load for testing
    
    # No edges needed as data loading steps are independent
    
    logger.info(f"Created DAG with {len(dag.nodes)} nodes")
    return dag


def create_pipeline(
    config_path: str,
    session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    pipeline_description: Optional[str] = None
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler]:
    """
    Create a SageMaker Pipeline from the DAG for Cradle data loading.
    
    Args:
        config_path: Path to the configuration file
        session: SageMaker pipeline session
        role: IAM role for pipeline execution
        pipeline_name: Custom name for the pipeline (optional)
        pipeline_description: Description for the pipeline (optional)
        
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
    
    # Compile the DAG into a pipeline
    pipeline, report = dag_compiler.compile_with_report(dag=dag)
    
    logger.info(f"Pipeline '{pipeline.name}' created successfully")
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


def save_execution_document(document: Dict[str, Any], output_path: str) -> None:
    """
    Save the execution document to a file.
    
    Args:
        document: The execution document to save
        output_path: Path where to save the document
    """
    # Ensure directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the document
    with open(output_path, "w") as f:
        json.dump(document, f, indent=2)
    
    logger.info(f"Execution document saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse
    from sagemaker import Session
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a Cradle data loading pipeline')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    parser.add_argument('--output-doc', type=str, default='cradle_execution_doc.json', 
                        help='Path to save the execution document')
    parser.add_argument('--upsert', action='store_true', help='Upsert the pipeline after creation')
    args = parser.parse_args()
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Use provided config path or fallback to default
    config_path = args.config_path
    if not config_path:
        config_dir = Path.cwd().parent / "pipeline_config"
        config_path = os.path.join(config_dir, "config_cradle.json")
        
        if not os.path.exists(config_path):
            # Try generic config as fallback
            config_path = os.path.join(config_dir, "config.json")
            logger.warning(f"Cradle-specific config not found, using generic config: {config_path}")
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"No config file found at {config_path}")
    
    # Create the pipeline
    pipeline, report, dag_compiler = create_pipeline(
        config_path=config_path,
        session=pipeline_session,
        role=role,
        pipeline_name="Cradle-Data-Loading",
        pipeline_description="Pipeline for loading data from Cradle"
    )
    
    # Fill the execution document with defaults
    try:
        execution_doc = fill_execution_document(
            pipeline=pipeline,
            document={
                "training_dataset": "my-training-dataset",
                "validation_dataset": "my-validation-dataset",
                "testing_dataset": "my-testing-dataset"
            },
            dag_compiler=dag_compiler
        )
        
        # Save the execution document
        save_execution_document(execution_doc, args.output_doc)
    except Exception as e:
        logger.error(f"Failed to create or save execution document: {e}")
    
    # Upsert the pipeline if requested
    if args.upsert:
        pipeline.upsert()
        logger.info(f"Pipeline '{pipeline.name}' upserted successfully")
