"""
PyTorch Basic Training Pipeline

This pipeline implements a workflow for training a PyTorch model:
1) Data Loading (training)
2) Preprocessing (training) 
3) PyTorch Model Training
4) Data Loading (validation)
5) Preprocessing (validation)
6) Model Evaluation

This pipeline provides a basic framework for training and evaluating PyTorch models.
It's suitable for most standard deep learning tasks where you need to train a model
and immediately evaluate its performance on a validation dataset.

Example:
    ```python
    from cursus.pipeline_catalog.frameworks.pytorch.training.basic_training import create_pipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create the pipeline
    pipeline, report, dag_compiler = create_pipeline(
        config_path="path/to/config_pytorch.json",
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
    Create a DAG for training a PyTorch model.
    
    This DAG represents a workflow that includes training a PyTorch model
    and evaluating it with a validation dataset.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()
    
    # Add nodes
    dag.add_node("CradleDataLoading_training")      # Data load for training
    dag.add_node("TabularPreprocessing_training")   # Preprocessing for training
    dag.add_node("PyTorchTraining")                 # PyTorch training step
    dag.add_node("CradleDataLoading_validation")    # Data load for validation
    dag.add_node("TabularPreprocessing_validation") # Preprocessing for validation
    dag.add_node("PyTorchModelEval")                # Model evaluation step
    
    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "PyTorchTraining")
    
    # Evaluation flow
    dag.add_edge("CradleDataLoading_validation", "TabularPreprocessing_validation")
    dag.add_edge("TabularPreprocessing_validation", "PyTorchModelEval")
    dag.add_edge("PyTorchTraining", "PyTorchModelEval")  # Model is input to evaluation
    
    logger.info(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def create_pipeline(
    config_path: str,
    session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    pipeline_description: Optional[str] = None,
    preview_resolution: bool = True
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler]:
    """
    Create a SageMaker Pipeline from the DAG for PyTorch training.
    
    Args:
        config_path: Path to the configuration file
        session: SageMaker pipeline session
        role: IAM role for pipeline execution
        pipeline_name: Custom name for the pipeline (optional)
        pipeline_description: Description for the pipeline (optional)
        preview_resolution: Whether to preview node resolution before compilation
        
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
    
    # Preview resolution if requested
    if preview_resolution:
        preview = dag_compiler.preview_resolution(dag)
        logger.info("DAG node resolution preview:")
        for node, config_type in preview.node_config_map.items():
            confidence = preview.resolution_confidence.get(node, 0.0)
            logger.info(f"  {node} → {config_type} (confidence: {confidence:.2f})")
        
        # Log recommendations if any
        if preview.recommendations:
            logger.info("Recommendations:")
            for recommendation in preview.recommendations:
                logger.info(f"  - {recommendation}")
    
    # Compile the DAG into a pipeline
    pipeline, report = dag_compiler.compile_with_report(dag=dag)
    
    # Log compilation details
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
    import argparse
    from sagemaker import Session
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a PyTorch training pipeline')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    parser.add_argument('--upsert', action='store_true', help='Upsert the pipeline after creation')
    parser.add_argument('--execute', action='store_true', help='Execute the pipeline after creation')
    args = parser.parse_args()
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Use provided config path or fallback to default
    config_path = args.config_path
    if not config_path:
        config_dir = Path.cwd().parent / "pipeline_config"
        config_path = os.path.join(config_dir, "config_pytorch.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Default config file not found: {config_path}")
    
    # Create the pipeline
    pipeline, report, dag_compiler = create_pipeline(
        config_path=config_path,
        session=pipeline_session,
        role=role,
        pipeline_name="PyTorch-Basic-Training",
        pipeline_description="PyTorch training pipeline with model evaluation"
    )
    
    # Fill execution document if needed
    if args.execute:
        execution_doc = fill_execution_document(
            pipeline=pipeline,
            document={
                "training_dataset": "my-training-dataset",
                "validation_dataset": "my-validation-dataset"
            },
            dag_compiler=dag_compiler
        )
    
    # Upsert if requested
    if args.upsert or args.execute:
        pipeline.upsert()
        logger.info(f"Pipeline '{pipeline.name}' upserted successfully")
        
    # Execute if requested
    if args.execute:
        execution = pipeline.start(execution_input=execution_doc)
        logger.info(f"Started pipeline execution: {execution.arn}")
