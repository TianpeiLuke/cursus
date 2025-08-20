"""
XGBoost Training with Evaluation Pipeline

This pipeline implements a workflow for training and evaluating an XGBoost model:
1) Data Loading (training)
2) Preprocessing (training) 
3) XGBoost Model Training
4) Data Loading (evaluation)
5) Preprocessing (evaluation)
6) Model Evaluation

This pipeline is ideal when you need to train an XGBoost model and immediately
evaluate its performance on a separate evaluation dataset. The evaluation results
can be used to assess model quality and make informed decisions about model deployment.

Example:
    ```python
    from cursus.pipeline_catalog.frameworks.xgboost.training.with_evaluation import create_pipeline
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
    Create a DAG for training and evaluating an XGBoost model.
    
    This DAG represents a workflow that includes training an XGBoost model
    and then evaluating it with a separate evaluation dataset.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()
    
    # Add nodes
    dag.add_node("CradleDataLoading_training")      # Data load for training
    dag.add_node("TabularPreprocessing_training")   # Tabular preprocessing for training
    dag.add_node("XGBoostTraining")                 # XGBoost training step
    dag.add_node("CradleDataLoading_evaluation")    # Data load for evaluation
    dag.add_node("TabularPreprocessing_evaluation") # Tabular preprocessing for evaluation
    dag.add_node("XGBoostModelEval")                # Model evaluation step
    
    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")
    
    # Evaluation flow
    dag.add_edge("CradleDataLoading_evaluation", "TabularPreprocessing_evaluation")
    dag.add_edge("TabularPreprocessing_evaluation", "XGBoostModelEval")
    dag.add_edge("XGBoostTraining", "XGBoostModelEval")  # Model is input to evaluation
    
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
    Create a SageMaker Pipeline from the DAG for XGBoost training with evaluation.
    
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
    for node, details in report.resolution_details.items():
        logger.debug(f"  {node} → {details['config_type']} ({details['builder_type']})")
    
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
    import json
    
    # Ensure directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the document
    with open(output_path, "w") as f:
        json.dump(document, f, indent=2)
    
    logger.info(f"Execution document saved to: {output_path}")


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
        pipeline_name="XGBoost-Training-With-Evaluation",
        pipeline_description="XGBoost training pipeline with model evaluation"
    )
    
    # Create and save execution document
    execution_doc = fill_execution_document(
        pipeline=pipeline, 
        document={"training_dataset": "my-dataset", "evaluation_dataset": "my-evaluation-dataset"}, 
        dag_compiler=dag_compiler
    )
    
    save_execution_document(
        document=execution_doc,
        output_path="output/xgboost_train_evaluate_execution.json"
    )
    
    # You can now upsert and execute the pipeline
    # pipeline.upsert()
    # execution = pipeline.start(execution_input=execution_doc)
