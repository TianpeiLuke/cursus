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
    from cursus.pipeline_catalog.pipelines.xgb_training_evaluation import create_pipeline
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

from ...api.dag.base_dag import PipelineDAG
from ...core.compiler.dag_compiler import PipelineDAGCompiler
from ..shared_dags.xgboost.training_with_evaluation_dag import create_xgboost_training_with_evaluation_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..utils.catalog_registry import CatalogRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dag() -> PipelineDAG:
    """
    Create a DAG for training and evaluating an XGBoost model.
    
    This function now uses the shared DAG definition to ensure consistency
    between regular and MODS pipeline variants.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = create_xgboost_training_with_evaluation_dag()
    logger.info(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def get_enhanced_dag_metadata() -> EnhancedDAGMetadata:
    """
    Get enhanced DAG metadata with Zettelkasten integration for xgb_training_evaluation.
    
    Returns:
        EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
    """
    # Create Zettelkasten metadata with comprehensive properties
    zettelkasten_metadata = ZettelkastenMetadata(
        atomic_id="xgb_training_evaluation",
        title="XGBoost Training with Evaluation Pipeline",
        single_responsibility="XGBoost model training with comprehensive evaluation",
        input_interface=["Training dataset path", "evaluation dataset path", "model hyperparameters"],
        output_interface=["Trained XGBoost model artifact", "evaluation metrics", "performance report"],
        side_effects="Creates model artifacts and evaluation reports in S3",
        independence_level="fully_self_contained",
        node_count=6,
        edge_count=5,
        framework="xgboost",
        complexity="standard",
        use_case="XGBoost training with model evaluation",
        features=["training", "xgboost", "evaluation", "supervised"],
        mods_compatible=False,
        source_file="pipelines/xgb_training_evaluation.py",
        migration_source="legacy_migration",
        created_date="2025-08-21",
        priority="high",
        framework_tags=["xgboost"],
        task_tags=["training", "evaluation", "supervised"],
        complexity_tags=["standard"],
        domain_tags=["machine_learning", "supervised_learning"],
        pattern_tags=["atomic_workflow", "evaluation", "independent"],
        integration_tags=["sagemaker", "s3"],
        quality_tags=["production_ready", "tested", "evaluated"],
        data_tags=["tabular", "structured"],
        creation_context="XGBoost training with comprehensive model evaluation",
        usage_frequency="high",
        stability="stable",
        maintenance_burden="low",
        estimated_runtime="20-45 minutes",
        resource_requirements="ml.m5.large or higher",
        use_cases=[
            "Model validation and performance assessment",
            "Production readiness evaluation",
            "Model comparison and selection"
        ],
        skill_level="intermediate"
    )
    
    # Create enhanced metadata using the new pattern
    enhanced_metadata = EnhancedDAGMetadata(
        dag_id="xgb_training_evaluation",
        description="XGBoost training pipeline with model evaluation",
        complexity="standard",
        features=["training", "xgboost", "evaluation", "supervised"],
        framework="xgboost",
        node_count=6,
        edge_count=5,
        zettelkasten_metadata=zettelkasten_metadata
    )
    
    return enhanced_metadata


def sync_to_registry() -> bool:
    """
    Synchronize this pipeline's metadata to the catalog registry.
    
    Returns:
        bool: True if synchronization was successful, False otherwise
    """
    try:
        registry = CatalogRegistry()
        enhanced_metadata = get_enhanced_dag_metadata()
        
        # Add or update the pipeline node using the enhanced metadata
        success = registry.add_or_update_enhanced_node(enhanced_metadata)
        
        if success:
            logger.info(f"Successfully synchronized {enhanced_metadata.zettelkasten_metadata.atomic_id} to registry")
        else:
            logger.warning(f"Failed to synchronize {enhanced_metadata.zettelkasten_metadata.atomic_id} to registry")
            
        return success
        
    except Exception as e:
        logger.error(f"Error synchronizing to registry: {e}")
        return False


def create_pipeline(
    config_path: str,
    session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    pipeline_description: Optional[str] = None,
    preview_resolution: bool = True
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler, Any]:
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
    
    # Get the pipeline template instance for further operations
    pipeline_template = dag_compiler.get_last_template()
    if pipeline_template is None:
        logger.warning("Pipeline template instance not found after compilation")
    else:
        logger.info("Pipeline template instance retrieved for further operations")
    
    # Log compilation details
    logger.info(f"Pipeline '{pipeline.name}' created successfully")
    logger.info(f"Average resolution confidence: {report.avg_confidence:.2f}")
    for node, details in report.resolution_details.items():
        logger.debug(f"  {node} → {details['config_type']} ({details['builder_type']})")
    
    # Sync to registry after successful pipeline creation
    sync_to_registry()
    
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
    import argparse
    from sagemaker import Session
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create an XGBoost training with evaluation pipeline')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    parser.add_argument('--upsert', action='store_true', help='Upsert the pipeline after creation')
    parser.add_argument('--execute', action='store_true', help='Execute the pipeline after creation')
    parser.add_argument('--sync-registry', action='store_true', help='Sync pipeline metadata to registry')
    parser.add_argument('--save-execution-doc', type=str, help='Save execution document to specified path')
    args = parser.parse_args()
    
    # Sync to registry if requested
    if args.sync_registry:
        success = sync_to_registry()
        if success:
            print("Successfully synchronized pipeline metadata to registry")
        else:
            print("Failed to synchronize pipeline metadata to registry")
        exit(0)
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Use provided config path or fallback to default
    config_path = args.config_path
    if not config_path:
        config_dir = Path.cwd().parent / "pipeline_config"
        config_path = os.path.join(config_dir, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Default config file not found: {config_path}")
    
    # Create the pipeline
    pipeline, report, dag_compiler, pipeline_template = create_pipeline(
        config_path=config_path,
        session=pipeline_session,
        role=role,
        pipeline_name="XGBoost-Training-With-Evaluation",
        pipeline_description="XGBoost training pipeline with model evaluation"
    )
    
    # Create execution document
    execution_doc = fill_execution_document(
        pipeline=pipeline, 
        document={
            "training_dataset": "my-dataset", 
            "evaluation_dataset": "my-evaluation-dataset"
        }, 
        dag_compiler=dag_compiler
    )
    
    # Save execution document if requested
    if args.save_execution_doc:
        save_execution_document(
            document=execution_doc,
            output_path=args.save_execution_doc
        )
    
    # Upsert if requested
    if args.upsert or args.execute:
        pipeline.upsert()
        logger.info(f"Pipeline '{pipeline.name}' upserted successfully")
        
    # Note: Pipeline execution is left to the user's environment
    # Users can execute the pipeline using:
    # pipeline.upsert()
    # execution = pipeline.start(execution_input=execution_doc)
