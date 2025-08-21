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
    from cursus.pipeline_catalog.pipelines.pytorch_training_basic import create_pipeline
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

from ...api.dag.base_dag import PipelineDAG
from ...core.compiler.dag_compiler import PipelineDAGCompiler
from ..shared_dags.pytorch.training_dag import create_pytorch_training_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..utils.catalog_registry import CatalogRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dag() -> PipelineDAG:
    """
    Create a DAG for training a PyTorch model.
    
    This function now uses the shared DAG definition to ensure consistency
    between regular and MODS pipeline variants.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = create_pytorch_training_dag()
    logger.info(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def get_enhanced_dag_metadata() -> EnhancedDAGMetadata:
    """
    Get enhanced DAG metadata with Zettelkasten integration for pytorch_training_basic.
    
    Returns:
        EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
    """
    # Create Zettelkasten metadata with comprehensive properties
    zettelkasten_metadata = ZettelkastenMetadata(
        atomic_id="pytorch_training_basic",
        title="PyTorch Basic Training Pipeline",
        single_responsibility="PyTorch model training with basic configuration",
        input_interface=["Training dataset path", "validation dataset path", "model hyperparameters"],
        output_interface=["Trained PyTorch model artifact", "evaluation metrics"],
        side_effects="Creates model artifacts and evaluation reports in S3",
        independence_level="fully_self_contained",
        node_count=6,
        edge_count=5,
        framework="pytorch",
        complexity="simple",
        use_case="Basic PyTorch training",
        features=["training", "pytorch", "deep_learning", "supervised"],
        mods_compatible=False,
        source_file="pipelines/pytorch_training_basic.py",
        migration_source="legacy_migration",
        created_date="2025-08-21",
        priority="high",
        framework_tags=["pytorch"],
        task_tags=["training", "supervised", "deep_learning"],
        complexity_tags=["simple", "basic"],
        domain_tags=["machine_learning", "deep_learning"],
        pattern_tags=["atomic_workflow", "independent"],
        integration_tags=["sagemaker", "s3"],
        quality_tags=["production_ready", "tested"],
        data_tags=["images", "text", "structured"],
        creation_context="Basic PyTorch training for deep learning tasks",
        usage_frequency="high",
        stability="stable",
        maintenance_burden="low",
        estimated_runtime="30-60 minutes",
        resource_requirements="ml.m5.large or higher",
        use_cases=[
            "Image classification with CNNs",
            "Text classification with transformers",
            "Regression with neural networks"
        ],
        skill_level="beginner"
    )
    
    # Create enhanced metadata using the new pattern
    enhanced_metadata = EnhancedDAGMetadata(
        dag_id="pytorch_training_basic",
        description="PyTorch training pipeline with model evaluation",
        complexity="simple",
        features=["training", "pytorch", "deep_learning", "supervised"],
        framework="pytorch",
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
            - Any: The pipeline template instance for further operations
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


if __name__ == "__main__":
    # Example usage
    import argparse
    from sagemaker import Session
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a PyTorch training pipeline')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    parser.add_argument('--upsert', action='store_true', help='Upsert the pipeline after creation')
    parser.add_argument('--execute', action='store_true', help='Execute the pipeline after creation')
    parser.add_argument('--sync-registry', action='store_true', help='Sync pipeline metadata to registry')
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
        config_path = os.path.join(config_dir, "config_pytorch.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Default config file not found: {config_path}")
    
    # Create the pipeline
    pipeline, report, dag_compiler, pipeline_template = create_pipeline(
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
