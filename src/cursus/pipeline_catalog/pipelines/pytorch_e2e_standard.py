"""
PyTorch End-to-End Pipeline

This pipeline implements a complete workflow for training, evaluating, packaging, 
and registering a PyTorch model:
1) Data Loading (training)
2) Preprocessing (training)
3) PyTorch Model Training
4) Package Model
5) Payload Generation
6) Model Registration
7) Data Loading (validation)
8) Preprocessing (validation) 
9) Model Evaluation

This comprehensive pipeline covers the entire ML lifecycle from data loading to
model registration and evaluation. Use this when you need a production-ready PyTorch
pipeline that handles model training through deployment.

Example:
    ```python
    from cursus.pipeline_catalog.pipelines.pytorch_e2e_standard import create_pipeline
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
from typing import Dict, Any, Tuple, Optional, Union

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from ...api.dag.base_dag import PipelineDAG
from ...core.compiler.dag_compiler import PipelineDAGCompiler
from ..shared_dags.pytorch.standard_e2e_dag import create_pytorch_standard_e2e_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..utils.catalog_registry import CatalogRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dag() -> PipelineDAG:
    """
    Create a complete end-to-end PyTorch pipeline DAG.
    
    This function now uses the shared DAG definition to ensure consistency
    between regular and MODS pipeline variants.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = create_pytorch_standard_e2e_dag()
    logger.info(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def get_enhanced_dag_metadata() -> EnhancedDAGMetadata:
    """
    Get enhanced DAG metadata with Zettelkasten integration for pytorch_e2e_standard.
    
    Returns:
        EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
    """
    # Create Zettelkasten metadata with comprehensive properties
    zettelkasten_metadata = ZettelkastenMetadata(
        atomic_id="pytorch_e2e_standard",
        title="PyTorch Standard End-to-End Pipeline",
        single_responsibility="Complete PyTorch ML lifecycle from data loading to model registration",
        input_interface=["Training dataset path", "validation dataset path", "model hyperparameters", "registration config"],
        output_interface=["Trained PyTorch model artifact", "evaluation metrics", "packaged model", "registered model"],
        side_effects="Creates model artifacts, evaluation reports, and registers model in SageMaker Model Registry",
        independence_level="fully_self_contained",
        node_count=9,
        edge_count=8,
        framework="pytorch",
        complexity="standard",
        use_case="Complete PyTorch ML lifecycle with production deployment",
        features=["training", "pytorch", "evaluation", "registration", "packaging", "end_to_end"],
        mods_compatible=False,
        source_file="pipelines/pytorch_e2e_standard.py",
        migration_source="legacy_migration",
        created_date="2025-08-21",
        priority="high",
        framework_tags=["pytorch"],
        task_tags=["end_to_end", "training", "evaluation", "registration", "packaging"],
        complexity_tags=["standard"],
        domain_tags=["deep_learning", "production_ml"],
        pattern_tags=["complete_lifecycle", "production_ready", "atomic_workflow", "independent"],
        integration_tags=["sagemaker", "s3", "model_registry"],
        quality_tags=["production_ready", "tested", "comprehensive"],
        data_tags=["images", "text", "structured"],
        creation_context="Complete PyTorch ML lifecycle for production deployment",
        usage_frequency="medium",
        stability="stable",
        maintenance_burden="medium",
        estimated_runtime="45-90 minutes",
        resource_requirements="ml.m5.xlarge or higher",
        use_cases=[
            "Production deployment with complete ML lifecycle",
            "Model governance and monitoring setup",
            "Automated retraining workflows"
        ],
        skill_level="intermediate"
    )
    
    # Create enhanced metadata using the new pattern
    enhanced_metadata = EnhancedDAGMetadata(
        dag_id="pytorch_e2e_standard",
        description="Complete PyTorch pipeline with training, evaluation, and registration",
        complexity="standard",
        features=["training", "pytorch", "evaluation", "registration", "packaging", "end_to_end"],
        framework="pytorch",
        node_count=9,
        edge_count=8,
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
    validate: bool = True
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler, Any]:
    """
    Create a SageMaker Pipeline from the DAG for a complete PyTorch workflow.
    
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
    import argparse
    from sagemaker import Session
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a complete PyTorch end-to-end pipeline')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    parser.add_argument('--output-doc', type=str, help='Path to save the execution document')
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
    logger.info(f"Creating pipeline with config: {config_path}")
    pipeline, report, dag_compiler, pipeline_template = create_pipeline(
        config_path=config_path,
        session=pipeline_session,
        role=role,
        pipeline_name="PyTorch-End-To-End",
        pipeline_description="Complete PyTorch pipeline with training, evaluation, and registration"
    )
    
    # Process execution documents and pipeline operations if requested
    execution_doc = None
    if args.output_doc or args.execute:
        execution_doc = fill_execution_document(
            pipeline=pipeline,
            document={
                "training_dataset": "dataset-training",
                "validation_dataset": "dataset-validation"
            },
            dag_compiler=dag_compiler
        )
        
        # Save the execution document if requested
        if args.output_doc:
            save_execution_document(
                document=execution_doc,
                output_path=args.output_doc
            )
    
    # Upsert if requested
    if args.upsert and not args.execute:
        pipeline.upsert()
        logger.info(f"Pipeline '{pipeline.name}' upserted successfully")
    
    # Note: Pipeline execution is left to the user's environment
    # Users can execute the pipeline using:
    # pipeline.upsert()
    # execution = pipeline.start(execution_input=execution_doc)
