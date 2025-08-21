"""
MODS PyTorch Basic Training Pipeline

This pipeline implements a MODS-enhanced version of the basic PyTorch training workflow:
1) Data Loading (training)
2) Preprocessing (training)
3) PyTorch Model Training
4) Data Loading (validation)
5) Preprocessing (validation)
6) Model Evaluation

This MODS variant provides enhanced functionality including:
- Automatic template registration in MODS global registry
- Enhanced metadata extraction and validation
- Integration with MODS operational tools
- Advanced pipeline tracking and monitoring

The pipeline uses the same shared DAG definition as the standard version,
ensuring consistency while providing MODS-specific features.

Example:
    ```python
    from cursus.pipeline_catalog.mods_pipelines.pytorch_mods_training_basic import create_pipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create MODS pipeline (automatically registers with MODS global registry)
    pipeline, report, dag_compiler, mods_template = create_pipeline(
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
    Create a basic PyTorch training pipeline DAG.
    
    This function uses the shared DAG definition to ensure consistency
    between regular and MODS pipeline variants.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = create_pytorch_training_dag()
    logger.info(f"Created MODS PyTorch basic training DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def get_enhanced_dag_metadata() -> EnhancedDAGMetadata:
    """
    Get enhanced DAG metadata with Zettelkasten integration for pytorch_mods_training_basic.
    
    Returns:
        EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
    """
    zettelkasten_metadata = ZettelkastenMetadata(
        atomic_id="pytorch_mods_training_basic",
        title="MODS PyTorch Basic Training Pipeline",
        single_responsibility="MODS-enhanced PyTorch model training with basic configuration",
        input_interface=["Training dataset path", "validation dataset path", "model hyperparameters"],
        output_interface=["Trained PyTorch model artifact", "evaluation metrics", "MODS template registration"],
        side_effects="Creates model artifacts and evaluation reports in S3 and registers template in MODS global registry",
        independence_level="high",
        node_count=4,
        edge_count=3,
        framework="pytorch",
        complexity="simple",
        use_case="MODS-enhanced basic PyTorch training",
        features=["training", "pytorch", "deep_learning", "supervised", "mods", "template_registration"],
        mods_compatible=True,
        source_file="mods_pipelines/pytorch_mods_training_basic.py",
        migration_source="mods_frameworks/pytorch/training/basic_training_mods.py",
        created_date="2025-08-20",
        priority="high",
        framework_tags=["pytorch"],
        task_tags=["training", "supervised", "deep_learning", "mods", "template_registration"],
        complexity_tags=["simple", "basic"],
        domain_tags=["machine_learning", "deep_learning", "mods_integration"],
        pattern_tags=["atomic_workflow", "independent", "mods_enhanced"],
        integration_tags=["sagemaker", "s3", "mods_registry"],
        quality_tags=["production_ready", "tested", "mods_validated"],
        data_tags=["images", "text", "structured"],
        creation_context="MODS-enhanced basic PyTorch training for deep learning tasks",
        usage_frequency="high",
        stability="stable",
        maintenance_burden="low",
        estimated_runtime="30-60 minutes",
        resource_requirements="ml.m5.large or higher",
        use_cases=[
            "MODS-enhanced image classification with CNNs",
            "Text classification with transformers and template registration",
            "Regression with neural networks and operational integration"
        ],
        skill_level="beginner"
    )
    
    return EnhancedDAGMetadata(
        pipeline_name="pytorch_mods_training_basic",
        description="MODS-enhanced PyTorch training pipeline with model evaluation and template registration",
        framework="pytorch",
        task_type="training",
        complexity_level="simple",
        estimated_duration_minutes=45,
        resource_requirements=["ml.m5.large"],
        dependencies=["pytorch", "sagemaker", "mods"],
        zettelkasten_metadata=zettelkasten_metadata
    )


def sync_to_registry() -> bool:
    """
    Synchronize this pipeline's metadata to the catalog registry.
    
    Returns:
        bool: True if synchronization was successful, False otherwise
    """
    try:
        registry = CatalogRegistry()
        enhanced_metadata = get_enhanced_dag_metadata()
        
        # Add or update the pipeline node using enhanced metadata
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
    enable_mods: bool = True,
    preview_resolution: bool = True
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler, Any]:
    """
    Create a SageMaker Pipeline from the DAG for basic PyTorch training with MODS features.
    
    Args:
        config_path: Path to the configuration file
        session: SageMaker pipeline session
        role: IAM role for pipeline execution
        pipeline_name: Custom name for the pipeline (optional)
        pipeline_description: Description for the pipeline (optional)
        enable_mods: Whether to enable MODS features (default: True)
        preview_resolution: Whether to preview node resolution before compilation
        
    Returns:
        Tuple containing:
            - Pipeline: The created SageMaker pipeline
            - Dict: Conversion report with details about the compilation
            - PipelineDAGCompiler: The compiler instance for further operations
            - Any: The template instance for further operations (MODS-decorated if enabled)
    """
    dag = create_dag()
    
    # Create compiler with the configuration
    if enable_mods:
        try:
            # Try to import MODS compiler
            from ...mods.compiler.mods_dag_compiler import MODSPipelineDAGCompiler
            dag_compiler = MODSPipelineDAGCompiler(
                config_path=config_path,
                sagemaker_session=session,
                role=role
            )
            logger.info("Using MODS-enhanced compiler")
        except ImportError:
            logger.warning("MODS compiler not available, falling back to standard compiler")
            dag_compiler = PipelineDAGCompiler(
                config_path=config_path,
                sagemaker_session=session,
                role=role
            )
    else:
        dag_compiler = PipelineDAGCompiler(
            config_path=config_path,
            sagemaker_session=session,
            role=role
        )
    
    # Set optional pipeline properties
    if pipeline_name:
        dag_compiler.pipeline_name = pipeline_name
    else:
        dag_compiler.pipeline_name = "MODS-PyTorch-Basic-Training"
        
    if pipeline_description:
        dag_compiler.pipeline_description = pipeline_description
    else:
        dag_compiler.pipeline_description = "MODS-enhanced basic PyTorch training pipeline with data loading and preprocessing"
    
    # Preview resolution if requested
    if preview_resolution:
        try:
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
        except Exception as e:
            logger.warning(f"Preview resolution failed: {e}")
    
    # Compile the DAG into a pipeline
    pipeline, report = dag_compiler.compile_with_report(dag=dag)
    
    # Get the template instance for further operations
    template_instance = dag_compiler.get_last_template()
    if template_instance is None:
        logger.warning("Template instance not found after compilation")
    else:
        if enable_mods and hasattr(template_instance, 'mods_metadata'):
            logger.info("MODS-decorated template instance retrieved for global registry")
        else:
            logger.info("Template instance retrieved for further operations")
    
    # Log compilation details
    logger.info(f"Pipeline '{pipeline.name}' created successfully")
    logger.info(f"Average resolution confidence: {report.avg_confidence:.2f}")
    if enable_mods:
        logger.info("MODS features enabled: template registration, enhanced metadata, operational integration")
    
    # Sync to registry after successful pipeline creation
    sync_to_registry()
    
    return pipeline, report, dag_compiler, template_instance


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
    parser = argparse.ArgumentParser(description='Create a MODS-enhanced PyTorch basic training pipeline')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    parser.add_argument('--upsert', action='store_true', help='Upsert the pipeline after creation')
    parser.add_argument('--execute', action='store_true', help='Execute the pipeline after creation')
    parser.add_argument('--sync-registry', action='store_true', help='Sync pipeline metadata to registry')
    parser.add_argument('--disable-mods', action='store_true', help='Disable MODS features')
    args = parser.parse_args()
    
    # Sync to registry if requested
    if args.sync_registry:
        success = sync_to_registry()
        if success:
            print("Successfully synchronized pipeline metadata to registry")
        else:
            print("Failed to synchronize pipeline metadata to registry")
        exit(0)
    
    try:
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
        pipeline, report, dag_compiler, template_instance = create_pipeline(
            config_path=config_path,
            session=pipeline_session,
            role=role,
            pipeline_name="MODS-PyTorch-Basic-Training",
            pipeline_description="MODS-enhanced basic PyTorch training pipeline with data loading and preprocessing",
            enable_mods=not args.disable_mods
        )
        
        logger.info("MODS pipeline created successfully!")
        logger.info(f"Pipeline name: {pipeline.name}")
        logger.info(f"Template available: {template_instance is not None}")
        if not args.disable_mods:
            logger.info("MODS features: Template registration, enhanced metadata, operational integration")
        
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
        
    except Exception as e:
        logger.error(f"Failed to create MODS pipeline: {e}")
        if "mods" in str(e).lower():
            logger.info("Consider using --disable-mods flag or install MODS dependencies")
