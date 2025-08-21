"""
MODS XGBoost Simple Training Pipeline

This pipeline implements a MODS-enhanced version of the simple XGBoost training workflow:
1) Data Loading (training)
2) Preprocessing (training)
3) XGBoost Model Training
4) Data Loading (calibration)
5) Preprocessing (calibration)

This MODS variant provides enhanced functionality including:
- Automatic template registration in MODS global registry
- Enhanced metadata extraction and validation
- Integration with MODS operational tools
- Advanced pipeline tracking and monitoring

The pipeline uses the same shared DAG definition as the standard version,
ensuring consistency while providing MODS-specific features.

Example:
    ```python
    from cursus.pipeline_catalog.mods_pipelines.xgb_mods_training_simple import create_pipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create MODS pipeline (automatically registers with MODS global registry)
    pipeline, report, dag_compiler, mods_template = create_pipeline(
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
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from ...api.dag.base_dag import PipelineDAG
from ...core.compiler.dag_compiler import PipelineDAGCompiler
from ..shared_dags.xgboost.simple_dag import create_xgboost_simple_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..utils.catalog_registry import CatalogRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dag() -> PipelineDAG:
    """
    Create a simple XGBoost training pipeline DAG.
    
    This function uses the shared DAG definition to ensure consistency
    between regular and MODS pipeline variants.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = create_xgboost_simple_dag()
    logger.info(f"Created MODS XGBoost simple DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def get_enhanced_dag_metadata() -> EnhancedDAGMetadata:
    """
    Get enhanced DAG metadata with Zettelkasten integration for xgb_mods_training_simple.
    
    Returns:
        EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
    """
    # Create Zettelkasten metadata with comprehensive properties
    zettelkasten_metadata = ZettelkastenMetadata(
        atomic_id="xgb_mods_training_simple",
        title="MODS XGBoost Simple Training Pipeline",
        single_responsibility="MODS-enhanced simple XGBoost model training with basic configuration",
        input_interface=["Training dataset path", "calibration dataset path", "model hyperparameters", "MODS configuration"],
        output_interface=["Trained XGBoost model artifact", "MODS template registration"],
        side_effects="Creates model artifacts in S3 and registers template in MODS global registry",
        independence_level="fully_self_contained",
        node_count=5,
        edge_count=4,
        framework="xgboost",
        complexity="simple",
        use_case="MODS-enhanced basic XGBoost training",
        features=["training", "xgboost", "mods_enhanced", "template_registration"],
        mods_compatible=True,
        source_file="mods_pipelines/xgb_mods_training_simple.py",
        migration_source="legacy_migration",
        created_date="2025-08-21",
        priority="high",
        framework_tags=["xgboost", "mods"],
        task_tags=["training", "mods_enhanced", "template_registration"],
        complexity_tags=["simple", "basic"],
        domain_tags=["machine_learning", "mods_operations"],
        pattern_tags=["atomic_workflow", "mods_enhanced", "independent"],
        integration_tags=["sagemaker", "s3", "mods_global_registry"],
        quality_tags=["production_ready", "mods_enhanced"],
        data_tags=["tabular", "structured"],
        creation_context="MODS-enhanced basic XGBoost training with template registration",
        usage_frequency="high",
        stability="stable",
        maintenance_burden="low",
        estimated_runtime="15-30 minutes",
        resource_requirements="ml.m5.large or equivalent",
        use_cases=[
            "MODS-enhanced basic XGBoost training",
            "Template registration for operational workflows",
            "Automated pipeline tracking and monitoring"
        ],
        skill_level="beginner"
    )
    
    # Create enhanced metadata using the new pattern
    enhanced_metadata = EnhancedDAGMetadata(
        dag_id="xgb_mods_training_simple",
        description="MODS-enhanced simple XGBoost training pipeline with template registration",
        complexity="simple",
        features=["training", "xgboost", "mods_enhanced", "template_registration"],
        framework="xgboost",
        node_count=5,
        edge_count=4,
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
    enable_mods: bool = True
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler, Any]:
    """
    Create a SageMaker Pipeline from the DAG for simple XGBoost training with MODS features.
    
    Args:
        config_path: Path to the configuration file
        session: SageMaker pipeline session
        role: IAM role for pipeline execution
        pipeline_name: Custom name for the pipeline (optional)
        pipeline_description: Description for the pipeline (optional)
        enable_mods: Whether to enable MODS features (default: True)
        
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
        dag_compiler.pipeline_name = "MODS-XGBoost-Simple-Training"
        
    if pipeline_description:
        dag_compiler.pipeline_description = pipeline_description
    else:
        dag_compiler.pipeline_description = "MODS-enhanced simple XGBoost training pipeline with data loading and preprocessing"
    
    # Preview resolution if requested
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
    parser = argparse.ArgumentParser(description='Create a MODS-enhanced XGBoost simple training pipeline')
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
            config_path = os.path.join(config_dir, "config.json")
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Default config file not found: {config_path}")
        
        # Create the pipeline
        pipeline, report, dag_compiler, template_instance = create_pipeline(
            config_path=config_path,
            session=pipeline_session,
            role=role,
            pipeline_name="MODS-XGBoost-Simple-Training",
            pipeline_description="MODS-enhanced simple XGBoost training pipeline with data loading and preprocessing",
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
                    "calibration_dataset": "my-calibration-dataset"
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
