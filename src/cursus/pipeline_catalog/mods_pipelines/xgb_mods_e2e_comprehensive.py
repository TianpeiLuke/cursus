"""
MODS XGBoost Complete End-to-End Pipeline

This pipeline implements a MODS-enhanced version of the complete XGBoost workflow:
1) Data Loading (training)
2) Preprocessing (training)
3) XGBoost Model Training
4) Model Calibration
5) Package Model
6) Payload Generation
7) Model Registration
8) Data Loading (calibration)
9) Preprocessing (calibration)
10) Model Evaluation (calibration)

This MODS variant provides enhanced functionality including:
- Automatic template registration in MODS global registry
- Enhanced metadata extraction and validation
- Integration with MODS operational tools
- Advanced pipeline tracking and monitoring

The pipeline uses the same shared DAG definition as the standard version,
ensuring consistency while providing MODS-specific features.

Example:
    ```python
    from cursus.pipeline_catalog.mods_pipelines.xgb_mods_e2e_comprehensive import create_pipeline
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
        role=role,
        enable_mods=True
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
from ..shared_dags.xgboost.complete_e2e_dag import create_xgboost_complete_e2e_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..utils.catalog_registry import CatalogRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MODS imports with fallback
try:
    from ...mods.compiler.mods_dag_compiler import MODSPipelineDAGCompiler
    MODS_AVAILABLE = True
except ImportError:
    MODS_AVAILABLE = False
    logger.warning("MODS framework not available. Pipeline will use standard compiler.")
    MODSPipelineDAGCompiler = None
    
    class MODSNotAvailableError(Exception):
        pass


def create_dag() -> PipelineDAG:
    """
    Create a complete XGBoost end-to-end pipeline DAG.
    
    This function uses the shared DAG definition to ensure consistency
    between regular and MODS pipeline variants.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = create_xgboost_complete_e2e_dag()
    logger.info(f"Created MODS XGBoost complete end-to-end DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def get_enhanced_dag_metadata() -> EnhancedDAGMetadata:
    """
    Get enhanced DAG metadata with Zettelkasten integration for xgb_mods_e2e_comprehensive.
    
    Returns:
        EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
    """
    zettelkasten_metadata = ZettelkastenMetadata(
        atomic_id="xgb_mods_e2e_comprehensive",
        title="MODS XGBoost Complete End-to-End Pipeline",
        single_responsibility="MODS-enhanced complete XGBoost ML lifecycle from data loading to model registration",
        input_interface=["Training dataset path", "calibration dataset path", "model hyperparameters", "registration config", "MODS configuration"],
        output_interface=["Trained XGBoost model artifact", "calibration metrics", "evaluation report", "registered model", "MODS template registration"],
        side_effects="Creates model artifacts, evaluation reports, registers model in SageMaker Model Registry, and registers template in MODS global registry",
        independence_level="high",
        node_count=10,
        edge_count=9,
        framework="xgboost",
        complexity="comprehensive",
        use_case="MODS-enhanced complete XGBoost ML lifecycle with production deployment and operational integration",
        features=["training", "xgboost", "calibration", "evaluation", "registration", "end_to_end", "mods_enhanced", "template_registration"],
        mods_compatible=True,
        source_file="mods_pipelines/xgb_mods_e2e_comprehensive.py",
        migration_source="mods_frameworks/xgboost/end_to_end/complete_e2e_mods.py",
        created_date="2025-08-20",
        priority="high",
        framework_tags=["xgboost", "mods"],
        task_tags=["end_to_end", "training", "calibration", "evaluation", "registration", "mods_enhanced"],
        complexity_tags=["comprehensive"],
        domain_tags=["machine_learning", "supervised_learning", "production_ml", "mods_operations"],
        pattern_tags=["complete_lifecycle", "production_ready", "mods_enhanced"],
        integration_tags=["sagemaker", "s3", "model_registry", "mods_global_registry"],
        quality_tags=["production_ready", "tested", "comprehensive", "mods_enhanced"],
        data_tags=["tabular", "structured"],
        creation_context="MODS-enhanced complete XGBoost ML lifecycle for production deployment with operational integration",
        usage_frequency="medium",
        stability="stable",
        maintenance_burden="medium",
        estimated_runtime="60-120 minutes",
        resource_requirements="ml.m5.xlarge or higher",
        use_cases=[
            "MODS-enhanced production deployment with complete ML lifecycle",
            "Model governance and monitoring with MODS operational tools",
            "Automated retraining workflows with MODS template management"
        ],
        skill_level="advanced"
    )
    
    return EnhancedDAGMetadata(
        pipeline_name="xgb_mods_e2e_comprehensive",
        description="MODS-enhanced complete XGBoost end-to-end pipeline with calibration, evaluation, and registration",
        framework="xgboost",
        task_type="end_to_end",
        complexity_level="comprehensive",
        estimated_duration_minutes=90,
        resource_requirements=["ml.m5.xlarge"],
        dependencies=["xgboost", "sagemaker", "pandas", "scikit-learn", "joblib", "mods"],
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
    validate: bool = True
) -> Tuple[Pipeline, Dict[str, Any], Union[PipelineDAGCompiler, Any], Any]:
    """
    Create a SageMaker Pipeline from the DAG for complete XGBoost end-to-end workflow with MODS features.
    
    Args:
        config_path: Path to the configuration file
        session: SageMaker pipeline session
        role: IAM role for pipeline execution
        pipeline_name: Custom name for the pipeline (optional)
        pipeline_description: Description for the pipeline (optional)
        enable_mods: Whether to enable MODS features (default: True)
        validate: Whether to validate the DAG before compilation
        
    Returns:
        Tuple containing:
            - Pipeline: The created SageMaker pipeline
            - Dict: Conversion report with details about the compilation
            - Union[PipelineDAGCompiler, MODSPipelineDAGCompiler]: The compiler instance for further operations
            - Any: The template instance (MODS template if MODS enabled, standard template otherwise)
            
    Raises:
        MODSNotAvailableError: If MODS is requested but not available in the environment
    """
    dag = create_dag()
    
    # Determine which compiler to use
    if enable_mods and MODS_AVAILABLE:
        try:
            if MODSPipelineDAGCompiler is None:
                raise MODSNotAvailableError("MODSPipelineDAGCompiler is not available")
            
            # Create MODS compiler with the configuration
            dag_compiler = MODSPipelineDAGCompiler(
                config_path=config_path,
                sagemaker_session=session,
                role=role
            )
            
            # Set optional pipeline properties
            if pipeline_name:
                dag_compiler.pipeline_name = pipeline_name
            else:
                dag_compiler.pipeline_name = "MODS-XGBoost-Complete-E2E-Pipeline"
                
            if pipeline_description:
                dag_compiler.pipeline_description = pipeline_description
            else:
                dag_compiler.pipeline_description = "MODS-enhanced complete XGBoost end-to-end pipeline with training, evaluation, registration, and deployment"
            
            logger.info("Using MODS compiler for enhanced functionality")
            
        except (MODSNotAvailableError, Exception) as e:
            logger.warning(f"MODS not available ({e}), falling back to standard compiler")
            enable_mods = False
    
    if not enable_mods or not MODS_AVAILABLE:
        # Create standard compiler with the configuration
        dag_compiler = PipelineDAGCompiler(
            config_path=config_path,
            sagemaker_session=session,
            role=role
        )
        
        # Set optional pipeline properties
        if pipeline_name:
            dag_compiler.pipeline_name = pipeline_name
        else:
            dag_compiler.pipeline_name = "XGBoost-Complete-E2E-Pipeline"
            
        if pipeline_description:
            dag_compiler.pipeline_description = pipeline_description
        else:
            dag_compiler.pipeline_description = "Complete XGBoost end-to-end pipeline with training, evaluation, registration, and deployment"
        
        logger.info("Using standard compiler")
    
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
    
    # Preview resolution (optional)
    try:
        preview = dag_compiler.preview_resolution(dag)
        logger.info("DAG node resolution preview:")
        for node, config_type in preview.node_config_map.items():
            confidence = preview.resolution_confidence.get(node, 0.0)
            logger.info(f"  {node} → {config_type} (confidence: {confidence:.2f})")
    except Exception as e:
        logger.warning(f"Preview resolution failed: {e}")
    
    # Compile the DAG into a pipeline
    pipeline, report = dag_compiler.compile_with_report(dag=dag)
    
    # Get the template instance for further operations
    if enable_mods and MODS_AVAILABLE:
        # Get the MODS decorated template instance for global registry
        template_instance = dag_compiler.get_last_template()
        if template_instance is None:
            logger.warning("MODS template instance not found after compilation")
        else:
            logger.info("MODS decorated template instance retrieved for global registry")
    else:
        # Get the standard pipeline template instance
        template_instance = dag_compiler.get_last_template()
        if template_instance is None:
            logger.warning("Pipeline template instance not found after compilation")
        else:
            logger.info("Pipeline template instance retrieved for further operations")
    
    logger.info(f"Pipeline '{pipeline.name}' created successfully")
    logger.info(f"Average resolution confidence: {report.avg_confidence:.2f}")
    
    if enable_mods and MODS_AVAILABLE:
        logger.info("MODS features enabled: template registration, enhanced metadata, operational integration")
    else:
        logger.info("Using standard pipeline features")
    
    # Sync to registry after successful pipeline creation
    sync_to_registry()
    
    return pipeline, report, dag_compiler, template_instance


def fill_execution_document(
    pipeline: Pipeline,
    document: Dict[str, Any],
    dag_compiler: Union[PipelineDAGCompiler, Any]
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
    parser = argparse.ArgumentParser(description='Create a MODS-enhanced complete XGBoost end-to-end pipeline')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    parser.add_argument('--output-doc', type=str, help='Path to save the execution document')
    parser.add_argument('--upsert', action='store_true', help='Upsert the pipeline after creation')
    parser.add_argument('--execute', action='store_true', help='Execute the pipeline after creation')
    parser.add_argument('--sync-registry', action='store_true', help='Sync pipeline metadata to registry')
    parser.add_argument('--disable-mods', action='store_true', help='Disable MODS features and use standard compiler')
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
        logger.info(f"Creating MODS pipeline with config: {config_path}")
        pipeline, report, dag_compiler, template_instance = create_pipeline(
            config_path=config_path,
            session=pipeline_session,
            role=role,
            pipeline_name="MODS-XGBoost-Complete-E2E-Pipeline",
            pipeline_description="MODS-enhanced complete XGBoost end-to-end pipeline with training, evaluation, registration, and deployment",
            enable_mods=not args.disable_mods
        )
        
        logger.info("MODS pipeline created successfully!")
        logger.info(f"Pipeline name: {pipeline.name}")
        logger.info(f"Template available: {template_instance is not None}")
        
        if not args.disable_mods and MODS_AVAILABLE:
            logger.info("MODS features: Template registration, enhanced metadata, operational integration")
        else:
            logger.info("Using standard pipeline features")
        
        # Process execution documents and pipeline operations if requested
        execution_doc = None
        if args.output_doc or args.execute:
            execution_doc = fill_execution_document(
                pipeline=pipeline,
                document={
                    "training_dataset": "dataset-training",
                    "calibration_dataset": "dataset-calibration"
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
        
    except MODSNotAvailableError as e:
        logger.error(f"MODS not available: {e}")
        logger.info("Please install MODS or use the --disable-mods flag to use standard compiler")
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise
