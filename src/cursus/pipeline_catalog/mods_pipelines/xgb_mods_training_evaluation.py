"""
MODS XGBoost Training with Evaluation Pipeline

This pipeline implements a MODS-enhanced version of the XGBoost training workflow with evaluation:
1) Data Loading (training)
2) Preprocessing (training) 
3) XGBoost Model Training
4) Data Loading (evaluation)
5) Preprocessing (evaluation)
6) Model Evaluation

This MODS variant provides enhanced functionality including:
- Automatic template registration in MODS global registry
- Enhanced metadata extraction and validation
- Integration with MODS operational tools
- Advanced pipeline tracking and monitoring

Example:
    ```python
    from cursus.pipeline_catalog.mods_pipelines.xgb_mods_training_evaluation import create_pipeline
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
from ..shared_dags.xgboost.training_with_evaluation_dag import create_xgboost_training_with_evaluation_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..utils.catalog_registry import CatalogRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dag() -> PipelineDAG:
    """Create an XGBoost training with evaluation pipeline DAG."""
    dag = create_xgboost_training_with_evaluation_dag()
    logger.info(f"Created MODS XGBoost training with evaluation DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def get_enhanced_dag_metadata() -> EnhancedDAGMetadata:
    """Get enhanced DAG metadata with Zettelkasten integration for xgb_mods_training_evaluation."""
    zettelkasten_metadata = ZettelkastenMetadata(
        atomic_id="xgb_mods_training_evaluation",
        title="MODS XGBoost Training with Evaluation Pipeline",
        single_responsibility="MODS-enhanced XGBoost model training with comprehensive evaluation",
        input_interface=["Training dataset path", "evaluation dataset path", "model hyperparameters"],
        output_interface=["Trained XGBoost model artifact", "evaluation metrics", "performance report", "MODS template registration"],
        side_effects="Creates model artifacts and evaluation reports in S3 and registers template in MODS global registry",
        independence_level="high",
        node_count=6,
        edge_count=5,
        framework="xgboost",
        complexity="standard",
        use_case="MODS-enhanced XGBoost training with model evaluation",
        features=["training", "xgboost", "evaluation", "supervised", "mods", "template_registration"],
        mods_compatible=True,
        source_file="mods_pipelines/xgb_mods_training_evaluation.py",
        migration_source="mods_frameworks/xgboost/training/with_evaluation_mods.py",
        created_date="2025-08-20",
        priority="high",
        framework_tags=["xgboost"],
        task_tags=["training", "evaluation", "supervised", "mods", "template_registration"],
        complexity_tags=["standard"],
        domain_tags=["machine_learning", "supervised_learning", "mods_integration"],
        pattern_tags=["atomic_workflow", "evaluation", "mods_enhanced"],
        integration_tags=["sagemaker", "s3", "mods_registry"],
        quality_tags=["production_ready", "tested", "evaluated", "mods_validated"],
        data_tags=["tabular", "structured"],
        creation_context="MODS-enhanced XGBoost training with comprehensive model evaluation",
        usage_frequency="high",
        stability="stable",
        maintenance_burden="low",
        estimated_runtime="20-45 minutes",
        resource_requirements="ml.m5.large or higher",
        use_cases=[
            "MODS-enhanced model validation and performance assessment",
            "Production readiness evaluation with template registration",
            "Model comparison and selection with operational integration"
        ],
        skill_level="intermediate"
    )
    
    return EnhancedDAGMetadata(
        pipeline_name="xgb_mods_training_evaluation",
        description="MODS-enhanced XGBoost training pipeline with model evaluation and template registration",
        framework="xgboost",
        task_type="training",
        complexity_level="standard",
        estimated_duration_minutes=35,
        resource_requirements=["ml.m5.large"],
        dependencies=["xgboost", "sagemaker", "mods"],
        zettelkasten_metadata=zettelkasten_metadata
    )


def sync_to_registry() -> bool:
    """Synchronize this pipeline's metadata to the catalog registry."""
    try:
        registry = CatalogRegistry()
        enhanced_metadata = get_enhanced_dag_metadata()
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
    """Create a SageMaker Pipeline from the DAG for XGBoost training with evaluation and MODS features."""
    dag = create_dag()
    
    # Create compiler with the configuration
    if enable_mods:
        try:
            from ...mods.compiler.mods_dag_compiler import MODSPipelineDAGCompiler
            dag_compiler = MODSPipelineDAGCompiler(config_path=config_path, sagemaker_session=session, role=role)
            logger.info("Using MODS-enhanced compiler")
        except ImportError:
            logger.warning("MODS compiler not available, falling back to standard compiler")
            dag_compiler = PipelineDAGCompiler(config_path=config_path, sagemaker_session=session, role=role)
    else:
        dag_compiler = PipelineDAGCompiler(config_path=config_path, sagemaker_session=session, role=role)
    
    # Set pipeline properties
    dag_compiler.pipeline_name = pipeline_name or "MODS-XGBoost-Training-With-Evaluation"
    dag_compiler.pipeline_description = pipeline_description or "MODS-enhanced XGBoost training pipeline with model evaluation"
    
    # Preview resolution if requested
    if preview_resolution:
        try:
            preview = dag_compiler.preview_resolution(dag)
            logger.info("DAG node resolution preview:")
            for node, config_type in preview.node_config_map.items():
                confidence = preview.resolution_confidence.get(node, 0.0)
                logger.info(f"  {node} → {config_type} (confidence: {confidence:.2f})")
            
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


def fill_execution_document(pipeline: Pipeline, document: Dict[str, Any], dag_compiler: PipelineDAGCompiler) -> Dict[str, Any]:
    """Fill an execution document for the pipeline with all necessary parameters."""
    execution_doc = dag_compiler.create_execution_document(document)
    return execution_doc


if __name__ == "__main__":
    import argparse
    from sagemaker import Session
    
    parser = argparse.ArgumentParser(description='Create a MODS-enhanced XGBoost training with evaluation pipeline')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    parser.add_argument('--upsert', action='store_true', help='Upsert the pipeline after creation')
    parser.add_argument('--execute', action='store_true', help='Execute the pipeline after creation')
    parser.add_argument('--sync-registry', action='store_true', help='Sync pipeline metadata to registry')
    parser.add_argument('--disable-mods', action='store_true', help='Disable MODS features')
    args = parser.parse_args()
    
    if args.sync_registry:
        success = sync_to_registry()
        print("Successfully synchronized pipeline metadata to registry" if success else "Failed to synchronize pipeline metadata to registry")
        exit(0)
    
    try:
        sagemaker_session = Session()
        role = sagemaker_session.get_caller_identity_arn()
        pipeline_session = PipelineSession()
        
        config_path = args.config_path
        if not config_path:
            config_dir = Path.cwd().parent / "pipeline_config"
            config_path = os.path.join(config_dir, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Default config file not found: {config_path}")
        
        pipeline, report, dag_compiler, template_instance = create_pipeline(
            config_path=config_path,
            session=pipeline_session,
            role=role,
            pipeline_name="MODS-XGBoost-Training-With-Evaluation",
            pipeline_description="MODS-enhanced XGBoost training pipeline with model evaluation",
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
                    "evaluation_dataset": "my-evaluation-dataset"
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
