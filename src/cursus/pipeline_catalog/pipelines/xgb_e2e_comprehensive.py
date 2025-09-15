"""
Refactored XGBoost Complete End-to-End Pipeline

This pipeline demonstrates the new BasePipeline class structure for XGBoost workflows.

Example:
    ```python
    from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive_refactored import XGBoostE2EComprehensivePipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create pipeline instance
    pipeline_instance = XGBoostE2EComprehensivePipeline(
        config_path="path/to/config.json",
        sagemaker_session=pipeline_session,
        execution_role=role
    )
    
    # Generate pipeline
    pipeline = pipeline_instance.generate_pipeline()
    
    # Execute the pipeline
    pipeline.upsert()
    execution = pipeline.start()
    ```
"""

import logging
from typing import Optional

from sagemaker.workflow.pipeline_context import PipelineSession

from ...api.dag.base_dag import PipelineDAG
from ..shared_dags.xgboost.complete_e2e_dag import create_xgboost_complete_e2e_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..core.base_pipeline import BasePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostE2EComprehensivePipeline(BasePipeline):
    """
    XGBoost Complete End-to-End Pipeline using the new BasePipeline structure.
    
    This pipeline implements a complete XGBoost workflow from demo/demo_pipeline.ipynb:
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
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        sagemaker_session: Optional[PipelineSession] = None,
        execution_role: Optional[str] = None,
        enable_mods: bool = False,  # Regular pipeline, not MODS-enhanced
        validate: bool = True,
        **kwargs
    ):
        """
        Initialize the XGBoost E2E Comprehensive Pipeline.

        Args:
            config_path: Path to the configuration file
            sagemaker_session: SageMaker pipeline session
            execution_role: IAM role for pipeline execution
            enable_mods: Whether to enable MODS features (default: False for regular pipelines)
            validate: Whether to validate the DAG before compilation
            **kwargs: Additional arguments for template constructor
        """
        super().__init__(
            config_path=config_path,
            sagemaker_session=sagemaker_session,
            execution_role=execution_role,
            enable_mods=enable_mods,
            validate=validate,
            **kwargs
        )
        logger.info("Initialized XGBoost E2E Comprehensive Pipeline")

    def create_dag(self) -> PipelineDAG:
        """
        Create a complete XGBoost end-to-end pipeline DAG.

        Returns:
            PipelineDAG: The directed acyclic graph for the pipeline
        """
        dag = create_xgboost_complete_e2e_dag()
        logger.info(
            f"Created XGBoost complete end-to-end DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
        )
        return dag

    def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
        """
        Get enhanced DAG metadata with Zettelkasten integration for xgb_e2e_comprehensive.

        Returns:
            EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
        """
        # Create Zettelkasten metadata with comprehensive properties (keeping original metadata)
        zettelkasten_metadata = ZettelkastenMetadata(
            atomic_id="xgb_e2e_comprehensive",
            title="XGBoost Complete End-to-End Pipeline",
            single_responsibility="Complete XGBoost ML lifecycle from data loading to model registration",
            input_interface=[
                "Training dataset path",
                "calibration dataset path",
                "model hyperparameters",
                "registration config",
            ],
            output_interface=[
                "Trained XGBoost model artifact",
                "calibration metrics",
                "evaluation report",
                "registered model",
            ],
            side_effects="Creates model artifacts, evaluation reports, and registers model in SageMaker Model Registry",
            independence_level="fully_self_contained",
            node_count=10,
            edge_count=9,
            framework="xgboost",
            complexity="comprehensive",
            use_case="Complete XGBoost ML lifecycle with production deployment",
            features=[
                "training",
                "xgboost",
                "calibration",
                "evaluation",
                "registration",
                "end_to_end",
            ],
            mods_compatible=False,
            source_file="pipelines/xgb_e2e_comprehensive.py",
            migration_source="legacy_migration",
            created_date="2025-08-21",
            priority="high",
            framework_tags=["xgboost"],
            task_tags=[
                "end_to_end",
                "training",
                "calibration",
                "evaluation",
                "registration",
            ],
            complexity_tags=["comprehensive"],
            domain_tags=["machine_learning", "supervised_learning", "production_ml"],
            pattern_tags=[
                "complete_lifecycle",
                "production_ready",
                "atomic_workflow",
                "independent",
            ],
            integration_tags=["sagemaker", "s3", "model_registry"],
            quality_tags=["production_ready", "tested", "comprehensive"],
            data_tags=["tabular", "structured"],
            creation_context="Complete XGBoost ML lifecycle for production deployment",
            usage_frequency="medium",
            stability="stable",
            maintenance_burden="medium",
            estimated_runtime="60-120 minutes",
            resource_requirements="ml.m5.xlarge or higher",
            use_cases=[
                "Production deployment with complete ML lifecycle",
                "Model governance and monitoring setup",
                "Automated retraining workflows",
            ],
            skill_level="advanced",
        )

        # Create enhanced metadata using the new pattern (keeping original metadata)
        enhanced_metadata = EnhancedDAGMetadata(
            dag_id="xgb_e2e_comprehensive",
            description="Complete XGBoost end-to-end pipeline with calibration, evaluation, and registration",
            complexity="comprehensive",
            features=[
                "training",
                "xgboost",
                "calibration",
                "evaluation",
                "registration",
                "end_to_end",
            ],
            framework="xgboost",
            node_count=10,
            edge_count=9,
            zettelkasten_metadata=zettelkasten_metadata,
        )

        return enhanced_metadata


if __name__ == "__main__":
    # Example usage with new class-based approach
    import argparse
    from sagemaker import Session

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a refactored complete XGBoost end-to-end pipeline"
    )
    parser.add_argument(
        "--config-path", type=str, help="Path to the configuration file"
    )
    parser.add_argument(
        "--output-doc", type=str, help="Path to save the execution document"
    )
    parser.add_argument(
        "--upsert", action="store_true", help="Upsert the pipeline after creation"
    )
    parser.add_argument(
        "--sync-registry",
        action="store_true",
        help="Sync pipeline metadata to registry",
    )
    args = parser.parse_args()

    try:
        # Initialize session
        sagemaker_session = Session()
        role = sagemaker_session.get_caller_identity_arn()
        pipeline_session = PipelineSession()

        # Use provided config path or fallback to default
        config_path = args.config_path
        if not config_path:
            from pathlib import Path
            config_dir = Path.cwd().parent / "pipeline_config"
            config_path = str(config_dir / "config.json")

        # Create the pipeline using new class-based approach
        logger.info(f"Creating refactored XGBoost pipeline with config: {config_path}")
        
        pipeline_instance = XGBoostE2EComprehensivePipeline(
            config_path=config_path,
            sagemaker_session=pipeline_session,
            execution_role=role,
            enable_mods=False,  # Regular pipeline
            validate=True,
        )

        # Sync to registry if requested
        if args.sync_registry:
            success = pipeline_instance.sync_to_registry()
            if success:
                print("Successfully synchronized pipeline metadata to registry")
            else:
                print("Failed to synchronize pipeline metadata to registry")
            exit(0)

        # Generate pipeline
        pipeline = pipeline_instance.generate_pipeline()

        logger.info("Refactored XGBoost pipeline created successfully!")
        logger.info(f"Pipeline name: {pipeline.name}")
        logger.info(f"Template available: {pipeline_instance.get_last_template() is not None}")

        # Process execution documents if requested
        if args.output_doc:
            execution_doc = pipeline_instance.fill_execution_document({
                "training_dataset": "dataset-training",
                "calibration_dataset": "dataset-calibration",
            })
            pipeline_instance.save_execution_document(execution_doc, args.output_doc)

        # Upsert if requested
        if args.upsert:
            pipeline.upsert()
            logger.info(f"Pipeline '{pipeline.name}' upserted successfully")

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise
