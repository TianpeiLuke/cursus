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
    from cursus.pipeline_catalog.pipelines.pytorch_e2e_standard import PyTorchE2EStandardPipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create pipeline instance
    pipeline_instance = PyTorchE2EStandardPipeline(
        config_path="path/to/config_pytorch.json",
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
from ..shared_dags.pytorch.standard_e2e_dag import create_pytorch_standard_e2e_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..core.base_pipeline import BasePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyTorchE2EStandardPipeline(BasePipeline):
    """
    PyTorch End-to-End Standard Pipeline using the new BasePipeline structure.
    
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
        Initialize the PyTorch E2E Standard Pipeline.

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
        logger.info("Initialized PyTorch E2E Standard Pipeline")

    def create_dag(self) -> PipelineDAG:
        """
        Create a complete end-to-end PyTorch pipeline DAG.

        This function now uses the shared DAG definition to ensure consistency
        between regular and MODS pipeline variants.

        Returns:
            PipelineDAG: The directed acyclic graph for the pipeline
        """
        dag = create_pytorch_standard_e2e_dag()
        logger.info(
            f"Created PyTorch standard end-to-end DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
        )
        return dag

    def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
        """
        Get enhanced DAG metadata with Zettelkasten integration for pytorch_e2e_standard.

        Returns:
            EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
        """
        # Create Zettelkasten metadata with comprehensive properties (keeping original metadata)
        zettelkasten_metadata = ZettelkastenMetadata(
            atomic_id="pytorch_e2e_standard",
            title="PyTorch Standard End-to-End Pipeline",
            single_responsibility="Complete PyTorch ML lifecycle from data loading to model registration",
            input_interface=[
                "Training dataset path",
                "validation dataset path",
                "model hyperparameters",
                "registration config",
            ],
            output_interface=[
                "Trained PyTorch model artifact",
                "evaluation metrics",
                "packaged model",
                "registered model",
            ],
            side_effects="Creates model artifacts, evaluation reports, and registers model in SageMaker Model Registry",
            independence_level="fully_self_contained",
            node_count=9,
            edge_count=8,
            framework="pytorch",
            complexity="standard",
            use_case="Complete PyTorch ML lifecycle with production deployment",
            features=[
                "training",
                "pytorch",
                "evaluation",
                "registration",
                "packaging",
                "end_to_end",
            ],
            mods_compatible=False,
            source_file="pipelines/pytorch_e2e_standard.py",
            migration_source="legacy_migration",
            created_date="2025-08-21",
            priority="high",
            framework_tags=["pytorch"],
            task_tags=["end_to_end", "training", "evaluation", "registration", "packaging"],
            complexity_tags=["standard"],
            domain_tags=["deep_learning", "production_ml"],
            pattern_tags=[
                "complete_lifecycle",
                "production_ready",
                "atomic_workflow",
                "independent",
            ],
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
                "Automated retraining workflows",
            ],
            skill_level="intermediate",
        )

        # Create enhanced metadata using the new pattern (keeping original metadata)
        enhanced_metadata = EnhancedDAGMetadata(
            dag_id="pytorch_e2e_standard",
            description="Complete PyTorch pipeline with training, evaluation, and registration",
            complexity="standard",
            features=[
                "training",
                "pytorch",
                "evaluation",
                "registration",
                "packaging",
                "end_to_end",
            ],
            framework="pytorch",
            node_count=9,
            edge_count=8,
            zettelkasten_metadata=zettelkasten_metadata,
        )

        return enhanced_metadata


if __name__ == "__main__":
    # Example usage with new class-based approach
    import argparse
    from sagemaker import Session

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a complete PyTorch end-to-end pipeline"
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
            config_path = str(config_dir / "config_pytorch.json")

        # Create the pipeline using new class-based approach
        logger.info(f"Creating PyTorch pipeline with config: {config_path}")
        
        pipeline_instance = PyTorchE2EStandardPipeline(
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

        logger.info("PyTorch pipeline created successfully!")
        logger.info(f"Pipeline name: {pipeline.name}")
        logger.info(f"Template available: {pipeline_instance.get_last_template() is not None}")

        # Process execution documents if requested
        if args.output_doc:
            execution_doc = pipeline_instance.fill_execution_document({
                "training_dataset": "dataset-training",
                "validation_dataset": "dataset-validation",
            })
            pipeline_instance.save_execution_document(execution_doc, args.output_doc)

        # Upsert if requested
        if args.upsert:
            pipeline.upsert()
            logger.info(f"Pipeline '{pipeline.name}' upserted successfully")

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise
