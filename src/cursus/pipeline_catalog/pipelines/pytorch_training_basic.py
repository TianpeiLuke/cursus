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
    from cursus.pipeline_catalog.pipelines.pytorch_training_basic import PyTorchTrainingBasicPipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create pipeline instance
    pipeline_instance = PyTorchTrainingBasicPipeline(
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
from ..shared_dags.pytorch.training_dag import create_pytorch_training_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..core.base_pipeline import BasePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyTorchTrainingBasicPipeline(BasePipeline):
    """
    PyTorch Basic Training Pipeline using the new BasePipeline structure.
    
    This pipeline implements a workflow for training a PyTorch model:
    1) Data Loading (training)
    2) Preprocessing (training) 
    3) PyTorch Model Training
    4) Data Loading (validation)
    5) Preprocessing (validation)
    6) Model Evaluation
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
        Initialize the PyTorch Training Basic Pipeline.

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
        logger.info("Initialized PyTorch Training Basic Pipeline")

    def create_dag(self) -> PipelineDAG:
        """
        Create a DAG for training a PyTorch model.

        This function now uses the shared DAG definition to ensure consistency
        between regular and MODS pipeline variants.

        Returns:
            PipelineDAG: The directed acyclic graph for the pipeline
        """
        dag = create_pytorch_training_dag()
        logger.info(
            f"Created PyTorch training DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
        )
        return dag

    def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
        """
        Get enhanced DAG metadata with Zettelkasten integration for pytorch_training_basic.

        Returns:
            EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
        """
        # Create Zettelkasten metadata with comprehensive properties (keeping original metadata)
        zettelkasten_metadata = ZettelkastenMetadata(
            atomic_id="pytorch_training_basic",
            title="PyTorch Basic Training Pipeline",
            single_responsibility="PyTorch model training with basic configuration",
            input_interface=[
                "Training dataset path",
                "validation dataset path",
                "model hyperparameters",
            ],
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
                "Regression with neural networks",
            ],
            skill_level="beginner",
        )

        # Create enhanced metadata using the new pattern (keeping original metadata)
        enhanced_metadata = EnhancedDAGMetadata(
            dag_id="pytorch_training_basic",
            description="PyTorch training pipeline with model evaluation",
            complexity="simple",
            features=["training", "pytorch", "deep_learning", "supervised"],
            framework="pytorch",
            node_count=6,
            edge_count=5,
            zettelkasten_metadata=zettelkasten_metadata,
        )

        return enhanced_metadata


if __name__ == "__main__":
    # Example usage with new class-based approach
    import argparse
    from sagemaker import Session

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create a PyTorch training pipeline")
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
        logger.info(f"Creating PyTorch training pipeline with config: {config_path}")
        
        pipeline_instance = PyTorchTrainingBasicPipeline(
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

        logger.info("PyTorch training pipeline created successfully!")
        logger.info(f"Pipeline name: {pipeline.name}")
        logger.info(f"Template available: {pipeline_instance.get_last_template() is not None}")

        # Process execution documents if requested
        if args.output_doc:
            execution_doc = pipeline_instance.fill_execution_document({
                "training_dataset": "my-training-dataset",
                "validation_dataset": "my-validation-dataset",
            })
            pipeline_instance.save_execution_document(execution_doc, args.output_doc)

        # Upsert if requested
        if args.upsert:
            pipeline.upsert()
            logger.info(f"Pipeline '{pipeline.name}' upserted successfully")

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise
