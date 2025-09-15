"""
Refactored Dummy End-to-End Basic Pipeline

This pipeline demonstrates the new BasePipeline class structure that incorporates
PipelineDAGCompiler, calls pipeline DAG generator, and maintains the same interface
as the original functional approach while keeping pipeline metadata and registry integration.

This refactored version provides:
- Class-based pipeline definition using BasePipeline
- Simplified interface with same functionality
- Integration with PipelineDAGCompiler
- Pipeline metadata and registry management
- Execution document handling

Example:
    ```python
    from cursus.pipeline_catalog.pipelines.dummy_e2e_basic_refactored import DummyE2EBasicPipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create pipeline instance
    pipeline_instance = DummyE2EBasicPipeline(
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
from ..shared_dags.dummy.e2e_basic_dag import create_dummy_e2e_basic_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..core.base_pipeline import BasePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyE2EBasicPipeline(BasePipeline):
    """
    Dummy End-to-End Basic Pipeline using the new BasePipeline structure.
    
    This pipeline implements a basic dummy workflow for testing and demonstration:
    1) DummyTraining - Training step using a pretrained model
    2) Package - Model packaging step
    3) Payload - Payload testing step
    4) Registration - Model registration step
    
    This refactored version provides enhanced functionality including:
    - Class-based pipeline definition
    - Integration with PipelineDAGCompiler
    - Enhanced metadata extraction and validation
    - Pipeline tracking and monitoring
    - Registry synchronization
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
        Initialize the Dummy E2E Basic Pipeline.

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
        logger.info("Initialized Dummy E2E Basic Pipeline")

    def create_dag(self) -> PipelineDAG:
        """
        Create a dummy end-to-end basic pipeline DAG.

        This function uses the shared DAG definition to ensure consistency
        between regular and MODS pipeline variants.

        Returns:
            PipelineDAG: The directed acyclic graph for the pipeline
        """
        dag = create_dummy_e2e_basic_dag()
        logger.info(
            f"Created dummy end-to-end basic DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
        )
        return dag

    def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
        """
        Get enhanced DAG metadata with Zettelkasten integration for dummy_e2e_basic.

        Returns:
            EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
        """
        # Create Zettelkasten metadata with comprehensive properties (keeping original metadata)
        zettelkasten_metadata = ZettelkastenMetadata(
            atomic_id="dummy_e2e_basic",
            title="Dummy End-to-End Basic Pipeline",
            single_responsibility="Basic dummy workflow for testing pipeline infrastructure",
            input_interface=[
                "Dummy model configuration",
                "packaging parameters",
                "payload configuration",
                "registration config",
            ],
            output_interface=[
                "Dummy model artifact",
                "packaged model",
                "payload test results",
                "registered model",
            ],
            side_effects="Creates dummy artifacts for testing purposes",
            independence_level="fully_self_contained",
            node_count=4,
            edge_count=4,
            framework="generic",
            complexity="simple",
            use_case="Testing and demonstration pipeline",
            features=["end_to_end", "dummy", "testing", "packaging", "registration"],
            mods_compatible=False,
            source_file="pipelines/dummy_e2e_basic.py",
            migration_source="legacy_migration",
            created_date="2025-08-21",
            priority="low",
            framework_tags=["generic", "dummy"],
            task_tags=["end_to_end", "testing", "packaging", "registration"],
            complexity_tags=["simple", "basic"],
            domain_tags=["testing", "infrastructure"],
            pattern_tags=[
                "diamond_dependency",
                "testing_framework",
                "atomic_workflow",
                "independent",
            ],
            integration_tags=["sagemaker", "s3"],
            quality_tags=["testing", "demonstration"],
            data_tags=["dummy", "synthetic"],
            creation_context="Basic dummy workflow for testing pipeline infrastructure",
            usage_frequency="low",
            stability="stable",
            maintenance_burden="low",
            estimated_runtime="10-15 minutes",
            resource_requirements="ml.m5.large",
            use_cases=[
                "Pipeline infrastructure testing",
                "Development and debugging",
                "Training and demonstration",
            ],
            skill_level="beginner",
        )

        # Create enhanced metadata using the new pattern (keeping original metadata)
        enhanced_metadata = EnhancedDAGMetadata(
            dag_id="dummy_e2e_basic",
            description="Basic end-to-end pipeline with dummy training, packaging, payload preparation, and registration for testing purposes",
            complexity="simple",
            features=["end_to_end", "dummy", "testing", "packaging", "registration"],
            framework="generic",
            node_count=4,
            edge_count=4,
            zettelkasten_metadata=zettelkasten_metadata,
        )

        return enhanced_metadata


if __name__ == "__main__":
    # Example usage with new class-based approach
    import argparse
    from sagemaker import Session

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a refactored dummy end-to-end basic pipeline"
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
        logger.info(f"Creating refactored dummy pipeline with config: {config_path}")
        
        pipeline_instance = DummyE2EBasicPipeline(
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

        logger.info("Refactored dummy pipeline created successfully!")
        logger.info(f"Pipeline name: {pipeline.name}")
        logger.info(f"Template available: {pipeline_instance.get_last_template() is not None}")

        # Process execution documents if requested
        if args.output_doc:
            execution_doc = pipeline_instance.fill_execution_document({
                "dummy_model_config": "basic-config",
                "packaging_params": "standard-packaging",
            })
            pipeline_instance.save_execution_document(execution_doc, args.output_doc)

        # Upsert if requested
        if args.upsert:
            pipeline.upsert()
            logger.info(f"Pipeline '{pipeline.name}' upserted successfully")

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise
