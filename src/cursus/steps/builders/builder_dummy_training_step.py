"""
MIMS DummyTraining Step Builder.

This module defines the builder that creates SageMaker processing steps
for the DummyTraining component, which processes a pretrained model with
hyperparameters to make it available for downstream packaging and payload steps.
"""

import logging
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.workflow.functions import Join
from sagemaker.s3 import S3Uploader
from botocore.exceptions import ClientError

from ..configs.config_dummy_training_step import DummyTrainingConfig
from ...core.base.builder_base import StepBuilderBase
from .s3_utils import S3PathHandler
from ..specs.dummy_training_spec import DUMMY_TRAINING_SPEC

logger = logging.getLogger(__name__)


class DummyTrainingStepBuilder(StepBuilderBase):
    """Builder for DummyTraining processing step that handles pretrained model processing with hyperparameters."""

    def __init__(
        self,
        config: DummyTrainingConfig,
        sagemaker_session=None,
        role=None,
        registry_manager=None,
        dependency_resolver=None,
    ):
        """Initialize the DummyTraining step builder.

        Args:
            config: Configuration for the DummyTraining step
            sagemaker_session: SageMaker session to use
            role: IAM role for SageMaker execution
            registry_manager: Registry manager for dependency injection
            dependency_resolver: Dependency resolver for dependency injection

        Raises:
            ValueError: If config is not a DummyTrainingConfig instance
        """
        if not isinstance(config, DummyTrainingConfig):
            raise ValueError(
                "DummyTrainingStepBuilder requires a DummyTrainingConfig instance."
            )

        super().__init__(
            config=config,
            spec=DUMMY_TRAINING_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: DummyTrainingConfig = config

    def validate_configuration(self):
        """
        Validate the provided configuration.
        
        For SOURCE nodes, we rely on the base class validation which already
        handles source directory validation and other required attributes.
        """
        self.log_info("Validating DummyTraining SOURCE configuration...")
        
        # The base class (ProcessingStepConfigBase) already validates:
        # - processing_framework_version
        # - processing_instance_count  
        # - processing_volume_size
        # - processing_entry_point (if provided)
        # - effective_source_dir existence
        
        # For SOURCE nodes, we just need to ensure we have an entry point
        if not self.config.processing_entry_point:
            raise ValueError("DummyTraining SOURCE node requires processing_entry_point")
        
        self.log_info("DummyTraining SOURCE configuration validation succeeded.")

    # Removed _upload_model_to_s3 and _prepare_hyperparameters_file methods
    # SOURCE node gets model and hyperparameters from source directory

    def _get_processor(self):
        """
        Get the processor for the step.

        Returns:
            SKLearnProcessor: Configured processor for running the step
        """
        return SKLearnProcessor(
            framework_version=self.config.processing_framework_version,
            role=self.role,
            instance_type=self.config.get_instance_type(),
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._generate_job_name(),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Create environment variables for the processing job.

        Returns:
            Dict[str, str]: Environment variables for the processing job
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Add any specific environment variables needed for DummyTraining
        # For example, we could add model paths or other configuration settings

        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the processor.
        
        For SOURCE nodes, return empty list since all data comes from source directory.
        
        Returns:
            Empty list - SOURCE node has no external inputs
        """
        self.log_info("DummyTraining is a SOURCE node - no external inputs required")
        return []

    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Get outputs for the processor using the specification and contract.

        Args:
            outputs: Dictionary of output destinations keyed by logical name

        Returns:
            List of ProcessingOutput objects for the processor

        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for output mapping")

        # Use the pipeline S3 location to construct output path using base output path and Join for parameter compatibility
        from sagemaker.workflow.functions import Join
        base_output_path = self._get_base_output_path()
        default_output_path = Join(on="/", values=[base_output_path, "dummy_training", "output"])
        output_path = outputs.get("model_input", default_output_path)

        # Handle PipelineVariable objects in output_path
        if hasattr(output_path, "expr"):
            self.log_info(
                "Processing PipelineVariable for output_path: %s", output_path.expr
            )

        # Get source path from contract
        source_path = self.contract.expected_output_paths.get("model_input")
        if not source_path:
            raise ValueError(
                "Script contract missing required output path: model_input"
            )

        return [
            ProcessingOutput(
                output_name="model_input",  # Using consistent name matching specification
                source=source_path,
                destination=output_path,
            )
        ]

    def _get_job_arguments(self) -> Optional[List[str]]:
        """
        Returns None as job arguments since the dummy training script now uses
        standard paths defined directly in the script.

        Returns:
            None since no arguments are needed
        """
        self.log_info("No command-line arguments needed for dummy training script")
        return None

    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Create the processing step following the pattern from XGBoostModelEvalStepBuilder.
        
        This implementation uses processor.run() with both code and source_dir parameters,
        which is the correct pattern for ProcessingSteps that need source directory access.
        
        Args:
            **kwargs: Additional keyword arguments for step creation including:
                     - inputs: Dictionary of input sources keyed by logical name (will be empty for SOURCE node)
                     - outputs: Dictionary of output destinations keyed by logical name
                     - dependencies: List of steps this step depends on
                     - enable_caching: Whether to enable caching for this step

        Returns:
            ProcessingStep: The configured processing step
        """
        try:
            # Extract parameters
            inputs_raw = kwargs.get("inputs", {})
            outputs = kwargs.get("outputs", {})
            dependencies = kwargs.get("dependencies", [])
            enable_caching = kwargs.get("enable_caching", True)

            # Handle inputs (should be empty for SOURCE node)
            inputs = {}
            inputs.update(inputs_raw)  # Should be empty but include for consistency

            # Create processor and get inputs/outputs
            processor = self._get_processor()
            processing_inputs = self._get_inputs(inputs)  # Returns empty list for SOURCE node
            processing_outputs = self._get_outputs(outputs)

            # Get step name using standardized method with auto-detection
            step_name = self._get_step_name()

            # Get job arguments from contract
            script_args = self._get_job_arguments()

            # CRITICAL: Follow XGBoostModelEvalStepBuilder pattern for source directory
            # Use processor.run() with both code and source_dir parameters
            # For processor.run(), code parameter should be just the entry point filename
            entry_point = self.config.processing_entry_point  # Just the filename
            # Use modernized effective_source_dir with comprehensive hybrid resolution
            source_dir = self.config.effective_source_dir
            self.log_info("Using entry point: %s", entry_point)
            self.log_info("Using resolved source directory: %s", source_dir)

            # Create step arguments using processor.run()
            step_args = processor.run(
                code=entry_point,
                source_dir=source_dir,  # This ensures source directory is available in container
                inputs=processing_inputs,
                outputs=processing_outputs,
                arguments=script_args,
            )

            # Create and return the step using step_args
            processing_step = ProcessingStep(
                name=step_name,
                step_args=step_args,
                depends_on=dependencies,
                cache_config=self._get_cache_config(enable_caching),
            )

            # Store specification in step for future reference
            setattr(processing_step, "_spec", self.spec)

            return processing_step

        except Exception as e:
            self.log_error(f"Error creating DummyTraining step: {e}")
            import traceback
            self.log_error(traceback.format_exc())
            raise ValueError(f"Failed to create DummyTraining step: {str(e)}") from e
