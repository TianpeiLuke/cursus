"""
MIMS DummyTraining Step Builder with Flexible Input Modes.

This module defines the builder that creates SageMaker processing steps
for the DummyTraining component. This step can operate in two modes:

1. INTERNAL mode: Accepts optional inputs from previous steps
   - model_artifacts_input: Model from training steps
   - hyperparameters_s3_uri: Hyperparameters from input channel

2. SOURCE mode (fallback): Reads from source directory when inputs not provided

The step processes the model by adding hyperparameters.json to model.tar.gz for
downstream packaging and payload steps.
"""

import logging
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List

from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor
from sagemaker.sklearn import SKLearn
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

        For INTERNAL nodes with optional dependencies, we validate:
        - Required processing configuration attributes
        - Entry point is specified
        - pretrained_model_path format (if provided)
        - Source directory exists (for SOURCE fallback when pretrained_model_path=None)
        """
        self.log_info("Validating DummyTraining INTERNAL configuration...")

        # The base class (ProcessingStepConfigBase) already validates:
        # - processing_framework_version
        # - processing_instance_count
        # - processing_volume_size
        # - processing_entry_point (if provided)
        # - effective_source_dir existence (for SOURCE fallback mode)

        # Validate entry point is specified
        if not self.config.processing_entry_point:
            raise ValueError("DummyTraining step requires processing_entry_point")

        # Validate pretrained_model_path (Tier 1 Essential Field)
        # Note: The config validator already handles format validation (None/S3/local)
        # Here we just log what mode will be used
        if self.config.pretrained_model_path is not None:
            self.log_info(
                f"Model artifacts will be sourced from config field: {self.config.pretrained_model_path}"
            )
        else:
            self.log_info(
                "pretrained_model_path is None - will check dependency injection or use SOURCE fallback"
            )

        self.log_info("DummyTraining INTERNAL configuration validation succeeded.")

    def _get_processor(self):
        """
        Get the processor for the step.

        Uses FrameworkProcessor with SKLearn estimator to support source_dir parameter
        while providing a Python environment suitable for dummy training processing.

        Returns:
            FrameworkProcessor: Configured processor for running the step
        """
        return FrameworkProcessor(
            estimator_cls=SKLearn,
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
        Get inputs for the processor with 3-tier priority resolution for model artifacts.

        Model Priority Resolution (3-tier system):
        1. Tier 1 (Highest): Config field pretrained_model_path
        2. Tier 2: Dependency injection via model_artifacts_input channel
        3. Tier 3 (Fallback): SOURCE mode - source_dir/models/model.tar.gz

        Hyperparameters Priority (unchanged):
        - Dependency injection via hyperparameters_s3_uri channel
        - Falls back to SOURCE mode if not provided

        Args:
            inputs: Dictionary of input sources keyed by logical name

        Returns:
            List of ProcessingInput objects (may be empty for SOURCE mode)
        """
        processing_inputs = []

        # === MODEL ARTIFACTS: 3-Tier Priority Resolution ===

        # Tier 1 (Highest): Config field pretrained_model_path
        if self.config.pretrained_model_path:
            model_source = self.config.pretrained_model_path
            model_dest = self.contract.expected_input_paths.get("model_artifacts_input")
            if model_dest:
                self.log_info(
                    f"[Tier 1 - Config] Using pretrained_model_path: {model_source}"
                )
                processing_inputs.append(
                    ProcessingInput(
                        input_name="model_artifacts_input",
                        source=model_source,
                        destination=model_dest,
                    )
                )
        # Tier 2: Dependency injection via model_artifacts_input channel
        elif "model_artifacts_input" in inputs and inputs["model_artifacts_input"]:
            model_source = inputs["model_artifacts_input"]
            model_dest = self.contract.expected_input_paths.get("model_artifacts_input")
            if model_dest:
                self.log_info(
                    f"[Tier 2 - Dependency Injection] Using model_artifacts_input channel: {model_source}"
                )
                processing_inputs.append(
                    ProcessingInput(
                        input_name="model_artifacts_input",
                        source=model_source,
                        destination=model_dest,
                    )
                )
        else:
            # Tier 3 (Fallback): SOURCE mode - handled by script
            self.log_info(
                "[Tier 3 - SOURCE Fallback] No model input provided - using source_dir/models/model.tar.gz"
            )

        # === HYPERPARAMETERS: Optional Dependency Injection ===

        # Optional: hyperparameters_s3_uri (for INTERNAL mode)
        if "hyperparameters_s3_uri" in inputs and inputs["hyperparameters_s3_uri"]:
            hparam_source = inputs["hyperparameters_s3_uri"]
            hparam_dest = self.contract.expected_input_paths.get(
                "hyperparameters_s3_uri"
            )
            if hparam_dest:
                self.log_info(
                    f"[Dependency Injection] Adding hyperparameters_s3_uri: {hparam_source}"
                )
                processing_inputs.append(
                    ProcessingInput(
                        input_name="hyperparameters_s3_uri",
                        source=hparam_source,
                        destination=hparam_dest,
                    )
                )
        else:
            self.log_info(
                "[SOURCE Fallback] No hyperparameters input - using code directory or source_dir/hyperparams/"
            )

        return processing_inputs

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

        # Use the pipeline S3 location to construct output path
        from sagemaker.workflow.functions import Join

        base_output_path = self._get_base_output_path()
        default_output_path = Join(
            on="/", values=[base_output_path, "dummy_training", "output"]
        )

        # Support both old and new logical names for backward compatibility
        output_path = outputs.get(
            "model_output", outputs.get("model_input", default_output_path)
        )

        # Handle PipelineVariable objects in output_path
        if hasattr(output_path, "expr"):
            self.log_info(
                "Processing PipelineVariable for output_path: %s", output_path.expr
            )

        # Get source path from contract (updated logical name)
        source_path = self.contract.expected_output_paths.get("model_output")
        if not source_path:
            raise ValueError(
                "Script contract missing required output path: model_output"
            )

        return [
            ProcessingOutput(
                output_name="model_output",  # Updated to match contract
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
            processing_inputs = self._get_inputs(
                inputs
            )  # Returns empty list for SOURCE node
            processing_outputs = self._get_outputs(outputs)

            # Get step name using standardized method with auto-detection
            step_name = self._get_step_name()

            # Get job arguments from contract
            script_args = self._get_job_arguments()

            # CRITICAL: Follow XGBoostModelEvalStepBuilder pattern for source directory
            # Use processor.run() with both code and source_dir parameters
            # For processor.run(), code parameter should be just the entry point filename

            # Get fully resolved script path using hybrid resolution
            # This ensures we get an absolute path even in Lambda where directories may not exist
            script_path = self.config.get_script_path()
            self.log_info("Using resolved script path: %s", script_path)

            # Split the resolved script path into source_dir and entry_point
            # This provides both the absolute directory path and the script filename
            from pathlib import Path

            script_path_obj = Path(script_path)
            source_dir = str(script_path_obj.parent)  # Absolute directory path
            entry_point = script_path_obj.name  # Just the filename

            self.log_info("Using entry point: %s", entry_point)
            self.log_info("Using source directory: %s", source_dir)

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
