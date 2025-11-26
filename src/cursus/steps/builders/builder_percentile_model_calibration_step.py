#!/usr/bin/env python
"""Builder for PercentileModelCalibration processing step.

This module defines the PercentileModelCalibrationStepBuilder class that builds a SageMaker
ProcessingStep for percentile model calibration, connecting the configuration, specification,
and script contract.
"""

import logging
from typing import Dict, List, Any

from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import ProcessingStep

from ...core.base.builder_base import StepBuilderBase
from ..configs.config_percentile_model_calibration_step import (
    PercentileModelCalibrationConfig,
)
from ..specs.percentile_model_calibration_spec import PERCENTILE_MODEL_CALIBRATION_SPEC

logger = logging.getLogger(__name__)


class PercentileModelCalibrationStepBuilder(StepBuilderBase):
    """Builder for PercentileModelCalibration processing step.

    This class builds a SageMaker ProcessingStep that performs percentile-based model
    calibration using ROC curves to map model scores to percentile rankings. This calibration
    is essential for converting raw model scores into interpretable percentile rankings
    that can be used for risk assessment and decision-making.

    Key Features:
    - ROC curve-based percentile mapping
    - Configurable binning for score discretization
    - Support for custom calibration dictionaries
    - Three types of outputs: calibration mapping, metrics, and calibrated data
    - Robust fallback to built-in calibration defaults

    Integration:
    - Works with: XGBoostTraining, XGBoostModelEval, ModelEvaluation, ModelCalibration
    - Depends on: PercentileModelCalibrationConfig, PERCENTILE_MODEL_CALIBRATION_SPEC
    - Produces: Percentile score mapping, calibration metrics, calibrated predictions

    Example:
        Single-task mode:
        ```python
        config = PercentileModelCalibrationConfig(
            score_field="prediction_score",
            n_bins=1000,
            accuracy=0.001
        )
        builder = PercentileModelCalibrationStepBuilder(config)
        step = builder.create_step(
            inputs={"evaluation_data": "s3://bucket/eval-data/"},
            outputs={
                "calibration_output": "s3://bucket/calibration/",
                "metrics_output": "s3://bucket/metrics/",
                "calibrated_data": "s3://bucket/calibrated/"
            }
        )
        ```

        Multi-task mode:
        ```python
        config = PercentileModelCalibrationConfig(
            score_fields=["task_0_prob", "task_1_prob", "task_2_prob"],
            n_bins=1000,
            accuracy=0.001
        )
        builder = PercentileModelCalibrationStepBuilder(config)
        step = builder.create_step(
            inputs={"evaluation_data": "s3://bucket/eval-data/"},
            outputs={
                "calibration_output": "s3://bucket/calibration/",
                "metrics_output": "s3://bucket/metrics/",
                "calibrated_data": "s3://bucket/calibrated/"
            }
        )
        ```

    See Also:
        ModelCalibrationStepBuilder, PercentileModelCalibrationConfig, PERCENTILE_MODEL_CALIBRATION_SPEC
    """

    def __init__(
        self,
        config,
        sagemaker_session=None,
        role=None,
        registry_manager=None,
        dependency_resolver=None,
    ):
        """Initialize the PercentileModelCalibrationStepBuilder.

        Args:
            config: Configuration object for this step
            sagemaker_session: SageMaker session
            role: IAM role for SageMaker execution
            registry_manager: Registry manager for steps
            dependency_resolver: Resolver for step dependencies

        Raises:
            ValueError: If config is not a PercentileModelCalibrationConfig instance
        """
        if not isinstance(config, PercentileModelCalibrationConfig):
            raise ValueError(
                "PercentileModelCalibrationStepBuilder requires a PercentileModelCalibrationConfig instance."
            )

        super().__init__(
            config=config,
            spec=PERCENTILE_MODEL_CALIBRATION_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: PercentileModelCalibrationConfig = config

    def validate_configuration(self) -> None:
        """Validate the provided configuration.

        This method performs comprehensive validation of all configuration parameters,
        ensuring they meet the requirements for the percentile calibration step.
        Supports both single-task (score_field) and multi-task (score_fields) modes.

        Raises:
            ValueError: If any configuration validation fails
        """
        self.log_info("Validating PercentileModelCalibrationConfig...")

        # Validate required attributes (excluding score_field/score_fields - handled separately)
        required_attrs = [
            "processing_entry_point",
            "processing_source_dir",
            "processing_instance_count",
            "processing_volume_size",
            "job_type",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [
                None,
                "",
            ]:
                raise ValueError(
                    f"PercentileModelCalibrationConfig missing required attribute: {attr}"
                )

        # Validate that at least one of score_field or score_fields is provided
        if not self.config.score_field and not self.config.score_fields:
            raise ValueError(
                "Either 'score_field' (single-task) or 'score_fields' (multi-task) must be provided"
            )

        # Validate score_fields if provided
        if self.config.score_fields:
            if not isinstance(self.config.score_fields, list):
                raise ValueError("score_fields must be a list of strings")
            if len(self.config.score_fields) == 0:
                raise ValueError("score_fields cannot be an empty list")
            if not all(isinstance(field, str) for field in self.config.score_fields):
                raise ValueError("All elements in score_fields must be strings")

        # Validate job_type
        valid_job_types = {"training", "calibration", "validation", "testing"}
        if self.config.job_type not in valid_job_types:
            raise ValueError(f"Invalid job_type: {self.config.job_type}")

        # Validate numeric parameters
        if self.config.n_bins <= 0:
            raise ValueError(f"n_bins must be > 0, got {self.config.n_bins}")

        if not 0 <= self.config.accuracy <= 1:
            raise ValueError(
                f"accuracy must be between 0 and 1, got {self.config.accuracy}"
            )

        self.log_info("PercentileModelCalibrationConfig validation succeeded.")

    def _get_environment_variables(self) -> Dict[str, str]:
        """Create environment variables for the processing job.

        Uses the config's get_environment_variables method which provides all
        environment variables needed for the percentile calibration script.

        Returns:
            Dict[str, str]: Environment variables for the processing job
        """
        # Use the config's method to get all environment variables
        env_vars = self.config.get_environment_variables()

        self.log_info("Processing environment variables: %s", env_vars)
        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for the processor using the specification and contract.

        This method maps logical input names from the step specification to
        SageMaker ProcessingInput objects required by the processing script.

        Args:
            inputs: Dictionary of input values

        Returns:
            List[ProcessingInput]: List of configured ProcessingInput objects

        Raises:
            ValueError: If spec or contract is missing
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for input mapping")

        processing_inputs = []

        # Process each dependency in the specification
        for _, dependency_spec in self.spec.dependencies.items():
            logical_name = dependency_spec.logical_name

            # Skip calibration_config as it's loaded from the script folder
            if logical_name == "calibration_config":
                self.log_info(
                    "Skipping calibration_config channel (calibration config loaded from script folder)"
                )
                continue

            # Skip if optional and not provided
            if not dependency_spec.required and logical_name not in inputs:
                continue

            # Make sure required inputs are present
            if dependency_spec.required and logical_name not in inputs:
                raise ValueError(f"Required input '{logical_name}' not provided")

            # Get container path from contract
            container_path = None
            if logical_name in self.contract.expected_input_paths:
                container_path = self.contract.expected_input_paths[logical_name]
            else:
                raise ValueError(f"No container path found for input: {logical_name}")

            # Use the input value directly - property references are handled by PipelineAssembler
            processing_inputs.append(
                ProcessingInput(
                    input_name=logical_name,
                    source=inputs[logical_name],
                    destination=container_path,
                )
            )

        return processing_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for the processor using the specification and contract.

        This method maps logical output names from the step specification to
        SageMaker ProcessingOutput objects that will be produced by the processing script.

        Args:
            outputs: Dictionary of output values

        Returns:
            List[ProcessingOutput]: List of configured ProcessingOutput objects

        Raises:
            ValueError: If spec or contract is missing
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for output mapping")

        processing_outputs = []

        # Process each output in the specification
        for _, output_spec in self.spec.outputs.items():
            logical_name = output_spec.logical_name

            # Get container path from contract
            container_path = None
            if logical_name in self.contract.expected_output_paths:
                container_path = self.contract.expected_output_paths[logical_name]
            else:
                raise ValueError(f"No container path found for output: {logical_name}")

            # Try to find destination in outputs
            destination = None

            # Look in outputs by logical name
            if logical_name in outputs:
                destination = outputs[logical_name]
            else:
                # Generate destination from config including job_type
                from sagemaker.workflow.functions import Join

                base_output_path = self._get_base_output_path()
                destination = Join(
                    on="/",
                    values=[
                        base_output_path,
                        "percentile_model_calibration",
                        self.config.job_type,
                        logical_name,
                    ],
                )
                self.log_info(
                    "Using generated destination for '%s': %s",
                    logical_name,
                    destination,
                )

            processing_outputs.append(
                ProcessingOutput(
                    output_name=logical_name,
                    source=container_path,
                    destination=destination,
                )
            )

        return processing_outputs

    def _get_processor(self) -> FrameworkProcessor:
        """Create and configure the processor for this step using FrameworkProcessor with SKLearn.

        This uses FrameworkProcessor with SKLearn estimator class to support source_dir parameter
        while maintaining the SKLearn processing environment.

        Returns:
            FrameworkProcessor: The configured processor for the step using SKLearn
        """
        # Get appropriate instance type based on configuration
        instance_type = (
            self.config.processing_instance_type_large
            if self.config.use_large_processing_instance
            else self.config.processing_instance_type_small
        )

        # Get framework version with fallback
        framework_version = getattr(
            self.config, "processing_framework_version", "1.2-1"
        )

        return FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=framework_version,
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._generate_job_name(),  # Use standardized method
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_job_arguments(self) -> List[str]:
        """Constructs the list of command-line arguments to be passed to the processing script.

        This implementation uses job_type from the configuration.

        Returns:
            A list of strings representing the command-line arguments.
        """
        # Get job_type from configuration
        job_type = self.config.job_type
        self.log_info("Setting job_type argument to: %s", job_type)

        # Return job_type argument
        return ["--job_type", job_type]

    def create_step(self, **kwargs) -> ProcessingStep:
        """Creates the final, fully configured SageMaker ProcessingStep for the pipeline.

        This method creates a ProcessingStep configured for percentile model calibration
        using the specification-driven approach.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Input data sources keyed by logical name
                - outputs: Output destinations keyed by logical name
                - dependencies: Optional list of steps that this step depends on
                - enable_caching: A boolean indicating whether to cache the results of this step

        Returns:
            ProcessingStep: A configured sagemaker.workflow.steps.ProcessingStep instance

        Raises:
            ValueError: If required inputs are missing or configuration is invalid
        """
        self.log_info("Creating PercentileModelCalibration ProcessingStep...")

        # Extract parameters
        inputs_raw = kwargs.get("inputs", {})
        outputs = kwargs.get("outputs", {})
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        # Handle inputs
        inputs = {}

        # If dependencies are provided, extract inputs from them
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)

        # Add explicitly provided inputs (overriding any extracted ones)
        inputs.update(inputs_raw)

        # Create processor and get inputs/outputs
        processor = self._get_processor()
        proc_inputs = self._get_inputs(inputs)
        proc_outputs = self._get_outputs(outputs)
        job_args = self._get_job_arguments()

        # Get step name using standardized method with auto-detection
        step_name = self._get_step_name()

        # Use FrameworkProcessor with get_run_args for pipeline compatibility
        # This supports source_dir parameter which SKLearnProcessor.run() doesn't support

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

        # Use FrameworkProcessor.run() which supports source_dir parameter and works with ProcessingStep
        step_args = processor.run(
            code=entry_point,
            source_dir=source_dir,  # FrameworkProcessor.run() supports this parameter
            inputs=proc_inputs,
            outputs=proc_outputs,
            arguments=job_args,
        )

        # Create and return the step using step_args
        processing_step = ProcessingStep(
            name=step_name,
            step_args=step_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching),
        )

        # Attach specification to the step for future reference
        if hasattr(self, "spec") and self.spec:
            setattr(processing_step, "_spec", self.spec)

        self.log_info("Created ProcessingStep with name: %s", processing_step.name)
        return processing_step
