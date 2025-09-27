from typing import Dict, Optional, Any, List
from pathlib import Path
import logging
import importlib
import tempfile
import json
import shutil
import os
import boto3
from botocore.exceptions import ClientError

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.s3 import S3Uploader

from ..configs.config_risk_table_mapping_step import RiskTableMappingConfig
from ...core.base.builder_base import StepBuilderBase
from .s3_utils import S3PathHandler

# Import step specifications
try:
    from ..specs.risk_table_mapping_training_spec import (
        RISK_TABLE_MAPPING_TRAINING_SPEC,
    )
    from ..specs.risk_table_mapping_validation_spec import (
        RISK_TABLE_MAPPING_VALIDATION_SPEC,
    )
    from ..specs.risk_table_mapping_testing_spec import RISK_TABLE_MAPPING_TESTING_SPEC
    from ..specs.risk_table_mapping_calibration_spec import (
        RISK_TABLE_MAPPING_CALIBRATION_SPEC,
    )

    SPECS_AVAILABLE = True
except ImportError:
    RISK_TABLE_MAPPING_TRAINING_SPEC = RISK_TABLE_MAPPING_VALIDATION_SPEC = (
        RISK_TABLE_MAPPING_TESTING_SPEC
    ) = RISK_TABLE_MAPPING_CALIBRATION_SPEC = None
    SPECS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RiskTableMappingStepBuilder(StepBuilderBase):
    """
    Builder for a Risk Table Mapping ProcessingStep.

    This implementation uses a specification-driven approach where inputs, outputs,
    and behavior are defined by step specifications and script contracts.
    The builder also handles generating and uploading hyperparameters.json for the step.
    """

    def __init__(
        self,
        config: RiskTableMappingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initialize with specification based on job type.

        Args:
            config: Configuration for the step
            sagemaker_session: SageMaker session
            role: IAM role
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection

        Raises:
            ValueError: If no specification is available for the job type
        """
        # Get the appropriate spec based on job type
        spec = None
        if not hasattr(config, "job_type"):
            raise ValueError("config.job_type must be specified")

        job_type = config.job_type.lower()

        # Get specification based on job type
        if job_type == "training" and RISK_TABLE_MAPPING_TRAINING_SPEC is not None:
            spec = RISK_TABLE_MAPPING_TRAINING_SPEC
        elif (
            job_type == "calibration"
            and RISK_TABLE_MAPPING_CALIBRATION_SPEC is not None
        ):
            spec = RISK_TABLE_MAPPING_CALIBRATION_SPEC
        elif (
            job_type == "validation" and RISK_TABLE_MAPPING_VALIDATION_SPEC is not None
        ):
            spec = RISK_TABLE_MAPPING_VALIDATION_SPEC
        elif job_type == "testing" and RISK_TABLE_MAPPING_TESTING_SPEC is not None:
            spec = RISK_TABLE_MAPPING_TESTING_SPEC
        else:
            # Try dynamic import
            try:
                module_path = (
                    f"..pipeline_step_specs.risk_table_mapping_{job_type}_spec"
                )
                module = importlib.import_module(module_path, package=__package__)
                spec_var_name = f"RISK_TABLE_MAPPING_{job_type.upper()}_SPEC"
                if hasattr(module, spec_var_name):
                    spec = getattr(module, spec_var_name)
            except (ImportError, AttributeError):
                self.log_warning(
                    "Could not import specification for job type: %s", job_type
                )

        if not spec:
            raise ValueError(f"No specification found for job type: {job_type}")

        self.log_info("Using specification for %s", job_type)

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: RiskTableMappingConfig = config

    def validate_configuration(self) -> None:
        """
        Validate required configuration.

        Raises:
            ValueError: If required attributes are missing
        """
        # Required processing configuration
        required_attrs = [
            "processing_instance_count",
            "processing_volume_size",
            "processing_instance_type_large",
            "processing_instance_type_small",
            "processing_framework_version",
            "use_large_processing_instance",
            "job_type",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
                raise ValueError(f"Missing required attribute: {attr}")

        # Validate job type
        if self.config.job_type not in [
            "training",
            "validation",
            "testing",
            "calibration",
        ]:
            raise ValueError(f"Invalid job_type: {self.config.job_type}")

        # Validate label name is provided
        if not self.config.label_name:
            raise ValueError("label_name must be provided")

        # For training job type, validate cat_field_list
        if self.config.job_type == "training" and not self.config.cat_field_list:
            self.log_warning(
                "cat_field_list is empty. Risk table mapping will validate fields at runtime."
            )

    def _create_processor(self) -> SKLearnProcessor:
        """
        Create the SKLearn processor for the processing job.

        Returns:
            SKLearnProcessor: Configured processor for the step
        """
        instance_type = (
            self.config.processing_instance_type_large
            if self.config.use_large_processing_instance
            else self.config.processing_instance_type_small
        )

        return SKLearnProcessor(
            framework_version=self.config.processing_framework_version,
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._generate_job_name(),  # Use standardized method with auto-detection
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Create environment variables for the processing job.

        Uses the base class implementation to get environment variables from the
        script contract (both required and optional), then adds any additional
        environment variables from config.env.

        Returns:
            Dict[str, str]: Environment variables for the processing job
        """
        # Get environment variables from contract using base class implementation
        env_vars = super()._get_environment_variables()

        # Add environment variables from config if they exist
        if hasattr(self.config, "env") and self.config.env:
            env_vars.update(self.config.env)

        self.log_info("Processing environment variables: %s", env_vars)
        return env_vars


    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.

        This method creates ProcessingInput objects for each dependency defined in the specification.
        After refactor: Only handles data inputs, hyperparameters are embedded in source directory.

        Args:
            inputs: Input data sources keyed by logical name

        Returns:
            List of ProcessingInput objects
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for input mapping")

        processing_inputs = []

        # Process each dependency in the specification
        for _, dependency_spec in self.spec.dependencies.items():
            logical_name = dependency_spec.logical_name

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
                    s3_data_distribution_type="FullyReplicated",
                )
            )
            self.log_info(
                "Added %s input from %s to %s",
                logical_name,
                inputs[logical_name],
                container_path,
            )

        return processing_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Get outputs for the step using specification and contract.

        This method creates ProcessingOutput objects for each output defined in the specification.

        Args:
            outputs: Output destinations keyed by logical name

        Returns:
            List of ProcessingOutput objects
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
                # Generate destination using base output path and Join for parameter compatibility
                from sagemaker.workflow.functions import Join
                base_output_path = self._get_base_output_path()
                destination = Join(on="/", values=[base_output_path, "risk_table_mapping", self.config.job_type, logical_name])
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

    def _get_job_arguments(self) -> List[str]:
        """
        Constructs the list of command-line arguments to be passed to the processing script.

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
        """
        Create the ProcessingStep following the pattern from XGBoostModelEvalStepBuilder.
        
        This implementation uses processor.run() with both code and source_dir parameters,
        which is the correct pattern for ProcessingSteps that need source directory access.

        Args:
            **kwargs: Step parameters including:
                - inputs: Input data sources
                - outputs: Output destinations
                - dependencies: Steps this step depends on
                - enable_caching: Whether to enable caching

        Returns:
            Configured ProcessingStep
        """
        try:
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

            # Add direct keyword arguments (e.g., DATA, METADATA from template)
            for key in ["data_input", "config_input", "risk_tables"]:
                if key in kwargs and key not in inputs:
                    inputs[key] = kwargs[key]

            # Create processor and get inputs/outputs
            processor = self._create_processor()
            proc_inputs = self._get_inputs(inputs)
            proc_outputs = self._get_outputs(outputs)
            job_args = self._get_job_arguments()

            # Get step name using standardized method with auto-detection
            step_name = self._get_step_name()

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
            setattr(processing_step, "_spec", self.spec)

            return processing_step

        except Exception as e:
            self.log_error(f"Error creating RiskTableMapping step: {e}")
            import traceback
            self.log_error(traceback.format_exc())
            raise ValueError(f"Failed to create RiskTableMapping step: {str(e)}") from e
