from typing import Dict, Optional, Any, List
from pathlib import Path
import logging
import importlib

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_missing_value_imputation_step import MissingValueImputationConfig
from ...core.base.builder_base import StepBuilderBase

# Import specifications based on job type
try:
    from ..specs.missing_value_imputation_spec import (
        MISSING_VALUE_IMPUTATION_SPEC,
    )
    from ..specs.missing_value_imputation_training_spec import (
        MISSING_VALUE_IMPUTATION_TRAINING_SPEC,
    )
    from ..specs.missing_value_imputation_validation_spec import (
        MISSING_VALUE_IMPUTATION_VALIDATION_SPEC,
    )
    from ..specs.missing_value_imputation_testing_spec import (
        MISSING_VALUE_IMPUTATION_TESTING_SPEC,
    )
    from ..specs.missing_value_imputation_calibration_spec import (
        MISSING_VALUE_IMPUTATION_CALIBRATION_SPEC,
    )

    SPECS_AVAILABLE = True
except ImportError:
    MISSING_VALUE_IMPUTATION_SPEC = MISSING_VALUE_IMPUTATION_TRAINING_SPEC = (
        MISSING_VALUE_IMPUTATION_VALIDATION_SPEC
    ) = MISSING_VALUE_IMPUTATION_TESTING_SPEC = MISSING_VALUE_IMPUTATION_CALIBRATION_SPEC = None
    SPECS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MissingValueImputationStepBuilder(StepBuilderBase):
    """
    Builder for a Missing Value Imputation ProcessingStep.

    This implementation uses the fully specification-driven approach where inputs, outputs,
    and behavior are defined by step specifications and script contracts. The builder handles
    statistical imputation methods (mean, median, mode, constant) with pandas-safe values.
    """

    def __init__(
        self,
        config: MissingValueImputationConfig,
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
        if job_type == "training" and MISSING_VALUE_IMPUTATION_TRAINING_SPEC is not None:
            spec = MISSING_VALUE_IMPUTATION_TRAINING_SPEC
        elif (
            job_type == "validation"
            and MISSING_VALUE_IMPUTATION_VALIDATION_SPEC is not None
        ):
            spec = MISSING_VALUE_IMPUTATION_VALIDATION_SPEC
        elif job_type == "testing" and MISSING_VALUE_IMPUTATION_TESTING_SPEC is not None:
            spec = MISSING_VALUE_IMPUTATION_TESTING_SPEC
        elif (
            job_type == "calibration"
            and MISSING_VALUE_IMPUTATION_CALIBRATION_SPEC is not None
        ):
            spec = MISSING_VALUE_IMPUTATION_CALIBRATION_SPEC
        else:
            # Fallback to default spec if available
            if MISSING_VALUE_IMPUTATION_SPEC is not None:
                spec = MISSING_VALUE_IMPUTATION_SPEC
                self.log_warning(
                    "Using default specification for job type: %s", job_type
                )
            else:
                # Try dynamic import
                try:
                    module_path = f"..pipeline_step_specs.missing_value_imputation_{job_type}_spec"
                    module = importlib.import_module(module_path, package=__package__)
                    spec_var_name = f"MISSING_VALUE_IMPUTATION_{job_type.upper()}_SPEC"
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
        self.config: MissingValueImputationConfig = config

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
            "label_field",
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

        # Validate label_field
        if not self.config.label_field or not self.config.label_field.strip():
            raise ValueError("label_field must be provided and non-empty")

        # Validate imputation strategies
        valid_numerical_strategies = {"mean", "median", "constant"}
        if self.config.default_numerical_strategy not in valid_numerical_strategies:
            raise ValueError(
                f"Invalid default_numerical_strategy: {self.config.default_numerical_strategy}. "
                f"Must be one of {valid_numerical_strategies}"
            )

        valid_categorical_strategies = {"mode", "constant"}
        if self.config.default_categorical_strategy not in valid_categorical_strategies:
            raise ValueError(
                f"Invalid default_categorical_strategy: {self.config.default_categorical_strategy}. "
                f"Must be one of {valid_categorical_strategies}"
            )

        valid_text_strategies = {"mode", "constant", "empty"}
        if self.config.default_text_strategy not in valid_text_strategies:
            raise ValueError(
                f"Invalid default_text_strategy: {self.config.default_text_strategy}. "
                f"Must be one of {valid_text_strategies}"
            )

        # Validate categorical_unique_ratio_threshold
        if not (0.0 <= self.config.categorical_unique_ratio_threshold <= 1.0):
            raise ValueError(
                "categorical_unique_ratio_threshold must be between 0.0 and 1.0"
            )

        # Validate column_strategies if provided
        if self.config.column_strategies:
            valid_strategies = {"mean", "median", "constant", "mode", "empty"}
            for column, strategy in self.config.column_strategies.items():
                if strategy not in valid_strategies:
                    raise ValueError(
                        f"Invalid strategy '{strategy}' for column '{column}'. "
                        f"Must be one of {valid_strategies}"
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

        Uses the configuration's environment_variables property which automatically
        generates all required environment variables from the config fields.

        Returns:
            Dict[str, str]: Environment variables for the processing job
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Get environment variables from config (includes all imputation settings)
        config_env_vars = self.config.environment_variables
        env_vars.update(config_env_vars)

        # Add environment variables from config.env if they exist
        if hasattr(self.config, "env") and self.config.env:
            env_vars.update(self.config.env)

        self.log_info("Processing environment variables: %s", env_vars)
        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.

        This method creates ProcessingInput objects for each dependency defined in the specification.
        For non-training modes, it handles both data_input and imputation_params_input.

        Args:
            inputs: Input data sources keyed by logical name

        Returns:
            List of ProcessingInput objects

        Raises:
            ValueError: If no specification or contract is available
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

        Raises:
            ValueError: If no specification or contract is available
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
                destination = Join(on="/", values=[base_output_path, "missing_value_imputation", self.config.job_type, logical_name])
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

        This implementation uses job_type from the configuration, which is required by the script
        and also included in the contract's expected_arguments.

        Returns:
            A list of strings representing the command-line arguments.
        """
        # Get job_type from configuration
        job_type = self.config.job_type
        self.log_info("Setting job_type argument to: %s", job_type)

        # Return job_type argument - the script uses this to determine processing mode
        return ["--job_type", job_type]

    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Create the ProcessingStep.

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

            # Add direct keyword arguments (e.g., data_input, imputation_params_input from template)
            for key in ["data_input", "imputation_params_input", "processed_data"]:
                if key in kwargs and key not in inputs:
                    inputs[key] = kwargs[key]

            # Create processor and get inputs/outputs
            processor = self._create_processor()
            proc_inputs = self._get_inputs(inputs)
            proc_outputs = self._get_outputs(outputs)
            job_args = self._get_job_arguments()

            # Get step name using standardized method with auto-detection
            step_name = self._get_step_name()

            # Get script path using modernized method with comprehensive fallbacks
            script_path = self.config.get_script_path()
            self.log_info("Using script path: %s", script_path)

            # Create step
            step = ProcessingStep(
                name=step_name,
                processor=processor,
                inputs=proc_inputs,
                outputs=proc_outputs,
                code=script_path,
                job_arguments=job_args,
                depends_on=dependencies,
                cache_config=self._get_cache_config(enable_caching),
            )

            # Attach specification to the step for future reference
            setattr(step, "_spec", self.spec)

            return step

        except Exception as e:
            self.log_error(f"Error creating MissingValueImputation step: {e}")
            import traceback
            self.log_error(traceback.format_exc())
            raise ValueError(f"Failed to create MissingValueImputation step: {str(e)}") from e
