from typing import Dict, Optional, Any, List
from pathlib import Path
import logging
import importlib

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_currency_conversion_step import CurrencyConversionConfig
from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Import specifications based on job type
try:
    from ..specs.currency_conversion_training_spec import (
        CURRENCY_CONVERSION_TRAINING_SPEC,
    )
    from ..specs.currency_conversion_calibration_spec import (
        CURRENCY_CONVERSION_CALIBRATION_SPEC,
    )
    from ..specs.currency_conversion_validation_spec import (
        CURRENCY_CONVERSION_VALIDATION_SPEC,
    )
    from ..specs.currency_conversion_testing_spec import (
        CURRENCY_CONVERSION_TESTING_SPEC,
    )

    SPECS_AVAILABLE = True
except ImportError:
    CURRENCY_CONVERSION_TRAINING_SPEC = CURRENCY_CONVERSION_CALIBRATION_SPEC = (
        CURRENCY_CONVERSION_VALIDATION_SPEC
    ) = CURRENCY_CONVERSION_TESTING_SPEC = None
    SPECS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CurrencyConversionStepBuilder(StepBuilderBase):
    """
    Builder for a Currency Conversion ProcessingStep.

    This implementation uses the fully specification-driven approach where inputs, outputs,
    and behavior are defined by step specifications and script contracts.
    """

    def __init__(
        self,
        config: CurrencyConversionConfig,
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
        if job_type == "training" and CURRENCY_CONVERSION_TRAINING_SPEC is not None:
            spec = CURRENCY_CONVERSION_TRAINING_SPEC
        elif (
            job_type == "calibration"
            and CURRENCY_CONVERSION_CALIBRATION_SPEC is not None
        ):
            spec = CURRENCY_CONVERSION_CALIBRATION_SPEC
        elif (
            job_type == "validation" and CURRENCY_CONVERSION_VALIDATION_SPEC is not None
        ):
            spec = CURRENCY_CONVERSION_VALIDATION_SPEC
        elif job_type == "testing" and CURRENCY_CONVERSION_TESTING_SPEC is not None:
            spec = CURRENCY_CONVERSION_TESTING_SPEC
        else:
            # Try dynamic import
            try:
                module_path = (
                    f"..pipeline_step_specs.currency_conversion_{job_type}_spec"
                )
                module = importlib.import_module(module_path, package=__package__)
                spec_var_name = f"CURRENCY_CONVERSION_{job_type.upper()}_SPEC"
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
        self.config: CurrencyConversionConfig = config

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

        # Currency-specific validation
        if self.config.enable_currency_conversion:
            if not self.config.marketplace_id_col:
                raise ValueError(
                    "marketplace_id_col must be provided when conversion is enabled"
                )
            if not self.config.currency_conversion_var_list:
                raise ValueError(
                    "currency_conversion_var_list cannot be empty when conversion is enabled"
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

        Returns:
            Dict[str, str]: Environment variables for the processing job
        """
        import json

        env_vars = {
            "CURRENCY_CONVERSION_VARS": json.dumps(
                self.config.currency_conversion_var_list
            ),
            "CURRENCY_CONVERSION_DICT": json.dumps(
                self.config.currency_conversion_dict
            ),
            "MARKETPLACE_INFO": json.dumps(self.config.marketplace_info),
            "LABEL_FIELD": self.config.label_field,
            "TRAIN_RATIO": str(self.config.train_ratio),
            "TEST_VAL_RATIO": str(self.config.test_val_ratio),
        }

        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.

        This method creates ProcessingInput objects for each dependency defined in the specification.

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

            processing_inputs.append(
                ProcessingInput(
                    input_name=logical_name,
                    source=inputs[logical_name],
                    destination=container_path,
                )
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
                destination = Join(on="/", values=[base_output_path, "currency_conversion", self.config.job_type, logical_name])
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
        Get command-line arguments for the script.

        Returns:
            List[str]: Command-line arguments
        """
        args = [
            "--job-type",
            self.config.job_type,
            "--mode",
            self.config.mode,
            "--marketplace-id-col",
            self.config.marketplace_id_col,
            "--default-currency",
            self.config.default_currency,
            "--enable-conversion",
            str(self.config.enable_currency_conversion).lower(),
        ]

        # Add optional arguments
        if hasattr(self.config, "currency_col") and self.config.currency_col:
            args.extend(["--currency-col", self.config.currency_col])

        if (
            hasattr(self.config, "skip_invalid_currencies")
            and self.config.skip_invalid_currencies
        ):
            args.append("--skip-invalid-currencies")

        return args

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

        # Add direct keyword arguments (e.g., data_input from template)
        for key in ["data_input"]:
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
