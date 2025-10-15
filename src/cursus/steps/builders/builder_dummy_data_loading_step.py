from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_dummy_data_loading_step import DummyDataLoadingConfig
from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Import the dummy data loading specification
try:
    from ..specs.dummy_data_loading_spec import DUMMY_DATA_LOADING_SPEC

    SPEC_AVAILABLE = True
except ImportError:
    DUMMY_DATA_LOADING_SPEC = None
    SPEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class DummyDataLoadingStepBuilder(StepBuilderBase):
    """
    Builder for a Dummy Data Loading ProcessingStep.

    This implementation uses the specification-driven approach where dependencies, outputs,
    and script contract are defined in the dummy data loading specification.
    This step processes user-provided data instead of calling internal Cradle services.
    """

    def __init__(
        self,
        config: DummyDataLoadingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initializes the builder with a specific configuration for the dummy data loading step.

        Args:
            config: A DummyDataLoadingConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        if not isinstance(config, DummyDataLoadingConfig):
            raise ValueError("DummyDataLoadingStepBuilder requires a DummyDataLoadingConfig instance.")

        # Use the dummy data loading specification if available
        spec = DUMMY_DATA_LOADING_SPEC if SPEC_AVAILABLE else None

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: DummyDataLoadingConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        self.log_info("Validating DummyDataLoadingConfig...")

        # Validate data source - this is the only essential user input for this step
        # All other processing settings are handled by ProcessingStepConfigBase
        if not hasattr(self.config, "data_source") or not self.config.data_source:
            raise ValueError("dummy data loading step requires a data_source")

        self.log_info("DummyDataLoadingConfig validation succeeded.")

    def _create_processor(self) -> SKLearnProcessor:
        """
        Creates and configures the SKLearnProcessor for the SageMaker Processing Job.
        This defines the execution environment for the script, including the instance
        type, framework version, and environment variables.

        Returns:
            An instance of sagemaker.sklearn.SKLearnProcessor.
        """
        # Get the appropriate instance type based on use_large_processing_instance
        instance_type = (
            self.config.processing_instance_type_large
            if self.config.use_large_processing_instance
            else self.config.processing_instance_type_small
        )

        # Get framework version
        framework_version = getattr(
            self.config, "processing_framework_version", "1.2-1"
        )

        return SKLearnProcessor(
            framework_version=framework_version,
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
        Constructs a dictionary of environment variables to be passed to the processing job.
        
        The dummy data loading script does not require any environment variables.
        It uses only standard SageMaker paths and processes data from the input channel.

        Returns:
            A dictionary of environment variables (only base environment variables from contract).
        """
        # Get base environment variables from contract (should be empty for this step)
        env_vars = super()._get_environment_variables()

        self.log_info("Dummy data loading environment variables: %s", env_vars)
        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.

        The dummy data loading step has only one input (INPUT_DATA) which always comes
        from the user-provided data source in the configuration.

        Args:
            inputs: Input data sources keyed by logical name (ignored - data source comes from config)

        Returns:
            List of ProcessingInput objects

        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for input mapping")

        # Dummy data loading has only one input: INPUT_DATA from user config
        data_source_path = self.config.get_data_source_uri()
        container_path = self.contract.expected_input_paths["INPUT_DATA"]
        
        self.log_info(
            "Using user-provided data source from configuration: %s -> %s",
            data_source_path,
            container_path,
        )

        return [
            ProcessingInput(
                input_name="INPUT_DATA",
                source=data_source_path,
                destination=container_path,
            )
        ]

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
                # Generate destination from base path using Join instead of f-string
                from sagemaker.workflow.functions import Join
                base_output_path = self._get_base_output_path()
                destination = Join(on="/", values=[base_output_path, "dummy_data_loading", logical_name])
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

    def _get_job_arguments(self) -> Optional[List[str]]:
        """
        Returns None as job arguments since the dummy data loading script uses
        standard paths defined directly in the script.

        Returns:
            None since no arguments are needed
        """
        self.log_info("No command-line arguments needed for dummy data loading script")
        return None

    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates the final, fully configured SageMaker ProcessingStep for the pipeline
        using the specification-driven approach.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Input data sources keyed by logical name (will be overridden by config data_source)
                - outputs: Output destinations keyed by logical name
                - dependencies: Optional list of steps that this step depends on
                - enable_caching: A boolean indicating whether to cache the results of this step

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        self.log_info("Creating Dummy Data Loading ProcessingStep...")

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

        # Note: The actual data source will come from config.data_source, not from inputs
        # This is handled in _get_inputs() method

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
        if hasattr(self, "spec") and self.spec:
            setattr(step, "_spec", self.spec)

        self.log_info("Created ProcessingStep with name: %s", step.name)
        return step
