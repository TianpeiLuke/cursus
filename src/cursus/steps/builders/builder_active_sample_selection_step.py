from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_active_sample_selection_step import ActiveSampleSelectionConfig
from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Import the active sample selection specification
try:
    from ..specs.active_sample_selection_spec import ACTIVE_SAMPLE_SELECTION_SPEC

    SPEC_AVAILABLE = True
except ImportError:
    ACTIVE_SAMPLE_SELECTION_SPEC = None
    SPEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class ActiveSampleSelectionStepBuilder(StepBuilderBase):
    """
    Builder for Active Sample Selection ProcessingStep.

    This step intelligently selects samples from model predictions for:
    - Semi-Supervised Learning: High-confidence samples for pseudo-labeling
    - Active Learning: Uncertain/diverse samples for human labeling

    Supports multiple selection strategies:
    - confidence_threshold: Samples with confidence >= threshold
    - top_k_per_class: Top-k confident samples per class
    - uncertainty: Uncertain samples (margin, entropy, least_confidence)
    - diversity: Diverse samples via k-center algorithm
    - badge: Combined uncertainty + diversity via gradient embeddings

    This implementation uses the specification-driven approach where dependencies,
    outputs, and script contract are defined in the active sample selection specification.
    """

    def __init__(
        self,
        config: ActiveSampleSelectionConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initializes the builder with a specific configuration for the active sample selection step.

        Args:
            config: An ActiveSampleSelectionConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        if not isinstance(config, ActiveSampleSelectionConfig):
            raise ValueError(
                "ActiveSampleSelectionStepBuilder requires an ActiveSampleSelectionConfig instance."
            )

        # Use the active sample selection specification if available
        spec = ACTIVE_SAMPLE_SELECTION_SPEC if SPEC_AVAILABLE else None

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: ActiveSampleSelectionConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        self.log_info("Validating ActiveSampleSelectionConfig...")

        # Validate required attributes
        required_attrs = [
            "processing_entry_point",
            "processing_source_dir",
            "processing_instance_count",
            "processing_volume_size",
            "pipeline_name",
            "selection_strategy",
            "use_case",
            "id_field",
            "label_field",
            "processing_framework_version",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [
                None,
                "",
            ]:
                # label_field can be empty string, which is valid
                if attr == "label_field" and getattr(self.config, attr) == "":
                    continue
                raise ValueError(
                    f"ActiveSampleSelectionConfig missing required attribute: {attr}"
                )

        # Validate instance type settings
        if not hasattr(self.config, "processing_instance_type_large"):
            raise ValueError(
                "Missing required attribute: processing_instance_type_large"
            )
        if not hasattr(self.config, "processing_instance_type_small"):
            raise ValueError(
                "Missing required attribute: processing_instance_type_small"
            )
        if not hasattr(self.config, "use_large_processing_instance"):
            raise ValueError(
                "Missing required attribute: use_large_processing_instance"
            )

        # Strategy-use case validation is handled by config Pydantic validators

        self.log_info("ActiveSampleSelectionConfig validation succeeded.")

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

        return SKLearnProcessor(
            framework_version=self.config.processing_framework_version,
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._generate_job_name(),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the processing job.
        These variables are used to control the behavior of the active sample selection script
        without needing to pass them as command-line arguments.

        Returns:
            A dictionary of environment variables.
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Add selection-specific environment variables from config
        selection_env_vars = self.config.get_environment_variables()
        env_vars.update(selection_env_vars)

        self.log_info("Active sampling environment variables: %s", env_vars)
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
                destination = Join(
                    on="/", values=[base_output_path, "active_sampling", logical_name]
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

    def _get_job_arguments(self) -> List[str]:
        """
        Constructs the list of command-line arguments to be passed to the processing script.

        This implementation uses job_type from the configuration, which is required by the script.

        Returns:
            A list of strings representing the command-line arguments.
        """
        # Get job_type from configuration or use default
        job_type = getattr(self.config, "job_type", "ssl_selection")
        self.log_info("Setting job_type argument to: %s", job_type)

        return ["--job_type", job_type]

    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates the final, fully configured SageMaker ProcessingStep for the pipeline
        using the specification-driven approach.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Input data sources keyed by logical name
                - outputs: Output destinations keyed by logical name
                - dependencies: Optional list of steps that this step depends on
                - enable_caching: A boolean indicating whether to cache the results of this step

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        self.log_info("Creating ActiveSampleSelection ProcessingStep...")

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
        processor = self._create_processor()
        proc_inputs = self._get_inputs(inputs)
        proc_outputs = self._get_outputs(outputs)
        job_args = self._get_job_arguments()

        # Get step name using standardized method with auto-detection
        step_name = self._get_step_name()

        # Get script path using modernized method with comprehensive fallbacks
        script_path = self.config.get_script_path()
        self.log_info("Using script path: %s", script_path)

        # Create step arguments
        step_args = processor.run(
            code=script_path,
            inputs=proc_inputs,
            outputs=proc_outputs,
            arguments=job_args,
        )

        # Create and return the step - use only step_args, not processor
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
