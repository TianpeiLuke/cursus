from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_package_step import PackageConfig
from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Import the packaging specification
try:
    from ..specs.package_spec import PACKAGE_SPEC

    SPEC_AVAILABLE = True
except ImportError:
    PACKAGE_SPEC = None
    SPEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class PackageStepBuilder(StepBuilderBase):
    """
    Builder for a Model Packaging ProcessingStep.

    This implementation uses the specification-driven approach where dependencies, outputs,
    and script contract are defined in the packaging specification.
    """

    def __init__(
        self,
        config: PackageConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initializes the builder with a specific configuration for the packaging step.

        Args:
            config: A PackageConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        if not isinstance(config, PackageConfig):
            raise ValueError("PackageStepBuilder requires a PackageConfig instance.")

        # Use the packaging specification if available
        spec = PACKAGE_SPEC if SPEC_AVAILABLE else None

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: PackageConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        self.log_info("Validating PackageConfig...")

        # Validate processing script settings
        if (
            not hasattr(self.config, "processing_entry_point")
            or not self.config.processing_entry_point
        ):
            raise ValueError("packaging step requires a processing_entry_point")

        # Validate other required attributes
        required_attrs = [
            "processing_instance_count",
            "processing_volume_size",
            "processing_instance_type_large",
            "processing_instance_type_small",
            "pipeline_name",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [
                None,
                "",
            ]:
                raise ValueError(f"PackageConfig missing required attribute: {attr}")

        self.log_info("PackageConfig validation succeeded.")

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
            self.config, "processing_framework_version", "1.0-1"
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

        Returns:
            A dictionary of environment variables.
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Add packaging-specific environment variables
        if hasattr(self.config, "pipeline_name"):
            env_vars["PIPELINE_NAME"] = self.config.pipeline_name

        if hasattr(self.config, "region"):
            env_vars["REGION"] = self.config.region

        # Add optional configurations
        for key, env_key in [
            ("model_type", "MODEL_TYPE"),
            ("bucket", "BUCKET_NAME"),
            ("pipeline_version", "PIPELINE_VERSION"),
            ("model_objective", "MODEL_OBJECTIVE"),
        ]:
            if hasattr(self.config, key) and getattr(self.config, key) is not None:
                env_vars[env_key] = str(getattr(self.config, key))

        self.log_info("Packaging environment variables: %s", env_vars)
        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.

        This method creates ProcessingInput objects for each dependency defined in the specification.
        Special handling is implemented for inference_scripts_input to always use a local path.

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
        matched_inputs = set()  # Track which inputs we've handled

        # SPECIAL CASE: Always handle inference_scripts_input from local path
        # This will take precedence over any dependency-resolved value
        inference_scripts_key = "inference_scripts_input"
        # Use source directory with hybrid resolution fallback
        inference_scripts_path = (
            self.config.resolved_source_dir or  # Hybrid resolution
            self.config.source_dir or           # Fallback to existing behavior
            "inference"                         # Final fallback
        )
        self.log_info("Using source dir: %s", inference_scripts_path)

        self.log_info(
            "[PACKAGING INPUT OVERRIDE] Using local inference scripts path from configuration: %s",
            inference_scripts_path,
        )
        self.log_info(
            "[PACKAGING INPUT OVERRIDE] This local path will be used regardless of any dependency-resolved values"
        )

        # Get container path from contract
        container_path = None
        if inference_scripts_key in self.contract.expected_input_paths:
            container_path = self.contract.expected_input_paths[inference_scripts_key]
        else:
            # Fallback container path if not in contract
            container_path = "/opt/ml/processing/input/script"

        # Use the input path directly - property references are handled by PipelineAssembler
        processing_inputs.append(
            ProcessingInput(
                input_name=inference_scripts_key,
                source=inference_scripts_path,
                destination=container_path,
            )
        )
        matched_inputs.add(
            inference_scripts_key
        )  # Mark as handled to skip in main loop
        self.log_info(
            "Added inference scripts input with local path: %s -> %s",
            inference_scripts_path,
            container_path,
        )

        # Create a copy of the inputs dictionary to ensure we don't modify the original
        # This ensures we don't affect subsequent steps if they need the original inputs
        working_inputs = inputs.copy()

        # Remove our special case from the inputs dictionary to ensure it doesn't get processed again
        # This is a stronger protection than just tracking in matched_inputs
        if inference_scripts_key in working_inputs:
            external_path = working_inputs[inference_scripts_key]
            self.log_info(
                "[PACKAGING INPUT OVERRIDE] Ignoring dependency-provided value: %s",
                external_path,
            )
            self.log_info(
                "[PACKAGING INPUT OVERRIDE] Using internal path %s instead",
                inference_scripts_path,
            )
            del working_inputs[inference_scripts_key]

        # Process each dependency in the specification
        for _, dependency_spec in self.spec.dependencies.items():
            logical_name = dependency_spec.logical_name

            # Skip inputs we've already handled
            if logical_name in matched_inputs:
                continue

            # Skip if optional and not provided
            if not dependency_spec.required and logical_name not in working_inputs:
                continue

            # Make sure required inputs are present
            if dependency_spec.required and logical_name not in working_inputs:
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
                    source=working_inputs[logical_name],
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
                destination = Join(on="/", values=[base_output_path, "packaging", logical_name])
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
        Returns None as job arguments since the packaging script now uses
        standard paths defined directly in the script.

        Returns:
            None since no arguments are needed
        """
        self.log_info("No command-line arguments needed for packaging script")
        return None

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
        self.log_info("Creating Packaging ProcessingStep...")

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
