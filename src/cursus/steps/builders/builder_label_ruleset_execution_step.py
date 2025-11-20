"""
Label Ruleset Execution Step Builder

Builder for creating Label Ruleset Execution processing steps that apply validated
rulesets to processed data to generate classification labels.
"""

from typing import Dict, Optional, Any, List
import logging

from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_label_ruleset_execution_step import LabelRulesetExecutionConfig
from ...core.base.builder_base import StepBuilderBase

# Import specification
try:
    from ..specs.label_ruleset_execution_spec import LABEL_RULESET_EXECUTION_SPEC

    SPEC_AVAILABLE = True
except ImportError:
    LABEL_RULESET_EXECUTION_SPEC = None
    SPEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class LabelRulesetExecutionStepBuilder(StepBuilderBase):
    """
    Builder for Label Ruleset Execution ProcessingStep.

    This implementation uses the specification-driven approach where inputs, outputs,
    and behavior are defined by step specifications and script contracts.
    """

    def __init__(
        self,
        config: LabelRulesetExecutionConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initialize with specification.

        Args:
            config: Configuration for the step
            sagemaker_session: SageMaker session
            role: IAM role
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection

        Raises:
            ValueError: If specification is not available
        """
        if not SPEC_AVAILABLE or LABEL_RULESET_EXECUTION_SPEC is None:
            raise ValueError(
                "Label Ruleset Execution specification not available. "
                "Ensure label_ruleset_execution_spec.py is properly installed."
            )

        super().__init__(
            config=config,
            spec=LABEL_RULESET_EXECUTION_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: LabelRulesetExecutionConfig = config
        self.log_info("Initialized Label Ruleset Execution step builder")

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

        self.log_info(
            "Configuration validation passed for job_type: %s", self.config.job_type
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

        self.log_info(
            "Creating processor with instance type: %s (large=%s)",
            instance_type,
            self.config.use_large_processing_instance,
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

        This method combines:
        1. Base environment variables from the contract
        2. Configuration-specific environment variables from config.execution_environment_variables

        Returns:
            Dict[str, str]: Environment variables for the processing job
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Add configuration-specific environment variables
        config_env_vars = self.config.execution_environment_variables
        env_vars.update(config_env_vars)

        self.log_info(
            "Label ruleset execution environment variables: FAIL_ON_MISSING_FIELDS=%s, RULE_MATCH_TRACKING=%s",
            config_env_vars.get("FAIL_ON_MISSING_FIELDS"),
            config_env_vars.get("ENABLE_RULE_MATCH_TRACKING"),
        )

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

            self.log_info(
                "Adding input '%s': %s -> %s",
                logical_name,
                inputs[logical_name],
                container_path,
            )

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
                # Generate destination using base output path and Join for parameter compatibility
                from sagemaker.workflow.functions import Join

                base_output_path = self._get_base_output_path()
                destination = Join(
                    on="/",
                    values=[
                        base_output_path,
                        "label_ruleset_execution",
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

    def _get_job_arguments(self) -> List[str]:
        """
        Constructs the list of command-line arguments to be passed to the processing script.

        Returns:
            A list of strings representing the command-line arguments.
        """
        # Get job_type from configuration (required by script)
        job_type = self.config.job_type
        self.log_info("Setting job-type argument to: %s", job_type)

        return ["--job-type", job_type]

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
                self.log_info(
                    "Extracted %s inputs from dependencies", len(extracted_inputs)
                )
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)

        # Add explicitly provided inputs (overriding any extracted ones)
        inputs.update(inputs_raw)

        # Log input summary
        self.log_info(
            "Creating step with inputs: %s",
            ", ".join(inputs.keys()) if inputs else "none",
        )

        # Create processor and get inputs/outputs
        processor = self._create_processor()
        proc_inputs = self._get_inputs(inputs)
        proc_outputs = self._get_outputs(outputs)
        job_args = self._get_job_arguments()

        # Get step name using standardized method
        step_name = self._get_step_name()

        # Get script path using config method with comprehensive fallbacks
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

        self.log_info(
            "Successfully created Label Ruleset Execution step: %s", step_name
        )

        return step
