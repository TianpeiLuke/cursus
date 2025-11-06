"""
Bedrock Prompt Template Generation Step Builder

This module implements the step builder for the Bedrock Prompt Template Generation step
following the specification-driven approach and standardization rules.
"""

from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_bedrock_prompt_template_generation_step import (
    BedrockPromptTemplateGenerationConfig,
)
from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Import the bedrock prompt template generation specification
try:
    from ..specs.bedrock_prompt_template_generation_spec import (
        BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC,
    )

    SPEC_AVAILABLE = True
except ImportError:
    BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC = None
    SPEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class BedrockPromptTemplateGenerationStepBuilder(StepBuilderBase):
    """
    Builder for a Bedrock Prompt Template Generation ProcessingStep.

    This implementation uses the specification-driven approach where dependencies, outputs,
    and script contract are defined in the bedrock prompt template generation specification.
    This step generates structured prompt templates for classification tasks using the
    5-component architecture pattern optimized for LLM performance.
    """

    def __init__(
        self,
        config: BedrockPromptTemplateGenerationConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initializes the builder with a specific configuration for the bedrock prompt template generation step.

        Args:
            config: A BedrockPromptTemplateGenerationConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        if not isinstance(config, BedrockPromptTemplateGenerationConfig):
            raise ValueError(
                "BedrockPromptTemplateGenerationStepBuilder requires a BedrockPromptTemplateGenerationConfig instance."
            )

        # Use the bedrock prompt template generation specification if available
        spec = BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC if SPEC_AVAILABLE else None

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: BedrockPromptTemplateGenerationConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        self.log_info("Validating BedrockPromptTemplateGenerationConfig...")

        # Validate processing script settings
        if (
            not hasattr(self.config, "processing_entry_point")
            or not self.config.processing_entry_point
        ):
            raise ValueError(
                "bedrock prompt template generation step requires a processing_entry_point"
            )

        # Validate template-specific configuration
        required_attrs = [
            "template_task_type",
            "template_style",
            "validation_level",
            "input_placeholders",
            "output_format_type",
            "required_output_fields",
            "template_version",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [
                None,
                "",
                [],
            ]:
                raise ValueError(
                    f"BedrockPromptTemplateGenerationConfig missing required attribute: {attr}"
                )

        # Validate input placeholders is not empty
        if not self.config.input_placeholders:
            raise ValueError("input_placeholders cannot be empty")

        # Validate required output fields is not empty
        if not self.config.required_output_fields:
            raise ValueError("required_output_fields cannot be empty")

        # Validate JSON configuration strings
        try:
            self.config.effective_system_prompt_config
            self.config.effective_output_format_config
            self.config.effective_instruction_config
        except Exception as e:
            raise ValueError(f"Invalid JSON configuration: {e}")

        self.log_info("BedrockPromptTemplateGenerationConfig validation succeeded.")

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

        This method combines:
        1. Base environment variables from the contract
        2. Configuration-specific environment variables from config.environment_variables

        Returns:
            A dictionary of environment variables for the processing job.
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Add configuration-specific environment variables
        config_env_vars = self.config.environment_variables
        env_vars.update(config_env_vars)

        self.log_info(
            "Bedrock prompt template generation environment variables: %s", env_vars
        )
        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.

        This method creates ProcessingInput objects for each dependency defined in the specification.
        For local file inputs, it uses user-specified paths from config when provided (non-default),
        otherwise allows dependency-provided inputs to override.

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

            # Handle prompt_configs input
            if logical_name == "prompt_configs":
                # Always use resolved_prompt_configs_path from config (has default "prompt_configs")
                try:
                    source_path = self.config.resolved_prompt_configs_path
                    container_path = self.contract.expected_input_paths[logical_name]

                    processing_inputs.append(
                        ProcessingInput(
                            input_name=logical_name,
                            source=source_path,
                            destination=container_path,
                        )
                    )

                    self.log_info(
                        "Added prompt_configs input: %s -> %s",
                        source_path,
                        container_path,
                    )
                    continue
                except ValueError as e:
                    raise ValueError(
                        f"Failed to resolve prompt_configs_path: {e}. "
                        f"Ensure effective_source_dir is configured properly."
                    )

            # Handle other dependency-provided inputs
            # Skip if optional and not provided
            if not dependency_spec.required and logical_name not in inputs:
                self.log_info(
                    "Optional input '%s' not provided, skipping", logical_name
                )
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
                "Added dependency input '%s': %s -> %s",
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
                # Generate destination from base path using Join instead of f-string
                from sagemaker.workflow.functions import Join

                base_output_path = self._get_base_output_path()
                destination = Join(
                    on="/",
                    values=[
                        base_output_path,
                        "bedrock_prompt_template_generation",
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

            self.log_info(
                "Added output '%s': %s -> %s",
                logical_name,
                container_path,
                destination,
            )

        return processing_outputs

    def _get_job_arguments(self) -> Optional[List[str]]:
        """
        Get job arguments for the bedrock prompt template generation script.

        The script accepts the following command-line arguments:
        - --include-examples: Include examples in template (boolean flag)
        - --generate-validation-schema: Generate validation schema (boolean flag)
        - --template-version: Template version identifier (string)

        Returns:
            List of command-line arguments for the script, or None if no arguments needed
        """
        job_args = []

        # Add boolean flags only if they're True (since False is the default)
        if self.config.include_examples:
            job_args.append("--include-examples")

        if self.config.generate_validation_schema:
            job_args.append("--generate-validation-schema")

        # Add template version (always include since it has a meaningful default)
        job_args.extend(["--template-version", self.config.template_version])

        self.log_info(
            "Job arguments for bedrock prompt template generation script: %s", job_args
        )
        return job_args if job_args else None

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
        self.log_info("Creating Bedrock Prompt Template Generation ProcessingStep...")

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
