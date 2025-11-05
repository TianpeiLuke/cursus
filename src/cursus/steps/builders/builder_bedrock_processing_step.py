"""
Bedrock Processing Step Builder

This module implements the step builder for the Bedrock Processing step
following the specification-driven approach and standardization rules.
"""

from typing import Dict, Optional, Any, List
from pathlib import Path
import logging
import importlib

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_bedrock_processing_step import BedrockProcessingConfig
from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Import the bedrock processing specification
try:
    from ..specs.bedrock_processing_spec import BEDROCK_PROCESSING_SPEC

    SPEC_AVAILABLE = True
except ImportError:
    BEDROCK_PROCESSING_SPEC = None
    SPEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class BedrockProcessingStepBuilder(StepBuilderBase):
    """
    Builder for a Bedrock Processing ProcessingStep.

    This implementation uses the specification-driven approach where dependencies, outputs,
    and script contract are defined in the bedrock processing specification.
    This step processes input data through AWS Bedrock models using generated prompt
    templates and validation schemas from the Bedrock Prompt Template Generation step.
    Supports template-driven response processing with dynamic Pydantic model creation
    and both sequential and concurrent processing modes.

    Key Features:
    - Template-driven processing using outputs from Bedrock Prompt Template Generation
    - Support for latest Claude models (Sonnet 4.5, Haiku 4.5, Sonnet 4.0, Opus 4.1)
    - Concurrent and sequential processing modes
    - Dynamic Pydantic model creation from validation schemas
    - Production-ready inference profile management
    - Comprehensive error handling and retry logic

    Integration:
    - Depends on: BedrockPromptTemplateGeneration (for templates and schemas)
    - Depends on: TabularPreprocessing or similar (for input data)
    - Produces: Processed data with LLM-generated classifications/analysis

    Example:
        ```python
        config = BedrockProcessingConfig(
            bedrock_inference_profile_arn="arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc123",
            bedrock_primary_model_id="anthropic.claude-sonnet-4-5-20250929-v1:0",  # Claude 4.5
            bedrock_concurrency_mode="concurrent",
            bedrock_max_concurrent_workers=8
        )
        builder = BedrockProcessingStepBuilder(config)
        step = builder.create_step(inputs={"input_data": data_source, "prompt_templates": template_source})
        ```

    See Also:
        BedrockPromptTemplateGenerationStepBuilder, BedrockProcessingConfig, TabularPreprocessingStepBuilder
    """

    def __init__(
        self,
        config: BedrockProcessingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initialize the Bedrock Processing step builder.

        Args:
            config: Configuration for the step
            sagemaker_session: SageMaker session
            role: IAM role
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection

        Raises:
            ValueError: If config is not a BedrockProcessingConfig instance or job_type is missing
        """
        if not isinstance(config, BedrockProcessingConfig):
            raise ValueError(
                "BedrockProcessingStepBuilder requires a BedrockProcessingConfig instance."
            )

        # job_type now has a default value, so no need to validate presence

        # Use the generic bedrock processing specification for all job types
        # since job type variants don't change inputs/outputs, only processing behavior
        spec = BEDROCK_PROCESSING_SPEC if SPEC_AVAILABLE else None

        if not spec:
            raise ValueError("Bedrock processing specification not available")

        self.log_info(
            "Using bedrock processing specification for job type: %s", config.job_type
        )

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: BedrockProcessingConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Validates:
        - Processing script settings
        - Bedrock model configuration
        - Inference profile settings
        - Concurrency configuration
        - Production readiness

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        self.log_info("Validating BedrockProcessingConfig...")

        # Validate processing script settings
        if (
            not hasattr(self.config, "processing_entry_point")
            or not self.config.processing_entry_point
        ):
            raise ValueError(
                "bedrock processing step requires a processing_entry_point"
            )

        # Validate Bedrock-specific configuration
        required_attrs = [
            "job_type",
            "bedrock_inference_profile_arn",
            "bedrock_primary_model_id",
            "bedrock_max_tokens",
            "bedrock_temperature",
            "bedrock_top_p",
            "bedrock_batch_size",
            "bedrock_max_retries",
            "bedrock_output_column_prefix",
            "bedrock_concurrency_mode",
            "bedrock_max_concurrent_workers",
            "bedrock_rate_limit_per_second",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise ValueError(
                    f"BedrockProcessingConfig missing required attribute: {attr}"
                )

        # Validate job type
        if self.config.job_type not in [
            "training",
            "validation",
            "testing",
            "calibration",
        ]:
            raise ValueError(f"Invalid job_type: {self.config.job_type}")

        # Validate required fields are not empty
        if not self.config.bedrock_inference_profile_arn:
            raise ValueError("bedrock_inference_profile_arn cannot be empty")

        if not self.config.bedrock_primary_model_id:
            raise ValueError("bedrock_primary_model_id cannot be empty")

        # Validate concurrency mode
        valid_modes = ["sequential", "concurrent"]
        if self.config.bedrock_concurrency_mode not in valid_modes:
            raise ValueError(f"bedrock_concurrency_mode must be one of {valid_modes}")

        # Validate numeric ranges
        if (
            self.config.bedrock_max_tokens <= 0
            or self.config.bedrock_max_tokens > 64000
        ):
            raise ValueError("bedrock_max_tokens must be between 1 and 64000")

        if not (0.0 <= self.config.bedrock_temperature <= 2.0):
            raise ValueError("bedrock_temperature must be between 0.0 and 2.0")

        if not (0.0 <= self.config.bedrock_top_p <= 1.0):
            raise ValueError("bedrock_top_p must be between 0.0 and 1.0")

        if self.config.bedrock_batch_size <= 0 or self.config.bedrock_batch_size > 100:
            raise ValueError("bedrock_batch_size must be between 1 and 100")

        if (
            self.config.bedrock_max_concurrent_workers <= 0
            or self.config.bedrock_max_concurrent_workers > 20
        ):
            raise ValueError("bedrock_max_concurrent_workers must be between 1 and 20")

        if (
            self.config.bedrock_rate_limit_per_second <= 0
            or self.config.bedrock_rate_limit_per_second > 100
        ):
            raise ValueError("bedrock_rate_limit_per_second must be between 1 and 100")

        # Validate inference profile required models
        if not isinstance(self.config.bedrock_inference_profile_required_models, list):
            raise ValueError("bedrock_inference_profile_required_models must be a list")

        # Validate model ID format
        valid_prefixes = [
            "anthropic.",
            "amazon.",
            "ai21.",
            "cohere.",
            "meta.",
            "mistral.",
            "stability.",
            "global.",
        ]
        if not any(
            self.config.bedrock_primary_model_id.startswith(prefix)
            for prefix in valid_prefixes
        ):
            self.log_warning(
                "Primary model ID '%s' doesn't match common Bedrock patterns",
                self.config.bedrock_primary_model_id,
            )

        # Check production readiness and log warnings
        if not self.config.is_production_ready():
            self.log_warning(
                "Configuration may not be production-ready. Consider adding a fallback model and reviewing concurrency settings."
            )

        # Validate derived properties can be accessed
        try:
            _ = self.config.effective_inference_profile_required_models
            _ = self.config.bedrock_environment_variables
            _ = self.config.processing_metadata
            _ = self.config.concurrency_configuration
        except Exception as e:
            raise ValueError(f"Failed to access derived configuration properties: {e}")

        self.log_info("BedrockProcessingConfig validation succeeded.")

    def _create_processor(self) -> SKLearnProcessor:
        """
        Creates and configures the SKLearnProcessor for the SageMaker Processing Job.
        This defines the execution environment for the script, including the instance
        type, framework version, and environment variables.

        Returns:
            An instance of sagemaker.sklearn.SKLearnProcessor configured for Bedrock processing.
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
        2. Configuration-specific environment variables from config.bedrock_environment_variables

        Returns:
            A dictionary of environment variables for the processing job.
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Add Bedrock-specific environment variables
        bedrock_env_vars = self.config.bedrock_environment_variables
        env_vars.update(bedrock_env_vars)

        self.log_info("Bedrock processing environment variables: %s", env_vars)
        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.

        This method creates ProcessingInput objects for each dependency defined in the specification.
        The Bedrock Processing step expects:
        1. input_data - The data to be processed (from TabularPreprocessing or similar)
        2. prompt_templates - Generated templates from BedrockPromptTemplateGeneration

        Args:
            inputs: Input data sources keyed by logical name

        Returns:
            List of ProcessingInput objects

        Raises:
            ValueError: If no specification or contract is available, or required inputs are missing
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
                "Added input '%s': %s -> %s",
                logical_name,
                inputs[logical_name],
                container_path,
            )

        return processing_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Get outputs for the step using specification and contract.

        This method creates ProcessingOutput objects for each output defined in the specification.
        The Bedrock Processing step produces processed data with LLM-generated classifications.

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
                    values=[base_output_path, "bedrock_processing", logical_name],
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
        Get job arguments for the bedrock processing script.

        The Bedrock processing script accepts the following command-line arguments:
        - --job_type: Job type for processing (training, validation, testing, calibration)
        - --batch-size: Batch size for processing (overrides environment variable)
        - --max-retries: Maximum retries for Bedrock calls (overrides environment variable)

        Returns:
            List of command-line arguments for the script, or None if no arguments needed
        """
        job_args = []

        # Add job_type argument (from config) - similar to tabular preprocessing
        job_args.extend(["--job_type", self.config.job_type])
        self.log_info("Setting job_type argument to: %s", self.config.job_type)

        # Add batch size argument (from config)
        job_args.extend(["--batch-size", str(self.config.bedrock_batch_size)])

        # Add max retries argument (from config)
        job_args.extend(["--max-retries", str(self.config.bedrock_max_retries)])

        # Log performance estimate for debugging
        performance = self.config.get_performance_estimate()
        self.log_info("Expected processing performance: %s", performance)

        self.log_info("Job arguments for bedrock processing script: %s", job_args)
        return job_args if job_args else None

    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates the final, fully configured SageMaker ProcessingStep for the pipeline
        using the specification-driven approach.

        This method creates a ProcessingStep that:
        1. Processes input data using AWS Bedrock models
        2. Uses prompt templates from BedrockPromptTemplateGeneration
        3. Supports both sequential and concurrent processing
        4. Handles inference profile management automatically
        5. Provides comprehensive error handling and retry logic

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Input data sources keyed by logical name
                - outputs: Output destinations keyed by logical name
                - dependencies: Optional list of steps that this step depends on
                - enable_caching: A boolean indicating whether to cache the results of this step

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.

        Raises:
            ValueError: If required inputs are missing or configuration is invalid
        """
        self.log_info("Creating Bedrock Processing ProcessingStep...")

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

        # Log configuration summary
        performance = self.config.get_performance_estimate()
        self.log_info("Created ProcessingStep with name: %s", step.name)
        self.log_info("Primary model: %s", self.config.bedrock_primary_model_id)
        self.log_info(
            "Fallback model: %s", self.config.bedrock_fallback_model_id or "None"
        )
        self.log_info("Processing mode: %s", self.config.bedrock_concurrency_mode)
        self.log_info("Expected performance: %s", performance)
        self.log_info("Production ready: %s", self.config.is_production_ready())

        return step
