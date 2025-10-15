from typing import Dict, Optional, Any, List
from pathlib import Path
import logging
import tempfile
import json
import shutil
import boto3
from botocore.exceptions import ClientError

from sagemaker.workflow.steps import TrainingStep, Step
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker import image_uris
from sagemaker.s3 import S3Uploader
from sagemaker.workflow.functions import Join

from ..configs.config_lightgbm_training_step import LightGBMTrainingConfig
from ...core.base.builder_base import StepBuilderBase
from .s3_utils import S3PathHandler
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Import LightGBM training specification
try:
    from ..specs.lightgbm_training_spec import LIGHTGBM_TRAINING_SPEC

    SPEC_AVAILABLE = True
except ImportError:
    LIGHTGBM_TRAINING_SPEC = None
    SPEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class LightGBMTrainingStepBuilder(StepBuilderBase):
    """
    Builder for a LightGBM Training Step using SageMaker built-in algorithm.
    This class creates a SageMaker TrainingStep using the generic Estimator
    with LightGBM built-in algorithm image URI.
    """

    def __init__(
        self,
        config: LightGBMTrainingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initializes the builder with a specific configuration for the training step.

        Args:
            config: A LightGBMTrainingConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Training Job.
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection

        Raises:
            ValueError: If specification is not available or config is invalid
        """
        if not isinstance(config, LightGBMTrainingConfig):
            raise ValueError(
                "LightGBMTrainingStepBuilder requires a LightGBMTrainingConfig instance."
            )

        # Load LightGBM training specification
        if not SPEC_AVAILABLE or LIGHTGBM_TRAINING_SPEC is None:
            raise ValueError("LightGBM training specification not available")

        self.log_info("Using LightGBM training specification")

        super().__init__(
            config=config,
            spec=LIGHTGBM_TRAINING_SPEC,  # Add specification
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: LightGBMTrainingConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        self.log_info("Validating LightGBMTrainingConfig...")

        # Validate required attributes
        required_attrs = [
            "training_instance_type",
            "training_instance_count",
            "training_volume_size",
            "training_entry_point",
            "source_dir",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [
                None,
                "",
            ]:
                raise ValueError(
                    f"LightGBMTrainingConfig missing required attribute: {attr}"
                )

        # Input/output validation is now handled by specifications
        self.log_info("Configuration validation relies on step specifications")

        self.log_info("LightGBMTrainingConfig validation succeeded.")

    def _get_lightgbm_image_uri(self) -> str:
        """Get LightGBM JumpStart model image URI for the current region."""
        # LightGBM is available as a JumpStart model, not a traditional built-in algorithm
        return image_uris.retrieve(
            region=self.aws_region,
            framework=None,
            model_id="lightgbm-classification-model",
            model_version="*",  # Latest version
            image_scope="training",
            instance_type=self.config.training_instance_type,
        )

    def _create_estimator(self, output_path=None) -> Estimator:
        """
        Creates and configures the LightGBM estimator using built-in algorithm.
        This defines the execution environment for the training job, including the instance
        type, built-in algorithm image, and environment variables.

        Args:
            output_path: Optional override for model output path. If provided, this will be used
                         instead of generating a default path.

        Returns:
            An instance of sagemaker.estimator.Estimator configured for LightGBM.
        """
        # Get LightGBM built-in algorithm image URI
        image_uri = self._get_lightgbm_image_uri()
        self.log_info("Using LightGBM built-in algorithm image: %s", image_uri)

        # Use modernized effective_source_dir with comprehensive hybrid resolution
        source_dir = self.config.effective_source_dir
        self.log_info("Using source directory: %s", source_dir)
        
        return Estimator(
            image_uri=image_uri,
            entry_point=self.config.training_entry_point,
            source_dir=source_dir,
            role=self.role,
            instance_type=self.config.training_instance_type,
            instance_count=self.config.training_instance_count,
            volume_size=self.config.training_volume_size,
            max_run=86400,  # 24 hours default
            input_mode='File',
            output_path=output_path,  # Use provided output_path directly
            base_job_name=self._generate_job_name(),  # Use standardized method with auto-detection
            sagemaker_session=self.session,
            environment=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the training job.
        For built-in algorithms, minimal environment variables are needed.

        Returns:
            A dictionary of environment variables.
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Add environment variables from config if they exist
        if hasattr(self.config, "env") and self.config.env:
            env_vars.update(self.config.env)

        self.log_info("Training environment variables: %s", env_vars)
        return env_vars

    def _create_data_channels_from_source(self, base_path):
        """
        Create train, validation, and test channel inputs from a base path.

        Args:
            base_path: Base S3 path containing train/val/test subdirectories

        Returns:
            Dictionary of channel name to TrainingInput
        """
        from sagemaker.workflow.functions import Join

        # Base path is used directly - property references are handled by PipelineAssembler

        channels = {
            "train": TrainingInput(s3_data=Join(on="/", values=[base_path, "train/"])),
            "val": TrainingInput(s3_data=Join(on="/", values=[base_path, "val/"])),
            "test": TrainingInput(s3_data=Join(on="/", values=[base_path, "test/"])),
        }

        return channels

    def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
        """
        Get inputs for the step using specification and contract.

        This method creates TrainingInput objects for each dependency defined in the specification.
        After refactor: Only handles data inputs, hyperparameters are embedded in source directory.

        Args:
            inputs: Input data sources keyed by logical name

        Returns:
            Dictionary of TrainingInput objects keyed by channel name

        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for input mapping")

        training_inputs = {}

        # Process each dependency in the specification
        for _, dependency_spec in self.spec.dependencies.items():
            logical_name = dependency_spec.logical_name

            # Skip hyperparameters_s3_uri if configured to do so
            if logical_name == "hyperparameters_s3_uri" and self.config.skip_hyperparameters_s3_uri:
                self.log_info("Skipping hyperparameters_s3_uri channel as configured (hyperparameters loaded from script folder)")
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

                # SPECIAL HANDLING FOR input_path
                # For '/opt/ml/input/data', we need to create train/val/test channels
                if logical_name == "input_path":
                    base_path = inputs[logical_name]

                    # Create separate channels for each data split using helper method
                    data_channels = self._create_data_channels_from_source(base_path)
                    training_inputs.update(data_channels)
                    self.log_info(
                        "Created data channels from %s: %s", logical_name, base_path
                    )
                else:
                    # For other inputs, extract the channel name from the container path
                    parts = container_path.split("/")
                    if (
                        len(parts) > 4
                        and parts[1] == "opt"
                        and parts[2] == "ml"
                        and parts[3] == "input"
                        and parts[4] == "data"
                    ):
                        if len(parts) > 5:
                            channel_name = parts[5]  # Extract channel name from path
                            training_inputs[channel_name] = TrainingInput(
                                s3_data=inputs[logical_name]
                            )
                            self.log_info(
                                "Created %s channel from %s: %s",
                                channel_name,
                                logical_name,
                                inputs[logical_name],
                            )
                        else:
                            # If no specific channel in path, use logical name as channel
                            training_inputs[logical_name] = TrainingInput(
                                s3_data=inputs[logical_name]
                            )
                            self.log_info(
                                "Created %s channel from %s: %s",
                                logical_name,
                                logical_name,
                                inputs[logical_name],
                            )
            else:
                raise ValueError(f"No container path found for input: {logical_name}")

        return training_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> str:
        """
        Get outputs for the step using specification and contract.

        For training steps, this returns the output path where model artifacts and evaluation results will be stored.
        SageMaker uses this single output_path parameter for both:
        - model.tar.gz (from /opt/ml/model/)
        - output.tar.gz (from /opt/ml/output/data/)

        Args:
            outputs: Output destinations keyed by logical name

        Returns:
            Output path for model artifacts and evaluation results

        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for output mapping")

        # First, check if any output path is explicitly provided in the outputs dictionary
        primary_output_path = None

        # Check if model_output or evaluation_output are in the outputs dictionary
        output_logical_names = [
            spec.logical_name for _, spec in self.spec.outputs.items()
        ]

        for logical_name in output_logical_names:
            if logical_name in outputs:
                primary_output_path = outputs[logical_name]
                self.log_info("Using provided output path from '%s': %s", 
                              logical_name, primary_output_path
                              )
                break

        # If no output path was provided, generate a default one
        if primary_output_path is None:
            # Generate a clean path using base output path and Join for parameter compatibility
            from sagemaker.workflow.functions import Join
            base_output_path = self._get_base_output_path()
            primary_output_path = Join(on="/", values=[base_output_path, "lightgbm_training"])
            self.log_info("Using generated base output path: %s", primary_output_path)

        # Get base job name for logging purposes
        base_job_name = self._generate_job_name()

        # Log how SageMaker will structure outputs under this path
        self.log_info(
            "SageMaker will organize outputs using base job name: %s", base_job_name
        )
        self.log_info("Full job name will be: %s-[timestamp]", base_job_name)
        self.log_info(
            "Output path structure will be: %s/%s-[timestamp]/", primary_output_path, base_job_name
        )
        self.log_info(
            "  - Model artifacts will be in: %s/%s-[timestamp]/output/model.tar.gz", primary_output_path, base_job_name
        )
        self.log_info(
            "  - Evaluation results will be in: %s/%s-[timestamp]/output/output.tar.gz", primary_output_path, base_job_name
        )

        return primary_output_path

    def create_step(self, **kwargs) -> TrainingStep:
        """
        Creates a SageMaker TrainingStep for the pipeline.

        This method creates the LightGBM estimator using built-in algorithm, sets up training inputs 
        from the input data, and creates the SageMaker TrainingStep. Hyperparameters are embedded 
        in the source directory.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Dictionary mapping input channel names to their S3 locations
                - input_path: Direct parameter for training data input path (for backward compatibility)
                - output_path: Direct parameter for model output path (for backward compatibility)
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: Whether to enable caching for this step.

        Returns:
            A configured TrainingStep instance.
        """
        # Extract common parameters
        inputs_raw = kwargs.get("inputs", {})
        input_path = kwargs.get("input_path")
        output_path = kwargs.get("output_path")
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        self.log_info("Creating LightGBM TrainingStep...")

        # Get the step name using standardized automatic step type detection
        step_name = self._get_step_name()

        # Handle inputs
        inputs = {}

        # If dependencies are provided, extract inputs from them using the resolver
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)

        # Add explicitly provided inputs (overriding any extracted ones)
        inputs.update(inputs_raw)

        # Add direct parameters if provided
        if input_path is not None:
            inputs["input_path"] = input_path

        # Get training inputs using specification-driven method
        # Note: Hyperparameters are now embedded in source directory
        training_inputs = self._get_inputs(inputs)

        # Make sure we have the inputs we need
        if len(training_inputs) == 0:
            raise ValueError(
                "No training inputs available. Provide input_path or ensure dependencies supply necessary outputs."
            )

        self.log_info("Final training inputs: %s", list(training_inputs.keys()))

        # Get output path using specification-driven method
        output_path = self._get_outputs({})

        # Create estimator
        estimator = self._create_estimator(output_path)

        # Create the training step
        try:
            training_step = TrainingStep(
                name=step_name,
                estimator=estimator,
                inputs=training_inputs,
                depends_on=dependencies,
                cache_config=self._get_cache_config(enable_caching),
            )

            # Attach specification to the step for future reference
            setattr(training_step, "_spec", self.spec)

            # Log successful creation
            self.log_info("Created TrainingStep with name: %s", training_step.name)

            return training_step

        except Exception as e:
            self.log_warning("Error creating LightGBM TrainingStep: %s", str(e))
            raise ValueError(f"Failed to create LightGBMTrainingStep: {str(e)}") from e
