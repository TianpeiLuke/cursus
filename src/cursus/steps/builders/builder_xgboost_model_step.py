from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import CreateModelStep, Step
from sagemaker.xgboost import XGBoostModel
from sagemaker.model import Model
from sagemaker import image_uris

from ..configs.config_xgboost_model_step import XGBoostModelStepConfig
from ...core.base.builder_base import StepBuilderBase
from ..specs.xgboost_model_spec import XGBOOST_MODEL_SPEC

logger = logging.getLogger(__name__)


class XGBoostModelStepBuilder(StepBuilderBase):
    """
    Builder for an XGBoost Model Step.
    This class is responsible for configuring and creating a SageMaker ModelStep
    that creates an XGBoost model from a trained model artifact.
    """

    def __init__(
        self,
        config: XGBoostModelStepConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initializes the builder with a specific configuration for the model step.

        Args:
            config: A XGBoostModelStepConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Model.
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        if not isinstance(config, XGBoostModelStepConfig):
            raise ValueError(
                "XGBoostModelStepBuilder requires a XGBoostModelStepConfig instance."
            )

        # Validate specification availability
        if XGBOOST_MODEL_SPEC is None:
            raise ValueError("XGBoost model specification not available")

        super().__init__(
            config=config,
            spec=XGBOOST_MODEL_SPEC,  # Add specification
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: XGBoostModelStepConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        self.log_info("Validating XGBoostModelStepConfig...")

        # Validate required attributes
        required_attrs = ["instance_type", "entry_point", "source_dir"]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [
                None,
                "",
            ]:
                raise ValueError(
                    f"XGBoostModelStepConfig missing required attribute: {attr}"
                )

        self.log_info("XGBoostModelStepConfig validation succeeded.")

    def _get_image_uri(self) -> str:
        """
        Generate the appropriate SageMaker XGBoost container image URI.
        Uses the SageMaker SDK's built-in image_uris.retrieve function.
        Forces the region to us-east-1 regardless of the configured region.

        Returns:
            A string containing the image URI for the XGBoost container.
        """
        # Get region from configuration but enforce us-east-1
        region = getattr(self.config, "aws_region", "us-east-1")
        if region != "us-east-1":
            self.log_info(
                "Region '%s' specified, but forcing to 'us-east-1' due to environment limitations", region
            )
        region = "us-east-1"

        # Retrieve the image URI using SageMaker SDK
        image_uri = image_uris.retrieve(
            framework="xgboost",
            region=region,
            version=self.config.framework_version,
            py_version=self.config.py_version,
            instance_type=self.config.instance_type,
            image_scope="inference",
        )

        self.log_info("Generated XGBoost image URI: %s", image_uri)
        return image_uri

    def _create_model(self, model_data: str) -> XGBoostModel:
        """
        Creates and configures the XGBoostModel.
        This defines the model that will be deployed, including the model artifacts,
        inference code, and environment.

        Args:
            model_data: The S3 URI of the model artifacts.

        Returns:
            An instance of sagemaker.xgboost.XGBoostModel.
        """
        # Generate the image URI automatically
        image_uri = self._get_image_uri()

        # Use source directory with hybrid resolution fallback
        source_dir = (
            self.config.effective_source_dir
        )
        self.log_info("Using source directory: %s", source_dir)
        
        return XGBoostModel(
            model_data=model_data,
            role=self.role,
            entry_point=self.config.entry_point,
            source_dir=source_dir,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            image_uri=image_uri,
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the model.
        These variables are used to control the behavior of the inference code.

        Returns:
            A dictionary of environment variables.
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Add environment variables from config if they exist
        if hasattr(self.config, "env") and self.config.env:
            env_vars.update(self.config.env)

        self.log_info("Model environment variables: %s", env_vars)
        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use specification dependencies to get model_data.

        Args:
            inputs: Dictionary of available inputs

        Returns:
            Dictionary containing processed inputs for model creation
        """
        # Spec defines: model_data dependency from XGBoostTraining, ProcessingStep, ModelArtifactsStep
        model_data_key = "model_data"  # From spec.dependencies

        if model_data_key not in inputs:
            raise ValueError(f"Required input '{model_data_key}' not found")

        return {model_data_key: inputs[model_data_key]}

    def _get_outputs(self, outputs: Dict[str, Any]) -> None:
        """
        Use specification outputs - returns model name.

        Args:
            outputs: Dictionary to store outputs (not used for CreateModelStep)

        Returns:
            None - CreateModelStep handles outputs automatically
        """
        # Spec defines: model output with property_path="properties.ModelName"
        # For CreateModelStep, we don't need to return specific outputs
        # The step automatically provides ModelName property
        return None

    def create_step(self, **kwargs) -> CreateModelStep:
        """
        Creates the final, fully configured SageMaker ModelStep for the pipeline.
        This method orchestrates the assembly of the model and its configuration
        into a single, executable pipeline step.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Dictionary mapping input channel names to their S3 locations
                - model_data: Direct parameter for model artifacts S3 URI (for backward compatibility)
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: Whether to enable caching for this step.

        Returns:
            A configured ModelStep instance.
        """
        # Extract common parameters
        inputs_raw = kwargs.get("inputs", {})
        model_data = kwargs.get("model_data")
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        self.log_info("Creating XGBoost ModelStep...")

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

        # Add direct parameters if provided (for backward compatibility)
        if model_data is not None:
            inputs["model_data"] = model_data

        # Use specification-driven input processing
        model_inputs = self._get_inputs(inputs)
        model_data_value = model_inputs["model_data"]

        # Create the model
        model = self._create_model(model_data_value)

        # Create the model step
        try:
            # Note: CreateModelStep does not support cache_config parameter
            # Log warning if caching was requested
            if enable_caching:
                self.log_warning(
                    "CreateModelStep does not support caching - ignoring enable_caching=True"
                )

            model_step = CreateModelStep(
                name=step_name,
                model=model,  # Pass model directly instead of step_args
                depends_on=dependencies,
                # Note: cache_config parameter removed - not supported by CreateModelStep
            )

            # Attach specification to the step for future reference
            setattr(model_step, "_spec", self.spec)

            # Log successful creation
            self.log_info("Created ModelStep with name: %s", model_step.name)

            return model_step

        except Exception as e:
            self.log_warning("Error creating XGBoost ModelStep: %s", str(e))
            raise ValueError(f"Failed to create XGBoostModelStep: {str(e)}") from e
