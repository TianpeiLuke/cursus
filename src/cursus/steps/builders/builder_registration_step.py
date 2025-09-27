from typing import Dict, Optional, Any, List, Set, Union
from pathlib import Path
import logging

from sagemaker.workflow.steps import Step
from sagemaker.processing import ProcessingInput
from sagemaker.workflow.properties import Properties

# Import the customized step
from secure_ai_sandbox_workflow_python_sdk.mims_model_registration.mims_model_registration_processing_step import (
    MimsModelRegistrationProcessingStep,
)

from ..configs.config_registration_step import RegistrationConfig
from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Import the registration specification
try:
    from ..specs.registration_spec import REGISTRATION_SPEC

    SPEC_AVAILABLE = True
except ImportError:
    REGISTRATION_SPEC = None
    SPEC_AVAILABLE = False

# Import the script contract
try:
    from ..contracts.mims_registration_contract import MIMS_REGISTRATION_CONTRACT

    CONTRACT_AVAILABLE = True
except ImportError:
    MIMS_REGISTRATION_CONTRACT = None
    CONTRACT_AVAILABLE = False

logger = logging.getLogger(__name__)


class RegistrationStepBuilder(StepBuilderBase):
    """
    Builder for a Model Registration ProcessingStep.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that registers a model with MIMS.
    """

    def __init__(
        self,
        config: RegistrationConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        """
        Initializes the builder with a specific configuration for the model registration step.

        Args:
            config: A RegistrationConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        if not isinstance(config, RegistrationConfig):
            raise ValueError(
                "RegistrationStepBuilder requires a RegistrationConfig instance."
            )

        # Use the registration specification if available
        spec = REGISTRATION_SPEC if SPEC_AVAILABLE else None

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: RegistrationConfig = config

        # Store contract reference
        self.contract = MIMS_REGISTRATION_CONTRACT if CONTRACT_AVAILABLE else None

        if self.spec and not self.contract:
            self.log_warning(
                "Script contract not available - path resolution will use hardcoded values"
            )

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        self.log_info("Validating RegistrationConfig...")

        # Validate required attributes that are actually defined in the config
        required_attrs = [
            "region",
            "model_domain",
            "model_objective",
            "framework",
            "inference_instance_type",
            "inference_entry_point",
            "source_model_inference_content_types",
            "source_model_inference_response_types",
            "source_model_inference_input_variable_list",
            "source_model_inference_output_variable_list",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [
                None,
                "",
            ]:
                raise ValueError(
                    f"RegistrationConfig missing required attribute: {attr}"
                )

        # Validate spec-contract alignment if both are available
        if self.spec and self.contract:
            # Check if all required dependencies have container paths in the contract
            for _, dependency in self.spec.dependencies.items():
                logical_name = dependency.logical_name
                if (
                    dependency.required
                    and logical_name not in self.contract.expected_input_paths
                ):
                    raise ValueError(
                        f"Required dependency '{logical_name}' in spec not found in contract expected_input_paths"
                    )

        self.log_info("RegistrationConfig validation succeeded.")

    # No special handling needed since upstream steps now provide paths with .tar.gz suffix

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.

        This method creates ProcessingInput objects with the exact structure required by the MIMS SDK.
        The MIMS SDK has strict requirements about the order and structure of ProcessingInput objects.

        Args:
            inputs: Input data sources keyed by logical name

        Returns:
            List of ProcessingInput objects in the specific order required by MIMS SDK

        Raises:
            ValueError: If required inputs are missing
        """
        if not self.contract:
            self.log_warning(
                "Script contract not available - path resolution will use hardcoded values"
            )

        # Create a new list to store the properly ordered ProcessingInput objects
        ordered_processing_inputs = []

        # CRITICAL: The MIMS SDK expects exactly 1 or 2 ProcessingInput objects in a specific order

        # 1. First (required): PackagedModel must be the first input
        model_logical_name = "PackagedModel"
        if model_logical_name not in inputs:
            raise ValueError(f"Required input '{model_logical_name}' not provided")

        # Get container path from contract (which we verified matches the MIMS script expectations)
        model_container_path = self.contract.expected_input_paths.get(
            model_logical_name,
            "/opt/ml/processing/input/model",  # Fallback if contract not available
        )

        # Use model source directly - upstream changes ensure it ends with .tar.gz
        model_source = inputs[model_logical_name]

        self.log_info(
            "Using source for '%s' directly without wrapper", model_logical_name
        )

        # Add the model input first (order matters for MIMS SDK validation)
        ordered_processing_inputs.append(
            ProcessingInput(
                input_name=model_logical_name,  # Use the logical name as input_name
                source=model_source,
                destination=model_container_path,
                s3_data_distribution_type="FullyReplicated",
                s3_input_mode="File",
            )
        )

        # 2. Second (may be optional depending on spec): Payload samples
        payload_logical_name = "GeneratedPayloadSamples"
        if payload_logical_name in inputs:
            payload_container_path = self.contract.expected_input_paths.get(
                payload_logical_name,
                "/opt/ml/processing/mims_payload",  # Fallback if contract not available
            )

            # Use payload source directly - upstream changes ensure it ends with .tar.gz
            payload_source = inputs[payload_logical_name]

            self.log_info(
                "Using source for '%s' directly without wrapper", payload_logical_name
            )

            # Add the payload input second
            ordered_processing_inputs.append(
                ProcessingInput(
                    input_name=payload_logical_name,  # Use the logical name as input_name
                    source=payload_source,
                    destination=payload_container_path,
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File",
                )
            )

        self.log_info(
            "Created %s ProcessingInput objects in required order",
            len(ordered_processing_inputs),
        )
        return ordered_processing_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> None:
        """
        Get outputs for the step.

        Registration step has no outputs - it registers the model as a side effect.

        Args:
            outputs: Output destinations (unused for registration step)

        Returns:
            None - registration step produces no outputs
        """
        return None

    def create_step(self, **kwargs) -> Step:
        """
        Creates a MimsModelRegistrationProcessingStep using specification-driven approach.

        This simplified method leverages the specification and dependency resolver to automatically
        handle input resolution, eliminating complex parameter handling logic.

        Args:
            **kwargs: Keyword arguments including:
                - dependencies: List of upstream steps (preferred approach)
                - inputs: Dictionary of input mappings (fallback)
                - performance_metadata_location: Optional S3 location of performance metadata

        Returns:
            A configured MimsModelRegistrationProcessingStep instance
        """
        self.log_info("Creating MimsModelRegistrationProcessingStep...")

        # Extract core parameters
        dependencies = kwargs.get("dependencies", [])
        performance_metadata_location = kwargs.get("performance_metadata_location")

        # Use specification-driven input resolution
        inputs = {}
        if dependencies:
            try:
                inputs = self.extract_inputs_from_dependencies(dependencies)
                self.log_info(
                    "Extracted inputs from dependencies: %s", list(inputs.keys())
                )
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)

        # Allow manual input override/supplement
        inputs.update(kwargs.get("inputs", {}))

        # Validate we have required inputs
        if not inputs:
            raise ValueError(
                "No inputs provided. Either specify 'dependencies' or 'inputs'."
            )

        # Get processing inputs using specification-driven method
        processing_inputs = self._get_inputs(inputs)

        # Create step with clean, simple logic using standardized auto-detection method
        step_name = self._get_step_name() + "-" + self.config.region

        try:
            # Create registration step
            registration_step = MimsModelRegistrationProcessingStep(
                step_name=step_name,
                role=self.role,
                sagemaker_session=self.session,
                processing_input=processing_inputs,
                performance_metadata_location=performance_metadata_location,
                depends_on=dependencies,
            )

            # Attach specification and contract for future reference
            if self.spec:
                setattr(registration_step, "_spec", self.spec)
            if self.contract:
                setattr(registration_step, "_contract", self.contract)

            self.log_info(
                "Created MimsModelRegistrationProcessingStep: %s",
                registration_step.name,
            )
            return registration_step

        except Exception as e:
            self.log_error("Error creating MimsModelRegistrationProcessingStep: %s", e)
            raise ValueError(
                f"Failed to create MimsModelRegistrationProcessingStep: {e}"
            ) from e
