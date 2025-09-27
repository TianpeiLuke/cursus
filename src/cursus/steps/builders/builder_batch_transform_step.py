from typing import Optional, List, Union, Dict, Any, Tuple
from pathlib import Path
import logging
import importlib

from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep, Step
from sagemaker.inputs import TransformInput
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.pipeline_context import PipelineSession

from ..configs.config_batch_transform_step import BatchTransformStepConfig
from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Import specifications based on job type
try:
    from ..specs.batch_transform_training_spec import BATCH_TRANSFORM_TRAINING_SPEC
    from ..specs.batch_transform_calibration_spec import (
        BATCH_TRANSFORM_CALIBRATION_SPEC,
    )
    from ..specs.batch_transform_validation_spec import BATCH_TRANSFORM_VALIDATION_SPEC
    from ..specs.batch_transform_testing_spec import BATCH_TRANSFORM_TESTING_SPEC

    SPECS_AVAILABLE = True
except ImportError:
    BATCH_TRANSFORM_TRAINING_SPEC = BATCH_TRANSFORM_CALIBRATION_SPEC = (
        BATCH_TRANSFORM_VALIDATION_SPEC
    ) = BATCH_TRANSFORM_TESTING_SPEC = None
    SPECS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BatchTransformStepBuilder(StepBuilderBase):
    """
    Builder for creating a SageMaker Batch Transform step in a workflow.

    This implementation uses the specification-driven approach where dependencies, outputs,
    and behavior are defined by step specifications.
    """

    def __init__(
        self,
        config: BatchTransformStepConfig,
        sagemaker_session: Optional[PipelineSession] = None,
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
            ValueError: If the configuration is invalid
        """
        if not isinstance(config, BatchTransformStepConfig):
            raise ValueError(
                "BatchTransformStepBuilder requires a BatchTransformStepConfig instance."
            )

        # Get the appropriate spec based on job type
        spec = None
        if not hasattr(config, "job_type"):
            raise ValueError("config.job_type must be specified")

        job_type = config.job_type.lower()

        # Get specification based on job type
        if (
            job_type == "training"
            and SPECS_AVAILABLE
            and BATCH_TRANSFORM_TRAINING_SPEC is not None
        ):
            spec = BATCH_TRANSFORM_TRAINING_SPEC
        elif (
            job_type == "calibration"
            and SPECS_AVAILABLE
            and BATCH_TRANSFORM_CALIBRATION_SPEC is not None
        ):
            spec = BATCH_TRANSFORM_CALIBRATION_SPEC
        elif (
            job_type == "validation"
            and SPECS_AVAILABLE
            and BATCH_TRANSFORM_VALIDATION_SPEC is not None
        ):
            spec = BATCH_TRANSFORM_VALIDATION_SPEC
        elif (
            job_type == "testing"
            and SPECS_AVAILABLE
            and BATCH_TRANSFORM_TESTING_SPEC is not None
        ):
            spec = BATCH_TRANSFORM_TESTING_SPEC
        else:
            # Try dynamic import
            try:
                module_path = f"..pipeline_step_specs.batch_transform_{job_type}_spec"
                module = importlib.import_module(module_path, package=__package__)
                spec_var_name = f"BATCH_TRANSFORM_{job_type.upper()}_SPEC"
                if hasattr(module, spec_var_name):
                    spec = getattr(module, spec_var_name)
            except (ImportError, AttributeError) as e:
                self.log_warning(
                    "Could not import specification for job type: %s, error: %s",
                    job_type,
                    e,
                )

        # Even if we don't have a spec, continue without one
        if spec:
            self.log_info("Using specification for batch transform %s", job_type)
        else:
            self.log_info(
                "No specification found for batch transform job type: %s, continuing with default behavior",
                job_type,
            )

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: BatchTransformStepConfig = config

    def validate_configuration(self) -> None:
        """
        Validate that all required transform settings are provided.
        """
        # Validate job type
        if self.config.job_type not in {
            "training",
            "testing",
            "validation",
            "calibration",
        }:
            raise ValueError(f"Unsupported job_type: {self.config.job_type}")

        # Validate other required fields
        required_attrs = ["transform_instance_type", "transform_instance_count"]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
                raise ValueError(f"Missing required attribute: {attr}")

        self.log_info(
            "BatchTransformStepBuilder configuration for '%s' validated.",
            self.config.job_type,
        )

    def _create_transformer(
        self, model_name: Union[str, Properties], output_path: Optional[str] = None
    ) -> Transformer:
        """
        Create the SageMaker Transformer object.

        Args:
            model_name: Name of the model to transform with
            output_path: Optional output path for transform job results

        Returns:
            Configured Transformer object
        """
        return Transformer(
            model_name=model_name,
            instance_type=self.config.transform_instance_type,
            instance_count=self.config.transform_instance_count,
            output_path=output_path,  # Will be determined by SageMaker if None
            accept=self.config.accept,
            assemble_with=self.config.assemble_with,
            sagemaker_session=self.session,
        )

    def _get_inputs(
        self, inputs: Dict[str, Any]
    ) -> Tuple[TransformInput, Union[str, Properties]]:
        """
        Create transform input using specification and provided inputs.

        This method creates a TransformInput object based on the specification
        dependencies and input data sources.

        Args:
            inputs: Input data sources keyed by logical name

        Returns:
            Tuple of (TransformInput object, model_name)

        Raises:
            ValueError: If required inputs are missing or specification is not available
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        model_name = None
        input_data = None

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
                raise ValueError(
                    f"Required input '{logical_name}' not provided. "
                    f"Expected from compatible sources: {dependency_spec.compatible_sources}"
                )

            # Handle specific dependency types based on logical name
            if logical_name == "model_name":
                model_name = inputs[logical_name]
                self.log_info(
                    "Using model_name from dependencies: %s (type: %s)",
                    model_name,
                    dependency_spec.dependency_type,
                )
            elif logical_name == "processed_data":
                input_data = inputs[logical_name]
                self.log_info(
                    "Using processed_data from dependencies: %s (type: %s)",
                    input_data,
                    dependency_spec.dependency_type,
                )
            else:
                # Log unexpected logical names for debugging
                self.log_warning(
                    "Unexpected logical name '%s' in specification dependencies",
                    logical_name,
                )

        # Final validation - ensure we got the required inputs
        if not model_name:
            raise ValueError(
                "model_name is required but not provided in inputs. "
                "Check that a model step (PytorchModel, XgboostModel) is properly connected."
            )

        if not input_data:
            raise ValueError(
                "processed_data is required but not provided in inputs. "
                "Check that a preprocessing step (TabularPreprocessing) is properly connected."
            )

        # Create the transform input
        transform_input = TransformInput(
            data=input_data,
            content_type=self.config.content_type,
            split_type=self.config.split_type,
            join_source=self.config.join_source,
            input_filter=self.config.input_filter,
            output_filter=self.config.output_filter,
        )

        return transform_input, model_name

    def _get_outputs(self, outputs: Dict[str, Any]) -> str:
        """
        Get the S3 output path for the transform job using specification.
        
        This path will be used by the Transformer and will be accessible via 
        properties.TransformOutput.S3OutputPath at runtime.

        Args:
            outputs: Output destinations keyed by logical name

        Returns:
            str: S3 URI where transform outputs will be stored
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        # Check for explicitly provided output path
        for _, output_spec in self.spec.outputs.items():
            logical_name = output_spec.logical_name
            if logical_name in outputs:
                self.log_info(
                    "Using provided output path from '%s': %s", 
                    logical_name, 
                    outputs[logical_name]
                )
                return outputs[logical_name]

        # Generate default output path using specification
        base_output_path = self._get_base_output_path()
        step_type = self.spec.step_type.lower() if hasattr(self.spec, 'step_type') else 'batch_transform'
        
        from sagemaker.workflow.functions import Join
        output_path = Join(on="/", values=[base_output_path, step_type, self.config.job_type])
        
        self.log_info("Generated default output path: %s", output_path)
        return output_path

    def create_step(self, **kwargs) -> TransformStep:
        """
        Create a TransformStep for a batch transform.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - model_name: The name of the SageMaker model (string or Properties) (required)
                - inputs: Input data sources keyed by logical name
                - outputs: Output destinations keyed by logical name
                - dependencies: Optional list of Pipeline Step dependencies
                - enable_caching: Whether to enable caching for this step (default: True)

        Returns:
            TransformStep: configured batch transform step.
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

        # Get transformer inputs and model name
        transform_input, model_name = self._get_inputs(inputs)

        # Get output path using specification-driven method
        output_path = self._get_outputs(outputs)

        # Build the transformer with the output path
        transformer = self._create_transformer(model_name, output_path)

        # Get step name using standardized method with auto-detection
        step_name = self._get_step_name()

        # Create the transform step
        transform_step = TransformStep(
            name=step_name,
            transformer=transformer,
            inputs=transform_input,
            depends_on=dependencies or [],
            cache_config=(
                self._get_cache_config(enable_caching) if enable_caching else None
            ),
        )

        # Attach specification to the step for future reference
        if hasattr(self, "spec") and self.spec:
            setattr(transform_step, "_spec", self.spec)

        self.log_info("Created TransformStep with name: %s", step_name)
        return transform_step
