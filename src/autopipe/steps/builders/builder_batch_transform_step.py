from typing import Optional, List, Union, Dict, Any
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
from ..registry.builder_registry import register_builder

# Import specifications based on job type
try:
    from ..specs.batch_transform_training_spec import BATCH_TRANSFORM_TRAINING_SPEC
    from ..specs.batch_transform_calibration_spec import BATCH_TRANSFORM_CALIBRATION_SPEC
    from ..specs.batch_transform_validation_spec import BATCH_TRANSFORM_VALIDATION_SPEC
    from ..specs.batch_transform_testing_spec import BATCH_TRANSFORM_TESTING_SPEC
    SPECS_AVAILABLE = True
except ImportError:
    BATCH_TRANSFORM_TRAINING_SPEC = BATCH_TRANSFORM_CALIBRATION_SPEC = BATCH_TRANSFORM_VALIDATION_SPEC = BATCH_TRANSFORM_TESTING_SPEC = None
    SPECS_AVAILABLE = False

logger = logging.getLogger(__name__)

@register_builder()
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
        notebook_root: Optional[Path] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None
    ):
        """
        Initialize with specification based on job type.
        
        Args:
            config: Configuration for the step
            sagemaker_session: SageMaker session
            role: IAM role
            notebook_root: Root directory of notebook
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
        if not hasattr(config, 'job_type'):
            raise ValueError("config.job_type must be specified")
            
        job_type = config.job_type.lower()
        
        # Get specification based on job type
        if job_type == "training" and SPECS_AVAILABLE and BATCH_TRANSFORM_TRAINING_SPEC is not None:
            spec = BATCH_TRANSFORM_TRAINING_SPEC
        elif job_type == "calibration" and SPECS_AVAILABLE and BATCH_TRANSFORM_CALIBRATION_SPEC is not None:
            spec = BATCH_TRANSFORM_CALIBRATION_SPEC
        elif job_type == "validation" and SPECS_AVAILABLE and BATCH_TRANSFORM_VALIDATION_SPEC is not None:
            spec = BATCH_TRANSFORM_VALIDATION_SPEC
        elif job_type == "testing" and SPECS_AVAILABLE and BATCH_TRANSFORM_TESTING_SPEC is not None:
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
                self.log_warning("Could not import specification for job type: %s, error: %s", job_type, e)
        
        # Even if we don't have a spec, continue without one
        if spec:
            self.log_info("Using specification for batch transform %s", job_type)
        else:
            self.log_info("No specification found for batch transform job type: %s, continuing with default behavior", job_type)
            
        super().__init__(
            config=config, 
            spec=spec,
            sagemaker_session=sagemaker_session, 
            role=role, 
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: BatchTransformStepConfig = config

    def validate_configuration(self) -> None:
        """
        Validate that all required transform settings are provided.
        """
        # Validate job type
        if self.config.job_type not in {"training", "testing", "validation", "calibration"}:
            raise ValueError(f"Unsupported job_type: {self.config.job_type}")
        
        # Validate other required fields
        required_attrs = [
            'transform_instance_type', 'transform_instance_count'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
                raise ValueError(f"Missing required attribute: {attr}")
                
        self.log_info("BatchTransformStepBuilder configuration for '%s' validated.", self.config.job_type)

    def _create_transformer(self, model_name: Union[str, Properties], output_path: Optional[str] = None) -> Transformer:
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

    def _get_inputs(self, inputs: Dict[str, Any]) -> tuple[TransformInput, Union[str, Properties]]:
        """
        Create transform input using specification and provided inputs.
        
        This method creates a TransformInput object based on the configuration
        and input dependencies.
        
        Args:
            inputs: Input data sources keyed by logical name
            
        Returns:
            TransformInput object
            
        Raises:
            ValueError: If required inputs are missing
        """
        # Process model_name input
        model_name = None
        if 'model_name' in inputs:
            model_name = inputs['model_name']
            self.log_info("Using model_name from dependencies: %s", model_name)
        
        if not model_name:
            raise ValueError("model_name is required but not provided in inputs")
            
        # Process data input (must come from dependencies)
        input_data = None
        
        # Check for processed_data or input_data in the inputs
        if 'processed_data' in inputs:
            input_data = inputs['processed_data']
            self.log_info("Using processed_data from dependencies: %s", input_data)
        elif 'input_data' in inputs:  # backward compatibility
            input_data = inputs['input_data']
            self.log_info("Using input_data from dependencies: %s", input_data)
        
        if not input_data:
            raise ValueError("Input data source (processed_data) is required but not provided in inputs")
            
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
        
    def _get_outputs(self, outputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Process outputs based on specification.
        
        For batch transform, this simply returns a dictionary of output information
        for reference, as the TransformStep doesn't take explicit output destinations.
        
        Args:
            outputs: Output destinations keyed by logical name
            
        Returns:
            Dictionary of output information
        """
        # No explicit outputs need to be configured for TransformStep
        # Just log the outputs that will be available
        result = {}
        
        # If we have a specification, include output information
        if self.spec:
            for output_spec in self.spec.outputs.values():
                logical_name = output_spec.logical_name
                if logical_name in outputs:
                    # If explicit output path provided
                    result[logical_name] = outputs[logical_name]
                else:
                    # Default transform output path will be determined by SageMaker
                    result[logical_name] = f"Will be available at: {output_spec.property_path}"
        
        self.log_info("Transform step will produce outputs: %s", list(result.keys()))
        return result

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
        inputs_raw = kwargs.get('inputs', {})
        outputs = kwargs.get('outputs', {})
        dependencies = kwargs.get('dependencies', [])
        enable_caching = kwargs.get('enable_caching', True)
        
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
        
        # Process outputs (mostly for logging in batch transform case)
        self._get_outputs(outputs)
        
        # Build the transformer
        transformer = self._create_transformer(model_name)

        # Get step name using standardized method with auto-detection
        step_name = self._get_step_name()
        
        # Create the transform step
        transform_step = TransformStep(
            name=step_name,
            transformer=transformer,
            inputs=transform_input,
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(enable_caching) if enable_caching else None
        )
        
        # Attach specification to the step for future reference
        if hasattr(self, 'spec') and self.spec:
            setattr(transform_step, '_spec', self.spec)
            
        self.log_info("Created TransformStep with name: %s", step_name)
        return transform_step
