# Step Builder Implementation

Step builders bridge the gap between declarative specifications and executable SageMaker steps. They transform configuration and specification information into concrete SageMaker pipeline steps that can be executed.

## Purpose of Step Builders

Step builders serve several important purposes:

1. **SageMaker Integration**: They create the actual SageMaker step objects
2. **Resource Configuration**: They apply configuration settings for SageMaker resources
3. **Input/Output Mapping**: They connect inputs and outputs based on specifications
4. **Environment Setup**: They configure environment variables for scripts
5. **Dependency Resolution**: They extract dependencies from connected steps

## Builder Structure

A step builder is implemented as a class that extends `StepBuilderBase` and includes the following key components:

```python
from typing import Dict, List, Any, Optional
from pathlib import Path

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from ...core.base.specification_base import StepSpecification
from ...core.base.contract_base import ScriptContract
from ...core.base.builder_base import StepBuilderBase
from ..configs.config_your_step import YourStepConfig
from ..specs.your_step_spec import YOUR_STEP_SPEC
from ...registry.hybrid.manager import UnifiedRegistryManager

class YourStepBuilder(StepBuilderBase):
    """Builder for YourStep processing step."""
    
    def __init__(
        self, 
        config: YourStepConfig, 
        spec: Optional[StepSpecification] = None,
        sagemaker_session: Optional[PipelineSession] = None, 
        role: Optional[str] = None, 
        notebook_root: Optional[Path] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None
    ):
        """Initialize the step builder with configuration and specification."""
        if not isinstance(config, YourStepConfig):
            raise ValueError("YourStepBuilder requires a YourStepConfig instance.")
        
        # Get the appropriate specification based on job type if not provided
        if spec is None:
            job_type = getattr(config, 'job_type', None)
            if job_type and hasattr(self, '_get_spec_for_job_type'):
                spec = self._get_spec_for_job_type(job_type)
            else:
                spec = YOUR_STEP_SPEC
        
        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: YourStepConfig = config
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for the processor using the specification and contract."""
        return self._get_spec_driven_processor_inputs(inputs)
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for the processor using the specification and contract."""
        return self._get_spec_driven_processor_outputs(outputs)
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processor."""
        # Use the standardized method from the base class
        env_vars = super()._get_environment_variables()
        
        # Add any additional environment variables needed by your script
        env_vars.update({
            "ADDITIONAL_PARAM": self.config.additional_param
        })
        
        return env_vars
    
    def _get_processor(self):
        """Create and return a SageMaker processor."""
        # Create the processor using SageMaker SDK
        from sagemaker.processing import Processor
        
        processor = Processor(
            role=self.role,
            image_uri=self.config.get_image_uri(),
            instance_count=self.config.instance_count,
            instance_type=self.config.instance_type,
            volume_size_in_gb=self.config.volume_size_gb,
            max_runtime_in_seconds=self.config.max_runtime_seconds,
            sagemaker_session=self.sagemaker_session
        )
        
        return processor
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create the processing step.
        
        Args:
            **kwargs: Additional keyword arguments for step creation.
                     Should include 'dependencies' list if step has dependencies.
        """
        # Extract inputs from dependencies using the resolver
        dependencies = kwargs.get('dependencies', [])
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        
        # Get processor inputs and outputs
        inputs = self._get_inputs(extracted_inputs)
        outputs = self._get_outputs({})
        
        # Create processor
        processor = self._get_processor()
        
        # Set environment variables
        env_vars = self._get_environment_variables()
        
        # Create and return the step
        step_name = kwargs.get('step_name', 'YourStep')
        step = processor.run(
            inputs=inputs,
            outputs=outputs,
            container_arguments=[],
            container_entrypoint=["python", self.config.get_script_path()],
            job_name=self._generate_job_name(),
            wait=False,
            environment=env_vars
        )
        
        # Store specification in step for future reference
        setattr(step, '_spec', self.spec)
        
        return step
```

## How to Develop a Step Builder

### 1. Choose the Right Base Class and Design Pattern

Select the appropriate base builder for your step type. **Important**: Different SageMaker step types require different design patterns and implementation approaches. Refer to the detailed design documents for step-type-specific patterns:

#### Base Classes
- **StepBuilderBase**: Base class for all step builders
- **ProcessingStepBuilder**: For processing steps
- **TrainingStepBuilder**: For training steps
- **ModelStepBuilder**: For model steps
- **TransformStepBuilder**: For transform steps

#### Step-Type-Specific Design Patterns

Each SageMaker step type has unique characteristics and requirements. Consult these design documents for detailed implementation patterns:

- **Processing Steps**: See [`slipbox/1_design/processing_step_builder_patterns.md`](../1_design/processing_step_builder_patterns.md)
  - ProcessingStep creation patterns
  - Input/output handling for processing jobs
  - Environment variable management
  - Container and script configuration

- **Training Steps**: See [`slipbox/1_design/training_step_builder_patterns.md`](../1_design/training_step_builder_patterns.md)
  - TrainingStep and estimator patterns
  - Hyperparameter handling
  - Training input channels
  - Model artifact management

- **Transform Steps**: See [`slipbox/1_design/transform_step_builder_patterns.md`](../1_design/transform_step_builder_patterns.md)
  - TransformStep creation patterns
  - Batch transform configuration
  - Input/output data handling
  - Transform job parameters

- **Model Creation Steps**: See [`slipbox/1_design/createmodel_step_builder_patterns.md`](../1_design/createmodel_step_builder_patterns.md)
  - CreateModelStep patterns
  - Model registration and deployment
  - Container image configuration
  - Model artifact handling

- **Comprehensive Overview**: See [`slipbox/1_design/step_builder_patterns_summary.md`](../1_design/step_builder_patterns_summary.md)
  - Summary of all step builder patterns
  - Comparison of different approaches
  - Best practices across step types

#### Why Step-Type-Specific Patterns Matter

Different SageMaker step types have fundamentally different:
- **SDK Integration**: Different SageMaker SDK classes (ProcessingStep, TrainingStep, TransformStep, etc.)
- **Resource Configuration**: Different resource types (processors, estimators, transformers, models)
- **Input/Output Handling**: Different data flow patterns and requirements
- **Parameter Management**: Different ways to pass parameters (environment variables, hyperparameters, etc.)
- **Dependency Patterns**: Different ways steps connect and share data

**Always consult the appropriate design pattern document** before implementing a step builder for a specific SageMaker step type.

### 2. Handle Job Type Variants

Many steps in the pipeline support different job types (e.g., "training", "calibration", "validation", "testing") that require different specifications and behaviors. Here's how to implement this pattern:

#### Understanding Job Type Variants

Job type variants allow the same step builder to handle different use cases:
- **Training**: Process data for model training
- **Calibration**: Process data for model calibration
- **Validation**: Process data for model validation
- **Testing**: Process data for final testing

Each job type may have:
- Different input/output requirements
- Different processing logic
- Different specifications

#### Implementation Pattern

```python
from typing import Dict, Optional, Any, List
from pathlib import Path
import logging
import importlib

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_your_step import YourStepConfig
from ...core.base.builder_base import StepBuilderBase
from ...registry.hybrid.manager import UnifiedRegistryManager

# Import different specifications for different job types
try:
    from ..specs.your_step_training_spec import YOUR_STEP_TRAINING_SPEC
    from ..specs.your_step_calibration_spec import YOUR_STEP_CALIBRATION_SPEC
    from ..specs.your_step_validation_spec import YOUR_STEP_VALIDATION_SPEC
    from ..specs.your_step_testing_spec import YOUR_STEP_TESTING_SPEC
    SPECS_AVAILABLE = True
except ImportError:
    YOUR_STEP_TRAINING_SPEC = YOUR_STEP_CALIBRATION_SPEC = YOUR_STEP_VALIDATION_SPEC = YOUR_STEP_TESTING_SPEC = None
    SPECS_AVAILABLE = False

class YourStepBuilder(StepBuilderBase):
    """Builder that supports multiple job type variants."""
    
    def __init__(self, config, **kwargs):
        # Get job type from config
        if not hasattr(config, 'job_type'):
            raise ValueError("config.job_type must be specified")
            
        job_type = config.job_type.lower()
        
        # Select appropriate specification based on job type
        spec = self._get_spec_for_job_type(job_type)
        
        super().__init__(config=config, spec=spec, **kwargs)
        self.config: YourStepConfig = config
        
        # Register with UnifiedRegistryManager (automatic discovery handles this)
        registry_manager = kwargs.get('registry_manager')
        if registry_manager is None:
            registry_manager = UnifiedRegistryManager()
        # Registration is handled automatically by the hybrid registry system
        # based on naming conventions and file location
    
    def _get_spec_for_job_type(self, job_type: str):
        """Get the appropriate specification for the given job type."""
        if job_type == "training" and YOUR_STEP_TRAINING_SPEC is not None:
            return YOUR_STEP_TRAINING_SPEC
        elif job_type == "calibration" and YOUR_STEP_CALIBRATION_SPEC is not None:
            return YOUR_STEP_CALIBRATION_SPEC
        elif job_type == "validation" and YOUR_STEP_VALIDATION_SPEC is not None:
            return YOUR_STEP_VALIDATION_SPEC
        elif job_type == "testing" and YOUR_STEP_TESTING_SPEC is not None:
            return YOUR_STEP_TESTING_SPEC
        else:
            # Try dynamic import as fallback
            try:
                module_path = f"..specs.your_step_{job_type}_spec"
                module = importlib.import_module(module_path, package=__package__)
                spec_var_name = f"YOUR_STEP_{job_type.upper()}_SPEC"
                if hasattr(module, spec_var_name):
                    return getattr(module, spec_var_name)
            except (ImportError, AttributeError):
                self.log_warning("Could not import specification for job type: %s", job_type)
                
        raise ValueError(f"No specification found for job type: {job_type}")
```

#### Real Example from TabularPreprocessingStepBuilder

Here's how the actual `TabularPreprocessingStepBuilder` handles job type variants:

```python
@register_builder()
class TabularPreprocessingStepBuilder(StepBuilderBase):
    def __init__(self, config, **kwargs):
        # Get job type from config
        if not hasattr(config, 'job_type'):
            raise ValueError("config.job_type must be specified")
            
        job_type = config.job_type.lower()
        
        # Get specification based on job type
        if job_type == "training" and TABULAR_PREPROCESSING_TRAINING_SPEC is not None:
            spec = TABULAR_PREPROCESSING_TRAINING_SPEC
        elif job_type == "calibration" and TABULAR_PREPROCESSING_CALIBRATION_SPEC is not None:
            spec = TABULAR_PREPROCESSING_CALIBRATION_SPEC
        elif job_type == "validation" and TABULAR_PREPROCESSING_VALIDATION_SPEC is not None:
            spec = TABULAR_PREPROCESSING_VALIDATION_SPEC
        elif job_type == "testing" and TABULAR_PREPROCESSING_TESTING_SPEC is not None:
            spec = TABULAR_PREPROCESSING_TESTING_SPEC
        else:
            raise ValueError(f"No specification found for job type: {job_type}")
                
        super().__init__(config=config, spec=spec, **kwargs)
        self.config: TabularPreprocessingConfig = config
```

#### Key Benefits of Job Type Variants

1. **Single Builder Class**: One builder handles multiple use cases
2. **Specification-Driven**: Different specifications define different behaviors
3. **Flexible Dependencies**: Each job type can have different input requirements
4. **Consistent Interface**: Same configuration class, different specifications

#### When to Use Job Type Variants

Use job type variants when:
- The same processing logic applies to different data types (training vs validation data)
- Different specifications are needed for different pipeline phases
- You want to reuse the same step builder for multiple purposes
- The core processing remains the same but inputs/outputs differ

#### Alternative: Separate Builders

If job types are very different, consider separate builders:
```python
@register_builder()
class YourStepTrainingBuilder(StepBuilderBase):
    """Dedicated builder for training job type."""
    pass

@register_builder()
class YourStepCalibrationBuilder(StepBuilderBase):
    """Dedicated builder for calibration job type."""
    pass
```

### 3. Implement Input/Output Handling

Use the specification-driven approach for inputs and outputs:

```python
def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    """Get inputs for the processor using the specification and contract."""
    # Use the built-in method that leverages the specification and contract
    return self._get_spec_driven_processor_inputs(inputs)

def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
    """Get outputs for the step using specification and contract."""
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
            destination = Join(on="/", values=[base_output_path, "your_step_type", self.config.job_type, logical_name])
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
```

### 4. Set Up Environment Variables

Configure environment variables required by your script:

```python
def _get_processor_env_vars(self) -> Dict[str, str]:
    """Get environment variables for the processor."""
    # Base environment variables
    env_vars = {
        # Map configuration parameters to environment variables
        "MODEL_TYPE": self.config.model_type,
        "NUM_EPOCHS": str(self.config.num_epochs),
        "LEARNING_RATE": str(self.config.learning_rate),
        "DEBUG_MODE": str(self.config.debug_mode).lower()
    }
    
    # Add job type specific variables if needed
    job_type = getattr(self.config, 'job_type', 'training')
    if job_type.lower() == "training":
        env_vars["TRAINING_MODE"] = "True"
    
    return env_vars
```

### 5. Create the Processor

Implement the processor creation logic:

```python
def _get_processor(self):
    """Create and return a SageMaker processor."""
    # For processing steps
    from sagemaker.processing import Processor
    
    processor = Processor(
        role=self.role,
        image_uri=self.config.get_image_uri(),
        instance_count=self.config.instance_count,
        instance_type=self.config.instance_type,
        volume_size_in_gb=self.config.volume_size_gb,
        max_runtime_in_seconds=self.config.max_runtime_seconds,
        sagemaker_session=self.sagemaker_session
    )
    
    return processor
    
    # For training steps
    # from sagemaker.estimator import Estimator
    # return Estimator(...)
```

### 6. Implement Step Creation

Override the `create_step` method to create the actual SageMaker step. Here's the proper implementation pattern based on the actual `PackageStepBuilder`:

```python
def create_step(self, **kwargs) -> ProcessingStep:
    """
    Create the processing step using the specification-driven approach.

    Args:
        **kwargs: Keyword arguments for configuring the step, including:
            - inputs: Input data sources keyed by logical name
            - outputs: Output destinations keyed by logical name
            - dependencies: Optional list of steps that this step depends on
            - enable_caching: A boolean indicating whether to cache the results of this step

    Returns:
        A configured sagemaker.workflow.steps.ProcessingStep instance.
    """
    self.log_info("Creating ProcessingStep...")

    # Extract parameters from kwargs
    inputs_raw = kwargs.get('inputs', {})
    outputs = kwargs.get('outputs', {})
    dependencies = kwargs.get('dependencies', [])
    enable_caching = kwargs.get('enable_caching', True)
    
    # Handle inputs - combine dependency-extracted and explicitly provided inputs
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
    
    # Add direct keyword arguments (e.g., DATA, METADATA from template)
    for key in ["DATA", "METADATA", "SIGNATURE"]:
        if key in kwargs and key not in inputs:
            inputs[key] = kwargs[key]
    
    # Create processor and get inputs/outputs using specification-driven methods
    processor = self._create_processor()
    proc_inputs = self._get_inputs(inputs)
    proc_outputs = self._get_outputs(outputs)
    job_args = self._get_job_arguments()

    # Get step name using standardized method with auto-detection
    step_name = self._get_step_name()
    
    # Get script path with portable path support and automatic fallback
    script_path = self.config.get_portable_script_path() or self.config.get_script_path()
    if not script_path and self.contract:
        script_path = self.contract.entry_point
    
    # Log which path type is being used for debugging
    self.log_info("Using script path: %s (portable: %s)", 
                 script_path, 
                 "yes" if self.config.get_portable_script_path() else "no")
    
    # Create the ProcessingStep using SageMaker SDK
    step = ProcessingStep(
        name=step_name,
        processor=processor,
        inputs=proc_inputs,
        outputs=proc_outputs,
        code=script_path,
        job_arguments=job_args,
        depends_on=dependencies,
        cache_config=self._get_cache_config(enable_caching)
    )
    
    # Attach specification to the step for future reference
    if hasattr(self, 'spec') and self.spec:
        setattr(step, '_spec', self.spec)
        
    self.log_info("Created ProcessingStep with name: %s", step.name)
    return step
```

#### Key Implementation Details

1. **Parameter Extraction**: Extract all parameters from `**kwargs` including inputs, outputs, dependencies, and caching settings

2. **Input Handling**: 
   - Extract inputs from dependencies using the dependency resolver
   - Merge with explicitly provided inputs
   - Handle special keyword arguments like DATA, METADATA, SIGNATURE

3. **Processor Creation**: Use the `_create_processor()` method to create the SageMaker processor

4. **Specification-Driven I/O**: Use `_get_inputs()` and `_get_outputs()` methods that leverage the step specification and script contract

5. **Step Configuration**: 
   - Use `_get_step_name()` for automatic step naming
   - Get script path from config or contract
   - Configure caching with `_get_cache_config()`

6. **ProcessingStep Creation**: Create the actual SageMaker `ProcessingStep` with all configured parameters

7. **Specification Attachment**: Attach the specification to the step for future reference

#### Error Handling

The implementation includes proper error handling:
- Graceful handling of dependency extraction failures
- Logging of warnings and information
- Fallback mechanisms for script paths

#### Flexibility

The method supports various input sources:
- Dependency-extracted inputs (automatic)
- Explicitly provided inputs (override)
- Direct keyword arguments (template support)

## Dependency Resolution

### Extracting Inputs from Dependencies

The `extract_inputs_from_dependencies` method is crucial for connecting steps:

```python
# In StepBuilderBase
def extract_inputs_from_dependencies(self, dependencies: List) -> Dict[str, str]:
    """Extract inputs from the given dependencies using the spec and resolver."""
    if not dependencies or not self.dependency_resolver:
        return {}
    
    # Use the dependency resolver to extract inputs based on the specification
    return self.dependency_resolver.resolve_dependencies(self.spec, dependencies)
```

This method:
1. Takes a list of dependency steps
2. Uses the dependency resolver to match outputs from those steps with dependencies in the specification
3. Returns a dictionary mapping logical input names to S3 URIs

### Understanding the Dependency Resolution Process

The dependency resolution process:

1. **Input Identification**: For each dependency in the specification, identify required inputs
2. **Output Matching**: For each dependency step, find outputs that match the required inputs
3. **Semantic Matching**: Use logical names, types, and semantic keywords to find the best matches
4. **URI Resolution**: Extract S3 URIs from the matched outputs
5. **Input Mapping**: Create a dictionary mapping logical input names to S3 URIs

## Processor Creation

Different step types require different SageMaker components:

### Processing Step

```python
from sagemaker.processing import Processor

processor = Processor(
    role=self.role,
    image_uri=self.config.get_image_uri(),
    instance_count=self.config.instance_count,
    instance_type=self.config.instance_type,
    volume_size_in_gb=self.config.volume_size_gb,
    max_runtime_in_seconds=self.config.max_runtime_seconds,
    sagemaker_session=self.sagemaker_session
)
```

### Training Step

```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    role=self.role,
    image_uri=self.config.get_image_uri(),
    instance_count=self.config.instance_count,
    instance_type=self.config.instance_type,
    volume_size=self.config.volume_size_gb,
    max_run=self.config.max_runtime_seconds,
    sagemaker_session=self.sagemaker_session,
    hyperparameters=self.config.get_hyperparameters()
)
```

### Model Step

```python
from sagemaker.model import Model

model = Model(
    image_uri=self.config.get_image_uri(),
    model_data=self.config.model_data,
    role=self.role,
    sagemaker_session=self.sagemaker_session
)
```

## Environment Variable Handling

Environment variables connect configuration parameters to script requirements. The `StepBuilderBase` class now includes a standardized `_get_environment_variables()` method that automatically extracts environment variables from the script contract:

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """
    Create environment variables for the processing job based on the script contract.
    
    This base implementation:
    1. Uses required_env_vars from the script contract
    2. Gets values from the config object
    3. Adds optional variables with defaults from the contract
    4. Can be overridden by child classes to add custom logic
    
    Returns:
        Dict[str, str]: Environment variables for the processing job
    """
    env_vars = {}
    
    if not hasattr(self, 'contract') or self.contract is None:
        self.log_warning("No script contract available for environment variable definition")
        return env_vars
    
    # Process required environment variables
    for env_var in self.contract.required_env_vars:
        # Convert from ENV_VAR_NAME format to config attribute style (env_var_name)
        config_attr = env_var.lower()
        
        # Try to get from config (direct attribute)
        if hasattr(self.config, config_attr):
            env_vars[env_var] = str(getattr(self.config, config_attr))
        # Try to get from config.hyperparameters
        elif hasattr(self.config, 'hyperparameters') and hasattr(self.config.hyperparameters, config_attr):
            env_vars[env_var] = str(getattr(self.config.hyperparameters, config_attr))
        else:
            self.log_warning(f"Required environment variable '{env_var}' not found in config")
    
    # Add optional environment variables with defaults
    for env_var, default_value in self.contract.optional_env_vars.items():
        # Convert from ENV_VAR_NAME format to config attribute style (env_var_name)
        config_attr = env_var.lower()
        
        # Try to get from config, fall back to default
        if hasattr(self.config, config_attr):
            env_vars[env_var] = str(getattr(self.config, config_attr))
        # Try to get from config.hyperparameters
        elif hasattr(self.config, 'hyperparameters') and hasattr(self.config.hyperparameters, config_attr):
            env_vars[env_var] = str(getattr(self.config.hyperparameters, config_attr))
        else:
            env_vars[env_var] = default_value
            self.log_debug(f"Using default value for optional environment variable '{env_var}': {default_value}")
    
    return env_vars
```

### How to Use the Standardized Method

In your step builder, you can use this method directly:

```python
def _get_processor(self):
    """Create and return a SageMaker processor."""
    from sagemaker.processing import ScriptProcessor
    
    processor = ScriptProcessor(
        role=self.role,
        image_uri=self.config.get_image_uri(),
        command=["python"],
        instance_count=self.config.instance_count,
        instance_type=self.config.instance_type,
        volume_size_in_gb=self.config.volume_size_gb,
        max_runtime_in_seconds=self.config.max_runtime_seconds,
        sagemaker_session=self.sagemaker_session,
        env=self._get_environment_variables()  # Use the standardized method
    )
    
    return processor
```

### Extending the Base Method

If you need additional environment variables beyond what's in the script contract:

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """Get environment variables for the processor."""
    # Get standard environment variables from contract
    env_vars = super()._get_environment_variables()
    
    # Add step-specific environment variables
    env_vars.update({
        "ADDITIONAL_PARAM": self.config.additional_param,
        "DEBUG_MODE": str(self.config.debug_mode).lower()
    })
    
    return env_vars
```

### Completely Overriding the Method

For cases where the standard approach doesn't fit:

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """Get custom environment variables for this specific step."""
    return {
        "CUSTOM_PARAM_1": self.config.custom_param1,
        "CUSTOM_PARAM_2": str(self.config.custom_param2),
        "JOB_TYPE": self.config.job_type
    }
```

Best practices:
1. Use the base implementation unless you have specific requirements
2. When extending, call `super()._get_environment_variables()` first
3. Handle type conversion (all environment variables are strings)
4. Log warnings for missing required variables
5. Provide sensible defaults for optional variables

## Standardized Job Name Generation

The `StepBuilderBase` class now includes a standardized `_generate_job_name()` method to create consistent job names with automatic step type detection and uniqueness:

```python
def _generate_job_name(self) -> str:
    """
    Generate a standardized job name for SageMaker processing/training jobs.
    
    This method automatically determines the step type from the class name
    using the centralized step name registry. It also adds a timestamp to 
    ensure uniqueness across executions.
        
    Returns:
        Sanitized job name suitable for SageMaker
    """
    import time
    
    # Determine step type from the class name
    class_name = self.__class__.__name__
    determined_step_type = None
    
    # Try to find a matching entry in the STEP_NAMES registry
    for canonical_name, info in self.STEP_NAMES.items():
        if info["builder_step_name"] == class_name or class_name.startswith(info["builder_step_name"]):
            determined_step_type = canonical_name
            break
    
    # If no match found, fall back to class name with "StepBuilder" removed
    if determined_step_type is None:
        if class_name.endswith("StepBuilder"):
            determined_step_type = class_name[:-11]  # Remove "StepBuilder"
        else:
            determined_step_type = class_name
            
    # Generate a timestamp for uniqueness (unix timestamp in seconds)
    timestamp = int(time.time())
    
    # Build the job name
    if hasattr(self.config, 'job_type') and self.config.job_type:
        job_name = f"{determined_step_type}-{self.config.job_type.capitalize()}-{timestamp}"
    else:
        job_name = f"{determined_step_type}-{timestamp}"
        
    # Sanitize and return
    return self._sanitize_name_for_sagemaker(job_name)
```

The updated method provides several key improvements:

1. **Automatic Step Type Detection**: The method can now automatically determine the step type from the class name using the centralized step name registry, eliminating the need to pass it as a parameter
2. **Guaranteed Uniqueness**: A timestamp is added to ensure unique job names across multiple executions
3. **Registry-Based Naming**: Uses the centralized `STEP_NAMES` registry to maintain consistent naming conventions
4. **Backward Compatibility**: Still supports explicitly passing a step type for legacy code

### How to Use the Standardized Method

In your step builder's `create_step()` method, you can now call the method without any parameters:

```python
def create_step(self, **kwargs) -> ProcessingStep:
    """Create the processing step."""
    # [...]
    
    # Create and return the step
    step = processor.run(
        inputs=processing_inputs,
        outputs=processing_outputs,
        arguments=script_args,
        job_name=self._generate_job_name(),  # No parameter needed!
        wait=False,
        cache_config=cache_config
    )
    
    return step
```

This approach ensures that all jobs created by your step builders will have consistent and unique naming, making them easier to identify in the SageMaker console and avoiding job name conflicts.

## Builder Examples

### Processing Step Builder

```python
class TabularPreprocessingStepBuilder(StepBuilderBase):
    """Builder for TabularPreprocessing step."""
    
    def __init__(self, config, **kwargs):
        job_type = getattr(config, 'job_type', 'training')
        
        # Select appropriate spec based on job type
        if job_type.lower() == "calibration":
            spec = PREPROCESSING_CALIBRATION_SPEC
        else:
            spec = PREPROCESSING_TRAINING_SPEC
            
        super().__init__(config=config, spec=spec, **kwargs)
    
    def _get_processor(self):
        """Create and return a SageMaker processor."""
        from sagemaker.processing import ScriptProcessor
        
        processor = ScriptProcessor(
            role=self.role,
            image_uri=self.config.get_image_uri(),
            command=["python"],
            instance_count=self.config.instance_count,
            instance_type=self.config.instance_type,
            volume_size_in_gb=self.config.volume_size_gb,
            max_runtime_in_seconds=self.config.max_runtime_seconds,
            sagemaker_session=self.sagemaker_session
        )
        
        return processor
    
    def _get_environment_variables(self):
        """Get environment variables for the processor."""
        # Get base environment variables from script contract
        env_vars = super()._get_environment_variables()
        
        # Add or override with specific variables for this step
        env_vars.update({
            "FEATURE_COLUMNS": ",".join(self.config.feature_columns),
            "DEBUG_MODE": str(self.config.debug_mode).lower()
        })
        
        return env_vars
```

### Training Step Builder

```python
class XGBoostTrainingStepBuilder(StepBuilderBase):
    """Builder for XGBoost training step."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config=config, spec=XGBOOST_TRAINING_SPEC, **kwargs)
    
    def _get_hyperparameters(self):
        """Get hyperparameters for the estimator."""
        return {
            "max_depth": str(self.config.max_depth),
            "eta": str(self.config.learning_rate),
            "gamma": str(self.config.gamma),
            "min_child_weight": str(self.config.min_child_weight),
            "subsample": str(self.config.subsample),
            "silent": "0",
            "objective": self.config.objective,
            "num_round": str(self.config.num_round)
        }
    
    def create_step(self, **kwargs):
        """Create the training step."""
        from sagemaker.xgboost.estimator import XGBoost
        from sagemaker.inputs import TrainingInput
        from sagemaker.workflow.steps import TrainingStep
        
        # Extract inputs from dependencies
        dependencies = kwargs.get('dependencies', [])
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        
        # Create training inputs
        inputs = {}
        for channel_name, s3_uri in extracted_inputs.items():
            inputs[channel_name] = TrainingInput(
                s3_data=s3_uri,
                content_type="csv"
            )
        
        # Create estimator
        estimator = XGBoost(
            entry_point=self.config.get_script_path(),
            hyperparameters=self._get_hyperparameters(),
            role=self.role,
            instance_count=self.config.instance_count,
            instance_type=self.config.instance_type,
            volume_size=self.config.volume_size_gb,
            max_run=self.config.max_runtime_seconds,
            sagemaker_session=self.sagemaker_session,
            framework_version=self.config.framework_version
        )
        
        # Create and return the step
        step_name = kwargs.get('step_name', 'XGBoostTraining')
        step = TrainingStep(
            name=step_name,
            estimator=estimator,
            inputs=inputs
        )
        
        # Store specification in step for future reference
        setattr(step, '_spec', self.spec)
        
        return step
```

## Step Builder Registration

> **📖 For comprehensive information about the step builder registry system, see:**
> - **[Step Builder Registry Guide](step_builder_registry_guide.md)** - Complete guide to the registry architecture and implementation
> - **[Step Builder Registry Usage](step_builder_registry_usage.md)** - Practical usage examples and best practices

### Automatic Registration with UnifiedRegistryManager

With the modern hybrid registry system, step registration is handled automatically through the UnifiedRegistryManager. Step builders are discovered and registered based on naming conventions and file location:

```python
from ...registry.hybrid.manager import UnifiedRegistryManager

class YourNewStepBuilder(StepBuilderBase):
    """Builder for your new step type."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        
        # Register with UnifiedRegistryManager (automatic discovery handles this)
        registry_manager = kwargs.get('registry_manager')
        if registry_manager is None:
            registry_manager = UnifiedRegistryManager()
        # Registration is handled automatically by the hybrid registry system
        # based on naming conventions and file location
```

The UnifiedRegistryManager provides:

1. **Automatic Discovery**: Discovers step builders based on naming conventions
2. **Workspace-Aware Registration**: Supports both main and isolated workspace contexts
3. **Hybrid Registry Support**: Integrates with both core and local registries
4. **Caching and Performance**: Efficient caching for improved performance
5. **Backward Compatibility**: Maintains compatibility with legacy registration patterns

### Manual Registration (For Custom Cases)

If you need explicit control over registration, use the registry's validation-enabled registration:

```python
from cursus.registry.step_names import add_new_step_with_validation

# Register your step with validation
warnings = add_new_step_with_validation(
    step_name="YourNewStep",
    config_class="YourNewStepConfig", 
    builder_name="YourNewStepBuilder",
    sagemaker_type="Processing",  # Based on create_step() return type
    description="Description of your new step",
    validation_mode="warn",  # Options: "warn", "strict", "auto_correct"
    workspace_id=None  # Use current workspace context
)
```

### Workspace-Specific Registration

For isolated workspace development:

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry manager with workspace context
registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")

# Register step in specific workspace
registry.register_step_definition(
    "YourNewStep",
    {
        "config_class": "YourNewStepConfig",
        "builder_step_name": "YourNewStepBuilder", 
        "spec_type": "YourNewStep",
        "sagemaker_step_type": "Processing",
        "description": "Description of your new step"
    }
)
```

### Registration Requirements

For successful registration, ensure:

1. **Class Name Convention**: Your builder class should end with `StepBuilder` (e.g., `TabularPreprocessingStepBuilder`)
2. **Step Names Registry**: Add your step to the `STEP_NAMES` registry in `src/cursus/registry/step_names.py`
3. **Module Import**: Your builder module should be importable from the builders package
4. **Base Class**: Your builder must extend `StepBuilderBase`

### Adding to Step Names Registry

Add your step to the centralized registry:

```python
# In src/cursus/registry/step_names.py
STEP_NAMES = {
    # ... existing entries ...
    "YourStepType": {
        "builder_step_name": "YourStepTypeStepBuilder",
        "config_class": "YourStepTypeConfig",
        "step_name_template": "your-step-{job_type}",
        "description": "Description of your step type"
    }
}
```

### Module Structure for New Step Builder

When creating a new step builder, follow this structure:

```
src/cursus/steps/builders/
├── __init__.py                          # Import and export your builder
├── builder_your_step_type.py           # Your step builder implementation
└── ...
```

Update `__init__.py` to include your builder:

```python
# In src/cursus/steps/builders/__init__.py
from .builder_your_step_type import YourStepTypeStepBuilder

__all__ = [
    # ... existing builders ...
    "YourStepTypeStepBuilder",
]
```

### Verification

To verify your step builder is properly registered:

```python
from cursus.steps.registry.builder_registry import get_global_registry

registry = get_global_registry()

# Check if your step type is supported
print(registry.is_step_type_supported("YourStepType"))

# List all supported step types
print(registry.list_supported_step_types())

# Get validation results
validation_results = registry.validate_registry()
print(validation_results)
```

## Best Practices

1. **Always Use the Registration Decorator**: Use `@register_builder()` on every step builder class
2. **Follow Naming Conventions**: End class names with `StepBuilder` and use PascalCase
3. **Update Central Registry**: Add entries to `STEP_NAMES` registry for proper integration
4. **Use Specification-Driven Methods**: Leverage built-in methods for input/output handling
5. **Handle Job Type Variants**: Implement proper selection of specifications for different job types
6. **Validate Configuration**: Add validation to ensure configuration is complete and valid
7. **Provide Meaningful Errors**: Include helpful error messages when validation fails
8. **Log Key Information**: Log important information for debugging
9. **Follow SageMaker Best Practices**: Adhere to SageMaker's conventions for resource creation
10. **Use Strong Typing**: Add type hints to improve code quality
11. **Test Edge Cases**: Write unit tests for various configuration scenarios
12. **Export from Module**: Always add new builders to the `__init__.py` exports

By following these guidelines, your step builders will provide a robust implementation that connects specifications to SageMaker steps while maintaining separation of concerns and proper integration with the pipeline system.
