# Property Reference

## Overview

The PropertyReference class is a critical component that bridges between definition-time specifications and runtime property references in the SageMaker pipeline context. It provides a way to lazily evaluate references to properties of pipeline steps, enabling the automatic wiring of pipeline steps based on specifications rather than manual coding.

## Class Definition

```python
class PropertyReference(BaseModel):
    """Lazy evaluation reference bridging definition-time and runtime."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    step_name: str = Field(
        ...,
        description="Name of the step that produces this output",
        min_length=1
    )
    output_spec: OutputSpec = Field(
        ...,
        description="Output specification for the referenced output"
    )
```

## Key Design Choices

### 1. Definition-Time vs. Runtime Bridging

The primary purpose of PropertyReference is to bridge between two worlds:

1. **Definition Time**: When the pipeline is being constructed and only specifications are available
2. **Runtime**: When the pipeline is executing and actual SageMaker step instances exist

This bridging enables the core capability of the specification-driven dependency resolution system: resolving dependencies based on specifications, but using them to create actual SageMaker property references at runtime.

### 2. Pydantic Integration

PropertyReference is implemented as a Pydantic model, which provides several benefits:

- **Validation**: Automatic validation of inputs at creation time
- **Serialization**: Easy serialization/deserialization for storage or transmission
- **Type Safety**: Strong typing for better IDE support and fewer runtime errors
- **Field Documentation**: Self-documenting field descriptions

### 3. Property Path Parsing

The class contains sophisticated property path parsing logic to handle the complex structure of SageMaker property references:

```python
def _parse_property_path(self, path: str) -> List[Union[str, Tuple[str, str]]]:
    """
    Parse a property path into a sequence of access operations.
    
    This method handles various property path formats, including:
    - Regular attribute access: "properties.ProcessingOutputConfig"
    - Dictionary access: "Outputs['DATA']"
    - Combined access: "properties.ProcessingOutputConfig.Outputs['DATA']"
    
    Args:
        path: Property path as a string
        
    Returns:
        List of access operations, where each operation is either:
        - A string for attribute access
        - A tuple (attr_name, key) for dictionary access
    """
    # Remove "properties." prefix if present
    if path.startswith("properties."):
        path = path[11:]  # Remove "properties."
    
    result = []
    
    # Improved pattern to match dictionary access like: 
    # Outputs['DATA'] or Outputs["DATA"] or Outputs[0]
    dict_pattern = re.compile(r'(\w+)\[([\'"]?)([^\]\'\"]+)\2\]')
    
    # Split by dots first, but preserve quoted parts
    parts = []
    current = ""
    in_brackets = False
    bracket_depth = 0
    
    for char in path:
        if char == '.' and not in_brackets:
            if current:
                parts.append(current)
                current = ""
        elif char == '[':
            in_brackets = True
            bracket_depth += 1
            current += char
        elif char == ']':
            bracket_depth -= 1
            if bracket_depth == 0:
                in_brackets = False
            current += char
        else:
            current += char
    
    if current:
        parts.append(current)
    
    # Process each part
    for part in parts:
        # Check if this part contains dictionary access
        dict_match = dict_pattern.match(part)
        if dict_match:
            # Extract the attribute name and key
            attr_name = dict_match.group(1)
            quote_type = dict_match.group(2)  # This will be ' or " or empty
            key = dict_match.group(3)
            
            # Handle numeric indices
            if not quote_type and key.isdigit():
                key = int(key)
                
            # Add a tuple for dictionary access
            result.append((attr_name, key))
        else:
            # Regular attribute access
            result.append(part)
    
    return result
```

This parsing logic is capable of handling complex nested property references with both attribute access (dot notation) and dictionary/list access (bracket notation).

### 4. Runtime Property Navigation

Once the property path is parsed, the class provides a method to navigate through the actual objects at runtime:

```python
def _get_property_value(self, obj: Any, path_parts: List[Union[str, Tuple[str, str]]]) -> Any:
    """
    Navigate through the property path to get the final value.
    
    Args:
        obj: The object to start navigation from
        path_parts: List of path parts from _parse_property_path
        
    Returns:
        The value at the end of the property path
        
    Raises:
        AttributeError: If any part of the path is invalid
        ValueError: If a path part has an invalid format
    """
    current_obj = obj
    
    # Navigate through each part of the path
    for part in path_parts:
        if isinstance(part, str):
            # Regular attribute access
            current_obj = getattr(current_obj, part)
        elif isinstance(part, tuple) and len(part) == 2:
            # Dictionary access with [key]
            attr_name, key = part
            if attr_name:  # If there's an attribute before the bracket
                current_obj = getattr(current_obj, attr_name)
            # Handle the key access
            if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):  
                # Array index - convert string digits to int if needed
                idx = key if isinstance(key, int) else int(key)
                current_obj = current_obj[idx]
            else:  # Dictionary key
                current_obj = current_obj[key]
        else:
            raise ValueError(f"Invalid path part: {part}")
    
    return current_obj
```

This method handles both attribute access and dictionary/list access with proper error handling.

### 5. SageMaker Integration

The class provides methods to convert property references to SageMaker-compatible formats:

```python
def to_sagemaker_property(self) -> Dict[str, str]:
    """Convert to SageMaker Properties dictionary format at pipeline definition time."""
    # Get the property path without 'properties.' prefix
    property_path = self.output_spec.property_path
    if property_path.startswith('properties.'):
        property_path = property_path[11:]
    
    return {"Get": f"Steps.{self.step_name}.{property_path}"}

def to_runtime_property(self, step_instances: Dict[str, Any]) -> Any:
    """
    Create an actual SageMaker property reference using step instances.
    
    This method navigates the property path to create a proper SageMaker
    Properties object that can be used at runtime.
    
    Args:
        step_instances: Dictionary mapping step names to step instances
        
    Returns:
        SageMaker Properties object for the referenced property
        
    Raises:
        ValueError: If the step is not found or property path is invalid
        AttributeError: If any part of the property path is invalid
    """
    if self.step_name not in step_instances:
        raise ValueError(f"Step '{self.step_name}' not found in step instances. Available steps: {list(step_instances.keys())}")
    
    # Get the step instance
    step_instance = step_instances[self.step_name]
    
    # Parse and navigate the property path
    path_parts = self._parse_property_path(self.output_spec.property_path)
    
    # Use helper method to navigate property path
    return self._get_property_value(step_instance.properties, path_parts)
```

These methods support both pipeline definition time (using a dictionary format) and runtime (using actual SageMaker Properties objects).

## Usage Patterns

### Creation by Dependency Resolver

PropertyReference objects are typically created by the dependency resolver during the resolution process:

```python
# In UnifiedDependencyResolver._resolve_single_dependency:
if confidence > 0.5:  # Threshold for viable candidates
    prop_ref = PropertyReference(
        step_name=provider_step,
        output_spec=output_spec
    )
    candidates.append((prop_ref, confidence, provider_step, output_name))
```

### Usage in PipelineAssembler

The PipelineAssembler uses PropertyReference objects to create actual SageMaker property references:

```python
# In PipelineAssembler._instantiate_step:
if output_spec:
    try:
        # Create a PropertyReference object
        prop_ref = PropertyReference(
            step_name=src_step,
            output_spec=output_spec
        )
        
        # Use the enhanced to_runtime_property method to get a SageMaker Properties object
        runtime_prop = prop_ref.to_runtime_property(self.step_instances)
        inputs[input_name] = runtime_prop
    except Exception as e:
        # Fallback handling...
```

## Error Handling and Fallbacks

The PropertyReference implementation includes robust error handling:

```python
# In PipelineAssembler._instantiate_step:
try:
    # Create a PropertyReference object
    prop_ref = PropertyReference(
        step_name=src_step,
        output_spec=output_spec
    )
    
    # Use the enhanced to_runtime_property method to get a SageMaker Properties object
    runtime_prop = prop_ref.to_runtime_property(self.step_instances)
    inputs[input_name] = runtime_prop
    
    logger.debug(f"Created runtime property reference for {step_name}.{input_name} -> {src_step}.{output_spec.property_path}")
except Exception as e:
    # Log the error and fall back to a safe string
    logger.warning(f"Error creating runtime property reference: {str(e)}")
    s3_uri = f"s3://pipeline-reference/{src_step}/{src_output}"
    inputs[input_name] = s3_uri
    logger.warning(f"Using S3 URI fallback: {s3_uri}")
```

This ensures that even if property resolution fails, the pipeline can still be created with placeholder values.

## String Representation

PropertyReference objects provide both string and representation methods for debugging:

```python
def __str__(self) -> str:
    return f"{self.step_name}.{self.output_spec.logical_name}"

def __repr__(self) -> str:
    return f"PropertyReference(step='{self.step_name}', output='{self.output_spec.logical_name}')"
```

## Example Usage

```python
# Create an output specification
model_output = OutputSpec(
    logical_name="model_output",
    output_type=DependencyType.MODEL_ARTIFACTS,
    property_path="properties.ModelArtifacts.S3ModelArtifacts",
    data_type="S3Uri"
)

# Create a property reference
prop_ref = PropertyReference(
    step_name="training_step",
    output_spec=model_output
)

# At definition time, get SageMaker property dictionary
sagemaker_prop = prop_ref.to_sagemaker_property()
print(sagemaker_prop)  # {"Get": "Steps.training_step.ModelArtifacts.S3ModelArtifacts"}

# At runtime, with step instances available
step_instances = {"training_step": training_step_instance}
runtime_prop = prop_ref.to_runtime_property(step_instances)
# This returns the actual SageMaker property object
```

## Integration with Step Specifications

The PropertyReference class integrates with the step specification system through the OutputSpec class:

```python
# In the StepSpecification class:
def get_output_by_name_or_alias(self, name: str) -> Optional[OutputSpec]:
    """
    Get output specification by logical name or alias.
    
    Args:
        name: The logical name or alias to search for
        
    Returns:
        OutputSpec if found, None otherwise
    """
    # First try exact logical name match
    if name in self.outputs:
        return self.outputs[name]
    
    # Then search through aliases (case-insensitive)
    name_lower = name.lower()
    for output_spec in self.outputs.values():
        for alias in output_spec.aliases:
            if alias.lower() == name_lower:
                return output_spec
    
    return None
```

This allows property references to be created from both logical names and aliases.

## Benefits

The PropertyReference design provides several key benefits:

1. **Abstraction**: Hides the complexity of SageMaker property references
2. **Lazy Evaluation**: Allows references to be created before the actual step instances exist
3. **Validation**: Ensures property paths are valid before runtime
4. **Error Handling**: Provides robust error handling and fallbacks
5. **Debugging**: Offers clear string representations for debugging
6. **Integration**: Works seamlessly with the dependency resolution system

## Related Components

- [Base Specifications](base_specifications.md): Defines OutputSpec and StepSpecification
- [Dependency Resolver](dependency_resolver.md): Creates PropertyReference objects during resolution
- [Pipeline Assembler](../pipeline_builder/pipeline_assembler.md): Uses PropertyReference objects for step wiring

## SageMaker Step Properties Reference

The following is a comprehensive list of SageMaker step properties organized by step type. This reference is valuable for understanding the property paths that need to be parsed by the PropertyReference class.

### Training Step Properties

```
properties.ModelArtifacts.S3ModelArtifacts       # Path to model artifacts
properties.TrainingJobName                       # Name of the training job
properties.TrainingJobArn                        # ARN of the training job
properties.TrainingJobStatus                     # Status of the training job
properties.SecondaryStatus                       # Secondary status details
properties.HyperParameters                       # Dictionary of hyperparameters
properties.HyperParameters['param_name']         # Access specific hyperparameter
properties.OutputDataConfig.S3OutputPath         # S3 path for output data
properties.ResourceConfig.InstanceType           # Instance type used
properties.ResourceConfig.InstanceCount          # Number of instances used
properties.StoppingCondition.MaxRuntimeInSeconds # Maximum runtime
```

### Processing Step Properties

```
properties.ProcessingInputs[0].S3Input.S3Uri     # First input S3 location
properties.ProcessingInputs['named'].S3Input.S3Uri  # Named input S3 location
properties.ProcessingOutputConfig.Outputs['output_name'].S3Output.S3Uri  # Output location
properties.ProcessingJobName                     # Name of the processing job
properties.ProcessingJobArn                      # ARN of the processing job
properties.ProcessingJobStatus                   # Status of the processing job
properties.ProcessingResources.ClusterConfig.InstanceType  # Instance type
properties.ProcessingResources.ClusterConfig.InstanceCount  # Instance count
properties.AppSpecification.ImageUri             # Docker image URI
properties.AppSpecification.ContainerArguments[0]  # First container argument
properties.RoleArn                               # IAM role ARN
```

### Model Step Properties

```
properties.ModelName                             # Name of the model
properties.ModelArn                              # ARN of the model
properties.PrimaryContainer.Image                # Primary container image URI
properties.PrimaryContainer.ModelDataUrl         # Model data S3 location
properties.PrimaryContainer.Environment['ENV_VAR']  # Environment variable
properties.Containers[0].Image                   # First container image (for multi-container)
properties.Containers[0].ModelDataUrl            # First container model data URL
properties.ExecutionRoleArn                      # IAM role ARN
```

### Transform Step Properties

```
properties.TransformJobName                      # Name of the transform job
properties.TransformJobArn                       # ARN of the transform job
properties.TransformJobStatus                    # Status of the transform job
properties.ModelName                             # Name of the model used
properties.TransformOutput.S3OutputPath          # Output S3 location
properties.TransformInput.DataSource.S3DataSource.S3Uri  # Input data source
properties.TransformResources.InstanceType       # Instance type
properties.TransformResources.InstanceCount      # Instance count
properties.BatchStrategy                         # Batch transform strategy
```

### Evaluation Step Properties

```
properties.ProcessingJobName                     # Name of the evaluation job
properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri  # Evaluation output
properties.ProcessingOutputConfig.Outputs['metrics'].S3Output.S3Uri  # Metrics output
properties.Statistics['accuracy']                # Accuracy metric
properties.Statistics['precision']               # Precision metric
properties.Statistics['recall']                  # Recall metric
properties.Statistics['f1']                      # F1 score
properties.Statistics['confusion_matrix'][0][0]  # Confusion matrix entry
properties.Statistics['roc_auc']                 # ROC AUC score
```

### Hyperparameter Tuning Step Properties

```
properties.BestTrainingJob.TrainingJobName       # Best training job name
properties.BestTrainingJob.TrainingJobArn        # Best training job ARN
properties.BestTrainingJob.ModelArtifacts.S3ModelArtifacts  # Best model artifacts
properties.BestTrainingJob.FinalHyperParameterTuningJobObjectiveMetric.Value  # Best metric value
properties.TrainingJobSummaries[0].TrainingJobName  # First training job name
properties.TrainingJobSummaries[0].TrainingJobArn   # First training job ARN
properties.TrainingJobSummaries[0].FinalHyperParameterTuningJobObjectiveMetric.Value  # First job metric
properties.TrainingJobSummaries[0].HyperParameters['param_name']  # Hyperparameter of first job
properties.TrainingJobSummaries.Length              # Number of training jobs
properties.HyperParameterTuningJobName           # Tuning job name
properties.HyperParameterTuningJobStatus         # Tuning job status
```

### Model Registry Step Properties

```
properties.ModelPackageName                      # Model package name
properties.ModelPackageArn                       # Model package ARN
properties.ModelPackageStatus                    # Status of model package
properties.ModelApprovalStatus                   # Approval status
properties.CreationTime                          # Creation timestamp
properties.ModelMetrics.ModelQuality.Statistics.RegressionMetrics.RSquared  # R-squared value
properties.ModelMetrics.ModelQuality.Statistics.ClassificationMetrics.Accuracy  # Accuracy metric
properties.InferenceSpecification.Containers[0].Image  # Container image
properties.InferenceSpecification.SupportedContentTypes[0]  # Supported content type
```

### Clarify Step Properties

```
properties.ProcessingJobName                     # Name of the clarify job
properties.ProcessingOutputConfig.Outputs['analysis'].S3Output.S3Uri  # Analysis output
properties.ProcessingOutputConfig.Outputs['shap_values'].S3Output.S3Uri  # SHAP values
properties.AnalysisResult.BiasMetrics['class_probability_difference_in_labels']  # Bias metric
properties.AnalysisResult.FeatureImportance[0].FeatureName  # Most important feature name
properties.AnalysisResult.FeatureImportance[0].Importance   # Importance score
properties.ExplainabilityReport.S3Uri           # Explainability report location
properties.DataConfig.S3Uri                     # Input data location
```

### Conditional Step Properties

```
properties.Outcome                               # Step outcome (true/false)
properties.ConditionOutcome                      # Detailed condition outcome
properties.IfSteps[0].StepName                   # Name of first 'if' step 
properties.IfSteps.Length                        # Number of 'if' steps
properties.ElseSteps[0].StepName                 # Name of first 'else' step
properties.ElseSteps.Length                      # Number of 'else' steps
```

### Lambda Step Properties

```
properties.FunctionArn                           # Lambda function ARN
properties.Payload                               # Lambda function payload
properties.Payload['output_key']                 # Access specific output key
properties.OutputFile                            # Output file S3 location
properties.StatusCode                            # Lambda execution status code
properties.ExecutedVersion                       # Lambda version executed
```

### Array Indexing in SageMaker Properties

Several SageMaker step properties use array indexing patterns:

1. **Accessing array elements with numeric indices**:
   ```
   properties.TrainingJobSummaries[0].TrainingJobName
   properties.ProcessingInputs[0].S3Input.S3Uri
   properties.Containers[0].Image
   properties.AppSpecification.ContainerArguments[0]
   properties.Statistics['confusion_matrix'][0][0]
   properties.AnalysisResult.FeatureImportance[0].FeatureName
   properties.IfSteps[0].StepName
   ```

2. **Mixed patterns with array indexing and subsequent property access**:
   ```
   properties.TrainingJobSummaries[0].FinalHyperParameterTuningJobObjectiveMetric.Value
   properties.AnalysisResult.FeatureImportance[0].FeatureName
   ```

3. **Multi-dimensional array access**:
   ```
   properties.Statistics['confusion_matrix'][0][0]
   ```

These examples confirm that property path parsing must support:
- Dictionary access with string keys (e.g., `Outputs['output_name']`)
- Array indexing with numeric indices (e.g., `Containers[0]`)
- Mixed patterns with both types of access (e.g., `TrainingJobSummaries[0].TrainingJobName`)
- Multiple levels of nesting (e.g., `BestTrainingJob.FinalHyperParameterTuningJobObjectiveMetric.Value`)
