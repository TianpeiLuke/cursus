# Common Pitfalls to Avoid

When implementing new pipeline steps, there are several common mistakes that developers often make. This document outlines these pitfalls and provides guidance on how to avoid them.

## Script Implementation Pitfalls

### 1. Hardcoded Paths

**Problem**: Hardcoding container paths in your script makes it inflexible and breaks the contract alignment.

```python
# WRONG ❌
input_path = "/opt/ml/processing/input/data"  # Hardcoded path
output_path = "/opt/ml/processing/output/results"  # Hardcoded path
```

**Solution**: Use the script contract to get paths. See [Script Contract Development](script_contract.md) for details, [Path Handling Best Practices](best_practices.md#path-handling) for examples, and [Script Development Guide](script_development_guide.md) for comprehensive script implementation patterns.

```python
# CORRECT ✅
def get_script_contract():
    from ..contracts.your_script_contract import YOUR_SCRIPT_CONTRACT
    return YOUR_SCRIPT_CONTRACT

contract = get_script_contract()
input_path = contract.expected_input_paths["input_data"]
output_path = contract.expected_output_paths["output_data"]
```

### 2. Missing Environment Variable Error Handling

**Problem**: Not handling missing environment variables can cause cryptic runtime errors.

```python
# WRONG ❌
learning_rate = float(os.environ["LEARNING_RATE"])  # Will fail if not set
```

**Solution**: Validate environment variables or provide defaults. For best practices on environment variable handling, see [Environment Variable Documentation](best_practices.md#environment-variable-documentation).

```python
# CORRECT ✅
try:
    learning_rate = float(os.environ["LEARNING_RATE"])
except (KeyError, ValueError) as e:
    raise ValueError(f"Required environment variable LEARNING_RATE is missing or invalid: {str(e)}")

# Alternative with default
learning_rate = float(os.environ.get("LEARNING_RATE", "0.01"))
```

### 3. Directory vs. File Path Confusion

**Problem**: Treating a directory path as a file or vice versa leads to errors.

```python
# WRONG ❌
# If output_path is a directory, this will fail
with open(output_path, 'w') as f:  # Trying to open a directory as a file
    json.dump(data, f)
```

**Solution**: Be explicit about directory vs. file paths. See [Path Handling Best Practices](best_practices.md#path-handling) for recommended patterns.

```python
# CORRECT ✅
output_dir = contract.expected_output_paths["output_data"]
output_file = os.path.join(output_dir, "results.json")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(data, f)
```

### 4. Insufficient Error Handling

**Problem**: Minimal error handling makes debugging difficult.

```python
# WRONG ❌
def process_data():
    data = pd.read_csv(input_path)
    result = data.groupby('category').sum()
    result.to_csv(output_path)
```

**Solution**: Add comprehensive error handling and logging. See [Error Handling Best Practices](best_practices.md#error-handling) for detailed examples and patterns.

```python
# CORRECT ✅
def process_data():
    try:
        logger.info(f"Reading data from {input_path}")
        data = pd.read_csv(input_path)
        
        if data.empty:
            raise ValueError("Input data is empty")
        
        logger.info(f"Processing data with {len(data)} records")
        result = data.groupby('category').sum()
        
        logger.info(f"Writing results to {output_path}")
        result.to_csv(output_path)
        
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {input_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
```

### 5. Not Creating Output Directories

**Problem**: Failing to create output directories before writing files.

```python
# WRONG ❌
with open(os.path.join(output_path, "output.csv"), 'w') as f:  # May fail if directory doesn't exist
    csv_writer = csv.writer(f)
    csv_writer.writerows(data)
```

**Solution**: Always create directories before writing files. See [Path Handling Best Practices](best_practices.md#path-handling) for more examples.

```python
# CORRECT ✅
output_file = os.path.join(output_path, "output.csv")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(data)
```

## Contract Development Pitfalls

### 1. Inconsistent Logical Names

**Problem**: Using different logical names in contract vs. specification leads to misalignment.

```python
# WRONG ❌
# In script contract
expected_input_paths={
    "data": "/opt/ml/processing/input/data"  # Logical name: "data"
}

# In step specification
dependencies={
    "input_data": DependencySpec(...)  # Different logical name: "input_data"
}
```

**Solution**: Use consistent logical names across contract and specification. See [Alignment Rules](alignment_rules.md) for detailed guidance, [Logical Name Clarity](best_practices.md#logical-name-clarity) for best practices, and [Level 1: Script Contract Alignment Design](../1_design/level1_script_contract_alignment_design.md) for alignment validation patterns.

```python
# CORRECT ✅
# In script contract
expected_input_paths={
    "input_data": "/opt/ml/processing/input/data"  # Logical name: "input_data"
}

# In step specification
dependencies={
    "input_data": DependencySpec(...)  # Same logical name: "input_data"
}
```

### 2. Missing Required Environment Variables

**Problem**: Not declaring all required environment variables in the contract.

```python
# WRONG ❌
# Script uses LEARNING_RATE but contract doesn't declare it
YOUR_SCRIPT_CONTRACT = ScriptContract(
    entry_point="your_script.py",
    expected_input_paths={...},
    expected_output_paths={...},
    required_env_vars=[
        # Missing LEARNING_RATE
    ],
    # ...
)
```

**Solution**: Declare all environment variables used by the script.

```python
# CORRECT ✅
YOUR_SCRIPT_CONTRACT = ScriptContract(
    entry_point="your_script.py",
    expected_input_paths={...},
    expected_output_paths={...},
    required_env_vars=[
        "LEARNING_RATE",
        "NUM_EPOCHS",
        # All required env vars
    ],
    optional_env_vars={
        "DEBUG_MODE": "False"
        # Optional env vars with defaults
    }
    # ...
)
```

### 3. Wrong Container Path Conventions

**Problem**: Using non-standard paths or incorrect path formats.

```python
# WRONG ❌
expected_input_paths={
    "data": "/data/input",  # Non-standard path
    "config": "/etc/config"  # Non-standard path
}
```

**Solution**: Follow SageMaker path conventions.

```python
# CORRECT ✅
expected_input_paths={
    "data": "/opt/ml/processing/input/data",  # Standard processing input path
    "config": "/opt/ml/processing/input/config"  # Standard processing input path
}

# For training scripts
expected_input_paths={
    "training": "/opt/ml/input/data/training",  # Standard training input path
    "validation": "/opt/ml/input/data/validation"  # Standard training input path
}
```

### 4. Overlapping Input/Output Paths

**Problem**: Using the same path for both input and output can cause conflicts.

```python
# WRONG ❌
expected_input_paths={
    "data": "/opt/ml/processing/data"  # Same base path as output
}

expected_output_paths={
    "results": "/opt/ml/processing/data/results"  # Subpath of input path
}
```

**Solution**: Use separate paths for inputs and outputs.

```python
# CORRECT ✅
expected_input_paths={
    "data": "/opt/ml/processing/input/data"  # Standard input path
}

expected_output_paths={
    "results": "/opt/ml/processing/output/results"  # Standard output path
}
```

### 5. File vs. Directory Path Confusion in Contracts

**Problem**: SageMaker creates directories at paths specified in `ProcessingOutput`, which can cause issues when scripts expect to create files.

```python
# WRONG ❌
expected_output_paths={
    "model": "/opt/ml/processing/output/model.tar.gz"  # File path, but SageMaker will create this as a directory
}
```

**Solution**: Use the parent directory for file outputs.

```python
# CORRECT ✅
expected_output_paths={
    "model": "/opt/ml/processing/output/model"  # Directory path
}

# Then in script:
model_dir = contract.expected_output_paths["model"]
model_file = os.path.join(model_dir, "model.tar.gz")
```

## Specification Development Pitfalls

### 1. Incomplete Compatible Sources

**Problem**: Not including all possible upstream steps that could provide a dependency.

```python
# WRONG ❌
"input_data": DependencySpec(
    # ...
    compatible_sources=["TabularPreprocessing"]  # Only one source listed
    # ...
)
```

**Solution**: List all compatible source steps.

```python
# CORRECT ✅
"input_data": DependencySpec(
    # ...
    compatible_sources=[
        "TabularPreprocessing", 
        "FeatureEngineering", 
        "DataLoading"
    ]  # All possible sources
    # ...
)
```

### 2. Incorrect Property Paths

**Problem**: Using incorrect property path formats for outputs.

```python
# WRONG ❌
"model_output": OutputSpec(
    logical_name="model_output",
    property_path="ModelArtifacts.S3Uri"  # Wrong format
    # ...
)
```

**Solution**: Use standard property path formats based on step type.

```python
# CORRECT ✅
# For Processing Steps - use the logical name in the path
"processed_data": OutputSpec(
    logical_name="processed_data",
    property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"  # Note: 'processed_data' matches logical_name
    # ...
)

# For Training Steps - model artifacts
"model_output": OutputSpec(
    logical_name="model_output",
    property_path="properties.ModelArtifacts.S3ModelArtifacts"  # Correct format for model artifacts
    # ...
)

# Key Rule: For Processing Steps, the property path must include the logical name:
# properties.ProcessingOutputConfig.Outputs['<logical_name>'].S3Output.S3Uri
```

### 3. Mismatch Between NodeType and Dependencies/Outputs

**Problem**: Using inconsistent NodeType, dependencies, and outputs.

```python
# WRONG ❌
YOUR_STEP_SPEC = StepSpecification(
    step_type=get_spec_step_type("YourStep"),
    node_type=NodeType.SOURCE,  # Source node type
    dependencies={  # But has dependencies
        "input_data": DependencySpec(...)
    },
    outputs={
        "output_data": OutputSpec(...)
    }
)
```

**Solution**: Ensure NodeType matches dependencies and outputs.

```python
# CORRECT ✅
# For source nodes (no dependencies)
SOURCE_STEP_SPEC = StepSpecification(
    step_type=get_spec_step_type("SourceStep"),
    node_type=NodeType.SOURCE,
    dependencies={},  # No dependencies
    outputs={
        "output_data": OutputSpec(...)
    }
)

# For internal nodes (dependencies and outputs)
INTERNAL_STEP_SPEC = StepSpecification(
    step_type=get_spec_step_type("InternalStep"),
    node_type=NodeType.INTERNAL,
    dependencies={
        "input_data": DependencySpec(...)
    },
    outputs={
        "output_data": OutputSpec(...)
    }
)

# For sink nodes (dependencies but no outputs)
SINK_STEP_SPEC = StepSpecification(
    step_type=get_spec_step_type("SinkStep"),
    node_type=NodeType.SINK,
    dependencies={
        "input_data": DependencySpec(...)
    },
    outputs={}  # No outputs
)
```

### 4. Missing Semantic Keywords

**Problem**: Not providing enough semantic keywords for dependency matching.

```python
# WRONG ❌
"input_data": DependencySpec(
    # ...
    semantic_keywords=["data"]  # Too vague
    # ...
)
```

**Solution**: Provide rich semantic keywords.

```python
# CORRECT ✅
"input_data": DependencySpec(
    # ...
    semantic_keywords=[
        "data", 
        "tabular", 
        "processed", 
        "features", 
        "training"
    ]  # Rich semantic context
    # ...
)
```

### 5. Missing Dependency Description

**Problem**: Not providing a clear description for dependencies.

```python
# WRONG ❌
"input_data": DependencySpec(
    # ...
    description=""  # Missing or empty description
    # ...
)
```

**Solution**: Add a clear, comprehensive description.

```python
# CORRECT ✅
"input_data": DependencySpec(
    # ...
    description="Processed tabular data for training, with features and label columns prepared for model training"  # Clear description
    # ...
)
```

## Builder Implementation Pitfalls

### 1. Hardcoded Container Paths

**Problem**: Hardcoding container paths instead of using the script contract.

```python
# WRONG ❌
def _get_inputs(self, inputs):
    return [
        ProcessingInput(
            input_name="data",
            source=inputs["data"],
            destination="/opt/ml/processing/input/data"  # Hardcoded path
        )
    ]
```

**Solution**: Get paths from the script contract.

```python
# CORRECT ✅
def _get_inputs(self, inputs):
    contract = self.spec.script_contract
    return [
        ProcessingInput(
            input_name="data",
            source=inputs["data"],
            destination=contract.expected_input_paths["data"]  # Path from contract
        )
    ]
```

### 2. Missing Environment Variables

**Problem**: Not setting all required environment variables.

```python
# WRONG ❌
def _get_processor_env_vars(self):
    return {
        # Missing required variables
    }
```

**Solution**: Set all required environment variables based on the contract.

```python
# CORRECT ✅
def _get_environment_variables(self) -> Dict[str, str]:
    """Get environment variables for the processor."""
    # Get base environment variables from contract
    env_vars = super()._get_environment_variables()
    
    # Add step-specific environment variables
    step_env_vars = {
        "LEARNING_RATE": str(self.config.learning_rate),
        "NUM_EPOCHS": str(self.config.num_epochs),
        "DEBUG_MODE": str(self.config.debug_mode).lower()
    }
    
    env_vars.update(step_env_vars)
    return env_vars
```

### 3. Not Using the Specification-Driven Approach

**Problem**: Manually defining inputs/outputs instead of using the specification-driven approach.

```python
# WRONG ❌
def _get_inputs(self, inputs):
    # Manual definition, not using specification
    return [
        ProcessingInput(
            input_name="data",
            source=inputs.get("data", ""),
            destination="/opt/ml/processing/input/data"
        )
    ]
```

**Solution**: Use specification-driven methods. See [Step Builder Implementation](step_builder.md) for details, follow [Standardization Rules](standardization_rules.md) for consistency, and refer to [Specification-Driven Design](../1_design/specification_driven_design.md) for architectural patterns.

```python
# CORRECT ✅
def _get_inputs(self, inputs):
    # Use specification-driven method
    return self._get_spec_driven_processor_inputs(inputs)
```

### 4. Missing Type Conversion for Environment Variables

**Problem**: Not converting non-string values to strings for environment variables.

```python
# WRONG ❌
def _get_processor_env_vars(self):
    return {
        "LEARNING_RATE": self.config.learning_rate,  # Not converted to string
        "NUM_EPOCHS": self.config.num_epochs  # Not converted to string
    }
```

**Solution**: Convert all values to strings. See [Environment Variable Setting](best_practices.md#environment-variable-setting) for best practices.

```python
# CORRECT ✅
def _get_processor_env_vars(self):
    return {
        "LEARNING_RATE": str(self.config.learning_rate),  # Converted to string
        "NUM_EPOCHS": str(self.config.num_epochs)  # Converted to string
    }
```

### 5. Not Handling Job Type Variants

**Problem**: Not selecting the appropriate specification for different job types.

```python
# WRONG ❌
def __init__(self, config, **kwargs):
    # Always using the same specification regardless of job type
    super().__init__(config=config, spec=YOUR_STEP_SPEC, **kwargs)
```

**Solution**: Select specification based on job type.

```python
# CORRECT ✅
def __init__(self, config, **kwargs):
    # Get job type if available
    job_type = getattr(config, 'job_type', None)
    
    # Select appropriate specification based on job type
    if job_type and job_type.lower() == "calibration":
        spec = YOUR_STEP_CALIBRATION_SPEC
    elif job_type and job_type.lower() == "validation":
        spec = YOUR_STEP_VALIDATION_SPEC
    else:
        spec = YOUR_STEP_TRAINING_SPEC  # Default to training
    
    super().__init__(config=config, spec=spec, **kwargs)
```

## Integration Pitfalls

### 1. Incomplete Step Registration

**Problem**: Not registering the step in all required places.

```python
# WRONG ❌
# Only added to step_names.py but not imported in __init__.py
```

**Solution**: Register the step in all required places.

```python
# CORRECT ✅
# In step_names.py
STEP_NAMES = {
    # ...
    "YourNewStep": {
        "config_class": "YourNewStepConfig",
        "builder_step_name": "YourNewStepBuilder",
        "spec_type": "YourNewStep"
    }
}

# In pipeline_steps/__init__.py
from .builder_your_new_step import YourNewStepBuilder

# In pipeline_step_specs/__init__.py
from .your_new_step_spec import YOUR_NEW_STEP_SPEC
```

### 2. Missing DAG Connections

**Problem**: Adding a node but forgetting to connect it with edges.

```python
# WRONG ❌
def _create_pipeline_dag(self):
    dag = PipelineDAG()
    dag.add_node("data_loading")
    dag.add_node("preprocessing")
    dag.add_node("your_new_step")  # Added node
    dag.add_node("training")
    
    # Missing connections for your_new_step
    dag.add_edge("data_loading", "preprocessing")
    dag.add_edge("preprocessing", "training")  # Direct connection bypassing your step
    
    return dag
```

**Solution**: Ensure proper connections in the DAG.

```python
# CORRECT ✅
def _create_pipeline_dag(self):
    dag = PipelineDAG()
    dag.add_node("data_loading")
    dag.add_node("preprocessing")
    dag.add_node("your_new_step")  # Added node
    dag.add_node("training")
    
    dag.add_edge("data_loading", "preprocessing")
    dag.add_edge("preprocessing", "your_new_step")  # Connect to your step
    dag.add_edge("your_new_step", "training")  # Connect from your step
    
    return dag
```

### 3. Forgetting to Add Configuration to Config Map

**Problem**: Adding a step to the DAG but forgetting to add its configuration to the config map.

```python
# WRONG ❌
def _create_config_map(self):
    config_map = {}
    config_map["data_loading"] = self._get_data_loading_config()
    config_map["preprocessing"] = self._get_preprocessing_config()
    # Missing your_new_step config
    config_map["training"] = self._get_training_config()
    return config_map
```

**Solution**: Add the step's configuration to the config map.

```python
# CORRECT ✅
def _create_config_map(self):
    config_map = {}
    config_map["data_loading"] = self._get_data_loading_config()
    config_map["preprocessing"] = self._get_preprocessing_config()
    config_map["your_new_step"] = self._get_your_new_step_config()  # Add your step's config
    config_map["training"] = self._get_training_config()
    return config_map
```

### 4. Not Adding Builder to Builder Map

**Problem**: Forgetting to add the step builder to the builder map.

```python
# WRONG ❌
def _create_step_builder_map(self):
    return {
        "data_loading": DataLoadingStepBuilder,
        "preprocessing": PreprocessingStepBuilder,
        # Missing your_new_step builder
        "training": TrainingStepBuilder
    }
```

**Solution**: Add the step builder to the builder map.

```python
# CORRECT ✅
def _create_step_builder_map(self):
    return {
        "data_loading": DataLoadingStepBuilder,
        "preprocessing": PreprocessingStepBuilder,
        "your_new_step": YourNewStepBuilder,  # Add your step's builder
        "training": TrainingStepBuilder
    }
```

### 5. Incompatible Dependencies

**Problem**: Connecting steps with incompatible dependencies.

```python
# WRONG ❌
# Step A output
"output_data": OutputSpec(
    logical_name="output_data",
    output_type=DependencyType.PROCESSING_OUTPUT
    # ...
)

# Step B dependency (incompatible type)
"input_model": DependencySpec(
    logical_name="input_model",
    dependency_type=DependencyType.MODEL_ARTIFACTS  # Different type than Step A output
    # ...
)

# Connecting A to B in DAG
dag.add_edge("step_a", "step_b")  # Incompatible connection
```

**Solution**: Ensure dependency compatibility or add intermediate steps.

```python
# CORRECT ✅
# Add intermediate step to convert between types, or
# Ensure output and dependency types match, or
# Modify specifications to be compatible
```

## Testing Pitfalls

### 1. Missing Three-Level Validation Tests

**Problem**: Not implementing comprehensive validation tests covering all three validation levels.

```python
# WRONG ❌
# Only testing basic functionality, missing validation framework tests
def test_step_creation(self):
    builder = YourStepBuilder(config)
    step = builder.create_step()
    self.assertIsNotNone(step)
```

**Solution**: Implement three-level validation tests. See [Validation Framework Guide](validation_framework_guide.md) for comprehensive testing approach and [Unit Testing Strategy](best_practices.md#unit-testing-strategy) for recommended test patterns.

```python
# CORRECT ✅
def test_three_level_validation(self):
    """Test comprehensive three-level validation."""
    
    # Level 1: Alignment Testing
    alignment_result = unified_alignment_tester.test_alignment(
        contract=YOUR_SCRIPT_CONTRACT,
        spec=YOUR_STEP_SPEC,
        builder_class=YourStepBuilder,
        config_class=YourStepConfig
    )
    self.assertTrue(alignment_result.is_valid, f"Alignment validation failed: {alignment_result.errors}")
    
    # Level 2: Builder Testing
    builder_result = universal_step_builder_tester.test_builder(
        builder_class=YourStepBuilder,
        config=self.test_config,
        test_inputs=self.test_inputs
    )
    self.assertTrue(builder_result.is_valid, f"Builder validation failed: {builder_result.errors}")
    
    # Level 3: Script Runtime Testing
    runtime_result = script_runtime_tester.test_script_execution(
        script_name="your_script",
        contract=YOUR_SCRIPT_CONTRACT,
        test_data=self.test_data
    )
    self.assertTrue(runtime_result.is_valid, f"Runtime validation failed: {runtime_result.errors}")

def test_contract_alignment(self):
    """Test that specification aligns with contract."""
    result = YOUR_STEP_SPEC.validate_contract_alignment()
    self.assertTrue(result.is_valid, f"Contract alignment failed: {result.errors}")
```

### 2. Missing Script Runtime Testing

**Problem**: Not testing actual script execution and data flow validation.

```python
# WRONG ❌
# No tests for script runtime behavior
```

**Solution**: Add comprehensive script runtime tests covering all three modes. See [Validation Framework Guide](validation_framework_guide.md#script-runtime-testing) for detailed implementation.

```python
# CORRECT ✅
def test_script_runtime_individual_mode(self):
    """Test individual script execution with mock data."""
    result = script_runtime_tester.test_individual_script(
        script_name="your_script",
        contract=YOUR_SCRIPT_CONTRACT,
        mock_data=self.mock_input_data,
        workspace_dir="./test_workspace"
    )
    self.assertTrue(result.is_valid, f"Individual script test failed: {result.errors}")

def test_script_runtime_compatibility_mode(self):
    """Test data compatibility between pipeline steps."""
    result = script_runtime_tester.test_data_compatibility(
        upstream_contract=UPSTREAM_CONTRACT,
        downstream_contract=YOUR_SCRIPT_CONTRACT,
        test_data=self.compatibility_test_data
    )
    self.assertTrue(result.is_valid, f"Data compatibility test failed: {result.errors}")

def test_script_runtime_pipeline_flow_mode(self):
    """Test complete pipeline flow with realistic data."""
    pipeline_dag = self.create_test_pipeline_dag()
    result = script_runtime_tester.test_pipeline_flow(
        pipeline_dag=pipeline_dag,
        test_data=self.realistic_test_data,
        workspace_dir="./test_workspace"
    )
    self.assertTrue(result.is_valid, f"Pipeline flow test failed: {result.errors}")
```

### 2. Missing Property Path Tests

**Problem**: Not testing property path consistency.

```python
# WRONG ❌
# No tests for property path consistency
```

**Solution**: Add property path consistency tests.

```python
# CORRECT ✅
def test_property_path_consistency(self):
    """Test property path consistency in outputs."""
    for output_name, output_spec in YOUR_STEP_SPEC.outputs.items():
        expected = f"properties.ProcessingOutputConfig.Outputs['{output_spec.logical_name}'].S3Output.S3Uri"
        self.assertEqual(output_spec.property_path, expected,
                       f"Property path inconsistency in {output_name}")
```

### 3. Not Testing Environment Variable Generation

**Problem**: Not testing if all required environment variables are correctly set.

```python
# WRONG ❌
# No tests for environment variable generation
```

**Solution**: Add environment variable generation tests.

```python
# CORRECT ✅
def test_environment_variable_generation(self):
    """Test that all required environment variables are set."""
    config = YourStepConfig(
        region="us-west-2",
        pipeline_s3_loc="s3://bucket/prefix",
        param1="value1",
        param2=42
    )
    builder = YourStepBuilder(config)
    
    env_vars = builder._get_processor_env_vars()
    
    # Check that all required variables are present
    for var_name in builder.spec.script_contract.required_env_vars:
        self.assertIn(var_name, env_vars, f"Missing required env var: {var_name}")
        
    # Check specific values
    self.assertEqual(env_vars["REQUIRED_PARAM_1"], "value1")
    self.assertEqual(env_vars["REQUIRED_PARAM_2"], "42")  # Note: converted to string
```

### 4. Not Testing with Job Type Variants

**Problem**: Only testing with default job type.

```python
# WRONG ❌
# Only testing with default job type
def test_step_creation(self):
    config = YourStepConfig(
        region="us-west-2",
        pipeline_s3_loc="s3://bucket/prefix"
    )  # No job_type specified
    builder = YourStepBuilder(config)
    step = builder.create_step()
    # ...
```

**Solution**: Test with all supported job types.

```python
# CORRECT ✅
def test_step_creation_with_training_job_type(self):
    config = YourStepConfig(
        region="us-west-2",
        pipeline_s3_loc="s3://bucket/prefix",
        job_type="training"
    )
    builder = YourStepBuilder(config)
    step = builder.create_step()
    # Test assertions...

def test_step_creation_with_calibration_job_type(self):
    config = YourStepConfig(
        region="us-west-2",
        pipeline_s3_loc="s3://bucket/prefix",
        job_type="calibration"
    )
    builder = YourStepBuilder(config)
    step = builder.create_step()
    # Test assertions...
```

### 5. Not Testing Edge Cases

**Problem**: Only testing the happy path, not edge cases.

```python
# WRONG ❌
# Only testing with valid inputs
```

**Solution**: Test edge cases and error conditions.

```python
# CORRECT ✅
def test_missing_required_dependency(self):
    """Test behavior when a required dependency is missing."""
    # Setup with missing dependency
    inputs = {}  # Missing required dependency
    
    # Act/Assert
    with self.assertRaises(ValueError):
        builder._get_inputs(inputs)
        
def test_empty_dependency_handling(self):
    """Test that empty dependencies are handled gracefully."""
    # Act
    extracted_inputs = builder.extract_inputs_from_dependencies([])
    
    # Assert
    self.assertEqual(extracted_inputs, {})
```

## 8. SageMaker Step Type Issues

### Issue: Wrong Property Path for Step Type
**Problem**: Using property paths that don't match the SageMaker step type being created.

```python
# WRONG ❌ - Using Training step property path in Processing step
"model_output": OutputSpec(
    logical_name="model_output",
    property_path="properties.ModelArtifacts.S3ModelArtifacts"  # Training step pattern
    # ...
)
```

**Solution**: Use correct property paths based on the SageMaker step type your builder creates.

```python
# CORRECT ✅ - Processing step property path
"processed_data": OutputSpec(
    logical_name="processed_data",
    property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"  # Processing step pattern
    # ...
)

# CORRECT ✅ - Training step property path  
"model_artifacts": OutputSpec(
    logical_name="model_artifacts",
    property_path="properties.ModelArtifacts.S3ModelArtifacts"  # Training step pattern
    # ...
)
```

### Issue: Mismatched Step Type Classification
**Problem**: Step builder creates a different SageMaker step type than expected based on its classification.

```python
# WRONG ❌ - Builder classified as Processing but creates Training step
class YourStepBuilder(StepBuilderBase):
    def create_step(self):
        # Creates TrainingStep but registry shows "Processing" type
        return TrainingStep(...)
```

**Solution**: Ensure step type classification matches actual SageMaker step creation. Check the [SageMaker Step Type Classification](../1_design/sagemaker_step_type_classification_design.md) for proper mapping.

```python
# CORRECT ✅ - Consistent classification and implementation
# In step_names.py registry:
"YourStep": {
    "sagemaker_step_type": "Training",  # Matches actual step creation
    # ...
}

# In builder:
class YourStepBuilder(StepBuilderBase):
    def create_step(self):
        return TrainingStep(...)  # Matches registry classification
```

### Issue: Invalid Property Path Syntax
**Problem**: Using incorrect syntax for property path references.

```python
# WRONG ❌ - Invalid syntax patterns
"output_data": OutputSpec(
    property_path="properties.ProcessingOutputConfig.Outputs.data.S3Output.S3Uri"  # Missing brackets
)

"metrics": OutputSpec(
    property_path="FinalMetricDataList['accuracy'].Value"  # Missing 'properties.' prefix
)
```

**Solution**: Follow correct property path syntax patterns from the [SageMaker Property Path Reference Database](sagemaker_property_path_reference_database.md).

```python
# CORRECT ✅ - Valid syntax patterns
"output_data": OutputSpec(
    property_path="properties.ProcessingOutputConfig.Outputs['data'].S3Output.S3Uri"  # Correct bracket notation
)

"metrics": OutputSpec(
    property_path="properties.FinalMetricDataList['accuracy'].Value"  # Correct prefix
)
```

## 9. Job Type Variant Issues

### Issue: Inconsistent Job Type Step Names
**Problem**: Job type variants create different step names that don't match pipeline DAG node names.

```python
# WRONG ❌ - Mismatch between config job_type and DAG node name
# Config with job_type creates step name: "CradleDataLoading_training"
config = CradleDataLoadConfig(job_type="training", ...)

# But DAG uses different node name
dag.add_node("train_data_load")  # Doesn't match generated step name
```

**Solution**: Ensure DAG node names align with job type variant step names or use consistent naming patterns.

```python
# CORRECT ✅ - Consistent naming approach
# Option 1: Use job type suffix in DAG node names
dag.add_node("CradleDataLoading_training")  # Matches generated step name

# Option 2: Use semantic node names and map appropriately
dag.add_node("train_data_load")
config_map["train_data_load"] = CradleDataLoadConfig(job_type="training", ...)
```

### Issue: Job Type Specification Mismatch
**Problem**: Using wrong specification for job type variant.

```python
# WRONG ❌ - Always using same spec regardless of job_type
class CradleDataLoadingStepBuilder(StepBuilderBase):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, spec=DATA_LOADING_SPEC, **kwargs)  # Wrong - ignores job_type
```

**Solution**: Select appropriate specification based on job type. See [Job Type Variant Handling](../1_design/job_type_variant_handling.md) for complete implementation patterns.

```python
# CORRECT ✅ - Dynamic specification selection
class CradleDataLoadingStepBuilder(StepBuilderBase):
    def __init__(self, config, **kwargs):
        job_type = getattr(config, 'job_type', 'training').lower()
        
        if job_type == 'calibration':
            spec = DATA_LOADING_CALIBRATION_SPEC
        elif job_type == 'validation':
            spec = DATA_LOADING_VALIDATION_SPEC
        elif job_type == 'testing':
            spec = DATA_LOADING_TESTING_SPEC
        else:
            spec = DATA_LOADING_TRAINING_SPEC
            
        super().__init__(config=config, spec=spec, **kwargs)
```

### Issue: Missing Job Type in Environment Variables
**Problem**: Not passing job type to processing scripts via environment variables.

```python
# WRONG ❌ - Missing JOB_TYPE environment variable
def _get_environment_variables(self):
    return {
        "REGION": self.config.region,
        "DATA_SOURCE_TYPE": self.config.data_source_type
        # Missing JOB_TYPE
    }
```

**Solution**: Include job type in environment variables as required by script contracts.

```python
# CORRECT ✅ - Include job type
def _get_environment_variables(self):
    return {
        "REGION": self.config.region,
        "JOB_TYPE": getattr(self.config, 'job_type', 'training'),
        "DATA_SOURCE_TYPE": self.config.data_source_type
    }
```

### Issue: Semantic Keyword Conflicts
**Problem**: Different job type variants using conflicting semantic keywords causing wrong dependency resolution.

```python
# WRONG ❌ - Same semantic keywords for different job types
DATA_LOADING_TRAINING_SPEC = StepSpecification(
    outputs=[
        OutputSpec(
            semantic_keywords=["data", "input", "raw"]  # Generic keywords
        )
    ]
)

DATA_LOADING_CALIBRATION_SPEC = StepSpecification(
    outputs=[
        OutputSpec(
            semantic_keywords=["data", "input", "raw"]  # Same keywords - causes conflicts
        )
    ]
)
```

**Solution**: Use job-type-specific semantic keywords to ensure proper dependency resolution.

```python
# CORRECT ✅ - Job-type-specific semantic keywords
DATA_LOADING_TRAINING_SPEC = StepSpecification(
    outputs=[
        OutputSpec(
            semantic_keywords=["training", "train", "data", "input", "raw", "dataset"]
        )
    ]
)

DATA_LOADING_CALIBRATION_SPEC = StepSpecification(
    outputs=[
        OutputSpec(
            semantic_keywords=["calibration", "calib", "eval", "data", "input"]
        )
    ]
)
```

### Issue: Config Types Metadata Missing Job Type Variants
**Problem**: Not including job type variant step names in config_types metadata.

```python
# WRONG ❌ - Missing job type variants in metadata
"config_types": {
    "CradleDataLoading": "CradleDataLoadConfig"  # Only base name
}
```

**Solution**: Include all job type variant step names in config_types metadata.

```python
# CORRECT ✅ - Include all variants
"config_types": {
    "CradleDataLoading_training": "CradleDataLoadConfig",
    "CradleDataLoading_calibration": "CradleDataLoadConfig", 
    "CradleDataLoading_validation": "CradleDataLoadConfig",
    "CradleDataLoading_testing": "CradleDataLoadConfig"
}
```

## Conclusion

By being aware of these common pitfalls, you can avoid many of the issues that often arise when implementing new pipeline steps. Remember to:

1. **Follow the Conventions**: Adhere to established path conventions and naming standards
2. **Ensure Alignment**: Maintain alignment between script, contract, and specification
3. **Handle Edge Cases**: Consider and test for edge cases and error conditions
4. **Complete Registration**: Register your step in all required places
5. **Provide Rich Documentation**: Document your step comprehensively
6. **Follow Standardization Rules**: Apply [standardization rules](standardization_rules.md) for universal patterns
7. **Validate SageMaker Step Types**: Ensure property paths match the actual SageMaker step type being created
8. **Handle Job Type Variants**: Properly implement job type variant handling with consistent naming and specification selection

When in doubt, refer to the [validation checklist](validation_checklist.md) to verify your implementation, follow the [standardization rules](standardization_rules.md) for universal patterns, consult the [best practices](best_practices.md) for guidance on recommended approaches, and check the [SageMaker Property Path Reference Database](sagemaker_property_path_reference_database.md) for correct property path patterns.

## Related Documentation

### Developer Guide References
- [Adding New Pipeline Step](adding_new_pipeline_step.md) - Quick start guide for new step implementation
- [Creation Process](creation_process.md) - Complete step-by-step creation workflow
- [Validation Framework Guide](validation_framework_guide.md) - Comprehensive three-level validation approach
- [Best Practices](best_practices.md) - Recommended patterns and approaches
- [Standardization Rules](standardization_rules.md) - Universal implementation standards
- [Alignment Rules](alignment_rules.md) - Contract-specification alignment requirements
- [Script Contract](script_contract.md) - Script contract development guide
- [Script Development Guide](script_development_guide.md) - Comprehensive script implementation patterns
- [Step Builder](step_builder.md) - Step builder implementation guide
- [Step Specification](step_specification.md) - Step specification development guide
- [Three Tier Config Design](three_tier_config_design.md) - Configuration class design patterns
- [Component Guide](component_guide.md) - System component overview
- [Step Catalog Integration Guide](step_catalog_integration_guide.md) - StepCatalog integration patterns
- [Pipeline Catalog Integration Guide](pipeline_catalog_integration_guide.md) - Pipeline catalog usage

### Design References
- [SageMaker Step Type Classification Design](../1_design/sagemaker_step_type_classification_design.md) - Step type mapping and validation
- [Job Type Variant Handling](../1_design/job_type_variant_handling.md) - Job type variant implementation patterns
- [Level 1: Script Contract Alignment Design](../1_design/level1_script_contract_alignment_design.md) - Contract alignment validation
- [Level 2: Contract Specification Alignment Design](../1_design/level2_contract_specification_alignment_design.md) - Specification alignment validation
- [Specification-Driven Design](../1_design/specification_driven_design.md) - Architectural patterns for specification-driven development
- [Universal Step Builder Test](../1_design/universal_step_builder_test.md) - Builder testing framework
- [Unified Alignment Tester Design](../1_design/unified_alignment_tester_design.md) - Alignment testing architecture
- [Pipeline Runtime Testing Simplified Design](../1_design/pipeline_runtime_testing_simplified_design.md) - Script runtime testing framework
- [Step Builder Registry Design](../1_design/step_builder_registry_design.md) - Registry architecture and patterns
- [Dependency Resolution System](../1_design/dependency_resolution_system.md) - Dependency matching and resolution
- [Config Types Format](../1_design/config_types_format.md) - Configuration metadata format
- [Registry Based Step Name Generation](../1_design/registry_based_step_name_generation.md) - Step name generation patterns
- [Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md) - Processing step implementation patterns
- [Training Step Builder Patterns](../1_design/training_step_builder_patterns.md) - Training step implementation patterns
- [Environment Variable Contract Enforcement](../1_design/environment_variable_contract_enforcement.md) - Environment variable validation
