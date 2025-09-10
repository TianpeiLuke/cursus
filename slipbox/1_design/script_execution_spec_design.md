---
tags:
  - design
  - pipeline_runtime_testing
  - script_execution_spec
  - data_model
  - dual_identity
keywords:
  - ScriptExecutionSpec
  - script parameters
  - path specifications
  - dual identity
  - file discovery
  - DAG node matching
topics:
  - runtime testing
  - data models
  - script configuration
  - path management
language: python
date of note: 2025-09-09
---

# ScriptExecutionSpec Design

## Overview

The ScriptExecutionSpec is a core data model that defines script execution parameters and path specifications for the pipeline runtime testing system. It manages the dual identity challenge where scripts need both file-based identity (for discovery) and DAG node identity (for matching), while providing comprehensive execution configuration.

## Core Challenge: Dual Identity Management

The ScriptExecutionSpec addresses a fundamental challenge in pipeline runtime testing:

- **File Identity**: `script_name` for filesystem operations and script discovery
- **DAG Node Identity**: `step_name` for pipeline DAG node matching and job type handling
- **Path Management**: Input/output path specifications with logical name mapping
- **Execution Context**: Environment variables, job arguments, and runtime parameters

## Data Model Architecture

### Basic Structure

```python
class ScriptExecutionSpec(BaseModel):
    """
    Comprehensive specification for script execution in runtime testing.
    
    Manages dual identity (script_name vs step_name) and provides complete
    execution context including paths, environment, and job parameters.
    """
    
    # Core Identity Fields
    script_name: str = Field(..., description="Script file name (snake_case)")
    step_name: str = Field(..., description="DAG node name (PascalCase with job type)")
    script_path: str = Field(..., description="Full path to script file")
    
    # Path Specifications
    input_paths: Dict[str, str] = Field(default_factory=dict, description="Logical name to input path mapping")
    output_paths: Dict[str, str] = Field(default_factory=dict, description="Logical name to output path mapping")
    
    # Execution Context
    environ_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    job_args: Dict[str, Any] = Field(default_factory=dict, description="Job-specific arguments")
    
    # Metadata
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
```

### Enhanced Structure with PathSpecs

```python
class PathSpec(BaseModel):
    """Enhanced path specification with alias support."""
    logical_name: str = Field(..., description="Primary logical name")
    path: str = Field(..., description="File system path")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    
    def matches_name_or_alias(self, name: str) -> bool:
        """Check if name matches logical_name or any alias"""
        return name == self.logical_name or name in self.aliases

class EnhancedScriptExecutionSpec(BaseModel):
    """Enhanced ScriptExecutionSpec with alias system for logical name matching."""
    
    # Core Identity (same as basic)
    script_name: str
    step_name: str
    script_path: str
    
    # Enhanced Path Specifications with alias support
    input_path_specs: Dict[str, PathSpec] = Field(default_factory=dict)
    output_path_specs: Dict[str, PathSpec] = Field(default_factory=dict)
    
    # Execution Context (same as basic)
    environ_vars: Dict[str, str] = Field(default_factory=dict)
    job_args: Dict[str, Any] = Field(default_factory=dict)
    
    # Backward Compatibility Properties
    @property
    def input_paths(self) -> Dict[str, str]:
        """Backward compatibility property for input paths"""
        return {name: spec.path for name, spec in self.input_path_specs.items()}
    
    @property 
    def output_paths(self) -> Dict[str, str]:
        """Backward compatibility property for output paths"""
        return {name: spec.path for name, spec in self.output_path_specs.items()}
```

## Identity Resolution Examples

### Example 1: Standard Processing Step

```python
# DAG Node: "TabularPreprocessing_training"
# Script File: "tabular_preprocessing.py"

spec = ScriptExecutionSpec(
    script_name="tabular_preprocessing",           # For file discovery
    step_name="TabularPreprocessing_training",     # For DAG node matching
    script_path="/path/to/scripts/tabular_preprocessing.py",
    
    input_paths={
        "data_input": "/test_data/input/raw_data",
        "config": "/test_data/config/preprocessing_config.json"
    },
    
    output_paths={
        "data_output": "/test_data/output/processed_data",
        "metrics": "/test_data/output/preprocessing_metrics"
    },
    
    environ_vars={
        "PYTHONPATH": "/path/to/cursus/src",
        "CURSUS_ENV": "testing"
    },
    
    job_args={
        "batch_size": 1000,
        "validation_split": 0.2
    }
)
```

### Example 2: Complex Technical Term with Job Type

```python
# DAG Node: "XGBoostModelEval_evaluation"
# Script File: "xgboost_model_evaluation.py"

spec = ScriptExecutionSpec(
    script_name="xgboost_model_evaluation",        # Resolved from XGBoostModelEval
    step_name="XGBoostModelEval_evaluation",       # Original DAG node name
    script_path="/path/to/scripts/xgboost_model_evaluation.py",
    
    input_paths={
        "model_input": "/test_data/models/trained_xgboost",
        "data_input": "/test_data/validation/test_dataset",
        "config": "/test_data/config/evaluation_config.json"
    },
    
    output_paths={
        "metrics_output": "/test_data/results/evaluation_metrics",
        "predictions_output": "/test_data/results/predictions"
    },
    
    environ_vars={
        "MODEL_TYPE": "xgboost",
        "EVALUATION_MODE": "full"
    },
    
    job_args={
        "metrics": ["accuracy", "precision", "recall", "f1"],
        "threshold": 0.5
    }
)
```

### Example 3: Enhanced Spec with Aliases

```python
# Enhanced spec with logical name aliases for intelligent matching

enhanced_spec = EnhancedScriptExecutionSpec(
    script_name="tabular_preprocessing",
    step_name="TabularPreprocessing_training",
    script_path="/path/to/scripts/tabular_preprocessing.py",
    
    input_path_specs={
        "data_input": PathSpec(
            logical_name="data_input",
            path="/test_data/input/raw_data",
            aliases=["input", "dataset", "training_data", "raw_dataset"]
        ),
        "config": PathSpec(
            logical_name="config",
            path="/test_data/config/preprocessing_config.json",
            aliases=["configuration", "params", "hyperparameters", "settings"]
        )
    },
    
    output_path_specs={
        "data_output": PathSpec(
            logical_name="data_output",
            path="/test_data/output/processed_data",
            aliases=["output", "result", "processed_data", "dataset"]
        ),
        "metrics": PathSpec(
            logical_name="metrics",
            path="/test_data/output/preprocessing_metrics",
            aliases=["stats", "statistics", "performance", "quality_metrics"]
        )
    }
)
```

## Core Methods and Operations

### 1. Serialization and Persistence

```python
class ScriptExecutionSpec(BaseModel):
    
    def save_to_file(self, file_path: Path) -> None:
        """Save spec to JSON file for persistence."""
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'ScriptExecutionSpec':
        """Load spec from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
    
    def get_spec_file_name(self) -> str:
        """Get the standard file name for this spec."""
        return f"{self.script_name}_runtime_test_spec.json"
```

### 2. Path Management

```python
def add_input_path(self, logical_name: str, path: str) -> None:
    """Add or update an input path specification."""
    self.input_paths[logical_name] = path
    self.update_timestamp()

def add_output_path(self, logical_name: str, path: str) -> None:
    """Add or update an output path specification."""
    self.output_paths[logical_name] = path
    self.update_timestamp()

def get_input_path(self, logical_name: str) -> Optional[str]:
    """Get input path by logical name."""
    return self.input_paths.get(logical_name)

def get_output_path(self, logical_name: str) -> Optional[str]:
    """Get output path by logical name."""
    return self.output_paths.get(logical_name)

def validate_paths(self) -> List[str]:
    """Validate that all specified paths exist and are accessible."""
    errors = []
    
    # Validate input paths
    for logical_name, path in self.input_paths.items():
        path_obj = Path(path)
        if not path_obj.exists():
            errors.append(f"Input path '{logical_name}' does not exist: {path}")
        elif not os.access(path, os.R_OK):
            errors.append(f"Input path '{logical_name}' is not readable: {path}")
    
    # Validate output paths (directories should exist or be creatable)
    for logical_name, path in self.output_paths.items():
        path_obj = Path(path)
        if not path_obj.exists():
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                errors.append(f"Cannot create output path '{logical_name}': {path} - {str(e)}")
        elif not os.access(path, os.W_OK):
            errors.append(f"Output path '{logical_name}' is not writable: {path}")
    
    return errors
```

### 3. Environment and Job Arguments

```python
def add_environ_var(self, key: str, value: str) -> None:
    """Add or update an environment variable."""
    self.environ_vars[key] = value
    self.update_timestamp()

def add_job_arg(self, key: str, value: Any) -> None:
    """Add or update a job argument."""
    self.job_args[key] = value
    self.update_timestamp()

def get_execution_environment(self) -> Dict[str, str]:
    """Get complete execution environment including system vars."""
    env = os.environ.copy()
    env.update(self.environ_vars)
    return env

def validate_job_args(self, required_args: List[str] = None) -> List[str]:
    """Validate job arguments against required parameters."""
    errors = []
    
    if required_args:
        missing_args = set(required_args) - set(self.job_args.keys())
        if missing_args:
            errors.append(f"Missing required job arguments: {', '.join(missing_args)}")
    
    return errors
```

## File Storage and Naming Convention

### Storage Location

ScriptExecutionSpec files are stored in the `.specs` directory within the test data directory:

```
test_data_dir/
├── .specs/                            # Hidden directory for specs
│   ├── tabular_preprocessing_runtime_test_spec.json
│   ├── xgboost_training_runtime_test_spec.json
│   ├── model_calibration_runtime_test_spec.json
│   └── ...
```

### Naming Convention

The file naming follows a consistent pattern:
- **Format**: `{script_name}_runtime_test_spec.json`
- **Examples**:
  - `tabular_preprocessing_runtime_test_spec.json`
  - `xgboost_model_evaluation_runtime_test_spec.json`
  - `model_calibration_runtime_test_spec.json`

### JSON Structure

```json
{
  "script_name": "tabular_preprocessing",
  "step_name": "TabularPreprocessing_training",
  "script_path": "/path/to/scripts/tabular_preprocessing.py",
  "input_paths": {
    "data_input": "/test_data/input/raw_data",
    "config": "/test_data/config/preprocessing_config.json"
  },
  "output_paths": {
    "data_output": "/test_data/output/processed_data",
    "metrics": "/test_data/output/preprocessing_metrics"
  },
  "environ_vars": {
    "PYTHONPATH": "/path/to/cursus/src",
    "CURSUS_ENV": "testing"
  },
  "job_args": {
    "batch_size": 1000,
    "validation_split": 0.2
  },
  "created_at": "2025-09-09T22:14:23.123456",
  "updated_at": "2025-09-09T22:14:23.123456"
}
```

## Integration with Pipeline Components

### 1. PipelineTestingSpecBuilder Integration

```python
# Builder creates and manages ScriptExecutionSpec instances
class PipelineTestingSpecBuilder:
    
    def resolve_script_execution_spec_from_node(self, node_name: str) -> ScriptExecutionSpec:
        """Create ScriptExecutionSpec from DAG node name."""
        # ... resolution logic ...
        
        return ScriptExecutionSpec(
            script_name=resolved_script_name,
            step_name=node_name,
            script_path=resolved_script_path,
            # ... other fields ...
        )
    
    def _load_or_create_script_spec(self, node_name: str) -> ScriptExecutionSpec:
        """Load existing or create new ScriptExecutionSpec."""
        spec = self.resolve_script_execution_spec_from_node(node_name)
        spec_file_path = self.specs_dir / spec.get_spec_file_name()
        
        if spec_file_path.exists():
            existing_spec = ScriptExecutionSpec.load_from_file(spec_file_path)
            existing_spec.step_name = node_name  # Update for current context
            return existing_spec
        else:
            return spec
```

### 2. RuntimeTester Integration

```python
# RuntimeTester uses ScriptExecutionSpec for script execution
class RuntimeTester:
    
    def test_script_with_spec(self, spec: ScriptExecutionSpec, 
                            main_params: Dict[str, Any]) -> ScriptTestResult:
        """Execute script using ScriptExecutionSpec configuration."""
        
        # Use spec for execution environment
        env = spec.get_execution_environment()
        
        # Use spec paths for input/output validation
        input_validation = self._validate_input_paths(spec)
        
        # Execute with spec configuration
        result = self._execute_script_safely(
            Path(spec.script_path), 
            main_params, 
            env
        )
        
        return result
```

### 3. Logical Name Matching Integration

```python
# Enhanced specs support intelligent path matching
def _create_enhanced_script_spec(self, original_spec: ScriptExecutionSpec) -> EnhancedScriptExecutionSpec:
    """Convert to enhanced spec with PathSpecs for matching."""
    
    input_path_specs = {}
    for logical_name, path in original_spec.input_paths.items():
        input_path_specs[logical_name] = PathSpec(
            logical_name=logical_name,
            path=path,
            aliases=self._get_aliases_for_logical_name(logical_name)
        )
    
    # ... similar for output_path_specs ...
    
    return EnhancedScriptExecutionSpec(
        script_name=original_spec.script_name,
        step_name=original_spec.step_name,
        script_path=original_spec.script_path,
        input_path_specs=input_path_specs,
        output_path_specs=output_path_specs,
        environ_vars=original_spec.environ_vars,
        job_args=original_spec.job_args
    )
```

## Validation and Error Handling

### 1. Comprehensive Validation

```python
def validate_spec(self) -> List[str]:
    """Comprehensive validation of the ScriptExecutionSpec."""
    errors = []
    
    # Validate core identity
    if not self.script_name:
        errors.append("script_name is required")
    elif not re.match(r'^[a-z][a-z0-9_]*$', self.script_name):
        errors.append("script_name must be in snake_case format")
    
    if not self.step_name:
        errors.append("step_name is required")
    elif not re.match(r'^[A-Z][A-Za-z0-9]*(_[A-Za-z]+)?$', self.step_name):
        errors.append("step_name must be in PascalCase format with optional job type suffix")
    
    # Validate script path
    if not self.script_path:
        errors.append("script_path is required")
    elif not Path(self.script_path).exists():
        errors.append(f"Script file does not exist: {self.script_path}")
    elif not self.script_path.endswith('.py'):
        errors.append("script_path must point to a Python file (.py)")
    
    # Validate paths
    errors.extend(self.validate_paths())
    
    # Validate job arguments if required
    errors.extend(self.validate_job_args())
    
    return errors

def is_valid(self) -> bool:
    """Check if the spec is valid."""
    return len(self.validate_spec()) == 0
```

### 2. Error Recovery

```python
def auto_fix_common_issues(self) -> List[str]:
    """Attempt to automatically fix common validation issues."""
    fixes_applied = []
    
    # Fix script_name format
    if self.script_name and not re.match(r'^[a-z][a-z0-9_]*$', self.script_name):
        original = self.script_name
        self.script_name = self._to_snake_case(self.script_name)
        fixes_applied.append(f"Fixed script_name format: {original} -> {self.script_name}")
    
    # Create missing output directories
    for logical_name, path in self.output_paths.items():
        path_obj = Path(path)
        if not path_obj.exists():
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                fixes_applied.append(f"Created output directory: {path}")
            except OSError:
                pass  # Will be caught in validation
    
    # Update timestamp if fixes were applied
    if fixes_applied:
        self.update_timestamp()
    
    return fixes_applied

def _to_snake_case(self, name: str) -> str:
    """Convert name to snake_case."""
    result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    result = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', result)
    return result.lower()
```

## Usage Patterns

### 1. Basic Creation and Usage

```python
# Create a new ScriptExecutionSpec
spec = ScriptExecutionSpec(
    script_name="tabular_preprocessing",
    step_name="TabularPreprocessing_training",
    script_path="/path/to/scripts/tabular_preprocessing.py"
)

# Add paths
spec.add_input_path("data_input", "/test_data/input/raw_data")
spec.add_output_path("data_output", "/test_data/output/processed_data")

# Add environment variables
spec.add_environ_var("PYTHONPATH", "/path/to/cursus/src")
spec.add_environ_var("CURSUS_ENV", "testing")

# Add job arguments
spec.add_job_arg("batch_size", 1000)
spec.add_job_arg("validation_split", 0.2)

# Validate and save
if spec.is_valid():
    spec.save_to_file(Path("/test_data/.specs") / spec.get_spec_file_name())
else:
    print("Validation errors:", spec.validate_spec())
```

### 2. Loading and Updating

```python
# Load existing spec
spec_file = Path("/test_data/.specs/tabular_preprocessing_runtime_test_spec.json")
spec = ScriptExecutionSpec.load_from_file(spec_file)

# Update for new context
spec.step_name = "TabularPreprocessing_evaluation"  # Different job type
spec.add_job_arg("evaluation_mode", "full")
spec.update_timestamp()

# Save updated spec
spec.save_to_file(spec_file)
```

### 3. Enhanced Spec Usage

```python
# Create enhanced spec with aliases
enhanced_spec = EnhancedScriptExecutionSpec(
    script_name="xgboost_training",
    step_name="XGBoostTraining_training",
    script_path="/path/to/scripts/xgboost_training.py",
    
    input_path_specs={
        "data_input": PathSpec(
            logical_name="data_input",
            path="/test_data/input/training_data",
            aliases=["training_data", "dataset", "input"]
        )
    },
    
    output_path_specs={
        "model_output": PathSpec(
            logical_name="model_output",
            path="/test_data/models/trained_xgboost",
            aliases=["model", "artifact", "trained_model"]
        )
    }
)

# Use for intelligent matching
matches = path_matcher.find_path_matches(source_spec, enhanced_spec)
```

## Performance Considerations

### 1. Memory Usage
- **Basic spec**: ~1-2KB per instance
- **Enhanced spec with aliases**: ~2-5KB per instance
- **JSON serialization**: ~1-3KB per file

### 2. I/O Performance
- **File save/load**: ~1-5ms per operation
- **Path validation**: ~0.1ms per path
- **Spec validation**: ~0.5-2ms per spec

### 3. Optimization Strategies
- **Lazy loading**: Load specs only when needed
- **Caching**: Cache frequently used specs in memory
- **Batch operations**: Process multiple specs together
- **Path caching**: Cache path existence checks

## Testing Strategy

### 1. Unit Tests
- Spec creation and validation
- Serialization/deserialization
- Path management operations
- Environment variable handling

### 2. Integration Tests
- Integration with PipelineTestingSpecBuilder
- Integration with RuntimeTester
- File system operations
- Error handling scenarios

### 3. Performance Tests
- Large spec handling
- Batch operations
- Memory usage monitoring
- I/O performance

## Future Enhancements

### 1. Advanced Features
- **Schema validation**: JSON schema validation for specs
- **Version management**: Spec versioning and migration
- **Template system**: Spec templates for common patterns
- **Inheritance**: Spec inheritance and composition

### 2. Enhanced Metadata
- **Execution history**: Track execution results and performance
- **Dependencies**: Explicit dependency tracking
- **Tags and labels**: Categorization and filtering
- **Documentation**: Embedded documentation and examples

### 3. Integration Improvements
- **IDE integration**: Editor support and validation
- **CI/CD integration**: Automated spec validation
- **Monitoring**: Runtime monitoring and alerting
- **Analytics**: Usage analytics and optimization suggestions

## References

### Foundation Documents
- **[Config Driven Design](config_driven_design.md)**: Core principles for specification-driven system architecture
- **[Design Principles](design_principles.md)**: Fundamental design patterns and architectural guidelines
- **[Config Types Format](config_types_format.md)**: Data model patterns and type system design

### Data Model and Serialization Patterns
- **[Config Field Manager Refactoring](config_field_manager_refactoring.md)**: Field management and validation patterns
- **[Config Tiered Design](config_tiered_design.md)**: Hierarchical configuration architecture
- **[Enhanced Property Reference](enhanced_property_reference.md)**: Property resolution and reference management

### Validation and Error Handling
- **[Enhanced Dependency Validation Design](enhanced_dependency_validation_design.md)**: Validation framework patterns
- **[Environment Variable Contract Enforcement](environment_variable_contract_enforcement.md)**: Environment variable validation and management
- **[Alignment Validation Data Structures](alignment_validation_data_structures.md)**: Data structure validation patterns

### Path Management and File Discovery
- **[Flexible File Resolver Design](flexible_file_resolver_design.md)**: File discovery and path resolution patterns
- **[Default Values Provider Revised](default_values_provider_revised.md)**: Default value management and path specifications
- **[Dependency Resolution System](dependency_resolution_system.md)**: Resource dependency and path management

### Pipeline Integration
- **[Pipeline Testing Spec Builder Design](pipeline_testing_spec_builder_design.md)**: Builder pattern integration for spec creation
- **[Runtime Tester Design](runtime_tester_design.md)**: Execution engine integration patterns
- **[Pipeline Testing Spec Design](pipeline_testing_spec_design.md)**: Top-level pipeline configuration integration

### Logical Name Matching and Semantic Resolution
- **[Logical Name Matching](../validation/logical_name_matching_design.md)**: Semantic matching algorithms for path resolution
- **[Enhanced Universal Step Builder Tester Design](enhanced_universal_step_builder_tester_design.md)**: Universal testing patterns

### Testing Framework Integration
- **[Pytest Unittest Compatibility Framework Design](pytest_unittest_compatibility_framework_design.md)**: Testing framework integration patterns
- **[Pipeline Runtime Testing Simplified Design](pipeline_runtime_testing_simplified_design.md)**: Overall runtime testing architecture

## Conclusion

The ScriptExecutionSpec provides a comprehensive, flexible data model for managing script execution in the pipeline runtime testing system. By addressing the dual identity challenge and providing rich configuration capabilities, it enables reliable, maintainable script testing while supporting advanced features like logical name matching and intelligent path resolution.

Key design principles:
- **Dual Identity**: Clear separation of file and DAG node identities
- **Comprehensive Configuration**: Complete execution context management
- **Validation**: Robust validation and error handling
- **Extensibility**: Support for enhanced features and future growth
- **Performance**: Efficient serialization and minimal overhead

The ScriptExecutionSpec serves as the foundation for reliable script configuration management, enabling the runtime testing system to execute scripts consistently and predictably across different contexts and environments.
