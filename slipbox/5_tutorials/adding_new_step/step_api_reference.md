---
tags:
  - code
  - step_development
  - api_reference
  - pipeline_step_creation
  - cursus_components
keywords:
  - step development API
  - cursus step API
  - script contract API
  - step specification API
  - step builder API
  - configuration API
  - pipeline integration
  - component alignment
topics:
  - Step development API reference
  - Cursus core components
  - Pipeline step integration
  - ML workflow development
language: python
date of note: 2025-09-11
---

# Adding New Pipeline Step: API Reference

## Overview

This API reference provides detailed documentation for all components involved in creating new pipeline steps in Cursus. It covers the key interfaces, classes, and methods you'll use when developing custom steps, with links to existing documentation where appropriate.

## Core Components

### Script Contract API

The script contract defines the interface between your processing script and the SageMaker container environment.

#### ScriptContract Class

```python
from cursus.core.base.contract_base import ScriptContract

class ScriptContract:
    def __init__(
        self,
        entry_point: str,
        expected_input_paths: Dict[str, str],
        expected_output_paths: Dict[str, str],
        required_env_vars: List[str],
        optional_env_vars: Dict[str, str] = None,
        framework_requirements: Dict[str, str] = None,
        description: str = ""
    )
```

**Parameters:**
- `entry_point`: Script filename (e.g., "feature_selection.py")
- `expected_input_paths`: Mapping of logical names to container paths
- `expected_output_paths`: Mapping of logical names to container paths
- `required_env_vars`: List of required environment variable names
- `optional_env_vars`: Dict of optional env vars with default values
- `framework_requirements`: Dict of package dependencies with version constraints
- `description`: Detailed description of the script's functionality

**Example:**
```python
FEATURE_SELECTION_CONTRACT = ScriptContract(
    entry_point="feature_selection.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/data",
        "config": "/opt/ml/processing/input/config"
    },
    expected_output_paths={
        "selected_features": "/opt/ml/processing/output/features",
        "feature_importance": "/opt/ml/processing/output/importance"
    },
    required_env_vars=["SELECTION_METHOD", "N_FEATURES", "TARGET_COLUMN"],
    optional_env_vars={
        "MIN_IMPORTANCE": "0.01",
        "RANDOM_SEED": "42",
        "DEBUG_MODE": "False"
    },
    framework_requirements={
        "pandas": ">=1.2.0,<2.0.0",
        "scikit-learn": ">=0.23.2,<1.0.0",
        "numpy": ">=1.19.0"
    }
)
```

**Related Documentation:**
- [Script Contract Guide](../../0_developer_guide/script_contract.md)
- [Container Path Standards](../../0_developer_guide/sagemaker_property_path_reference_database.md)

### Processing Script API

All processing scripts must implement the unified main function interface for testability and container compatibility.

#### Unified Main Function Interface

```python
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger=None
) -> Dict[str, Any]:
    """
    Main processing function with unified interface.
    
    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger function
    
    Returns:
        Processing results dictionary
    """
```

**Parameters:**
- `input_paths`: Maps logical names to actual input paths
- `output_paths`: Maps logical names to actual output paths
- `environ_vars`: All environment variables as key-value pairs
- `job_args`: Parsed command line arguments
- `logger`: Optional logging function (defaults to print)

**Return Value:**
Must return a dictionary with processing results, typically including:
- `status`: "success" or "failed"
- `processed_records`: Number of records processed
- `output_files`: List of generated output files
- Any step-specific metrics or metadata

**Example:**
```python
def main(input_paths, output_paths, environ_vars, job_args, logger=None):
    log = logger or print
    
    # Extract parameters from environment variables
    selection_method = environ_vars.get("SELECTION_METHOD")
    n_features = int(environ_vars.get("N_FEATURES", "10"))
    
    # Process data
    result = process_feature_selection(input_paths, output_paths, selection_method, n_features)
    
    return {
        "status": "success",
        "selected_features_count": result.feature_count,
        "selection_method": selection_method,
        "output_files": result.files
    }
```

**Related Documentation:**
- [Script Development Guide](../../0_developer_guide/script_development_guide.md)
- [Script Testability Implementation](../../0_developer_guide/script_testability_implementation.md)

### Step Specification API

Step specifications define how your step connects with other steps through logical input/output names.

#### StepSpecification Class

```python
from cursus.core.base.specification_base import (
    StepSpecification, DependencySpec, OutputSpec, 
    DependencyType, NodeType
)

class StepSpecification:
    def __init__(
        self,
        step_type: str,
        node_type: NodeType,
        script_contract: ScriptContract,
        dependencies: Dict[str, DependencySpec],
        outputs: Dict[str, OutputSpec]
    )
```

**Parameters:**
- `step_type`: Unique identifier for the step type
- `node_type`: NodeType enum (SOURCE, INTERNAL, SINK)
- `script_contract`: Associated script contract
- `dependencies`: Dict of input dependencies
- `outputs`: Dict of output specifications

#### DependencySpec Class

```python
class DependencySpec:
    def __init__(
        self,
        logical_name: str,
        dependency_type: DependencyType,
        required: bool,
        compatible_sources: List[str],
        semantic_keywords: List[str],
        data_type: str,
        description: str
    )
```

**Parameters:**
- `logical_name`: Logical name for the dependency (must match contract)
- `dependency_type`: Type of dependency (PROCESSING_OUTPUT, TRAINING_OUTPUT, etc.)
- `required`: Whether this dependency is required
- `compatible_sources`: List of compatible step types
- `semantic_keywords`: Keywords for automatic dependency resolution
- `data_type`: Expected data type (S3Uri, String, etc.)
- `description`: Human-readable description

#### OutputSpec Class

```python
class OutputSpec:
    def __init__(
        self,
        logical_name: str,
        aliases: List[str],
        output_type: DependencyType,
        property_path: str,
        data_type: str,
        description: str
    )
```

**Parameters:**
- `logical_name`: Logical name for the output (must match contract)
- `aliases`: Alternative names for the output
- `output_type`: Type of output (PROCESSING_OUTPUT, TRAINING_OUTPUT, etc.)
- `property_path`: SageMaker property path for accessing the output
- `data_type`: Output data type
- `description`: Human-readable description

**Example:**
```python
FEATURE_SELECTION_SPEC = StepSpecification(
    step_type=get_spec_step_type("FeatureSelection"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_feature_selection_contract(),
    dependencies={
        "input_data": DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep"],
            semantic_keywords=["data", "processed", "tabular", "features"],
            data_type="S3Uri",
            description="Preprocessed tabular data for feature selection"
        )
    },
    outputs={
        "selected_features": OutputSpec(
            logical_name="selected_features",
            aliases=["features", "reduced_features"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['selected_features'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Selected subset of features"
        )
    }
)
```

**Related Documentation:**
- [Step Specification Guide](../../0_developer_guide/step_specification.md)
- [Dependency Resolution System](../../1_design/dependency_resolution_system.md)

### Configuration API

Configuration classes implement the three-tier design pattern for managing step parameters with portable path support.

#### ProcessingStepConfigBase Class

```python
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
from pydantic import BaseModel, Field, field_validator, PrivateAttr

class YourStepConfig(ProcessingStepConfigBase):
    """
    Configuration following three-tier design with portable path support:
    - Tier 1: Essential fields (required user inputs)
    - Tier 2: System fields (defaults, can be overridden)
    - Tier 3: Derived fields (computed from other fields)
    
    Inherits portable path capabilities:
    - portable_processing_source_dir: Relative path to processing source directory
    - portable_effective_source_dir: Effective source directory with portable paths
    - get_portable_script_path(): Script path with portable path support
    """
```

#### Three-Tier Field Classification

**Tier 1: Essential Fields (Required User Inputs)**
```python
# No defaults, must be provided by user
selection_method: str = Field(..., description="Selection method")
n_features: int = Field(..., ge=1, description="Number of features")
target_column: str = Field(..., description="Target column name")
```

**Tier 2: System Fields (Defaults, Can Be Overridden)**
```python
# Sensible defaults, user can override
instance_type: str = Field(default="ml.m5.xlarge", description="Instance type")
instance_count: int = Field(default=1, ge=1, description="Instance count")
debug_mode: bool = Field(default=False, description="Enable debug mode")
```

**Tier 3: Derived Fields (Computed Properties)**
```python
# Private attributes with property access
_script_path: Optional[str] = PrivateAttr(default=None)
_output_path: Optional[str] = PrivateAttr(default=None)

@property
def script_path(self) -> str:
    """Get script path."""
    if self._script_path is None:
        self._script_path = "your_script.py"
    return self._script_path

@property
def output_path(self) -> str:
    """Get output path."""
    if self._output_path is None:
        self._output_path = f"{self.pipeline_s3_loc}/your_step/{self.region}"
    return self._output_path
```

#### Field Validation

```python
@field_validator('selection_method')
@classmethod
def validate_selection_method(cls, v: str) -> str:
    """Validate selection method is supported."""
    valid_methods = ["mutual_info", "correlation", "tree_based"]
    if v not in valid_methods:
        raise ValueError(f"Selection method must be one of: {valid_methods}")
    return v
```

#### Contract Integration

```python
def get_script_contract(self):
    """Return the script contract for this step."""
    from ..contracts.your_step_contract import YOUR_STEP_CONTRACT
    return YOUR_STEP_CONTRACT
```

**Related Documentation:**
- [Three-Tier Config Design](../../0_developer_guide/three_tier_config_design.md)
- [Config Field Manager Guide](../../0_developer_guide/config_field_manager_guide.md)

### Step Builder API

Step builders connect all components and create SageMaker steps.

#### StepBuilderBase Class

```python
from cursus.core.base.builder_base import StepBuilderBase
from sagemaker.workflow.steps import ProcessingStep

class YourStepBuilder(StepBuilderBase):
    def __init__(
        self,
        config: YourStepConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None
    ):
        """Initialize the step builder with configuration and specification."""
        super().__init__(
            config=config,
            spec=YOUR_STEP_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
```

#### Key Methods

**validate_configuration()**
```python
def validate_configuration(self) -> None:
    """Validate the provided configuration."""
    required_attrs = ['selection_method', 'n_features', 'target_column']
    
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"Config missing required attribute: {attr}")
```

**_create_processor()**
```python
def _create_processor(self):
    """Create and return a SageMaker processor."""
    from sagemaker.sklearn import SKLearnProcessor
    
    return SKLearnProcessor(
        framework_version="0.23-1",
        role=self.role,
        instance_type=self.config.instance_type,
        instance_count=self.config.instance_count,
        volume_size_in_gb=self.config.volume_size_gb,
        max_runtime_in_seconds=self.config.max_runtime_seconds,
        base_job_name=self._generate_job_name(),
        sagemaker_session=self.session,
        env=self._get_environment_variables()
    )
```

**_get_environment_variables()**
```python
def _get_environment_variables(self) -> Dict[str, str]:
    """Get environment variables for the processor."""
    # Get base environment variables from contract
    env_vars = super()._get_environment_variables()
    
    # Add step-specific environment variables
    step_env_vars = {
        "SELECTION_METHOD": self.config.selection_method,
        "N_FEATURES": str(self.config.n_features),
        "TARGET_COLUMN": self.config.target_column
    }
    
    env_vars.update(step_env_vars)
    return env_vars
```

**_get_inputs()**
```python
def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    """Get inputs for the step using specification and contract."""
    processing_inputs = []
    
    for logical_name, dependency_spec in self.spec.dependencies.items():
        if not dependency_spec.required and logical_name not in inputs:
            continue
            
        if dependency_spec.required and logical_name not in inputs:
            raise ValueError(f"Required input '{logical_name}' not provided")
        
        container_path = self.contract.expected_input_paths[logical_name]
        processing_inputs.append(ProcessingInput(
            input_name=logical_name,
            source=inputs[logical_name],
            destination=container_path
        ))
        
    return processing_inputs
```

**_get_outputs()**
```python
def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
    """Get outputs for the step using specification and contract."""
    processing_outputs = []
    
    for logical_name, output_spec in self.spec.outputs.items():
        container_path = self.contract.expected_output_paths[logical_name]
        
        destination = outputs.get(logical_name, f"{self.config.output_path}/{logical_name}/")
        
        processing_outputs.append(ProcessingOutput(
            output_name=logical_name,
            source=container_path,
            destination=destination
        ))
        
    return processing_outputs
```

**create_step()**
```python
def create_step(self, **kwargs) -> ProcessingStep:
    """Create the ProcessingStep with portable path support."""
    inputs_raw = kwargs.get('inputs', {})
    outputs = kwargs.get('outputs', {})
    dependencies = kwargs.get('dependencies', [])
    enable_caching = kwargs.get('enable_caching', True)
    
    # Handle inputs
    inputs = {}
    if dependencies:
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        inputs.update(extracted_inputs)
    inputs.update(inputs_raw)
    
    # Create processor and get inputs/outputs
    processor = self._create_processor()
    proc_inputs = self._get_inputs(inputs)
    proc_outputs = self._get_outputs(outputs)
    
    # Get script path with portable path support
    script_path = self.config.get_portable_script_path() or self.config.get_script_path()
    
    self.log_info("Using script path: %s (portable: %s)", 
                 script_path, 
                 "yes" if self.config.get_portable_script_path() else "no")
    
    # Create step
    step = ProcessingStep(
        name=self._get_step_name(),
        processor=processor,
        inputs=proc_inputs,
        outputs=proc_outputs,
        code=script_path,
        depends_on=dependencies,
        cache_config=self._get_cache_config(enable_caching)
    )
    
    # Attach specification for future reference
    setattr(step, '_spec', self.spec)
    return step
```

**Related Documentation:**
- [Step Builder Guide](../../0_developer_guide/step_builder.md)
- [Step Builder Registry Guide](../../0_developer_guide/step_builder_registry_guide.md)

## Registry API

The registry system manages step discovery and registration.

### UnifiedRegistryManager

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry instance
registry = UnifiedRegistryManager()

# Set workspace context
registry.set_workspace_context("main")  # or project name

# Register step builder (automatic discovery)
# Steps are automatically registered based on naming conventions

# Manual registration (if needed)
registry.register_step_builder("YourStepType", YourStepStepBuilder)
```

### Step Registration with Validation

```python
from cursus.registry.step_names import add_new_step_with_validation

warnings = add_new_step_with_validation(
    step_name="YourStep",
    config_class="YourStepConfig",
    builder_name="YourStepStepBuilder",
    sagemaker_type="Processing",
    description="Your step description",
    validation_mode="warn"
)
```

**Related Documentation:**
- [Registry API Reference](../registry/api_reference.md)
- [Step Builder Registry Usage](../../0_developer_guide/step_builder_registry_usage.md)

## Pipeline Integration API

### PipelineDAG

```python
from cursus.api.dag.base_dag import PipelineDAG

# Create DAG
dag = PipelineDAG()

# Add nodes
dag.add_node("PreprocessingStep")
dag.add_node("YourStep")
dag.add_node("TrainingStep")

# Add edges (dependencies)
dag.add_edge("PreprocessingStep", "YourStep")
dag.add_edge("YourStep", "TrainingStep")
```

### PipelineDAGCompiler

```python
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler
from sagemaker.workflow.parameters import ParameterString

# Define pipeline parameters for runtime configuration
pipeline_parameters = [
    ParameterString(name="EXECUTION_S3_PREFIX", default_value="s3://your-bucket/temp"),
    ParameterString(name="KMS_ENCRYPTION_KEY_PARAM", default_value=""),
    ParameterString(name="SECURITY_GROUP_ID", default_value=""),
    ParameterString(name="VPC_SUBNET", default_value=""),
]

# Create compiler with pipeline parameters
compiler = PipelineDAGCompiler(
    config_path="config.json",
    pipeline_parameters=pipeline_parameters,  # Enable runtime parameter injection
    sagemaker_session=pipeline_session,
    role=role
)

# Validate DAG
validation = compiler.validate_dag_compatibility(dag)

# Compile pipeline with runtime parameter support
pipeline, report = compiler.compile_with_report(dag=dag)
```

**Related Documentation:**
- [SageMaker Pipeline API Reference](../main/sagemaker_pipeline_api_reference.md)
- [Pipeline Compiler Design](../../1_design/pipeline_compiler.md)

## Validation API

### Alignment Validation

```python
# CLI commands for validation
python -m cursus.cli.alignment_cli validate your_step --verbose --show-scoring
python -m cursus.cli.alignment_cli visualize your_step --output-dir ./reports
```

### Builder Testing

```python
# CLI commands for builder testing
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_step.YourStepBuilder --scoring --verbose
python -m cursus.cli.builder_test_cli test-by-type Processing --verbose
```

### Runtime Testing

```python
# CLI commands for runtime testing
cursus runtime test-script your_step --workspace-dir ./test_workspace --verbose
cursus runtime test-compatibility preprocessing_step your_step --workspace-dir ./test_workspace
```

**Related Documentation:**
- [Validation Framework Guide](../../0_developer_guide/validation_framework_guide.md)
- [Alignment Rules](../../0_developer_guide/alignment_rules.md)

## Error Handling

### Common Exceptions

```python
from cursus.core.exceptions import (
    ConfigurationError,
    CompilationError,
    ValidationError,
    DependencyResolutionError
)

try:
    step = builder.create_step(inputs=inputs)
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Validation Results

```python
from cursus.core.compiler.validation import ValidationResult

validation = compiler.validate_dag_compatibility(dag)

if not validation.is_valid:
    print(f"Validation failed: {validation.issues}")
    print(f"Missing configs: {validation.missing_configs}")
    print(f"Unresolvable builders: {validation.unresolvable_builders}")
```

## Best Practices

### Naming Conventions

- **Files**: `builder_your_step.py`, `config_your_step.py`, `your_step_spec.py`
- **Classes**: `YourStepBuilder`, `YourStepConfig`, `YOUR_STEP_SPEC`
- **Logical Names**: Use snake_case, be descriptive (`input_data`, `selected_features`)
- **Environment Variables**: Use UPPER_CASE with underscores (`SELECTION_METHOD`)

### Error Handling Patterns

```python
def create_step(self, **kwargs):
    """Create step with comprehensive error handling."""
    try:
        # Validate configuration
        self.validate_configuration()
        
        # Create step
        step = self._create_sagemaker_step(**kwargs)
        
        return step
        
    except ValueError as e:
        raise ConfigurationError(f"Invalid configuration: {e}")
    except Exception as e:
        raise CompilationError(f"Failed to create step: {e}")
```

### Testing Patterns

```python
import unittest
from unittest.mock import MagicMock, patch

class TestYourStepBuilder(unittest.TestCase):
    def setUp(self):
        self.config = YourStepConfig(
            selection_method="mutual_info",
            n_features=20,
            target_column="target"
        )
        self.builder = YourStepBuilder(self.config)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        self.builder.validate_configuration()  # Should not raise
        
        # Test invalid config
        invalid_config = YourStepConfig(
            selection_method="invalid",
            n_features=20,
            target_column="target"
        )
        with self.assertRaises(ValueError):
            invalid_config.validate_selection_method("invalid")
    
    @patch('your_module.SKLearnProcessor')
    def test_create_step(self, mock_processor):
        """Test step creation."""
        inputs = {"input_data": "s3://bucket/input"}
        step = self.builder.create_step(inputs=inputs)
        
        self.assertIsNotNone(step)
        self.assertTrue(hasattr(step, '_spec'))
```

## Integration Examples

### Complete Step Implementation

```python
# 1. Contract
YOUR_STEP_CONTRACT = ScriptContract(
    entry_point="your_script.py",
    expected_input_paths={"input_data": "/opt/ml/processing/input/data"},
    expected_output_paths={"output_data": "/opt/ml/processing/output/data"},
    required_env_vars=["PARAM1", "PARAM2"]
)

# 2. Specification
YOUR_STEP_SPEC = StepSpecification(
    step_type="YourStep",
    node_type=NodeType.INTERNAL,
    script_contract=YOUR_STEP_CONTRACT,
    dependencies={"input_data": DependencySpec(...)},
    outputs={"output_data": OutputSpec(...)}
)

# 3. Configuration
class YourStepConfig(BasePipelineConfig):
    param1: str = Field(..., description="Required parameter")
    param2: int = Field(default=10, description="Optional parameter")

# 4. Builder
class YourStepBuilder(StepBuilderBase):
    def __init__(self, config: YourStepConfig, **kwargs):
        super().__init__(config=config, spec=YOUR_STEP_SPEC, **kwargs)
    
    def create_step(self, **kwargs) -> ProcessingStep:
        # Implementation here
        pass

# 5. Usage
config = YourStepConfig(param1="value")
builder = YourStepBuilder(config)
step = builder.create_step(inputs={"input_data": "s3://bucket/input"})
```

### Pipeline Integration

```python
# Create DAG with your step
dag = PipelineDAG()
dag.add_node("PreprocessingStep")
dag.add_node("YourStep")
dag.add_node("TrainingStep")

dag.add_edge("PreprocessingStep", "YourStep")
dag.add_edge("YourStep", "TrainingStep")

# Compile and execute
compiler = PipelineDAGCompiler(config_path="config.json", session=session, role=role)
pipeline, report = compiler.compile_with_report(dag=dag)

pipeline.upsert()
execution = pipeline.start()
```

## Related Documentation

### Core Documentation
- **[Getting Started Tutorial](getting_started.md)** - Step-by-step tutorial for creating new steps
- **[Developer Guide](../../0_developer_guide/README.md)** - Comprehensive development documentation
- **[Design Principles](../../0_developer_guide/design_principles.md)** - Core architectural principles

### Component Guides
- **[Script Contract Guide](../../0_developer_guide/script_contract.md)** - Detailed contract development
- **[Step Specification Guide](../../0_developer_guide/step_specification.md)** - Specification patterns
- **[Step Builder Guide](../../0_developer_guide/step_builder.md)** - Builder implementation
- **[Three-Tier Config Design](../../0_developer_guide/three_tier_config_design.md)** - Configuration architecture

### Advanced Topics
- **[Validation Framework](../../0_developer_guide/validation_framework_guide.md)** - Comprehensive validation
- **[Registry System](../../0_developer_guide/step_builder_registry_guide.md)** - Step registration
- **[Dependency Resolution](../../1_design/dependency_resolution_system.md)** - How dependencies work
- **[Alignment Rules](../../0_developer_guide/alignment_rules.md)** - Component alignment requirements

### Reference Materials
- **[SageMaker Property Paths](../../0_developer_guide/sagemaker_property_path_reference_database.md)** - Complete property reference
- **[Best Practices](../../0_developer_guide/best_practices.md)** - Development best practices
- **[Common Pitfalls](../../0_developer_guide/common_pitfalls.md)** - Avoiding common mistakes
- **[Standardization Rules](../../0_developer_guide/standardization_rules.md)** - Coding standards

### Architecture and Design References
- **[Unified Step Catalog System Design](../../1_design/unified_step_catalog_system_design.md)** - Core discovery system architecture
- **[Cursus Package Portability Architecture Design](../../1_design/cursus_package_portability_architecture_design.md)** - Universal deployment compatibility and runtime parameter support
- **[Config Portability Path Resolution Design](../../1_design/config_portability_path_resolution_design.md)** - Portable path resolution system design
- **[Pipeline Execution Temp Dir Integration](../../1_design/pipeline_execution_temp_dir_integration.md)** - Runtime parameter flow architecture

### API References
- **[Core API Reference](../../core/api_reference.md)** - Core component APIs
- **[Registry API Reference](../registry/api_reference.md)** - Registry system APIs
- **[Validation API Reference](../validation/api_reference.md)** - Validation framework APIs
- **[SageMaker Pipeline API](../main/sagemaker_pipeline_api_reference.md)** - Pipeline compilation APIs

This API reference provides the essential interfaces and patterns for developing custom pipeline steps in Cursus. For detailed examples and step-by-step guidance, see the [Getting Started Tutorial](getting_started.md).
