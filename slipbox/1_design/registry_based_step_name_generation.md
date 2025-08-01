---
tags:
  - design
  - implementation
  - pipeline_registry
  - naming_conventions
keywords:
  - step name registry
  - centralized naming
  - job type variants
  - single source of truth
  - name generation
  - consistent naming
  - registry integration
topics:
  - registry design
  - naming consistency
  - pipeline integration
  - system-wide standardization
language: python
date of note: 2025-07-31
---

# Registry-Based Step Name Generation

## Overview

This design document describes the comprehensive registry-based step name generation system implemented across the entire pipeline framework. This system ensures consistent step naming across all pipeline components by using the pipeline registry as the single source of truth for step names.

## Purpose

The purpose of registry-based step name generation is to:

1. **Establish a single source of truth** for step names across all pipeline components
2. **Eliminate naming mismatches** that cause validation failures
3. **Support job type variants** with a reliable naming scheme
4. **Ensure consistency** between saving and loading configurations
5. **Simplify maintenance** by centralizing step name definitions

This design complements the [step builder registry design](step_builder_registry_design.md) and provides a foundation for consistent naming across [configurations](config.md), [step builders](step_builder.md), [step specifications](step_specification.md), and the [step config resolver](step_config_resolver.md).

## Central Registry Architecture

### Core Registry: Single Source of Truth

The core of the system is the central registry defined in `src/pipeline_registry/step_names.py`:

```python
# src/pipeline_registry/step_names.py
STEP_NAMES = {
    "PyTorchTraining": {
        "config_class": "PyTorchTrainingConfig",           # For config registry
        "builder_step_name": "PyTorchTrainingStepBuilder", # For builder registry
        "spec_type": "PyTorchTraining",                    # For StepSpecification.step_type
        "description": "PyTorch model training step"
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig", 
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "spec_type": "XGBoostTraining",
        "description": "XGBoost model training step"
    },
    # ... other steps
}

# Convenient mappings for different components
CONFIG_STEP_REGISTRY = {
    info["config_class"]: step_name 
    for step_name, info in STEP_NAMES.items()
}

BUILDER_STEP_NAMES = {
    info["builder_step_name"]: step_name
    for step_name, info in STEP_NAMES.items()
}

# Helper function for step specifications
def get_spec_step_type(step_name: str) -> str:
    """Get the step_type for step specifications."""
    if step_name in STEP_NAMES:
        return STEP_NAMES[step_name]["spec_type"]
    return step_name

# Helper for job type variants
def get_spec_step_type_with_job_type(step_name: str, job_type: str = None) -> str:
    """Get step_type with optional job_type suffix."""
    base_type = get_spec_step_type(step_name)
    if job_type:
        return f"{base_type}_{job_type.capitalize()}"
    return base_type
```

### System-wide Integration

This central registry is integrated across all pipeline components:

1. **Configuration System**: For serialization, deserialization, and loading through [config resolution](config_resolution_enhancements.md) and the [step config resolver](step_config_resolver.md)
2. **Builder System**: For step builder creation and registry through the [step builder registry](step_builder_registry_design.md)
3. **Step Specification System**: For defining step types and dependencies in [step specifications](step_specification.md)
4. **Pipeline Template System**: For building pipelines with consistent names via [step builders](step_builder.md)

## Core Components Integration

### 1. Configuration System Integration

#### 1.1. Type-Aware Config Serializer

The `TypeAwareConfigSerializer` class uses the registry to generate consistent step names:

```python
def generate_step_name(self, config: Any) -> str:
    """Generate a step name for a config, including job type and other attributes."""
    # First check for step_name_override - highest priority
    if hasattr(config, "step_name_override") and config.step_name_override != config.__class__.__name__:
        return config.step_name_override
        
    # Get class name
    class_name = config.__class__.__name__
    
    # Look up the step name from the registry (primary source of truth)
    try:
        from src.pipeline_registry.step_names import CONFIG_STEP_REGISTRY
        if class_name in CONFIG_STEP_REGISTRY:
            base_step = CONFIG_STEP_REGISTRY[class_name]
        else:
            # Fall back to the old behavior if not in registry
            from src.pipeline_steps.config_base import BasePipelineConfig
            base_step = BasePipelineConfig.get_step_name(class_name)
    except (ImportError, AttributeError):
        # If registry not available, fall back to the old behavior
        from src.pipeline_steps.config_base import BasePipelineConfig
        base_step = BasePipelineConfig.get_step_name(class_name)
    
    # Append distinguishing attributes - essential for job type variants
    for attr in ("job_type", "data_type", "mode"):
        if hasattr(config, attr):
            val = getattr(config, attr)
            if val is not None:
                step_name = f"{step_name}_{val}"
                
    return step_name
```

#### 1.2. Config Base

The base [configuration](config.md) class directly imports from the central registry:

```python
# src/pipeline_steps/config_base.py
from ..pipeline_registry.step_names import CONFIG_STEP_REGISTRY as STEP_REGISTRY

class BasePipelineConfig(BaseModel):
    """Base configuration for pipeline steps."""
    
    @staticmethod
    def get_step_name(class_name: str) -> str:
        """Get the step name for a class using the registry."""
        if class_name in STEP_REGISTRY:
            return STEP_REGISTRY[class_name]
            
        # Fall back to removing Config suffix
        if class_name.endswith("Config"):
            return class_name[:-6]
        return class_name
```

#### 1.3. Config Merger

The `ConfigMerger` class uses the type-aware serializer for step name generation:

```python
def _generate_step_name(self, config: Any) -> str:
    """Generate a consistent step name for a config object."""
    serializer = TypeAwareConfigSerializer()
    return serializer.generate_step_name(config)
```

### 2. Builder System Integration

#### 2.1. Builder Step Base

The [step builder](step_builder.md) base imports step names from the central registry, as described in the [step builder registry design](step_builder_registry_design.md):

```python
# src/pipeline_steps/builder_step_base.py
from ..pipeline_registry.step_names import BUILDER_STEP_NAMES as STEP_NAMES

class StepBuilderBase:
    """Base class for all step builders."""
    
    def __init__(self, config, spec, sagemaker_session=None, role=None, notebook_root=None):
        """Initialize with configuration and specification."""
        self.config = config
        self.spec = spec
        self.sagemaker_session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root
        
        # Use step name from registry
        self.step_name = self._get_step_name_from_registry()
        
    def _get_step_name_from_registry(self) -> str:
        """Get step name from builder class name using registry."""
        builder_class_name = self.__class__.__name__
        if builder_class_name in STEP_NAMES:
            return STEP_NAMES[builder_class_name]
        return builder_class_name
```

#### 2.2. Builder Factory

The builder factory uses the registry to find the correct builder class:

```python
# src/pipeline_steps/builder_factory.py
from ..pipeline_registry.step_names import STEP_NAMES, CONFIG_STEP_REGISTRY

class StepBuilderFactory:
    """Factory for creating step builders."""
    
    @classmethod
    def create_builder(cls, config, sagemaker_session=None, role=None):
        """Create a step builder for the config."""
        config_type = config.__class__.__name__
        
        # Get step name from registry
        step_name = None
        if config_type in CONFIG_STEP_REGISTRY:
            step_name = CONFIG_STEP_REGISTRY[config_type]
        
        # Find builder class
        builder_name = None
        if step_name and step_name in STEP_NAMES:
            builder_name = STEP_NAMES[step_name]["builder_step_name"]
            
        # Get builder class
        builder_class = BUILDER_MAP.get(builder_name)
        if not builder_class:
            raise ValueError(f"No builder for config type {config_type}")
            
        # Create builder
        return builder_class(config, sagemaker_session=sagemaker_session, role=role)
```

### 3. Step Specification Integration

#### 3.1. Step Specification Base

The [step specification](step_specification.md) base uses the central registry:

```python
# src/pipeline_step_specs/step_specification.py
from ..pipeline_registry.step_names import get_spec_step_type

class StepSpecification:
    """Base class for step specifications."""
    
    def __init__(self, step_type, node_type, dependencies, outputs):
        """Initialize with step type, node type, dependencies, and outputs."""
        self.step_type = step_type
        self.node_type = node_type
        self.dependencies = dependencies
        self.outputs = outputs
```

#### 3.2. Step Specification Examples

Example of using the registry in step specifications:

```python
# src/pipeline_step_specs/pytorch_training_spec.py
from ..pipeline_registry.step_names import get_spec_step_type

PYTORCH_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("PytorchTraining"),
    node_type=NodeType.PROCESSING,
    dependencies=[
        # Dependencies...
    ],
    outputs=[
        # Outputs...
    ]
)
```

#### 3.3. Job Type Variant Specifications

Example of using job type variants in step specifications:

```python
# src/pipeline_step_specs/preprocessing_training_spec.py
from ..pipeline_registry.step_names import get_spec_step_type_with_job_type

TABULAR_PREPROCESSING_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("TabularPreprocessing", "training"),
    node_type=NodeType.PROCESSING,
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            source_step_type=get_spec_step_type_with_job_type("CradleDataLoading", "training"),
            source_output="DATA",
            description="Training data from Cradle",
            semantic_keywords=["training", "data", "input"]
        )
    ],
    outputs=[
        # Outputs...
    ]
)
```

### 4. Pipeline Template Integration

Pipeline templates reference step names consistently through the central registry, leveraging the [configuration resolution](config_resolution_enhancements.md) system:

```python
# src/pipeline_builder/template_pipeline_xgboost_end_to_end.py
from ..pipeline_registry.step_names import get_spec_step_type

class XGBoostEndToEndTemplate(PipelineTemplateBase):
    """Template for XGBoost end-to-end pipeline."""
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        """Create the pipeline DAG structure."""
        dag = PipelineDAG()
        
        # Add nodes with job type variants in node names
        dag.add_node(get_spec_step_type_with_job_type("CradleDataLoading", "training"))
        dag.add_node(get_spec_step_type_with_job_type("TabularPreprocessing", "training"))
        dag.add_node(get_spec_step_type("XGBoostTraining"))
        dag.add_node(get_spec_step_type("Package"))
        dag.add_node(get_spec_step_type("Payload"))
        dag.add_node(get_spec_step_type("Registration"))
        dag.add_node(get_spec_step_type_with_job_type("CradleDataLoading", "calibration"))
        dag.add_node(get_spec_step_type_with_job_type("TabularPreprocessing", "calibration"))
        
        # Add edges
        # ...
        
        return dag
```

## Job Type Variant Handling

### 1. Job Type Variant Naming Convention

Step names with job type variants follow a consistent naming convention:

```
{base_step_name}_{job_type}
```

Examples:
- `CradleDataLoading_training`
- `CradleDataLoading_calibration`
- `TabularPreprocessing_training`
- `TabularPreprocessing_validation`

### 2. Helper Functions

The central registry provides helper functions for job type variants:

```python
def get_spec_step_type_with_job_type(step_name: str, job_type: str = None) -> str:
    """Get step_type with optional job_type suffix."""
    base_type = get_spec_step_type(step_name)
    if job_type:
        return f"{base_type}_{job_type.capitalize()}"
    return base_type
```

### 3. Dynamic Specification Selection

Step builders dynamically select the appropriate specification based on job type:

```python
class CradleDataLoadingStepBuilder(StepBuilderBase):
    """Builder for Cradle Data Loading processing step."""
    
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None):
        # Determine which specification to use based on job type
        job_type = getattr(config, 'job_type', 'training').lower()
        
        # Select appropriate specification
        if job_type == 'calibration':
            spec = DATA_LOADING_CALIBRATION_SPEC
        elif job_type == 'validation':
            spec = DATA_LOADING_VALIDATION_SPEC
        elif job_type == 'testing':
            spec = DATA_LOADING_TESTING_SPEC
        else:  # Default to training
            spec = DATA_LOADING_TRAINING_SPEC
            
        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
```

## Script Contract Integration

Script contracts ensure consistent job type handling in processing scripts:

```python
# src/pipeline_script_contracts/data_loading_contract.py
CRADLE_DATA_LOADING_CONTRACT = ScriptContract(
    script_path="cradle_data_loading/process.py",
    expected_output_paths={
        "DATA": "/opt/ml/processing/output/data",
        "METADATA": "/opt/ml/processing/output/metadata",
        "SIGNATURE": "/opt/ml/processing/output/signature"
    },
    required_env_vars=[
        "JOB_TYPE",  # Must be one of: training, calibration, validation, testing
        "REGION",
        "DATA_SOURCE_TYPE"
    ],
    # ...
)
```

## Validation Tools

The system includes validation tools to ensure consistency:

```python
# tools/validate_step_names.py
def validate_step_name_consistency():
    """Validate consistency of step names across all components."""
    # Load central registry
    from src.pipeline_registry.step_names import STEP_NAMES, CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES
    
    # 1. Validate config registry
    for config_class, step_name in CONFIG_STEP_REGISTRY.items():
        if step_name not in STEP_NAMES:
            print(f"ERROR: Config class {config_class} maps to unknown step name {step_name}")
            
    # 2. Validate builder registry
    for builder_class, step_name in BUILDER_STEP_NAMES.items():
        if step_name not in STEP_NAMES:
            print(f"ERROR: Builder class {builder_class} maps to unknown step name {step_name}")
            
    # 3. Validate step specifications
    # ... code to validate spec_type values ...
    
    # 4. Validate templates
    # ... code to validate template step references ...
```

## Implementation Benefits

1. **Consistent Naming**: Uses the same canonical names across the system
2. **Improved Reliability**: Prevents validation failures during loading
3. **Better Maintainability**: Changes to step names only need to be made in the registry
4. **Single Source of Truth**: One definitive place for step name definitions
5. **Systematic Job Type Support**: Standardized approach to job type variants

This design works in harmony with the [step builder registry](step_builder_registry_design.md) system to ensure consistent naming throughout the [configuration](config.md), [step building](step_builder.md), and [specification](step_specification.md) subsystems.

## Error Handling

1. **Registry Not Found**: Falls back to legacy naming scheme (remove "Config" suffix)
2. **Class Not in Registry**: Falls back to legacy naming scheme
3. **Multiple Job Types**: Handles job type variants consistently by appending the job type
4. **Job Type Capitalization**: Consistent capitalization for job type variants

## Backward Compatibility

The implementation maintains backward compatibility through:

1. **Fallback Mechanisms**: When registry is not available or incomplete
2. **Legacy Support**: Continues to support removing "Config" suffix as fallback
3. **Gradual Integration**: Components adopt registry in phases
4. **Format Preservation**: Maintains expected step name format for all consumers

## Future Improvements

1. **Namespace Support**: Add support for namespaces to avoid name collisions
2. **Automatic Discovery**: Automatically register step classes based on naming conventions
3. **Configuration Validation**: Add validation to ensure all used step names are in registry
4. **Documentation Generation**: Generate documentation from registry metadata
5. **Job Type Registry**: Extend registry with explicit job type variant support

## References

- [Step Builder Registry Design](step_builder_registry_design.md) - Complementary registry design for step builders
- [Step Builder](step_builder.md) - Details on how step builders use the registry for naming
- [Configuration System](config.md) - Overview of the configuration system that uses these names
- [Step Specification](step_specification.md) - Specification system that relies on consistent step naming
- [Configuration Resolution](config_resolution_enhancements.md) - How configurations are resolved using the registry
- [Step Config Resolver](step_config_resolver.md) - Resolution of node names to configuration objects
- [Config Types Format](config_types_format.md) - Format standards for configuration types
- [Simplified Config Field Categorization](simplified_config_field_categorization.md) - Field organization in configs
- [Step Name Consistency Implementation Plan](../slipbox/project_planning/2025-07-07_step_name_consistency_implementation_plan.md)
- [Job Type Variant Solution](../slipbox/project_planning/2025-07-04_job_type_variant_solution.md)
