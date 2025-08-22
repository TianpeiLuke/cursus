---
tags:
  - code
  - mods_integration
  - compiler
  - dag_compiler
  - implementation
keywords:
  - MODSPipelineDAGCompiler
  - compile_mods_dag_to_pipeline
  - MODSTemplate decorator
  - metaclass conflict
  - pipeline compilation
  - MODS metadata extraction
topics:
  - MODS compiler implementation
  - pipeline compilation
  - template decoration
language: python
date of note: 2025-08-22
---

# MODS DAG Compiler Implementation

## Overview

The `mods_dag_compiler.py` module provides the core implementation for MODS-integrated pipeline compilation. This module contains the `MODSPipelineDAGCompiler` class and the `compile_mods_dag_to_pipeline` convenience function that enable seamless integration between Cursus pipelines and the Model Operations and Deployment Service (MODS).

## Module Structure

### Location
- **File**: `src/cursus/mods/compiler/mods_dag_compiler.py`
- **Module**: `cursus.mods.compiler.mods_dag_compiler`

### Exports
- `MODSPipelineDAGCompiler`: Advanced API for DAG-to-template compilation with MODS integration
- `compile_mods_dag_to_pipeline`: Convenience function for simple one-call compilation

## Key Components

### 1. MODSPipelineDAGCompiler Class

#### Purpose
Advanced API for DAG-to-template compilation with MODS integration. This class extends `PipelineDAGCompiler` to enable MODS integration with dynamically generated pipelines, solving the metaclass conflict issue that occurs when trying to apply the `MODSTemplate` decorator to an instance of `DynamicPipelineTemplate`.

#### Class Hierarchy
```python
MODSPipelineDAGCompiler(PipelineDAGCompiler)
```

#### Constructor Parameters
- `config_path: str` - Path to configuration file
- `sagemaker_session: Optional[PipelineSession]` - SageMaker session for pipeline execution
- `role: Optional[str]` - IAM role for pipeline execution
- `config_resolver: Optional[StepConfigResolver]` - Custom config resolver
- `builder_registry: Optional[StepBuilderRegistry]` - Custom builder registry
- `**kwargs` - Additional arguments for template constructor

#### Core Methods

##### _get_base_config()
```python
def _get_base_config(self) -> BasePipelineConfig
```

**Purpose**: Extract base configuration from the configuration file for MODS metadata extraction.

**Implementation Details**:
- Creates a minimal test DAG to validate config loading
- Uses parent's `create_template` with `skip_validation=True`
- Attempts multiple strategies to locate base configuration:
  1. Direct access via `_get_base_config()` method
  2. Search by name ('base') in configs dictionary
  3. Search by type (`BasePipelineConfig`)
  4. Fallback to first available configuration
- Provides comprehensive error handling and logging

**Returns**: `BasePipelineConfig` object containing pipeline metadata

**Raises**: `ConfigurationError` if base configuration cannot be found or loaded

##### create_decorated_class()
```python
def create_decorated_class(
    self,
    dag=None,
    author: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None
) -> Type
```

**Purpose**: Create and return the MODSTemplate-decorated DynamicPipelineTemplate class.

**Implementation Details**:
- Imports `DynamicPipelineTemplate` to avoid circular imports
- Extracts metadata from base configuration if not provided
- Applies fallback defaults for missing metadata:
  - `author`: 'Unknown'
  - `version`: '1.0.0'
  - `description`: 'MODS Pipeline'
- Applies `MODSTemplate` decorator to the class (not instance)
- Includes comprehensive error handling and logging

**Parameters**:
- `dag`: Optional pipeline DAG (not used for class creation)
- `author`: Author name for MODS metadata
- `version`: Version for MODS metadata
- `description`: Description for MODS metadata

**Returns**: The DynamicPipelineTemplate class decorated with MODSTemplate

##### create_template_params()
```python
def create_template_params(self, dag: PipelineDAG, **template_kwargs) -> Dict[str, Any]
```

**Purpose**: Create and return parameters needed to instantiate a template.

**Implementation Details**:
- Merges standard parameters with provided kwargs
- Standard parameters include:
  - `dag`: Pipeline DAG
  - `config_path`: Configuration file path
  - `config_resolver`: Step configuration resolver
  - `builder_registry`: Step builder registry
  - `sagemaker_session`: SageMaker session
  - `role`: IAM execution role

**Parameters**:
- `dag`: Pipeline DAG to compile
- `**template_kwargs`: Additional template parameters

**Returns**: Dictionary of parameters for template instantiation

##### create_template()
```python
def create_template(self, dag: PipelineDAG, **kwargs) -> Any
```

**Purpose**: Create a MODS template instance with the given DAG. Overrides the parent method to handle MODS integration.

**Implementation Details**:
- Extracts MODS metadata parameters from kwargs
- Gets the decorated class using `create_decorated_class()`
- Creates template parameters using `create_template_params()`
- Instantiates the decorated template class
- Includes comprehensive logging and error handling

**Parameters**:
- `dag`: PipelineDAG instance to create a template for
- `**kwargs`: Additional arguments including author, version, description

**Returns**: MODS-decorated template instance ready for pipeline generation

##### compile_and_fill_execution_doc()
```python
def compile_and_fill_execution_doc(
    self, 
    dag: PipelineDAG, 
    execution_doc: Dict[str, Any],
    pipeline_name: Optional[str] = None,
    **kwargs
) -> Tuple[Pipeline, Dict[str, Any]]
```

**Purpose**: Compile a DAG to MODS pipeline and fill an execution document in one step.

**Implementation Details**:
- Ensures proper sequencing of pipeline generation and execution document filling
- Compiles the pipeline first (stores template internally)
- Uses stored template to fill execution document
- Addresses timing issues with template metadata

**Parameters**:
- `dag`: PipelineDAG instance to compile
- `execution_doc`: Execution document template to fill
- `pipeline_name`: Optional pipeline name override
- `**kwargs`: Additional arguments for template

**Returns**: Tuple of (compiled_pipeline, filled_execution_doc)

### 2. compile_mods_dag_to_pipeline Function

#### Purpose
Main entry point for users who want simple, one-call compilation from DAG to MODS-compatible pipeline.

#### Function Signature
```python
def compile_mods_dag_to_pipeline(
    dag: PipelineDAG,
    config_path: str,
    sagemaker_session: Optional[PipelineSession] = None,
    role: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    **kwargs
) -> Pipeline
```

#### Implementation Details
- Validates input parameters before processing
- Checks DAG instance type and node count
- Validates configuration file existence
- Creates `MODSPipelineDAGCompiler` instance
- Uses compiler's `compile` method for pipeline generation
- Includes comprehensive error handling and logging

#### Parameters
- `dag`: PipelineDAG instance defining the pipeline structure
- `config_path`: Path to configuration file containing step configs
- `sagemaker_session`: SageMaker session for pipeline execution
- `role`: IAM role for pipeline execution
- `pipeline_name`: Optional pipeline name override
- `**kwargs`: Additional arguments passed to template constructor

#### Returns
Generated SageMaker Pipeline ready for execution, decorated with MODSTemplate

#### Error Handling
- `ValueError`: If DAG is invalid or empty
- `FileNotFoundError`: If configuration file doesn't exist
- `ConfigurationError`: If configuration validation fails
- `RegistryError`: If step builders not found for config types
- `PipelineAPIError`: General compilation failures

## Technical Implementation Details

### Metaclass Conflict Resolution

The core technical challenge addressed by this implementation is the metaclass conflict that occurs when applying the `MODSTemplate` decorator to instances of `DynamicPipelineTemplate`. The solution involves:

1. **Class-Level Decoration**: Apply the decorator to the class before instantiation
2. **Proper Sequencing**: Ensure correct order of decoration and template creation
3. **Metadata Extraction**: Automatically extract MODS metadata from configuration

### MODS Import Handling

The module includes graceful fallback handling for MODS dependencies:

```python
try:
    from mods.mods_template import MODSTemplate
except ImportError:
    # Placeholder decorator for testing when MODS is not available
    def MODSTemplate(author=None, description=None, version=None):
        def decorator(cls):
            return cls
        return decorator
```

This ensures the module can function even when MODS is not installed, making it suitable for development and testing environments.

### Error Handling Strategy

The implementation includes multiple layers of error handling:

1. **Input Validation**: Comprehensive validation of all input parameters
2. **Configuration Errors**: Specific handling for configuration-related issues
3. **Import Errors**: Graceful fallback for missing dependencies
4. **Template Creation Errors**: Detailed error reporting for template instantiation
5. **Compilation Errors**: Clear error messages for pipeline compilation failures

### Logging Strategy

The module uses structured logging throughout:

- **Info Level**: Successful operations and progress updates
- **Warning Level**: Fallback scenarios and missing metadata
- **Error Level**: Compilation failures and error details

## Dependencies

### Internal Dependencies
- `cursus.api.dag.base_dag.PipelineDAG`
- `cursus.core.compiler.dag_compiler.PipelineDAGCompiler`
- `cursus.core.compiler.config_resolver.StepConfigResolver`
- `cursus.steps.registry.builder_registry.StepBuilderRegistry`
- `cursus.core.compiler.exceptions.PipelineAPIError`
- `cursus.core.compiler.exceptions.ConfigurationError`
- `cursus.core.base.config_base.BasePipelineConfig`
- `cursus.core.compiler.dynamic_template.DynamicPipelineTemplate`

### External Dependencies
- `mods.mods_template.MODSTemplate` (with graceful fallback)
- `sagemaker.workflow.pipeline.Pipeline`
- `sagemaker.workflow.pipeline_context.PipelineSession`
- `typing` - Type hints
- `logging` - Logging functionality
- `pathlib.Path` - File path handling

## Usage Patterns

### Simple Compilation Pattern
```python
from cursus.mods.compiler import compile_mods_dag_to_pipeline

pipeline = compile_mods_dag_to_pipeline(
    dag=my_dag,
    config_path="config.json",
    sagemaker_session=session,
    role=role
)
```

### Advanced Compilation Pattern
```python
from cursus.mods.compiler import MODSPipelineDAGCompiler

compiler = MODSPipelineDAGCompiler(
    config_path="config.json",
    sagemaker_session=session,
    role=role
)

decorated_class = compiler.create_decorated_class(
    author="Team",
    version="1.0",
    description="Pipeline"
)

template = compiler.create_template(dag)
pipeline = compiler.compile(dag)
```

### Execution Document Pattern
```python
compiler = MODSPipelineDAGCompiler(config_path="config.json")

pipeline, filled_doc = compiler.compile_and_fill_execution_doc(
    dag=dag,
    execution_doc=template_doc,
    pipeline_name="my-pipeline"
)
```

## Testing Considerations

The implementation is designed with testability in mind:

1. **Dependency Injection**: All dependencies can be injected for testing
2. **Graceful Fallbacks**: MODS dependency is optional for testing
3. **Error Simulation**: Comprehensive error handling enables error simulation
4. **Logging Verification**: Structured logging enables test verification
5. **Method Isolation**: Individual methods can be tested independently

## Performance Considerations

The implementation includes several performance optimizations:

1. **Lazy Imports**: Dynamic template import to avoid circular dependencies
2. **Minimal DAG Creation**: Test DAG creation only when needed for config extraction
3. **Efficient Error Handling**: Early validation to avoid expensive operations
4. **Logging Optimization**: Conditional logging to reduce overhead

## Related Documentation

- [MODS Compiler Overview](README.md) - High-level compiler documentation
- [MODS Integration Module](../README.md) - Module overview and architecture
- [MODS DAG Compiler Design](../../1_design/mods_dag_compiler_design.md) - Design decisions
- [Implementation Plan](../../2_project_planning/2025-08-19_mods_pipeline_dag_compiler_implementation_plan.md) - Development roadmap

## Conclusion

The `mods_dag_compiler.py` module provides a robust, flexible implementation for MODS-integrated pipeline compilation. By solving the metaclass conflict challenge and providing comprehensive error handling, it enables seamless integration between Cursus pipelines and the MODS ecosystem while maintaining high code quality and testability standards.
