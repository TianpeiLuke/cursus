---
tags:
  - entry_point
  - code
  - mods_integration
  - compiler
  - documentation
keywords:
  - MODS compiler
  - DAG compilation
  - MODSTemplate decorator
  - metaclass conflict resolution
  - pipeline template decoration
topics:
  - MODS compiler
  - pipeline compilation
  - template decoration
language: python
date of note: 2025-08-22
---

# MODS Compiler

## Overview

The MODS Compiler provides specialized DAG-to-pipeline compilation with MODS (Model Operations and Deployment Service) integration. This compiler extends the standard `PipelineDAGCompiler` to resolve metaclass conflicts when applying the `MODSTemplate` decorator to dynamically generated pipeline templates.

## Architecture

The MODS compiler consists of two main components:

```
compiler/
├── __init__.py              # Module exports
└── mods_dag_compiler.py     # Core implementation
```

### Core Components

1. **MODSPipelineDAGCompiler**: Advanced API for DAG-to-template compilation with MODS integration
2. **compile_mods_dag_to_pipeline**: Convenience function for simple one-call compilation

## Key Features

### 1. Metaclass Conflict Resolution
The primary technical challenge solved by this compiler is the metaclass conflict that occurs when trying to apply the `MODSTemplate` decorator to an instance of `DynamicPipelineTemplate`. The solution involves:

- **Class-Level Decoration**: Apply the decorator to the class before instantiation
- **Proper Sequencing**: Ensure correct order of decoration and template creation
- **Metadata Extraction**: Automatically extract MODS metadata from configuration

### 2. Enhanced API Flexibility
Beyond basic compilation, the compiler provides:

- **Decorated Class Access**: Expose the decorated class for advanced use cases
- **Custom Template Parameters**: Support for specialized template instantiation
- **Execution Document Integration**: Proper handling of MODS execution documents

### 3. Robust Error Handling
- Graceful fallback when MODS is not available
- Comprehensive validation of inputs and configurations
- Clear error messages for troubleshooting

## API Reference

### MODSPipelineDAGCompiler Class

#### Constructor
```python
MODSPipelineDAGCompiler(
    config_path: str,
    sagemaker_session: Optional[PipelineSession] = None,
    role: Optional[str] = None,
    config_resolver: Optional[StepConfigResolver] = None,
    builder_registry: Optional[StepBuilderRegistry] = None,
    **kwargs
)
```

**Parameters:**
- `config_path`: Path to configuration file containing step configs
- `sagemaker_session`: SageMaker session for pipeline execution
- `role`: IAM role for pipeline execution
- `config_resolver`: Custom config resolver (optional)
- `builder_registry`: Custom builder registry (optional)
- `**kwargs`: Additional arguments for template constructor

#### Key Methods

##### create_decorated_class()
```python
create_decorated_class(
    dag=None,
    author: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None
) -> Type
```

Creates and returns the MODSTemplate-decorated DynamicPipelineTemplate class.

**Parameters:**
- `dag`: Optional pipeline DAG (for metadata extraction)
- `author`: Author name for MODS metadata (auto-extracted if None)
- `version`: Version for MODS metadata (auto-extracted if None)
- `description`: Description for MODS metadata (auto-extracted if None)

**Returns:** The DynamicPipelineTemplate class decorated with MODSTemplate

##### create_template_params()
```python
create_template_params(dag: PipelineDAG, **template_kwargs) -> Dict[str, Any]
```

Creates and returns parameters needed to instantiate a template.

**Parameters:**
- `dag`: Pipeline DAG to compile
- `**template_kwargs`: Additional template parameters

**Returns:** Dictionary of parameters for template instantiation

##### create_template()
```python
create_template(dag: PipelineDAG, **kwargs) -> Any
```

Creates a MODS template instance with the given DAG. This method overrides the parent method to handle MODS integration.

**Parameters:**
- `dag`: PipelineDAG instance to create a template for
- `**kwargs`: Additional arguments for template (including author, version, description)

**Returns:** MODS-decorated template instance ready for pipeline generation

##### compile_and_fill_execution_doc()
```python
compile_and_fill_execution_doc(
    dag: PipelineDAG, 
    execution_doc: Dict[str, Any],
    pipeline_name: Optional[str] = None,
    **kwargs
) -> Tuple[Pipeline, Dict[str, Any]]
```

Compiles a DAG to MODS pipeline and fills an execution document in one step.

**Parameters:**
- `dag`: PipelineDAG instance to compile
- `execution_doc`: Execution document template to fill
- `pipeline_name`: Optional pipeline name override
- `**kwargs`: Additional arguments for template

**Returns:** Tuple of (compiled_pipeline, filled_execution_doc)

### Convenience Function

#### compile_mods_dag_to_pipeline()
```python
compile_mods_dag_to_pipeline(
    dag: PipelineDAG,
    config_path: str,
    sagemaker_session: Optional[PipelineSession] = None,
    role: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    **kwargs
) -> Pipeline
```

Main entry point for users who want simple, one-call compilation from DAG to MODS-compatible pipeline.

**Parameters:**
- `dag`: PipelineDAG instance defining the pipeline structure
- `config_path`: Path to configuration file containing step configs
- `sagemaker_session`: SageMaker session for pipeline execution
- `role`: IAM role for pipeline execution
- `pipeline_name`: Optional pipeline name override
- `**kwargs`: Additional arguments passed to template constructor

**Returns:** Generated SageMaker Pipeline ready for execution, decorated with MODSTemplate

**Raises:**
- `ValueError`: If DAG nodes don't have corresponding configurations
- `ConfigurationError`: If configuration validation fails
- `RegistryError`: If step builders not found for config types

## Usage Examples

### Simple Usage
```python
from cursus.mods.compiler import compile_mods_dag_to_pipeline

# Create your DAG
dag = create_xgboost_pipeline_dag()

# Compile to MODS pipeline
pipeline = compile_mods_dag_to_pipeline(
    dag=dag,
    config_path="configs/my_pipeline.json",
    sagemaker_session=session,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Execute pipeline
pipeline.upsert()
```

### Advanced Usage
```python
from cursus.mods.compiler import MODSPipelineDAGCompiler

# Create compiler
compiler = MODSPipelineDAGCompiler(
    config_path="configs/my_pipeline.json",
    sagemaker_session=session,
    role=role
)

# Get decorated class for custom instantiation
decorated_class = compiler.create_decorated_class(
    author="Data Science Team",
    version="2.1.0",
    description="Production XGBoost Pipeline"
)

# Create template with custom parameters
template = compiler.create_template(
    dag=dag,
    custom_param="value",
    debug_mode=True
)

# Compile to pipeline
pipeline = compiler.compile(dag, pipeline_name="custom-pipeline")
```

### Execution Document Integration
```python
from cursus.mods.compiler import MODSPipelineDAGCompiler

compiler = MODSPipelineDAGCompiler(
    config_path="configs/my_pipeline.json",
    sagemaker_session=session,
    role=role
)

# Execution document template
execution_doc = {
    "pipeline_name": "{{pipeline_name}}",
    "execution_role": "{{execution_role}}",
    "parameters": "{{parameters}}"
}

# Compile and fill in one step
pipeline, filled_doc = compiler.compile_and_fill_execution_doc(
    dag=dag,
    execution_doc=execution_doc,
    pipeline_name="production-pipeline"
)
```

## Technical Implementation

### Metadata Extraction
The compiler automatically extracts MODS metadata from the base configuration:

1. **Base Config Discovery**: Locates the base configuration in the config file
2. **Metadata Extraction**: Extracts author, version, and description fields
3. **Fallback Handling**: Provides sensible defaults if metadata is missing

### Error Handling
The compiler includes comprehensive error handling:

- **Import Errors**: Graceful fallback when MODS is not available
- **Configuration Errors**: Clear messages for configuration issues
- **Validation Errors**: Detailed validation error reporting
- **Template Creation Errors**: Specific error handling for template instantiation

### Logging
Detailed logging throughout the compilation process:

- **Info Level**: Successful operations and progress updates
- **Warning Level**: Fallback scenarios and missing metadata
- **Error Level**: Compilation failures and error details

## Integration Points

### With Standard DAG Compiler
- Extends `PipelineDAGCompiler` for consistency
- Maintains all standard compilation features
- Overrides specific methods for MODS integration

### With MODS Service
- Applies `MODSTemplate` decorator correctly
- Extracts required metadata automatically
- Ensures proper MODS registration and execution

### With Configuration System
- Reads MODS metadata from pipeline configurations
- Supports flexible configuration formats
- Provides robust fallback mechanisms

## Dependencies

### Internal Dependencies
- `cursus.api.dag.base_dag.PipelineDAG`
- `cursus.core.compiler.dag_compiler.PipelineDAGCompiler`
- `cursus.core.compiler.config_resolver.StepConfigResolver`
- `cursus.steps.registry.builder_registry.StepBuilderRegistry`
- `cursus.core.compiler.exceptions.PipelineAPIError`
- `cursus.core.base.config_base.BasePipelineConfig`

### External Dependencies
- `mods.mods_template.MODSTemplate` (with graceful fallback)
- `sagemaker.workflow.pipeline.Pipeline`
- `sagemaker.workflow.pipeline_context.PipelineSession`

## Testing

The MODS compiler includes comprehensive testing:

### Unit Tests
- Core functionality testing
- Metadata extraction validation
- Error handling verification
- Method parameter validation

### Integration Tests
- Real pipeline DAG compilation
- MODS compatibility verification
- Advanced usage pattern testing

### End-to-End Tests
- Complete pipeline lifecycle testing
- MODS metadata propagation verification
- Execution environment compatibility

## Related Documentation

- [MODS Integration Overview](../README.md) - Module overview and architecture
- [MODS DAG Compiler Design](../../1_design/mods_dag_compiler_design.md) - Design decisions and architecture
- [Implementation Plan](../../2_project_planning/2025-08-19_mods_pipeline_dag_compiler_implementation_plan.md) - Development roadmap
- [Pipeline DAG Compiler](../../core/compiler/README.md) - Base compiler documentation

## Future Enhancements

### Planned Features
1. **Custom Metadata Providers**: Support for external metadata sources
2. **Enhanced Validation**: More comprehensive validation of MODS metadata
3. **Template Caching**: Caching of decorated classes for performance
4. **Multi-Version Support**: Support for different MODS API versions

### Performance Optimizations
- Lazy loading of MODS dependencies
- Template class caching
- Optimized metadata extraction
- Reduced memory footprint

## Conclusion

The MODS Compiler provides a robust, flexible solution for integrating Cursus pipelines with the MODS ecosystem. By solving the metaclass conflict challenge and providing enhanced API flexibility, it enables seamless adoption of MODS capabilities while maintaining the simplicity and power of the standard pipeline compilation process.
