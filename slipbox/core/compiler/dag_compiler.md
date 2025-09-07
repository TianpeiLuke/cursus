---
tags:
  - code
  - core
  - compiler
  - dag_compilation
  - pipeline_generation
keywords:
  - PipelineDAGCompiler
  - compile_dag_to_pipeline
  - DAG compilation
  - SageMaker pipeline
  - pipeline template
  - validation
topics:
  - DAG compilation
  - pipeline generation
  - SageMaker integration
language: python
date of note: 2025-09-07
---

# DAG Compiler

Main API functions for compiling PipelineDAG structures into executable SageMaker pipelines.

## Overview

The `dag_compiler` module provides the main API functions for compiling PipelineDAG structures into executable SageMaker pipelines. It offers both simple one-call compilation and advanced compilation with detailed control over the process, including validation, debugging, and customization options.

The module handles the complete compilation pipeline from DAG structure to executable SageMaker Pipeline, including configuration resolution, step builder mapping, validation, and template generation. It provides comprehensive error handling and detailed reporting for troubleshooting compilation issues.

## Classes and Methods

### Classes
- [`PipelineDAGCompiler`](#pipelinedagcompiler) - Advanced API for DAG-to-template compilation with additional control

### Functions
- [`compile_dag_to_pipeline`](#compile_dag_to_pipeline) - Simple one-call compilation from DAG to pipeline

## API Reference

### compile_dag_to_pipeline

compile_dag_to_pipeline(_dag_, _config_path_, _sagemaker_session=None_, _role=None_, _pipeline_name=None_, _**kwargs_)

Compile a PipelineDAG into a complete SageMaker Pipeline. This is the main entry point for users who want a simple, one-call compilation from DAG to pipeline.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG instance defining the pipeline structure
- **config_path** (_str_) – Path to configuration file containing step configs
- **sagemaker_session** (_Optional[PipelineSession]_) – SageMaker session for pipeline execution
- **role** (_Optional[str]_) – IAM role for pipeline execution
- **pipeline_name** (_Optional[str]_) – Optional pipeline name override
- ****kwargs** – Additional arguments passed to template constructor

**Returns:**
- **Pipeline** – Generated SageMaker Pipeline ready for execution

**Raises:**
- **ValueError** – If DAG nodes don't have corresponding configurations
- **ConfigurationError** – If configuration validation fails
- **RegistryError** – If step builders not found for config types

```python
from cursus.core.compiler.dag_compiler import compile_dag_to_pipeline
from cursus.api.dag.base_dag import PipelineDAG

# Create a simple DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_edge("data_load", "preprocess")

# Compile to SageMaker Pipeline
pipeline = compile_dag_to_pipeline(
    dag=dag,
    config_path="configs/my_pipeline.json",
    sagemaker_session=session,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Execute the pipeline
pipeline.upsert()
```

### PipelineDAGCompiler

_class_ cursus.core.compiler.dag_compiler.PipelineDAGCompiler(_config_path_, _sagemaker_session=None_, _role=None_, _config_resolver=None_, _builder_registry=None_, _**kwargs_)

Advanced API for DAG-to-template compilation with additional control. This class provides more control over the compilation process, including validation, debugging, and customization options.

**Parameters:**
- **config_path** (_str_) – Path to configuration file
- **sagemaker_session** (_Optional[PipelineSession]_) – SageMaker session for pipeline execution
- **role** (_Optional[str]_) – IAM role for pipeline execution
- **config_resolver** (_Optional[StepConfigResolver]_) – Custom config resolver (optional)
- **builder_registry** (_Optional[StepBuilderRegistry]_) – Custom builder registry (optional)
- ****kwargs** – Additional arguments for template constructor

```python
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler

# Create compiler with custom settings
compiler = PipelineDAGCompiler(
    config_path="configs/pipeline.json",
    sagemaker_session=session,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Validate DAG before compilation
validation_result = compiler.validate_dag_compatibility(dag)
if not validation_result.is_valid:
    print(f"Validation failed: {validation_result.summary()}")

# Preview resolution
preview = compiler.preview_resolution(dag)
print(f"Node mappings: {preview.node_config_map}")

# Compile with detailed report
pipeline, report = compiler.compile_with_report(dag)
print(f"Compilation report: {report.summary()}")
```

#### validate_dag_compatibility

validate_dag_compatibility(_dag_)

Validate that DAG nodes have corresponding configurations. Returns detailed validation results including missing configurations, unresolvable step builders, configuration validation errors, and dependency resolution issues.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG instance to validate

**Returns:**
- **ValidationResult** – ValidationResult with detailed validation information

```python
# Validate DAG compatibility
validation_result = compiler.validate_dag_compatibility(dag)

if validation_result.is_valid:
    print("DAG is valid and ready for compilation")
else:
    print(f"Validation issues found:")
    for error in validation_result.config_errors:
        print(f"  - {error}")
    
    if validation_result.missing_configs:
        print(f"Missing configs: {validation_result.missing_configs}")
```

#### preview_resolution

preview_resolution(_dag_)

Preview how DAG nodes will be resolved to configs and builders. Returns a detailed preview showing node → configuration mappings, configuration → step builder mappings, detected step types and dependencies, and potential issues or ambiguities.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG instance to preview

**Returns:**
- **ResolutionPreview** – ResolutionPreview with detailed resolution information

```python
# Preview resolution before compilation
preview = compiler.preview_resolution(dag)

# Examine node-to-config mappings
for node, config_type in preview.node_config_map.items():
    confidence = preview.resolution_confidence.get(node, 0.0)
    builder = preview.config_builder_map.get(config_type, 'Unknown')
    print(f"Node '{node}' -> {config_type} -> {builder} (confidence: {confidence:.2f})")

# Check for issues
if preview.ambiguous_resolutions:
    print("Ambiguous resolutions found:")
    for issue in preview.ambiguous_resolutions:
        print(f"  - {issue}")

# Review recommendations
for recommendation in preview.recommendations:
    print(f"Recommendation: {recommendation}")
```

#### compile

compile(_dag_, _pipeline_name=None_, _**kwargs_)

Compile DAG to pipeline with full control.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG instance to compile
- **pipeline_name** (_Optional[str]_) – Optional pipeline name override
- ****kwargs** – Additional arguments for template

**Returns:**
- **Pipeline** – Generated SageMaker Pipeline

**Raises:**
- **PipelineAPIError** – If compilation fails

```python
# Compile with custom pipeline name
pipeline = compiler.compile(
    dag=dag,
    pipeline_name="my-custom-pipeline",
    skip_validation=False
)

print(f"Compiled pipeline: {pipeline.name}")
```

#### compile_with_report

compile_with_report(_dag_, _pipeline_name=None_, _**kwargs_)

Compile DAG to pipeline and return detailed compilation report.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG instance to compile
- **pipeline_name** (_Optional[str]_) – Optional pipeline name override
- ****kwargs** – Additional arguments for template

**Returns:**
- **Tuple[Pipeline, ConversionReport]** – Tuple of (Pipeline, ConversionReport)

```python
# Compile with detailed reporting
pipeline, report = compiler.compile_with_report(dag)

print(f"Pipeline: {report.pipeline_name}")
print(f"Steps: {len(report.steps)}")
print(f"Average confidence: {report.avg_confidence:.2f}")

# Review resolution details
for node, details in report.resolution_details.items():
    print(f"Node '{node}': {details['config_type']} -> {details['builder_type']}")

# Check warnings
if report.warnings:
    print("Warnings:")
    for warning in report.warnings:
        print(f"  - {warning}")
```

#### create_template

create_template(_dag_, _**kwargs_)

Create a pipeline template from the DAG without generating the pipeline. This allows inspecting or modifying the template before pipeline generation.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG instance to create a template for
- ****kwargs** – Additional arguments for template

**Returns:**
- **DynamicPipelineTemplate** – DynamicPipelineTemplate instance ready for pipeline generation

**Raises:**
- **PipelineAPIError** – If template creation fails

```python
# Create template for inspection
template = compiler.create_template(dag)

# Inspect template properties
print(f"Base config: {type(template.base_config).__name__}")
print(f"Available configs: {list(template.configs.keys())}")

# Generate pipeline when ready
pipeline = template.generate_pipeline()
```

#### compile_and_fill_execution_doc

compile_and_fill_execution_doc(_dag_, _execution_doc_, _pipeline_name=None_, _**kwargs_)

Compile a DAG to pipeline and fill an execution document in one step. This method ensures proper sequencing of the pipeline generation and execution document filling, addressing timing issues with template metadata.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG instance to compile
- **execution_doc** (_Dict[str, Any]_) – Execution document template to fill
- **pipeline_name** (_Optional[str]_) – Optional pipeline name override
- ****kwargs** – Additional arguments for template

**Returns:**
- **Tuple[Pipeline, Dict[str, Any]]** – Tuple of (compiled_pipeline, filled_execution_doc)

```python
# Execution document template
execution_doc = {
    "pipeline_name": "{{pipeline_name}}",
    "steps": "{{step_names}}",
    "execution_role": "{{execution_role}}"
}

# Compile and fill document
pipeline, filled_doc = compiler.compile_and_fill_execution_doc(
    dag=dag,
    execution_doc=execution_doc,
    pipeline_name="my-pipeline"
)

print(f"Filled document: {filled_doc}")
```

#### get_supported_step_types

get_supported_step_types()

Get list of supported step types.

**Returns:**
- **List[str]** – List of supported step type names

```python
# Check supported step types
supported_types = compiler.get_supported_step_types()
print(f"Supported step types: {supported_types}")
```

#### validate_config_file

validate_config_file()

Validate the configuration file structure.

**Returns:**
- **Dict[str, Any]** – Dictionary with validation results

```python
# Validate configuration file
config_validation = compiler.validate_config_file()

if config_validation['valid']:
    print(f"Config file is valid with {config_validation['config_count']} configurations")
    print(f"Config types: {config_validation['config_types']}")
else:
    print(f"Config file validation failed: {config_validation['error']}")
```

#### get_last_template

get_last_template()

Get the last template used during compilation. This template will have its pipeline_metadata populated from the generation process.

**Returns:**
- **Optional[DynamicPipelineTemplate]** – The last template used in compilation, or None if no compilation has occurred

```python
# Compile pipeline
pipeline = compiler.compile(dag)

# Get the template used for compilation
template = compiler.get_last_template()
if template:
    print(f"Template metadata: {template.pipeline_metadata}")
```

## Related Documentation

- [Configuration Resolver](config_resolver.md) - Used for resolving DAG nodes to configurations
- [Dynamic Template](dynamic_template.md) - Template system used for pipeline generation
- [Validation](validation.md) - Validation engine for DAG compatibility checking
- [Compiler Exceptions](exceptions.md) - Exception classes used in compilation
- [Compiler Overview](README.md) - System overview and integration
