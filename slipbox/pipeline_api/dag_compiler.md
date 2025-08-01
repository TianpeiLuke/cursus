---
tags:
  - code
  - pipeline_api
  - dag_compiler
  - pipeline_generation
keywords:
  - DAG
  - compiler
  - pipeline
  - SageMaker
  - conversion
topics:
  - pipeline API
  - DAG compilation
language: python
date of note: 2025-07-31
---

# DAG Compiler

## Purpose

The `PipelineDAGCompiler` is the main entry point of the Pipeline API, responsible for compiling PipelineDAG structures into executable SageMaker pipelines. It provides both simple one-call functions and advanced APIs for validation, debugging, and customization.

## Core Problem Solved

Converting abstract DAG representations into concrete SageMaker pipelines requires multiple steps:
1. Matching DAG nodes to appropriate configurations
2. Resolving configurations to step builders
3. Building and connecting pipeline steps according to the DAG structure
4. Managing pipeline parameters and execution properties

The DAG Compiler abstracts away this complexity, providing a simple interface for pipeline generation.

## Usage

### Simple Usage

```python
from src.pipeline_api.dag_compiler import compile_dag_to_pipeline

# Create a DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_node("train")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "train")

# Compile to pipeline
pipeline = compile_dag_to_pipeline(
    dag=dag,
    config_path="configs/my_pipeline.json",
    sagemaker_session=session,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)
```

### Advanced Usage

```python
from src.pipeline_api.dag_compiler import PipelineDAGCompiler

# Create compiler
compiler = PipelineDAGCompiler(
    config_path="configs/my_pipeline.json",
    sagemaker_session=session,
    role=role
)

# Validate DAG compatibility
validation_result = compiler.validate_dag_compatibility(dag)
if validation_result.is_valid:
    # Preview resolution
    preview = compiler.preview_resolution(dag)
    print(preview.display())
    
    # Compile with detailed reporting
    pipeline, report = compiler.compile_with_report(dag)
    print(report.summary())
```

## Implementation Details

The compiler:

1. Creates a `DynamicPipelineTemplate` with the DAG structure
2. Uses `StepConfigResolver` to map DAG nodes to configurations
3. Uses `StepBuilderRegistry` to map configurations to step builders
4. Validates compatibility between DAG, configs, and builders
5. Generates a SageMaker pipeline with proper connectivity

## Key Components

### Core Functions

- **`compile_dag_to_pipeline()`**: One-call compilation function
- **`PipelineDAGCompiler`**: Advanced class with validation and debugging

### Advanced APIs

- **`validate_dag_compatibility()`**: Validate DAG compatibility
- **`preview_resolution()`**: Preview node-to-config resolution
- **`compile_with_report()`**: Compile with detailed reporting

## Validation and Debugging

The compiler provides comprehensive validation and debugging:

1. **Validation checks**:
   - Missing configurations detection
   - Unresolvable step builders
   - Configuration validation errors
   - Dependency resolution issues

2. **Resolution preview**:
   - Node-to-config mappings
   - Config-to-builder mappings
   - Confidence scores
   - Resolution method used
   - Potential alternatives

3. **Compilation reports**:
   - Resolution details
   - Average confidence
   - Warnings and recommendations
   - Metadata

## Integration with Other Components

The DAG Compiler integrates with:

- **Config Resolver**: For DAG node-to-config mapping
- **Builder Registry**: For config-to-builder mapping
- **Dynamic Template**: For pipeline generation
- **Validation Engine**: For compatibility validation

It serves as the central orchestrator, connecting all pieces of the pipeline generation process into a cohesive workflow.
