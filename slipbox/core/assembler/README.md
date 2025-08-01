# Pipeline Builder

## Overview

The Pipeline Builder is a specification-driven system for creating SageMaker pipelines. It provides a declarative approach to defining pipeline structure and leverages intelligent dependency resolution to automatically connect steps, eliminating the need for manual wiring of inputs and outputs.

## Key Components

### 1. [Pipeline Template Base](pipeline_template_base.md)

The Pipeline Template Base is an abstract base class that provides a consistent structure and common functionality for all pipeline templates. It handles configuration loading, component lifecycle management, and pipeline generation:

- Abstract methods for DAG structure, configuration mapping, and step builder mapping
- Factory methods for creating templates with proper dependency components
- Support for context management and thread safety

### 2. [Pipeline Assembler](pipeline_assembler.md)

The Pipeline Assembler is responsible for assembling pipeline steps using a DAG structure and specification-based dependency resolution:

- Step builder initialization
- Specification-based message propagation
- Runtime property reference handling
- Topological step instantiation

### 3. [Pipeline DAG](../pipeline_dag/pipeline_dag.md)

The Pipeline DAG (Directed Acyclic Graph) represents the structure of a pipeline where nodes are pipeline steps and edges represent dependencies between steps. It provides methods for:

- Adding nodes and edges
- Querying dependencies
- Topological sorting
- Cycle detection

### 4. [Pipeline Examples](pipeline_examples.md)

The Pipeline Examples document provides an overview of the example pipelines built using the template system, including:

- XGBoost End-to-End Pipeline
- XGBoost Simple Pipeline
- PyTorch End-to-End Pipeline
- PyTorch Model Registration Pipeline
- XGBoost Data Load and Preprocess Pipeline
- XGBoost Training and Evaluation Pipeline
- XGBoost Training and Evaluation (No Registration) Pipeline
- Cradle-Only Pipeline

### 5. [Template Implementation](template_implementation.md)

The Template Implementation document explains how the template-based implementation uses specification-driven dependency resolution to connect pipeline steps through:

- Step Specifications
- DAG-Based Message Propagation
- Property Reference Resolution
- Compatibility Scoring

## Specification-Driven Dependency Resolution

### Step Specifications

Each step builder provides a specification that declares its inputs and outputs:

```python
self.spec = StepSpecification(
    step_type="XGBoostTrainingStep",
    node_type=NodeType.INTERNAL,
    dependencies={
        "training_data": DependencySpec(
            logical_name="training_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["PreprocessingStep"],
            semantic_keywords=["data", "training", "processed"],
            data_type="S3Uri"
        )
    },
    outputs={
        "model_output": OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            aliases=["ModelArtifacts", "model_data"]
        )
    }
)
```

### Dependency Resolution Process

1. **Specification Registration**: Each step registers its specification with the registry.
2. **Dependency Analysis**: The dependency resolver analyzes the specifications of all steps.
3. **Compatibility Scoring**: The resolver calculates compatibility scores between dependencies and outputs.
4. **Message Propagation**: Messages are propagated from source steps to destination steps based on the DAG structure.
5. **Property Reference Creation**: Property references are created to bridge definition-time and runtime.

### Component Relationships

The dependency resolution system consists of several interrelated components:

- **Registry Manager**: Manages multiple isolated specification registries
- **Dependency Resolver**: Resolves dependencies between steps using specifications
- **Semantic Matcher**: Calculates similarity between dependency names and output names
- **Property Reference**: Bridges definition-time and runtime property references
- **Factory Module**: Creates and manages component instances

These components work together to provide a powerful system for automatically connecting pipeline steps.

## Creating Custom Pipeline Templates

To create a custom pipeline template, extend the `PipelineTemplateBase` class:

```python
class MyCustomTemplate(PipelineTemplateBase):
    # Define the configuration classes expected in the config file
    CONFIG_CLASSES = {
        'Base': BasePipelineConfig,
        'DataLoading': CradleDataLoadingConfig,
        'Preprocessing': TabularPreprocessingConfig,
        'Training': XGBoostTrainingConfig,
    }
    
    def _validate_configuration(self) -> None:
        # Custom validation logic
        pass
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        # Create the DAG structure
        pass
    
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        # Map step names to configurations
        pass
    
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        # Map step types to builder classes
        pass
```

## Benefits of Using the Pipeline Builder

1. **Automatic Step Connection**: Dependencies between steps are automatically resolved based on specifications.
2. **Semantic Matching**: Inputs and outputs are matched based on semantic similarity, not just exact names.
3. **Type Compatibility**: The system ensures that connected steps have compatible input/output types.
4. **Configuration-Driven**: Pipelines are configured through JSON files, making them easy to modify.
5. **Declarative Definition**: Pipeline structure is defined declaratively through the DAG.
6. **Modular Design**: Pipelines can be composed of reusable components.
7. **Context Isolation**: Multiple pipelines can run in isolated contexts.
8. **Thread Safety**: Components can be used safely in multi-threaded environments.

## Usage Examples

### Basic Template Usage

```python
from src.pipeline_builder.template_pipeline_xgboost_end_to_end import XGBoostEndToEndTemplate

# Create pipeline with context management
pipeline = XGBoostEndToEndTemplate.build_with_context(
    config_path="configs/xgboost_config.json",
    sagemaker_session=sagemaker_session,
    role=execution_role
)

# Execute the pipeline
pipeline.upsert()
execution = pipeline.start()
```

### Advanced Template Usage with Execution Document

```python
from src.pipeline_builder.template_pipeline_xgboost_end_to_end import XGBoostEndToEndTemplate

# Create template instance
template = XGBoostEndToEndTemplate(
    config_path="configs/xgboost_config.json",
    sagemaker_session=sagemaker_session,
    role=execution_role
)

# Generate the pipeline
pipeline = template.generate_pipeline()

# Create an execution document template
execution_doc = {
    "execution": {
        "name": "XGBoost Training Pipeline",
        "steps": []
    }
}

# Fill execution document with pipeline metadata
filled_doc = template.fill_execution_document(execution_doc)

# Execute the pipeline
pipeline.upsert()
execution = pipeline.start()
```

## Related Documentation

### Pipeline Builder Components
- [Pipeline Template Base](pipeline_template_base.md): Core abstract class for pipeline templates
- [Pipeline Assembler](pipeline_assembler.md): Assembles steps using specifications
- [Template Implementation](template_implementation.md): How templates are implemented
- [Pipeline Examples](pipeline_examples.md): Example pipeline implementations

### Pipeline Structure
- [Pipeline DAG](../pipeline_dag/pipeline_dag.md): DAG structure for pipeline steps
- [Pipeline DAG Overview](../pipeline_dag/README.md): DAG-based pipeline structure concepts

### Dependency Resolution
- [Pipeline Dependencies](../pipeline_deps/README.md): Overview of dependency resolution
- [Dependency Resolver](../pipeline_deps/dependency_resolver.md): Resolves step dependencies
- [Base Specifications](../pipeline_deps/base_specifications.md): Core specification structures
- [Semantic Matcher](../pipeline_deps/semantic_matcher.md): Name matching algorithms
- [Property Reference](../pipeline_deps/property_reference.md): Runtime property bridge
- [Registry Manager](../pipeline_deps/registry_manager.md): Multi-context registry management

### Pipeline Components
- [Pipeline Steps](../pipeline_steps/README.md): Available steps and their specifications
- [Script Contracts](../pipeline_script_contracts/README.md): Script contracts and validation
- [Base Script Contract](../pipeline_script_contracts/base_script_contract.md): Foundation for script contracts
