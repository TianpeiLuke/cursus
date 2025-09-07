---
tags:
  - entry_point
  - code
  - core
  - pipeline_assembler
  - specification_driven
keywords:
  - pipeline assembler
  - specification driven
  - dependency resolution
  - template system
  - declarative approach
  - automatic connection
  - SageMaker pipelines
  - DAG structure
topics:
  - pipeline assembler system
  - specification-driven development
  - template architecture
  - dependency resolution
language: python
date of note: 2024-12-07
---

# Pipeline Assembler

## Overview

The Pipeline Assembler is a specification-driven system for creating SageMaker pipelines. It provides a declarative approach to defining pipeline structure and leverages intelligent dependency resolution to automatically connect steps, eliminating the need for manual wiring of inputs and outputs.

The system consists of two main components: the Pipeline Template Base for consistent template structure, and the Pipeline Assembler for step assembly and dependency resolution.

## Key Components

### 1. [Pipeline Template Base](pipeline_template_base.md)

The Pipeline Template Base is an abstract base class that provides a consistent structure and common functionality for all pipeline templates. It handles configuration loading, component lifecycle management, and pipeline generation:

- **Abstract Template Structure**: Enforces consistent implementation across pipeline templates
- **Configuration Management**: Loads and validates pipeline configurations from JSON files
- **Component Lifecycle**: Manages dependency resolution components (registry manager, dependency resolver)
- **Factory Methods**: Provides multiple creation patterns for different use cases
- **Context Management**: Supports scoped contexts and thread-local components for thread safety

Key features:
- Configuration loading with automatic hyperparameter class resolution
- Dependency injection support for registry manager and dependency resolver
- Abstract methods for DAG structure, configuration mapping, and step builder mapping
- Pipeline metadata storage for execution documents and step-specific data
- Multiple instantiation patterns: standard, with components, with context, in thread

### 2. [Pipeline Assembler](pipeline_assembler.md)

The Pipeline Assembler is responsible for assembling pipeline steps using a DAG structure and specification-based dependency resolution:

- **Step Builder Management**: Initializes and manages step builders for all pipeline steps
- **Specification-Based Matching**: Uses step specifications to intelligently match inputs to outputs
- **Message Propagation**: Propagates dependency information between connected steps
- **Runtime Property Handling**: Creates SageMaker property references for step connections
- **Topological Assembly**: Instantiates steps in dependency order using topological sorting

Key features:
- Automatic step builder initialization with dependency injection
- Intelligent input/output matching based on compatibility scoring
- Runtime property reference creation for SageMaker pipeline execution
- Support for Cradle data loading request storage
- Comprehensive error handling and validation
- Component-based architecture with factory method support

## Architecture Integration

### Dependency Resolution System
The assembler integrates with the core dependency resolution system:
- **Registry Manager**: Manages isolated specification registries for different contexts
- **Dependency Resolver**: Resolves dependencies between steps using specifications
- **Semantic Matcher**: Calculates compatibility scores between step inputs and outputs
- **Property Reference**: Creates runtime property references for SageMaker execution

### Configuration System
Integration with the configuration management system:
- **Config Classes**: Automatic loading of pipeline and hyperparameter configurations
- **Step Registry**: Uses centralized step name registry for consistent step type mapping
- **Validation**: Lightweight configuration validation with dependency validation handled by resolver

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

### Option 1: Use Dynamic Pipeline Template (Recommended)

The easiest way to create pipelines is using the `DynamicPipelineTemplate` which automatically handles configuration resolution and step builder mapping:

```python
from src.cursus.core.compiler.dynamic_template import DynamicPipelineTemplate
from src.cursus.api.dag.base_dag import PipelineDAG

# Create your DAG structure
dag = PipelineDAG()
dag.add_node("CradleDataLoading_data_loading")
dag.add_node("XGBoostTraining_training")
dag.add_node("Package_packaging")
dag.add_edge("CradleDataLoading_data_loading", "XGBoostTraining_training")
dag.add_edge("XGBoostTraining_training", "Package_packaging")

# Create dynamic template - automatically detects required config classes
template = DynamicPipelineTemplate(
    dag=dag,
    config_path="configs/my_pipeline.json",
    sagemaker_session=sagemaker_session,
    role=execution_role
)

# Generate pipeline
pipeline = template.generate_pipeline()
```

### Option 2: Create Custom Template Class

For more control, extend the `PipelineTemplateBase` class:

```python
from src.cursus.core.assembler.pipeline_template_base import PipelineTemplateBase
from src.cursus.core.base.config_base import BasePipelineConfig
from src.cursus.steps.configs.config_cradle_data_loading_step import CradleDataLoadConfig
from src.cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingConfig
from src.cursus.steps.configs.config_package_step import PackageConfig

class MyCustomTemplate(PipelineTemplateBase):
    # Define the configuration classes expected in the config file
    CONFIG_CLASSES = {
        'BasePipelineConfig': BasePipelineConfig,
        'CradleDataLoadConfig': CradleDataLoadConfig,
        'XGBoostTrainingConfig': XGBoostTrainingConfig,
        'PackageConfig': PackageConfig,
    }
    
    def _validate_configuration(self) -> None:
        """Validate that required configurations are present."""
        # Check for required configurations
        required_configs = ['CradleDataLoadConfig', 'XGBoostTrainingConfig']
        for config_type in required_configs:
            if not any(isinstance(cfg, self.CONFIG_CLASSES[config_type]) 
                      for cfg in self.configs.values()):
                raise ValueError(f"Missing required configuration: {config_type}")
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        """Create the DAG structure for the pipeline."""
        dag = PipelineDAG()
        dag.add_node("data_loading")
        dag.add_node("training")
        dag.add_node("packaging")
        dag.add_edge("data_loading", "training")
        dag.add_edge("training", "packaging")
        return dag
    
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        """Map step names to configuration instances."""
        config_map = {}
        
        # Find configurations by type
        for config_name, config in self.configs.items():
            if isinstance(config, CradleDataLoadConfig):
                config_map["data_loading"] = config
            elif isinstance(config, XGBoostTrainingConfig):
                config_map["training"] = config
            elif isinstance(config, PackageConfig):
                config_map["packaging"] = config
                
        return config_map
    
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """Map step types to builder classes."""
        from src.cursus.steps.builders.builder_cradle_data_loading_step import CradleDataLoadingStepBuilder
        from src.cursus.steps.builders.builder_xgboost_training_step import XGBoostTrainingStepBuilder
        from src.cursus.steps.builders.builder_package_step import PackageStepBuilder
        
        return {
            "CradleDataLoading": CradleDataLoadingStepBuilder,
            "XGBoostTraining": XGBoostTrainingStepBuilder,
            "Package": PackageStepBuilder,
        }

# Usage
template = MyCustomTemplate(
    config_path="configs/my_pipeline.json",
    sagemaker_session=sagemaker_session,
    role=execution_role
)

pipeline = template.generate_pipeline()
```

### Key Differences Between Approaches

#### Dynamic Template Advantages:
- **Automatic Configuration Detection**: Analyzes config file to determine required classes
- **Intelligent Resolution**: Uses `StepConfigResolver` to match DAG nodes to configurations
- **Registry Integration**: Automatically maps configuration types to step builders
- **Validation**: Built-in validation with detailed error reporting
- **Preview Capabilities**: Can preview how nodes will be resolved before building

#### Custom Template Advantages:
- **Full Control**: Complete control over DAG structure and mappings
- **Custom Validation**: Implement domain-specific validation logic
- **Static Configuration**: Explicit configuration class definitions
- **Predictable Behavior**: No automatic resolution - everything is explicit

### Configuration File Requirements

Both approaches require properly formatted configuration files created with `merge_and_save_configs`:

```python
from src.cursus.steps.configs import merge_and_save_configs
from src.cursus.steps.configs.config_cradle_data_loading_step import CradleDataLoadConfig
from src.cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingConfig

# Create configuration instances
configs = [
    CradleDataLoadConfig(
        pipeline_name="my_pipeline",
        job_type="data_loading",
        input_path="s3://bucket/data/"
    ),
    XGBoostTrainingConfig(
        pipeline_name="my_pipeline", 
        job_type="training",
        hyperparameters={"max_depth": 6}
    )
]

# Save to file with proper structure
merge_and_save_configs(configs, "configs/my_pipeline.json")
```

### Advanced Dynamic Template Features

The `DynamicPipelineTemplate` provides additional capabilities:

```python
# Create template with custom resolver
from src.cursus.core.compiler.config_resolver import StepConfigResolver

custom_resolver = StepConfigResolver(confidence_threshold=0.8)
template = DynamicPipelineTemplate(
    dag=dag,
    config_path="configs/my_pipeline.json",
    config_resolver=custom_resolver,
    sagemaker_session=sagemaker_session,
    role=execution_role
)

# Preview resolution before building
preview = template.get_resolution_preview()
for node, resolution in preview['resolutions'].items():
    print(f"{node} → {resolution['config_type']} (confidence: {resolution['confidence']:.2f})")

# Validate configuration
if template.validate_before_build():
    pipeline = template.generate_pipeline()
else:
    print("Configuration validation failed")

# Get execution order
execution_order = template.get_execution_order()
print(f"Steps will execute in order: {execution_order}")
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

## Pipeline Assembly Process

The Pipeline Assembler follows a systematic approach to build SageMaker pipelines:

### 1. Step Builder Initialization
```python
# Initialize step builders for all steps in the DAG
for step_name in self.dag.nodes:
    config = self.config_map[step_name]
    step_type = CONFIG_STEP_REGISTRY.get(type(config).__name__)
    builder_cls = self.step_builder_map[step_type]
    
    builder = builder_cls(
        config=config,
        sagemaker_session=self.sagemaker_session,
        role=self.role,
        registry_manager=self._registry_manager,
        dependency_resolver=self._dependency_resolver
    )
    self.step_builders[step_name] = builder
```

### 2. Message Propagation
```python
# Propagate messages between steps using specifications
for src_step, dst_step in self.dag.edges:
    src_builder = self.step_builders[src_step]
    dst_builder = self.step_builders[dst_step]
    
    # Match outputs to inputs based on compatibility
    for dep_name, dep_spec in dst_builder.spec.dependencies.items():
        for out_name, out_spec in src_builder.spec.outputs.items():
            compatibility = resolver._calculate_compatibility(dep_spec, out_spec, src_builder.spec)
            if compatibility > 0.5:
                # Store connection information
                self.step_messages[dst_step][dep_name] = {
                    'source_step': src_step,
                    'source_output': out_name,
                    'compatibility': compatibility
                }
```

### 3. Step Instantiation
```python
# Instantiate steps in topological order
build_order = self.dag.topological_sort()
for step_name in build_order:
    builder = self.step_builders[step_name]
    
    # Extract inputs from message connections
    inputs = {}
    for input_name, message in self.step_messages[step_name].items():
        src_step = message['source_step']
        src_output = message['source_output']
        
        # Create runtime property reference
        prop_ref = PropertyReference(
            step_name=src_step,
            output_spec=src_builder.spec.get_output_by_name_or_alias(src_output)
        )
        inputs[input_name] = prop_ref.to_runtime_property(self.step_instances)
    
    # Generate outputs and create step
    outputs = self._generate_outputs(step_name)
    step = builder.create_step(inputs=inputs, outputs=outputs)
    self.step_instances[step_name] = step
```

## Usage Examples

### Basic Template Usage

```python
from cursus.core.assembler import PipelineTemplateBase

class MyPipelineTemplate(PipelineTemplateBase):
    CONFIG_CLASSES = {
        'Base': BasePipelineConfig,
        'Training': XGBoostTrainingConfig,
        'Preprocessing': TabularPreprocessingConfig
    }
    
    def _validate_configuration(self):
        # Validate required configurations exist
        if 'Training' not in self.configs:
            raise ValueError("Training configuration required")
    
    def _create_pipeline_dag(self):
        dag = PipelineDAG()
        dag.add_node("preprocessing")
        dag.add_node("training")
        dag.add_edge("preprocessing", "training")
        return dag
    
    def _create_config_map(self):
        return {
            "preprocessing": self.configs['Preprocessing'],
            "training": self.configs['Training']
        }
    
    def _create_step_builder_map(self):
        return {
            "TabularPreprocessing": TabularPreprocessingStepBuilder,
            "XGBoostTraining": XGBoostTrainingStepBuilder
        }

# Create and use template
template = MyPipelineTemplate(
    config_path="configs/my_pipeline.json",
    sagemaker_session=sagemaker_session,
    role=execution_role
)

pipeline = template.generate_pipeline()
pipeline.upsert()
execution = pipeline.start()
```

### Advanced Template Usage with Context Management

```python
# Create pipeline with scoped context
pipeline = MyPipelineTemplate.build_with_context(
    config_path="configs/my_pipeline.json",
    sagemaker_session=sagemaker_session,
    role=execution_role
)

# Context automatically cleans up dependency resolution components
pipeline.upsert()
execution = pipeline.start()
```

### Thread-Safe Template Usage

```python
# Create pipeline with thread-local components
pipeline = MyPipelineTemplate.build_in_thread(
    config_path="configs/my_pipeline.json",
    sagemaker_session=sagemaker_session,
    role=execution_role
)

# Safe for use in multi-threaded environments
pipeline.upsert()
execution = pipeline.start()
```

### Direct Assembler Usage

```python
from cursus.core.assembler import PipelineAssembler
from cursus.api.dag.base_dag import PipelineDAG

# Create DAG structure
dag = PipelineDAG()
dag.add_node("step1")
dag.add_node("step2")
dag.add_edge("step1", "step2")

# Create configuration and builder mappings
config_map = {
    "step1": preprocessing_config,
    "step2": training_config
}

step_builder_map = {
    "TabularPreprocessing": TabularPreprocessingStepBuilder,
    "XGBoostTraining": XGBoostTrainingStepBuilder
}

# Create assembler with managed components
assembler = PipelineAssembler.create_with_components(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    context_name="my_pipeline",
    sagemaker_session=sagemaker_session,
    role=execution_role
)

# Generate pipeline
pipeline = assembler.generate_pipeline("my-pipeline")
```

## Error Handling and Validation

The assembler provides comprehensive error handling:

### Configuration Validation
```python
# Missing configs for DAG nodes
missing_configs = [node for node in self.dag.nodes if node not in self.config_map]
if missing_configs:
    raise ValueError(f"Missing configs for nodes: {missing_configs}")

# Missing step builders for config types
for step_name, config in self.config_map.items():
    step_type = CONFIG_STEP_REGISTRY.get(type(config).__name__)
    if step_type not in self.step_builder_map:
        raise ValueError(f"Missing step builder for step type: {step_type}")
```

### DAG Validation
```python
# Validate DAG edges connect existing nodes
for src, dst in self.dag.edges:
    if src not in self.dag.nodes:
        raise ValueError(f"Edge source node not in DAG: {src}")
    if dst not in self.dag.nodes:
        raise ValueError(f"Edge destination node not in DAG: {dst}")
```

### Step Building Error Handling
```python
try:
    step = builder.create_step(**kwargs)
    logger.info(f"Built step {step_name}")
    return step
except Exception as e:
    logger.error(f"Error building step {step_name}: {e}")
    raise ValueError(f"Failed to build step {step_name}: {e}") from e
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
