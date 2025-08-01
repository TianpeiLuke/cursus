# Template-Based Pipeline Implementation

This document explains the template-based implementation of pipelines in the MODS Pipeline framework and how they use specification-driven dependency resolution to connect pipeline steps.

## Overview

The template-based implementations in `src/pipeline_builder/` build on two core abstractions:

1. `PipelineTemplateBase`: An abstract base class providing standardized structure for pipeline templates
2. `PipelineAssembler`: A concrete class that assembles pipeline steps using a DAG and specification-based dependency resolution

These components work together to automatically handle connections between pipeline steps, eliminating the need for manual wiring of inputs and outputs.

## Available Template Implementations

The framework provides several ready-to-use template implementations:

| Template File | Purpose |
|---------------|---------|
| `template_pipeline_xgboost_end_to_end.py` | End-to-end XGBoost pipeline with data loading, preprocessing, training, evaluation, and registration |
| `template_pipeline_xgboost_simple.py` | Simplified XGBoost pipeline for quick experimentation |
| `template_pipeline_pytorch_end_to_end.py` | End-to-end PyTorch pipeline with data loading, preprocessing, training, and registration |
| `template_pipeline_pytorch_model_registration.py` | PyTorch pipeline focused on model registration |
| `template_pipeline_xgboost_dataload_preprocess.py` | XGBoost pipeline focused on data loading and preprocessing |
| `template_pipeline_xgboost_train_evaluate_e2e.py` | XGBoost pipeline focused on training and evaluation |
| `template_pipeline_xgboost_train_evaluate_no_registration.py` | XGBoost pipeline for training and evaluation without registration |
| `template_pipeline_cradle_only.py` | Pipeline focused on Cradle data operations |

## Specification-Driven Dependency Resolution

The new template implementations use specification-driven dependency resolution to automatically connect pipeline steps:

### 1. Step Specifications

Each step builder provides a specification that declares its inputs and outputs:

```python
# In a step builder's __init__ method:
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

### 2. DAG-Based Message Propagation

The `PipelineAssembler` uses the DAG to determine which steps can provide inputs to other steps:

```python
# In PipelineAssembler._propagate_messages
for src_step, dst_step in self.dag.edges:
    # Skip if builders don't exist
    if src_step not in self.step_builders or dst_step not in self.step_builders:
        continue
        
    # Get specs
    src_builder = self.step_builders[src_step]
    dst_builder = self.step_builders[dst_step]
    
    # Let resolver match outputs to inputs
    for dep_name, dep_spec in dst_builder.spec.dependencies.items():
        matches = []
        
        # Check if source step can provide this dependency
        for out_name, out_spec in src_builder.spec.outputs.items():
            compatibility = resolver._calculate_compatibility(dep_spec, out_spec, src_builder.spec)
            if compatibility > 0.5:  # Same threshold as resolver
                matches.append((out_name, out_spec, compatibility))
        
        # Use best match if found
        if matches:
            # Sort by compatibility score
            matches.sort(key=lambda x: x[2], reverse=True)
            best_match = matches[0]
            
            # Store in step_messages
            self.step_messages[dst_step][dep_name] = {
                'source_step': src_step,
                'source_output': best_match[0],
                'match_type': 'specification_match',
                'compatibility': best_match[2]
            }
```

### 3. Property Reference Resolution

During step instantiation, the `PipelineAssembler` creates `PropertyReference` objects to bridge definition-time and runtime:

```python
# In PipelineAssembler._instantiate_step
if output_spec:
    try:
        # Create a PropertyReference object
        prop_ref = PropertyReference(
            step_name=src_step,
            output_spec=output_spec
        )
        
        # Use the enhanced to_runtime_property method to get a SageMaker Properties object
        runtime_prop = prop_ref.to_runtime_property(self.step_instances)
        inputs[input_name] = runtime_prop
    except Exception as e:
        # Fallback handling...
```

## Creating a Custom Template Implementation

To create a custom template implementation, extend the `PipelineTemplateBase` class:

```python
from src.pipeline_builder.pipeline_template_base import PipelineTemplateBase
from src.pipeline_dag.base_dag import PipelineDAG
from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadingConfig
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig
from src.pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder

class MyCustomXGBoostTemplate(PipelineTemplateBase):
    CONFIG_CLASSES = {
        'Base': BasePipelineConfig,
        'DataLoading': CradleDataLoadingConfig,
        'Preprocessing': TabularPreprocessingConfig,
        'Training': XGBoostTrainingConfig
    }
    
    def _validate_configuration(self) -> None:
        # Validation logic here
        pass
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        dag = PipelineDAG()
        
        # Define steps
        dag.add_node("data_loading")
        dag.add_node("preprocessing")
        dag.add_node("training")
        
        # Define dependencies
        dag.add_edge("data_loading", "preprocessing")
        dag.add_edge("preprocessing", "training")
        
        return dag
    
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        return {
            "data_loading": self.configs['DataLoading'],
            "preprocessing": self.configs['Preprocessing'],
            "training": self.configs['Training'],
        }
    
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        return {
            "CradleDataLoading": CradleDataLoadingStepBuilder,
            "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
            "XGBoostTrainingStep": XGBoostTrainingStepBuilder,
        }
```

## Example Usage

There are two main patterns for using template implementations:

### 1. Direct Usage

```python
from src.pipeline_builder.template_pipeline_xgboost_end_to_end import XGBoostEndToEndTemplate

# Create the template
template = XGBoostEndToEndTemplate(
    config_path="configs/xgboost_config.json",
    sagemaker_session=pipeline_session,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
)

# Generate the pipeline
pipeline = template.generate_pipeline()

# Optional: Execute the pipeline
pipeline.upsert()
execution = pipeline.start()
```

### 2. Factory Method Usage

```python
from src.pipeline_builder.template_pipeline_xgboost_end_to_end import XGBoostEndToEndTemplate

# Use factory method with context management
pipeline = XGBoostEndToEndTemplate.build_with_context(
    config_path="configs/xgboost_config.json",
    sagemaker_session=pipeline_session,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
)

# Alternative: Thread-safe factory method
pipeline = XGBoostEndToEndTemplate.build_in_thread(
    config_path="configs/xgboost_config.json",
    sagemaker_session=pipeline_session,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
)
```

## Execution Document Support

Many template implementations support filling execution documents for external systems:

```python
# Create execution document template
execution_doc = {
    "execution": {
        "name": "My Pipeline Execution",
        "steps": []
    }
}

# Fill execution document with pipeline metadata
filled_doc = template.fill_execution_document(execution_doc)

# The document now contains step-specific metadata, such as Cradle data loading requests
```

## Advantages of Template-Based Implementations

1. **Declarative Pipeline Definition**: Pipelines are defined declaratively using configurations, DAGs, and specifications
2. **Automatic Dependency Resolution**: Step connections are automatically determined using intelligent matching
3. **Separation of Concerns**: Pipeline structure is separated from step implementations
4. **Configuration-Driven**: Pipeline variations can be created by changing configuration files
5. **Reusable Components**: Templates can be composed of reusable step builders
6. **Context Management**: Templates support proper resource management and isolation
7. **Thread Safety**: Templates can be used safely in multi-threaded environments

## Performance Considerations

Creating pipeline templates incurs some overhead due to specification registration and dependency resolution. For most use cases, this overhead is negligible, but there are ways to optimize:

```python
# Use factory method with manual component management
components = create_pipeline_components("my_pipeline")
template = MyCustomTemplate(
    config_path="config.json",
    registry_manager=components["registry_manager"],
    dependency_resolver=components["resolver"],
)

# Reuse the template for multiple pipeline variations
pipeline1 = template.generate_pipeline()
pipeline2 = template.generate_pipeline()  # Reuses cached dependency resolution
```

## Related Documentation

### Pipeline Building
- [Pipeline Template Base](pipeline_template_base.md): Core abstract class for pipeline templates
- [Pipeline Assembler](pipeline_assembler.md): Assembles pipeline steps using specifications
- [Pipeline Builder Overview](README.md): Introduction to the template-based system
- [Pipeline Examples](pipeline_examples.md): Example pipeline implementations

### Pipeline Structure
- [Pipeline DAG Overview](../pipeline_dag/README.md): Introduction to the DAG-based pipeline structure
- [Base Pipeline DAG](../pipeline_dag/base_dag.md): Core DAG implementation
- [Enhanced Pipeline DAG](../pipeline_dag/enhanced_dag.md): Advanced DAG with port-level dependency resolution
- [Edge Types](../pipeline_dag/edge_types.md): Types of edges used in the DAG

### Dependency System
- [Dependency Resolver](../pipeline_deps/dependency_resolver.md): Core resolver for dependencies
- [Base Specifications](../pipeline_deps/base_specifications.md): Core specification structures
- [Property Reference](../pipeline_deps/property_reference.md): Runtime property bridge
- [Semantic Matcher](../pipeline_deps/semantic_matcher.md): Name matching algorithms

### Pipeline Components
- [Pipeline Steps](../pipeline_steps/README.md): Available steps for pipeline construction
- [Script Contracts](../pipeline_script_contracts/README.md): Script validation system
