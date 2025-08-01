# Pipeline Steps Documentation

This directory contains documentation for each step in the MODS pipeline. Each markdown file provides a detailed description of a specific pipeline step, including its purpose, inputs, outputs, configuration parameters, and usage examples.

> **New Feature**: These pipeline steps can now be used with the [Pipeline Builder Template](../pipeline_builder/README.md) system, which provides a declarative approach to defining pipeline structure and automatically handles the connections between steps using specification-driven dependency resolution.

## Available Pipeline Steps

### Data Loading and Preprocessing
- [Cradle Data Load Step](data_load_step_cradle.md): Loads data from various sources (MDS, EDX, or ANDES) using the Cradle service
- [Tabular Preprocessing Step](tabular_preprocessing_step.md): Prepares tabular data for model training
- [Risk Table Mapping Step](risk_table_map_step.md): Processes raw data and applies risk table mappings
- [Currency Conversion Step](currency_conversion_step.md): Performs currency normalization on monetary values

### Model Configuration
- [Hyperparameter Preparation Step](hyperparameter_prep_step.md): Serializes model hyperparameters to JSON and uploads them to S3

### Model Training
- [PyTorch Training Step](training_step_pytorch.md): Configures and executes a PyTorch model training job
- [XGBoost Training Step](training_step_xgboost.md): Configures and executes an XGBoost model training job

### Model Evaluation and Transformation
- [XGBoost Model Evaluation Step](model_eval_step_xgboost.md): Evaluates a trained XGBoost model on a specified dataset
- [Batch Transform Step](batch_transform_step.md): Generates predictions using a trained model

### Model Packaging and Registration
- [XGBoost Model Step](model_step_xgboost.md): Creates a SageMaker model artifact from a trained XGBoost model
- [PyTorch Model Step](model_step_pytorch.md): Creates a SageMaker model artifact from a trained PyTorch model
- [MIMS Packaging Step](mims_packaging_step.md): Prepares a trained model for deployment in MIMS
- [MIMS Payload Step](mims_payload_step.md): Generates and uploads test payloads for model testing
- [MIMS Registration Step](mims_registration_step.md): Registers a packaged model with MIMS

## Pipeline Architecture

Each step in the pipeline follows a consistent pattern:
- **Config Class (`config_xxx_step.py`)**: Defines the configuration parameters for the step using Pydantic models
- **Builder Class (`builder_xxx_step.py`)**: Implements the logic to create a SageMaker Pipeline step using the configuration
- **Step Specification**: Declares the step's inputs (dependencies) and outputs for automated connection

## Specification-Driven Dependency Resolution

Each step builder now includes a specification that declares its inputs and outputs:

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

These specifications enable the automatic connection of steps based on:
- Semantic matching of dependency and output names
- Type compatibility
- Explicit compatibility declarations
- Context-aware resolution

## Common Base Classes
- **BasePipelineConfig**: Base configuration class with common parameters
- **ProcessingStepConfigBase**: Base configuration for processing steps
- **StepBuilderBase**: Base builder class with common functionality
- **ModelHyperparameters**: Base class for model hyperparameters
- **XGBoostModelHyperparameters**: XGBoost-specific hyperparameters

## Usage Patterns

### Traditional Pattern

The traditional usage pattern for these steps is:

1. Create a configuration object with the required parameters
2. Create a builder object with the configuration
3. Use the builder to create a step with appropriate inputs and dependencies
4. Add the step to the pipeline

Example:
```python
# Create configuration
config = StepConfig(param1="value1", param2="value2")

# Create builder
builder = StepBuilder(config=config)

# Create step
step = builder.create_step(
    input_data=previous_step.properties.OutputPath,
    dependencies=[previous_step]
)

# Add to pipeline
pipeline.add_step(step)
```

### Template-Based Pattern

With the new [Pipeline Builder Template](../pipeline_builder/README.md) system, you can use a more declarative approach:

1. Define the pipeline structure as a DAG
2. Create a config map that maps step names to configuration instances
3. Create a step builder map that maps step types to step builder classes
4. Use the template to generate the pipeline

Example:
```python
from src.pipeline_builder.pipeline_template_base import PipelineTemplateBase
from src.pipeline_dag.base_dag import PipelineDAG

class MyPipelineTemplate(PipelineTemplateBase):
    # Define configuration classes
    CONFIG_CLASSES = {
        'Base': BasePipelineConfig,
        'DataLoading': CradleDataLoadingConfig,
        'Preprocessing': TabularPreprocessingConfig,
        'Training': XGBoostTrainingConfig
    }
    
    def _validate_configuration(self) -> None:
        # Validation logic
        pass
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        # Create DAG
        dag = PipelineDAG()
        dag.add_node("data_loading")
        dag.add_node("preprocessing")
        dag.add_edge("data_loading", "preprocessing")
        return dag
    
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        # Map steps to configurations
        return {
            "data_loading": self.configs['DataLoading'],
            "preprocessing": self.configs['Preprocessing']
        }
    
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        # Map step types to builder classes
        return {
            "CradleDataLoading": CradleDataLoadingStepBuilder,
            "TabularPreprocessingStep": TabularPreprocessingStepBuilder
        }

# Create the template and generate pipeline
template = MyPipelineTemplate(
    config_path="configs/pipeline_config.json",
    sagemaker_session=sagemaker_session,
    role=role
)
pipeline = template.generate_pipeline()
```

This template-based approach automatically handles the connections between steps using specification-driven dependency resolution, eliminating the need for manual wiring of inputs and outputs. It's particularly valuable for handling complex SageMaker property references.

## Related Documentation

### Pipeline Building
- [Pipeline Template Base](../pipeline_builder/pipeline_template_base.md): Core abstract class for pipeline templates
- [Pipeline Assembler](../pipeline_builder/pipeline_assembler.md): Assembles pipeline steps using a DAG and specifications
- [Pipeline Builder Overview](../pipeline_builder/README.md): Introduction to the template-based pipeline building system
- [Template Implementation](../pipeline_builder/template_implementation.md): How templates are implemented using specifications
- [Pipeline Examples](../pipeline_builder/pipeline_examples.md): Example pipeline implementations

### Dependency Resolution
- [Dependency Resolver](../pipeline_deps/dependency_resolver.md): Resolves dependencies between steps using specifications
- [Base Specifications](../pipeline_deps/base_specifications.md): Core specification data structures
- [Semantic Matcher](../pipeline_deps/semantic_matcher.md): Multi-metric semantic matching for step connections
- [Property Reference](../pipeline_deps/property_reference.md): Bridging definition and runtime properties

### Pipeline Structure
- [Base Pipeline DAG](../pipeline_dag/base_dag.md): Core DAG implementation
- [Enhanced Pipeline DAG](../pipeline_dag/enhanced_dag.md): Advanced DAG with port-level dependency resolution  
- [Pipeline DAG Overview](../pipeline_dag/README.md): Introduction to the DAG-based pipeline structure
- [Edge Types](../pipeline_dag/edge_types.md): Types of edges used in the DAG

### Script Contracts
- [Script Contracts Overview](../pipeline_script_contracts/README.md): Introduction to script contracts
- [Base Script Contract](../pipeline_script_contracts/base_script_contract.md): Foundation for script contracts
- [Contract Validator](../pipeline_script_contracts/contract_validator.md): Validation of script implementations
