# MODS_BSM Pipeline Examples

This directory contains documentation for the various pipeline examples in the MODS_BSM system. Each markdown file provides a detailed description of a specific pipeline, including its steps, connections between steps, and input/output relationships.

> **New Feature**: A new template-based approach for building pipelines is now available. See the [Pipeline Builder](../pipeline_builder/README.md) documentation for details on how to use the template system to simplify pipeline creation and automatically handle connections between steps.

## Available Pipelines

### XGBoost Pipelines

- [XGBoost End-to-End Pipeline](mods_pipeline_xgboost_end_to_end.md): A complete machine learning workflow including data loading, preprocessing, model training, model creation, packaging, and registration.

- [XGBoost End-to-End Simple Pipeline](mods_pipeline_xgboost_end_to_end_simple.md): A streamlined version of the end-to-end pipeline that skips the explicit model creation step.

### PyTorch Pipelines

- [PyTorch BSM Pipeline](mods_pipeline_bsm_pytorch.md): A model deployment pipeline that takes a pre-trained PyTorch model and prepares it for deployment.

## Pipeline Architecture

### Traditional Approach

The traditional pipelines follow a consistent design pattern:

1. **Configuration-Driven**: Pipelines extract their settings from JSON configuration files
2. **Builder Pattern**: Each pipeline uses a builder class to construct the pipeline
3. **Step-Based**: Pipelines are composed of individual steps that are connected together
4. **Execution Document Support**: Pipelines provide methods to fill execution documents with step configurations

### Template-Based Approach

The new template-based pipelines use a more declarative approach:

1. **DAG-Based Structure**: Pipeline structure is defined as a Directed Acyclic Graph (DAG)
2. **Automatic Connection**: Steps are automatically connected based on the DAG structure
3. **Message Passing**: A message passing algorithm propagates information between steps
4. **Placeholder Handling**: Placeholder variables are automatically handled by the template

See the [Pipeline Builder](../pipeline_builder/README.md) documentation for more details on the template-based approach.

## Common Pipeline Components

Most pipelines include some combination of these components:

- **Data Loading**: Using the [Cradle Data Load Step](../pipeline_steps/data_load_step_cradle.md)
- **Data Preprocessing**: Using the [Tabular Preprocessing Step](../pipeline_steps/tabular_preprocessing_step.md)
- **Model Training**: Using either [XGBoost Training Step](../pipeline_steps/training_step_xgboost.md) or [PyTorch Training Step](../pipeline_steps/training_step_pytorch.md)
- **Model Creation**: Using either [XGBoost Model Step](../pipeline_steps/model_step_xgboost.md) or [PyTorch Model Step](../pipeline_steps/model_step_pytorch.md)
- **Model Packaging**: Using the [MIMS Packaging Step](../pipeline_steps/mims_packaging_step.md)
- **Model Registration**: Using the [MIMS Registration Step](../pipeline_steps/mims_registration_step.md)

## Pipeline Flow Patterns

The pipelines demonstrate several common flow patterns:

1. **Training Flow**: Data Loading → Preprocessing → Training → Model Creation → Packaging → Registration
2. **Calibration Flow**: Data Loading → Preprocessing
3. **Deployment Flow**: Model Creation → Packaging → Registration

## Usage Patterns

### Traditional Pattern

The typical usage pattern for traditional pipelines is:

1. Create a pipeline builder with a configuration file
2. Generate the pipeline
3. Execute the pipeline
4. Fill the execution document with step configurations

Example:
```python
# Create builder
builder = MDSXGBoostPipelineBuilder(config_path="path/to/config.json")

# Generate pipeline
pipeline = builder.generate_pipeline()

# Execute pipeline
execution = pipeline.start()

# Fill execution document
execution_doc = builder.fill_execution_document(execution.describe())
```

### Template-Based Pattern

The usage pattern for template-based pipelines is:

1. Define the pipeline structure as a DAG
2. Create a config map that maps step names to configuration instances
3. Create a step builder map that maps step types to step builder classes
4. Use the template to generate the pipeline

Example:
```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_node("train")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "train")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "preprocess": preprocess_config,
    "train": train_config,
}

# Create the step builder map
step_builder_map = {
    "DataLoadStep": DataLoadStepBuilder,
    "PreprocessStep": PreprocessStepBuilder,
    "TrainStep": TrainStepBuilder,
}

# Create the template
template = PipelineBuilderTemplate(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    sagemaker_session=sagemaker_session,
    role=role,
)

# Generate the pipeline
pipeline = template.generate_pipeline("my-pipeline")
```

This template-based approach automatically handles the connections between steps, eliminating the need for manual wiring of inputs and outputs. It's particularly valuable for handling placeholder variables like `dependency_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri`.

See the [Pipeline Builder](../pipeline_builder/README.md) documentation for more details.
