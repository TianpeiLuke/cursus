# XGBoost Model Step

## Task Summary
The XGBoost Model Step creates a SageMaker model artifact from a trained XGBoost model. This step:

1. Takes trained model artifacts from a previous training step
2. Creates a SageMaker model with the specified inference configuration
3. Configures the model with appropriate environment variables and resource limits
4. Prepares the model for deployment or batch transformation

The step now uses step specifications and script contracts to standardize input/output paths and dependencies.

## Input and Output Format

### Input
- **Model Data**: S3 path to trained XGBoost model artifacts
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

Note: The step can automatically extract model data input from dependencies using the dependency resolver.

### Output
- **Model**: SageMaker model artifact that can be deployed to an endpoint or used for batch transformation
- **ModelStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| instance_type | Instance type for inference endpoint/transform job | ml.m5.large |
| entry_point | Entry point script for inference | inference.py |
| source_dir | Directory containing inference scripts | Required |
| framework_version | XGBoost framework version | Inherited from base |
| py_version | Python version for the SageMaker XGBoost container | py3 |
| use_xgboost_framework | Whether to use XGBoost framework or custom image | True |
| image_uri | Custom image URI for inference (when not using XGBoost framework) | None (Required when use_xgboost_framework=False) |
| accelerator_type | Optional accelerator type for the endpoint | None |
| tags | Optional tags for the model | None |

## Validation Rules
- entry_point must be provided
- instance_type must be a valid SageMaker instance type
- When use_xgboost_framework is False, image_uri must be provided
- If source_dir is provided and not an S3 URI, the entry point script must exist

## Environment Variables
The model step can set custom environment variables for the model container via the `env` configuration dictionary. These variables can be used to control the behavior of the inference code.

## Specification and Contract Support

The XGBoost Model Step uses:
- **Step Specification**: Defines input/output relationships and dependencies
- **Script Contract**: Defines expected container paths for script inputs/outputs

These help standardize integration with the Pipeline Builder Template and ensure consistent handling of inputs and outputs.

## Usage Example
```python
from src.pipeline_steps.config_model_step_xgboost import XGBoostModelStepConfig
from src.pipeline_steps.builder_model_step_xgboost import XGBoostModelStepBuilder

# Create configuration
config = XGBoostModelStepConfig(
    instance_type="ml.m5.large",
    entry_point="inference.py",
    source_dir="s3://my-bucket/inference-scripts/",
    framework_version="1.5-1",
    py_version="py3",
    use_xgboost_framework=True
)

# Create builder and step
builder = XGBoostModelStepBuilder(config=config)
model_step = builder.create_step(
    # The step can extract model_data from dependencies automatically
    dependencies=[training_step]
)

# Add to pipeline
pipeline.add_step(model_step)
```

## Custom Image Usage Example
```python
from src.pipeline_steps.config_model_step_xgboost import XGBoostModelStepConfig
from src.pipeline_steps.builder_model_step_xgboost import XGBoostModelStepBuilder

# Create configuration using custom image
config = XGBoostModelStepConfig(
    instance_type="ml.m5.large",
    entry_point="inference.py",
    source_dir="s3://my-bucket/inference-scripts/",
    use_xgboost_framework=False,
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:latest"
)

# Create builder and step
builder = XGBoostModelStepBuilder(config=config)
model_step = builder.create_step(
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    dependencies=[training_step]
)

# Add to pipeline
pipeline.add_step(model_step)
```

## Integration with Pipeline Builder Template

### Input Arguments

The `XGBoostModelStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| model_data | Model artifacts location | Yes | Previous step's model_artifacts output |

### Output Properties

The `XGBoostModelStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| model_name | Model name | `step.properties.ModelName` |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the inputs and outputs are automatically connected based on the DAG structure:

```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_node("train")
dag.add_node("model")
dag.add_node("transform")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "train")
dag.add_edge("train", "model")
dag.add_edge("model", "transform")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "preprocess": preprocess_config,
    "train": train_config,
    "model": model_config,
    "transform": transform_config,
}

# Create the step builder map
step_builder_map = {
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
    "XGBoostTrainingStep": XGBoostTrainingStepBuilder,
    "XGBoostModelStep": XGBoostModelStepBuilder,
    "BatchTransformStep": BatchTransformStepBuilder,
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

For more details on how the Pipeline Builder Template handles connections between steps, see the [Pipeline Builder documentation](../pipeline_builder/README.md).
