# Hyperparameter Preparation Step

## Task Summary
The Hyperparameter Preparation Step serializes model hyperparameters to JSON and uploads them to S3, making them available for the training step. This step:

1. Takes a set of hyperparameters defined in the configuration
2. Serializes them to a JSON format
3. Uploads the JSON file to the specified S3 location
4. Returns the S3 URI to the uploaded hyperparameters file

The step uses a Lambda function which is more lightweight than a ProcessingStep for this simple task.

## Input and Output Format

### Input
- **Hyperparameters**: XGBoostModelHyperparameters object containing the hyperparameters to be used for training
- **S3 URI**: Target S3 location where the hyperparameters file should be uploaded
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

### Output
- **Hyperparameters S3 URI**: The full S3 URI to the uploaded hyperparameters.json file

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| hyperparameters | XGBoost model hyperparameters to be prepared and uploaded to S3 | Required |
| hyperparameters_s3_uri | S3 URI prefix under which hyperparameters.json will be uploaded | Required (or auto-generated) |
| lambda_timeout | Timeout for the Lambda function in seconds | 60 |
| lambda_memory_size | Memory size for the Lambda function in MB | 128 |

## Validation Rules
- hyperparameters must be provided and non-empty
- hyperparameters_s3_uri must be a valid S3 URI
- The output_names dictionary must contain the key 'hyperparameters_s3_uri'

## Usage Example
```python
from src.pipeline_steps.config_hyperparameter_prep_step import HyperparameterPrepConfig
from src.pipeline_steps.builder_hyperparameter_prep_step import HyperparameterPrepStepBuilder
from src.pipeline_steps.hyperparameters_xgboost import XGBoostModelHyperparameters

# Create hyperparameters object
hyperparams = XGBoostModelHyperparameters(
    max_depth=6,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    silent=0,
    objective="binary:logistic",
    num_round=100
)

# Create configuration
config = HyperparameterPrepConfig(
    hyperparameters=hyperparams,
    hyperparameters_s3_uri="s3://my-bucket/my-pipeline/training_config/2025-07-13"
)

# Create builder and step
builder = HyperparameterPrepStepBuilder(config=config)
hyperparameter_prep_step = builder.create_step(
    dependencies=[data_loading_step]
)

# Add to pipeline
pipeline.add_step(hyperparameter_prep_step)
```

## Integration with Pipeline Builder Template

### Input Arguments

The `HyperparameterPrepStepBuilder` doesn't require any direct inputs from previous steps as the hyperparameters are defined in the configuration.

### Output Properties

The `HyperparameterPrepStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| hyperparameters_s3_uri | S3 URI to the hyperparameters JSON file | `step.properties.Outputs["hyperparameters_s3_uri"]` |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the step can be easily integrated into your pipeline DAG:

```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("hyperparameter_prep")
dag.add_node("train")
dag.add_edge("data_load", "hyperparameter_prep")
dag.add_edge("hyperparameter_prep", "train")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "hyperparameter_prep": hyperparameter_prep_config,
    "train": train_config,
}

# Create the step builder map
step_builder_map = {
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "HyperparameterPrepStep": HyperparameterPrepStepBuilder, 
    "XGBoostTrainingStep": XGBoostTrainingStepBuilder,
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
