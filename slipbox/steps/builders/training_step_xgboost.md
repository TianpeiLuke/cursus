# XGBoost Training Step

## Task Summary
The XGBoost Training Step configures and executes an XGBoost model training job in SageMaker. This step:

1. Automatically serializes and uploads hyperparameters as a JSON file to S3
2. Creates an XGBoost estimator with the specified configuration
3. Configures input channels for training, validation, and test data
4. Executes the training job with the specified instance type and count
5. Outputs the trained model artifacts to S3

The step now uses step specifications and script contracts to standardize input/output paths and dependencies.

## Input and Output Format

### Input
- **Train Data**: Training dataset from S3 (train/ subfolder)
- **Validation Data**: Validation dataset from S3 (val/ subfolder)
- **Test Data**: Test dataset from S3 (test/ subfolder)
- **Config**: Hyperparameters JSON file automatically generated and uploaded to S3
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

Note: The step can automatically extract inputs from dependencies using the dependency resolver.

### Output
- **Model Artifacts**: Trained XGBoost model artifacts stored in S3
- **TrainingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| input_path | S3 path for input training data | Generated from bucket and pipeline name |
| output_path | S3 path for output model artifacts | Generated from bucket and pipeline name |
| checkpoint_path | Optional S3 path for model checkpoints | Generated from bucket and pipeline name |
| training_instance_type | Instance type for XGBoost training job | ml.m5.xlarge |
| training_instance_count | Number of instances for XGBoost training job | 1 |
| training_volume_size | Volume size (GB) for training instances | 30 |
| training_entry_point | Entry point script for XGBoost training | train_xgb.py |
| framework_version | SageMaker XGBoost framework version | 1.7-1 |
| py_version | Python version for the SageMaker XGBoost container | py3 |
| hyperparameters | XGBoost model hyperparameters | Required |
| hyperparameters_s3_uri | S3 URI prefix for hyperparameters.json | Generated from bucket and pipeline name |

## Validation Rules
- input_path, output_path, and checkpoint_path must be valid S3 URIs
- All defined paths (input, output, checkpoint) must be unique
- Paths must have at least 2 levels of hierarchy (bucket + prefix)
- training_instance_type must be a valid SageMaker instance type for XGBoost
- All fields in tab_field_list and cat_field_list must be in full_field_list
- label_name and id_name must be in full_field_list

## Environment Variables
The training step sets the following environment variables for the training job:
- **CA_REPOSITORY_ARN**: ARN for the secure PyPI repository
- Additional environment variables can be specified in the config.env dictionary

## Usage Example
```python
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig
from src.pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from src.pipeline_steps.hyperparameters_xgboost import XGBoostModelHyperparameters

# Create hyperparameters
hyperparams = XGBoostModelHyperparameters(
    id_name="customer_id",
    label_name="target",
    full_field_list=["customer_id", "target", "feature1", "feature2", "feature3"],
    tab_field_list=["feature1", "feature2"],
    cat_field_list=["feature3"],
    xgb_params={
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.3,
        "num_round": 100
    }
)

# Create configuration
config = XGBoostTrainingConfig(
    input_path="s3://my-bucket/preprocessed-data/",
    output_path="s3://my-bucket/model-artifacts/",
    training_instance_type="ml.m5.2xlarge",
    training_instance_count=1,
    hyperparameters=hyperparams
)

# Create builder and step
builder = XGBoostTrainingStepBuilder(config=config)
training_step = builder.create_step(
    # The step can extract inputs from dependencies automatically
    dependencies=[preprocessing_step]
)

# Add to pipeline
pipeline.add_step(training_step)
```

## Input Channels
The training step configures the following input channels for the XGBoost estimator:
- **train**: Training data from {input_path}/train/
- **val**: Validation data from {input_path}/val/
- **test**: Test data from {input_path}/test/
- **config**: Hyperparameters JSON file from {hyperparameters_s3_uri}/hyperparameters.json

## Specification and Contract Support

The XGBoost Training Step now uses:
- **Step Specification**: Defines input/output relationships and dependencies
- **Script Contract**: Defines expected container paths for script inputs/outputs

These help standardize integration with the Pipeline Builder Template and ensure consistent handling of inputs and outputs.

## Integration with Pipeline Builder Template

### Input Arguments

The `XGBoostTrainingStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| input_path | Path to preprocessed data | Yes | Previous step's processed_data output |
| hyperparameters_s3_uri | S3 URI for hyperparameters | No | Generated internally (previous hyperparameter prep step ignored) |

### Output Properties

The `XGBoostTrainingStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| model_artifacts | Trained model artifacts | `step.properties.ModelArtifacts` |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the inputs and outputs are automatically connected based on the DAG structure:

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
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
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
