# PyTorch Training Step

## Task Summary
The PyTorch Training Step configures and executes a PyTorch model training job in SageMaker. This step:

1. Creates a PyTorch estimator with the specified configuration and hyperparameters
2. Configures a single data channel containing training, validation, and test data
3. Sets up metric monitoring for tracking training progress
4. Executes the training job with the specified instance type and count
5. Outputs the trained model artifacts to S3

The step now uses step specifications and script contracts to standardize input/output paths and dependencies.

## Input and Output Format

### Input
- **Input Data**: Directory containing train, val, and test subdirectories
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

Note: The step can automatically extract inputs from dependencies using the dependency resolver.

### Output
- **Model Artifacts**: Trained PyTorch model artifacts stored in S3
- **TrainingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| training_instance_type | Instance type for training job | ml.g5.12xlarge |
| training_instance_count | Number of instances for training job | 1 |
| training_volume_size | Volume size (GB) for training instances | 30 |
| training_entry_point | Entry point script for training | train.py |
| source_dir | Directory containing training scripts | Required |
| framework_version | PyTorch framework version | Required |
| py_version | Python version | Required |
| hyperparameters | Model hyperparameters | Optional |
| env | Environment variables for the training job | {} |

## Metrics Monitoring
The training step monitors the following metrics during training:
- **Train Loss**: Training loss value
- **Validation Loss**: Validation loss value
- **Validation F1 Score**: F1 score on validation data
- **Validation AUC ROC**: Area under ROC curve on validation data

## Validation Rules
- training_instance_type must be a valid SageMaker instance type
- training_entry_point must be provided
- source_dir must be provided
- framework_version must be provided
- py_version must be provided

## Specification and Contract Support

The PyTorch Training Step uses:
- **Step Specification**: Defines input/output relationships and dependencies
- **Script Contract**: Defines expected container paths for script inputs/outputs

These help standardize integration with the Pipeline Builder Template and ensure consistent handling of inputs and outputs.

## Usage Example
```python
from src.pipeline_steps.config_training_step_pytorch import PyTorchTrainingConfig
from src.pipeline_steps.builder_training_step_pytorch import PyTorchTrainingStepBuilder
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters

# Create hyperparameters
hyperparams = ModelHyperparameters(
    id_name="customer_id",
    label_name="target",
    full_field_list=["customer_id", "target", "feature1", "feature2", "feature3"],
    tab_field_list=["feature1", "feature2"],
    cat_field_list=["feature3"],
    learning_rate=0.001,
    batch_size=64,
    epochs=10
)

# Create configuration
config = PyTorchTrainingConfig(
    training_instance_type="ml.g5.12xlarge",
    training_instance_count=1,
    training_volume_size=30,
    training_entry_point="train.py",
    source_dir="s3://my-bucket/scripts/",
    framework_version="1.13.1",
    py_version="py39",
    hyperparameters=hyperparams,
    env={
        "CA_REPOSITORY_ARN": "arn:aws:codeartifact:us-east-1:123456789012:repository/my-domain/my-repo"
    }
)

# Create builder and step
builder = PyTorchTrainingStepBuilder(config=config)
training_step = builder.create_step(
    # The step can extract inputs from dependencies automatically
    dependencies=[preprocessing_step]
)

# Add to pipeline
pipeline.add_step(training_step)
```

## Data Channel Structure

Unlike the XGBoost training step which uses separate channels for train/val/test data, the PyTorch training step expects:
1. A single "data" channel pointing to a directory
2. This directory should contain train/, val/, and test/ subdirectories
3. The PyTorch training script is responsible for loading data from these subdirectories

This structure allows for more flexibility in data loading within the PyTorch script.

## Environment Variables
The training step can set custom environment variables for the training job via the `env` configuration dictionary. Common environment variables include:
- **CA_REPOSITORY_ARN**: ARN for a secure PyPI repository

## Integration with Pipeline Builder Template

### Input Arguments

The `PyTorchTrainingStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| input_path | Path containing train/val/test data | Yes | Previous step's processed_data output |

### Output Properties

The `PyTorchTrainingStepBuilder` provides the following output properties that can be used by subsequent steps:

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
dag.add_node("model")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "train")
dag.add_edge("train", "model")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "preprocess": preprocess_config,
    "train": train_config,
    "model": model_config,
}

# Create the step builder map
step_builder_map = {
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
    "PyTorchTrainingStep": PyTorchTrainingStepBuilder,
    "PyTorchModelStep": PyTorchModelStepBuilder,
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
