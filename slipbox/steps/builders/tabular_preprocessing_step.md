# Tabular Preprocessing Step

## Task Summary
The Tabular Preprocessing Step prepares tabular data for model training by performing various preprocessing operations. This step:

1. Takes raw tabular data as input
2. Performs data cleaning, transformation, and feature engineering
3. Optionally splits the data into training, validation, and testing sets
4. Outputs the processed data to S3 for use in subsequent pipeline steps

The step now uses step specifications and script contracts to standardize input/output paths and dependencies, with different specifications based on job type (training, testing, validation, or calibration).

## Input and Output Format

### Input
- **Raw Data**: Raw tabular data from a previous step or S3 location
- **Optional Metadata**: Metadata about the dataset (optional)
- **Optional Signature**: Data signatures for verification (optional)
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

Note: The step can automatically extract inputs from dependencies using the dependency resolver.

### Output
- **Processed Data**: Preprocessed tabular data ready for model training
- **Full Data**: Complete dataset before splitting (if applicable)
- **ProcessingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| processing_entry_point | Relative path to preprocessing script | tabular_preprocess.py |
| processing_source_dir | Directory containing processing scripts | Required |
| processing_instance_type_small | Instance type for small processing | Inherited from base |
| processing_instance_type_large | Instance type for large processing | Inherited from base |
| processing_instance_count | Number of instances for processing | Inherited from base |
| processing_volume_size | EBS volume size for processing | Inherited from base |
| processing_framework_version | SKLearn framework version for processing | Required |
| use_large_processing_instance | Whether to use large instance type | False |
| hyperparameters | Model hyperparameters (only label_name is used) | ModelHyperparameters() |
| job_type | Dataset type ('training', 'validation', 'testing', 'calibration') | training |
| train_ratio | Fraction of data for training set | 0.7 |
| test_val_ratio | Fraction of holdout for test vs validation | 0.5 |
| categorical_columns | List of categorical column names | [] |
| numerical_columns | List of numerical column names | [] |
| text_columns | List of text column names | [] |
| date_columns | List of date column names | [] |

## Environment Variables
The preprocessing step sets the following environment variables for the processing job:
- **LABEL_FIELD**: The name of the label field from hyperparameters
- **TRAIN_RATIO**: The fraction of data to allocate to the training set
- **TEST_VAL_RATIO**: The fraction of the holdout to allocate to the test set vs. validation
- **CATEGORICAL_COLUMNS**: Comma-separated list of categorical column names (if provided)
- **NUMERICAL_COLUMNS**: Comma-separated list of numerical column names (if provided)
- **TEXT_COLUMNS**: Comma-separated list of text column names (if provided)
- **DATE_COLUMNS**: Comma-separated list of date column names (if provided)

## Validation Rules
- processing_entry_point must be provided
- job_type must be one of: 'training', 'validation', 'testing', 'calibration'
- train_ratio and test_val_ratio must be strictly between 0 and 1
- hyperparameters.label_name must be provided and non-empty
- Processing attributes (instance count, volume size, etc.) must be provided

## Specification and Contract Support

The Tabular Preprocessing Step uses different specifications based on job type:
- **PREPROCESSING_TRAINING_SPEC**: For preprocessing jobs on training data
- **PREPROCESSING_TESTING_SPEC**: For preprocessing jobs on testing data
- **PREPROCESSING_VALIDATION_SPEC**: For preprocessing jobs on validation data
- **PREPROCESSING_CALIBRATION_SPEC**: For preprocessing jobs on calibration data

These specifications define input/output relationships and dependencies, helping standardize integration with the Pipeline Builder Template.

## Usage Example
```python
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters

# Create hyperparameters
hyperparams = ModelHyperparameters(
    label_name="target",
    feature_names=["feature1", "feature2", "feature3"]
)

# Create configuration
config = TabularPreprocessingConfig(
    processing_entry_point="tabular_preprocess.py",
    processing_source_dir="s3://my-bucket/scripts/",
    processing_framework_version="1.0-1",
    processing_instance_count=1,
    processing_volume_size=30,
    processing_instance_type_small="ml.m5.xlarge",
    processing_instance_type_large="ml.m5.2xlarge",
    hyperparameters=hyperparams,
    job_type="training",
    train_ratio=0.8,
    test_val_ratio=0.5,
    categorical_columns=["category1", "category2"],
    numerical_columns=["numeric1", "numeric2"]
)

# Create builder and step
builder = TabularPreprocessingStepBuilder(config=config)
preprocessing_step = builder.create_step(
    # The step can extract inputs from dependencies automatically
    dependencies=[data_loading_step]
)

# Add to pipeline
pipeline.add_step(preprocessing_step)
```

## Command-line Arguments
The preprocessing step passes the following command-line arguments to the processing script:
- `--job_type`: Dataset type (training, validation, testing, calibration)

## Integration with Pipeline Builder Template

### Input Arguments

The `TabularPreprocessingStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| DATA | Raw data input location | Yes | Previous step's DATA output |
| METADATA | Metadata input location | No | Previous step's METADATA output |
| SIGNATURE | Signature input location | No | Previous step's SIGNATURE output |

### Output Properties

The `TabularPreprocessingStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| processed_data | Processed data location | `step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri` |
| full_data | Full data location | `step.properties.ProcessingOutputConfig.Outputs["full_data"].S3Output.S3Uri` |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the inputs and outputs are automatically connected based on the DAG structure:

```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_edge("data_load", "preprocess")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "preprocess": preprocess_config,
}

# Create the step builder map
step_builder_map = {
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
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
