# Batch Transform Step

## Task Summary
The Batch Transform Step creates a SageMaker Batch Transform job to generate predictions using a trained model. This step:

1. Configures a SageMaker Transformer object with specified instance type, count, and output path
2. Sets up the transform input with data location and content negotiation parameters
3. Creates a TransformStep in the SageMaker pipeline with appropriate dependencies
4. Validates the configuration parameters before execution

The step now uses step specifications and script contracts to standardize input/output paths and dependencies, with different specifications based on job type (training, testing, validation, or calibration).

## Input and Output Format

### Input
- **Model Name**: Name or Properties reference to a trained SageMaker model
- **Input Data**: CSV or other formatted data in S3 that needs predictions
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

Note: The step can automatically extract inputs from dependencies using the dependency resolver.

### Output
- **Predictions**: Model predictions stored in the specified S3 output location
- **TransformStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| job_type | Dataset slice to transform ('training', 'testing', 'validation', 'calibration') | Required |
| transform_instance_type | Instance type for the transform job | ml.m5.large |
| transform_instance_count | Number of instances for the transform job | 1 |
| content_type | MIME type of input data | text/csv |
| accept | Response MIME type | text/csv |
| split_type | How to split the input file | Line |
| assemble_with | How to reassemble input+output | Line |
| input_filter | JMESPath filter on each input record | $[1:] |
| output_filter | JMESPath filter on each joined record | $[-1] |
| join_source | Whether to join on 'Input' or 'Output' stream | Input |

## Validation Rules
- job_type must be one of: 'training', 'testing', 'validation', 'calibration'
- transform_instance_type and transform_instance_count must be provided
- When join_source='Input', assemble_with must equal split_type

## Specification and Contract Support

The Batch Transform Step uses different specifications based on job type:
- **BATCH_TRANSFORM_TRAINING_SPEC**: For transform jobs on training data
- **BATCH_TRANSFORM_TESTING_SPEC**: For transform jobs on testing data
- **BATCH_TRANSFORM_VALIDATION_SPEC**: For transform jobs on validation data
- **BATCH_TRANSFORM_CALIBRATION_SPEC**: For transform jobs on calibration data

These specifications define input/output relationships and dependencies, helping standardize integration with the Pipeline Builder Template.

## Usage Example
```python
from src.pipeline_steps.config_batch_transform_step import BatchTransformStepConfig
from src.pipeline_steps.builder_batch_transform_step import BatchTransformStepBuilder

# Create configuration
config = BatchTransformStepConfig(
    job_type="testing",
    transform_instance_type="ml.c5.xlarge",
    transform_instance_count=1,
    content_type="text/csv",
    accept="text/csv",
    split_type="Line",
    assemble_with="Line",
    input_filter="$[1:]",
    output_filter="$[-1]",
    join_source="Input"
)

# Create builder and step
builder = BatchTransformStepBuilder(config=config)
transform_step = builder.create_step(
    # The step can extract inputs from dependencies automatically
    dependencies=[model_step, preprocessing_step]
)

# Add to pipeline
pipeline.add_step(transform_step)
```

## Integration with Pipeline Builder Template

### Input Arguments

The `BatchTransformStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| model_name | Model name or ARN to use for transform | Yes | Previous step's ModelName property |
| processed_data | Input data location | Yes | Previous step's processed_data output |

### Output Properties

The `BatchTransformStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| transform_output | Transform output location | `step.properties.TransformOutput.S3OutputPath` |

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
dag.add_edge("preprocess", "transform")  # For input data

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
