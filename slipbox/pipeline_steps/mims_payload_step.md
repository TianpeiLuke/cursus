# MIMS Payload Step

## Task Summary
The MIMS Payload Step generates and uploads test payloads for a model to be registered with the Model Inventory Management System (MIMS). This step:

1. Creates sample payloads based on the model's input schema
2. Packages the payloads into a tar.gz archive
3. Uploads the archive to an S3 location
4. Provides the S3 URI for use in model testing and registration

The step now uses step specifications and script contracts to standardize input/output paths and dependencies.

This step is a key component in the MIMS registration workflow:
- It generates sample payloads based on the model's input and output schema
- Its output (payload S3 URI) is a required input for the [MIMS Registration Step](mims_registration_step.md)
- It typically runs after model training and before or in parallel with the packaging step

## Input and Output Format

### Input
- **Model Schema**: Input and output variable definitions from the model configuration
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

Note: The step can automatically extract necessary information from dependencies using the dependency resolver.

### Output
- **Payload Archive**: Generated payloads packaged in a tar.gz archive, stored in S3
- **ProcessingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| pipeline_name | Name of the pipeline | Required |
| bucket | S3 bucket for output storage | Required |
| region | AWS region | Inherited from base |
| expected_tps | Expected transactions per second | 2 |
| max_latency_in_millisecond | Maximum acceptable latency in milliseconds | 800 |
| max_acceptable_error_rate | Maximum acceptable error rate (0-1) | 0.2 |
| sample_payload_s3_key | S3 key for sample payload file | Auto-generated if not provided |
| default_numeric_value | Default value for numeric fields | 0.0 |
| default_string_value | Default value for text fields | "DEFAULT_TEXT" |
| processing_entry_point | Entry point script for payload generation | mims_payload.py |
| processing_source_dir | Directory containing processing scripts | Required |
| processing_framework_version | SKLearn framework version | 1.0-1 |
| processing_instance_type_small | Instance type for small processing | Inherited from base |
| processing_instance_type_large | Instance type for large processing | Inherited from base |
| processing_instance_count | Number of instances for processing | Inherited from base |
| processing_volume_size | EBS volume size for processing | Inherited from base |
| use_large_processing_instance | Whether to use large instance type | Inherited from base |
| special_field_values | Optional dictionary of special TEXT fields and their template values | None |
| processing_script_arguments | Optional arguments for the payload generation script | None |
| source_model_inference_content_types | Content type for model inference input | ["text/csv"] |
| source_model_inference_response_types | Response type for model inference output | ["application/json"] |
| source_model_inference_output_variable_list | Dictionary of output variables and their types | Required |
| source_model_inference_input_variable_list | Input variables and their types | Required |

## Environment Variables
The payload step sets the following environment variables for the processing job:
- **PIPELINE_NAME**: Name of the pipeline from configuration
- **REGION**: AWS region from configuration
- **CONTENT_TYPES**: Comma-separated list of content types from configuration
- **DEFAULT_NUMERIC_VALUE**: Default value for numeric fields (if provided)
- **DEFAULT_STRING_VALUE**: Default value for text fields (if provided)
- **PAYLOAD_S3_KEY**: S3 key for sample payload file (if provided)
- **BUCKET_NAME**: S3 bucket name (if provided)

## Validation Rules
- pipeline_name and bucket must be provided
- source_model_inference_content_types must be specified
- processing_instance_count and processing_volume_size must be provided
- If special_field_values is provided, all fields must exist in source_model_inference_input_variable_list and be of type TEXT
- expected_tps must be >= 1
- max_latency_in_millisecond must be between 100 and 10000
- max_acceptable_error_rate must be between 0.0 and 1.0

## Specification and Contract Support

The MIMS Payload Step uses:
- **Step Specification**: Defines input/output relationships and dependencies
- **Script Contract**: Defines expected container paths for script inputs/outputs

These help standardize integration with the Pipeline Builder Template and ensure consistent handling of inputs and outputs.

## Usage Example
```python
from src.pipeline_steps.config_mims_payload_step import PayloadConfig, VariableType
from src.pipeline_steps.builder_mims_payload_step import MIMSPayloadStepBuilder

# Create configuration
config = PayloadConfig(
    expected_tps=5,
    max_latency_in_millisecond=500,
    max_acceptable_error_rate=0.1,
    model_registration_objective="FraudDetection",
    source_model_inference_content_types=["text/csv"],
    source_model_inference_response_types=["application/json"],
    source_model_inference_output_variable_list={
        "fraud_score": VariableType.NUMERIC,
        "risk_category": VariableType.TEXT
    },
    source_model_inference_input_variable_list={
        "account_age": VariableType.NUMERIC,
        "transaction_count": VariableType.NUMERIC,
        "country_code": VariableType.TEXT
    },
    special_field_values={
        "country_code": "US-{timestamp}"
    },
    bucket="my-bucket",
    pipeline_name="fraud-detection",
    processing_source_dir="s3://my-bucket/scripts/",
    processing_entry_point="mims_payload.py"
)

# Create builder and step
builder = MIMSPayloadStepBuilder(config=config)
payload_step = builder.create_step(
    # The step can extract needed information from dependencies
    dependencies=[training_step]
)

# Add to pipeline
pipeline.add_step(payload_step)
```

## Payload Generation
The payload step generates sample payloads for each content type specified in the configuration:

### CSV Format
For `text/csv` content type, the payload is a comma-separated string of values following the order in `source_model_inference_input_variable_list`.

Example:
```
0.0,0.0,DEFAULT_TEXT
```

### JSON Format
For `application/json` content type, the payload is a JSON object with field names and values.

Example:
```json
{
  "account_age": "0.0",
  "transaction_count": "0.0",
  "country_code": "DEFAULT_TEXT"
}
```

### Special Field Values
If `special_field_values` is provided, the specified fields will use the template values instead of the default values. Templates can include placeholders like `{timestamp}` which will be replaced with the current timestamp.

## Integration with Pipeline Builder Template

### Input Arguments

The `MIMSPayloadStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| model_schema | Model schema information | Optional | Previous step's schema output |

Note: Most payload generation is based on configuration rather than inputs from previous steps.

### Output Properties

The `MIMSPayloadStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| payload_output | S3 URI of the generated payload archive | `step.properties.ProcessingOutputConfig.Outputs["payload_output"].S3Output.S3Uri` |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the inputs and outputs are automatically connected based on the DAG structure:

```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_node("train")
dag.add_node("payload")
dag.add_node("package")
dag.add_node("register")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "train")
dag.add_edge("train", "payload")
dag.add_edge("train", "package")
dag.add_edge("payload", "register")
dag.add_edge("package", "register")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "preprocess": preprocess_config,
    "train": train_config,
    "payload": payload_config,
    "package": package_config,
    "register": register_config,
}

# Create the step builder map
step_builder_map = {
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
    "XGBoostTrainingStep": XGBoostTrainingStepBuilder,
    "MIMSPayloadStep": MIMSPayloadStepBuilder,
    "MIMSPackagingStep": MIMSPackagingStepBuilder,
    "ModelRegistrationStep": ModelRegistrationStepBuilder,
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

## Related Steps
- [MIMS Packaging Step](mims_packaging_step.md): Prepares a trained model for deployment in MIMS
- [MIMS Registration Step](mims_registration_step.md): Registers a packaged model with MIMS
