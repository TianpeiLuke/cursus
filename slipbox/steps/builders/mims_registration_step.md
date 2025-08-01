# MIMS Registration Step

## Task Summary
The MIMS Registration Step registers a packaged model with the Model Inventory Management System (MIMS). This step:

1. Takes a packaged model artifact from a previous packaging step
2. Optionally takes generated payload samples from a previous payload step
3. Registers the model with MIMS, providing metadata about the model's inputs, outputs, and usage
4. Optionally registers the model in multiple AWS regions
5. Creates a model package group in SageMaker Model Registry

The step now uses step specifications and script contracts to standardize input/output paths and dependencies.

This step is the final component in the MIMS registration workflow:
- It takes the packaged model from the [MIMS Packaging Step](mims_packaging_step.md) as a required input
- It takes the test payloads from the [MIMS Payload Step](mims_payload_step.md) as an optional input
- It combines these inputs to register the model with MIMS and make it available for deployment

## Input and Output Format

### Input
- **PackagedModel**: Model packaged according to MIMS requirements from a previous packaging step
- **GeneratedPayloadSamples** (optional): Custom MIMS payload configuration
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

Note: The step can automatically extract these inputs from dependencies using the dependency resolver.

### Output
- **Registered Model**: Model registered in MIMS and SageMaker Model Registry (this is a side-effect of the step, not an actual file output)
- No actual processing output files are produced by this step

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| region | AWS region for model registration | Required |
| framework | ML framework used for the model | xgboost |
| inference_instance_type | Instance type for inference endpoint/transform job | ml.m5.large |
| inference_entry_point | Entry point script for inference | inference.py |
| model_registration_domain | Domain for model registration | BuyerSellerMessaging |
| model_registration_objective | Objective of model registration | Required |
| source_model_inference_content_types | Content type for model inference input | ["text/csv"] |
| source_model_inference_response_types | Response type for model inference output | ["application/json"] |
| source_model_inference_output_variable_list | Dictionary of output variables and their types | {"legacy-score": "NUMERIC"} |
| source_model_inference_input_variable_list | Input variables and their types | {} |

## Special Input Handling

The MIMS Registration Step has specific requirements for its inputs to comply with the MIMS SDK:
- **Input Order Matters**: The PackagedModel input must be first, followed by GeneratedPayloadSamples if provided
- **Input Naming**: Input names must match exactly with the expected MIMS SDK naming conventions
- **Container Paths**: Container destination paths must match MIMS SDK expectations
  - Model: `/opt/ml/processing/input/model`
  - Payload: `/opt/ml/processing/mims_payload`

## Validation Rules
- model_registration_domain and model_registration_objective must be provided
- region must be a valid AWS region
- framework must be one of: 'xgboost', 'sklearn', 'pytorch', 'tensorflow'
- source_model_inference_content_types and source_model_inference_response_types must be properly formatted
- At least one output variable must be defined
- Variable types must be either 'NUMERIC' or 'TEXT'

## Specification and Contract Support

The MIMS Registration Step uses:
- **Step Specification**: Defines input/output relationships and dependencies
- **Script Contract**: Defines expected container paths for script inputs/outputs

These help standardize integration with the Pipeline Builder Template and ensure consistent handling of inputs and outputs.

## Usage Example
```python
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig, VariableType
from src.pipeline_steps.builder_mims_registration_step import ModelRegistrationStepBuilder

# Create configuration
config = ModelRegistrationConfig(
    region="us-west-2",
    framework="xgboost",
    inference_instance_type="ml.m5.large",
    inference_entry_point="inference.py",
    model_registration_domain="BuyerSellerMessaging",
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
    }
)

# Create builder and step
builder = ModelRegistrationStepBuilder(config=config)

# Standard approach with automatic input extraction from dependencies
registration_step = builder.create_step(
    # The step can extract inputs from dependencies automatically
    dependencies=[packaging_step, payload_step]
)

# Add to pipeline
pipeline.add_step(registration_step)
```

## Legacy Parameter Support

For backward compatibility, the builder also supports legacy parameter names:
- `packaged_model_output`, `PackagedModel`, or `packaged_model` for the model artifacts
- `GeneratedPayloadSamples`, `generated_payload_samples`, `payload_sample`, `payload_s3_key`, or `payload_s3_uri` for the payload samples

## Variable Schema
The registration step generates a variable schema for MIMS registration with the following structure:

```json
{
  "input": {
    "variables": [
      {"name": "variable_name", "type": "NUMERIC|TEXT"}
    ]
  },
  "output": {
    "variables": [
      {"name": "variable_name", "type": "NUMERIC|TEXT"}
    ]
  }
}
```

## Integration with Pipeline Builder Template

### Input Arguments

The `ModelRegistrationStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| PackagedModel | Packaged model location | Yes | Previous packaging step's packaged_model_output |
| GeneratedPayloadSamples | Test payload location | No | Previous payload step's payload output |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the inputs and outputs are automatically connected based on the DAG structure:

```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_node("train")
dag.add_node("package")
dag.add_node("payload")
dag.add_node("register")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "train")
dag.add_edge("train", "package")
dag.add_edge("train", "payload")
dag.add_edge("package", "register")
dag.add_edge("payload", "register")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "preprocess": preprocess_config,
    "train": train_config,
    "package": package_config,
    "payload": payload_config,
    "register": register_config,
}

# Create the step builder map
step_builder_map = {
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
    "XGBoostTrainingStep": XGBoostTrainingStepBuilder,
    "MIMSPackagingStep": MIMSPackagingStepBuilder,
    "MIMSPayloadStep": MIMSPayloadStepBuilder,
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
- [MIMS Payload Step](mims_payload_step.md): Generates test payloads for the model
