# MIMS Packaging Step

## Task Summary
The MIMS Packaging Step prepares a trained model for deployment in the Model Inventory Management System (MIMS). This step:

1. Takes a trained model artifact and inference scripts as inputs
2. Packages them together in a format compatible with MIMS requirements
3. Outputs the packaged model to an S3 location
4. Prepares the model for subsequent registration with MIMS

The step now uses step specifications and script contracts to standardize input/output paths and dependencies.

This step is a critical part of the MIMS registration workflow:
- It receives model artifacts from a previous training or model creation step
- Its output (packaged model) is a required input for the [MIMS Registration Step](mims_registration_step.md)
- It typically runs in parallel with the [MIMS Payload Step](mims_payload_step.md), which generates test payloads for the model

## Input and Output Format

### Input
- **Model Artifacts**: Trained model artifacts from a previous training or model step
- **Inference Scripts**: Scripts needed for model inference (inference.py, etc.)
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

Note: The step can automatically extract model artifacts from dependencies using the dependency resolver, but it will always use the local inference scripts path from configuration rather than any dependency-provided values.

### Output
- **Packaged Model**: Model packaged according to MIMS requirements, stored in S3
- **ProcessingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| processing_entry_point | Entry point script for packaging | mims_package.py |
| processing_source_dir | Directory containing processing scripts | Required |
| source_dir | Directory containing inference scripts | Required (falls back to "inference" in notebook root) |
| processing_framework_version | SKLearn framework version | 1.0-1 |
| processing_instance_type_small | Instance type for small processing | Inherited from base |
| processing_instance_type_large | Instance type for large processing | Inherited from base |
| processing_instance_count | Number of instances for processing | Inherited from base |
| processing_volume_size | EBS volume size for processing | Inherited from base |
| use_large_processing_instance | Whether to use large instance type | Inherited from base |
| pipeline_name | Name of the pipeline | Required |
| region | AWS region | Inherited from base |
| model_type | Type of model being packaged | Optional |
| pipeline_version | Version of the pipeline | Optional |
| model_registration_objective | Registration objective | Optional |

## Environment Variables
The packaging step sets the following environment variables for the processing job:
- **PIPELINE_NAME**: Name of the pipeline from configuration
- **REGION**: AWS region from configuration
- **MODEL_TYPE**: Type of model (if provided)
- **BUCKET_NAME**: S3 bucket name (if provided)
- **PIPELINE_VERSION**: Pipeline version (if provided)
- **MODEL_OBJECTIVE**: Model registration objective (if provided)

## Validation Rules
- processing_entry_point must be provided
- Required attributes: processing_instance_count, processing_volume_size, processing_instance_type_large, processing_instance_type_small, pipeline_name

## Specification and Contract Support

The MIMS Packaging Step uses:
- **Step Specification**: Defines input/output relationships and dependencies
- **Script Contract**: Defines expected container paths for script inputs/outputs

These help standardize integration with the Pipeline Builder Template and ensure consistent handling of inputs and outputs.

## Usage Example
```python
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder

# Create configuration
config = PackageStepConfig(
    processing_entry_point="mims_package.py",
    source_dir="inference/",  # Local path, always preferred over dependency inputs
    processing_source_dir="s3://my-bucket/processing-scripts/",
    pipeline_name="MyModelPipeline",
    pipeline_s3_loc="s3://my-bucket/pipeline-outputs/",
    model_type="XGBoost",
    pipeline_version="1.0.0",
    model_registration_objective="fraud_detection"
)

# Create builder and step
builder = MIMSPackagingStepBuilder(config=config)
packaging_step = builder.create_step(
    # The step can extract model artifacts from dependencies automatically
    dependencies=[training_step]
)

# Add to pipeline
pipeline.add_step(packaging_step)
```

## Special Input Handling

The MIMS Packaging Step has special handling for inference scripts:
- It will **always** use the local inference scripts path from the configuration (`source_dir`)
- Any inference scripts input from dependencies will be ignored
- If no `source_dir` is provided, it falls back to an "inference" directory in the notebook root

This ensures that packaging always uses the correct inference scripts from the local environment.

## Integration with Pipeline Builder Template

### Input Arguments

The `MIMSPackagingStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| model_input | Model artifacts location | Yes | Previous step's model_artifacts output |

Note: inference_scripts_input is always taken from the local `source_dir` configuration and ignores any dependency-provided values.

### Output Properties

The `MIMSPackagingStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| packaged_model_output | Packaged model location | `step.properties.ProcessingOutputConfig.Outputs["packaged_model_output"].S3Output.S3Uri` |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the inputs and outputs are automatically connected based on the DAG structure:

```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_node("train")
dag.add_node("package")
dag.add_node("register")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "train")
dag.add_edge("train", "package")
dag.add_edge("package", "register")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "preprocess": preprocess_config,
    "train": train_config,
    "package": package_config,
    "register": register_config,
}

# Create the step builder map
step_builder_map = {
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
    "XGBoostTrainingStep": XGBoostTrainingStepBuilder,
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
- [MIMS Payload Step](mims_payload_step.md): Generates test payloads for the model
- [MIMS Registration Step](mims_registration_step.md): Registers the packaged model with MIMS
