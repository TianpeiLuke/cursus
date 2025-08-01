# PyTorch BSM Pipeline

## Pipeline Summary
The PyTorch BSM Pipeline is a model deployment pipeline that takes a pre-trained PyTorch model and prepares it for deployment. It focuses on the deployment aspects of the machine learning workflow, including model creation, packaging, and registration, without including the training steps.

## Steps Involved

1. **PyTorch Model Step**
   - Uses [PyTorch Model Step](../pipeline_steps/model_step_pytorch.md)
   - Creates a SageMaker model artifact from a pre-trained PyTorch model

2. **MIMS Packaging Step**
   - Uses [MIMS Packaging Step](../pipeline_steps/mims_packaging_step.md)
   - Packages the model for deployment in MIMS

3. **MIMS Registration Step**
   - Uses [MIMS Registration Step](../pipeline_steps/mims_registration_step.md)
   - Registers the model with MIMS

## Step Connections (Adjacency List)

```
- PyTorch Model Step → MIMS Packaging Step
- MIMS Packaging Step → MIMS Registration Step
```

## Input/Output Connections

### External Input → PyTorch Model Step
- **Input**: The pipeline takes an external S3 path to a pre-trained model artifact (model.tar.gz)
- **Output**: The model step creates a SageMaker model and stores the model path in `model_artifacts_path`

### PyTorch Model Step → MIMS Packaging Step
- **Output**: PyTorch model step provides the model artifacts path via `model_artifacts_path`
- **Input**: MIMS packaging step takes this path as input for packaging
- **Note**: The model step also serves as a dependency for the packaging step

### MIMS Packaging Step → MIMS Registration Step
- **Output**: MIMS packaging outputs packaged model to S3 via `ProcessingOutputConfig.Outputs[0].S3Output.S3Uri`
- **Input**: Model registration takes the packaged model S3 URI as input
- **Note**: The packaging step also serves as a dependency for the registration step

## Notes
- This pipeline is designed for deployment of pre-trained models, not for training
- The pipeline enforces region settings, forcing model creation to occur in the NA region (us-east-1)
- The pipeline handles execution document configuration for model registration
- The pipeline validates the provided model S3 path to ensure it's a valid S3 URI
- The pipeline supports multiple regions for model registration
- The pipeline uses a configuration-driven approach, extracting all settings from a JSON configuration file
- The pipeline generates and uploads payloads for model testing during registration

## Usage Example
The pipeline is initialized with a configuration file and then generated with a specific model S3 path:

```python
builder = BSMPytorchPipelineBuilder(config_path="path/to/config.json")
pipeline = builder.generate_pipeline(model_s3_path="s3://bucket/path/to/model.tar.gz")
```
