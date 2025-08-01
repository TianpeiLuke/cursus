# MODS MIMS Model Registration Script

## Overview

The MIMS (Model Inference Management Service) Model Registration script is responsible for registering trained ML models with the MIMS service, which handles model governance, versioning, and deployment. This registration step is a critical part of the ML model deployment pipeline, ensuring models are properly tracked and can be deployed in production environments.

## Script Location

```
secure_ai_sandbox_workflow_python_sdk/mims_model_registration/scripts/script.py
```

## Purpose

The script performs several key functions:
1. Upload model artifacts to S3 for registration
2. Process model registration configuration
3. Handle optional payload samples for inference testing
4. Register the model with the MIMS service
5. Track workflow execution for lineage and auditing

## Inputs

### Required Inputs

- **Model Artifacts**
  - Container Path: `/opt/ml/processing/input/model`
  - Description: Packaged model artifacts (typically .tar.gz) containing the trained model
  - Handling: The script extracts the model from this location and uploads it to a temporary S3 location

- **Configuration**
  - Container Path: `/opt/ml/processing/config/config`
  - Description: JSON configuration file with model registration parameters
  - Required Fields:
    - `model_domain`: Domain of the model (e.g., "buyer_abuse")
    - `model_objective`: Objective of the model (e.g., "risk_scoring")
    - `source_model_inference_content_types`: Content types supported by the model
    - `source_model_inference_response_types`: Response types supported by the model
    - `source_model_inference_input_variable_list`: Input variables for inference
    - `source_model_inference_output_variable_list`: Output variables from inference
    - `model_registration_region`: AWS region for model registration
    - `model_owner`: Owner of the model
    - `source_model_inference_image_arn`: ECR image ARN for inference

### Optional Inputs

- **Payload Samples**
  - Container Path: `/opt/ml/processing/mims_payload`
  - Description: Sample payloads for testing model inference
  - Handling: If provided, uploaded to S3 and included in registration

- **Performance Metadata**
  - Container Path: `/opt/ml/processing/input/metadata`
  - Description: Model performance metrics and metadata
  - Environment Variable: `PERFORMANCE_METADATA_PATH`
  - Handling: Downloaded from S3 if path is provided via environment variable

## Environment Variables

- **MODS_WORKFLOW_EXECUTION_ID**
  - Purpose: Links the model registration to a specific training workflow execution
  - Usage: Added to registration request for traceability

- **PERFORMANCE_METADATA_PATH**
  - Purpose: Specifies the S3 location of performance metadata
  - Usage: Optional; used to include performance metrics in registration

## Execution Flow

1. **Initialize Services**
   ```python
   sandbox_session = Session(session_folder="/tmp/")
   s3_resource = sandbox_session.resource("SharedBucketS3DataLoader")
   mims_resource = sandbox_session.resource("MIMSModelRegistrar")
   ```

2. **Upload Model Artifacts**
   ```python
   temp_uploaded_s3_path = upload_artifact(model_dir, "model.tar.gz")
   ```

3. **Process Optional Payload**
   ```python
   if os.path.exists(payload_dir):
       payload_s3_path = upload_artifact(payload_dir, "payload.tar.gz")
   ```

4. **Load Configuration**
   ```python
   with open("/opt/ml/processing/config/config") as config_file:
       model_registration_config = json.load(config_file)
   ```

5. **Register Model with MIMS**
   ```python
   sagemaker_model = mims_resource.register_model(
       model_domain=model_registration_config["model_domain"],
       model_objective=model_registration_config["model_objective"],
       # Additional parameters...
   )
   ```

6. **Wait for Registration Completion**
   ```python
   mims_resource.wait_for_done(sagemaker_model)
   ```

7. **Cleanup Temporary Resources**
   ```python
   s3_resource.delete_file(temp_uploaded_s3_path)
   ```

## Output

The script does not produce output files but instead registers the model with MIMS as a side effect. After successful execution:

1. The model is registered in MIMS with a unique model ID
2. The registration workflow execution ARN is printed for reference
3. The model ID is printed for reference

Example output:
```
Workflow execution arn for model registration process: arn:aws:states:us-west-2:012345678910:execution:mims-registration-workflow:a1b2c3d4
Model id for model being registered: model-12345678
```

## Error Handling

The script provides error handling for:
- Missing model files
- Invalid configuration parameters
- Multiple files in the model directory
- Failed uploads to S3
- Failed registration with MIMS

## Integration with MODS

This script is designed to work within the MODS (Model Operations and Development System) framework as a processing step. It is typically executed as part of a SageMaker processing step orchestrated by the MimsModelRegistrationProcessingStep class, which handles environment setup and input configuration.

## Related Components

1. **MimsModelRegistrationProcessor**: Creates a processing job for model registration
2. **MimsModelRegistrationProcessingStep**: Creates a SageMaker pipeline step for model registration
3. **MIMS_REGISTRATION_CONTRACT**: Defines the contract for the registration script
4. **REGISTRATION_SPEC**: Defines the step specification for the registration step

## Security Considerations

- The script uses a secure sandbox session for resource access
- Temporary S3 locations are used and cleaned up after registration
- Model artifacts are processed within the container's filesystem
- All connections to MIMS are authenticated via IAM roles
