# MIMS Registration Script Contract

## Overview
The MIMS Registration Script Contract defines the execution requirements for the MIMS model registration script (`script.py`). This contract ensures the script complies with SageMaker processing job conventions for registering trained models with the MIMS (Model Inference Management System) service.

## Contract Details

### Script Information
- **Entry Point**: `script.py`
- **Container Type**: SageMaker Processing Job
- **Framework**: Python with secure_ai_sandbox_python_lib
- **Purpose**: Model registration with MIMS service

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| PackagedModel | `/opt/ml/processing/input/model` | Packaged model artifacts (.tar.gz) |
| GeneratedPayloadSamples | `/opt/ml/processing/mims_payload` | Optional payload samples for inference testing |
| config | `/opt/ml/processing/config/config` | Configuration file for registration |
| metadata | `/opt/ml/processing/input/metadata` | Optional performance metadata |

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| N/A | N/A | No output paths - registration is a side effect |

### Environment Variables

#### Required Variables
- `MODS_WORKFLOW_EXECUTION_ID` - Workflow execution ID for tracking and lineage

#### Optional Variables
- `PERFORMANCE_METADATA_PATH` - S3 path to performance metadata

### Framework Requirements

#### Core Dependencies
```python
python>=3.7
# Uses secure_ai_sandbox_python_lib libraries and standard modules
```

## Script Functionality

Based on the contract definition in `src/pipeline_script_contracts/mims_registration_contract.py`:

### Model Registration Process
1. **Secure Session Management**:
   - Creates SandboxSession to interact with secure resources
   - Handles authentication and authorization for MIMS service
   - Manages secure communication channels

2. **Model Artifact Upload**:
   - Uploads packaged model artifacts to temporary S3 location
   - Handles large model files efficiently
   - Ensures secure transfer of model data

### Payload Sample Processing
1. **Optional Payload Upload**:
   - Uploads payload samples if provided for inference testing
   - Validates payload format and structure
   - Ensures payload compatibility with model

2. **Inference Testing Setup**:
   - Prepares payload samples for model validation
   - Configures inference testing parameters
   - Validates model-payload compatibility

### Registration Configuration
1. **Configuration Processing**:
   - Reads registration configuration from config file
   - Validates configuration parameters
   - Sets appropriate environment variables for registration

2. **Metadata Handling**:
   - Processes optional performance metadata
   - Uploads metadata to specified S3 location
   - Associates metadata with model registration

### Key Implementation Concepts

#### Secure Session Creation
```python
def create_sandbox_session():
    """Create secure sandbox session for MIMS interaction"""
    from secure_ai_sandbox_python_lib import SandboxSession
    return SandboxSession()
```

#### Model Upload Process
```python
def upload_model_artifacts(session, model_path: str, s3_location: str):
    """Upload model artifacts to temporary S3 location"""
    session.upload_file(model_path, s3_location)
    return s3_location
```

#### MIMS Registration
```python
def register_with_mims(session, config: dict, model_s3_uri: str):
    """Register model with MIMS service"""
    mims_resource = session.get_mims_resource()
    registration_result = mims_resource.register_model(
        model_uri=model_s3_uri,
        config=config,
        workflow_id=os.environ["MODS_WORKFLOW_EXECUTION_ID"]
    )
    return registration_result
```

#### Registration Monitoring
```python
def wait_for_registration_completion(session, registration_id: str):
    """Wait for model registration to complete"""
    mims_resource = session.get_mims_resource()
    while True:
        status = mims_resource.get_registration_status(registration_id)
        if status in ["COMPLETED", "FAILED"]:
            return status
        time.sleep(30)
```

### Registration Workflow
1. **Preparation Phase**:
   - Create secure sandbox session
   - Validate input files and configuration
   - Set up temporary S3 locations

2. **Upload Phase**:
   - Upload model artifacts to S3
   - Upload payload samples if provided
   - Upload performance metadata if available

3. **Registration Phase**:
   - Submit registration request to MIMS
   - Monitor registration progress
   - Handle registration completion or failure

4. **Cleanup Phase**:
   - Clean up temporary S3 resources
   - Log registration results
   - Handle error conditions

### Configuration Structure

#### Registration Config Format
```json
{
  "model_name": "my-model",
  "model_version": "1.0.0",
  "framework": "xgboost",
  "inference_config": {
    "instance_type": "ml.t2.medium",
    "initial_instance_count": 1
  },
  "tags": {
    "environment": "production",
    "team": "ml-team"
  }
}
```

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import MIMS_REGISTRATION_CONTRACT

# Access contract details
print(f"Entry Point: {MIMS_REGISTRATION_CONTRACT.entry_point}")
print(f"Required Env Vars: {MIMS_REGISTRATION_CONTRACT.required_env_vars}")
```

### Integration with Step Builder
```python
from src.pipeline_steps import MIMSRegistrationStepBuilder

class MIMSRegistrationStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        validation = MIMS_REGISTRATION_CONTRACT.validate_implementation(
            'script.py'
        )
        if not validation.is_valid:
            self.logger.warning(f"Script validation warnings: {validation.errors}")
```

## Integration Points

### Pipeline Position
- **Registration Node** - Final step in model deployment pipeline
- **Input Dependencies** - Requires packaged model and optional payload samples
- **Output Consumers** - Registers model for inference service consumption

### Data Flow
```
Packaged Model + Payload Samples → script.py → MIMS Registration → Deployed Model
```

## Best Practices

### Script Development
1. **Secure Communication** - Use secure channels for all MIMS interactions
2. **Error Handling** - Handle registration failures and retries gracefully
3. **Logging** - Log all registration steps for debugging and auditing
4. **Resource Cleanup** - Clean up temporary resources after registration

### Model Registration
1. **Configuration Validation** - Validate all registration parameters
2. **Metadata Management** - Include comprehensive model metadata
3. **Version Control** - Implement proper model versioning strategy
4. **Testing** - Validate model functionality before registration

## Related Contracts

### Upstream Contracts
- `MIMS_PACKAGE_CONTRACT` - Provides packaged model for registration
- `MIMS_PAYLOAD_CONTRACT` - Provides payload samples for testing

### Downstream Contracts
- MIMS inference service (external system)

## Troubleshooting

### Common Issues
1. **Authentication Failures** - Ensure proper credentials for MIMS service
2. **Upload Failures** - Handle S3 upload errors and retries
3. **Registration Timeouts** - Monitor registration progress and handle timeouts
4. **Configuration Errors** - Validate registration configuration format

### Validation Failures
1. **Path Validation** - Ensure all input paths exist and are accessible
2. **Environment Variables** - Check required environment variables are set
3. **Model Format** - Validate packaged model format and structure
4. **Service Availability** - Ensure MIMS service is available and responsive

## Performance Considerations

### Optimization Strategies
- **Efficient Uploads** - Optimize S3 upload performance for large models
- **Parallel Processing** - Upload model and payload samples in parallel
- **Retry Logic** - Implement intelligent retry strategies for failures
- **Resource Management** - Optimize memory usage during uploads

### Monitoring Metrics
- Registration success rate
- Upload performance metrics
- Registration completion time
- Error rates and retry attempts

## Security Considerations

### Data Protection
- Secure handling of model artifacts during upload
- Encrypted communication with MIMS service
- Proper cleanup of temporary S3 resources
- No sensitive information in logs

### Access Control
- Appropriate IAM roles for MIMS registration
- Secure sandbox session management
- Audit trail for registration operations
- Protection against unauthorized registration

## Error Handling

### Registration Failures
- Handle MIMS service unavailability
- Retry failed registration attempts
- Validate registration parameters
- Monitor registration status

### Upload Failures
- Handle S3 upload errors
- Implement retry logic for failed uploads
- Validate uploaded file integrity
- Clean up failed uploads

## Monitoring and Alerting

### Registration Monitoring
- Track registration success rates
- Monitor registration completion times
- Alert on registration failures
- Track model deployment status

### Performance Monitoring
- Monitor upload performance
- Track resource utilization
- Monitor service response times
- Alert on performance degradation
