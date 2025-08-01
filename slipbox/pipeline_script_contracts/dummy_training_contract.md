# Dummy Training Script Contract

## Overview
The Dummy Training Script Contract defines the execution requirements for the dummy training script (`dummy_training.py`). This contract ensures the script complies with SageMaker processing job conventions for copying pretrained models to make them available for downstream packaging and registration steps.

## Contract Details

### Script Information
- **Entry Point**: `dummy_training.py`
- **Container Type**: SageMaker Processing Job
- **Framework**: Python with boto3 and pathlib
- **Purpose**: Copy pretrained model artifacts for pipeline compatibility

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| pretrained_model_path | `/opt/ml/processing/input/model/model.tar.gz` | Pretrained model archive |

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| model_input | `/opt/ml/processing/output/model` | Copied model artifacts for downstream steps |

### Environment Variables

#### Required Variables
- None specified

#### Optional Variables
- None specified

### Framework Requirements

#### Core Dependencies
```python
boto3>=1.26.0
pathlib>=1.0.0
```

## Script Functionality

Based on the contract definition in `src/pipeline_script_contracts/dummy_training_contract.py`:

### Model Copying Operations
1. **Pretrained Model Access**:
   - Locates pretrained model.tar.gz in input directory
   - Validates model archive existence and accessibility
   - Handles missing or corrupted model files gracefully

2. **Model Transfer**:
   - Copies pretrained model to output directory
   - Maintains model archive format and structure
   - Preserves model metadata and permissions

### Pipeline Integration
1. **Downstream Compatibility**:
   - Ensures output format matches expectations of packaging steps
   - Provides consistent model artifact structure
   - Maintains compatibility with MIMS packaging requirements

2. **Workflow Continuity**:
   - Enables pipeline execution without actual training
   - Supports testing and validation workflows
   - Facilitates model deployment pipeline testing

### Key Implementation Concepts

#### Model File Validation
```python
def validate_model_file(model_path: Path) -> bool:
    """Validate that model file exists and is accessible"""
    return model_path.exists() and model_path.is_file()
```

#### Model Copying Process
```python
def copy_model_artifacts(input_path: Path, output_path: Path):
    """Copy model artifacts from input to output location"""
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path / "model.tar.gz")
```

#### Error Handling
```python
def handle_missing_model(input_path: Path):
    """Handle cases where pretrained model is not found"""
    logger.error(f"Pretrained model not found at {input_path}")
    raise FileNotFoundError(f"Required model file missing: {input_path}")
```

### Use Cases

#### Pipeline Testing
- **Development Testing**: Test pipeline without training overhead
- **Integration Testing**: Validate downstream steps with known model
- **Performance Testing**: Measure pipeline performance without training time

#### Model Deployment
- **Pretrained Models**: Deploy existing trained models through pipeline
- **Model Updates**: Update deployed models with new pretrained versions
- **A/B Testing**: Deploy alternative pretrained models for comparison

### Output Structure
```
/opt/ml/processing/output/model/
└── model.tar.gz  # Copied pretrained model archive
```

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import DUMMY_TRAINING_CONTRACT

# Access contract details
print(f"Entry Point: {DUMMY_TRAINING_CONTRACT.entry_point}")
print(f"Input Paths: {DUMMY_TRAINING_CONTRACT.expected_input_paths}")
```

### Integration with Step Builder
```python
from src.pipeline_steps import DummyTrainingStepBuilder

class DummyTrainingStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        validation = DUMMY_TRAINING_CONTRACT.validate_implementation(
            'dummy_training.py'
        )
        if not validation.is_valid:
            self.logger.warning(f"Script validation warnings: {validation.errors}")
```

## Integration Points

### Pipeline Position
- **Training Substitute Node** - Replaces actual training step
- **Input Dependencies** - Requires pretrained model archive
- **Output Consumers** - Provides model for packaging and deployment steps

### Data Flow
```
Pretrained Model → dummy_training.py → Model Artifacts → Packaging
```

## Best Practices

### Script Development
1. **Robust Validation** - Validate model file existence and format
2. **Error Handling** - Handle missing or corrupted model files gracefully
3. **Logging** - Log all file operations for debugging
4. **Performance** - Optimize file copying for large models

### Model Management
1. **Version Control** - Track pretrained model versions
2. **Validation** - Validate model compatibility before copying
3. **Metadata** - Preserve model metadata during copying
4. **Security** - Ensure secure handling of model files

## Related Contracts

### Upstream Contracts
- Model storage system (provides pretrained models)

### Downstream Contracts
- `MIMS_PACKAGE_CONTRACT` - Uses copied model for packaging
- `MIMS_REGISTRATION_CONTRACT` - Uses model for registration
- `MODEL_EVALUATION_CONTRACT` - May evaluate copied model

## Troubleshooting

### Common Issues
1. **Missing Model File** - Ensure pretrained model exists at expected path
2. **Permission Issues** - Check file permissions for copying operations
3. **Disk Space** - Ensure sufficient disk space for model copying
4. **File Corruption** - Validate model file integrity before copying

### Validation Failures
1. **Path Validation** - Ensure input and output paths are accessible
2. **File Format** - Validate model.tar.gz format and structure
3. **Copy Operation** - Handle file copying failures and retries
4. **Output Verification** - Verify copied model integrity

## Performance Considerations

### Optimization Strategies
- **Efficient Copying** - Use optimized file copying methods
- **Memory Management** - Handle large model files efficiently
- **Parallel Operations** - Consider parallel copying for multiple models
- **Caching** - Cache frequently used pretrained models

### Monitoring Metrics
- Model copying time
- File operation success rates
- Disk space utilization
- Error rates and retry attempts

## Security Considerations

### Model Security
- Validate model file integrity
- Secure handling of model artifacts
- Proper cleanup of temporary files
- Protection against model tampering

### Access Control
- Appropriate permissions for model access
- Secure model storage and transfer
- Audit trail for model operations
- Protection against unauthorized access

## Testing Strategies

### Unit Testing
- Test model file validation
- Test copying operations
- Test error handling scenarios
- Test output verification

### Integration Testing
- Test with actual pretrained models
- Test pipeline integration
- Test downstream compatibility
- Test error recovery scenarios

## Deployment Considerations

### Environment Setup
- Ensure required dependencies are available
- Configure appropriate file system permissions
- Set up logging and monitoring
- Prepare error handling and recovery

### Monitoring and Alerting
- Monitor file operation success rates
- Alert on model copying failures
- Track performance metrics
- Monitor disk space usage
