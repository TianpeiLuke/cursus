# Cradle Data Loading Script Contract

## Overview
The Cradle Data Loading Script Contract defines the execution requirements for the Cradle data loading script (`scripts.py`). This contract ensures the script complies with SageMaker processing job conventions for securely downloading data from Amazon's internal Cradle data service.

## Contract Details

### Script Information
- **Entry Point**: `scripts.py`
- **Container Type**: SageMaker Processing Job
- **Framework**: Python with secure_ai_sandbox_python_lib
- **Purpose**: Source node for ML pipeline data ingestion

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| config | `/opt/ml/processing/config/config` | Data loading configuration (provided by step creation) |

*Note: This is a source node with no data inputs - configuration is provided via job configuration*

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| SIGNATURE | `/opt/ml/processing/output/signature` | Schema information for loaded data |
| METADATA | `/opt/ml/processing/output/metadata` | Field type information and metadata |
| DATA | `/opt/ml/processing/output/place_holder` | Placeholder (actual data goes to S3) |

### Environment Variables

#### Required Variables
- None (configuration provided via config file)

#### Optional Variables
| Variable | Default | Description |
|----------|---------|-------------|
| OUTPUT_PATH | "" | Optional override for data output path |

### Framework Requirements

#### Core Dependencies
```python
python>=3.7
secure_ai_sandbox_python_lib  # Core dependency for Cradle integration
```

## Script Functionality

*Note: This contract is defined but the actual script implementation was not found in the examined docker directories. The functionality described below is based on the contract specification.*

### Data Loading Pipeline
1. **Configuration Reading** - Reads data loading configuration from config file
2. **Schema Generation** - Writes output signature for data schema
3. **Metadata Creation** - Writes metadata file with field type information
4. **Cradle Integration** - Creates and executes Cradle data load job
5. **Job Monitoring** - Waits for job completion

### Cradle Service Integration
- **SandboxSession** - Creates secure session to interact with protected resources
- **Data Download Job** - Starts Cradle data download job
- **Job Completion** - Monitors and waits for download completion
- **S3 Integration** - Data is loaded directly to S3 by Cradle service

### Output Artifacts
- **signature/signature** - Schema information for the loaded data
- **metadata/metadata** - Metadata about fields including type information
- **S3 Data** - Actual data loaded directly to S3 by Cradle service

## Configuration Structure

### Data Loading Configuration
The script reads configuration from `/opt/ml/processing/config/config` which includes:
- Data source specifications
- Field definitions and types
- Output schema requirements
- Cradle service parameters

### Output Schema Format
```json
{
  "fields": [
    {
      "name": "field_name",
      "type": "field_type",
      "nullable": true/false
    }
  ]
}
```

### Metadata Format
```json
{
  "field_name": {
    "type": "categorical|numerical|text|date",
    "description": "Field description",
    "constraints": {}
  }
}
```

## Security Considerations

### Secure Access
- Uses `secure_ai_sandbox_python_lib` for secure resource access
- Creates SandboxSession for authenticated Cradle interactions
- Handles secure data transfer from internal sources

### Data Protection
- Data remains within Amazon's secure infrastructure
- No sensitive data exposed in logs or intermediate files
- Secure transfer to S3 storage

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import CRADLE_DATA_LOADING_CONTRACT

# Access contract details
print(f"Entry Point: {CRADLE_DATA_LOADING_CONTRACT.entry_point}")
print(f"Expected Outputs: {CRADLE_DATA_LOADING_CONTRACT.expected_output_paths}")
```

### Integration with Step Builder
```python
from src.pipeline_steps import CradleDataLoadingStepBuilder

class CradleDataLoadingStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        # Validate script compliance
        validation = CRADLE_DATA_LOADING_CONTRACT.validate_implementation(
            'scripts/scripts.py'
        )
        if not validation.is_valid:
            self.logger.warning(f"Script validation warnings: {validation.errors}")
```

## Integration Points

### Pipeline Position
- **Source Node** - First step in ML pipeline
- **No Dependencies** - Doesn't depend on other pipeline steps
- **Output Consumers** - Provides data for preprocessing and training steps

### Data Flow
```
Cradle Service → scripts.py → S3 Storage → Downstream Steps
                           ↓
                    Schema + Metadata Files
```

## Best Practices

### Script Development
1. **Error Handling** - Implement robust error handling for Cradle service failures
2. **Logging** - Log job progress without exposing sensitive data
3. **Timeout Handling** - Handle long-running download jobs gracefully
4. **Resource Cleanup** - Clean up temporary resources after job completion

### Configuration Management
1. **Schema Validation** - Validate output schema matches expected format
2. **Field Mapping** - Ensure field types are correctly mapped
3. **Metadata Completeness** - Provide comprehensive field metadata
4. **Version Control** - Track configuration changes for reproducibility

## Related Contracts

### Downstream Contracts
- `TABULAR_PREPROCESS_CONTRACT` - Processes loaded data
- `HYPERPARAMETER_PREP_CONTRACT` - Prepares training configuration

### Processing Contracts
- `MODEL_EVALUATION_CONTRACT` - Uses processed data for evaluation
- `PYTORCH_TRAIN_CONTRACT` - Uses data for model training

## Troubleshooting

### Common Issues
1. **Cradle Service Unavailable** - Check service status and retry logic
2. **Authentication Failures** - Verify SandboxSession configuration
3. **Schema Mismatches** - Validate field definitions in configuration
4. **S3 Access Issues** - Check permissions for data output location

### Validation Failures
1. **Missing Outputs** - Ensure all required output paths are created
2. **Configuration Errors** - Validate config file format and content
3. **Service Integration** - Check secure_ai_sandbox_python_lib installation
4. **Path Validation** - Ensure output paths follow SageMaker conventions

## Monitoring and Logging

### Job Monitoring
- Track Cradle job status and progress
- Monitor data transfer completion
- Log schema generation and validation

### Performance Metrics
- Data loading time and throughput
- Schema generation performance
- Job completion rates and failures

### Alerting
- Failed data loading jobs
- Schema validation errors
- Service availability issues
