# MIMS Payload Script Contract

## Overview
The MIMS Payload Script Contract defines the execution requirements for the MIMS payload generation script (`mims_payload.py`). This contract ensures the script complies with SageMaker processing job conventions for creating sample payloads and metadata for model inference testing.

## Contract Details

### Script Information
- **Entry Point**: `mims_payload.py`
- **Container Type**: SageMaker Processing Job
- **Framework**: Python with standard library only
- **Purpose**: Generate sample payloads for model inference testing

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| model_input | `/opt/ml/processing/input/model` | Model artifacts containing hyperparameters.json |

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| payload_sample | `/opt/ml/processing/output` | Output directory containing payload.tar.gz file |

### Environment Variables

#### Required Variables
- None (script has defaults for all parameters)

#### Optional Variables
- `CONTENT_TYPES` - Comma-separated list of content types (default: "application/json")
- `DEFAULT_NUMERIC_VALUE` - Default value for numeric fields (default: "0.0")
- `DEFAULT_TEXT_VALUE` - Default value for text fields (default: "DEFAULT_TEXT")
- `SPECIAL_FIELD_<fieldname>` - Custom values for specific fields

### Framework Requirements

#### Core Dependencies
```python
# Standard library only - no external dependencies
json
logging
os
tarfile
tempfile
pathlib
enum
typing
datetime
```

## Script Functionality

Based on the contract definition in `src/pipeline_script_contracts/mims_payload_contract.py`:

### Hyperparameter Extraction
1. **Model Artifact Processing**:
   - Extracts hyperparameters from model.tar.gz or directory structure
   - Locates and parses hyperparameters.json file
   - Handles both compressed and uncompressed model artifacts

2. **Field Information Parsing**:
   - Creates model variable list from field information
   - Identifies field types (numeric, text, categorical)
   - Maps field names to appropriate data types

### Payload Generation
1. **Sample Data Creation**:
   - Generates sample payloads based on model field requirements
   - Uses default values for different data types
   - Supports custom values for specific fields via environment variables

2. **Multiple Format Support**:
   - Creates payloads in JSON format by default
   - Supports CSV format generation
   - Handles multiple content types as specified

### Archive Creation
1. **Payload Packaging**:
   - Archives payload files into payload.tar.gz
   - Organizes files in structured directory format
   - Ensures compatibility with MIMS deployment requirements

2. **Temporary File Management**:
   - Uses temporary working directory for payload creation
   - Cleans up intermediate files after archiving
   - Handles file system operations safely

### Key Implementation Concepts

#### Hyperparameter Extraction
```python
def extract_hyperparameters(model_path: Path) -> dict:
    """Extract hyperparameters from model artifacts"""
    if model_path.is_file() and model_path.suffix == '.gz':
        # Extract from tar.gz file
        with tarfile.open(model_path, 'r:gz') as tar:
            hyperparams_file = tar.extractfile('hyperparameters.json')
            return json.load(hyperparams_file)
    else:
        # Read from directory
        hyperparams_path = model_path / 'hyperparameters.json'
        with open(hyperparams_path, 'r') as f:
            return json.load(f)
```

#### Payload Generation
```python
def generate_sample_payload(field_info: dict, defaults: dict) -> dict:
    """Generate sample payload based on field information"""
    payload = {}
    for field_name, field_type in field_info.items():
        if field_name in os.environ:
            # Use custom value from environment
            payload[field_name] = os.environ[f"SPECIAL_FIELD_{field_name}"]
        elif field_type == "numeric":
            payload[field_name] = float(defaults["numeric"])
        elif field_type == "text":
            payload[field_name] = defaults["text"]
        else:
            payload[field_name] = defaults["text"]  # Default fallback
    return payload
```

#### Archive Creation
```python
def create_payload_archive(payload_dir: Path, output_path: Path):
    """Create tar.gz archive of payload files"""
    with tarfile.open(output_path / "payload.tar.gz", "w:gz") as tar:
        for file_path in payload_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(payload_dir)
                tar.add(file_path, arcname=arcname)
```

### Payload Structure Examples

#### JSON Payload Format
```json
{
  "instances": [
    {
      "feature1": 0.0,
      "feature2": "DEFAULT_TEXT",
      "feature3": 1.5,
      "category_field": "category_a"
    }
  ]
}
```

#### CSV Payload Format
```csv
feature1,feature2,feature3,category_field
0.0,DEFAULT_TEXT,1.5,category_a
```

### Output Structure
```
/opt/ml/processing/output/
└── payload.tar.gz  # Archive containing:
    ├── application_json/
    │   └── sample_payload.json
    ├── text_csv/
    │   └── sample_payload.csv
    └── metadata.json
```

### Environment Variable Examples
```bash
export CONTENT_TYPES="application/json,text/csv"
export DEFAULT_NUMERIC_VALUE="1.0"
export DEFAULT_TEXT_VALUE="SAMPLE_TEXT"
export SPECIAL_FIELD_customer_id="CUST_12345"
export SPECIAL_FIELD_amount="100.50"
```

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import MIMS_PAYLOAD_CONTRACT

# Access contract details
print(f"Entry Point: {MIMS_PAYLOAD_CONTRACT.entry_point}")
print(f"Optional Env Vars: {MIMS_PAYLOAD_CONTRACT.optional_env_vars}")
```

### Integration with Step Builder
```python
from src.pipeline_steps import MIMSPayloadStepBuilder

class MIMSPayloadStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        validation = MIMS_PAYLOAD_CONTRACT.validate_implementation(
            'mims_payload.py'
        )
        if not validation.is_valid:
            self.logger.warning(f"Script validation warnings: {validation.errors}")
```

## Integration Points

### Pipeline Position
- **Payload Generation Node** - Prepares test data for model registration
- **Input Dependencies** - Requires model artifacts with hyperparameters
- **Output Consumers** - Provides payload samples for MIMS registration

### Data Flow
```
Model Artifacts → mims_payload.py → Payload Samples → MIMS Registration
```

## Best Practices

### Script Development
1. **Robust Parsing** - Handle various hyperparameter formats gracefully
2. **Default Values** - Provide sensible defaults for all field types
3. **Error Handling** - Handle missing or malformed hyperparameters
4. **File Management** - Clean up temporary files properly

### Payload Generation
1. **Realistic Data** - Generate realistic sample data for testing
2. **Data Types** - Ensure correct data types for all fields
3. **Format Consistency** - Maintain consistent format across content types
4. **Validation** - Validate generated payloads before archiving

## Related Contracts

### Upstream Contracts
- `MIMS_PACKAGE_CONTRACT` - Provides model artifacts with hyperparameters
- `XGBOOST_TRAIN_CONTRACT` - May provide hyperparameters in model artifacts
- `PYTORCH_TRAIN_CONTRACT` - May provide hyperparameters in model artifacts

### Downstream Contracts
- `MIMS_REGISTRATION_CONTRACT` - Uses payload samples for model testing

## Troubleshooting

### Common Issues
1. **Missing Hyperparameters** - Ensure hyperparameters.json exists in model artifacts
2. **Format Errors** - Validate JSON format of hyperparameters file
3. **Field Type Mapping** - Ensure correct mapping of field names to types
4. **Archive Creation** - Handle file system permissions for archive creation

### Validation Failures
1. **Path Validation** - Ensure model input path exists and is accessible
2. **File Format** - Validate model artifact format (tar.gz or directory)
3. **Hyperparameter Structure** - Check hyperparameters.json structure
4. **Output Directory** - Ensure output directory is writable

## Performance Considerations

### Optimization Strategies
- **Efficient Extraction** - Optimize tar file extraction for large models
- **Memory Management** - Handle large hyperparameter files efficiently
- **File I/O** - Minimize file system operations during generation
- **Archive Compression** - Balance compression ratio with processing time

### Monitoring Metrics
- Payload generation time
- Archive creation performance
- Memory usage during processing
- File operation success rates

## Security Considerations

### Data Protection
- Secure handling of model hyperparameters
- No sensitive information in generated payloads
- Proper cleanup of temporary files
- Secure archive creation

### File Security
- Validate input file integrity
- Secure temporary file handling
- Proper file permissions for outputs
- Protection against path traversal attacks

## Testing Strategies

### Unit Testing
- Test hyperparameter extraction
- Test payload generation logic
- Test archive creation
- Test error handling scenarios

### Integration Testing
- Test with actual model artifacts
- Test multiple content type generation
- Test custom field value handling
- Test pipeline integration

## Deployment Considerations

### Environment Setup
- Ensure sufficient disk space for temporary files
- Configure appropriate file system permissions
- Set up logging and monitoring
- Prepare error handling and recovery

### Monitoring and Alerting
- Monitor payload generation success rates
- Alert on generation failures
- Track processing performance
- Monitor disk space usage
