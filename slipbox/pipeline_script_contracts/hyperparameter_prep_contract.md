# Hyperparameter Preparation Script Contract

## Overview
The Hyperparameter Preparation Script Contract defines the execution requirements for the hyperparameter preparation process. Unlike other contracts, this represents a Lambda function that serializes hyperparameters from configuration to JSON format and uploads them to S3 for use by downstream pipeline steps.

## Contract Details

### Script Information
- **Entry Point**: `hyperparameter_prep.py` (Virtual - represents Lambda function)
- **Container Type**: AWS Lambda Function
- **Framework**: Python with boto3 for S3 operations
- **Purpose**: Hyperparameter serialization and S3 upload

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| N/A | N/A | No input paths - uses hyperparameters directly from config |

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| hyperparameters_s3_uri | `/opt/ml/processing/output/hyperparameters` | Virtual output path - actual output is S3 URI |

### Environment Variables

#### Required Variables
- None specified

#### Optional Variables
- None specified

### Framework Requirements

#### Core Dependencies
```python
# Lambda runtime provides:
boto3  # For S3 operations
json   # For hyperparameter serialization
```

## Script Functionality

Based on the contract definition in `src/pipeline_script_contracts/hyperparameter_prep_contract.py`:

### Hyperparameter Processing
1. **Configuration Access**:
   - Receives hyperparameters directly from pipeline configuration
   - No file system input required - operates on in-memory configuration
   - Supports complex nested hyperparameter structures

2. **JSON Serialization**:
   - Converts hyperparameter dictionary to JSON format
   - Handles various data types (strings, numbers, booleans, lists, dicts)
   - Ensures JSON compatibility for downstream consumption

### S3 Upload Operations
1. **S3 Storage**:
   - Uploads serialized hyperparameters to designated S3 bucket
   - Generates unique S3 key for hyperparameter file
   - Returns S3 URI for downstream pipeline steps

2. **URI Generation**:
   - Creates standardized S3 URI format
   - Ensures URI accessibility for SageMaker processing jobs
   - Provides consistent naming convention

### Key Implementation Concepts

#### Hyperparameter Serialization
```python
def serialize_hyperparameters(hyperparams: dict) -> str:
    """Serialize hyperparameters to JSON string"""
    return json.dumps(hyperparams, indent=2, sort_keys=True)
```

#### S3 Upload Process
```python
def upload_to_s3(content: str, bucket: str, key: str) -> str:
    """Upload content to S3 and return URI"""
    s3_client = boto3.client('s3')
    s3_client.put_object(Bucket=bucket, Key=key, Body=content)
    return f"s3://{bucket}/{key}"
```

#### Lambda Handler Pattern
```python
def lambda_handler(event, context):
    """Lambda function handler for hyperparameter preparation"""
    hyperparams = event.get('hyperparameters', {})
    json_content = serialize_hyperparameters(hyperparams)
    s3_uri = upload_to_s3(json_content, bucket, key)
    return {'hyperparameters_s3_uri': s3_uri}
```

### Output Format
- **S3 URI**: `s3://bucket-name/path/to/hyperparameters.json`
- **JSON Content**: Serialized hyperparameters in JSON format
- **Return Value**: Dictionary containing S3 URI for downstream consumption

### Hyperparameter Structure Examples

#### XGBoost Hyperparameters
```json
{
  "objective": "binary:logistic",
  "max_depth": 6,
  "learning_rate": 0.1,
  "n_estimators": 100,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 42
}
```

#### PyTorch Hyperparameters
```json
{
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 10,
  "hidden_size": 128,
  "dropout": 0.2,
  "optimizer": "adam",
  "weight_decay": 0.0001
}
```

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import HYPERPARAMETER_PREP_CONTRACT

# Access contract details
print(f"Entry Point: {HYPERPARAMETER_PREP_CONTRACT.entry_point}")
print(f"Description: {HYPERPARAMETER_PREP_CONTRACT.description}")
```

### Integration with Step Builder
```python
from src.pipeline_steps import HyperparameterPrepStepBuilder

class HyperparameterPrepStepBuilder(StepBuilderBase):
    def create_lambda_step(self, hyperparameters: dict) -> LambdaStep:
        """Create Lambda step for hyperparameter preparation"""
        return LambdaStep(
            name="hyperparameter-prep",
            lambda_func=self.hyperparameter_prep_function,
            inputs={"hyperparameters": hyperparameters}
        )
```

## Integration Points

### Pipeline Position
- **Configuration Node** - Early step in pipeline execution
- **Input Dependencies** - Requires hyperparameter configuration
- **Output Consumers** - Provides S3 URI for training and evaluation steps

### Data Flow
```
Pipeline Config → Lambda Function → S3 Hyperparameters → Training Steps
```

## Best Practices

### Lambda Development
1. **Error Handling** - Implement robust error handling for S3 operations
2. **Logging** - Use CloudWatch logging for debugging and monitoring
3. **Timeout Management** - Set appropriate timeout for S3 upload operations
4. **Memory Optimization** - Optimize memory usage for large hyperparameter sets

### Hyperparameter Management
1. **Validation** - Validate hyperparameter format and values
2. **Versioning** - Include version information in S3 key naming
3. **Documentation** - Document hyperparameter meanings and ranges
4. **Consistency** - Ensure consistent hyperparameter naming across frameworks

## Related Contracts

### Upstream Contracts
- Pipeline configuration system (not a script contract)

### Downstream Contracts
- `XGBOOST_TRAIN_CONTRACT` - Uses hyperparameters from S3 URI
- `PYTORCH_TRAIN_CONTRACT` - Uses hyperparameters from S3 URI
- `MODEL_EVALUATION_CONTRACT` - May use hyperparameters for evaluation context

## Troubleshooting

### Common Issues
1. **S3 Permissions** - Ensure Lambda has proper S3 write permissions
2. **JSON Serialization** - Handle non-serializable hyperparameter values
3. **S3 Bucket Access** - Verify S3 bucket exists and is accessible
4. **URI Format** - Ensure S3 URI format is compatible with SageMaker

### Validation Failures
1. **Hyperparameter Format** - Validate hyperparameter dictionary structure
2. **S3 Upload** - Handle S3 upload failures and retries
3. **Lambda Timeout** - Ensure sufficient timeout for S3 operations
4. **Memory Limits** - Monitor Lambda memory usage for large hyperparameter sets

## Performance Considerations

### Optimization Strategies
- **Efficient Serialization** - Optimize JSON serialization for large hyperparameter sets
- **S3 Upload Optimization** - Use appropriate S3 upload strategies
- **Lambda Cold Start** - Minimize Lambda cold start impact
- **Caching** - Consider caching for repeated hyperparameter uploads

### Monitoring Metrics
- Lambda execution time
- S3 upload success rate
- Memory usage during serialization
- Error rates and retry attempts

## Security Considerations

### Data Protection
- Secure handling of hyperparameter values
- No sensitive information in hyperparameters
- Proper S3 bucket security configuration

### Access Control
- Appropriate IAM roles for Lambda execution
- S3 bucket access controls
- Audit trail for hyperparameter uploads
- Protection against unauthorized access

## Lambda Function Configuration

### Runtime Settings
- **Runtime**: Python 3.9+
- **Memory**: 128-512 MB (depending on hyperparameter size)
- **Timeout**: 30-60 seconds
- **Environment**: Production Lambda environment

### IAM Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:PutObjectAcl"
      ],
      "Resource": "arn:aws:s3:::hyperparameter-bucket/*"
    }
  ]
}
