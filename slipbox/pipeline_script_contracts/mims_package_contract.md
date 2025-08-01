# MIMS Package Script Contract

## Overview
The MIMS Package Script Contract defines the execution requirements for the MIMS packaging script (`mims_package.py`). This contract ensures the script complies with SageMaker processing job conventions for packaging trained models and inference scripts into deployable MIMS (Model Inference Management System) packages.

## Contract Details

### Script Information
- **Entry Point**: `mims_package.py`
- **Container Type**: SageMaker Processing Job
- **Framework**: Python with standard library (shutil, tarfile, pathlib)
- **Purpose**: Model and inference script packaging for deployment

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| model_input | `/opt/ml/processing/input/model` | Trained model artifacts (may include model.tar.gz) |
| script_input | `/opt/ml/processing/input/script` | Inference scripts and supporting code |

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| package_output | `/opt/ml/processing/output` | Packaged model.tar.gz ready for deployment |

### Environment Variables

#### Required Variables
- None (script uses fixed paths)

#### Optional Variables
- None specified in current implementation

### Framework Requirements

#### Core Dependencies
```python
# Standard library only - no external dependencies
shutil
tarfile
pathlib
logging
os
typing
sys
```

## Script Functionality

Based on the actual implementation in `dockers/xgboost_atoz/pipeline_scripts/mims_package.py`:

### Model Artifact Processing
1. **Model Extraction**:
   - Checks for existing `model.tar.gz` in input model directory
   - Extracts compressed model artifacts if present
   - Falls back to copying individual files if no tar archive exists
   - Preserves file permissions and metadata during extraction

2. **File Validation**:
   - Validates existence and accessibility of all input files
   - Logs detailed file information (size, permissions, timestamps)
   - Provides comprehensive error handling for missing files

### Script Integration
1. **Inference Script Copying**:
   - Recursively copies all inference scripts from script input directory
   - Maintains directory structure and file organization
   - Places scripts in `code/` subdirectory within package
   - Handles nested directory structures properly

2. **Code Organization**:
   - Creates standardized directory structure for deployment
   - Ensures proper code placement for MIMS compatibility
   - Maintains relative paths and import structures

### Package Creation
1. **Tar Archive Generation**:
   - Creates compressed tar.gz archive of complete package
   - Includes both model artifacts and inference code
   - Optimizes compression for deployment efficiency
   - Validates package integrity after creation

2. **Directory Management**:
   - Uses temporary working directory for package assembly
   - Ensures clean separation of inputs and outputs
   - Provides detailed logging of all file operations

### Key Implementation Details

#### File Existence Validation
```python
def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists and log its details"""
    exists = path.exists() and path.is_file()
    if exists:
        stats = path.stat()
        size_mb = stats.st_size / 1024 / 1024
        logger.info(f"{description}: {path} ({size_mb:.2f}MB)")
    return exists
```

#### Robust File Copying
```python
def copy_file_robust(src: Path, dst: Path):
    """Copy a file and log the operation, ensuring destination directory exists"""
    ensure_directory(dst.parent)
    shutil.copy2(src, dst)  # Preserves metadata
    return check_file_exists(dst, "Destination file after copy")
```

#### Model Extraction Logic
```python
def extract_tarfile(tar_path: Path, extract_path: Path):
    """Extract a tar file to the specified path"""
    with tarfile.open(tar_path, "r:*") as tar:
        # Log contents before extraction
        for member in tar.getmembers():
            logger.info(f"  {member.name} ({member.size / 1024 / 1024:.2f}MB)")
        tar.extractall(path=extract_path)
```

#### Package Assembly
```python
def create_tarfile(output_tar_path: Path, source_dir: Path):
    """Create a tar file from the contents of a directory"""
    with tarfile.open(output_tar_path, "w:gz") as tar:
        for item in source_dir.rglob("*"):
            if item.is_file():
                arcname = item.relative_to(source_dir)
                tar.add(item, arcname=arcname)
```

### Output Structure
```
/opt/ml/processing/output/
└── model.tar.gz  # Complete MIMS package containing:
    ├── model_artifacts/     # Extracted model files
    │   ├── xgboost_model.bst
    │   ├── risk_table_map.pkl
    │   ├── impute_dict.pkl
    │   └── hyperparameters.json
    └── code/               # Inference scripts
        ├── inference.py
        ├── processing/
        └── requirements.txt
```

### Logging and Monitoring
1. **Comprehensive Logging**:
   - Detailed file operation logs with timestamps
   - File size and permission tracking
   - Directory content listings before and after operations
   - Compression ratio and package statistics

2. **Error Handling**:
   - Graceful handling of missing files and directories
   - Detailed error messages with stack traces
   - Validation of all file operations
   - Recovery strategies for common failures

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import MIMS_PACKAGE_CONTRACT

# Access contract details
print(f"Entry Point: {MIMS_PACKAGE_CONTRACT.entry_point}")
print(f"Input Paths: {MIMS_PACKAGE_CONTRACT.expected_input_paths}")
```

### Integration with Step Builder
```python
from src.pipeline_steps import MIMSPackageStepBuilder

class MIMSPackageStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        validation = MIMS_PACKAGE_CONTRACT.validate_implementation(
            'dockers/xgboost_atoz/pipeline_scripts/mims_package.py'
        )
        if not validation.is_valid:
            self.logger.warning(f"Script validation warnings: {validation.errors}")
```

## Integration Points

### Pipeline Position
- **Packaging Node** - Final step before model deployment
- **Input Dependencies** - Requires trained model artifacts and inference scripts
- **Output Consumers** - Provides deployable package for MIMS deployment

### Data Flow
```
Trained Model + Inference Scripts → mims_package.py → Deployable MIMS Package
```

## Best Practices

### Script Development
1. **Robust File Handling** - Handle missing files and directories gracefully
2. **Comprehensive Logging** - Log all file operations for debugging
3. **Validation** - Validate package integrity before completion
4. **Error Recovery** - Implement recovery strategies for common failures

### Package Management
1. **Directory Structure** - Maintain consistent package structure
2. **File Permissions** - Preserve necessary file permissions
3. **Compression** - Optimize package size for deployment
4. **Validation** - Verify package completeness and integrity

## Related Contracts

### Upstream Contracts
- `XGBOOST_TRAIN_CONTRACT` - Provides trained model artifacts
- `PYTORCH_TRAIN_CONTRACT` - Provides trained model artifacts
- `MODEL_EVALUATION_CONTRACT` - May provide evaluation results

### Downstream Contracts
- `MIMS_REGISTRATION_CONTRACT` - Uses packaged model for registration
- `MIMS_PAYLOAD_CONTRACT` - Uses package for payload testing

## Troubleshooting

### Common Issues
1. **Missing Model Files** - Ensure all required model artifacts are present
2. **Script Dependencies** - Verify all inference scripts and dependencies are included
3. **Permission Issues** - Check file permissions for packaging operations
4. **Disk Space** - Ensure sufficient disk space for package creation

### Validation Failures
1. **Path Validation** - Ensure all input paths exist and are accessible
2. **File Integrity** - Verify file integrity during copying and extraction
3. **Package Structure** - Validate final package structure meets MIMS requirements
4. **Compression Issues** - Handle tar archive creation and compression errors

## Performance Considerations

### Optimization Strategies
- **Efficient File Operations** - Use optimized file copying and compression
- **Memory Management** - Handle large model files efficiently
- **Parallel Processing** - Consider parallel file operations for large packages
- **Compression Optimization** - Balance compression ratio with processing time

### Monitoring Metrics
- Package creation time
- File operation success rates
- Package size and compression ratios
- Disk space utilization during packaging

## Security Considerations

### Package Security
- Validate all input files before packaging
- Ensure no sensitive information in logs
- Secure handling of model artifacts
- Proper cleanup of temporary files

### Deployment Security
- Validate package integrity before deployment
- Ensure proper file permissions in package
- Protect against package tampering
- Audit trail for packaging operations
