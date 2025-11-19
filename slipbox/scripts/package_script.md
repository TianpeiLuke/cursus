---
tags:
  - code
  - processing_script
  - model_packaging
  - deployment
keywords:
  - model packaging
  - MIMS deployment
  - tar.gz creation
  - inference scripts
  - calibration model
  - model deployment
  - artifact packaging
topics:
  - model deployment
  - packaging workflows
  - MIMS integration
language: python
date of note: 2025-11-18
---

# Package Script Documentation

## Overview

The `package.py` script implements the MIMS (Model Inference Management System) packaging workflow that combines model artifacts, inference scripts, and optional calibration models into a single deployable tar.gz file for production deployment.

The script serves as the final preparation step before model registration and deployment, ensuring all necessary components are properly structured and packaged according to MIMS requirements. It handles both pre-packaged models (model.tar.gz) and loose model artifacts, automatically detecting and processing the appropriate format.

The packaging process includes comprehensive logging and verification at each step, making it easy to debug issues and ensure successful packaging. Optional calibration model integration enables deployment of calibrated models without modifying the core packaging workflow.

## Purpose and Major Tasks

### Primary Purpose
Package model artifacts, inference scripts, and optional calibration models into a single MIMS-compliant tar.gz file ready for production deployment to SageMaker endpoints.

### Major Tasks

1. **Working Directory Setup**: Create and prepare temporary working directory for packaging operations with proper permissions and structure

2. **Model Artifact Extraction**: Extract model artifacts from input model.tar.gz or copy loose files from model directory with validation

3. **Calibration Model Integration**: Process and integrate optional calibration model artifacts into the package structure matching inference script expectations

4. **Inference Script Copying**: Recursively copy inference scripts and dependencies to the code directory within the package structure

5. **Directory Structure Validation**: Verify all components are correctly placed and accessible in the working directory with detailed logging

6. **Tar Archive Creation**: Create compressed tar.gz archive from the working directory contents with compression statistics

7. **Output Verification**: Verify the created package is valid, accessible, and contains all expected components

8. **Logging and Diagnostics**: Provide comprehensive logging throughout the process including file sizes, permissions, and operation results

## Script Contract

### Entry Point
```
package.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `model_input` | `/opt/ml/processing/input/model` | Model artifacts directory (may contain model.tar.gz or loose files) |
| `inference_scripts_input` | `/opt/ml/processing/input/script` | Inference scripts and code for model deployment |
| `calibration_model` | `/opt/ml/processing/input/calibration` | Optional calibration model artifacts (calibration_model.pkl, calibration_summary.json) |

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `packaged_model` | `/opt/ml/processing/output` | Output directory containing the packaged model.tar.gz file |

### Required Environment Variables

None - This script uses only the standardized SageMaker Processing paths from the contract.

### Optional Environment Variables

#### Working Directory Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKING_DIRECTORY` | `"/tmp/mims_packaging_directory"` | Temporary directory for packaging operations and intermediate files |

### Job Arguments

No command-line arguments are required. The script uses the standardized contract-based path structure.

## Input Data Structure

### Model Input Structure

The script accepts two input formats:

**Format 1: Pre-packaged Model**
```
/opt/ml/processing/input/model/
└── model.tar.gz
```

**Format 2: Loose Model Files**
```
/opt/ml/processing/input/model/
├── model_file1.pkl
├── model_file2.json
├── hyperparameters.json
└── additional_artifacts/
    └── ...
```

### Inference Scripts Input Structure

```
/opt/ml/processing/input/script/
├── inference.py
├── requirements.txt
├── utils/
│   ├── __init__.py
│   └── helpers.py
└── config/
    └── inference_config.json
```

All files and subdirectories are recursively copied to maintain structure.

### Calibration Model Input Structure (Optional)

```
/opt/ml/processing/input/calibration/
├── calibration_model.pkl          # Binary calibration (sklearn isotonic/sigmoid)
└── calibration_summary.json       # Calibration metadata and statistics
```

OR for multi-class models:

```
/opt/ml/processing/input/calibration/
├── calibration_models/
│   ├── class_0_calibration.pkl
│   ├── class_1_calibration.pkl
│   └── ...
└── calibration_summary.json
```

## Output Data Structure

### Output Directory Structure

```
/opt/ml/processing/output/
└── model.tar.gz
```

### Package Contents (model.tar.gz)

When extracted, the tar.gz file contains:

```
working_directory/
├── model_file1.pkl              # Model artifacts
├── model_file2.json
├── hyperparameters.json
├── code/                        # Inference scripts directory
│   ├── inference.py
│   ├── requirements.txt
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── config/
│       └── inference_config.json
└── calibration/                 # Optional calibration directory
    ├── calibration_model.pkl
    └── calibration_summary.json
```

This structure matches SageMaker's expected format for model deployment:
- Model artifacts at root level
- Inference code in `code/` subdirectory
- Calibration models in `calibration/` subdirectory

## Key Functions and Tasks

### Directory Management Component

#### `ensure_directory(directory: Path) -> bool`
**Purpose**: Create directory with parent directories if needed, ensuring proper permissions

**Algorithm**:
```python
1. Create directory using mkdir(parents=True, exist_ok=True)
2. Log directory creation with full path
3. Log directory permissions in octal format
4. Return True on success, False on exception
5. Log detailed error information on failure
```

**Parameters**:
- `directory` (Path): Directory path to create

**Returns**: `bool` - True if directory exists/created successfully, False otherwise

**Error Handling**: Catches all exceptions, logs with traceback, returns False

#### `list_directory_contents(path: Path, description: str) -> None`
**Purpose**: List and log complete directory contents with detailed file statistics

**Algorithm**:
```python
1. Validate path exists and is a directory
2. Recursively iterate through all items using rglob("*")
3. For each item:
   a. Calculate indentation based on depth
   b. If file: log with size in MB, increment counters
   c. If directory: log with folder icon, increment counter
4. Calculate and log summary statistics:
   - Total files count
   - Total directories count
   - Total size in MB
```

**Complexity**:
- Time: O(n) where n is total number of files/directories
- Space: O(1) for iteration

### File Operations Component

#### `check_file_exists(path: Path, description: str) -> bool`
**Purpose**: Verify file existence and log comprehensive file metadata

**Algorithm**:
```python
1. Check if path exists and is a file
2. If exists:
   a. Get file statistics (size, permissions, mtime)
   b. Calculate size in MB
   c. Log all metadata with formatting
3. If not exists:
   a. Log warning message
4. Return existence status
```

**Parameters**:
- `path` (Path): File path to check
- `description` (str): Description for logging

**Returns**: `bool` - True if file exists, False otherwise

#### `copy_file_robust(src: Path, dst: Path) -> bool`
**Purpose**: Copy file with validation, logging, and directory creation

**Algorithm**:
```python
1. Log copy operation details (source and destination)
2. Verify source file exists using check_file_exists()
3. If source doesn't exist: log warning, return False
4. Ensure destination parent directory exists
5. Copy file using shutil.copy2() (preserves metadata)
6. Verify copied file using check_file_exists()
7. Return True if verified, False otherwise
```

**Error Handling**: Comprehensive exception catching with detailed logging

#### `copy_scripts(src_dir: Path, dst_dir: Path) -> None`
**Purpose**: Recursively copy all scripts from source to destination with statistics

**Algorithm**:
```python
1. Log operation header with source and destination
2. List source directory contents
3. Validate source directory exists
4. Ensure destination directory exists
5. Initialize counters (files_copied=0, total_size_mb=0)
6. For each item in src_dir.rglob("*"):
   a. If item is file:
      - Calculate relative path
      - Construct destination path
      - Copy file using copy_file_robust()
      - If successful: increment counters
7. Log summary statistics
8. List destination directory contents for verification
```

**Complexity**:
- Time: O(n) where n is total number of files
- Space: O(d) where d is maximum directory depth

### Archive Operations Component

#### `extract_tarfile(tar_path: Path, extract_path: Path) -> None`
**Purpose**: Extract tar.gz archive with validation and logging

**Algorithm**:
```python
1. Verify tar file exists using check_file_exists()
2. If not exists: log error, return early
3. Ensure extraction directory exists
4. Open tar file in read mode (supports all compression formats)
5. Log tar contents before extraction:
   a. For each member: log name and size in MB
   b. Calculate and log total size
6. Extract all contents to extraction path
7. List extracted contents for verification
```

**Parameters**:
- `tar_path` (Path): Path to tar.gz file
- `extract_path` (Path): Directory to extract contents

**Error Handling**: Catches exceptions during tar operations, logs with traceback

#### `create_tarfile(output_tar_path: Path, source_dir: Path) -> None`
**Purpose**: Create compressed tar.gz archive from directory contents

**Algorithm**:
```python
1. Log operation header with output path and source
2. Ensure output parent directory exists
3. Initialize counters (total_size=0, files_added=0)
4. Open tar file in write-gzip mode
5. For each item in source_dir.rglob("*"):
   a. If item is file:
      - Calculate relative path (arcname)
      - Get file size in MB
      - Log addition to tar
      - Add file to tar with relative path
      - Increment counters
6. Close tar file
7. Log summary statistics:
   - Files added count
   - Total uncompressed size
   - Compressed tar size
   - Compression ratio
```

**Complexity**:
- Time: O(n) where n is total number of files
- Space: O(1) for streaming compression

### Main Orchestration Component

#### `main(input_paths, output_paths, environ_vars, job_args) -> Path`
**Purpose**: Orchestrate the complete packaging workflow with validation and error handling

**Algorithm**:
```python
1. Validate required input paths exist in dictionaries
2. Extract paths from parameters
3. Initialize working directory path from environment or default
4. Log system information (Python version, disk space)
5. Log all configured paths
6. Create working and output directories
7. Process model input:
   IF model.tar.gz exists:
      - Extract to working directory
   ELSE:
      - Copy all files from model_path to working directory
8. Process optional calibration model:
   IF calibration_path exists:
      - Create calibration subdirectory
      - Copy all calibration artifacts to subdirectory
9. Copy inference scripts to code subdirectory
10. Create output tar.gz from working directory
11. Verify final output and log summary
12. Return path to packaged model
```

**Parameters**:
- `input_paths` (Dict[str, str]): Input path mappings (model_input, inference_scripts_input, calibration_model)
- `output_paths` (Dict[str, str]): Output path mapping (packaged_model)
- `environ_vars` (Dict[str, str]): Environment variables (WORKING_DIRECTORY)
- `job_args` (Optional[argparse.Namespace]): Command-line arguments (not used)

**Returns**: `Path` - Path to the created model.tar.gz file

**Error Handling**: Top-level exception handler with traceback logging and re-raise

## Algorithms and Data Structures

### Tar File Processing Algorithm

**Problem**: Need to package diverse file structures (pre-packaged tar, loose files, optional components) into a single deployment-ready archive

**Solution Strategy**:
1. Use temporary working directory as staging area
2. Normalize all input formats to common directory structure
3. Assemble components in correct hierarchy
4. Create final compressed archive

**Algorithm**:
```python
# Stage 1: Extract or copy model artifacts
if model.tar.gz exists:
    extract_to_working_directory()
else:
    copy_all_files_to_working_directory()

# Stage 2: Add optional calibration (if provided)
if calibration_path exists:
    create_calibration_subdirectory()
    copy_calibration_artifacts()

# Stage 3: Add inference scripts
create_code_subdirectory()
copy_scripts_recursively()

# Stage 4: Create final package
create_compressed_tar_gz()
```

**Complexity**:
- Time: O(n + m + k) where n=model files, m=calibration files, k=script files
- Space: O(n + m + k) for working directory contents

**Key Features**:
- Format-agnostic: Handles both pre-packaged and loose files
- Incremental assembly: Builds package step-by-step
- Verification at each stage: Ensures correctness before proceeding

### Recursive Directory Copying Algorithm

**Problem**: Copy entire directory trees while preserving structure and maintaining comprehensive logs

**Solution Strategy**:
1. Use Path.rglob("*") for recursive traversal
2. Calculate relative paths to preserve structure
3. Track statistics during copying

**Algorithm**:
```python
def copy_scripts(src_dir, dst_dir):
    files_copied = 0
    total_size = 0
    
    for item in src_dir.rglob("*"):
        if item.is_file():
            relative_path = item.relative_to(src_dir)
            destination = dst_dir / relative_path
            
            if copy_file_robust(item, destination):
                files_copied += 1
                total_size += destination.stat().st_size
    
    return files_copied, total_size
```

**Complexity**:
- Time: O(n) where n is total number of files
- Space: O(d) where d is maximum directory depth (for relative path calculation)

**Key Features**:
- Structure preservation: Maintains directory hierarchy
- Metadata preservation: Uses shutil.copy2() to preserve timestamps and permissions
- Progress tracking: Counts files and sizes during operation

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|-----------------|------------------|-------|
| Directory creation | O(d) | O(1) | d = directory depth |
| File existence check | O(1) | O(1) | Single stat() call |
| Single file copy | O(s) | O(b) | s = file size, b = buffer size |
| Recursive directory copy | O(n) | O(d) | n = files, d = max depth |
| Tar extraction | O(n * s) | O(s) | n = files, s = avg file size |
| Tar creation | O(n * s) | O(b) | b = compression buffer |
| Complete packaging | O(n * s) | O(n * s) | Dominated by working directory |

### Processing Time Estimates

Based on typical model sizes:

| Model Size Category | File Count | Typical Time | Peak Memory |
|---------------------|------------|--------------|-------------|
| Small (< 100MB) | < 50 | 2-5s | ~200MB |
| Medium (100MB-1GB) | 50-500 | 10-30s | ~1GB |
| Large (1-10GB) | 500-5000 | 1-5min | ~10GB |
| Very Large (> 10GB) | > 5000 | 5-15min | ~20GB |

**Factors Affecting Performance**:
- File system I/O speed
- Compression level (gzip default)
- Number of small files vs. few large files
- Available disk space and memory

### Compression Ratio

Typical compression ratios for different content types:

| Content Type | Compression Ratio | Notes |
|--------------|-------------------|-------|
| Text files (.py, .json) | 60-80% | Highly compressible |
| Binary models (.pkl, .pt) | 5-20% | Already somewhat compressed |
| Mixed content | 30-50% | Average across types |

## Error Handling

### Error Types

#### Input Validation Errors

**Missing Required Input Path**
- **Cause**: `model_input` or `inference_scripts_input` not provided in input_paths
- **Handling**: Raises ValueError with specific missing key message
- **Prevention**: Contract validation ensures paths are provided

**Source File Not Found**
- **Cause**: Expected input files or directories don't exist
- **Handling**: Logs warning, continues with available inputs
- **Impact**: May result in incomplete package if critical files missing

#### Processing Errors

**Directory Creation Failure**
- **Cause**: Permission issues, disk full, invalid path
- **Handling**: Logs error with traceback, returns False from ensure_directory()
- **Impact**: Prevents further processing in affected path

**File Copy Failure**
- **Cause**: Permission issues, disk full, source file locked
- **Handling**: Logs error with traceback, skips file, continues with others
- **Impact**: Reduces files in final package, may cause deployment issues

**Tar Extraction Failure**
- **Cause**: Corrupted tar file, unsupported compression, permission issues
- **Handling**: Logs error with traceback, raises exception
- **Impact**: Halts processing, requires manual intervention

**Tar Creation Failure**
- **Cause**: Disk full, permission issues, invalid source directory
- **Handling**: Logs error with traceback, raises exception
- **Impact**: No output file created, entire operation fails

#### Output Errors

**Output Directory Not Writable**
- **Cause**: Permission issues on /opt/ml/processing/output
- **Handling**: ensure_directory() logs error and returns False
- **Impact**: Cannot create final package

**Insufficient Disk Space**
- **Cause**: Working directory or output path on full filesystem
- **Handling**: Exception caught during tar creation, logged with traceback
- **Impact**: Partial files created, operation fails

### Error Response Strategy

The script uses a multi-layered error handling approach:

1. **Validation Layer**: Check inputs before processing
2. **Operation Layer**: Try-catch around each major operation
3. **Logging Layer**: Comprehensive logging for debugging
4. **Recovery Layer**: Continue with partial success where possible

**Example Error Flow**:
```python
try:
    if not check_file_exists(input_tar, "Input tar"):
        logger.warning("Tar not found, trying loose files")
        # Attempt alternative approach
        copy_loose_files()
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    raise  # Re-raise for step failure
```

## Best Practices

### For Production Deployments

1. **Validate Inputs Before Execution**
   - Verify all required files exist in input directories
   - Check available disk space (recommend 3x final package size)
   - Ensure proper permissions on input and output paths

2. **Monitor Working Directory Size**
   - Working directory can grow to 2-3x final package size
   - Use `/tmp` with sufficient space or configure custom WORKING_DIRECTORY
   - Clean up working directory after successful packaging

3. **Verify Package Contents**
   - Always check the final tar.gz can be extracted
   - Verify inference.py and other critical files are present
   - Test package deployment in staging before production

4. **Handle Calibration Models Appropriately**
   - If using calibration, ensure calibration_summary.json is present
   - Verify calibration model format matches inference script expectations
   - Test inference with calibrated predictions before deployment

### For Development and Testing

1. **Use Comprehensive Logging**
   - Keep logging at INFO level during development
   - Review logs for file counts and sizes
   - Check compression ratios for optimization opportunities

2. **Test with Various Input Formats**
   - Test with pre-packaged model.tar.gz
   - Test with loose model files
   - Test with and without calibration models
   - Test with complex inference script structures

3. **Verify Directory Structure**
   - Extract packaged tar.gz locally and inspect structure
   - Ensure code/ and calibration/ subdirectories are correct
   - Check that relative imports in inference scripts still work

4. **Handle Edge Cases**
   - Empty directories in inference scripts
   - Very large model files (>5GB)
   - Special characters in filenames
   - Symbolic links (note: currently not followed)

### For Performance Optimization

1. **Optimize Working Directory Location**
   - Use local SSD for working directory when available
   - Avoid network file systems for temporary directories
   - Configure WORKING_DIRECTORY on fast storage

2. **Pre-package When Possible**
   - If model artifacts are stable, pre-package as model.tar.gz
   - Reduces processing time by skipping individual file copies
   - Enables faster extraction with tar streaming

3. **Minimize Small Files**
   - Combine multiple small configuration files when possible
   - Use single requirements.txt instead of multiple
   - Archive rarely-used utilities separately

## Example Configurations

### Basic Model Packaging

```bash
# Standard SageMaker Processing paths (defaults)
# No environment variables needed - uses contract defaults

# Input structure:
# /opt/ml/processing/input/model/
#   └── model.tar.gz
# /opt/ml/processing/input/script/
#   ├── inference.py
#   └── requirements.txt

# Output:
# /opt/ml/processing/output/model.tar.gz
```

**Use Case**: Simple model with basic inference script, no calibration

### Model with Calibration

```bash
# Standard paths with optional calibration input

# Input structure:
# /opt/ml/processing/input/model/
#   └── trained_model.pkl
# /opt/ml/processing/input/script/
#   ├── inference.py
#   └── utils/
# /opt/ml/processing/input/calibration/
#   ├── calibration_model.pkl
#   └── calibration_summary.json

# Output:
# /opt/ml/processing/output/model.tar.gz
#   (contains model + code + calibration subdirectory)
```

**Use Case**: Calibrated model requiring probability adjustment during inference

### Custom Working Directory

```bash
export WORKING_DIRECTORY="/mnt/fast-storage/packaging_temp"

# Use custom working directory for better I/O performance
# Useful for very large models or when /tmp is small
```

**Use Case**: Large models requiring more space or faster I/O than default /tmp

### Complex Inference Scripts

```bash
# Input structure with nested dependencies:
# /opt/ml/processing/input/script/
#   ├── inference.py
#   ├── requirements.txt
#   ├── config/
#   │   ├── model_config.json
#   │   └── preprocessing_config.json
#   ├── utils/
#   │   ├── __init__.py
#   │   ├── preprocessing.py
#   │   └── postprocessing.py
#   └── models/
#       └── feature_extractor.py

# All structure preserved in final package
```

**Use Case**: Complex multi-component inference pipeline with dependencies

## Integration Patterns

### Upstream Integration

```
TrainingStep (XGBoost/LightGBM/PyTorch)
   ↓ (outputs: model.tar.gz)
PackageStep
   ↓ (outputs: packaged_model.tar.gz)
```

OR with calibration:

```
TrainingStep
   ↓ (model.tar.gz)
ModelCalibrationStep
   ↓ (calibration artifacts)
PackageStep (combines both inputs)
   ↓ (packaged_model.tar.gz with calibration)
```

### Downstream Integration

```
PackageStep
   ↓ (packaged_model.tar.gz)
ModelRegistrationStep (MIMS)
   ↓ (registered model)
DeploymentStep
   ↓ (live endpoint)
```

### Complete Workflow Example

```
1. XGBoostTraining
   ↓ model.tar.gz
2. ModelCalibration (optional)
   ↓ calibration_model.pkl, calibration_summary.json
3. Package (THIS STEP)
   Inputs:
   - model.tar.gz from step 1
   - inference scripts from local config
   - calibration artifacts from step 2 (optional)
   Output:
   ↓ packaged_model.tar.gz (complete deployment package)
4. MimsModelRegistration
   ↓ model registered in MIMS
5. SageMaker Endpoint Deployment
   ↓ live inference endpoint
```

**Key Integration Points**:
- **Training outputs**: Accepts model.tar.gz or loose files
- **Calibration outputs**: Optional calibration directory
- **Local inference scripts**: Always from config.source_dir
- **Registration input**: Single packaged tar.gz for MIMS

## Troubleshooting

### Package Creation Issues

**Symptom**: "Failed to create directory" errors

**Common Causes**:
1. **Insufficient permissions**: Output directory not writable
2. **Disk space full**: No space for working directory
3. **Invalid path**: Working directory path contains invalid characters

**Solution**:
```bash
# Check permissions
ls -ld /opt/ml/processing/output

# Check disk space
df -h /tmp

# Verify working directory path
echo $WORKING_DIRECTORY

# Use alternative working directory if needed
export WORKING_DIRECTORY="/mnt/larger-volume/packaging"
```

### Missing Files in Package

**Symptom**: Extracted package missing expected files

**Common Causes**:
1. **Copy failures**: Files failed to copy but script continued
2. **Incorrect input paths**: Wrong source directory provided
3. **Symbolic links**: Script doesn't follow symlinks

**Solution**:
```bash
# Review logs for copy failures
grep "Error copying" processing_logs.txt

# Verify input directory contents
ls -laR /opt/ml/processing/input/script

# Check for symbolic links (not followed)
find /opt/ml/processing/input/script -type l

# Verify final package contents
tar -tzf /opt/ml/processing/output/model.tar.gz | head -20
```

### Tar Extraction Failures

**Symptom**: "Error during tar extraction" in logs

**Common Causes**:
1. **Corrupted input tar**: model.tar.gz is corrupted
2. **Unsupported format**: Non-gzip compression
3. **Disk space**: Not enough space to extract

**Solution**:
```bash
# Verify tar file integrity
tar -tzf /opt/ml/processing/input/model/model.tar.gz > /dev/null

# Check compression format
file /opt/ml/processing/input/model/model.tar.gz

# Check available space
df -h /tmp

# Try manual extraction for debugging
tar -xzf /opt/ml/processing/input/model/model.tar.gz -C /tmp/test_extract
```

### Compression Issues

**Symptom**: Final package much larger than expected

**Common Causes**:
1. **Already compressed files**: Binary models don't compress well
2. **Duplicate large files**: Same files copied multiple times
3. **Unnecessary files**: Large files not needed for inference

**Solution**:
```bash
# Analyze package contents by size
tar -tzf model.tar.gz | xargs -I {} sh -c 'tar -xzOf model.tar.gz {} | wc -c' | \
  paste <(tar -tzf model.tar.gz) - | sort -k2 -n | tail -10

# Check for duplicates
tar -tzf model.tar.gz | sort | uniq -d

# Review inference script dependencies
# Remove unnecessary large files before packaging
```

### Calibration Integration Issues

**Symptom**: Calibration model not found during inference

**Common Causes**:
1. **Wrong directory structure**: Calibration files not in expected location
2. **Missing summary file**: calibration_summary.json not provided
3. **Path mismatch**: Inference script looking in wrong location

**Solution**:
```bash
# Verify calibration directory structure in package
tar -tzf model.tar.gz | grep calibration

# Should see:
# calibration/calibration_model.pkl
# calibration/calibration_summary.json

# Check inference script expectations
grep -r "calibration" /opt/ml/processing/input/script/inference.py

# Ensure calibration path matches (typically /opt/ml/model/calibration/)
```

## References

### Related Scripts

- [`model_calibration.py`](model_calibration_script.md): Generates the optional calibration artifacts consumed by this script
- [`xgboost_training.py`](xgboost_training_script.md): Produces model.tar.gz that can be packaged by this script
- [`lightgbm_training.py`](lightgbm_training_script.md): Produces model.tar.gz that can be packaged by this script
- [`pytorch_training.py`](pytorch_training_script.md): Produces model.tar.gz that can be packaged by this script

### Related Documentation

- **Step Builder**: `src/cursus/steps/builders/builder_package_step.py`
- **Config Class**: `src/cursus/steps/configs/config_package_step.py`
- **Contract**: [`src/cursus/steps/contracts/package_contract.py`](../../src/cursus/steps/contracts/package_contract.py)
- **Step Specification**: Defines packaging step dependencies and outputs

### Related Design Documents

- **[Packaging Step Improvements](../1_design/packaging_step_improvements.md)**: Design document covering dependency handling, special case processing for inference scripts, and priority handling for local paths vs dependency-provided values

### External References

- [SageMaker Model Packaging](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html): Official guide for SageMaker model package structure
- [MIMS Documentation](https://w.amazon.com/bin/view/MIMS/): Internal MIMS deployment system documentation
- [Python tarfile Module](https://docs.python.org/3/library/tarfile.html): Standard library tar file handling
