---
tags:
  - code
  - processing_script
  - dummy_training
  - model_processing
  - source_node
keywords:
  - dummy training
  - pretrained model
  - model repacking
  - hyperparameters injection
  - source node
  - model archive
topics:
  - model processing
  - pipeline integration
  - ML pipelines
language: python
date of note: 2025-11-18
---

# Dummy Training Script Documentation

## Overview

The `dummy_training.py` script serves as a specialized INTERNAL node processing step with flexible input modes that packages a pretrained model by unpacking `model.tar.gz`, adding `hyperparameters.json` inside it, then repacking it for downstream consumption. It bypasses actual training while maintaining full integration with MIMS packaging and payload steps, enabling the use of externally trained or pretrained models in the pipeline framework.

The script implements a unified path assignment strategy in `__main__` that supports both dependency injection (INTERNAL mode) and SOURCE fallback. Paths are pre-configured based on whether input channels exist, simplifying the find functions to simple existence checks with a final relative path fallback. This design provides robust file discovery with comprehensive validation and detailed logging throughout the model processing workflow.

This approach is particularly valuable for transfer learning scenarios, model serving pipelines, and situations where training has already been completed externally. The flexible input modes allow the step to work as either an INTERNAL node (receiving inputs from previous training steps) or a SOURCE node (reading from embedded deployment package files).

## Purpose and Major Tasks

### Primary Purpose
Process pretrained models by conditionally injecting hyperparameters into the model archive, enabling seamless integration with downstream MIMS packaging and payload steps without performing actual training.

**Conditional Injection Logic**:
- If model.tar.gz already contains hyperparameters.json (e.g., from PyTorch/XGBoost training), the original hyperparameters are preserved
- If model.tar.gz does NOT contain hyperparameters.json, the script requires and injects hyperparameters from the input channel
- The script only fails if BOTH the model lacks hyperparameters AND no input hyperparameters are provided

### Major Tasks
1. **Model Discovery**: Search multiple fallback locations for `model.tar.gz` file
2. **Hyperparameters Discovery**: Search multiple fallback locations for `hyperparameters.json` (optional)
3. **Model Validation**: Verify tar.gz format and archive integrity
4. **Archive Extraction**: Unpack model archive to temporary working directory
5. **Hyperparameters Check**: Verify if hyperparameters.json exists in extracted model
6. **Conditional Hyperparameters Injection**: 
   - If hyperparameters.json found in model → Keep original, skip injection
   - If hyperparameters.json NOT in model + input provided → Inject from input
   - If hyperparameters.json NOT in model + no input → Fail with clear error
7. **Archive Creation**: Repack model (with hyperparameters if injected) into new tar.gz
8. **Output Management**: Write processed model to output directory
9. **Logging**: Provide detailed logs including file sizes, compression ratios, and conditional logic decisions
10. **Error Handling**: Gracefully handle missing files, invalid formats, and processing errors
11. **Cleanup**: Automatically clean temporary directories regardless of success/failure

## Script Contract

### Entry Point
```
dummy_training.py
```

### Input Paths
**None** - This is an INTERNAL node with flexible input modes supporting both dependency injection and SOURCE fallback

The script uses a unified path assignment strategy in `__main__`:

**Model Artifacts Path Assignment**:
1. **Input Channel** (if exists): `/opt/ml/input/data/model_artifacts_input/` (from dependency injection)
2. **SOURCE Fallback**: `/opt/ml/code/models/` (embedded in deployment package)
3. **Final Fallback**: Relative to script location (`Path(__file__).parent`)

**Hyperparameters Path Assignment**:
1. **Input Channel** (if exists): `/opt/ml/input/data/hyperparameters_s3_uri/` (from dependency injection)
2. **SOURCE Fallback**: `/opt/ml/code/hyperparams/` (embedded in deployment package)

### Output Paths
| Path | Location | Description |
|------|----------|-------------|
| `model_output` | `/opt/ml/processing/output/model` | Processed model archive directory |

**Output Files**:
- `output/model/model.tar.gz`: Repacked model archive containing original model + hyperparameters.json

### Required Environment Variables

**None** - All environment variables are optional.

### Optional Environment Variables

**None** - Script uses hard-coded paths from contract.

### Job Arguments
**None** - This script does not require command-line arguments.

## Input Data Structure

### Expected Input Format
```
/opt/ml/code/
├── models/
│   └── model.tar.gz          # Pretrained model archive
└── hyperparams/
    └── hyperparameters.json  # Model hyperparameters
```

**Alternative structure** (fallback):
```
/opt/ml/code/
├── model.tar.gz              # Pretrained model at root
└── hyperparameters.json      # Hyperparameters at root
```

### Model Archive Structure
**model.tar.gz** (input):
```
model.tar.gz
├── saved_model.pb            # TensorFlow SavedModel
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── assets/
    └── vocab.txt
```

Or for PyTorch:
```
model.tar.gz
├── model.pth                 # PyTorch model weights
├── config.json               # Model configuration
└── tokenizer.json            # Tokenizer config
```

### Hyperparameters File Structure
**hyperparameters.json**:
```json
{
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 10,
  "optimizer": "adam",
  "model_type": "bert-base-uncased",
  "max_seq_length": 128,
  "num_labels": 3,
  "training_date": "2025-11-18",
  "framework": "pytorch",
  "version": "1.0.0"
}
```

### Configuration Example

**Directory Setup**:
```bash
# SOURCE node structure
mkdir -p /opt/ml/code/models
mkdir -p /opt/ml/code/hyperparams
mkdir -p /opt/ml/processing/output/model

# Place files
cp pretrained_model.tar.gz /opt/ml/code/models/model.tar.gz
cp model_hyperparams.json /opt/ml/code/hyperparams/hyperparameters.json
```

## Output Data Structure

### Output Directory Structure
```
output/
└── model/
    └── model.tar.gz          # Repacked model with hyperparameters
```

### Output Model Archive Structure
**model.tar.gz** (output):
```
model.tar.gz
├── saved_model.pb            # Original model files
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
├── assets/
│   └── vocab.txt
└── hyperparameters.json      # ← Added by script
```

**Key Properties**:
- Contains all original model files
- Includes hyperparameters.json at root level
- Maintains original file permissions and structure
- Compressed as tar.gz for size optimization
- Compatible with downstream MIMS packaging step

### Output File Metadata

**Logging Output Example**:
```
2025-11-18 16:00:01 - INFO - Tar file contents before extraction:
  saved_model.pb (5.23MB)
  variables/variables.data-00000-of-00001 (120.45MB)
  variables/variables.index (0.01MB)
  assets/vocab.txt (0.50MB)
Total size in tar: 126.19MB

2025-11-18 16:00:15 - INFO - Tar creation summary:
  Files added: 5
  Total uncompressed size: 126.21MB
  Compressed tar size: 85.43MB
  Compression ratio: 67.68%
```

## Key Functions and Tasks

### Main Orchestration Component

#### `main(input_paths, output_paths, environ_vars, job_args)`
**Purpose**: Standardized entry point orchestrating the complete model processing workflow

**Algorithm**:
```python
1. Log processing mode and input paths
2. Search for model file:
   - Call find_model_file(input_paths)
   - If not found: raise FileNotFoundError with assigned path
3. Search for hyperparameters file:
   - Call find_hyperparams_file(input_paths)
   - If not found: raise FileNotFoundError with assigned path
4. Get output directory from output_paths
5. Log all resolved paths
6. Process model:
   - Call process_model_with_hyperparameters(model_path, hyperparams_path, output_dir)
7. Return processed model path
8. Handle exceptions with logging
```

**Parameters**:
- `input_paths` (dict): Pre-configured paths from __main__ with keys:
  - "model_artifacts_input": Path to directory containing model.tar.gz
  - "hyperparameters_s3_uri": Path to directory containing hyperparameters.json
- `output_paths` (dict): Output directory paths
- `environ_vars` (dict): Environment variables (unused)
- `job_args` (Namespace | None): Command-line arguments (unused)

**Returns**: `Path` - Path to processed model.tar.gz

**Raises**: `FileNotFoundError` if required files not found, `Exception` for processing errors

**Note**: Paths in input_paths are pre-resolved in __main__ to point to either input channels or SOURCE fallback directories

### File Discovery Components

#### `find_model_file(input_paths)`
**Purpose**: Check for model.tar.gz at pre-configured path with relative fallback

**Algorithm**:
```python
1. Get pre-configured path from input_paths["model_artifacts_input"]
2. Construct full path: model_path = Path(model_artifacts_input) / "model.tar.gz"
3. Check if file exists:
   - If exists: log found location, return model_path
   - If not: log warning
4. Final fallback - relative to script location:
   a. script_dir = Path(__file__).parent
   b. code_fallback_path = script_dir / "model.tar.gz"
   c. If exists: log found location, return code_fallback_path
5. Return None if not found in any location
```

**Parameters**:
- `input_paths` (dict): Dictionary with pre-configured paths from __main__
  - "model_artifacts_input": Path to directory (input channel or SOURCE fallback)

**Returns**: `Optional[Path]` - Path to model file or None

**Complexity**: O(1) - checks exactly 2 locations (pre-configured + relative fallback)

**Note**: Path is pre-resolved in __main__ to point to input channel or /opt/ml/code/models/

#### `find_hyperparams_file(input_paths)`
**Purpose**: Check for hyperparameters.json at pre-configured path

**Algorithm**:
```python
1. Get pre-configured path from input_paths["hyperparameters_s3_uri"]
2. Construct full path: hparam_path = Path(hyperparameters_s3_uri) / "hyperparameters.json"
3. Check if file exists:
   - If exists: log found location, return hparam_path
   - If not: log warning
4. Return None if not found
```

**Parameters**:
- `input_paths` (dict): Dictionary with pre-configured paths from __main__
  - "hyperparameters_s3_uri": Path to directory (input channel or SOURCE fallback)

**Returns**: `Optional[Path]` - Path to hyperparameters file or None

**Complexity**: O(1) - checks exactly 1 location (pre-configured path)

**Note**: Path is pre-resolved in __main__ to point to input channel or /opt/ml/code/hyperparams/

### Validation Component

#### `validate_model(input_path)`
**Purpose**: Validate model file format and archive integrity

**Algorithm**:
```python
1. Log validation start
2. Check file extension:
   - suffix = input_path.suffix
   - If not ".tar.gz" and not str(input_path).endswith(".tar.gz"):
     * Raise ValueError with error code: INVALID_FORMAT
3. Check if valid tar archive:
   - If not tarfile.is_tarfile(input_path):
     * Raise ValueError with error code: INVALID_ARCHIVE
4. Log validation success
5. Return True
```

**Parameters**:
- `input_path` (Path): Path to model archive

**Returns**: `bool` - True if validation passes

**Raises**: `ValueError` with error codes for invalid format or archive

**Validation Checks**:
- File extension must be .tar.gz
- File must be a valid tar archive (readable by tarfile module)
- Additional checks can be added (file size, required contents, etc.)

### Archive Processing Components

#### `extract_tarfile(tar_path, extract_path)`
**Purpose**: Extract tar.gz archive with detailed logging

**Algorithm**:
```python
1. Log extraction start
2. Validate tar_path exists:
   - If not exists: raise FileNotFoundError
3. Ensure extract_path directory exists
4. Try to extract:
   a. Open tar file: with tarfile.open(tar_path, "r:*") as tar:
   b. Log archive contents:
      - Initialize total_size = 0
      - For each member in tar.getmembers():
        * Calculate size_mb = member.size / 1024 / 1024
        * Accumulate total_size += size_mb
        * Log member name and size
      - Log total size
   c. Extract all: tar.extractall(path=extract_path)
   d. Log extraction complete
5. Catch exceptions:
   - Log error with traceback
   - Re-raise exception
```

**Parameters**:
- `tar_path` (Path): Path to tar.gz file
- `extract_path` (Path): Directory to extract to

**Returns**: None

**Raises**: `FileNotFoundError` if tar file missing, `Exception` for extraction errors

**Complexity**: O(n × m) where n = number of files, m = average file size

#### `create_tarfile(output_tar_path, source_dir)`
**Purpose**: Create compressed tar.gz from directory contents

**Algorithm**:
```python
1. Log tar creation start
2. Ensure output directory exists
3. Try to create tar:
   a. Initialize counters: total_size = 0, files_added = 0
   b. Open tar for writing: with tarfile.open(output_tar_path, "w:gz") as tar:
   c. Recursively iterate source_dir:
      - For each item in source_dir.rglob("*"):
        * If item.is_file():
          - Calculate arcname = item.relative_to(source_dir)
          - Calculate size_mb = item.stat().st_size / 1024 / 1024
          - Accumulate total_size += size_mb
          - Increment files_added += 1
          - Log file being added
          - Add to tar: tar.add(item, arcname=arcname)
   d. Log summary: files added, total uncompressed size
   e. If output file exists:
      - Calculate compressed_size
      - Log compressed size and compression ratio
4. Catch exceptions:
   - Log error with traceback
   - Re-raise exception
```

**Parameters**:
- `output_tar_path` (Path): Path for output tar.gz
- `source_dir` (Path): Directory to compress

**Returns**: None

**Raises**: `Exception` for creation errors

**Complexity**: O(n × m) where n = number of files, m = average file size

#### `process_model_with_hyperparameters(model_path, hyperparams_path, output_dir)`
**Purpose**: Main processing function - unpack, conditionally add hyperparameters, repack

**Algorithm**:
```python
1. Log processing start with all paths
2. Validate model input:
   a. If model_path not exists: raise FileNotFoundError
3. Create temporary working directory:
   - with tempfile.TemporaryDirectory() as temp_dir:
4. Extract model archive:
   - extract_tarfile(model_path, working_dir)
5. Check for existing hyperparameters in model:
   - hyperparams_dest = working_dir / "hyperparameters.json"
   - If hyperparams_dest.exists():
     a. Log "HYPERPARAMETERS ALREADY IN MODEL"
     b. If hyperparams_path provided:
        - Log warning that input will be IGNORED
        - Log reason: model already contains hyperparameters
     c. Else: Log that no input needed
   - Else (hyperparameters NOT in model):
     a. Log "HYPERPARAMETERS NOT IN MODEL"
     b. If hyperparams_path provided:
        - Log injection from input
        - copy_file(hyperparams_path, hyperparams_dest)
        - Log success
     c. Else (no input provided):
        - Log error
        - Raise FileNotFoundError with clear message
6. Ensure output directory exists
7. Create output archive:
   - output_path = output_dir / "model.tar.gz"
   - create_tarfile(output_path, working_dir)
8. Log completion with output path
9. Return output_path
10. Temporary directory automatically cleaned up on context exit
```

**Parameters**:
- `model_path` (Path): Input model.tar.gz path
- `hyperparams_path` (Optional[Path]): Optional hyperparameters.json path (None if not provided)
- `output_dir` (Path): Output directory for processed model

**Returns**: `Path` - Path to processed model.tar.gz

**Raises**: 
- `FileNotFoundError` if model doesn't exist, or if hyperparameters missing from both model and input
- `Exception` for other processing errors

**Complexity**: O(n × m) where n = number of files, m = average file size

**Conditional Behavior**:
- Model contains hyperparameters.json → Preserves original (injection skipped)
- Model missing hyperparameters.json + input provided → Injects from input
- Model missing hyperparameters.json + no input → Fails with clear error

### Utility Components

#### `ensure_directory(directory)`
**Purpose**: Create directory if it doesn't exist

**Algorithm**:
```python
1. Try to create directory:
   - directory.mkdir(parents=True, exist_ok=True)
   - Log success
   - Return True
2. Catch exceptions:
   - Log error with traceback
   - Return False
```

**Parameters**:
- `directory` (Path): Directory to create

**Returns**: `bool` - Success status

**Note**: Uses exist_ok=True to avoid errors if already exists

#### `copy_file(src, dst)`
**Purpose**: Copy file with directory creation and error handling

**Algorithm**:
```python
1. Log copy operation
2. Validate source exists:
   - If not src.exists(): raise FileNotFoundError
3. Ensure destination directory exists
4. Try to copy:
   - shutil.copy2(src, dst)  # Preserves metadata
   - Log success
5. Catch exceptions:
   - Log error with traceback
   - Re-raise exception
```

**Parameters**:
- `src` (Path): Source file path
- `dst` (Path): Destination file path

**Returns**: None

**Raises**: `FileNotFoundError` if source missing, `Exception` for copy errors

**Note**: Uses copy2 to preserve file metadata (timestamps, permissions)

## Algorithms and Data Structures

### Unified Path Assignment Strategy
**Problem**: Support both dependency injection (INTERNAL mode) and SOURCE fallback seamlessly

**Solution Strategy**:
1. Path resolution happens in `__main__` before calling main()
2. Check if input channels exist (dependency injection)
3. If not, assign SOURCE fallback paths
4. Pass pre-configured paths to main() via input_paths dict
5. Find functions simply check existence at pre-configured path

**Algorithm**:
```python
# In __main__
if os.path.exists(CONTAINER_PATHS["MODEL_ARTIFACTS_INPUT"]):
    input_paths["model_artifacts_input"] = CONTAINER_PATHS["MODEL_ARTIFACTS_INPUT"]
    logger.info("[Input Channel] Using model artifacts from: ...")
else:
    input_paths["model_artifacts_input"] = "/opt/ml/code/models"
    logger.info("[SOURCE Fallback] Using model artifacts from: ...")

# In find_model_file()
model_path = Path(input_paths["model_artifacts_input"]) / "model.tar.gz"
if model_path.exists():
    return model_path
# Final fallback: relative to script
return Path(__file__).parent / "model.tar.gz" if exists else None
```

**Complexity**: O(1) - path assignment is constant time, existence check is O(1)

**Key Features**:
- Centralized path resolution in `__main__`
- Clear [Input Channel] vs [SOURCE Fallback] logging
- Simplified find functions (just existence checks)
- Relative path fallback for model (script-location independent)
- No iteration over multiple paths - pre-resolved

### Temporary Directory Pattern
**Problem**: Safely manipulate archive contents without affecting original files or leaving artifacts

**Data Structure**:
```python
# Context manager ensures cleanup
with tempfile.TemporaryDirectory() as temp_dir:
    working_dir = Path(temp_dir)
    # Extract, modify, repack
    # Automatic cleanup on exit (success or failure)
```

**Key Features**:
- Automatic cleanup regardless of success/failure
- Isolated workspace per execution
- No conflicts with concurrent executions
- Security through OS-managed temp directories

### Archive Processing Pipeline
**Problem**: Transform model archive by adding hyperparameters while preserving structure

**Pipeline Stages**:
```
1. Validation → Check format and integrity
2. Extraction → Unpack to temporary directory
3. Injection → Add hyperparameters.json
4. Compression → Repack with all files
5. Output → Write to destination
6. Cleanup → Remove temporary files
```

**Algorithm**:
```python
def process_archive(model_path, hyperparams_path, output_dir):
    # Stage 1: Validate
    validate_model(model_path)
    
    # Stage 2-5: Process with temp directory
    with tempfile.TemporaryDirectory() as temp:
        working_dir = Path(temp)
        
        # Stage 2: Extract
        extract_tarfile(model_path, working_dir)
        
        # Stage 3: Inject
        copy_file(hyperparams_path, working_dir / "hyperparameters.json")
        
        # Stage 4-5: Compress and output
        output_path = output_dir / "model.tar.gz"
        create_tarfile(output_path, working_dir)
    
    # Stage 6: Cleanup (automatic via context manager)
    return output_path
```

**Complexity**: O(n × m) where n = files, m = average file size

**Key Features**:
- Linear pipeline with clear stages
- Fail-fast validation
- Isolated processing environment
- Automatic resource cleanup

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Find File | O(n) | O(1) | n = search paths |
| Validate Archive | O(1) | O(1) | Format checks only |
| Extract Archive | O(f × s) | O(f × s) | f = files, s = avg size |
| Copy File | O(s) | O(1) | s = file size |
| Create Archive | O(f × s) | O(f × s) | Includes compression |
| Full Pipeline | O(f × s) | O(f × s) | Dominated by archive ops |

**Overall Complexity**: O(f × s) where f = number of files in archive, s = average file size

**Typical Performance**:
- Small models (<100MB, <50 files): < 5 seconds
- Medium models (100MB-1GB, 50-500 files): 5-30 seconds
- Large models (>1GB, >500 files): 30-180 seconds

### Memory Usage

**Peak Memory**: O(f × s) for archive contents in temporary directory

**Components**:
- Temporary directory: O(f × s) for extracted files
- In-memory buffers: O(b) where b = buffer size (typically 64KB-1MB)
- Archive creation: O(f × s) for reading and compressing files
- Metadata tracking: O(f) for file list

**Memory Optimization Opportunities**:
1. Stream processing for very large files
2. Incremental compression (process files individually)
3. Memory-mapped file operations
4. Configurable buffer sizes

### Compression Performance

**Typical Compression Ratios**:
- Text files (JSON, config): 70-90% compression
- Binary models (SavedModel, PyTorch): 20-40% compression
- Mixed content: 40-70% compression

**Example**:
```
Input: 126.21MB uncompressed (5 files)
Output: 85.43MB compressed tar.gz
Ratio: 67.68% of original size
Time: ~15 seconds on standard hardware
```

**Factors Affecting Performance**:
- File count (more files = more overhead)
- File sizes (larger files = longer processing)
- File types (text compresses better than binary)
- CPU speed (compression is CPU-bound)
- Disk I/O speed (extraction/creation disk-bound)

## Error Handling

### Error Types

#### File Discovery Errors
- **Model Not Found**: Cannot locate model.tar.gz in any search path
  - **Handling**: Raises FileNotFoundError with list of searched locations
  - **Exit Code**: 1

- **Hyperparameters Not Found**: Cannot locate hyperparameters.json
  - **Handling**: Raises FileNotFoundError with list of searched locations
  - **Exit Code**: 1

#### Validation Errors
- **Invalid Format**: File extension not .tar.gz
  - **Handling**: Raises ValueError with ERROR_CODE: INVALID_FORMAT
  - **Exit Code**: 1

- **Invalid Archive**: File exists but is not a valid tar archive
  - **Handling**: Raises ValueError with ERROR_CODE: INVALID_ARCHIVE
  - **Exit Code**: 1

#### Processing Errors
- **Extraction Failed**: Cannot extract tar archive
  - **Handling**: Logs error with traceback, propagates exception
  - **Exit Code**: 1

- **Copy Failed**: Cannot copy hyperparameters file
  - **Handling**: Logs error with traceback, propagates exception
  - **Exit Code**: 1

- **Creation Failed**: Cannot create output tar archive
  - **Handling**: Logs error with traceback, propagates exception
  - **Exit Code**: 1

#### I/O Errors
- **Output Directory Creation Failed**: Cannot create output directory
  - **Handling**: ensure_directory logs error, returns False
  - **Impact**: Subsequent operations will fail

- **Insufficient Disk Space**: Not enough space for extraction or output
  - **Handling**: Propagates OSError with system message
  - **Exit Code**: 1

### Error Response Structure

When processing fails, detailed logging provides:

```python
logger.error(f"Required file not found: {e}")
logger.error(f"Error in dummy training: {e}")
logger.error(traceback.format_exc())
sys.exit(1)
```

**Exit Codes**:
- 0: Success
- 1: General failure (file not found, validation, processing)

**Resilience Features**:
- Multi-path fallback for file discovery
- Detailed error messages with search locations
- Automatic temporary directory cleanup
- Comprehensive logging for debugging

## Best Practices

### For Production Deployments
1. **Source Directory Structure**: Use recommended `/opt/ml/code/models/` and `/opt/ml/code/hyperparams/` structure
2. **Pre-validate Archives**: Verify model.tar.gz integrity before deployment
3. **Monitor Logs**: Check logs for compression ratios and processing times
4. **Disk Space**: Ensure sufficient space (3× model size for extraction, processing, output)
5. **Version Hyperparameters**: Include version info in hyperparameters.json

### For Development
1. **Test Fallback Paths**: Verify script finds files in all configured locations
2. **Validate Archive Contents**: Check extracted model has expected structure
3. **Test Error Handling**: Verify graceful failure with missing/invalid files
4. **Review Compression**: Ensure compression ratios are reasonable
5. **Inspect Output**: Validate hyperparameters.json is correctly placed in archive

### For Performance Optimization
1. **Minimize File Count**: Fewer large files better than many small files
2. **Pre-compress Large Files**: Some frameworks (e.g., ONNX) can be pre-compressed
3. **Use Fast Storage**: SSD/NVMe significantly faster than network storage
4. **Monitor Temp Space**: Ensure temp directory has sufficient space
5. **Optimize Archive Contents**: Remove unnecessary files before packaging

## Example Configurations

### Basic SOURCE Node Setup
```bash
# Directory structure
mkdir -p /opt/ml/code/models
mkdir -p /opt/ml/code/hyperparams
mkdir -p /opt/ml/processing/output/model

# Place files
cp pretrained_bert.tar.gz /opt/ml/code/models/model.tar.gz
cat > /opt/ml/code/hyperparams/hyperparameters.json <<EOF
{
  "model_type": "bert-base-uncased",
  "max_seq_length": 128,
  "num_labels": 3,
  "framework": "pytorch"
}
EOF

# Run script
python dummy_training.py
```

**Use Case**: Standard pretrained model integration

### Fallback Path Testing
```bash
# Test fallback to root directory
mkdir -p /opt/ml/code
mkdir -p /opt/ml/processing/output/model

cp model.tar.gz /opt/ml/code/
cp hyperparameters.json /opt/ml/code/

# Run script (will use fallback paths)
python dummy_training.py
```

**Use Case**: Simplified directory structure, testing fallback mechanism

### Large Model Processing
```bash
# For models >1GB
mkdir -p /opt/ml/code/models
mkdir -p /opt/ml/code/hyperparams
mkdir -p /opt/ml/processing/output/model

# Ensure sufficient disk space (3× model size)
df -h /opt/ml

# Place large model
cp large_pretrained_model.tar.gz /opt/ml/code/models/model.tar.gz
cp hyperparams.json /opt/ml/code/hyperparams/hyperparameters.json

# Run with monitoring
time python dummy_training.py
```

**Use Case**: Large model processing with performance monitoring

### Debug Mode Setup
```bash
# Enable detailed Python logging
export PYTHONUNBUFFERED=1

# Run with full output
python -u dummy_training.py 2>&1 | tee processing.log

# Review logs
cat processing.log | grep -E "(Found|Extracting|Adding|Compressed)"
```

**Use Case**: Troubleshooting, performance analysis

## Integration Patterns

### Upstream Integration (SOURCE Node)
```
Source Directory (/opt/ml/code/) → DummyTraining
   ↓
model.tar.gz + hyperparameters.json
```

**Input Sources**:
- Deployment package embedded files
- Pre-downloaded pretrained models
- Externally trained model artifacts

### Downstream Integration
```
DummyTraining → model.tar.gz (with hyperparameters)
   ↓
MIMS Package → packaged model
   ↓
MIMS Payload → deployment artifact
```

**Output Consumers**:
- **MIMS Package Step**: Uses model.tar.gz for packaging
- **MIMS Payload Step**: Uses packaged model for deployment
- **Model Registration**: May consume model for versioning

### Pipeline Integration Pattern

**Step Dependencies**:
```python
# Pipeline definition
dummy_training_step = DummyTrainingStepBuilder(config).create_step()

# Downstream steps use output
package_step = PackageStepBuilder(config).create_step(
    inputs={
        'model': dummy_training_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
    }
)

payload_step = PayloadStepBuilder(config).create_step(
    inputs={
        'packaged_model': package_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
    }
)
```

### Workflow Example
1. **Prepare**: Embed model.tar.gz and hyperparameters.json in source directory
2. **Deploy**: Deploy SOURCE node with embedded files
3. **Execute**: DummyTraining processes files, outputs combined archive
4. **Package**: MIMS Package step creates deployment artifact
5. **Test**: MIMS Payload step validates deployment
6. **Register**: Optional model registration for versioning

## Troubleshooting

### Model File Not Found

**Symptom**: Script fails with "Model file (model.tar.gz) not found in any of these locations"

**Common Causes**:
1. **Wrong directory structure**: Files not in expected locations
2. **Incorrect filename**: File named differently (e.g., "model.tar", "model.tgz")
3. **Missing deployment**: Files not included in deployment package

**Solution**:
1. Check search paths from error message
2. Verify file exists: `ls -la /opt/ml/code/models/model.tar.gz`
3. Check filename exactly matches "model.tar.gz"
4. Verify deployment package includes model
5. Review logs for attempted search locations

### Hyperparameters File Not Found

**Symptom**: Script fails with "Hyperparameters file (hyperparameters.json) not found"

**Common Causes**:
1. **Wrong directory structure**: File not in expected location
2. **Incorrect filename**: File named differently (e.g., "hyperparams.json", "config.json")
3. **Missing from deployment**: File not included in package

**Solution**:
1. Check search paths from error message
2. Verify file exists: `ls -la /opt/ml/code/hyperparams/hyperparameters.json`
3. Check filename exactly matches "hyperparameters.json"
4. Verify deployment includes hyperparameters
5. Use fallback location: `/opt/ml/code/hyperparameters.json`

### Invalid Archive Format

**Symptom**: Script fails with "ERROR_CODE: INVALID_FORMAT" or "ERROR_CODE: INVALID_ARCHIVE"

**Common Causes**:
1. **Wrong file format**: Not a tar.gz file (e.g., plain tar, zip)
2. **Corrupted archive**: File damaged during transfer
3. **Incomplete download**: Archive not fully downloaded

**Solution**:
1. Verify file extension: `file /opt/ml/code/models/model.tar.gz`
2. Test archive integrity: `tar -tzf /opt/ml/code/models/model.tar.gz`
3. Check file size matches expected
4. Re-download or re-create archive if corrupted
5. Ensure file is compressed tar (tar.gz, not just tar)

### Insufficient Disk Space

**Symptom**: Script fails with OSError or "No space left on device"

**Common Causes**:
1. **Temp directory full**: /tmp partition too small
2. **Output directory full**: Insufficient space in /opt/ml/processing/output
3. **Large model**: Model requires 3× its size (extract + process + output)

**Solution**:
1. Check available space: `df -h /tmp /opt/ml/processing`
2. Clean temp directory: `rm -rf /tmp/*` (careful!)
3. Use larger instance with more disk space
4. Optimize model archive (remove unnecessary files)
5. Mount additional storage if available

### Slow Processing Performance

**Symptom**: Script takes much longer than expected

**Common Causes**:
1. **Large file count**: Many small files cause overhead
2. **Slow storage**: Network storage slower than local SSD
3. **CPU bottleneck**: Compression CPU-intensive
4. **Large individual files**: Multi-GB files take time to compress

**Solution**:
1. Review logs for file count and sizes
2. Check if using network storage (slower)
3. Monitor CPU usage during execution
4. Consider pre-compressing large files
5. Use faster instance type if critical

## References

### Related Scripts
- [`tabular_preprocessing.py`](tabular_preprocess_script.md): Upstream data preprocessing
- [`package.py`](package_script.md): Downstream MIMS packaging step
- [`payload.py`](payload_script.md): Downstream MIMS payload testing
- [`mims_registration.py`](mims_registration_script.md): Model registration step

### Related Documentation
- **Step Builder**: Dummy training step builder implementation
- **Config Class**: Configuration class for dummy training step
- **Contract**: [`src/cursus/steps/contracts/dummy_training_contract.py`](../../src/cursus/steps/contracts/dummy_training_contract.py)
- **Step Specification**: SOURCE node specification with empty inputs

### Related Design Documents
- **[Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)**: Step builder patterns
- **[SOURCE Node Design Pattern](../1_design/source_node_pattern.md)**: SOURCE node architecture (if exists)

### External References
- [Python tarfile Documentation](https://docs.python.org/3/library/tarfile.html): tar archive handling
- [Python tempfile Documentation](https://docs.python.org/3/library/tempfile.html): Temporary directory management
- [Python pathlib Documentation](https://docs.python.org/3/library/pathlib.html): Path manipulation
- [SageMaker Processing Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html): Processing job architecture
