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

The `dummy_training.py` script serves as a specialized SOURCE node processing step that packages a pretrained model by unpacking `model.tar.gz`, adding `hyperparameters.json` inside it, then repacking it for downstream consumption. It bypasses actual training while maintaining full integration with MIMS packaging and payload steps, enabling the use of externally trained or pretrained models in the pipeline framework.

The script implements robust file discovery with multi-path fallback mechanisms, comprehensive validation, and detailed logging throughout the model processing workflow. A key distinguishing feature is its SOURCE node design pattern - it reads files from the source directory (`/opt/ml/code/`) rather than processing inputs, making it suitable for scenarios where models and hyperparameters are embedded in the deployment package.

This approach is particularly valuable for transfer learning scenarios, model serving pipelines, and situations where training has already been completed externally.

## Purpose and Major Tasks

### Primary Purpose
Process pretrained models by injecting hyperparameters into the model archive, enabling seamless integration with downstream MIMS packaging and payload steps without performing actual training.

### Major Tasks
1. **Model Discovery**: Search multiple fallback locations for `model.tar.gz` file
2. **Hyperparameters Discovery**: Search multiple fallback locations for `hyperparameters.json`  
3. **Model Validation**: Verify tar.gz format and archive integrity
4. **Archive Extraction**: Unpack model archive to temporary working directory
5. **Hyperparameters Injection**: Copy hyperparameters.json into extracted model contents
6. **Archive Creation**: Repack model with hyperparameters into new tar.gz
7. **Output Management**: Write processed model to output directory
8. **Logging**: Provide detailed logs including file sizes, compression ratios, and operation status
9. **Error Handling**: Gracefully handle missing files, invalid formats, and processing errors
10. **Cleanup**: Automatically clean temporary directories regardless of success/failure

## Script Contract

### Entry Point
```
dummy_training.py
```

### Input Paths
**None** - This is a SOURCE node that reads from source directory

The script searches for files in these locations with fallback:

**Model Search Paths** (in priority order):
1. `/opt/ml/code/models/model.tar.gz` (Primary: source directory models folder)
2. `/opt/ml/code/model.tar.gz` (Fallback: source directory root)
3. `/opt/ml/processing/input/model/model.tar.gz` (Legacy: processing input if somehow provided)

**Hyperparameters Search Paths** (in priority order):
1. `/opt/ml/code/hyperparams/hyperparameters.json` (Primary: source directory hyperparams folder)
2. `/opt/ml/code/hyperparameters.json` (Fallback: source directory root)
3. `/opt/ml/processing/input/config/hyperparameters.json` (Legacy: processing input if somehow provided)

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
1. Define model search paths (priority order):
   - /opt/ml/code/models/model.tar.gz
   - /opt/ml/code/model.tar.gz  
   - /opt/ml/processing/input/model/model.tar.gz
2. Define hyperparams search paths (priority order):
   - /opt/ml/code/hyperparams/hyperparameters.json
   - /opt/ml/code/hyperparameters.json
   - /opt/ml/processing/input/config/hyperparameters.json
3. Search for model file:
   - Call find_model_file(model_search_paths)
   - If not found: raise FileNotFoundError with search locations
4. Search for hyperparameters file:
   - Call find_hyperparams_file(hyperparams_search_paths)
   - If not found: raise FileNotFoundError with search locations
5. Get output directory from output_paths
6. Log all resolved paths
7. Process model:
   - Call process_model_with_hyperparameters(model_path, hyperparams_path, output_dir)
8. Return processed model path
9. Handle exceptions with logging
```

**Parameters**:
- `input_paths` (dict): Empty dict for SOURCE node
- `output_paths` (dict): Output directory paths
- `environ_vars` (dict): Environment variables (unused)
- `job_args` (Namespace | None): Command-line arguments (unused)

**Returns**: `Path` - Path to processed model.tar.gz

**Raises**: `FileNotFoundError` if required files not found, `Exception` for processing errors

### File Discovery Components

#### `find_model_file(base_paths)`
**Purpose**: Search multiple locations for model.tar.gz with fallback

**Algorithm**:
```python
1. For each base_path in base_paths:
   a. Construct full path: model_path = Path(base_path) / "model.tar.gz"
   b. Check if file exists:
      - If exists: log found location, return model_path
2. If loop completes without finding:
   - Return None
```

**Parameters**:
- `base_paths` (list[str]): List of base directories to search

**Returns**: `Optional[Path]` - Path to model file or None

**Complexity**: O(n) where n = number of search paths

**Note**: Checks paths in order, returns first match

#### `find_hyperparams_file(base_paths)`
**Purpose**: Search multiple locations for hyperparameters.json with fallback

**Algorithm**:
```python
1. For each base_path in base_paths:
   a. Construct full path: hyperparams_path = Path(base_path) / "hyperparameters.json"
   b. Check if file exists:
      - If exists: log found location, return hyperparams_path
2. If loop completes without finding:
   - Return None
```

**Parameters**:
- `base_paths` (list[str]): List of base directories to search

**Returns**: `Optional[Path]` - Path to hyperparameters file or None

**Complexity**: O(n) where n = number of search paths

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
**Purpose**: Main processing function - unpack, add hyperparameters, repack

**Algorithm**:
```python
1. Log processing start with all paths
2. Validate inputs:
   a. If model_path not exists: raise FileNotFoundError
   b. If hyperparams_path not exists: raise FileNotFoundError
3. Create temporary working directory:
   - with tempfile.TemporaryDirectory() as temp_dir:
4. Extract model archive:
   - extract_tarfile(model_path, working_dir)
5. Copy hyperparameters to working directory:
   - hyperparams_dest = working_dir / "hyperparameters.json"
   - copy_file(hyperparams_path, hyperparams_dest)
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
- `hyperparams_path` (Path): Hyperparameters.json path
- `output_dir` (Path): Output directory for processed model

**Returns**: `Path` - Path to processed model.tar.gz

**Raises**: `FileNotFoundError` if inputs missing, `Exception` for processing errors

**Complexity**: O(n × m) where n = number of files, m = average file size

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

### Multi-Path Fallback Search Algorithm
**Problem**: Locate required files across multiple potential locations with graceful fallback

**Solution Strategy**:
1. Define priority-ordered search paths
2. Iterate paths in order
3. Return first match found
4. Return None if no match (caller handles error)

**Algorithm**:
```python
def find_file(base_paths, filename):
    for base_path in base_paths:
        file_path = Path(base_path) / filename
        if file_path.exists():
            log(f"Found {filename} at: {file_path}")
            return file_path
    return None
```

**Complexity**: O(n) where n = number of search paths

**Key Features**:
- Priority-based search (checks paths in order)
- Early termination on first match
- Graceful failure (returns None, not exception)
- Detailed logging for debugging

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
