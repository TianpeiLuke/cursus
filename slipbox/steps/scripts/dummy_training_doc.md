# Dummy Training Module Documentation

## Overview

The `dummy_training.py` script is a specialized utility in the MODS pipeline that simulates a training step without actually performing model training. Instead, it processes a pre-trained model artifact by unpacking it, adding hyperparameters, and repacking it for downstream steps. This enables integration with the rest of the pipeline framework, particularly the MIMS packaging and payload steps, when a model has been trained externally or pre-trained models are being used.

## Key Features

- Validates existing pre-trained model archive format
- Extracts model archive contents to add additional files
- Integrates hyperparameters.json into the model archive
- Repacks the model for downstream steps
- Maintains contract compatibility with standard training outputs
- Provides comprehensive logging for debugging and traceability

## Core Functions

### Validation and Verification

- **validate_model(input_path)**: Validates that the provided file is a valid tar.gz archive with expected format. Performs checks on file extension and archive validity.

### File and Directory Operations

- **ensure_directory(directory)**: Creates a directory if it doesn't exist, ensuring output paths are available.
- **extract_tarfile(tar_path, extract_path)**: Extracts a tar.gz archive while logging detailed information about its contents.
- **create_tarfile(output_tar_path, source_dir)**: Creates a compressed tar.gz archive from a directory, with detailed logging of files added and compression statistics.
- **copy_file(src, dst)**: Copies a file with appropriate error handling and directory creation.

### Core Processing

- **process_model_with_hyperparameters(model_path, hyperparams_path, output_dir)**: The main processing function that:
  1. Extracts the model archive to a temporary directory
  2. Adds hyperparameters.json to the extracted contents
  3. Repacks everything into a new model.tar.gz archive
  4. Outputs the new archive to the specified directory

### Main Execution Flow

- **main()**: Orchestrates the complete workflow:
  - Retrieves paths defined in the script contract
  - Checks if hyperparameters file exists
  - Either processes the model with hyperparameters or falls back to simple validation and copy
  - Handles exceptions with appropriate error codes

## Input and Output

### Input Files

1. **Model Archive**: 
   - Path: `/opt/ml/processing/input/model/model.tar.gz`
   - Format: Compressed tar.gz archive containing a pre-trained model
   - Validation: Must be a valid tar.gz archive

2. **Hyperparameters File**:
   - Path: `/opt/ml/processing/input/config/hyperparameters.json`
   - Format: JSON file containing model hyperparameters
   - Optional: Script handles cases where this file is missing

### Output Files

1. **Processed Model Archive**:
   - Path: `/opt/ml/processing/output/model/model.tar.gz`
   - Format: Compressed tar.gz archive containing the original model plus hyperparameters.json
   - Purpose: Ready for consumption by downstream MIMS packaging step

## Error Handling

The script includes comprehensive error handling with specific error codes:

| Error Code | Description | Cause |
|------------|-------------|-------|
| 1 | File Not Found | Input files (model or hyperparameters) cannot be found |
| 2 | Validation Error | Model file format is invalid |
| 3 | Runtime Error | General errors during processing |
| 4 | Unexpected Error | Unhandled exceptions with full traceback |

## Use Cases

### 1. Pre-trained Model Integration

Use when you have an externally trained model that needs to be integrated into the MODS pipeline workflow:

```
# Prepare directories
mkdir -p /opt/ml/processing/input/model
mkdir -p /opt/ml/processing/input/config
mkdir -p /opt/ml/processing/output/model

# Copy files
cp pretrained-model.tar.gz /opt/ml/processing/input/model/model.tar.gz
cp hyperparams.json /opt/ml/processing/input/config/hyperparameters.json

# Run the script
python dummy_training.py
```

### 2. Fallback Mode

When hyperparameters aren't available but you need to validate and prepare a model for downstream steps:

```
# Prepare directories
mkdir -p /opt/ml/processing/input/model
mkdir -p /opt/ml/processing/output/model

# Copy model only
cp pretrained-model.tar.gz /opt/ml/processing/input/model/model.tar.gz

# Run the script (will use fallback mode)
python dummy_training.py
```

## Implementation Details

### Temporary Directory Usage

The script uses Python's `tempfile.TemporaryDirectory()` context manager to create a secure workspace for extracting and modifying the model archive. This ensures:

- Clean-up after processing regardless of success or failure
- No conflicts with other processes
- Security of temporary files

### Comprehensive Logging

The script implements detailed logging throughout the process:
- File sizes and counts
- Compression ratios
- Archive content listings
- Success/failure of each operation

### Contract Compatibility

The script adheres to the input/output contract expected by other pipeline steps, particularly:
- Input paths match what would be provided to a real training step
- Output paths and file naming match what downstream MIMS steps expect

## Best Practices Demonstrated

1. **Path Handling**: Uses `pathlib.Path` for robust cross-platform path manipulation
2. **Contextual Processing**: Uses context managers for resource cleanup
3. **Comprehensive Logging**: Detailed logging for debugging and traceability
4. **Explicit Error Codes**: Different exit codes for different error types
5. **Error Handling**: Specific exception types with meaningful messages
6. **Fallback Behavior**: Graceful handling when optional files are missing
