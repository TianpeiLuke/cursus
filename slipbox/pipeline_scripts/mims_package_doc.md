# MIMS Package Module Documentation

## Overview

The `mims_package.py` script is responsible for packaging model artifacts and inference code into a standardized format for deployment through the Machine Intelligence Model System (MIMS). This script takes a trained model (either as a tar.gz archive or individual files) along with inference scripts and combines them into a single deployment-ready package.

## Key Features

- Handles both pre-packaged models (model.tar.gz) and loose model files
- Copies inference scripts to a standardized location within the package
- Creates a properly structured model.tar.gz file for MIMS deployment
- Provides comprehensive logging for debugging and traceability
- Robust error handling and verification throughout the packaging process

## Core Functions

### Directory and File Management

- **ensure_directory(directory)**: Creates a directory if it doesn't exist and logs its details.
- **check_file_exists(path, description)**: Validates file existence and logs detailed information including size, permissions, and modification time.
- **list_directory_contents(path, description)**: Recursively lists directory contents, providing file counts, sizes, and hierarchical structure visualization.

### File Operations

- **copy_file_robust(src, dst)**: Safely copies a file with verification and detailed logging.
- **copy_scripts(src_dir, dst_dir)**: Recursively copies inference scripts from the source to destination directory, preserving the directory structure.

### Archive Handling

- **extract_tarfile(tar_path, extract_path)**: Extracts a tar.gz archive with detailed logging of contents and verification.
- **create_tarfile(output_tar_path, source_dir)**: Creates a compressed tar.gz archive from a directory, providing detailed statistics about the archive contents and compression ratio.

### Main Execution Flow

- **main()**: Orchestrates the complete packaging workflow:
  1. Prepares working and output directories
  2. Processes the input model (extracts tar.gz or copies individual files)
  3. Copies inference scripts to the code directory
  4. Packages everything into the final model.tar.gz
  5. Verifies the output and provides a summary

## Input and Output Structure

### Input Directories

1. **Model Directory**:
   - Path: `/opt/ml/processing/input/model`
   - Contents: Either a model.tar.gz file or individual model files and artifacts
   - Purpose: Contains the trained model to be packaged

2. **Script Directory**:
   - Path: `/opt/ml/processing/input/script`
   - Contents: Inference scripts required for model deployment
   - Purpose: Provides the code needed to run inference with the model

### Output Directory

- Path: `/opt/ml/processing/output`
- Output File: `model.tar.gz`
- Format: A compressed tar archive containing:
  - The model artifacts at the root level
  - Inference scripts in the `code/` directory

### Working Directory

- Path: `/tmp/mims_packaging_directory`
- Purpose: Temporary workspace for extracting, organizing, and packaging files
- Structure:
  - Root: Contains extracted model artifacts
  - `code/`: Contains copied inference scripts

## Implementation Details

### Robust File Operations

The script emphasizes robustness through:
- Detailed logging of file operations
- Size and permission verification after copies
- Exception handling for all file system operations
- Directory existence checks and creation when needed

### Detailed Logging

The script provides comprehensive logging to aid debugging and verification:
- File sizes in MB
- File permissions
- Directory hierarchies with visual indicators
- Compression ratios
- Total file counts and sizes
- Operation summaries

### Error Handling

The script implements thorough error handling:
- Specific exception handling for different file operations
- Graceful fallbacks when expected files are missing
- Traceback logging for unexpected errors
- Verification after key operations

## Usage in the MODS Pipeline

This script is typically used as a processing step after model training or dummy training:

1. **After Training**: Takes the model artifacts produced by a training step and packages them with inference code
2. **After Dummy Training**: Takes a pre-trained model that has been processed by the dummy training step and packages it with inference code

## Best Practices Demonstrated

1. **Comprehensive Logging**: Detailed logging for debugging and traceability
2. **Robust Error Handling**: Exception handling with specific error messages
3. **Verification**: Validation of operations through file existence and size checks
4. **Clean Organization**: Standardized directory structure for model artifacts and code
5. **Path Handling**: Uses `pathlib.Path` for cross-platform path manipulation
6. **Resource Reporting**: Reports disk space and other system resources
7. **Operation Summaries**: Provides concise summaries of operations performed

## Typical Workflow Example

```
Input:
  /opt/ml/processing/input/model/model.tar.gz (trained model)
  /opt/ml/processing/input/script/inference.py
  /opt/ml/processing/input/script/preprocessing.py

Working Directory (/tmp/mims_packaging_directory):
  extracted model files...
  code/
    inference.py
    preprocessing.py

Output:
  /opt/ml/processing/output/model.tar.gz (MIMS-compatible package)
