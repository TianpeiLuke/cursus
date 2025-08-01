# Tabular Preprocessing Module Documentation

## Overview

The `tabular_preprocess.py` script provides a robust data preprocessing framework for tabular data in SageMaker processing jobs. It handles the ingestion, cleaning, and preparation of data from multiple file formats (CSV, TSV, JSON, Parquet) including compressed files, combines data shards, normalizes column names, and creates appropriate train/test/validation splits.

## Key Features

- Multi-format support (CSV, TSV, JSON, JSON Lines, Parquet)
- Automatic handling of compressed (gzipped) files
- Intelligent delimiter detection for CSV/TSV files
- Support for different JSON formats (regular and JSON Lines)
- Automatic combining of data shards from distributed sources
- Label cleaning and normalization
- Stratified data splitting for training workflows
- Column name normalization

## Core Components

### File Format Handling

- **Format Detection**: Automatically identifies and handles various file formats
- **Compression Support**: Transparently processes gzipped files
- **Delimiter Detection**: Uses CSV Sniffer to automatically detect delimiters in CSV/TSV files
- **JSON Format Recognition**: Distinguishes between regular JSON and JSON Lines formats

### Data Processing Functions

- **_is_gzipped(path)**: Detects if a file is gzipped based on extension
- **_detect_separator_from_sample(sample_lines)**: Infers CSV/TSV delimiter from content
- **peek_json_format(file_path, open_func)**: Determines if a JSON file is in regular or lines format
- **_read_json_file(file_path)**: Reads JSON files of either format into a DataFrame
- **_read_file_to_df(file_path)**: Universal file reader that handles all supported formats
- **combine_shards(input_dir)**: Combines multiple data shards into a single DataFrame

### Main Workflow

- **main(job_type, label_field, train_ratio, test_val_ratio, input_base_dir, output_dir)**:
  1. Sets up paths for input data and output
  2. Combines all data shards from the input directory
  3. Normalizes column names and cleans the label field
  4. Creates appropriate data splits based on job type
  5. Saves the processed data to output locations

## Workflow Details

### 1. Data Shards Combination

The script locates and reads all data shards in the input directory matching patterns like:
- `part-*.csv`
- `part-*.csv.gz`
- `part-*.json`
- `part-*.json.gz`
- `part-*.parquet`
- `part-*.snappy.parquet`
- `part-*.parquet.gz`

It then combines these shards into a single DataFrame for further processing.

### 2. Column Name Normalization

The script normalizes column names by replacing special characters:
- Replaces `__DOT__` with `.` in column names to handle dot notation

### 3. Label Field Processing

For the target variable (label field):
- Ensures the label field exists in the data
- Converts categorical labels to numeric indices if needed
- Converts to numeric and handles any invalid values
- Removes rows with missing label values

### 4. Data Splitting

- **For training job type**:
  - Performs stratified train/test/validation split
  - Uses environment-configured split ratios
  - Ensures class distribution is maintained across splits
- **For other job types** (validation, testing):
  - Uses the entire dataset as a single split named after the job type

### 5. Output Generation

Saves each split as a CSV file in the appropriate output directory:
- `/opt/ml/processing/output/{split_name}/{split_name}_processed_data.csv`

## Configuration

### Command Line Arguments

- **--job_type**: Type of job to perform (one of 'training', 'validation', 'testing')

### Environment Variables

- **LABEL_FIELD**: Name of the target/label column (required)
- **TRAIN_RATIO**: Proportion of data for training (default: 0.7)
- **TEST_VAL_RATIO**: Test/validation split ratio (default: 0.5)

### Standard SageMaker Paths

- **Input**: `/opt/ml/processing/input/data/` (contains data shards)
- **Output**: `/opt/ml/processing/output/` (destination for processed splits)

## Usage Examples

### Training Workflow

```bash
# Set up environment variables
export LABEL_FIELD="fraud_flag"
export TRAIN_RATIO="0.8"
export TEST_VAL_RATIO="0.5"

# Run preprocessing script for training
python tabular_preprocess.py --job_type training
```

This creates:
- `/opt/ml/processing/output/train/train_processed_data.csv`
- `/opt/ml/processing/output/test/test_processed_data.csv`
- `/opt/ml/processing/output/val/val_processed_data.csv`

### Validation Workflow

```bash
# Set up environment variables
export LABEL_FIELD="fraud_flag"

# Run preprocessing script for validation
python tabular_preprocess.py --job_type validation
```

This creates:
- `/opt/ml/processing/output/validation/validation_processed_data.csv`

## Error Handling

The script includes robust error handling for:

- Missing input directories
- No data shards found
- Unsupported file formats
- Missing or invalid label field
- JSON parsing errors
- Data combination failures

## Best Practices

1. **Consistent Label Field**: Ensure the LABEL_FIELD environment variable matches the column name in your data

2. **Format Compatibility**: The script is designed to work with most common data formats, but if using custom formats, ensure they're compatible with Pandas readers

3. **Stratification**: For imbalanced datasets, the script performs stratified splitting to maintain class distribution across train/test/val splits

4. **Column Naming**: If your data uses dot notation in column names, the script properly handles transformation from `__DOT__` to `.`

## Integration in the Pipeline

This script is typically:

1. The first step in a model development pipeline
2. Followed by feature preprocessing (e.g., risk table mapping)
3. Used before model training to prepare train/test/val splits
4. Used in validation/testing workflows to prepare evaluation data
