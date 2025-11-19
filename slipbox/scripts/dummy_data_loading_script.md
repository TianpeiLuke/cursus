---
tags:
  - code
  - processing_script
  - data_loading
  - schema_generation
  - data_sharding
keywords:
  - dummy data loading
  - user data processing
  - schema signature
  - metadata generation
  - format detection
  - data sharding
topics:
  - data loading
  - data preprocessing
  - ML pipelines
language: python
date of note: 2025-11-18
---

# Dummy Data Loading Script Documentation

## Overview

The `dummy_data_loading.py` script serves as a drop-in replacement for the CradleDataLoadingStep by processing user-provided data instead of calling internal Cradle services. It automatically detects data formats (CSV, Parquet, JSON), generates schema signatures and metadata, and outputs processed data in configurable formats for downstream pipeline steps.

The script operates in two modes: **legacy mode** (default, backward compatible) and **enhanced mode** (writes actual data shards). It performs format auto-detection, combines multiple input files, generates comprehensive metadata with column statistics, and writes outputs in a format compatible with tabular preprocessing steps.

A key feature is its flexibility in handling multiple data formats and its ability to scale from small datasets (single file) to large datasets (multiple shards) while maintaining full compatibility with the Cursus framework's processing pipeline.

## Purpose and Major Tasks

### Primary Purpose
Process user-provided data files to replace Cradle service calls, enabling local data loading for ML pipelines with automatic schema inference and metadata generation.

### Major Tasks
1. **Data Discovery**: Recursively search input directory for supported data files
2. **Format Detection**: Auto-detect file format (CSV, Parquet, JSON) based on extension and content
3. **Data Reading**: Read multiple data files and combine into unified DataFrame
4. **Schema Generation**: Extract column names to create schema signature
5. **Metadata Generation**: Compute comprehensive column statistics and data info
6. **Signature Output**: Write schema signature in CSV format for downstream compatibility
7. **Metadata Output**: Write JSON metadata file with detailed column information
8. **Data Output**: Write processed data as single file or multiple shards
9. **Format Conversion**: Support CSV, JSON, and Parquet output formats
10. **Error Handling**: Gracefully handle missing files, format errors, and processing failures

## Script Contract

### Entry Point
```
dummy_data_loading.py
```

### Input Paths
| Path | Location | Description |
|------|----------|-------------|
| `INPUT_DATA` | `/opt/ml/processing/input/data` | User-provided data files (CSV/Parquet/JSON) |

**Expected Input Files**:
- `input/data/**/*.csv`: CSV data files
- `input/data/**/*.parquet` or `*.pq`: Parquet data files  
- `input/data/**/*.json` or `*.jsonl`: JSON/JSONL data files

### Output Paths
| Path | Location | Description |
|------|----------|-------------|
| `SIGNATURE` | `/opt/ml/processing/output/signature` | Schema signature file |
| `METADATA` | `/opt/ml/processing/output/metadata` | Metadata JSON file |
| `DATA` | `/opt/ml/processing/output/data` | Processed data output |

**Output Files**:
- `output/signature/signature`: Comma-separated column names
- `output/metadata/metadata`: JSON file with column statistics
- `output/data/part-00000.{format}`: Data file(s) in configured format

### Required Environment Variables

**None** - All environment variables are optional with sensible defaults.

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WRITE_DATA_SHARDS` | `"false"` | Enable enhanced data sharding mode (true/false) |
| `SHARD_SIZE` | `"10000"` | Number of rows per shard file |
| `OUTPUT_FORMAT` | `"CSV"` | Output format: CSV, JSON, or PARQUET |

**Mode Selection**:
- `WRITE_DATA_SHARDS=false` (default): Legacy mode - writes placeholder for backward compatibility
- `WRITE_DATA_SHARDS=true`: Enhanced mode - writes actual data as shards

**Output Format Options**:
- `CSV`: Comma-separated values (default)
- `JSON`: JSON Lines format (one object per line)
- `PARQUET`: Apache Parquet columnar format

### Job Arguments
**None** - This script does not require command-line arguments.

## Input Data Structure

### Expected Input Format
```
input/data/
├── dataset1.csv
├── dataset2.parquet
└── subfolder/
    └── dataset3.json
```

### Input Data Example

**CSV Input** (`dataset1.csv`):
```csv
customer_id,transaction_amount,currency,marketplace_id
C001,100.50,USD,1
C002,85.20,EUR,3
C003,120.00,GBP,2
```

**Parquet Input** (`dataset2.parquet`):
Binary Parquet file with same schema

**JSON Input** (`dataset3.json`):
```json
{"customer_id": "C004", "transaction_amount": 50.00, "currency": "USD", "marketplace_id": 1}
{"customer_id": "C005", "transaction_amount": 75.00, "currency": "EUR", "marketplace_id": 3}
```

### Configuration Example

**Environment Variables**:
```bash
# Legacy Mode (default)
export WRITE_DATA_SHARDS="false"
export OUTPUT_FORMAT="CSV"

# Enhanced Mode with sharding
export WRITE_DATA_SHARDS="true"
export SHARD_SIZE="10000"
export OUTPUT_FORMAT="PARQUET"
```

## Output Data Structure

### Output Directory Structure

**Legacy Mode** (WRITE_DATA_SHARDS=false):
```
output/
├── signature/
│   └── signature
├── metadata/
│   └── metadata
└── data/
    └── part-00000.csv  # Placeholder or small dataset
```

**Enhanced Mode** (WRITE_DATA_SHARDS=true, large dataset):
```
output/
├── signature/
│   └── signature
├── metadata/
│   └── metadata
└── data/
    ├── part-00000.parquet
    ├── part-00001.parquet
    ├── part-00002.parquet
    └── part-00003.parquet
```

### Output File Examples

**Signature File** (`signature`):
```
customer_id,transaction_amount,currency,marketplace_id
```

**Metadata File** (`metadata`):
```json
{
  "version": "1.0",
  "data_info": {
    "total_rows": 5,
    "total_columns": 4,
    "memory_usage_bytes": 2048
  },
  "column_info": {
    "customer_id": {
      "data_type": "object",
      "null_count": 0,
      "unique_count": 5,
      "memory_usage": 512
    },
    "transaction_amount": {
      "data_type": "float64",
      "null_count": 0,
      "unique_count": 5,
      "memory_usage": 512,
      "min": 50.0,
      "max": 120.0,
      "mean": 86.14,
      "std": 26.32
    },
    "currency": {
      "data_type": "object",
      "null_count": 0,
      "unique_count": 3,
      "memory_usage": 512
    },
    "marketplace_id": {
      "data_type": "int64",
      "null_count": 0,
      "unique_count": 3,
      "memory_usage": 512,
      "min": 1.0,
      "max": 3.0,
      "mean": 2.0,
      "std": 0.82
    }
  }
}
```

**Data File** (`part-00000.csv`):
```csv
customer_id,transaction_amount,currency,marketplace_id
C001,100.5,USD,1
C002,85.2,EUR,3
C003,120.0,GBP,2
C004,50.0,USD,1
C005,75.0,EUR,3
```

**Key Properties**:
- Signature contains all column names from combined data
- Metadata includes data type, null counts, unique counts, and statistics
- Data files use part-* naming convention for compatibility
- Multiple input files are combined into unified output

## Key Functions and Tasks

### Main Orchestration Component

#### `main(input_paths, output_paths, environ_vars, job_args)`
**Purpose**: Standardized entry point orchestrating the complete data loading workflow

**Algorithm**:
```python
1. Extract configuration from environment variables:
   - WRITE_DATA_SHARDS (boolean flag)
   - SHARD_SIZE (integer rows per shard)
   - OUTPUT_FORMAT (CSV/JSON/PARQUET)
2. Validate output format against supported formats
3. Extract input/output directory paths from parameters
4. Log configuration and paths for debugging
5. Find all data files in input directory:
   - Call find_data_files() to search recursively
   - Returns list of supported file paths
6. Process all data files:
   - Call process_data_files() to read and combine
   - Returns unified DataFrame
7. Generate schema and metadata:
   - Call generate_schema_signature() for column names
   - Call generate_metadata() for statistics
8. Write output files:
   - write_signature_file() → signature output
   - write_metadata_file() → metadata output
   - write_data_output() → data output (mode-dependent)
9. Return dictionary of output file paths
10. Handle exceptions with logging and re-raise
```

**Parameters**:
- `input_paths` (dict): Input directory paths
- `output_paths` (dict): Output directory paths
- `environ_vars` (dict): Environment variable configuration
- `job_args` (Namespace | None): Command-line arguments (unused)

**Returns**: `Dict[str, Union[Path, List[Path]]]` - Output file paths

**Raises**: `ValueError` for configuration/data errors, `Exception` for processing failures

### Data Discovery Component

#### `find_data_files(input_dir)`
**Purpose**: Recursively search directory for supported data files

**Algorithm**:
```python
1. Check if input directory exists:
   - If not exists: log error, return empty list
2. Initialize data_files empty list
3. Define supported extensions:
   - extensions = {'.csv', '.parquet', '.pq', '.json', '.jsonl'}
4. Recursively search directory with rglob('*'):
   - For each path in results:
     * Check if path is file (not directory)
     * Check if suffix.lower() in supported_extensions
     * If both true: append to data_files, log found file
5. Log total count of files found
6. Return data_files list
```

**Parameters**:
- `input_dir` (Path): Directory to search

**Returns**: `List[Path]` - List of data file paths

**Complexity**: O(n) where n = total files in directory tree

#### `detect_file_format(file_path)`
**Purpose**: Auto-detect data file format based on extension and content

**Algorithm**:
```python
1. Log detection start
2. Get file suffix: suffix = file_path.suffix.lower()
3. Check extension-based detection:
   - If suffix in ['.csv']: return 'csv'
   - If suffix in ['.parquet', '.pq']: return 'parquet'
   - If suffix in ['.json', '.jsonl']: return 'json'
4. If extension unclear, try content-based detection:
   a. Try pd.read_csv(file_path, nrows=1):
      - If succeeds: return 'csv'
   b. Try pd.read_parquet(file_path):
      - If succeeds: return 'parquet'
   c. Try pd.read_json(file_path, lines=True, nrows=1):
      - If succeeds: return 'json'
5. If all methods fail:
   - Log warning
   - Return 'unknown'
```

**Parameters**:
- `file_path` (Path): Path to data file

**Returns**: `str` - Format string: 'csv', 'parquet', 'json', or 'unknown'

**Decision Logic**:
- Priority 1: File extension (fast, reliable)
- Priority 2: Content-based reading attempts (fallback)
- Priority 3: Unknown if all methods fail

### Data Reading Component

#### `read_data_file(file_path, file_format)`
**Purpose**: Read data file based on detected format

**Algorithm**:
```python
1. Log reading start with format
2. Switch on file_format:
   - If 'csv': df = pd.read_csv(file_path)
   - If 'parquet': df = pd.read_parquet(file_path)
   - If 'json': df = pd.read_json(file_path, lines=True)
   - Else: raise ValueError(f"Unsupported format: {file_format}")
3. Log successful read with row/column counts
4. Return DataFrame
5. Catch exceptions:
   - Log error with file path and format
   - Re-raise exception
```

**Parameters**:
- `file_path` (Path): Path to data file
- `file_format` (str): Format string from detection

**Returns**: `pd.DataFrame` - Loaded data

**Raises**: `ValueError` for unsupported format, `Exception` for read failures

**Complexity**: O(n × m) where n = rows, m = columns

#### `process_data_files(data_files)`
**Purpose**: Process multiple data files and combine into single DataFrame

**Algorithm**:
```python
1. Validate input:
   - If data_files empty: raise ValueError("No data files")
2. Log processing start with file count
3. Initialize combined_df = None
4. For each file_path in data_files:
   a. Detect format: file_format = detect_file_format(file_path)
   b. If format == 'unknown':
      - Log warning
      - Continue to next file (skip)
   c. Try to read file:
      - df = read_data_file(file_path, file_format)
   d. Combine with existing data:
      - If combined_df is None: combined_df = df
      - Else: combined_df = pd.concat([combined_df, df], ignore_index=True)
   e. Log file processed with row count
   f. Catch exceptions:
      - Log error
      - Continue to next file (don't fail completely)
5. Validate result:
   - If combined_df is None: raise ValueError("No valid data processed")
6. Log final combined shape
7. Return combined_df
```

**Parameters**:
- `data_files` (List[Path]): List of data file paths

**Returns**: `pd.DataFrame` - Combined DataFrame

**Raises**: `ValueError` if no valid data

**Complexity**: O(f × n × m) where f = files, n = rows per file, m = columns

### Schema Generation Component

#### `generate_schema_signature(df)`
**Purpose**: Generate schema signature as list of column names

**Algorithm**:
```python
1. Log generation start
2. Extract column names: signature = list(df.columns)
3. Log completion with column count and names
4. Return signature list
```

**Parameters**:
- `df` (pd.DataFrame): DataFrame to analyze

**Returns**: `List[str]` - List of column names

**Complexity**: O(m) where m = columns

**Note**: Simple extraction - just column names for downstream compatibility

#### `generate_metadata(df)`
**Purpose**: Generate comprehensive metadata with column statistics

**Algorithm**:
```python
1. Log generation start
2. Initialize metadata structure:
   - version: "1.0"
   - data_info: {total_rows, total_columns, memory_usage_bytes}
   - column_info: {}
3. For each column in df.columns:
   a. Create base column info:
      - data_type: str(df[column].dtype)
      - null_count: int(df[column].isnull().sum())
      - unique_count: int(df[column].nunique())
      - memory_usage: int(df[column].memory_usage(deep=True))
   b. If column is numeric (pd.api.types.is_numeric_dtype):
      - Add min: float(df[column].min())
      - Add max: float(df[column].max())
      - Add mean: float(df[column].mean())
      - Add std: float(df[column].std())
   c. Store in metadata['column_info'][column]
4. Log completion with column count
5. Return metadata dictionary
```

**Parameters**:
- `df` (pd.DataFrame): DataFrame to analyze

**Returns**: `Dict[str, Any]` - Metadata dictionary

**Complexity**: O(n × m) where n = rows, m = columns (for statistics computation)

### Output Writing Components

#### `write_signature_file(signature, output_dir)`
**Purpose**: Write schema signature as comma-separated values

**Algorithm**:
```python
1. Ensure output directory exists
2. Define signature_file = output_dir / "signature"
3. Log write start
4. Try to write:
   - Open file in write mode
   - Write: ",".join(signature)
   - Close file
5. Log success with column count
6. Return signature_file path
7. Catch exceptions:
   - Log error
   - Re-raise exception
```

**Parameters**:
- `signature` (List[str]): Schema signature (column names)
- `output_dir` (Path): Output directory

**Returns**: `Path` - Path to written signature file

**Raises**: `Exception` on write failure

**File Format**: CSV (comma-separated, no header)

#### `write_metadata_file(metadata, output_dir)`
**Purpose**: Write metadata as JSON file

**Algorithm**:
```python
1. Ensure output directory exists
2. Define metadata_file = output_dir / "metadata"
3. Log write start
4. Try to write:
   - Open file in write mode
   - json.dump(metadata, f, indent=2)
   - Close file
5. Log success
6. Return metadata_file path
7. Catch exceptions:
   - Log error
   - Re-raise exception
```

**Parameters**:
- `metadata` (Dict[str, Any]): Metadata dictionary
- `output_dir` (Path): Output directory

**Returns**: `Path` - Path to written metadata file

**Raises**: `Exception` on write failure

**File Format**: JSON with 2-space indentation

#### `write_single_shard(df, output_dir, shard_index, output_format)`
**Purpose**: Write single data shard in specified format

**Algorithm**:
```python
1. Map format to extension:
   - format_extensions = {'CSV': 'csv', 'JSON': 'json', 'PARQUET': 'parquet'}
2. Validate format:
   - If output_format not in format_extensions: raise ValueError
3. Build shard path:
   - extension = format_extensions[output_format]
   - filename = f"part-{shard_index:05d}.{extension}"
   - shard_path = output_dir / filename
4. Log write start
5. Switch on output_format:
   - If 'CSV': df.to_csv(shard_path, index=False)
   - If 'JSON': df.to_json(shard_path, orient='records', lines=True)
   - If 'PARQUET': df.to_parquet(shard_path, index=False)
6. Log success with row count
7. Return shard_path
8. Catch exceptions:
   - Log error
   - Re-raise exception
```

**Parameters**:
- `df` (pd.DataFrame): Data to write
- `output_dir` (Path): Output directory
- `shard_index` (int): Shard number for filename
- `output_format` (str): CSV/JSON/PARQUET

**Returns**: `Path` - Path to written shard file

**Raises**: `ValueError` for unsupported format, `Exception` for write failure

**Complexity**: O(n × m) where n = rows, m = columns

#### `write_data_shards(df, output_dir, shard_size, output_format)`
**Purpose**: Write DataFrame as multiple data shards

**Algorithm**:
```python
1. Ensure output directory exists
2. Initialize written_files = []
3. Get total_rows = len(df)
4. Log sharding start with parameters
5. Check if sharding needed:
   a. If total_rows <= shard_size:
      - Single shard: write_single_shard(df, output_dir, 0, output_format)
      - Append to written_files
   b. Else (multiple shards):
      - For i in range(0, total_rows, shard_size):
        * shard_df = df.iloc[i:i+shard_size]
        * shard_index = i // shard_size
        * shard_file = write_single_shard(shard_df, output_dir, shard_index, output_format)
        * Append shard_file to written_files
6. Log completion with shard count
7. Return written_files list
```

**Parameters**:
- `df` (pd.DataFrame): DataFrame to shard
- `output_dir` (Path): Output directory
- `shard_size` (int): Rows per shard
- `output_format` (str): CSV/JSON/PARQUET

**Returns**: `List[Path]` - List of shard file paths

**Complexity**: O(n × m) where n = rows, m = columns

#### `write_data_output(df, output_dir, write_shards, shard_size, output_format)`
**Purpose**: Write data output based on mode configuration

**Algorithm**:
```python
1. Check write_shards flag:
   a. If write_shards == False (legacy mode):
      - Log single file write
      - Return write_single_data_file(df, output_dir, output_format)
   b. If write_shards == True (enhanced mode):
      - Log shard write with parameters
      - Return write_data_shards(df, output_dir, shard_size, output_format)
```

**Parameters**:
- `df` (pd.DataFrame): Processed DataFrame
- `output_dir` (Path): Output directory
- `write_shards` (bool): Mode flag
- `shard_size` (int): Rows per shard
- `output_format` (str): CSV/JSON/PARQUET

**Returns**: `Union[Path, List[Path]]` - Single file or list of shard files

**Mode Selection**:
- Legacy (write_shards=False): Single file for compatibility
- Enhanced (write_shards=True): Multiple shards for scalability

### Utility Components

#### `ensure_directory(directory)`
**Purpose**: Ensure directory exists, creating if necessary

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
- `directory` (Path): Directory to ensure

**Returns**: `bool` - Success status

**Note**: Uses exist_ok=True to avoid errors if already exists

## Algorithms and Data Structures

### Format Detection Algorithm
**Problem**: Identify data file format without explicit user specification

**Solution Strategy**:
1. Check file extension first (fast, reliable for most cases)
2. Try content-based reading if extension unclear
3. Return 'unknown' if all methods fail

**Algorithm**:
```python
def detect_format(file_path):
    # Phase 1: Extension-based detection
    suffix = file_path.suffix.lower()
    if suffix in ['.csv']: return 'csv'
    if suffix in ['.parquet', '.pq']: return 'parquet'
    if suffix in ['.json', '.jsonl']: return 'json'
    
    # Phase 2: Content-based detection
    try: pd.read_csv(file_path, nrows=1); return 'csv'
    except: pass
    
    try: pd.read_parquet(file_path); return 'parquet'
    except: pass
    
    try: pd.read_json(file_path, lines=True, nrows=1); return 'json'
    except: pass
    
    return 'unknown'
```

**Complexity**: O(1) for extension check, O(k) for content check where k = sample size

**Key Features**:
- Fast path for standard extensions
- Fallback for ambiguous cases
- Graceful degradation to 'unknown'

### Data Combination Algorithm
**Problem**: Merge multiple data files of potentially different formats into unified DataFrame

**Solution Strategy**:
1. Process files sequentially
2. Auto-detect format for each file
3. Concatenate DataFrames with ignore_index
4. Continue on individual file failures (don't fail entire pipeline)

**Algorithm**:
```python
def combine_data(files):
    combined = None
    
    for file in files:
        format = detect_format(file)
        if format == 'unknown': continue
        
        try:
            df = read_file(file, format)
            if combined is None:
                combined = df
            else:
                combined = pd.concat([combined, df], ignore_index=True)
        except Exception:
            continue  # Skip problematic files
    
    return combined
```

**Complexity**: O(f × n × m) where f = files, n = rows per file, m = columns

**Key Features**:
- Format-agnostic processing
- Resilient to individual file failures
- Index reset for unified numbering

### Sharding Strategy
**Problem**: Split large DataFrame into multiple files for efficient processing

**Solution Strategy**:
1. Determine if sharding needed based on total rows vs. shard size
2. Use iloc for efficient row slicing
3. Generate sequential shard indices
4. Use part-* naming convention for compatibility

**Data Structure**:
```python
# Shard metadata
shards = [
    {
        'index': 0,
        'start_row': 0,
        'end_row': 10000,
        'filename': 'part-00000.csv'
    },
    {
        'index': 1,
        'start_row': 10000,
        'end_row': 20000,
        'filename': 'part-00001.csv'
    }
]
```

**Algorithm**:
```python
def shard_data(df, shard_size):
    total_rows = len(df)
    
    if total_rows <= shard_size:
        return [write_shard(df, 0)]  # Single shard
    
    shards = []
    for i in range(0, total_rows, shard_size):
        shard_df = df.iloc[i:i+shard_size]
        shard_index = i // shard_size
        shard_path = write_shard(shard_df, shard_index)
        shards.append(shard_path)
    
    return shards
```

**Complexity**: O(n × m) where n = rows, m = columns

**Key Features**:
- Configurable shard size
- Sequential numbering (00000, 00001, ...)
- Compatible with distributed processing

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Find Files | O(n) | O(n) | n = files in directory tree |
| Format Detection | O(1) to O(k) | O(k) | k = sample size for content check |
| Read Single File | O(n × m) | O(n × m) | n = rows, m = columns |
| Combine Files | O(f × n × m) | O(f × n × m) | f = files |
| Generate Signature | O(m) | O(m) | m = columns |
| Generate Metadata | O(n × m) | O(m) | Statistics computation |
| Write Signature | O(m) | O(1) | Small file |
| Write Metadata | O(m) | O(1) | JSON serialization |
| Write Single Shard | O(n × m) | O(1) | Stream to file |
| Write All Shards | O(n × m) | O(s × n × m) | s = shard size |

**Overall Complexity**: O(f × n × m) where f = input files, n = rows, m = columns

**Typical Performance**:
- Small datasets (<10K rows, <50 cols): < 5 seconds
- Medium datasets (10K-100K rows, 50-200 cols): 5-30 seconds
- Large datasets (>100K rows, >200 cols): 30-180 seconds

### Memory Usage

**Peak Memory**: O(n × m) for combined DataFrame

**Components**:
- Input DataFrame: O(n × m) per file
- Combined DataFrame: O(f × n × m) worst case
- Metadata dictionary: O(m)
- Shard DataFrames: O(shard_size × m) temporary

**Optimization Opportunities**:
1. Stream processing for very large files (chunked reading)
2. Incremental sharding (write as reading progresses)
3. Format-specific optimizations (Parquet columnar reads)
4. Memory-mapped files for large Parquet files

### Sharding Benefits

**Storage Efficiency**:
- Multiple small files vs. single large file
- Better for distributed storage systems
- Easier to parallelize downstream processing

**Processing Efficiency**:
- Downstream steps can process shards in parallel
- Reduced memory footprint per worker
- Better cache locality

**Example Scaling**:
```
Dataset: 1M rows, 100 columns
Shard size: 10K rows

Sequential: 1 file, 1M rows → single reader
Sharded: 100 files, 10K rows each → 100 parallel readers

Speedup: ~100× with sufficient workers
```

## Error Handling

### Error Types

#### Configuration Errors
- **Invalid OUTPUT_FORMAT**: Format not in [CSV, JSON, PARQUET]
  - **Handling**: Raises ValueError with supported formats list
  
- **Invalid SHARD_SIZE**: Non-numeric or negative value
  - **Handling**: Raises ValueError during int() conversion

#### Data Errors
- **No Data Files Found**: Input directory empty or no supported formats
  - **Handling**: Raises ValueError after search completes
  
- **Unknown File Format**: Cannot detect format via extension or content
  - **Handling**: Logs warning, skips file, continues with others

- **Corrupt Data File**: File exists but cannot be read
  - **Handling**: Logs error, skips file, continues with others

- **Schema Mismatch**: Different columns across files
  - **Handling**: pd.concat handles missing columns (fills with NaN)

#### I/O Errors
- **Input Directory Not Found**: Specified path doesn't exist
  - **Handling**: find_data_files returns empty list, raises ValueError

- **Output Directory Creation Failed**: Permission or disk space issues
  - **Handling**: ensure_directory logs error, returns False

- **Write Failure**: Cannot write to output location
  - **Handling**: Propagates exception with detailed error message

#### Processing Errors
- **Empty DataFrame**: All files skipped or no rows after combination
  - **Handling**: Raises ValueError("No valid data could be processed")

- **Memory Error**: Dataset too large for available memory
  - **Handling**: Propagates MemoryError (user must use chunked processing or increase resources)

### Error Response Structure

When processing fails, detailed logging provides:

```python
logger.error(f"Error in dummy data loading: {str(e)}")
logger.error(traceback.format_exc())
sys.exit(1)
```

**Exit Codes**:
- 0: Success
- 1: General failure (configuration, data, or processing error)

**Resilience Features**:
- Individual file failures don't stop entire process
- Skip-and-continue strategy for robust processing
- Detailed logging for all error types

## Best Practices

### For Production Deployments
1. **Use Enhanced Mode**: Enable WRITE_DATA_SHARDS=true for large datasets
2. **Tune Shard Size**: Adjust based on downstream processing capabilities
3. **Choose Appropriate Format**: Use Parquet for large datasets (better compression, faster I/O)
4. **Monitor Logs**: Check logs for skipped files or format detection issues
5. **Validate Outputs**: Verify signature and metadata match expectations

### For Development
1. **Start with Legacy Mode**: Test with default settings first
2. **Test Multiple Formats**: Verify CSV, Parquet, and JSON compatibility
3. **Check Schema Inference**: Review generated signature for accuracy
4. **Validate Metadata**: Ensure statistics match expected data distribution
5. **Test Error Handling**: Verify behavior with missing/corrupt files

### For Performance Optimization
1. **Use Parquet Format**: 2-5× faster than CSV for large datasets
2. **Optimize Shard Size**: Balance parallelism vs. overhead (typically 10K-100K rows)
3. **Minimize File Count**: Fewer large files often better than many small files
4. **Pre-validate Data**: Check data quality before pipeline execution
5. **Use Appropriate Instance Type**: Memory-optimized for large datasets

## Example Configurations

### Basic CSV Loading (Legacy Mode)
```bash
export WRITE_DATA_SHARDS="false"
export OUTPUT_FORMAT="CSV"

python dummy_data_loading.py
```

**Use Case**: Small datasets, backward compatibility, simple testing

### Enhanced Parquet Sharding
```bash
export WRITE_DATA_SHARDS="true"
export SHARD_SIZE="50000"
export OUTPUT_FORMAT="PARQUET"

python dummy_data_loading.py
```

**Use Case**: Large datasets (>100K rows), distributed processing, production pipelines

### JSON Lines Processing
```bash
export WRITE_DATA_SHARDS="true"
export SHARD_SIZE="10000"
export OUTPUT_FORMAT="JSON"

python dummy_data_loading.py
```

**Use Case**: Log file processing, streaming data, JSON-native pipelines

### Mixed Format Loading
```bash
# Input directory contains mix of CSV, Parquet, JSON files
export WRITE_DATA_SHARDS="true"
export SHARD_SIZE="25000"
export OUTPUT_FORMAT="PARQUET"

python dummy_data_loading.py
```

**Use Case**: Data consolidation from multiple sources, format standardization

## Integration Patterns

### Upstream Integration
```
User Data Files → DummyDataLoading
   ↓
Schema Signature + Metadata + Data
```

**Input Sources**: Local files, S3 buckets, data lakes

### Downstream Integration
```
DummyDataLoading → signature + metadata + data
   ↓
TabularPreprocessing → processed_data
   ↓
FeatureEngineering → engineered_data
   ↓
TrainingStep
```

**Output Consumers**:
- **TabularPreprocessing**: Uses signature for schema validation
- **Feature Engineering**: Processes data shards in parallel
- **Training Steps**: Consumes prepared data

### Cradle Replacement Pattern

The script serves as a drop-in replacement for CradleDataLoadingStep:

```python
# Original: Cradle data loading
cradle_step = CradleDataLoadingStepBuilder(config).create_step()

# Replacement: Dummy data loading with user files
dummy_step = DummyDataLoadingStepBuilder(config).create_step(
    inputs={'user_data': user_data_source}
)

# Downstream steps work identically
preprocessing_step = TabularPreprocessingStepBuilder(config).create_step(
    inputs={'signature': dummy_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri,
            'data': dummy_step.properties.ProcessingOutputConfig.Outputs[2].S3Output.S3Uri}
)
```

### Workflow Example
1. **User Upload**: Upload CSV/Parquet/JSON files to S3
2. **Data Loading**: DummyDataLoading processes files, generates signature/metadata
3. **Preprocessing**: TabularPreprocessing validates schema, processes data
4. **Feature Engineering**: Additional transformations applied
5. **Training**: Model training with prepared data

## Troubleshooting

### No Data Files Found

**Symptom**: Script fails with "No supported data files found"

**Common Causes**:
1. **Wrong input directory**: Files not in expected location
2. **Unsupported format**: Files have unexpected extensions
3. **Empty directory**: No files uploaded to input location

**Solution**:
1. Verify input path points to correct S3 bucket/local directory
2. Check file extensions are .csv, .parquet, .pq, .json, or .jsonl
3. List directory contents to confirm files exist
4. Check file permissions (read access required)

### Format Detection Failures

**Symptom**: Files skipped with "unknown format" warnings

**Common Causes**:
1. **Ambiguous extension**: File has no extension or uncommon extension
2. **Corrupt file**: File exists but cannot be read
3. **Binary format**: Unsupported binary format (e.g., Excel)

**Solution**:
1. Rename files with standard extensions (.csv, .parquet, .json)
2. Validate file integrity (try opening manually)
3. Convert unsupported formats to CSV/Parquet/JSON
4. Check logs for specific error messages during content detection

### Schema Mismatch Across Files

**Symptom**: Combined data has unexpected columns or NaN values

**Common Causes**:
1. **Different schemas**: Files have different column sets
2. **Column name variations**: Same data, different column names
3. **Partial files**: Some files missing expected columns

**Solution**:
1. Standardize column names across all files before upload
2. Use consistent schema for all input files
3. Review metadata output to identify column inconsistencies
4. Pre-process files to ensure schema alignment

### Memory Errors

**Symptom**: Script crashes with MemoryError or OOM

**Common Causes**:
1. **Dataset too large**: Combined data exceeds available memory
2. **Too many files**: Loading all files simultaneously
3. **Inefficient format**: CSV uses more memory than Parquet

**Solution**:
1. Use chunked processing (modify script for streaming)
2. Process files in batches instead of all at once
3. Convert to Parquet format (better compression)
4. Increase instance memory or use distributed processing
5. Enable sharding to reduce per-worker memory requirements

### Shard Write Failures

**Symptom**: Enhanced mode fails to write shards

**Common Causes**:
1. **Disk space**: Insufficient storage for output
2. **Permission issues**: Cannot write to output directory
3. **Invalid format**: OUTPUT_FORMAT not in supported list

**Solution**:
1. Check available disk space in output location
2. Verify write permissions on output directory
3. Confirm OUTPUT_FORMAT is CSV, JSON, or PARQUET
4. Check logs for specific write errors
5. Ensure output directory path is correct

## References

### Related Scripts
- [`tabular_preprocessing.py`](tabular_preprocess_script.md): Downstream data preprocessing step
- [`currency_conversion.py`](currency_conversion_script.md): Currency normalization step
- [`cradle_data_loading.py`]: Internal Cradle service-based loading (replaced by this script)

### Related Documentation
- **Step Builder**: Dummy data loading step builder implementation
- **Config Class**: Configuration class for dummy data loading step
- **Contract**: [`src/cursus/steps/contracts/dummy_data_loading_contract.py`](../../src/cursus/steps/contracts/dummy_data_loading_contract.py)
- **Step Specification**: Specification defining inputs, outputs, and step behavior

### Related Design Documents
- **[Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md)**: Format detection and preservation strategy
- **[Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)**: Step builder patterns for processing steps
- **[Cradle Data Load Config Helper Design](../1_design/cradle_data_load_config_helper_design.md)**: Cradle data loading configuration design

### External References
- [Pandas DataFrame Documentation](https://pandas.pydata.org/docs/reference/frame.html): DataFrame operations reference
- [Apache Parquet Format](https://parquet.apache.org/): Parquet format specification
- [JSON Lines Format](https://jsonlines.org/): JSON Lines (JSONL) format specification
- [AWS S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html): S3 performance optimization
