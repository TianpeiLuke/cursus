---
tags:
  - code
  - scripts
  - tabular_preprocess_script
  - script_documentation
keywords:
  - tabular preprocess script
  - processing script
  - pipeline script
  - script documentation
  - streaming mode
  - memory optimization
topics:
  - pipeline scripts
  - script implementation
  - memory-efficient processing
language: python
date of note: 2026-01-13
---

# Tabular Preprocessing Module Documentation

## Overview

The `tabular_preprocess.py` script provides a robust data preprocessing framework for tabular data in SageMaker processing jobs. It handles the ingestion, cleaning, and preparation of data from multiple file formats (CSV, TSV, JSON, Parquet) including compressed files, combines data shards, normalizes column names, and creates appropriate train/test/validation splits.

**NEW**: The script now supports **two processing modes** - Batch Mode for smaller datasets that fit in memory, and True Streaming Mode for large datasets that require memory-efficient processing.

## Key Features

- **Dual Processing Modes**: Batch mode (loads full DataFrame) and True Streaming mode (incremental processing)
- **Multi-format support**: CSV, TSV, JSON, JSON Lines, Parquet
- **Automatic compressed file handling**: Gzipped files transparently processed
- **Intelligent delimiter detection**: CSV Sniffer for CSV/TSV files
- **Parallel processing**: Multi-threaded shard reading for improved performance
- **Batch concatenation**: Memory-efficient DataFrame combination
- **Direct write optimization**: Streaming mode writes directly to final files (2-3x faster)
- **Memory optimization**: Automatic dtype downcast and low-cardinality categorization
- **Stratified splitting**: Maintains class distribution in batch mode (for training jobs)
- **Random splitting**: Efficient random assignment in streaming mode
- **Column name normalization**: Handles dot notation and special characters
- **Multiple output formats**: CSV, TSV, Parquet
- **Optional label processing**: Works with or without label columns

## Processing Modes

### Batch Mode (Default)

**When to Use**:
- Datasets < 50GB that fit in memory
- Sufficient memory available (dataset size × 3 as rule of thumb)
- Need stratified splits for imbalanced classes
- Development and debugging

**How It Works**:
1. Loads ALL data shards into memory
2. Combines into single DataFrame
3. Applies transformations
4. Splits data (stratified if labels present)
5. Writes output files

**Characteristics**:
- **Memory**: High (full dataset + intermediate copies)
- **Speed**: Fast once loaded (in-memory operations)
- **Splits**: Stratified (preserves class distribution)
- **Best for**: Datasets that comfortably fit in memory

**Enable**: Default behavior when `ENABLE_TRUE_STREAMING="false"` (or not set)

### True Streaming Mode (NEW)

**When to Use**:
- Large datasets > 50GB
- Memory-constrained environments
- Very large number of shards (100+)
- Need to process datasets larger than available memory

**How It Works**:
1. Processes data in batches (never loads full dataset)
2. Reads batch of shards → Process → Write directly to final files
3. Repeats for all batches
4. Uses direct write (no temporary files, no consolidation step)
5. Memory usage stays constant regardless of total data size

**Characteristics**:
- **Memory**: Fixed ~2-5GB per batch (independent of dataset size)
- **Speed**: Slower per-batch but enables processing of unlimited data
- **Splits**: Random assignment (fast, approximates stratified for large data)
- **Best for**: Very large datasets or memory-constrained instances

**Enable**: Set `ENABLE_TRUE_STREAMING="true"`

**Performance**:
- **Direct Write Optimization**: 2-3x faster than old consolidation approach
- **No temp files**: Writes directly to final output files
- **Memory-efficient**: Can process TB-scale data on small instances

### Mode Comparison

| Feature | Batch Mode | Streaming Mode |
|---------|------------|----------------|
| Memory Usage | High (full dataset) | Fixed (~2-5GB) |
| Speed | Fast (in-memory) | Moderate (batch I/O) |
| Max Data Size | Limited by RAM | Unlimited |
| Split Strategy | Stratified | Random |
| Best Use Case | <50GB datasets | >50GB datasets |
| Complexity | Simple | More complex |

## Core Components

### Shared Utility Functions (Used by Both Modes)

#### Memory and Data Optimization
- **optimize_dtypes(df, log_func)**: Optimizes DataFrame memory usage
  - Downcasts int64→int32, float64→float32
  - Converts low-cardinality objects to category
  - Typical reduction: 30-50% memory usage

#### File and Schema Handling
- **load_signature_columns(signature_path)**: Loads column names from signature file
  - Enables consistent column naming across CSV/TSV files
  - Returns list of column names or None if no signature found

- **_is_gzipped(path)**: Detects if a file is gzipped based on extension

- **_detect_separator_from_sample(sample_lines)**: Infers CSV/TSV delimiter using csv.Sniffer

- **peek_json_format(file_path, open_func)**: Determines if JSON is regular or lines format

- **_read_json_file(file_path)**: Reads JSON files of either format into DataFrame

- **_read_file_to_df(file_path, column_names)**: Universal file reader for all supported formats
  - Handles CSV, TSV, JSON, Parquet
  - Supports gzipped versions automatically
  - Uses signature columns if provided

#### Data Processing
- **process_label_column(df, label_field, log_func)**: Processes and validates label column
  - Converts categorical labels to numeric
  - Handles missing values
  - Used by both batch and streaming modes

### Batch Mode Components

#### Shard Reading and Combination
- **_read_shard_wrapper(args)**: Wrapper for parallel shard reading using multiprocessing
  - Enables parallel processing across multiple workers
  - Logs progress per shard

- **_batch_concat_dataframes(dfs, batch_size)**: Hierarchical DataFrame concatenation
  - Minimizes memory copies (O(n log n) instead of O(n²))
  - Avoids PyArrow 2GB column limit

- **_combine_shards_streaming(shard_args, max_workers, concat_batch_size, streaming_batch_size)**: 
  - Incremental shard loading for batch mode
  - Processes shards in batches to reduce memory peaks
  - Note: Still accumulates full DataFrame by end

- **combine_shards(input_dir, signature_columns, max_workers, batch_size, streaming_batch_size)**:
  - Main shard combination function for batch mode
  - Coordinates parallel reading and batch concatenation
  - Returns single unified DataFrame

#### Batch Mode Orchestration
- **process_batch_mode_preprocessing(input_data_dir, output_dir, signature_columns, job_type, ...)**:
  - Main orchestrator for batch mode processing
  - Loads full DataFrame into memory
  - Applies stratified splits (for training jobs)
  - Writes output files

### Streaming Mode Components (NEW)

#### Streaming Mode Orchestration
- **process_streaming_mode_preprocessing(input_dir, output_dir, signature_columns, job_type, ...)**:
  - Main orchestrator for true streaming mode
  - Never loads full DataFrame into memory
  - Routes to training or single-split processing

#### Shard Discovery and Batch Processing
- **find_input_shards(input_dir, log_func)**: Discovers all input shards
  - Finds part-* files with supported extensions
  - Returns sorted list of shard paths

- **process_single_batch(shard_files, signature_columns, batch_size, optimize_memory, label_field, log_func)**:
  - Processes a single batch of shards
  - Applies memory optimization if enabled
  - Processes labels if provided
  - Returns processed DataFrame for the batch

#### Split Assignment
- **assign_random_splits(df, train_ratio, test_val_ratio)**: 
  - Randomly assigns rows to train/test/val splits
  - Fast alternative to stratified splits for large data
  - Adds '_split' column to DataFrame

#### Training Splits Processing (Direct Write)
- **process_training_splits_streaming(all_shards, output_path, signature_columns, ...)**:
  - Processes training data in streaming mode
  - Uses direct write (no temp files)
  - Assigns random splits per batch
  - 2-3x faster than consolidation approach

#### Single Split Processing (Direct Write)
- **process_single_split_streaming(all_shards, output_path, job_type, signature_columns, ...)**:
  - Processes validation/testing/calibration data in streaming mode
  - Uses direct write optimization
  - No split assignment needed

#### Direct Write Functions - CSV/TSV (Training)
- **_init_csv_writers_training(output_path, output_format, log_func)**: 
  - Opens file handles for train/test/val CSV/TSV files
  - Handles remain open across batches

- **_write_splits_to_csv(batch_df, writers, first_batch, output_format, log_func)**:
  - Appends batch data directly to open file handles
  - Writes header only on first batch

- **_close_csv_writers(writers, log_func)**:
  - Closes file handles and logs final sizes

#### Direct Write Functions - Parquet (Training)
- **_init_parquet_writers_training(output_path, log_func)**:
  - Initializes PyArrow Parquet writers
  - Schema captured on first batch

- **_write_splits_to_parquet(batch_df, writers, first_batch, log_func)**:
  - Streams data to Parquet files using PyArrow
  - Incremental writing without loading full dataset

- **_close_parquet_writers(writers, log_func)**:
  - Closes Parquet writers and logs final sizes

#### Direct Write Functions - Single Split
- **_init_csv_writer_single_split(split_dir, split_name, output_format, log_func)**: 
  - Opens file handle for single CSV/TSV output

- **_write_to_csv_single(batch_df, writer, first_batch, output_format, log_func)**:
  - Appends to single CSV/TSV file

- **_close_csv_writer_single(writer, log_func)**:
  - Closes single CSV/TSV writer

- **_init_parquet_writer_single_split(split_dir, split_name, log_func)**:
  - Initializes single Parquet writer

- **_write_to_parquet_single(batch_df, writer, first_batch, log_func)**:
  - Streams to single Parquet file

- **_close_parquet_writer_single(writer, log_func)**:
  - Closes single Parquet writer

#### Legacy Functions (Deprecated)
- **write_single_shard(df, output_dir, shard_number, output_format)**: 
  - Writes single shard to file
  - Used in old consolidation approach (no longer used with direct write)

- **write_splits_to_shards(df, output_base, split_counters, shard_size, output_format, log_func)**:
  - Writes splits to temporary shards
  - No longer used with direct write optimization

- **consolidate_shards_to_single_files(output_path, job_type, output_format, log_func)**:
  - Consolidates temp shards into final files
  - No longer used with direct write optimization

### Main Entry Point

- **main(input_paths, output_paths, environ_vars, job_args, logger)**:
  1. Loads configuration from environment variables
  2. Determines processing mode (batch vs streaming)
  3. Routes to appropriate mode handler:
     - **Batch Mode**: `process_batch_mode_preprocessing()`
     - **Streaming Mode**: `process_streaming_mode_preprocessing()`
  4. Returns dictionary of DataFrames (batch mode) or empty dict (streaming mode)

## Performance Optimizations

### Parallel Shard Reading

The script uses Python's `multiprocessing.Pool` to read multiple data shards in parallel:

- **Automatic Worker Detection**: By default, uses `min(cpu_count(), num_shards)` workers
- **Parallel Execution**: Multiple shards are read simultaneously, reducing I/O wait time
- **Progress Logging**: Each worker logs progress as shards are processed
- **Expected Speedup**: 2-4x faster for typical workloads with multiple shards

### Batch Concatenation

Instead of iteratively concatenating DataFrames (O(n²) complexity), the script uses a hierarchical batch approach:

- **Batch Size**: Concatenates DataFrames in batches of 10 (default)
- **Hierarchical Processing**: Repeatedly batches results until a single DataFrame remains
- **Memory Efficiency**: Minimizes intermediate copies and memory spikes
- **PyArrow Compatibility**: Avoids the 2GB column limit error through efficient batching

### Performance Characteristics

| Metric | Sequential (Old) | Parallel (New) | Improvement |
|--------|-----------------|----------------|-------------|
| Shard Reading | Sequential | Parallel (N workers) | 2-4x faster |
| Concatenation | O(n²) iterations | O(n log n) batches | 3-5x faster |
| Memory Usage | High spikes | Controlled batching | 40-60% reduction |
| Large Datasets | Slow/crashes | Efficient/stable | Handles 10x+ data |

### When to Expect Maximum Benefits

- **Many Shards**: 10+ data shards benefit most from parallel reading
- **Large Shards**: Individual shards >100MB each show significant improvement
- **Multi-core Systems**: 4+ CPU cores allow full parallelization
- **I/O Bound**: Network/distributed storage benefits from concurrent reads

## Workflow Details

### 1. Data Shards Combination (Optimized)

The script locates and reads all data shards in the input directory matching patterns like:
- `part-*.csv`
- `part-*.csv.gz`
- `part-*.json`
- `part-*.json.gz`
- `part-*.parquet`
- `part-*.snappy.parquet`
- `part-*.parquet.gz`

**Parallel Processing Flow:**
1. Identifies all matching shard files in the input directory
2. Determines optimal number of workers (default: `min(cpu_count(), num_shards)`)
3. Distributes shards across worker processes using `multiprocessing.Pool`
4. Each worker reads and parses its assigned shards independently
5. Collects all DataFrames and combines using hierarchical batch concatenation
6. Returns a single unified DataFrame ready for downstream processing

**Logging Output:**
- Reports total number of shards found
- Shows number of parallel workers being used
- Logs progress as each shard is processed (with row counts)
- Reports total rows loaded before concatenation
- Confirms final combined shape

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

Saves each split in the configured output format to the appropriate output directory. The output structure depends on the processing mode and CONSOLIDATE_SHARDS setting.

#### Batch Mode Output (Default)

Always creates single consolidated files per split:

**Output Formats:**
- **CSV** (default): `/opt/ml/processing/output/{split_name}/{split_name}_processed_data.csv`
- **TSV**: `/opt/ml/processing/output/{split_name}/{split_name}_processed_data.tsv`
- **Parquet**: `/opt/ml/processing/output/{split_name}/{split_name}_processed_data.parquet`

#### Streaming Mode Output

Output structure controlled by `CONSOLIDATE_SHARDS` environment variable:

**Consolidated Output (CONSOLIDATE_SHARDS="true", default)**:
- Single file per split (same as batch mode)
- **CSV**: `/opt/ml/processing/output/{split_name}/{split_name}_processed_data.csv`
- **TSV**: `/opt/ml/processing/output/{split_name}/{split_name}_processed_data.tsv`
- **Parquet**: `/opt/ml/processing/output/{split_name}/{split_name}_processed_data.parquet`
- **Recommended** for most use cases

**Sharded Output (CONSOLIDATE_SHARDS="false")**:
- Multiple shard files per split
- **CSV**: `/opt/ml/processing/output/{split_name}/part-00000.csv`, `part-00001.csv`, ...
- **TSV**: `/opt/ml/processing/output/{split_name}/part-00000.tsv`, `part-00001.tsv`, ...
- **Parquet**: `/opt/ml/processing/output/{split_name}/part-00000.parquet`, `part-00001.parquet`, ...
- Use only when downstream consumers explicitly require sharded input

**Configuration:**
- The output format is controlled by the `OUTPUT_FORMAT` environment variable
- The output structure (consolidated vs sharded) is controlled by `CONSOLIDATE_SHARDS` (streaming mode only)
- If an invalid format is specified, the script defaults to CSV with a warning message

## Configuration

### Command Line Arguments

- **--job_type**: Type of job to perform (one of 'training', 'validation', 'testing')

### Environment Variables

#### Basic Configuration

- **LABEL_FIELD**: Name of the target/label column (optional - if not provided, label processing is skipped)
- **TRAIN_RATIO**: Proportion of data for training (default: 0.7)
- **TEST_VAL_RATIO**: Test/validation split ratio (default: 0.5)
- **OUTPUT_FORMAT**: Output file format - 'csv', 'tsv', or 'parquet' (default: 'csv')

#### Memory Optimization (Batch Mode)

- **MAX_WORKERS**: Number of parallel workers for shard reading (default: `auto` - uses `min(cpu_count(), num_shards)`)
  - Set to `0` or omit for automatic detection
  - Set to specific number (e.g., `4`) to limit parallelism
- **BATCH_SIZE**: Number of DataFrames to concatenate at once (default: `5`)
  - Lower values: Less memory usage but slightly slower
  - Higher values: Faster but more memory peaks
  - Recommended range: 5-10
- **OPTIMIZE_MEMORY**: Enable dtype optimization and categorization (default: `"true"`)
  - Set to `"false"` to disable memory optimization
  - When enabled: 30-50% memory reduction typical
- **STREAMING_BATCH_SIZE**: Enable incremental loading in batch mode (default: `0` - disabled)
  - Set to number of shards to process per batch (e.g., `10`)
  - Loads and processes shards in batches instead of all at once
  - Useful for very large datasets in batch mode
  - Note: Still loads full dataset by end (not true streaming)

#### True Streaming Mode (NEW)

- **ENABLE_TRUE_STREAMING**: Enable true streaming mode (default: `"false"`)
  - Set to `"true"` to enable
  - Never loads full DataFrame into memory
  - Processes data in batches, writes directly to final files
  - **Most important setting for large datasets**
- **CONSOLIDATE_SHARDS**: Control output file format in streaming mode (default: `"true"`)
  - Set to `"true"` for single file per split (e.g., `train_processed_data.csv`)
  - Set to `"false"` for multiple shard files per split (e.g., `part-00000.csv`, `part-00001.csv`, ...)
  - **Only applies when ENABLE_TRUE_STREAMING=true**
  - **Recommended**: Use `"true"` (default) for compatibility with downstream steps
  - Use `"false"` only when downstream consumers explicitly require sharded output
- **SHARD_SIZE**: Rows per output shard in streaming mode (default: `100000`)
  - Only used when ENABLE_TRUE_STREAMING=true AND CONSOLIDATE_SHARDS=false
  - Controls granularity of output sharding

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

### Using Alternative Output Formats

**Parquet Output (recommended for large datasets):**
```bash
# Set up environment variables for Parquet output
export LABEL_FIELD="fraud_flag"
export TRAIN_RATIO="0.8"
export TEST_VAL_RATIO="0.5"
export OUTPUT_FORMAT="parquet"

# Run preprocessing script
python tabular_preprocess.py --job_type training
```

This creates:
- `/opt/ml/processing/output/train/train_processed_data.parquet`
- `/opt/ml/processing/output/test/test_processed_data.parquet`
- `/opt/ml/processing/output/val/val_processed_data.parquet`

**TSV Output:**
```bash
# Set up environment variables for TSV output
export LABEL_FIELD="fraud_flag"
export OUTPUT_FORMAT="tsv"

# Run preprocessing script
python tabular_preprocess.py --job_type validation
```

This creates:
- `/opt/ml/processing/output/validation/validation_processed_data.tsv`

### Processing Without Labels (Feature Engineering Only)

```bash
# No LABEL_FIELD specified - skips label processing
export OUTPUT_FORMAT="parquet"

# Run preprocessing script
python tabular_preprocess.py --job_type validation
```

This is useful for:
- Feature engineering pipelines without supervised learning
- Unsupervised learning workflows
- Data transformation steps before labeling

### True Streaming Mode Examples (NEW)

**Large Dataset Training (>50GB):**
```bash
# Enable true streaming mode for memory-efficient processing
export LABEL_FIELD="fraud_flag"
export TRAIN_RATIO="0.8"
export TEST_VAL_RATIO="0.5"
export ENABLE_TRUE_STREAMING="true"
export STREAMING_BATCH_SIZE="10"    # Process 10 shards at a time
export OUTPUT_FORMAT="parquet"       # Recommended for streaming

# Run preprocessing script
python tabular_preprocess.py --job_type training
```

**Key Benefits**:
- Memory usage: Fixed at ~2-5GB regardless of dataset size
- Can process TB-scale datasets on ml.m5.xlarge instances
- Direct write: 2-3x faster than old consolidation approach
- No temporary files created

**Memory-Constrained Environment:**
```bash
# Minimal memory footprint configuration
export LABEL_FIELD="target"
export ENABLE_TRUE_STREAMING="true"
export STREAMING_BATCH_SIZE="5"      # Smaller batches for less memory
export OPTIMIZE_MEMORY="true"        # Enable dtype optimization
export OUTPUT_FORMAT="parquet"

python tabular_preprocess.py --job_type training
```

**Memory Usage Comparison**:
| Configuration | Memory Usage | Max Dataset Size |
|--------------|--------------|------------------|
| Batch Mode (default) | Dataset × 3 | ~30GB on 128GB instance |
| Batch + STREAMING_BATCH_SIZE | Dataset × 1.5 | ~60GB on 128GB instance |
| True Streaming Mode | Fixed 2-5GB | Unlimited |

### Batch Mode with Memory Optimization

**Moderate Dataset (10-50GB):**
```bash
# Use batch mode with memory optimizations
export LABEL_FIELD="target"
export TRAIN_RATIO="0.8"
export OPTIMIZE_MEMORY="true"        # Reduce memory by 30-50%
export STREAMING_BATCH_SIZE="15"     # Load in batches
export MAX_WORKERS="8"               # Parallel processing
export BATCH_SIZE="5"                # Concatenation batch size
export OUTPUT_FORMAT="parquet"

python tabular_preprocess.py --job_type training
```

**Benefits**:
- Stratified splits (maintains class distribution)
- 30-50% memory reduction from dtype optimization
- Incremental loading reduces peak memory
- Good balance between speed and memory

### Instance Type Recommendations

| Dataset Size | Mode | Instance Type | Memory | vCPUs |
|-------------|------|---------------|--------|-------|
| <10GB | Batch | ml.m5.xlarge | 16GB | 4 |
| 10-30GB | Batch + Opt | ml.m5.2xlarge | 32GB | 8 |
| 30-50GB | Batch + Streaming Batch | ml.m5.4xlarge | 64GB | 16 |
| 50-200GB | True Streaming | ml.m5.2xlarge | 32GB | 8 |
| >200GB | True Streaming | ml.m5.xlarge | 16GB | 4 |

**Note**: True streaming mode enables processing of datasets much larger than instance memory, making it cost-effective for very large datasets.

## Error Handling

The script includes robust error handling for:

- Missing input directories
- No data shards found
- Unsupported file formats
- Missing or invalid label field
- JSON parsing errors
- Data combination failures

## Troubleshooting Memory Issues

### Problem: "Please use an instance type with more memory"

**Error Message**:
```
ClientError: Please use an instance type with more memory, or reduce the size of job data processed on an instance.
```

This error occurs when the preprocessing job runs out of memory. Here are solutions ordered from quickest to most comprehensive:

#### Solution 1: Enable True Streaming Mode (RECOMMENDED)

**Fastest fix for large datasets**:
```bash
export ENABLE_TRUE_STREAMING="true"
export STREAMING_BATCH_SIZE="10"
export OUTPUT_FORMAT="parquet"
```

**Why it works**:
- Memory usage fixed at ~2-5GB regardless of data size
- Can process unlimited dataset sizes
- No need to upgrade instance type
- 2-3x faster than old consolidation approach

**When to use**: Dataset >50GB or memory errors in batch mode

#### Solution 2: Enable Memory Optimizations (Batch Mode)

**For moderate datasets (10-50GB)**:
```bash
export OPTIMIZE_MEMORY="true"
export STREAMING_BATCH_SIZE="10"
export BATCH_SIZE="5"
```

**Why it works**:
- Reduces memory by 30-50% through dtype optimization
- Incremental loading prevents loading full dataset at once
- Maintains stratified splits

**When to use**: Dataset 10-50GB with imbalanced classes needing stratification

#### Solution 3: Upgrade Instance Type

**Instance upgrade path**:
```
ml.m5.large (8GB)
  ↓
ml.m5.xlarge (16GB) ← Start here for most cases
  ↓
ml.m5.2xlarge (32GB)
  ↓
ml.m5.4xlarge (64GB)
  ↓
ml.m5.8xlarge (128GB)
```

**Cost-effective approach**:
- Use **True Streaming Mode** with ml.m5.xlarge instead of upgrading
- Only upgrade if you need stratified splits and can't use streaming

### Memory Usage Estimation

**Rule of Thumb**:
```
Required Memory = Dataset Size × Memory Multiplier
```

| Mode | Memory Multiplier | Example (100GB dataset) |
|------|------------------|------------------------|
| Batch (no optimization) | 3.0x | 300GB RAM needed |
| Batch + OPTIMIZE_MEMORY | 2.0x | 200GB RAM needed |
| Batch + STREAMING_BATCH_SIZE | 1.5x | 150GB RAM needed |
| True Streaming | Fixed | 2-5GB RAM (any size) |

### Symptom: Job Runs Slowly Then Crashes

**Diagnosis**: Memory exhaustion during concatenation phase

**Solution**:
```bash
# Reduce concatenation batch size
export BATCH_SIZE="3"  # Default is 5
export OPTIMIZE_MEMORY="true"
```

Or switch to streaming mode:
```bash
export ENABLE_TRUE_STREAMING="true"
```

### Symptom: Out of Memory During Shard Reading

**Diagnosis**: Too many parallel workers loading shards simultaneously

**Solution**:
```bash
# Limit parallel workers
export MAX_WORKERS="2"  # Reduce from auto-detected
export BATCH_SIZE="3"
```

### Symptom: Memory Issues with Parquet Files

**Diagnosis**: PyArrow memory allocation issues

**Solution**:
```bash
# Use streaming mode with smaller batches
export ENABLE_TRUE_STREAMING="true"
export STREAMING_BATCH_SIZE="5"  # Reduce from 10
export OUTPUT_FORMAT="parquet"   # Still use parquet for output
```

### Memory Monitoring Commands

**Check instance memory**:
```bash
# Available memory
free -h

# Monitor memory during job
watch -n 1 free -h

# Process memory usage
top -o %MEM
```

### Decision Tree: Which Mode to Use?

```
Dataset > 50GB?
├─ YES → Use True Streaming Mode (ENABLE_TRUE_STREAMING=true)
└─ NO → Dataset 10-50GB?
    ├─ YES → Need stratified splits?
    │   ├─ YES → Batch + Optimizations (OPTIMIZE_MEMORY=true, STREAMING_BATCH_SIZE=10)
    │   └─ NO → True Streaming Mode (faster)
    └─ NO → Use Batch Mode (default)
```

### Quick Reference: Memory Reduction Strategies

| Strategy | Memory Reduction | Maintains Stratification | Complexity |
|----------|-----------------|-------------------------|------------|
| ENABLE_TRUE_STREAMING | 90-95% | No (random splits) | Low |
| OPTIMIZE_MEMORY | 30-50% | Yes | Low |
| STREAMING_BATCH_SIZE | 40-60% | Yes | Low |
| Upgrade Instance | N/A | Yes | Low (cost) |
| Reduce MAX_WORKERS | 20-30% | Yes | Low |

## Best Practices

### Data Quality

1. **Consistent Label Field**: Ensure the LABEL_FIELD environment variable matches the column name in your data

2. **Format Compatibility**: The script is designed to work with most common data formats, but if using custom formats, ensure they're compatible with Pandas readers

3. **Stratification**: For imbalanced datasets, the script performs stratified splitting to maintain class distribution across train/test/val splits

4. **Column Naming**: If your data uses dot notation in column names, the script properly handles transformation from `__DOT__` to `.`

### Performance Optimization

5. **Output Format Selection**: 
   - Use **Parquet** for large datasets (>1GB) - faster I/O and better compression
   - Use **CSV** for debugging and human readability
   - Use **TSV** for compatibility with systems that expect tab-delimited files

6. **Instance Type Selection**:
   - For datasets with many shards (50+): Use instances with 8+ vCPUs to maximize parallel processing
   - For large individual shards (>500MB each): Use memory-optimized instances (r5, r6i series)
   - Typical recommended instance: ml.m5.2xlarge or ml.m5.4xlarge

7. **Data Sharding Strategy**:
   - Optimal shard size: 100-500MB per shard for best parallel performance
   - Too many small shards (<10MB): Overhead from parallel processing may outweigh benefits
   - Too few large shards (>2GB): Limited parallelization, memory pressure

8. **Monitoring Performance**:
   - Check logs for parallel worker count - should match available CPU cores
   - Monitor memory usage during concatenation phase
   - Track total processing time to measure optimization impact

9. **Signature Files**: 
   - Provide signature files for CSV/TSV inputs when column names are consistent
   - Reduces header parsing overhead and ensures consistent column naming

## Integration in the Pipeline

This script is typically:

1. The first step in a model development pipeline
2. Followed by feature preprocessing (e.g., risk table mapping)
3. Used before model training to prepare train/test/val splits
4. Used in validation/testing workflows to prepare evaluation data
