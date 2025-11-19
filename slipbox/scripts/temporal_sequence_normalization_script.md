---
tags:
  - code
  - processing_script
  - temporal_sequences
  - sequence_normalization
  - time_series_processing
keywords:
  - temporal sequence normalization
  - sequence padding
  - time delta computation
  - sequence validation
  - attention masks
  - multi-sequence processing
  - temporal feature engineering
  - fraud detection sequences
topics:
  - temporal data processing
  - sequence normalization
  - time series ML
  - fraud detection
language: python
date of note: 2025-11-18
---

# Temporal Sequence Normalization Script Documentation

## Overview

The `temporal_sequence_normalization.py` script normalizes temporal sequence data for machine learning models by transforming variable-length sequences into fixed-length representations suitable for deep learning architectures. The script provides five configurable processing operations: sequence ordering, data validation, missing value handling, time delta computation, and sequence padding/truncation.

The script is specifically designed for fraud detection temporal sequences where customer and credit card transaction histories need to be normalized for neural network consumption. It supports multi-format data loading (CSV/TSV/JSON/Parquet), dual-entity sequence processing (e.g., customer ID + credit card ID), and generates attention masks to indicate valid sequence positions for transformer-based models.

Key capabilities:
- **Multi-format loading**: Automatic format detection and loading from CSV, TSV, JSON, Parquet
- **Sequence ordering**: Temporal sorting with duplicate handling
- **Data validation**: Configurable strict/lenient validation strategies
- **Missing value handling**: Standardized missing value normalization
- **Time delta computation**: Relative temporal relationship encoding
- **Sequence normalization**: Padding/truncation to fixed length with attention masks
- **Multi-sequence support**: Dual-entity processing for complex fraud patterns
- **Memory efficiency**: Optional chunked processing for large datasets

## Purpose and Major Tasks

### Primary Purpose
Transform variable-length temporal sequences into fixed-length normalized representations suitable for machine learning models, with configurable processing operations for sequence ordering, validation, missing value handling, time delta computation, and padding/truncation.

### Major Tasks

1. **Data Loading**: Load temporal sequence data from multiple formats with signature column support

2. **Sequence Field Detection**: Automatically detect categorical and numerical sequence fields based on naming patterns

3. **Sequence Ordering**: Sort sequences by temporal field and handle duplicate records

4. **Data Validation**: Validate sequence integrity using configurable strict or lenient strategies

5. **Missing Value Handling**: Standardize missing value indicators across all sequences

6. **Sequence Parsing**: Parse string-encoded sequences into numpy arrays with proper encoding

7. **Time Delta Computation**: Compute relative time deltas with configurable capping

8. **Sequence Normalization**: Apply padding/truncation to achieve target sequence length

9. **Attention Mask Generation**: Create attention masks indicating valid sequence positions

10. **Format Preservation**: Save normalized sequences in specified output format with metadata

## Script Contract

### Entry Point
```
temporal_sequence_normalization.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `DATA` | `/opt/ml/processing/input/data` | Temporal sequence data in CSV/TSV/JSON/Parquet format |
| `SIGNATURE` | `/opt/ml/processing/input/signature` | Optional column signature file |

**Input Structure**:
```
/opt/ml/processing/input/data/
├── part-00000.csv (or .tsv, .json, .parquet)
├── part-00001.csv
└── ... (sharded data files)

/opt/ml/processing/input/signature/
└── signature.txt (comma-separated column names)
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `normalized_sequences` | `/opt/ml/processing/output` | Normalized sequences with metadata |

**Output Structure**:
```
/opt/ml/processing/output/
├── categorical.npy (or .parquet, .csv based on format)
├── numerical.npy
├── categorical_attention_mask.npy (if enabled)
├── numerical_attention_mask.npy (if enabled)
└── metadata.json
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SEQUENCE_LENGTH` | Target sequence length for normalization | `"51"` |
| `SEQUENCE_SEPARATOR` | Separator used in sequence string encoding | `"~"` |
| `TEMPORAL_FIELD` | Column name containing timestamps | `"orderDate"` |
| `SEQUENCE_GROUPING_FIELD` | Column for grouping records into sequences | `"customerId"` |
| `RECORD_ID_FIELD` | Column uniquely identifying records | `"objectId"` |

### Optional Environment Variables

#### Core Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `MISSING_INDICATORS` | `["", "My Text String", null]` | JSON array of missing value indicators |
| `TIME_DELTA_MAX_SECONDS` | `"10000000"` | Maximum time delta cap in seconds |
| `PADDING_STRATEGY` | `"pre"` | Padding position: `pre` or `post` |
| `TRUNCATION_STRATEGY` | `"post"` | Truncation position: `pre` or `post` |

#### Multi-Sequence Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MULTI_SEQUENCE` | `"false"` | Enable dual-entity sequence processing |
| `SECONDARY_ENTITY_FIELD` | `"creditCardId"` | Secondary entity field for dual sequences |
| `SEQUENCE_NAMING_PATTERN` | `"*_seq_by_{entity}.*"` | Pattern for sequence field detection |

#### Processing Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `VALIDATION_STRATEGY` | `"strict"` | Validation mode: `strict` or `lenient` |
| `OUTPUT_FORMAT` | `"numpy"` | Output format: `numpy`, `parquet`, or `csv` |
| `INCLUDE_ATTENTION_MASKS` | `"true"` | Generate attention masks for sequences |
| `ENABLE_DISTRIBUTED_PROCESSING` | `"false"` | Enable chunked processing |
| `CHUNK_SIZE` | `"10000"` | Chunk size for distributed processing |
| `MAX_WORKERS` | `"auto"` | Number of parallel workers |

### Job Arguments

| Argument | Type | Required | Description | Choices |
|----------|------|----------|-------------|---------|
| `--job_type` | `str` | Yes | Processing mode for data splits | `training`, `validation`, `testing`, `calibration` |

### Framework Dependencies

- **pandas** >= 1.3.0 (DataFrame operations, data loading)
- **numpy** >= 1.21.0 (Array operations, sequence processing)
- **scikit-learn** >= 1.0.0 (LabelEncoder for categorical encoding)

## Input Data Structure

### Expected Input Format

**Standard Sequence Data**:
```
/opt/ml/processing/input/data/
├── part-00000.csv
├── part-00001.csv
└── _SUCCESS
```

**With Signature**:
```
/opt/ml/processing/input/signature/
└── columns.txt  # Comma-separated: "customerId,orderDate,amount_seq_by_customerId,..."
```

### Required Columns

- **Record ID**: Specified by `RECORD_ID_FIELD` (default: `objectId`)
  - Uniquely identifies each record
  - Used for duplicate handling

- **Temporal Field**: Specified by `TEMPORAL_FIELD` (default: `orderDate`)
  - Contains timestamps for ordering
  - Converted to numeric for sorting

- **Entity ID**: Specified by `SEQUENCE_GROUPING_FIELD` (default: `customerId`)
  - Groups records into sequences
  - Primary entity for single-sequence mode

### Sequence Field Format

Sequences are stored as string-encoded values separated by `SEQUENCE_SEPARATOR`:

**Categorical Sequences**:
```csv
id,category_seq_by_customerId
1,"Electronics~Books~Clothing~Electronics"
2,"Grocery~Grocery~Health"
```

**Numerical Sequences**:
```csv
id,amount_seq_by_customerId
1,"100.50~200.00~150.75~300.25"
2,"50.00~55.50~60.00"
```

### Multi-Sequence Format

When `ENABLE_MULTI_SEQUENCE=true`:
```csv
customerId,creditCardId,amount_seq_by_customerId,amount_seq_by_creditCardId
C1,CC1,"100~200~150","100~200"
C2,CC2,"50~55~60","50~55~60~45"
```

### Supported File Formats

1. **CSV**: Comma-separated values (`.csv`, `.csv.gz`)
2. **TSV**: Tab-separated values (`.tsv`, `.tsv.gz`)
3. **JSON**: JSON Lines or regular JSON (`.json`, `.json.gz`)
4. **Parquet**: Apache Parquet (`.parquet`, `.snappy.parquet`, `.parquet.gz`)

## Output Data Structure

### Output Directory Structure

**NumPy Format** (default):
```
/opt/ml/processing/output/
├── categorical.npy           # Shape: (batch_size, seq_len, num_cat_features)
├── numerical.npy             # Shape: (batch_size, seq_len, num_num_features + 1)
├── categorical_attention_mask.npy  # Shape: (batch_size, seq_len)
├── numerical_attention_mask.npy    # Shape: (batch_size, seq_len)
└── metadata.json
```

**Parquet Format**:
```
/opt/ml/processing/output/
├── categorical.parquet       # Flattened 3D array to 2D
├── numerical.parquet
├── categorical_attention_mask.parquet
├── numerical_attention_mask.parquet
└── metadata.json
```

### Array Shapes

**Categorical Sequences**:
- Shape: `(batch_size, sequence_length, num_categorical_features)`
- Dtype: `int64` (label-encoded categories)
- Example: `(10000, 51, 3)` for 10K samples, 51-length sequences, 3 categorical fields

**Numerical Sequences**:
- Shape: `(batch_size, sequence_length, num_numerical_features + 1)`
- Dtype: `float64`
- Last dimension: Padding indicator column (1.0 = valid, 0.0 = padding)
- Example: `(10000, 51, 5)` for 10K samples, 51-length sequences, 4 numerical fields + 1 padding indicator

**Attention Masks**:
- Shape: `(batch_size, sequence_length)`
- Dtype: `int8`
- Values: 1 = valid position, 0 = padded position
- Example: `(10000, 51)` for 10K samples with 51-length sequences

### Metadata Output

**metadata.json**:
```json
{
  "sequence_length": 51,
  "sequence_separator": "~",
  "temporal_field": "orderDate",
  "entity_id_field": "customerId",
  "id_field": "objectId",
  "output_format": "numpy",
  "include_attention_masks": true,
  "shapes": {
    "categorical": [10000, 51, 3],
    "numerical": [10000, 51, 5],
    "categorical_attention_mask": [10000, 51],
    "numerical_attention_mask": [10000, 51]
  }
}
```

## Key Functions and Tasks

### Data Loading Component

#### `combine_shards(input_dir, signature_columns) -> pd.DataFrame`

**Purpose**: Detect and combine all data shards from input directory

**Algorithm**:
```python
1. Validate input directory exists
2. Scan for files matching patterns:
   - part-*.{csv,tsv,json,parquet}
   - *.{csv,tsv,json,parquet}
   - Compressed variants (.gz)
3. FOR each shard:
   a. Detect format from extension
   b. Read file using appropriate reader
   c. Apply signature columns if provided
4. Concatenate all DataFrames
5. RETURN combined DataFrame
```

**Returns**: Combined DataFrame with all records

**Complexity**: O(n) where n = total records across all shards

#### `load_signature_columns(signature_path) -> Optional[list]`

**Purpose**: Load column names from signature file for schema enforcement

**Algorithm**:
```python
1. Check if signature directory exists
2. Find first signature file in directory
3. Read file content
4. Parse comma-separated column names
5. RETURN list of column names or None
```

### Sequence Field Detection Component

#### `detect_sequence_fields(df) -> Dict[str, List[str]]`

**Purpose**: Automatically detect categorical and numerical sequence fields

**Algorithm**:
```python
1. Initialize result dict with keys: categorical, numerical, temporal

2. Create entity pattern from configuration:
   IF ENABLE_MULTI_SEQUENCE:
      pattern = f"({ENTITY_ID_FIELD}|{SECONDARY_ENTITY_FIELD})"
   ELSE:
      pattern = ENTITY_ID_FIELD

3. FOR each column in DataFrame:
   a. Check if matches sequence naming pattern
   b. IF matches:
      - Check for categorical indicators (cat_seq, categorical)
      - Check for numerical indicators (num_seq, numerical, amount)
      - Classify accordingly
   
4. IF no explicit sequence fields found:
   a. Infer from data:
      - Sample first value
      - Check if contains SEQUENCE_SEPARATOR
      - Try parsing as numerical
      - Classify based on parse success

5. RETURN sequence_fields dictionary
```

**Returns**: Dictionary with categorical, numerical, and temporal field lists

**Example**:
```python
{
  "categorical": ["category_seq_by_customerId", "merchant_seq_by_customerId"],
  "numerical": ["amount_seq_by_customerId", "orderDate_seq_by_customerId"],
  "temporal": ["orderDate"]
}
```

### Sequence Processing Operations

#### `SequenceOrderingOperation.process(df) -> pd.DataFrame`

**Purpose**: Sort sequences by temporal field and handle duplicates

**Algorithm**:
```python
1. Convert temporal field to numeric:
   df[temporal_field] = pd.to_numeric(df[temporal_field], errors='coerce')

2. Sort by temporal field ascending:
   df = df.sort_values(by=temporal_field, ascending=True)

3. Handle duplicates:
   df = df.drop_duplicates(subset=[id_field], keep='last')

4. Log ordering statistics

5. RETURN ordered DataFrame
```

**Returns**: DataFrame sorted by temporal field with duplicates removed

**Complexity**: O(n log n) for sorting

#### `DataValidationOperation.process(df, sequence_fields) -> pd.DataFrame`

**Purpose**: Validate sequence data integrity based on strategy

**Algorithm**:
```python
1. Check required fields present:
   required = [temporal_field, id_field]
   missing = [f for f in required if f not in df.columns]

2. Handle missing fields based on strategy:
   IF validation_strategy == "strict":
      RAISE RuntimeError if any missing
   ELSE:
      LOG warning and continue

3. Validate sequence field consistency:
   FOR entity, fields in sequence_fields:
      FOR field in fields:
         IF field in df.columns:
            a. Check for empty sequences
            b. IF strict mode:
               Remove empty sequences
            c. IF lenient mode:
               Log warning but keep

4. Log validation statistics

5. RETURN validated DataFrame
```

**Returns**: Validated DataFrame (potentially filtered if strict mode)

**Validation Strategies**:
- **Strict**: Raises errors on missing fields, removes invalid sequences
- **Lenient**: Logs warnings, attempts to proceed with available data

#### `MissingValueHandlingOperation.process(df, sequence_fields) -> pd.DataFrame`

**Purpose**: Standardize missing value indicators across sequences

**Algorithm**:
```python
1. FOR each entity type (categorical, numerical):
   FOR each field in entity:
      IF field exists in DataFrame:
         a. FOR each missing indicator:
            - Replace with standardized empty string ""
         b. Handle None/NaN values:
            df[field] = df[field].fillna("")

2. Log completion

3. RETURN DataFrame with standardized missing values
```

**Returns**: DataFrame with standardized missing value representation

**Complexity**: O(n×m) where n = rows, m = sequence fields

#### `parse_sequence_data(df, sequence_fields) -> Dict[str, np.ndarray]`

**Purpose**: Parse string-encoded sequences into numpy arrays

**Algorithm**:
```python
1. Process categorical sequences:
   FOR each categorical field:
      a. Parse string sequences:
         sequences = [str(seq).split(SEPARATOR) for seq in df[field]]
      
      b. Find max sequence length
      
      c. Pad sequences to max length
      
      d. Label encode categorical values:
         encoder = LabelEncoder()
         encoder.fit(all_values)
         encoded = encoder.transform(sequences)
      
      e. Stack encoded sequences
   
   f. RETURN shape: (batch_size, max_seq_len, num_cat_features)

2. Process numerical sequences:
   FOR each numerical field:
      a. Parse string sequences:
         sequences = [str(seq).split(SEPARATOR) for seq in df[field]]
         Convert to float, handle errors
      
      b. Find max sequence length
      
      c. Pad sequences to max length with 0.0
      
      d. Stack numerical sequences
   
   e. Add padding indicator column (all 1.0)
   
   f. RETURN shape: (batch_size, max_seq_len, num_num_features + 1)

3. RETURN sequence_data dictionary
```

**Returns**: Dictionary mapping sequence types to numpy arrays

**Example**:
```python
{
  "categorical": np.array(shape=(10000, 45, 3)),  # Before padding
  "numerical": np.array(shape=(10000, 45, 5))      # Before padding
}
```

#### `TimeDeltaComputationOperation.process(sequence_data) -> Dict[str, np.ndarray]`

**Purpose**: Compute relative time deltas for temporal relationships

**Algorithm**:
```python
1. FOR each sequence type in numerical sequences:
   IF sequence has multiple features:
      a. Identify temporal column (assume -2, before padding indicator)
      
      b. Extract most recent timestamp for each sample:
         recent_time = seq_array[:, -1, temporal_col]
      
      c. Compute relative time deltas:
         seq_array[:, :, temporal_col] = recent_time - seq_array[:, :, temporal_col]
      
      d. Cap time deltas to max value:
         seq_array[:, :, temporal_col] = np.clip(seq_array, 0, MAX_SECONDS)
      
      e. Log completion

2. RETURN updated sequence_data
```

**Returns**: Sequence data with time deltas instead of absolute timestamps

**Time Delta Strategy**:
- Computes relative time from most recent event
- Recent event has delta = 0
- Older events have positive deltas
- Capped to prevent extreme values affecting normalization

**Complexity**: O(batch_size × seq_len)

#### `SequencePaddingOperation.process(sequence_data) -> Dict[str, np.ndarray]`

**Purpose**: Normalize all sequences to target length with attention masks

**Algorithm**:
```python
1. Initialize result dictionaries:
   padded_data = {}
   attention_masks = {}

2. FOR each sequence type (categorical, numerical):
   a. Get current shape: (batch_size, seq_len, feature_dim)
   
   b. IF seq_len == target_length:
      - Use sequences as-is
      - Create all-ones attention mask
   
   c. ELIF seq_len < target_length:
      # Padding needed
      pad_width = target_length - seq_len
      
      IF padding_strategy == "pre":
         pad_config = ((0,0), (pad_width,0), (0,0))
         mask[:, pad_width:] = 1  # Valid positions
      ELSE:  # post
         pad_config = ((0,0), (0,pad_width), (0,0))
         mask[:, :seq_len] = 1  # Valid positions
      
      padded_array = np.pad(seq_array, pad_config, constant_values=0)
   
   d. ELSE:  # seq_len > target_length
      # Truncation needed
      IF truncation_strategy == "pre":
         truncated = seq_array[:, -target_length:, :]
      ELSE:  # post
         truncated = seq_array[:, :target_length, :]
      
      mask = np.ones((batch_size, target_length))

3. Add attention masks to output if enabled

4. RETURN padded_data with attention masks
```

**Returns**: Dictionary with normalized sequences and attention masks

**Padding Strategies**:
- **Pre-padding**: Add zeros at beginning `[0,0,0,5,10,15]`
- **Post-padding**: Add zeros at end `[5,10,15,0,0,0]`

**Truncation Strategies**:
- **Pre-truncation**: Keep most recent events
- **Post-truncation**: Keep oldest events

**Complexity**: O(batch_size × target_length × feature_dim)

## Algorithms and Data Structures

### Algorithm 1: Sequence Field Auto-Detection

**Problem**: Identify which columns contain temporal sequences without explicit configuration

**Solution Strategy**:
1. Pattern-based detection using naming conventions
2. Content-based inference for ambiguous cases
3. Multi-entity support with configurable patterns

**Algorithm**:
```python
# Primary: Pattern-based detection
pattern = SEQUENCE_NAMING_PATTERN  # "*_seq_by_{entity}.*"
FOR column in df.columns:
   IF column matches pattern:
      IF "cat_seq" or "categorical" in column:
         → categorical sequence
      ELIF "num_seq" or "numerical" or numeric_indicator in column:
         → numerical sequence

# Secondary: Content-based inference
IF no sequences found:
   FOR column in df.columns:
      sample = df[column].iloc[0]
      IF SEQUENCE_SEPARATOR in sample:
         TRY:
            parse as numerical
            → numerical sequence
         EXCEPT:
            → categorical sequence
```

**Benefit**: Zero-configuration sequence detection for standard naming conventions

### Algorithm 2: Label Encoding for Categorical Sequences

**Problem**: Convert variable-length string sequences into numerical arrays

**Solution**:
```python
# Step 1: Parse and pad sequences
sequences = []
FOR seq_str in df[field]:
   values = seq_str.split(SEPARATOR)
   sequences.append(values)

max_len = max(len(seq) for seq in sequences)

padded = []
FOR seq in sequences:
   IF len(seq) < max_len:
      seq.extend([""] * (max_len - len(seq)))
   padded.append(seq[:max_len])

# Step 2: Fit label encoder on all unique values
encoder = LabelEncoder()
all_values = [val for seq in padded for val in seq]
encoder.fit(all_values)

# Step 3: Transform each sequence
encoded = []
FOR seq in padded:
   encoded.append(encoder.transform(seq))

result = np.array(encoded)  # Shape: (batch_size, max_len)
```

**Complexity**: O(n×m) where n = samples, m = sequence length

### Algorithm 3: Attention Mask Generation

**Problem**: Indicate which positions in padded sequences contain valid data

**Solution**:
```python
IF seq_len < target_length:
   mask = np.zeros((batch_size, target_length))
   
   IF padding_strategy == "pre":
      # Padding at start: [0,0,0,1,1,1]
      mask[:, pad_width:] = 1
   ELSE:
      # Padding at end: [1,1,1,0,0,0]
      mask[:, :seq_len] = 1

ELIF seq_len >= target_length:
   # No padding needed or truncated
   mask = np.ones((batch_size, target_length))
```

**Use Case**: Transformer models use attention masks to ignore padded positions

### Data Structure: Sequence Data Dictionary

```python
{
  "categorical": np.ndarray,  # Shape: (B, L, C)
  "numerical": np.ndarray,    # Shape: (B, L, N+1)  # +1 for padding indicator
  "categorical_attention_mask": np.ndarray,  # Shape: (B, L)
  "numerical_attention_mask": np.ndarray     # Shape: (B, L)
}

# Where:
# B = batch size
# L = sequence length (target_length after padding/truncation)
# C = number of categorical features
# N = number of numerical features
```

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Data Loading | O(n) | O(n) | n = total records |
| Sequence Ordering | O(n log n) | O(n) | Sorting dominant |
| Field Detection | O(m) | O(m) | m = number of columns |
| Sequence Parsing | O(n×s×f) | O(n×L×f) | s=avg seq len, L=target len, f=features |
| Label Encoding | O(n×s) | O(v) | v = unique values |
| Time Delta Computation | O(n×L×f) | O(1) | In-place operation |
| Padding/Truncation | O(n×L×f) | O(n×L×f) | Creates new arrays |
| **Total** | **O(n×L×f)** | **O(n×L×f)** | Dominated by array operations |

### Memory Requirements

**Peak Memory Usage**:
```
Total ≈ Input Data + Parsed Arrays + Padded Output

Example (10K samples, 51 length, 8 features):
- Input DataFrame: ~10 MB
- Parsed arrays: ~4 MB
- Padded arrays: ~4 MB
- Attention masks: ~1 MB
Total: ~19 MB
```

### Processing Time Estimates

| Dataset Size | Seq Length | Features | Time (single-thread) |
|--------------|------------|----------|---------------------|
| 10K samples  | 51         | 8        | ~5-10 seconds       |
| 100K samples | 51         | 8        | ~30-60 seconds      |
| 1M samples   | 51         | 8        | ~5-10 minutes       |

**Optimization**: Enable `ENABLE_DISTRIBUTED_PROCESSING=true` for large datasets

## Error Handling

### Input Validation Errors

**Missing Required Fields**:
- **Cause**: Required columns not present in input data
- **Handling**: 
  - Strict mode: Raises `RuntimeError` immediately
  - Lenient mode: Logs warning and attempts to proceed

**Invalid Temporal Data**:
- **Cause**: Temporal field contains non-numeric values
- **Handling**: `pd.to_numeric(..., errors='coerce')` converts to NaN, logged as warning

### Processing Errors

**Empty Sequences**:
- **Cause**: All values in sequence match missing indicators
- **Handling**:
  - Strict mode: Removes records with empty sequences
  - Lenient mode: Keeps records with empty sequences (encoded as zeros)

**Sequence Length Mismatch**:
- **Cause**: Different fields have different max sequence lengths
- **Handling**: Each field padded independently to its max length before stacking

**Label Encoding Errors**:
- **Cause**: Unseen categories during transform
- **Handling**: LabelEncoder fitted on all values including empty strings

### File Format Errors

**Unsupported Format**:
- **Cause**: File extension not recognized
- **Handling**: Raises `ValueError` with supported format list

**Corrupted Compressed Files**:
- **Cause**: Invalid gzip compression
- **Handling**: Propagates decompression error with file path context

## Best Practices

### For Production Deployments

1. **Use Signature Files**: Always provide column signature for schema enforcement
   ```bash
   echo "customerId,orderDate,amount_seq,category_seq" > signature/columns.txt
   ```

2. **Enable Strict Validation**: Catch data quality issues early
   ```bash
   export VALIDATION_STRATEGY="strict"
   ```

3. **Cap Time Deltas**: Prevent extreme values from affecting normalization
   ```bash
   export TIME_DELTA_MAX_SECONDS="31536000"  # 1 year in seconds
   ```

4. **Use NumPy Format**: Most efficient for downstream ML training
   ```bash
   export OUTPUT_FORMAT="numpy"
   ```

### For Development

1. **Start with Lenient Validation**: Understand data issues without blocking
   ```bash
   export VALIDATION_STRATEGY="lenient"
   ```

2. **Use CSV Output**: Easier to inspect normalized sequences
   ```bash
   export OUTPUT_FORMAT="csv"
   ```

3. **Disable Attention Masks**: Reduce output size during testing
   ```bash
   export INCLUDE_ATTENTION_MASKS="false"
   ```

### For Performance Optimization

1. **Use Parquet Input**: Faster loading than CSV for large datasets
2. **Enable Distributed Processing**: For datasets > 100K samples
   ```bash
   export ENABLE_DISTRIBUTED_PROCESSING="true"
   export CHUNK_SIZE="50000"
   ```

3. **Pre-sort Data**: Provide pre-sorted data to skip ordering operation

## Example Configurations

### Example 1: Basic Temporal Sequence Normalization
```bash
export SEQUENCE_LENGTH="51"
export SEQUENCE_SEPARATOR="~"
export TEMPORAL_FIELD="orderDate"
export SEQUENCE_GROUPING_FIELD="customerId"
export RECORD_ID_FIELD="objectId"
export VALIDATION_STRATEGY="strict"
export OUTPUT_FORMAT="numpy"

python temporal_sequence_normalization.py --job_type training
```

**Use Case**: Standard fraud detection sequence normalization

### Example 2: Multi-Sequence Processing (Customer + Credit Card)
```bash
export SEQUENCE_LENGTH="51"
export SEQUENCE_SEPARATOR="~"
export TEMPORAL_FIELD="orderDate"
export SEQUENCE_GROUPING_FIELD="customerId"
export RECORD_ID_FIELD="objectId"
export ENABLE_MULTI_SEQUENCE="true"
export SECONDARY_ENTITY_FIELD="creditCardId"
export SEQUENCE_NAMING_PATTERN="*_seq_by_{entity}.*"

python temporal_sequence_normalization.py --job_type training
```

**Use Case**: Dual-entity fraud detection with both customer and credit card histories

### Example 3: Large Dataset Processing
```bash
export SEQUENCE_LENGTH="100"
export ENABLE_DISTRIBUTED_PROCESSING="true"
export CHUNK_SIZE="100000"
export MAX_WORKERS="4"
export OUTPUT_FORMAT="parquet"

python temporal_sequence_normalization.py --job_type training
```

**Use Case**: Processing millions of sequences with parallel chunks

## Integration Patterns

### Upstream Integration
```
TabularPreprocessing
   ↓ (outputs: processed sequences in string format)
TemporalSequenceNormalization
   ↓ (outputs: normalized numpy arrays)
```

### Downstream Integration
```
TemporalSequenceNormalization
   ↓ (outputs: normalized sequences + attention masks)
TemporalFeatureEngineering
   ↓ (outputs: engineered temporal features)
PyTorchTraining (TSA Model)
```

### Complete Fraud Detection Pipeline
```
DataLoading → TabularPreprocessing → TemporalSequenceNormalization → 
TemporalFeatureEngineering → PyTorchTraining → ModelInference → ActiveSampleSelection
```

## Troubleshooting

### Issue: Sequence Field Detection Fails

**Symptom**: No sequence fields detected, script fails

**Common Causes**:
1. Column names don't match `SEQUENCE_NAMING_PATTERN`
2. Separator not found in sequence values
3. Multi-sequence enabled but pattern doesn't match entities

**Solution**: 
1. Check column naming conventions
2. Adjust `SEQUENCE_NAMING_PATTERN` to match your data
3. Verify `SEQUENCE_SEPARATOR` matches actual data

### Issue: Memory Overflow

**Symptom**: Script crashes with out-of-memory error

**Common Causes**:
1. Large dataset with long sequences
2. Too many features creating large arrays

**Solution**:
1. Enable distributed processing:
   ```bash
   export ENABLE_DISTRIBUTED_PROCESSING="true"
   export CHUNK_SIZE="50000"
   ```
2. Use Parquet output format (more memory efficient)
3. Process data in smaller batches

### Issue: Attention Mask Mismatch

**Symptom**: Attention masks don't align with sequences

**Common Causes**:
1. Padding strategy inconsistent with model expectations
2. Pre/post truncation confusion

**Solution**:
1. Verify padding strategy matches model:
   - Transformer models typically use pre-padding
   - RNN models can use either
2. Check attention mask values match sequence validity

## References

### Related Scripts
- [`tabular_preprocessing.py`](../steps/tabular_preprocessing_step.md): Upstream preprocessing for sequence generation

### Related Documentation
- **Contract**: `src/cursus/steps/contracts/temporal_sequence_normalization_contract.py`
- **Step Specification**: Temporal sequence normalization step specification

### Related Design Documents
- **[Temporal Sequence Normalization Design](../1_design/temporal_sequence_normalization_design.md)**: Complete design document with architecture, processing operations, and validation strategies
- **[Temporal Feature Engineering Design](../1_design/temporal_feature_engineering_design.md)**: Downstream feature engineering from normalized sequences
- **[Multi-Sequence Preprocessing Design](../1_design/multi_sequence_preprocessing_design.md)**: Multi-entity sequence processing patterns

---

**Document Metadata**:
- **Author**: Cursus Framework Team
- **Last Updated**: 2025-11-18
- **Script Version**: 2025-11-18
- **Documentation Version**: 1.0
