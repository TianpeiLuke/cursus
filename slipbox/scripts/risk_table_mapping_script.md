---
tags:
  - code
  - processing_script
  - risk_mapping
  - categorical_features
  - feature_engineering
keywords:
  - risk table mapping
  - categorical encoding
  - target encoding
  - weight of evidence
  - smoothing
  - parameter accumulator
  - format preservation
topics:
  - categorical feature engineering
  - risk-based encoding
  - preprocessing pipeline
language: python
date of note: 2025-11-18
---

# Risk Table Mapping Script Documentation

## Overview

The `risk_table_mapping.py` script transforms categorical features into continuous risk scores based on their correlation with the target variable, implementing a sophisticated weight-of-evidence (WoE) encoding strategy. This script is a critical component in the preprocessing pipeline that bridges categorical data to numerical ML models while preserving target-feature relationships.

The script operates in dual modes: training mode fits risk tables on labeled data with smoothing and count thresholds, while inference mode applies pre-trained risk tables to new data. It implements the parameter accumulator pattern, copying all artifacts from previous preprocessing steps and adding its own risk mappings, ensuring downstream steps have access to the complete preprocessing parameter set.

Key capabilities:
- **Risk-based encoding**: Maps categorical values to continuous risk scores based on target correlation
- **Robust estimation**: Applies Laplace smoothing and count thresholds to prevent overfitting
- **Dual-mode operation**: Training (fit + transform) and inference (transform only)
- **Parameter accumulation**: Preserves all artifacts from previous preprocessing steps
- **Format preservation**: Auto-detects and maintains input format (CSV/TSV/Parquet)
- **Job type variants**: Supports training, validation, testing, and calibration workflows

## Purpose and Major Tasks

### Primary Purpose
Transform categorical features into continuous risk scores that quantify the probability of the positive target class, enabling numerical ML models to leverage categorical information while maintaining interpretability and preventing overfitting through smoothing.

### Major Tasks

1. **Hyperparameter Loading**: Load preprocessing configuration including categorical field list, smoothing factors, and count thresholds from hyperparameters.json

2. **Data Loading with Format Detection**: Auto-detect and load input data in CSV, TSV, or Parquet formats across multiple splits (train/test/val for training, single split for others)

3. **Risk Table Generation** (Training Mode):
   - Compute target-conditional probabilities for each categorical value
   - Apply Laplace smoothing to prevent zero-probability estimates
   - Apply count thresholds to filter unreliable estimates
   - Calculate default risk scores for unseen categories

4. **Risk Mapping Transformation**:
   - Map categorical values to their computed risk scores
   - Handle missing categories with default risk scores
   - Preserve non-categorical columns unchanged

5. **Parameter Accumulation**: Copy all existing artifacts from previous preprocessing steps (imputation dictionaries, schemas, etc.) to output directory

6. **Artifact Persistence**: Save fitted risk tables, hyperparameters, and accumulated artifacts for reuse in inference workflows

7. **Multi-Split Processing**: Handle train/test/val splits in training mode or single split in inference modes

## Script Contract

### Entry Point
```
risk_table_mapping.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `input_data` | `/opt/ml/processing/input/data` | Preprocessed data from upstream steps |
| `hyperparameters_s3_uri` | `/opt/ml/code/hyperparams` | Configuration directory |
| `model_artifacts_input` | `/opt/ml/processing/input/model_artifacts` | Previous preprocessing artifacts (non-training) |

**Training Mode Data Structure**:
```
/opt/ml/processing/input/data/
├── train/
│   └── train_processed_data.{csv|tsv|parquet}
├── test/
│   └── test_processed_data.{csv|tsv|parquet}
└── val/
    └── val_processed_data.{csv|tsv|parquet}
```

**Non-Training Mode Data Structure**:
```
/opt/ml/processing/input/data/
└── {job_type}/
    └── {job_type}_processed_data.{csv|tsv|parquet}
```

**Model Artifacts Input** (Non-Training):
```
/opt/ml/processing/input/model_artifacts/
├── risk_table_map.pkl          # Pre-trained risk tables
├── impute_dict.pkl             # From missing_value_imputation
├── hyperparameters.json        # From training
└── [other artifacts]           # From previous steps
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `processed_data` | `/opt/ml/processing/output/data` | Risk-mapped data by split |
| `model_artifacts_output` | `/opt/ml/processing/output/model_artifacts` | Risk tables + accumulated artifacts |

**Output Structure**:
```
/opt/ml/processing/output/
├── data/
│   ├── train/
│   │   └── train_processed_data.{format}
│   ├── test/
│   │   └── test_processed_data.{format}
│   └── val/
│       └── val_processed_data.{format}
└── model_artifacts/
    ├── risk_table_map.pkl      # Fitted risk tables
    ├── hyperparameters.json    # Copy of config
    ├── impute_dict.pkl         # Accumulated from previous step
    └── [other artifacts]       # Accumulated from previous steps
```

### Required Environment Variables

None - all configuration via hyperparameters.json

### Optional Environment Variables

None - script operates entirely from hyperparameters and command-line arguments

### Job Arguments

| Argument | Type | Required | Description | Choices |
|----------|------|----------|-------------|---------|
| `--job_type` | `str` | Yes | Type of processing job | `training`, `validation`, `testing`, `calibration` |

### Hyperparameters (from hyperparameters.json)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cat_field_list` | `List[str]` | `[]` | List of categorical feature names to encode |
| `label_name` | `str` | `"target"` | Name of target variable column |
| `smooth_factor` | `float` | `0.01` | Laplace smoothing factor (0.0 to 1.0) |
| `count_threshold` | `int` | `5` | Minimum count for computed risk (vs default) |

**Example hyperparameters.json**:
```json
{
  "cat_field_list": ["merchant_category", "payment_method", "device_type"],
  "label_name": "is_fraud",
  "smooth_factor": 0.01,
  "count_threshold": 5
}
```

### Framework Dependencies

- **pandas** >= 1.3.0 (data manipulation)
- **numpy** >= 1.21.0 (numerical operations)
- **scikit-learn** >= 1.0.0 (SimpleImputer utilities)

## Input Data Structure

### Expected Input Format

**Training Mode**:
```
/opt/ml/processing/input/data/
├── train/train_processed_data.{csv|tsv|parquet}
├── test/test_processed_data.{csv|tsv|parquet}
└── val/val_processed_data.{csv|tsv|parquet}
```

**Inference Modes**:
```
/opt/ml/processing/input/data/
└── validation/validation_processed_data.{csv|tsv|parquet}
```

### Required Columns

- **Target Column**: Specified by `label_name` (default: `"target"`)
  - Must be binary (0 or 1) for risk calculation
  - Can contain -1 or NaN (filtered out during fitting)
  
- **Categorical Columns**: Specified in `cat_field_list`
  - Can be string, numeric, or pandas categorical types
  - Validation checks for < 100 unique values
  - Missing values handled via default risk score

### Optional Columns

All other columns are preserved unchanged through the transformation. Common examples:
- ID columns (customer_id, transaction_id)
- Numerical features (already numeric, not in cat_field_list)
- Timestamp columns
- Metadata fields

### Input Data Requirements

1. **Target Variable**: Binary classification (0/1) for training mode
2. **Categorical Features**: Should have reasonable cardinality (< 100 unique values recommended)
3. **Data Quality**: Script handles missing values in categorical features
4. **Format Consistency**: All splits must use the same format (CSV, TSV, or Parquet)

## Output Data Structure

### Output Directory Structure

```
/opt/ml/processing/output/
├── data/
│   ├── {split_1}/
│   │   └── {split_1}_processed_data.{format}
│   └── {split_2}/
│       └── {split_2}_processed_data.{format}
└── model_artifacts/
    ├── risk_table_map.pkl
    ├── hyperparameters.json
    └── [accumulated artifacts from previous steps]
```

### Output Data Format

**Transformed Data Characteristics**:
- Original non-categorical columns: **Preserved unchanged**
- Categorical columns (in cat_field_list): **Replaced with continuous risk scores**
- Format: **Same as input** (CSV/TSV/Parquet)
- Schema: **Column names unchanged**, data types converted to float for risk scores

**Example Transformation**:

*Input*:
```csv
id,merchant_category,amount,is_fraud
001,electronics,150.0,0
002,groceries,45.0,1
003,electronics,200.0,1
```

*Output (after risk mapping)*:
```csv
id,merchant_category,amount,is_fraud
001,0.3333,150.0,0
002,0.5000,45.0,1
003,0.3333,200.0,1
```

Where `0.3333` = risk score for "electronics", `0.5000` = risk score for "groceries"

### Risk Table Map Structure

**File**: `risk_table_map.pkl` (Python pickle format)

**Structure**:
```python
{
    "merchant_category": {
        "varName": "merchant_category",
        "type": "categorical",
        "mode": "categorical",  # or "numeric" for numeric categoricals
        "default_bin": 0.42,    # Default risk score
        "bins": {
            "electronics": 0.3333,
            "groceries": 0.5000,
            "fashion": 0.2500,
            # ... more category mappings
        }
    },
    "payment_method": {
        # ... similar structure
    }
}
```

### Hyperparameters Output

**File**: `hyperparameters.json`

Contains a copy of the input hyperparameters for reproducibility and downstream processing.

## Key Functions and Tasks

### Configuration Loading Component

#### `load_json_config(config_path)`
**Purpose**: Load and validate JSON configuration files with comprehensive error handling

**Algorithm**:
```python
1. Attempt to open and parse JSON file
2. Handle specific error types:
   - FileNotFoundError: Config file missing
   - PermissionError: Access denied
   - JSONDecodeError: Invalid JSON syntax
3. Return parsed configuration dict
```

**Error Handling**:
- Logs specific error type and path
- Re-raises with context for debugging
- Distinguishes file system vs format errors

### Categorical Field Validation Component

#### `validate_categorical_fields(df, cat_field_list)`
**Purpose**: Validate that specified fields are suitable for categorical encoding

**Algorithm**:
```python
1. Initialize valid_fields = []
2. FOR each field in cat_field_list:
   a. Check if field exists in DataFrame
   b. Count unique values
   c. IF (is_categorical_dtype OR unique_count < 100):
      - Add to valid_fields
      - Log validation success
   d. ELSE:
      - Log warning about high cardinality
3. Return valid_fields list
```

**Validation Criteria**:
- Field must exist in DataFrame
- Must be pandas categorical OR have < 100 unique values
- Logs warnings for potentially unsuitable fields

**Time Complexity**: O(n × d) where n=rows, d=categorical fields

### Risk Table Generation Component

#### `OfflineBinning` Class

**Purpose**: Core class for creating and applying risk table mappings

**Attributes**:
- `risk_tables`: Dict mapping variable names to risk table structures
- `target`: Target variable name
- `variables`: List of categorical variable names

#### `OfflineBinning.fit(df, smooth_factor, count_threshold)`
**Purpose**: Fit risk tables on training data with smoothing

**Algorithm**:
```python
1. Filter training data (remove target == -1 or NaN)
2. Calculate default_risk = mean(target)
3. Calculate smooth_samples = len(data) * smooth_factor

4. FOR each categorical variable:
   a. Infer data type (numeric vs categorical)
   b. Initialize risk table structure:
      {
        "varName": variable,
        "type": "categorical",
        "mode": data_type,
        "default_bin": default_risk,
        "bins": {}
      }
   c. IF all values are NULL:
      - Set bins = {}
      - Continue to next variable
   d. ELSE:
      - Call _create_risk_table()
      - Store result in bins

5. Store risk_tables for transform phase
```

**Smoothing Strategy**: Laplace smoothing with dataset-proportional pseudo-counts

#### `OfflineBinning._create_risk_table(df, variable, default_risk, samples, count_threshold)`
**Purpose**: Calculate risk scores for individual categorical variable

**Algorithm**:
```python
1. Create crosstab: variable × target (with margins)
   Example:
   merchant_cat    0    1    _count_
   electronics    20   10      30
   groceries       5   15      20
   _count_        25   25      50

2. Calculate raw risk:
   risk = count(target=1) / (count(target=0) + count(target=1))
   
3. Apply smoothing:
   smooth_risk = (count * risk + smooth_samples * default_risk) 
                 / (count + smooth_samples)
   
4. Apply count threshold:
   IF count < count_threshold:
       use default_risk
   ELSE:
       use smooth_risk

5. Remove margin row (_count_)
6. Return dict mapping categories to risk scores
```

**Mathematical Formulation**:

For category $c$ with $n_c^+$ positive samples and $n_c^-$ negative samples:

$$\text{raw\_risk}_c = \frac{n_c^+}{n_c^+ + n_c^-}$$

With smoothing parameter $\alpha$ (smooth_factor) and dataset size $N$:

$$\text{smooth\_risk}_c = \frac{n_c^+ + \alpha N \cdot \text{default\_risk}}{n_c^+ + n_c^- + \alpha N}$$

**Time Complexity**: O(n × d × u) where n=rows, d=features, u=unique values per feature

#### `OfflineBinning.transform(df)`
**Purpose**: Apply fitted risk tables to transform categorical features

**Algorithm**:
```python
1. Create copy of input DataFrame
2. FOR each variable in risk_tables:
   a. IF variable exists in DataFrame:
      - Extract bins dict
      - Extract default_bin value
      - Map categorical values: df[var].map(bins).fillna(default_bin)
      - Replace original column with risk scores
3. Return transformed DataFrame
```

**Key Property**: Non-categorical columns remain unchanged

**Time Complexity**: O(n × d) where n=rows, d=categorical features

### Data I/O Component

#### `load_split_data(job_type, input_dir)`
**Purpose**: Load data with automatic format detection and job-type-aware splitting

**Algorithm**:
```python
1. Initialize result = {}
2. IF job_type == "training":
   a. FOR split in ["train", "test", "val"]:
      - Call _detect_file_format(split_dir, split_name)
      - Load DataFrame based on detected format
      - Store in result[split]
   b. Store detected format in result["_format"]
3. ELSE:  # validation/testing/calibration
   a. Call _detect_file_format(job_type_dir, job_type)
   b. Load DataFrame based on detected format
   c. Store in result[job_type]
   d. Store detected format in result["_format"]
4. Return result dict
```

**Format Detection Priority**:
1. CSV (`.csv`)
2. TSV (`.tsv`)
3. Parquet (`.parquet`)

#### `save_output_data(job_type, output_dir, data_dict)`
**Purpose**: Save processed data preserving input format

**Algorithm**:
```python
1. Extract output_format from data_dict["_format"]
2. FOR each (split_name, df) in data_dict:
   a. Skip "_format" metadata key
   b. Create output directory: output_dir/split_name/
   c. Save based on format:
      - csv: df.to_csv(path, index=False)
      - tsv: df.to_csv(path, sep='\t', index=False)
      - parquet: df.to_parquet(path, index=False)
3. Log save locations and shapes
```

**Format Preservation**: Maintains input format throughout pipeline

### Parameter Accumulator Component

#### `copy_existing_artifacts(src_dir, dst_dir)`
**Purpose**: Implement parameter accumulator pattern by copying all artifacts from previous steps

**Algorithm**:
```python
1. Validate src_dir exists and is accessible
2. Create dst_dir if not exists
3. Initialize copied_count = 0
4. FOR each filename in listdir(src_dir):
   a. IF is_file(src_file):
      - Copy file to destination (preserving metadata)
      - Increment copied_count
      - Log copy operation
5. Log total copied_count
```

**Purpose**: Ensures downstream steps have access to all preprocessing parameters in one location

**Accumulated Artifacts**:
- `impute_dict.pkl` from MissingValueImputation
- `schema.json` from TabularPreprocessing
- `feature_columns.txt` from FeatureSelection
- Any other artifacts from previous preprocessing steps

### Core Processing Logic

#### `process_data(data_dict, cat_field_list, label_name, job_type, risk_tables_dict, smooth_factor, count_threshold)`
**Purpose**: Orchestrate risk table fitting or loading and transformation

**Algorithm**:
```python
1. IF job_type == "training":
   a. Validate categorical fields on training data
   b. IF no valid fields:
      - Return original data unchanged
      - Create empty binner for consistency
   c. ELSE:
      - Initialize OfflineBinning with valid fields
      - Fit on training data (with smoothing, thresholds)
      - Transform all splits (train, test, val)
      
2. ELSE:  # validation/testing/calibration
   a. IF risk_tables_dict not provided:
      - Raise ValueError
   b. Extract categorical fields from risk_tables
   c. Initialize OfflineBinning with fields
   d. Load pre-trained risk tables
   e. Transform data using loaded tables

3. Return (transformed_data_dict, binner)
```

**Mode Differentiation**:
- Training: Fit + Transform
- Inference: Load + Transform

### Artifact Management Component

#### `save_artifacts(binner, hyperparams, output_path)`
**Purpose**: Save risk tables and hyperparameters for reuse

**Algorithm**:
```python
1. Save risk tables:
   - Format: {var_name: {"bins": {...}, "default_bin": value}}
   - File: risk_table_map.pkl
   - Method: pickle.dump()
   
2. Save hyperparameters:
   - Format: JSON
   - File: hyperparameters.json
   - Content: Copy of input hyperparameters
   
3. Log save locations and formats
```

**Compatibility**: Output format matches XGBoost training artifact structure

#### `load_risk_tables(risk_table_path)`
**Purpose**: Load pre-trained risk tables for inference mode

**Algorithm**:
```python
1. Validate risk_table_path exists
2. Open pickle file
3. Load risk_tables dict
4. Validate structure (has bins, default_bin)
5. Log loaded table count
6. Return risk_tables
```

### Main Entry Points

#### `internal_main(job_type, input_dir, output_dir, hyperparams, ...)`
**Purpose**: Core processing logic independent of CLI parsing (enables testing)

**Algorithm**:
```python
1. Extract hyperparameters:
   - cat_field_list
   - label_name  
   - smooth_factor
   - count_threshold

2. Setup output directories

3. IF model_artifacts_input_dir provided:
   - Copy existing artifacts (parameter accumulator)

4. Load data with format detection:
   - Training: train/test/val splits
   - Others: single split

5. IF job_type != "training":
   - Load pre-trained risk tables from artifacts

6. Process data:
   - Training: fit + transform
   - Inference: load + transform

7. Save processed data (format-preserved)

8. Save artifacts:
   - risk_table_map.pkl
   - hyperparameters.json

9. Return (transformed_data, binner)
```

**Dependency Injection**: Accepts custom load/save functions for testing

#### `main(input_paths, output_paths, environ_vars, job_args)`
**Purpose**: Standardized entry point for SageMaker Processing

**Algorithm**:
```python
1. Validate required paths present:
   - input_data
   - processed_data

2. Extract job_type from job_args

3. Setup model artifacts paths:
   - Input: for non-training modes
   - Output: for all modes

4. Load hyperparameters with fallback:
   - Primary: input_paths["hyperparameters_s3_uri"]
   - Fallback: /opt/ml/code/hyperparams/hyperparameters.json
   - Fail if neither exists

5. Call internal_main() with extracted parameters

6. Handle exceptions with specific error codes:
   - FileNotFoundError: exit 1
   - ValueError: exit 2
   - General Exception: exit 3
```

## Algorithms and Data Structures

### Algorithm 1: Smoothed Risk Calculation with Count Threshold

**Problem**: Raw target-conditional probabilities overfit on rare categories and fail on unseen categories

**Solution Strategy**:
1. Apply Laplace smoothing to blend category-specific risk with global risk
2. Use count thresholds to fallback to global risk for rare categories
3. Provide default risk for completely unseen categories

**Algorithm**:
```python
# Given category c with observations
n_pos = count(target == 1 for category c)
n_neg = count(target == 0 for category c)
n_total = n_pos + n_neg

# Global statistics
global_risk = count(target == 1) / count(all samples)
smooth_samples = dataset_size * smooth_factor

# Raw risk
raw_risk = n_pos / n_total

# Smoothed risk (Laplace smoothing)
smooth_risk = (n_pos + smooth_samples * global_risk) / 
              (n_total + smooth_samples)

# Apply count threshold
IF n_total >= count_threshold:
    final_risk = smooth_risk
ELSE:
    final_risk = global_risk
```

**Mathematical Properties**:
- As $n_{\text{total}} \rightarrow \infty$: smooth_risk → raw_risk (data dominates)
- As smooth_factor → 0: smooth_risk → raw_risk (no smoothing)
- As smooth_factor → 1: smooth_risk → global_risk (full smoothing)

**Complexity**: O(u) per variable where u = unique categories

**Example**:

Given:
- Category "rare_merchant" with 3 positive, 2 negative samples
- global_risk = 0.40
- dataset_size = 10000
- smooth_factor = 0.01
- count_threshold = 5

```python
n_pos, n_neg = 3, 2
n_total = 5
raw_risk = 3/5 = 0.60

smooth_samples = 10000 * 0.01 = 100
smooth_risk = (3 + 100*0.40) / (5 + 100)
            = 43 / 105
            = 0.4095

# n_total (5) >= count_threshold (5): TRUE
final_risk = 0.4095  # use smoothed risk
```

### Algorithm 2: Parameter Accumulator Pattern

**Problem**: Preprocessing pipeline consists of multiple steps, each producing artifacts needed by downstream steps

**Solution Strategy**:
1. Each step copies all artifacts from previous steps to its output
2. Each step adds its own artifacts to the accumulated set
3. Downstream steps receive complete parameter set in one location

**Algorithm**:
```python
# Step N (e.g., RiskTableMapping)
INPUT: model_artifacts_from_step_N-1/
OUTPUT: model_artifacts_from_step_N/

PROCEDURE:
1. Create output artifacts directory
2. Copy ALL files from input artifacts:
   FOR each file in model_artifacts_input:
       copy(file, model_artifacts_output)
3. Add own artifacts:
   save(risk_table_map.pkl, model_artifacts_output)
   save(hyperparameters.json, model_artifacts_output)
4. Result: model_artifacts_output contains:
   - All artifacts from steps 1 through N-1
   - New artifacts from step N
```

**Benefits**:
- Single artifact location for all preprocessing parameters
- Simplifies downstream step inputs
- Enables flexible pipeline composition
- Maintains artifact provenance

**Example Pipeline**:
```
TabularPreprocessing
  └─> model_artifacts/
      ├─ schema.json
      └─ preprocessor_config.json
  
MissingValueImputation  
  └─> model_artifacts/
      ├─ schema.json (copied)
      ├─ preprocessor_config.json (copied)
      └─ impute_dict.pkl (new)

RiskTableMapping
  └─> model_artifacts/
      ├─ schema.json (copied)
      ├─ preprocessor_config.json (copied)
      ├─ impute_dict.pkl (copied)
      ├─ risk_table_map.pkl (new)
      └─ hyperparameters.json (new)
```

### Algorithm 3: Format-Preserving Data Pipeline

**Problem**: Pipeline should maintain data format (CSV/TSV/Parquet) for consistency and efficiency

**Solution Strategy**:
1. Detect format during initial load
2. Store format metadata alongside data
3. Use same format for all saves

**Algorithm**:
```python
# Load phase
def load_split_data(job_type, input_dir):
    result = {}
    # Try formats in order: CSV, TSV, Parquet
    file_path, detected_format = _detect_file_format(...)
    
    # Load based on format
    IF detected_format == "csv":
        df = pd.read_csv(file_path)
    ELIF detected_format == "tsv":
        df = pd.read_csv(file_path, sep='\t')
    ELIF detected_format == "parquet":
        df = pd.read_parquet(file_path)
    
    # Store format for later use
    result["_format"] = detected_format
    result[split_name] = df
    return result

# Save phase
def save_output_data(job_type, output_dir, data_dict):
    format = data_dict["_format"]  # Retrieve stored format
    
    FOR split_name, df in data_dict:
        IF format == "csv":
            df.to_csv(output_path, index=False)
        ELIF format == "tsv":
            df.to_csv(output_path, sep='\t', index=False)
        ELIF format == "parquet":
            df.to_parquet(output_path, index=False)
```

**Benefits**:
- Zero configuration for format handling
- Maintains storage efficiency of Parquet
- Preserves human-readability of CSV/TSV
- Consistent with upstream/downstream steps

### Data Structure 1: Risk Table Dictionary

```python
risk_tables: Dict[str, Dict[str, Any]] = {
    "categorical_feature_1": {
        "varName": "categorical_feature_1",
        "type": "categorical",
        "mode": "categorical",  # or "numeric" for numeric categoricals
        "default_bin": 0.42,    # Global risk score
        "bins": {
            "category_a": 0.35,  # Risk score for category_a
            "category_b": 0.67,  # Risk score for category_b
            "category_c": 0.21,  # Risk score for category_c
            # ... more categories
        }
    },
    "categorical_feature_2": {
        # ... similar structure
    }
}
```

**Properties**:
- Nested dictionary structure
- Outer key: variable name
- Inner dict: metadata + bins mapping
- Compatible with sklearn transformers
- Serializable with pickle

### Data Structure 2: Processed Data Dictionary

```python
data_dict: Dict[str, Union[pd.DataFrame, str]] = {
    "train": pd.DataFrame(...),      # Training split
    "test": pd.DataFrame(...),       # Test split
    "val": pd.DataFrame(...),        # Validation split
    "_format": "csv",                # Format metadata
}

# OR for inference modes:
data_dict: Dict[str, Union[pd.DataFrame, str]] = {
    "validation": pd.DataFrame(...), # Single split
    "_format": "parquet",            # Format metadata
}
```

**Design Pattern**: Dictionary with special metadata key `"_format"`

**Benefits**:
- Flexible number of splits
- Format information travels with data
- Enables generic save function
- Clear separation of data vs metadata

## Performance Characteristics

### Processing Time

| Operation | Typical Time (10K samples, 10 categorical features, avg 20 unique values) |
|-----------|--------------------------------------------------------------------------|
| Hyperparameter loading | <0.1 seconds |
| Data loading (CSV) | 0.5-1 second per split |
| Data loading (Parquet) | 0.2-0.5 seconds per split |
| Categorical field validation | 0.1-0.2 seconds |
| Risk table fitting | 1-2 seconds |
| Risk mapping transformation | 0.5-1 second per split |
| Artifact copying | 0.1-0.3 seconds |
| Artifact saving | 0.1-0.2 seconds |
| Data saving (CSV) | 0.5-1 second per split |
| Data saving (Parquet) | 0.3-0.7 seconds per split |
| **Total (training, CSV)** | **4-8 seconds** |
| **Total (inference, Parquet)** | **2-4 seconds** |

### Memory Usage

**Typical Memory Profile**:
```
Input data: n × d × 8 bytes (float64)
Risk tables: f × u × (key + value) ~ f × u × 100 bytes
  where f = categorical features, u = avg unique values per feature
Transformed data: n × d × 8 bytes (copy)
Peak: ~2× input data size + risk tables

Example (10K samples, 100 features, 10 categorical with 20 unique each):
- Input data: 10K × 100 × 8 = ~8 MB
- Risk tables: 10 × 20 × 100 = ~20 KB (negligible)
- Peak: ~16 MB
```

**Scalability**:

| Dataset Size | Memory (est.) | Time (est.) |
|--------------|---------------|-------------|
| 1K samples | ~2 MB | 1-2 seconds |
| 10K samples | ~16 MB | 4-8 seconds |
| 100K samples | ~160 MB | 30-60 seconds |
| 1M samples | ~1.6 GB | 5-10 minutes |

**Recommendation**: For datasets > 1M samples with many categorical features:
- Use Parquet format for efficiency
- Consider batch processing
- Monitor memory usage during crosstab operations

## Error Handling

### Error Types

#### Configuration Errors

**Missing Hyperparameters File**:
- **Cause**: hyperparameters.json not found at specified or default paths
- **Handling**: Raises `FileNotFoundError` with detailed message
- **Response**: Script fails with exit code 1, logs available paths tried

**Invalid Hyperparameters JSON**:
- **Cause**: Malformed JSON syntax in configuration file
- **Handling**: Raises `json.JSONDecodeError` with line/column information
- **Response**: Script fails with exit code 1, logs JSON error details

**Permission Denied on Configuration**:
- **Cause**: Cannot read hyperparameters.json due to file permissions
- **Handling**: Raises `PermissionError` with path information
- **Response**: Script fails with exit code 1, logs permission issue

#### Data Validation Errors

**Input Data Not Found**:
- **Cause**: Expected data files missing from input directory
- **Handling**: Raises `RuntimeError` listing expected file patterns
- **Response**: Clear error showing which formats were attempted

**High Cardinality Warning**:
- **Cause**: Categorical field has > 100 unique values
- **Handling**: Logs warning but continues processing
- **Response**: Field still processed but may not be ideal for risk mapping

**Missing Categorical Fields**:
- **Cause**: Field in cat_field_list not found in DataFrame
- **Handling**: Logs warning and skips field
- **Response**: Processing continues with available fields

#### Processing Errors

**Missing Risk Tables** (Non-Training):
- **Cause**: risk_table_map.pkl not found in model_artifacts_input
- **Handling**: Logs warning if file missing
- **Response**: For non-training jobs, will fail when trying to transform

**Empty Categorical Fields**:
- **Cause**: All values in categorical field are NULL
- **Handling**: Creates empty bins dict, uses default_bin for all values
- **Response**: Processing continues, all values map to default risk

**Target Variable Issues**:
- **Cause**: Target not binary (0/1) or all values are -1/NaN
- **Handling**: Filters non-binary values during fitting
- **Response**: May produce empty risk tables if no valid training data

### Error Response Structure

**Exit Codes**:
- **0**: Success
- **1**: FileNotFoundError (missing config or data files)
- **2**: ValueError (invalid parameters or paths)
- **3**: General Exception (unexpected errors)

**Error Message Format**:
```
ERROR: [Error type]: [Detailed message]
[Stack trace if applicable]
```

**Example Error Messages**:
```
ERROR: Hyperparameters file not found at /opt/ml/code/hyperparams/hyperparameters.json. 
Risk table mapping requires hyperparameters to be provided either via 
input channel or in source directory at /opt/ml/code/hyperparams/hyperparameters.json

ERROR: For non-training job types, risk_tables_dict must be provided

WARNING: Field 'country_code' may not be suitable for risk mapping (500 unique values)
```

## Best Practices

### For Production Deployments

1. **Use Appropriate Smoothing**
   - Start with smooth_factor=0.01 (1% of dataset as pseudo-counts)
   - Increase for small datasets or noisy features
   - Decrease for large, clean datasets

2. **Set Reasonable Count Thresholds**
   - Default count_threshold=5 is reasonable for most cases
   - Increase for datasets with many rare categories
   - Monitor distribution of category counts

3. **Validate Categorical Feature Selection**
   - Review cardinality: aim for 5-100 unique values per feature
   - Very low cardinality (<5): consider one-hot encoding instead
   - Very high cardinality (>100): consider binning or grouping first

4. **Monitor Risk Table Quality**
   - Check default_bin vs category-specific risks
   - If most categories use default risk, smoothing may be too aggressive
   - If risk scores are extreme (close to 0 or 1), may need more smoothing

5. **Ensure Artifact Persistence**
   - Verify model_artifacts_output contains all accumulated artifacts
   - Confirm risk_table_map.pkl is accessible for downstream steps
   - Test inference workflow with saved artifacts

### For Development

1. **Start with Small Sample**
   - Validate logic on subset before full dataset
   - Check risk table contents for reasonableness
   - Verify format preservation works correctly

2. **Inspect Risk Tables**
   - Load risk_table_map.pkl and examine contents:
     ```python
     with open('risk_table_map.pkl', 'rb') as f:
         risk_tables = pkl.load(f)
     print(risk_tables)
     ```
   - Verify default_bin is reasonable (should be close to overall positive rate)
   - Check that bins contain expected categories

3. **Test Both Modes**
   - Train with job_type=training
   - Validate with job_type=validation using saved artifacts
   - Ensure inference mode uses correct risk tables

4. **Validate Output**
   - Check that categorical columns now contain continuous values
   - Verify non-categorical columns unchanged
   - Confirm splits maintain correct shapes

### For Performance Optimization

1. **Use Parquet Format**
   - 40-60% size reduction vs CSV
   - Faster I/O operations
   - Better compression for large datasets

2. **Optimize Feature Selection**
   - Remove high-cardinality features before risk mapping
   - Group rare categories into "other" category upstream
   - Consider feature selection to reduce cat_field_list

3. **Minimize Artifact Copying**
   - Parameter accumulator is efficient but avoid unnecessary artifacts
   - Clean up artifacts from steps that are no longer needed
   - Use symbolic links for very large artifact directories (if supported)

## Example Configurations

### Example 1: Standard Training with Moderate Smoothing
```json
{
  "cat_field_list": ["merchant_category", "payment_method", "card_type"],
  "label_name": "is_fraud",
  "smooth_factor": 0.01,
  "count_threshold": 5
}
```

**Use Case**: Fraud detection with 10K training samples, moderate category diversity

**Execution**:
```bash
python risk_table_mapping.py --job_type training
```

### Example 2: Heavy Smoothing for Small Dataset
```json
{
  "cat_field_list": ["product_category", "region", "customer_segment"],
  "label_name": "churn",
  "smooth_factor": 0.05,
  "count_threshold": 10
}
```

**Use Case**: Customer churn with 1K training samples, needs aggressive smoothing

### Example 3: Light Smoothing for Large Dataset
```json
{
  "cat_field_list": ["device_type", "browser", "os", "country"],
  "label_name": "clicked",
  "smooth_factor": 0.001,
  "count_threshold": 50
}
```

**Use Case**: Click prediction with 1M training samples, minimal smoothing needed

### Example 4: Validation with Pre-Trained Risk Tables
```bash
# Training phase already completed, using saved artifacts
python risk_table_mapping.py --job_type validation
```

**Use Case**: Apply pre-trained risk tables to validation split

**Requirements**:
- `/opt/ml/processing/input/model_artifacts/risk_table_map.pkl` must exist
- Uses same categorical features as training

## Integration Patterns

### Upstream Integration

```
TabularPreprocessing
   ↓ (train/test/val splits, schema.json)
MissingValueImputation
   ↓ (imputed data, impute_dict.pkl)
RiskTableMapping
   ↓ (risk-mapped data, risk_table_map.pkl + accumulated artifacts)
```

**Data Flow**:
1. TabularPreprocessing creates initial feature set
2. MissingValueImputation fills missing numerical values
3. RiskTableMapping encodes categorical features as risks

### Downstream Integration

```
RiskTableMapping
   ↓ (risk-mapped data + all accumulated artifacts)
   ├→ XGBoostTraining (uses risk-encoded features)
   ├→ LightGBMTraining (uses risk-encoded features)
   └→ PyTorchTraining (uses risk-encoded features)
```

**Artifact Usage**:
- Training steps receive complete preprocessing parameter set
- risk_table_map.pkl + impute_dict.pkl enable consistent inference preprocessing
- All artifacts packaged with trained model for deployment

### Complete Preprocessing Pipeline

```
1. TabularPreprocessing (job_type=training)
   → train/test/val splits with engineered features
   → model_artifacts: schema.json

2. MissingValueImputation (job_type=training)
   → imputed train/test/val splits
   → model_artifacts: schema.json + impute_dict.pkl

3. RiskTableMapping (job_type=training)
   → risk-mapped train/test/val splits
   → model_artifacts: schema.json + impute_dict.pkl + risk_table_map.pkl

4. XGBoostTraining
   → trained model with all preprocessing artifacts embedded
```

## Troubleshooting

### Issue 1: High Cardinality Warning

**Symptom**:
```
WARNING: Field 'zip_code' may not be suitable for risk mapping (2000 unique values)
```

**Common Causes**:
1. Field truly has high cardinality (zip codes, IDs, etc.)
2. Field needs preprocessing/grouping before risk mapping
3. Field accidentally included in cat_field_list

**Solutions**:
1. Remove high-cardinality field from cat_field_list
2. Group rare categories into "other" in upstream preprocessing
3. Use different encoding strategy (target mean encoding with regularization)
4. Consider binning strategies (geographic clustering for zip codes)

### Issue 2: All Risk Scores Equal to Default

**Symptom**:
All transformed values equal to global risk (e.g., all 0.42)

**Common Causes**:
1. count_threshold too high relative to data size
2. smooth_factor too aggressive
3. Categories all below count threshold

**Solutions**:
1. Lower count_threshold (e.g., from 10 to 5 or 3)
2. Reduce smooth_factor (e.g., from 0.05 to 0.01)
3. Check category distribution: `df[col].value_counts()`
4. Ensure sufficient training data per category

### Issue 3: Risk Tables Not Found in Inference

**Symptom**:
```
WARNING: Risk tables not found at /opt/ml/processing/input/model_artifacts/risk_table_map.pkl
```

**Common Causes**:
1. Training step didn't save artifacts correctly
2. Wrong model_artifacts_input path
3. Artifacts not copied from S3 to container

**Solutions**:
1. Verify training step completed successfully
2. Check S3 location of model_artifacts
3. Ensure ProcessingInput configured correctly in step builder
4. List files in model_artifacts_input directory for debugging

### Issue 4: Format Mismatch Error

**Symptom**:
```
RuntimeError: No processed data file found in /opt/ml/processing/input/data/train
Looked for: ['train_processed_data.csv', 'train_processed_data.tsv', 'train_processed_data.parquet']
```

**Common Causes**:
1. Upstream step saved with different naming convention
2. Files in wrong directory structure
3. Format not supported (e.g., .xlsx)

**Solutions**:
1. Check upstream step output format and naming
2. Verify directory structure matches expectations
3. Convert unsupported formats to CSV/TSV/Parquet
4. Align file naming across preprocessing steps

### Issue 5: Memory Error with Large Dataset

**Symptom**:
```
MemoryError: Unable to allocate array for crosstab operation
```

**Common Causes**:
1. Too many categorical features processed simultaneously
2. Very high cardinality features
3. Insufficient instance memory

**Solutions**:
1. Process categorical features in batches
2. Remove or bin high-cardinality features
3. Use larger processing instance
4. Switch to Parquet format for better memory efficiency

## References

### Related Scripts

- **[`missing_value_imputation.py`](./missing_value_imputation_script.md)**: Upstream imputation script
- **[`tabular_preprocessing.py`](./tabular_preprocessing_script.md)**: Initial preprocessing script
- **[`xgboost_training.py`](./xgboost_training_script.md)**: Downstream training that consumes risk-mapped features

### Related Documentation

- **Contract**: `src/cursus/steps/contracts/risk_table_mapping_contract.py`
- **Step Specification**: RiskTableMapping step specification
- **Config Class**: `src/cursus/steps/configs/config_risk_table_mapping_step.py`
- **Builder**: `src/cursus/steps/builders/builder_risk_table_mapping_step.py`

### Related Design Documents

No specific design documents currently exist for this script. General patterns apply from:
- **[Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)**: General processing step architecture
- **[Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md)**: Format detection and preservation strategy
- **[Job Type Variant Handling](../1_design/job_type_variant_handling.md)**: Multi-variant processing patterns

### External References

- **Weight of Evidence Encoding**: Commonly used in credit scoring and risk modeling
- **Laplace Smoothing**: https://en.wikipedia.org/wiki/Additive_smoothing
- **Target Encoding**: https://contrib.scikit-learn.org/category_encoders/targetencoder.html
- **Pandas Crosstab**: https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html

---

## Document Metadata

**Author**: Cursus Framework Team  
**Last Updated**: 2025-11-18  
**Script Version**: 2025-11-18  
**Documentation Version**: 1.0  
**Review Status**: Complete

**Change Log**:
- 2025-11-18: Initial comprehensive documentation created
- 2025-11-18: Risk table mapping script documented

**Related Scripts**: 
- Upstream: `tabular_preprocessing.py`, `missing_value_imputation.py`
- Downstream: `xgboost_training.py`, `lightgbm_training.py`, `pytorch_training.py`
- Related: Feature engineering and encoding scripts
