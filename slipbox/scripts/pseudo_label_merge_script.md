---
tags:
  - code
  - processing_script
  - semi_supervised_learning
  - active_learning
  - data_merge
  - pseudo_labeling
  - split_aware_merge
keywords:
  - pseudo label merge
  - split-aware merge
  - auto-inferred ratios
  - data augmentation
  - semi-supervised learning
  - active learning
  - provenance tracking
  - schema alignment
  - format preservation
topics:
  - semi-supervised learning
  - active learning
  - data merging
language: python
date of note: 2025-11-18
---

# Pseudo Label Merge Script Documentation

## Overview

The `pseudo_label_merge.py` script implements intelligent merging of original labeled training data with pseudo-labeled or augmented samples for Semi-Supervised Learning (SSL) and Active Learning workflows. The script provides two distinct merge strategies: **split-aware merge** for training jobs that maintains train/test/val boundaries with auto-inferred split ratios, and **simple merge** for validation/testing/calibration jobs. It includes comprehensive schema alignment, data format preservation, and provenance tracking to ensure data integrity and auditability across the ML pipeline.

The script's **auto-inferred split ratios** feature (enabled by default) automatically calculates proportions from base data and applies them to augmentation distribution, eliminating manual configuration while ensuring augmentation follows base data characteristics. For example, if base data has 10K train / 2K test / 2K val samples (71.4% / 14.3% / 14.3%), the script automatically distributes augmentation samples using these exact proportions with stratified sampling to maintain class balance.

Key capabilities:
- **Split-aware merge**: Maintains train/test/val boundaries for training jobs
- **Auto-inferred ratios**: Adapts augmentation distribution to base data proportions
- **Simple merge**: Concatenation for non-training jobs (validation/testing/calibration)
- **Format preservation**: Auto-detects and maintains CSV/TSV/Parquet formats
- **Schema alignment**: Handles pseudo_label→label conversion and type compatibility
- **Provenance tracking**: Distinguishes original vs pseudo-labeled samples
- **Stratified splitting**: Maintains class balance across splits
- **Metadata generation**: Comprehensive merge operation documentation

## Purpose and Major Tasks

### Primary Purpose
Intelligently merge original labeled data with pseudo-labeled or augmented samples for Semi-Supervised Learning and Active Learning workflows while maintaining data split integrity, preserving formats, aligning schemas, and tracking provenance for audit and analysis.

### Major Tasks

1. **Input Data Loading**: Auto-detect split structure (train/test/val or single dataset) and load data with format detection (CSV/TSV/Parquet)

2. **Merge Strategy Selection**: Determine strategy based on job type and input structure (split-aware vs simple)

3. **Schema Alignment**: Convert pseudo_label to label column, extract common columns, align data types for compatibility

4. **Split Ratio Calculation**: Auto-infer proportions from base data splits (default) or use manual ratios

5. **Augmentation Distribution**: Split augmentation data using calculated ratios with stratified sampling to maintain class balance

6. **Data Merging**: Combine base and augmentation datasets while maintaining split boundaries (training) or simple concatenation (non-training)

7. **Provenance Tracking**: Add data_source column distinguishing "original" vs "pseudo_labeled" samples

8. **Format Preservation**: Save merged data in original format maintaining directory structure

9. **Metadata Generation**: Create comprehensive merge operation documentation with statistics

10. **Validation**: Verify provenance columns, split ratios, and output structure

## Script Contract

### Entry Point
```
pseudo_label_merge.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `base_data` | `/opt/ml/processing/input/base_data` | Original labeled training data |
| `augmentation_data` | `/opt/ml/processing/input/augmentation_data` | Pseudo-labeled or augmented samples |

**Input Structure - Training Job with Splits**:
```
/opt/ml/processing/input/base_data/
├── train/
│   └── train_processed_data.{csv|tsv|parquet}
├── test/
│   └── test_processed_data.{csv|tsv|parquet}
└── val/
    └── val_processed_data.{csv|tsv|parquet}

/opt/ml/processing/input/augmentation_data/
└── selected_samples.{csv|tsv|parquet}  # or predictions.*, labeled_data.*
```

**Input Structure - Non-Training Jobs**:
```
/opt/ml/processing/input/base_data/
└── {job_type}/
    └── {job_type}_processed_data.{csv|tsv|parquet}

/opt/ml/processing/input/augmentation_data/
└── selected_samples.{csv|tsv|parquet}
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `merged_data` | `/opt/ml/processing/output/merged_data` | Merged data by split with metadata |

**Output Structure - Training Job**:
```
/opt/ml/processing/output/merged_data/
├── train/
│   └── train_processed_data.{format}
├── test/
│   └── test_processed_data.{format}
├── val/
│   └── val_processed_data.{format}
└── merge_metadata.json
```

**Output Structure - Non-Training Jobs**:
```
/opt/ml/processing/output/merged_data/
├── {job_type}/
│   └── {job_type}_processed_data.{format}
└── merge_metadata.json
```

### Required Environment Variables

| Variable | Type | Description |
|----------|------|-------------|
| `LABEL_FIELD` | `str` | Name of the label column in both datasets (e.g., "label", "target", "class") |

### Optional Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_AUTO_SPLIT_RATIOS` | `bool` | `"true"` | Auto-infer split ratios from base data (RECOMMENDED) |
| `TRAIN_RATIO` | `float` | `""` (None) | Train split proportion (only used if auto-infer disabled, e.g., "0.7") |
| `TEST_VAL_RATIO` | `float` | `""` (None) | Test vs val holdout proportion (only used if auto-infer disabled, e.g., "0.5") |
| `OUTPUT_FORMAT` | `str` | `"csv"` | Output format: `csv`, `tsv`, or `parquet` |
| `ADD_PROVENANCE` | `bool` | `"true"` | Add data_source column to track origin |
| `PSEUDO_LABEL_COLUMN` | `str` | `"pseudo_label"` | Column name for pseudo-labels in augmentation data |
| `ID_FIELD` | `str` | `"id"` | ID column name for schema validation |
| `PRESERVE_CONFIDENCE` | `bool` | `"true"` | Keep confidence scores from augmentation data |
| `STRATIFY` | `bool` | `"true"` | Use stratified splits to maintain class balance |
| `RANDOM_SEED` | `int` | `"42"` | Random seed for reproducibility |

### Job Arguments

| Argument | Type | Required | Description | Choices |
|----------|------|----------|-------------|---------|
| `--job_type` | `str` | Yes | Type of merge job | `training`, `validation`, `testing`, `calibration` |

**Job Type Behavior**:
- `training`: Uses split-aware merge with auto-inferred or manual ratios, expects train/test/val structure
- `validation`: Simple concatenation merge, single dataset
- `testing`: Simple concatenation merge, single dataset
- `calibration`: Simple concatenation merge, single dataset

### Framework Dependencies

- **pandas** >= 1.3.0 (data manipulation, DataFrame operations)
- **scikit-learn** >= 1.0.0 (train_test_split for stratified splitting)

## Input Data Structure

### Expected Input Format

**Training Job - Base Data**:
```
/opt/ml/processing/input/base_data/
├── train/train_processed_data.csv     # 10,000 labeled samples
├── test/test_processed_data.csv       #  2,000 labeled samples  
└── val/val_processed_data.csv         #  2,000 labeled samples
```

**Training Job - Augmentation Data**:
```
/opt/ml/processing/input/augmentation_data/
└── selected_samples.csv                # 5,000 pseudo-labeled samples
```

**Non-Training Jobs - Base Data**:
```
/opt/ml/processing/input/base_data/validation/
└── validation_processed_data.csv       # Validation dataset
```

### Required Columns

**Base Data**:
- **Label Column**: Specified by `LABEL_FIELD` (e.g., "label", "target", "is_fraud")
- **ID Column**: Specified by `ID_FIELD` (default: "id") for tracking
- **Feature Columns**: All feature columns used for model training

**Augmentation Data**:
- **Pseudo-Label Column**: Specified by `PSEUDO_LABEL_COLUMN` (default: "pseudo_label")
  - Will be converted to label column during schema alignment
- **ID Column**: Same as base data
- **Feature Columns**: Must have overlap with base data for schema alignment
- **Confidence Scores** (optional): Preserved if `PRESERVE_CONFIDENCE=true`

### Schema Requirements

1. **Common Columns**: Base and augmentation data must have overlapping feature columns
2. **Label Compatibility**: Pseudo-labels must match base label domain (e.g., same classes)
3. **Type Compatibility**: Column types should be compatible or convertible
4. **ID Uniqueness**: IDs should be unique within each dataset (not enforced across datasets)

### Example Data

**Base Training Data** (train split):
```csv
id,age,income,category,label
1,25,50000,A,0
2,30,60000,B,1
3,35,55000,A,0
...
```

**Augmentation Data**:
```csv
id,age,income,category,pseudo_label,confidence
10001,28,52000,A,0,0.92
10002,32,58000,B,1,0.88
10003,29,54000,A,0,0.95
...
```

## Output Data Structure

### Output Directory Structure

```
/opt/ml/processing/output/merged_data/
├── train/
│   └── train_processed_data.csv     # Base train + augmentation train portion
├── test/
│   └── test_processed_data.csv      # Base test + augmentation test portion
├── val/
│   └── val_processed_data.csv       # Base val + augmentation val portion
└── merge_metadata.json              # Merge operation metadata
```

### Output Data Characteristics

**Training Job - Split-Aware Merge**:
- **Train split**: Base train (10K) + Aug train portion (3,571 samples) = 13,571 total
- **Test split**: Base test (2K) + Aug test portion (714 samples) = 2,714 total
- **Val split**: Base val (2K) + Aug val portion (715 samples) = 2,715 total
- **Ratios**: Maintains 71.4% / 14.3% / 14.3% from base data
- **Provenance**: All records have data_source column

**Non-Training Job - Simple Merge**:
- Base samples + All augmentation samples concatenated
- Single output file in job_type subdirectory
- Provenance tracking included

### Merged Output Example

**Merged Train Split**:
```csv
id,age,income,category,label,data_source
1,25,50000,A,0,original
2,30,60000,B,1,original
...
10001,28,52000,A,0,pseudo_labeled
10002,32,58000,B,1,pseudo_labeled
...
```

### Merge Metadata Output

**merge_metadata.json**:
```json
{
  "job_type": "training",
  "merge_strategy": "split_aware",
  "base_splits": {
    "train": {"count": 10000, "shape": [10000, 5]},
    "test": {"count": 2000, "shape": [2000, 5]},
    "val": {"count": 2000, "shape": [2000, 5]}
  },
  "augmentation_count": 5000,
  "merged_splits": {
    "train": {"count": 13571, "shape": [13571, 6]},
    "test": {"count": 2714, "shape": [2714, 6]},
    "val": {"count": 2715, "shape": [2715, 6]}
  },
  "configuration": {
    "label_field": "label",
    "use_auto_split_ratios": true,
    "train_ratio": null,
    "test_val_ratio": null,
    "stratify": true,
    "preserve_confidence": true,
    "random_seed": 42
  },
  "output_paths": {
    "train": "/opt/ml/processing/output/merged_data/train/train_processed_data.csv",
    "test": "/opt/ml/processing/output/merged_data/test/test_processed_data.csv",
    "val": "/opt/ml/processing/output/merged_data/val/val_processed_data.csv"
  },
  "timestamp": "2025-11-18T19:30:45.123456"
}
```

### Job-Type-Specific Behavior

**Training Job** (`job_type=training`):
- Expects: train/test/val split structure in base data
- Strategy: Split-aware merge with auto-inferred ratios
- Output: Three split directories with merged data
- Augmentation: Distributed proportionally across splits

**Validation Job** (`job_type=validation`):
- Expects: Single validation dataset
- Strategy: Simple concatenation merge
- Output: Single validation directory
- Augmentation: All samples added to validation set

**Testing Job** (`job_type=testing`):
- Expects: Single test dataset
- Strategy: Simple concatenation merge
- Output: Single test directory
- Augmentation: All samples added to test set

**Calibration Job** (`job_type=calibration`):
- Expects: Single calibration dataset
- Strategy: Simple concatenation merge
- Output: Single calibration directory
- Augmentation: All samples added to calibration set

## Key Functions and Tasks

### Data Loading Component

#### `load_dataframe_with_format(file_path: Path) -> Tuple[pd.DataFrame, str]`

**Purpose**: Load DataFrame and detect its format

**Algorithm**:
```python
1. Detect file format from extension:
   suffix = file_path.suffix.lower()
   IF suffix == ".csv": format = "csv"
   ELIF suffix == ".tsv": format = "tsv"
   ELIF suffix == ".parquet": format = "parquet"
   ELSE: RAISE RuntimeError

2. Load DataFrame based on format:
   IF format == "csv":
      df = pd.read_csv(file_path)
   ELIF format == "tsv":
      df = pd.read_csv(file_path, sep='\t')
   ELIF format == "parquet":
      df = pd.read_parquet(file_path)

3. Log loading info

4. RETURN (df, format)
```

**Returns**: Tuple of (DataFrame, format_string)

**Complexity**: O(n) where n = number of rows (file I/O dominant)

#### `_load_split_data(split_dir: Path, split_name: str) -> pd.DataFrame`

**Purpose**: Load data from a specific split directory with flexible file discovery

**Algorithm**:
```python
1. Try specific file names first:
   FOR ext in [".csv", ".tsv", ".parquet"]:
      file_path = split_dir / f"{split_name}_processed_data{ext}"
      IF file_path.exists():
         RETURN _read_file(file_path)

2. Fall back to pattern matching:
   csv_files = list(split_dir.glob("*.csv"))
   tsv_files = list(split_dir.glob("*.tsv"))
   parquet_files = list(split_dir.glob("*.parquet"))

3. Load first available file:
   IF parquet_files: RETURN _read_file(parquet_files[0])
   ELIF csv_files: RETURN _read_file(csv_files[0])
   ELIF tsv_files: RETURN _read_file(tsv_files[0])
   ELSE: RAISE FileNotFoundError
```

**File Name Priority**:
1. Exact match: `{split_name}_processed_data.{ext}`
2. First parquet file found
3. First CSV file found
4. First TSV file found

**Complexity**: O(n) for file loading after O(m) for file discovery where m = files in directory

#### `load_base_data(base_data_dir: str, job_type: str) -> Dict[str, pd.DataFrame]`

**Purpose**: Load base training data with automatic split structure detection

**Algorithm**:
```python
1. Initialize base_path = Path(base_data_dir)

2. IF job_type == "training":
   a. Check for split structure:
      train_dir = base_path / "train"
      test_dir = base_path / "test"
      val_dir = base_path / "val"
   
   b. IF all three directories exist:
      Log "Detected split structure"
      RETURN {
         "train": _load_split_data(train_dir, "train"),
         "test": _load_split_data(test_dir, "test"),
         "val": _load_split_data(val_dir, "val")
      }
   
   c. ELSE (single dataset):
      Log warning about expected splits
      RETURN {job_type: _load_single_dataset(base_path)}

3. ELSE (non-training):
   a. Try job_type subdirectory:
      job_dir = base_path / job_type
      IF job_dir.exists():
         RETURN {job_type: _load_split_data(job_dir, job_type)}
   
   b. Fall back to root directory:
      RETURN {job_type: _load_single_dataset(base_path)}
```

**Returns**: Dictionary mapping split names to DataFrames

**Example Returns**:
```python
# Training with splits
{"train": df_train, "test": df_test, "val": df_val}

# Training without splits (fallback)
{"training": df_single}

# Validation job
{"validation": df_validation}
```

#### `load_augmentation_data(aug_data_dir: str) -> pd.DataFrame`

**Purpose**: Load augmentation data (always single dataset)

**Algorithm**:
```python
1. Initialize aug_path = Path(aug_data_dir)

2. Try common file names:
   FOR filename in ["selected_samples", "predictions", "labeled_data"]:
      FOR ext in [".parquet", ".csv", ".tsv"]:
         file_path = aug_path / f"{filename}{ext}"
         IF file_path.exists():
            Log loading info
            RETURN _read_file(file_path)

3. Fall back to any data file:
   RETURN _load_single_dataset(aug_path)
```

**File Discovery Priority**:
1. `selected_samples.parquet` (from active sample selection)
2. `predictions.parquet` (from model inference)
3. `labeled_data.parquet` (from labeling steps)
4. Any other data file in directory

**Complexity**: O(n) for file loading

### Split Detection Component

#### `detect_merge_strategy(base_splits: Dict[str, pd.DataFrame], job_type: str) -> str`

**Purpose**: Determine merge strategy based on input structure

**Algorithm**:
```python
1. Check for training job with complete splits:
   IF job_type == "training" AND 
      set(base_splits.keys()) == {"train", "test", "val"}:
      Log "Using split-aware merge strategy"
      RETURN "split_aware"

2. All other cases:
   Log "Using simple merge strategy"
   RETURN "simple"
```

**Returns**: `"split_aware"` or `"simple"`

**Decision Table**:
```
Job Type   | Base Splits           | Strategy
-----------|----------------------|-------------
training   | {train, test, val}   | split_aware
training   | {training}           | simple
validation | {validation}         | simple
testing    | {testing}            | simple
calibration| {calibration}        | simple
```

**Complexity**: O(1)

#### `extract_split_ratios(base_splits: Dict[str, pd.DataFrame]) -> Dict[str, float]`

**Purpose**: Calculate split proportions from base data for auto-inferred distribution

**Algorithm**:
```python
1. Calculate total samples:
   total = sum(len(df) for df in base_splits.values())

2. Calculate proportions:
   ratios = {}
   FOR name, df in base_splits.items():
      ratios[name] = len(df) / total

3. Log extracted ratios

4. RETURN ratios
```

**Returns**: Dictionary with split proportions summing to 1.0

**Example**:
```python
# Input
base_splits = {
   "train": df_10000_rows,
   "test": df_2000_rows,
   "val": df_2000_rows
}

# Output
{
   "train": 0.714,  # 10000/14000
   "test": 0.143,   # 2000/14000
   "val": 0.143     # 2000/14000
}
```

**Complexity**: O(k) where k = number of splits (typically 3)

### Schema Alignment Component

#### `_infer_common_dtype(dtype1, dtype2)`

**Purpose**: Infer common data type for two columns with different types

**Algorithm**:
```python
1. Check for numeric types:
   IF both are numeric:
      IF either is float:
         RETURN "float64"
      ELSE:
         RETURN "int64"

2. Check for string types:
   IF either is string:
      RETURN "object"

3. Default fallback:
   RETURN dtype1
```

**Type Priority**:
- Numeric + Numeric → float64 (if either is float) or int64
- String + Any → object
- Other → first type

**Complexity**: O(1)

#### `align_schemas(base_df, aug_df, label_field, pseudo_label_column, id_field) -> Tuple[pd.DataFrame, pd.DataFrame]`

**Purpose**: Align schemas between base and augmentation data

**Algorithm**:
```python
1. Create copies to avoid modifying originals:
   base_aligned = base_df.copy()
   aug_aligned = aug_df.copy()

2. Handle pseudo-label column conversion:
   IF pseudo_label_column in aug_aligned.columns:
      IF label_field not in aug_aligned.columns:
         # Convert pseudo_label → label
         aug_aligned[label_field] = aug_aligned[pseudo_label_column]
         Log conversion
      
      # Drop pseudo_label to avoid duplication
      aug_aligned = aug_aligned.drop(columns=[pseudo_label_column])

3. Validate label field exists:
   IF label_field not in aug_aligned.columns:
      RAISE ValueError with available columns

4. Find common columns:
   common_columns = sorted(
      set(base_aligned.columns) & set(aug_aligned.columns)
   )

5. Validate essential columns:
   IF id_field not in common_columns:
      Log warning
   
   IF label_field not in common_columns:
      RAISE ValueError

6. Select only common columns:
   base_aligned = base_aligned[common_columns]
   aug_aligned = aug_aligned[common_columns]

7. Align data types:
   FOR col in common_columns:
      IF base_aligned[col].dtype != aug_aligned[col].dtype:
         TRY:
            common_type = _infer_common_dtype(
               base_aligned[col].dtype,
               aug_aligned[col].dtype
            )
            base_aligned[col] = base_aligned[col].astype(common_type)
            aug_aligned[col] = aug_aligned[col].astype(common_type)
            Log dtype alignment
         EXCEPT Exception:
            Log warning about alignment failure

8. RETURN (base_aligned, aug_aligned)
```

**Returns**: Tuple of (aligned_base_df, aligned_aug_df)

**Example**:
```python
# Before alignment
base: [id, age, income, category, label]
aug:  [id, age, income, category, pseudo_label, confidence]

# After alignment
base: [age, category, id, income, label]  # Sorted alphabetically
aug:  [age, category, id, income, label]  # pseudo_label converted, confidence dropped
```

**Complexity**: O(n×m) where n = rows, m = columns (dominant operations are column selection and type conversion)

### Merge Strategy Component

#### `_split_by_ratios(df, ratios, label_field, stratify, random_seed) -> Dict[str, pd.DataFrame]`

**Purpose**: Split DataFrame into three parts using specified ratios (core auto-inferred distribution function)

**Algorithm**:
```python
1. Normalize ratios to sum to 1.0:
   total_ratio = sum(ratios.values())
   IF NOT abs(total_ratio - 1.0) < 1e-6:
      Log warning
      ratios = {k: v/total_ratio for k, v in ratios.items()}

2. Extract split proportions:
   train_ratio = ratios.get("train", 0.7)
   test_ratio = ratios.get("test", 0.15)
   val_ratio = ratios.get("val", 0.15)

3. First split: train vs (test+val):
   IF stratify AND label_field in df.columns:
      train_df, holdout_df = train_test_split(
         df,
         train_size=train_ratio,
         random_state=random_seed,
         stratify=df[label_field]
      )
   ELSE:
      train_df, holdout_df = train_test_split(
         df,
         train_size=train_ratio,
         random_state=random_seed
      )

4. Second split: test vs val from holdout:
   test_proportion = test_ratio / (test_ratio + val_ratio)
   
   IF stratify AND label_field in holdout_df.columns:
      test_df, val_df = train_test_split(
         holdout_df,
         train_size=test_proportion,
         random_state=random_seed,
         stratify=holdout_df[label_field]
      )
   ELSE:
      test_df, val_df = train_test_split(
         holdout_df,
         train_size=test_proportion,
         random_state=random_seed
      )

5. Log split sizes

6. RETURN {"train": train_df, "test": test_df, "val": val_df}
```

**Returns**: Dictionary mapping split names to DataFrames

**Example**:
```python
# Input
df = 5000 samples
ratios = {"train": 0.714, "test": 0.143, "val": 0.143}

# Output
{
   "train": 3570 samples,  # 5000 * 0.714
   "test": 715 samples,    # 5000 * 0.143
   "val": 715 samples      # 5000 * 0.143
}
```

**Stratification Benefit**: Maintains class distribution across all splits

**Complexity**: O(n log n) for stratified splitting (sorting required)

#### `merge_with_splits(base_splits, augmentation_df, label_field, ...) -> Dict[str, pd.DataFrame]`

**Purpose**: Merge with proportional augmentation distribution across splits (split-aware merge)

**Algorithm**:
```python
1. Log merge initiation with parameters

2. Determine split strategy:
   IF use_auto_split_ratios OR train_ratio is None:
      a. Log "Auto-inferring split ratios"
      
      b. Extract ratios from base data:
         ratios = extract_split_ratios(base_splits)
      
      c. Split augmentation using actual base ratios:
         aug_splits = _split_by_ratios(
            augmentation_df,
            ratios=ratios,
            label_field=label_field,
            stratify=stratify,
            random_seed=random_seed
         )
   
   ELSE (manual ratios - backward compatibility):
      a. Log "Using manual ratios"
      
      b. Two-step split:
         IF stratify AND label_field in augmentation_df:
            aug_train, aug_holdout = train_test_split(
               augmentation_df,
               train_size=train_ratio,
               stratify=augmentation_df[label_field],
               random_state=random_seed
            )
            
            aug_test, aug_val = train_test_split(
               aug_holdout,
               test_size=test_val_ratio,
               stratify=aug_holdout[label_field],
               random_state=random_seed
            )
         ELSE:
            # Non-stratified splits
            ...
      
      c. Package splits:
         aug_splits = {
            "train": aug_train,
            "test": aug_test,
            "val": aug_val
         }

3. Log augmentation split sizes

4. Merge each split with provenance:
   merged_splits = {}
   
   FOR split_name in ["train", "test", "val"]:
      a. Create copies:
         base_df = base_splits[split_name].copy()
         aug_df = aug_splits[split_name].copy()
      
      b. Add provenance:
         base_df["data_source"] = "original"
         aug_df["data_source"] = "pseudo_labeled"
      
      c. Remove confidence columns if not preserving:
         IF NOT preserve_confidence:
            confidence_cols = [col for col in aug_df.columns 
                              if "confidence" in col.lower() 
                              or "score" in col.lower()]
            IF confidence_cols:
               aug_df = aug_df.drop(columns=confidence_cols)
      
      d. Combine:
         merged_df = pd.concat([base_df, aug_df], ignore_index=True)
      
      e. Log merge stats
      
      f. Store:
         merged_splits[split_name] = merged_df

5. RETURN merged_splits
```

**Returns**: Dictionary with merged train/test/val DataFrames

**Key Feature - Auto-Inferred Ratios**:
- Calculates actual proportions from base data
- Example: 10K train / 2K test / 2K val → 71.4% / 14.3% / 14.3%
- Applies same ratios to augmentation distribution
- Zero configuration needed

**Example**:
```python
# Base splits
base = {
   "train": 10000 samples,
   "test": 2000 samples,
   "val": 2000 samples
}

# Augmentation
aug = 5000 samples

# Auto-inferred ratios: 71.4% / 14.3% / 14.3%
# Aug distribution: 3571 train / 714 test / 715 val

# Merged result
{
   "train": 13571 samples (10K + 3571),
   "test": 2714 samples (2K + 714),
   "val": 2715 samples (2K + 715)
}
```

**Complexity**: O(n log n) for stratified splitting + O(n) for merging

#### `merge_simple(base_df, augmentation_df, preserve_confidence) -> pd.DataFrame`

**Purpose**: Simple concatenation merge for non-training jobs

**Algorithm**:
```python
1. Create copies:
   base_merged = base_df.copy()
   aug_merged = augmentation_df.copy()

2. Add provenance:
   base_merged["data_source"] = "original"
   aug_merged["data_source"] = "pseudo_labeled"

3. Remove confidence columns if not preserving:
   IF NOT preserve_confidence:
      confidence_cols = [col for col in aug_merged.columns 
                        if "confidence" or "score" in col.lower()]
      IF confidence_cols:
         aug_merged = aug_merged.drop(columns=confidence_cols)
         Log removal

4. Combine:
   merged_df = pd.concat([base_merged, aug_merged], ignore_index=True)

5. Log merge stats

6. RETURN merged_df
```

**Returns**: Merged DataFrame with provenance

**Complexity**: O(n) for concatenation

### Main Function

#### `main(input_paths, output_paths, environ_vars, job_args) -> Dict[str, pd.DataFrame]`

**Purpose**: Main orchestration function

**Algorithm**:
```python
1. Extract and validate configuration
2. Load base data (detect split structure)
3. Load augmentation data
4. Detect merge strategy
5. Perform schema alignment
6. Execute appropriate merge (split-aware or simple)
7. Save merged data
8. Generate and save metadata
9. RETURN merged_splits
```

**Returns**: Dictionary of merged DataFrames by split

## Algorithms and Data Structures

### Algorithm 1: Auto-Inferred Split Ratios

**Problem**: Distribute augmentation data to match base data proportions

**Solution**:
1. Calculate base ratios: `ratio = len(split) / total`
2. Apply to augmentation with stratification
3. Merge corresponding splits

**Benefit**: Zero configuration, adapts to any base data distribution

### Algorithm 2: Stratified Three-Way Split

**Problem**: Split data into train/test/val maintaining class balance

**Solution**:
1. First split: train vs (test+val) stratified by label
2. Second split: test vs val from holdout stratified by label
3. Return three balanced splits

**Benefit**: Prevents class imbalance across splits

### Data Structure: Merge Metadata

```python
{
    "job_type": str,
    "merge_strategy": "split_aware" | "simple",
    "base_splits": Dict[str, {"count": int, "shape": list}],
    "augmentation_count": int,
    "merged_splits": Dict[str, {"count": int, "shape": list}],
    "configuration": Dict[str, any],
    "output_paths": Dict[str, str],
    "timestamp": str (ISO format)
}
```

## Performance Characteristics

| Operation | Time (10K base + 5K aug) |
|-----------|--------------------------|
| Load data | 0.5-1.0 sec |
| Extract ratios | <0.01 sec |
| Split augmentation | 0.1-0.2 sec |
| Schema alignment | 0.05-0.1 sec |
| Merge splits | 0.2-0.3 sec |
| Save output | 0.5-1.0 sec |
| **Total** | **2-4 seconds** |

**Memory**: ~2× input data (copies during merge)

## Example Configurations

### Example 1: Auto-Inferred Ratios (Recommended)
```bash
export LABEL_FIELD="label"
export USE_AUTO_SPLIT_RATIOS="true"  # DEFAULT

python pseudo_label_merge.py --job_type training
```

### Example 2: Manual Ratios
```bash
export LABEL_FIELD="target"
export USE_AUTO_SPLIT_RATIOS="false"
export TRAIN_RATIO="0.8"
export TEST_VAL_RATIO="0.5"

python pseudo_label_merge.py --job_type training
```

### Example 3: Simple Merge
```bash
export LABEL_FIELD="label"

python pseudo_label_merge.py --job_type validation
```

## Integration Patterns

### SSL Pipeline Flow
```
TabularPreprocessing → XGBoostTraining (pretrain) → 
XGBoostInference → ActiveSampleSelection → 
PseudoLabelMerge → XGBoostTraining (fine-tune)
```

### Active Learning Flow
```
Initial Training → Inference → Selection → 
Manual Labeling → PseudoLabelMerge → Retrain
```

## Best Practices

1. **Use Auto-Inferred Ratios**: Adapts to your data automatically
2. **Enable Stratification**: Maintains class balance
3. **Preserve Confidence**: Useful for analysis
4. **Track Provenance**: Essential for debugging
5. **Validate Metadata**: Check output statistics

## Troubleshooting

**Issue**: Label field not found  
**Solution**: Check `PSEUDO_LABEL_COLUMN` setting

**Issue**: No common columns  
**Solution**: Verify feature names match

**Issue**: Ratios don't sum to 1.0  
**Solution**: Script auto-normalizes, check logs

## References

- **Design**: [Pseudo Label Merge Script Design](../1_design/pseudo_label_merge_script_design.md)
- **Contract**: `src/cursus/steps/contracts/pseudo_label_merge_contract.py`
- **Related**: [Active Sample Selection](./active_sample_selection_script.md)

---

**Document Metadata**:
- **Author**: Cursus Framework Team
- **Last Updated**: 2025-11-18
- **Script Version**: 2025-11-18
- **Documentation Version**: 1.0
