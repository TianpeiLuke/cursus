---
tags:
  - code
  - processing_script
  - missing_values
  - imputation
  - data_preprocessing
  - statistical_methods
keywords:
  - missing value imputation
  - mean imputation
  - median imputation
  - mode imputation
  - categorical imputation
  - text imputation
  - pandas-safe values
  - parameter accumulator
  - format preservation
topics:
  - data preprocessing
  - missing data handling
  - statistical imputation
language: python
date of note: 2025-11-18
---

# Missing Value Imputation Script Documentation

## Overview

The `missing_value_imputation.py` script implements comprehensive missing value handling using simple statistical methods (mean, median, mode, constant) for tabular data. The script intelligently handles numerical, categorical, and text data types with appropriate imputation strategies while ensuring pandas-safe fill values that won't be misinterpreted as NA/NULL. It operates within the preprocessing pipeline, supporting both training mode (fit and transform) and inference mode (transform only with pre-trained parameters).

The script implements the **parameter accumulator pattern**, copying existing preprocessing artifacts from upstream steps and adding its own imputation parameters, enabling downstream steps to access all preprocessing parameters from a single location. It auto-detects data formats (CSV/TSV/Parquet) and preserves them throughout processing, maintaining consistency across the pipeline.

Key capabilities:
- **Multi-type imputation**: Numerical (mean/median/constant), categorical (mode/constant), text (mode/constant/empty)
- **Pandas-safe validation**: Prevents fill values that pandas interprets as NA/NULL
- **Auto-detection**: Distinguishes categorical from text data based on unique value ratio
- **Training mode**: Fits on train split, transforms all splits, saves parameters
- **Inference mode**: Uses pre-trained parameters for consistent transformation
- **Parameter accumulator**: Copies and augments upstream preprocessing artifacts
- **Format preservation**: Auto-detects and maintains CSV/TSV/Parquet formats
- **Comprehensive reporting**: Quality metrics, recommendations, and detailed logs

## Purpose and Major Tasks

### Primary Purpose
Handle missing values in tabular data using simple statistical imputation methods, providing consistent and reproducible transformations across training and inference workflows while ensuring pandas-safe imputation values and maintaining all upstream preprocessing parameters.

### Major Tasks

1. **Configuration Loading**: Extract imputation strategies, fill values, and validation settings from environment variables including column-specific overrides

2. **Format Detection and Data Loading**: Auto-detect data format (CSV/TSV/Parquet) and load data from split-based directory structure

3. **Missing Value Analysis**: Comprehensive analysis of missing patterns, data types, and recommendations for imputation strategies

4. **Artifact Accumulation**: Copy all existing preprocessing artifacts from upstream steps (parameter accumulator pattern)

5. **Data Type Detection**: Auto-detect numerical, categorical, and text types with configurable unique value ratio threshold

6. **Strategy Selection**: Choose appropriate imputation strategy per column based on data type and configuration

7. **Imputation Execution**:
   - **Training mode**: Fit imputation parameters on training data, transform all splits
   - **Inference mode**: Load pre-trained parameters, transform data consistently

8. **Pandas-Safe Validation**: Validate fill values against pandas NA interpretation list and replace problematic values

9. **Artifact Saving**: Save imputation parameters in XGBoost-compatible format (impute_dict.pkl)

10. **Report Generation**: Generate comprehensive quality metrics, recommendations, and human-readable summaries

## Script Contract

### Entry Point
```
missing_value_imputation.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `input_data` | `/opt/ml/processing/input/data` | Preprocessed data from tabular_preprocessing |
| `model_artifacts_input` | `/opt/ml/processing/input/model_artifacts` | Pre-trained parameters (non-training jobs only) |

**Input Structure**:
```
/opt/ml/processing/input/data/
├── train/
│   └── train_processed_data.{csv|tsv|parquet}
├── val/
│   └── val_processed_data.{csv|tsv|parquet}
└── test/
    └── test_processed_data.{csv|tsv|parquet}

/opt/ml/processing/input/model_artifacts/  # Non-training jobs only
├── impute_dict.pkl                         # XGBoost-compatible format
├── imputation_summary.json
└── [other artifacts from upstream steps]
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `processed_data` | `/opt/ml/processing/output/data` | Imputed data by split |
| `model_artifacts_output` | `/opt/ml/processing/output/model_artifacts` | Imputation parameters + accumulated artifacts |

**Output Structure**:
```
/opt/ml/processing/output/
├── data/
│   ├── train/
│   │   └── train_processed_data.{format}
│   ├── val/
│   │   └── val_processed_data.{format}
│   └── test/
│       └── test_processed_data.{format}
├── model_artifacts/
│   ├── impute_dict.pkl                    # Simple dict {column: value}
│   ├── imputation_summary.json
│   └── [accumulated artifacts from upstream]
├── imputation_report.json
└── imputation_summary.txt
```

### Required Environment Variables

| Variable | Type | Description |
|----------|------|-------------|
| `LABEL_FIELD` | `str` | Target column name to exclude from imputation |

### Optional Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEFAULT_NUMERICAL_STRATEGY` | `str` | `"mean"` | Default imputation strategy for numerical columns: `mean`, `median`, `constant` |
| `DEFAULT_CATEGORICAL_STRATEGY` | `str` | `"mode"` | Default strategy for categorical: `mode`, `constant` |
| `DEFAULT_TEXT_STRATEGY` | `str` | `"mode"` | Default strategy for text: `mode`, `constant`, `empty` |
| `NUMERICAL_CONSTANT_VALUE` | `float` | `"0"` | Constant value for numerical constant imputation |
| `CATEGORICAL_CONSTANT_VALUE` | `str` | `"Unknown"` | Constant value for categorical constant imputation |
| `TEXT_CONSTANT_VALUE` | `str` | `"Unknown"` | Constant value for text constant imputation |
| `CATEGORICAL_PRESERVE_DTYPE` | `bool` | `"true"` | Preserve categorical dtype after imputation |
| `AUTO_DETECT_CATEGORICAL` | `bool` | `"true"` | Enable automatic categorical vs text detection |
| `CATEGORICAL_UNIQUE_RATIO_THRESHOLD` | `float` | `"0.1"` | Threshold for categorical detection (unique/total < threshold) |
| `VALIDATE_FILL_VALUES` | `bool` | `"true"` | Validate fill values aren't pandas NA values |
| `EXCLUDE_COLUMNS` | `str` | `""` | Comma-separated list of columns to exclude |
| `COLUMN_STRATEGY_<column>` | `str` | N/A | Column-specific strategy override (e.g., `COLUMN_STRATEGY_age=median`) |

### Job Arguments

| Argument | Type | Required | Description | Choices |
|----------|------|----------|-------------|---------|
| `--job_type` | `str` | Yes | Type of processing job | `training`, `validation`, `testing`, `calibration` |

### Framework Dependencies

- **pandas** >= 1.3.0 (data manipulation)
- **numpy** >= 1.21.0 (numerical operations)
- **scikit-learn** >= 1.0.0 (SimpleImputer implementation)

## Input Data Structure

### Expected Input Format

**Training Job**:
```
/opt/ml/processing/input/data/
├── train/train_processed_data.{csv|tsv|parquet}
├── val/val_processed_data.{csv|tsv|parquet}
└── test/test_processed_data.{csv|tsv|parquet}

/opt/ml/processing/input/model_artifacts/  # Not used in training
```

**Non-Training Jobs** (validation/testing/calibration):
```
/opt/ml/processing/input/data/
└── {job_type}/{job_type}_processed_data.{csv|tsv|parquet}

/opt/ml/processing/input/model_artifacts/
├── impute_dict.pkl                         # REQUIRED
├── imputation_summary.json
└── [other accumulated artifacts]
```

### Required Columns

- **Label Field**: Specified by `LABEL_FIELD` environment variable
  - Automatically excluded from imputation
  - Examples: target, label, is_fraud, click

### Optional Columns

All feature columns are candidates for imputation based on:
- Presence of missing values
- Data type (numerical/categorical/text)
- Configuration (exclude list, column-specific strategies)

### Column Classification

**Numerical Columns**:
- Detected via `pd.api.types.is_numeric_dtype()`
- Examples: age, amount, score, count
- Strategies: mean, median, constant

**Categorical Columns**:
- Detected via `pd.api.types.is_categorical_dtype()` OR
- Auto-detected from object dtype if `unique_ratio < threshold` (default 0.1)
- Examples: category, region, product_type (low cardinality)
- Strategies: mode, constant

**Text Columns**:
- Object dtype with `unique_ratio >= threshold`
- Examples: description, comments, user_id (high cardinality)
- Strategies: mode, constant, empty

### Input Data Requirements

1. **Format Consistency**: All splits must use same format (CSV, TSV, or Parquet)
2. **Label Field**: Must exist in data (will be excluded from imputation)
3. **Missing Values**: At least one column should have missing values for meaningful imputation
4. **Data Quality**: Should not have excessive missing values (>50% triggers warning)

## Output Data Structure

### Output Directory Structure

```
/opt/ml/processing/output/
├── data/
│   ├── train/
│   │   └── train_processed_data.{format}
│   ├── val/
│   │   └── val_processed_data.{format}
│   └── test/
│       └── test_processed_data.{format}
├── model_artifacts/
│   ├── impute_dict.pkl                    # XGBoost-compatible format
│   ├── imputation_summary.json
│   └── [accumulated upstream artifacts]
├── imputation_report.json
└── imputation_summary.txt
```

### Output Data Characteristics

- **Schema**: Identical to input (all columns preserved)
- **Format**: Same as input (CSV/TSV/Parquet)
- **Missing Values**: Eliminated in imputed columns (may remain in excluded columns)
- **Data Types**: Preserved (numerical remains numerical, categorical preserved if configured)

**Example Imputation**:

*Input (train split)*:
```
   age  income  category  description
0  25.0   50000  A        Good product
1  NaN    60000  B        Excellent
2  30.0   NaN    NaN      Great
3  NaN    55000  A        NaN
```

*Output (mean for numerical, mode for categorical, mode for text)*:
```
   age  income  category  description
0  25.0   50000  A        Good product
1  27.5   60000  B        Excellent
2  30.0   55000  A        Great
3  27.5   55000  A        Good product
```

### Imputation Parameters Output

**impute_dict.pkl** (XGBoost-compatible format):
```python
{
    'age': 27.5,           # Mean imputation value
    'income': 55000.0,     # Mean imputation value  
    'category': 'A',       # Mode imputation value
    'description': 'Good product'  # Mode imputation value
}
```

This simple dictionary format is:
- Compatible with XGBoost training script expectations
- Easily loadable with pickle
- Used directly with pandas `fillna()` in non-training modes

### Job-Type-Specific Output

**Training Job**:
- Fits on: train split
- Transforms: train, val, test splits
- Saves: imputation parameters + comprehensive reports
- Copies: existing upstream artifacts (parameter accumulator)

**Non-Training Jobs**:
- Loads: pre-trained parameters from model_artifacts_input
- Transforms: only the specified job_type split
- Saves: transformed data (no new parameters)
- Example: `job_type=validation` only imputes validation split

## Key Functions and Tasks

### Configuration Management Component

#### `load_imputation_config(environ_vars)`

**Purpose**: Load and parse imputation configuration from environment variables

**Algorithm**:
```python
1. Initialize config dict with defaults
2. Parse default strategies:
   - DEFAULT_NUMERICAL_STRATEGY (default: "mean")
   - DEFAULT_CATEGORICAL_STRATEGY (default: "mode")
   - DEFAULT_TEXT_STRATEGY (default: "mode")
3. Parse constant fill values:
   - NUMERICAL_CONSTANT_VALUE (default: 0)
   - CATEGORICAL_CONSTANT_VALUE (default: "Unknown")
   - TEXT_CONSTANT_VALUE (default: "Unknown")
4. Parse validation and detection settings:
   - VALIDATE_FILL_VALUES (default: true)
   - AUTO_DETECT_CATEGORICAL (default: true)
   - CATEGORICAL_UNIQUE_RATIO_THRESHOLD (default: 0.1)
5. Parse exclude columns (comma-separated)
6. Parse column-specific strategies:
   FOR each env var starting with "COLUMN_STRATEGY_":
       Extract column name from key
       Store strategy in column_strategies dict
7. Return comprehensive config dict
```

**Returns**: Configuration dictionary with all imputation settings

### Data Type Detection Component

#### `detect_column_type(df, column, config)`

**Purpose**: Enhanced data type detection for appropriate imputation strategy selection

**Algorithm**:
```python
1. IF pd.api.types.is_numeric_dtype(df[column]):
   RETURN "numerical"

2. IF pd.api.types.is_categorical_dtype(df[column]):
   RETURN "categorical"

3. IF df[column].dtype == "object":
   IF config.auto_detect_categorical:
      a. Calculate unique value ratio:
         non_null_count = df[column].dropna().shape[0]
         unique_ratio = df[column].nunique() / non_null_count
      
      b. Compare with threshold:
         IF unique_ratio < config.threshold:
            RETURN "categorical"  # Low cardinality → categorical
         ELSE:
            RETURN "text"         # High cardinality → text
   ELSE:
      RETURN "text"  # Auto-detection disabled

4. ELSE (other dtypes):
   RETURN "text"  # Default fallback
```

**Example Classifications**:
```
Column: 'age' (dtype: float64) → "numerical"
Column: 'category' (dtype: object, 3 unique out of 1000) → "categorical" (ratio=0.003)
Column: 'user_id' (dtype: object, 950 unique out of 1000) → "text" (ratio=0.95)
Column: 'region' (dtype: category) → "categorical" (explicit dtype)
```

**Complexity**: O(1) for numerical/categorical, O(n) for object type with auto-detection

### Pandas-Safe Validation Component

#### `get_pandas_na_values()`

**Purpose**: Get set of values that pandas interprets as NA/NULL

**Returns**:
```python
{
    "N/A", "NA", "NULL", "NaN", "nan", "NAN",
    "#N/A", "#N/A N/A", "#NA",
    "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN",
    "<NA>", "null", "Null", "none", "None", "NONE"
}
```

**Usage**: Validate fill values won't be misinterpreted as missing

#### `validate_text_fill_value(value)`

**Purpose**: Validate that a text fill value won't be interpreted as NA by pandas

**Algorithm**:
```python
1. Get pandas NA values set
2. Check if value in pandas_na_values
3. RETURN (value not in pandas_na_values)
```

**Example**:
```python
validate_text_fill_value("Unknown")  # True - safe
validate_text_fill_value("NA")       # False - will be interpreted as NA
validate_text_fill_value("null")     # False - will be interpreted as NA
```

### Strategy Management Component

#### `ImputationStrategyManager` Class

**Purpose**: Manage imputation strategy selection and creation for different data types

**Attributes**:
- `config`: Imputation configuration dictionary
- `pandas_na_values`: Set of pandas NA interpretation values

#### `ImputationStrategyManager.get_strategy_for_column(df, column)`

**Purpose**: Select and create appropriate imputation strategy for a column

**Algorithm**:
```python
1. Detect column type:
   column_type = detect_column_type(df, column, config)

2. Check for explicit column-specific strategy:
   IF column in config.column_strategies:
      strategy_name = config.column_strategies[column]
      RETURN _create_strategy_from_name(df, column, column_type, strategy_name)

3. Auto-select based on detected type:
   IF column_type == "numerical":
      default_strategy = config.default_numerical_strategy
   ELIF column_type == "categorical":
      default_strategy = config.default_categorical_strategy
   ELSE:  # text
      default_strategy = config.default_text_strategy

4. Create and return strategy:
   RETURN _create_strategy_from_name(df, column, column_type, default_strategy)
```

**Returns**: Configured `SimpleImputer` instance

#### `ImputationStrategyManager._create_numerical_strategy(strategy_name)`

**Purpose**: Create numerical imputation strategy

**Algorithm**:
```python
1. IF strategy_name == "mean":
   RETURN SimpleImputer(strategy="mean")

2. ELIF strategy_name == "median":
   RETURN SimpleImputer(strategy="median")

3. ELIF strategy_name == "constant":
   fill_value = config.numerical_constant_value
   RETURN SimpleImputer(strategy="constant", fill_value=fill_value)

4. ELSE:
   Log warning about unknown strategy
   RETURN SimpleImputer(strategy="mean")  # Safe default
```

**Complexity**: O(1) - strategy creation

#### `ImputationStrategyManager._create_categorical_strategy(df, column, strategy_name)`

**Purpose**: Create categorical imputation strategy with pandas-safe validation

**Algorithm**:
```python
1. IF strategy_name == "mode":
   RETURN SimpleImputer(strategy="most_frequent")

2. ELIF strategy_name == "constant":
   fill_value = config.categorical_constant_value
   
   # Pandas-safe validation
   IF config.validate_fill_values AND fill_value in pandas_na_values:
      Log warning
      fill_value = "Missing"  # Safe replacement
   
   RETURN SimpleImputer(strategy="constant", fill_value=fill_value)

3. ELSE:
   Log warning
   RETURN SimpleImputer(strategy="most_frequent")  # Safe default
```

**Pandas-Safe Protection**: Automatically replaces problematic values like "NA", "null", "None"

#### `ImputationStrategyManager._create_text_strategy(strategy_name)`

**Purpose**: Create text-specific imputation strategy with pandas-safe validation

**Algorithm**:
```python
1. IF strategy_name == "mode":
   RETURN SimpleImputer(strategy="most_frequent")

2. ELIF strategy_name == "constant":
   fill_value = config.text_constant_value
   
   # Pandas-safe validation
   IF config.validate_fill_values AND fill_value in pandas_na_values:
      Log warning
      fill_value = "Unknown"  # Safe replacement
   
   RETURN SimpleImputer(strategy="constant", fill_value=fill_value)

3. ELIF strategy_name == "empty":
   RETURN SimpleImputer(strategy="constant", fill_value="")

4. ELSE:
   Log warning
   RETURN SimpleImputer(strategy="most_frequent")  # Safe default
```

**Strategy Options**:
- `"mode"`: Most frequent value
- `"constant"`: User-specified constant (validated for pandas safety)
- `"empty"`: Empty string (useful for text fields where blank is meaningful)

### Imputation Engine Component

#### `SimpleImputationEngine` Class

**Purpose**: Core engine for simple statistical imputation methods with fit/transform pattern

**Attributes**:
- `strategy_manager`: ImputationStrategyManager instance
- `label_field`: Target column to exclude
- `fitted_imputers`: Dict mapping columns to fitted SimpleImputer instances
- `imputation_statistics`: Dict storing statistics per column
- `last_transformation_log`: Log of most recent transformation

#### `SimpleImputationEngine.fit(df)`

**Purpose**: Fit imputation parameters on training data

**Algorithm**:
```python
1. Identify columns to impute:
   exclude_cols = [label_field] + config.exclude_columns
   imputable_columns = [col for col in df.columns 
                       if col not in exclude_cols 
                       AND df[col].isnull().any()]

2. Log imputable columns

3. FOR each column in imputable_columns:
   a. Get appropriate strategy:
      imputer = strategy_manager.get_strategy_for_column(df, column)
   
   b. Fit the imputer:
      column_data = df[[column]]
      imputer.fit(column_data)
   
   c. Store fitted imputer:
      fitted_imputers[column] = imputer
   
   d. Store statistics:
      imputation_statistics[column] = {
          'strategy': imputer.strategy,
          'fill_value': getattr(imputer, 'fill_value', None),
          'statistics': getattr(imputer, 'statistics_', None),
          'missing_count_training': df[column].isnull().sum(),
          'missing_percentage_training': (missing / len(df)) * 100,
          'data_type': str(df[column].dtype)
      }
   
   e. Log fitted imputer info
```

**Time Complexity**: O(n×k) where n=rows, k=imputable columns

**Space Complexity**: O(k) for storing fitted imputers

#### `SimpleImputationEngine.transform(df)`

**Purpose**: Apply fitted imputation to data

**Algorithm**:
```python
1. Create copy of DataFrame:
   df_imputed = df.copy()

2. Initialize transformation log:
   transformation_log = {}

3. FOR each (column, imputer) in fitted_imputers.items():
   IF column exists in df_imputed:
      a. Count missing before:
         missing_before = df_imputed[column].isnull().sum()
      
      b. IF missing_before > 0:
         - Apply imputation:
           column_data = df_imputed[[column]]
           imputed_data = imputer.transform(column_data)
           df_imputed[column] = imputed_data[:, 0]
         
         - Count missing after:
           missing_after = df_imputed[column].isnull().sum()
         
         - Log transformation:
           transformation_log[column] = {
               'missing_before': missing_before,
               'missing_after': missing_after,
               'imputed_count': missing_before - missing_after,
               'strategy_used': imputer.strategy
           }
         
         - Log progress
      
      c. ELSE:
         - Log no missing values

4. Store transformation log

5. RETURN df_imputed
```

**Time Complexity**: O(n×k) where n=rows, k=imputed columns

**Properties**:
- Non-destructive (returns copy)
- Handles columns missing in new data gracefully
- Logs detailed transformation statistics

#### `SimpleImputationEngine.fit_transform(df)`

**Purpose**: Fit imputation parameters and transform data in one step

**Algorithm**:
```python
1. Call fit(df) to learn parameters
2. Call transform(df) to apply imputation
3. RETURN transformed DataFrame
```

**Usage**: Convenience method for training workflow

#### `SimpleImputationEngine.get_imputation_summary()`

**Purpose**: Get comprehensive summary of imputation process

**Returns**:
```python
{
    'fitted_columns': ['age', 'income', 'category'],
    'imputation_statistics': {
        'age': {
            'strategy': 'mean',
            'statistics': [27.5],
            'missing_count_training': 150,
            'missing_percentage_training': 15.0,
            'data_type': 'float64'
        },
        ...
    },
    'last_transformation_log': {
        'age': {
            'missing_before': 150,
            'missing_after': 0,
            'imputed_count': 150,
            'strategy_used': 'mean'
        },
        ...
    },
    'total_imputers': 3
}
```

### Missing Value Analysis Component

#### `analyze_missing_values(df)`

**Purpose**: Comprehensive missing value analysis for imputation planning

**Algorithm**:
```python
1. Initialize analysis dict:
   missing_analysis = {
       'total_records': len(df),
       'columns_with_missing': {},
       'missing_patterns': {},
       'data_types': {},
       'imputation_recommendations': {}
   }

2. FOR each column in df.columns:
   a. Calculate missing statistics:
      missing_count = df[column].isnull().sum()
      missing_percentage = (missing_count / len(df)) * 100
   
   b. IF missing_count > 0:
      - Store column stats:
        columns_with_missing[column] = {
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
            'data_type': str(df[column].dtype),
            'unique_values': df[column].nunique(),
            'sample_values': df[column].dropna().head(5).tolist()
        }
      
      - Generate recommendation:
        IF numeric:
            IF abs(skewness) > 1:
                recommend "median"  # Skewed distribution
            ELSE:
                recommend "mean"    # Normal distribution
        ELSE:
            recommend "mode"
   
   c. Store data type

3. Analyze missing patterns:
   missing_per_record = df.isnull().sum(axis=1)
   missing_patterns = {
       'records_with_no_missing': (missing_per_record == 0).sum(),
       'records_with_missing': (missing_per_record > 0).sum(),
       'max_missing_per_record': missing_per_record.max(),
       'avg_missing_per_record': missing_per_record.mean()
   }

4. RETURN missing_analysis
```

**Output Example**:
```python
{
    'total_records': 1000,
    'columns_with_missing': {
        'age': {
            'missing_count': 150,
            'missing_percentage': 15.0,
            'data_type': 'float64',
            'unique_values': 45,
            'sample_values': [25, 30, 35, 40, 45]
        }
    },
    'missing_patterns': {
        'records_with_no_missing': 800,
        'records_with_missing': 200,
        'max_missing_per_record': 3,
        'avg_missing_per_record': 0.6
    },
    'imputation_recommendations': {
        'age': 'mean',
        'category': 'mode'
    }
}
```

**Complexity**: O(n×m) where n=rows, m=columns

#### `validate_imputation_data(df, label_field, exclude_columns)`

**Purpose**: Validate data for imputation processing

**Algorithm**:
```python
1. Initialize validation report:
   validation_report = {
       'is_valid': True,
       'errors': [],
       'warnings': [],
       'imputable_columns': [],
       'excluded_columns': exclude_columns.copy()
   }

2. Validate label field:
   IF label_field in df.columns:
      Add to excluded_columns
   ELSE:
      Add warning

3. Identify imputable columns:
   FOR each column in df.columns:
      IF column not excluded AND has missing values:
         Add to imputable_columns

4. Check if any columns need imputation:
   IF no imputable_columns:
      Add warning

5. RETURN validation_report
```

**Returns**: Validation report with errors, warnings, and imputable columns list

### Artifact Management Component

#### `save_imputation_artifacts(imputation_engine, imputation_config, output_path)`

**Purpose**: Save imputation artifacts in XGBoost-compatible format

**Algorithm**:
```python
1. Extract simple imputation dictionary:
   impute_dict = {}
   FOR each (column, imputer) in fitted_imputers:
      IF imputer has statistics_:
         # For mean/median/mode strategies
         impute_dict[column] = float(imputer.statistics_[0])
      ELIF imputer has fill_value:
         # For constant strategy
         impute_dict[column] = imputer.fill_value
      ELSE:
         Log warning

2. Save imputation dictionary (XGBoost format):
   params_path = output_path / "impute_dict.pkl"
   WITH open(params_path, 'wb') as f:
      pkl.dump(impute_dict, f)
   
   Log save location and format

3. Save human-readable summary:
   summary = imputation_engine.get_imputation_summary()
   summary_path = output_path / "imputation_summary.json"
   WITH open(summary_path, 'w') as f:
      json.dump(summary, f, indent=2, default=str)
   
   Log save location
```

**Output Format**:
```python
# impute_dict.pkl
{
    'age': 27.5,
    'income': 55000.0,
    'category': 'A',
    'description': 'Good product'
}
```

**Compatibility**: Format matches XGBoost training script expectations

#### `load_imputation_parameters(imputation_params_path)`

**Purpose**: Load imputation parameters from pickle file

**Algorithm**:
```python
1. Validate file exists:
   IF NOT imputation_params_path.exists():
      RAISE FileNotFoundError

2. Load parameters:
   WITH open(imputation_params_path, 'rb') as f:
      impute_dict = pkl.load(f)

3. Validate format:
   IF NOT isinstance(impute_dict, dict):
      RAISE ValueError

4. Log loaded parameters count

5. RETURN impute_dict
```

**Expected Format**: Simple dictionary `{column_name: imputation_value}`

**Compatibility**: Loads XGBoost training output format

#### `copy_existing_artifacts(src_dir, dst_dir)`

**Purpose**: Copy all existing model artifacts from previous processing steps (parameter accumulator pattern)

**Algorithm**:
```python
1. Validate source directory:
   IF NOT src_dir OR NOT exists(src_dir):
      Log "no existing artifacts"
      RETURN

2. Create destination directory:
   makedirs(dst_dir, exist_ok=True)

3. Copy all files:
   copied_count = 0
   FOR each filename in listdir(src_dir):
      IF is_file(src_file):
         Copy src_file to dst_file
         copied_count += 1
         Log copied artifact

4. Log total copied count
```

**Purpose**: Enables parameter accumulator pattern where each preprocessing step:
1. Copies artifacts from previous steps
2. Adds its own artifacts
3. Passes all artifacts to the next step

**Benefits**:
- Downstream steps can access all preprocessing parameters
- Single source of truth for all transformations
- Simplifies deployment (one artifact directory)

### Data I/O Components

#### `_detect_file_format(split_dir, split_name)`

**Purpose**: Auto-detect data file format

**Algorithm**:
```python
1. Define format priority:
   formats = [
       (f"{split_name}_processed_data.csv", "csv"),
       (f"{split_name}_processed_data.tsv", "tsv"),
       (f"{split_name}_processed_data.parquet", "parquet")
   ]

2. FOR each (filename, format) in formats:
   a. Construct file_path = split_dir / filename
   b. IF file_path.exists():
      - RETURN (file_path, format)

3. IF no file found:
   - Raise RuntimeError with formats tried
```

**Format Detection Priority**: CSV → TSV → Parquet

#### `load_split_data(job_type, input_dir)`

**Purpose**: Load data according to job_type with automatic format detection

**Algorithm**:
```python
1. Initialize result dict

2. IF job_type == "training":
   a. FOR each split in ["train", "test", "val"]:
      - Detect format and file path
      - Load data based on format
      - Store in result dict
   
   b. Store detected format in result["_format"]
   c. Log loaded shapes

3. ELSE (non-training):
   a. Detect format and file path for job_type split
   b. Load data based on format
   c. Store in result dict
   d. Store detected format in result["_format"]
   e. Log loaded shape

4. RETURN result dict
```

**Returns**: Dictionary with DataFrames and `_format` key storing detected format

#### `save_output_data(job_type, output_dir, data_dict)`

**Purpose**: Save processed data according to job_type, preserving input format

**Algorithm**:
```python
1. Extract format from data_dict:
   output_format = data_dict.get("_format", "csv")

2. FOR each (split_name, df) in data_dict.items():
   IF split_name == "_format":
      CONTINUE  # Skip metadata
   
   a. Create split output directory
   
   b. Save in detected format:
      IF format == "csv":
         df.to_csv(path, index=False)
      ELIF format == "tsv":
         df.to_csv(path, sep='\t', index=False)
      ELIF format == "parquet":
         df.to_parquet(path, index=False)
   
   c. Log save location and shape
```

**Format Preservation**: Output format matches input format

### Main Processing Logic

#### `process_data(data_dict, label_field, job_type, imputation_config, imputation_parameters)`

**Purpose**: Core data processing logic for missing value imputation

**Algorithm**:
```python
1. Initialize strategy manager and imputation engine:
   strategy_manager = ImputationStrategyManager(imputation_config)
   imputation_engine = SimpleImputationEngine(strategy_manager, label_field)

2. IF job_type == "training":
   a. Log "training mode"
   
   b. Fit imputation parameters on training data:
      imputation_engine.fit(data_dict["train"])
   
   c. Transform all splits:
      transformed_data = {}
      FOR each (split_name, df) in data_dict.items():
         IF split_name != "_format":
            df_imputed = imputation_engine.transform(df)
            transformed_data[split_name] = df_imputed
            Log shape

3. ELSE (non-training):
   a. Validate imputation_parameters provided
   
   b. Log "using pre-fitted parameters"
   
   c. Transform using simple fillna:
      transformed_data = {}
      FOR each (split_name, df) in data_dict.items():
         IF split_name != "_format":
            df_imputed = df.copy()
            FOR each (column, value) in imputation_parameters.items():
               IF column in df_imputed:
                  df_imputed[column] = df_imputed[column].fillna(value)
            
            transformed_data[split_name] = df_imputed
            Log shape
   
   d. Create minimal engine for consistency:
      imputation_engine.imputation_statistics = {...}

4. RETURN (transformed_data, imputation_engine)
```

**Key Logic**:
- **Training**: Fit on train, transform all splits, save parameters
- **Inference**: Load parameters, transform with simple fillna

#### `internal_main(job_type, input_dir, output_dir, imputation_config, label_field, model_artifacts_input_dir, model_artifacts_output_dir, load_data_func, save_data_func)`

**Purpose**: Main logic for missing value imputation, handling both training and inference modes

**Algorithm**:
```python
1. Create output directories:
   output_path.mkdir(parents=True, exist_ok=True)
   artifacts_output_dir.mkdir(parents=True, exist_ok=True)

2. Log configuration

3. Copy existing artifacts (parameter accumulator pattern):
   IF model_artifacts_input_dir:
      copy_existing_artifacts(model_artifacts_input_dir, artifacts_output_dir)

4. Load data according to job type:
   data_dict = load_data_func(job_type, input_dir)

5. Load imputation parameters if needed (non-training):
   IF job_type != "training" AND model_artifacts_input_dir:
      imputation_params_path = Path(model_artifacts_input_dir) / "impute_dict.pkl"
      IF exists:
         imputation_parameters = load_imputation_parameters(...)
      ELSE:
         Log warning

6. Process the data:
   transformed_data, imputation_engine = process_data(...)

7. Save processed data:
   save_data_func(job_type, output_dir, transformed_data)

8. Save fitted artifacts (training only):
   IF job_type == "training":
      save_imputation_artifacts(imputation_engine, imputation_config, artifacts_output_dir)

9. Generate comprehensive report:
   missing_analysis = analyze_missing_values(sample_df)
   validation_report = validate_imputation_data(sample_df, label_field)
   generate_imputation_report(...)

10. Log completion

11. RETURN (transformed_data, imputation_engine)
```

**Dependency Injection**: Accepts load_data_func and save_data_func for testability

#### `main(input_paths, output_paths, environ_vars, job_args)`

**Purpose**: Standardized main entry point for missing value imputation script

**Algorithm**:
```python
1. Validate required paths:
   IF "input_data" not in input_paths:
      RAISE ValueError
   IF "processed_data" not in output_paths:
      RAISE ValueError

2. Validate job_args:
   IF job_args is None OR not hasattr(job_args, "job_type"):
      RAISE ValueError

3. Extract paths:
   job_type = job_args.job_type
   input_dir = input_paths["input_data"]
   output_dir = output_paths["processed_data"]
   model_artifacts_input_dir = input_paths.get("model_artifacts_input")
   model_artifacts_output_dir = output_paths.get("model_artifacts_output")

4. Log input/output paths

5. Load imputation configuration:
   imputation_config = load_imputation_config(environ_vars)
   label_field = environ_vars.get("LABEL_FIELD", "target")

6. Execute internal main logic:
   RETURN internal_main(...)

7. Exception handling:
   CATCH Exception:
      Log error with traceback
      RAISE
```

**Contract Compliance**: Follows standardized main() signature for all processing scripts

## Algorithms and Data Structures

### Algorithm 1: Auto-Detection of Column Types

**Problem**: Distinguish between categorical and text columns in object dtype data

**Solution Strategy**: Use unique value ratio threshold

**Algorithm**:
```python
# Calculate uniqueness ratio
non_null_count = df[column].dropna().shape[0]
unique_count = df[column].nunique()
unique_ratio = unique_count / non_null_count

# Classify based on threshold
IF unique_ratio < threshold (default 0.1):
    column_type = "categorical"  # Low cardinality
ELSE:
    column_type = "text"          # High cardinality
```

**Example**:
```
Column: 'region' (10 unique out of 1000 records)
Ratio: 10/1000 = 0.01 < 0.1 → "categorical"

Column: 'user_id' (950 unique out of 1000 records)
Ratio: 950/1000 = 0.95 > 0.1 → "text"
```

**Complexity**: O(n) for calculating unique values

**Benefits**:
- Automatic appropriate strategy selection
- Configurable threshold for flexibility
- Handles mixed object dtype columns correctly

### Algorithm 2: Pandas-Safe Fill Value Validation

**Problem**: Prevent fill values that pandas interprets as NA/NULL

**Solution Strategy**: Maintain set of pandas NA values and validate before use

**Algorithm**:
```python
# Define pandas NA values
pandas_na_values = {
    "N/A", "NA", "NULL", "NaN", "nan", "NAN",
    "#N/A", "#N/A N/A", "#NA",
    "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN",
    "<NA>", "null", "Null", "none", "None", "NONE"
}

# Validate fill value
IF fill_value in pandas_na_values:
    Log warning
    fill_value = safe_replacement  # "Missing" or "Unknown"

RETURN SimpleImputer(strategy="constant", fill_value=fill_value)
```

**Complexity**: O(1) for set membership check

**Benefits**:
- Prevents silent data corruption
- Automatic safe replacement
- User-friendly warnings

### Algorithm 3: Parameter Accumulator Pattern

**Problem**: Downstream steps need access to all upstream preprocessing parameters

**Solution Strategy**: Copy all existing artifacts and add new ones

**Algorithm**:
```python
# Step 1: Copy existing artifacts
IF model_artifacts_input_dir:
    FOR each file in model_artifacts_input_dir:
        Copy file to model_artifacts_output_dir

# Step 2: Add own artifacts
save_imputation_artifacts(engine, config, model_artifacts_output_dir)

# Result: model_artifacts_output_dir contains:
# - All upstream artifacts (risk tables, normalization params, etc.)
# - New imputation artifacts (impute_dict.pkl, imputation_summary.json)
```

**Benefits**:
- Single source of truth for all preprocessing
- Simplifies deployment (one artifact directory)
- Enables complex preprocessing pipelines
- Maintains artifact provenance

### Data Structure 1: Imputation Configuration Dictionary

```python
imputation_config = {
    'default_numerical_strategy': 'mean',
    'default_categorical_strategy': 'mode',
    'default_text_strategy': 'mode',
    'numerical_constant_value': 0.0,
    'categorical_constant_value': 'Unknown',
    'text_constant_value': 'Unknown',
    'categorical_preserve_dtype': True,
    'auto_detect_categorical': True,
    'categorical_unique_ratio_threshold': 0.1,
    'validate_fill_values': True,
    'column_strategies': {
        'age': 'median',        # Column-specific override
        'income': 'mean',
        'category': 'mode'
    },
    'exclude_columns': ['id', 'timestamp']
}
```

**Properties**:
- Hierarchical: default strategies + column-specific overrides
- Type-specific: separate strategies for numerical/categorical/text
- Validation settings: pandas-safe validation configuration
- Auto-detection: categorical vs text detection settings

### Data Structure 2: Imputation Statistics Dictionary

```python
imputation_statistics = {
    'age': {
        'strategy': 'mean',
        'fill_value': None,
        'statistics': [27.5],                    # Learned mean value
        'missing_count_training': 150,
        'missing_percentage_training': 15.0,
        'data_type': 'float64'
    },
    'category': {
        'strategy': 'most_frequent',
        'fill_value': None,
        'statistics': ['A'],                     # Mode value
        'missing_count_training': 50,
        'missing_percentage_training': 5.0,
        'data_type': 'object'
    },
    'description': {
        'strategy': 'constant',
        'fill_value': 'Unknown',                 # Constant fill value
        'statistics': None,
        'missing_count_training': 80,
        'missing_percentage_training': 8.0,
        'data_type': 'object'
    }
}
```

**Usage**: Stores metadata about fitted imputation parameters

### Data Structure 3: Simple Imputation Dictionary (XGBoost Format)

```python
impute_dict = {
    'age': 27.5,
    'income': 55000.0,
    'category': 'A',
    'description': 'Good product',
    'region': 'US'
}
```

**Properties**:
- Simple `{column: value}` mapping
- Compatible with XGBoost training script
- Easy to use with pandas `fillna()`
- Serialized with pickle for efficiency

**Usage in Inference**:
```python
# Load parameters
with open('impute_dict.pkl', 'rb') as f:
    impute_dict = pkl.load(f)

# Apply imputation
for column, value in impute_dict.items():
    df[column] = df[column].fillna(value)
```

## Performance Characteristics

### Processing Time

| Operation | Typical Time (10K samples, 50 features, 20% missing) |
|-----------|------------------------------------------------------|
| Configuration loading | <0.01 seconds |
| Data loading (CSV) | 0.2-0.5 seconds |
| Data loading (Parquet) | 0.1-0.3 seconds |
| Missing value analysis | 0.1-0.2 seconds |
| Data type detection | 0.05-0.1 seconds (with auto-detect) |
| Strategy fitting (training) | 0.1-0.2 seconds |
| Imputation execution | 0.2-0.4 seconds |
| Artifact saving | 0.05-0.1 seconds |
| Report generation | 0.1-0.2 seconds |
| **Total (training job, 3 splits)** | **2-4 seconds** |
| **Total (inference job, 1 split)** | **0.8-1.5 seconds** |

### Memory Usage

**Memory Profile**:
```
Input data: n × m × 8 bytes (float64 DataFrame)
Fitted imputers: k × O(1) bytes (k=imputed columns, SimpleImputer is lightweight)
Imputed data: n × m × 8 bytes (DataFrame copy)
Impute dict: k × 8 bytes (simple dict)
Peak: ~2× input data + impute dict

Example (10K samples, 50 features, 20% missing → 10 imputed columns):
- Input: 10K × 50 × 8 = 4 MB
- Fitted imputers: 10 × negligible
- Imputed: 10K × 50 × 8 = 4 MB
- Impute dict: 10 × 8 = 80 bytes
- Peak: ~8 MB
```

**Scalability**:

| Input Size | Missing % | Memory (est.) | Time (est.) |
|------------|-----------|---------------|-------------|
| 1K samples | 10% | ~1 MB | 0.5 seconds |
| 10K samples | 20% | ~8 MB | 2 seconds |
| 100K samples | 20% | ~80 MB | 15 seconds |
| 1M samples | 20% | ~800 MB | 2-3 minutes |

**Recommendations**:
- For datasets > 1M samples: Use Parquet format
- For many missing columns: Consider filtering low-variance columns first
- For memory constraints: Process splits sequentially
- For speed: Use Parquet + parallel processing where possible

### Strategy-Specific Performance

| Strategy | Computation Overhead | Best For |
|----------|---------------------|----------|
| Mean | O(n) - single pass | Normal distributions, fast computation |
| Median | O(n log n) - sorting | Skewed distributions, outlier robustness |
| Mode | O(n) - frequency count | Categorical data, preserves distribution |
| Constant | O(1) - no computation | Known values, domain knowledge |

**Where** n = number of non-null values in column

## Error Handling

### Error Types

#### Configuration Errors

**Missing Required Environment Variable**:
- **Cause**: `LABEL_FIELD` not set
- **Handling**: Raises `ValueError` immediately
- **Response**: Script exits with code 2, clear error message

**Invalid Strategy Name**:
- **Cause**: Unknown strategy in configuration
- **Handling**: Logs warning, uses safe default (mean for numerical, mode for categorical/text)
- **Response**: Processing continues with default strategy

**Invalid Constant Value**:
- **Cause**: Non-numeric value for `NUMERICAL_CONSTANT_VALUE`
- **Handling**: Raises `ValueError` during config parsing
- **Response**: Script exits with code 2

#### Pandas-Safe Validation Warnings

**Problematic Fill Value Detected**:
- **Cause**: Fill value in pandas NA interpretation set (e.g., "NA", "null", "None")
- **Handling**: Logs warning, automatically replaces with safe value
- **Response**: Processing continues with safe replacement

**Example**:
```
WARNING: Categorical fill value 'NA' may be interpreted as NA by pandas. Using 'Missing' instead.
WARNING: Text fill value 'null' may be interpreted as NA by pandas. Using 'Unknown' instead.
```

#### Data Validation Errors

**Label Field Not Found**:
- **Cause**: Column specified in `LABEL_FIELD` doesn't exist in DataFrame
- **Handling**: Logs warning (not fatal)
- **Response**: Processing continues, but imputation may be applied to all columns

**No Missing Values Found**:
- **Cause**: Data has no missing values in any column
- **Handling**: Logs warning
- **Response**: Processing continues, returns data unchanged

**Format Detection Failure**:
- **Cause**: No processed data file found in expected location
- **Handling**: Raises `RuntimeError` listing attempted formats
- **Response**: Script exits with code 1

#### Artifact Loading Errors

**Imputation Parameters Not Found**:
- **Cause**: Non-training job but `impute_dict.pkl` not found in model_artifacts_input
- **Handling**: Logs warning, raises `ValueError`
- **Response**: Script exits with code 2

**Invalid Imputation Parameters Format**:
- **Cause**: Loaded parameters not a dictionary
- **Handling**: Raises `ValueError` with format description
- **Response**: Script exits with code 2

### Error Response Structure

**Exit Codes**:
- **0**: Success
- **1**: File not found error
- **2**: Value/validation error  
- **3**: Unexpected exception

**Error Message Format**:
```
ERROR: [Error description]
[Detailed context including available options/columns/files]
[Stack trace if unexpected]
```

**Example Error Messages**:
```
ERROR: Missing required input path: input_data

ERROR: LABEL_FIELD environment variable must be set.

ERROR: Imputation parameters file not found: /opt/ml/processing/input/model_artifacts/impute_dict.pkl

ERROR: No processed data file found in /opt/ml/processing/input/data/train. Looked for: ['train_processed_data.csv', 'train_processed_data.tsv', 'train_processed_data.parquet']

WARNING: Categorical fill value 'NA' may be interpreted as NA by pandas. Using 'Missing' instead.

WARNING: Column 'age' has 65% missing values. Consider investigating data collection issues.
```

## Best Practices

### For Production Deployments

1. **Choose Appropriate Strategies**
   - **Mean**: Use for normally distributed numerical data
   - **Median**: Use for skewed numerical data or presence of outliers
   - **Mode**: Use for categorical and text data
   - **Constant**: Use when domain knowledge suggests specific value

2. **Configure Pandas-Safe Validation**
   - Keep `VALIDATE_FILL_VALUES=true` (default)
   - Avoid fill values like "NA", "NULL", "none", "NaN"
   - Use descriptive values like "Unknown", "Missing", "Not Provided"

3. **Set Appropriate Thresholds**
   - Default categorical threshold (0.1) works for most cases
   - Increase threshold for high-cardinality categorical (e.g., 0.2)
   - Decrease threshold for strict categorical detection (e.g., 0.05)

4. **Monitor Missing Value Patterns**
   - Review generated reports for high missing percentages (>50%)
   - Investigate systematic missing patterns
   - Consider data collection improvements for excessive missing

5. **Use Column-Specific Strategies**
   - Override defaults for known problematic columns
   - Example: `COLUMN_STRATEGY_age=median` for age with outliers
   - Example: `COLUMN_STRATEGY_user_id=constant` for optional fields

### For Development

1. **Start with Analysis**
   - Run missing value analysis first to understand patterns
   - Review recommendations in generated reports
   - Adjust strategies based on data characteristics

2. **Test with Small Samples**
   - Validate imputation logic on subset of data
   - Verify pandas-safe validation is working
   - Check output format preservation

3. **Validate Artifact Accumulation**
   - Verify existing artifacts are copied correctly
   - Check new artifacts are added properly
   - Ensure downstream steps can load accumulated artifacts

4. **Test Both Modes**
   - Training mode: Verify parameters are saved correctly
   - Inference mode: Verify parameters are loaded and applied correctly
   - Check consistency between training and inference

### For Performance Optimization

1. **Use Parquet Format**
   - 40-60% faster I/O vs CSV
   - Smaller file sizes (50-80% compression)
   - Better for large datasets

2. **Optimize Strategy Selection**
   - Mean/mode are faster than median (O(n) vs O(n log n))
   - Consider speed vs accuracy tradeoff
   - Use constant strategy when domain knowledge is strong

3. **Filter Columns Early**
   - Exclude columns that don't need imputation via `EXCLUDE_COLUMNS`
   - Reduces processing time and memory usage
   - Improves clarity of imputation logic

4. **Monitor Memory Usage**
   - For very large datasets, consider batch processing
   - Process splits sequentially if memory constrained
   - Use data sampling for strategy selection if needed

## Example Configurations

### Example 1: Mean Imputation for Numerical Data
```bash
export LABEL_FIELD="target"
export DEFAULT_NUMERICAL_STRATEGY="mean"
export DEFAULT_CATEGORICAL_STRATEGY="mode"
export DEFAULT_TEXT_STRATEGY="mode"

python missing_value_imputation.py --job_type training
```

**Use Case**: Standard numerical data with normal distributions
**Result**: Numerical columns imputed with mean, categorical with mode

### Example 2: Median Imputation for Skewed Data
```bash
export LABEL_FIELD="is_fraud"
export DEFAULT_NUMERICAL_STRATEGY="median"
export COLUMN_STRATEGY_transaction_amount="median"
export COLUMN_STRATEGY_account_age="median"

python missing_value_imputation.py --job_type training
```

**Use Case**: Financial data with outliers (transaction amounts, account age)
**Result**: Robust imputation using median for outlier resistance

### Example 3: Column-Specific Strategies
```bash
export LABEL_FIELD="click"
export DEFAULT_NUMERICAL_STRATEGY="mean"
export COLUMN_STRATEGY_age="median"
export COLUMN_STRATEGY_income="median"
export COLUMN_STRATEGY_user_id="constant"
export TEXT_CONSTANT_VALUE="UNKNOWN_USER"
export CATEGORICAL_CONSTANT_VALUE="OTHER"

python missing_value_imputation.py --job_type training
```

**Use Case**: Mixed data with specific requirements per column
**Result**: Customized imputation per column type and business logic

### Example 4: Inference Mode with Pre-Trained Parameters
```bash
export LABEL_FIELD="target"

python missing_value_imputation.py --job_type validation
```

**Use Case**: Apply training-learned imputation to validation data
**Result**: Consistent transformation using saved impute_dict.pkl

### Example 5: High-Cardinality Categorical Detection
```bash
export LABEL_FIELD="label"
export AUTO_DETECT_CATEGORICAL="true"
export CATEGORICAL_UNIQUE_RATIO_THRESHOLD="0.2"
export DEFAULT_CATEGORICAL_STRATEGY="mode"
export DEFAULT_TEXT_STRATEGY="constant"
export TEXT_CONSTANT_VALUE=""

python missing_value_imputation.py --job_type training
```

**Use Case**: Data with mix of low and high cardinality text columns
**Result**: Columns with <20% unique values treated as categorical, rest as text

## Integration Patterns

### Upstream Integration

```
TabularPreprocessing
   ↓ (train/val/test splits with potential missing values)
MissingValueImputation
   ↓ (imputed data + accumulated artifacts)
```

**Data Flow**:
1. TabularPreprocessing creates initial splits
2. MissingValueImputation handles missing values
3. Preserves format and structure for downstream steps

### Downstream Integration

```
MissingValueImputation
   ↓ (imputed data + accumulated artifacts)
   ├→ RiskTableMapping (uses imputed data for encoding)
   ├→ FeatureSelection (uses imputed data for feature analysis)
   └→ XGBoostTraining (uses accumulated artifacts including impute_dict.pkl)
```

**Benefits**:
- Complete data for feature engineering
- No missing values in training data
- Accumulated artifacts available for inference

### Complete Pipeline Example

```
1. TabularPreprocessing (job_type=training)
   → train/val/test splits (may have missing values)

2. MissingValueImputation (job_type=training)
   → Imputed splits + impute_dict.pkl
   
3. RiskTableMapping (job_type=training)
   → Risk-mapped splits + accumulated artifacts

4. FeatureSelection (job_type=training)
   → Selected features + accumulated artifacts

5. XGBoostTraining
   → Trains on complete data
   → Can access all preprocessing artifacts from single location
```

### Parameter Accumulator Pattern in Action

```
Step 1: TabularPreprocessing
   Output: /opt/ml/processing/output/model_artifacts/
   └── preprocessing_params.json

Step 2: MissingValueImputation
   Input: /opt/ml/processing/input/model_artifacts/
   └── preprocessing_params.json
   
   Output: /opt/ml/processing/output/model_artifacts/
   ├── preprocessing_params.json          # Copied from input
   ├── impute_dict.pkl                    # Added
   └── imputation_summary.json            # Added

Step 3: RiskTableMapping
   Input: /opt/ml/processing/input/model_artifacts/
   ├── preprocessing_params.json
   ├── impute_dict.pkl
   └── imputation_summary.json
   
   Output: /opt/ml/processing/output/model_artifacts/
   ├── preprocessing_params.json          # Copied
   ├── impute_dict.pkl                    # Copied
   ├── imputation_summary.json            # Copied
   ├── risk_table.pkl                     # Added
   └── risk_mapping_summary.json          # Added
```

## Troubleshooting

### Issue 1: All Missing Values Remain After Imputation

**Symptom**:
Output data still has missing values in columns that should be imputed

**Common Causes**:
1. Column excluded via `EXCLUDE_COLUMNS` or is label field
2. Imputation failed silently (check logs)
3. Pandas-safe validation replaced fill value with NA value

**Solutions**:
1. Check excluded columns: Review `EXCLUDE_COLUMNS` and `LABEL_FIELD`
2. Check logs for imputation warnings
3. Verify fill values are pandas-safe (not "NA", "null", etc.)
4. Review imputation_summary.txt for column-specific issues

### Issue 2: Categorical Data Treated as Text

**Symptom**:
Low-cardinality categorical columns receiving text imputation strategy

**Common Causes**:
1. Auto-detection disabled (`AUTO_DETECT_CATEGORICAL=false`)
2. Threshold too low (e.g., 0.01)
3. Column has slightly higher unique ratio than threshold

**Solutions**:
1. Enable auto-detection: `AUTO_DETECT_CATEGORICAL=true`
2. Increase threshold: `CATEGORICAL_UNIQUE_RATIO_THRESHOLD=0.15`
3. Use column-specific override: `COLUMN_STRATEGY_category=mode`
4. Check unique value ratio in missing_analysis report

### Issue 3: "NA" or "null" Fill Values Interpreted as Missing

**Symptom**:
Imputed values disappear or show as NaN after saving/loading

**Common Causes**:
1. Fill value in pandas NA interpretation set
2. Pandas-safe validation disabled
3. Manual constant value conflicting with pandas

**Solutions**:
1. Enable validation: `VALIDATE_FILL_VALUES=true` (default)
2. Use safe values: "Unknown", "Missing", "Not Provided"
3. Review warnings in logs for automatic replacements
4. Check get_pandas_na_values() set for problematic values

### Issue 4: Imputation Parameters Not Found in Inference Mode

**Symptom**:
ValueError: "Imputation parameters file not found"

**Common Causes**:
1. Training job didn't complete successfully
2. Wrong model_artifacts_input path
3. File naming mismatch

**Solutions**:
1. Verify training job completed: Check for impute_dict.pkl in training output
2. Verify path: Check `model_artifacts_input` parameter
3. Check expected filename: Should be "impute_dict.pkl"
4. Review training logs for artifact saving errors

### Issue 5: Memory Error with Large Datasets

**Symptom**:
Script crashes with MemoryError or OutOfMemoryError

**Common Causes**:
1. Dataset too large for available memory
2. Multiple splits loaded simultaneously
3. Inefficient data format (CSV vs Parquet)

**Solutions**:
1. Use Parquet format for better compression
2. Process splits sequentially if possible
3. Increase available memory (larger instance type)
4. Sample data for strategy selection if needed
5. Monitor memory usage with smaller test dataset first

## References

### Related Scripts

- **[`tabular_preprocessing.py`](./tabular_preprocessing_script.md)**: Upstream preprocessing script
- **[`risk_table_mapping.py`](./risk_table_mapping_script.md)**: Downstream risk mapping script
- **[`feature_selection.py`](./feature_selection_script.md)**: Downstream feature selection
- **[`xgboost_training.py`](./xgboost_training_script.md)**: Uses accumulated artifacts

### Related Documentation

- **Contract**: `src/cursus/steps/contracts/missing_value_imputation_contract.py`
- **Step Specification**: MissingValueImputation step specification
- **Config Class**: `src/cursus/steps/configs/config_missing_value_imputation_step.py`
- **Builder**: `src/cursus/steps/builders/builder_missing_value_imputation_step.py`

### Related Design Documents

- **[Missing Value Imputation Design](../1_design/missing_value_imputation_design.md)**: Complete system design with architecture
- **[Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md)**: Format detection and preservation strategy
- **[Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)**: General processing step architecture

### External References

- **Scikit-learn SimpleImputer**: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- **Pandas Missing Data**: https://pandas.pydata.org/docs/user_guide/missing_data.html
- **Missing Value Imputation Methods**: https://en.wikipedia.org/wiki/Imputation_(statistics)
- **Pandas NA Values**: https://pandas.pydata.org/docs/reference/api/pandas.isna.html

---

## Document Metadata

**Author**: Cursus Framework Team  
**Last Updated**: 2025-11-18  
**Script Version**: 2025-11-18  
**Documentation Version**: 1.0  
**Review Status**: Complete

**Change Log**:
- 2025-11-18: Initial comprehensive documentation created
- 2025-11-18: Missing value imputation script documented with multi-type imputation support and pandas-safe validation

**Related Scripts**: 
- Upstream: `tabular_preprocessing.py`
- Downstream: `risk_table_mapping.py`, `feature_selection.py`, `xgboost_training.py`
- Related: Data preprocessing and feature engineering scripts
