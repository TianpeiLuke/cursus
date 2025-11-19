---
tags:
  - code
  - training_script
  - xgboost
  - gradient_boosting
keywords:
  - XGBoost training
  - risk table mapping
  - numerical imputation
  - tabular classification
  - binary classification
  - multiclass classification
  - feature preprocessing
  - gradient boosting
topics:
  - XGBoost training
  - tabular machine learning
  - preprocessing integration
  - model training workflows
language: python
date of note: 2025-11-18
---

# XGBoost Training Script

## Overview

The `xgboost_training.py` script implements a comprehensive XGBoost training pipeline for tabular data classification with integrated preprocessing capabilities including numerical imputation and risk table mapping for categorical features.

The script provides a production-ready training workflow that supports both inline preprocessing computation and pre-computed artifact reuse from upstream steps. It handles binary and multiclass classification with configurable hyperparameters, class imbalance handling, comprehensive evaluation metrics, and format-preserving I/O across CSV, TSV, and Parquet formats.

Key capabilities include flexible preprocessing artifact control (compute inline vs reuse pre-computed), risk table mapping for categorical feature transformation, mean-based numerical imputation, XGBoost model training with early stopping, comprehensive evaluation with ROC/PR curves, and packaged output artifacts in tarball format for deployment.

## Purpose and Major Tasks

### Primary Purpose
Train XGBoost gradient boosting models for tabular classification tasks with integrated preprocessing, supporting flexible artifact reuse patterns for efficient pipeline orchestration and production deployment.

### Major Tasks

1. **Package Installation Management**: Install required packages from secure CodeArtifact or public PyPI based on configuration
2. **Configuration Loading**: Load and validate hyperparameters from JSON with Pydantic schema validation
3. **Data Loading**: Load train/validation/test datasets with automatic format detection (CSV, TSV, Parquet)
4. **Preprocessing Artifact Control**: Detect and load pre-computed artifacts or compute inline based on environment flags
5. **Numerical Imputation**: Apply mean-based imputation for missing values (inline or pre-computed)
6. **Risk Table Mapping**: Fit and apply risk tables for categorical feature transformation (inline or pre-computed)
7. **Feature Selection Integration**: Apply pre-computed feature selection if available
8. **Model Training**: Train XGBoost model with configurable hyperparameters and early stopping
9. **Comprehensive Evaluation**: Compute metrics, generate predictions, and create ROC/PR visualizations
10. **Artifact Packaging**: Save model, preprocessing artifacts, and evaluation outputs

## Script Contract

### Entry Point
```
xgboost_training.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `input_path` | `/opt/ml/input/data` | Root directory containing train/val/test subdirectories |
| `train` | `/opt/ml/input/data/train` | Training data files (.csv, .tsv, .parquet) |
| `val` | `/opt/ml/input/data/val` | Validation data files |
| `test` | `/opt/ml/input/data/test` | Test data files |
| `hyperparameters_s3_uri` | `/opt/ml/code/hyperparams/hyperparameters.json` | Model configuration and hyperparameters |
| `model_artifacts_input` | `/opt/ml/input/data/model_artifacts_input` | Optional: Pre-computed preprocessing artifacts |

**Optional Preprocessing Artifacts** (in model_artifacts_input):
- `impute_dict.pkl`: Pre-computed imputation parameters (mean values per column)
- `risk_table_map.pkl`: Pre-computed risk tables for categorical features
- `selected_features.json`: Pre-computed feature selection results

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `model_output` | `/opt/ml/model` | Primary model artifacts directory |
| `evaluation_output` | `/opt/ml/output/data` | Evaluation results and metrics |

**Model Output Contents**:
- `xgboost_model.bst`: Trained XGBoost model in binary format
- `risk_table_map.pkl`: Risk table mappings for categorical features
- `impute_dict.pkl`: Imputation values for numerical features
- `feature_importance.json`: Feature importance scores from model
- `feature_columns.txt`: Ordered feature column names with indices
- `hyperparameters.json`: Complete model hyperparameters

**Evaluation Output Contents**:
- `val.tar.gz`: Validation predictions and metrics tarball
  - `val/predictions.{csv,tsv,parquet}`: Validation predictions
  - `val/metrics.json`: Validation metrics
  - `val_metrics/val_roc.jpg`: ROC curve
  - `val_metrics/val_pr.jpg`: Precision-Recall curve
- `test.tar.gz`: Test predictions and metrics tarball (same structure)

### Required Environment Variables

None strictly required - all configuration via hyperparameters.json

### Optional Environment Variables

#### Package Installation Control
| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SECURE_PYPI` | `"true"` | Controls PyPI source (true=CodeArtifact, false=public PyPI) |

#### Preprocessing Artifact Control
| Variable | Default | Description | Use Case |
|----------|---------|-------------|----------|
| `USE_PRECOMPUTED_IMPUTATION` | `"false"` | Use pre-computed imputation artifacts | When data already imputed upstream |
| `USE_PRECOMPUTED_RISK_TABLES` | `"false"` | Use pre-computed risk table artifacts | When risk mapping done upstream |
| `USE_PRECOMPUTED_FEATURES` | `"false"` | Use pre-computed feature selection | When features already selected upstream |

**Artifact Control Behavior**:
- When `USE_PRECOMPUTED_*=true`: Script loads artifacts from `model_artifacts_input` and skips transformation (data must already be processed)
- When `USE_PRECOMPUTED_*=false`: Script computes preprocessing inline and transforms data
- Script validates data state matches environment variable flags
- All artifacts (pre-computed or inline) are packaged into final model output

### Job Arguments

**No command-line arguments** - Script follows container contract pattern with fixed paths and configuration via hyperparameters.json

### Hyperparameters (via JSON Configuration)

#### Data Configuration
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `tab_field_list` | `List[str]` | Yes | Numerical/tabular feature column names | `["price", "quantity", "rating"]` |
| `cat_field_list` | `List[str]` | Yes | Categorical feature column names | `["marketplace", "category", "seller_type"]` |
| `label_name` | `str` | Yes | Column name for classification target | `"is_fraud"` |
| `id_name` | `str` | No | Column name for record IDs (default: "id") | `"order_id"` |
| `multiclass_categories` | `List[Union[int, str]]` | Yes | List of all class labels | `[0, 1]` or `["low", "medium", "high"]` |

#### Model Architecture
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `is_binary` | `bool` | No | Whether task is binary classification (auto-detected if omitted) | `true`, `false` |
| `num_classes` | `int` | No | Number of classes (auto-computed from multiclass_categories) | `2` (binary), `3+` (multiclass) |
| `class_weights` | `List[float]` | No | Class weights for imbalanced data | Example: `[1.0, 5.0]` |

#### XGBoost Hyperparameters
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `eta` | `float` | No | Learning rate (step size shrinkage) | Default: `0.1`, Range: `0.0-1.0` |
| `gamma` | `float` | No | Minimum loss reduction for split | Default: `0`, Range: `0+` |
| `max_depth` | `int` | No | Maximum tree depth | Default: `6`, Range: `3-10` typical |
| `subsample` | `float` | No | Subsample ratio of training instances | Default: `1`, Range: `0.0-1.0` |
| `colsample_bytree` | `float` | No | Subsample ratio of columns per tree | Default: `1`, Range: `0.0-1.0` |
| `lambda_xgb` | `float` | No | L2 regularization term on weights | Default: `1`, Range: `0+` |
| `alpha_xgb` | `float` | No | L1 regularization term on weights | Default: `0`, Range: `0+` |

#### Training Control
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `num_round` | `int` | No | Number of boosting rounds | Default: `100`, Typical: `50-500` |
| `early_stopping_rounds` | `int` | No | Early stopping patience | Default: `10`, Typical: `5-20` |

#### Risk Table Parameters
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `smooth_factor` | `float` | No | Smoothing factor for risk table estimation | Default: `0.0`, Range: `0.0-1.0` |
| `count_threshold` | `int` | No | Minimum count threshold for risk table bins | Default: `0`, Range: `0+` |

## Input Data Structure

### Expected Input Format

```
/opt/ml/input/data/
├── train/
│   ├── training_data.csv (or .tsv, .parquet)
│   └── _SUCCESS (optional marker)
├── val/
│   ├── validation_data.csv
│   └── _SUCCESS
├── test/
│   ├── test_data.csv
│   └── _SUCCESS
└── model_artifacts_input/ (optional)
    ├── impute_dict.pkl
    ├── risk_table_map.pkl
    └── selected_features.json

/opt/ml/code/hyperparams/
└── hyperparameters.json
```

### Required Columns in Data Files

**Essential Columns**:
- Label column (name specified by `label_name`): Classification target
- ID column (name specified by `id_name`, optional): Unique record identifier

**Feature Columns**:
- Tabular features (specified in `tab_field_list`): Numerical features for XGBoost
- Categorical features (specified in `cat_field_list`): Categorical features for risk table mapping

### Supported Data Formats

1. **CSV Files**: Comma-separated values
2. **TSV Files**: Tab-separated values
3. **Parquet Files**: Columnar format for efficient storage

**Format Preservation**:
- Script automatically detects input format from training data
- All outputs (predictions, metrics) use the same format as input
- Format warning logged if mixed formats detected across splits

### Data Requirements

**Numerical Features** (`tab_field_list`):
- May contain missing values (will be imputed)
- Should be numeric type or convertible to float
- After imputation, must not contain NaN or inf values

**Categorical Features** (`cat_field_list`):
- Can be string or numeric types
- Will be transformed to risk scores via risk table mapping
- After transformation, will be numeric type

**Labels**:
- Must match values in `multiclass_categories`
- Can be integers or strings
- Will be encoded to sequential integers (0, 1, 2, ...)

### Pre-Computed Data State Requirements

When using pre-computed artifacts, input data must match expected state:

**USE_PRECOMPUTED_IMPUTATION=true**:
- Data must NOT contain NaN values in `tab_field_list` columns
- Data already imputed upstream
- Script validates no missing values exist

**USE_PRECOMPUTED_RISK_TABLES=true**:
- Categorical columns in `cat_field_list` must be numeric type
- Data already risk-mapped upstream
- Script validates numeric dtype

**USE_PRECOMPUTED_FEATURES=true**:
- Data may contain additional columns beyond selected features
- Script filters to selected features only
- Validates selected features exist in data

## Output Data Structure

### Model Output Directory Structure

```
/opt/ml/model/
├── xgboost_model.bst              # XGBoost model (binary format)
├── risk_table_map.pkl             # Risk tables dictionary
├── impute_dict.pkl                # Imputation values dictionary
├── feature_importance.json        # Feature importance scores
├── feature_columns.txt            # Ordered feature column names
└── hyperparameters.json           # Model hyperparameters
```

**xgboost_model.bst**:
- Binary format XGBoost model
- Can be loaded with `xgb.Booster().load_model()`
- Contains all tree structures and parameters

**risk_table_map.pkl**:
```python
{
    "marketplace": {
        "US": 0.15,
        "UK": 0.23,
        "DE": 0.18,
        "__default__": 0.20  # For unseen categories
    },
    "category": {...},
    ...
}
```

**impute_dict.pkl**:
```python
{
    "price": 29.99,        # Mean value for price
    "quantity": 2.5,       # Mean value for quantity
    "rating": 4.2,         # Mean value for rating
    ...
}
```

**feature_columns.txt**:
```
# Feature columns in exact order required for XGBoost model inference
# DO NOT MODIFY THE ORDER OF THESE COLUMNS
# Each line contains: <column_index>,<column_name>
0,price
1,quantity
2,rating
3,marketplace
4,category
```

**feature_importance.json**:
```json
{
    "price": 45,
    "marketplace": 32,
    "quantity": 28,
    "rating": 15,
    "category": 10
}
```

### Evaluation Output Directory Structure

```
/opt/ml/output/data/
├── val.tar.gz                     # Validation results tarball
│   ├── val/
│   │   ├── predictions.csv        # Validation predictions
│   │   └── metrics.json           # Validation metrics
│   └── val_metrics/
│       ├── val_roc.jpg            # ROC curve
│       └── val_pr.jpg             # Precision-Recall curve
└── test.tar.gz                    # Test results tarball (same structure)
```

**predictions.{csv,tsv,parquet} Contents**:
| Column | Description |
|--------|-------------|
| `id_name` | Record identifier (preserves from input) |
| `label_name` | True label values |
| `prob_class_0` | Predicted probability for class 0 |
| `prob_class_1` | Predicted probability for class 1 |
| ... | Additional prob columns for multiclass |

**metrics.json Contents** (Binary):
```json
{
    "auc_roc": 0.8542,
    "average_precision": 0.8012,
    "f1_score": 0.7823
}
```

**metrics.json Contents** (Multiclass):
```json
{
    "auc_roc_class_0": 0.85,
    "auc_roc_class_1": 0.82,
    "auc_roc_class_2": 0.88,
    "auc_roc_micro": 0.85,
    "auc_roc_macro": 0.85,
    "average_precision_micro": 0.82,
    "average_precision_macro": 0.81,
    "f1_score_micro": 0.78,
    "f1_score_macro": 0.77
}
```

## Key Functions and Tasks

### Package Installation Component

#### `install_packages(packages, use_secure)`
**Purpose**: Install Python packages from secure CodeArtifact or public PyPI

**Algorithm**:
```python
1. Log installation configuration (source, package count)
2. If use_secure:
   a. Get CodeArtifact access token via AWS STS and boto3
   b. Construct secure PyPI index URL with token
   c. Install packages using pip with --index-url
3. Else:
   a. Install packages using pip from public PyPI
4. Log success or error with detailed information
```

**Parameters**:
- `packages` (list): Package specifications (e.g., ["pandas==1.5.0", "numpy"])
- `use_secure` (bool): If True, use CodeArtifact; if False, use public PyPI

**Returns**: None (raises exception on failure)

**Security Features**:
- Uses AWS STS role assumption for CodeArtifact access
- Token-based authentication for secure PyPI
- Configurable via `USE_SECURE_PYPI` environment variable

### Configuration Management Component

#### `load_and_validate_config(hparam_path)`
**Purpose**: Load and validate hyperparameters from JSON file

**Algorithm**:
```python
1. Load JSON configuration from file
2. Validate required keys exist:
   a. tab_field_list, cat_field_list
   b. label_name, multiclass_categories
3. Auto-compute num_classes from multiclass_categories length
4. Auto-detect is_binary (num_classes == 2)
5. Validate class_weights length matches num_classes
6. Return validated configuration dictionary
```

**Parameters**:
- `hparam_path` (str): Path to hyperparameters.json

**Returns**: `dict` - Validated configuration

**Validation Checks**:
- Required keys present
- class_weights length consistency
- Proper type conversion

### Data Loading Component

#### `load_datasets(input_path)`
**Purpose**: Load train/val/test datasets with automatic format detection

**Algorithm**:
```python
1. Find first data file in each directory (train/val/test):
   a. Sort files alphabetically
   b. Match extensions: .csv, .tsv, .parquet
   c. Return first match or raise FileNotFoundError
2. Load each dataset with format detection:
   a. Detect format from file extension
   b. Load using appropriate pandas function
   c. Track format for output preservation
3. Use training data format as primary format
4. Log warning if mixed formats detected
5. Return all datasets and detected format
```

**Parameters**:
- `input_path` (str): Root input directory

**Returns**: `Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]` - (train_df, val_df, test_df, format)

**Format Detection**:
- `.csv` → `pd.read_csv()`
- `.tsv` → `pd.read_csv(sep='\t')`
- `.parquet` → `pd.read_parquet()`

#### `save_dataframe_with_format(df, output_path, format_str)`
**Purpose**: Save DataFrame in specified format with proper extension

**Algorithm**:
```python
1. Convert output_path to Path object
2. Based on format_str:
   a. If 'csv': Save with .csv extension
   b. If 'tsv': Save with .tsv extension, tab separator
   c. If 'parquet': Save with .parquet extension
3. Save DataFrame without index
4. Return path to saved file
```

**Parameters**:
- `df` (pd.DataFrame): DataFrame to save
- `output_path` (str): Base output path (without extension)
- `format_str` (str): Format ('csv', 'tsv', or 'parquet')

**Returns**: `str` - Path to saved file

### Preprocessing Component

#### `apply_numerical_imputation(config, train_df, val_df, test_df)`
**Purpose**: Apply mean-based numerical imputation to all datasets

**Algorithm**:
```python
1. Create empty imputation_processors dictionary
2. For each numerical column in tab_field_list:
   a. Create NumericalVariableImputationProcessor with strategy='mean'
   b. Fit processor on training data column
   c. Store processor in dictionary
   d. Transform train/val/test splits
3. Build impute_dict from fitted processors
4. Return transformed datasets and impute_dict
```

**Parameters**:
- `config` (dict): Configuration with tab_field_list
- `train_df`, `val_df`, `test_df` (pd.DataFrame): Datasets to impute

**Returns**: `Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]` - (train, val, test, impute_dict)

**Single-Column Architecture**:
- One processor per column (consistent with risk table mapping)
- Each processor fits independently
- Enables granular control and debugging

#### `fit_and_apply_risk_tables(config, train_df, val_df, test_df)`
**Purpose**: Fit risk tables on training data and apply to all splits

**Algorithm**:
```python
1. Create empty risk_processors dictionary
2. For each categorical column in cat_field_list:
   a. Create RiskTableMappingProcessor with:
      - column_name
      - label_name
      - smooth_factor (from config)
      - count_threshold (from config)
   b. Fit processor on full training DataFrame
   c. Store processor in dictionary
   d. Transform all splits using fitted processor
3. Build consolidated_risk_tables from fitted processors
4. Return transformed datasets and risk_tables
```

**Parameters**:
- `config` (dict): Configuration with cat_field_list, label_name, smoothing params
- `train_df`, `val_df`, `test_df` (pd.DataFrame): Datasets to transform

**Returns**: `Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]` - (train, val, test, risk_tables)

**Risk Table Format**:
```python
{
    "marketplace": {
        "US": 0.15,     # Target rate for US
        "UK": 0.23,     # Target rate for UK
        "__default__": 0.20  # Default for unseen values
    }
}
```

### Feature Selection Integration Component

#### `detect_feature_selection_artifacts(model_artifacts_dir)`
**Purpose**: Conservatively detect if feature selection was applied upstream

**Algorithm**:
```python
1. Check if model_artifacts_dir exists
2. If not exists: return None (no artifacts)
3. Check for selected_features.json in directory
4. If found: return path to file
5. Else: return None
```

**Parameters**:
- `model_artifacts_dir` (Optional[str]): Path to model artifacts directory

**Returns**: `Optional[str]` - Path to selected_features.json or None

**Conservative Approach**:
- Only detects if file explicitly exists
- Does not make assumptions
- Clear logging of detection results

#### `get_effective_feature_columns(config, input_paths, train_df)`
**Purpose**: Get feature columns with fallback-first approach

**Algorithm**:
```python
1. ALWAYS start with original behavior:
   a. Extract features from config (tab_field_list + cat_field_list)
   b. This is the baseline/fallback
2. Check if feature selection artifacts exist:
   a. Call detect_feature_selection_artifacts()
   b. If None: return original features, False
3. Try to load selected features from artifacts:
   a. Parse selected_features.json
   b. If fails: fallback to original features
4. Validate selected features:
   a. Check all features exist in train_df columns
   b. Check reasonable subset (not more than original)
   c. If validation fails: fallback to original features
5. Success: return selected features, True
```

**Parameters**:
- `config` (dict): Configuration object
- `input_paths` (Dict[str, str]): Input paths
- `train_df` (pd.DataFrame): Training data for validation

**Returns**: `Tuple[List[str], bool]` - (feature_columns, feature_selection_applied)

**Fallback Strategy**:
- Prioritizes reliability over optimization
- Multiple validation checks
- Explicit logging at each decision point

### Model Training Component

#### `prepare_dmatrices(config, train_df, val_df)`
**Purpose**: Prepare XGBoost DMatrix objects from DataFrames

**Algorithm**:
```python
1. Maintain exact ordering of features:
   a. feature_columns = tab_field_list + cat_field_list
2. Extract feature matrices:
   a. X_train = train_df[feature_columns].astype(float)
   b. X_val = val_df[feature_columns].astype(float)
3. Validate no NaN/inf values remain
4. Create DMatrix objects:
   a. dtrain = xgb.DMatrix(X_train, label=y_train)
   b. dval = xgb.DMatrix(X_val, label=y_val)
5. Set feature names in DMatrix for preservation
6. Return dtrain, dval, feature_columns
```

**Parameters**:
- `config` (dict): Configuration with feature lists
- `train_df`, `val_df` (pd.DataFrame): Processed datasets

**Returns**: `Tuple[xgb.DMatrix, xgb.DMatrix, List[str]]` - (dtrain, dval, feature_columns)

**Critical**: Feature ordering is preserved for consistent inference

#### `train_model(config, dtrain, dval)`
**Purpose**: Train XGBoost model with configured hyperparameters

**Algorithm**:
```python
1. Build base parameter dictionary:
   a. eta, gamma, max_depth
   b. subsample, colsample_bytree
   c. lambda, alpha
2. Set objective based on classification type:
   a. If is_binary:
      - objective = "binary:logistic"
      - Handle class weights via scale_pos_weight
   b. Else:
      - objective = "multi:softprob"
      - num_class = num_classes
      - Handle class weights via sample weights
3. Log training configuration and label distributions
4. Call xgb.train() with:
   a. params dictionary
   b. dtrain (training data)
   c. num_boost_round
   d. evals (train and validation)
   e. early_stopping_rounds
   f. verbose_eval
5. Return trained model
```

**Parameters**:
- `config` (dict): Configuration with XGBoost hyperparameters
- `dtrain` (xgb.DMatrix): Training data
- `dval` (xgb.DMatrix): Validation data

**Returns**: `xgb.Booster` - Trained XGBoost model

**Class Weight Handling**:
- Binary: Uses `scale_pos_weight = weight[1] / weight[0]`
- Multiclass: Applies sample weights to training data

### Evaluation Component

#### `save_preds_and_metrics(ids, y_true, y_prob, id_col, label_col, out_dir, is_binary, output_format)`
**Purpose**: Save predictions and compute metrics with format preservation

**Algorithm**:
```python
1. Compute metrics based on classification type:
   a. Binary: AUC-ROC, Average Precision, F1-Score
   b. Multiclass: Per-class + micro/macro averaged metrics
2. Log all metric values
3. Save metrics to metrics.json
4. Build predictions DataFrame:
   a. Include ID and true label columns
   b. Add prob_class_i for each class
5. Save predictions using format-preserving function
```

**Parameters**:
- `ids`: Record identifiers
- `y_true`: True labels
- `y_prob`: Predicted probabilities
- `id_col`, `label_col`: Column names
- `out_dir`: Output directory
- `is_binary`: Classification type
- `output_format`: Format to save ('csv', 'tsv', 'parquet')

**Returns**: None (saves files)

#### `plot_curves(y_true, y_prob, out_dir, prefix, is_binary)`
**Purpose**: Generate and save ROC and Precision-Recall curves

**Algorithm**:
```python
1. If binary:
   a. Compute ROC curve (FPR, TPR, AUC)
   b. Plot ROC with diagonal reference line
   c. Compute PR curve (precision, recall, AP)
   d. Plot PR curve
2. If multiclass:
   a. For each class:
      - Create binary labels (one-vs-rest)
      - Compute and plot ROC curve
      - Compute and plot PR curve
   b. Save class-specific curves
3. Save all figures as JPEG files
```

**Parameters**:
- `y_true`: True labels
- `y_prob`: Predicted probabilities
- `out_dir`: Output directory
- `prefix`: Filename prefix
- `is_binary`: Classification type

**Returns**: None (saves plots)

#### `evaluate_split(name, df, feats, model, cfg, output_format, prefix)`
**Purpose**: Comprehensive evaluation of a data split

**Algorithm**:
```python
1. Extract IDs, true labels from DataFrame
2. Build DMatrix with feature names
3. Generate predictions using model
4. Reshape predictions if needed (binary)
5. Save predictions and metrics
6. Generate ROC and PR curve plots
7. Package outputs into tarball:
   a. Create tar.gz archive
   b. Include predictions directory
   c. Include metrics directory
8. Log completion
```

**Parameters**:
- `name` (str): Split name ("val" or "test")
- `df` (pd.DataFrame): Data to evaluate
- `feats` (List[str]): Feature column names
- `model` (xgb.Booster): Trained model
- `cfg` (dict): Configuration
- `output_format` (str): Output format
- `prefix` (str): Base output directory

**Returns**: None (saves tarball)

### Pre-Computed Artifact Management Component

#### `load_precomputed_artifacts(model_artifacts_dir, use_imputation, use_risk_tables, use_features)`
**Purpose**: Auto-detect and load pre-computed artifacts from directory

**Algorithm**:
```python
1. Initialize result dictionary with None values
2. If directory doesn't exist: return empty result
3. If use_imputation:
   a. Check for impute_dict.pkl
   b. If exists: load with pickle, mark as loaded
   c. If fails: log warning, continue
4. If use_risk_tables:
   a. Check for risk_table_map.pkl
   b. If exists: load with pickle, mark as loaded
   c. If fails: log warning, continue
5. If use_features:
   a. Check for selected_features.json
   b. If exists: load with json, mark as loaded
   c. If fails: log warning, continue
6. Return result with loaded artifacts and flags
```

**Parameters**:
- `model_artifacts_dir` (Optional[str]): Artifacts directory
- `use_imputation` (bool): Whether to load imputation
- `use_risk_tables` (bool): Whether to load risk tables
- `use_features` (bool): Whether to load features

**Returns**: `Dict[str, Any]` - Dictionary with loaded artifacts and flags

**Result Structure**:
```python
{
    "impute_dict": dict or None,
    "risk_tables": dict or None,
    "selected_features": list or None,
    "loaded": {
        "imputation": bool,
        "risk_tables": bool,
        "features": bool
    }
}
```

#### `validate_precomputed_data_state(train_df, config, imputation_used, risk_tables_used)`
**Purpose**: Validate data state matches pre-computed artifact flags

**Algorithm**:
```python
1. If imputation_used:
   a. Check tab_field_list columns for NaN values
   b. If NaN found: raise ValueError with column names
   c. Log validation success
2. If risk_tables_used:
   a. Check cat_field_list columns are numeric dtype
   b. If non-numeric found: raise ValueError with column name
   c. Log validation success
```

**Parameters**:
- `train_df` (pd.DataFrame): Training data to validate
- `config` (dict): Configuration dictionary
- `imputation_used` (bool): Whether pre-computed imputation is used
- `risk_tables_used` (bool): Whether pre-computed risk tables are used

**Returns**: None (raises ValueError on validation failure)

**Critical Validation**:
- Prevents incorrect artifact usage
- Catches data/config mismatches early
- Provides clear error messages

### Main Orchestration Component

#### `main(input_paths, output_paths, environ_vars, job_args)`
**Purpose**: Main entry point orchestrating complete training workflow

**Workflow**:
```
1. Extract and log all paths
   ↓
2. Load and validate configuration
   ↓
3. Load train/val/test datasets with format detection
   ↓
4. Preprocessing Artifact Control Decision:
   - Load pre-computed artifacts (if enabled)
   - OR compute inline
   ↓
5. Numerical Imputation:
   - Use pre-computed (skip transformation)
   - OR compute inline (transform data)
   ↓
6. Risk Table Mapping:
   - Use pre-computed (skip transformation)
   - OR compute inline (transform data)
   ↓
7. Feature Selection:
   - Apply if enabled and valid
   - OR use original features
   ↓
8. Prepare DMatrices for XGBoost
   ↓
9. Train XGBoost model
   ↓
10. Save model and artifacts
   ↓
11. Evaluate on validation set
   ↓
12. Evaluate on test set
   ↓
13. Package outputs
```

**Parameters**:
- `input_paths` (Dict[str, str]): Input path mapping
- `output_paths` (Dict[str, str]): Output path mapping
- `environ_vars` (Dict[str, str]): Environment variables
- `job_args` (argparse.Namespace): Command line arguments

**Returns**: None (exits with code 0 on success, 1 on failure)

## Algorithms and Data Structures

### Risk Table Mapping Algorithm

**Problem**: Transform categorical features to numerical risk scores based on target correlation

**Solution Strategy**:
1. Compute target rate for each category value
2. Apply smoothing to handle small sample sizes
3. Use default value for unseen categories
4. Store mapping for inference

**Algorithm**:
```python
# Fit phase (on training data)
for category_value in categorical_column.unique():
    # Count occurrences
    count = (categorical_column == category_value).sum()
    
    # Count positive targets
    positive_count = ((categorical_column == category_value) & 
                     (label_column == 1)).sum()
    
    # Apply smoothing
    if count >= count_threshold:
        risk_score = (positive_count + smooth_factor) / (count + 2 * smooth_factor)
    else:
        risk_score = default_risk
    
    risk_table[category_value] = risk_score

# Transform phase
transformed_column = categorical_column.map(risk_table).fillna(default_risk)
```

**Complexity**: O(n) where n is number of rows

**Key Features**:
- Handles class imbalance via target rate
- Smoothing prevents overfitting on rare categories
- Default value handles unseen categories

### Mean Imputation Algorithm

**Problem**: Fill missing numerical values with stable estimates

**Solution Strategy**:
1. Compute mean on training data only
2. Store mean value for each column
3. Apply same mean to validation/test
4. Validate no NaN remains

**Algorithm**:
```python
# Fit phase (on training data)
for column in tab_field_list:
    mean_value = train_df[column].mean()
    impute_dict[column] = mean_value

# Transform phase
for column in tab_field_list:
    df[column] = df[column].fillna(impute_dict[column])
```

**Complexity**: O(n * m) where n=rows, m=columns

**Advantages**:
- Simple and robust
- No data leakage (fit on train only)
- Preserves distribution center

### Feature Selection Integration Algorithm

**Problem**: Safely integrate pre-computed feature selection with fallback

**Solution Strategy**:
1. Start with original feature list (fallback)
2. Attempt to load selected features
3. Validate loaded features
4. Use selected or fallback based on validation

**Algorithm**:
```python
# Always start with fallback
original_features = config["tab_field_list"] + config["cat_field_list"]

# Try to detect and load
fs_path = detect_feature_selection_artifacts(input_dir)
if fs_path is None:
    return original_features, False

selected_features = load_selected_features(fs_path)
if selected_features is None:
    return original_features, False

# Validate
missing = [f for f in selected_features if f not in train_df.columns]
if missing or len(selected_features) > len(original_features):
    return original_features, False

# Success
return selected_features, True
```

**Complexity**: O(n) where n is number of features

**Safety Features**:
- Multiple validation checks
- Conservative fallback strategy
- Explicit logging at each decision

### Format-Preserving I/O Algorithm

**Problem**: Maintain data format consistency across pipeline

**Solution Strategy**:
1. Detect format from input file extension
2. Store format in configuration
3. Use same format for all outputs

**Algorithm**:
```python
# Detection
def detect_format(file_path):
    suffix = Path(file_path).suffix.lower()
    format_map = {'.csv': 'csv', '.tsv': 'tsv', '.parquet': 'parquet'}
    return format_map.get(suffix, 'csv')

# Save with format
def save_with_format(df, output_path, format_str):
    if format_str == 'csv':
        df.to_csv(f"{output_path}.csv", index=False)
    elif format_str == 'tsv':
        df.to_csv(f"{output_path}.tsv", sep='\t', index=False)
    elif format_str == 'parquet':
        df.to_parquet(f"{output_path}.parquet", index=False)
```

**Complexity**: O(n) for I/O operations

**Benefits**:
- Consistent format throughout pipeline
- Handles mixed formats gracefully
- Clear logging of format decisions

## Performance Characteristics

### Training Performance

| Dataset Size | Training Time | Memory Usage | Best Configuration |
|--------------|---------------|--------------|-------------------|
| 10K samples | ~30s | ~500MB | Default params |
| 100K samples | ~3min | ~2GB | Increase num_round |
| 1M samples | ~30min | ~10GB | Use subsample, colsample |
| 10M samples | ~5hrs | ~50GB | Distributed training |

### Preprocessing Performance

| Operation | 10K rows | 100K rows | 1M rows | Complexity |
|-----------|----------|-----------|---------|------------|
| Load CSV | 0.5s | 3s | 30s | O(n) |
| Load Parquet | 0.1s | 0.5s | 5s | O(n) |
| Imputation | 0.1s | 0.5s | 3s | O(n*m) |
| Risk Tables | 0.2s | 1s | 10s | O(n*c) |
| Feature Selection | 0.01s | 0.05s | 0.5s | O(f) |

**Optimization Tips**:
- Use Parquet format (6-10x faster than CSV)
- Enable pre-computed artifacts for large datasets
- Use subsample/colsample for very large datasets
- Consider distributed training for 10M+ samples

## Error Handling

### Configuration Errors

**ValidationError: Missing required key**
```python
ValueError: Missing required key in config: tab_field_list
```
**Cause**: Required configuration parameter missing

**Handling**: Raised during config validation before training

**Resolution**: Add missing parameter to hyperparameters.json

### Data Loading Errors

**FileNotFoundError: Data file not found**
```python
FileNotFoundError: Training, validation, or test data file not found
```
**Cause**: Missing data files in expected directories

**Handling**: Raised during data loading

**Resolution**: Ensure data files exist in train/val/test subdirectories

### Preprocessing Errors

**ValueError: Pre-computed imputation but data has NaN**
```python
ValueError: USE_PRECOMPUTED_IMPUTATION=true but data contains NaN values in columns: ['price', 'quantity']
```
**Cause**: Environment variable indicates data is imputed but NaN values exist

**Handling**: Raised during data state validation

**Resolution**: Either set USE_PRECOMPUTED_IMPUTATION=false or ensure data is pre-imputed

**ValueError: Pre-computed risk tables but data not numeric**
```python
ValueError: USE_PRECOMPUTED_RISK_TABLES=true but column 'marketplace' is not numeric
```
**Cause**: Environment variable indicates data is risk-mapped but categorical columns not numeric

**Handling**: Raised during data state validation

**Resolution**: Either set USE_PRECOMPUTED_RISK_TABLES=false or ensure data is pre-transformed

### Training Errors

**ValueError: NaN or inf values in training data**
```python
ValueError: Training data contains NaN or inf values after preprocessing
```
**Cause**: Incomplete preprocessing or data quality issues

**Handling**: Raised during DMatrix preparation

**Resolution**: Check preprocessing pipeline, verify imputation completeness

### Feature Selection Errors

**Warning: Selected features missing from data**
```python
WARNING: Selected features missing from data: ['missing_feature']
```
**Cause**: Feature selection artifacts reference columns not in data

**Handling**: Logged as warning, falls back to original features

**Resolution**: Verify feature selection output matches training data schema

## Best Practices

### For Production Deployments

1. **Use Pre-Computed Artifacts**
   ```bash
   export USE_PRECOMPUTED_IMPUTATION=true
   export USE_PRECOMPUTED_RISK_TABLES=true
   export USE_PRECOMPUTED_FEATURES=true
   ```
   - Separates preprocessing from training
   - Enables independent scaling
   - Faster training iterations

2. **Enable Early Stopping**
   ```json
   {
       "num_round": 500,
       "early_stopping_rounds": 10
   }
   ```
   - Prevents overfitting
   - Reduces training time
   - Automatically finds optimal iterations

3. **Use Appropriate Regularization**
   ```json
   {
       "lambda_xgb": 1.0,
       "alpha_xgb": 0.1,
       "subsample": 0.8,
       "colsample_bytree": 0.8
   }
   ```
   - Controls model complexity
   - Improves generalization
   - Reduces overfitting risk

4. **Handle Class Imbalance**
   ```json
   {
       "class_weights": [1.0, 5.0]
   }
   ```
   - Critical for imbalanced datasets
   - Weight inversely proportional to frequency
   - Improves minority class performance

5. **Use Parquet Format**
   - 6-10x faster I/O than CSV
   - Smaller file sizes
   - Better compression

### For Development

1. **Start with Small Datasets**
   - Faster iteration cycles
   - Easier debugging
   - Validate pipeline end-to-end

2. **Use Inline Preprocessing Initially**
   ```bash
   export USE_PRECOMPUTED_IMPUTATION=false
   export USE_PRECOMPUTED_RISK_TABLES=false
   ```
   - Simpler setup
   - Easier to debug
   - Self-contained script

3. **Monitor Training Logs**
   - Check label distributions
   - Validate preprocessing
   - Track convergence

4. **Test with Default Hyperparameters**
   - Establish baseline performance
   - Identify issues early
   - Guide hyperparameter tuning

### For Performance Optimization

1. **Tune Learning Rate**
   ```json
   {
       "eta": 0.1,  # Start here
       "num_round": 100
   }
   ```
   - Lower eta requires more rounds
   - Higher eta trains faster but may miss optimum
   - Typical range: 0.01-0.3

2. **Use Subsampling for Large Datasets**
   ```json
   {
       "subsample": 0.8,
       "colsample_bytree": 0.8
   }
   ```
   - Reduces training time
   - Acts as regularization
   - Improves generalization

3. **Optimize Tree Depth**
   ```json
   {
       "max_depth": 6  # Default
   }
   ```
   - Deeper trees: more complex models
   - Shallower trees: faster, more regularized
   - Typical range: 3-10

4. **Use Format-Appropriate I/O**
   - Parquet for large datasets
   - CSV for small datasets or debugging
   - TSV for tab-separated data

## Example Configurations

### Example 1: Binary Fraud Detection
```json
{
    "tab_field_list": ["price", "quantity", "seller_rating", "buyer_age"],
    "cat_field_list": ["marketplace", "category", "payment_method"],
    "label_name": "is_fraud",
    "id_name": "transaction_id",
    "multiclass_categories": [0, 1],
    "is_binary": true,
    "class_weights": [1.0, 10.0],
    "eta": 0.1,
    "max_depth": 6,
    "num_round": 100,
    "early_stopping_rounds": 10,
    "smooth_factor": 0.1,
    "count_threshold": 10
}
```
**Use Case**: Fraud detection with severe class imbalance

### Example 2: Multiclass Risk Scoring
```json
{
    "tab_field_list": ["amount", "frequency", "recency"],
    "cat_field_list": ["country", "merchant_category"],
    "label_name": "risk_level",
    "multiclass_categories": ["low", "medium", "high", "critical"],
    "is_binary": false,
    "num_classes": 4,
    "class_weights": [1.0, 2.0, 5.0, 10.0],
    "eta": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "num_round": 200
}
```
**Use Case**: Multi-level risk scoring with ordered classes

### Example 3: With Pre-Computed Artifacts
```bash
# Environment variables
export USE_PRECOMPUTED_IMPUTATION=true
export USE_PRECOMPUTED_RISK_TABLES=true
export USE_PRECOMPUTED_FEATURES=true
```
```json
{
    "tab_field_list": ["price", "quantity", "rating"],
    "cat_field_list": ["marketplace", "category"],
    "label_name": "target",
    "multiclass_categories": [0, 1],
    "eta": 0.1,
    "max_depth": 6,
    "num_round": 150
}
```
**Use Case**: Production pipeline with upstream preprocessing

## Integration Patterns

### Upstream Integration (Preprocessing)

```
MissingValueImputation
   ↓ (outputs: imputed train/val/test + impute_dict.pkl)
RiskTableMapping
   ↓ (outputs: transformed train/val/test + risk_table_map.pkl)
FeatureSelection
   ↓ (outputs: filtered train/val/test + selected_features.json)
XGBoostTraining (USE_PRECOMPUTED_*=true)
   ↓ (outputs: model + evaluation)
```

**Artifact Flow**:
1. Each preprocessing step outputs artifacts
2. Training loads artifacts from model_artifacts_input
3. Training validates data state matches artifacts
4. All artifacts packaged into final model

### Downstream Integration (Inference)

```
XGBoostTraining
   ↓ (outputs: xgboost_model.bst + preprocessing artifacts)
XGBoostModelInference
   ↓ (uses: model + artifacts for transformation)
ModelMetricsComputation
   ↓ (computes: detailed metrics)
```

**Deployment Flow**:
1. Training produces complete model package
2. Inference applies same preprocessing
3. Metrics computation validates performance

### Complete Pipeline Example

```
1. DummyDataLoading/CradleDataLoading
   ↓ (raw_data)
2. TabularPreprocessing
   ↓ (cleaned data, train/val/test splits)
3. MissingValueImputation
   ↓ (imputed data + impute_dict.pkl)
4. RiskTableMapping
   ↓ (transformed data + risk_table_map.pkl)
5. FeatureSelection
   ↓ (filtered data + selected_features.json)
6. XGBoostTraining
   ↓ (model + all artifacts)
7. ModelMetricsComputation
   ↓ (comprehensive metrics)
8. Package
   ↓ (deployment package)
```

## Troubleshooting

### Issue 1: Slow Training

**Symptom**: Training taking much longer than expected

**Common Causes**:
1. Large dataset without subsampling
2. Too many boosting rounds
3. Deep trees
4. CSV I/O instead of Parquet

**Solution**:
```json
{
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_depth": 6,  # Reduce if still slow
    "num_round": 100  # Reduce if appropriate
}
```
Also: Convert data to Parquet format

### Issue 2: Poor Performance on Test Set

**Symptom**: Good validation metrics but poor test metrics

**Common Causes**:
1. Overfitting
2. Data leakage
3. Different distributions

**Solution**:
```json
{
    "lambda_xgb": 2.0,  # Increase regularization
    "alpha_xgb": 0.5,
    "subsample": 0.7,
    "max_depth": 5,  # Reduce complexity
    "early_stopping_rounds": 10
}
```

### Issue 3: Memory Errors

**Symptom**: Script crashes with out-of-memory error

**Common Causes**:
1. Dataset too large for memory
2. Too many features
3. Multiple copies of data

**Solution**:
- Use Parquet format (more memory efficient)
- Enable feature selection
- Process data in chunks if possible
- Use instance with more memory

### Issue 4: Class Imbalance Issues

**Symptom**: Model predicts only majority class

**Common Causes**:
1. No class weights specified
2. Insufficient minority class samples
3. Too aggressive early stopping

**Solution**:
```json
{
    "class_weights": [1.0, 10.0],  # Boost minority class
    "scale_pos_weight": 10.0,  # Alternative for binary
    "early_stopping_rounds": 15  # More patience
}
```

### Issue 5: Pre-Computed Artifact Errors

**Symptom**: Validation errors about data state

**Common Causes**:
1. Environment variables don't match data state
2. Artifacts from different preprocessing
3. Missing artifact files

**Solution**:
1. Verify data is preprocessed if USE_PRECOMPUTED_*=true
2. Check artifact files exist in model_artifacts_input
3. Ensure artifacts match current data schema
4. Set USE_PRECOMPUTED_*=false if unsure

## References

### Related Scripts

- **Training Scripts:**
  - [`pytorch_training.py`](pytorch_training_script.md): PyTorch training (comparison)
  - [`lightgbm_training.py`](../../projects/ab_lightgbm/docker/lightgbm_training.py): LightGBM training (similar)

- **Preprocessing Scripts:**
  - [`missing_value_imputation.py`](missing_value_imputation_script.md): Numerical imputation
  - [`risk_table_mapping.py`](risk_table_mapping_script.md): Categorical feature transformation
  - [`feature_selection.py`](feature_selection_script.md): Feature selection

- **Evaluation Scripts:**
  - [`xgboost_model_eval.py`](xgboost_model_eval_script.md): Model evaluation
  - [`xgboost_model_inference.py`](xgboost_model_inference_script.md): Inference only
  - [`model_metrics_computation.py`](model_metrics_computation_script.md): Comprehensive metrics

- **Deployment Scripts:**
  - [`package.py`](package_script.md): Model packaging
  - [`payload.py`](payload_script.md): Test payload generation

### Related Documentation

- **Contract**: [`src/cursus/steps/contracts/xgboost_training_contract.py`](../../src/cursus/steps/contracts/xgboost_training_contract.py) - Complete contract specification
- **Hyperparameters**: [`projects/atoz_xgboost/docker/hyperparams/hyperparameters_xgboost.py`](../../projects/atoz_xgboost/docker/hyperparams/hyperparameters_xgboost.py) - Hyperparameter definitions
- **Config**: XGBoost training configuration class in step configs
- **Builder**: XGBoost training step builder
- **Specification**: XGBoost training step specification in registry

### Related Design Documents

No specific design documents currently exist for this training script. General training step patterns are documented in:

- **[Training Step Builder Patterns](../1_design/training_step_builder_patterns.md)**: Common patterns for training step implementation
- **[Training Step Alignment Validation Patterns](../1_design/training_step_alignment_validation_patterns.md)**: Validation patterns for training steps

### External References

- **[XGBoost Documentation](https://xgboost.readthedocs.io/)**: Official XGBoost documentation
- **[XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)**: Complete parameter reference
- **[scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)**: Metrics computation reference
- **[Pandas Documentation](https://pandas.pydata.org/docs/)**: DataFrame operations
- **[AWS CodeArtifact](https://docs.aws.amazon.com/codeartifact/)**: Secure package repository documentation
