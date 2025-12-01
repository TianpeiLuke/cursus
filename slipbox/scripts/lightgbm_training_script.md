---
tags:
  - code
  - training_script
  - lightgbm
  - gradient_boosting
keywords:
  - LightGBM training
  - risk table mapping
  - numerical imputation
  - tabular classification
  - binary classification
  - multiclass classification
  - feature preprocessing
  - gradient boosting
topics:
  - LightGBM training
  - tabular machine learning
  - preprocessing integration
  - model training workflows
language: python
date of note: 2025-11-18
---

# LightGBM Training Script

## Overview

The `lightgbm_training.py` script implements a comprehensive LightGBM training pipeline for tabular data classification with integrated preprocessing capabilities including numerical imputation and dual-mode categorical feature handling (native categorical or risk table mapping).

The script provides a production-ready training workflow that supports both inline preprocessing computation and pre-computed artifact reuse from upstream steps. It handles binary and multiclass classification with configurable hyperparameters, class imbalance handling, comprehensive evaluation metrics, and format-preserving I/O across CSV, TSV, and Parquet formats.

Key capabilities include flexible preprocessing artifact control (compute inline vs reuse pre-computed), **dual-mode categorical handling** (LightGBM's native categorical features via dictionary encoding OR traditional risk table mapping), mean-based numerical imputation, LightGBM model training with early stopping, comprehensive evaluation with ROC/PR curves, and packaged output artifacts in tarball format for deployment. The native categorical mode leverages LightGBM's built-in categorical feature support for better performance and accuracy, while risk table mode maintains backward compatibility with XGBoost-style workflows.

## Purpose and Major Tasks

### Primary Purpose
Train LightGBM gradient boosting models for tabular classification tasks with integrated preprocessing, supporting flexible artifact reuse patterns for efficient pipeline orchestration and production deployment.

### Major Tasks

1. **Package Installation Management**: Install required packages from secure CodeArtifact or public PyPI based on configuration
2. **Configuration Loading**: Load and validate hyperparameters from JSON with Pydantic schema validation
3. **Data Loading**: Load train/validation/test datasets with automatic format detection (CSV, TSV, Parquet)
4. **Preprocessing Artifact Control**: Detect and load pre-computed artifacts or compute inline based on environment flags
5. **Numerical Imputation**: Apply mean-based imputation for missing values (inline or pre-computed)
6. **Categorical Feature Handling**: Dual-mode support:
   - **Native Mode** (recommended): Dictionary encoding for LightGBM's native categorical features
   - **Risk Table Mode**: Traditional risk table mapping for backward compatibility
7. **Feature Selection Integration**: Apply pre-computed feature selection if available
8. **Model Training**: Train LightGBM model with configurable hyperparameters and early stopping
9. **Comprehensive Evaluation**: Compute metrics, generate predictions, and create ROC/PR visualizations
10. **Artifact Packaging**: Save model, preprocessing artifacts, and evaluation outputs

## Script Contract

### Entry Point
```
lightgbm_training.py
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
- `risk_table_map.pkl`: Pre-computed risk tables (risk table mode only)
- `categorical_mappings.pkl`: Pre-computed dictionary encodings (native categorical mode only)
- `selected_features.json`: Pre-computed feature selection results

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `model_output` | `/opt/ml/model` | Primary model artifacts directory |
| `evaluation_output` | `/opt/ml/output/data` | Evaluation results and metrics |

**Model Output Contents**:
- `lightgbm_model.txt`: Trained LightGBM model in text format (human-readable, version-stable)
- `categorical_config.json`: Categorical mode configuration (native vs risk table)
- **Mode-specific artifacts**:
  - Native mode: `categorical_mappings.pkl` (dictionary encodings) + `categorical_mappings.json` (human-readable)
  - Risk table mode: `risk_table_map.pkl` (risk table mappings)
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
| `USE_PRECOMPUTED_RISK_TABLES` | `"false"` | Use pre-computed risk table artifacts | When risk mapping done upstream (risk table mode only) |
| `USE_PRECOMPUTED_FEATURES` | `"false"` | Use pre-computed feature selection | When features already selected upstream |
| `USE_NATIVE_CATEGORICAL` | `"true"` | Use LightGBM's native categorical features | **Recommended** for better performance and accuracy |

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

#### LightGBM Hyperparameters

**Note**: Script maps XGBoost parameter names to LightGBM equivalents for consistency:

| XGBoost Parameter (Config) | LightGBM Parameter (Internal) | Description | Range/Default |
|---------------------------|-------------------------------|-------------|---------------|
| `eta` | `learning_rate` | Learning rate (shrinkage rate) | Default: `0.1`, Range: `0.0-1.0` |
| `gamma` | `min_split_gain` | Minimum loss reduction for split | Default: `0`, Range: `0+` |
| `max_depth` | `max_depth` | Maximum tree depth | Default: `6`, Range: `-1` (no limit) or `1+` |
| `subsample` | `bagging_fraction` | Subsample ratio of training instances | Default: `1`, Range: `0.0-1.0` |
| `colsample_bytree` | `feature_fraction` | Subsample ratio of features per tree | Default: `1`, Range: `0.0-1.0` |
| `lambda_xgb` | `lambda_l2` | L2 regularization term | Default: `1`, Range: `0+` |
| `alpha_xgb` | `lambda_l1` | L1 regularization term | Default: `0`, Range: `0+` |

**Additional LightGBM-Specific Parameters**:
- `bagging_freq`: Automatically set to `1` if `subsample < 1`, otherwise `0`
- `verbose`: Set to `-1` (silent mode)

#### Training Control
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `num_round` | `int` | No | Number of boosting rounds | Default: `100`, Typical: `50-500` |
| `early_stopping_rounds` | `int` | No | Early stopping patience | Default: `10`, Typical: `5-20` |

#### Categorical Feature Parameters

**Native Categorical Mode** (use_native_categorical=True, recommended):
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `use_native_categorical` | `bool` | No | Use LightGBM's native categorical features | Default: `true` (recommended) |
| `min_data_per_group` | `int` | No | Minimum data per categorical group | Default: `100`, Range: `1+` |
| `cat_smooth` | `float` | No | Categorical feature smoothing | Default: `10.0`, Range: `0+` |
| `max_cat_threshold` | `int` | No | Max number of categorical bins | Default: `32`, Range: `1-255` |

**Risk Table Mode** (use_native_categorical=False, backward compatibility):
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `use_native_categorical` | `bool` | No | Use risk table mapping instead | Default: `false` |
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
- Tabular features (specified in `tab_field_list`): Numerical features for LightGBM
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
- **Native mode** (recommended): Encoded to integers, LightGBM handles natively
- **Risk table mode**: Transformed to risk scores (float)
- After transformation, will be numeric type (int32 for native, float32 for risk table)

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

**USE_PRECOMPUTED_RISK_TABLES=true** (risk table mode only):
- Categorical columns in `cat_field_list` must be numeric type (float)
- Data already risk-mapped upstream
- Script validates numeric dtype
- Not applicable in native categorical mode

**USE_PRECOMPUTED_FEATURES=true**:
- Data may contain additional columns beyond selected features
- Script filters to selected features only
- Validates selected features exist in data

## Output Data Structure

### Model Output Directory Structure

```
/opt/ml/model/
├── lightgbm_model.txt             # LightGBM model (text format)
├── categorical_config.json        # Categorical mode configuration
├── categorical_mappings.pkl       # Dictionary encodings (native mode)
├── categorical_mappings.json      # Human-readable encodings (native mode)
├── risk_table_map.pkl             # Risk tables (risk table mode)
├── impute_dict.pkl                # Imputation values dictionary
├── feature_importance.json        # Feature importance scores
├── feature_columns.txt            # Ordered feature column names
└── hyperparameters.json           # Model hyperparameters
```

**lightgbm_model.txt**:
- Text format LightGBM model (human-readable, version-stable)
- Can be loaded with `lgb.Booster(model_file='lightgbm_model.txt')`
- Alternative formats: `.bin` (binary), `.pkl`/`.joblib` (scikit-learn API)
- Contains all tree structures and parameters

**categorical_config.json**:
```json
{
    "use_native_categorical": true,
    "categorical_features": ["marketplace", "category", "payment_method"],
    "min_data_per_group": 100,
    "cat_smooth": 10.0,
    "max_cat_threshold": 32
}
```

**categorical_mappings.pkl** (native mode):
```python
{
    "marketplace": {
        "US": 0,
        "UK": 1,
        "DE": 2,
        "FR": 3,
        "__unknown__": -1  # For unseen categories
    },
    "category": {
        "Electronics": 0,
        "Books": 1,
        "__unknown__": -1
    },
    ...
}
```

**risk_table_map.pkl** (risk table mode):
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
# Feature columns in exact order required for LightGBM model inference
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
    "price": 145,
    "marketplace": 102,
    "quantity": 89,
    "rating": 67,
    "category": 34
}
```
Note: LightGBM uses split-based importance by default (number of times feature used in splits)

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

### Parameter Mapping Component

#### XGBoost to LightGBM Parameter Mapping
**Purpose**: Maintain consistency with XGBoost training configuration while using LightGBM

**Mapping Table**:
```python
{
    "eta": "learning_rate",           # Learning rate
    "gamma": "min_split_gain",        # Min loss reduction
    "max_depth": "max_depth",         # Tree depth (same name)
    "subsample": "bagging_fraction",  # Row sampling
    "colsample_bytree": "feature_fraction",  # Feature sampling
    "lambda_xgb": "lambda_l2",        # L2 regularization
    "alpha_xgb": "lambda_l1"          # L1 regularization
}
```

**Additional Logic**:
- `bagging_freq` = 1 if `subsample < 1` else 0
- `verbose` = -1 (silent mode for CloudWatch logs)

### Dataset Preparation Component

#### `prepare_datasets(config, train_df, val_df)`
**Purpose**: Prepare LightGBM Dataset objects from DataFrames with categorical feature support

**Algorithm**:
```python
1. Maintain exact ordering of features:
   a. feature_columns = tab_field_list + cat_field_list
2. Extract feature matrices:
   a. X_train = train_df[feature_columns].copy()
   b. X_val = val_df[feature_columns].copy()
3. Apply proper data types based on categorical mode:
   a. If use_native_categorical:
      - Numerical features: float32
      - Categorical features: int32
   b. Else (risk table mode):
      - All features: float32
4. Validate no NaN/inf values remain
5. Get label arrays:
   a. y_train, y_val from label_name column
6. Handle class weights for multiclass:
   a. If multiclass and class_weights provided:
      - Create sample_weights array
      - Map class weights to each sample
7. Specify categorical features for LightGBM:
   a. If use_native_categorical:
      - categorical_feature = cat_field_list
   b. Else:
      - categorical_feature = None
8. Create LightGBM Dataset objects:
   a. train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                               categorical_feature=categorical_feature)
   b. val_set = lgb.Dataset(X_val, label=y_val, reference=train_set,
                             categorical_feature=categorical_feature)
9. Set feature names in Dataset
10. Return train_set, val_set, feature_columns
```

**Parameters**:
- `config` (dict): Configuration with feature lists
- `train_df`, `val_df` (pd.DataFrame): Processed datasets

**Returns**: `Tuple[lgb.Dataset, lgb.Dataset, List[str]]` - (train_set, val_set, feature_columns)

**Key Differences from XGBoost**:
- Uses `lgb.Dataset` instead of `xgb.DMatrix`
- Sample weights set during Dataset creation for multiclass
- Reference parameter links validation set to training set
- **Categorical feature support**: `categorical_feature` parameter for native mode
- **Data types**: Supports int32 for categorical features (native mode)

### Model Training Component

#### `train_model(config, train_set, val_set)`
**Purpose**: Train LightGBM model with configured hyperparameters

**Algorithm**:
```python
1. Map XGBoost parameters to LightGBM:
   a. learning_rate = eta
   b. min_split_gain = gamma
   c. max_depth = max_depth
   d. bagging_fraction = subsample
   e. feature_fraction = colsample_bytree
   f. lambda_l2 = lambda_xgb
   g. lambda_l1 = alpha_xgb
2. Add categorical parameters if using native categorical:
   a. If use_native_categorical:
      - min_data_per_group (default: 100)
      - cat_smooth (default: 10.0)
      - max_cat_threshold (default: 32)
3. Set bagging_freq:
   a. If subsample < 1: bagging_freq = 1
   b. Else: bagging_freq = 0
4. Set objective based on classification type:
   a. If is_binary:
      - objective = "binary"
      - Handle class weights via scale_pos_weight
   b. Else:
      - objective = "multiclass"
      - num_class = num_classes
      - Class weights via sample weights (already in Dataset)
4. Configure callbacks:
   a. log_evaluation(period=1)
   b. early_stopping if configured
6. Call lgb.train() with:
   a. params dictionary
   b. train_set (training data with categorical_feature info)
   c. num_boost_round
   d. valid_sets (train and validation)
   e. callbacks
7. Return trained model
```

**Native Categorical Feature Benefits**:
- LightGBM can find optimal splits on categorical features directly
- No information loss from encoding (maintains semantic meaning)
- Better accuracy and generalization
- Faster training (no need for risk table computation)
- Memory efficient (int32 vs float32/64)

**Parameters**:
- `config` (dict): Configuration with LightGBM hyperparameters
- `train_set` (lgb.Dataset): Training data
- `val_set` (lgb.Dataset): Validation data

**Returns**: `lgb.Booster` - Trained LightGBM model

**Class Weight Handling**:
- Binary: Uses `scale_pos_weight = weight[1] / weight[0]`
- Multiclass: Sample weights applied during Dataset creation

**Key Differences from XGBoost**:
- Uses `lgb.train()` instead of `xgb.train()`
- Different parameter names and structure
- Callbacks instead of inline arguments

### Model Saving Component

#### `save_artifacts(model, risk_tables, impute_dict, model_path, feature_columns, config, categorical_mappings)`
**Purpose**: Save trained model and preprocessing artifacts with mode-specific handling

**Algorithm**:
```python
1. Create model_path directory
2. Save LightGBM model:
   a. model.save_model(os.path.join(model_path, "lightgbm_model.txt"))
   b. Text format for version stability
3. Save categorical configuration:
   a. categorical_config.json with mode and parameters
4. Save mode-specific categorical artifacts:
   a. If use_native_categorical:
      - Save categorical_mappings.pkl (pickle format)
      - Save categorical_mappings.json (human-readable)
   b. Else (risk table mode):
      - Save risk_table_map.pkl (pickle format)
5. Save imputation dictionary (pickle format)
6. Save feature importance:
   a. importance = model.feature_importance()
   b. Map to feature names
   c. Save as JSON
7. Save feature columns with ordering
8. Save hyperparameters as JSON
```

**Parameters**:
- `model` (lgb.Booster): Trained model
- `risk_tables` (dict): Risk table mappings
- `impute_dict` (dict): Imputation values
- `model_path` (str): Output directory
- `feature_columns` (List[str]): Feature column names
- `config` (dict): Hyperparameters

**Returns**: None (saves files)

**Key Differences from XGBoost**:
- Saves as `lightgbm_model.txt` (text) instead of `.bst` (binary)
- Feature importance extracted via `model.feature_importance()` (returns array)
- **Dual-mode artifacts**: Saves categorical_mappings.pkl (native) OR risk_table_map.pkl (risk table)
- **Mode configuration**: Saves categorical_config.json for inference consistency

## Algorithms and Data Structures

### LightGBM Tree Growth Algorithm

**Problem**: Build efficient gradient boosting trees for classification

**LightGBM Strategy** (Leaf-Wise Growth):
1. Grow trees leaf-by-leaf (best-first)
2. Choose leaf with maximum delta loss
3. Split that leaf to minimize loss
4. Repeat until max_depth or min_split_gain threshold

**Comparison with XGBoost** (Level-Wise Growth):
- XGBoost: Grows all leaves at same level before moving to next level
- LightGBM: Grows best leaf first (potentially deeper, more accurate trees)
- LightGBM: Faster training, better accuracy, higher risk of overfitting on small data

**Algorithm**:
```python
# Simplified leaf-wise growth
leaves = [root_leaf]
while len(leaves) < num_leaves:
    # Find leaf with maximum gain
    best_leaf = max(leaves, key=lambda leaf: leaf.split_gain)
    
    if best_leaf.split_gain < min_split_gain:
        break  # Stop if no significant gain
    
    # Split best leaf
    left_child, right_child = split_leaf(best_leaf)
    leaves.remove(best_leaf)
    leaves.extend([left_child, right_child])
```

**Complexity**: O(n * d) per tree where n=samples, d=features

**Key Features**:
- Histogram-based splitting (faster than exact XGBoost)
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB) for high-dimensional data

### Parameter Mapping Data Structure

**Problem**: Maintain consistent configuration across XGBoost and LightGBM

**Solution Strategy**:
1. Accept XGBoost parameter names in config
2. Map to LightGBM equivalents internally
3. Apply LightGBM-specific logic (e.g., bagging_freq)

**Mapping Dictionary**:
```python
PARAM_MAPPING = {
    # Learning parameters
    "eta": "learning_rate",
    "gamma": "min_split_gain",
    
    # Tree structure
    "max_depth": "max_depth",  # Same name
    
    # Sampling parameters
    "subsample": "bagging_fraction",
    "colsample_bytree": "feature_fraction",
    
    # Regularization
    "lambda_xgb": "lambda_l2",
    "alpha_xgb": "lambda_l1"
}

# Additional logic
if params["bagging_fraction"] < 1:
    params["bagging_freq"] = 1
```

**Benefits**:
- Consistent configuration interface
- Easy migration from XGBoost to LightGBM
- Backward compatibility

### Class Weight Handling

**Problem**: Handle imbalanced datasets in binary and multiclass scenarios

**Binary Classification**:
```python
if class_weights and len(class_weights) == 2:
    params["scale_pos_weight"] = class_weights[1] / class_weights[0]
```

**Multiclass Classification**:
```python
# During Dataset creation
sample_weights = np.ones(len(y_train))
for class_idx, weight in enumerate(class_weights):
    sample_weights[y_train == class_idx] = weight

train_set = lgb.Dataset(X, y, weight=sample_weights)
```

**Complexity**: O(n) for multiclass weight assignment

## Performance Characteristics

### Training Performance

| Dataset Size | Training Time | Memory Usage | Best Configuration |
|--------------|---------------|--------------|-------------------|
| 10K samples | ~20s | ~400MB | Default params |
| 100K samples | ~2min | ~1.5GB | Faster than XGBoost |
| 1M samples | ~20min | ~8GB | Use histogram mode |
| 10M samples | ~3hrs | ~40GB | Distributed training |

**LightGBM vs XGBoost Speed**:
- LightGBM typically 2-3x faster than XGBoost
- Memory usage ~20-30% lower
- Histogram-based splitting more efficient

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
- LightGBM handles high-dimensional data better than XGBoost
- Consider GOSS and EFB for very large datasets

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

### Training Errors

**ValueError: NaN or inf values in training data**
```python
ValueError: Training data contains NaN or inf values after preprocessing
```
**Cause**: Incomplete preprocessing or data quality issues

**Handling**: Raised during Dataset preparation

**Resolution**: Check preprocessing pipeline, verify imputation completeness

## Best Practices

### For Production Deployments

1. **Use Native Categorical Mode (Recommended)**
   ```json
   {
       "use_native_categorical": true,
       "min_data_per_group": 100,
       "cat_smooth": 10.0,
       "max_cat_threshold": 32
   }
   ```
   - Better accuracy and generalization
   - Faster training (no risk table computation)
   - Memory efficient (int32 encoding)
   - Leverages LightGBM's built-in categorical handling

2. **Use Pre-Computed Artifacts**
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
   - Reduces overfitting risk (especially with leaf-wise growth)

4. **Control Tree Depth**
   ```json
   {
       "max_depth": 6
   }
   ```
   - LightGBM's leaf-wise growth can create very deep trees
   - Set max_depth to prevent overfitting
   - Typical range: 3-10 for most datasets

6. **Use Parquet Format**
   - 6-10x faster I/O than CSV
   - Smaller file sizes
   - Better compression

### For Development

1. **Start with Conservative Parameters**
   ```json
   {
       "eta": 0.1,
       "max_depth": 6,
       "num_round": 100
   }
   ```
   - Prevent overfitting during development
   - Establish baseline performance
   - Tune gradually

2. **Monitor Training Logs**
   - Watch for overfitting (train vs validation gap)
   - Check convergence behavior
   - Validate label distributions

3. **Start with Native Categorical Mode**
   ```json
   {
       "use_native_categorical": true
   }
   ```
   - Simpler than risk table mapping
   - Better performance out-of-the-box
   - Less preprocessing complexity

4. **Use Inline Preprocessing Initially**
   ```bash
   export USE_PRECOMPUTED_IMPUTATION=false
   export USE_PRECOMPUTED_RISK_TABLES=false
   ```
   - Simpler setup
   - Easier to debug
   - Self-contained script

### For Performance Optimization

1. **Tune Learning Rate**
   ```json
   {
       "eta": 0.1,
       "num_round": 100
   }
   ```
   - Lower eta requires more rounds
   - LightGBM typically converges faster than XGBoost
   - Typical range: 0.01-0.3

2. **Use Subsampling**
   ```json
   {
       "subsample": 0.8,
       "colsample_bytree": 0.8
   }
   ```
   - Reduces training time
   - Acts as regularization
   - Improves generalization

3. **Leverage LightGBM-Specific Features**
   - Use histogram-based splitting (automatic)
   - Consider GOSS for very large datasets
   - Use EFB for high-dimensional sparse data

## Example Configurations

### Example 1: Binary Fraud Detection (Native Categorical Mode)
```json
{
    "tab_field_list": ["price", "quantity", "seller_rating", "buyer_age"],
    "cat_field_list": ["marketplace", "category", "payment_method"],
    "label_name": "is_fraud",
    "id_name": "transaction_id",
    "multiclass_categories": [0, 1],
    "is_binary": true,
    "class_weights": [1.0, 10.0],
    "use_native_categorical": true,
    "min_data_per_group": 100,
    "cat_smooth": 10.0,
    "eta": 0.1,
    "max_depth": 6,
    "num_round": 100,
    "early_stopping_rounds": 10
}
```
**Use Case**: Fraud detection with severe class imbalance using native categorical features

### Example 2: Multiclass Risk Scoring (Native Categorical Mode)
```json
{
    "tab_field_list": ["amount", "frequency", "recency"],
    "cat_field_list": ["country", "merchant_category"],
    "label_name": "risk_level",
    "multiclass_categories": ["low", "medium", "high", "critical"],
    "is_binary": false,
    "num_classes": 4,
    "class_weights": [1.0, 2.0, 5.0, 10.0],
    "use_native_categorical": true,
    "eta": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "num_round": 200
}
```
**Use Case**: Multi-level risk scoring with ordered classes using native categorical features

### Example 3: High-Dimensional Data with Native Categorical
```json
{
    "tab_field_list": ["feature1", "feature2", ..., "feature500"],
    "cat_field_list": ["category1", "category2"],
    "label_name": "target",
    "multiclass_categories": [0, 1],
    "use_native_categorical": true,
    "max_cat_threshold": 64,
    "eta": 0.05,
    "max_depth": -1,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "num_round": 300
}
```
**Use Case**: High-dimensional dataset leveraging LightGBM's efficiency with native categorical features

### Example 4: Legacy Risk Table Mode (Backward Compatibility)
```json
{
    "tab_field_list": ["price", "quantity"],
    "cat_field_list": ["marketplace", "category"],
    "label_name": "target",
    "multiclass_categories": [0, 1],
    "use_native_categorical": false,
    "smooth_factor": 0.1,
    "count_threshold": 10,
    "eta": 0.1,
    "max_depth": 6,
    "num_round": 100
}
```
**Use Case**: XGBoost-style workflow using risk table mapping for backward compatibility

## Integration Patterns

### Upstream Integration (Preprocessing)

**Native Categorical Mode (Recommended)**:
```
MissingValueImputation
   ↓ (outputs: imputed train/val/test + impute_dict.pkl)
DictionaryEncoding (optional upstream)
   ↓ (outputs: encoded train/val/test + categorical_mappings.pkl)
FeatureSelection
   ↓ (outputs: filtered train/val/test + selected_features.json)
LightGBMTraining (USE_NATIVE_CATEGORICAL=true)
   ↓ (outputs: model + categorical_mappings.pkl + evaluation)
```

**Risk Table Mode (Backward Compatibility)**:
```
MissingValueImputation
   ↓ (outputs: imputed train/val/test + impute_dict.pkl)
RiskTableMapping
   ↓ (outputs: transformed train/val/test + risk_table_map.pkl)
FeatureSelection
   ↓ (outputs: filtered train/val/test + selected_features.json)
LightGBMTraining (USE_PRECOMPUTED_RISK_TABLES=true)
   ↓ (outputs: model + risk_table_map.pkl + evaluation)
```

**Artifact Flow**:
1. Each preprocessing step outputs artifacts
2. Training loads artifacts from model_artifacts_input
3. Training validates data state matches artifacts
4. All artifacts packaged into final model

### Downstream Integration (Inference)

```
LightGBMTraining
   ↓ (outputs: lightgbm_model.txt + preprocessing artifacts)
LightGBMModelInference
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
6. LightGBMTraining
   ↓ (model + all artifacts)
7. ModelMetricsComputation
   ↓ (comprehensive metrics)
8. Package
   ↓ (deployment package)
```

## Troubleshooting

### Issue 1: Overfitting with Leaf-Wise Growth

**Symptom**: Training accuracy high but validation accuracy poor

**Common Causes**:
1. LightGBM's leaf-wise growth creating overly deep trees
2. Insufficient regularization
3. Too many boosting rounds

**Solution**:
```json
{
    "max_depth": 5,
    "lambda_xgb": 2.0,
    "alpha_xgb": 0.5,
    "subsample": 0.7,
    "num_round": 100
}
```
Also: Increase early_stopping_rounds

### Issue 2: Slow Training Despite LightGBM

**Symptom**: Training slower than expected

**Common Causes**:
1. Dataset not leveraging histogram optimization
2. Too many features without sampling
3. Validation set too large

**Solution**:
```json
{
    "colsample_bytree": 0.8,
    "subsample": 0.8
}
```
Also: Use Parquet format, enable pre-computed artifacts

### Issue 3: Memory Errors

**Symptom**: Out-of-memory errors during training

**Common Causes**:
1. Dataset too large for memory
2. Too many trees in memory

**Solution**:
- Use Parquet format (more memory efficient)
- Reduce max_depth
- Use subsampling
- Process data in chunks if possible

### Issue 4: Class Imbalance Issues

**Symptom**: Model predicts only majority class

**Common Causes**:
1. No class weights specified
2. Insufficient minority class samples

**Solution**:
```json
{
    "class_weights": [1.0, 10.0],
    "early_stopping_rounds": 15
}
```

### Issue 5: Model Format Compatibility

**Symptom**: Cannot load saved model

**Common Causes**:
1. Version mismatch between save and load
2. Binary format incompatibility

**Solution**:
- Use text format (`.txt`) for version stability
- Avoid binary format for production
- Document LightGBM version in artifacts

## References

### Related Scripts

- **Training Scripts:**
  - [`xgboost_training.py`](xgboost_training_script.md): XGBoost training (comparison)
  - [`pytorch_training.py`](pytorch_training_script.md): PyTorch training
  - [`lightgbmmt_training.py`](../../projects/pfw_lightgbmmt_legacy/docker/lightgbmmt_training.py): Multi-task LightGBM

- **Preprocessing Scripts:**
  - [`missing_value_imputation.py`](missing_value_imputation_script.md): Numerical imputation
  - [`risk_table_mapping.py`](risk_table_mapping_script.md): Categorical feature transformation
  - [`feature_selection.py`](feature_selection_script.md): Feature selection

- **Evaluation Scripts:**
  - [`lightgbm_model_eval.py`](lightgbm_model_eval_script.md): Model evaluation
  - [`lightgbm_model_inference.py`](lightgbm_model_inference_script.md): Inference only
  - [`model_metrics_computation.py`](model_metrics_computation_script.md): Comprehensive metrics

- **Deployment Scripts:**
  - [`package.py`](package_script.md): Model packaging
  - [`payload.py`](payload_script.md): Test payload generation

### Related Documentation

- **Contract**: [`src/cursus/steps/contracts/lightgbm_training_contract.py`](../../src/cursus/steps/contracts/lightgbm_training_contract.py) - Complete contract specification
- **Hyperparameters**: [`projects/ab_lightgbm/docker/hyperparams/hyperparameters_lightgbm.py`](../../projects/ab_lightgbm/docker/hyperparams/hyperparameters_lightgbm.py) - Hyperparameter definitions
- **Config**: LightGBM training configuration class in step configs
- **Builder**: LightGBM training step builder
- **Specification**: LightGBM training step specification in registry

### Related Design Documents

No specific design documents currently exist for this training script. General training step patterns are documented in the developer guide.

### External References

- **[LightGBM Documentation](https://lightgbm.readthedocs.io/)**: Official LightGBM documentation
- **[LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)**: Complete parameter reference
- **[LightGBM vs XGBoost](https://lightgbm.readthedocs.io/en/latest/Experiments.html)**: Performance comparison
- **[scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)**: Metrics computation reference
- **[Pandas Documentation](https://pandas.pydata.org/docs/)**: DataFrame operations
- **[AWS CodeArtifact](https://docs.aws.amazon.com/codeartifact/)**: Secure package repository documentation
