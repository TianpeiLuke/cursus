---
tags:
  - code
  - training_script
  - lightgbmmt
  - multi_task_learning
  - gradient_boosting
keywords:
  - LightGBMMT training
  - multi-task learning
  - adaptive task weighting
  - knowledge distillation
  - risk table mapping
  - numerical imputation
  - tabular classification
topics:
  - Multi-task learning
  - gradient boosting
  - adaptive weighting
  - knowledge distillation
  - tabular machine learning
language: python
date of note: 2025-11-18
---

# LightGBMMT Multi-Task Training Script

## Overview

The `lightgbmmt_training.py` script implements a comprehensive LightGBM multi-task (MT) training pipeline for multi-label tabular data classification with adaptive task weighting and optional knowledge distillation.

The script provides a production-ready multi-task learning workflow that supports both inline preprocessing computation and pre-computed artifact reuse from upstream steps. It handles multiple binary classification tasks simultaneously with shared tree structures, adaptive task weight computation based on similarity, optional knowledge distillation for struggling tasks, and comprehensive per-task evaluation with aggregate metrics.

Key capabilities include flexible preprocessing artifact control (compute inline vs reuse pre-computed), risk table mapping for categorical feature transformation, mean-based numerical imputation, multi-task LightGBM model training with custom Python loss functions, adaptive task weighting via JS divergence, knowledge distillation strategy, comprehensive per-task and aggregate evaluation, and packaged output artifacts including weight evolution tracking.

## Purpose and Major Tasks

### Primary Purpose
Train LightGBM multi-task gradient boosting models for multi-label tabular classification with adaptive task weighting, knowledge distillation, and shared representation learning across related tasks.

### Major Tasks

1. **Package Installation Management**: Install required packages from secure CodeArtifact or public PyPI
2. **Configuration Loading**: Load and validate multi-task hyperparameters with Pydantic schema
3. **Data Loading**: Load train/validation/test datasets with multi-label targets
4. **Preprocessing Artifact Control**: Detect and load pre-computed artifacts or compute inline
5. **Numerical Imputation**: Apply mean-based imputation for missing values
6. **Risk Table Mapping**: Fit and apply risk tables for categorical features
7. **Task Column Identification**: Identify and validate multi-label task columns
8. **Task Index Creation**: Create task-specific positive sample indices
9. **Loss Function Initialization**: Create multi-task loss function via LossFactory
10. **Model Training**: Train LightGBM with custom loss function and adaptive weighting
11. **Weight Evolution Tracking**: Track task weight changes over training iterations
12. **Comprehensive Evaluation**: Compute per-task and aggregate metrics with visualizations
13. **Artifact Packaging**: Save model, preprocessing artifacts, training state, and weight evolution

## Script Contract

### Entry Point
```
lightgbmmt_training.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `input_path` | `/opt/ml/input/data` | Root directory containing train/val/test subdirectories |
| `train` | `/opt/ml/input/data/train` | Multi-label training data files (.csv, .tsv, .parquet) |
| `val` | `/opt/ml/input/data/val` | Multi-label validation data files |
| `test` | `/opt/ml/input/data/test` | Multi-label test data files (optional) |
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
- `lightgbmmt_model.txt`: Trained multi-task LightGBM model in text format
- `risk_table_map.pkl`: Risk table mappings for categorical features
- `impute_dict.pkl`: Imputation values for numerical features
- `training_state.json`: Training state for checkpointing and resumption
- `feature_columns.txt`: Ordered feature column names with indices
- `hyperparameters.json`: Complete model hyperparameters including loss configuration
- `weight_evolution.json`: Task weight evolution over training iterations

**Evaluation Output Contents**:
- `val.tar.gz`: Validation predictions and metrics tarball
  - `val/predictions.{csv,tsv,parquet}`: Per-task validation predictions
  - `val/metrics.json`: Per-task and aggregate validation metrics
  - `val_metrics/val_task_i_taskname_roc.jpg`: Per-task ROC curves
  - `val_metrics/val_task_i_taskname_pr.jpg`: Per-task Precision-Recall curves
- `test.tar.gz`: Test predictions and metrics tarball (same structure, optional)

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
- When `USE_PRECOMPUTED_*=true`: Script loads artifacts from `model_artifacts_input` and skips transformation
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
| `cat_field_list` | `List[str]` | Yes | Categorical feature column names | `["marketplace", "category"]` |
| `label_name` | `str` | Yes | Main task label column name | `"isFraud"` |
| `task_label_names` | `List[str]` | Yes | All task label column names (ordered) | `["isFraud", "isCCfrd", "isDDfrd"]` |
| `id_name` | `str` | No | Column name for record IDs (default: "id") | `"order_id"` |

#### Multi-Task Configuration
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `num_tasks` | `int` | No | Number of tasks (auto-derived from task_label_names length) | Example: `3` |
| `loss_type` | `str` | Yes | Loss function type | `"fixed"`, `"adaptive"`, `"adaptive_kd"` |
| `loss_beta` | `float` | No | Fixed weight for subtasks | Default: `0.5`, Range: `0.0-1.0` |
| `loss_main_task_weight` | `float` | No | Weight for main task | Default: `0.5`, Range: `0.0-1.0` |

#### Adaptive Weighting Parameters
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `loss_weight_lr` | `float` | No | Learning rate for weight updates | Default: `0.1`, Range: `0.0-1.0` |
| `loss_weight_method` | `str` | No | Weight update method | `"standard"`, `"tenIters"`, `"sqrt"`, `"delta"` |
| `loss_weight_update_frequency` | `int` | No | Iterations between weight updates | Default: `10` |
| `loss_patience` | `int` | No | Patience for KD triggering | Default: `5` |

#### LightGBM Hyperparameters
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `num_leaves` | `int` | No | Maximum number of leaves per tree | Default: `31` |
| `learning_rate` | `float` | No | Learning rate (shrinkage rate) | Default: `0.1`, Range: `0.0-1.0` |
| `num_iterations` | `int` | No | Number of boosting iterations | Default: `100`, Typical: `50-500` |
| `max_depth` | `int` | No | Maximum tree depth | Default: `-1` (no limit) |
| `feature_fraction` | `float` | No | Subsample ratio of features per tree | Default: `1.0`, Range: `0.0-1.0` |
| `bagging_fraction` | `float` | No | Subsample ratio of training instances | Default: `1.0`, Range: `0.0-1.0` |

#### Performance Optimization
| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `loss_cache_predictions` | `bool` | No | Cache predictions for performance | `true` |
| `loss_precompute_indices` | `bool` | No | Pre-compute task indices | `true` |

## Input Data Structure

### Expected Input Format

```
/opt/ml/input/data/
├── train/
│   ├── multi_label_training_data.csv (or .tsv, .parquet)
│   └── _SUCCESS (optional marker)
├── val/
│   ├── multi_label_validation_data.csv
│   └── _SUCCESS
├── test/ (optional)
│   ├── multi_label_test_data.csv
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
- Task label columns (names specified by `task_label_names`): Binary labels (0/1) for each task
- ID column (name specified by `id_name`, optional): Unique record identifier

**Feature Columns**:
- Tabular features (specified in `tab_field_list`): Numerical features
- Categorical features (specified in `cat_field_list`): Categorical features for risk table mapping

### Multi-Label Data Format

**Task Label Structure**:
```
Each row can have multiple positive labels (multi-label, not mutually exclusive):
- isFraud=1, isCCfrd=1, isDDfrd=0  # Fraud via credit card
- isFraud=1, isCCfrd=0, isDDfrd=1  # Fraud via direct debit
- isFraud=0, isCCfrd=0, isDDfrd=0  # Not fraud
```

**Task Relationships**:
- Main task (e.g., `isFraud`): Overall fraud indicator
- Subtasks (e.g., `isCCfrd`, `isDDfrd`): Specific fraud types
- Hierarchical: Subtask positive → Main task typically positive
- Multi-label: Sample can belong to multiple subtasks

### Supported Data Formats

1. **CSV Files**: Comma-separated values
2. **TSV Files**: Tab-separated values
3. **Parquet Files**: Columnar format for efficient storage

**Format Preservation**:
- Script automatically detects input format from training data
- All outputs (predictions, metrics) use the same format as input

### Data Requirements

**Numerical Features** (`tab_field_list`):
- May contain missing values (will be imputed)
- Should be numeric type or convertible to float

**Categorical Features** (`cat_field_list`):
- Can be string or numeric types
- Will be transformed to risk scores via risk table mapping

**Task Labels** (`task_label_names`):
- Must be binary (0 or 1)
- Multiple tasks can be positive simultaneously (multi-label)
- At least one task must have positive samples in training data

## Output Data Structure

### Model Output Directory Structure

```
/opt/ml/model/
├── lightgbmmt_model.txt           # Multi-task LightGBM model (text format)
├── risk_table_map.pkl             # Risk tables dictionary
├── impute_dict.pkl                # Imputation values dictionary
├── training_state.json            # Training state for checkpointing
├── feature_columns.txt            # Ordered feature column names
├── hyperparameters.json           # Model hyperparameters + loss config
└── weight_evolution.json          # Task weight evolution over training
```

**lightgbmmt_model.txt**:
- Text format multi-task LightGBM model
- Contains shared tree structures for all tasks
- Can be loaded with custom LightGBMMT model class

**training_state.json**:
```json
{
    "current_iteration": 100,
    "best_iteration": 85,
    "best_score": 0.8542,
    "weight_evolution": [[0.5, 0.5, 0.5], ...],
    "task_performances": {...}
}
```

**weight_evolution.json**:
```json
[
    [0.5, 0.5, 0.5],      # Initial weights (iteration 0)
    [0.6, 0.4, 0.4],      # After 10 iterations
    [0.65, 0.35, 0.38],   # After 20 iterations
    ...
]
```
Note: Tracks how task weights adapt during training based on task similarity

### Evaluation Output Directory Structure

```
/opt/ml/output/data/
├── val.tar.gz                                    # Validation results tarball
│   ├── val/
│   │   ├── predictions.csv                       # Multi-task predictions
│   │   └── metrics.json                          # Per-task + aggregate metrics
│   └── val_metrics/
│       ├── val_task_0_isFraud_roc.jpg           # ROC curve for task 0
│       ├── val_task_0_isFraud_pr.jpg            # PR curve for task 0
│       ├── val_task_1_isCCfrd_roc.jpg           # ROC curve for task 1
│       └── val_task_1_isCCfrd_pr.jpg            # PR curve for task 1
└── test.tar.gz                                   # Test results (same structure)
```

**predictions.{csv,tsv,parquet} Contents**:
| Column | Description |
|--------|-------------|
| `id_name` | Record identifier |
| `isFraud_true` | True label for main task |
| `isFraud_prob` | Predicted probability for main task |
| `isCCfrd_true` | True label for subtask 1 |
| `isCCfrd_prob` | Predicted probability for subtask 1 |
| ... | Additional task columns |

**metrics.json Contents**:
```json
{
    "task_0_isFraud": {
        "auc_roc": 0.8542,
        "average_precision": 0.8012,
        "f1_score": 0.7823
    },
    "task_1_isCCfrd": {
        "auc_roc": 0.8201,
        "average_precision": 0.7856,
        "f1_score": 0.7654
    },
    "aggregate": {
        "mean_auc_roc": 0.8372,
        "median_auc_roc": 0.8372,
        "mean_average_precision": 0.7934,
        "median_average_precision": 0.7934,
        "mean_f1_score": 0.7739,
        "median_f1_score": 0.7739
    }
}
```

## Key Functions and Tasks

### Task Column Identification

#### `identify_task_columns(df, hyperparams)`
**Purpose**: Identify task label columns with priority-based detection

**Algorithm**:
```python
1. Priority 1: Use explicit task_label_names from hyperparameters
   a. If hyperparams.task_label_names provided:
      - Use specified column names
      - Validate all columns exist in dataframe
      - Return task columns
2. Priority 2: Auto-detection (backward compatibility fallback)
   a. Strategy 1: Look for columns starting with 'task_'
   b. Strategy 2: Look for common fraud patterns
      - isFraud, isCCfrd, isDDfrd, isGCfrd, etc.
   c. If no columns found: raise ValueError
3. Validate against num_tasks if provided
4. Log detection method and column names
5. Return task_columns list
```

**Parameters**:
- `df` (pd.DataFrame): Input dataframe to search
- `hyperparams` (LightGBMMtModelHyperparameters): Configuration object

**Returns**: `List[str]` - Ordered list of task label column names

**Priority Logic**:
- Explicit configuration preferred over auto-detection
- Clear error messages if detection fails
- Backward compatibility maintained

#### `create_task_indices(train_df, val_df, task_columns)`
**Purpose**: Create task-specific positive sample indices

**Algorithm**:
```python
1. Initialize empty dictionaries for train and validation indices
2. For each task index i and task column:
   a. trn_sublabel_idx[i] = np.where(train_df[task_col] == 1)[0]
   b. val_sublabel_idx[i] = np.where(val_df[task_col] == 1)[0]
3. Log positive sample counts per task
4. Return train and validation index dictionaries
```

**Parameters**:
- `train_df`, `val_df` (pd.DataFrame): Training and validation data
- `task_columns` (List[str]): Task label column names

**Returns**: `Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]` - (train_indices, val_indices)

**Output Structure**:
```python
{
    0: np.array([0, 5, 12, ...]),  # Indices where task 0 is positive
    1: np.array([1, 7, 15, ...]),  # Indices where task 1 is positive
    2: np.array([3, 8, 20, ...]),  # Indices where task 2 is positive
}
```

### Multi-Task Prediction and Evaluation

#### `predict_multitask(model, df, feature_columns)`
**Purpose**: Generate multi-task predictions

**Algorithm**:
```python
1. Extract feature matrix X from df using feature_columns
2. Call model.predict(X) to get predictions
3. Return predictions as numpy array of shape (n_samples, n_tasks)
```

**Parameters**:
- `model`: Trained LightGBMMT model
- `df` (pd.DataFrame): Input data
- `feature_columns` (List[str]): Feature column names

**Returns**: `np.ndarray` - Shape (n_samples, n_tasks) with probabilities

#### `compute_multitask_metrics(y_true_tasks, y_pred_tasks, task_columns)`
**Purpose**: Compute per-task and aggregate metrics

**Algorithm**:
```python
1. Initialize metrics dictionary and aggregate lists
2. For each task i:
   a. Extract y_true[i] and y_pred[:, i]
   b. Compute AUC-ROC, Average Precision, F1-Score
   c. Store in metrics["task_i_taskname"]
   d. Append to aggregate lists
   e. Handle ValueError (single class) gracefully
3. Compute aggregate statistics:
   a. Mean and median of AUC-ROC
   b. Mean and median of Average Precision
   c. Mean and median of F1-Score
4. Return comprehensive metrics dictionary
```

**Parameters**:
- `y_true_tasks` (Dict[int, np.ndarray]): True labels per task
- `y_pred_tasks` (np.ndarray): Predicted probabilities (n_samples, n_tasks)
- `task_columns` (List[str]): Task names

**Returns**: `Dict[str, Any]` - Per-task and aggregate metrics

**Error Handling**:
- Handles tasks with single class (returns default scores)
- Logs warnings for problematic tasks
- Continues with remaining tasks

#### `plot_multitask_curves(y_true_tasks, y_pred_tasks, task_columns, out_dir, prefix)`
**Purpose**: Generate ROC and PR curves for each task

**Algorithm**:
```python
1. Create output directory
2. For each task i:
   a. Extract y_true and y_pred for task
   b. Check if at least 2 classes present
   c. Generate ROC curve:
      - Compute FPR, TPR, AUC
      - Plot curve with diagonal reference
      - Save as task_i_taskname_roc.jpg
   d. Generate PR curve:
      - Compute precision, recall, AP
      - Plot curve
      - Save as task_i_taskname_pr.jpg
   e. Handle exceptions gracefully
3. Close all figures
```

**Parameters**:
- `y_true_tasks` (Dict[int, np.ndarray]): True labels per task
- `y_pred_tasks` (np.ndarray): Predicted probabilities
- `task_columns` (List[str]): Task names
- `out_dir` (str): Output directory
- `prefix` (str): Filename prefix

**Returns**: None (saves plots to disk)

### Model Training and Artifact Saving

#### `save_artifacts(model, risk_tables, impute_dict, model_path, feature_columns, hyperparams, training_state)`
**Purpose**: Save all model and training artifacts

**Algorithm**:
```python
1. Create model_path directory
2. Save LightGBM model:
   a. model.save(lightgbmmt_model.txt)
3. Save preprocessing artifacts:
   a. risk_table_map.pkl
   b. impute_dict.pkl
4. Save training state:
   a. training_state.json (for checkpointing)
5. Save feature columns with ordering
6. Save hyperparameters as JSON
7. Save weight evolution:
   a. Convert numpy arrays to lists
   b. Save as weight_evolution.json
8. Log all save operations
```

**Parameters**:
- `model`: Trained LightGBMMT model
- `risk_tables` (dict): Risk table mappings
- `impute_dict` (dict): Imputation values
- `model_path` (str): Output directory
- `feature_columns` (List[str]): Feature column names
- `hyperparams`: Hyperparameters object
- `training_state` (TrainingState): Training state object

**Returns**: None (saves files to disk)

**Multi-Task Specific Artifacts**:
- `training_state.json`: Checkpoint for resuming training
- `weight_evolution.json`: Task weight history

## Algorithms and Data Structures

### Multi-Task Loss Functions

The script uses three loss function types via LossFactory:

#### 1. Fixed Loss
**Strategy**: Static weight allocation

**Algorithm**:
```python
weights = [main_task_weight, beta, beta, ..., beta]
# Example: [0.5, 0.5, 0.5] for 3 tasks

loss = sum(weight[i] * binary_logloss(y_true[i], y_pred[:, i]) for i in tasks)
```

**Use Case**: Baseline multi-task learning without adaptation

#### 2. Adaptive Loss
**Strategy**: Dynamic weight adjustment based on task similarity

**Algorithm**:
```python
# Every K iterations:
1. Compute task similarity via JS divergence:
   for i, j in task_pairs:
       similarity[i][j] = 1 - js_divergence(pred[i], pred[j])

2. Update weights based on similarity:
   # Weight update strategies:
   - standard: w[i] += lr * (1 - similarity[i][main])
   - tenIters: Apply update every 10 iterations
   - sqrt: Use sqrt of similarity difference
   - delta: Delta-based adjustment

3. Normalize weights to sum to 1.0

4. Compute weighted loss
```

**Complexity**: O(n * t²) where n=samples, t=tasks

**Key Features**:
- Adapts to task relationships during training
- Reduces weight for highly similar tasks (avoid redundancy)
- Increases weight for dissimilar tasks (learn unique patterns)

#### 3. Adaptive KD (Knowledge Distillation) Loss
**Strategy**: Adaptive weights + KD for struggling tasks

**Algorithm**:
```python
1. Perform adaptive weight updates (as in Adaptive Loss)

2. Identify struggling tasks:
   if task_performance not improving for patience iterations:
       mark task as "struggling"

3. Apply knowledge distillation:
   for struggling_task in struggling_tasks:
       # Soft labels from main task
       soft_labels = main_task_predictions
       
       # KD loss (MSE between predictions)
       kd_loss = mse(struggling_task_pred, soft_labels)
       
       # Combined loss
       total_loss += kd_weight * kd_loss

4. Compute final weighted loss
```

**Complexity**: O(n * t²) + O(n * s) where s=struggling tasks

**Benefits**:
- Stabilizes training for weak tasks
- Transfers knowledge from strong (main) task
- Prevents catastrophic forgetting

### Task Similarity Computation

**Problem**: Measure similarity between task predictions to adjust weights

**Solution**: Jensen-Shannon (JS) Divergence

**Algorithm**:
```python
def js_divergence(p, q):
    """
    Compute JS divergence between two probability distributions.
    
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    """
    # Add small epsilon to avoid log(0)
    p = p + 1e-10
    q = q + 1e-10
    
    # Compute midpoint distribution
    m = 0.5 * (p + q)
    
    # Compute KL divergences
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    
    # JS divergence
    js = 0.5 * kl_pm + 0.5 * kl_qm
    
    # Convert to similarity (0 = identical, 1 = completely different)
    similarity = 1 - js
    
    return similarity
```

**Properties**:
- Symmetric: JS(P||Q) = JS(Q||P)
- Bounded: 0 ≤ JS ≤ 1
- Square root is a metric

### Weight Update Strategies

The script supports multiple weight update methods:

#### Strategy 1: Standard
```python
w[i] += lr * (1 - similarity[i][main_task])
```
- Direct similarity-based adjustment
- Higher weight for dissimilar tasks

#### Strategy 2: TenIters
```python
if iteration % 10 == 0:
    w[i] += lr * (1 - similarity[i][main_task])
```
- Updates every 10 iterations
- Reduces update frequency
- More stable training

#### Strategy 3: Sqrt
```python
w[i] += lr * sqrt(1 - similarity[i][main_task])
```
- Square root dampens large adjustments
- Smoother weight evolution

#### Strategy 4: Delta
```python
delta = current_similarity - previous_similarity
w[i] += lr * delta
```
- Change-based adjustment
- Reacts to similarity trends

### Training State Management

**Problem**: Support checkpointing and resumption of multi-task training

**Solution**: TrainingState object with serialization

**Data Structure**:
```python
class TrainingState:
    current_iteration: int
    best_iteration: int
    best_score: float
    weight_evolution: List[np.ndarray]  # Weight history
    task_performances: Dict[int, List[float]]  # Per-task metrics
    patience_counters: Dict[int, int]  # For KD triggering
    
    def to_checkpoint_dict(self) -> dict:
        """Serialize to JSON-compatible dictionary"""
        
    @classmethod
    def from_checkpoint_dict(cls, data: dict):
        """Deserialize from dictionary"""
```

**Benefits**:
- Enables training resumption
- Tracks weight evolution for analysis
- Supports debugging and monitoring

## Performance Characteristics

### Training Performance

| Dataset Size | Tasks | Training Time | Memory Usage | Notes |
|--------------|-------|---------------|--------------|-------|
| 10K samples | 3 | ~45s | ~600MB | Overhead from multi-task loss |
| 100K samples | 3 | ~4min | ~2.5GB | Adaptive weighting adds 20% overhead |
| 1M samples | 3 | ~35min | ~12GB | Benefit from shared representations |
| 10M samples | 5 | ~6hrs | ~60GB | Scales linearly with task count |

**Multi-Task Overhead**:
- Fixed loss: +10% vs single-task
- Adaptive loss: +20-30% vs single-task (similarity computation)
- Adaptive KD: +30-40% vs single-task (additional KD loss)

**Performance Optimization**:
- `loss_cache_predictions=true`: 30-50% speedup
- `loss_precompute_indices=true`: Faster index operations
- Shared tree structures: Better generalization, no redundant trees

### Preprocessing Performance

Same as LightGBM single-task (see lightgbm_training_script.md)

## Error Handling

### Task Column Errors

**ValueError: Task columns not found**
```python
ValueError: Could not auto-detect task columns. Expected columns starting with 'task_' or common fraud patterns. 
Please specify 'task_label_names' explicitly in hyperparameters.
```
**Cause**: Auto-detection failed and no explicit task_label_names provided

**Resolution**: Add `task_label_names` to hyperparameters.json

### Training Errors

**ValueError: Task has single class**
```python
ValueError: Only one class present in task 2 (isDDfrd)
```
**Cause**: Task has no positive or no negative samples in data split

**Handling**: Logged as warning, task gets default scores

**Resolution**: Check data distribution, may need to merge rare tasks

### Configuration Errors

**ValidationError: Missing multi-task parameters**
```python
ValueError: Missing required key in hyperparameters: task_label_names
```
**Cause**: Multi-task specific configuration missing

**Resolution**: Add required multi-task parameters to hyperparameters.json

## Best Practices

### For Production Deployments

1. **Explicit Task Configuration**
   ```json
   {
       "task_label_names": ["isFraud", "isCCfrd", "isDDfrd"],
       "loss_type": "adaptive",
       "num_tasks": 3
   }
   ```
   - Always specify task_label_names explicitly
   - Don't rely on auto-detection in production
   - Validate task count matches data

2. **Use Adaptive Loss with Patience**
   ```json
   {
       "loss_type": "adaptive",
       "loss_weight_lr": 0.1,
       "loss_weight_update_frequency": 10,
       "loss_weight_method": "standard"
   }
   ```
   - Adapts to task relationships during training
   - Better than fixed weights for related tasks
   - Monitor weight evolution for insights

3. **Enable Knowledge Distillation for Weak Tasks**
   ```json
   {
       "loss_type": "adaptive_kd",
       "loss_patience": 5,
       "loss_weight_lr": 0.1
   }
   ```
   - Helps struggling tasks learn from main task
   - Stabilizes training
   - Prevents catastrophic forgetting

4. **Performance Optimization**
   ```json
   {
       "loss_cache_predictions": true,
       "loss_precompute_indices": true
   }
   ```
   - 30-50% speedup with caching
   - Pre-computed indices reduce overhead
   - Essential for large datasets

5. **Use Pre-Computed Artifacts**
   ```bash
   export USE_PRECOMPUTED_IMPUTATION=true
   export USE_PRECOMPUTED_RISK_TABLES=true
   ```
   - Separates preprocessing from training
   - Enables independent scaling
   - Faster training iterations

### For Development

1. **Start with Fixed Loss**
   ```json
   {
       "loss_type": "fixed",
       "loss_beta": 0.5,
       "loss_main_task_weight": 0.5
   }
   ```
   - Establish baseline performance
   - Simpler to debug
   - Understand task difficulties

2. **Monitor Task Performance**
   - Check per-task metrics regularly
   - Identify struggling tasks early
   - Adjust task weights if needed

3. **Track Weight Evolution**
   - Analyze weight_evolution.json after training
   - Understand task relationships
   - Validate adaptive weighting behavior

4. **Test with Smaller Task Sets**
   - Start with 2-3 tasks
   - Add tasks incrementally
   - Validate performance doesn't degrade

### For Multi-Task Design

1. **Task Hierarchy**
   ```
   Main Task (isFraud)
       ├── Subtask 1 (isCCfrd)
       ├── Subtask 2 (isDDfrd)
       └── Subtask 3 (isGCfrd)
   ```
   - Define clear task hierarchy
   - Main task should be most general
   - Subtasks should be specific patterns

2. **Task Balance**
   - Ensure all tasks have sufficient positive samples
   - Aim for at least 100 positive samples per task
   - Consider merging rare tasks

3. **Task Similarity**
   - Group related tasks together
   - Separate unrelated tasks into different models
   - Monitor JS divergence between tasks

## Example Configurations

### Example 1: Fraud Detection with 3 Tasks
```json
{
    "tab_field_list": ["price", "quantity", "seller_rating"],
    "cat_field_list": ["marketplace", "category", "payment_method"],
    "label_name": "isFraud",
    "task_label_names": ["isFraud", "isCCfrd", "isDDfrd"],
    "id_name": "transaction_id",
    "num_tasks": 3,
    "loss_type": "adaptive",
    "loss_beta": 0.5,
    "loss_main_task_weight": 0.5,
    "loss_weight_lr": 0.1,
    "loss_weight_method": "standard",
    "loss_weight_update_frequency": 10,
    "learning_rate": 0.1,
    "num_iterations": 200,
    "num_leaves": 31,
    "max_depth": -1
}
```
**Use Case**: Payment fraud with payment-method specific subtasks

### Example 2: Multi-Task with Knowledge Distillation
```json
{
    "tab_field_list": ["amount", "frequency", "recency"],
    "cat_field_list": ["country", "merchant_category"],
    "label_name": "isFraud",
    "task_label_names": ["isFraud", "isATO", "isPhishing", "isScam"],
    "num_tasks": 4,
    "loss_type": "adaptive_kd",
    "loss_patience": 5,
    "loss_weight_lr": 0.05,
    "loss_weight_method": "tenIters",
    "learning_rate": 0.05,
    "num_iterations": 300
}
```
**Use Case**: Multiple fraud types with KD for weak tasks

### Example 3: Fixed Weights for Unrelated Tasks
```json
{
    "tab_field_list": ["feature1", "feature2", "feature3"],
    "cat_field_list": ["category1", "category2"],
    "label_name": "task_1",
    "task_label_names": ["task_1", "task_2"],
    "num_tasks": 2,
    "loss_type": "fixed",
    "loss_beta": 0.3,
    "loss_main_task_weight": 0.7,
    "learning_rate": 0.1,
    "num_iterations": 150
}
```
**Use Case**: Two independent tasks trained together for efficiency

## Integration Patterns

### Upstream Integration (Preprocessing)

```
MissingValueImputation
   ↓ (outputs: imputed train/val/test + impute_dict.pkl)
RiskTableMapping
   ↓ (outputs: transformed train/val/test + risk_table_map.pkl)
FeatureSelection (optional)
   ↓ (outputs: filtered train/val/test + selected_features.json)
LightGBMMTTraining (USE_PRECOMPUTED_*=true)
   ↓ (outputs: multi-task model + evaluation)
```

**Artifact Flow**:
1. Each preprocessing step outputs artifacts
2. Training loads artifacts from model_artifacts_input
3. Training validates data state matches artifacts
4. All artifacts packaged into final multi-task model

### Downstream Integration (Inference)

```
LightGBMMTTraining
   ↓ (outputs: lightgbmmt_model.txt + preprocessing artifacts + training state)
LightGBMMTModelInference
   ↓ (uses: model + artifacts for multi-task transformation)
ModelMetricsComputation (per-task)
   ↓ (computes: per-task detailed metrics)
```

**Deployment Flow**:
1. Training produces complete multi-task model package
2. Inference applies same preprocessing and generates N task predictions
3. Metrics computation validates per-task and aggregate performance

### Complete Multi-Task Pipeline Example

```
1. DummyDataLoading/CradleDataLoading
   ↓ (multi-label data)
2. TabularPreprocessing
   ↓ (cleaned data with all task labels, train/val/test splits)
3. MissingValueImputation
   ↓ (imputed data + impute_dict.pkl)
4. RiskTableMapping
   ↓ (transformed data + risk_table_map.pkl)
5. LightGBMMTTraining
   ↓ (multi-task model + training state + weight evolution)
6. ModelMetricsComputation (per-task)
   ↓ (per-task comprehensive metrics)
7. Package
   ↓ (deployment package with multi-task support)
```

## Troubleshooting

### Issue 1: Unbalanced Task Weights

**Symptom**: One task dominates, others perform poorly

**Common Causes**:
1. Fixed weights not appropriate for task difficulties
2. Adaptive learning rate too high
3. Task similarities not captured

**Solution**:
```json
{
    "loss_type": "adaptive",
    "loss_weight_lr": 0.05,
    "loss_weight_method": "tenIters"
}
```
Also: Analyze weight_evolution.json to understand dynamics

### Issue 2: Poor Subtask Performance

**Symptom**: Main task good, subtasks poor

**Common Causes**:
1. Insufficient subtask samples
2. Tasks not related enough for knowledge transfer
3. Need knowledge distillation

**Solution**:
```json
{
    "loss_type": "adaptive_kd",
    "loss_patience": 5
}
```
Also: Check per-task positive sample counts

### Issue 3: Slow Multi-Task Training

**Symptom**: Training much slower than single-task

**Common Causes**:
1. Prediction caching disabled
2. Too many tasks
3. Frequent weight updates

**Solution**:
```json
{
    "loss_cache_predictions": true,
    "loss_precompute_indices": true,
    "loss_weight_update_frequency": 20
}
```
Also: Consider reducing task count or splitting into multiple models

### Issue 4: Task Auto-Detection Fails

**Symptom**: Script can't find task columns

**Common Causes**:
1. Inconsistent column naming
2. No task_label_names specified
3. Columns don't match patterns

**Solution**:
```json
{
    "task_label_names": ["task_1", "task_2", "task_3"]
}
```
Always specify explicitly in production

### Issue 5: Weight Evolution Unstable

**Symptom**: Task weights oscillating wildly

**Common Causes**:
1. Learning rate too high
2. Update frequency too low
3. Tasks too dissimilar

**Solution**:
```json
{
    "loss_weight_lr": 0.01,
    "loss_weight_method": "sqrt",
    "loss_weight_update_frequency": 20
}
```
Also: Consider using fixed weights if tasks truly unrelated

## References

### Related Scripts

- **Training Scripts:**
  - [`xgboost_training.py`](xgboost_training_script.md): Single-task XGBoost training
  - [`lightgbm_training.py`](lightgbm_training_script.md): Single-task LightGBM training
  - [`pytorch_training.py`](pytorch_training_script.md): PyTorch training

- **Preprocessing Scripts:**
  - [`missing_value_imputation.py`](missing_value_imputation_script.md): Numerical imputation
  - [`risk_table_mapping.py`](risk_table_mapping_script.md): Categorical feature transformation
  - [`feature_selection.py`](feature_selection_script.md): Feature selection

- **Evaluation Scripts:**
  - [`lightgbmmt_model_inference.py`](../../projects/pfw_lightgbmmt_legacy/docker/lightgbmmt_model_inference.py): Multi-task inference
  - [`model_metrics_computation.py`](model_metrics_computation_script.md): Comprehensive metrics

- **Deployment Scripts:**
  - [`package.py`](package_script.md): Model packaging
  - [`payload.py`](payload_script.md): Test payload generation

### Related Documentation

- **Contract**: [`src/cursus/steps/contracts/lightgbmmt_training_contract.py`](../../src/cursus/steps/contracts/lightgbmmt_training_contract.py) - Complete contract specification
- **Hyperparameters**: [`projects/pfw_lightgbmmt_legacy/docker/hyperparams/hyperparameters_lightgbmmt.py`](../../projects/pfw_lightgbmmt_legacy/docker/hyperparams/hyperparameters_lightgbmmt.py) - Hyperparameter definitions
- **Loss Functions**: LossFactory and loss implementations in `projects/pfw_lightgbmmt_legacy/docker/models/loss/`
- **Model Factory**: ModelFactory for multi-task model creation
- **Training State**: TrainingState for checkpointing

### Related Design Documents

**Multi-Task Learning Architecture:**
- **[LightGBM Multi-Task Training Step Design](../1_design/lightgbm_multi_task_training_step_design.md)**: Complete step design for multi-task LightGBM training with adaptive weighting and knowledge distillation
- **[MTGBM Multi-Task Learning Design](../1_design/mtgbm_multi_task_learning_design.md)**: Multi-task learning design patterns, loss functions, and weight adaptation strategies
- **[MTGBM Model Classes Refactoring Design](../1_design/mtgbm_model_classes_refactoring_design.md)**: Refactored model class architecture with template method and factory patterns
- **[MTGBM Models Refactoring Design](../1_design/mtgbm_models_refactoring_design.md)**: Complete refactoring design for multi-task gradient boosting models (67% code reduction, 91% quality score)
- **[LightGBMMT Model Inference Design](../1_design/lightgbmmt_model_inference_design.md)**: Multi-task model inference design with per-task predictions

**Implementation Analysis:**
- **[LightGBMMT Multi-Task Implementation Analysis](../4_analysis/2025-11-10_lightgbmmt_multi_task_implementation_analysis.md)**: Comprehensive analysis of multi-task implementation patterns and integration
- **[MTGBM Models Optimization Analysis](../4_analysis/2025-11-11_mtgbm_models_optimization_analysis.md)**: Performance optimization analysis for multi-task models with caching and pre-computation
- **[MTGBM Pipeline Reusability Analysis](../4_analysis/2025-11-11_mtgbm_pipeline_reusability_analysis.md)**: Analysis of pipeline reusability patterns and multi-task workflow integration
- **[Python Refactored vs LightGBMMT Fork Comparison](../4_analysis/2025-11-12_python_refactored_vs_lightgbmmt_fork_comparison.md)**: Comparison analysis of refactored Python implementation vs original fork
- **[MTGBM Missing Features Recovery](../4_analysis/2025-11-13_mtgbm_missing_features_recovery.md)**: Analysis of feature recovery and migration from legacy implementations

### External References

- **[LightGBM Documentation](https://lightgbm.readthedocs.io/)**: Official LightGBM documentation
- **[LightGBM Custom Loss](https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api)**: Custom objective functions
- **[Multi-Task Learning Survey](https://arxiv.org/abs/1706.05098)**: Overview of multi-task learning
- **[Knowledge Distillation](https://arxiv.org/abs/1503.02531)**: Distilling knowledge in neural networks
- **[Jensen-Shannon Divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)**: Similarity measurement
- **[scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)**: Metrics computation reference

## Architecture Highlights

### Refactored Design Benefits

**Code Quality Improvements**:
- 67% code reduction in loss functions (360 → 120 lines)
- Quality score: 53% → 91% (Poor → Excellent)
- Comprehensive Pydantic v2 validation
- Template method pattern for training workflow

**Performance Optimizations**:
- Prediction caching: 30-50% speedup
- Pre-computed indices: Reduced overhead
- Efficient similarity computation

**Design Patterns**:
- Factory pattern: LossFactory, ModelFactory
- Strategy pattern: Weight update methods
- Template method: Training workflow
- State pattern: TrainingState for checkpointing

### Multi-Task Architecture

**Key Components**:
1. **LossFactory**: Creates appropriate loss function (Fixed/Adaptive/Adaptive_KD)
2. **ModelFactory**: Creates multi-task LightGBM model
3. **TrainingState**: Manages training state and checkpointing
4. **Weight Evolution**: Tracks adaptive weight changes

**Workflow**:
```
1. Identify tasks → Create indices
2. Initialize loss function via factory
3. Create model via factory
4. Train with adaptive weighting
5. Track weight evolution
6. Evaluate per-task performance
7. Save comprehensive artifacts
```
