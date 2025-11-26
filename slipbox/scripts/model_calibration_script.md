---
tags:
  - code
  - processing_script
  - model_calibration
  - probability_calibration
keywords:
  - model calibration
  - probability calibration
  - GAM calibration
  - isotonic regression
  - Platt scaling
  - expected calibration error
  - reliability diagram
  - binary classification
  - multi-class calibration
topics:
  - model calibration
  - prediction reliability
  - risk-based decision making
  - probability estimation
language: python
date of note: 2025-11-18
---

# Model Calibration Script Documentation

## Overview

The `model_calibration.py` script calibrates model prediction scores to accurate probabilities, transforming raw model outputs into reliable probability estimates essential for risk-based decision-making and threshold setting. The script implements three calibration methods (GAM, Isotonic Regression, Platt Scaling) and supports binary classification, multi-class classification, and multi-task/multi-label scenarios where multiple independent binary classifiers are calibrated with a shared calibration approach.

Well-calibrated probabilities ensure that predicted probabilities match true frequencies - for example, among all predictions with 70% confidence, approximately 70% should be correct. This is crucial for applications requiring trustworthy confidence scores, such as fraud detection, credit risk assessment, and medical diagnosis systems where decisions are based on probability thresholds.

The script provides a complete calibration workflow including data loading with format preservation, multiple calibration method options with configurable constraints, comprehensive evaluation metrics (ECE, MCE, Brier score), visualization generation with reliability diagrams, and support for complex deployment scenarios including nested tarball extraction from SageMaker training outputs.

## Purpose and Major Tasks

### Primary Purpose
Transform raw model prediction scores into well-calibrated probabilities that accurately reflect true event likelihoods, enabling reliable risk-based decision-making and threshold setting in production ML systems.

### Major Tasks

1. **Package Installation Management**: Install required dependencies from either public PyPI or secure CodeArtifact based on environment configuration with automatic credential handling

2. **Data Loading with Format Preservation**: Load evaluation data from multiple sources (direct files, nested tarballs from SageMaker) while detecting and preserving original format (CSV/TSV/Parquet)

3. **Binary Calibration**: Train and apply calibration models for binary classification using GAM, Isotonic Regression, or Platt Scaling with optional monotonicity constraints

4. **Multi-Class Calibration**: Train independent per-class calibration models in one-vs-rest fashion and normalize calibrated probabilities to ensure valid probability distributions

5. **Calibration Metrics Computation**: Calculate comprehensive metrics including Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier score, and AUC to quantify calibration quality

6. **Reliability Diagram Generation**: Create visual reliability diagrams comparing uncalibrated vs calibrated probabilities with histogram distributions for model assessment

7. **Model Persistence**: Save trained calibration models as pickle files for deployment alongside trained models for inference-time calibration

8. **Calibrated Data Output**: Generate complete datasets with both original and calibrated probability columns in original format for downstream processing

## Script Contract

### Entry Point
```
model_calibration.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `evaluation_data` | `/opt/ml/processing/input/eval_data` | Evaluation dataset with ground truth labels and model predictions (supports CSV/TSV/Parquet/JSON formats, direct files or nested tarballs) |

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `calibration_output` | `/opt/ml/processing/output/calibration` | Trained calibration models (.pkl files) and summary JSON |
| `metrics_output` | `/opt/ml/processing/output/metrics` | Calibration quality metrics (JSON) and reliability diagrams (PNG) |
| `calibrated_data` | `/opt/ml/processing/output/calibrated_data` | Dataset with calibrated probability columns in original format |

### Required Environment Variables

| Variable | Description | Valid Values |
|----------|-------------|--------------|
| `CALIBRATION_METHOD` | Calibration method to use | `"gam"`, `"isotonic"`, `"platt"` |
| `LABEL_FIELD` | Name of ground truth label column | Any string (e.g., `"label"`, `"ground_truth"`) |
| `IS_BINARY` | Whether this is binary classification | `"True"` or `"False"` (case-insensitive) |

**Note**: At least one of `SCORE_FIELD` or `SCORE_FIELDS` must be provided:
- **Single-Task Mode**: Use `SCORE_FIELD` for calibrating a single binary classifier
- **Multi-Task Mode**: Use `SCORE_FIELDS` for calibrating multiple independent binary classifiers with shared calibration method

### Optional Environment Variables

#### PyPI Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SECURE_PYPI` | `"false"` | Use secure CodeArtifact PyPI (`"true"`) or public PyPI (`"false"`) |

#### Single-Task / Multi-Task Mode Selection

| Variable | Default | Description | Notes |
|----------|---------|-------------|-------|
| `SCORE_FIELD` | `None` | Name of single score column | For single-task binary classification |
| `SCORE_FIELDS` | `None` | Comma-separated list of score columns | For multi-task binary classification (takes precedence) |

**Mode Determination**:
- If `SCORE_FIELDS` is provided: Multi-task mode (calibrates each score field independently)
- Else if `SCORE_FIELD` is provided: Single-task mode (calibrates one score field)
- If neither provided: Error (at least one required)

**Multi-Task Example**:
```bash
export SCORE_FIELDS="task_0_prob,task_1_prob,task_2_prob"
# Results in 3 independent calibrators, one per task
```

#### Binary Calibration Parameters

| Variable | Default | Description | Notes |
|----------|---------|-------------|-------|
| `MONOTONIC_CONSTRAINT` | `"True"` | Enforce monotonicity in GAM calibration | Only applies to GAM method |
| `GAM_SPLINES` | `"10"` | Number of splines for GAM | Only applies to GAM method, range: 5-25 |
| `ERROR_THRESHOLD` | `"0.05"` | Acceptable calibration error threshold | Used for warnings, range: 0.0-1.0 |
| `CALIBRATION_SAMPLE_POINTS` | `"1000"` | Number of sample points for lookup table | Used in percentile calibration |

#### Multi-Class Calibration Parameters

| Variable | Default | Description | Notes |
|----------|---------|-------------|-------|
| `NUM_CLASSES` | `"2"` | Number of classes | Must match number of probability columns |
| `SCORE_FIELD_PREFIX` | `"prob_class_"` | Prefix for probability columns | E.g., `"prob_class_0"`, `"prob_class_1"` |
| `MULTICLASS_CATEGORIES` | `"[0, 1]"` | JSON array of class names/indices | Must match NUM_CLASSES length |

### Job Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--job_type` | `str` | No | `"calibration"` | Job type determining data loading strategy: `"training"` (nested tarballs), `"calibration"` (direct files), `"validation"`, `"testing"` |

## Input Data Structure

### Expected Input Format

The script accepts evaluation data in multiple formats and packaging structures:

**Format 1: Direct Files (calibration/validation/testing jobs)**
```
/opt/ml/processing/input/eval_data/
├── predictions.csv (or .tsv, .parquet, .json)
└── _SUCCESS (optional marker)
```

**Format 2: Nested Tarballs (training jobs from XGBoostTraining)**
```
/opt/ml/processing/input/eval_data/
└── output.tar.gz
    ├── val.tar.gz
    │   ├── val/
    │   │   └── predictions.csv
    │   └── val_metrics/
    └── test.tar.gz
        ├── test/
        │   └── predictions.csv
        └── test_metrics/
```

**Format 3: Job Directory Structure**
```
/opt/ml/processing/input/eval_data/
└── {job_id}/
    └── output/
        └── output.tar.gz (contains nested tarballs)
```

### Required Columns

**Single-Task Binary Classification:**
- **Label Column**: Configured via `LABEL_FIELD` (default: `"label"`)
  - Type: Integer (0 or 1) or Float
  - Contains ground truth binary labels

- **Score Column**: Configured via `SCORE_FIELD` (default: `"prob_class_1"`)
  - Type: Float in range [0.0, 1.0]
  - Contains model prediction probabilities for positive class

**Multi-Task Binary Classification:**
- **Label Column**: Configured via `LABEL_FIELD` (default: `"label"`)
  - Type: Integer (0 or 1) or Float
  - Contains ground truth binary labels (shared across all tasks)

- **Score Columns**: Configured via `SCORE_FIELDS` (comma-separated list)
  - Examples: `"task_0_prob,task_1_prob,task_2_prob"`
  - Type: Float in range [0.0, 1.0] for each column
  - Each column represents predictions for an independent binary task

**Multi-Class Classification:**
- **Label Column**: Configured via `LABEL_FIELD` (default: `"label"`)
  - Type: Integer (0 to NUM_CLASSES-1)
  - Contains ground truth class indices

- **Probability Columns**: One per class, named as `{SCORE_FIELD_PREFIX}{class_name}`
  - Examples: `"prob_class_0"`, `"prob_class_1"`, `"prob_class_2"`
  - Type: Float in range [0.0, 1.0]
  - Should ideally sum to 1.0 per row (enforced after calibration)

### Optional Columns

- **ID Column**: Any identifier column (preserved in output)
- **Dataset Origin**: Added by script when loading from nested tarballs (`"dataset_origin": "val"` or `"test"`)
- **Additional Features**: All other columns preserved in calibrated output

### Supported Input Sources

1. **XGBoostModelEval Output**: Standard predictions from model evaluation
   - Format: CSV/TSV/Parquet with `prob_class_*` columns
   - Direct file structure

2. **XGBoostTraining Output**: Validation/test predictions from training jobs
   - Format: Nested tar.gz archives
   - Requires `job_type="training"` argument

3. **LightGBMModelEval Output**: Model evaluation predictions
   - Format: CSV/TSV/Parquet/JSON
   - Multi-format support with auto-detection

4. **PyTorchModelEval Output**: PyTorch model predictions
   - Format: CSV/TSV/Parquet
   - Compatible probability column naming

## Output Data Structure

### Calibration Output Directory Structure

```
/opt/ml/processing/output/calibration/
├── calibration_model.pkl                    # Binary calibration model
├── calibration_models/                      # Multi-class calibration models
│   ├── calibration_model_class_0.pkl
│   ├── calibration_model_class_1.pkl
│   └── ...
└── calibration_summary.json                 # Summary with improvement metrics
```

### Metrics Output Directory Structure

```
/opt/ml/processing/output/metrics/
├── calibration_metrics.json                 # Detailed calibration metrics
└── reliability_diagram.png                  # Visual calibration comparison
    # OR for multi-class:
    └── multiclass_reliability_diagram.png   # Per-class reliability plots
```

### Calibrated Data Output Structure

```
/opt/ml/processing/output/calibrated_data/
└── calibrated_data.{format}                 # Same format as input
```

**Binary Classification - Added Columns**:
- `calibrated_{SCORE_FIELD}`: Calibrated probability for positive class
  - Example: `calibrated_prob_class_1`

**Multi-Class Classification - Added Columns**:
- `calibrated_{SCORE_FIELD_PREFIX}{class_name}`: One per class
  - Examples: `calibrated_prob_class_0`, `calibrated_prob_class_1`
  - Normalized to sum to 1.0 per row

### Calibration Summary JSON

```json
{
  "status": "success",
  "mode": "binary",  // or "multi-class"
  "calibration_method": "gam",
  "uncalibrated_ece": 0.0532,
  "calibrated_ece": 0.0124,
  "improvement_percentage": 76.7,
  "output_files": {
    "metrics": "/opt/ml/processing/output/metrics/calibration_metrics.json",
    "calibrator": "/opt/ml/processing/output/calibration/calibration_model.pkl",
    "calibrated_data": "/opt/ml/processing/output/calibrated_data/calibrated_data.csv"
  }
}
```

### Calibration Metrics JSON

**Binary Classification**:
```json
{
  "mode": "binary",
  "calibration_method": "gam",
  "uncalibrated": {
    "expected_calibration_error": 0.0532,
    "maximum_calibration_error": 0.1234,
    "brier_score": 0.1543,
    "auc_roc": 0.8765,
    "reliability_diagram": {
      "true_probs": [0.1, 0.2, ...],
      "pred_probs": [0.15, 0.25, ...]
    },
    "bin_statistics": {
      "bin_counts": [100, 150, ...],
      "bin_predicted_probs": [0.05, 0.15, ...],
      "bin_true_probs": [0.1, 0.2, ...],
      "calibration_errors": [0.05, 0.05, ...]
    }
  },
  "calibrated": {
    // Same structure as uncalibrated
  },
  "improvement": {
    "ece_reduction": 0.0408,
    "mce_reduction": 0.0987,
    "brier_reduction": 0.0234,
    "auc_change": -0.0012  // Small AUC change expected
  }
}
```

**Multi-Class Classification**:
```json
{
  "mode": "multi-class",
  "num_classes": 3,
  "class_names": ["class_0", "class_1", "class_2"],
  "multiclass_brier_score": 0.2345,
  "macro_expected_calibration_error": 0.0456,
  "per_class_metrics": [
    {
      "class_index": 0,
      "class_name": "class_0",
      "metrics": {
        "expected_calibration_error": 0.0432,
        // Full metrics like binary case
      }
    }
  ]
}
```

## Key Functions and Tasks

### Package Installation Component

#### `install_packages(packages: list, use_secure: bool) -> None`
**Purpose**: Install Python packages from either public PyPI or secure CodeArtifact based on configuration

**Algorithm**:
```python
1. Log installation configuration
   - PyPI source (PUBLIC or SECURE)
   - Package count and names
2. If use_secure is True:
   a. Call _get_secure_pypi_access_token()
   b. Construct CodeArtifact index URL with token
   c. Execute pip install with --index-url flag
3. Else:
   a. Execute standard pip install
4. Log success or raise exception on failure
```

**Parameters**:
- `packages` (list): Package specifications (e.g., `["numpy==1.24.4", "scipy==1.10.1"]`)
- `use_secure` (bool): If True, use secure CodeArtifact; if False, use public PyPI

**Required Packages**:
- `numpy==1.24.4`: Numerical operations
- `scipy==1.10.1`: Statistical functions
- `matplotlib>=3.3.0,<3.7.0`: Visualization
- `pygam==0.8.1`: GAM calibration (optional)

#### `_get_secure_pypi_access_token() -> str`
**Purpose**: Obtain CodeArtifact authorization token for secure package installation

**Algorithm**:
```python
1. Set AWS_STS_REGIONAL_ENDPOINTS to "regional"
2. Create STS client for us-east-1
3. Get caller identity to obtain current account
4. Assume SecurePyPIReadRole in account 675292366480
   Role format: "SecurePyPIReadRole_{current_account}"
5. Extract temporary credentials from assumed role
6. Create CodeArtifact client with temporary credentials
7. Get authorization token from "amazon" domain
8. Return authorization token
9. On error: Log and re-raise exception
```

**Returns**: `str` - Authorization token for CodeArtifact

**Error Handling**: Raises exception if role assumption or token retrieval fails

### Configuration Component

#### `CalibrationConfig`
**Purpose**: Configuration class managing all calibration parameters with environment variable integration

**Attributes**:
```python
# I/O Paths
input_data_path: str
output_calibration_path: str
output_metrics_path: str
output_calibrated_data_path: str

# Calibration parameters
calibration_method: str  # "gam", "isotonic", "platt"
label_field: str
score_field: str  # Binary only
is_binary: bool
monotonic_constraint: bool  # GAM only
gam_splines: int  # GAM only
error_threshold: float

# Multi-class parameters
num_classes: int
score_field_prefix: str
multiclass_categories: List[str]

# Internal state
_input_format: str  # Detected format for output preservation
```

#### `CalibrationConfig.from_env() -> CalibrationConfig`
**Purpose**: Factory method creating configuration from environment variables

**Algorithm**:
```python
1. Parse IS_BINARY to determine classification mode
2. If multi-class:
   a. Get MULTICLASS_CATEGORIES environment variable
   b. Try parsing as JSON array
   c. Fallback to comma-separated parsing on failure
3. Create CalibrationConfig instance with:
   - Fixed paths from global variables
   - Calibration method from CALIBRATION_METHOD
   - Field names from LABEL_FIELD, SCORE_FIELD
   - Binary mode from IS_BINARY
   - Constraints from MONOTONIC_CONSTRAINT
   - GAM parameters from GAM_SPLINES
   - Multi-class parameters if applicable
4. Return configured instance
```

**Returns**: `CalibrationConfig` - Configured instance

### Data Loading Component

#### `load_dataframe_with_format(file_path) -> Tuple[pd.DataFrame, str]`
**Purpose**: Load DataFrame and automatically detect its format for preservation

**Algorithm**:
```python
1. Call _detect_file_format(file_path) to get format
2. Based on detected format:
   - "csv": pd.read_csv(file_path)
   - "tsv": pd.read_csv(file_path, sep="\t")
   - "parquet": pd.read_parquet(file_path)
3. Return (DataFrame, format_string)
```

**Returns**: `Tuple[pd.DataFrame, str]` - Loaded data and format identifier

**Complexity**: O(n) where n = number of rows

#### `extract_and_load_nested_tarball_data(config) -> pd.DataFrame`
**Purpose**: Extract and load data from nested SageMaker tar.gz output structure

**Algorithm**:
```python
1. Check for direct data file first (non-tarball case)
   - If found, return using standard loading
2. Search for output.tar.gz in multiple locations:
   a. Direct in input directory
   b. In job-specific directories (job_id/output/)
   c. Recursive search as fallback
3. If no tarball found, fallback to standard loading
4. Create temporary directories for extraction
5. Extract outer archive (output.tar.gz)
6. Find inner archives (val.tar.gz, test.tar.gz)
7. For each inner archive:
   a. Extract to temporary directory
   b. Find predictions.csv in archive_name/ subdirectory
   c. Load predictions with pandas
   d. Add "dataset_origin" column
   e. Concatenate with combined DataFrame
8. Clean up temporary directories
9. Return combined DataFrame with all datasets
```

**Nested Structure Handled**:
```
output.tar.gz
├── val.tar.gz
│   ├── val/predictions.csv       ← Extracted
│   └── val_metrics/              ← Ignored
└── test.tar.gz
    ├── test/predictions.csv      ← Extracted
    └── test_metrics/             ← Ignored
```

**Returns**: `pd.DataFrame` - Combined dataset with dataset_origin column

**Error Handling**:
- Falls back to standard loading if tarballs not found
- Logs warnings for missing files
- Handles column mismatches between datasets

**Complexity**: O(n * k) where n = rows, k = number of inner archives

#### `load_and_prepare_data(config, job_type: str) -> Tuple`
**Purpose**: Load evaluation data and prepare features/labels based on classification type

**Algorithm**:
```python
1. Determine loading strategy based on job_type:
   - "training": Use extract_and_load_nested_tarball_data()
   - Other: Use standard load_data()
2. Store detected format in config._input_format
3. If binary classification:
   a. Extract y_true from label_field
   b. Extract y_prob from score_field
   c. Return (df, y_true, y_prob, None)
4. Elif multi-class classification:
   a. Extract y_true from label_field
   b. Build list of probability column names
   c. Extract y_prob_matrix from probability columns
   d. Return (df, y_true, None, y_prob_matrix)
```

**Returns**:
- Binary: `Tuple[pd.DataFrame, np.ndarray, np.ndarray, None]`
- Multi-class: `Tuple[pd.DataFrame, np.ndarray, None, np.ndarray]`

**Validation**:
- Checks required columns exist
- Warns if multi-class columns partially missing
- Raises ValueError if critical columns absent

### Binary Calibration Component

#### `train_gam_calibration(scores, labels, config) -> LogisticGAM`
**Purpose**: Train a GAM (Generalized Additive Model) calibration with optional monotonicity constraint

**Algorithm**:
```python
1. Reshape scores to column vector (n, 1)
2. If monotonic_constraint is True:
   a. Create LogisticGAM with monotonic_inc constraint
   b. Use s(0, n_splines=gam_splines, constraints="monotonic_inc")
3. Else:
   a. Create LogisticGAM without constraints
   b. Use s(0, n_splines=gam_splines)
4. Fit GAM on (scores, labels)
5. Log training completion with deviance statistic
6. Return trained GAM model
```

**Parameters**:
- `scores` (np.ndarray): Raw prediction scores (n,)
- `labels` (np.ndarray): Binary labels (n,)
- `config` (CalibrationConfig): Configuration object

**Returns**: `LogisticGAM` - Trained GAM model

**GAM Advantages**:
- Smooth non-parametric calibration
- Monotonicity ensures logical probability ordering
- Flexible spline-based modeling

**Complexity**: O(n * s) where n = samples, s = splines

#### `train_isotonic_calibration(scores, labels, config) -> IsotonicRegression`
**Purpose**: Train isotonic regression calibration (piecewise constant monotonic)

**Algorithm**:
```python
1. Create IsotonicRegression(out_of_bounds="clip")
2. Fit on (scores, labels)
3. Log completion
4. Return trained model
```

**Parameters**:
- `scores` (np.ndarray): Raw prediction scores
- `labels` (np.ndarray): Binary labels
- `config` (CalibrationConfig): Optional configuration

**Returns**: `IsotonicRegression` - Trained model

**Isotonic Advantages**:
- Non-parametric, minimal assumptions
- Guaranteed monotonic mapping
- No hyperparameters to tune

**Complexity**: O(n log n) for fitting

#### `train_platt_scaling(scores, labels, config) -> LogisticRegression`
**Purpose**: Train Platt scaling (logistic regression) calibration

**Algorithm**:
```python
1. Reshape scores to column vector (n, 1)
2. Create LogisticRegression(C=1e5)  # Minimal regularization
3. Fit on (scores, labels)
4. Log completion
5. Return trained model
```

**Parameters**:
- `scores` (np.ndarray): Raw prediction scores
- `labels` (np.ndarray): Binary labels
- `config` (CalibrationConfig): Optional configuration

**Returns**: `LogisticRegression` - Trained model

**Platt Scaling Advantages**:
- Simple parametric model
- Fast training and inference
- Works well when miscalibration is roughly sigmoidal

**Complexity**: O(n) for fitting

### Multi-Class Calibration Component

#### `train_multiclass_calibration(y_prob_matrix, y_true, method, config) -> List[Any]`
**Purpose**: Train independent calibration models for each class in one-vs-rest fashion

**Algorithm**:
```python
1. Initialize calibrators list
2. One-hot encode y_true:
   - Create zero matrix (n_samples, n_classes)
   - For each sample, set y_true[i, class_idx] = 1
3. For each class i in range(n_classes):
   a. Log training for class i
   b. If method == "gam":
      - Train GAM on y_prob_matrix[:, i] vs y_true_onehot[:, i]
   c. Elif method == "isotonic":
      - Train IsotonicRegression
   d. Elif method == "platt":
      - Train LogisticRegression
   e. Append calibrator to list
4. Return list of calibrators
```

**Returns**: `List[Any]` - List of calibration models (one per class)

**Complexity**: O(k * c) where k = calibration cost, c = number of classes

**Key Insight**: One-vs-rest approach treats each class independently, then normalizes

#### `apply_multiclass_calibration(y_prob_matrix, calibrators, config) -> np.ndarray`
**Purpose**: Apply per-class calibration and normalize to valid probability distribution

**Algorithm**:
```python
1. Initialize calibrated_probs matrix (n_samples, n_classes)
2. For each class i:
   a. Log application for class i
   b. If calibrator is IsotonicRegression:
      - calibrated_probs[:, i] = calibrator.transform(y_prob_matrix[:, i])
   c. Elif calibrator is LogisticRegression:
      - calibrated_probs[:, i] = calibrator.predict_proba(...)[:, 1]
   d. Else (GAM):
      - calibrated_probs[:, i] = calibrator.predict_proba(...)
3. Normalize each row:
   a. Compute row_sums = sum across columns
   b. Divide each row by its sum
   c. Ensures Σ P(class_i) = 1.0 for each sample
4. Return normalized calibrated_probs
```

**Returns**: `np.ndarray` - Calibrated probability matrix (n_samples, n_classes)

**Normalization Necessity**: Per-class calibration doesn't guarantee sum-to-one, so explicit normalization required

**Complexity**: O(n * c) where n = samples, c = classes

### Calibration Metrics Component

#### `compute_calibration_metrics(y_true, y_prob, n_bins: int = 10) -> Dict`
**Purpose**: Compute comprehensive calibration quality metrics

**Algorithm**:
```python
1. Compute calibration curve using sklearn.calibration_curve
   - Returns (prob_true, prob_pred) for plotting
2. Assign samples to bins:
   - bin_indices = floor(y_prob * n_bins)
   - Clamp to [0, n_bins-1]
3. For each bin:
   a. Count samples in bin
   b. Compute mean predicted probability
   c. Compute mean true label (fraction of positives)
4. Compute Expected Calibration Error (ECE):
   - ECE = Σ (n_bin / n_total) * |mean_pred - mean_true|
   - Weighted average of absolute calibration errors
5. Compute Maximum Calibration Error (MCE):
   - MCE = max(|mean_pred - mean_true|) across bins
6. Compute Brier score:
   - Brier = (1/n) * Σ (y_pred - y_true)²
   - Quadratic scoring rule
7. Compute AUC-ROC for discrimination
8. Create detailed bin statistics
9. Return comprehensive metrics dictionary
```

**Returns**: `Dict` containing:
- `expected_calibration_error` (float): Weighted avg calibration error
- `maximum_calibration_error` (float): Max bin calibration error
- `brier_score` (float): Quadratic scoring rule
- `auc_roc` (float): Discrimination metric
- `reliability_diagram` (dict): Points for plotting
- `bin_statistics` (dict): Detailed per-bin information

**Metrics Interpretation**:
- **ECE < 0.05**: Well-calibrated
- **ECE 0.05-0.10**: Moderately calibrated
- **ECE > 0.10**: Poorly calibrated
- **Brier score**: Lower is better (0 = perfect)
- **AUC**: Should remain similar before/after (discrimination preservation)

**Complexity**: O(n + b) where n = samples, b = bins

#### `compute_multiclass_calibration_metrics(y_true, y_prob_matrix, n_bins, config) -> Dict`
**Purpose**: Compute calibration metrics for multi-class scenario

**Algorithm**:
```python
1. One-hot encode y_true
2. For each class:
   a. Compute binary calibration metrics
   b. Store in class_metrics list
3. Compute multi-class Brier score:
   - For each sample:
      - For each class:
         - If true_class: add (1 - P(class))²
         - Else: add P(class)²
   - Divide by n_samples
4. Compute macro-averaged metrics:
   - macro_ece = mean of per-class ECEs
   - macro_mce = mean of per-class MCEs
   - max_mce = maximum MCE across all classes
5. Return aggregated metrics
```

**Returns**: `Dict` containing:
- `multiclass_brier_score` (float): Overall Brier score
- `macro_expected_calibration_error` (float): Average ECE
- `per_class_metrics` (list): Detailed metrics per class
- Aggregated statistics

**Complexity**: O(n * c * b) where n = samples, c = classes, b = bins

### Visualization Component

#### `plot_reliability_diagram(y_true, y_prob_uncalibrated, y_prob_calibrated, n_bins, config) -> str`
**Purpose**: Create reliability diagram comparing uncalibrated and calibrated probabilities

**Algorithm**:
```python
1. Create figure with 2 subplots:
   - Top: Calibration curves (2/3 height)
   - Bottom: Histograms (1/3 height)
2. Top subplot (calibration curves):
   a. Plot y=x line (perfect calibration)
   b. Compute and plot uncalibrated calibration curve
   c. Compute and plot calibrated calibration curve
   d. Add legend and labels
3. Bottom subplot (histograms):
   a. Histogram of uncalibrated predictions
   b. Histogram of calibrated predictions (overlay)
   c. Add legend
4. Apply tight_layout()
5. Save to output_metrics_path/reliability_diagram.png
6. Close figure
7. Return file path
```

**Returns**: `str` - Path to saved figure

**Visual Interpretation**:
- Curves closer to diagonal = better calibration
- Histograms show prediction distribution changes
- Ideal: calibrated curve hugs diagonal

**Example Output**: 
```
reliability_diagram.png showing:
- Diagonal perfect calibration line
- Uncalibrated curve (likely deviated)
- Calibrated curve (closer to diagonal)
- Prediction histograms
```

#### `plot_multiclass_reliability_diagram(y_true, y_prob_uncalibrated, y_prob_calibrated, n_bins, config) -> str`
**Purpose**: Create per-class reliability diagrams in grid layout

**Algorithm**:
```python
1. Determine grid layout:
   - n_cols = min(3, num_classes)
   - n_rows = ceil(num_classes / n_cols)
2. Create figure with subplots grid
3. One-hot encode y_true
4. For each class i:
   a. Get appropriate subplot axis
   b. Plot y=x perfect calibration line
   c. Compute and plot uncalibrated curve for class i
   d. Compute and plot calibrated curve for class i
   e. Set title with class name
   f. Add legend
5. Hide unused subplots if any
6. Apply tight_layout()
7. Save to multiclass_reliability_diagram.png
8. Return file path
```

**Returns**: `str` - Path to saved figure

**Grid Layout Example**:
- 3 classes: 3x1 grid
- 5 classes: 3x2 grid (one empty)
- 10 classes: 3x4 grid (two empty)

### Main Orchestration Component

#### `main(input_paths, output_paths, environ_vars, job_args) -> dict`
**Purpose**: Orchestrate complete calibration workflow from data loading to output generation

**Algorithm**:
```python
1. Parse multiclass_categories from environment
2. Create CalibrationConfig from parameters
3. Log calibration start with mode (binary/multi-class)
4. Create output directories
5. Get job_type from command line arguments (default: "calibration")
6. Branch based on classification mode:

   === BINARY CLASSIFICATION ===
   7. Load and prepare data with load_and_prepare_data()
   8. Select calibration method:
      - "gam": train_gam_calibration()
      - "isotonic": train_isotonic_calibration()
      - "platt": train_platt_scaling()
   9. Apply calibration to get y_prob_calibrated
   10. Compute metrics before and after:
       - uncalibrated_metrics = compute_calibration_metrics(y_true, y_prob_uncalibrated)
       - calibrated_metrics = compute_calibration_metrics(y_true, y_prob_calibrated)
   11. Create reliability diagram visualization
   12. Build metrics report with improvement stats
   13. Save metrics report JSON
   14. Save calibrator model as pickle
   15. Add calibrated column to DataFrame
   16. Save calibrated data in original format
   17. Write summary JSON with ECE improvement percentage
   18. Check improvement and log warnings if < 5%
   
   === MULTI-CLASS CLASSIFICATION ===
   7. Load and prepare data (gets y_prob_matrix)
   8. Train per-class calibrators: train_multiclass_calibration()
   9. Apply calibration: apply_multiclass_calibration()
   10. Compute metrics before and after:
       - uncalibrated_metrics = compute_multiclass_calibration_metrics()
       - calibrated_metrics = compute_multiclass_calibration_metrics()
   11. Create multi-class reliability diagram
   12. Build metrics report with per-class details
   13. Save metrics report JSON
   14. Save per-class calibrator models
   15. Add calibrated columns to DataFrame (one per class)
   16. Save calibrated data in original format
   17. Write summary JSON with macro ECE improvement
   18. Check improvement and log warnings if insufficient

19. Log completion message
20. Return results dictionary
```

**Returns**: `dict` containing:
- `status`: "success" or error status
- `mode`: "binary" or "multi-class"
- `calibration_method`: Method used
- `metrics_report`: Full metrics dictionary
- `summary`: Summary statistics

**Error Handling**: Top-level try-except catches all exceptions, logs with traceback, exits with code 1

## Algorithms and Data Structures

### One-vs-Rest Multi-Class Calibration Algorithm

**Problem**: Multi-class classifiers output probability distributions, but per-class probabilities may not be well-calibrated. Need to calibrate each class probability independently while maintaining valid probability distribution (sum to 1).

**Solution Strategy**:
1. Treat each class as binary problem (class vs. all others)
2. Train independent calibrator for each class
3. Apply calibration to each class probability
4. Normalize to ensure valid probability distribution

**Algorithm**:
```python
def multiclass_calibration(y_prob_matrix, y_true, method):
    # Step 1: One-hot encode labels
    y_onehot = zeros(n_samples, n_classes)
    for i in range(n_samples):
        y_onehot[i, y_true[i]] = 1
    
    # Step 2: Train per-class calibrators
    calibrators = []
    for class_i in range(n_classes):
        # Binary problem: class_i vs others
        calibrator = train_binary_calibrator(
            y_prob_matrix[:, class_i],  # Predicted P(class_i)
            y_onehot[:, class_i],       # True indicator for class_i
            method
        )
        calibrators.append(calibrator)
    
    # Step 3: Apply calibration
    calibrated_probs = zeros(n_samples, n_classes)
    for class_i in range(n_classes):
        calibrated_probs[:, class_i] = calibrators[class_i].predict(
            y_prob_matrix[:, class_i]
        )
    
    # Step 4: Normalize to valid distribution
    row_sums = sum(calibrated_probs, axis=1)
    calibrated_probs = calibrated_probs / row_sums[:, newaxis]
    
    return calibrated_probs, calibrators
```

**Complexity**:
- Training: O(n * c * k) where n=samples, c=classes, k=calibration cost
- Inference: O(n * c)
- Space: O(n * c) for probability matrices

**Key Features**:
- Independent per-class calibration preserves class-specific patterns
- Normalization ensures valid probability distribution
- Works with any binary calibration method

### Nested Tarball Extraction Algorithm

**Problem**: SageMaker training jobs output predictions in nested tar.gz structure. Need robust extraction handling multiple directory structures and fallback strategies.

**Solution Strategy**:
1. Try direct file loading first (fast path)
2. Search for output.tar.gz in multiple locations
3. Extract outer archive, then inner archives
4. Combine data from multiple inner archives
5. Clean up temporary files

**Algorithm**:
```python
def extract_nested_tarball(input_dir):
    # Fast path: Check for direct data file
    try:
        file = find_first_data_file(input_dir)
        return load_direct(file)
    except FileNotFoundError:
        pass
    
    # Search for output.tar.gz
    outer_tar = None
    # Location 1: Direct in input_dir
    if exists(input_dir / "output.tar.gz"):
        outer_tar = input_dir / "output.tar.gz"
    # Location 2: In job directories
    elif exists(input_dir / "*" / "output" / "output.tar.gz"):
        outer_tar = find_match(...)
    # Location 3: Recursive search
    else:
        outer_tar = recursive_find(input_dir, "output.tar.gz")
    
    if not outer_tar:
        raise FileNotFoundError
    
    # Extract outer archive
    temp_outer = create_temp_dir()
    extract(outer_tar, temp_outer)
    
    # Find and process inner archives
    combined_df = None
    for inner_tar in find_tarballs(temp_outer):
        archive_name = get_name(inner_tar)  # "val" or "test"
        temp_inner = create_temp_dir()
        extract(inner_tar, temp_inner)
        
        # Load predictions
        predictions_path = temp_inner / archive_name / "predictions.csv"
        df = load_csv(predictions_path)
        df["dataset_origin"] = archive_name
        
        # Combine
        if combined_df is None:
            combined_df = df
        else:
            combined_df = concat(combined_df, df)
    
    # Cleanup
    remove_temp_dirs()
    return combined_df
```

**Complexity**:
- Best case (direct file): O(n) where n = rows
- Worst case (recursive search + extraction): O(n * k + f) where k = archives, f = files searched
- Space: O(n) for final DataFrame, temporary space for extraction

**Robustness Features**:
- Multiple fallback strategies
- Graceful degradation
- Automatic cleanup
- Detailed logging at each step

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|-----------------|------------------|-------|
| Data loading (direct) | O(n) | O(n) | n = rows |
| Data loading (nested) | O(n * k) | O(n) | k = inner archives |
| GAM training | O(n * s * i) | O(s) | s = splines, i = iterations |
| Isotonic training | O(n log n) | O(n) | Sorting-based |
| Platt training | O(n * i) | O(1) | i = iterations |
| GAM inference | O(n * s) | O(n) | Per-sample spline evaluation |
| Isotonic inference | O(n log m) | O(n) | m = unique scores |
| Platt inference | O(n) | O(n) | Linear transform |
| Metrics computation | O(n + b) | O(b) | b = bins |
| Visualization | O(b) | O(1) | Fixed-size plots |
| Multi-class (all) | O(n * c * k) | O(n * c) | c = classes, k = base cost |

### Processing Time Estimates

| Dataset Size | Classes | Method | Typical Time | Notes |
|--------------|---------|--------|--------------|-------|
| 1K samples | Binary | Any | < 1s | Instant |
| 10K samples | Binary | GAM | 2-5s | Spline fitting |
| 10K samples | Binary | Isotonic | 1-2s | Fastest |
| 10K samples | Binary | Platt | 1-2s | Simple fit |
| 100K samples | Binary | GAM | 10-30s | Larger dataset |
| 100K samples | Binary | Isotonic | 5-10s | Still fast |
| 10K samples | 10 classes | GAM | 20-50s | 10x slower |
| 10K samples | 10 classes | Isotonic | 10-20s | Linear scaling |

**Factors Affecting Performance**:
- Dataset size (dominant factor)
- Number of classes (multi-class multiplier)
- Calibration method (GAM slowest, Platt fastest)
- GAM splines count (more splines = slower)
- Input format (tarball extraction overhead)

### Memory Usage

| Component | Memory Usage | Scaling | Notes |
|-----------|--------------|---------|-------|
| Input DataFrame | 8 * n * f bytes | O(n * f) | n=rows, f=fields |
| Probability arrays | 8 * n bytes | O(n) | Binary |
| Probability matrices | 8 * n * c bytes | O(n * c) | Multi-class |
| GAM model | ~1MB | O(s) | s=splines |
| Isotonic model | 16 * m bytes | O(m) | m=unique scores |
| Platt model | <1KB | O(1) | Two parameters |
| Metrics | ~100KB | O(b) | b=bins |
| Peak usage | ~3x input | O(n * f) | Temporary copies |

## Error Handling

### Error Types

#### Configuration Errors

**Missing Required Environment Variable**
- **Cause**: CALIBRATION_METHOD, LABEL_FIELD, SCORE_FIELD, or IS_BINARY not set
- **Handling**: Script expects these via contract, will fail at config creation
- **Prevention**: Ensure step builder sets all required env vars

**Invalid Calibration Method**
- **Cause**: CALIBRATION_METHOD not in ["gam", "isotonic", "platt"]
- **Handling**: Raises ValueError in main() with unknown method message
- **Impact**: Calibration cannot proceed

**Missing pygam Package**
- **Cause**: pygam not installed when CALIBRATION_METHOD="gam"
- **Handling**: Falls back to Platt scaling with warning
- **Impact**: Uses different calibration method than requested

#### Data Loading Errors

**Hyperparameters Not Found (Nested Tarball)**
- **Cause**: output.tar.gz structure doesn't contain val.tar.gz or test.tar.gz
- **Handling**: Falls back to standard data loading
- **Impact**: May fail if no direct data files exist

**Missing Label Column**
- **Cause**: LABEL_FIELD column not in DataFrame
- **Handling**: Raises ValueError with specific column name
- **Impact**: Cannot proceed without labels

**Missing Probability Columns**
- **Cause**: SCORE_FIELD (binary) or prob_class_* (multi-class) not found
- **Handling**: Raises ValueError specifying missing columns
- **Impact**: Cannot calibrate without predictions

#### Calibration Quality Errors

**Negative Improvement**
- **Cause**: Calibration increased ECE instead of decreasing
- **Handling**: Logs warning, continues with results
- **Impact**: Saves poorly calibrated model, user should investigate

**Marginal Improvement (< 5%)**
- **Cause**: ECE improvement less than 5%
- **Handling**: Logs warning about minimal improvement
- **Impact**: Calibration may not be beneficial

**AUC Degradation**
- **Cause**: Calibration significantly changed AUC (> 0.01 drop)
- **Handling**: Logged in metrics but no explicit warning
- **Impact**: May indicate overfitting or data issues

### Error Response Structure

Errors result in exit code 1 with traceback logged. No structured error JSON is created.

## Best Practices

### For Production Deployments

1. **Validate Input Quality**
   - Ensure predictions and labels are from same evaluation set
   - Check that probability columns sum to ~1.0
   - Verify no missing values in required columns
   - Confirm sufficient samples (> 1000 recommended)

2. **Choose Appropriate Method**
   - **GAM**: Best for smooth calibration with monotonicity
   - **Isotonic**: Best when non-parametric flexibility needed
   - **Platt**: Best for simple, fast calibration

3. **Monitor Calibration Quality**
   - Track ECE over time across model versions
   - Set alerts for ECE > 0.10 (poorly calibrated)
   - Review reliability diagrams regularly
   - Compare calibrated vs uncalibrated AUC (should be similar)

4. **Deploy with Model**
   - Save calibration model alongside trained model
   - Apply calibration at inference time
   - Version calibration models with training models
   - Test calibrated predictions on hold-out set

### For Development and Testing

1. **Start with Binary Calibration**
   - Test on binary problems first
   - Verify metrics make sense
   - Check reliability diagrams visually
   - Confirm AUC preservation

2. **Test Multiple Methods**
   - Compare GAM, Isotonic, and Platt on your data
   - Evaluate ECE reduction for each
   - Consider speed vs quality tradeoffs
   - Choose based on production requirements

3. **Validate Format Preservation**
   - Test with CSV, TSV, and Parquet inputs
   - Verify output format matches input
   - Check all columns preserved
   - Confirm calibrated columns added correctly

4. **Test Nested Tarball Loading**
   - Use actual SageMaker training outputs
   - Verify extraction works correctly
   - Check dataset_origin column added
   - Confirm multiple archives combined properly

### For Performance Optimization

1. **Optimize Dataset Size**
   - Use stratified sampling if > 100K samples
   - Maintain class balance in sample
   - Keep at least 10K samples for reliable calibration
   - More samples = better but slower

2. **Tune GAM Parameters**
   - Start with 10 splines (default)
   - Increase to 15-20 for complex relationships
   - Decrease to 5-7 for faster training
   - Always use monotonic constraint for production

3. **Batch Processing**
   - Process multiple models in parallel if possible
   - Share data loading across calibrations
   - Cache format-detected DataFrames
   - Reuse calibration models when appropriate

## Example Configurations

### Basic Binary GAM Calibration

```bash
export CALIBRATION_METHOD="gam"
export LABEL_FIELD="label"
export SCORE_FIELD="prob_class_1"
export IS_BINARY="True"
export MONOTONIC_CONSTRAINT="True"
export GAM_SPLINES="10"

python model_calibration.py
```

**Use Case**: Standard binary classification calibration with smooth monotonic mapping

**Expected Output**:
- calibration_model.pkl (GAM model)
- calibrated_prob_class_1 column added
- ECE typically reduced by 50-80%

### Binary Isotonic Calibration (Fast)

```bash
export CALIBRATION_METHOD="isotonic"
export LABEL_FIELD="label"
export SCORE_FIELD="score"
export IS_BINARY="True"

python model_calibration.py
```

**Use Case**: Fast calibration for large datasets or when non-parametric flexibility needed

**Advantages**:
- Fastest method
- No hyperparameters
- Non-parametric

### Multi-Class Calibration

```bash
export CALIBRATION_METHOD="gam"
export LABEL_FIELD="label"
export IS_BINARY="False"
export NUM_CLASSES="3"
export SCORE_FIELD_PREFIX="prob_class_"
export MULTICLASS_CATEGORIES='["class_0", "class_1", "class_2"]'
export MONOTONIC_CONSTRAINT="True"
export GAM_SPLINES="10"

python model_calibration.py
```

**Use Case**: Multi-class classification with named classes

**Generated Files**:
- calibration_models/calibration_model_class_0.pkl
- calibration_models/calibration_model_class_1.pkl
- calibration_models/calibration_model_class_2.pkl
- multiclass_reliability_diagram.png

**Columns Added**:
- calibrated_prob_class_0
- calibrated_prob_class_1
- calibrated_prob_class_2

### Training Job with Nested Tarballs

```bash
export CALIBRATION_METHOD="gam"
export LABEL_FIELD="label"
export SCORE_FIELD="prob_class_1"
export IS_BINARY="True"

python model_calibration.py --job_type training
```

**Use Case**: Calibrate using validation/test predictions from XGBoost training job

**Input Structure**:
```
/opt/ml/processing/input/eval_data/output.tar.gz
├── val.tar.gz → val/predictions.csv
└── test.tar.gz → test/predictions.csv
```

**Output**: Combined calibration from both val and test sets with dataset_origin column

### Multi-Task Binary Calibration

```bash
export CALIBRATION_METHOD="gam"
export LABEL_FIELD="label"
export SCORE_FIELDS="task_0_prob,task_1_prob,task_2_prob"
export IS_BINARY="True"
export MONOTONIC_CONSTRAINT="True"
export GAM_SPLINES="10"

python model_calibration.py
```

**Use Case**: Multi-task learning with multiple independent binary classifiers that need calibration

**Input Data**:
- Single label column (shared across all tasks)
- Multiple score columns (one per task)

**Generated Files**:
- calibration_models/calibration_model_task_0_prob.pkl
- calibration_models/calibration_model_task_1_prob.pkl
- calibration_models/calibration_model_task_2_prob.pkl
- Separate metrics and diagrams for each task

**Columns Added**:
- calibrated_task_0_prob
- calibrated_task_1_prob
- calibrated_task_2_prob

**Key Features**:
- Independent calibration per task
- Same calibration method applied to all
- Shared labels across tasks

### Secure PyPI Installation

```bash
export USE_SECURE_PYPI="true"
export CALIBRATION_METHOD="gam"
export LABEL_FIELD="label"
export SCORE_FIELD="prob_class_1"
export IS_BINARY="True"

python model_calibration.py
```

**Use Case**: Enterprise deployment requiring secure package sources

**Behavior**: 
- Assumes SecurePyPIReadRole
- Uses CodeArtifact for package installation
- All other behavior identical

## Integration Patterns

### Upstream Integration

```
XGBoostModelEval (or LightGBMModelEval, PyTorchModelEval)
   ↓ (outputs: predictions with prob_class_* columns)
ModelCalibration
   ↓ (outputs: calibration models, calibrated predictions)
```

**Key Connection**: Evaluation step produces predictions with ground truth labels that calibration consumes

### Downstream Integration

```
ModelCalibration
   ↓ (calibration models, calibrated data)
ModelDeployment
   ↓ (deployed model + calibrator)
InferenceEndpoint (applies calibration at inference time)
```

**Usage**: Calibration models deployed alongside trained models for production inference

### Complete Training Pipeline

```
1. XGBoostTraining
   ↓ model.tar.gz + nested predictions (output.tar.gz)
2. ModelCalibration (job_type=training)
   Inputs:
   - Nested tarball from step 1
   - Environment config
   Output:
   ↓ calibration_model.pkl, metrics, calibrated_data
3. ModelPackage
   Inputs:
   - model.tar.gz from step 1
   - calibration_model.pkl from step 2
   Output:
   ↓ packaged model with embedded calibrator
4. ModelRegistration (MIMS)
   ↓ registered model in MIMS with calibration
5. Deployment
   ↓ endpoint applies calibration automatically
```

**Key Integration Points**:
- Training predictions flow to calibration
- Calibration models packaged with trained model
- Deployment applies calibration transparently

## Troubleshooting

### ECE Not Improving

**Symptom**: Calibrated ECE similar to or worse than uncalibrated ECE

**Common Causes**:
1. **Insufficient data**: < 1000 samples
2. **Already well-calibrated**: Uncalibrated ECE < 0.05
3. **Overfitting**: Too many GAM splines for dataset size
4. **Wrong method**: Method not suited for data distribution

**Solution**:
```bash
# Check uncalibrated ECE
cat /opt/ml/processing/output/metrics/calibration_metrics.json | jq '.uncalibrated.expected_calibration_error'

# If < 0.05: Already well-calibrated, no improvement needed
# If > 0.10: Try different method
export CALIBRATION_METHOD="isotonic"  # More flexible

# If GAM overfitting: Reduce splines
export GAM_SPLINES="5"

# Ensure sufficient data
wc -l /opt/ml/processing/input/eval_data/predictions.csv
# Should be > 1000
```

### Probability Columns Not Found

**Symptom**: "Score field 'prob_class_1' not found in data" or similar error

**Common Causes**:
1. **Wrong column name**: SCORE_FIELD doesn't match actual column
2. **Multi-class mismatch**: IS_BINARY=True but only multi-class columns present
3. **Upstream failure**: Evaluation step didn't generate predictions

**Solution**:
```bash
# Check actual columns
head -1 /opt/ml/processing/input/eval_data/predictions.csv

# Correct binary case
export SCORE_FIELD="score"  # Match actual column name

# Or switch to multi-class
export IS_BINARY="False"
export NUM_CLASSES="2"
export SCORE_FIELD_PREFIX="prob_class_"

# Verify upstream step completed
ls -la /opt/ml/processing/input/eval_data/
```

### Nested Tarball Extraction Fails

**Symptom**: "hyperparameters.json not found" or "No val.tar.gz found in output.tar.gz"

**Common Causes**:
1. **Wrong job_type**: Using job_type=training but input is direct files
2. **Corrupted tarball**: Incomplete or damaged archive
3. **Wrong structure**: Tarball doesn't follow expected SageMaker structure

**Solution**:
```bash
# Check if tarball exists
ls -la /opt/ml/processing/input/eval_data/output.tar.gz

# If no tarball, use standard loading
# Remove --job_type argument or set to "calibration"

# If tarball exists, check contents
tar -tzf /opt/ml/processing/input/eval_data/output.tar.gz | head -20

# Should see val.tar.gz and test.tar.gz
# If not, input structure is wrong

# Try direct file loading
ls /opt/ml/processing/input/eval_data/*.csv
# If direct files exist, don't use job_type=training
```

### AUC Significantly Changed

**Symptom**: AUC drops by > 0.01 after calibration

**Common Causes**:
1. **Data leakage**: Calibration set overlaps with test set
2. **Overfitting calibration**: Too complex calibration model
3. **Bug in application**: Calibration applied incorrectly

**Solution**:
```bash
# Check metrics
cat /opt/ml/processing/output/metrics/calibration_metrics.json | \
  jq '.improvement.auc_change'

# If negative and magnitude > 0.01: Problem

# Use simpler method
export CALIBRATION_METHOD="platt"  # Simplest

# Verify data separation
# Ensure calibration data != test data

# Check for duplicate rows
sort /opt/ml/processing/input/eval_data/predictions.csv | uniq -d
```

### Multi-Class Normalization Issues

**Symptom**: Calibrated probabilities don't sum to 1.0

**Common Causes**:
1. **Script bug**: Normalization not applied (shouldn't happen)
2. **Floating point errors**: Very small deviations (acceptable)
3. **Corrupted output**: File writing error

**Solution**:
```bash
# Check output probabilities
python3 << EOF
import pandas as pd
df = pd.read_csv('/opt/ml/processing/output/calibrated_data/calibrated_data.csv')
# Check sum of calibrated columns
prob_cols = [c for c in df.columns if c.startswith('calibrated_prob_class_')]
sums = df[prob_cols].sum(axis=1)
print(f"Mean sum: {sums.mean():.10f}")
print(f"Min sum: {sums.min():.10f}")
print(f"Max sum: {sums.max():.10f}")
# Should all be very close to 1.0 (within 1e-10)
EOF

# If significantly off: Report bug
# If < 1e-6 deviation: Acceptable floating point error
```

## References

### Related Scripts

- [`xgboost_model_eval.py`](xgboost_model_eval_script.md): Generates predictions for calibration input
- [`lightgbm_model_eval.py`](lightgbm_model_eval_script.md): Alternative evaluation producing calibration inputs
- [`package.py`](package_script.md): Packages calibration models with trained models for deployment

### Related Documentation

- **Step Builder**: `src/cursus/steps/builders/builder_model_calibration_step.py`
- **Config Class**: `src/cursus/steps/configs/config_model_calibration_step.py`
- **Contract**: [`src/cursus/steps/contracts/model_calibration_contract.py`](../../src/cursus/steps/contracts/model_calibration_contract.py)

### Related Design Documents

No specific design documents currently exist for this script. General calibration concepts are referenced throughout the codebase in 254 locations covering:
- Job type variant handling
- Multi-class calibration patterns
- Integration with evaluation steps
- Pipeline workflows with calibration

### External References

- [Scikit-learn Calibration](https://scikit-learn.org/stable/modules/calibration.html): Calibration curves and methods
- [pygam Documentation](https://pygam.readthedocs.io/): GAM calibration implementation
- [Platt Scaling Paper](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf): Original Platt scaling method
- [Expected Calibration Error](https://arxiv.org/abs/1706.04599): ECE metric definition and properties
