---
tags:
  - code
  - processing_script
  - model_calibration
  - percentile_mapping
  - roc_analysis
keywords:
  - percentile model calibration
  - ROC curve analysis
  - score mapping
  - percentile mapping
  - calibration dictionary
  - score normalization
  - format preservation
  - model post-processing
topics:
  - model calibration workflows
  - percentile score mapping
  - ROC curve calibration
  - score normalization
language: python
date of note: 2025-11-18
---

# Percentile Model Calibration Script

## Overview

The `percentile_model_calibration.py` script performs percentile score mapping calibration to convert raw model prediction scores to calibrated percentile values using ROC (Receiver Operating Characteristic) curve analysis. This script provides an alternative calibration approach to traditional probability calibration methods by mapping scores to percentiles based on empirical data distribution.

The script supports automatic data format detection and preservation (CSV/TSV/Parquet), flexible calibration dictionary configuration, and comprehensive quality metrics. It integrates seamlessly with upstream model evaluation/inference steps (XGBoost, LightGBM, PyTorch) and provides standardized calibrated outputs for downstream deployment workflows.

The percentile mapping approach is particularly useful for creating consistent risk interpretation across different models and time periods, as percentiles provide a stable reference frame for comparing scores even when underlying score distributions shift.

## Purpose and Major Tasks

### Primary Purpose

Convert raw model prediction scores to calibrated percentile values using ROC curve analysis, enabling consistent risk interpretation and score comparison across models, datasets, and time periods.

### Major Tasks

1. **Calibration Dictionary Loading**: Load calibration dictionary from external configuration or use built-in default with 1000 calibration points spanning the full [0,1] probability range

2. **Evaluation Data Loading**: Load model prediction scores from upstream evaluation/inference steps with automatic format detection (CSV/TSV/Parquet/JSON)

3. **Data Quality Validation**: Validate score ranges, handle missing values, check for constant scores, and verify data diversity requirements

4. **ROC Curve Analysis**: Compute ROC curve on evaluation data to establish empirical relationship between score thresholds and percentile rankings

5. **Percentile Score Mapping**: Generate calibrated score map by interpolating between ROC curve points to find score thresholds corresponding to target percentiles

6. **Score Calibration Application**: Apply percentile mapping function to convert all raw scores to calibrated percentile values

7. **Calibration Artifact Generation**: Save pickled percentile score mapping for deployment and downstream inference applications

8. **Metrics Computation**: Calculate calibration quality metrics including score statistics, calibration range, and mapping characteristics

9. **Calibrated Dataset Generation**: Create output dataset with original features plus calibrated percentile scores in preserved data format

10. **Quality Assurance**: Validate calibration mapping continuity, range coverage, and interpolation accuracy

## Script Contract

### Entry Point

```
percentile_model_calibration.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `evaluation_data` | `/opt/ml/processing/input/eval_data` | Evaluation dataset with model prediction scores (CSV/TSV/Parquet/JSON) |
| `calibration_config` | `/opt/ml/code/calibration` | Optional directory containing `standard_calibration_dictionary.json` for custom calibration points |

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `calibration_output` | `/opt/ml/processing/output/calibration` | Calibration artifacts including `percentile_score.pkl` mapping file |
| `metrics_output` | `/opt/ml/processing/output/metrics` | Calibration quality metrics in `calibration_metrics.json` |
| `calibrated_data` | `/opt/ml/processing/output/calibrated_data` | Dataset with calibrated percentile scores in original format |

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SCORE_FIELD` | Name of the prediction score column to calibrate | `"prob_class_1"` |

### Optional Environment Variables

#### Core Calibration Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `N_BINS` | `"1000"` | Number of bins for calibration analysis (historical parameter, maintained for compatibility) |
| `ACCURACY` | `"1e-5"` | Accuracy threshold for calibration mapping interpolation |

### Job Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--job_type` | `str` | No | Job type identifier (default: `"calibration"`). Supported values: `training`, `calibration`, `validation`, `testing` |

## Input Data Structure

### Expected Input Format

The script accepts evaluation data from various upstream sources with flexible file detection:

```
evaluation_data/
├── eval_predictions.csv                          # XGBoost evaluation (standard)
├── eval_predictions_with_comparison.csv          # XGBoost evaluation (comparison mode)
├── predictions.csv                               # XGBoost/LightGBM/PyTorch inference (CSV)
├── predictions.parquet                           # XGBoost/LightGBM/PyTorch inference (Parquet)
├── predictions.json                              # XGBoost/LightGBM/PyTorch inference (JSON)
└── _SUCCESS                                      # Optional success marker
```

**File Priority**: The script uses intelligent file selection with the following priority:
1. `eval_predictions.csv` (XGBoost standard evaluation)
2. `eval_predictions_with_comparison.csv` (XGBoost comparison)
3. `predictions.csv` (inference CSV output)
4. `predictions.parquet` (inference Parquet output)
5. `predictions.json` (inference JSON output)
6. First supported file alphabetically (fallback)

### Required Columns

- **Score Column**: Configurable via `SCORE_FIELD` environment variable (default: `"prob_class_1"`)
  - Must contain probability scores in range [0, 1]
  - Will be validated for range, missing values, and diversity
  - Example column names: `prob_class_1`, `confidence_score`, `prediction_score`

### Optional Columns

All other columns in the input dataset are preserved in the calibrated output, enabling downstream processing to utilize original features alongside calibrated scores.

### Calibration Dictionary Format

Optional external calibration dictionary in JSON format:

```json
{
  "0.001": 0.995014,
  "0.002": 0.990047,
  "0.005": 0.975267,
  ...
  "0.999": 0.000001
}
```

**Dictionary Structure**:
- **Keys**: Target percentile scores (0.0 to 1.0)
- **Values**: Corresponding population percentile ranks (0.0 to 1.0)
- **Interpretation**: A key of 0.01 with value 0.99 means "1% of the population should score above this threshold"

### Supported Input Sources

1. **XGBoost Model Evaluation**: Standard evaluation output with predictions and optional comparison baselines
2. **XGBoost Model Inference**: Pure inference output without evaluation metrics
3. **LightGBM Model Evaluation/Inference**: Similar structure to XGBoost outputs
4. **PyTorch Model Evaluation/Inference**: Deep learning model outputs with prediction scores
5. **Custom Prediction Files**: Any CSV/TSV/Parquet file containing a score column

## Output Data Structure

### Output Directory Structure

```
calibration_output/
└── percentile_score.pkl              # Pickled percentile score mapping

metrics_output/
└── calibration_metrics.json          # Calibration quality metrics

calibrated_data/
└── calibrated_data.{csv|tsv|parquet} # Dataset with calibrated scores
```

### Calibrated Data Output

**Columns in Output**:
- All original input columns (preserved)
- `{SCORE_FIELD}_percentile`: Calibrated percentile scores (0.0 to 1.0)
  - Example: If `SCORE_FIELD="prob_class_1"`, output includes `prob_class_1_percentile`

**Format Preservation**: Output format matches input format (CSV → CSV, TSV → TSV, Parquet → Parquet)

### Percentile Score Mapping Artifact

**File**: `percentile_score.pkl`

**Contents**: Pickled list of tuples representing the calibrated score map:
```python
[
    (0.0, 0.0),              # (raw_score_threshold, percentile)
    (0.123, 0.001),          # First calibration point
    (0.234, 0.01),
    ...
    (0.987, 0.999),          # Last calibration point
    (1.0, 1.0)
]
```

**Usage**: The mapping is used during inference to convert raw scores to percentiles via linear interpolation between adjacent points.

### Calibration Metrics Output

**File**: `calibration_metrics.json`

**Metrics Contents**:
```json
{
  "calibration_method": "percentile_score_mapping",
  "num_calibration_points": 1002,
  "num_input_scores": 50000,
  "score_statistics": {
    "min_score": 0.001234,
    "max_score": 0.987654,
    "mean_score": 0.523456,
    "std_score": 0.234567
  },
  "calibration_range": {
    "min_percentile": 0.0,
    "max_percentile": 1.0,
    "min_score_threshold": 0.0,
    "max_score_threshold": 1.0
  },
  "config": {
    "n_bins": 1000,
    "score_field": "prob_class_1",
    "accuracy": 1e-05,
    "calibration_dict_size": 1000,
    "job_type": "calibration"
  }
}
```

## Key Functions and Tasks

### Data Loading Component

#### `load_calibration_dictionary(input_paths)`

**Purpose**: Load calibration dictionary from external configuration or use built-in default

**Algorithm**:
```python
1. Check for calibration_config input path
2. If exists:
   a. Look for standard_calibration_dictionary.json
   b. Load and parse JSON
   c. Convert string keys to float keys
3. If not found or fails:
   a. Log fallback message
   b. Return built-in default dictionary (1000 points)
4. Return calibration dictionary
```

**Parameters**:
- `input_paths` (Dict[str, str]): Dictionary of input paths with logical names

**Returns**: `Dict[float, float]` - Calibration dictionary mapping percentile scores to population ranks

**Built-in Default**: 1000-point calibration dictionary spanning [0.001, 0.999] range with empirically derived percentile mappings

**Example**:
```python
calibration_dict = load_calibration_dictionary({
    "calibration_config": "/opt/ml/code/calibration"
})
# Returns: {0.001: 0.995014, 0.002: 0.990047, ...}
```

#### `find_first_data_file(data_dir)`

**Purpose**: Find the most appropriate data file in directory with intelligent priority ordering

**Algorithm**:
```python
1. Validate directory exists
2. Get all files in directory
3. Check for priority files in order:
   - eval_predictions.csv
   - eval_predictions_with_comparison.csv
   - predictions.csv
   - predictions.parquet
   - predictions.json
4. If priority file found:
   a. Log selection
   b. Warn if multiple priority files exist
   c. Return highest priority file
5. Fallback to any supported file:
   a. Find .csv, .parquet, .json files
   b. Sort alphabetically
   c. Warn if multiple files exist
   d. Return first file
6. Raise error if no supported file found
```

**Parameters**:
- `data_dir` (str): Directory to search for data files

**Returns**: `str` - Path to the most appropriate data file

**Error Handling**:
- Raises `FileNotFoundError` if directory doesn't exist
- Raises `FileNotFoundError` if no supported files found
- Logs warnings when multiple candidates exist

**Example**:
```python
data_file = find_first_data_file("/opt/ml/processing/input/eval_data")
# Returns: "/opt/ml/processing/input/eval_data/eval_predictions.csv"
```

#### `load_dataframe_with_format(file_path)`

**Purpose**: Load DataFrame and automatically detect its format

**Algorithm**:
```python
1. Detect format from file extension:
   - .csv → "csv"
   - .tsv → "tsv"
   - .parquet → "parquet"
2. Load data using appropriate reader:
   - CSV: pd.read_csv()
   - TSV: pd.read_csv(sep="\t")
   - Parquet: pd.read_parquet()
3. Return (DataFrame, format_string)
```

**Parameters**:
- `file_path` (str): Path to the file

**Returns**: `Tuple[pd.DataFrame, str]` - DataFrame and detected format

**Complexity**:
- Time: O(n) where n is file size
- Space: O(n) for DataFrame storage

### Format Preservation Component

#### `_detect_file_format(file_path)`

**Purpose**: Detect the format of a data file based on its extension

**Algorithm**:
```python
1. Extract file suffix using pathlib
2. Convert to lowercase
3. Map suffix to format:
   - .csv → "csv"
   - .tsv → "tsv"
   - .parquet → "parquet"
4. Raise error if unsupported
```

**Parameters**:
- `file_path` (str): Path to the file

**Returns**: `str` - Format string ('csv', 'tsv', or 'parquet')

**Example**:
```python
format_str = _detect_file_format("data.parquet")
# Returns: "parquet"
```

#### `save_dataframe_with_format(df, output_path, format_str)`

**Purpose**: Save DataFrame in specified format with appropriate extension

**Algorithm**:
```python
1. Create Path object from output_path
2. Based on format_str:
   - "csv": Add .csv extension, save with to_csv()
   - "tsv": Add .tsv extension, save with to_csv(sep="\t")
   - "parquet": Add .parquet extension, save with to_parquet()
3. Return path to saved file
```

**Parameters**:
- `df` (pd.DataFrame): DataFrame to save
- `output_path` (str): Base output path (without extension)
- `format_str` (str): Format to save in

**Returns**: `str` - Path to saved file

**Complexity**:
- Time: O(n) where n is DataFrame size
- Space: O(1) (streaming write)

### ROC Curve Calibration Component

#### `get_calibrated_score_map(df, score_field, calibration_dictionary, weight_field)`

**Purpose**: Calculate calibrated score map using ROC curve analysis and target percentiles

**Algorithm**:
```python
1. Add "all" column set to 1 (for ROC curve calculation)
2. Compute ROC curve:
   - If weight_field: Use sample_weight parameter
   - Else: Standard ROC curve
   - Returns: (false_positive_rate, true_positive_rate, thresholds)
3. Augment ROC curve arrays:
   - Prepend: (0.0, 1.0) to pct, 1.0 to thresholds
   - Append: (1.0, 0.0) to pct, 0.0 to thresholds
4. Initialize score_map with (0.0, 0.0)
5. For each calibrated score in sorted dictionary keys:
   a. Find bracketing percentile range in ROC curve
   b. Linear interpolation to find exact score threshold:
      score = threshold[i] + (threshold[i+1] - threshold[i]) * 
              (target_pct - pct[i]) / (pct[i+1] - pct[i])
   c. Append (score_threshold, calibrated_score) to map
6. Append (1.0, 1.0) to score_map
7. Return score_map
```

**Parameters**:
- `df` (pd.DataFrame): Input data with scores
- `score_field` (str): Name of score column
- `calibration_dictionary` (Dict[float, float]): Target percentile mappings
- `weight_field` (Optional[str]): Optional column for sample weights

**Returns**: `List[Tuple[float, float]]` - Sorted list of (score_threshold, percentile) pairs

**Mathematical Foundation**:

The ROC curve provides the empirical relationship:
```
P(score > t | data) = percentile_rank(t)
```

For each target percentile p in calibration_dictionary:
1. Find percentile rank on ROC curve
2. Interpolate to find corresponding score threshold
3. Create mapping: raw_score → calibrated_percentile

**Complexity**:
- Time: O(n log n + m log n) where n = data size, m = calibration points
- Space: O(n) for ROC curve arrays

**Example**:
```python
score_map = get_calibrated_score_map(
    df=calibration_data,
    score_field="prob_class_1",
    calibration_dictionary={0.01: 0.99, 0.05: 0.95, 0.1: 0.90},
    weight_field=None
)
# Returns: [(0.0, 0.0), (0.123, 0.01), (0.234, 0.05), (0.345, 0.1), (1.0, 1.0)]
```

### Score Mapping Application Component

#### `apply_percentile_mapping(score, score_map)`

**Purpose**: Apply percentile mapping to a single score using linear interpolation

**Algorithm**:
```python
1. If score <= min_threshold: return min_percentile
2. If score >= max_threshold: return max_percentile
3. Find bracketing range:
   - Search score_map for i where:
     score_map[i][0] <= score <= score_map[i+1][0]
4. Linear interpolation:
   - x1, y1 = score_map[i]
   - x2, y2 = score_map[i+1]
   - If x2 == x1: return y1
   - Else: return y1 + (y2 - y1) * (score - x1) / (x2 - x1)
5. Return interpolated percentile
```

**Parameters**:
- `score` (float): Raw score to map
- `score_map` (List[Tuple[float, float]]): Calibrated score mapping

**Returns**: `float` - Calibrated percentile value

**Mathematical Formula**:
```
percentile = y1 + (y2 - y1) * (score - x1) / (x2 - x1)
```

**Complexity**:
- Time: O(m) where m = number of calibration points
- Space: O(1)

**Example**:
```python
percentile = apply_percentile_mapping(
    score=0.567,
    score_map=[(0.0, 0.0), (0.5, 0.3), (0.6, 0.4), (1.0, 1.0)]
)
# Interpolates between (0.5, 0.3) and (0.6, 0.4)
# Returns: ~0.367
```

### Data Quality Validation Component

#### Data Validation in `main()`

**Purpose**: Validate data quality and suitability for calibration

**Validation Checks**:

1. **Missing Values**:
```python
missing_count = pd.isna(scores).sum()
if missing_count > 0:
    # Remove missing values, log warning
    valid_mask = ~pd.isna(scores)
    data = data[valid_mask]
```

2. **Score Range Validation**:
```python
if min_score < 0 or max_score > 1:
    # Log warning
    if min_score < -0.1 or max_score > 1.1:
        # Raise error for extreme violations
    else:
        # Clip to [0, 1] range
        scores = np.clip(scores, 0.0, 1.0)
```

3. **Constant Score Detection**:
```python
if std_score < 1e-10:
    # Raise error - cannot perform calibration
```

4. **Diversity Check**:
```python
unique_scores = len(np.unique(scores))
if unique_scores < 10:
    # Log warning about low diversity
```

**Error Handling**:
- Missing values: Removed with warning
- Out-of-range scores: Clipped with warning (errors if extreme)
- Constant scores: Fatal error (calibration impossible)
- Low diversity: Warning only

### Main Orchestration Component

#### `main(input_paths, output_paths, environ_vars, job_args)`

**Purpose**: Orchestrate the complete percentile calibration workflow

**Workflow Steps**:
```python
1. Parse configuration from environment variables
2. Extract job_type from command line arguments
3. Create output directories
4. Load calibration dictionary
5. Find and load evaluation data
6. Validate data quality
7. Perform ROC-based calibration
8. Generate calibrated score map
9. Apply mapping to all scores
10. Save calibration artifacts
11. Save metrics
12. Save calibrated dataset
13. Return results summary
```

**Parameters**:
- `input_paths` (Dict[str, str]): Input path mappings
- `output_paths` (Dict[str, str]): Output path mappings
- `environ_vars` (Dict[str, str]): Environment variable configuration
- `job_args` (argparse.Namespace): Command line arguments

**Returns**: `Dict[str, Any]` - Results dictionary with status, metrics, and output paths

**Error Handling**:
- All exceptions caught and returned in results dictionary
- Detailed error messages with traceback
- Status field indicates success/error

## Algorithms and Data Structures

### ROC Curve-Based Percentile Mapping Algorithm

**Problem**: Convert raw model scores to calibrated percentiles that reflect empirical data distribution while maintaining monotonicity and providing stable risk interpretation.

**Solution Strategy**:
1. Use ROC curve to establish empirical relationship between scores and population percentiles
2. Define target percentile breakpoints via calibration dictionary
3. Interpolate ROC curve to find score thresholds for each target percentile
4. Create piecewise linear mapping function from thresholds

**Algorithm**:
```python
# Step 1: Compute ROC curve on calibration data
df["all"] = 1  # Create binary label (all positive for percentile computation)
fpr, tpr, thresholds = roc_curve(df["all"], df[score_field], sample_weight=weights)

# Step 2: Augment ROC curve with boundary points
pct = [0.0] + list(tpr) + [1.0]  # Percentile ranks
thresholds = [1.0] + list(thresholds) + [0.0]  # Score thresholds

# Step 3: For each target percentile in calibration dictionary
score_map = [(0.0, 0.0)]
for target_score in sorted(calibration_dict.keys()):
    target_percentile = calibration_dict[target_score]
    
    # Find bracketing range in ROC curve
    for i in range(len(pct) - 1):
        if pct[i] <= target_percentile <= pct[i+1]:
            # Linear interpolation to find exact threshold
            if pct[i+1] == pct[i]:
                score_threshold = thresholds[i]
            else:
                score_threshold = thresholds[i] + \
                                (thresholds[i+1] - thresholds[i]) * \
                                (target_percentile - pct[i]) / \
                                (pct[i+1] - pct[i])
            
            score_map.append((score_threshold, target_score))
            break

score_map.append((1.0, 1.0))
return score_map
```

**Complexity Analysis**:
- **Time**: O(n log n + m log n)
  - n log n: Sorting for ROC curve computation
  - m log n: Finding m calibration points via search
- **Space**: O(n + m)
  - n: ROC curve arrays
  - m: Score map entries

**Key Features**:
- **Monotonicity**: Preserved by ROC curve properties and sorted interpolation
- **Empirical**: Based on actual data distribution, not parametric assumptions
- **Stable**: Percentiles provide consistent interpretation across datasets
- **Flexible**: Calibration dictionary allows custom percentile breakpoints

### Linear Interpolation for Score Mapping

**Problem**: Convert arbitrary raw scores to calibrated percentiles using pre-computed mapping.

**Solution Strategy**: Piecewise linear interpolation between adjacent calibration points.

**Algorithm**:
```python
def apply_percentile_mapping(score, score_map):
    # Boundary cases
    if score <= score_map[0][0]:
        return score_map[0][1]
    if score >= score_map[-1][0]:
        return score_map[-1][1]
    
    # Find bracketing range
    for i in range(len(score_map) - 1):
        x1, y1 = score_map[i]
        x2, y2 = score_map[i+1]
        
        if x1 <= score <= x2:
            # Linear interpolation
            if x2 == x1:
                return y1
            return y1 + (y2 - y1) * (score - x1) / (x2 - x1)
    
    return score_map[-1][1]  # fallback
```

**Complexity**:
- **Time**: O(m) where m = number of calibration points
- **Space**: O(1)

**Optimization Opportunity**: Binary search could reduce to O(log m) but not implemented since m is typically small (1000 points).

### Default Calibration Dictionary Structure

**Design**: 1000-point percentile mapping covering [0.001, 0.999] range

**Properties**:
- **Coverage**: Dense coverage from 0.1% to 99.9% percentiles
- **Granularity**: 0.001 spacing in lower range, gradually coarser
- **Empirical**: Derived from historical production data distributions
- **Stable**: Provides consistent risk interpretation across models

**Structure**:
```python
{
    0.001: 0.995014,  # Top 0.1% → 99.5th percentile
    0.002: 0.990047,  # Top 0.2% → 99.0th percentile
    ...
    0.5: 0.031717,    # Median → 3.2nd percentile
    ...
    0.999: 0.000001   # Bottom 0.1% → 0.0001st percentile
}
```

**Usage**: Maps desired calibrated scores (keys) to population percentile ranks (values).

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Load calibration dict | O(m) | O(m) | m = dict size (1000) |
| Find data file | O(k) | O(1) | k = files in directory |
| Load DataFrame | O(n) | O(n) | n = dataset size |
| Data validation | O(n) | O(1) | Linear scan of scores |
| ROC curve computation | O(n log n) | O(n) | Sorting dominates |
| Score map generation | O(m log n) | O(m) | m searches in ROC curve |
| Apply mapping (single) | O(m) | O(1) | Linear search |
| Apply mapping (batch) | O(n * m) | O(n) | n scores × m points |
| Save artifacts | O(1) | O(1) | Small pickle |
| Save calibrated data | O(n) | O(1) | Streaming write |
| **Overall** | **O(n log n + n*m)** | **O(n + m)** | Dominated by batch mapping |

**Typical Performance**:
- **n** (dataset size): 10,000 - 1,000,000 records
- **m** (calibration points): 1,000 (constant)
- **Effective complexity**: O(n) with large constant factor (~1000x)

### Memory Usage

| Component | Memory Usage | Description |
|-----------|-------------|-------------|
| Input DataFrame | ~100-500 MB | Depends on dataset size and columns |
| ROC curve arrays | ~50-200 KB | Three arrays of length n |
| Calibration dictionary | ~8 KB | 1000 float pairs |
| Score map | ~16 KB | 1002 tuples |
| Calibrated scores | ~8-40 MB | Array of n floats |
| Output DataFrame | ~100-500 MB | Copy of input + percentile column |
| **Peak Memory** | **2-3x input size** | During DataFrame operations |

### Processing Time Estimates

| Dataset Size | ROC Computation | Mapping Application | Total Time | Notes |
|--------------|----------------|-------------------|------------|-------|
| 10K records | <1 second | 1-2 seconds | ~3 seconds | Fast |
| 100K records | 1-2 seconds | 10-20 seconds | ~25 seconds | Good |
| 1M records | 10-15 seconds | 100-200 seconds | ~220 seconds | Acceptable |
| 10M records | 100-150 seconds | 1000-2000 seconds | ~35 minutes | Slow |

**Optimization Note**: Batch mapping could be vectorized but current implementation prioritizes code clarity.

### Format-Specific Performance

| Format | Read Speed | Write Speed | Compression | Best For |
|--------|-----------|-------------|-------------|----------|
| CSV | Moderate | Fast | None | Human-readable, debugging |
| TSV | Moderate | Fast | None | Tab-separated legacy systems |
| Parquet | Fast | Fast | High | Large datasets, production |

## Error Handling

### Error Types

#### Input Validation Errors

- **Missing Score Field**: `ValueError` raised if `SCORE_FIELD` column not found in data
  - **Cause**: Incorrect score field name in configuration
  - **Handling**: Script terminates with clear error message listing available columns

- **Missing Values in Scores**: Handled automatically with data cleaning
  - **Cause**: Upstream prediction failures or data quality issues
  - **Handling**: Missing values removed with warning, count logged

- **Out-of-Range Scores**: Scores outside [0, 1] probability range
  - **Cause**: Uncalibrated raw model outputs or data errors
  - **Handling**: 
    - Minor violations (-0.1 to 1.1): Clipped with warning
    - Major violations (outside [-0.1, 1.1]): Fatal error

- **Constant Scores**: All scores identical (std < 1e-10)
  - **Cause**: Model failure or data corruption
  - **Handling**: Fatal error - calibration impossible

#### Data Quality Errors

- **No Data File Found**: `FileNotFoundError` when no supported files exist
  - **Cause**: Incorrect input path or missing upstream output
  - **Handling**: Script terminates with list of expected file names

- **Low Score Diversity**: < 10 unique score values
  - **Cause**: Limited evaluation dataset or quantized predictions
  - **Handling**: Warning logged but processing continues

- **Insufficient Data**: Very small datasets (< 100 records)
  - **Cause**: Upstream filtering or sampling issues
  - **Handling**: Warning logged - calibration may be unreliable

#### Configuration Errors

- **Missing Required Environment Variable**: `SCORE_FIELD` not set
  - **Cause**: Pipeline configuration error
  - **Handling**: Script uses default `"prob_class_1"` with warning

- **Invalid Calibration Dictionary**: Malformed JSON or invalid structure
  - **Cause**: Corrupted config file or incorrect format
  - **Handling**: Falls back to built-in default with warning

#### Processing Errors

- **ROC Curve Computation Failure**: Numerical issues during ROC calculation
  - **Cause**: Extreme score distributions or numerical instability
  - **Handling**: Full traceback logged, returns error status

- **Interpolation Errors**: Issues during score mapping generation
  - **Cause**: Numerical precision issues or edge cases
  - **Handling**: Falls back to nearest neighbor mapping

### Error Response Structure

```python
{
    "status": "error",
    "error_message": "Descriptive error message",
    "traceback": "Full Python traceback",
    # Original configuration for debugging
    "config": {
        "n_bins": 1000,
        "score_field": "prob_class_1",
        "accuracy": 1e-05
    }
}
```

## Best Practices

### For Production Deployments

1. **Validate Calibration Dictionary**: Use external calibration dictionary with empirically validated percentile mappings rather than relying on built-in defaults
   - Rationale: Built-in defaults may not match your specific use case or population

2. **Monitor Score Distribution Shifts**: Track score statistics over time to detect distribution drift
   - Rationale: Percentile calibration assumes stable score distributions

3. **Use Consistent Score Fields**: Standardize score field naming across all pipelines
   - Rationale: Reduces configuration errors and simplifies maintenance

4. **Preserve Format for Performance**: Use Parquet format for large-scale production pipelines
   - Rationale: Better compression and faster I/O than CSV

5. **Archive Calibration Artifacts**: Version and archive percentile_score.pkl files
   - Rationale: Enables reproducibility and audit trails

### For Development

1. **Start with Small Datasets**: Test calibration logic on small samples before full-scale runs
   - Rationale: Faster iteration and easier debugging

2. **Inspect Calibration Metrics**: Review calibration_metrics.json for data quality issues
   - Rationale: Catches upstream problems early

3. **Validate Score Ranges**: Check that input scores are in [0, 1] range
   - Rationale: Prevents cryptic errors during calibration

4. **Test with Edge Cases**: Verify behavior with constant scores, missing values, extreme distributions
   - Rationale: Ensures robust error handling

5. **Compare with Probability Calibration**: Run both percentile and probability calibration for comparison
   - Rationale: Understand trade-offs between different calibration approaches

### For Performance Optimization

1. **Use Parquet for Large Datasets**: Switch to Parquet format for datasets > 100K records
   - Rationale: 10-50x faster I/O than CSV

2. **Reduce Calibration Points**: Use fewer calibration dictionary entries (e.g., 100 instead of 1000)
   - Rationale: Linear speedup in mapping application (1000x → 100x)
   - Trade-off: Slightly less precise percentile mapping

3. **Batch Processing**: For multiple models, process sequentially to avoid memory pressure
   - Rationale: Each calibration job holds 2-3x input data in memory

4. **Monitor Memory Usage**: Track peak memory for large datasets
   - Rationale: Prevents OOM errors in production

5. **Pre-filter Input Data**: Remove unnecessary columns before calibration
   - Rationale: Reduces memory footprint and I/O time

## Example Configurations

### Standard XGBoost Evaluation Output

```bash
export SCORE_FIELD="prob_class_1"
export N_BINS="1000"
export ACCURACY="1e-5"
```

**Use Case**: Standard binary classification with XGBoost model evaluation output. Uses default built-in calibration dictionary for general-purpose percentile mapping.

### LightGBM Multi-Class Prediction

```bash
export SCORE_FIELD="prob_class_2"  # Target class 2
export N_BINS="1000"
export ACCURACY="1e-5"
```

**Use Case**: Multi-class classification focusing on specific class probability (class 2). Calibrates one class at a time.

### PyTorch Deep Learning Model

```bash
export SCORE_FIELD="confidence_score"
export N_BINS="1000"
export ACCURACY="1e-5"
```

**Use Case**: Deep learning model with custom score field name. Same calibration logic applies regardless of model type.

### Custom Calibration Dictionary

```bash
export SCORE_FIELD="prob_class_1"
export N_BINS="500"
export ACCURACY="1e-6"
```

**Directory Structure**:
```
/opt/ml/code/calibration/
└── standard_calibration_dictionary.json
```

**Use Case**: Domain-specific calibration with custom percentile breakpoints. Useful when default dictionary doesn't match target application (e.g., fraud detection with different precision requirements).

### High-Precision Calibration

```bash
export SCORE_FIELD="risk_score"
export N_BINS="2000"
export ACCURACY="1e-7"
```

**Use Case**: Applications requiring very fine-grained percentile distinctions (e.g., credit risk scoring). Increased accuracy parameter ensures precise interpolation.

## Integration Patterns

### Upstream Integration (Model Evaluation)

```
XGBoostModelEval / LightGBMModelEval / PyTorchModelEval
   ↓ (outputs: eval_predictions.csv with prob_class_1)
PercentileModelCalibration
   ↓ (outputs: percentile_score.pkl, calibrated_data.csv)
```

**Data Flow**:
1. Model evaluation produces predictions with probability scores
2. Percentile calibration loads predictions
3. ROC curve analysis generates percentile mapping
4. Calibrated scores added to output dataset

### Upstream Integration (Model Inference)

```
XGBoostModelInference / LightGBMModelInference / PyTorchModelInference
   ↓ (outputs: predictions.csv/parquet with scores)
PercentileModelCalibration
   ↓ (outputs: percentile_score.pkl for deployment)
```

**Data Flow**:
1. Model inference produces raw prediction scores
2. Percentile calibration creates mapping artifact
3. Mapping deployed with model for consistent scoring

### Downstream Integration (MIMS Deployment)

```
PercentileModelCalibration
   ↓ (outputs: percentile_score.pkl)
MIMSPackaging
   ↓ (includes: model + percentile mapping)
MIMSRegistration
   ↓ (deploys: inference endpoint with calibration)
```

**Deployment Flow**:
1. Percentile mapping artifact created during training
2. Packaging bundles model + mapping
3. Inference endpoint applies mapping to raw scores
4. API returns calibrated percentiles to clients

### Complete Pipeline Example

```
DataLoading
   ↓
Preprocessing
   ↓
ModelTraining
   ↓
ModelEvaluation (produces prob_class_1)
   ↓
PercentileModelCalibration
   ├─→ percentile_score.pkl (for deployment)
   └─→ calibrated_data.csv (for analysis)
```

**End-to-End Workflow**:
1. Train model on prepared data
2. Evaluate on held-out test set
3. Calibrate scores to percentiles
4. Deploy model + calibration mapping
5. Monitor percentile stability over time

## Troubleshooting

### Issue: "Score field not found in data"

**Symptom**: `ValueError: Score field 'prob_class_X' not found in data columns`

**Common Causes**:
1. Incorrect `SCORE_FIELD` environment variable
2. Mismatch between upstream model output and configuration
3. Multi-class model with different class naming

**Solution**:
1. Check upstream model evaluation/inference output columns
2. Update `SCORE_FIELD` to match actual column name
3. For multi-class: Use `prob_class_0`, `prob_class_1`, etc.

**Example**:
```bash
# Check actual columns in upstream output
head -1 /opt/ml/processing/input/eval_data/*.csv

# Update configuration
export SCORE_FIELD="confidence_score"  # Match actual column
```

### Issue: "All scores are essentially constant"

**Symptom**: `ValueError: All scores are essentially constant (std=X.XXe-YY), cannot perform calibration`

**Common Causes**:
1. Model failed to train properly (predicts same value for all inputs)
2. Data corruption or preprocessing error
3. Incorrect score field selected (e.g., using ID column instead of probability)

**Solution**:
1. Verify upstream model training succeeded
2. Check model evaluation metrics for reasonableness
3. Inspect score distribution: `df['prob_class_1'].describe()`
4. Retrain model if necessary

### Issue: "Multiple XGBoost output files found"

**Symptom**: Warning message about multiple files, script uses first one

**Common Causes**:
1. Multiple evaluation runs in same directory
2. Both evaluation and inference outputs present
3. Leftover files from previous runs

**Solution**:
1. Clean input directory before calibration
2. Use explicit file selection if needed
3. Review priority order in documentation
4. Consider: Does your pipeline need both eval and inference outputs?

**Prevention**:
```bash
# Clean input directory
rm /opt/ml/processing/input/eval_data/*

# Run single upstream step
# Then run calibration
```

### Issue: "Calibration produces unexpected percentiles"

**Symptom**: Calibrated percentiles don't match expectations (e.g., median score → 90th percentile)

**Common Causes**:
1. Calibration dictionary not matched to use case
2. Skewed evaluation dataset (not representative)
3. Score distribution shift between training and calibration

**Solution**:
1. Review calibration dictionary mappings
2. Verify evaluation data is representative
3. Check score distribution statistics in metrics
4. Consider using custom calibration dictionary
5. Compare with probability calibration results

**Diagnostic Commands**:
```python
# Examine calibration metrics
import json
with open('metrics_output/calibration_metrics.json') as f:
    metrics = json.load(f)
    print(metrics['score_statistics'])
    print(metrics['calibration_range'])
```

### Issue: "Out of memory errors with large datasets"

**Symptom**: Script crashes with `MemoryError` or system OOM killer

**Common Causes**:
1. Dataset too large for available memory (> 5M records with many columns)
2. Multiple DataFrame copies in memory
3. Inefficient format (CSV vs Parquet)

**Solution**:
1. Switch to Parquet format for better memory efficiency
2. Remove unnecessary columns from input data
3. Increase instance memory
4. Process in batches if needed (requires code modification)

**Memory Estimates**:
- 1M records × 50 columns: ~2-3 GB peak memory
- 10M records × 50 columns: ~20-30 GB peak memory

## References

### Related Scripts

- [`model_calibration.py`](model_calibration_script.md): Probability calibration using GAM, Isotonic Regression, and Platt Scaling methods

### Related Documentation

- **Step Builder**: No separate step builder documentation (uses general Processing pattern)
- **Contract**: [`src/cursus/steps/contracts/percentile_model_calibration_contract.py`](../../src/cursus/steps/contracts/percentile_model_calibration_contract.py)
- **Step Specification**: Defined in step registry as `PercentileModelCalibration`

### Related Design Documents

- **[Percentile Model Calibration Design](../1_design/percentile_model_calibration_design.md)**: Design document for percentile calibration approach and ROC curve methodology

### External References

- [scikit-learn ROC Curve Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html): Documentation for ROC curve computation
- [Percentile vs Probability Calibration](https://en.wikipedia.org/wiki/Calibration_(statistics)): Comparison of calibration approaches
- [ROC Analysis](https://en.wikipedia.org/wiki/Receiver_operating_characteristic): Background on ROC curves and their applications
