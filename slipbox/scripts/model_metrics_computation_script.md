---
tags:
  - code
  - processing_script
  - model_evaluation
  - metrics_computation
keywords:
  - model metrics
  - performance evaluation
  - ROC curve
  - precision recall
  - dollar recall
  - count recall
  - model comparison
  - A/B testing
  - binary classification
  - multiclass classification
topics:
  - model evaluation
  - performance metrics
  - business impact analysis
  - model comparison
language: python
date of note: 2025-11-18
---

# Model Metrics Computation Script Documentation

## Overview

The `model_metrics_computation.py` script computes comprehensive model performance metrics from prediction outputs, enabling thorough evaluation of binary and multiclass classifiers with both standard ML metrics and domain-specific business impact measures.

The script consumes prediction outputs from inference steps (XGBoost, LightGBM, PyTorch, or Bedrock), validates data structure, computes metrics across multiple dimensions, generates performance visualizations, and produces detailed reports. It supports comparison mode for A/B testing between model versions, providing statistical significance tests and side-by-side performance analysis.

Key capabilities:
- **Standard ML Metrics**: AUC-ROC, precision, recall, F1 score, average precision with threshold analysis
- **Domain-Specific Metrics**: Dollar recall (abuse amount caught) and count recall (abuse orders caught)
- **Comparison Mode**: Statistical comparison between new and previous model scores with significance tests
- **Comprehensive Visualizations**: ROC curves, PR curves, score distributions, threshold analysis, comparison plots
- **Flexible Input**: Auto-detects CSV, TSV, Parquet, or JSON prediction formats
- **Detailed Reporting**: JSON metrics, text summaries, actionable insights, and recommendations

## Purpose and Major Tasks

### Primary Purpose
Compute comprehensive performance metrics from model predictions to enable data-driven evaluation of classifier effectiveness, business impact, and comparative performance between model versions.

### Major Tasks

1. **Data Loading and Validation**: Load prediction files with automatic format detection and validate schema

2. **Standard Metrics Computation**: Calculate core ML metrics (AUC-ROC, F1, precision, recall) for binary and multiclass

3. **Domain Metrics Computation**: Calculate business impact metrics (dollar recall, count recall) for fraud detection

4. **Threshold Analysis**: Analyze performance across prediction thresholds to find optimal operating points

5. **Visualization Generation**: Create ROC curves, PR curves, score distributions, and threshold analysis plots

6. **Comparison Analysis**: Perform A/B testing between new and previous model versions with statistical tests

7. **Insight Generation**: Generate actionable insights and recommendations based on metric analysis

8. **Report Generation**: Produce comprehensive JSON and text reports with metrics, plots, and guidance

9. **Health Checking**: Create success markers and health check files for pipeline monitoring

## Script Contract

### Entry Point
```
model_metrics_computation.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `processed_data` | `/opt/ml/processing/input/eval_data` | Predictions from inference step |

**Input Directory Structure**:
```
/opt/ml/processing/input/eval_data/
├── predictions.{format} (csv, tsv, parquet, or json)
├── eval_predictions.csv (alternative from xgboost_model_eval)
└── _SUCCESS (optional marker)
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `metrics_output` | `/opt/ml/processing/output/metrics` | Computed metrics and reports |
| `plots_output` | `/opt/ml/processing/output/plots` | Generated visualization plots |

**Output Structure**:
```
/opt/ml/processing/output/metrics/
├── metrics.json                 # All computed metrics
├── metrics_summary.txt          # Human-readable summary
├── metrics_report.json          # Comprehensive report with insights
├── _SUCCESS                     # Success marker
└── _HEALTH                      # Health check file

/opt/ml/processing/output/plots/
├── roc_curve.jpg               # ROC curve
├── pr_curve.jpg                # Precision-Recall curve
├── score_distribution.jpg      # Score distributions by class
├── threshold_analysis.jpg      # Metrics vs threshold
├── comparison_roc_curves.jpg   # Model comparison (if enabled)
├── comparison_pr_curves.jpg    # PR comparison (if enabled)
├── score_scatter_plot.jpg      # Score correlation plot (if enabled)
└── score_distributions.jpg     # Detailed distributions (if enabled)
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ID_FIELD` | Name of ID column in predictions | `"customer_id"` |
| `LABEL_FIELD` | Name of label column in predictions | `"is_fraud"` |

### Optional Environment Variables

#### Core Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `AMOUNT_FIELD` | `None` | Column name for transaction amounts (enables dollar recall) |
| `INPUT_FORMAT` | `"auto"` | Input format: `auto`, `csv`, `tsv`, `parquet`, or `json` |

#### Metrics Configuration
| Variable | Default | Description | Range |
|----------|---------|-------------|-------|
| `COMPUTE_DOLLAR_RECALL` | `"true"` | Whether to compute dollar recall | `true` or `false` |
| `COMPUTE_COUNT_RECALL` | `"true"` | Whether to compute count recall | `true` or `false` |
| `DOLLAR_RECALL_FPR` | `"0.1"` | False positive rate for dollar recall threshold | 0.0 - 1.0 |
| `COUNT_RECALL_CUTOFF` | `"0.1"` | Top percentile cutoff for count recall | 0.0 - 1.0 |
| `GENERATE_PLOTS` | `"true"` | Whether to generate visualization plots | `true` or `false` |

#### Comparison Mode Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `COMPARISON_MODE` | `"false"` | Enable A/B testing comparison mode |
| `PREVIOUS_SCORE_FIELD` | `""` | Column name for previous model scores (required if comparison enabled) |
| `COMPARISON_METRICS` | `"all"` | Metrics to compute: `all` or `basic` |
| `STATISTICAL_TESTS` | `"true"` | Perform statistical significance tests |
| `COMPARISON_PLOTS` | `"true"` | Generate comparison visualizations |

### Job Arguments

| Argument | Type | Required | Description | Choices |
|----------|------|----------|-------------|---------|
| `--job_type` | `str` | Yes | Type of evaluation job | `evaluation`, `validation`, `testing`, `calibration` |

### Framework Dependencies

- **pandas** >= 1.3.0 (DataFrame operations)
- **numpy** >= 1.21.0 (Array operations)
- **scikit-learn** >= 1.0.0 (Metrics computation)
- **scipy** >= 1.7.0 (Statistical tests)
- **matplotlib** == 3.7.0 (Visualization)

## Input Data Structure

### Expected Input Format

**Predictions File** (from inference steps):
```
/opt/ml/processing/input/eval_data/
├── predictions.csv (or .tsv, .parquet, .json)
└── _SUCCESS (optional marker)
```

### Required Columns

The script requires columns specified by `ID_FIELD` and `LABEL_FIELD`, plus probability columns from inference.

**Typical Structure**:
- **ID Column**: Unique identifier (configurable via `ID_FIELD`, default: `"id"`)
- **Label Column**: Ground truth labels (configurable via `LABEL_FIELD`, default: `"label"`)
- **Probability Columns**: `prob_class_0`, `prob_class_1`, ... `prob_class_N`
- **Amount Column** (optional): Transaction amounts for dollar recall (configurable via `AMOUNT_FIELD`)
- **Previous Score Column** (optional): Previous model scores for comparison (configurable via `PREVIOUS_SCORE_FIELD`)

**Example Binary Classification Input**:
```csv
customer_id,is_fraud,prob_class_0,prob_class_1,amount
12345,0,0.85,0.15,100.50
67890,1,0.12,0.88,250.00
```

**Example Multiclass Input**:
```csv
id,label,prob_class_0,prob_class_1,prob_class_2
1,0,0.70,0.20,0.10
2,2,0.15,0.25,0.60
```

**Example with Comparison** (for A/B testing):
```csv
customer_id,is_fraud,prob_class_0,prob_class_1,prev_model_score,amount
12345,0,0.85,0.15,0.80,100.50
67890,1,0.12,0.88,0.15,250.00
```

### Supported Input Formats

- **CSV**: Comma-separated values (`.csv`)
- **TSV**: Tab-separated values (`.tsv`)
- **Parquet**: Apache Parquet binary format (`.parquet`)
- **JSON**: JSON format (`.json`)

**Auto-Detection**: Script automatically detects format from file extension

## Output Data Structure

### Metrics Output Files

**metrics.json**:
Complete metrics in JSON format with all computed values
```json
{
  "auc_roc": 0.8523,
  "average_precision": 0.7891,
  "f1_score": 0.7234,
  "count_recall": 0.8100,
  "dollar_recall": 0.8756,
  "optimal_threshold": 0.4321
}
```

**metrics_summary.txt**:
Human-readable text summary with key metrics formatted for easy reading

**metrics_report.json**:
Comprehensive report with metrics, insights, and recommendations
```json
{
  "timestamp": "2025-11-18T20:00:00",
  "data_summary": {
    "total_records": 10000,
    "prediction_columns": ["prob_class_0", "prob_class_1"],
    "label_distribution": {"0": 9500, "1": 500}
  },
  "standard_metrics": {...},
  "domain_metrics": {...},
  "visualizations": {...},
  "performance_insights": [
    "Excellent discrimination capability (AUC ≥ 0.9)",
    "Model is particularly effective at catching high-value abuse cases"
  ],
  "recommendations": [
    "Consider lowering decision threshold to catch more abuse cases"
  ]
}
```

### Visualization Outputs

**Standard Visualizations** (binary classification):
- `roc_curve.jpg`: ROC curve with AUC annotation
- `pr_curve.jpg`: Precision-Recall curve with AP annotation
- `score_distribution.jpg`: Score distributions separated by class
- `threshold_analysis.jpg`: Precision, recall, F1 vs threshold

**Comparison Visualizations** (when comparison mode enabled):
- `comparison_roc_curves.jpg`: Side-by-side ROC curves for new vs previous model
- `comparison_pr_curves.jpg`: Side-by-side PR curves with delta annotations
- `score_scatter_plot.jpg`: Correlation plot of new vs previous scores
- `score_distributions.jpg`: 4-panel comparison of score distributions

**Multiclass Visualizations**:
- `class_{i}_roc_curve.jpg`: Per-class ROC curves
- `class_{i}_pr_curve.jpg`: Per-class PR curves
- `multiclass_roc_curves.jpg`: Combined ROC curves for all classes

### Success Markers

**_SUCCESS**: Empty file indicating successful completion

**_HEALTH**: Health check file with timestamp
```
healthy: 2025-11-18T20:00:00.123456
```

## Key Functions and Tasks

### Data Loading Component

#### `detect_and_load_predictions(input_dir, preferred_format=None)`

**Purpose**: Auto-detect and load predictions file with intelligent format detection and graceful fallback

**Algorithm**:
```python
1. Determine formats to try (preferred format first if specified)
2. FOR each format in [parquet, csv, tsv, json]:
      file_path = join(input_dir, f"predictions.{format}")
      IF file exists:
         Detect actual format from extension
         Load using appropriate pandas method
         RETURN DataFrame
3. Try fallback: eval_predictions.csv
4. IF no file found:
      RAISE FileNotFoundError
```

**Returns**: `pd.DataFrame` with loaded predictions

**Format Detection**: Uses `_detect_file_format()` to identify actual format from extension

#### `validate_prediction_data(df, id_field, label_field, amount_field=None)`

**Purpose**: Validate prediction data schema and return comprehensive validation report

**Algorithm**:
```python
1. Initialize validation_report with is_valid=True
2. Check required columns:
   - ID field
   - Label field
   - Probability columns (prob_class_*)
3. Check optional amount field
4. Generate data summary:
   - Total records
   - Prediction columns
   - Label distribution
5. Collect errors and warnings
6. RETURN validation_report
```

**Returns**: `Dict[str, Any]` with validation results and data summary

### Standard Metrics Component

#### `compute_standard_metrics(y_true, y_prob, is_binary=True)`

**Purpose**: Compute comprehensive standard ML metrics matching xgboost_model_eval.py exactly

**Binary Classification Metrics**:
```python
1. Extract positive class probability (y_score = y_prob[:, 1])
2. Core metrics:
   - AUC-ROC
   - Average Precision
   - F1 Score
3. Precision-Recall curve analysis
4. Threshold-based metrics for [0.3, 0.5, 0.7]:
   - F1, Precision, Recall at each threshold
5. ROC curve analysis:
   - Find optimal threshold (max TPR - FPR)
6. RETURN metrics dictionary
```

**Multiclass Classification Metrics**:
```python
1. FOR each class i:
      Binarize labels (y_true == i)
      Compute per-class AUC, AP, F1
2. Compute micro and macro averages:
   - AUC-ROC (micro, macro)
   - Average Precision (micro, macro)
   - F1 Score (micro, macro)
3. Class distribution metrics
4. RETURN metrics dictionary
```

**Returns**: `Dict[str, float]` with all standard metrics

### Domain Metrics Component

#### `compute_domain_metrics(scores, labels, amounts, ...)`

**Purpose**: Compute domain-specific business impact metrics for fraud detection

**Algorithm**:
```python
1. IF compute_count_recall:
      count_recall = calculate_count_recall(
          scores, labels, cutoff=count_recall_cutoff
      )
      # Percentage of abuse orders caught at top percentile

2. IF compute_dollar_recall AND amounts available:
      dollar_recall = calculate_dollar_recall(
          scores, labels, amounts, fpr=dollar_recall_fpr
      )
      # Percentage of abuse dollar amount caught

3. Compute additional amount-based metrics:
   - Total abuse amount
   - Average abuse amount
   - Amount ratio abuse to total

4. RETURN domain_metrics dictionary
```

**Count Recall Algorithm**:
```python
threshold = quantile(scores, 1 - cutoff)  # Top cutoff% threshold
abuse_total = count(labels == 1)
abuse_caught = count((labels == 1) AND (scores >= threshold))
count_recall = abuse_caught / abuse_total
```

**Dollar Recall Algorithm**:
```python
threshold = quantile(scores[labels==0], 1 - fpr)  # FPR threshold
abuse_amount_total = sum(amounts[labels == 1])
abuse_amount_caught = sum(amounts[(labels==1) AND (scores>threshold)])
dollar_recall = abuse_amount_caught / abuse_amount_total
```

**Returns**: `Dict[str, float]` with domain metrics

### Visualization Component

#### `generate_performance_visualizations(y_true, y_prob, metrics, output_dir, is_binary=True)`

**Purpose**: Generate comprehensive performance visualizations matching xgboost_model_eval.py

**Binary Classification Plots**:
```python
1. ROC Curve:
   - Plot TPR vs FPR
   - Add diagonal (random baseline)
   - Annotate with AUC score

2. Precision-Recall Curve:
   - Plot precision vs recall
   - Annotate with AP score

3. Score Distribution:
   - Histogram of scores by class
   - Separate distributions for legitimate vs abuse
   - Density normalization

4. Threshold Analysis:
   - Plot F1, precision, recall vs threshold
   - Mark optimal threshold
   - Show trade-offs across thresholds
```

**Multiclass Plots**:
```python
1. Per-class ROC and PR curves
2. Combined multiclass ROC curves
3. Class-specific score distributions
```

**Returns**: `Dict[str, str]` mapping plot name to file path

#### `plot_and_save_roc_curve(y_true, y_score, output_dir, prefix="")`

**Purpose**: Plot and save ROC curve as JPG (exact match with xgboost_model_eval.py)

**Algorithm**:
```python
1. Compute ROC curve: fpr, tpr, thresholds = roc_curve(y_true, y_score)
2. Compute AUC: auc = roc_auc_score(y_true, y_score)
3. Create plot:
   - Plot fpr vs tpr
   - Add random baseline (diagonal)
   - Add legend with AUC
4. Save as JPG with 300 DPI
5. RETURN file path
```

### Comparison Mode Component

#### `compute_comparison_metrics(y_true, y_new_score, y_prev_score, is_binary=True)`

**Purpose**: Compute comparison metrics between new and previous model scores

**Algorithm**:
```python
1. Correlation analysis:
   - Pearson correlation coefficient
   - Spearman rank correlation
   - P-values for significance

2. Performance comparison:
   - New model AUC vs Previous model AUC
   - AUC delta and lift percent
   - Average Precision comparison
   - F1 score comparison at multiple thresholds

3. Score distribution comparison:
   - Mean and std of scores
   - Score mean delta

4. Agreement metrics:
   - Prediction agreement at different thresholds
   - Percentage of samples with same prediction

5. RETURN comparison_metrics dictionary
```

**Returns**: `Dict[str, float]` with comparison metrics

#### `perform_statistical_tests(y_true, y_new_score, y_prev_score, is_binary=True)`

**Purpose**: Perform statistical significance tests comparing model performances

**Algorithm**:
```python
1. McNemar's Test (binary classification):
   - Create contingency table
   - Compute test statistic
   - Calculate p-value
   - Determine significance (p < 0.05)

2. Paired t-test:
   - Test if score differences are significant
   - Calculate t-statistic and p-value

3. Wilcoxon signed-rank test:
   - Non-parametric alternative to t-test
   - Compute test statistic and p-value

4. RETURN test_results dictionary
```

**Returns**: `Dict[str, float]` with statistical test results

### Insight Generation Component

#### `generate_performance_insights(metrics)`

**Purpose**: Generate actionable performance insights based on metrics

**Algorithm**:
```python
1. AUC analysis:
   - Excellent (≥ 0.9), Good (≥ 0.8), Fair (≥ 0.7), Poor (< 0.7)

2. Dollar vs Count recall comparison:
   - IF dollar_recall > count_recall * 1.2:
        "Effective at catching high-value abuse cases"
   - ELIF count_recall > dollar_recall * 1.2:
        "Catches many cases but may miss high-value ones"

3. Threshold analysis:
   - IF optimal_threshold < 0.3:
        "Low threshold - check business tolerance"
   - ELIF optimal_threshold > 0.7:
        "High threshold - model is conservative"

4. RETURN list of insight strings
```

#### `generate_recommendations(metrics)`

**Purpose**: Generate actionable recommendations based on performance analysis

**Algorithm**:
```python
1. IF AUC < 0.75:
      Recommend feature engineering
      Recommend data quality investigation

2. IF dollar_recall < 0.6:
      Recommend focus on high-value cases
      Recommend amount-weighted loss functions

3. IF count_recall < 0.7:
      Recommend lowering threshold
      Recommend additional features

4. IF max_f1 < 0.6:
      Recommend class balancing techniques

5. RETURN list of recommendation strings
```

### Reporting Component

#### `generate_comprehensive_report(standard_metrics, domain_metrics, plot_paths, validation_report, output_dir)`

**Purpose**: Generate comprehensive metrics report with insights and recommendations

**Algorithm**:
```python
1. Combine all metrics
2. Create JSON report:
   - Timestamp
   - Data summary
   - Standard metrics
   - Domain metrics
   - Visualization paths
   - Performance insights
   - Recommendations
3. Generate text summary from JSON report
4. Save both JSON and text reports
5. RETURN report paths
```

**Returns**: `Dict[str, str]` with paths to JSON report and text summary

## Algorithms and Data Structures

### Algorithm 1: Count Recall Computation

**Problem**: Measure what percentage of abuse orders are caught when flagging top X% of orders by score

**Solution Strategy**:
1. Sort all orders by prediction score
2. Take top X% as flagged
3. Count how many actual abuse orders are in flagged set
4. Divide by total abuse orders

**Algorithm**:
```python
# Count Recall at cutoff percentile (e.g., 10%)
threshold = np.quantile(scores, 1 - cutoff)  # Top 10% threshold

abuse_order_total = len(labels[labels == 1])
abuse_order_above_threshold = len(
    labels[(labels == 1) & (scores >= threshold)]
)

order_count_recall = abuse_order_above_threshold / abuse_order_total
```

**Use Case**: Operational metric for teams with limited review capacity - "If we can only review top 10% of orders, what % of abuse do we catch?"

**Complexity**: O(n log n) for quantile calculation, O(n) overall

### Algorithm 2: Dollar Recall Computation

**Problem**: Measure what dollar amount of abuse is caught at a specific false positive rate threshold

**Solution Strategy**:
1. Find score threshold that gives desired FPR on legitimate orders
2. Calculate total abuse dollar amount caught above that threshold
3. Divide by total abuse dollar amount

**Algorithm**:
```python
# Dollar Recall at FPR threshold (e.g., 10%)
# Threshold is set based on LEGITIMATE orders only
threshold = np.quantile(scores[labels == 0], 1 - fpr)

abuse_amount_total = amounts[labels == 1].sum()
abuse_amount_above_threshold = amounts[
    (labels == 1) & (scores > threshold)
].sum()

dollar_recall = abuse_amount_above_threshold / abuse_amount_total
```

**Key Difference from Count Recall**: Threshold is based on legitimate orders (to control FPR) and measures dollar impact, not order count

**Use Case**: Business impact metric - "At 10% FPR, what % of fraud dollar amount do we prevent?"

**Complexity**: O(n log n) for quantile, O(n) overall

### Algorithm 3: McNemar's Test for Model Comparison

**Problem**: Determine if difference in prediction errors between two models is statistically significant

**Solution Strategy**:
1. Create 2x2 contingency table of predictions (both correct, both wrong, only one correct)
2. Apply McNemar's test focusing on discordant pairs
3. Calculate chi-square statistic and p-value

**Algorithm**:
```python
# Binarize predictions at threshold
new_pred = (y_new_score >= 0.5).astype(int)
prev_pred = (y_prev_score >= 0.5).astype(int)

# Create contingency table
correct_both = sum((new_pred == y_true) & (prev_pred == y_true))
new_correct_prev_wrong = sum((new_pred == y_true) & (prev_pred != y_true))
new_wrong_prev_correct = sum((new_pred != y_true) & (prev_pred == y_true))
wrong_both = sum((new_pred != y_true) & (prev_pred != y_true))

# McNemar's test statistic (with continuity correction)
n_discordant = new_correct_prev_wrong + new_wrong_prev_correct
IF n_discordant > 0:
    mcnemar_stat = (abs(new_correct_prev_wrong - new_wrong_prev_correct) - 1)^2 / n_discordant
    mcnemar_p_value = 1 - chi2.cdf(mcnemar_stat, 1)
ELSE:
    mcnemar_stat = 0.0
    mcnemar_p_value = 1.0

# Significant if p < 0.05
is_significant = (mcnemar_p_value < 0.05)
```

**Interpretation**:
- Low p-value (< 0.05): Models perform significantly differently
- High p-value: No significant difference in performance

**Complexity**: O(n) for contingency table, O(1) for test statistic

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Data Loading | O(n×f) | O(n×f) | n = rows, f = features |
| Format Detection | O(1) | O(1) | File extension check |
| Data Validation | O(n×f) | O(1) | Column checks |
| Standard Metrics | O(n log n) | O(n) | Sorting for thresholds |
| Domain Metrics | O(n log n) | O(n) | Quantile calculations |
| ROC/PR Curves | O(n log n) | O(n) | Sorting predictions |
| Comparison Metrics | O(n) | O(n) | Correlation, statistics |
| Visualization | O(n log n) | O(n) | Histogram binning |
| Report Generation | O(n) | O(m) | m = metrics count |
| **Total** | **O(n log n)** | **O(n×f)** | Dominated by sorting |

### Memory Requirements

**Peak Memory Usage**:
```
Total ≈ Input Data + Metrics Storage + Visualization Buffers

Example (100K records, 10 features):
- Input DataFrame: ~8 MB (100K × 10 × 8 bytes)
- Predictions array: ~1.6 MB (100K × 2 × 8 bytes)
- Metrics storage: < 1 MB (100s of metrics)
- Visualization buffers: ~10 MB (matplotlib figures)
Total: ~20 MB
```

### Processing Time Estimates

| Dataset Size | Features | Visualizations | Time (typical) |
|--------------|----------|----------------|----------------|
| 1K records   | 10       | All            | ~2 seconds     |
| 10K records  | 10       | All            | ~5 seconds     |
| 100K records | 10       | All            | ~15 seconds    |
| 1M records   | 10       | All            | ~1-2 minutes   |
| 1M records   | 10       | Disabled       | ~30-45 seconds |

**Note**: Visualization generation accounts for 50-60% of processing time

## Error Handling

### Input Validation Errors

**Missing Required Columns**:
- **Cause**: ID field, label field, or probability columns not found
- **Handling**: Raises `ValueError` with detailed error message listing missing columns
- **User Action**: Verify upstream inference step outputs correct columns, check environment variable field names

**No Prediction Files Found**:
- **Cause**: No CSV, TSV, Parquet, or JSON files in input directory
- **Handling**: Raises `FileNotFoundError` with supported formats list
- **User Action**: Verify inference step completed successfully, check input path configuration

### Data Quality Errors

**Invalid Prediction Probabilities**:
- **Cause**: Probabilities outside [0, 1] range or not summing to 1
- **Handling**: Logs warning but continues processing
- **Impact**: May produce unreliable metrics

**Missing Labels**:
- **Cause**: NaN values in label column
- **Handling**: Pandas operations will fail with clear error
- **User Action**: Ensure evaluation data has valid labels

### Comparison Mode Errors

**Previous Score Field Not Found**:
- **Cause**: `COMPARISON_MODE=true` but `PREVIOUS_SCORE_FIELD` column doesn't exist
- **Handling**: Logs warning, disables comparison mode, continues with standard evaluation
- **User Action**: Verify previous model scores included in input data

**Empty Previous Score Field**:
- **Cause**: `PREVIOUS_SCORE_FIELD` environment variable is empty string
- **Handling**: Detects and disables comparison mode with warning log
- **User Action**: Provide valid column name or disable comparison mode

### Visualization Errors

**Matplotlib Subplot Creation Failure**:
- **Cause**: Backend issues in headless environments
- **Handling**: Multi-level fallback strategy, creates minimal error plot
- **Impact**: May produce simplified plots but script continues

**SciPy Correlation Computation Error**:
- **Cause**: Type incompatibility with scipy version
- **Handling**: Falls back to numpy correlation, logs warning
- **Impact**: P-values not available but correlation computed

### Success and Failure Markers

**Success Flow**:
```python
# After successful computation
success_path = join(output_metrics_dir, "_SUCCESS")
Path(success_path).touch()

health_path = join(output_metrics_dir, "_HEALTH")
with open(health_path, "w") as f:
    f.write(f"healthy: {datetime.now().isoformat()}")
```

**Failure Flow**:
```python
# On exception
failure_path = join(output_metrics_dir, "_FAILURE")
with open(failure_path, "w") as f:
    f.write(f"Error: {str(exception)}")
sys.exit(1)
```

## Best Practices

### For Production Deployments

1. **Enable All Metrics**: Use default settings to capture comprehensive evaluation
   ```bash
   export COMPUTE_DOLLAR_RECALL="true"
   export COMPUTE_COUNT_RECALL="true"
   export GENERATE_PLOTS="true"
   ```

2. **Use Parquet Input**: More efficient for large datasets
   ```bash
   # Inference step should output Parquet
   export INPUT_FORMAT="parquet"
   ```

3. **Configure Business Metrics**: Align FPR and cutoff with business requirements
   ```bash
   export DOLLAR_RECALL_FPR="0.05"  # 5% FPR for conservative threshold
   export COUNT_RECALL_CUTOFF="0.10"  # Top 10% review capacity
   ```

4. **Enable Comparison for A/B Testing**: Track model improvements over time
   ```bash
   export COMPARISON_MODE="true"
   export PREVIOUS_SCORE_FIELD="baseline_score"
   ```

### For Development and Testing

1. **Start with Standard Metrics**: Disable domain metrics initially
   ```bash
   export COMPUTE_DOLLAR_RECALL="false"
   export COMPUTE_COUNT_RECALL="false"
   ```

2. **Disable Plots for Faster Iteration**: Skip visualization during development
   ```bash
   export GENERATE_PLOTS="false"
   ```

3. **Use Small Samples**: Test with subset before full dataset
   ```bash
   # Generate sample predictions for testing
   head -n 1001 predictions.csv > test_predictions.csv
   ```

4. **Check Metrics JSON**: Verify output structure
   ```bash
   cat /opt/ml/processing/output/metrics/metrics.json | jq .
   ```

### For Performance Optimization

1. **Disable Unnecessary Visualizations**: Significant time savings
   ```bash
   export GENERATE_PLOTS="false"  # Saves 50-60% of processing time
   ```

2. **Use Parquet Format**: Faster I/O than CSV
   ```bash
   export INPUT_FORMAT="parquet"
   ```

3. **Minimize Comparison Tests**: Only compute what's needed
   ```bash
   export COMPARISON_MODE="true"
   export STATISTICAL_TESTS="false"  # Skip statistical tests if not needed
   export COMPARISON_PLOTS="false"   # Skip comparison plots if not needed
   ```

4. **Batch Processing**: For very large datasets, consider chunking upstream

## Example Configurations

### Example 1: Standard Binary Classification Evaluation
```bash
export ID_FIELD="customer_id"
export LABEL_FIELD="is_fraud"
export AMOUNT_FIELD="transaction_amount"
export COMPUTE_DOLLAR_RECALL="true"
export COMPUTE_COUNT_RECALL="true"
export GENERATE_PLOTS="true"

python model_metrics_computation.py --job_type evaluation
```

**Use Case**: Standard fraud detection model evaluation with business impact metrics

**Expected Output**: All standard metrics, dollar/count recall, 4 visualization plots

### Example 2: A/B Testing with Model Comparison
```bash
export ID_FIELD="order_id"
export LABEL_FIELD="label"
export COMPARISON_MODE="true"
export PREVIOUS_SCORE_FIELD="baseline_model_score"
export STATISTICAL_TESTS="true"
export COMPARISON_PLOTS="true"

python model_metrics_computation.py --job_type evaluation
```

**Use Case**: Compare new model against baseline for A/B testing decision

**Expected Output**: All standard metrics plus comparison metrics, statistical tests, comparison visualizations

### Example 3: Fast Development Mode
```bash
export ID_FIELD="id"
export LABEL_FIELD="label"
export COMPUTE_DOLLAR_RECALL="false"
export COMPUTE_COUNT_RECALL="false"
export GENERATE_PLOTS="false"

python model_metrics_computation.py --job_type validation
```

**Use Case**: Quick validation during development iterations

**Expected Output**: Standard metrics only, no visualizations, ~2x faster execution

### Example 4: Multiclass Classification Evaluation
```bash
export ID_FIELD="sample_id"
export LABEL_FIELD="category"
export INPUT_FORMAT="parquet"
export GENERATE_PLOTS="true"

python model_metrics_computation.py --job_type evaluation
```

**Use Case**: Multiclass classification with 3+ classes

**Expected Output**: Per-class and macro/micro averaged metrics, multiclass visualizations

## Integration Patterns

### Upstream Integration (From Inference)

```
XGBoostModelInference / LightGBMModelInference / PyTorchModelInference
   ↓ (outputs: predictions.csv with ID + label + prob_class_*)
ModelMetricsComputation
   ↓ (outputs: metrics.json, plots, reports)
```

**Inference Output → Metrics Input**:
- Prediction probabilities: `prob_class_0`, `prob_class_1`, ...
- Ground truth labels: Configurable via `LABEL_FIELD`
- Optional: Transaction amounts for dollar recall
- Optional: Previous model scores for comparison

### Downstream Integration (Reporting)

```
ModelMetricsComputation
   ↓ (outputs: metrics.json + visualizations)
ModelWikiGenerator
   ↓ (outputs: comprehensive model documentation)
```

**OR**

```
ModelMetricsComputation
   ↓ (outputs: metrics.json)
Business Dashboard / Monitoring System
   ↓ (displays: model performance tracking)
```

### Complete Pipeline Flow

```
TabularPreprocessing → XGBoostTraining → XGBoostModelInference →
ModelMetricsComputation → ModelWikiGenerator → Package → Registration
```

**Key Integration Points**:
1. **Inference → Metrics**: Predictions with probabilities
2. **Metrics → Wiki**: Performance metrics for documentation
3. **Metrics → Monitoring**: Automated performance tracking
4. **Metrics → Decision**: A/B test results for model selection

### Modular Design Benefits

**Separation of Concerns**:
- Inference step: Generate predictions only
- Metrics step: Compute comprehensive evaluation
- Wiki step: Generate documentation
- Each step can be run independently

**Flexibility**:
- Compute metrics on cached predictions multiple times
- Try different thresholds without re-running inference
- Generate additional visualizations on demand
- Compare multiple models against same baseline

**Reproducibility**:
- Metrics computed from frozen prediction outputs
- Consistent evaluation across model versions
- Auditable performance tracking

## Troubleshooting

### Issue: Missing Probability Columns

**Symptom**: `No prediction probability columns found`

**Common Causes**:
1. Inference step didn't output probability columns
2. Column names don't match expected pattern (`prob_class_*`)
3. Using raw predictions instead of probabilities

**Solution**:
1. Verify inference step outputs probability columns:
   ```bash
   head -n 2 predictions.csv
   # Should show: id, label, prob_class_0, prob_class_1, ...
   ```
2. Check inference step configuration
3. If using different column names, update inference step to use standard naming

### Issue: Dollar Recall Not Computed

**Symptom**: `dollar_recall` missing from metrics

**Common Causes**:
1. `AMOUNT_FIELD` not specified
2. Amount column doesn't exist in input data
3. `COMPUTE_DOLLAR_RECALL` set to `"false"`

**Solution**:
1. Set amount field environment variable:
   ```bash
   export AMOUNT_FIELD="transaction_amount"
   ```
2. Verify column exists in predictions:
   ```bash
   head -n 2 predictions.csv | grep -o "transaction_amount"
   ```
3. Enable dollar recall computation:
   ```bash
   export COMPUTE_DOLLAR_RECALL="true"
   ```

### Issue: Comparison Mode Not Working

**Symptom**: No comparison metrics generated despite `COMPARISON_MODE=true`

**Common Causes**:
1. `PREVIOUS_SCORE_FIELD` not set or empty
2. Previous score column doesn't exist in data
3. Script automatically disabled comparison mode

**Solution**:
1. Check logs for warning message about disabled comparison mode
2. Set previous score field:
   ```bash
   export PREVIOUS_SCORE_FIELD="baseline_model_prob_class_1"
   ```
3. Verify column exists:
   ```bash
   head -n 2 predictions.csv | grep -o "baseline_model_prob_class_1"
   ```

### Issue: Matplotlib Visualization Errors

**Symptom**: Plots missing or error messages about matplotlib backend

**Common Causes**:
1. Headless environment without display
2. Matplotlib backend not configured
3. Memory issues with large datasets

**Solution**:
1. Script handles this automatically with fallback
2. Check generated plots:
   ```bash
   ls -lh /opt/ml/processing/output/plots/
   ```
3. If plots are minimal error plots, check logs for specific error
4. Try disabling plots and focus on metrics:
   ```bash
   export GENERATE_PLOTS="false"
   ```

### Issue: Low Performance Scores

**Symptom**: All metrics show poor performance (AUC < 0.6)

**Common Causes**:
1. Model training issues
2. Label leakage in features
3. Incorrect label mapping
4. Data distribution shift

**Solution**:
1. Check data summary in metrics_report.json:
   ```bash
   cat metrics_report.json | jq .data_summary
   ```
2. Verify label distribution makes sense
3. Review training logs for issues
4. Check if test data is from same distribution as training
5. Inspect score distributions plot for anomalies

### Issue: Comparison Shows No Significant Difference

**Symptom**: McNemar's test shows `p_value > 0.05` despite visible AUC improvement

**Common Causes**:
1. Small sample size (insufficient statistical power)
2. Models make similar predictions despite different scores
3. Improvements in specific segments not captured in overall test

**Solution**:
1. Check sample size in data summary
2. Review prediction agreement metrics
3. Examine score scatter plot for correlation patterns
4. Consider segment-specific analysis
5. Look at delta metrics even if not statistically significant

## References

### Related Scripts

- [`xgboost_model_inference.py`](xgboost_model_inference_script.md): XGBoost inference script that generates input predictions
- [`lightgbm_model_inference.py`](lightgbm_model_inference_script.md): LightGBM inference script for predictions
- [`xgboost_model_eval.py`](xgboost_model_eval_script.md): Combined training+evaluation script (legacy pattern)
- [`lightgbm_model_eval.py`](lightgbm_model_eval_script.md): Combined LightGBM evaluation script
- [`model_wiki_generator.py`](): Wiki generation script that consumes metrics output
- [`tabular_preprocess.py`](tabular_preprocess_script.md): Preprocessing script in upstream pipeline

### Related Documentation

- **Contract**: [`src/cursus/steps/contracts/model_metrics_computation_contract.py`](../../src/cursus/steps/contracts/model_metrics_computation_contract.py)
- **Step Builder**: Documented in `slipbox/steps/` (if exists)
- **Step Specification**: Part of ModelMetricsComputation step implementation

### Related Design Documents

- **[Model Metrics Computation Design](../1_design/model_metrics_computation_design.md)**: Comprehensive design document (if exists)
- **[Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md)**: Format detection and handling patterns used in this script

### External References

- [scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html): Official documentation for metrics functions
- [ROC Analysis](https://en.wikipedia.org/wiki/Receiver_operating_characteristic): Understanding ROC curves and AUC
- [Precision-Recall Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html): Understanding PR curves
- [McNemar's Test](https://en.wikipedia.org/wiki/McNemar%27s_test): Statistical test for paired nominal data
- [Statistical Hypothesis Testing](https://docs.scipy.org/doc/scipy/reference/stats.html): SciPy statistical tests documentation
