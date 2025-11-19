---
tags:
  - code
  - processing_script
  - temporal_features
  - feature_engineering
  - time_series_features
keywords:
  - temporal feature engineering
  - time window aggregations
  - rolling statistics
  - lag features
  - exponential smoothing
  - behavioral features
  - feature quality control
  - fraud detection features
topics:
  - temporal feature extraction
  - feature engineering
  - time series ML
  - fraud detection
language: python
date of note: 2025-11-18
---

# Temporal Feature Engineering Script Documentation

## Overview

The `temporal_feature_engineering.py` script extracts comprehensive temporal features from normalized sequence data produced by `temporal_sequence_normalization.py`. It combines two complementary feature extraction approaches: **generic temporal features** (statistical, temporal patterns, behavioral) and **time window aggregations** (rolling windows, lag features, exponential smoothing). The script includes robust feature quality control with validation, correlation analysis, and automated recommendations.

Designed specifically for fraud detection and temporal modeling, the script transforms fixed-length normalized sequences into rich feature representations suitable for machine learning model consumption. It processes attention masks to handle padded sequences correctly, computes multi-scale temporal patterns, and provides comprehensive quality metrics for production deployment.

Key capabilities:
- **Generic temporal features**: Statistical summaries, temporal patterns, behavioral indicators
- **Time window aggregations**: Multi-scale rolling statistics, historical lag features, exponential smoothing
- **Feature quality control**: Missing value analysis, correlation detection, variance analysis, outlier detection
- **Multi-format support**: Load from and save to numpy, parquet, or CSV formats
- **Attention mask processing**: Correctly handles padded sequences using attention masks
- **Production-ready**: Comprehensive validation, quality scoring, and automated recommendations

## Purpose and Major Tasks

### Primary Purpose
Extract comprehensive temporal features from normalized sequence data, combining generic temporal features with time window aggregations to create rich feature representations for machine learning models, with integrated quality control and validation.

### Major Tasks

1. **Input Loading**: Load normalized sequences from TemporalSequenceNormalization output

2. **Input Validation**: Validate structure and consistency of normalized sequence data

3. **Generic Feature Extraction**: Extract statistical, temporal pattern, and behavioral features

4. **Window Aggregations**: Compute multi-scale rolling statistics, lag features, and exponential smoothing

5. **Feature Combination**: Merge generic and window features into unified feature tensor

6. **Quality Validation**: Perform comprehensive feature quality assessment

7. **Quality Scoring**: Compute overall feature quality metrics

8. **Recommendation Generation**: Generate actionable feature selection recommendations

9. **Metadata Creation**: Create detailed feature metadata with configurations

10. **Output Saving**: Save feature tensors with metadata and quality reports

## Script Contract

### Entry Point
```
temporal_feature_engineering.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `normalized_sequences` | `/opt/ml/processing/input/normalized_sequences` | Normalized sequences from TemporalSequenceNormalization |

**Input Structure**:
```
/opt/ml/processing/input/normalized_sequences/
├── categorical.npy (or .parquet, .csv)
├── numerical.npy
├── categorical_attention_mask.npy (optional)
├── numerical_attention_mask.npy (optional)
└── metadata.json
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `temporal_feature_tensors` | `/opt/ml/processing/output` | Temporal features with metadata and quality report |

**Output Structure**:
```
/opt/ml/processing/output/
├── features.npy (or .parquet, .csv)
├── feature_names.json
├── feature_metadata.json
└── quality_report.json
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SEQUENCE_GROUPING_FIELD` | Entity field for grouping | `"customerId"` |
| `TIMESTAMP_FIELD` | Temporal field name | `"orderDate"` |
| `VALUE_FIELDS` | JSON array of numerical fields | `'["transactionAmount", "merchantRiskScore"]'` |

### Optional Environment Variables

#### Feature Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `FEATURE_TYPES` | `["statistical", "temporal", "behavioral"]` | Feature types to extract |
| `CATEGORICAL_FIELDS` | `["merchantCategory", "paymentMethod"]` | Categorical fields for features |
| `WINDOW_SIZES` | `[7, 14, 30, 90]` | Time window sizes for aggregations |
| `AGGREGATION_FUNCTIONS` | `["mean", "sum", "std", "min", "max", "count"]` | Aggregation functions |
| `LAG_FEATURES` | `[1, 7, 14, 30]` | Lag periods for historical features |

#### Processing Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `EXPONENTIAL_SMOOTHING_ALPHA` | `"0.3"` | Alpha for exponential smoothing |
| `TIME_UNIT` | `"days"` | Time unit: `days` or `hours` |
| `INPUT_FORMAT` | `"numpy"` | Input format: `numpy`, `parquet`, `csv` |
| `OUTPUT_FORMAT` | `"numpy"` | Output format: `numpy`, `parquet`, `csv` |

#### Quality Control Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_VALIDATION` | `"true"` | Enable feature quality validation |
| `MISSING_VALUE_THRESHOLD` | `"0.95"` | Threshold for high missing values |
| `CORRELATION_THRESHOLD` | `"0.99"` | Threshold for high correlation |
| `VARIANCE_THRESHOLD` | `"0.01"` | Threshold for low variance |
| `OUTLIER_DETECTION` | `"true"` | Enable outlier detection |

#### Distributed Processing
| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_DISTRIBUTED_PROCESSING` | `"false"` | Enable chunked processing |
| `CHUNK_SIZE` | `"5000"` | Chunk size for processing |
| `MAX_WORKERS` | `"auto"` | Number of parallel workers |
| `FEATURE_PARALLELISM` | `"true"` | Parallel feature computation |
| `CACHE_INTERMEDIATE` | `"true"` | Cache intermediate results |

### Job Arguments

| Argument | Type | Required | Description | Choices |
|----------|------|----------|-------------|---------|
| `--job_type` | `str` | Yes | Processing mode for data splits | `training`, `validation`, `testing`, `calibration` |

### Framework Dependencies

- **pandas** >= 1.3.0 (DataFrame operations)
- **numpy** >= 1.21.0 (Array operations)
- **scikit-learn** >= 1.0.0 (Feature utilities)
- **scipy** >= 1.7.0 (Statistical functions)

## Input Data Structure

### Expected Input Format

Normalized sequences from `temporal_sequence_normalization.py`:

```
/opt/ml/processing/input/normalized_sequences/
├── categorical.npy          # Shape: (batch_size, seq_len, num_cat_features)
├── numerical.npy            # Shape: (batch_size, seq_len, num_num_features+1)
├── categorical_attention_mask.npy  # Shape: (batch_size, seq_len)
├── numerical_attention_mask.npy    # Shape: (batch_size, seq_len)
└── metadata.json
```

### Metadata Structure

```json
{
  "sequence_length": 51,
  "sequence_separator": "~",
  "temporal_field": "orderDate",
  "entity_id_field": "customerId",
  "output_format": "numpy",
  "include_attention_masks": true,
  "shapes": {
    "categorical": [10000, 51, 3],
    "numerical": [10000, 51, 5]
  }
}
```

### Attention Mask Interpretation

- **1**: Valid sequence position with real data
- **0**: Padded position (should be ignored in feature computation)

## Output Data Structure

### Feature Tensor Output

**features.npy** (or .parquet, .csv):
- Shape: `(batch_size, total_feature_count)`
- Dtype: `float32`
- Contains: Combined generic + window aggregation features

### Feature Names Output

**feature_names.json**:
```json
[
  "generic_count_transactionAmount",
  "generic_mean_transactionAmount",
  "generic_std_transactionAmount",
  "window_rolling_7_mean_transactionAmount",
  "window_rolling_30_sum_transactionAmount",
  "window_lag_1_transactionAmount",
  ...
]
```

### Feature Metadata Output

**feature_metadata.json**:
```json
{
  "feature_count": 250,
  "entity_count": 10000,
  "output_format": "numpy",
  "feature_metadata": {
    "generic_feature_count": 120,
    "window_feature_count": 130,
    "total_feature_count": 250,
    "entity_count": 10000,
    "configuration": {
      "feature_types": ["statistical", "temporal", "behavioral"],
      "value_fields": ["transactionAmount", "merchantRiskScore"],
      "window_sizes": [7, 14, 30, 90],
      "aggregation_functions": ["mean", "sum", "std", "min", "max", "count"]
    }
  },
  "tensor_shapes": {
    "features": [10000, 250]
  }
}
```

### Quality Report Output

**quality_report.json**:
```json
{
  "validation_results": {
    "missing_values": {
      "missing_rates": {...},
      "problematic_features": [],
      "max_missing_rate": 0.02,
      "avg_missing_rate": 0.005
    },
    "correlations": {
      "high_correlation_pairs": [],
      "max_correlation": 0.87
    },
    "variance": {
      "low_variance_features": [],
      "min_variance": 0.15,
      "avg_variance": 12.5
    },
    "outliers": {...}
  },
  "quality_metrics": {
    "overall_score": 0.92
  },
  "recommendations": []
}
```

## Key Functions and Tasks

### Input Loading Component

#### `load_normalized_sequences(input_dir, input_format) -> Dict[str, np.ndarray]`

**Purpose**: Load normalized sequences from TemporalSequenceNormalization output

**Algorithm**:
```python
1. Validate input directory exists
2. Load metadata.json for configuration
3. Based on input_format:
   IF numpy:
      - Load .npy files for categorical, numerical
      - Load attention masks
   ELIF parquet:
      - Load .parquet files
      - Reshape based on metadata shapes
   ELIF csv:
      - Load .csv files
      - Reshape based on metadata shapes
4. Return dictionary with sequences and masks
```

**Returns**: Dictionary with categorical, numerical sequences and attention masks

#### `validate_input_data(normalized_data) -> None`

**Purpose**: Validate structure of normalized sequence data

**Validation Checks**:
- Required keys present (categorical, numerical)
- Batch sizes match across sequence types
- Sequence lengths match across sequence types
- Attention masks align with sequences

### Generic Feature Extraction Component

#### `GenericTemporalFeaturesOperation.process(normalized_data) -> Dict[str, np.ndarray]`

**Purpose**: Extract statistical, temporal, and behavioral features

**Algorithm**:
```python
FOR each entity in batch:
   entity_features = {}
   
   IF "statistical" in feature_types:
      - Extract count, sum, mean, std, min, max
      - Compute percentiles (25th, 50th, 75th)
      - Calculate skew, kurtosis, range, CV
      - Process categorical diversity metrics
   
   IF "temporal" in feature_types:
      - Analyze time deltas
      - Compute temporal span, frequency
      - Calculate interval regularity
   
   IF "behavioral" in feature_types:
      - Compute Gini coefficient for concentration
      - Calculate consistency scores
      - Analyze trend slopes
      - Compute volatility metrics
   
   Append entity_features to all_features

RETURN features_array, feature_names
```

**Returns**: Dictionary with features array and feature names list

#### `_extract_statistical_features(cat_seq, num_seq, masks) -> Dict[str, float]`

**Purpose**: Extract statistical summaries from sequences

**Features Extracted**:
- **Numerical**: count, sum, mean, std, min, max, p25, p50, p75, skew, kurtosis, range, CV
- **Categorical**: unique_count, diversity, mode_freq

**Complexity**: O(n×m) where n = sequence length, m = features

#### `_extract_temporal_patterns(num_seq, mask) -> Dict[str, float]`

**Purpose**: Extract temporal pattern features

**Features Extracted**:
- avg_time_delta, std_time_delta, min_time_delta, max_time_delta
- temporal_span, event_frequency
- interval_regularity

**Algorithm**:
```python
1. Extract time delta column (assumed -2, before padding)
2. Apply attention mask to get valid deltas
3. Compute time interval statistics
4. Calculate temporal span and frequency
5. Compute regularity: 1 / (1 + cv_of_deltas)
```

#### `_extract_behavioral_features(cat_seq, num_seq, masks) -> Dict[str, float]`

**Purpose**: Extract behavioral pattern features

**Features Extracted**:
- **Activity concentration**: Gini coefficient
- **Consistency score**: Based on coefficient of variation
- **Trend analysis**: Linear regression slopes
- **Volatility**: Standard deviation of returns

**Gini Coefficient Algorithm**:
```python
1. Sort values
2. Compute cumulative sum
3. Apply Gini formula:
   gini = (2 * Σ((i+1) * value[i])) / (n * Σ(values)) - (n+1)/n
4. Return max(0, gini)
```

### Time Window Aggregations Component

#### `TimeWindowAggregationsOperation.process(normalized_data) -> Dict[str, np.ndarray]`

**Purpose**: Compute multi-scale time window aggregations

**Algorithm**:
```python
FOR each entity in batch:
   entity_features = {}
   
   # Rolling window features
   FOR each window_size in window_sizes:
      FOR each agg_func in aggregation_functions:
         Compute rolling aggregation on recent window
   
   # Lag features
   FOR each lag in lag_features:
      Extract historical value at lag position
   
   # Exponential smoothing
   FOR each value_field:
      Compute EWMA and EWMA std
   
   Append entity_features to all_features

RETURN features_array, feature_names
```

**Returns**: Dictionary with window features and feature names

#### `_compute_rolling_features(num_seq, mask) -> Dict[str, float]`

**Purpose**: Compute rolling window aggregations

**Algorithm**:
```python
FOR each value_field:
   FOR each window_size:
      effective_window = min(window_size, len(valid_values))
      window_values = values[-effective_window:]  # Most recent
      
      FOR each agg_func:
         result = agg_func(window_values)
         feature_name = f"rolling_{window_size}_{agg_func}_{field_name}"
         features[feature_name] = result
```

**Supported Aggregations**: mean, sum, std, min, max, count

**Complexity**: O(w×f×a) where w = windows, f = fields, a = aggregations

#### `_compute_lag_features(num_seq, mask) -> Dict[str, float]`

**Purpose**: Compute lag features for historical values

**Algorithm**:
```python
FOR each value_field:
   FOR each lag in lag_features:
      IF lag < len(values):
         lag_value = values[-(lag+1)]  # lag=1 means previous
      ELSE:
         lag_value = 0.0  # Insufficient history
      
      features[f"lag_{lag}_{field_name}"] = lag_value
```

**Example**: `lag_1_transactionAmount` = previous transaction amount

#### `_compute_exponential_smoothing(num_seq, mask) -> Dict[str, float]`

**Purpose**: Compute exponential smoothing features

**Algorithm**:
```python
alpha = exponential_smoothing_alpha

FOR each value_field:
   # Initialize with first value
   ewm_values = [values[0]]
   
   # Compute EWMA
   FOR i in range(1, len(values)):
      ewm_val = alpha * values[i] + (1-alpha) * ewm_values[-1]
      ewm_values.append(ewm_val)
   
   # Compute EWMA standard deviation
   squared_diffs = [(values[i] - ewm_values[i])**2 for i in range(len(values))]
   ewm_var = squared_diffs[0]
   FOR i in range(1, len(squared_diffs)):
      ewm_var = alpha * squared_diffs[i] + (1-alpha) * ewm_var
   ewm_std = sqrt(ewm_var)
   
   features[f"exp_smooth_{field_name}"] = ewm_values[-1]
   features[f"exp_smooth_std_{field_name}"] = ewm_std
```

### Feature Quality Control Component

#### `FeatureQualityController.validate_features(features, feature_names) -> Dict[str, Any]`

**Purpose**: Comprehensive feature quality assessment

**Algorithm**:
```python
1. Convert to DataFrame for analysis
2. Analyze missing values
3. Analyze correlations
4. Analyze variance
5. Detect outliers (if enabled)
6. Generate feature selection recommendations
7. Compute overall quality score
8. Return quality report
```

**Returns**: Quality report with validation results, metrics, and recommendations

#### `_analyze_missing_values(df) -> Dict[str, Any]`

**Purpose**: Analyze missing value patterns

**Metrics**:
- missing_rates per feature
- problematic_features (> threshold)
- max_missing_rate, avg_missing_rate

#### `_analyze_correlations(df) -> Dict[str, Any]`

**Purpose**: Identify redundant features via correlation

**Algorithm**:
```python
1. Compute correlation matrix
2. Find pairs with |correlation| > threshold
3. Return high correlation pairs with values
```

#### `_analyze_variance(df) -> Dict[str, Any]`

**Purpose**: Identify low-variance features

**Metrics**:
- variances per feature
- low_variance_features (< threshold)
- min_variance, avg_variance

#### `_detect_outliers(df) -> Dict[str, Any]`

**Purpose**: Detect outliers using IQR method

**Algorithm**:
```python
FOR each feature:
   Q1 = 25th percentile
   Q3 = 75th percentile
   IQR = Q3 - Q1
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
   outliers = values outside [lower_bound, upper_bound]
   
   Store outlier_count, outlier_rate, bounds
```

#### `_compute_quality_score(validation_results) -> float`

**Purpose**: Compute overall feature quality score

**Algorithm**:
```python
score_components = []

# Missing value score (lower missing = higher score)
missing_score = 1 - avg_missing_rate
score_components.append(missing_score)

# Variance score (higher variance = higher score, capped at 1.0)
variance_score = min(1.0, avg_variance / 10.0)
score_components.append(variance_score)

# Correlation score (fewer high correlations = higher score)
correlation_score = max(0, 1 - high_corr_count / 10.0)
score_components.append(correlation_score)

RETURN mean(score_components)
```

**Score Range**: 0.0 to 1.0 (higher is better)

## Algorithms and Data Structures

### Algorithm 1: Gini Coefficient for Activity Concentration

**Problem**: Measure inequality/concentration in temporal activity patterns

**Formula**:
```
gini = (2 * Σ((i+1) * sorted_value[i])) / (n * Σ(sorted_values)) - (n+1)/n
```

**Use Case**: Identify users with concentrated vs. distributed activity patterns

**Interpretation**:
- 0.0: Perfect equality (uniform distribution)
- 1.0: Perfect inequality (all activity at one point)

### Algorithm 2: Exponential Weighted Moving Average (EWMA)

**Problem**: Track temporal trends with exponential decay

**Recursive Formula**:
```
EWMA[0] = value[0]
EWMA[t] = α * value[t] + (1-α) * EWMA[t-1]
```

**Properties**:
- α near 1: Fast adaptation to recent changes
- α near 0: Smooth, slow-changing trend
- Default α=0.3: Balanced responsiveness

### Algorithm 3: Feature Quality Scoring

**Components**:
1. **Missing Value Score**: `1 - avg_missing_rate`
2. **Variance Score**: `min(1.0, avg_variance / 10.0)`
3. **Correlation Score**: `max(0, 1 - high_corr_count / 10.0)`

**Overall Score**: Average of component scores

**Thresholds**:
- Score > 0.8: Excellent quality
- Score 0.6-0.8: Good quality
- Score 0.4-0.6: Fair quality (review recommended)
- Score < 0.4: Poor quality (action required)

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Input Loading | O(n×s) | O(n×s) | n=entities, s=seq_len |
| Generic Features | O(n×s×f) | O(n×g) | f=features, g=generic_count |
| Window Aggregations | O(n×w×f×a) | O(n×w×a) | w=windows, a=aggregations |
| Quality Validation | O(g²) | O(g²) | Correlation matrix |
| Output Saving | O(n×g) | O(n×g) | Writing features |
| **Total** | **O(n×(s×f + w×f×a + g²))** | **O(n×(s + g) + g²)** | Dominated by aggregations |

### Memory Requirements

**Peak Memory Usage**:
```
Total ≈ Input Sequences + Feature Matrices + Quality Analysis

Example (10K entities, 51 seq_len, 250 features):
- Input sequences: ~20 MB
- Feature matrices: ~10 MB
- Quality analysis: ~5 MB
- Temporary buffers: ~5 MB
Total: ~40 MB
```

### Processing Time Estimates

| Dataset Size | Seq Length | Features | Time (single-thread) |
|--------------|------------|----------|---------------------|
| 10K entities | 51         | 250      | ~10-20 seconds      |
| 100K entities| 51         | 250      | ~2-4 minutes        |
| 1M entities  | 51         | 250      | ~20-40 minutes      |

**Optimization**: Enable `FEATURE_PARALLELISM=true` for 2-3x speedup

## Error Handling

### Input Validation Errors

**Missing Required Sequences**:
- **Cause**: categorical or numerical sequences not in input
- **Handling**: Raises `RuntimeError` with missing keys list

**Inconsistent Shapes**:
- **Cause**: Batch sizes or sequence lengths don't match
- **Handling**: Raises `RuntimeError` with shape mismatch details

### Feature Extraction Errors

**Insufficient Data**:
- **Cause**: All values masked or empty after attention mask application
- **Handling**: Returns 0.0 for affected features, logs warning

**Numerical Errors**:
- **Cause**: Division by zero, NaN values
- **Handling**: Graceful fallback to 0.0, continues processing

### Quality Validation Warnings

**High Missing Values**:
- **Severity**: Warning
- **Action**: Logged in quality report, included in recommendations

**High Correlation**:
- **Severity**: Warning
- **Action**: Pairs logged, redundancy noted in recommendations

## Best Practices

### For Production Deployments

1. **Enable Quality Validation**: Catch feature quality issues
   ```bash
   export ENABLE_VALIDATION="true"
   ```

2. **Use Strict Thresholds**: Enforce quality standards
   ```bash
   export MISSING_VALUE_THRESHOLD="0.90"
   export CORRELATION_THRESHOLD="0.95"
   export VARIANCE_THRESHOLD="0.05"
   ```

3. **Match Input Format**: Use same format as normalization output
   ```bash
   export INPUT_FORMAT="numpy"  # Match normalization output
   export OUTPUT_FORMAT="numpy"
   ```

4. **Configure Time Windows**: Align with business logic
   ```bash
   export WINDOW_SIZES="[7, 14, 30, 90, 180]"  # 1 week to 6 months
   ```

### For Development

1. **Start with Basic Features**: Test with subset
   ```bash
   export FEATURE_TYPES='["statistical"]'
   export WINDOW_SIZES="[7, 30]"
   ```

2. **Use CSV Output**: Easier inspection
   ```bash
   export OUTPUT_FORMAT="csv"
   ```

3. **Disable Quality Validation**: Faster iteration
   ```bash
   export ENABLE_VALIDATION="false"
   ```

### For Performance Optimization

1. **Enable Feature Parallelism**: For large feature sets
   ```bash
   export FEATURE_PARALLELISM="true"
   export MAX_WORKERS="4"
   ```

2. **Cache Intermediate Results**: For repeated processing
   ```bash
   export CACHE_INTERMEDIATE="true"
   ```

3. **Use Chunked Processing**: For very large datasets
   ```bash
   export ENABLE_DISTRIBUTED_PROCESSING="true"
   export CHUNK_SIZE="10000"
   ```

## Example Configurations

### Example 1: Basic Fraud Detection Features
```bash
export SEQUENCE_GROUPING_FIELD="customerId"
export TIMESTAMP_FIELD="orderDate"
export VALUE_FIELDS='["transactionAmount", "merchantRiskScore"]'
export CATEGORICAL_FIELDS='["merchantCategory", "paymentMethod"]'
export FEATURE_TYPES='["statistical", "temporal", "behavioral"]'
export WINDOW_SIZES="[7, 14, 30, 90]"
export ENABLE_VALIDATION="true"

python temporal_feature_engineering.py --job_type training
```

**Use Case**: Standard fraud detection feature engineering

### Example 2: High-Frequency Trading Features
```bash
export SEQUENCE_GROUPING_FIELD="traderId"
export TIMESTAMP_FIELD="tradeTimestamp"
export VALUE_FIELDS='["tradeVolume", "tradePrice", "orderBookDepth"]'
export WINDOW_SIZES="[5, 10, 20, 50, 100]"  # Short windows
export LAG_FEATURES="[1, 2, 5, 10]"
export EXPONENTIAL_SMOOTHING_ALPHA="0.5"  # Fast adaptation
export TIME_UNIT="hours"

python temporal_feature_engineering.py --job_type training
```

**Use Case**: High-frequency trading pattern analysis

### Example 3: Large-Scale Processing
```bash
export FEATURE_TYPES='["statistical", "temporal"]'  # Skip behavioral
export WINDOW_SIZES="[7, 30]"  # Fewer windows
export ENABLE_DISTRIBUTED_PROCESSING="true"
export CHUNK_SIZE="50000"
export MAX_WORKERS="8"
export FEATURE_PARALLELISM="true"
export OUTPUT_FORMAT="parquet"  # More efficient

python temporal_feature_engineering.py --job_type training
```

**Use Case**: Processing millions of entities with performance optimization

## Integration Patterns

### Upstream Integration
```
TemporalSequenceNormalization
   ↓ (outputs: normalized sequences + attention masks)
TemporalFeatureEngineering
   ↓ (outputs: temporal feature tensors)
```

### Downstream Integration
```
TemporalFeatureEngineering
   ↓ (outputs: feature tensors)
[TabularPreprocessing] ← Merge with tabular features
   ↓
XGBoost/LightGBM/PyTorchTraining
```

### Complete Fraud Detection Pipeline
```
TabularPreprocessing → TemporalSequenceNormalization → 
TemporalFeatureEngineering → [Merge Features] → 
XGBoostTraining → ModelEvaluation → ModelRegistration
```

## Troubleshooting

### Issue: No Features Generated

**Symptom**: Empty feature tensor or 0 features

**Common Causes**:
1. All sequence values masked by attention masks
2. Insufficient valid data after masking
3. Configuration mismatch with input data

**Solution**:
1. Check attention masks align with sequences
2. Verify `VALUE_FIELDS` match input numerical columns
3. Review input data quality and validation

### Issue: High Missing Value Rates

**Symptom**: Quality report shows high missing values

**Common Causes**:
1. Sparse sequences with many padding
2. Attention masks not properly utilized
3. Invalid data in input sequences

**Solution**:
1. Review sequence normalization configuration
2. Check attention mask generation
3. Adjust `MISSING_VALUE_THRESHOLD` if appropriate

### Issue: Memory Overflow

**Symptom**: Out of memory error during processing

**Common Causes**:
1. Large window sizes with many aggregations
2. Too many entities processed at once
3. Quality validation on very large feature sets

**Solution**:
1. Enable distributed processing:
   ```bash
   export ENABLE_DISTRIBUTED_PROCESSING="true"
   export CHUNK_SIZE="5000"
   ```
2. Reduce window configurations
3. Use parquet output format (more memory efficient)

### Issue: Low Quality Score

**Symptom**: Quality score < 0.6 in quality report

**Common Causes**:
1. High correlation between features
2. Low variance in some features
3. High missing value rates

**Solution**:
1. Review quality report recommendations
2. Consider feature selection based on correlation analysis
3. Adjust configuration to reduce redundant features
4. Check input data quality

## References

### Related Scripts
- [`temporal_sequence_normalization.py`](temporal_sequence_normalization_script.md): Upstream normalization for sequence preparation
- [`tabular_preprocessing.py`](../steps/tabular_preprocessing_step.md): Parallel tabular feature processing

### Related Documentation
- **Contract**: `src/cursus/steps/contracts/temporal_feature_engineering_contract.py`
- **Step Specification**: Temporal feature engineering step specification

### Related Design Documents
- **[Temporal Feature Engineering Design](../1_design/temporal_feature_engineering_design.md)**: Complete design document with feature taxonomy, aggregation strategies, and quality control framework
- **[Temporal Sequence Normalization Design](../1_design/temporal_sequence_normalization_design.md)**: Upstream sequence normalization architecture

---

**Document Metadata**:
- **Author**: Cursus Framework Team
- **Last Updated**: 2025-11-18
- **Script Version**: 2025-11-18
- **Documentation Version**: 1.0
