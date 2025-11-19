---
tags:
  - code
  - processing_script
  - model_inference
  - xgboost
keywords:
  - xgboost inference
  - model inference
  - prediction generation
  - format preservation
  - multi-format output
  - pure inference
  - embedded preprocessing
topics:
  - model inference
  - xgboost predictions
  - modular ML pipelines
language: python
date of note: 2025-11-18
---

# XGBoost Model Inference Script Documentation

## Overview

The `xgboost_model_inference.py` script generates predictions from trained XGBoost models without computing evaluation metrics, enabling modular pipeline architectures where inference results can be cached, reused, and processed by different downstream components.

The script loads trained model artifacts (model weights, preprocessing parameters, feature configurations), applies the same preprocessing transformations used during training (risk table mapping, numerical imputation), and generates class probability predictions. It preserves the original input data structure while adding probability columns, supporting multiple output formats (CSV, TSV, Parquet, JSON) for flexible integration with downstream components.

Key capabilities:
- **Pure Inference**: Generates predictions without metrics computation for modular design
- **Model Artifact Loading**: Automatic extraction from model.tar.gz or direct loading
- **Embedded Preprocessing**: Self-contained risk table mapping and imputation processors
- **Format Preservation**: Maintains input format or converts to specified output format
- **Multi-Format Output**: Supports CSV, TSV, Parquet, and JSON with configurable orientations
- **Data Preservation**: Retains all original columns including ID, label, and metadata
- **Binary and Multiclass**: Handles both classification scenarios with consistent output

## Purpose and Major Tasks

### Primary Purpose
Generate class probability predictions from trained XGBoost models by loading model artifacts, applying preprocessing transformations, and producing structured prediction outputs that preserve original data for downstream processing.

### Major Tasks

1. **Model Artifact Loading**: Load and extract trained model, preprocessing parameters, and feature configurations

2. **Data Loading**: Load evaluation data with automatic format detection (CSV, TSV, Parquet)

3. **Risk Table Mapping**: Apply categorical feature transformations using trained risk tables

4. **Numerical Imputation**: Impute missing numerical values using trained imputation parameters

5. **Feature Preparation**: Ensure features match training configuration and format

6. **Prediction Generation**: Generate class probability predictions using XGBoost model

7. **Output Formatting**: Structure predictions with original data and probability columns

8. **Multi-Format Saving**: Save predictions in specified format (CSV, TSV, Parquet, JSON)

9. **Health Checking**: Create success and health markers for pipeline monitoring

## Script Contract

### Entry Point
```
xgboost_model_inference.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `model_input` | `/opt/ml/processing/input/model` | Trained XGBoost model artifacts |
| `processed_data` | `/opt/ml/processing/input/eval_data` | Evaluation data for inference |

**Model Input Structure**:
```
/opt/ml/processing/input/model/
├── model.tar.gz (or extracted files below)
├── xgboost_model.bst
├── risk_table_map.pkl
├── impute_dict.pkl
├── feature_columns.txt
└── hyperparameters.json
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `eval_output` | `/opt/ml/processing/output/eval` | Predictions with probabilities |

**Output Structure**:
```
/opt/ml/processing/output/eval/
├── predictions.{format}  # csv, tsv, parquet, or json
├── _SUCCESS              # Success marker
└── _HEALTH               # Health check file
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ID_FIELD` | Name of ID column in evaluation data | `"customer_id"` |
| `LABEL_FIELD` | Name of label column in evaluation data | `"is_fraud"` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_FORMAT` | `"csv"` | Output format: `csv`, `tsv`, `parquet`, or `json` |
| `JSON_ORIENT` | `"records"` | JSON orientation: `records`, `index`, `values`, `split`, `table` |

**JSON Orientation Options**:
- `records`: `[{column -> value}, ..., {column -> value}]` (default)
- `index`: `{index -> {column -> value}}`
- `values`: `[[row values], [row values], ...]`
- `split`: `{'index': [index], 'columns': [columns], 'data': [values]}`
- `table`: `{'schema': {schema}, 'data': [{row}, {row}, ...]}`

### Job Arguments

| Argument | Type | Required | Description | Choices |
|----------|------|----------|-------------|---------|
| `--job_type` | `str` | Yes | Type of inference job | `inference`, `validation`, `testing`, `calibration` |

### Framework Dependencies

- **pandas** >= 1.3.0 (DataFrame operations)
- **numpy** >= 1.21.0 (Array operations)
- **xgboost** >= 1.6.0 (Model inference)

## Input Data Structure

### Expected Input Format

**Evaluation Data**:
```
/opt/ml/processing/input/eval_data/
├── data.csv (or .tsv, .parquet)
└── _SUCCESS (optional marker)
```

### Required Columns

The script requires columns specified by `ID_FIELD` and `LABEL_FIELD`, plus features used during training.

**Typical Structure**:
- **ID Column**: Unique identifier for each record (configurable via `ID_FIELD`)
- **Label Column**: Ground truth labels for evaluation (configurable via `LABEL_FIELD`)
- **Feature Columns**: All features from `feature_columns.txt` in model artifacts

### Model Artifacts

**xgboost_model.bst**: Trained XGBoost booster model

**risk_table_map.pkl**: Dictionary mapping categorical feature names to risk tables
```python
{
    'feature_name': {
        'bins': {'category1': 0.75, 'category2': 0.45, ...},
        'default_bin': 0.5
    }
}
```

**impute_dict.pkl**: Dictionary mapping feature names to imputation values
```python
{
    'numerical_feature1': 0.0,
    'numerical_feature2': 42.5,
    ...
}
```

**feature_columns.txt**: Ordered list of feature names
```
# Feature columns in model training order
0,feature_1
1,feature_2
2,feature_3
...
```

**hyperparameters.json**: Model hyperparameters and metadata

### Supported Input Formats

- **CSV**: Comma-separated values (`.csv`)
- **TSV**: Tab-separated values (`.tsv`)
- **Parquet**: Apache Parquet binary format (`.parquet`)

## Output Data Structure

### Output File Structure

**predictions.{format}**:
- All original input columns preserved
- Added probability columns: `prob_class_0`, `prob_class_1`, ... `prob_class_N`
- Format determined by `OUTPUT_FORMAT` env var or input format

### Output Columns

**Original Columns** (preserved):
- ID column (specified by `ID_FIELD`)
- Label column (specified by `LABEL_FIELD`)
- All feature columns from input
- Any metadata columns from input

**Added Prediction Columns**:
- `prob_class_0`: Probability for class 0
- `prob_class_1`: Probability for class 1
- `prob_class_N`: Probability for class N (multiclass)

### Output Format Examples

**CSV Output** (`OUTPUT_FORMAT=csv`):
```csv
customer_id,is_fraud,feature_1,feature_2,prob_class_0,prob_class_1
12345,0,0.75,42.5,0.85,0.15
67890,1,0.45,38.2,0.12,0.88
```

**JSON Output** (`OUTPUT_FORMAT=json`, `JSON_ORIENT=records`):
```json
[
  {
    "customer_id": 12345,
    "is_fraud": 0,
    "feature_1": 0.75,
    "feature_2": 42.5,
    "prob_class_0": 0.85,
    "prob_class_1": 0.15
  },
  {
    "customer_id": 67890,
    "is_fraud": 1,
    "feature_1": 0.45,
    "feature_2": 38.2,
    "prob_class_0": 0.12,
    "prob_class_1": 0.88
  }
]
```

### Success Markers

**_SUCCESS**: Empty file indicating successful completion

**_HEALTH**: Health check file with timestamp
```
healthy: 2025-11-18T20:10:00.123456
```

## Key Functions and Tasks

### Model Loading Component

#### `load_model_artifacts(model_dir) -> Tuple`

**Purpose**: Load trained XGBoost model and all preprocessing artifacts from directory

**Algorithm**:
```python
1. Check for model.tar.gz
   IF exists AND xgboost_model.bst not found:
      Extract tar archive to model_dir
2. Load individual artifacts:
   a. Load XGBoost booster from xgboost_model.bst
   b. Load risk_table_map.pkl (pickle)
   c. Load impute_dict.pkl (pickle)
   d. Load feature_columns.txt (parse column names)
   e. Load hyperparameters.json
3. Return all artifacts as tuple
```

**Returns**: `(model, risk_tables, impute_dict, feature_columns, hyperparams)`

**Handles**:
- Automatic tar.gz extraction
- Both extracted and compressed model artifacts
- Validation of required files

### Format Detection Component

#### `_detect_file_format(file_path) -> str`

**Purpose**: Detect data file format from extension

**Algorithm**:
```python
suffix = file_path.suffix.lower()
IF suffix == ".csv": RETURN "csv"
ELIF suffix == ".tsv": RETURN "tsv"
ELIF suffix == ".parquet": RETURN "parquet"
ELSE: RAISE RuntimeError
```

**Returns**: Format string (`csv`, `tsv`, or `parquet`)

#### `load_dataframe_with_format(file_path) -> Tuple[pd.DataFrame, str]`

**Purpose**: Load DataFrame with automatic format detection

**Algorithm**:
```python
1. Detect format from file extension
2. Based on format:
   IF csv: pd.read_csv(file_path)
   ELIF tsv: pd.read_csv(file_path, sep="\t")
   ELIF parquet: pd.read_parquet(file_path)
3. Return (DataFrame, format_string)
```

**Format Preservation**: Returns format for downstream use

#### `save_dataframe_with_format(df, output_path, format_str) -> Path`

**Purpose**: Save DataFrame in specified format

**Algorithm**:
```python
1. Determine output file extension from format_str
2. Based on format:
   IF csv: df.to_csv(path, index=False)
   ELIF tsv: df.to_csv(path, sep="\t", index=False)
   ELIF parquet: df.to_parquet(path, index=False)
3. Return saved file path
```

**Complexity**: O(n×m) where n = rows, m = columns

### Preprocessing Component

#### `preprocess_inference_data(df, feature_columns, risk_tables, impute_dict) -> pd.DataFrame`

**Purpose**: Apply risk table mapping and numerical imputation to match training preprocessing

**Algorithm**:
```python
1. Create copy of input DataFrame (preserve original)
2. Identify available features from feature_columns
3. FOR each feature WITH risk_table:
      Create RiskTableMappingProcessor
      Transform feature using risk table
      Update column in DataFrame
4. Extract feature columns for imputation
5. Create NumericalVariableImputationProcessor
6. Transform features using imputation dict
7. Update feature columns in DataFrame
8. Convert features to numeric, fill remaining NaN with 0
9. Return preprocessed DataFrame
```

**Returns**: DataFrame with preprocessed features, preserving non-feature columns

**Key Operations**:
- Risk table mapping for categorical features
- Numerical imputation for missing values
- Type conversion and NaN handling
- Column preservation for ID, label, metadata

### Embedded Preprocessing Classes

#### `RiskTableMappingProcessor`

**Purpose**: Apply risk-table-based categorical encoding

**Key Methods**:
- `process(value)`: Map single value to risk score
- `transform(data)`: Transform Series or DataFrame column
- `set_risk_tables(risk_tables)`: Set risk table configuration

**Algorithm for transform**:
```python
1. Convert values to strings
2. Map each value using risk_tables['bins']
3. Fill unmapped values with risk_tables['default_bin']
4. Return transformed values
```

#### `NumericalVariableImputationProcessor`

**Purpose**: Impute missing numerical values using trained parameters

**Key Methods**:
- `process(input_data)`: Impute single record dictionary
- `transform(X)`: Impute DataFrame or Series
- `get_params()`: Return configuration parameters

**Algorithm for transform**:
```python
1. Create copy of input data
2. FOR each variable in imputation_dict:
      IF variable in DataFrame:
         Create mask for NaN values
         Replace NaN values with imputation value
3. Return imputed DataFrame
```

### Prediction Generation Component

#### `generate_predictions(model, df, feature_columns, hyperparams) -> np.ndarray`

**Purpose**: Generate class probability predictions using XGBoost model

**Algorithm**:
```python
1. Extract available features from DataFrame
2. Create feature matrix X from DataFrame
3. Create XGBoost DMatrix with feature names
4. Generate predictions: y_prob = model.predict(dmatrix)
5. Handle output format:
   IF binary (1D output):
      Convert to 2-column: [1-p, p]
   ELSE:
      Use multiclass output as-is
6. Return probability array
```

**Returns**: `np.ndarray` of shape `(n_samples, n_classes)`

**Binary Classification Handling**:
- XGBoost returns single probability for positive class
- Script converts to two-column format: `[prob_class_0, prob_class_1]`
- Ensures consistent output format for downstream processing

**Multiclass Classification**:
- XGBoost returns probabilities for all classes
- Output shape: `(n_samples, n_classes)`

### Output Saving Component

#### `save_predictions(df, predictions, output_dir, input_format, ...) -> str`

**Purpose**: Save predictions with original data in specified format

**Algorithm**:
```python
1. Create copy of original DataFrame
2. Add probability columns:
   FOR i in range(n_classes):
      output_df[f"prob_class_{i}"] = predictions[:, i]
3. Based on output format:
   IF json:
      Convert numpy types to Python types
      Save as JSON with specified orientation
   ELSE (csv, tsv, parquet):
      Use save_dataframe_with_format()
4. Return output file path
```

**Returns**: Path to saved predictions file

**Format Handling**:
- Preserves input format by default
- Overrides with `OUTPUT_FORMAT` env var if specified
- Special handling for JSON with type conversion

## Algorithms and Data Structures

### Algorithm 1: Model Artifact Loading with Tar Extraction

**Problem**: Load model artifacts that may be compressed in tar.gz or already extracted

**Solution Strategy**:
1. Check for existence of both tar.gz and extracted files
2. Extract if needed, use directly if already extracted
3. Validate all required artifacts present

**Algorithm**:
```python
model_tar_path = join(model_dir, "model.tar.gz")
model_bst_path = join(model_dir, "xgboost_model.bst")

IF exists(model_tar_path) AND NOT exists(model_bst_path):
   # Need to extract
   WITH tarfile.open(model_tar_path, "r:gz") as tar:
      tar.extractall(path=model_dir)
ELIF exists(model_bst_path):
   # Already extracted, use directly
   pass
ELSE:
   # Error: neither format available
   RAISE FileNotFoundError

# Load extracted artifacts
model = xgb.Booster()
model.load_model(join(model_dir, "xgboost_model.bst"))
risk_tables = pickle.load(join(model_dir, "risk_table_map.pkl"))
impute_dict = pickle.load(join(model_dir, "impute_dict.pkl"))
# ... load remaining artifacts
```

**Complexity**: O(1) for file operations, O(n) for tar extraction where n = compressed size

### Algorithm 2: Binary to Two-Column Probability Conversion

**Problem**: XGBoost binary classification returns single probability, but consistent output needs two columns

**Solution Strategy**:
```python
y_prob = model.predict(dmatrix)  # Shape: (n_samples,)

IF len(y_prob.shape) == 1:
   # Binary classification - convert to two-column
   y_prob = np.column_stack([1 - y_prob, y_prob])
   # Result shape: (n_samples, 2)
   # Column 0: P(class=0) = 1 - P(class=1)
   # Column 1: P(class=1) = original prediction
```

**Use Case**: Ensures consistent downstream processing regardless of binary vs multiclass

**Output Format**:
- Binary: Always `(n_samples, 2)` with `prob_class_0` and `prob_class_1`
- Multiclass: `(n_samples, n_classes)` with `prob_class_0` through `prob_class_N`

### Algorithm 3: Format Preservation Pattern

**Problem**: Maintain data format consistency across pipeline stages

**Solution Strategy**: Three-function pattern
```python
# 1. Detection
detected_format = _detect_file_format(input_path)

# 2. Loading
df, format_str = load_dataframe_with_format(input_path)

# 3. Saving (preserve or override)
final_format = OUTPUT_FORMAT if OUTPUT_FORMAT != "csv" else format_str
save_dataframe_with_format(df, output_path, final_format)
```

**Benefits**:
- Automatic format detection
- Format preservation by default
- Override capability with env var
- Consistent I/O across all formats

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Model Loading | O(m) | O(m) | m = model size |
| Tar Extraction | O(t) | O(t) | t = tar size (if needed) |
| Data Loading | O(n×f) | O(n×f) | n = rows, f = features |
| Risk Table Mapping | O(n×c) | O(n×f) | c = categorical features |
| Numerical Imputation | O(n×f) | O(n×f) | f = features |
| XGBoost Prediction | O(n×d×t) | O(n×k) | d = depth, t = trees, k = classes |
| Output Saving | O(n×f) | O(n×f) | Writing to disk |
| **Total** | **O(n×(f + d×t))** | **O(n×f + m)** | Dominated by prediction |

### Memory Requirements

**Peak Memory Usage**:
```
Total ≈ Model + Input Data + Predictions + Output Buffer

Example (100K records, 50 features, binary classification):
- Model artifacts: ~10-50 MB (depending on model complexity)
- Input DataFrame: ~40 MB (100K × 50 × 8 bytes)
- Predictions array: ~1.6 MB (100K × 2 × 8 bytes)
- Output buffer: ~45 MB (input + predictions)
Total: ~100-140 MB
```

### Processing Time Estimates

| Dataset Size | Features | Model Complexity | Time (typical) |
|--------------|----------|------------------|----------------|
| 1K records   | 50       | 100 trees, depth 6 | ~0.1 seconds |
| 10K records  | 50       | 100 trees, depth 6 | ~0.5 seconds |
| 100K records | 50       | 100 trees, depth 6 | ~2-3 seconds |
| 1M records   | 50       | 100 trees, depth 6 | ~20-30 seconds |
| 1M records   | 100      | 500 trees, depth 8 | ~2-3 minutes |

**Note**: Times vary based on CPU, model complexity, and I/O speed

## Error Handling

### Model Loading Errors

**Missing Model Artifacts**:
- **Cause**: Neither `model.tar.gz` nor `xgboost_model.bst` found
- **Handling**: Raises `FileNotFoundError` with available files list
- **User Action**: Verify model input path and model training completion

**Tar Extraction Failure**:
- **Cause**: Corrupted or incomplete tar.gz file
- **Handling**: Raises `RuntimeError` with extraction error details
- **User Action**: Re-run model training or verify model artifacts

### Data Loading Errors

**No Evaluation Data**:
- **Cause**: No CSV/TSV/Parquet files in eval_data directory
- **Handling**: Raises `RuntimeError` with error message
- **User Action**: Verify eval_data input path

**Unsupported Format**:
- **Cause**: File with unsupported extension (not .csv, .tsv, .parquet)
- **Handling**: Raises `RuntimeError` listing unsupported format
- **User Action**: Convert data to supported format

### Preprocessing Errors

**Missing Features**:
- **Severity**: Warning (non-critical)
- **Handling**: Uses available features, logs warning about missing features
- **Impact**: Predictions still generated with available features

**Risk Table Missing Category**:
- **Handling**: Uses `default_bin` value from risk table
- **Impact**: Graceful fallback for unseen categories

### Prediction Errors

**Feature Mismatch**:
- **Cause**: Features in DataFrame don't match model expectations
- **Handling**: Uses intersection of available and expected features
- **User Action**: Review preprocessing and feature engineering

**XGBoost Errors**:
- **Cause**: Invalid DMatrix or model configuration
- **Handling**: Propagates XGBoost exception with stack trace
- **User Action**: Check model training logs and feature compatibility

### Success and Failure Markers

**Success Flow**:
```python
# After successful inference
success_path = join(output_dir, "_SUCCESS")
Path(success_path).touch()

health_path = join(output_dir, "_HEALTH")
with open(health_path, "w") as f:
    f.write(f"healthy: {datetime.now().isoformat()}")
```

**Failure Flow**:
```python
# On exception
failure_path = join(output_dir, "_FAILURE")
with open(failure_path, "w") as f:
    f.write(f"Error: {str(exception)}")
sys.exit(1)
```

## Best Practices

### For Production Deployments

1. **Use Parquet Format**: More efficient for large datasets
   ```bash
   export OUTPUT_FORMAT="parquet"
   ```

2. **Verify Model Artifacts**: Ensure all required files present
   ```bash
   # Check model directory
   ls -la /opt/ml/processing/input/model/
   # Should see: xgboost_model.bst, risk_table_map.pkl, impute_dict.pkl, etc.
   ```

3. **Monitor Resource Usage**: Track memory and CPU for large datasets
   ```python
   # Typical usage: 100-150 MB per 100K records
   ```

4. **Use Consistent Field Names**: Match training configuration
   ```bash
   export ID_FIELD="customer_id"  # Must match training
   export LABEL_FIELD="is_fraud"   # Must match training
   ```

### For Development and Testing

1. **Start with CSV**: Easier to inspect and debug
   ```bash
   export OUTPUT_FORMAT="csv"
   ```

2. **Use Small Samples**: Test with subset before full dataset
   ```bash
   # Test with first 1000 records
   head -n 1001 large_data.csv > test_data.csv
   ```

3. **Check Success Markers**: Verify completion
   ```bash
   # Check for success
   test -f /opt/ml/processing/output/eval/_SUCCESS && echo "Success"
   ```

4. **Inspect Predictions**: Verify output structure
   ```python
   df = pd.read_csv("predictions.csv")
   print(df.columns)  # Should see: id, label, features, prob_class_0, prob_class_1
   print(df[['prob_class_0', 'prob_class_1']].describe())
   ```

### For Performance Optimization

1. **Use Parquet for Large Datasets**: 2-5x faster than CSV
   ```bash
   export OUTPUT_FORMAT="parquet"
   ```

2. **Minimize Features**: Remove unnecessary features before inference
   - Reduces memory usage
   - Speeds up preprocessing
   - Faster prediction generation

3. **Batch Processing**: Split very large datasets if memory-constrained
   ```bash
   # Process in chunks of 100K records
   split -l 100000 large_data.csv chunk_
   ```

4. **Monitor Preprocessing Time**: Risk table mapping can be slow for many categorical features
   - Consider reducing number of categorical features
   - Pre-compute risk tables offline if possible

## Example Configurations

### Example 1: Standard Binary Classification Inference
```bash
export ID_FIELD="customer_id"
export LABEL_FIELD="is_fraud"
export OUTPUT_FORMAT="csv"

python xgboost_model_inference.py --job_type inference
```

**Use Case**: Standard fraud detection inference with CSV output

**Expected Output**: `predictions.csv` with customer_id, is_fraud, features, prob_class_0, prob_class_1

### Example 2: Large-Scale Parquet Processing
```bash
export ID_FIELD="transaction_id"
export LABEL_FIELD="label"
export OUTPUT_FORMAT="parquet"

python xgboost_model_inference.py --job_type validation
```

**Use Case**: Processing millions of records efficiently

**Benefits**:
- 50-70% smaller file size
- 2-5x faster I/O
- Better compression with type information

### Example 3: JSON Output for API Integration
```bash
export ID_FIELD="request_id"
export LABEL_FIELD="ground_truth"
export OUTPUT_FORMAT="json"
export JSON_ORIENT="records"

python xgboost_model_inference.py --job_type inference
```

**Use Case**: API endpoints consuming predictions as JSON

**Output Format**:
```json
[
  {"request_id": 1, "ground_truth": 0, "prob_class_0": 0.85, "prob_class_1": 0.15},
  {"request_id": 2, "ground_truth": 1, "prob_class_0": 0.12, "prob_class_1": 0.88}
]
```

### Example 4: Calibration Dataset Preparation
```bash
export ID_FIELD="sample_id"
export LABEL_FIELD="true_label"
export OUTPUT_FORMAT="csv"

python xgboost_model_inference.py --job_type calibration
```

**Use Case**: Generate predictions for model calibration step

**Downstream**: Output consumed by `model_calibration.py` or `percentile_model_calibration.py`

## Integration Patterns

### Upstream Integration

```
XGBoostTraining
   ↓ (outputs: model artifacts in model.tar.gz)
XGBoostModelInference
   ↓ (outputs: predictions with probabilities)
```

**Training Output → Inference Input**:
- Model artifacts: `xgboost_model.bst`, `risk_table_map.pkl`, `impute_dict.pkl`, `feature_columns.txt`, `hyperparameters.json`
- Packaged in `model.tar.gz` or extracted

### Downstream Integration

```
XGBoostModelInference
   ↓ (outputs: predictions.csv with ID + label + prob_class_*)
ModelMetricsComputation
   ↓ (outputs: comprehensive metrics)
```

**OR**

```
XGBoostModelInference
   ↓ (outputs: predictions.csv with label + prob_class_*)
ModelCalibration/PercentileModelCalibration
   ↓ (outputs: calibrated model or percentile mapping)
```

### Complete Pipeline Flow

```
TabularPreprocessing → XGBoostTraining → XGBoostModelInference →
[Branch 1] → ModelMetricsComputation → ModelWikiGenerator
[Branch 2] → ModelCalibration → Package → Registration
```

**Key Integration Points**:
1. **Training → Inference**: Model artifacts
2. **Inference → Metrics**: Predictions with labels
3. **Inference → Calibration**: Predictions with labels
4. **Inference → Multiple Downstream**: Modular design allows caching and reuse

### Modular Design Benefits

**Cache and Reuse**:
- Generate predictions once
- Use same predictions for multiple downstream tasks
- Avoid redundant inference computation

**Parallel Processing**:
- Run metrics computation and calibration in parallel
- Both consume same inference output
- Reduces overall pipeline time

**Testing and Validation**:
- Test metrics computation without re-running inference
- Validate calibration with frozen predictions
- Reproducible results

## Troubleshooting

### Issue: Model Artifacts Not Found

**Symptom**: `FileNotFoundError: Model artifacts not found in /opt/ml/processing/input/model`

**Common Causes**:
1. Training step didn't complete successfully
2. Model input path misconfigured
3. Model.tar.gz not properly packaged

**Solution**:
1. Check training step logs for completion
2. Verify model output path from training matches inference input path
3. List model directory contents:
   ```bash
   ls -la /opt/ml/processing/input/model/
   ```
4. If model.tar.gz exists, verify it's not corrupted:
   ```bash
   tar -tzf /opt/ml/processing/input/model/model.tar.gz
   ```

### Issue: Feature Mismatch Warnings

**Symptom**: `Found X out of Y expected feature columns`

**Common Causes**:
1. Preprocessing step configuration changed between training and inference
2. Input data missing expected features
3. Feature engineering step not applied before inference

**Solution**:
1. Verify preprocessing configuration matches training
2. Check input data has all expected features:
   ```python
   df = pd.read_csv("eval_data.csv")
   print(df.columns.tolist())
   ```
3. Ensure same preprocessing pipeline as training
4. If intentional, inference will use available features (may impact accuracy)

### Issue: Predictions All Same Value

**Symptom**: All predictions show same probability (e.g., all 0.5)

**Common Causes**:
1. Model not properly loaded
2. Features not properly preprocessed
3. All features missing or NaN

**Solution**:
1. Verify model.bst file integrity
2. Check preprocessing logs for errors
3. Inspect feature values after preprocessing:
   ```python
   print(df[feature_columns].describe())
   print(df[feature_columns].isna().sum())
   ```

### Issue: Memory Error with Large Datasets

**Symptom**: `MemoryError` or killed process

**Common Causes**:
1. Dataset too large for available memory
2. Model too complex
3. Multiple copies of data in memory

**Solution**:
1. Use Parquet format (smaller memory footprint)
2. Process in batches:
   ```bash
   split -l 100000 large_data.csv chunk_
   # Process each chunk separately
   ```
3. Increase instance size
4. Monitor memory usage: `top` or `htop`

### Issue: Output Format Mismatch

**Symptom**: Expected CSV but got Parquet (or vice versa)

**Common Causes**:
1. `OUTPUT_FORMAT` env var not set correctly
2. Input format detection failing
3. Format override logic not working

**Solution**:
1. Explicitly set output format:
   ```bash
   export OUTPUT_FORMAT="csv"  # or "parquet" or "json"
   ```
2. Verify env var is set:
   ```bash
   echo $OUTPUT_FORMAT
   ```
3. Check input file extension matches content

### Issue: JSON Serialization Error

**Symptom**: `TypeError: Object of type X is not JSON serializable`

**Common Causes**:
1. NumPy types not converted to Python types
2. NaN or Inf values in output

**Solution**:
1. Script handles this automatically, but if it fails:
   ```python
   # Convert types manually
   df = df.astype({col: float for col in prob_columns})
   ```
2. Check for NaN/Inf values:
   ```python
   print(df[prob_columns].isna().sum())
   print(np.isinf(df[prob_columns]).sum())
   ```

## References

### Related Scripts

- [`xgboost_training.py`](xgboost_training_script.md): Training script that produces model artifacts consumed by this script
- [`xgboost_model_eval.py`](xgboost_model_eval_script.md): Evaluation script that computes metrics from inference predictions
- [`lightgbm_model_inference.py`](lightgbm_model_inference_script.md): Similar inference script for LightGBM models with parallel architecture
- [`pytorch_model_inference.py`](pytorch_model_inference_script.md): Inference script for PyTorch models (if exists)
- [`model_calibration.py`](model_calibration_script.md): Calibration script that consumes inference predictions
- [`tabular_preprocess.py`](tabular_preprocess_script.md): Preprocessing script that generates features for inference

### Related Documentation

- **Step Builder**: Documented in `slipbox/steps/` (if exists)
- **Contract**: [`src/cursus/steps/contracts/xgboost_model_inference_contract.py`](../../src/cursus/steps/contracts/xgboost_model_inference_contract.py)
- **Embedded Processors**: `RiskTableMappingProcessor` and `NumericalVariableImputationProcessor` are self-contained in this script

### Related Design Documents

- **[XGBoost Model Inference Design](../1_design/xgboost_model_inference_design.md)**: Comprehensive design document covering pure inference architecture, modular pipeline patterns, and multi-format output strategies
- **[Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md)**: Design patterns for automatic format detection and preservation across pipeline scripts
- **[Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)**: General patterns for building processing steps including inference steps

### External References

- [XGBoost Documentation](https://xgboost.readthedocs.io/): Official XGBoost documentation for model training and inference
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html): Python API reference for Booster.predict() and DMatrix
- [Pandas to_json Orientations](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html): JSON orientation options documentation
- [Parquet Format](https://parquet.apache.org/docs/): Apache Parquet columnar storage format documentation
