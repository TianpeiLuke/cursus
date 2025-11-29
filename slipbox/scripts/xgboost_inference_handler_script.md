---
tags:
  - code
  - inference_handler
  - sagemaker_endpoint
  - xgboost
  - gradient_boosting
  - real_time_inference
keywords:
  - xgboost inference handler
  - sagemaker endpoint
  - real-time prediction
  - tabular inference
  - calibrated predictions
  - model serving
topics:
  - real-time inference
  - model serving
  - xgboost endpoints
  - calibration integration
language: python
date of note: 2025-11-28
---

# XGBoost Real-Time Inference Handler Documentation

## Overview

The `xgboost_inference_handler.py` script implements the SageMaker inference handler functions (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) for deploying trained XGBoost models as real-time prediction endpoints. This handler serves HTTP requests with serialized tabular data, applies preprocessing transformations (risk table mapping, numerical imputation), generates predictions, and optionally applies calibration to improve probability estimates.

The handler loads all model artifacts (XGBoost model, preprocessing parameters, calibration models) once during endpoint initialization, processes incoming requests through preprocessing pipelines matching training, and returns both raw and calibrated predictions in JSON or CSV format for downstream consumption.

Key capabilities:
- **SageMaker Integration**: Standard handler functions for real-time endpoint deployment
- **Tabular ML Support**: Optimized for structured/tabular data with categorical and numerical features
- **Single-Record Fast Path**: 10-100x faster preprocessing for single-record requests
- **Embedded Preprocessing Artifacts**: Self-contained risk table mapping and numerical imputation
- **Calibration Support**: Optional probability calibration (binary, multiclass, percentile)
- **Dual Output Format**: Returns both raw and calibrated predictions
- **Multiple Input Formats**: Accepts CSV, JSON, and Parquet input data
- **Multiple Output Formats**: Returns JSON or CSV based on Accept header
- **Feature Alignment**: Strict feature column ordering for model compatibility
- **Optimized Inference**: Lightweight preprocessing with minimal overhead

## Purpose and Major Tasks

### Primary Purpose
Serve real-time predictions from trained XGBoost models via SageMaker endpoints by loading model artifacts once, processing incoming HTTP requests through preprocessing pipelines (risk tables + imputation), applying optional calibration, and returning structured predictions with both raw and calibrated probabilities.

### Major Tasks

1. **Model Artifact Loading** (`model_fn`): Load XGBoost model, preprocessing artifacts, and optional calibration models

2. **Feature Column Loading**: Load feature columns in correct order from feature_columns.txt

3. **Risk Table Loading**: Load categorical feature risk tables from training

4. **Imputation Dictionary Loading**: Load numerical imputation values from training

5. **Calibration Model Loading**: Load optional calibration models (binary, multiclass, percentile)

6. **Request Deserialization** (`input_fn`): Parse incoming HTTP requests into DataFrames

7. **Format Detection**: Auto-detect input format (CSV, JSON, Parquet) from content-type

8. **Input Validation**: Validate feature columns match training requirements

9. **Preprocessing Application** (`predict_fn`): Apply risk tables and imputation

10. **Single-Record Fast Path**: Bypass pandas for 10-100x faster single-record preprocessing

11. **Batch Prediction**: Generate raw predictions through XGBoost forward pass

12. **Calibration Application**: Apply optional calibration to improve probability estimates

13. **Response Formatting** (`output_fn`): Serialize predictions into JSON or CSV response

14. **Dual Output**: Return both raw and calibrated predictions for downstream choice

## Script Contract

### Entry Point
```
xgboost_inference_handler.py
```

### Deployment Context

**SageMaker Endpoint Initialization**:
```
1. SageMaker calls model_fn(model_dir="/opt/ml/model")
2. Handler loads all artifacts once
3. Returns model_data dict to be reused across requests
```

**SageMaker Request Processing**:
```
1. HTTP request arrives at endpoint
2. SageMaker calls input_fn(request_body, content_type)
3. Handler parses input into DataFrame
4. SageMaker calls predict_fn(input_df, model_data)
5. Handler generates predictions
6. SageMaker calls output_fn(predictions, accept)
7. Handler formats and returns response
```

### Model Directory Structure

```
/opt/ml/model/
├── xgboost_model.bst              # Trained XGBoost model
├── risk_table_map.pkl             # Risk tables for categorical encoding
├── impute_dict.pkl                # Imputation values for numerical features
├── feature_columns.txt            # Ordered feature list (REQUIRED)
├── hyperparameters.json           # Training hyperparameters (optional)
├── feature_importance.json        # Feature importance scores (optional)
├── calibration/                   # Optional calibration directory
│   ├── calibration_model.pkl      # Binary calibration lookup table
│   ├── percentile_score.pkl       # Percentile calibration mapping
│   ├── calibration_summary.json   # Calibration metadata
│   └── calibration_models/        # Multiclass calibration directory
│       ├── calibration_model_class_0.pkl
│       ├── calibration_model_class_1.pkl
│       └── ...
```

### Input Formats (Content-Type)

| Content-Type | Description | Example |
|--------------|-------------|---------|
| `text/csv` | CSV with NO header | `12345,5.2,category_A,3.7,...` |
| `application/json` | Single or multi-record JSON | `{"feature1": 5.2, "feature2": "A"}` |
| `application/x-parquet` | Parquet binary format | Binary parquet data |

### Output Formats (Accept Header)

| Accept | Description | Structure |
|--------|-------------|-----------|
| `application/json` | JSON with predictions array | `{"predictions": [{...}, {...}]}` |
| `text/csv` | CSV with probabilities (no header) | `raw_score,calib_score,label` |

### Required Columns

- **Tabular Features**: All numerical and categorical features used during training
- **NO ID Column**: IDs not required (pure feature input)
- **NO Label**: Labels not required for inference
- **Strict Ordering**: Features must match training order (from feature_columns.txt)

## Input Data Structure

### Expected Request Format

**CSV Request** (text/csv) - NO HEADER:
```csv
5.2,category_A,3.7,100,status_B
8.1,category_C,2.1,250,status_A
```

**JSON Request** (application/json):
```json
{
  "feature1": 5.2,
  "feature2": "category_A",
  "feature3": 3.7,
  "feature4": 100,
  "feature5": "status_B"
}
```

**Multi-Record JSON** (NDJSON):
```json
{"feature1": 5.2, "feature2": "category_A", "feature3": 3.7}
{"feature1": 8.1, "feature2": "category_C", "feature3": 2.1}
```

### Model Artifacts Structure

**feature_columns.txt** (REQUIRED - defines feature order):
```
# Feature columns in model training order
0,feature1
1,feature2
2,feature3
3,feature4
4,feature5
```

**risk_table_map.pkl** (categorical encoding):
```python
{
    'feature2': {
        'bins': {'category_A': 0.75, 'category_B': 0.45, 'category_C': 0.60},
        'default_bin': 0.5
    },
    'feature5': {
        'bins': {'status_A': 0.65, 'status_B': 0.35},
        'default_bin': 0.5
    }
}
```

**impute_dict.pkl** (numerical imputation):
```python
{
    'feature1': 0.0,
    'feature3': 5.2,
    'feature4': 100.0
}
```

**Calibration Models** (same structure as PyTorch handler):

*Binary Calibration* (`calibration/calibration_model.pkl`):
```python
[
  (0.0, 0.0),
  (0.1, 0.05),
  (0.5, 0.45),
  (0.9, 0.92),
  (1.0, 1.0)
]  # List[(raw_score, calibrated_score)]
```

*Percentile Calibration* (`calibration/percentile_score.pkl`):
```python
[
  (0.0, 0.0),
  (0.5, 0.65),  # 50th percentile → 65% calibrated
  (0.9, 0.95),
  (1.0, 1.0)
]  # List[(raw_score, percentile)]
```

## Output Data Structure

### JSON Response Format (application/json)

**Binary Classification**:
```json
{
  "predictions": [
    {
      "legacy-score": "0.85",           // Raw class-1 probability
      "score-percentile": "0.78",       // Calibrated class-1 probability
      "calibrated-score": "0.78",       // Calibrated class-1 probability (duplicate)
      "custom-output-label": "class-1"
    }
  ]
}
```

**Multiclass Classification**:
```json
{
  "predictions": [
    {
      "prob_01": "0.2",                     // Raw probabilities
      "calibrated_prob_01": "0.18",         // Calibrated probabilities
      "prob_02": "0.5",
      "calibrated_prob_02": "0.52",
      "prob_03": "0.3",
      "calibrated_prob_03": "0.30",
      "custom-output-label": "class-1"
    }
  ]
}
```

### CSV Response Format (text/csv) - NO HEADER

**Binary**:
```csv
0.85,0.78,0.78,class-1
0.12,0.08,0.08,class-0
```

**Multiclass**:
```csv
0.2,0.18,0.5,0.52,0.3,0.30,class-1
```

## Key Functions and Handler Architecture

### Handler Function 1: Model Loading

#### `model_fn(model_dir: str) -> Dict[str, Any]`

**Purpose**: Load all model artifacts once during endpoint initialization

**Algorithm**:
```python
1. Validate all required files exist:
   - xgboost_model.bst (REQUIRED)
   - risk_table_map.pkl (REQUIRED)
   - impute_dict.pkl (REQUIRED)
   - feature_columns.txt (REQUIRED)
2. Load XGBoost model from .bst file
3. Load risk tables from pickle:
   - Parse risk table dictionary
   - Create RiskTableMappingProcessor for each categorical feature
4. Load imputation dictionary from pickle:
   - Parse imputation values
   - Create NumericalVariableImputationProcessor for each numerical feature
5. Load feature columns in correct order:
   - Parse feature_columns.txt (format: "index,column_name")
   - Maintain strict ordering for XGBoost compatibility
6. Load optional hyperparameters.json
7. Load optional feature_importance.json
8. Load optional calibration model:
   - Binary: calibration/calibration_model.pkl
   - Multiclass: calibration/calibration_models/*.pkl
   - Percentile: calibration/percentile_score.pkl
9. Create model configuration:
   - Detect multiclass vs binary from model
   - Store feature columns and hyperparameters
10. Return model_data dictionary
```

**Returns**:
```python
{
    "model": xgb.Booster,
    "risk_processors": Dict[str, RiskTableMappingProcessor],
    "numerical_processors": Dict[str, NumericalVariableImputationProcessor],
    "feature_importance": Dict[str, float],
    "config": {
        "is_multiclass": bool,
        "num_classes": int,
        "feature_columns": List[str],
        "hyperparameters": Dict
    },
    "calibrator": Optional[Dict]
}
```

**Key Requirements**:
- All required files MUST exist or FileNotFoundError raised
- Feature columns MUST preserve training order
- Processors created for ALL categorical and numerical features

### Handler Function 2: Input Parsing

#### `input_fn(request_body, content_type, context=None) -> pd.DataFrame`

**Purpose**: Parse HTTP request body into DataFrame based on content-type

**Algorithm**: Same as PyTorch handler (CSV, JSON, Parquet support)

**Returns**: `pd.DataFrame` with all input columns

**Error Handling**: 
- Empty input raises ValueError
- Malformed data raises ValueError with details
- Unsupported content-type raises ValueError

### Handler Function 3: Prediction Generation

#### `predict_fn(input_data: pd.DataFrame, model_artifacts: Dict) -> Dict[str, np.ndarray]`

**Purpose**: Generate raw and calibrated predictions from input DataFrame

**Algorithm**:
```python
1. Extract configuration:
   - model, risk_processors, numerical_processors
   - feature_columns, is_multiclass, calibrator
2. Validate input data:
   - Check not empty
   - Verify feature count matches (if headerless CSV)
   - Verify required features present (if named columns)
3. Assign column names if headerless:
   - Use feature_columns to name columns
4. Determine preprocessing path:
   
   IF len(input_data) == 1:  # FAST PATH
      a. Assign column names
      b. Call preprocess_single_record_fast():
         - Extract single row values
         - Apply risk table mapping (direct dict lookup)
         - Apply numerical imputation (simple null check)
         - Convert to float array
         - Return np.ndarray [n_features]
      c. Create XGBoost DMatrix from array
   
   ELSE:  # BATCH PATH
      a. Assign column names
      b. Call apply_preprocessing():
         - Apply risk table processors to DataFrame
         - Apply numerical imputation processors to DataFrame
      c. Call convert_to_numeric():
         - Convert all columns to numeric dtype
         - Fill NaN with defaults
         - Validate all numeric
      d. Create XGBoost DMatrix from DataFrame
5. Generate raw predictions:
   raw_predictions = model.predict(dmatrix)
6. Format binary predictions if needed:
   IF binary AND raw_predictions is 1D:
      raw_predictions = [[1-p, p], ...] # Create 2-column format
7. Apply calibration if available:
   IF calibrator exists:
      calibrated_predictions = apply_calibration(
         raw_predictions, calibrator, is_multiclass
      )
   ELSE:
      calibrated_predictions = raw_predictions.copy()
8. Return both predictions:
   {
      "raw_predictions": raw_predictions,
      "calibrated_predictions": calibrated_predictions
   }
```

**Returns**: Dictionary with raw and calibrated predictions (both `np.ndarray`)

**Key Optimization**: Single-record fast path bypasses pandas operations for 10-100x speedup

### Handler Function 4: Response Formatting

#### `output_fn(prediction_output, accept="application/json") -> Tuple[str, str]`

**Purpose**: Serialize predictions into response format based on Accept header

**Algorithm**: Similar to PyTorch handler with XGBoost-specific field names

**Output Format Differences from PyTorch**:
- Binary JSON: `legacy-score`, `score-percentile`, `calibrated-score`, `custom-output-label`
- Multiclass JSON: `prob_01`, `calibrated_prob_01`, ..., `custom-output-label`
- CSV: No header, values only

**Returns**: `Tuple[response_body: str, content_type: str]`

## Calibration Integration Architecture

### Calibration Functions (Shared with PyTorch Handler)

The XGBoost handler uses the same calibration architecture as the PyTorch handler:

1. **`load_calibration_model()`**: Load calibration from various formats
2. **`apply_calibration()`**: Dispatcher for calibration methods
3. **`apply_percentile_calibration()`**: Percentile score mapping
4. **`apply_regular_binary_calibration()`**: Binary lookup table or legacy model
5. **`apply_regular_multiclass_calibration()`**: Per-class calibration with renormalization
6. **`_interpolate_score()` (implicit)**: Linear interpolation for lookup tables

See [PyTorch Inference Handler](pytorch_inference_handler_script.md) for detailed calibration algorithms.

## Single-Record Fast Path Optimization

### Algorithm: Fast Path Preprocessing

**Problem**: Single-record requests (most common in real-time serving) spend 90%+ time in pandas operations

**Solution**: Bypass pandas for single records, use direct value processing

**Algorithm**:
```python
def preprocess_single_record_fast(df, feature_columns, risk_processors, numerical_processors):
   """
   Fast path: 10-100x faster than pandas for single records.
   
   Complexity: O(n) where n = number of features
   Memory: O(n) - single array allocation
   """
   processed = np.zeros(len(feature_columns), dtype=np.float32)
   
   FOR i, col in enumerate(feature_columns):
      val = df[col].iloc[0]  # Extract single value
      
      # Apply risk table if categorical (O(1) dict lookup)
      IF col in risk_processors:
         val = risk_processors[col].process(val)  # Direct dict lookup
      
      # Apply imputation if numerical (O(1) null check)
      IF col in numerical_processors:
         val = numerical_processors[col].process(val)  # Simple null check
      
      # Convert to float
      TRY:
         val = float(val)
      EXCEPT:
         val = 0.0
      
      processed[i] = val
   
   RETURN processed  # Ready for XGBoost DMatrix
```

**Performance Comparison**:

| Path | Records | Preprocessing Time | Speedup |
|------|---------|-------------------|---------|
| Batch | 1 | 5-10 ms | 1x (baseline) |
| Fast | 1 | 50-100 μs | 50-100x |
| Batch | 100 | 10-20 ms | N/A |
| Fast | 100 | N/A (uses batch path) | N/A |

**Key Optimizations**:
1. **No DataFrame Operations**: Direct value extraction and processing
2. **Pre-allocated Array**: Single numpy array for results
3. **Processor.process() Method**: Optimized single-value processing
4. **Minimal Type Conversions**: Direct float conversion
5. **Zero Copying**: In-place array population

**When Used**:
- Automatically triggered for `len(input_data) == 1`
- Transparent to caller (same output format)
- Falls back to batch path for multiple records

## Performance Characteristics

### Latency Analysis

| Component | Latency (single-record) | Latency (batch-100) | Notes |
|-----------|------------------------|---------------------|-------|
| Model Loading (cold start) | 500 ms - 2 sec | - | One-time cost per instance |
| Request Parsing | 1-5 ms | 5-20 ms | Depends on format |
| Preprocessing (fast path) | 50-100 μs | N/A | Single-record only |
| Preprocessing (batch path) | 5-10 ms | 10-20 ms | Multiple records |
| XGBoost Prediction | 1-5 ms | 5-15 ms | Depends on model size |
| Calibration Application | 10-50 μs | 100-500 μs | Lookup table interpolation |
| Response Formatting | 100-500 μs | 1-5 ms | JSON serialization |
| **Total (single, fast path)** | **2-10 ms** | - | P50-P99 with optimization |
| **Total (single, batch path)** | **7-21 ms** | - | P50-P99 without optimization |
| **Total (batch-100)** | - | **20-60 ms** | Amortized cost per record |

### Throughput Analysis

| Configuration | Throughput (req/s) | Notes |
|--------------|-------------------|-------|
| Single instance, CPU (fast path) | 100-500 | Single-record requests |
| Single instance, CPU (batch path) | 50-150 | Single-record without optimization |
| Single instance, CPU (batch=100) | 1,000-5,000 records/s | Batch requests |
| 3 instances, CPU (fast path) | 300-1,500 | Linear scaling |
| 3 instances, auto-scaling | 500-3,000+ | Dynamic scaling under load |

**Key Insight**: Fast path provides 2-3x higher throughput for typical single-record serving patterns

### Memory Requirements

**Endpoint Instance Memory**:
```
Total = Model + Framework + Preprocessing Artifacts + Request Buffer

Example (Binary Classification, 100 features):
- XGBoost model: 10-50 MB (depends on trees)
- XGBoost framework: 50-100 MB
- Risk tables: 1-5 MB (categorical features)
- Imputation dict: <1 MB (numerical features)
- Feature columns: <1 MB
- Calibration models: 1-5 MB
- Request buffer: 1-10 MB (per concurrent request)

Total: ~65-170 MB RAM (cold) + ~5-15 MB per request
```

**Recommended Instance Types**:
- **Small models (<20 MB)**: ml.t3.medium (2 vCPU, 4 GB RAM) - $0.05/hr
- **Medium models (<50 MB)**: ml.m5.large (2 vCPU, 8 GB RAM) - $0.10/hr
- **Large models (>50 MB)**: ml.m5.xlarge (4 vCPU, 16 GB RAM) - $0.23/hr

## Error Handling

### Model Loading Errors

**Missing Required Files**:
- **Symptom**: FileNotFoundError listing missing files
- **Handling**: Fails fast during endpoint startup
- **User Action**: Verify model training completion and tar.gz contents

**Invalid Feature Columns File**:
- **Symptom**: ValueError "No valid feature columns found"
- **Handling**: Endpoint fails to start
- **User Action**: Check feature_columns.txt format

### Request Processing Errors

**Invalid Content-Type**:
- **Symptom**: HTTP 400 Bad Request
- **Response**: `{"error": "Unsupported content type"}`
- **User Action**: Use text/csv, application/json, or application/x-parquet

**Feature Count Mismatch**:
- **Symptom**: ValueError "Input data has X columns but model expects Y"
- **Handling**: Request rejected
- **User Action**: Verify input has all required features

**Missing Features**:
- **Symptom**: ValueError "Missing required features: {...}"
- **Handling**: Request rejected with list of missing features
- **User Action**: Include all features from training

### Prediction Errors

**Non-Numeric Values After Preprocessing**:
- **Symptom**: ValueError "Following columns contain non-numeric values"
- **Handling**: Request rejected with column names
- **User Action**: Check categorical values are in risk tables

**Calibration Application Failure**:
- **Symptom**: Warning logged, falls back to raw predictions
- **Handling**: Returns raw predictions only
- **Impact**: No calibrated predictions in response

## Best Practices

### For Production Deployments

1. **Use Fast Path for Single-Record Serving**:
   - Automatically enabled for single records
   - 10-100x faster preprocessing
   - Lower latency, higher throughput

2. **Enable Auto-Scaling**:
   ```python
   # Scale based on invocation metrics
   min_instances = 2
   max_instances = 20
   target_invocations_per_instance = 100  # Higher than PyTorch
   ```

3. **Use Model Calibration**:
   - Train calibration model separately
   - Include in model.tar.gz
   - Improves probability quality

4. **Monitor Endpoint Metrics**:
   - Invocations per minute
   - Model latency (P50, P95, P99)
   - CPU utilization
   - 4XX/5XX error rates

5. **Batch Requests When Possible**:
   - Send multiple records in single request
   - Amortize overhead costs
   - 20-50x records/second improvement

### For Cost Optimization

1. **Right-Size Instances**:
   - Start with ml.m5.large for most models
   - Monitor CPU utilization (<70% is over-provisioned)
   - Scale down to ml.t3.medium for small models

2. **Use Serverless Inference** (if available):
   - Pay per invocation
   - No idle costs
   - Cold start latency acceptable for XGBoost (fast loading)

3. **Optimize for Fast Path**:
   - Design API for single-record requests
   - Leverage 2-3x throughput improvement
   - Reduce instance count needed

### For Development and Testing

1. **Test with Local Endpoint**:
   ```python
   from sagemaker.local import LocalSession
   local_session = LocalSession()
   ```

2. **Validate Preprocessing**:
   - Compare inference preprocessing with training
   - Check risk table mappings for unseen categories
   - Verify imputation values

3. **Test Calibration**:
   - Verify calibration files loaded
   - Compare raw vs calibrated outputs
   - Validate probability ranges

4. **Test Fast Path**:
   - Send single-record requests
   - Monitor latency (should be <5 ms)
   - Verify results match batch path

## Example Usage Patterns

### Example 1: Single Record CSV Request (Fast Path)
```python
import requests

endpoint_url = "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-xgboost-endpoint/invocations"

# CSV payload - NO HEADER
# Features in exact training order
data = "5.2,category_A,3.7,100,status_B"

response = requests.post(
    endpoint_url,
    data=data,
    headers={
        "Content-Type": "text/csv",
        "Accept": "application/json"
    }
)

print(response.json())
# {
#   "predictions": [{
#     "legacy-score": "0.85",
#     "score-percentile": "0.78",
#     "calibrated-score": "0.78",
#     "custom-output-label": "class-1"
#   }]
# }
```

### Example 2: Single Record JSON Request
```python
import requests
import json

endpoint_url = "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-xgboost-endpoint/invocations"

# JSON payload with feature names
data = {
    "feature1": 5.2,
    "feature2": "category_A",
    "feature3": 3.7,
    "feature4": 100,
    "feature5": "status_B"
}

response = requests.post(
    endpoint_url,
    data=json.dumps(data),
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
)

print(response.json())
```

### Example 3: Batch Request for Throughput
```python
import requests
import json

endpoint_url = "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-xgboost-endpoint/invocations"

# Batch of 100 records
batch_data = [
    {"feature1": 5.2, "feature2": "category_A", ...},
    {"feature1": 8.1, "feature2": "category_C", ...},
    # ... 98 more records
]

response = requests.post(
    endpoint_url,
    data=json.dumps(batch_data),
    headers={
        "Content-Type": "application/json",
        "Accept": "text/csv"  # CSV output for efficiency
    }
)

# Parse CSV response
predictions = response.text.strip().split('\n')
print(f"Predicted {len(predictions)} records")
```

### Example 4: CSV Output for Downstream Processing
```python
import requests
import pandas as pd
from io import StringIO

endpoint_url = "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-xgboost-endpoint/invocations"

# CSV input
csv_data = """5.2,category_A,3.7,100,status_B
8.1,category_C,2.1,250,status_A"""

response = requests.post(
    endpoint_url,
    data=csv_data,
    headers={
        "Content-Type": "text/csv",
        "Accept": "text/csv"
    }
)

# Parse CSV response (no header)
# Columns: legacy-score, score-percentile, calibrated-score, custom-output-label
df = pd.read_csv(StringIO(response.text), header=None)
df.columns = ['raw_score', 'score_percentile', 'calibrated_score', 'label']
print(df)
```

## Integration Patterns

### Upstream Integration

```
XGBoostTraining
   ↓ (outputs: xgboost_model.bst, risk_table_map.pkl, impute_dict.pkl, feature_columns.txt)
XGBoostInferenceHandler (Endpoint)
   ↓ (outputs: real-time predictions with raw + calibrated scores)
```

**Training Output → Inference Input**:
- Model artifacts: `xgboost_model.bst`, risk tables, imputation dict, feature columns
- Packaged in `model.tar.gz`
- Feature order MUST match training

### Downstream Integration

```
XGBoostInferenceHandler (Endpoint)
   ↓ (outputs: JSON/CSV with raw + calibrated predictions)
[Branch 1] → Business Logic → Decision System
[Branch 2] → Monitoring → Alerting
[Branch 3] → Data Lake → Analysis
```

### Complete Pipeline Flow

```
TabularPreprocessing → XGBoostTraining → XGBoostModelCalibration →
Package → Registration → Endpoint Deployment →
XGBoostInferenceHandler → Real-Time Serving
```

**Key Integration Points**:
1. **Training → Calibration**: Predictions with labels for calibration
2. **Calibration → Packaging**: Calibrated model + original model
3. **Packaging → Endpoint**: model.tar.gz with all artifacts
4. **Endpoint → Applications**: Real-time predictions via HTTP/HTTPS

### Comparison with PyTorch Handler

| Feature | XGBoost Handler | PyTorch Handler |
|---------|----------------|-----------------|
| **Model Format** | xgboost_model.bst | model.pth + model_artifacts.pth |
| **Preprocessing** | Risk tables + imputation | Hyperparameter-driven pipelines |
| **Text Support** | Not applicable | Full text pipelines (tokenization) |
| **Architecture** | Tabular only | Bimodal, trimodal, text-only |
| **Calibration** | Lookup tables (binary, multiclass, percentile) | Same as XGBoost |
| **Fast Path** | 50-100x speedup for single records | Not implemented |
| **Typical Latency** | 2-10 ms (with fast path) | 20-200 ms |
| **Use Case** | Pure tabular features | Multi-modal, text-heavy |

## Related Documentation

- [XGBoost Training Script](xgboost_train_script.md) - Training implementation
- [XGBoost Model Evaluation Script](xgboost_model_eval_script.md) - Batch evaluation
- [Model Calibration Script](model_calibration_script.md) - Calibration training
- [Percentile Model Calibration Design](../1_design/percentile_model_calibration_design.md) - Percentile calibration
- [PyTorch Inference Handler](pytorch_inference_handler_script.md) - PyTorch equivalent
- [XGBoost Pipelines Index](../00_entry_points/xgboost_pipelines_index.md) - Pipeline catalog

---

## Maintenance Notes

**Last Updated:** 2025-11-28

**Update Triggers**:
- New calibration methods
- Fast path optimizations
- Feature alignment changes
- Performance improvements

**Maintenance Guidelines**:
- Keep calibration logic synced with PyTorch handler
- Update fast path implementation with new optimizations
- Document feature_columns.txt requirements
- Track performance benchmarks
