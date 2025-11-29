---
tags:
  - code
  - inference_handler
  - sagemaker_endpoint
  - pytorch
  - deep_learning
  - real_time_inference
keywords:
  - pytorch inference handler
  - sagemaker endpoint
  - real-time prediction
  - multi-modal inference
  - calibrated predictions
  - model serving
topics:
  - real-time inference
  - model serving
  - pytorch endpoints
  - calibration integration
language: python
date of note: 2025-11-28
---

# PyTorch Real-Time Inference Handler Documentation

## Overview

The `pytorch_inference_handler.py` script implements the SageMaker inference handler functions (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) for deploying trained PyTorch Lightning models as real-time prediction endpoints. This handler serves HTTP requests with serialized input data, applies the same preprocessing transformations used during training (text processing, risk table mapping, numerical imputation), generates predictions, and optionally applies calibration to improve probability estimates.

The handler loads all model artifacts (model weights, tokenizer, preprocessing parameters, calibration models) once during endpoint initialization, processes incoming requests through a complete preprocessing pipeline matching training, and returns both raw and calibrated predictions in JSON or CSV format for downstream consumption.

Key capabilities:
- **SageMaker Integration**: Standard handler functions for real-time endpoint deployment
- **Multi-Modal Support**: Handles text, tabular, bimodal, and trimodal architectures
- **Hyperparameter-Driven Preprocessing**: Reconstructs text pipelines from training metadata
- **Embedded Preprocessing Artifacts**: Self-contained risk table mapping and imputation
- **Calibration Support**: Optional probability calibration (binary, multiclass, percentile)
- **Dual Output Format**: Returns both raw and calibrated predictions
- **Multiple Input Formats**: Accepts CSV, JSON, and Parquet input data
- **Multiple Output Formats**: Returns JSON or CSV based on Accept header
- **Device Flexibility**: Automatic GPU detection or explicit CPU/GPU configuration
- **ONNX Support**: Can serve ONNX-converted models for faster inference

## Purpose and Major Tasks

### Primary Purpose
Serve real-time predictions from trained PyTorch Lightning models via SageMaker endpoints by loading model artifacts once, processing incoming HTTP requests through complete preprocessing pipelines, applying optional calibration, and returning structured predictions with both raw and calibrated probabilities.

### Major Tasks

1. **Model Artifact Loading** (`model_fn`): Load trained model, tokenizer, preprocessing artifacts, and optional calibration models

2. **Hyperparameter Loading**: Load training hyperparameters to reconstruct text processing pipelines

3. **Text Pipeline Reconstruction**: Build text preprocessing pipelines from hyperparameter-driven configurations

4. **Preprocessing Artifact Loading**: Load risk tables and imputation dictionaries from training

5. **Calibration Model Loading**: Load optional calibration models (binary, multiclass, percentile)

6. **Request Deserialization** (`input_fn`): Parse incoming HTTP requests into DataFrames

7. **Format Detection**: Auto-detect input format (CSV, JSON, Parquet) from content-type

8. **BSMDataset Creation**: Initialize dataset with configuration from model artifacts

9. **Pipeline Application**: Apply text, numerical, and categorical preprocessing pipelines

10. **Batch Prediction** (`predict_fn`): Generate raw predictions through model forward pass

11. **Calibration Application**: Apply optional calibration to improve probability estimates

12. **Response Formatting** (`output_fn`): Serialize predictions into JSON or CSV response

13. **Dual Output**: Return both raw and calibrated predictions for downstream choice

## Script Contract

### Entry Point
```
pytorch_inference_handler.py
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
├── model.pth                      # Trained PyTorch model weights
├── model_artifacts.pth            # Config, embeddings, vocab
├── risk_table_map.pkl             # Risk tables for categorical encoding
├── impute_dict.pkl                # Imputation values for numerical features
├── hyperparameters.json           # Training hyperparameters (NEW)
├── feature_columns.txt            # Ordered feature list (optional)
├── calibration/                   # Optional calibration directory
│   ├── calibration_model.pkl      # Binary calibration lookup table
│   ├── percentile_score.pkl       # Percentile calibration mapping
│   └── calibration_models/        # Multiclass calibration directory
│       ├── calibration_model_class_0.pkl
│       ├── calibration_model_class_1.pkl
│       └── ...
└── model.onnx                     # Optional ONNX model
```

### Input Formats (Content-Type)

| Content-Type | Description | Example |
|--------------|-------------|---------|
| `text/csv` | CSV with NO header | `customer_id,text,feature1,...` |
| `application/json` | Single or multi-record JSON | `{"customer_id": 123, "text": "..."}` |
| `application/x-parquet` | Parquet binary format | Binary parquet data |

### Output Formats (Accept Header)

| Accept | Description | Structure |
|--------|-------------|-----------|
| `application/json` | JSON with predictions array | `{"predictions": [{...}, {...}]}` |
| `text/csv` | CSV with probabilities | `raw_probs,calibrated_probs,label` |

### Required Columns

- **ID Column**: Unique identifier (optional, for tracking)
- **Text Columns**: Raw text fields for text subnetwork
- **Tabular Columns**: Numerical and categorical features
- **NO Label**: Labels not required for inference

## Input Data Structure

### Expected Request Format

**CSV Request** (text/csv):
```csv
12345,Hello customer service...,5.2,category_A,3
67890,I need help with...,8.1,category_B,7
```

**JSON Request** (application/json):
```json
{
  "customer_id": 12345,
  "text": "Hello customer service...",
  "feature1": 5.2,
  "feature2": "category_A",
  "feature3": 3
}
```

**Multi-Record JSON** (NDJSON):
```json
{"customer_id": 12345, "text": "Hello..."}
{"customer_id": 67890, "text": "I need help..."}
```

### Model Artifacts Structure

**hyperparameters.json** (NEW - Critical for pipeline reconstruction):
```json
{
  "model_class": "bimodal_bert",
  "is_binary": true,
  "text_name": "chat_messages",
  "text_processing_steps": [
    "dialogue_splitter",
    "html_normalizer",
    "emoji_remover",
    "text_normalizer",
    "dialogue_chunker",
    "tokenizer"
  ],
  "primary_text_name": "chat",
  "primary_text_processing_steps": ["dialogue_splitter", "html_normalizer", ...],
  "secondary_text_name": "events",
  "secondary_text_processing_steps": ["dialogue_splitter", "text_normalizer", ...],
  "max_sen_len": 512,
  "tokenizer": "bert-base-multilingual-cased",
  ...
}
```

**Calibration Models**:

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

*Multiclass Calibration* (`calibration/calibration_models/`):
```
calibration_model_class_0.pkl  # Calibration for class 0
calibration_model_class_1.pkl  # Calibration for class 1
calibration_model_class_N.pkl  # Calibration for class N
```

## Output Data Structure

### JSON Response Format (application/json)

**Binary Classification**:
```json
{
  "predictions": [
    {
      "legacy-score": "0.85",           // Raw class-1 probability
      "calibrated-score": "0.78",       // Calibrated class-1 probability
      "output-label": "class-1"
    },
    {
      "legacy-score": "0.12",
      "calibrated-score": "0.08",
      "output-label": "class-0"
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
      "prob_02": "0.5",
      "prob_03": "0.3",
      "calibrated_prob_01": "0.18",         // Calibrated probabilities
      "calibrated_prob_02": "0.52",
      "calibrated_prob_03": "0.30",
      "output-label": "class-1"
    }
  ]
}
```

### CSV Response Format (text/csv)

**Binary**:
```csv
0.85,0.15,0.78,0.22,class-0
0.12,0.88,0.08,0.92,class-1
```

**Multiclass**:
```csv
0.2,0.5,0.3,0.18,0.52,0.30,class-1
```

## Key Functions and Handler Architecture

### Handler Function 1: Model Loading

#### `model_fn(model_dir, context=None) -> Dict`

**Purpose**: Load all model artifacts once during endpoint initialization

**Algorithm**:
```python
1. Detect model type (ONNX vs PyTorch):
   IF model.onnx exists:
      Load ONNX model
   ELSE:
      Load PyTorch model from model.pth
2. Load model_artifacts.pth:
   - config (model configuration)
   - embedding_mat (optional)
   - vocab (optional)
   - model_class (architecture name)
3. Create Config object with validation
4. Load hyperparameters.json:  # NEW
   - Text processing steps
   - Pipeline configurations
5. Reconstruct text preprocessing pipelines:  # NEW
   - Use hyperparameter-driven steps
   - Build bimodal or trimodal pipelines
   - Configure tokenizer
6. Load preprocessing artifacts:
   - risk_table_map.pkl → create risk processors
   - impute_dict.pkl → create numerical processors
7. Load optional calibration model:  # NEW
   - Binary: calibration/calibration_model.pkl
   - Multiclass: calibration/calibration_models/*.pkl
   - Percentile: calibration/percentile_score.pkl
8. Load optional feature_columns.txt
9. Return model_data dictionary
```

**Returns**:
```python
{
    "model": torch.nn.Module,
    "config": Config,
    "embedding_mat": np.ndarray,
    "vocab": dict,
    "model_class": str,
    "pipelines": Dict[str, Processor],
    "calibrator": Optional[Dict],  # NEW
    "feature_columns": Optional[List[str]],  # NEW
    "hyperparameters": Dict,  # NEW
    "risk_processors": Dict[str, Processor],
    "numerical_processors": Dict[str, Processor]
}
```

**Key Innovation**: Hyperparameter-driven pipeline reconstruction ensures inference preprocessing matches training exactly

### Handler Function 2: Input Parsing

#### `input_fn(request_body, content_type, context=None) -> pd.DataFrame`

**Purpose**: Parse HTTP request body into DataFrame based on content-type

**Algorithm**:
```python
IF content_type == "text/csv":
   1. Decode bytes to UTF-8 string
   2. Parse CSV with pandas (NO header, NO index)
   3. Return DataFrame
ELIF content_type == "application/json":
   1. Decode bytes to UTF-8 string
   2. IF contains newlines:
         Parse as NDJSON (multi-record)
      ELSE:
         Parse as single JSON object
   3. Convert to DataFrame
   4. Return DataFrame
ELIF content_type == "application/x-parquet":
   1. Read Parquet from bytes buffer
   2. Return DataFrame
ELSE:
   Raise ValueError("Unsupported content type")
```

**Returns**: `pd.DataFrame` with all input columns

**Error Handling**: Logs parsing errors, raises ValueError with details

### Handler Function 3: Prediction Generation

#### `predict_fn(input_object, model_data, context=None) -> Dict`

**Purpose**: Generate raw and calibrated predictions from input DataFrame

**Algorithm**:
```python
1. Extract model, config, and processors from model_data
2. Remove label field from config (inference only)
3. Create BSMDataset from input DataFrame
4. Apply preprocessing pipelines in order:
   a. Numerical imputation processors
   b. Risk table mapping processors (excluding text)
   c. Text processing pipelines
5. Create DataLoader with BSM collate function
6. Generate raw predictions:
   raw_probs = model_online_inference(model, dataloader)
7. Apply calibration if available:  # NEW
   IF calibrator exists:
      is_multiclass = not config.is_binary
      calibrated_probs = apply_calibration(
         raw_probs, calibrator, is_multiclass
      )
   ELSE:
      calibrated_probs = raw_probs.copy()
8. Return dictionary with both predictions:  # NEW
   {
      "raw_predictions": raw_probs,
      "calibrated_predictions": calibrated_probs
   }
```

**Returns**: Dictionary with raw and calibrated predictions (both `np.ndarray`)

**Key Feature**: Dual output allows downstream systems to choose which predictions to use

### Handler Function 4: Response Formatting

#### `output_fn(prediction_output, accept="application/json") -> Tuple[str, str]`

**Purpose**: Serialize predictions into response format based on Accept header

**Algorithm**:
```python
1. Extract raw and calibrated predictions:
   IF prediction_output is dict:  # NEW format
      raw_predictions = prediction_output["raw_predictions"]
      calibrated_predictions = prediction_output["calibrated_predictions"]
   ELSE:  # Legacy format (backward compatible)
      raw_predictions = prediction_output
      calibrated_predictions = prediction_output

2. Convert to list format:
   raw_scores_list = raw_predictions.tolist()
   calibrated_scores_list = calibrated_predictions.tolist()

3. Determine classification type:
   is_multiclass = len(raw_scores_list[0]) > 2

4. Format based on accept header:
   IF accept == "application/json":
      FOR each (raw_probs, cal_probs) pair:
         IF binary:
            record = {
               "legacy-score": str(raw_probs[1]),
               "calibrated-score": str(cal_probs[1]),
               "output-label": f"class-{max_idx}"
            }
         ELSE:  # multiclass
            record = {
               "prob_01": str(raw_probs[0]),
               "prob_02": str(raw_probs[1]),
               ...
               "calibrated_prob_01": str(cal_probs[0]),
               "calibrated_prob_02": str(cal_probs[1]),
               ...
               "output-label": f"class-{max_idx}"
            }
      Return (json.dumps({"predictions": records}), "application/json")
   
   ELIF accept == "text/csv":
      FOR each (raw_probs, cal_probs) pair:
         Format as: raw_str,cal_str,class-label
      Return (csv_string, "text/csv")
   
   ELSE:
      Raise ValueError("Unsupported accept type")
```

**Returns**: `Tuple[response_body: str, content_type: str]`

**Output Format Design**:
- Binary: `legacy-score` (raw) + `calibrated-score` for backward compatibility
- Multiclass: Separate columns for raw and calibrated per class
- CSV: Comma-separated values for both raw and calibrated

## Calibration Integration Architecture

### Calibration Model Loading

#### `load_calibration_model(model_dir) -> Optional[Dict]`

**Purpose**: Load calibration model from various formats

**Algorithm**:
```python
CALIBRATION_DIR = "calibration"

# Priority 1: Percentile calibration
percentile_path = model_dir / CALIBRATION_DIR / "percentile_score.pkl"
IF percentile_path exists:
   Load percentile mapping
   RETURN {"type": "percentile", "data": mapping}

# Priority 2: Binary calibration
binary_path = model_dir / CALIBRATION_DIR / "calibration_model.pkl"
IF binary_path exists:
   Load binary calibration lookup table
   RETURN {"type": "regular", "data": calibrator}

# Priority 3: Multiclass calibration
multiclass_dir = model_dir / CALIBRATION_DIR / "calibration_models"
IF multiclass_dir exists AND is directory:
   calibrators = {}
   FOR each .pkl file in directory:
      class_name = extract_class_from_filename(file)
      calibrators[class_name] = load_pkl(file)
   IF calibrators:
      RETURN {"type": "regular_multiclass", "data": calibrators}

# No calibration found
RETURN None
```

**Returns**: Dict with `"type"` and `"data"` keys, or `None`

**Supported Types**:
1. **Percentile**: Maps raw scores to percentile ranks
2. **Regular Binary**: Lookup table for binary classification
3. **Regular Multiclass**: Per-class lookup tables

### Calibration Application

#### `apply_calibration(scores, calibrator, is_multiclass) -> np.ndarray`

**Purpose**: Apply appropriate calibration method to raw scores

**Algorithm**:
```python
IF calibrator is None:
   RETURN scores  # No calibration

IF calibrator["type"] == "percentile":
   IF is_multiclass:
      Log warning: percentile not supported for multiclass
      RETURN scores
   ELSE:
      RETURN apply_percentile_calibration(scores, calibrator["data"])

ELIF calibrator["type"] == "regular":
   RETURN apply_regular_binary_calibration(scores, calibrator["data"])

ELIF calibrator["type"] == "regular_multiclass":
   RETURN apply_regular_multiclass_calibration(scores, calibrator["data"])

ELSE:
   Log warning: unknown calibrator type
   RETURN scores
```

**Error Handling**: Gracefully falls back to raw scores on calibration failure

#### `apply_percentile_calibration(scores, mapping) -> np.ndarray`

**Purpose**: Map raw class-1 probabilities to percentile ranks

**Algorithm**:
```python
1. Create output array (same shape as input)
2. FOR each sample i:
      raw_class1_prob = scores[i, 1]
      calibrated_class1 = interpolate_score(raw_class1_prob, mapping)
      calibrated[i, 1] = calibrated_class1
      calibrated[i, 0] = 1 - calibrated_class1  # Complement
3. RETURN calibrated
```

**Use Case**: Convert raw probabilities to percentile ranks for better discrimination

#### `apply_regular_binary_calibration(scores, calibrator) -> np.ndarray`

**Purpose**: Apply lookup table calibration to binary scores

**Algorithm**:
```python
1. Create output array (same shape as input)
2. FOR each sample i:
      raw_class1_prob = scores[i, 1]
      calibrated_class1 = interpolate_score(raw_class1_prob, calibrator)
      calibrated[i, 1] = calibrated_class1
      calibrated[i, 0] = 1 - calibrated_class1
3. RETURN calibrated
```

**Calibrator Format**: List of `(raw_score, calibrated_score)` tuples

#### `apply_regular_multiclass_calibration(scores, calibrators) -> np.ndarray`

**Purpose**: Apply per-class calibration and renormalize

**Algorithm**:
```python
1. Create output array (same shape as input)
2. FOR each class i:
      class_name = str(i)
      IF class_name in calibrators:
         FOR each sample j:
            calibrated[j, i] = interpolate_score(
               scores[j, i], 
               calibrators[class_name]
            )
      ELSE:
         calibrated[:, i] = scores[:, i]  # No calibrator
3. Normalize to sum to 1:
   row_sums = calibrated.sum(axis=1)
   calibrated = calibrated / row_sums[:, np.newaxis]
4. RETURN calibrated
```

**Key Step**: Renormalization ensures probabilities sum to 1.0 after per-class calibration

#### `_interpolate_score(raw_score, lookup_table) -> float`

**Purpose**: Linear interpolation between lookup table points

**Algorithm**:
```python
# Boundary cases
IF raw_score <= lookup_table[0][0]:
   RETURN lookup_table[0][1]
IF raw_score >= lookup_table[-1][0]:
   RETURN lookup_table[-1][1]

# Find bracketing points
FOR i in range(len(lookup_table) - 1):
   IF lookup_table[i][0] <= raw_score <= lookup_table[i+1][0]:
      x1, y1 = lookup_table[i]
      x2, y2 = lookup_table[i+1]
      IF x2 == x1:
         RETURN y1
      # Linear interpolation
      RETURN y1 + (y2 - y1) * (raw_score - x1) / (x2 - x1)

RETURN lookup_table[-1][1]
```

**Complexity**: O(n) where n = number of points in lookup table

## Hyperparameter-Driven Text Pipeline Reconstruction

### Algorithm: Build Text Pipelines from Hyperparameters

**Problem**: Inference must use exact same preprocessing as training

**Solution**: Store processing steps in hyperparameters.json during training, reconstruct during inference

**Algorithm**:
```python
1. Load hyperparameters from hyperparameters.json
2. Extract pipeline configuration:
   - text_processing_steps (for bimodal)
   - primary_text_processing_steps (for trimodal primary)
   - secondary_text_processing_steps (for trimodal secondary)
   - text_name / primary_text_name / secondary_text_name

3. Determine architecture:
   IF primary_text_name exists:
      architecture = "trimodal"
   ELSE:
      architecture = "bimodal"

4. Build pipelines based on architecture:
   
   IF bimodal:
      steps = hyperparameters.get("text_processing_steps", default_steps)
      pipeline = build_text_pipeline_from_steps(
         processing_steps=steps,
         tokenizer=tokenizer,
         max_sen_len=config.max_sen_len,
         ...
      )
      pipelines[config.text_name] = pipeline
   
   ELIF trimodal:
      # Primary text pipeline (e.g., chat - full cleaning)
      primary_steps = hyperparameters.get(
         "primary_text_processing_steps", 
         default_full_steps
      )
      pipelines[primary_text_name] = build_text_pipeline_from_steps(
         processing_steps=primary_steps,
         ...
      )
      
      # Secondary text pipeline (e.g., events - minimal cleaning)
      secondary_steps = hyperparameters.get(
         "secondary_text_processing_steps",
         default_minimal_steps
      )
      pipelines[secondary_text_name] = build_text_pipeline_from_steps(
         processing_steps=secondary_steps,
         ...
      )

5. RETURN pipelines dict
```

**Default Steps**:
- **Full cleaning**: `['dialogue_splitter', 'html_normalizer', 'emoji_remover', 'text_normalizer', 'dialogue_chunker', 'tokenizer']`
- **Minimal cleaning**: `['dialogue_splitter', 'text_normalizer', 'dialogue_chunker', 'tokenizer']`

**Key Benefit**: Guarantees preprocessing consistency between training and inference

## Performance Characteristics

### Latency Analysis

| Component | Latency (typical) | Notes |
|-----------|------------------|-------|
| Model Loading (cold start) | 2-5 seconds | One-time cost per endpoint instance |
| Hyperparameter Loading | 10-50 ms | One-time, small JSON file |
| Calibration Loading | 10-100 ms | One-time, depends on model size |
| Request Parsing | 1-10 ms | Per request, depends on input size |
| Text Tokenization | 5-50 ms | Per request, depends on text length |
| Preprocessing | 2-20 ms | Per request, risk tables + imputation |
| Model Forward Pass | 10-100 ms | Per request, depends on model size |
| Calibration Application | 1-5 ms | Per request, lookup table interpolation |
| Response Formatting | 1-5 ms | Per request, JSON serialization |
| **Total (warm)** | **20-200 ms** | Typical P50-P99 latency range |

### Throughput Analysis

| Configuration | Throughput (req/s) | Notes |
|--------------|-------------------|-------|
| Single instance, CPU | 5-20 | Depends on model size |
| Single instance, GPU (T4) | 20-50 | GPU accelerates model inference |
| Single instance, GPU (A10G) | 50-100 | Faster GPU improves throughput |
| 3 instances, GPU (T4) | 60-150 | Linear scaling with instances |
| Batch inference (batch=32) | 2-5× faster | Batch requests when possible |

### Memory Requirements

**Endpoint Instance Memory**:
```
Total = Model + Framework + Preprocessing Artifacts + Request Buffer

Example (BimodalBERT):
- Model weights: 150-400 MB
- PyTorch/Lightning: 200-500 MB
- Tokenizer: 50-100 MB
- Risk tables: 5-20 MB
- Imputation dict: 1-5 MB
- Calibration models: 1-10 MB
- Request buffer: 10-50 MB (per concurrent request)
- GPU memory (if used): 2-4 GB

Total: ~400-600 MB RAM (cold) + ~50-100 MB per request
GPU Total: 2-4 GB VRAM
```

**Recommended Instance Types**:
- **CPU**: ml.m5.xlarge (4 vCPU, 16 GB RAM) - $0.23/hr
- **GPU**: ml.g4dn.xlarge (1 T4 GPU, 4 vCPU, 16 GB RAM) - $0.74/hr
- **GPU**: ml.g5.xlarge (1 A10G GPU, 4 vCPU, 16 GB RAM) - $1.01/hr

## Error Handling

### Model Loading Errors

**Missing Hyperparameters File**:
- **Symptom**: `hyperparameters.json not found`
- **Handling**: Logs warning, uses default pipeline steps
- **Impact**: May use incorrect preprocessing if model trained with custom steps

**Calibration Loading Failure**:
- **Symptom**: Error loading calibration pickle files
- **Handling**: Logs warning, continues without calibration
- **Impact**: Returns only raw predictions (no calibrated predictions)

### Request Processing Errors

**Invalid Content-Type**:
- **Symptom**: HTTP 400 Bad Request
- **Response**: `{"error": "Unsupported content type"}`
- **User Action**: Use text/csv, application/json, or application/x-parquet

**Malformed JSON**:
- **Symptom**: JSON parsing error
- **Response**: `{"error": "Invalid JSON format"}`
- **User Action**: Validate JSON syntax

**Missing Required Columns**:
- **Symptom**: KeyError during preprocessing
- **Response**: `{"error": "Missing required feature columns"}`
- **User Action**: Include all features used during training

### Prediction Errors

**CUDA Out of Memory**:
- **Symptom**: torch.cuda.OutOfMemoryError
- **Handling**: Endpoint restarts (health check failure)
- **User Action**: Use larger instance or reduce batch size

**Calibration Application Failure**:
- **Symptom**: Error during calibration
- **Handling**: Logs warning, falls back to raw predictions
- **Impact**: Returns raw predictions only

## Best Practices

### For Production Deployments

1. **Use GPU Instances for Large Models**:
   ```python
   # In endpoint config
   instance_type = "ml.g4dn.xlarge"  # T4 GPU
   instance_type = "ml.g5.xlarge"    # A10G GPU (better)
   ```

2. **Enable Auto-Scaling**:
   ```python
   # Scale based on invocation metrics
   min_instances = 2
   max_instances = 10
   target_invocations_per_instance = 50
   ```

3. **Use Model Calibration**:
   - Train calibration model separately
   - Include in model.tar.gz
   - Improves probability quality

4. **Monitor Endpoint Metrics**:
   - Invocations per minute
   - Model latency (P50, P99)
   - CPU/GPU utilization
   - 4XX/5XX error rates

### For Cost Optimization

1. **Use CPU for Small Models**:
   - Trimodal models → GPU
   - Bimodal with small text → CPU may suffice

2. **Use Serverless Inference** (if available):
   - Pay per invocation
   - No idle costs
   - Cold start latency acceptable

3. **Batch Requests When Possible**:
   - Send multiple records in single request
   - Amortize overhead costs

### For Development and Testing

1. **Test with Local Endpoint**:
   ```bash
   # Use SageMaker local mode
   from sagemaker.local import LocalSession
   local_session = LocalSession()
   ```

2. **Validate Preprocessing**:
   - Compare inference preprocessing with training
   - Check hyperparameters.json contents

3. **Test Calibration**:
   - Verify calibration files loaded
   - Compare raw vs calibrated outputs

## Example Usage Patterns

### Example 1: Single Record JSON Request
```python
import requests
import json

endpoint_url = "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-pytorch-endpoint/invocations"

# Request payload
data = {
    "customer_id": 12345,
    "text": "Hello, I need help with my order",
    "feature1": 5.2,
    "feature2": "category_A",
    "feature3": 3
}

# Make request
response = requests.post(
    endpoint_url,
    data=json.dumps(data),
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
)

print(response.json())
# {
#   "predictions": [{
#     "legacy-score": "0.85",
#     "calibrated-score": "0.78",
#     "output-label": "class-1"
#   }]
# }
```

### Example 2: Batch Request CSV
```python
import requests

endpoint_url = "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-pytorch-endpoint/invocations"

# CSV payload (NO HEADER)
csv_data = """12345,Hello customer service...,5.2,category_A,3
67890,I need help with...,8.1,category_B,7"""

response = requests.post(
    endpoint_url,
    data=csv_data,
    headers={
        "Content-Type": "text/csv",
        "Accept": "text/csv"
    }
)

# Parse CSV response
predictions = response.text.strip().split('\n')
print(f"Predicted {len(predictions)} records")
```

### Example 3: Testing Calibration
```python
import requests
import json

endpoint_url = "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-pytorch-endpoint/invocations"

data = {"customer_id": 12345, "text": "Test message", "feature1": 5.2}

response = requests.post(
    endpoint_url,
    data=json.dumps(data),
    headers={"Content-Type": "application/json", "Accept": "application/json"}
)

result = response.json()["predictions"][0]
raw_score = float(result["legacy-score"])
calibrated_score = float(result["calibrated-score"])

print(f"Raw score: {raw_score:.4f}")
print(f"Calibrated score: {calibrated_score:.4f}")
print(f"Calibration delta: {calibrated_score - raw_score:.4f}")
```

## Integration Patterns

### Upstream Integration

```
PyTorchTraining
   ↓ (outputs: model.pth, model_artifacts.pth, hyperparameters.json, risk_table_map.pkl, impute_dict.pkl)
[Optional] ModelCalibration
   ↓ (outputs: calibration/calibration_model.pkl or percentile_score.pkl)
PyTorchInferenceHandler (Endpoint)
   ↓ (outputs: real-time predictions with raw + calibrated scores)
```

**Training Output → Inference Input**:
- Model weights: `model.pth`, `model_artifacts.pth`
- Preprocessing artifacts: `risk_table_map.pkl`, `impute_dict.pkl`
- Pipeline configuration: `hyperparameters.json` (CRITICAL)
- Optional calibration: `calibration/*.pkl`

### Downstream Integration

```
PyTorchInferenceHandler (Endpoint)
   ↓ (outputs: JSON/CSV with raw + calibrated predictions)
[Branch 1] → Business Logic → Decision System
[Branch 2] → Monitoring → Alerting
[Branch 3] → Data Lake → Analysis
```

### Complete Pipeline Flow

```
DataPreprocessing → PyTorchTraining → PyTorchModelEvaluation →
[Optional] ModelCalibration → Package → Registration → Endpoint Deployment →
PyTorchInferenceHandler → Real-Time Serving
```

**Key Integration Points**:
1. **Training → Calibration**: Predictions with labels for calibration training
2. **Calibration → Packaging**: Calibrated model + original model
3. **Packaging → Endpoint**: model.tar.gz with all artifacts
4. **Endpoint → Applications**: Real-time predictions via HTTP/HTTPS

### Comparison with XGBoost Handler

| Feature | PyTorch Handler | XGBoost Handler |
|---------|----------------|-----------------|
| **Model Format** | model.pth + model_artifacts.pth | xgboost_model.bst |
| **Preprocessing** | Hyperparameter-driven pipelines | Risk tables + imputation |
| **Text Support** | Full text pipelines (tokenization) | Not applicable |
| **Architecture** | Bimodal, trimodal, text-only | Tabular only |
| **Calibration** | Lookup tables (binary, multiclass, percentile) | Same as PyTorch |
| **Fast Path** | Not implemented | 50-100x speedup for single records |
| **Typical Latency** | 20-200 ms | 2-10 ms (with fast path) |
| **Use Case** | Multi-modal, text-heavy | Pure tabular features |

## Related Documentation

- [PyTorch Training Script](pytorch_training_script.md) - Training implementation
- [PyTorch Model Inference Script](pytorch_model_inference_script.md) - Batch inference
- [PyTorch Inference Calibration Integration](../1_design/pytorch_inference_calibration_integration.md) - Calibration design
- [XGBoost Inference Handler](xgboost_inference_handler_script.md) - XGBoost equivalent
- [Model Architecture Design Index](../00_entry_points/model_architecture_design_index.md) - Model architectures

---

## Maintenance Notes

**Last Updated:** 2025-11-28

**Update Triggers**:
- New calibration methods
- Hyperparameter format changes
- Pipeline preprocessing updates
- Performance optimizations

**Maintenance Guidelines**:
- Keep calibration logic synced with XGBoost handler
- Update hyperparameter examples when format changes
- Document new preprocessing steps
- Track performance benchmarks
