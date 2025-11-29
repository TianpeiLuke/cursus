---
tags:
  - design
  - implementation
  - pytorch
  - model_inference
  - calibration
  - lookup_table
keywords:
  - pytorch inference
  - model calibration
  - calibration integration
  - lookup table
  - probability calibration
  - inference handler
topics:
  - model inference
  - calibration
  - pytorch
  - prediction output
language: python
date of note: 2025-11-28
---

# PyTorch Inference Handler Calibration Integration Design

## What is the Purpose of PyTorch Inference Calibration Integration?

The PyTorch Inference Handler Calibration Integration adds **model probability calibration** support to PyTorch inference handlers, bringing them into alignment with the XGBoost inference handler's comprehensive calibration system. Well-calibrated probabilities are essential for risk-based decision-making, threshold optimization, and reliable uncertainty quantification in production ML systems.

The core purpose is to:
1. **Load calibration models** automatically when available in model artifacts
2. **Apply calibration** to raw model predictions using efficient lookup tables
3. **Output dual scores** - both raw and calibrated probabilities for downstream analysis
4. **Maintain compatibility** - Zero breaking changes to existing inference behavior
5. **Support multiple calibration methods** - Percentile, isotonic, GAM, and Platt scaling
6. **Enable production reliability** - Consistent probability calibration across all model types

## Core Design Principles

The PyTorch Inference Calibration Integration is built on several key design principles:

1. **Backward Compatibility** - Works seamlessly without calibration files (no-op mode)
2. **Unified Interface** - Same calibration system as XGBoost handler for consistency
3. **Lookup Table Optimization** - Fast inference using pre-computed lookup tables (~2-5 μs per prediction)
4. **Graceful Degradation** - Falls back to raw scores on calibration errors
5. **Format Consistency** - Matches XGBoost output format for downstream compatibility
6. **Zero Dependencies** - No sklearn/pygam required at inference time

## Current State Analysis

### XGBoost Inference Handler (`projects/atoz_xgboost/docker/xgboost_inference_handler.py`)

**Artifacts Loaded in `model_fn()`:**
```python
✅ xgboost_model.bst              # Trained model
✅ risk_table_map.pkl             # Categorical mappings
✅ impute_dict.pkl                # Numerical imputation
✅ feature_columns.txt            # Ordered features (CRITICAL!)
✅ feature_importance.json        # Optional
✅ hyperparameters.json           # Optional

✅ calibration/                   # Calibration directory
   ├── percentile_score.pkl       # Percentile calibration
   ├── calibration_model.pkl      # Binary calibration (lookup table)
   └── calibration_models/        # Multiclass calibration
       ├── calibration_model_class_0.pkl
       ├── calibration_model_class_1.pkl
       └── ...
```

**Calibration System Features:**

1. **Loading** (`load_calibration_model()`):
   - Checks `calibration/` subdirectory
   - Supports multiple formats:
     - Percentile: `percentile_score.pkl`
     - Binary: `calibration_model.pkl`
     - Multiclass: `calibration_models/calibration_model_class_X.pkl`
   - Returns structured dict: `{"type": "percentile|regular|regular_multiclass", "data": model_or_mapping}`

2. **Application** (`apply_calibration()`):
   - Handles percentile calibration via lookup table interpolation
   - Supports regular calibration (isotonic, GAM, Platt) via lookup tables
   - Supports multiclass with per-class calibration
   - Graceful fallback to raw scores on error

3. **Output Format**:
   - **Binary**: Returns both raw and calibrated in `predict_fn()`
     ```python
     return {
         "raw_predictions": raw_predictions,
         "calibrated_predictions": calibrated_predictions
     }
     ```
   - **Output fields**:
     - `legacy-score`: Raw class-1 probability
     - `score-percentile`: Calibrated score (descriptive name)
     - `calibrated-score`: Calibrated score
     - `custom-output-label`: Predicted class

4. **Lookup Table Format**:
   - All calibration models are converted to lookup tables: `List[Tuple[float, float]]`
   - Format: `[(raw_score_1, calibrated_score_1), (raw_score_2, calibrated_score_2), ...]`
   - Uses linear interpolation for fast inference (~2-5 μs per prediction)
   - No model object dependencies at inference time

### PyTorch Inference Handler (`projects/rnr_pytorch_bedrock/docker/pytorch_inference_handler.py`)

**Current Artifacts Loaded:**
```python
✅ model.pth                      # Model weights
✅ model_artifacts.pth            # Config, embeddings, vocab, model_class
✅ model.onnx (optional)          # ONNX model

❌ NO calibration loading
❌ NO preprocessing artifacts (risk tables, imputation)
❌ NO feature_columns.txt
```

**Missing Components:**
- No calibration model loading
- No calibration application
- No dual output (raw + calibrated)
- No preprocessing artifact loading (relies on config reconstruction)
- Single output format (raw predictions only)

### Model Calibration Script (`src/cursus/steps/scripts/model_calibration.py`)

**Key Findings:**

1. **Calibration Model Storage Format**:
   - All models (GAM, Isotonic, Platt) are **converted to lookup tables** before saving
   - Function: `_model_to_lookup_table(model, method, config)`
   - Sample points: Configurable via `CALIBRATION_SAMPLE_POINTS` (default: 1000)
   - Output format: `List[Tuple[float, float]]` - same as percentile calibration

2. **Benefits of Lookup Table Format**:
   - **Fast inference**: ~2-5 μs per prediction (vs. ~100-500 μs for model objects)
   - **No dependencies**: No need for pygam, sklearn at inference time
   - **Unified interface**: Same code handles all calibration methods
   - **Easy to serialize**: Simple pickle of Python list

3. **Calibration Output Structure**:
   ```
   /opt/ml/processing/output/calibration/
   ├── calibration_model.pkl              # Binary: lookup table
   ├── calibration_summary.json           # Metadata
   └── calibration_models/                # Multiclass
       ├── calibration_model_class_0.pkl  # Per-class lookup tables
       ├── calibration_model_class_1.pkl
       └── ...
   ```

4. **Multi-task Support**:
   - Supports multiple score fields via `SCORE_FIELDS` environment variable
   - Each task gets its own calibrator: `calibration_model_{score_field}.pkl`
   - Binary classification only for multi-task mode

## Architecture Comparison

| Aspect | XGBoost Handler | PyTorch Handler | Gap |
|--------|----------------|-----------------|-----|
| **Preprocessing** | Load from pkl files | Rebuild from config | ⚠️ Different approach |
| **Feature Order** | feature_columns.txt | Derived from config | ⚠️ Different approach |
| **Calibration Loading** | ✅ Comprehensive | ❌ Missing | 🔴 Critical gap |
| **Calibration Application** | ✅ With fallback | ❌ Missing | 🔴 Critical gap |
| **Output Format** | Raw + Calibrated | Raw only | 🔴 Critical gap |
| **Lookup Tables** | ✅ Used | ❌ N/A | 🔴 Critical gap |
| **Multi-task** | ✅ Supported | ❌ N/A | 🟡 Feature gap |

## Implementation Plan

### Phase 1: Add Calibration Loading (High Priority)

**Goal**: Load calibration models if available in model directory.

**Changes to `model_fn()`:**

```python
def model_fn(model_dir: str) -> Dict[str, Any]:
    # ... existing model loading ...
    
    # Load calibration model if available
    calibrator = load_calibration_model(model_dir)
    if calibrator:
        logger.info("Calibration model loaded successfully")
    
    return {
        "model": model,
        "config": config,
        "embedding_mat": embedding_mat,
        "vocab": vocab,
        "model_class": model_class,
        "pipelines": pipelines,
        "calibrator": calibrator,  # ← ADD THIS
    }
```

**New Function** (copy from XGBoost handler):

```python
def load_calibration_model(model_dir: str) -> Optional[Dict]:
    """
    Load calibration model if it exists. Supports both regular calibration models
    (calibration_model.pkl) and percentile calibration (percentile_score.pkl).
    
    Args:
        model_dir: Directory containing model artifacts
    
    Returns:
        Calibration model if found, None otherwise. Returns a dictionary with
        'type' and 'data' keys.
    """
    import pickle as pkl
    
    # Define calibration file constants
    CALIBRATION_DIR = "calibration"
    CALIBRATION_MODEL_FILE = "calibration_model.pkl"
    PERCENTILE_SCORE_FILE = "percentile_score.pkl"
    CALIBRATION_MODELS_DIR = "calibration_models"
    
    # Check for percentile calibration first
    percentile_path = os.path.join(model_dir, CALIBRATION_DIR, PERCENTILE_SCORE_FILE)
    if os.path.exists(percentile_path):
        logger.info(f"Loading percentile calibration from {percentile_path}")
        try:
            with open(percentile_path, "rb") as f:
                percentile_mapping = pkl.load(f)
                return {"type": "percentile", "data": percentile_mapping}
        except Exception as e:
            logger.warning(f"Failed to load percentile calibration: {e}")
    
    # Check for binary calibration model
    calibration_path = os.path.join(model_dir, CALIBRATION_DIR, CALIBRATION_MODEL_FILE)
    if os.path.exists(calibration_path):
        logger.info(f"Loading binary calibration model from {calibration_path}")
        try:
            with open(calibration_path, "rb") as f:
                return {"type": "regular", "data": pkl.load(f)}
        except Exception as e:
            logger.warning(f"Failed to load binary calibration model: {e}")
    
    # Check for multiclass calibration models
    multiclass_dir = os.path.join(model_dir, CALIBRATION_DIR, CALIBRATION_MODELS_DIR)
    if os.path.exists(multiclass_dir) and os.path.isdir(multiclass_dir):
        logger.info(f"Loading multiclass calibration models from {multiclass_dir}")
        try:
            calibrators = {}
            for file in os.listdir(multiclass_dir):
                if file.endswith(".pkl"):
                    class_name = file.replace("calibration_model_class_", "").replace(".pkl", "")
                    with open(os.path.join(multiclass_dir, file), "rb") as f:
                        calibrators[class_name] = pkl.load(f)
            if calibrators:
                return {"type": "regular_multiclass", "data": calibrators}
        except Exception as e:
            logger.warning(f"Failed to load multiclass calibration models: {e}")
    
    logger.info("No calibration model found")
    return None
```

### Phase 2: Add Calibration Application (High Priority)

**Changes to `predict_fn()`:**

```python
def predict_fn(input_object, model_data, context=None):
    # ... existing preprocessing and prediction ...
    
    # Generate raw predictions
    raw_probs = model_online_inference(model, predict_dataloader)
    
    # Apply calibration if available
    calibrator = model_data.get("calibrator")
    if calibrator:
        try:
            calibrated_probs = apply_calibration(
                raw_probs, 
                calibrator, 
                config.is_binary
            )
            logger.info("Applied calibration to predictions")
        except Exception as e:
            logger.warning(f"Failed to apply calibration: {e}")
            calibrated_probs = raw_probs.copy()
    else:
        logger.info("No calibration model available, using raw predictions")
        calibrated_probs = raw_probs.copy()
    
    return {
        "raw_predictions": raw_probs,
        "calibrated_predictions": calibrated_probs
    }
```

**New Functions** (adapted from XGBoost handler):

```python
def _interpolate_score(raw_score: float, lookup_table: List[Tuple[float, float]]) -> float:
    """
    Interpolate calibrated score from lookup table.
    
    Args:
        raw_score: Raw model score (0-1)
        lookup_table: List of (raw_score, calibrated_score) tuples
    
    Returns:
        Interpolated calibrated score
    """
    # Boundary cases
    if raw_score <= lookup_table[0][0]:
        return lookup_table[0][1]
    if raw_score >= lookup_table[-1][0]:
        return lookup_table[-1][1]
    
    # Find bracketing points and perform linear interpolation
    for i in range(len(lookup_table) - 1):
        if lookup_table[i][0] <= raw_score <= lookup_table[i + 1][0]:
            x1, y1 = lookup_table[i]
            x2, y2 = lookup_table[i + 1]
            if x2 == x1:
                return y1
            return y1 + (y2 - y1) * (raw_score - x1) / (x2 - x1)
    
    return lookup_table[-1][1]


def apply_percentile_calibration(
    scores: np.ndarray, 
    percentile_mapping: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Apply percentile score mapping to raw scores.
    
    Args:
        scores: Raw model prediction scores (N x 2 for binary classification)
        percentile_mapping: List of (raw_score, percentile) tuples
    
    Returns:
        Calibrated scores with same shape as input
    """
    # Apply percentile calibration to class-1 probabilities
    calibrated = np.zeros_like(scores)
    for i in range(scores.shape[0]):
        raw_class1_prob = scores[i, 1]
        calibrated_class1_prob = _interpolate_score(raw_class1_prob, percentile_mapping)
        calibrated[i, 1] = calibrated_class1_prob
        calibrated[i, 0] = 1 - calibrated_class1_prob
    
    return calibrated


def apply_regular_binary_calibration(
    scores: np.ndarray, 
    calibrator: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Apply regular calibration to binary classification scores using lookup table.
    
    Args:
        scores: Raw model prediction scores (N x 2)
        calibrator: Lookup table List[Tuple[float, float]]
    
    Returns:
        Calibrated scores with same shape as input
    """
    calibrated = np.zeros_like(scores)
    
    # Apply lookup table calibration
    for i in range(scores.shape[0]):
        calibrated[i, 1] = _interpolate_score(scores[i, 1], calibrator)
        calibrated[i, 0] = 1 - calibrated[i, 1]
    
    return calibrated


def apply_regular_multiclass_calibration(
    scores: np.ndarray,
    calibrators: Dict[str, List[Tuple[float, float]]]
) -> np.ndarray:
    """
    Apply regular calibration to multiclass scores.
    
    Args:
        scores: Raw model prediction scores (N x num_classes)
        calibrators: Dictionary of calibration lookup tables, one per class
    
    Returns:
        Calibrated and normalized scores with same shape as input
    """
    calibrated = np.zeros_like(scores)
    
    # Apply calibration to each class
    for i in range(scores.shape[1]):
        class_name = str(i)
        if class_name in calibrators:
            for j in range(scores.shape[0]):
                calibrated[j, i] = _interpolate_score(scores[j, i], calibrators[class_name])
        else:
            calibrated[:, i] = scores[:, i]  # No calibrator for this class
    
    # Normalize probabilities to sum to 1
    row_sums = calibrated.sum(axis=1)
    calibrated = calibrated / row_sums[:, np.newaxis]
    
    return calibrated


def apply_calibration(
    scores: np.ndarray,
    calibrator: Dict[str, Any],
    is_multiclass: bool
) -> np.ndarray:
    """
    Apply calibration to raw model scores. Supports both regular calibration models
    and percentile calibration.
    
    Args:
        scores: Raw model prediction scores
        calibrator: Loaded calibration model(s) or percentile mapping
        is_multiclass: Whether this is a multiclass model
    
    Returns:
        Calibrated scores
    """
    if calibrator is None:
        return scores
    
    try:
        # Handle percentile calibration
        if calibrator.get("type") == "percentile":
            if is_multiclass:
                logger.warning("Percentile calibration not supported for multiclass, using raw scores")
                return scores
            else:
                logger.info("Applying percentile calibration")
                return apply_percentile_calibration(scores, calibrator["data"])
        
        # Handle regular calibration models
        elif calibrator.get("type") in ["regular", "regular_multiclass"]:
            actual_calibrator = calibrator["data"]
            
            if calibrator.get("type") == "regular_multiclass" or is_multiclass:
                logger.info("Applying regular multiclass calibration")
                return apply_regular_multiclass_calibration(scores, actual_calibrator)
            else:
                logger.info("Applying regular binary calibration")
                return apply_regular_binary_calibration(scores, actual_calibrator)
        
        else:
            logger.warning(f"Unknown calibrator type: {calibrator.get('type')}")
            return scores
    
    except Exception as e:
        logger.error(f"Error applying calibration: {str(e)}", exc_info=True)
        return scores
```

### Phase 3: Update Output Format (High Priority)

**Changes to `output_fn()`:**

```python
def output_fn(prediction_output, accept="application/json"):
    """
    Serializes the multi-class prediction output with both raw and calibrated scores.
    
    Args:
        prediction_output: Dict with "raw_predictions" and "calibrated_predictions"
        accept: The requested response MIME type
    
    Returns:
        tuple: (response_body, content_type)
    """
    logger.info(f"Received prediction output for accept type: {accept}")
    
    # Extract raw and calibrated predictions
    if isinstance(prediction_output, dict):
        raw_predictions = prediction_output.get("raw_predictions")
        calibrated_predictions = prediction_output.get("calibrated_predictions")
    else:
        # Backward compatibility: treat as raw predictions
        raw_predictions = prediction_output
        calibrated_predictions = prediction_output
    
    # Convert to list format
    raw_scores_list = raw_predictions.tolist() if isinstance(raw_predictions, np.ndarray) else raw_predictions
    calibrated_scores_list = calibrated_predictions.tolist() if isinstance(calibrated_predictions, np.ndarray) else calibrated_predictions
    
    # Ensure list of lists format
    if not isinstance(raw_scores_list[0], list):
        raw_scores_list = [[score] for score in raw_scores_list]
        calibrated_scores_list = [[score] for score in calibrated_scores_list]
    
    is_multiclass = len(raw_scores_list[0]) > 2
    
    # Format output based on accept type
    if accept.lower() == "application/json":
        return format_json_response(raw_scores_list, calibrated_scores_list, is_multiclass)
    elif accept.lower() == "text/csv":
        return format_csv_response(raw_scores_list, calibrated_scores_list, is_multiclass)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


def format_json_response(raw_scores_list, calibrated_scores_list, is_multiclass):
    """Format predictions as JSON with both raw and calibrated scores."""
    output_records = []
    
    for raw_probs, cal_probs in zip(raw_scores_list, calibrated_scores_list):
        max_idx = raw_probs.index(max(raw_probs))
        
        if not is_multiclass:
            # Binary classification
            record = {
                "legacy-score": str(raw_probs[1]),  # Raw class-1
                "calibrated-score": str(cal_probs[1]),  # Calibrated class-1
                "output-label": f"class-{max_idx}"
            }
        else:
            # Multiclass
            record = {}
            for i in range(len(raw_probs)):
                record[f"prob_{str(i+1).zfill(2)}"] = str(raw_probs[i])
                record[f"calibrated_prob_{str(i+1).zfill(2)}"] = str(cal_probs[i])
            record["output-label"] = f"class-{max_idx}"
        
        output_records.append(record)
    
    response = json.dumps({"predictions": output_records})
    return response, "application/json"
```

### Phase 4: Testing and Validation (High Priority)

**Test Cases:**

1. **Without Calibration**:
   - Model directory with no `calibration/` subdirectory
   - Should work as before, returning raw predictions as both raw and calibrated

2. **With Percentile Calibration**:
   - Model directory with `calibration/percentile_score.pkl`
   - Should apply percentile calibration and return dual outputs

3. **With Regular Binary Calibration**:
   - Model directory with `calibration/calibration_model.pkl`
   - Should apply lookup table calibration

4. **With Multiclass Calibration**:
   - Model directory with `calibration/calibration_models/` subdirectory
   - Should apply per-class calibration and normalize

5. **Error Handling**:
   - Corrupted calibration file → Should fall back to raw scores
   - Missing calibration file → Should use raw scores
   - Invalid calibration format → Should log warning and continue

## File Locations to Modify

1. **Primary Implementation**:
   - `projects/rnr_pytorch_bedrock/docker/pytorch_inference_handler.py`
   - `projects/bsm_pytorch/docker/pytorch_inference_handler.py` (if exists)

2. **Reference Files** (DO NOT MODIFY - for copying code):
   - `projects/atoz_xgboost/docker/xgboost_inference_handler.py` (lines 200-450 for calibration code)
   - `src/cursus/steps/scripts/model_calibration.py` (for understanding calibration output format)

3. **Documentation**:
   - This file: `slipbox/1_design/pytorch_inference_calibration_integration.md`

## Dependencies

**New Imports Required:**
```python
import pickle as pkl
from typing import List, Tuple
```

**No Additional Packages Required:**
- Calibration uses lookup tables (no sklearn, pygam needed at inference)
- All dependencies already present in PyTorch handler

## Migration Path

### Backward Compatibility

The implementation maintains full backward compatibility:

1. **Without calibration files**: Works exactly as before
   - Returns raw predictions as both raw and calibrated
   - No changes to existing behavior

2. **With calibration files**: Enhanced functionality
   - Loads and applies calibration automatically
   - Returns dual outputs

3. **Output format**: Compatible with existing consumers
   - Binary: Adds `calibrated-score` field
   - Multiclass: Adds `calibrated_prob_XX` fields
   - Existing fields remain unchanged

### Deployment Strategy

1. **Phase 1**: Deploy calibration loading (no-op if files don't exist)
2. **Phase 2**: Update model training/calibration pipelines to generate calibration files
3. **Phase 3**: Verify calibrated outputs in production
4. **Phase 4**: Update downstream consumers to use calibrated scores

## Success Criteria

- ✅ PyTorch handler loads calibration models when available
- ✅ Calibration is applied correctly to predictions
- ✅ Both raw and calibrated scores are returned
- ✅ Output format matches XGBoost handler format
- ✅ Backward compatibility maintained
- ✅ Error handling prevents failures
- ✅ Performance impact < 10ms per request
- ✅ All test cases pass

## Future Enhancements

1. **Multi-task Support**: Support multiple score fields (like XGBoost)
2. **Preprocessing Artifacts**: Load risk tables and imputation dicts
3. **Feature Column Order**: Use feature_columns.txt for consistency
4. **Calibration Metrics**: Log calibration quality metrics
5. **A/B Testing**: Support dual calibration models for experimentation

## References

### Related Design Documents
- [XGBoost Model Inference Design](xgboost_model_inference_design.md) - Reference implementation for inference with calibration
- [PyTorch Model Eval Design](pytorch_model_eval_design.md) - PyTorch evaluation step design
- [Inference Handler Spec Design](inference_handler_spec_design.md) - General inference handler specifications
- [Multitask Percentile Model Calibration Design](multitask_percentile_model_calibration_design.md) - Multi-task calibration patterns

### Related Script Documentation
- [Model Calibration Script](../scripts/model_calibration_script.md) - Standard calibration implementation with lookup tables
- [Percentile Model Calibration Script](../scripts/percentile_model_calibration_script.md) - Percentile-based calibration
- [PyTorch Model Eval Script](../scripts/pytorch_model_eval_script.md) - PyTorch evaluation script implementation
- [PyTorch Model Inference Script](../scripts/pytorch_model_inference_script.md) - PyTorch inference script patterns
- [XGBoost Model Inference Script](../scripts/xgboost_model_inference_script.md) - XGBoost inference reference implementation

### Implementation Files
- XGBoost Inference Handler: `projects/atoz_xgboost/docker/xgboost_inference_handler.py`
- PyTorch Inference Handler: `projects/rnr_pytorch_bedrock/docker/pytorch_inference_handler.py`
- Model Calibration Script: `src/cursus/steps/scripts/model_calibration.py`
- Percentile Calibration Script: `src/cursus/steps/scripts/percentile_model_calibration.py`

### Related Contracts and Specs
- Model Calibration Contract: `src/cursus/steps/contracts/model_calibration_contract.py`
- Model Calibration Spec: `src/cursus/steps/specs/model_calibration_spec.py`
- Percentile Model Calibration Spec: `src/cursus/steps/specs/percentile_model_calibration_spec.py`
