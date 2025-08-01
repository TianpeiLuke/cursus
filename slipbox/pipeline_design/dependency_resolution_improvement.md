# Dependency Resolution Improvements

## Background

The pipeline system faced an issue when resolving dependencies between steps, specifically with model evaluation steps. The error manifested as:

```
AttributeError: 'dict' object has no attribute 'decode'
```

This occurred when SageMaker tried to parse a URI during execution, but received a dictionary object (property reference) instead of a string.

## Root Cause Analysis

### Issue 1: Conflicting Matches for Dependencies

The logs showed two conflicting matches for the evaluation step's input:

```
Matched model_evaluation.eval_data_input to xgboost_train.evaluation_output (score: 0.74)
Matched model_evaluation.eval_data_input to calib_preprocess.processed_data (score: 0.92)
```

While the second match had a higher score (0.92) and was the correct one, the presence of both matches created potential confusion in the dependency resolution system.

### Issue 2: Property References vs. String URIs

SageMaker's internal URL parser expected string URIs but received property reference objects that didn't have a `decode()` method.

## Solution

We implemented a multi-part solution:

1. **Removed Unused Outputs from Training Specs**:
   - Removed `training_job_name` and `metrics_output` from XGBoost training specification
   - Removed `checkpoints`, `training_job_name`, and `metrics_output` from PyTorch training specification
   - Kept essential outputs like `model_output` and data outputs

2. **Direct Logical Name Matching**:
   - Changed the logical name in `model_eval_spec.py` from `eval_data_input` to `processed_data` to match exactly with the preprocessing output name
   - This creates a direct match by name, eliminating the need for semantic matching or aliases

3. **Updated Script Contract**:
   - Modified `MODEL_EVALUATION_CONTRACT` to use the new logical name `processed_data` while maintaining the same container path `/opt/ml/processing/input/eval_data`

## Benefits

1. **Clearer Dependency Resolution**:
   - Direct matching of logical names makes the dependencies more explicit
   - Less reliance on semantic matching and scoring reduces ambiguity

2. **Simplified Specifications**:
   - Removed unused outputs that could cause confusion or unexpected matches
   - Cleaner, more focused specifications that better reflect the actual pipeline needs

3. **Improved Error Prevention**:
   - By ensuring direct matches, we reduce the chance of resolution errors
   - Property references are properly handled when passed between steps

## Technical Details

### Changed Files:
- `src/v2/pipeline_step_specs/model_eval_spec.py`
- `src/v2/pipeline_script_contracts/model_evaluation_contract.py`
- `src/v2/pipeline_step_specs/xgboost_training_spec.py`
- `src/v2/pipeline_step_specs/pytorch_training_spec.py`

### Key Changes:

```python
# Before - In model_eval_spec.py
DependencySpec(
    logical_name="eval_data_input",
    # ...
)

# After - In model_eval_spec.py
DependencySpec(
    logical_name="processed_data",
    # ...
)
```

```python
# Before - In model_evaluation_contract.py
expected_input_paths={
    "model_input": "/opt/ml/processing/input/model",
    "eval_data_input": "/opt/ml/processing/input/eval_data"
},

# After - In model_evaluation_contract.py
expected_input_paths={
    "model_input": "/opt/ml/processing/input/model",
    "processed_data": "/opt/ml/processing/input/eval_data"
},
```

This change ensures that the logical name in the model evaluation dependency spec exactly matches the output name from the preprocessing step, making the connection explicit and preventing issues with property reference handling.
