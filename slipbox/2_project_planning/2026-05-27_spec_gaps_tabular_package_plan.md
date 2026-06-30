# Plan: Fix Spec Gaps — TabularPreprocessing + Package `compatible_sources`

**Date**: 2026-05-27
**Status**: ✅ COMPLETE — commit `0dcf5eb`, build https://build.amazon.com/7856942542
**Motivation**: FZ 29d11a/29d11b identified 2 spec gaps that block dependency resolution in the Munged Address DAG.
**Effort**: 3 lines total

## Gap 1: TabularPreprocessing Missing `BedrockProcessing` + `StratifiedSampling`

**File**: `src/cursus/steps/specs/tabular_preprocessing_spec.py`

The edge `BedrockProcessing_scoring → TabularPreprocessing_training` cannot resolve because `BedrockProcessing` is not in the `DATA` dependency's `compatible_sources`.

**Fix** (line ~37, in `DATA` DependencySpec):
```python
compatible_sources=[
    "CradleDataLoading",
    "DummyDataLoading",
    "DataLoad",
    "ProcessingStep",
    "BedrockProcessing",      # NEW — LLM-scored data as preprocessing input
    "StratifiedSampling",     # NEW — sampled data as preprocessing input
],
```

## Gap 2: Package Missing `PyTorchTraining`

**File**: `src/cursus/steps/specs/package_spec.py`

The edge `PyTorchTraining → Package` cannot resolve because `PyTorchTraining` is not in `model_input`'s `compatible_sources`.

**Fix** (line ~36, in `model_input` DependencySpec):
```python
compatible_sources=["XGBoostTraining", "TrainingStep", "ModelStep", "PyTorchTraining"],
```

## Verification

After fix, run alignment check:
- `BedrockProcessing_scoring → TabularPreprocessing_training`: score ~0.90 ✅
- `PyTorchTraining → Package`: score ~0.95 ✅

## Implementation

Both fixes are single-line additions. Can be done in parallel:
1. Edit both spec files
2. ruff format
3. python3 syntax check
4. brazil-build release
5. brazil pb build
