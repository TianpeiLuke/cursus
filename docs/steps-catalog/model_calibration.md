# ModelCalibration

**Calibrates model prediction scores to accurate probabilities**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `model_calibration.py` |
| **Interface file** | `steps/interfaces/model_calibration.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Model calibration step that calibrates raw prediction scores to true probabilities. Supports GAM, isotonic, and Platt methods. Handles binary, multi-class, and multi-task scenarios with per-task calibrators and aggregate metrics.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `evaluation_data` | `processing_output` | yes | [XGBoostTraining](xgboost_training.md), [XGBoostModelEval](xgboost_model_eval.md), [XGBoostModelInference](xgboost_model_inference.md), [LightGBMTraining](lightgbm_training.md), [LightGBMModelEval](lightgbm_model_eval.md), [LightGBMModelInference](lightgbm_model_inference.md), [LightGBMMTTraining](lightgbmmt_training.md), [LightGBMMTModelEval](lightgbmmt_model_eval.md), [PyTorchTraining](pytorch_training.md), [PyTorchModelEval](pytorch_model_eval.md), [PyTorchModelInference](pytorch_model_inference.md), ModelEvaluation, TrainingEvaluation, CrossValidation, [XgboostMtModelEval](xgboost_mt_model_eval.md) |

## Outputs

| Output | Type |
|--------|------|
| `calibration_output` | `processing_output` |
| `metrics_output` | `processing_output` |
| `calibrated_data` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [Package](package.md)
- [PercentileModelCalibration](percentile_model_calibration.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `scikit-learn` | `>=0.23.2,<1.0.0` |
| `pandas` | `>=1.2.0,<2.0.0` |
| `numpy` | `>=1.20.0` |
| `pygam` | `>=0.8.0` |
| `matplotlib` | `>=3.3.0` |

---

← [Back to the Step Catalog](index.md)
