# PercentileModelCalibration

**Creates percentile mapping from model scores using ROC curve analysis for consistent risk interpretation**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `percentile_model_calibration.py` |
| **Interface file** | `steps/interfaces/percentile_model_calibration.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `SKLearn` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

Percentile model calibration that converts raw model scores to calibrated percentile values using ROC curve analysis. Supports single-task and multi-task calibration with configurable calibration dictionary.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `evaluation_data` | `processing_output` | yes | [XGBoostTraining](xgboost_training.md), [XGBoostModelEval](xgboost_model_eval.md), [XGBoostModelInference](xgboost_model_inference.md), [LightGBMTraining](lightgbm_training.md), [LightGBMModelEval](lightgbm_model_eval.md), [LightGBMModelInference](lightgbm_model_inference.md), [LightGBMMTTraining](lightgbmmt_training.md), [LightGBMMTModelEval](lightgbmmt_model_eval.md), [PyTorchTraining](pytorch_training.md), [PyTorchModelEval](pytorch_model_eval.md), [PyTorchModelInference](pytorch_model_inference.md), ModelEvaluation, TrainingEvaluation, CrossValidation, [ModelCalibration](model_calibration.md) |
| `calibration_config` | `processing_output` | no | ConfigurationStep, DataPreprocessing, FeatureEngineering, ModelConfiguration |

## Outputs

| Output | Type |
|--------|------|
| `calibration_output` | `processing_output` |
| `metrics_output` | `processing_output` |
| `calibrated_data` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [Package](package.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `scikit-learn` | `>=0.23.2,<1.0.0` |
| `pandas` | `>=1.2.0,<2.0.0` |
| `numpy` | `>=1.20.0` |

---

← [Back to the Step Catalog](index.md)
