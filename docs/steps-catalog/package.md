# Package

**Model packaging step**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `package.py` |
| **Interface file** | `steps/interfaces/package.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

MIMS packaging script that extracts model artifacts, includes calibration model if available, copies inference scripts, and creates a packaged model.tar.gz for deployment.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `model_input` | `model_artifacts` | yes | [XGBoostTraining](xgboost_training.md), TrainingStep, ModelStep, [PyTorchTraining](pytorch_training.md), [XgboostMtTraining](xgboost_mt_training.md), [LightGBMMTTraining](lightgbmmt_training.md) |
| `inference_scripts_input` | `custom_property` | no | ProcessingStep, ScriptStep |
| `calibration_model` | `processing_output` | no | [ModelCalibration](model_calibration.md), [PercentileModelCalibration](percentile_model_calibration.md) |

## Outputs

| Output | Type |
|--------|------|
| `packaged_model` | `model_artifacts` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [PyTorchModel](pytorch_model.md)
- [Registration](registration.md)
- [XGBoostModel](xgboost_model.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `python` | `>=3.7` |

---

← [Back to the Step Catalog](index.md)
