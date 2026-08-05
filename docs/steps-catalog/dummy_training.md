# DummyTraining

**Training step that uses a pretrained model**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `dummy_training.py` |
| **Interface file** | `steps/interfaces/dummy_training.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `SKLearn` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

Dummy training step with flexible input modes. Adds hyperparameters.json to model.tar.gz for downstream packaging. Supports INTERNAL mode (accepts inputs) or SOURCE mode (reads from source directory).

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `hyperparameters_s3_uri` | `hyperparameters` | no | HyperparameterPrep, ProcessingStep |
| `model_artifacts_input` | `model_artifacts` | no | [PyTorchTraining](pytorch_training.md), [XGBoostTraining](xgboost_training.md), [LightGBMTraining](lightgbm_training.md), ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `model_output` | `model_artifacts` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [LightGBMMTModelEval](lightgbmmt_model_eval.md)
- [LightGBMMTModelInference](lightgbmmt_model_inference.md)
- [LightGBMModelEval](lightgbm_model_eval.md)
- [LightGBMModelInference](lightgbm_model_inference.md)
- [Payload](payload.md)
- [PyTorchModelEval](pytorch_model_eval.md)
- [PyTorchModelInference](pytorch_model_inference.md)
- [TSAModelEval](tsa_model_eval.md)
- [XGBoostModelEval](xgboost_model_eval.md)
- [XGBoostModelInference](xgboost_model_inference.md)
- [XgboostMtModelEval](xgboost_mt_model_eval.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `python` | `>=3.7` |

---

← [Back to the Step Catalog](index.md)
