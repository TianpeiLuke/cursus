# PyTorchModel

**PyTorch model creation step**

| | |
|---|---|
| **SageMaker step type** | `CreateModel` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Interface file** | `steps/interfaces/pytorch_model.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `model` |
| **SDK class** | `PyTorchModel` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

SageMaker Model creation step for PyTorch. No script — managed by SageMaker ModelStep. Creates a deployable model from training artifacts.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `model_data` | `model_artifacts` | yes | [PyTorchTraining](pytorch_training.md), ProcessingStep, ModelArtifactsStep, [Package](package.md) |

## Outputs

| Output | Type |
|--------|------|
| `model_name` | `custom_property` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [BatchTransform](batch_transform.md)
- [LightGBMMTModelInference](lightgbmmt_model_inference.md)
- [LightGBMModelEval](lightgbm_model_eval.md)
- [LightGBMModelInference](lightgbm_model_inference.md)
- [PyTorchModelEval](pytorch_model_eval.md)
- [PyTorchModelInference](pytorch_model_inference.md)
- [SlipboxKnowledgeRouting](slipbox_knowledge_routing.md)
- [TSAModelEval](tsa_model_eval.md)
- [XGBoostModelEval](xgboost_model_eval.md)
- [XGBoostModelInference](xgboost_model_inference.md)

---

← [Back to the Step Catalog](index.md)
