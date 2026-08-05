# BatchTransform

**Batch transform step**

| | |
|---|---|
| **SageMaker step type** | `Transform` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Interface file** | `steps/interfaces/batch_transform.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `transformer` |

## Functionality

SageMaker Batch Transform step. Uses a registered model to run batch inference on preprocessed data. No script — managed by SageMaker TransformStep.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `model_name` | `custom_property` | yes | [PyTorchModel](pytorch_model.md), [XGBoostModel](xgboost_model.md) |
| `processed_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md) |

## Outputs

| Output | Type |
|--------|------|
| `transform_output` | `custom_property` |

## Consumers (downstream steps)

_No cataloged step lists this step as a compatible source (it may be a terminal/sink step, or consumed via a generic source name)._

---

← [Back to the Step Catalog](index.md)
