# Payload

**Payload testing step**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `payload.py` |
| **Interface file** | `steps/interfaces/payload.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

MIMS payload generation script that extracts hyperparameters from model artifacts, detects model type (tabular/bimodal/trimodal), generates sample payloads with text field support, and archives payload files for deployment. Per-field overrides can also be supplied dynamically via SPECIAL_FIELD_<field_name> environment variables (discovered at runtime, not pre-declared).

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `model_input` | `model_artifacts` | yes | [XGBoostTraining](xgboost_training.md), [LightGBMTraining](lightgbm_training.md), [LightGBMMTTraining](lightgbmmt_training.md), [PyTorchTraining](pytorch_training.md), [DummyTraining](dummy_training.md), TrainingStep, ModelStep, [XgboostMtTraining](xgboost_mt_training.md) |
| `custom_payload_input` | `processing_output` | no | ProcessingStep, S3Source, UserProvided |

## Outputs

| Output | Type |
|--------|------|
| `payload_sample` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [Registration](registration.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `python` | `>=3.7` |

---

← [Back to the Step Catalog](index.md)
