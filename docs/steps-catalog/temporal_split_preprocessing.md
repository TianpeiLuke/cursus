# TemporalSplitPreprocessing

**Temporal split preprocessing step with customer-level splitting and OOT validation**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `temporal_split_preprocessing.py` |
| **Interface file** | `steps/interfaces/temporal_split_preprocessing.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Temporal split preprocessing script. Handles data loading, temporal splitting, customer-level splitting, and main task label generation.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `DATA` | `processing_output` | yes | [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md), DataLoad, ProcessingStep |
| `SIGNATURE` | `processing_output` | no | [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md) |

## Outputs

| Output | Type |
|--------|------|
| `training_data` | `training_data` |
| `oot_data` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [LightGBMMTModelEval](lightgbmmt_model_eval.md)
- [LightGBMMTTraining](lightgbmmt_training.md)
- [XgboostMtModelEval](xgboost_mt_model_eval.md)
- [XgboostMtTraining](xgboost_mt_training.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |

---

← [Back to the Step Catalog](index.md)
