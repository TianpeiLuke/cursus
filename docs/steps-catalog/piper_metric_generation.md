# PiperMetricGeneration

**PIPER metric generation step; recomputes ROC/PR curves and emits PIPER .metric + paired data CSVs flat to the output root for PIPER rendering**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `piper_metric_generation.py` |
| **Interface file** | `steps/interfaces/piper_metric_generation.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

PIPER metric generation script. Loads prediction data, recomputes ROC and PR curves, and emits PIPER .metric JSON files with paired 2-column data CSVs written flat to the output root for PIPER rendering.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `eval_output` | `processing_output` | yes | [XGBoostModelInference](xgboost_model_inference.md), [XGBoostModelEval](xgboost_model_eval.md), [LightGBMMTModelInference](lightgbmmt_model_inference.md), [LightGBMModelInference](lightgbm_model_inference.md), [PyTorchModelInference](pytorch_model_inference.md), [TabularPreprocessing](tabular_preprocessing.md), [CradleDataLoading](cradle_data_loading.md), [RiskTableMapping](risk_table_mapping.md), [CurrencyConversion](currency_conversion.md) |

## Outputs

| Output | Type |
|--------|------|
| `metric_output` | `processing_output` |

## Consumers (downstream steps)

_No cataloged step lists this step as a compatible source (it may be a terminal/sink step, or consumed via a generic source name)._

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |
| `scikit-learn` | `>=1.0.0` |

---

← [Back to the Step Catalog](index.md)
