# ModelMetricsComputation

**Model metrics computation step for comprehensive performance evaluation**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `model_metrics_computation.py` |
| **Interface file** | `steps/interfaces/model_metrics_computation.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Model metrics computation script. Loads prediction data, computes comprehensive performance metrics, generates visualizations and detailed reports.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `eval_output` | `processing_output` | yes | [XGBoostModelInference](xgboost_model_inference.md), [XGBoostModelEval](xgboost_model_eval.md), [LightGBMMTModelInference](lightgbmmt_model_inference.md), [LightGBMModelInference](lightgbm_model_inference.md), [PyTorchModelInference](pytorch_model_inference.md), [TabularPreprocessing](tabular_preprocessing.md), [CradleDataLoading](cradle_data_loading.md), [RiskTableMapping](risk_table_mapping.md), [CurrencyConversion](currency_conversion.md) |

## Outputs

| Output | Type |
|--------|------|
| `metrics_output` | `processing_output` |
| `plots_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [ModelWikiGenerator](model_wiki_generator.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |
| `scikit-learn` | `>=1.0.0` |
| `matplotlib` | `>=3.5.0` |

---

← [Back to the Step Catalog](index.md)
