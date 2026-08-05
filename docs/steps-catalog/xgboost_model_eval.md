# XGBoostModelEval

**XGBoost model evaluation step**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `xgboost_model_eval.py` |
| **Interface file** | `steps/interfaces/xgboost_model_eval.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `xgboost` |

## Functionality

XGBoost model evaluation script. Loads trained model, processes evaluation data, generates performance metrics and visualizations.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `model_input` | `model_artifacts` | yes | [XGBoostTraining](xgboost_training.md), [PyTorchTraining](pytorch_training.md), [DummyTraining](dummy_training.md), [XGBoostModel](xgboost_model.md), [PyTorchModel](pytorch_model.md) |
| `processed_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [CradleDataLoading](cradle_data_loading.md), [RiskTableMapping](risk_table_mapping.md), [CurrencyConversion](currency_conversion.md) |

## Outputs

| Output | Type |
|--------|------|
| `eval_output` | `processing_output` |
| `metrics_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [ActiveSampleSelection](active_sample_selection.md)
- [ModelCalibration](model_calibration.md)
- [ModelMetricsComputation](model_metrics_computation.md)
- [ModelWikiGenerator](model_wiki_generator.md)
- [PercentileModelCalibration](percentile_model_calibration.md)
- [PiperMetricGeneration](piper_metric_generation.md)
- [PseudoLabelMerge](pseudo_label_merge.md)
- [TSAModelCalibration](tsa_model_calibration.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |
| `xgboost` | `>=1.6.0` |
| `matplotlib` | `>=3.5.0` |

---

← [Back to the Step Catalog](index.md)
