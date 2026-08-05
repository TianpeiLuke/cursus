# PyTorchModelInference

**PyTorch model inference step for prediction generation without metrics**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `pytorch_model_inference.py` |
| **Interface file** | `steps/interfaces/pytorch_model_inference.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

PyTorch model inference script. Loads trained model, preprocesses evaluation data, generates predictions without metrics computation.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `model_input` | `model_artifacts` | yes | [PyTorchTraining](pytorch_training.md), [XGBoostTraining](xgboost_training.md), [DummyTraining](dummy_training.md), [PyTorchModel](pytorch_model.md), [XGBoostModel](xgboost_model.md) |
| `processed_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [CradleDataLoading](cradle_data_loading.md), [RiskTableMapping](risk_table_mapping.md), [CurrencyConversion](currency_conversion.md), [BedrockProcessing](bedrock_processing.md) |

## Outputs

| Output | Type |
|--------|------|
| `eval_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [ActiveSampleSelection](active_sample_selection.md)
- [ModelCalibration](model_calibration.md)
- [ModelMetricsComputation](model_metrics_computation.md)
- [ModelWikiGenerator](model_wiki_generator.md)
- [PercentileModelCalibration](percentile_model_calibration.md)
- [PiperMetricGeneration](piper_metric_generation.md)
- [PseudoLabelMerge](pseudo_label_merge.md)
- [XGBoostModelInference](xgboost_model_inference.md)
- [XGBoostTraining](xgboost_training.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `torch` | `==2.1.0` |
| `transformers` | `==4.37.2` |
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |
| `pydantic` | `==2.11.2` |

---

← [Back to the Step Catalog](index.md)
