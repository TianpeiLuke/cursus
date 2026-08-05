# LightGBMMTModelInference

**LightGBM multi-task model inference step for prediction generation without metrics**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `lightgbmmt_model_inference.py` |
| **Interface file** | `steps/interfaces/lightgbmmt_model_inference.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

LightGBMMT multi-task model inference. Generates per-task predictions without evaluation, metrics, or plots.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `model_input` | `model_artifacts` | yes | [LightGBMMTTraining](lightgbmmt_training.md), [LightGBMTraining](lightgbm_training.md), LightGBMMTModel, LightGBMModel, [XGBoostTraining](xgboost_training.md), [PyTorchTraining](pytorch_training.md), [DummyTraining](dummy_training.md), [XGBoostModel](xgboost_model.md), [PyTorchModel](pytorch_model.md) |
| `processed_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [CradleDataLoading](cradle_data_loading.md), [RiskTableMapping](risk_table_mapping.md), [CurrencyConversion](currency_conversion.md), [LabelRulesetExecution](label_ruleset_execution.md), [BedrockBatchProcessing](bedrock_batch_processing.md), [BedrockProcessing](bedrock_processing.md) |

## Outputs

| Output | Type |
|--------|------|
| `eval_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [ModelMetricsComputation](model_metrics_computation.md)
- [PiperMetricGeneration](piper_metric_generation.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.2.0,<2.0.0` |
| `numpy` | `>=1.21.0` |
| `scikit-learn` | `>=0.23.2,<1.0.0` |

---

← [Back to the Step Catalog](index.md)
