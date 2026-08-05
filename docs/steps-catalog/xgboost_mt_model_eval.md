# XgboostMtModelEval

**XGBoost multi-task model evaluation step**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `xgboost_mt_model_eval.py` |
| **Interface file** | `steps/interfaces/xgboost_mt_model_eval.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

XgboostMt multi-task model evaluation. Generates per-task and aggregate metrics with visualizations.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `model_input` | `model_artifacts` | yes | [XgboostMtTraining](xgboost_mt_training.md), [XGBoostTraining](xgboost_training.md), XgboostMtModel, [XGBoostModel](xgboost_model.md), [DummyTraining](dummy_training.md) |
| `processed_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [CradleDataLoading](cradle_data_loading.md), [RiskTableMapping](risk_table_mapping.md), [CurrencyConversion](currency_conversion.md), [LabelRulesetExecution](label_ruleset_execution.md), [BedrockBatchProcessing](bedrock_batch_processing.md), [BedrockProcessing](bedrock_processing.md), [TemporalSplitPreprocessing](temporal_split_preprocessing.md) |

## Outputs

| Output | Type |
|--------|------|
| `eval_output` | `processing_output` |
| `metrics_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [ModelCalibration](model_calibration.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.2.0,<2.0.0` |
| `numpy` | `>=1.21.0` |
| `scikit-learn` | `>=0.23.2,<1.0.0` |
| `matplotlib` | `>=3.0.0` |

---

← [Back to the Step Catalog](index.md)
