# CurrencyConversion

**Currency conversion processing step**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `currency_conversion.py` |
| **Interface file** | `steps/interfaces/currency_conversion.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Currency conversion script. Converts monetary values across currencies based on marketplace information and exchange rates.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), ProcessingStep, [CradleDataLoading](cradle_data_loading.md), [MissingValueImputation](missing_value_imputation.md), [RiskTableMapping](risk_table_mapping.md), [StratifiedSampling](stratified_sampling.md), [FeatureSelection](feature_selection.md) |

## Outputs

| Output | Type |
|--------|------|
| `processed_data` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [BedrockBatchProcessing](bedrock_batch_processing.md)
- [BedrockProcessing](bedrock_processing.md)
- [LabelRulesetExecution](label_ruleset_execution.md)
- [LightGBMMTModelEval](lightgbmmt_model_eval.md)
- [LightGBMMTModelInference](lightgbmmt_model_inference.md)
- [LightGBMModelEval](lightgbm_model_eval.md)
- [LightGBMModelInference](lightgbm_model_inference.md)
- [ModelMetricsComputation](model_metrics_computation.md)
- [PiperMetricGeneration](piper_metric_generation.md)
- [PyTorchModelEval](pytorch_model_eval.md)
- [PyTorchModelInference](pytorch_model_inference.md)
- [XGBoostModelEval](xgboost_model_eval.md)
- [XGBoostModelInference](xgboost_model_inference.md)
- [XgboostMtModelEval](xgboost_mt_model_eval.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |

---

← [Back to the Step Catalog](index.md)
