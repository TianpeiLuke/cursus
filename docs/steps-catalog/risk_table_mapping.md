# RiskTableMapping

**Risk table mapping step for categorical features**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `risk_table_mapping.py` |
| **Interface file** | `steps/interfaces/risk_table_mapping.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `SKLearn` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

Risk table mapping script. Creates risk tables for categorical features and handles missing value imputation for numeric features. Training mode creates tables; inference applies.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [MissingValueImputation](missing_value_imputation.md), ProcessingStep |
| `hyperparameters_s3_uri` | `hyperparameters` | no | HyperparameterPrep, ProcessingStep, ConfigurationStep |
| `model_artifacts_input` | `processing_output` | no | RiskTableMapping_Training, ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `processed_data` | `processing_output` |
| `model_artifacts_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [CurrencyConversion](currency_conversion.md)
- [FeatureSelection](feature_selection.md)
- [LabelRulesetExecution](label_ruleset_execution.md)
- [LightGBMMTModelEval](lightgbmmt_model_eval.md)
- [LightGBMMTModelInference](lightgbmmt_model_inference.md)
- [LightGBMMTTraining](lightgbmmt_training.md)
- [LightGBMModelEval](lightgbm_model_eval.md)
- [LightGBMModelInference](lightgbm_model_inference.md)
- [LightGBMTraining](lightgbm_training.md)
- [MissingValueImputation](missing_value_imputation.md)
- [ModelMetricsComputation](model_metrics_computation.md)
- [PiperMetricGeneration](piper_metric_generation.md)
- [PseudoLabelMerge](pseudo_label_merge.md)
- [PyTorchModelEval](pytorch_model_eval.md)
- [PyTorchModelInference](pytorch_model_inference.md)
- [PyTorchTraining](pytorch_training.md)
- [XGBoostModelEval](xgboost_model_eval.md)
- [XGBoostModelInference](xgboost_model_inference.md)
- [XGBoostTraining](xgboost_training.md)
- [XgboostMtModelEval](xgboost_mt_model_eval.md)
- [XgboostMtTraining](xgboost_mt_training.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |
| `scikit-learn` | `>=1.0.0` |

---

← [Back to the Step Catalog](index.md)
