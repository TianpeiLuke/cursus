# MissingValueImputation

**Missing value imputation step using statistical methods (mean, median, mode, constant) with pandas-safe values**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `missing_value_imputation.py` |
| **Interface file** | `steps/interfaces/missing_value_imputation.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Missing value imputation script. Handles missing values using statistical methods (mean, median, mode, constant). Training mode fits imputers; inference applies pre-fitted. Per-column strategies can also be supplied dynamically via COLUMN_STRATEGY_<column_name> environment variables (discovered at runtime, not pre-declared).

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [StratifiedSampling](stratified_sampling.md), [RiskTableMapping](risk_table_mapping.md), ProcessingStep |
| `model_artifacts_input` | `processing_output` | no | MissingValueImputation_Training, ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `processed_data` | `processing_output` |
| `model_artifacts_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [BedrockBatchProcessing](bedrock_batch_processing.md)
- [BedrockProcessing](bedrock_processing.md)
- [CurrencyConversion](currency_conversion.md)
- [FeatureSelection](feature_selection.md)
- [LabelRulesetExecution](label_ruleset_execution.md)
- [LightGBMMTTraining](lightgbmmt_training.md)
- [LightGBMTraining](lightgbm_training.md)
- [PseudoLabelMerge](pseudo_label_merge.md)
- [PyTorchTraining](pytorch_training.md)
- [RiskTableMapping](risk_table_mapping.md)
- [XGBoostTraining](xgboost_training.md)
- [XgboostMtTraining](xgboost_mt_training.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |
| `scikit-learn` | `>=1.0.0` |

---

← [Back to the Step Catalog](index.md)
