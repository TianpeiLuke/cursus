# FeatureSelection

**Feature selection step using multiple statistical and ML-based methods with ensemble combination strategies**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `feature_selection.py` |
| **Interface file** | `steps/interfaces/feature_selection.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Feature selection script. Applies statistical and ML-based feature selection methods for dimensionality reduction. Training mode fits selectors; inference applies pre-computed.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [StratifiedSampling](stratified_sampling.md), [RiskTableMapping](risk_table_mapping.md), [MissingValueImputation](missing_value_imputation.md), ProcessingStep |
| `model_artifacts_input` | `processing_output` | no | FeatureSelection_Training, [FeatureSelection](feature_selection.md), ProcessingStep |

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
