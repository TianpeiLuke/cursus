# PseudoLabelMerge

**Pseudo label merge step that intelligently combines labeled base data with pseudo-labeled or augmented samples for Semi-Supervised Learning (SSL) and Active Learning workflows with split-aware merge, auto-inferred split ratios, and provenance tracking**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `pseudo_label_merge.py` |
| **Interface file** | `steps/interfaces/pseudo_label_merge.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Pseudo label merge script. Intelligently merges labeled base data with pseudo-labeled or augmented samples for SSL and Active Learning workflows.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `base_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [RiskTableMapping](risk_table_mapping.md), [MissingValueImputation](missing_value_imputation.md), [FeatureSelection](feature_selection.md), [StratifiedSampling](stratified_sampling.md), [TemporalSequenceNormalization](temporal_sequence_normalization.md), [TemporalFeatureEngineering](temporal_feature_engineering.md), [LabelRulesetExecution](label_ruleset_execution.md) |
| `augmentation_data` | `processing_output` | yes | [ActiveSampleSelection](active_sample_selection.md), [XGBoostModelInference](xgboost_model_inference.md), [LightGBMModelInference](lightgbm_model_inference.md), [PyTorchModelInference](pytorch_model_inference.md), [XGBoostModelEval](xgboost_model_eval.md), [LightGBMModelEval](lightgbm_model_eval.md), [PyTorchModelEval](pytorch_model_eval.md), [BedrockBatchProcessing](bedrock_batch_processing.md), [BedrockProcessing](bedrock_processing.md), [LabelRulesetExecution](label_ruleset_execution.md) |

## Outputs

| Output | Type |
|--------|------|
| `merged_data` | `processing_output` |

## Consumers (downstream steps)

_No cataloged step lists this step as a compatible source (it may be a terminal/sink step, or consumed via a generic source name)._

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |

---

← [Back to the Step Catalog](index.md)
