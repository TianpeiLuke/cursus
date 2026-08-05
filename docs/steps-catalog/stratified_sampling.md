# StratifiedSampling

**Stratified sampling step with multiple allocation strategies for class imbalance, causal analysis, and variance optimization**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `stratified_sampling.py` |
| **Interface file** | `steps/interfaces/stratified_sampling.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Stratified sampling with four allocation strategies (balanced, proportional_min, optimal, external_proportional). Handles class imbalance correction, causal analysis, and variance optimization with per-split diagnostics.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `processed_data` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [BedrockBatchProcessing](bedrock_batch_processing.md)
- [BedrockProcessing](bedrock_processing.md)
- [CurrencyConversion](currency_conversion.md)
- [DataUploading](data_uploading.md)
- [EdxUploading](edx_uploading.md)
- [FeatureSelection](feature_selection.md)
- [GraphSubgraphExtraction](graph_subgraph_extraction.md)
- [LabelRulesetExecution](label_ruleset_execution.md)
- [LightGBMMTTraining](lightgbmmt_training.md)
- [LightGBMTraining](lightgbm_training.md)
- [MissingValueImputation](missing_value_imputation.md)
- [PseudoLabelMerge](pseudo_label_merge.md)
- [PyTorchTraining](pytorch_training.md)
- [TSATabularPreprocessing](tsa_tabular_preprocessing.md)
- [TabularPreprocessing](tabular_preprocessing.md)
- [XGBoostTraining](xgboost_training.md)
- [XgboostMtTraining](xgboost_mt_training.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |

---

← [Back to the Step Catalog](index.md)
