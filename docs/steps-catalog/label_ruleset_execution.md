# LabelRulesetExecution

**Label ruleset execution step that applies validated rulesets to processed data to generate classification labels using priority-based rule evaluation with execution-time field validation**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `label_ruleset_execution.py` |
| **Interface file** | `steps/interfaces/label_ruleset_execution.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Label ruleset execution script. Applies validated rulesets to processed data to generate classification labels using priority-based rule evaluation.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `validated_ruleset` | `processing_output` | yes | [LabelRulesetGeneration](label_ruleset_generation.md), [SlipboxKnowledgeRouting](slipbox_knowledge_routing.md) |
| `input_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [BedrockProcessing](bedrock_processing.md), [BedrockBatchProcessing](bedrock_batch_processing.md), [TemporalSequenceNormalization](temporal_sequence_normalization.md), [TemporalFeatureEngineering](temporal_feature_engineering.md), [StratifiedSampling](stratified_sampling.md), [MissingValueImputation](missing_value_imputation.md), [FeatureSelection](feature_selection.md), [CurrencyConversion](currency_conversion.md), [RiskTableMapping](risk_table_mapping.md) |

## Outputs

| Output | Type |
|--------|------|
| `processed_data` | `processing_output` |
| `execution_report` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [ActiveSampleSelection](active_sample_selection.md)
- [LightGBMMTModelEval](lightgbmmt_model_eval.md)
- [LightGBMMTModelInference](lightgbmmt_model_inference.md)
- [PseudoLabelMerge](pseudo_label_merge.md)
- [PyTorchTraining](pytorch_training.md)
- [XGBoostTraining](xgboost_training.md)
- [XgboostMtModelEval](xgboost_mt_model_eval.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `pydantic` | `>=2.0.0` |

---

← [Back to the Step Catalog](index.md)
