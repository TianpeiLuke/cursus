# BedrockBatchProcessing

**Bedrock batch processing step that provides AWS Bedrock batch inference capabilities with automatic fallback to real-time processing for cost-efficient large dataset processing**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `bedrock_batch_processing.py` |
| **Interface file** | `steps/interfaces/bedrock_batch_processing.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

Bedrock batch processing script with batch inference and automatic fallback to real-time. Integrates with prompt templates and validation schemas from BedrockPromptTemplateGeneration.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_data` | `processing_output` | yes | [DummyDataLoading](dummy_data_loading.md), [CradleDataLoading](cradle_data_loading.md), [TabularPreprocessing](tabular_preprocessing.md), [TemporalSequenceNormalization](temporal_sequence_normalization.md), [TemporalFeatureEngineering](temporal_feature_engineering.md), [StratifiedSampling](stratified_sampling.md), [MissingValueImputation](missing_value_imputation.md), [FeatureSelection](feature_selection.md), [CurrencyConversion](currency_conversion.md), [SlipboxKnowledgeRouting](slipbox_knowledge_routing.md) |
| `prompt_templates` | `processing_output` | yes | [BedrockPromptTemplateGeneration](bedrock_prompt_template_generation.md), [SlipboxKnowledgeRouting](slipbox_knowledge_routing.md) |

## Outputs

| Output | Type |
|--------|------|
| `processed_data` | `processing_output` |
| `analysis_summary` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [ActiveSampleSelection](active_sample_selection.md)
- [LabelRulesetExecution](label_ruleset_execution.md)
- [LightGBMMTModelEval](lightgbmmt_model_eval.md)
- [LightGBMMTModelInference](lightgbmmt_model_inference.md)
- [PseudoLabelMerge](pseudo_label_merge.md)
- [XgboostMtModelEval](xgboost_mt_model_eval.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.2.0` |
| `boto3` | `>=1.26.0` |
| `pydantic` | `>=2.0.0` |
| `tenacity` | `>=8.0.0` |

---

← [Back to the Step Catalog](index.md)
