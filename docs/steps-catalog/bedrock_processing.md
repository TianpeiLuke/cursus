# BedrockProcessing

**Bedrock processing step that processes input data through AWS Bedrock models using generated prompt templates and validation schemas**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `bedrock_processing.py` |
| **Interface file** | `steps/interfaces/bedrock_processing.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

Bedrock processing script with invoke_model, structured output, and Converse API modes. Supports self-contained mode via env var templates. Circuit breaker and adaptive rate limiting.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_data` | `processing_output` | yes | [DummyDataLoading](dummy_data_loading.md), [CradleDataLoading](cradle_data_loading.md), [TabularPreprocessing](tabular_preprocessing.md), [TemporalSequenceNormalization](temporal_sequence_normalization.md), [TemporalFeatureEngineering](temporal_feature_engineering.md), [StratifiedSampling](stratified_sampling.md), [MissingValueImputation](missing_value_imputation.md), [FeatureSelection](feature_selection.md), [CurrencyConversion](currency_conversion.md), [SlipboxKnowledgeRouting](slipbox_knowledge_routing.md) |
| `prompt_templates` | `processing_output` | no | [BedrockPromptTemplateGeneration](bedrock_prompt_template_generation.md), [SlipboxKnowledgeRouting](slipbox_knowledge_routing.md) |

## Outputs

| Output | Type |
|--------|------|
| `processed_data` | `processing_output` |
| `analysis_summary` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [ActiveSampleSelection](active_sample_selection.md)
- [EdxUploading](edx_uploading.md)
- [LabelRulesetExecution](label_ruleset_execution.md)
- [LightGBMMTModelEval](lightgbmmt_model_eval.md)
- [LightGBMMTModelInference](lightgbmmt_model_inference.md)
- [LightGBMTraining](lightgbm_training.md)
- [PseudoLabelMerge](pseudo_label_merge.md)
- [PyTorchModelEval](pytorch_model_eval.md)
- [PyTorchModelInference](pytorch_model_inference.md)
- [PyTorchTraining](pytorch_training.md)
- [TSATabularPreprocessing](tsa_tabular_preprocessing.md)
- [TabularPreprocessing](tabular_preprocessing.md)
- [XGBoostTraining](xgboost_training.md)
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
