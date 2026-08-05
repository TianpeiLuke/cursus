# CradleDataLoading

**Cradle data loading step**

| | |
|---|---|
| **SageMaker step type** | `CradleDataLoading` |
| **Node type** | source (no inputs — originates data) |
| **Container entry point** | `scripts.py` |
| **Build-time requirement** | `secure_ai_sandbox_workflow_python_sdk` (SAIS SDK — fatal on load if absent) |
| **Interface file** | `steps/interfaces/cradle_data_loading.step.yaml` |

## Functionality

Cradle data loading script that reads config, writes output signature/metadata, creates and executes a Cradle data load job, and waits for completion. Data is loaded directly to S3 by the Cradle service.

## Inputs (dependencies)

_None — this is a source step; it originates data with no pipeline inputs._

## Outputs

| Output | Type |
|--------|------|
| `DATA` | `processing_output` |
| `METADATA` | `processing_output` |
| `SIGNATURE` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [BedrockBatchProcessing](bedrock_batch_processing.md)
- [BedrockProcessing](bedrock_processing.md)
- [CurrencyConversion](currency_conversion.md)
- [DataUploading](data_uploading.md)
- [EdxUploading](edx_uploading.md)
- [GraphFeatureProcessing](graph_feature_processing.md)
- [GraphStormGNNInferenceEval](graphstorm_gnn_inference_eval.md)
- [GraphSubgraphExtraction](graph_subgraph_extraction.md)
- [LightGBMMTModelEval](lightgbmmt_model_eval.md)
- [LightGBMMTModelInference](lightgbmmt_model_inference.md)
- [LightGBMModelEval](lightgbm_model_eval.md)
- [LightGBMModelInference](lightgbm_model_inference.md)
- [ModelMetricsComputation](model_metrics_computation.md)
- [PiperMetricGeneration](piper_metric_generation.md)
- [PyTorchModelEval](pytorch_model_eval.md)
- [PyTorchModelInference](pytorch_model_inference.md)
- [SlipboxKnowledgeRouting](slipbox_knowledge_routing.md)
- [TSAPreprocessing](tsa_preprocessing.md)
- [TSATabularPreprocessing](tsa_tabular_preprocessing.md)
- [TabularPreprocessing](tabular_preprocessing.md)
- [TemporalSequenceNormalization](temporal_sequence_normalization.md)
- [TemporalSplitPreprocessing](temporal_split_preprocessing.md)
- [XGBoostModelEval](xgboost_model_eval.md)
- [XGBoostModelInference](xgboost_model_inference.md)
- [XgboostMtModelEval](xgboost_mt_model_eval.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `python` | `>=3.7` |

---

← [Back to the Step Catalog](index.md)
