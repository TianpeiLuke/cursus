# TabularPreprocessing

**Tabular data preprocessing step**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `tabular_preprocessing.py` |
| **Interface file** | `steps/interfaces/tabular_preprocessing.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Tabular preprocessing script that combines data shards, loads column signature, cleans/processes label field, splits data into train/test/val, and outputs in configurable format (CSV/TSV/Parquet). Supports streaming mode for large datasets.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `DATA` | `processing_output` | yes | [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md), [RedshiftDataLoading](redshift_data_loading.md), DataLoad, ProcessingStep, [BedrockProcessing](bedrock_processing.md), [StratifiedSampling](stratified_sampling.md) |
| `DATA_SECONDARY` | `processing_output` | no | [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md), [RedshiftDataLoading](redshift_data_loading.md), DataLoad, ProcessingStep, [BedrockProcessing](bedrock_processing.md), [StratifiedSampling](stratified_sampling.md) |
| `SIGNATURE` | `processing_output` | no | [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md) |

## Outputs

| Output | Type |
|--------|------|
| `processed_data` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [BatchTransform](batch_transform.md)
- [BedrockBatchProcessing](bedrock_batch_processing.md)
- [BedrockProcessing](bedrock_processing.md)
- [CurrencyConversion](currency_conversion.md)
- [DataUploading](data_uploading.md)
- [EdxUploading](edx_uploading.md)
- [FeatureSelection](feature_selection.md)
- [GraphSubgraphExtraction](graph_subgraph_extraction.md)
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
- [RiskTableMapping](risk_table_mapping.md)
- [SOPAInstructionTuning](sopa_instruction_tuning.md)
- [SlipboxKnowledgeRouting](slipbox_knowledge_routing.md)
- [StratifiedSampling](stratified_sampling.md)
- [TSAModelCalibration](tsa_model_calibration.md)
- [TSAModelEval](tsa_model_eval.md)
- [TSAPreprocessing](tsa_preprocessing.md)
- [TSATraining](tsa_training.md)
- [TemporalSequenceNormalization](temporal_sequence_normalization.md)
- [TokenizerTraining](tokenizer_training.md)
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
