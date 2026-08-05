# Step Catalog

Every pipeline step that cursus supports — **60 steps** — generated directly from the `.step.yaml` interface files. Each row links to that step's page: its purpose, its inputs (with the upstream steps that can produce them), its outputs, and the downstream steps that consume it.

A cursus pipeline is a DAG of these steps. An edge is valid when a downstream step's input **type** matches an upstream step's output, and the upstream step is listed among the input's *compatible producers* — see [The DAG + Config → Pipeline model](../concepts/dag_and_compilation.md) and [Registry and Step Catalog](../concepts/registry_and_discovery.md).

The **Compute** column names how a step's container is built: an SDK-managed DLC (`sklearn` / `xgboost` / `framework` / `estimator` / `model`), the SAIS `script` image, or **`byo_container`** — a user-supplied ECR image run verbatim (no `image_uris.retrieve`), which is how a non-DLC framework (e.g. GraphStorm/DGL) enters cursus. A step may also declare a per-step VPC (`network_mode: config`) to reach a VPC-only data source — shown on its page's **Compute** section.

## CradleDataLoading

_Cradle data loading — pull source data via the SAIS/Cradle SDK._

| Step | Node | Compute | Purpose | Consumes | Produces |
|------|------|---------|---------|----------|----------|
| [CradleDataLoading](cradle_data_loading.md) | source | `—` | Cradle data loading step. | — | `processing_output` |

## RedshiftDataLoading

_Redshift data loading — load data from Redshift via the SAIS SDK._

| Step | Node | Compute | Purpose | Consumes | Produces |
|------|------|---------|---------|----------|----------|
| [RedshiftDataLoading](redshift_data_loading.md) | source | `—` | Redshift SQL data loading step (source node with optional EDX upload). | — | `processing_output` |

## Processing

_Processing jobs — data prep, feature engineering, evaluation, packaging._

| Step | Node | Compute | Purpose | Consumes | Produces |
|------|------|---------|---------|----------|----------|
| [ActiveSampleSelection](active_sample_selection.md) | internal | `sklearn` | Active sample selection step that intelligently selects high-value samples from model predictions for Semi-Supervised Learning (SSL) or Active Learning workflows using confidence-based, uncertainty-based, diversity-based, or hybrid strategies. | `processing_output` | `processing_output` |
| [BedrockBatchProcessing](bedrock_batch_processing.md) | internal | `framework` | Bedrock batch processing step that provides AWS Bedrock batch inference capabilities with automatic fallback to real-time processing for cost-efficient large dataset processing. | `processing_output` | `processing_output` |
| [BedrockProcessing](bedrock_processing.md) | internal | `framework` | Bedrock processing step that processes input data through AWS Bedrock models using generated prompt templates and validation schemas. | `processing_output` | `processing_output` |
| [BedrockPromptTemplateGeneration](bedrock_prompt_template_generation.md) | internal | `sklearn` | Bedrock prompt template generation step that assembles the prompt-config bundle (system prompt, category rules, output schema) into ONE standardized prompts.json prompt ruleset ({ruleset, rules} shape) with the output schema embedded in the prompt — the same contract the knowledge-routing producer emits, consumed by the Bedrock processing steps. | `processing_output` | `processing_output` |
| [CurrencyConversion](currency_conversion.md) | internal | `sklearn` | Currency conversion processing step. | `processing_output` | `processing_output` |
| [DataUploading](data_uploading.md) | sink | `—` | Upload processed data to BDT (EDX/Andes) via SAIS SDK delegation. | `processing_output` | — |
| [DummyDataLoading](dummy_data_loading.md) | internal | `sklearn` | Dummy data loading step that processes user-provided data instead of calling Cradle services. | `processing_output` | `processing_output` |
| [DummyTraining](dummy_training.md) | internal | `framework` | Training step that uses a pretrained model. | `hyperparameters`, `model_artifacts` | `model_artifacts` |
| [EdxUploading](edx_uploading.md) | sink | `script` | Upload S3 data to EDX via EdxDataLoader (SINK node, no Kale required). | `processing_output` | — |
| [FeatureSelection](feature_selection.md) | internal | `sklearn` | Feature selection step using multiple statistical and ML-based methods with ensemble combination strategies. | `processing_output` | `processing_output` |
| [GraphConstruction](graph_construction.md) | internal | `byo_container` | GraphStorm gconstruct — build a partitioned DGL heterograph from the node/edge parquets + gconstruct schema emitted by GraphFeatureProcessing. | `processing_output` | `processing_output` |
| [GraphFeatureProcessing](graph_feature_processing.md) | internal | `byo_container` | Turn per-seed subgraph pickles + labelled seeds into the GraphStorm GConstruct input (per-type node/edge parquets, reverse edges, node-ID-keyed masks, gconstruct_config.json). | `processing_output` | `processing_output` |
| [GraphStormGNNInferenceEval](graphstorm_gnn_inference_eval.md) | internal | `byo_container` | GraphStorm/DGL GNN out-of-time inference + evaluation on a custom GraphStorm GPU container (online-inference simulator → ROC-AUC / PR-AUC / Recall@Precision report + plots). | `model_artifacts`, `processing_output` | `processing_output` |
| [GraphSubgraphExtraction](graph_subgraph_extraction.md) | source | `byo_container` | Point-in-time k-hop subgraph pull from a property-graph DB for seed order IDs (BYO GraphStorm image, VPC-bound). | `processing_output` | `processing_output` |
| [LabelRulesetExecution](label_ruleset_execution.md) | internal | `sklearn` | Label ruleset execution step that applies validated rulesets to processed data to generate classification labels using priority-based rule evaluation with execution-time field validation. | `processing_output` | `processing_output` |
| [LabelRulesetGeneration](label_ruleset_generation.md) | internal | `sklearn` | Label ruleset generation step that validates and optimizes user-defined classification rules for transparent, maintainable rule-based label mapping in ML training pipelines. | `processing_output` | `processing_output` |
| [LightGBMMTModelEval](lightgbmmt_model_eval.md) | internal | `framework` | LightGBM multi-task model evaluation step. | `model_artifacts`, `processing_output` | `processing_output` |
| [LightGBMMTModelInference](lightgbmmt_model_inference.md) | internal | `framework` | LightGBM multi-task model inference step for prediction generation without metrics. | `model_artifacts`, `processing_output` | `processing_output` |
| [LightGBMModelEval](lightgbm_model_eval.md) | internal | `framework` | LightGBM model evaluation step. | `model_artifacts`, `processing_output` | `processing_output` |
| [LightGBMModelInference](lightgbm_model_inference.md) | internal | `framework` | LightGBM model inference step for prediction generation without metrics. | `model_artifacts`, `processing_output` | `processing_output` |
| [MissingValueImputation](missing_value_imputation.md) | internal | `sklearn` | Missing value imputation step using statistical methods (mean, median, mode, constant) with pandas-safe values. | `processing_output` | `processing_output` |
| [ModelCalibration](model_calibration.md) | internal | `sklearn` | Calibrates model prediction scores to accurate probabilities. | `processing_output` | `processing_output` |
| [ModelMetricsComputation](model_metrics_computation.md) | internal | `sklearn` | Model metrics computation step for comprehensive performance evaluation. | `processing_output` | `processing_output` |
| [ModelWikiGenerator](model_wiki_generator.md) | internal | `sklearn` | Model wiki generator step for automated documentation creation. | `processing_output` | `processing_output` |
| [Package](package.md) | internal | `sklearn` | Model packaging step. | `custom_property`, `model_artifacts`, `processing_output` | `model_artifacts` |
| [Payload](payload.md) | internal | `sklearn` | Payload testing step. | `model_artifacts`, `processing_output` | `processing_output` |
| [PercentileModelCalibration](percentile_model_calibration.md) | internal | `framework` | Creates percentile mapping from model scores using ROC curve analysis for consistent risk interpretation. | `processing_output` | `processing_output` |
| [PiperMetricGeneration](piper_metric_generation.md) | internal | `sklearn` | PIPER metric generation step; recomputes ROC/PR curves and emits PIPER .metric + paired data CSVs flat to the output root for PIPER rendering. | `processing_output` | `processing_output` |
| [PseudoLabelMerge](pseudo_label_merge.md) | internal | `sklearn` | Pseudo label merge step that intelligently combines labeled base data with pseudo-labeled or augmented samples for Semi-Supervised Learning (SSL) and Active Learning workflows with split-aware merge, auto-inferred split ratios, and provenance tracking. | `processing_output` | `processing_output` |
| [PyTorchModelEval](pytorch_model_eval.md) | internal | `framework` | PyTorch model evaluation step. | `model_artifacts`, `processing_output` | `processing_output` |
| [PyTorchModelInference](pytorch_model_inference.md) | internal | `framework` | PyTorch model inference step for prediction generation without metrics. | `model_artifacts`, `processing_output` | `processing_output` |
| [RiskTableMapping](risk_table_mapping.md) | internal | `framework` | Risk table mapping step for categorical features. | `hyperparameters`, `processing_output` | `processing_output` |
| [SlipboxKnowledgeRouting](slipbox_knowledge_routing.md) | internal | `framework` | Slipbox knowledge routing step that hosts the DKS knowledge+ruleset corpus and runs compile→index→route internally, emitting a compiled prompt ruleset plus per-record routed rule names and routing confidence for downstream Bedrock processing. | `custom_property`, `model_artifacts`, `processing_output` | `processing_output` |
| [StratifiedSampling](stratified_sampling.md) | internal | `sklearn` | Stratified sampling step with multiple allocation strategies for class imbalance, causal analysis, and variance optimization. | `processing_output` | `processing_output` |
| [TSAModelCalibration](tsa_model_calibration.md) | internal | `sklearn` | TSA (Temporal Self-Attention) model calibration step using monotone B-spline calibration for converting raw prediction scores to well-calibrated probabilities for fraud detection. | `processing_output` | `processing_output` |
| [TSAModelEval](tsa_model_eval.md) | internal | `framework` | TSA (Temporal Self-Attention) model evaluation step for dual-task PyTorch models with comprehensive metrics and visualizations. | `model_artifacts`, `processing_output` | `processing_output` |
| [TSAPreprocessing](tsa_preprocessing.md) | internal | `framework` | TSA (Temporal Self-Attention) data preprocessing step that performs sequence processing with feature transformation and scaling for fraud detection models. | `processing_output` | `processing_output` |
| [TSATabularPreprocessing](tsa_tabular_preprocessing.md) | internal | `framework` | TSA (Temporal Self-Attention) tabular preprocessing with explicit output declarations for processed_data and preprocessor artifacts. | `processing_output` | `processing_output` |
| [TabularPreprocessing](tabular_preprocessing.md) | internal | `sklearn` | Tabular data preprocessing step. | `processing_output` | `processing_output` |
| [TemporalFeatureEngineering](temporal_feature_engineering.md) | internal | `sklearn` | Temporal feature engineering step that extracts comprehensive temporal features from normalized sequences for machine learning models. | `processing_output` | `processing_output` |
| [TemporalSequenceNormalization](temporal_sequence_normalization.md) | internal | `sklearn` | Temporal sequence normalization step for machine learning models with configurable sequence operations. | `processing_output` | `processing_output` |
| [TemporalSplitPreprocessing](temporal_split_preprocessing.md) | internal | `sklearn` | Temporal split preprocessing step with customer-level splitting and OOT validation. | `processing_output` | `processing_output`, `training_data` |
| [TokenizerTraining](tokenizer_training.md) | internal | `framework` | BPE tokenizer training step for customer name data with automatic vocabulary size tuning. | `processing_output` | `processing_output` |
| [XGBoostModelEval](xgboost_model_eval.md) | internal | `xgboost` | XGBoost model evaluation step. | `model_artifacts`, `processing_output` | `processing_output` |
| [XGBoostModelInference](xgboost_model_inference.md) | internal | `xgboost` | XGBoost model inference step for prediction generation without metrics. | `model_artifacts`, `processing_output` | `processing_output` |
| [XgboostMtModelEval](xgboost_mt_model_eval.md) | internal | `framework` | XGBoost multi-task model evaluation step. | `model_artifacts`, `processing_output` | `processing_output` |

## Training

_Training jobs — fit a model from prepared data + hyperparameters._

| Step | Node | Compute | Purpose | Consumes | Produces |
|------|------|---------|---------|----------|----------|
| [GraphStormGNNTraining](graphstorm_gnn_training.md) | internal | `byo_container` | GraphStorm/DGL R-GCN GNN training on a partitioned heterograph, run in a bring-your-own GraphStorm ECR container. | `hyperparameters`, `processing_output` | `model_artifacts`, `processing_output` |
| [LightGBMMTTraining](lightgbmmt_training.md) | internal | `estimator` | LightGBM multi-task training with adaptive weighting and knowledge distillation. | `hyperparameters`, `processing_output`, `training_data` | `model_artifacts`, `processing_output` |
| [LightGBMTraining](lightgbm_training.md) | internal | `estimator` | LightGBM model training step using built-in algorithm. | `hyperparameters`, `processing_output`, `training_data` | `model_artifacts`, `processing_output` |
| [PyTorchTraining](pytorch_training.md) | internal | `estimator` | PyTorch model training step. | `hyperparameters`, `processing_output`, `training_data` | `model_artifacts`, `processing_output` |
| [SOPAInstructionTuning](sopa_instruction_tuning.md) | internal | `estimator` | SOPA Stage 2 instruction fine-tuning step for BLIP2-based model (Q-Former + Phi-3 LLM) with tabular-to-text instruction following. | `model_artifacts`, `training_data` | `model_artifacts`, `processing_output` |
| [TSATraining](tsa_training.md) | internal | `estimator` | TSA (Temporal Self-Attention) model training step for PyTorch-based temporal attention models. | `hyperparameters`, `training_data` | `model_artifacts`, `processing_output` |
| [XGBoostTraining](xgboost_training.md) | internal | `estimator` | XGBoost model training step. | `hyperparameters`, `processing_output`, `training_data` | `model_artifacts`, `processing_output` |
| [XgboostMtTraining](xgboost_mt_training.md) | internal | `estimator` | XGBoost multi-task training with one_output_per_tree strategy. | `hyperparameters`, `processing_output`, `training_data` | `model_artifacts`, `processing_output` |

## Transform

_Batch transform jobs — run inference over a dataset with a model._

| Step | Node | Compute | Purpose | Consumes | Produces |
|------|------|---------|---------|----------|----------|
| [BatchTransform](batch_transform.md) | internal | `transformer` | Batch transform step. | `custom_property`, `processing_output` | `custom_property` |

## CreateModel

_Model creation — wrap trained artifacts into a deployable SageMaker model._

| Step | Node | Compute | Purpose | Consumes | Produces |
|------|------|---------|---------|----------|----------|
| [PyTorchModel](pytorch_model.md) | internal | `model` | PyTorch model creation step. | `model_artifacts` | `custom_property` |
| [XGBoostModel](xgboost_model.md) | internal | `model` | XGBoost model creation step. | `model_artifacts` | `custom_property` |

## MimsModelRegistrationProcessing

_Model registration — register a model with MIMS._

| Step | Node | Compute | Purpose | Consumes | Produces |
|------|------|---------|---------|----------|----------|
| [Registration](registration.md) | sink | `—` | Model registration step. | `model_artifacts`, `payload_samples` | — |

---

*This catalog is generated from `src/cursus/steps/interfaces/*.step.yaml` by `docs/gen_step_catalog.py`. To change a step's catalog entry, edit its `.step.yaml` and re-run the generator.*

```{toctree}
:hidden:
:maxdepth: 1

cradle_data_loading
redshift_data_loading
active_sample_selection
bedrock_batch_processing
bedrock_processing
bedrock_prompt_template_generation
currency_conversion
data_uploading
dummy_data_loading
dummy_training
edx_uploading
feature_selection
graph_construction
graph_feature_processing
graphstorm_gnn_inference_eval
graph_subgraph_extraction
label_ruleset_execution
label_ruleset_generation
lightgbmmt_model_eval
lightgbmmt_model_inference
lightgbm_model_eval
lightgbm_model_inference
missing_value_imputation
model_calibration
model_metrics_computation
model_wiki_generator
package
payload
percentile_model_calibration
piper_metric_generation
pseudo_label_merge
pytorch_model_eval
pytorch_model_inference
risk_table_mapping
slipbox_knowledge_routing
stratified_sampling
tsa_model_calibration
tsa_model_eval
tsa_preprocessing
tsa_tabular_preprocessing
tabular_preprocessing
temporal_feature_engineering
temporal_sequence_normalization
temporal_split_preprocessing
tokenizer_training
xgboost_model_eval
xgboost_model_inference
xgboost_mt_model_eval
graphstorm_gnn_training
lightgbmmt_training
lightgbm_training
pytorch_training
sopa_instruction_tuning
tsa_training
xgboost_training
xgboost_mt_training
batch_transform
pytorch_model
xgboost_model
registration
```