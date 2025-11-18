---
tags:
  - code
  - steps
  - registry
  - pipeline_reference
keywords:
  - step registry
  - pipeline steps
  - step catalog
  - step types
  - step builders
  - step configurations
topics:
  - pipeline steps
  - step organization
  - ML workflow
language: python
date of note: 2025-11-18
---

# Registered Steps Pipeline Reference

## Purpose

This document provides a comprehensive reference of all registered pipeline steps in the Cursus framework, organized by their functional role and position in the ML pipeline execution flow. Each step is categorized by its SageMaker step type and grouped according to the phase of the ML workflow where it operates.

## Organization Principle

Steps are organized into **8 pipeline execution phases** that mirror the natural ML workflow progression:

1. **Data Acquisition & Loading** - Initial data retrieval
2. **Data Preparation & Feature Engineering** - Data transformation and feature creation
3. **Label Generation & Data Augmentation** - Label creation and data enhancement
4. **Model Training** - Model learning and optimization
5. **Model Inference & Evaluation** - Prediction and performance assessment
6. **Model Post-Processing & Calibration** - Score calibration and adjustments
7. **Model Deployment & Serving** - Model packaging and registration
8. **Utilities & Cross-Cutting Concerns** - Helper steps used across phases

## Step Registry Location

All steps are registered in: `src/cursus/registry/step_names_original.py`

---

## Step Component Architecture

Each pipeline step in the Cursus framework is built from **up to 5 core data structures** depending on the step type:

### Processing & Training Steps (Full Architecture)

1. **Script** (`src/cursus/steps/scripts/`) - The executable Python script that performs the actual work
2. **Script Contract** (`src/cursus/steps/contracts/`) - Defines the contract between script inputs/outputs
3. **Step Specification** (`src/cursus/steps/specs/`) - Formal specification of step behavior and dependencies
4. **Step Configuration** (`src/cursus/steps/configs/`) - Pydantic configuration class with three-tier design
5. **Step Builder** (`src/cursus/steps/builders/`) - Builder class that constructs the SageMaker step

### CreateModel, Lambda & Transform Steps (No Script)

1. ~~Script~~ - Not applicable (uses SageMaker built-in functionality)
2. ~~Script Contract~~ - Not applicable
3. **Step Specification** - Formal specification of step behavior
4. **Step Configuration** - Configuration class
5. **Step Builder** - Builder class

### Special Cases

**CradleDataLoading** (Customized Processing):
- Has Contract but no Script (uses external Cradle service)
- Has Specification, Configuration, Builder

**MimsModelRegistration** (Customized Processing):
- No Script (uses MIMS service)
- Custom sagemaker_step_type: `MimsModelRegistrationProcessing`
- Has Specification, Configuration, Builder

### Component Notation

In the step entries below, components are marked as:
- ✅ Component exists with file location
- ➖ Component not applicable for this step type
- ⚠️ Special implementation (see notes)

---

## Component Architecture Summary Table

This table provides a quick overview of which components exist for each registered step:

| Step Name | Script | Contract | Spec | Config | Builder | Notes |
|-----------|--------|----------|------|--------|---------|-------|
| **Phase 1: Data Acquisition & Loading** |
| DummyDataLoading | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| CradleDataLoading | - | ✓ | ✓ | ✓ | ✓ | Special: External service |
| **Phase 2: Data Preparation & Feature Engineering** |
| TabularPreprocessing | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| TemporalSequenceNormalization | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| TemporalFeatureEngineering | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| StratifiedSampling | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| RiskTableMapping | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| MissingValueImputation | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| FeatureSelection | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| CurrencyConversion | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| **Phase 3: Label Generation & Data Augmentation** |
| BedrockPromptTemplateGeneration | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| BedrockProcessing | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| BedrockBatchProcessing | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| LabelRulesetGeneration | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| LabelRulesetExecution | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| ActiveSampleSelection | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| PseudoLabelMerge | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| **Phase 4: Model Training** |
| PyTorchTraining | ✓ | ✓ | ✓ | ✓ | ✓ | Full Training |
| XGBoostTraining | ✓ | ✓ | ✓ | ✓ | ✓ | Full Training |
| LightGBMTraining | ✓ | ✓ | ✓ | ✓ | ✓ | Full Training |
| LightGBMMTTraining | ✓ | ✓ | ✓ | ✓ | ✓ | Full Training |
| DummyTraining | ✓ | ✓ | ✓ | ✓ | ✓ | Special: Processing type |
| **Phase 5: Model Inference & Evaluation** |
| XGBoostModelEval | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| XGBoostModelInference | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| LightGBMModelEval | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| LightGBMModelInference | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| LightGBMMTModelInference | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| PyTorchModelEval | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| PyTorchModelInference | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| ModelMetricsComputation | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| ModelWikiGenerator | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| PyTorchModel | - | - | ✓ | ✓ | ✓ | CreateModel: No script |
| XGBoostModel | - | - | ✓ | ✓ | ✓ | CreateModel: No script |
| **Phase 6: Model Post-Processing & Calibration** |
| ModelCalibration | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| PercentileModelCalibration | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| **Phase 7: Model Deployment & Serving** |
| Package | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| Registration | - | - | ✓ | ✓ | ✓ | Special: MIMS service |
| Payload | ✓ | ✓ | ✓ | ✓ | ✓ | Full Processing |
| BatchTransform | - | - | ✓ | ✓ | ✓ | Transform: No script |
| **Phase 8: Utilities & Cross-Cutting** |
| HyperparameterPrep | - | - | ✓ | ✓ | ✓ | Lambda: No script |

**Legend:**
- ✓ = Component exists
- \- = Component not applicable for this step type

**Summary Statistics:**
- **Total Steps**: 37
- **Steps with Full Architecture** (5 components): 32 steps
  - 28 Processing steps (including DummyTraining)
  - 4 Training steps
- **Steps with Partial Architecture** (3 components): 5 steps
  - 2 CreateModel steps (PyTorchModel, XGBoostModel)
  - 1 Transform step (BatchTransform)
  - 1 Lambda step (HyperparameterPrep)
  - 1 Special MIMS step (Registration)
- **Special Cases**: 2 steps
  - CradleDataLoading: Contract but no script (external service)
  - DummyTraining: Processing type despite training functionality

---

## Phase 1: Data Acquisition & Loading

**Purpose**: Initial data retrieval and loading from various sources into the pipeline.

### Processing Steps

#### DummyDataLoading
- **Type**: Processing
- **Description**: Dummy data loading step that processes user-provided data instead of calling Cradle services

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/dummy_data_loading.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/dummy_data_loading_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_dummy_data_loading_step.py` (`DummyDataLoadingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_dummy_data_loading_step.py` (`DummyDataLoadingStepBuilder`)

**Use Case**: Testing and development without Cradle dependencies

#### CradleDataLoading
- **Type**: Processing (custom sagemaker_step_type)
- **Description**: Cradle data loading step for production data retrieval

**Component Architecture** (Special Case - External Service):
1. ➖ **Script**: Not applicable (uses external Cradle service)
2. ⚠️ **Contract**: `src/cursus/steps/contracts/cradle_data_loading_contract.md` (Special: contract without script)
3. ✅ **Specification**: Defined in step specification system
4. ✅ **Configuration**: `src/cursus/steps/configs/config_cradle_data_load_step.py` (`CradleDataLoadingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_cradle_data_load_step.py` (`CradleDataLoadingStepBuilder`)

**Use Case**: Production data loading from Cradle services

---

## Phase 2: Data Preparation & Feature Engineering

**Purpose**: Transform raw data into features suitable for model training through preprocessing, normalization, and feature engineering operations.

### Processing Steps

#### TabularPreprocessing
- **Type**: Processing
- **Description**: Tabular data preprocessing with multi-variant support for training, testing, validation, and calibration

**Component Architecture**:
1. ✅ **Script**: `src/cursus/steps/scripts/tabular_preprocessing.py`
2. ✅ **Contract**: `src/cursus/steps/contracts/tabular_preprocess_contract.md`
3. ✅ **Specification**: Defined in step specification system
4. ✅ **Configuration**: `src/cursus/steps/configs/config_tabular_preprocessing_step.py` (`TabularPreprocessingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_tabular_preprocessing_step.py` (`TabularPreprocessingStepBuilder`)

**Key Features**:
- Multi-variant support (training, testing, validation, calibration)
- Job type-based processing
- Portable path support
- Framework: Pandas/Scikit-learn

**Use Case**: Standard tabular data preprocessing for ML pipelines

#### TemporalSequenceNormalization
- **Type**: Processing
- **Description**: Temporal sequence normalization with configurable sequence operations for time-series data

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/temporal_sequence_normalization.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/temporal_sequence_normalization_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_temporal_sequence_normalization_step.py` (`TemporalSequenceNormalizationConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_temporal_sequence_normalization_step.py` (`TemporalSequenceNormalizationStepBuilder`)

**Use Case**: Time-series data normalization for temporal models

#### TemporalFeatureEngineering
- **Type**: Processing
- **Description**: Extracts comprehensive temporal features from normalized sequences for ML models

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/temporal_feature_engineering.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/temporal_feature_engineering_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_temporal_feature_engineering_step.py` (`TemporalFeatureEngineeringConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_temporal_feature_engineering_step.py` (`TemporalFeatureEngineeringStepBuilder`)

**Use Case**: Feature extraction from temporal sequences

#### StratifiedSampling
- **Type**: Processing
- **Description**: Stratified sampling with multiple allocation strategies for class imbalance, causal analysis, and variance optimization

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/stratified_sampling.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/stratified_sampling_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_stratified_sampling_step.py` (`StratifiedSamplingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_stratified_sampling_step.py` (`StratifiedSamplingStepBuilder`)

**Use Case**: Balanced sampling for imbalanced datasets

#### RiskTableMapping
- **Type**: Processing
- **Description**: Risk table mapping for categorical features with job type variant support

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/risk_table_mapping.py`
2. ✅ **Contract**: `src/cursus/steps/contracts/risk_table_mapping_contract.md`
3. ✅ **Specification**: `src/cursus/steps/specs/risk_table_mapping_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_risk_table_mapping_step.py` (`RiskTableMappingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_risk_table_mapping_step.py` (`RiskTableMappingStepBuilder`)

**Key Features**:
- Risk score mapping
- Job type variants
- Custom transformation logic

**Use Case**: Categorical feature transformation using risk tables

#### MissingValueImputation
- **Type**: Processing
- **Description**: Missing value imputation using statistical methods (mean, median, mode, constant) with pandas-safe values

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/missing_value_imputation.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/missing_value_imputation_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_missing_value_imputation_step.py` (`MissingValueImputationConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_missing_value_imputation_step.py` (`MissingValueImputationStepBuilder`)

**Use Case**: Handling missing data in datasets

#### FeatureSelection
- **Type**: Processing
- **Description**: Feature selection using multiple statistical and ML-based methods with ensemble combination strategies

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/feature_selection.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/feature_selection_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_feature_selection_step.py` (`FeatureSelectionConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_feature_selection_step.py` (`FeatureSelectionStepBuilder`)

**Use Case**: Dimensionality reduction and feature importance analysis

#### CurrencyConversion
- **Type**: Processing
- **Description**: Currency conversion processing for financial data

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/currency_conversion.py`
2. ✅ **Contract**: `src/cursus/steps/contracts/currency_conversion_contract.md`
3. ✅ **Specification**: `src/cursus/steps/specs/currency_conversion_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_currency_conversion_step.py` (`CurrencyConversionConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_currency_conversion_step.py` (`CurrencyConversionStepBuilder`)

**Use Case**: Multi-currency data normalization

---

## Phase 3: Label Generation & Data Augmentation

**Purpose**: Generate labels through AI models, rule-based systems, or active learning, and augment training data for improved model performance.

### Processing Steps

#### BedrockPromptTemplateGeneration
- **Type**: Processing
- **Description**: Creates structured prompt templates for classification tasks using the 5-component architecture pattern

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/bedrock_prompt_template_generation.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/bedrock_prompt_template_generation_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_bedrock_prompt_template_generation_step.py` (`BedrockPromptTemplateGenerationConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_bedrock_prompt_template_generation_step.py` (`BedrockPromptTemplateGenerationStepBuilder`)

**Key Features**:
- 5-component architecture pattern
- Classification task support
- Validation schema generation

**Use Case**: Automated prompt template creation for Bedrock AI models

#### BedrockProcessing
- **Type**: Processing
- **Description**: Processes input data through AWS Bedrock models using generated prompt templates and validation schemas

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/bedrock_processing.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/bedrock_processing_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_bedrock_processing_step.py` (`BedrockProcessingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_bedrock_processing_step.py` (`BedrockProcessingStepBuilder`)

**Use Case**: Real-time AI inference with Bedrock models

#### BedrockBatchProcessing
- **Type**: Processing
- **Description**: AWS Bedrock batch inference with automatic fallback to real-time processing for cost-efficient large dataset processing

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/bedrock_batch_processing.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/bedrock_batch_processing_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_bedrock_batch_processing_step.py` (`BedrockBatchProcessingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_bedrock_batch_processing_step.py` (`BedrockBatchProcessingStepBuilder`)

**Use Case**: Cost-efficient batch inference with Bedrock

#### LabelRulesetGeneration
- **Type**: Processing
- **Description**: Validates and optimizes user-defined classification rules for transparent, maintainable rule-based label mapping

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/label_ruleset_generation.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/label_ruleset_generation_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_label_ruleset_generation_step.py` (`LabelRulesetGenerationConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_label_ruleset_generation_step.py` (`LabelRulesetGenerationStepBuilder`)

**Use Case**: Rule-based label generation system setup

#### LabelRulesetExecution
- **Type**: Processing
- **Description**: Applies validated rulesets to processed data to generate classification labels using priority-based rule evaluation

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/label_ruleset_execution.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/label_ruleset_execution_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_label_ruleset_execution_step.py` (`LabelRulesetExecutionConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_label_ruleset_execution_step.py` (`LabelRulesetExecutionStepBuilder`)

**Use Case**: Applying rule-based labels to data

#### ActiveSampleSelection
- **Type**: Processing
- **Description**: Intelligently selects high-value samples from model predictions for Semi-Supervised Learning (SSL) or Active Learning workflows

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/active_sample_selection.py`
2. ✅ **Contract**: `src/cursus/steps/contracts/active_sample_selection_contract.md`
3. ✅ **Specification**: `src/cursus/steps/specs/active_sample_selection_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_active_sample_selection_step.py` (`ActiveSamplingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_active_sample_selection_step.py` (`ActiveSamplingStepBuilder`)

**Key Features**:
- Three sampling strategies: uncertainty, diversity, BADGE
- Model-agnostic design
- Works with XGBoost/LightGBM/PyTorch/Bedrock outputs
- Embedding support for enhanced diversity
- Configurable batch selection

**Use Case**: Active learning sample selection for SSL pipelines

#### PseudoLabelMerge
- **Type**: Processing
- **Description**: Intelligently combines labeled base data with pseudo-labeled or augmented samples for Semi-Supervised Learning (SSL) and Active Learning workflows

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/pseudo_label_merge.py`
2. ✅ **Contract**: `src/cursus/steps/contracts/pseudo_label_merge_contract.md`
3. ✅ **Specification**: `src/cursus/steps/specs/pseudo_label_merge_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_pseudo_label_merge_step.py` (`PseudoLabelMergeConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_pseudo_label_merge_step.py` (`PseudoLabelMergeStepBuilder`)

**Key Features**:
- Split-aware merge for training jobs
- Auto-inferred split ratios
- Simple merge for validation/testing jobs
- Data format preservation (CSV/TSV/Parquet)
- Schema alignment and provenance tracking

**Use Case**: Merging labeled and pseudo-labeled data in SSL pipelines

**Pipeline Flow**: TabularPreprocessing → (ModelInference → ActiveSampleSelection) → PseudoLabelMerge → Training

---

## Phase 4: Model Training

**Purpose**: Train machine learning models using prepared data and hyperparameters.

### Training Steps

#### PyTorchTraining
- **Type**: Training
- **Description**: PyTorch model training step for deep learning models

**Component Architecture** (Training Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/pytorch_training.py`
2. ✅ **Contract**: `src/cursus/steps/contracts/pytorch_train_contract.md`
3. ✅ **Specification**: `src/cursus/steps/specs/pytorch_training_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_pytorch_training_step.py` (`PyTorchTrainingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_pytorch_training_step.py` (`PyTorchTrainingStepBuilder`)

**Use Case**: Deep learning model training with PyTorch

#### XGBoostTraining
- **Type**: Training
- **Description**: XGBoost model training with support for Semi-Supervised Learning

**Component Architecture** (Training Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/xgboost_training.py`
2. ✅ **Contract**: `src/cursus/steps/contracts/xgboost_train_contract.md`
3. ✅ **Specification**: Defined in step specification system
4. ✅ **Configuration**: `src/cursus/steps/configs/config_xgboost_training_step.py` (`XGBoostTrainingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_xgboost_training_step.py` (`XGBoostTrainingStepBuilder`)

**Key Features**:
- SSL support with job_type parameter
- Multi-variant training
- Integration with pseudo-labeling workflows

**Use Case**: Gradient boosting model training

#### LightGBMTraining
- **Type**: Training
- **Description**: LightGBM model training using built-in SageMaker algorithm

**Component Architecture** (Training Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/lightgbm_training.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/lightgbm_training_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_lightgbm_training_step.py` (`LightGBMTrainingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_lightgbm_training_step.py` (`LightGBMTrainingStepBuilder`)

**Use Case**: Efficient gradient boosting with LightGBM

#### LightGBMMTTraining
- **Type**: Training
- **Description**: LightGBM multi-task training with adaptive weighting and knowledge distillation

**Component Architecture** (Training Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/lightgbmmt_training.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/lightgbmmt_training_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_lightgbmmt_training_step.py` (`LightGBMMTTrainingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_lightgbmmt_training_step.py` (`LightGBMMTTrainingStepBuilder`)

**Key Features**:
- Multi-task learning support
- Adaptive task weighting
- Knowledge distillation capabilities

**Use Case**: Multi-task learning scenarios

### Processing Steps (Special)

#### DummyTraining
- **Type**: Processing
- **Description**: Training step that uses a pretrained model

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/dummy_training.py`
2. ✅ **Contract**: `src/cursus/steps/contracts/dummy_training_contract.md`
3. ✅ **Specification**: `src/cursus/steps/specs/dummy_training_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_dummy_training_step.py` (`DummyTrainingConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_dummy_training_step.py` (`DummyTrainingStepBuilder`)

**Note**: Classified as Processing type despite training-related functionality

**Use Case**: Testing pipelines with pretrained models

---

## Phase 5: Model Inference & Evaluation

**Purpose**: Generate predictions from trained models and evaluate their performance using various metrics.

### Processing Steps

#### XGBoostModelEval
- **Type**: Processing
- **Description**: XGBoost model evaluation with job type support

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/model_evaluation_xgb.py`
2. ✅ **Contract**: `src/cursus/steps/contracts/model_evaluation_contract.md`
3. ✅ **Specification**: `src/cursus/steps/specs/xgboost_model_eval_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_xgboost_model_eval_step.py` (`XGBoostModelEvalConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_xgboost_model_eval_step.py` (`XGBoostModelEvalStepBuilder`)

**Key Features**:
- Model evaluation processing
- Job type support
- Integration with training pipeline

**Use Case**: Evaluating XGBoost model performance

#### XGBoostModelInference
- **Type**: Processing
- **Description**: XGBoost model inference for prediction generation without metrics computation

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/xgboost_model_inference.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/xgboost_model_inference_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_xgboost_model_inference_step.py` (`XGBoostModelInferenceConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_xgboost_model_inference_step.py` (`XGBoostModelInferenceStepBuilder`)

**Use Case**: Generating predictions with XGBoost models

#### LightGBMModelEval
- **Type**: Processing
- **Description**: LightGBM model evaluation step

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/lightgbm_model_eval.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/lightgbm_model_eval_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_lightgbm_model_eval_step.py` (`LightGBMModelEvalConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_lightgbm_model_eval_step.py` (`LightGBMModelEvalStepBuilder`)

**Use Case**: Evaluating LightGBM model performance

#### LightGBMModelInference
- **Type**: Processing
- **Description**: LightGBM model inference for prediction generation without metrics

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/lightgbm_model_inference.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/lightgbm_model_inference_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_lightgbm_model_inference_step.py` (`LightGBMModelInferenceConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_lightgbm_model_inference_step.py` (`LightGBMModelInferenceStepBuilder`)

**Use Case**: Generating predictions with LightGBM models

#### LightGBMMTModelInference
- **Type**: Processing
- **Description**: LightGBMMT multi-task model inference for generating N independent binary task predictions

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/lightgbmmt_model_inference.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/lightgbmmt_model_inference_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_lightgbmmt_model_inference_step.py` (`LightGBMMTModelInferenceConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_lightgbmmt_model_inference_step.py` (`LightGBMMTModelInferenceStepBuilder`)

**Key Features**:
- Multi-task prediction
- Per-task calibration support
- XGBoost multiclass output format compatibility

**Use Case**: Multi-task prediction generation

#### PyTorchModelEval
- **Type**: Processing
- **Description**: PyTorch model evaluation step for deep learning models

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/pytorch_model_eval.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/pytorch_model_eval_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_pytorch_model_eval_step.py` (`PyTorchModelEvalConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_pytorch_model_eval_step.py` (`PyTorchModelEvalStepBuilder`)

**Use Case**: Evaluating PyTorch model performance

#### PyTorchModelInference
- **Type**: Processing
- **Description**: PyTorch model inference for prediction generation without metrics

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/pytorch_model_inference.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/pytorch_model_inference_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_pytorch_model_inference_step.py` (`PyTorchModelInferenceConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_pytorch_model_inference_step.py` (`PyTorchModelInferenceStepBuilder`)

**Use Case**: Generating predictions with PyTorch models

#### ModelMetricsComputation
- **Type**: Processing
- **Description**: Comprehensive model performance evaluation with multiple metrics

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/model_metrics_computation.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/model_metrics_computation_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_model_metrics_computation_step.py` (`ModelMetricsComputationConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_model_metrics_computation_step.py` (`ModelMetricsComputationStepBuilder`)

**Use Case**: Computing comprehensive evaluation metrics

#### ModelWikiGenerator
- **Type**: Processing
- **Description**: Automated model documentation creation

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/model_wiki_generator.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/model_wiki_generator_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_model_wiki_generator_step.py` (`ModelWikiGeneratorConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_model_wiki_generator_step.py` (`ModelWikiGeneratorStepBuilder`)

**Use Case**: Generating model documentation automatically

### CreateModel Steps

#### PyTorchModel
- **Type**: CreateModel
- **Description**: PyTorch model creation step for SageMaker model registration

**Component Architecture** (CreateModel Step - No Script):
1. ➖ **Script**: Not applicable (uses SageMaker built-in CreateModel)
2. ➖ **Contract**: Not applicable
3. ✅ **Specification**: Defined in step specification system
4. ✅ **Configuration**: `src/cursus/steps/configs/config_pytorch_model_step.py` (`PyTorchModelConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_pytorch_model_step.py` (`PyTorchModelStepBuilder`)

**Use Case**: Creating PyTorch model artifacts for deployment

#### XGBoostModel
- **Type**: CreateModel
- **Description**: XGBoost model creation step for SageMaker model registration

**Component Architecture** (CreateModel Step - No Script):
1. ➖ **Script**: Not applicable (uses SageMaker built-in CreateModel)
2. ➖ **Contract**: Not applicable
3. ✅ **Specification**: `src/cursus/steps/specs/xgboost_model_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_xgboost_model_step.py` (`XGBoostModelConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_xgboost_model_step.py` (`XGBoostModelStepBuilder`)

**Use Case**: Creating XGBoost model artifacts for deployment

---

## Phase 6: Model Post-Processing & Calibration

**Purpose**: Calibrate model scores to improve probability estimates and create percentile mappings for consistent risk interpretation.

### Processing Steps

#### ModelCalibration
- **Type**: Processing
- **Description**: Calibrates model prediction scores to accurate probabilities

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/model_calibration.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/model_calibration_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_model_calibration_step.py` (`ModelCalibrationConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_model_calibration_step.py` (`ModelCalibrationStepBuilder`)

**Recent Enhancements**:
- Job type variant support
- Alignment with other processing steps
- Improved dependency resolution

**Use Case**: Probability calibration for better uncertainty estimates

#### PercentileModelCalibration
- **Type**: Processing
- **Description**: Creates percentile mapping from model scores using ROC curve analysis

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/percentile_model_calibration.py`
2. ✅ **Contract**: TBD (check contracts directory)
3. ✅ **Specification**: `src/cursus/steps/specs/percentile_model_calibration_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_percentile_model_calibration_step.py` (`PercentileModelCalibrationConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_percentile_model_calibration_step.py` (`PercentileModelCalibrationStepBuilder`)

**Use Case**: Creating consistent risk interpretation scores

---

## Phase 7: Model Deployment & Serving

**Purpose**: Package, register, and deploy trained models for production serving.

### Processing Steps

#### Package
- **Type**: Processing
- **Description**: Model packaging step for deployment preparation

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/mims_package.py`
2. ✅ **Contract**: `src/cursus/steps/contracts/mims_package_contract.md`
3. ✅ **Specification**: `src/cursus/steps/specs/package_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_package_step.py` (`PackageConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_package_step.py` (`PackageStepBuilder`)

**Use Case**: Packaging models for MIMS registration

#### Registration
- **Type**: MimsModelRegistrationProcessing (custom)
- **Description**: Model registration step for MIMS

**Component Architecture** (Special Case - MIMS Service):
1. ➖ **Script**: Not applicable (uses MIMS service)
2. ➖ **Contract**: Not applicable
3. ✅ **Specification**: Defined in step specification system
4. ✅ **Configuration**: `src/cursus/steps/configs/config_mims_registration_step.py` (`RegistrationConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_mims_registration_step.py` (`RegistrationStepBuilder`)

**Note**: Custom sagemaker_step_type: `MimsModelRegistrationProcessing`

**Use Case**: Registering models in MIMS for production deployment

#### Payload
- **Type**: Processing
- **Description**: Payload testing step for model endpoints

**Component Architecture** (Processing Step - Full):
1. ✅ **Script**: `src/cursus/steps/scripts/mims_payload.py`
2. ✅ **Contract**: `src/cursus/steps/contracts/mims_payload_contract.md`
3. ✅ **Specification**: `src/cursus/steps/specs/payload_spec.py`
4. ✅ **Configuration**: `src/cursus/steps/configs/config_payload_step.py` (`PayloadConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_payload_step.py` (`PayloadStepBuilder`)

**Use Case**: Testing model endpoint responses

### Transform Steps

#### BatchTransform
- **Type**: Transform
- **Description**: Batch transform step for large-scale inference

**Component Architecture** (Transform Step - No Script):
1. ➖ **Script**: Not applicable (uses SageMaker built-in Transform)
2. ➖ **Contract**: Not applicable
3. ✅ **Specification**: Defined in step specification system
4. ✅ **Configuration**: `src/cursus/steps/configs/config_batch_transform_step.py` (`BatchTransformConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_batch_transform_step.py` (`BatchTransformStepBuilder`)

**Use Case**: Batch inference on large datasets

---

## Phase 8: Utilities & Cross-Cutting Concerns

**Purpose**: Helper steps and utilities used across different pipeline phases.

### Lambda Steps

#### HyperparameterPrep
- **Type**: Lambda
- **Description**: Hyperparameter preparation step for training configuration

**Component Architecture** (Lambda Step - No Script):
1. ➖ **Script**: Not applicable (uses Lambda function)
2. ➖ **Contract**: Not applicable
3. ✅ **Specification**: Defined in step specification system
4. ✅ **Configuration**: `src/cursus/steps/configs/config_hyperparameter_prep_step.py` (`HyperparameterPrepConfig`)
5. ✅ **Builder**: `src/cursus/steps/builders/builder_hyperparameter_prep_step.py` (`HyperparameterPrepStepBuilder`)

**Use Case**: Preparing hyperparameters for training steps

---

## Step Type Summary

### By SageMaker Step Type

**Processing Steps** (28 total):
- Data Loading: 2 steps
- Data Preprocessing: 8 steps
- Bedrock Processing: 3 steps
- Label Generation: 2 steps
- Active Learning: 2 steps
- Model Evaluation: 8 steps
- Model Calibration: 2 steps
- Model Deployment: 3 steps
- Special: 1 step (DummyTraining)

**Training Steps** (4 total):
- PyTorchTraining
- XGBoostTraining
- LightGBMTraining
- LightGBMMTTraining

**CreateModel Steps** (2 total):
- PyTorchModel
- XGBoostModel

**Transform Steps** (1 total):
- BatchTransform

**Lambda Steps** (1 total):
- HyperparameterPrep

**Custom Type Steps** (1 total):
- Registration (MimsModelRegistrationProcessing)

### Total: 37 Registered Steps

---

## Step Categories by Functionality

### Data Steps (10)
Data acquisition, preprocessing, and feature engineering
- CradleDataLoading, DummyDataLoading
- TabularPreprocessing, TemporalSequenceNormalization, TemporalFeatureEngineering
- StratifiedSampling, RiskTableMapping, MissingValueImputation
- FeatureSelection, CurrencyConversion

### Label Generation Steps (7)
AI-based and rule-based label generation with active learning
- BedrockPromptTemplateGeneration, BedrockProcessing, BedrockBatchProcessing
- LabelRulesetGeneration, LabelRulesetExecution
- ActiveSampleSelection, PseudoLabelMerge

### Training Steps (5)
Model training including multi-task learning
- PyTorchTraining, XGBoostTraining, LightGBMTraining, LightGBMMTTraining
- DummyTraining

### Evaluation Steps (11)
Model inference, evaluation, and metrics computation
- XGBoostModelEval, XGBoostModelInference
- LightGBMModelEval, LightGBMModelInference, LightGBMMTModelInference
- PyTorchModelEval, PyTorchModelInference
- ModelMetricsComputation, ModelWikiGenerator

### Calibration Steps (2)
Score calibration and risk mapping
- ModelCalibration, PercentileModelCalibration

### Deployment Steps (6)
Model packaging, registration, and serving
- Package, Registration, Payload
- PyTorchModel, XGBoostModel
- BatchTransform

### Utility Steps (1)
Cross-cutting helper functionality
- HyperparameterPrep

---

## Common Pipeline Patterns

### Standard Training Pipeline
```
CradleDataLoading → TabularPreprocessing → XGBoostTraining → 
XGBoostModelEval → ModelCalibration → Package → Registration
```

### Semi-Supervised Learning Pipeline
```
CradleDataLoading → TabularPreprocessing → 
XGBoostTraining (base) → XGBoostModelInference → 
ActiveSampleSelection → PseudoLabelMerge → 
XGBoostTraining (fine-tune) → XGBoostModelEval
```

### AI-Assisted Labeling Pipeline
```
CradleDataLoading → TabularPreprocessing → 
BedrockPromptTemplateGeneration → BedrockBatchProcessing → 
PseudoLabelMerge → XGBoostTraining → XGBoostModelEval
```

### Rule-Based Labeling Pipeline
```
CradleDataLoading → TabularPreprocessing → 
LabelRulesetGeneration → LabelRulesetExecution → 
XGBoostTraining → XGBoostModelEval
```

### Deep Learning Pipeline
```
CradleDataLoading → TabularPreprocessing → 
PyTorchTraining → PyTorchModelEval → 
ModelCalibration → PyTorchModel → Registration
```

---

## Quick Reference by Use Case

### For Data Scientists
- **Data Preparation**: TabularPreprocessing, FeatureSelection, MissingValueImputation
- **Model Training**: XGBoostTraining, PyTorchTraining, LightGBMTraining
- **Model Evaluation**: XGBoostModelEval, PyTorchModelEval, ModelMetricsComputation
- **Calibration**: ModelCalibration, PercentileModelCalibration

### For ML Engineers
- **Pipeline Setup**: CradleDataLoading, HyperparameterPrep
- **Model Deployment**: Package, Registration, Payload, BatchTransform
- **Model Creation**: XGBoostModel, PyTorchModel

### For Active Learning
- **Sample Selection**: ActiveSampleSelection (uncertainty/diversity/BADGE strategies)
- **Data Merging**: PseudoLabelMerge
- **Inference**: XGBoostModelInference, PyTorchModelInference

### For AI/LLM Integration
- **Prompt Engineering**: BedrockPromptTemplateGeneration
- **AI Inference**: BedrockProcessing, BedrockBatchProcessing
- **Rule-Based Labels**: LabelRulesetGeneration, LabelRulesetExecution

---

## Related Documentation

### Entry Points
- [Step Design and Documentation Index](../00_entry_points/step_design_and_documentation_index.md)
- [Processing Steps Index](../00_entry_points/processing_steps_index.md)
- [Cursus Package Overview](../00_entry_points/cursus_package_overview.md)

### Implementation Details
- Step Builders: `slipbox/steps/builders/`
- Step Contracts: `slipbox/steps/contracts/`
- Step Scripts: `slipbox/steps/scripts/`
- Step Specifications: `slipbox/steps/specs/`

### Design Documentation
- Design Documents: `slipbox/1_design/`
- Implementation Plans: `slipbox/2_project_planning/`
- Analysis Reports: `slipbox/4_analysis/`

---

## Maintenance

**Last Updated**: 2025-11-18

**Update Triggers**:
- New step registration
- Step deprecation
- Functionality changes
- Pipeline pattern updates

**Maintenance Process**:
1. Update step registry in `src/cursus/registry/step_names_original.py`
2. Update this reference document
3. Update related index documents
4. Update design documentation if needed
