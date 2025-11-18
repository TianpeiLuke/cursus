---
tags:
  - entry_point
  - documentation
  - overview
  - pipeline_steps
keywords:
  - step design patterns
  - step builders
  - step contracts
  - step specifications
  - pipeline steps
  - documentation index
topics:
  - pipeline step design
  - step documentation
  - design patterns
language: python
date of note: 2025-11-18
---

# Step Design and Documentation Index

## Purpose

This entry point document provides a comprehensive index of all step-related design documentation and implementation references organized by **pipeline execution flow**. It serves as a navigation hub for understanding pipeline step architecture, design patterns, contracts, and implementations following the natural progression from data loading through model deployment.

## Organization Principle

This index is organized to mirror the **actual pipeline execution order**, grouping steps by their functional phase in the ML workflow. This makes it easier to:
- Understand how steps work together in sequence
- Find related functionality quickly
- Design complete end-to-end pipelines
- Navigate the full ML lifecycle from data to deployment

## Table of Contents

### [Part 1: Pipeline Flow - Design Documents by Execution Phase](#part-1-pipeline-flow---design-documents-by-execution-phase)
- [Phase 1: Data Acquisition & Loading](#phase-1-data-acquisition--loading)
- [Phase 2: Data Preparation & Feature Engineering](#phase-2-data-preparation--feature-engineering)
- [Phase 3: Label Generation & Data Augmentation](#phase-3-label-generation--data-augmentation)
- [Phase 4: Model Training](#phase-4-model-training)
- [Phase 5: Model Inference & Evaluation](#phase-5-model-inference--evaluation)
- [Phase 6: Model Post-Processing & Calibration](#phase-6-model-post-processing--calibration)
- [Phase 7: Model Deployment & Serving](#phase-7-model-deployment--serving)
- [Phase 8: Utilities & Cross-Cutting Concerns](#phase-8-utilities--cross-cutting-concerns)

### [Part 2: Step Implementation Documentation](#part-2-step-implementation-documentation)
- [Step Builders](#step-builders-slipboxstepsbuilders)
- [Step Contracts](#step-contracts-slipboxstepscontracts)
- [Step Scripts](#step-scripts-slipboxstepsscripts)
- [Step Specifications](#step-specifications-slipboxstepsspecs)

### [Part 3: System Architecture & General Principles](#part-3-system-architecture--general-principles)
- [Core Step Builder Patterns](#core-step-builder-patterns)
- [Step Alignment and Validation Patterns](#step-alignment-and-validation-patterns)
- [Step Architecture and Design](#step-architecture-and-design)
- [Configuration System](#configuration-system)
- [Step Improvements and Extensions](#step-improvements-and-extensions)
- [Step Testing and Validation](#step-testing-and-validation)
- [SageMaker Step Integration](#sagemaker-step-integration)
- [Step Catalog Systems](#step-catalog-systems)

### [Part 4: Developer Guides & Best Practices](#part-4-developer-guides--best-practices)
- [Getting Started](#getting-started)
- [Integration & Development](#integration--development)
- [Core Design Documents](#core-design-documents)

### Additional Sections
- [Navigation Tips](#navigation-tips)
- [Document Organization Principles](#document-organization-principles)
- [Contributing](#contributing)
- [See Also](#see-also)

---

## Part 1: Pipeline Flow - Design Documents by Execution Phase

### Phase 1: Data Acquisition & Loading

#### Registered Steps in This Phase

| Step Name | Step Type | Description |
|-----------|-----------|-------------|
| CradleDataLoading | Processing | Cradle data loading step |
| DummyDataLoading | Processing | Dummy data loading step that processes user-provided data instead of calling Cradle services |

**Data Loading Steps**
- [Cradle Data Load Config Helper Design](../1_design/cradle_data_load_config_helper_design.md) - Helper design for Cradle data loading configuration
- [Cradle Data Load Config UI Design](../1_design/cradle_data_load_config_ui_design.md) - UI design for Cradle data configuration
- [Cradle Data Load Config Single Page UI Design](../1_design/cradle_data_load_config_single_page_ui_design.md) - Single-page UI for Cradle configuration

**Implementation Documentation**
- [Data Load Step Builder (Cradle)](../steps/builders/data_load_step_cradle.md) - Step builder implementation
- [Cradle Data Loading Contract](../steps/contracts/cradle_data_loading_contract.md) - Contract definition
- [Data Loading Training Spec](../steps/specs/data_loading_training_spec.md) - Step specification

**Data Format & Quality**
- [Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md) - Design for automatic format preservation across pipeline scripts (CSV/TSV/Parquet)

### Phase 2: Data Preparation & Feature Engineering

#### Registered Steps in This Phase

| Step Name | Step Type | Description |
|-----------|-----------|-------------|
| TabularPreprocessing | Processing | Tabular data preprocessing step |
| TemporalSequenceNormalization | Processing | Temporal sequence normalization step for machine learning models with configurable sequence operations |
| TemporalFeatureEngineering | Processing | Temporal feature engineering step that extracts comprehensive temporal features from normalized sequences for machine learning models |
| StratifiedSampling | Processing | Stratified sampling step with multiple allocation strategies for class imbalance, causal analysis, and variance optimization |
| RiskTableMapping | Processing | Risk table mapping step for categorical features |
| MissingValueImputation | Processing | Missing value imputation step using statistical methods (mean, median, mode, constant) with pandas-safe values |
| FeatureSelection | Processing | Feature selection step using multiple statistical and ML-based methods with ensemble combination strategies |
| CurrencyConversion | Processing | Currency conversion processing step |

**Core Preprocessing**
- [Multi Sequence Preprocessing Design](../1_design/multi_sequence_preprocessing_design.md) - Design for multi-sequence preprocessing scripts
- [Temporal Sequence Normalization Design](../1_design/temporal_sequence_normalization_design.md) - Design for temporal sequence normalization scripts
- [Temporal Self Attention Scripts Analysis](../4_analysis/temporal_self_attention_scripts_analysis.md) - Detailed analysis of TSA scripts and preprocessing requirements

**Implementation Documentation**
- [Tabular Preprocessing Step Builder](../steps/builders/tabular_preprocessing_step.md) - Step builder implementation
- [Tabular Preprocess Contract](../steps/contracts/tabular_preprocess_contract.md) - Contract definition
- [Tabular Preprocess Documentation](../steps/scripts/tabular_preprocess_doc.md) - Script documentation
- [Risk Table Map Step Builder](../steps/builders/risk_table_map_step.md) - Risk table mapping step builder
- [Risk Table Mapping Contract](../steps/contracts/risk_table_mapping_contract.md) - Risk table mapping contract
- [Risk Table Mapping Documentation](../steps/scripts/risk_table_mapping_doc.md) - Risk table mapping script
- [Currency Conversion Step Builder](../steps/builders/currency_conversion_step.md) - Currency conversion step builder
- [Currency Conversion Contract](../steps/contracts/currency_conversion_contract.md) - Currency conversion contract
- [Currency Conversion Documentation](../steps/scripts/currency_conversion_doc.md) - Currency conversion script

**Feature Engineering & Selection**
- [Multi Sequence Feature Engineering Design](../1_design/multi_sequence_feature_engineering_design.md) - Design for multi-sequence feature engineering scripts
- [Temporal Feature Engineering Design](../1_design/temporal_feature_engineering_design.md) - Design for temporal feature engineering scripts
- [Feature Selection Script Design](../1_design/feature_selection_script_design.md) - Design for feature selection scripts

**Data Quality & Transformation**
- [Missing Value Imputation Design](../1_design/missing_value_imputation_design.md) - Design for missing value imputation scripts

**Sampling & Balancing**
- [Active Sampling Step Builder Patterns](../1_design/active_sampling_step_patterns.md) - Complete step builder patterns for intelligent sample selection in semi-supervised and active learning pipelines
- [Active Sampling Script Design](../1_design/active_sampling_script_design.md) - Modular sample selection engine with uncertainty, diversity, and BADGE strategies
- [Active Sampling BADGE Design](../1_design/active_sampling_badge.md) - BADGE algorithm for combining uncertainty and diversity in sample selection
- [Active Sampling Uncertainty Design](../1_design/active_sampling_uncertainty_margin_entropy.md) - Uncertainty-based sampling strategies
- [Active Sampling Core-Set Design](../1_design/active_sampling_core_set_leaf_core_set.md) - Diversity sampling with core-set and leaf core-set algorithms

### Phase 3: Label Generation & Data Augmentation

#### Registered Steps in This Phase

| Step Name | Step Type | Description |
|-----------|-----------|-------------|
| BedrockPromptTemplateGeneration | Processing | Bedrock prompt template generation step that creates structured prompt templates for classification tasks using the 5-component architecture pattern |
| BedrockProcessing | Processing | Bedrock processing step that processes input data through AWS Bedrock models using generated prompt templates and validation schemas |
| BedrockBatchProcessing | Processing | Bedrock batch processing step that provides AWS Bedrock batch inference capabilities with automatic fallback to real-time processing for cost-efficient large dataset processing |
| LabelRulesetGeneration | Processing | Label ruleset generation step that validates and optimizes user-defined classification rules for transparent, maintainable rule-based label mapping in ML training pipelines |
| LabelRulesetExecution | Processing | Label ruleset execution step that applies validated rulesets to processed data to generate classification labels using priority-based rule evaluation with execution-time field validation |
| ActiveSampleSelection | Processing | Active sample selection step that intelligently selects high-value samples from model predictions for Semi-Supervised Learning (SSL) or Active Learning workflows using confidence-based, uncertainty-based, diversity-based, or hybrid strategies |
| PseudoLabelMerge | Processing | Pseudo label merge step that intelligently combines labeled base data with pseudo-labeled or augmented samples for Semi-Supervised Learning (SSL) and Active Learning workflows with split-aware merge, auto-inferred split ratios, and provenance tracking |

**Bedrock AI Processing**
- [Bedrock Prompt Template Generation Step Patterns](../1_design/bedrock_prompt_template_generation_step_patterns.md) - Patterns for generating prompt templates in pipeline steps
- [Bedrock Prompt Template Generation Buyer Seller Example](../1_design/bedrock_prompt_template_generation_buyer_seller_example.md) - Concrete example of buyer-seller prompt generation
- [Bedrock Prompt Template Generation Input Formats](../1_design/bedrock_prompt_template_generation_input_formats.md) - Input format specifications for prompt generation
- [Bedrock Prompt Template Generation Output Design](../1_design/bedrock_prompt_template_generation_output_design.md) - Output design for prompt generation steps
- [Bedrock Processing Step Builder Patterns](../1_design/bedrock_processing_step_builder_patterns.md) - General processing step patterns for Bedrock integration
- [Bedrock Batch Processing Step Builder Patterns](../1_design/bedrock_batch_processing_step_builder_patterns.md) - Design patterns for building batch processing steps with Bedrock

**Label Ruleset System**
- [Label Ruleset Generation Step Patterns](../1_design/label_ruleset_generation_step_patterns.md) - Patterns for generating and validating label rulesets within pipeline steps
- [Label Ruleset Execution Step Patterns](../1_design/ruleset_execution_step_patterns.md) - Design patterns for executing label rulesets in pipeline steps
- [Label Ruleset Optimization Patterns](../1_design/label_ruleset_optimization_patterns.md) - Optimization strategies for label ruleset performance
- [Label Ruleset Generation Configuration Examples](../1_design/label_ruleset_generation_configuration_examples.md) - Complete working examples for binary, multiclass, and multilabel classification
- [Label Ruleset Multilabel Extension Design](../1_design/label_ruleset_multilabel_extension_design.md) - Design for multilabel support in rule-based labeling
- [Label Ruleset Multilabel Extension Implementation Plan](../2_project_planning/2025-11-11_label_ruleset_multilabel_extension_implementation_plan.md) - 5-phase implementation plan for multilabel support

**Semi-Supervised Learning & Data Augmentation**
- [Pseudo Label Merge Script Design](../1_design/pseudo_label_merge_script_design.md) - Unified data combination engine for merging labeled and pseudo-labeled data in SSL pipelines

### Phase 4: Model Training

#### Registered Steps in This Phase

| Step Name | Step Type | Description |
|-----------|-----------|-------------|
| PyTorchTraining | Training | PyTorch model training step |
| XGBoostTraining | Training | XGBoost model training step |
| LightGBMTraining | Training | LightGBM model training step using built-in algorithm |
| LightGBMMTTraining | Training | LightGBM multi-task training with adaptive weighting and knowledge distillation |
| DummyTraining | Processing | Training step that uses a pretrained model |

**Training Pipeline Architecture**
- [XGBoost Semi-Supervised Learning Pipeline Design](../1_design/xgboost_semi_supervised_learning_pipeline_design.md) - Complete SSL pipeline architecture with pseudo-labeling workflow
- [XGBoost Semi-Supervised Learning Training Design](../1_design/xgboost_semi_supervised_learning_training_design.md) - Training step extension with job_type support for SSL phases

**Implementation Documentation**
- [Training Step Builder (XGBoost)](../steps/builders/training_step_xgboost.md) - XGBoost training step builder
- [XGBoost Train Contract](../steps/contracts/xgboost_train_contract.md) - XGBoost training contract
- [Training Step Builder (PyTorch)](../steps/builders/training_step_pytorch.md) - PyTorch training step builder
- [PyTorch Train Contract](../steps/contracts/pytorch_train_contract.md) - PyTorch training contract
- [Dummy Training Contract](../steps/contracts/dummy_training_contract.md) - Dummy training contract
- [Dummy Training Documentation](../steps/scripts/dummy_training_doc.md) - Dummy training script

**Deep Learning Training**
- [Temporal Self Attention Model Design](../1_design/temporal_self_attention_model_design.md) - Design for temporal self-attention model architecture and training scripts

**Multi-Task Learning**
- [LightGBM Multi-Task Training Step Design](../1_design/lightgbm_multi_task_training_step_design.md) - Design for LightGBM multi-task/multi-label training
- [MTGBM Loss Functions Refactoring Design](../1_design/mtgbm_models_refactoring_design.md) - Comprehensive refactoring design for MTGBM loss function implementations
- [MTGBM Model Classes Refactoring Design](../1_design/mtgbm_model_classes_refactoring_design.md) - Architectural design for refactoring MTGBM model class implementations

**Implementation Plans**
- [LightGBMMT Implementation Part 1: Script Contract & Hyperparameters](../2_project_planning/2025-11-12_lightgbmmt_implementation_part1_script_contract_hyperparams.md) - Phase 1 implementation plan
- [LightGBMMT Implementation Part 2: Training Script Alignment](../2_project_planning/2025-11-12_lightgbmmt_implementation_part2_training_script_alignment.md) - Phase 2 implementation plan

**Analysis Documents**
- [LightGBMMT Multi-Task Implementation Analysis](../4_analysis/2025-11-10_lightgbmmt_multi_task_implementation_analysis.md) - Analysis of LightGBMMT framework
- [MTGBM Models Optimization Analysis](../4_analysis/2025-11-11_mtgbm_models_optimization_analysis.md) - Optimization opportunities analysis
- [MTGBM Pipeline Reusability Analysis](../4_analysis/2025-11-11_mtgbm_pipeline_reusability_analysis.md) - Pipeline reusability patterns
- [Python Refactored vs LightGBMMT Fork Comparison](../4_analysis/2025-11-12_python_refactored_vs_lightgbmmt_fork_comparison.md) - Comprehensive architecture comparison

### Phase 5: Model Inference & Evaluation

#### Registered Steps in This Phase

| Step Name | Step Type | Description |
|-----------|-----------|-------------|
| XGBoostModelEval | Processing | XGBoost model evaluation step |
| XGBoostModelInference | Processing | XGBoost model inference step for prediction generation without metrics |
| LightGBMModelEval | Processing | LightGBM model evaluation step |
| LightGBMModelInference | Processing | LightGBM model inference step for prediction generation without metrics |
| PyTorchModelEval | Processing | PyTorch model evaluation step |
| PyTorchModelInference | Processing | PyTorch model inference step for prediction generation without metrics |
| ModelMetricsComputation | Processing | Model metrics computation step for comprehensive performance evaluation |
| ModelWikiGenerator | Processing | Model wiki generator step for automated documentation creation |
| PyTorchModel | CreateModel | PyTorch model creation step |
| XGBoostModel | CreateModel | XGBoost model creation step |

**Model Inference**
- [XGBoost Model Inference Design](../1_design/xgboost_model_inference_design.md) - Design for XGBoost model inference scripts
- [PyTorch Model Evaluation Design](../1_design/pytorch_model_eval_design.md) - Design for PyTorch deep learning model evaluation with GPU/CPU support

**Implementation Documentation**
- [Model Evaluation Step Builder (XGBoost)](../steps/builders/model_eval_step_xgboost.md) - XGBoost evaluation step builder
- [Model Evaluation Contract](../steps/contracts/model_evaluation_contract.md) - Model evaluation contract
- [Model Evaluation XGBoost Documentation](../steps/scripts/model_evaluation_xgb_doc.md) - XGBoost evaluation script
- [Model Step Builder (PyTorch)](../steps/builders/model_step_pytorch.md) - PyTorch model step builder
- [Model Step Builder (XGBoost)](../steps/builders/model_step_xgboost.md) - XGBoost model step builder

**Model Evaluation & Metrics**
- [Model Metrics Computation Design](../1_design/model_metrics_computation_design.md) - Design for model metrics computation scripts
- [Model Wiki Generator Design](../1_design/model_wiki_generator_design.md) - Design for model wiki generation scripts
- [Inference Handler Spec Design](../1_design/inference_handler_spec_design.md) - Design for inference handler specifications
- [Inference Test Result Design](../1_design/inference_test_result_design.md) - Design for inference testing result models

### Phase 6: Model Post-Processing & Calibration

#### Registered Steps in This Phase

| Step Name | Step Type | Description |
|-----------|-----------|-------------|
| ModelCalibration | Processing | Calibrates model prediction scores to accurate probabilities |
| PercentileModelCalibration | Processing | Creates percentile mapping from model scores using ROC curve analysis for consistent risk interpretation |

**Model Calibration**
- [Percentile Model Calibration Design](../1_design/percentile_model_calibration_design.md) - Design for percentile model calibration scripts

**Implementation Documentation**
- [Model Calibration Documentation](../steps/scripts/model_calibration_doc.md) - Model calibration script

### Phase 7: Model Deployment & Serving

#### Registered Steps in This Phase

| Step Name | Step Type | Description |
|-----------|-----------|-------------|
| Package | Processing | Model packaging step |
| Registration | MimsModelRegistrationProcessing | Model registration step |
| Payload | Processing | Payload testing step |
| BatchTransform | Transform | Batch transform step |

**Model Packaging & Registration**
- [MIMS Packaging Step Builder](../steps/builders/mims_packaging_step.md) - MIMS packaging step builder
- [MIMS Package Contract](../steps/contracts/mims_package_contract.md) - MIMS package contract
- [MIMS Package Documentation](../steps/scripts/mims_package_doc.md) - MIMS packaging script
- [MIMS Payload Step Builder](../steps/builders/mims_payload_step.md) - MIMS payload step builder
- [MIMS Payload Contract](../steps/contracts/mims_payload_contract.md) - MIMS payload contract
- [MIMS Payload Documentation](../steps/scripts/mims_payload_doc.md) - MIMS payload script
- [MIMS Registration Step Builder](../steps/builders/mims_registration_step.md) - MIMS registration step builder
- [MIMS Registration Contract](../steps/contracts/mims_registration_contract.md) - MIMS registration contract
- [MODS MIMS Model Registration](../steps/scripts/MODS_MIMS_Model_Registration.md) - MODS MIMS registration
- [Batch Transform Step Builder](../steps/builders/batch_transform_step.md) - Batch transform step builder

### Phase 8: Utilities & Cross-Cutting Concerns

#### Registered Steps in This Phase

| Step Name | Step Type | Description |
|-----------|-----------|-------------|
| HyperparameterPrep | Lambda | Hyperparameter preparation step |

**Pipeline Utilities**
- [Conditional Step Builder Patterns](../1_design/conditional_step_builder_patterns.md) - Patterns for building conditional execution steps

**Implementation Documentation**
- [Hyperparameter Prep Step Builder](../steps/builders/hyperparameter_prep_step.md) - Hyperparameter preparation step builder
- [Hyperparameter Prep Contract](../steps/contracts/hyperparameter_prep_contract.md) - Hyperparameter prep contract
- [Contract Utils Documentation](../steps/scripts/contract_utils_doc.md) - Contract utilities

---

## Part 2: Step Implementation Documentation

### Step Builders (slipbox/steps/builders)

- [README](../steps/builders/README.md) - Overview of step builders
- [Batch Transform Step](../steps/builders/batch_transform_step.md)
- [Currency Conversion Step](../steps/builders/currency_conversion_step.md)
- [Data Load Step (Cradle)](../steps/builders/data_load_step_cradle.md)
- [Hyperparameter Prep Step](../steps/builders/hyperparameter_prep_step.md)
- [MIMS Packaging Step](../steps/builders/mims_packaging_step.md)
- [MIMS Payload Step](../steps/builders/mims_payload_step.md)
- [MIMS Registration Step](../steps/builders/mims_registration_step.md)
- [Model Evaluation Step (XGBoost)](../steps/builders/model_eval_step_xgboost.md)
- [Model Step (PyTorch)](../steps/builders/model_step_pytorch.md)
- [Model Step (XGBoost)](../steps/builders/model_step_xgboost.md)
- [Risk Table Map Step](../steps/builders/risk_table_map_step.md)
- [Tabular Preprocessing Step](../steps/builders/tabular_preprocessing_step.md)
- [Training Step (PyTorch)](../steps/builders/training_step_pytorch.md)
- [Training Step (XGBoost)](../steps/builders/training_step_xgboost.md)

### Step Contracts (slipbox/steps/contracts)

- [README](../steps/contracts/README.md) - Overview of step contracts
- [Cradle Data Loading Contract](../steps/contracts/cradle_data_loading_contract.md)
- [Currency Conversion Contract](../steps/contracts/currency_conversion_contract.md)
- [Dummy Training Contract](../steps/contracts/dummy_training_contract.md)
- [Hyperparameter Prep Contract](../steps/contracts/hyperparameter_prep_contract.md)
- [MIMS Package Contract](../steps/contracts/mims_package_contract.md)
- [MIMS Payload Contract](../steps/contracts/mims_payload_contract.md)
- [MIMS Registration Contract](../steps/contracts/mims_registration_contract.md)
- [Model Evaluation Contract](../steps/contracts/model_evaluation_contract.md)
- [PyTorch Train Contract](../steps/contracts/pytorch_train_contract.md)
- [Risk Table Mapping Contract](../steps/contracts/risk_table_mapping_contract.md)
- [Tabular Preprocess Contract](../steps/contracts/tabular_preprocess_contract.md)
- [XGBoost Train Contract](../steps/contracts/xgboost_train_contract.md)

### Step Scripts (slipbox/steps/scripts)

- [Contract Utils Documentation](../steps/scripts/contract_utils_doc.md)
- [Currency Conversion Documentation](../steps/scripts/currency_conversion_doc.md)
- [Dummy Training Documentation](../steps/scripts/dummy_training_doc.md)
- [MIMS Package Documentation](../steps/scripts/mims_package_doc.md)
- [MIMS Payload Documentation](../steps/scripts/mims_payload_doc.md)
- [Model Calibration Documentation](../steps/scripts/model_calibration_doc.md)
- [Model Evaluation XGBoost Documentation](../steps/scripts/model_evaluation_xgb_doc.md)
- [MODS MIMS Model Registration](../steps/scripts/MODS_MIMS_Model_Registration.md)
- [Risk Table Mapping Documentation](../steps/scripts/risk_table_mapping_doc.md)
- [Tabular Preprocess Documentation](../steps/scripts/tabular_preprocess_doc.md)

### Step Specifications (slipbox/steps/specs)

- [README](../steps/specs/README.md) - Overview of step specifications
- [Data Loading Training Spec](../steps/specs/data_loading_training_spec.md)

---

## Part 3: System Architecture & General Principles

### Core Step Builder Patterns

- [Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md) - General patterns for building processing steps
- [Training Step Builder Patterns](../1_design/training_step_builder_patterns.md) - Patterns for building training steps
- [Transform Step Builder Patterns](../1_design/transform_step_builder_patterns.md) - Patterns for building transform steps
- [Tuning Step Builder Patterns](../1_design/tuning_step_builder_patterns.md) - Patterns for building hyperparameter tuning steps
- [CreateModel Step Builder Patterns](../1_design/createmodel_step_builder_patterns.md) - Patterns for building model creation steps
- [Step Builder Patterns Summary](../1_design/step_builder_patterns_summary.md) - Comprehensive summary of all step builder patterns

### Step Alignment and Validation Patterns

- [Processing Step Alignment Validation Patterns](../1_design/processing_step_alignment_validation_patterns.md) - Validation patterns for processing steps
- [Training Step Alignment Validation Patterns](../1_design/training_step_alignment_validation_patterns.md) - Validation patterns for training steps
- [Transform Step Alignment Validation Patterns](../1_design/transform_step_alignment_validation_patterns.md) - Validation patterns for transform steps
- [CreateModel Step Alignment Validation Patterns](../1_design/createmodel_step_alignment_validation_patterns.md) - Validation patterns for model creation steps
- [RegisterModel Step Alignment Validation Patterns](../1_design/registermodel_step_alignment_validation_patterns.md) - Validation patterns for model registration steps
- [Utility Step Alignment Validation Patterns](../1_design/utility_step_alignment_validation_patterns.md) - Validation patterns for utility steps

### Step Architecture and Design

- [Step Builder](../1_design/step_builder.md) - Core step builder architecture and design
- [Step Specification](../1_design/step_specification.md) - Step specification format and requirements
- [Step Contract](../1_design/step_contract.md) - Contract definitions for pipeline steps
- [Step Config Resolver](../1_design/step_config_resolver.md) - Configuration resolution for steps
- [Step Builder Validation Rulesets Design](../1_design/step_builder_validation_rulesets_design.md) - Validation ruleset architecture

### Configuration System

- [Config Tiered Design](../1_design/config_tiered_design.md) - Three-tier configuration architecture
- [Config Manager Three Tier Implementation](../1_design/config_manager_three_tier_implementation.md) - Implementation of three-tier config system
- [Three Tier Config Design](../0_developer_guide/three_tier_config_design.md) - Developer guide for three-tier configuration

### Step Improvements and Extensions

- [Training Step Improvements](../1_design/training_step_improvements.md) - Enhancements to training step functionality
- [Packaging Step Improvements](../1_design/packaging_step_improvements.md) - Enhancements to packaging step functionality
- [Step Type Enhancement System Design](../1_design/step_type_enhancement_system_design.md) - System design for step type enhancements

### Step Testing and Validation

- [Universal Step Builder Test](../1_design/universal_step_builder_test.md) - Universal testing framework for step builders
- [Enhanced Universal Step Builder Tester Design](../1_design/enhanced_universal_step_builder_tester_design.md) - Enhanced testing capabilities
- [Universal Step Builder Test Scoring](../1_design/universal_step_builder_test_scoring.md) - Scoring system for step builder tests
- [Universal Step Builder Test Step Catalog Integration](../1_design/universal_step_builder_test_step_catalog_integration.md) - Integration with step catalog
- [Unified Alignment Tester Design](../1_design/unified_alignment_tester_design.md) - Unified testing for step alignment
- [Unified Alignment Tester Architecture](../1_design/unified_alignment_tester_architecture.md) - Architecture for alignment testing
- [Unified Alignment Tester Master Design](../1_design/unified_alignment_tester_master_design.md) - Master design document
- [Unified Alignment Tester Validation Ruleset](../1_design/unified_alignment_tester_validation_ruleset.md) - Validation rulesets
- [Unified Alignment Tester Import Resolution Design](../1_design/unified_alignment_tester_import_resolution_design.md) - Import resolution

### SageMaker Step Integration

- [SageMaker Step Type Classification Design](../1_design/sagemaker_step_type_classification_design.md) - Classification system for SageMaker step types
- [SageMaker Step Type Aware Unified Alignment Tester Design](../1_design/sagemaker_step_type_aware_unified_alignment_tester_design.md) - Type-aware testing
- [SageMaker Step Type Universal Builder Tester Design](../1_design/sagemaker_step_type_universal_builder_tester_design.md) - Universal builder testing
- [SageMaker Step Validation Requirements Specification](../1_design/sagemaker_step_validation_requirements_specification.md) - Validation requirements

### Step Catalog Systems

- [Unified Step Catalog Component Architecture Design](../1_design/unified_step_catalog_component_architecture_design.md) - Component architecture
- [Unified Step Catalog System Design](../1_design/unified_step_catalog_system_design.md) - System design overview
- [Unified Step Catalog System Expansion Design](../1_design/unified_step_catalog_system_expansion_design.md) - Expansion capabilities
- [Unified Step Catalog System Search Space Management Design](../1_design/unified_step_catalog_system_search_space_management_design.md) - Search space management

---

## Part 4: Developer Guides & Best Practices

### Getting Started

- [Adding New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md) - Guide for adding new pipeline steps
- [Step Builder Guide](../0_developer_guide/step_builder.md) - Developer guide for step builders
- [Step Specification Guide](../0_developer_guide/step_specification.md) - Guide for step specifications

### Integration & Development

- [Step Catalog Integration Guide](../0_developer_guide/step_catalog_integration_guide.md) - Integrating steps with the catalog
- [Script Contract Guide](../0_developer_guide/script_contract.md) - Understanding script contracts
- [Script Development Guide](../0_developer_guide/script_development_guide.md) - Developing step scripts

### Core Design Documents

- [Script Contract](../1_design/script_contract.md) - Core script contract design
- [Script Execution Spec Design](../1_design/script_execution_spec_design.md) - Script execution specification
- [Script Testability Refactoring](../1_design/script_testability_refactoring.md) - Testability improvements
- [Script Integration Testing System Design](../1_design/script_integration_testing_system_design.md) - Integration testing

---

## Navigation Tips

### For Pipeline Designers

1. **Start with Phase 1** (Data Loading) and progress through phases sequentially
2. Review relevant design documents for each phase you need
3. Check pipeline examples in `src/cursus/pipeline_catalog/shared_dags/`
4. Consult Step Implementation docs for concrete examples

### For New Developers

1. Start with **Part 4: Developer Guides** to understand the system
2. Review **Part 3: System Architecture** for design principles
3. Explore **Part 1: Pipeline Flow** to understand step functionality
4. Examine **Part 2: Step Implementation** for concrete code examples

### For Specific Functionality

**Data Processing**: See Phase 2 design documents

**AI/ML Features**: See Phase 3 (Bedrock, Label Ruleset)

**Training Workflows**: See Phase 4 design documents

**Model Evaluation**: See Phase 5 design documents

**Deployment**: See Phase 7 design documents

### For Validation and Testing

1. Review testing documents in **Part 3: System Architecture**
2. Check step-specific alignment validation patterns
3. Explore [Unified Alignment Tester Master Design](../1_design/unified_alignment_tester_master_design.md)

---

## Document Organization Principles

This index organizes step-related documentation into:

1. **Pipeline Flow Design** - Architectural patterns organized by pipeline execution phase
2. **Step Implementation** - Concrete implementations (builders, contracts, scripts, specs)
3. **System Architecture** - Core principles, patterns, and frameworks
4. **Developer Guides** - How-to guides and best practices

## Contributing

When adding new step-related documentation:

1. Follow the [Documentation YAML Frontmatter Standard](../6_resources/documentation_yaml_frontmatter_standard.md)
2. Place design documents in slipbox/1_design
3. Place implementation docs in appropriate slipbox/steps subdirectories
4. Update this index in the appropriate **pipeline phase section**
5. Use appropriate tags: `design`, `code`, `test`, etc.

## See Also

- [Cursus Package Overview](cursus_package_overview.md) - Main package entry point
- [Registered Steps Pipeline Reference](../steps/registered_steps_pipeline_reference.md) - Complete reference of all 37 registered steps with component architecture details
- [Processing Steps Index](processing_steps_index.md) - Processing step documentation
- [Pipeline Catalog Design](../1_design/pipeline_catalog_design.md) - Pipeline catalog architecture
- [Step Catalog Integration Guide](../0_developer_guide/step_catalog_integration_guide.md) - Catalog integration
- [XGBoost Pipelines Index](xgboost_pipelines_index.md) - XGBoost-specific pipeline documentation
