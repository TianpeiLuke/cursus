---
tags:
  - entry_point
  - processing_steps
  - step_patterns
  - documentation_index
keywords:
  - processing step
  - data processing
  - step builder patterns
  - validation patterns
topics:
  - processing steps
  - step implementation
  - design patterns
language: python
date of note: 2025-11-09
---

# Processing Steps Index & Documentation Hub

## Purpose

This index serves as the comprehensive navigation hub for all Processing Step-related documentation across the Cursus framework. Processing steps are fundamental components that handle data transformation, preprocessing, feature engineering, model evaluation, and deployment operations within ML pipelines.

## Organization Principle

This index is organized to mirror the **actual pipeline execution order**, grouping processing steps by their functional phase in the ML workflow. Additional sections cover design patterns, validation frameworks, implementation plans, and best practices.

## Table of Contents

### [Part 1: Pipeline Flow - Processing Steps by Execution Phase](#part-1-pipeline-flow---processing-steps-by-execution-phase)
- [Phase 1: Data Acquisition & Loading](#phase-1-data-acquisition--loading)
- [Phase 2: Data Preparation & Feature Engineering](#phase-2-data-preparation--feature-engineering)
- [Phase 3: Label Generation & Data Augmentation](#phase-3-label-generation--data-augmentation)
- [Phase 4: Model Training](#phase-4-model-training)
- [Phase 5: Model Inference & Evaluation](#phase-5-model-inference--evaluation)
- [Phase 6: Model Post-Processing & Calibration](#phase-6-model-post-processing--calibration)
- [Phase 7: Model Deployment & Serving](#phase-7-model-deployment--serving)

### [Part 2: Core Design Documentation](#part-2-core-design-documentation)
- [Primary Design Patterns](#21-primary-design-patterns)
- [Configuration & Architecture](#22-configuration--architecture)

### [Part 3: Validation & Testing](#part-3-validation--testing)
- [Validation Framework](#31-validation-framework)
- [Test Infrastructure](#32-test-infrastructure)

### [Part 4: Implementation Plans & Roadmaps](#part-4-implementation-plans--roadmaps)
- [Core Implementation Plans](#41-core-implementation-plans)
- [Feature Enhancement Plans](#42-feature-enhancement-plans)
- [UI & Configuration Plans](#43-ui--configuration-plans)

### [Part 5: Analysis & Optimization](#part-5-analysis--optimization)
- [System Analysis](#51-system-analysis)
- [Alignment & Standardization](#52-alignment--standardization)
- [Configuration & Path Resolution](#53-configuration--path-resolution)

### [Part 6: Integration & Usage](#part-6-integration--usage)
- [Pipeline Integration](#61-pipeline-integration)
- [Dependency Resolution](#62-dependency-resolution)

### [Part 7: Validation & Quality Assurance](#part-7-validation--quality-assurance)
- [Validation Frameworks](#71-validation-frameworks)
- [Test Infrastructure](#72-test-infrastructure)

### [Part 8: Historical Context & Evolution](#part-8-historical-context--evolution)
- [Implementation History](#81-implementation-history)
- [Feature Evolution](#82-feature-evolution)

### [Part 9: Migration & Modernization](#part-9-migration--modernization)
- [Registry Migration](#91-registry-migration)
- [Step Catalog Integration](#92-step-catalog-integration)

### [Part 10: Best Practices & Guidelines](#part-10-best-practices--guidelines)
- [Development Guidelines](#101-development-guidelines)
- [Testing Best Practices](#102-testing-best-practices)

### Additional Sections
- [Related Documentation](#part-11-related-documentation)
- [Processing Step Statistics](#part-12-processing-step-statistics)
- [Quick Reference](#quick-reference)

---

## Part 1: Pipeline Flow - Processing Steps by Execution Phase

This section organizes all 28 registered processing steps by their position and function in the ML pipeline execution flow.

### Phase 1: Data Acquisition & Loading

#### Registered Processing Steps in This Phase

| Step Name | Description | Has Design Docs |
|-----------|-------------|-----------------|
| DummyDataLoading | Dummy data loading step that processes user-provided data instead of calling Cradle services | ✓ |
| CradleDataLoading | Cradle data loading step | ✓ |

**Key Documentation:**
- Builder: `src/cursus/steps/builders/builder_cradle_data_load_step.py`
- Config: `src/cursus/steps/configs/config_cradle_data_load_step.py`
- [Cradle Data Load Config Helper Design](../1_design/cradle_data_load_config_helper_design.md)
- [Cradle Data Load Config UI Design](../1_design/cradle_data_load_config_ui_design.md)

### Phase 2: Data Preparation & Feature Engineering

#### Registered Processing Steps in This Phase

| Step Name | Description | Has Design Docs |
|-----------|-------------|-----------------|
| TabularPreprocessing | Tabular data preprocessing step with multi-variant support | ✓ |
| TemporalSequenceNormalization | Temporal sequence normalization with configurable operations | ✓ |
| TemporalFeatureEngineering | Temporal feature engineering for ML models | ✓ |
| StratifiedSampling | Stratified sampling with multiple allocation strategies | - |
| RiskTableMapping | Risk table mapping for categorical features | ✓ |
| MissingValueImputation | Missing value imputation using statistical methods | ✓ |
| FeatureSelection | Feature selection with statistical and ML-based methods | ✓ |
| CurrencyConversion | Currency conversion processing | - |

**Key Documentation:**
- [Tabular Preprocessing Step Guide](../steps/tabular_preprocessing_step.md)
- [Temporal Sequence Normalization Design](../1_design/temporal_sequence_normalization_design.md)
- [Temporal Feature Engineering Design](../1_design/temporal_feature_engineering_design.md)
- [Missing Value Imputation Design](../1_design/missing_value_imputation_design.md)
- [Feature Selection Script Design](../1_design/feature_selection_script_design.md)
- [Risk Table Mapping Step Guide](../steps/risk_table_map_step.md)

### Phase 3: Label Generation & Data Augmentation

#### Registered Processing Steps in This Phase

| Step Name | Description | Has Design Docs |
|-----------|-------------|-----------------|
| BedrockPromptTemplateGeneration | Bedrock prompt template generation with 5-component architecture | ✓ |
| BedrockProcessing | Bedrock processing through AWS Bedrock models | ✓ |
| BedrockBatchProcessing | Bedrock batch inference with automatic fallback | ✓ |
| LabelRulesetGeneration | Label ruleset generation and validation | ✓ |
| LabelRulesetExecution | Label ruleset execution with priority-based evaluation | ✓ |
| ActiveSampleSelection | Active sample selection for SSL/Active Learning | ✓ |
| PseudoLabelMerge | Intelligent merge of labeled and pseudo-labeled data | ✓ |

**Key Documentation:**
- [Bedrock Prompt Template Generation Step Patterns](../1_design/bedrock_prompt_template_generation_step_patterns.md)
- [Bedrock Processing Step Builder Patterns](../1_design/bedrock_processing_step_builder_patterns.md)
- [Bedrock Batch Processing Step Builder Patterns](../1_design/bedrock_batch_processing_step_builder_patterns.md)
- [Label Ruleset Generation Script](../scripts/label_ruleset_generation_script.md)
- [Label Ruleset Execution Script](../scripts/label_ruleset_execution_script.md)
- [Active Sampling Step Builder Patterns](../1_design/active_sampling_step_patterns.md)
- [Pseudo Label Merge Script Design](../1_design/pseudo_label_merge_script_design.md)

### Phase 4: Model Training

#### Registered Processing Steps in This Phase

| Step Name | Description | Has Design Docs |
|-----------|-------------|-----------------|
| DummyTraining | SOURCE node that packages pretrained models with hyperparameters | ✓ |

**Key Documentation:**
- **Script Documentation:** [Dummy Training Script](../scripts/dummy_training_script.md)
- Script: `src/cursus/steps/scripts/dummy_training.py`
- **Key Features:** Multi-path file discovery, model.tar.gz repacking, hyperparameters.json injection, temporary directory processing, automatic cleanup, detailed logging with compression ratios

**Note:** DummyTraining is classified as Processing type but operates as a SOURCE node (reads from `/opt/ml/code/` rather than processing inputs).

### Phase 5: Model Inference & Evaluation

#### Registered Processing Steps in This Phase

| Step Name | Description | Has Design Docs |
|-----------|-------------|-----------------|
| XGBoostModelEval | XGBoost model evaluation with job type support | - |
| LightGBMModelEval | LightGBM model evaluation with comparison mode | ✓ |
| XGBoostModelInference | XGBoost model inference for prediction generation | ✓ |
| LightGBMMTModelInference | LightGBMMT multi-task model inference | ✓ |
| PyTorchModelEval | PyTorch model evaluation step | - |
| PyTorchModelInference | PyTorch model inference for prediction generation | ✓ |
| ModelMetricsComputation | Comprehensive model performance evaluation | ✓ |
| ModelWikiGenerator | Automated model documentation creation | ✓ |

**Key Documentation:**
- Builder: `src/cursus/steps/builders/builder_xgboost_model_eval_step.py`
- Config: `src/cursus/steps/configs/config_xgboost_model_eval_step.py`
- **[LightGBM Model Evaluation Script](../scripts/lightgbm_model_eval_script.md)** - Comprehensive model evaluation with comparison mode
- [XGBoost Model Inference Design](../1_design/xgboost_model_inference_design.md)
- [LightGBMMT Model Inference Design](../1_design/lightgbmmt_model_inference_design.md)
- [PyTorch Model Inference Design](../1_design/pytorch_model_inference_design.md)
- [Model Metrics Computation Design](../1_design/model_metrics_computation_design.md)
- [Model Wiki Generator Design](../1_design/model_wiki_generator_design.md)

### Phase 6: Model Post-Processing & Calibration

#### Registered Processing Steps in This Phase

| Step Name | Description | Has Design Docs |
|-----------|-------------|-----------------|
| ModelCalibration | Calibrates model prediction scores to accurate probabilities | - |
| PercentileModelCalibration | Creates percentile mapping using ROC curve analysis | ✓ |

**Key Documentation:**
- Builder: `src/cursus/steps/builders/builder_model_calibration_step.py`
- Config: `src/cursus/steps/configs/config_model_calibration_step.py`
- [Percentile Model Calibration Design](../1_design/percentile_model_calibration_design.md)

### Phase 7: Model Deployment & Serving

#### Registered Processing Steps in This Phase

| Step Name | Description | Has Design Docs |
|-----------|-------------|-----------------|
| Package | Model packaging step | - |
| Registration | Model registration step (MimsModelRegistrationProcessing) | - |
| Payload | Payload testing step | - |

**Key Documentation:**
- Builder: `src/cursus/steps/builders/builder_package_step.py`
- Config: `src/cursus/steps/configs/config_package_step.py`
- Config: `src/cursus/steps/configs/config_payload_step.py`

---

## Part 2: Core Design Documentation

### 2.1 Primary Design Patterns

**[Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)**
- Comprehensive processing step builder implementation patterns
- SageMaker ProcessingStep creation patterns
- Input/output handling for data transformation
- Framework-specific processing implementations (Pandas, Spark, custom)

**[Processing Step Alignment Validation Patterns](../1_design/processing_step_alignment_validation_patterns.md)**
- Validation patterns specific to processing steps
- Specification-contract-builder alignment rules
- Property path validation for processing outputs
- Integration with unified alignment tester

### 2.2 Configuration & Architecture

**[Config Portability Path Resolution](../1_design/config_portability_path_resolution_design.md)**
- Portable path support for processing step scripts
- Universal deployment compatibility (dev, PyPI, Docker, Lambda)
- Hybrid path resolution strategies

**[Processing Step Config Base](../0_developer_guide/three_tier_config_design.md)**
- Three-tier configuration architecture for processing steps
- Essential, System, and Derived field patterns
- Pydantic v2 implementation guide

**[Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md)**
- Format preservation strategy for processing scripts
- Automatic CSV/TSV/Parquet format detection and propagation
- Three-function pattern for consistent file I/O
- Storage optimization (50% reduction in Bedrock scripts)

**Related Configuration Documents:**
- [Config Manager Three-Tier Implementation](../1_design/config_manager_three_tier_implementation.md)
- [Deployment Context Agnostic Path Resolution](../1_design/deployment_context_agnostic_path_resolution_design.md)
- [Hybrid Strategy Deployment Path Resolution](../1_design/hybrid_strategy_deployment_path_resolution_design.md)

---

## Part 3: Validation & Testing

### 3.1 Validation Framework

**[Script Execution Spec](../1_design/script_execution_spec_design.md)**
- Script validation specification for processing steps
- Contract enforcement patterns
- Environment variable handling

**[Processing Step Enhancer](../validation/alignment/step_type_enhancers/processing_enhancer.md)**
- Processing step-specific validation enhancement
- Data transformation pattern validation
- SageMaker processing path validation
- Framework-specific validation rules

### 3.2 Test Infrastructure

**Processing Test Modules:**
- [Processing Test](../validation/builders/variants/processing_test.md) - Main processing step test suite
- [Processing Interface Tests](../validation/builders/variants/processing_interface_tests.md) - Interface compliance testing
- [Processing Specification Tests](../validation/builders/variants/processing_specification_tests.md) - Specification compliance
- [Processing Step Creation Tests](../validation/builders/variants/processing_step_creation_tests.md) - Step creation validation
- [Processing Integration Tests](../validation/builders/variants/processing_integration_tests.md) - Integration testing

**Universal Testing:**
- [Universal Step Builder Test](../1_design/universal_step_builder_test.md)
- [Enhanced Universal Step Builder Tester](../1_design/enhanced_universal_step_builder_tester_design.md)
- [SageMaker Step Type Aware Alignment Tester](../1_design/sagemaker_step_type_aware_unified_alignment_tester_design.md)

---

## Part 4: Implementation Plans & Roadmaps

### 4.1 Core Implementation Plans

**[Config Portability Path Resolution Implementation Plan](../2_project_planning/2025-09-20_config_portability_path_resolution_implementation_plan.md)**
- Portable path implementation for processing step configs
- Migration strategy and testing approach
- Backward compatibility considerations

**[Hybrid Strategy Deployment Path Resolution Implementation Plan](../2_project_planning/2025-09-22_hybrid_strategy_deployment_path_resolution_implementation_plan.md)**
- Hybrid path resolution strategy
- Processing step builder updates
- Modernization of config base classes

**[Universal Step Builder Test Enhancement Plan](../2_project_planning/2025-08-07_universal_step_builder_test_enhancement_plan.md)**
- Comprehensive processing step validation implementation
- Pattern-based testing framework
- Processing-specific test coverage

### 4.2 Feature Enhancement Plans

**[SageMaker Step Type Variants 4-Level Validation Implementation](../2_project_planning/2025-08-15_sagemaker_step_type_variants_4level_validation_implementation.md)**
- Processing step-specific validation requirements
- 4-level validation hierarchy implementation
- Step type-aware enhancement system

**[Model Calibration Job Type Variant Expansion Plan](../2_project_planning/2025-08-21_model_calibration_job_type_variant_expansion_plan.md)**
- Extending ModelCalibration step with job type variants
- Alignment with other processing steps (TabularPreprocessing, RiskTableMapping)
- Dependency resolution improvements

**[Unified Script Path Resolver Implementation Plan](../2_project_planning/2025-10-16_unified_script_path_resolver_implementation_plan.md)**
- Consolidated script path resolution for processing steps
- Proven config-based approach
- Hybrid resolution system integration

**[Bedrock Prompt Template Generation Step Implementation Plan](../2_project_planning/2025-11-02_bedrock_prompt_template_generation_step_implementation_plan.md)**
- Implementation of BedrockPromptTemplateGeneration processing step
- 5-component architecture pattern for structured prompt templates
- Classification task support with validation schemas
- Integration with BedrockProcessing and BedrockBatchProcessing steps

**[LightGBM Training Step Implementation Plan](../2_project_planning/2025-10-14_lightgbm_training_step_implementation_plan.md)**
- Implementation of LightGBM training step following XGBoost patterns
- SageMaker built-in algorithm integration
- Specification-driven architecture support
- Complete alignment with existing training step patterns

### 4.3 UI & Configuration Plans

**[DAG Config Factory Implementation Plan](../2_project_planning/2025-10-15_dag_config_factory_implementation_plan.md)**
- Interactive configuration for processing steps
- Widget-based configuration collection
- Processing step-specific UI components

**[Generalized Config UI Implementation Plan](../2_project_planning/2025-10-07_generalized_config_ui_implementation_plan.md)**
- Generic configuration UI for processing steps
- Three-tier config integration
- Dynamic field discovery

**[Cradle Dynamic Data Sources Hybrid Implementation Plan](../2_project_planning/2025-10-09_cradle_dynamic_data_sources_hybrid_implementation_plan.md)**
- TabularPreprocessing step UI enhancements
- Method delegation patterns
- Navigation flow improvements

---

## Part 5: Analysis & Optimization

### 5.1 System Analysis

**[Step Builder Methods Comprehensive Analysis](../4_analysis/step_builder_methods_comprehensive_analysis.md)**
- Analysis of processing step builder methods
- Pattern identification and optimization opportunities
- Method usage across different processing step types

**[Validation System Efficiency and Purpose Analysis](../4_analysis/validation_system_efficiency_and_purpose_analysis.md)**
- Validation system performance for processing steps
- Purpose and effectiveness analysis
- Optimization recommendations

**[Step Builder Local Override Patterns Analysis](../4_analysis/step_builder_local_override_patterns_analysis.md)**
- Processing step override patterns
- Pure pipeline integration patterns
- Mid-pipeline processing step design

**[Unified Testers Comparative Analysis](../4_analysis/unified_testers_comparative_analysis.md)**
- Comparison of validation approaches for processing steps
- Processing step alignment validation patterns
- Test framework effectiveness

### 5.2 Alignment & Standardization

**[Code Alignment Standardization Plan](../2_project_planning/2025-08-11_code_alignment_standardization_plan.md)**
- Processing step alignment improvements
- Tabular preprocessing standardization
- Validation success rate improvements (25% → 87.5%)

**[Alignment Validation Implementation Plan](../2_project_planning/2025-07-05_alignment_validation_implementation_plan.md)**
- Processing step alignment validation
- Script-specification alignment
- Multi-level validation framework

**[Alignment Validation Refactoring Plan](../2_project_planning/2025-08-10_alignment_validation_refactoring_plan.md)**
- Processing step validation refactoring
- Builder-aware detection fixes
- Naming variation logic improvements

### 5.3 Configuration & Path Resolution

**[Runtime Script Discovery Redundancy Analysis](../4_analysis/2025-10-16_runtime_script_discovery_redundancy_analysis.md)**
- Processing step config base proven approach
- Script path resolution analysis
- Hybrid resolution system evaluation

**[ImportLib Usage Systemic Deployment Portability Analysis](../4_analysis/2025-09-19_importlib_usage_systemic_deployment_portability_analysis.md)**
- TabularPreprocessing step portability analysis
- Deployment compatibility evaluation
- ImportLib usage patterns

**[Config Field Management System Analysis](../4_analysis/config_field_management_system_analysis.md)**
- Processing step config base class handling
- Field categorization for processing configs
- Type-aware serialization

---

## Part 6: Complete Processing Steps Registry

This section lists ALL processing steps registered in `src/cursus/registry/step_names_original.py` with `sagemaker_step_type="Processing"`.

### 6.1 Data Loading Steps

**DummyDataLoading**
- Description: Drop-in replacement for CradleDataLoadingStep with auto-format detection and dual-mode output
- **Script Documentation:** [Dummy Data Loading Script](../scripts/dummy_data_loading_script.md)
- Config: `DummyDataLoadingConfig`
- Builder: `DummyDataLoadingStepBuilder`
- Script: `src/cursus/steps/scripts/dummy_data_loading.py`
- **Key Features:** Multi-format support (CSV/Parquet/JSON), auto-detection, schema signature generation, metadata with statistics, dual-mode output (legacy/enhanced sharding), format preservation

**CradleDataLoading** (Note: Uses custom sagemaker_step_type)
- Description: Cradle data loading step
- Config: `CradleDataLoadingConfig`
- Builder: `CradleDataLoadingStepBuilder`
- Builder: `src/cursus/steps/builders/builder_cradle_data_load_step.py`
- Config: `src/cursus/steps/configs/config_cradle_data_load_step.py`

### 6.2 Data Preprocessing Steps

**TabularPreprocessing**
- Description: Tabular data preprocessing step
- **Documentation:** [Tabular Preprocessing Step Guide](../steps/tabular_preprocessing_step.md)
- Builder: `src/cursus/steps/builders/builder_tabular_preprocessing_step.py`
- Config: `src/cursus/steps/configs/config_tabular_preprocessing_step.py`
- Script: `src/cursus/steps/scripts/tabular_preprocessing.py`
- **Key Features:** Multi-variant support (training, testing, validation, calibration), job type-based processing, portable path support, Framework: Pandas/Scikit-learn

**TemporalSequenceNormalization**
- Description: Temporal sequence normalization step for machine learning models with configurable sequence operations
- **Script Documentation:** [Temporal Sequence Normalization Script](../scripts/temporal_sequence_normalization_script.md)
- **Design Doc:** [Temporal Sequence Normalization Design](../1_design/temporal_sequence_normalization_design.md)
- Config: `TemporalSequenceNormalizationConfig`
- Builder: `TemporalSequenceNormalizationStepBuilder`

**TemporalFeatureEngineering**
- Description: Temporal feature engineering step that extracts comprehensive temporal features from normalized sequences for machine learning models
- **Script Documentation:** [Temporal Feature Engineering Script](../scripts/temporal_feature_engineering_script.md)
- **Design Doc:** [Temporal Feature Engineering Design](../1_design/temporal_feature_engineering_design.md)
- Config: `TemporalFeatureEngineeringConfig`
- Builder: `TemporalFeatureEngineeringStepBuilder`

**StratifiedSampling**
- Description: Stratified sampling step with multiple allocation strategies for class imbalance, causal analysis, and variance optimization
- **Script Documentation:** [Stratified Sampling Script](../scripts/stratified_sampling_script.md)
- Builder: `src/cursus/steps/builders/builder_stratified_sampling_step.py`
- Config: `src/cursus/steps/configs/config_stratified_sampling_step.py`
- Script: `src/cursus/steps/scripts/stratified_sampling.py`
- **Key Features:** Three allocation strategies (balanced, proportional with minimum, optimal Neyman), class imbalance handling, causal analysis support, variance optimization, format preservation, job-type awareness, test set protection

**RiskTableMapping**
- Description: Risk table mapping step for categorical features
- **Documentation:** [Risk Table Mapping Step Guide](../steps/risk_table_map_step.md)
- **Script Documentation:** [Risk Table Mapping Script](../scripts/risk_table_mapping_script.md)
- Builder: `src/cursus/steps/builders/builder_risk_table_mapping_step.py`
- Config: `src/cursus/steps/configs/config_risk_table_mapping_step.py`
- Script: `src/cursus/steps/scripts/risk_table_mapping.py`
- **Key Features:** Risk score mapping, weight-of-evidence encoding, Laplace smoothing, parameter accumulator pattern, job type variants, format preservation, custom transformation logic

**MissingValueImputation**
- Description: Missing value imputation step using statistical methods (mean, median, mode, constant) with pandas-safe values
- **Script Documentation:** [Missing Value Imputation Script](../scripts/missing_value_imputation_script.md)
- **Design Doc:** [Missing Value Imputation Design](../1_design/missing_value_imputation_design.md)
- Config: `MissingValueImputationConfig`
- Builder: `MissingValueImputationStepBuilder`
- Script: `src/cursus/steps/scripts/missing_value_imputation.py`
- **Key Features:** Multi-type imputation (numerical/categorical/text), pandas-safe validation, auto-detection of column types, training/inference modes, parameter accumulator pattern, format preservation, comprehensive reporting

**FeatureSelection**
- Description: Feature selection step using multiple statistical and ML-based methods with ensemble combination strategies
- **Design Doc:** [Feature Selection Script Design](../1_design/feature_selection_script_design.md)
- Config: `FeatureSelectionConfig`
- Builder: `FeatureSelectionStepBuilder`

**CurrencyConversion**
- Description: Currency conversion processing step with dual lookup methods and parallel processing
- **Script Documentation:** [Currency Conversion Script](../scripts/currency_conversion_script.md)
- Builder: `src/cursus/steps/builders/builder_currency_conversion_step.py`
- Config: `src/cursus/steps/configs/config_currency_conversion_step.py`
- Script: `src/cursus/steps/scripts/currency_conversion.py`
- **Key Features:** Dual currency lookup (direct codes + marketplace ID), parallel processing with multiprocessing, format preservation (CSV/TSV/Parquet), exchange rate conversion, job type support

### 6.3 Bedrock Processing Steps

**BedrockPromptTemplateGeneration**
- Description: Bedrock prompt template generation step that creates structured prompt templates for classification tasks using the 5-component architecture pattern
- **Script Documentation:** [Bedrock Prompt Template Generation Script](../scripts/bedrock_prompt_template_generation_script.md)
- **Design Docs:**
  - [Bedrock Prompt Template Generation Step Patterns](../1_design/bedrock_prompt_template_generation_step_patterns.md)
  - [Bedrock Prompt Template Generation Input Formats](../1_design/bedrock_prompt_template_generation_input_formats.md)
  - [Bedrock Prompt Template Generation Output Design](../1_design/bedrock_prompt_template_generation_output_design.md)
  - [Bedrock Prompt Template Generation Buyer Seller Example](../1_design/bedrock_prompt_template_generation_buyer_seller_example.md)
- Config: `BedrockPromptTemplateGenerationConfig`
- Builder: `BedrockPromptTemplateGenerationStepBuilder`

**BedrockProcessing**
- Description: Bedrock processing step that processes input data through AWS Bedrock models using generated prompt templates and validation schemas
- **Script Documentation:** [Bedrock Processing Script](../scripts/bedrock_processing_script.md)
- **Design Doc:** [Bedrock Processing Step Builder Patterns](../1_design/bedrock_processing_step_builder_patterns.md)
- Config: `BedrockProcessingConfig`
- Builder: `BedrockProcessingStepBuilder`

**BedrockBatchProcessing**
- Description: Bedrock batch processing step that provides AWS Bedrock batch inference capabilities with automatic fallback to real-time processing for cost-efficient large dataset processing
- **Script Documentation:** [Bedrock Batch Processing Script](../scripts/bedrock_batch_processing_script.md)
- **Design Doc:** [Bedrock Batch Processing Step Builder Patterns](../1_design/bedrock_batch_processing_step_builder_patterns.md)
- Config: `BedrockBatchProcessingConfig`
- Builder: `BedrockBatchProcessingStepBuilder`

### 6.4 Model Evaluation & Inference Steps

**XGBoostModelEval**
- Description: XGBoost model evaluation step
- Builder: `src/cursus/steps/builders/builder_xgboost_model_eval_step.py`
- Config: `src/cursus/steps/configs/config_xgboost_model_eval_step.py`
- **Key Features:** Model evaluation processing, job type support, integration with training pipeline

**XGBoostModelInference**
- Description: XGBoost model inference step for prediction generation without metrics
- **Script Documentation:** [XGBoost Model Inference Script](../scripts/xgboost_model_inference_script.md)
- **Design Doc:** [XGBoost Model Inference Design](../1_design/xgboost_model_inference_design.md)
- Config: `XGBoostModelInferenceConfig`
- Builder: `XGBoostModelInferenceStepBuilder`

**LightGBMModelInference**
- Description: LightGBM model inference for prediction generation with multi-format output
- **Script Documentation:** [LightGBM Model Inference Script](../scripts/lightgbm_model_inference_script.md)
- Config: `LightGBMModelInferenceConfig`
- Builder: `LightGBMModelInferenceStepBuilder`
- Script: `src/cursus/steps/scripts/lightgbm_model_inference.py`
- **Key Features:** Pure inference without metrics, multi-format output (CSV/TSV/Parquet/JSON), modular design for caching, lightweight (~600 lines), automatic tar.gz extraction, format preservation, graceful feature handling

**LightGBMMT Model Inference**
- Description: LightGBMMT multi-task model inference step for generating N independent binary task predictions
- **Design Doc:** [LightGBMMT Model Inference Design](../1_design/lightgbmmt_model_inference_design.md)
- Config: `LightGBMMTModelInferenceConfig`
- Builder: `LightGBMMTModelInferenceStepBuilder`
- **Key Features:** Multi-task prediction, per-task calibration support, reuses XGBoost multiclass output format for compatibility

**PyTorchModelEval**
- Description: PyTorch model evaluation step
- Config: `PyTorchModelEvalConfig`
- Builder: `PyTorchModelEvalStepBuilder`

**PyTorchModelInference**
- Description: PyTorch model inference step for prediction generation without metrics
- **Design Doc:** [PyTorch Model Inference Design](../1_design/pytorch_model_inference_design.md)
- Config: `PyTorchModelInferenceConfig`
- Builder: `PyTorchModelInferenceStepBuilder`

**ModelMetricsComputation**
- Description: Model metrics computation step for comprehensive performance evaluation
- **Design Doc:** [Model Metrics Computation Design](../1_design/model_metrics_computation_design.md)
- Config: `ModelMetricsComputationConfig`
- Builder: `ModelMetricsComputationStepBuilder`

**ModelWikiGenerator**
- Description: Model wiki generator step for automated documentation creation
- **Design Doc:** [Model Wiki Generator Design](../1_design/model_wiki_generator_design.md)
- Config: `ModelWikiGeneratorConfig`
- Builder: `ModelWikiGeneratorStepBuilder`

### 6.5 Model Calibration Steps

**ModelCalibration**
- Description: Calibrates model prediction scores to accurate probabilities
- Builder: `src/cursus/steps/builders/builder_model_calibration_step.py`
- Config: `src/cursus/steps/configs/config_model_calibration_step.py`
- **Recent Enhancements:** Job type variant support added, alignment with other processing steps, improved dependency resolution

**PercentileModelCalibration**
- Description: Creates percentile mapping from model scores using ROC curve analysis for consistent risk interpretation
- **Design Doc:** [Percentile Model Calibration Design](../1_design/percentile_model_calibration_design.md)
- Config: `PercentileModelCalibrationConfig`
- Builder: `PercentileModelCalibrationStepBuilder`

### 6.6 Deployment & Testing Steps

**Package**
- Description: Model packaging step
- Builder: `src/cursus/steps/builders/builder_package_step.py`
- Config: `src/cursus/steps/configs/config_package_step.py`

**Registration**
- Description: Model registration step
- Config: `RegistrationConfig`
- Builder: `RegistrationStepBuilder`
- Note: Uses custom `sagemaker_step_type="MimsModelRegistrationProcessing"`

**Payload**
- Description: Payload testing step
- Config: `src/cursus/steps/configs/config_payload_step.py`
- Builder: `PayloadStepBuilder`

### 6.7 Special Processing Steps

**DummyTraining**
- Description: SOURCE node that packages pretrained models by injecting hyperparameters into model archives
- **Script Documentation:** [Dummy Training Script](../scripts/dummy_training_script.md)
- Config: `DummyTrainingConfig`
- Builder: `DummyTrainingStepBuilder`
- Script: `src/cursus/steps/scripts/dummy_training.py`
- **Key Features:** Multi-path file discovery (3 fallback locations), model.tar.gz repacking, hyperparameters.json injection, temporary directory processing with automatic cleanup, detailed logging with compression ratios, format validation
- Note: Classified as Processing type but operates as SOURCE node (reads from `/opt/ml/code/` instead of processing inputs)

### 6.8 Semi-Supervised Learning & Active Learning Steps

**ActiveSampleSelection**
- Description: Modular sample selection engine for semi-supervised and active learning pipelines that intelligently selects high-value samples from unlabeled data pools
- **Script Documentation:** [Active Sample Selection Script](../scripts/active_sample_selection_script.md)
- **Design Docs:**
  - [Active Sampling Step Builder Patterns](../1_design/active_sampling_step_patterns.md) - Complete step builder patterns with contract, spec, config, and builder implementations
  - [Active Sampling Script Design](../1_design/active_sampling_script_design.md) - Main script design with uncertainty, diversity, and BADGE strategies
  - [Active Sampling BADGE Design](../1_design/active_sampling_badge.md) - BADGE (Batch Active learning by Diverse Gradient Embeddings) algorithm for hybrid sampling
  - [Active Sampling Uncertainty Design](../1_design/active_sampling_uncertainty_margin_entropy.md) - Uncertainty-based strategies (margin, entropy, least confidence)
  - [Active Sampling Core-Set Design](../1_design/active_sampling_core_set_leaf_core_set.md) - Diversity sampling with core-set and k-center algorithms
- Config: `ActiveSamplingConfig`
- Builder: `ActiveSamplingStepBuilder`
- **Key Features:** Three sampling strategies (uncertainty/diversity/BADGE), model-agnostic design, works with XGBoost/LightGBM/PyTorch/Bedrock inference outputs, embedding support for enhanced diversity, configurable batch selection, Pydantic validation for strategy-use case compatibility
- **Integration:** Used in semi-supervised learning pipelines between model inference and pseudo-label merging steps, automatic dependency resolution via specification-driven architecture

**PseudoLabelMerge**
- Description: Unified data combination engine that intelligently merges original labeled training data with pseudo-labeled or augmented samples for Semi-Supervised Learning (SSL) and Active Learning workflows
- **Script Documentation:** [Pseudo Label Merge Script](../scripts/pseudo_label_merge_script.md) - Complete script documentation with algorithms, data structures, and integration patterns
- **Design Doc:** [Pseudo Label Merge Script Design](../1_design/pseudo_label_merge_script_design.md) - Complete script design with split-aware merge, auto-inferred ratios, format preservation, and schema alignment
- Config: `PseudoLabelMergeConfig`
- Builder: `PseudoLabelMergeStepBuilder`
- **Key Features:** Split-aware merge for training jobs (maintains train/test/val boundaries), auto-inferred split ratios (adapts to base data proportions), simple merge for validation/testing jobs, data format preservation (CSV/TSV/Parquet), schema alignment and provenance tracking, symmetric input handling with semantic differentiation
- **Integration:** Used in semi-supervised learning pipelines after active sample selection to combine base labeled data with pseudo-labeled samples, supports multiple upstream sources (ActiveSampleSelection, ModelInference, LabelRulesetExecution, BedrockBatchProcessing), feeds into training steps (XGBoostTraining, LightGBMTraining, PyTorchTraining)
- **Pipeline Flow:** TabularPreprocessing → (XGBoostInference → ActiveSampleSelection) → PseudoLabelMerge → XGBoostTraining (fine-tune)

---

## Part 7: Integration & Usage

### 7.1 Pipeline Integration

**[Specification-Driven Step Builder Plan](../2_project_planning/2025-07-07_specification_driven_step_builder_plan.md)**
- Processing steps in specification-driven architecture
- Dependency resolution for processing steps
- Pipeline assembly patterns

**[Specification-Driven XGBoost Pipeline Plan](../2_project_planning/2025-07-04_specification_driven_xgboost_pipeline_plan.md)**
- TabularPreprocessing integration
- Data preparation workflows
- End-to-end pipeline examples

**[XGBoost End-to-End Pipeline Example](../examples/mods_pipeline_xgboost_end_to_end.md)**
- TabularPreprocessing step usage
- Training and calibration variants
- Complete pipeline implementation

### 7.2 Dependency Resolution

**[Comprehensive Dependency Matching Analysis](../2_project_planning/2025-07-08_comprehensive_dependency_matching_analysis.md)**
- Processing step output matching
- Dependency resolution for preprocessing steps
- Alias support and exact matching

**[Dependency Resolver Benefits](../2_project_planning/2025-07-07_dependency_resolver_benefits.md)**
- Processing step output matching patterns
- Automatic dependency resolution
- Semantic matching capabilities

---

## Part 8: Validation & Quality Assurance

### 8.1 Validation Frameworks

**[Validation Alignment Builders Analysis](../4_analysis/validation_alignment_builders_analysis.md)**
- Processing test suite structure
- Processing-specific validation modules
- Integration test patterns

**[Validation System Common Foundational Layer Analysis](../4_analysis/validation_system_common_foundational_layer_analysis.md)**
- Processing step builder patterns validation
- Common validation layer for processing steps
- Alignment validation integration

**[Validation Alignment Module Code Redundancy Analysis](../4_analysis/validation_alignment_module_code_redundancy_analysis.md)**
- Processing step alignment validation
- Redundancy elimination strategies
- Test code optimization

### 8.2 Test Infrastructure

**[Universal Step Builder Simplified Approach Analysis](../4_analysis/universal_step_builder_simplified_approach_analysis.md)**
- Processing step creation tests
- Configuration validity testing
- Simplified testing approach

**[Level3 Step Creation Tests Refactoring Analysis](../4_analysis/level3_step_creation_tests_refactoring_analysis.md)**
- Processing step creation test refactoring
- Test methodology improvements
- Failure analysis and resolution

**[Level3 Path Mapping Test Responsibility Analysis](../4_analysis/level3_path_mapping_test_responsibility_analysis.md)**
- Processing step path mapping validation
- Test responsibility allocation
- Coverage improvement strategies

---

## Part 9: Historical Context & Evolution

### 9.1 Implementation History

**[Phase 1 Solution Summary](../2_project_planning/phase1_solution_summary.md)**
- Initial processing step modernization
- Alignment validation implementation
- Foundation establishment

**[Phase 5 Training Step Modernization Summary](../2_project_planning/2025-07-07_phase5_training_step_modernization_summary.md)**
- Processing step patterns influence on training steps
- Pattern consistency across step types
- Shared architecture evolution

**[Corrected Alignment Architecture Plan](../2_project_planning/2025-07-05_corrected_alignment_architecture_plan.md)**
- Processing step builders alignment updates
- Specification-driven approach adoption
- Architecture corrections

### 9.2 Feature Evolution

**[Contract Alignment Implementation Summary](../2_project_planning/2025-07-04_contract_alignment_implementation_summary.md)**
- Processing step contract validation
- Alignment validation extension
- Component coverage expansion

**[Job Type Variant Solution](../2_project_planning/2025-07-04_job_type_variant_solution.md)**
- Job type variant support for processing steps
- Multi-variant processing implementation
- Pattern establishment

---

## Part 10: Migration & Modernization

### 10.1 Registry Migration

**[Registry Migration Implementation Analysis](../4_analysis/registry_migration_implementation_analysis.md)**
- TabularPreprocessing step registry updates
- Builder registry integration
- Import standardization

**[Workspace-Aware Hybrid Registry Migration Plan](../2_project_planning/2025-09-02_workspace_aware_hybrid_registry_migration_plan.md)**
- Processing step hybrid registry integration
- Custom processing step examples
- Migration strategy

**[Hybrid Registry Migration Plan Analysis](../4_analysis/2025-09-02_hybrid_registry_migration_plan_analysis.md)**
- Processing step builder patterns and registry
- Migration approach analysis
- Implementation recommendations

### 10.2 Step Catalog Integration

**[Step Catalog System Integration Analysis](../4_analysis/step_catalog_system_integration_analysis.md)**
- Processing step specializations
- Auto-discovery integration
- Catalog system compatibility

**[Unified Step Catalog System Implementation Plan](../2_project_planning/2025-09-10_unified_step_catalog_system_implementation_plan.md)**
- ProcessingStepConfigBase modernization
- Pydantic V2 migration
- Catalog integration improvements

**[Universal Step Builder Test Step Catalog Integration Plan](../2_project_planning/2025-09-28_universal_step_builder_test_step_catalog_integration_plan.md)**
- Processing test modules enhancement
- Step catalog integration
- Code redundancy elimination

---

## Part 11: Best Practices & Guidelines

### 11.1 Development Guidelines

**Key Principles:**
1. **Three-Tier Configuration**: Use Essential, System, and Derived field pattern
2. **Portable Paths**: Implement portable path support for universal deployment
3. **Job Type Variants**: Support multiple job types (training, testing, validation, calibration)
4. **Contract Compliance**: Ensure script contracts align with specifications
5. **Framework Independence**: Design for framework-agnostic processing where possible

**Common Patterns:**
```python
# Processing step builder pattern
class CustomProcessingStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session, role, 
                 registry_manager, dependency_resolver):
        self.config = config
        self.spec = CUSTOM_PROCESSING_SPEC
        
    def create_step(self, inputs=None, outputs=None, dependencies=None):
        # Use portable script path with fallback
        script_path = self.config.get_portable_script_path() or \
                     self.config.get_script_path()
        
        return ProcessingStep(
            name=self._get_step_name(),
            processor=self._create_processor(),
            code=script_path,
            inputs=self._process_inputs(inputs),
            outputs=self._process_outputs(outputs)
        )
```

### 11.2 Testing Best Practices

**Validation Levels:**
1. **Level 1**: Script contract alignment
2. **Level 2**: Contract specification alignment
3. **Level 3**: Specification dependency alignment
4. **Level 4**: Builder configuration alignment

**Test Coverage:**
- Unit tests for builder methods
- Integration tests with upstream/downstream steps
- Contract compliance validation
- Property path validation
- Configuration validity checks

---

## Part 12: Related Documentation

### Core Framework

- [Cursus Package Overview](./cursus_package_overview.md) - System architecture overview
- [Cursus Code Structure Index](./cursus_code_structure_index.md) - Complete code-to-doc mapping
- [Registered Steps Pipeline Reference](0_registered_steps_pipeline_reference.md) - Complete reference of all 37 registered steps with component architecture details
- [Step Design and Documentation Index](./step_design_and_documentation_index.md) - All step types index

### Step Builder Patterns

- [Step Builder Patterns Summary](../1_design/step_builder_patterns_summary.md) - Comprehensive pattern analysis
- [Training Step Builder Patterns](../1_design/training_step_builder_patterns.md) - Training step patterns
- [Transform Step Builder Patterns](../1_design/transform_step_builder_patterns.md) - Transform step patterns
- [CreateModel Step Builder Patterns](../1_design/createmodel_step_builder_patterns.md) - CreateModel patterns

### Validation Patterns

- [Training Step Alignment Validation Patterns](../1_design/training_step_alignment_validation_patterns.md)
- [Transform Step Alignment Validation Patterns](../1_design/transform_step_alignment_validation_patterns.md)
- [CreateModel Step Alignment Validation Patterns](../1_design/createmodel_step_alignment_validation_patterns.md)

### Developer Guides

- [Adding New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md) - Step creation workflow
- [Step Builder Guide](../0_developer_guide/step_builder.md) - Builder implementation
- [Script Development Guide](../0_developer_guide/script_development_guide.md) - Script implementation
- [Validation Framework Guide](../0_developer_guide/validation_framework_guide.md) - Validation system

### Documentation Standards

- **[Script Documentation Guide](../6_resources/script_documentation_guide.md)**: Comprehensive guide for documenting processing scripts with standardized 15-section format
- **[YAML Frontmatter Standard](../6_resources/documentation_yaml_frontmatter_standard.md)**: Standard format for YAML frontmatter in documentation files
- **[Script Documentation Standardization Project Plan](../2_project_planning/2025-11-18_script_documentation_standardization_project_plan.md)**: Complete project plan for documenting all 31 processing scripts in the Cursus framework

### Script Documentation Examples

- **[Active Sample Selection Script](../scripts/active_sample_selection_script.md)**: Complete example of script documentation following standardized format
- **[Bedrock Batch Processing Script](../scripts/bedrock_batch_processing_script.md)**: Complete example of complex script documentation with proper markdown links
- **[Bedrock Processing Script](../scripts/bedrock_processing_script.md)**: Complete example demonstrating all documentation sections and best practices

---

## Part 13: Processing Step Statistics

### Line Count Analysis (as of 2025-09-29)

**Builder Files:**
- `builder_tabular_preprocessing_step.py` - 304 lines
- `builder_risk_table_mapping_step.py` - 356 lines
- `builder_stratified_sampling_step.py` - 318 lines
- `builder_model_calibration_step.py` - ~250 lines
- `builder_xgboost_model_eval_step.py` - 271 lines

**Config Files:**
- `config_processing_step_base.py` - 337 lines (base class)
- `config_tabular_preprocessing_step.py` - 156 lines
- `config_payload_step.py` - 225 lines
- `config_stratified_sampling_step.py` - 194 lines
- `config_xgboost_model_eval_step.py` - 136 lines

**Validation Files:**
- `processing_specification_tests.py` - 323 lines
- `processing_test.py` - 311 lines
- `processing_step_creation_tests.py` - 52 lines

### Test Coverage

**Processing Step Builder Tests:**
- `test_processing_step_builders.py` - 454 lines
- Comprehensive validation across all processing step types
- Dynamic discovery and testing
- 87.5% alignment success rate

---

## Maintenance Notes

**Last Updated:** 2025-11-09

**Update Triggers:**
- New processing step implementation
- Pattern updates or improvements
- Validation framework changes
- Test infrastructure modifications

**Maintenance Guidelines:**
- Keep implementation examples current
- Update validation patterns as they evolve
- Document new processing step types
- Maintain link accuracy
- Track alignment success rates

---

## Quick Reference

### Common Processing Step Tasks

**Creating a New Processing Step:**
1. Define specification → [Step Specification](../1_design/step_specification.md)
2. Implement config with three-tier pattern → [Three-Tier Config](../0_developer_guide/three_tier_config_design.md)
3. Create builder following patterns → [Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)
4. Write processing script → [Script Development Guide](../0_developer_guide/script_development_guide.md)
5. Add contract → [Script Contract](../1_design/script_contract.md)
6. Validate alignment → [Validation Framework](../0_developer_guide/validation_framework_guide.md)

**Debugging Processing Steps:**
1. Check alignment → [Processing Step Alignment Validation](../1_design/processing_step_alignment_validation_patterns.md)
2. Verify paths → [Config Portability Path Resolution](../1_design/config_portability_path_resolution_design.md)
3. Test validation → [Processing Test Modules](../validation/builders/variants/)
4. Review patterns → [Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)

**Extending Existing Processing Steps:**
1. Understand current implementation → [Code Structure Index](./cursus_code_structure_index.md)
2. Review patterns → [Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)
3. Check job type support → [Job Type Variant Solution](../2_project_planning/2025-07-04_job_type_variant_solution.md)
4. Follow best practices → This document Section 10
