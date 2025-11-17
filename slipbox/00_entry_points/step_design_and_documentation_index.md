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
date of note: 2025-11-09
---

# Step Design and Documentation Index

## Purpose

This entry point document provides a comprehensive index of all step-related design documentation and implementation references. It serves as a navigation hub for understanding pipeline step architecture, design patterns, contracts, and implementations.

## Design Documents (slipbox/1_design)

### Bedrock Step Patterns

- [Bedrock Batch Processing Step Builder Patterns](../1_design/bedrock_batch_processing_step_builder_patterns.md) - Design patterns for building batch processing steps with Bedrock
- [Bedrock Processing Step Builder Patterns](../1_design/bedrock_processing_step_builder_patterns.md) - General processing step patterns for Bedrock integration
- [Bedrock Prompt Template Generation Step Patterns](../1_design/bedrock_prompt_template_generation_step_patterns.md) - Patterns for generating prompt templates in pipeline steps
- [Bedrock Prompt Template Generation Buyer Seller Example](../1_design/bedrock_prompt_template_generation_buyer_seller_example.md) - Concrete example of buyer-seller prompt generation
- [Bedrock Prompt Template Generation Input Formats](../1_design/bedrock_prompt_template_generation_input_formats.md) - Input format specifications for prompt generation
- [Bedrock Prompt Template Generation Output Design](../1_design/bedrock_prompt_template_generation_output_design.md) - Output design for prompt generation steps

### Label Ruleset Step Patterns

- [Label Ruleset Execution Step Patterns](../1_design/ruleset_execution_step_patterns.md) - Design patterns for executing label rulesets in pipeline steps
- [Label Ruleset Generation Step Patterns](../1_design/label_ruleset_generation_step_patterns.md) - Patterns for generating and validating label rulesets within pipeline steps
- [Label Ruleset Multilabel Extension Design](../1_design/label_ruleset_multilabel_extension_design.md) - Design for extending label ruleset system to support multi-label output for multi-task learning with category-conditional rules
- [Label Ruleset Multilabel Extension Implementation Plan](../2_project_planning/2025-11-11_label_ruleset_multilabel_extension_implementation_plan.md) - Detailed 5-phase implementation plan for multilabel support with unified design approach, configuration models, validators, execution engine, and comprehensive examples
- [Label Ruleset Optimization Patterns](../1_design/label_ruleset_optimization_patterns.md) - Optimization strategies for label ruleset performance (selectivity, field grouping, complexity ordering)
- [Label Ruleset Generation Configuration Examples](../1_design/label_ruleset_generation_configuration_examples.md) - Complete working examples for binary, multiclass, and multilabel classification with DAGConfigFactory integration, JSON configurations, and best practices

### MTGBM (Multi-Task Gradient Boosting Machine) Refactoring

#### Design Documents

- [LightGBM Multi-Task Training Step Design](../1_design/lightgbm_multi_task_training_step_design.md) - Design for LightGBM multi-task/multi-label training with adaptive weighting and knowledge distillation
- [MTGBM Loss Functions Refactoring Design](../1_design/mtgbm_models_refactoring_design.md) - Comprehensive refactoring design for MTGBM loss function implementations with abstract base classes, strategy pattern, and factory pattern (complements model classes refactoring)
- [MTGBM Model Classes Refactoring Design](../1_design/mtgbm_model_classes_refactoring_design.md) - Architectural design for refactoring MTGBM model class implementations with template method pattern, base model abstractions, and state management (complements loss function refactoring)

#### Implementation Plans

- [LightGBMMT Implementation Part 1: Script Contract & Hyperparameters](../2_project_planning/2025-11-12_lightgbmmt_implementation_part1_script_contract_hyperparams.md) - Phase 1 implementation plan covering script contract definition, hyperparameter classes, loss function refactoring with base class architecture, and factory patterns (67% code reduction achieved)
- [LightGBMMT Implementation Part 2: Training Script Alignment](../2_project_planning/2025-11-12_lightgbmmt_implementation_part2_training_script_alignment.md) - Phase 2 implementation plan for training script development aligned with XGBoost patterns, model architecture integration, and multi-task evaluation framework

#### Analysis Documents

- [LightGBMMT Multi-Task Implementation Analysis](../4_analysis/2025-11-10_lightgbmmt_multi_task_implementation_analysis.md) - Analysis of LightGBMMT framework implementation, custom loss functions, and multi-task learning patterns
- [MTGBM Models Optimization Analysis](../4_analysis/2025-11-11_mtgbm_models_optimization_analysis.md) - Detailed analysis identifying optimization opportunities, code duplication issues, and performance bottlenecks in MTGBM implementations
- [MTGBM Pipeline Reusability Analysis](../4_analysis/2025-11-11_mtgbm_pipeline_reusability_analysis.md) - Analysis of MTGBM pipeline architecture reusability patterns and extensibility opportunities
- [Python Refactored vs LightGBMMT Fork Architecture Comparison](../4_analysis/2025-11-12_python_refactored_vs_lightgbmmt_fork_comparison.md) - Comprehensive comparison analysis demonstrating Python refactored approach achieves 4-30% better performance, 67% code reduction, eliminates fork dependency, and provides $635K cost savings over 5 years compared to C++ lightgbmmt fork

### Core Step Builder Patterns

- [Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md) - General patterns for building processing steps
- [Training Step Builder Patterns](../1_design/training_step_builder_patterns.md) - Patterns for building training steps
- [Transform Step Builder Patterns](../1_design/transform_step_builder_patterns.md) - Patterns for building transform steps
- [Tuning Step Builder Patterns](../1_design/tuning_step_builder_patterns.md) - Patterns for building hyperparameter tuning steps
- [Conditional Step Builder Patterns](../1_design/conditional_step_builder_patterns.md) - Patterns for building conditional execution steps
- [CreateModel Step Builder Patterns](../1_design/createmodel_step_builder_patterns.md) - Patterns for building model creation steps

### Step Alignment and Validation Patterns

- [Processing Step Alignment Validation Patterns](../1_design/processing_step_alignment_validation_patterns.md) - Validation patterns for processing steps
- [Training Step Alignment Validation Patterns](../1_design/training_step_alignment_validation_patterns.md) - Validation patterns for training steps
- [Transform Step Alignment Validation Patterns](../1_design/transform_step_alignment_validation_patterns.md) - Validation patterns for transform steps
- [CreateModel Step Alignment Validation Patterns](../1_design/createmodel_step_alignment_validation_patterns.md) - Validation patterns for model creation steps
- [RegisterModel Step Alignment Validation Patterns](../1_design/registermodel_step_alignment_validation_patterns.md) - Validation patterns for model registration steps
- [Utility Step Alignment Validation Patterns](../1_design/utility_step_alignment_validation_patterns.md) - Validation patterns for utility steps

### Step Architecture and Design

- [Step Builder](../1_design/step_builder.md) - Core step builder architecture and design
- [Step Builder Patterns Summary](../1_design/step_builder_patterns_summary.md) - Comprehensive summary of all step builder patterns
- [Step Specification](../1_design/step_specification.md) - Step specification format and requirements
- [Step Contract](../1_design/step_contract.md) - Contract definitions for pipeline steps
- [Step Config Resolver](../1_design/step_config_resolver.md) - Configuration resolution for steps
- [Step Builder Validation Rulesets Design](../1_design/step_builder_validation_rulesets_design.md) - Validation ruleset architecture
- [Config Tiered Design](../1_design/config_tiered_design.md) - Three-tier configuration architecture
- [Config Manager Three Tier Implementation](../1_design/config_manager_three_tier_implementation.md) - Implementation of three-tier config system
- [Three Tier Config Design](../0_developer_guide/three_tier_config_design.md) - Developer guide for three-tier configuration

### Step Improvements and Extensions

- [Training Step Improvements](../1_design/training_step_improvements.md) - Enhancements to training step functionality
- [Packaging Step Improvements](../1_design/packaging_step_improvements.md) - Enhancements to packaging step functionality
- [Step Type Enhancement System Design](../1_design/step_type_enhancement_system_design.md) - System design for step type enhancements

### Step-Specific Script Design

- [Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md) - Design for automatic format preservation across pipeline scripts (CSV/TSV/Parquet)
- [Multi Sequence Preprocessing Design](../1_design/multi_sequence_preprocessing_design.md) - Design for multi-sequence preprocessing scripts
- [Multi Sequence Feature Engineering Design](../1_design/multi_sequence_feature_engineering_design.md) - Design for multi-sequence feature engineering scripts
- [Model Metrics Computation Design](../1_design/model_metrics_computation_design.md) - Design for model metrics computation scripts
- [Model Wiki Generator Design](../1_design/model_wiki_generator_design.md) - Design for model wiki generation scripts
- [Percentile Model Calibration Design](../1_design/percentile_model_calibration_design.md) - Design for percentile model calibration scripts
- [Feature Selection Script Design](../1_design/feature_selection_script_design.md) - Design for feature selection scripts
- [Missing Value Imputation Design](../1_design/missing_value_imputation_design.md) - Design for missing value imputation scripts
- [Temporal Feature Engineering Design](../1_design/temporal_feature_engineering_design.md) - Design for temporal feature engineering scripts
- [Temporal Sequence Normalization Design](../1_design/temporal_sequence_normalization_design.md) - Design for temporal sequence normalization scripts
- [Temporal Self Attention Model Design](../1_design/temporal_self_attention_model_design.md) - Design for temporal self-attention model architecture and training scripts
- [Temporal Self Attention Scripts Analysis](../4_analysis/temporal_self_attention_scripts_analysis.md) - Detailed analysis of TSA scripts and preprocessing requirements (related to multi-sequence processing, temporal feature engineering, and temporal sequence normalization)
- [Inference Handler Spec Design](../1_design/inference_handler_spec_design.md) - Design for inference handler specifications
- [Inference Test Result Design](../1_design/inference_test_result_design.md) - Design for inference testing result models
- [XGBoost Model Inference Design](../1_design/xgboost_model_inference_design.md) - Design for XGBoost model inference scripts
- [XGBoost Semi-Supervised Learning Pipeline Design](../1_design/xgboost_semi_supervised_learning_pipeline_design.md) - Complete semi-supervised learning pipeline architecture with pseudo-labeling workflow for XGBoost
- [XGBoost Semi-Supervised Learning Training Design](../1_design/xgboost_semi_supervised_learning_training_design.md) - Training step extension with job_type support for SSL pretraining and fine-tuning phases

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

## Step Implementation Documentation (slipbox/steps)

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

## Related Documentation

### Developer Guides

- [Adding New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md) - Guide for adding new pipeline steps
- [Step Builder Guide](../0_developer_guide/step_builder.md) - Developer guide for step builders
- [Step Specification Guide](../0_developer_guide/step_specification.md) - Guide for step specifications
- [Step Catalog Integration Guide](../0_developer_guide/step_catalog_integration_guide.md) - Integrating steps with the catalog
- [Script Contract Guide](../0_developer_guide/script_contract.md) - Understanding script contracts
- [Script Development Guide](../0_developer_guide/script_development_guide.md) - Developing step scripts

### Core Design Documents

- [Script Contract](../1_design/script_contract.md) - Core script contract design
- [Script Execution Spec Design](../1_design/script_execution_spec_design.md) - Script execution specification
- [Script Testability Refactoring](../1_design/script_testability_refactoring.md) - Testability improvements
- [Script Integration Testing System Design](../1_design/script_integration_testing_system_design.md) - Integration testing

## Navigation Tips

### For New Developers

1. Start with [Step Builder Guide](../0_developer_guide/step_builder.md)
2. Review [Adding New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md)
3. Understand [Step Specification](../1_design/step_specification.md)
4. Explore specific step builder patterns relevant to your use case

### For Step Pattern Selection

1. Identify the type of step you need (processing, training, transform, etc.)
2. Review the corresponding builder pattern document
3. Check alignment validation patterns for your step type
4. Examine existing implementations in slipbox/steps/builders

### For Validation and Testing

1. Review [Universal Step Builder Test](../1_design/universal_step_builder_test.md)
2. Check step-specific alignment validation patterns
3. Explore [Unified Alignment Tester Master Design](../1_design/unified_alignment_tester_master_design.md)

### For Specific Step Types

**Bedrock Steps**: Start with bedrock_processing_step_builder_patterns.md

**Label Ruleset Steps**: Review label_ruleset_execution_step_patterns.md and label_ruleset_generation_step_patterns.md

**Training Steps**: Check training_step_builder_patterns.md and training_step_alignment_validation_patterns.md

**Processing Steps**: Begin with processing_step_builder_patterns.md

## Document Organization

This index organizes step-related documentation into:

1. **Design Documents** - Architectural patterns and design decisions
2. **Implementation Documentation** - Concrete implementations (builders, contracts, scripts, specs)
3. **Developer Guides** - How-to guides and best practices
4. **Testing & Validation** - Testing frameworks and validation patterns

## Contributing

When adding new step-related documentation:

1. Follow the [Documentation YAML Frontmatter Standard](../6_resources/documentation_yaml_frontmatter_standard.md)
2. Place design documents in slipbox/1_design
3. Place implementation docs in appropriate slipbox/steps subdirectories
4. Update this index with links to new documentation
5. Use appropriate tags: `design`, `code`, `test`, etc.

## See Also

- [Cursus Package Overview](cursus_package_overview.md) - Main package entry point
- [Pipeline Catalog Design](../1_design/pipeline_catalog_design.md) - Pipeline catalog architecture
- [Step Catalog Integration Guide](../0_developer_guide/step_catalog_integration_guide.md) - Catalog integration
