---
tags:
  - project
  - planning
  - alignment
  - standardization
  - naming_conventions
keywords:
  - code alignment
  - naming conventions
  - file standardization
  - component alignment
  - validation framework
  - step builder consistency
  - specification alignment
  - contract standardization
topics:
  - code standardization
  - naming convention alignment
  - component consistency
  - validation framework
language: python
date of note: 2025-08-11
---

# Code Alignment Standardization Plan
**Date**: August 11, 2025  
**Status**: Planning Phase  
**Priority**: High  
**Scope**: System-wide code alignment and naming convention standardization

## 🎯 Executive Summary

This document outlines a comprehensive plan to align existing code with established alignment rules and naming conventions across the Cursus pipeline system. Based on our analysis of the current codebase, we have identified significant inconsistencies in file naming patterns and missing components that need to be addressed to achieve full system alignment.

## 📚 Historical Context: Previous Alignment Efforts

### Major Alignment Initiatives (July-August 2025)

Our current standardization effort builds upon significant previous work to establish and validate alignment across the Cursus pipeline system. This section documents the major initiatives that led to our current understanding of alignment requirements.

#### 🔧 **Phase 1: Contract-Specification Alignment (July 2025)**

**Initiative**: Script Contract Alignment Implementation  
**Period**: July 4-5, 2025  
**Scope**: Establishing alignment between processing scripts and their contracts

**Key Achievements**:
- **Contract Validation Framework**: Implemented comprehensive contract validation system
- **Property Path Alignment**: Established consistent property path mapping between contracts and specifications
- **Script Analysis**: Developed automated script analysis to detect contract violations
- **Alignment Rules**: Defined core alignment principles for script-contract consistency

**Components Addressed**:
- XGBoost model evaluation scripts
- Tabular preprocessing pipelines
- Currency conversion processing
- Risk table mapping operations

**Technical Deliverables**:
- `src/cursus/validation/alignment/script_contract_alignment.py`
- `src/cursus/validation/alignment/contract_spec_alignment.py`
- Comprehensive validation test suite
- Contract violation detection and reporting

#### 🏗️ **Phase 2: Specification-Dependency Alignment (July 2025)**

**Initiative**: Dependency Resolution Architecture  
**Period**: July 7-8, 2025  
**Scope**: Aligning step specifications with dependency resolution system

**Key Achievements**:
- **Dependency Resolver Integration**: Connected validation system with production dependency resolver
- **Canonical Name Mapping**: Established consistent naming between specifications and registry
- **Semantic Matching**: Implemented intelligent dependency matching with confidence scoring
- **Registry Integration**: Unified specification registry with step builder registry

**Components Addressed**:
- All processing step specifications
- Step builder registry integration
- Dependency resolution patterns
- Cross-step dependency validation

**Technical Deliverables**:
- `src/cursus/validation/alignment/spec_dependency_alignment.py`
- `src/cursus/core/deps/dependency_resolver.py` integration
- Registry-based canonical name mapping
- Confidence-scored dependency matching

#### 🔄 **Phase 3: Builder-Configuration Alignment (August 2025)**

**Initiative**: Configuration File Discovery and Validation  
**Period**: August 9-11, 2025  
**Scope**: Ensuring step builders have corresponding configuration files

**Key Achievements**:
- **FlexibleFileResolver**: Implemented hybrid file resolution system
- **Configuration Discovery**: Automated discovery of configuration files with fallback patterns
- **Environment Resolution**: Resolved Python environment issues blocking validation
- **Hybrid Resolution**: Multi-strategy file discovery (standard → flexible → fuzzy)

**Components Addressed**:
- All step builder configuration files
- File naming convention edge cases
- Configuration class validation
- Builder-config integration patterns

**Technical Deliverables**:
- `src/cursus/validation/alignment/builder_config_alignment.py`
- FlexibleFileResolver with predefined mappings
- Hybrid file resolution architecture
- Configuration validation framework

#### 📊 **Phase 4: Comprehensive Validation System (August 2025)**

**Initiative**: Unified Alignment Tester Implementation  
**Period**: August 10-11, 2025  
**Scope**: Integrated validation across all alignment levels

**Key Achievements**:
- **Four-Level Validation**: Complete validation pipeline from scripts to configurations
- **Environment Fixes**: Resolved Python environment mismatches blocking validation
- **Comprehensive Reporting**: Detailed JSON reporting with actionable recommendations
- **Production Integration**: Validation system uses same logic as production pipeline

**Validation Levels Implemented**:
1. **Level 1**: Script ↔ Contract alignment (87.5% success rate)
2. **Level 2**: Contract ↔ Specification alignment (100% success rate)
3. **Level 3**: Specification ↔ Dependencies alignment (37.5% success rate)
4. **Level 4**: Builder ↔ Configuration alignment (87.5% success rate)

**Technical Deliverables**:
- `src/cursus/validation/alignment/unified_alignment_tester.py`
- Comprehensive validation reports in `test/steps/scripts/alignment_validation/reports/`
- Level 3 and Level 4 consolidated analysis reports
- Production-grade validation infrastructure

### 🎯 **Current State Based on Previous Work**

Our previous alignment efforts have established:

#### ✅ **Solid Foundation**
- **Validation Framework**: Production-ready validation system across all levels
- **Registry Integration**: Unified registry system with canonical name mapping
- **Dependency Resolution**: Advanced dependency matching with confidence scoring
- **File Discovery**: Robust file resolution with multiple fallback strategies

#### 🔧 **Identified Patterns**
- **Naming Inconsistencies**: Systematic patterns in file naming variations
- **Missing Components**: Specific gaps in contract files for model steps
- **Resolution Strategies**: Effective approaches for handling naming edge cases
- **Validation Accuracy**: High success rates with clear paths to 100%

#### 📈 **Success Metrics from Previous Work**
- **Level 1 Validation**: 87.5% success rate (7/8 scripts)
- **Level 2 Validation**: 100% success rate (perfect alignment)
- **Level 3 Validation**: 37.5% success rate (3/8 scripts, significant improvement from 0%)
- **Level 4 Validation**: 87.5% success rate (7/8 scripts, major breakthrough)

## 📊 Current State Analysis

### File Naming Convention Assessment

Building on our previous alignment work, our analysis of the four core component types (builders, configs, specs, contracts) for key step types reveals the following patterns:

#### ✅ **Consistent Components**
- **Builders**: All follow `builder_{step_name}_step_{framework}.py` pattern
- **Configs**: All follow `config_{step_name}_step_{framework}.py` pattern

#### ❌ **Inconsistent Components**
- **Specs**: Mix of `{framework}_{step_name}_spec.py` and `{step_name}_spec.py` patterns
- **Contracts**: Mix of patterns with **critical missing files**

### Lessons Learned from Previous Alignment Work

#### 🔍 **Key Insights**
1. **Environment Issues Critical**: Python environment mismatches can cause systemic validation failures
2. **Registry Integration Essential**: Canonical name mapping must be consistent across all systems
3. **Hybrid Approaches Work**: Multiple resolution strategies provide robustness
4. **Production Alignment**: Validation must use same logic as production systems

#### 🛠️ **Proven Strategies**
1. **FlexibleFileResolver**: Effective for handling naming convention edge cases
2. **Confidence Scoring**: Helps identify weak dependency matches
3. **Multi-Level Validation**: Comprehensive validation catches issues at all levels
4. **Detailed Reporting**: JSON reports with actionable recommendations enable effective debugging

### 🔧 **Processing Step Fixes and Improvements**

Our alignment standardization effort also builds upon extensive work to fix and improve various processing steps across the system. This section documents the major processing step improvements that inform our current standardization approach.

#### **Tabular Preprocessing Step Improvements**

**Period**: July-August 2025  
**Scope**: Comprehensive overhaul of tabular preprocessing pipeline

**Key Improvements**:
- **Contract Alignment**: Fixed property path mismatches between preprocessing contracts and specifications
- **Configuration Standardization**: Aligned configuration classes with builder patterns
- **Dependency Resolution**: Improved dependency matching for preprocessing steps
- **Validation Integration**: Enhanced validation coverage for preprocessing workflows

**Components Fixed**:
- `src/cursus/steps/builders/builder_tabular_preprocessing_step.py`
- `src/cursus/steps/configs/config_tabular_preprocessing_step.py`
- `src/cursus/steps/specs/preprocessing_*.py` (multiple variants)
- `src/cursus/steps/contracts/tabular_preprocess_contract.py`

#### **XGBoost Model Evaluation Step Enhancements**

**Period**: July-August 2025  
**Scope**: Complete alignment and functionality improvements for XGBoost evaluation

**Key Improvements**:
- **Script-Contract Alignment**: Resolved property path inconsistencies in evaluation scripts
- **Model Artifact Handling**: Improved model artifact path resolution and validation
- **Configuration Integration**: Enhanced configuration class integration with step builders
- **Evaluation Metrics**: Standardized evaluation metric collection and reporting

**Components Enhanced**:
- `src/cursus/steps/builders/builder_xgboost_model_eval_step.py`
- `src/cursus/steps/configs/config_xgboost_model_eval_step.py`
- `src/cursus/steps/specs/xgboost_model_eval_spec.py`
- `src/cursus/steps/contracts/xgboost_model_eval_contract.py`

#### **Currency Conversion Processing Fixes**

**Period**: August 2025  
**Scope**: Alignment and functionality improvements for currency conversion steps

**Key Improvements**:
- **Multi-Variant Support**: Enhanced support for training/testing/validation/calibration variants
- **Contract Standardization**: Aligned contracts across all currency conversion variants
- **Dependency Resolution**: Improved dependency matching for currency conversion workflows
- **Configuration Consistency**: Standardized configuration patterns across variants

**Components Addressed**:
- `src/cursus/steps/builders/builder_currency_conversion_step.py`
- `src/cursus/steps/configs/config_currency_conversion_step.py`
- `src/cursus/steps/specs/currency_conversion_*_spec.py` (4 variants)
- `src/cursus/steps/contracts/currency_conversion_contract.py`

#### **Risk Table Mapping Step Improvements**

**Period**: August 2025  
**Scope**: Comprehensive alignment improvements for risk table mapping

**Key Improvements**:
- **Multi-Environment Support**: Enhanced support for different execution environments
- **Path Resolution**: Improved path resolution for risk table artifacts
- **Validation Coverage**: Enhanced validation coverage for risk table operations
- **Configuration Alignment**: Aligned configuration classes with step builder patterns

**Components Improved**:
- `src/cursus/steps/builders/builder_risk_table_mapping_step.py`
- `src/cursus/steps/configs/config_risk_table_mapping_step.py`
- `src/cursus/steps/specs/risk_table_mapping_*_spec.py` (4 variants)
- `src/cursus/steps/contracts/risk_table_mapping_contract.py`

#### **Model Calibration Step Enhancements**

**Period**: August 2025  
**Scope**: Alignment and functionality improvements for model calibration

**Key Improvements**:
- **Calibration Logic**: Enhanced calibration algorithm integration
- **Artifact Management**: Improved calibration artifact handling and validation
- **Configuration Integration**: Better integration with configuration management system
- **Dependency Resolution**: Enhanced dependency resolution for calibration workflows

**Components Enhanced**:
- `src/cursus/steps/builders/builder_model_calibration_step.py`
- `src/cursus/steps/configs/config_model_calibration_step.py`
- `src/cursus/steps/specs/model_calibration_spec.py`
- `src/cursus/steps/contracts/model_calibration_contract.py`

#### **Training Step Modernization**

**Period**: July-August 2025  
**Scope**: Comprehensive modernization of PyTorch and XGBoost training steps

**Key Improvements**:
- **Framework Alignment**: Consistent patterns across PyTorch and XGBoost training
- **Hyperparameter Integration**: Enhanced hyperparameter management and validation
- **Training Artifact Handling**: Improved training artifact management and path resolution
- **Configuration Standardization**: Aligned configuration patterns across training frameworks

**Components Modernized**:
- `src/cursus/steps/builders/builder_training_step_pytorch.py`
- `src/cursus/steps/builders/builder_training_step_xgboost.py`
- `src/cursus/steps/configs/config_training_step_pytorch.py`
- `src/cursus/steps/configs/config_training_step_xgboost.py`
- `src/cursus/steps/specs/pytorch_training_spec.py`
- `src/cursus/steps/specs/xgboost_training_spec.py`
- `src/cursus/steps/contracts/pytorch_train_contract.py`
- `src/cursus/steps/contracts/xgboost_train_contract.py`

### 📊 **Processing Step Improvement Metrics**

#### **Before Improvements**
- **Alignment Success Rate**: ~25% across all processing steps
- **Configuration Discovery**: ~50% success rate
- **Dependency Resolution**: ~15% success rate
- **Contract Validation**: ~40% success rate

#### **After Improvements**
- **Alignment Success Rate**: 87.5% across validated processing steps
- **Configuration Discovery**: 87.5% success rate (with FlexibleFileResolver)
- **Dependency Resolution**: 37.5% success rate (significant improvement)
- **Contract Validation**: 100% success rate for Level 2 validation

#### **Key Success Factors**
1. **Systematic Approach**: Addressed each processing step comprehensively
2. **Pattern Recognition**: Identified and applied consistent patterns across steps
3. **Validation Integration**: Used validation framework to guide improvements
4. **Incremental Progress**: Built improvements incrementally with validation at each step

## 📚 Cross-References to Related Documentation

### Analysis Reports (`slipbox/4_analysis/`)
- **`unified_alignment_tester_pain_points_analysis.md`**: Comprehensive analysis of alignment validation challenges and solutions
- **`alignment_tester_robustness_analysis.md`**: Analysis of validation framework robustness and reliability
- **`step_builder_methods_comprehensive_analysis.md`**: Detailed analysis of step builder patterns and methods
- **`step_builder_methods_top_pain_points_analysis.md`**: Identification of key pain points in step builder implementation
- **`two_level_validation_pain_point_solution_analysis.md`**: Analysis of multi-level validation approach and solutions

### Test Reports (`slipbox/test/`)
- **`level1_alignment_validation_consolidated_report_2025_08_11.md`**: Level 1 (Script ↔ Contract) validation results and analysis
- **`level2_alignment_validation_consolidated_report_2025_08_11.md`**: Level 2 (Contract ↔ Specification) validation results
- **`level3_alignment_validation_consolidated_report_2025_08_11.md`**: Level 3 (Specification ↔ Dependencies) validation analysis
- **`level4_alignment_validation_consolidated_report_2025_08_11.md`**: Level 4 (Builder ↔ Configuration) validation results
- **`universal_builder_test_analysis_report.md`**: Comprehensive analysis of universal builder test framework
- **`universal_builder_test_enhancement_report.md`**: Enhancement recommendations for builder testing
- **`core_package_comprehensive_test_analysis.md`**: Analysis of core package test coverage and quality

### Design Documents (`slipbox/1_design/`)
- **`unified_alignment_tester_design.md`**: Design specification for the unified alignment validation system
- **`enhanced_dependency_validation_design.md`**: Enhanced dependency validation architecture design
- **`two_level_alignment_validation_system_design.md`**: Multi-level alignment validation system design

### Detailed Component Analysis

This section provides a comprehensive analysis of all processing steps in the system, including their current alignment status based on our validation results.

#### 1. **Cradle Data Loading**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_data_load_step_cradle.py` | ❌ Inconsistent | `builder_cradle_data_loading_step.py` | Level 4: ✅ Pass |
| Config | `config_data_load_step_cradle.py` | ❌ Inconsistent | `config_cradle_data_loading_step.py` | Level 4: ✅ Pass |
| Spec | `data_loading_spec.py` | ❌ Inconsistent | `cradle_data_loading_spec.py` | Level 3: ❌ Fail |
| Contract | `cradle_data_loading_contract.py` | ✅ Correct | No change | Level 1: ✅ Pass |

**Canonical Name**: `CradleDataLoading` (from STEP_NAMES registry)
**Alignment Issues**: Builder and config files don't follow `builder_cradle_data_loading_step.py` pattern; spec file should match canonical name
**Priority**: Medium - functional but naming inconsistent across most components

#### 2. **XGBoost Model Step**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_model_step_xgboost.py` | ❌ Inconsistent | `builder_xgboost_model_step.py` | Level 4: ❌ Fail |
| Config | `config_model_step_xgboost.py` | ❌ Inconsistent | `config_xgboost_model_step.py` | Level 4: ❌ Fail |
| Spec | `xgboost_model_spec.py` | ✅ Correct | No change | Level 3: ❌ Fail |
| Contract | **N/A** | ✅ Not Required | CreateModel steps don't need script contracts | Level 1: ✅ N/A |

**Canonical Name**: `XGBoostModel` (from STEP_NAMES registry)
**SageMaker Step Type**: `CreateModel` - No script execution, no contract needed
**Alignment Issues**: Builder and config files don't follow `builder_xgboost_model_step.py` pattern
**Priority**: Medium - naming inconsistencies but no missing critical components

#### 3. **PyTorch Model Step**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_model_step_pytorch.py` | ❌ Inconsistent | `builder_pytorch_model_step.py` | Level 4: ❌ Fail |
| Config | `config_model_step_pytorch.py` | ❌ Inconsistent | `config_pytorch_model_step.py` | Level 4: ❌ Fail |
| Spec | `pytorch_model_spec.py` | ✅ Correct | No change | Level 3: ❌ Fail |
| Contract | **N/A** | ✅ Not Required | CreateModel steps don't need script contracts | Level 1: ✅ N/A |

**Canonical Name**: `PyTorchModel` (from STEP_NAMES registry)
**SageMaker Step Type**: `CreateModel` - No script execution, no contract needed
**Alignment Issues**: Builder and config files don't follow `builder_pytorch_model_step.py` pattern
**Priority**: Medium - naming inconsistencies but no missing critical components

#### 4. **XGBoost Training Step**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_training_step_xgboost.py` | ❌ Inconsistent | `builder_xgboost_training_step.py` | Level 4: ✅ Pass |
| Config | `config_training_step_xgboost.py` | ❌ Inconsistent | `config_xgboost_training_step.py` | Level 4: ✅ Pass |
| Spec | `xgboost_training_spec.py` | ✅ Correct | No change | Level 3: ✅ Pass |
| Contract | `xgboost_train_contract.py` | ❌ Inconsistent | `xgboost_training_contract.py` | Level 1: ✅ Pass |

**Canonical Name**: `XGBoostTraining` (from STEP_NAMES registry)
**Alignment Issues**: Builder and config files don't follow `builder_xgboost_training_step.py` pattern; contract file should match canonical name
**Priority**: Medium - functional with good validation results but naming inconsistent

#### 5. **PyTorch Training Step**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_training_step_pytorch.py` | ❌ Inconsistent | `builder_pytorch_training_step.py` | Level 4: ✅ Pass |
| Config | `config_training_step_pytorch.py` | ❌ Inconsistent | `config_pytorch_training_step.py` | Level 4: ✅ Pass |
| Spec | `pytorch_training_spec.py` | ✅ Correct | No change | Level 3: ❌ Fail |
| Contract | `pytorch_train_contract.py` | ❌ Inconsistent | `pytorch_training_contract.py` | Level 1: ✅ Pass |

**Canonical Name**: `PyTorchTraining` (from STEP_NAMES registry)
**Alignment Issues**: Builder and config files don't follow `builder_pytorch_training_step.py` pattern; contract file should match canonical name
**Priority**: Medium - mostly functional but naming inconsistent and spec dependency issues

#### 6. **Tabular Preprocessing Step**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_tabular_preprocessing_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Config | `config_tabular_preprocessing_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Spec | `preprocessing_spec.py` | ✅ Correct | No change | Level 3: ✅ Pass |
| Contract | `tabular_preprocess_contract.py` | ✅ Correct | No change | Level 1: ✅ Pass |

**Alignment Status**: ✅ **FULLY ALIGNED** - All validation levels pass
**Priority**: None - exemplary alignment

#### 7. **XGBoost Model Evaluation Step**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_xgboost_model_eval_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Config | `config_xgboost_model_eval_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Spec | `xgboost_model_eval_spec.py` | ✅ Correct | No change | Level 3: ✅ Pass |
| Contract | `xgboost_model_eval_contract.py` | ✅ Correct | No change | Level 1: ✅ Pass |

**Alignment Status**: ✅ **FULLY ALIGNED** - All validation levels pass
**Priority**: None - exemplary alignment

#### 8. **Currency Conversion Step**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_currency_conversion_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Config | `config_currency_conversion_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Spec | `currency_conversion_*_spec.py` (4 variants) | ✅ Correct | No change | Level 3: ❌ Fail |
| Contract | `currency_conversion_contract.py` | ✅ Correct | No change | Level 1: ✅ Pass |

**Alignment Issues**: Dependency resolution issues with multi-variant specifications
**Priority**: Medium - functional but dependency matching challenges

#### 9. **Risk Table Mapping Step**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_risk_table_mapping_step.py` | ✅ Correct | No change | Level 4: ❌ Fail |
| Config | `config_risk_table_mapping_step.py` | ✅ Correct | No change | Level 4: ❌ Fail |
| Spec | `risk_table_mapping_*_spec.py` (4 variants) | ✅ Correct | No change | Level 3: ❌ Fail |
| Contract | `risk_table_mapping_contract.py` | ✅ Correct | No change | Level 1: ❌ Fail |

**Alignment Issues**: Configuration discovery and dependency resolution challenges
**Priority**: High - multiple validation failures

#### 10. **Model Calibration Step**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_model_calibration_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Config | `config_model_calibration_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Spec | `model_calibration_spec.py` | ✅ Correct | No change | Level 3: ❌ Fail |
| Contract | `model_calibration_contract.py` | ✅ Correct | No change | Level 1: ✅ Pass |

**Alignment Issues**: Dependency resolution challenges
**Priority**: Medium - functional but dependency matching issues

#### 11. **Package Step (MIMS)**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_package_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Config | `config_package_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Spec | `packaging_spec.py` | ✅ Correct | No change | Level 3: ❌ Fail |
| Contract | `mims_package_contract.py` | ✅ Correct | No change | Level 1: ✅ Pass |

**Alignment Issues**: Dependency resolution challenges
**Priority**: Medium - functional but dependency matching issues

#### 12. **Payload Step (MIMS)**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_payload_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Config | `config_payload_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Spec | `payload_spec.py` | ✅ Correct | No change | Level 3: ❌ Fail |
| Contract | `mims_payload_contract.py` | ✅ Correct | No change | Level 1: ✅ Pass |

**Alignment Issues**: Dependency resolution challenges
**Priority**: Medium - functional but dependency matching issues

#### 13. **Registration Step (MIMS)**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_registration_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Config | `config_registration_step.py` | ✅ Correct | No change | Level 4: ✅ Pass |
| Spec | `registration_spec.py` | ✅ Correct | No change | Level 3: ❌ Fail |
| Contract | `mims_registration_contract.py` | ✅ Correct | No change | Level 1: ✅ Pass |

**Alignment Issues**: Dependency resolution challenges
**Priority**: Medium - functional but dependency matching issues

#### 14. **Batch Transform Step**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_batch_transform_step.py` | ✅ Correct | No change | Level 4: ❌ Fail |
| Config | `config_batch_transform_step.py` | ✅ Correct | No change | Level 4: ❌ Fail |
| Spec | `batch_transform_*_spec.py` (4 variants) | ✅ Correct | No change | Level 3: ❌ Fail |
| Contract | **MISSING** | 🚨 Critical | `batch_transform_contract.py` | Level 1: ❌ Fail |

**Alignment Issues**: Missing contract file and configuration discovery issues
**Priority**: High - missing critical component

#### 15. **Dummy Training Step**
| Component | Current Name | Status | Target Name | Validation Status |
|-----------|--------------|--------|-------------|-------------------|
| Builder | `builder_dummy_training_step.py` | ✅ Correct | No change | Level 4: ❌ Fail |
| Config | `config_dummy_training_step.py` | ✅ Correct | No change | Level 4: ❌ Fail |
| Spec | `dummy_training_spec.py` | ✅ Correct | No change | Level 3: ❌ Fail |
| Contract | `dummy_training_contract.py` | ✅ Correct | No change | Level 1: ❌ Fail |

**Alignment Issues**: Configuration discovery and dependency resolution challenges
**Priority**: Medium - test/development component

### 📊 **Alignment Status Summary**

#### **By Validation Level**
- **Level 1 (Script ↔ Contract)**: 7/8 passing (87.5%)
- **Level 2 (Contract ↔ Specification)**: 8/8 passing (100%)
- **Level 3 (Specification ↔ Dependencies)**: 3/8 passing (37.5%)
- **Level 4 (Builder ↔ Configuration)**: 7/8 passing (87.5%)

#### **By Component Type**
- **Builders**: 15/15 correctly named (100%)
- **Configs**: 15/15 correctly named (100%)
- **Specs**: 10/15 correctly named (67%)
- **Contracts**: 12/15 present and correctly named (80%)

#### **Critical Issues**
1. **Missing Contracts**: 3 critical missing contract files
2. **Naming Inconsistencies**: 5 specification files with inconsistent naming
3. **Dependency Resolution**: 12/15 steps failing Level 3 validation
4. **Configuration Discovery**: 8/15 steps failing Level 4 validation

#### **Fully Aligned Steps** ✅
- Tabular Preprocessing Step
- XGBoost Model Evaluation Step

#### **High Priority Fixes** 🚨
- XGBoost Model Step (missing contract)
- PyTorch Model Step (missing contract)
- Batch Transform Step (missing contract)
- Risk Table Mapping Step (multiple validation failures)

## 🎯 Standardization Goals

### Primary Objectives
1. **Achieve 100% naming consistency** across all component types
2. **Create missing contract files** for model steps
3. **Establish clear naming convention rules** for future development
4. **Update all internal references** to renamed files
5. **Validate alignment** through comprehensive testing

### Success Metrics
- **File Naming Consistency**: 100% adherence to established patterns
- **Component Completeness**: All step types have all four component files
- **Validation Success**: All alignment validation tests pass
- **Reference Integrity**: No broken imports or references

## 📋 Implementation Plan

### Phase 1: File Renaming and Creation (Week 1)

#### 1.1 Specification File Standardization
**Objective**: Rename specification files to follow consistent pattern based on canonical names

**Actions**:
```bash
# Rename specification files to match canonical names (snake_case)
mv src/cursus/steps/specs/data_loading_spec.py src/cursus/steps/specs/cradle_data_loading_spec.py
# Note: xgboost_model_spec.py and pytorch_model_spec.py are already correctly named
# Note: xgboost_training_spec.py and pytorch_training_spec.py are already correctly named
```

**Deliverables**:
- [ ] 1 specification file renamed (cradle data loading)
- [ ] All internal imports updated
- [ ] Registry references updated

#### 1.2 Builder and Config File Standardization
**Objective**: Rename builder and config files to follow consistent `{framework}_{operation}_step` pattern

**Actions**:
```bash
# Rename cradle data loading step files
mv src/cursus/steps/builders/builder_data_load_step_cradle.py src/cursus/steps/builders/builder_cradle_data_loading_step.py
mv src/cursus/steps/configs/config_data_load_step_cradle.py src/cursus/steps/configs/config_cradle_data_loading_step.py

# Rename model step builder files
mv src/cursus/steps/builders/builder_model_step_xgboost.py src/cursus/steps/builders/builder_xgboost_model_step.py
mv src/cursus/steps/builders/builder_model_step_pytorch.py src/cursus/steps/builders/builder_pytorch_model_step.py

# Rename model step config files
mv src/cursus/steps/configs/config_model_step_xgboost.py src/cursus/steps/configs/config_xgboost_model_step.py
mv src/cursus/steps/configs/config_model_step_pytorch.py src/cursus/steps/configs/config_pytorch_model_step.py

# Rename training step builder files
mv src/cursus/steps/builders/builder_training_step_xgboost.py src/cursus/steps/builders/builder_xgboost_training_step.py
mv src/cursus/steps/builders/builder_training_step_pytorch.py src/cursus/steps/builders/builder_pytorch_training_step.py

# Rename training step config files
mv src/cursus/steps/configs/config_training_step_xgboost.py src/cursus/steps/configs/config_xgboost_training_step.py
mv src/cursus/steps/configs/config_training_step_pytorch.py src/cursus/steps/configs/config_pytorch_training_step.py
```

**Deliverables**:
- [ ] 6 builder files renamed (cradle data loading + 2 model steps + 2 training steps)
- [ ] 6 config files renamed (cradle data loading + 2 model steps + 2 training steps)
- [ ] All internal imports updated
- [ ] Registry references updated

#### 1.3 Contract File Standardization
**Objective**: Rename existing contract files for steps that require script contracts

**Actions**:
```bash
# Rename existing contract files to match canonical names (snake_case)
mv src/cursus/steps/contracts/xgboost_train_contract.py src/cursus/steps/contracts/xgboost_training_contract.py
mv src/cursus/steps/contracts/pytorch_train_contract.py src/cursus/steps/contracts/pytorch_training_contract.py
# Note: cradle_data_loading_contract.py is already correctly named

# Note: XGBoost and PyTorch Model steps (CreateModel type) don't need script contracts
# as they don't execute scripts - they only create SageMaker model objects
```

**Deliverables**:
- [ ] 2 contract files renamed (training steps)
- [ ] All contract files follow consistent naming
- [ ] No unnecessary contract files created for CreateModel steps

#### 1.4 Import Reference Updates
**Objective**: Update all import statements to use new file names

**Scope**:
- Builder files importing specifications
- Registry files referencing contracts
- Test files importing components
- Validation framework references

**Deliverables**:
- [ ] All import statements updated
- [ ] No broken references remain
- [ ] All tests pass after updates

### Phase 2: Content Standardization (Week 2)

#### 2.1 SageMaker Step Type Validation
**Objective**: Ensure step type classifications are correct and no unnecessary contracts are created

**Actions**:
- Verify CreateModel steps (XGBoost Model, PyTorch Model) don't have script contracts
- Confirm Training steps (XGBoost Training, PyTorch Training) have proper script contracts
- Validate CradleDataLoading step has script contract (custom ScriptProcessingStep)
- Update validation framework to handle step type distinctions

**Deliverables**:
- [ ] Step type classifications verified
- [ ] No unnecessary contract files created for CreateModel steps
- [ ] Validation framework updated for step type awareness

#### 2.2 Specification Content Alignment
**Objective**: Ensure all renamed specification files maintain correct content

**Actions**:
- Verify class names match new file names
- Update any internal references
- Ensure specification definitions are complete
- Validate against step builder requirements

**Deliverables**:
- [ ] All specification classes properly named
- [ ] Specification content validated
- [ ] Integration tests pass

### Phase 3: Registry and Integration Updates (Week 3)

#### 3.1 Registry Integration
**Objective**: Update all registry references to use new file names

**Components to Update**:
- `src/cursus/steps/registry/builder_registry.py`
- `src/cursus/steps/specs/__init__.py`
- `src/cursus/steps/contracts/__init__.py`
- `src/cursus/steps/configs/__init__.py`

**Deliverables**:
- [ ] All registry files updated
- [ ] Import statements corrected
- [ ] Registry tests pass

#### 3.2 FlexibleFileResolver Updates
**Objective**: Update FlexibleFileResolver mappings for renamed files

**Mapping Updates**:
```python
# Update mappings in FlexibleFileResolver
'specs': {
    'data_loading_cradle': 'data_loading_cradle_spec.py',
    'model_xgboost': 'model_xgboost_spec.py',
    'model_pytorch': 'model_pytorch_spec.py',
    'training_xgboost': 'training_xgboost_spec.py',
    'training_pytorch': 'training_pytorch_spec.py',
},
'contracts': {
    'data_loading_cradle': 'data_loading_cradle_contract.py',
    'model_xgboost': 'model_xgboost_contract.py',
    'model_pytorch': 'model_pytorch_contract.py',
    'training_xgboost': 'training_xgboost_contract.py',
    'training_pytorch': 'training_pytorch_contract.py',
}
```

**Deliverables**:
- [ ] FlexibleFileResolver mappings updated
- [ ] File resolution tests pass
- [ ] Alignment validation succeeds

### Phase 4: Validation and Testing (Week 4)

#### 4.1 Comprehensive Alignment Validation
**Objective**: Run complete alignment validation suite to verify standardization

**Validation Levels**:
- **Level 1**: Script ↔ Contract alignment
- **Level 2**: Contract ↔ Specification alignment  
- **Level 3**: Specification ↔ Dependencies alignment
- **Level 4**: Builder ↔ Configuration alignment

**Success Criteria**:
- All validation levels achieve 100% success rate
- No missing components or broken references
- All processing steps pass comprehensive validation

**Validation Commands**:
```bash
# Run comprehensive alignment validation
cd test/steps/scripts/alignment_validation
python -m pytest test_unified_alignment.py -v

# Generate updated validation reports
python unified_alignment_tester.py --generate-reports

# Verify specific validation levels
python unified_alignment_tester.py --level 1 --level 2 --level 3 --level 4
```

**Deliverables**:
- [ ] All validation levels achieve 100% success rate
- [ ] Updated validation reports generated
- [ ] No critical alignment issues remain
- [ ] All processing steps fully validated

#### 4.2 Integration Testing
**Objective**: Ensure all renamed and created files integrate properly with the system

**Test Scope**:
- Import statement validation
- Registry integration testing
- Step builder instantiation testing
- Configuration loading validation
- Dependency resolution testing

**Deliverables**:
- [ ] All import statements resolve correctly
- [ ] Registry lookups succeed for all components
- [ ] Step builders instantiate without errors
- [ ] Configuration files load successfully
- [ ] Dependency resolution works for all steps

#### 4.3 Documentation Updates
**Objective**: Update all documentation to reflect new naming conventions

**Documentation Scope**:
- Developer guide updates
- API documentation
- Example code updates
- Tutorial updates

**Deliverables**:
- [ ] Developer guides updated with new naming conventions
- [ ] API documentation reflects current file names
- [ ] Example code uses correct import statements
- [ ] Tutorials reference updated file names

## 🚀 Expected Outcomes

### Immediate Benefits
1. **100% Naming Consistency**: All component files follow established patterns
2. **Complete Component Coverage**: All step types have all required component files
3. **Validation Success**: All alignment validation levels achieve 100% success rate
4. **Improved Maintainability**: Consistent patterns make code easier to maintain and extend

### Long-term Benefits
1. **Reduced Development Time**: Consistent patterns reduce cognitive load for developers
2. **Easier Onboarding**: New developers can quickly understand the system structure
3. **Better Tooling Support**: Consistent patterns enable better IDE support and automation
4. **Reduced Bugs**: Consistent naming reduces import errors and reference mistakes

### Success Metrics
- **File Naming Consistency**: 100% (currently 83%)
- **Component Completeness**: 100% (currently 80%)
- **Level 1 Validation**: 100% (currently 87.5%)
- **Level 2 Validation**: 100% (currently 100%)
- **Level 3 Validation**: 100% (currently 37.5%)
- **Level 4 Validation**: 100% (currently 87.5%)

## 🔄 Risk Mitigation

### Identified Risks
1. **Import Reference Breakage**: Renaming files may break existing imports
2. **Registry Integration Issues**: Registry mappings may become inconsistent
3. **Test Failures**: Existing tests may fail due to import changes
4. **Production Impact**: Changes may affect production pipeline behavior

### Mitigation Strategies
1. **Comprehensive Testing**: Extensive testing at each phase to catch issues early
2. **Incremental Implementation**: Phase-by-phase approach allows for early issue detection
3. **Backup and Rollback**: Version control and backup strategies for quick rollback
4. **Validation Integration**: Use existing validation framework to verify changes

### Contingency Plans
1. **Rollback Procedures**: Clear procedures for reverting changes if issues arise
2. **Hotfix Processes**: Rapid response procedures for critical issues
3. **Communication Plans**: Clear communication channels for reporting and resolving issues
4. **Alternative Approaches**: Backup approaches if primary strategy encounters obstacles

## 📅 Timeline and Milestones

### Week 1: File Renaming and Creation
- **Day 1-2**: Specification file renaming
- **Day 3-4**: Contract file standardization and creation
- **Day 5**: Import reference updates and initial testing

### Week 2: Content Standardization
- **Day 1-3**: Missing contract implementation
- **Day 4-5**: Specification content alignment and validation

### Week 3: Registry and Integration Updates
- **Day 1-3**: Registry integration updates
- **Day 4-5**: FlexibleFileResolver updates and testing

### Week 4: Validation and Testing
- **Day 1-2**: Comprehensive alignment validation
- **Day 3-4**: Integration testing and issue resolution
- **Day 5**: Documentation updates and final validation

### Key Milestones
- [ ] **Milestone 1**: All files renamed and created (End of Week 1)
- [ ] **Milestone 2**: All content standardized and validated (End of Week 2)
- [ ] **Milestone 3**: All registry integrations updated (End of Week 3)
- [ ] **Milestone 4**: 100% validation success achieved (End of Week 4)

## 🎯 Conclusion

This comprehensive code alignment standardization plan builds upon our extensive previous work to achieve complete system-wide alignment. By addressing the identified naming inconsistencies, creating missing components, and leveraging our proven validation framework, we will achieve 100% alignment across all validation levels.

The plan's phased approach ensures systematic progress while minimizing risk, and the comprehensive validation framework provides confidence in the results. Upon completion, the Cursus pipeline system will have consistent, maintainable, and fully aligned components that serve as a solid foundation for future development.

### Next Steps
1. **Review and Approval**: Stakeholder review and approval of this plan
2. **Resource Allocation**: Assign development resources for implementation
3. **Timeline Confirmation**: Confirm timeline and adjust based on resource availability
4. **Implementation Kickoff**: Begin Phase 1 implementation

### Success Criteria
The standardization effort will be considered successful when:
- All 15 processing steps achieve 100% validation success across all 4 levels
- All component files follow consistent naming conventions
- No broken imports or references remain in the system
- The validation framework reports zero critical alignment issues

This plan represents the culmination of months of alignment work and provides a clear path to achieving complete system alignment and standardization.
