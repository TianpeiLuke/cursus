# Level-3 and Level-4 Issues Summary Table
**Generated:** 2025-08-11T08:38:13

## Quick Reference: Script-by-Script Issues

| Script | Level-3 Status | Level-3 Issues | Level-4 Status | Level-4 Issues |
|--------|----------------|----------------|----------------|----------------|
| **currency_conversion** | ✅ PASS | None | ❌ FAIL | Missing config file |
| **dummy_training** | ❌ FAIL | 2 dependency resolution failures | ❌ FAIL | Missing config file |
| **mims_package** | ❌ FAIL | Missing specification | ❌ FAIL | Missing config file |
| **mims_payload** | ❌ FAIL | Missing specification | ❌ FAIL | Missing config file |
| **model_calibration** | ❌ FAIL | Missing specification | ❌ FAIL | Missing config file |
| **model_evaluation_xgb** | ❌ FAIL | Missing specification + dependency | ❌ FAIL | Missing config file |
| **risk_table_mapping** | ✅ PASS | None | ❌ FAIL | Missing config file |
| **tabular_preprocess** | ❌ FAIL | Missing specification | ❌ FAIL | Missing config file |

## Level-3 Detailed Issues

### Specification Registry Mismatches
| Script | Expected Specification | Available in Registry | Status |
|--------|----------------------|---------------------|---------|
| dummy_training | `Dummy_Training` | `DummyTraining` | ❌ Mismatch |
| mims_package | `MimsPackage` | `Package` | ❌ Mismatch |
| mims_payload | `MimsPayload` | `Payload` | ❌ Mismatch |
| model_calibration | `Model_Calibration` | `ModelCalibration` | ❌ Mismatch |
| model_evaluation_xgb | `ModelEvaluationXgb` | `XGBoostModelEval` | ❌ Mismatch |
| tabular_preprocess | `TabularPreprocess` | `TabularPreprocessing` | ❌ Mismatch |

### Dependency Resolution Details
| Script | Dependency | Compatible Sources | Resolution Status |
|--------|------------|-------------------|------------------|
| **currency_conversion** | `data_input` | TabularPreprocessing, CradleDataLoading, ProcessingStep | ✅ Resolved → Pytorch.data_output |
| **dummy_training** | `pretrained_model_path` | XGBoostTraining, ProcessingStep, PytorchTraining, TabularPreprocessing | ❌ Cannot resolve |
| **dummy_training** | `hyperparameters_s3_uri` | ProcessingStep, HyperparameterPrep | ❌ Cannot resolve |
| **risk_table_mapping** | `data_input` | TabularPreprocessing, CradleDataLoading, ProcessingStep | ✅ Resolved → Pytorch.data_output |
| **risk_table_mapping** | `risk_tables` | TabularPreprocessing, CradleDataLoading, ProcessingStep | ✅ Resolved → Preprocessing.processed_data |
| **risk_table_mapping** | `hyperparameters_s3_uri` | ProcessingStep, HyperparameterPrep | ⚠️ Optional - not resolved |

## Level-4 Missing Configuration Files

| Script | Expected Config File | Status |
|--------|---------------------|---------|
| currency_conversion | `config_currency_conversion_step.py` | ❌ Missing |
| dummy_training | `config_dummy_training_step.py` | ❌ Missing |
| mims_package | `config_mims_package_step.py` | ❌ Missing |
| mims_payload | `config_mims_payload_step.py` | ❌ Missing |
| model_calibration | `config_model_calibration_step.py` | ❌ Missing |
| model_evaluation_xgb | `config_model_evaluation_xgb_step.py` | ❌ Missing |
| risk_table_mapping | `config_risk_table_mapping_step.py` | ❌ Missing |
| tabular_preprocess | `config_tabular_preprocess_step.py` | ❌ Missing |

## Available Registry Steps
The following steps are currently registered and available for dependency resolution:
```
DataLoading, Preprocessing, CurrencyConversion, ModelEval, XgboostModel, 
Registration, RiskTableMapping, BatchTransform, Dummy, Model, Payload, 
Xgboost, Pytorch, Packaging, PytorchModel
```

## Fix Priority Matrix

### Critical (Blocks All Scripts)
- **Level-4:** Create all 8 missing configuration files
- **Level-3:** Fix specification registry name mappings

### High Priority (Specific Scripts)
- **dummy_training:** Fix `Dummy_Training` → `DummyTraining` mapping
- **mims_package:** Fix `MimsPackage` → `Package` mapping  
- **mims_payload:** Fix `MimsPayload` → `Payload` mapping
- **model_calibration:** Fix `Model_Calibration` → `ModelCalibration` mapping
- **model_evaluation_xgb:** Fix `ModelEvaluationXgb` → `XGBoostModelEval` mapping
- **tabular_preprocess:** Fix `TabularPreprocess` → `TabularPreprocessing` mapping

### Success Examples (Keep Working)
- **currency_conversion:** Level-3 dependency resolution working correctly
- **risk_table_mapping:** Level-3 dependency resolution working correctly

## Validation Statistics

### Level-3 Statistics
- **Total Dependencies:** 11 across all scripts
- **Successfully Resolved:** 3 (27%)
- **Failed to Resolve:** 8 (73%)
- **Scripts with All Dependencies Resolved:** 2/8 (25%)

### Level-4 Statistics  
- **Total Configuration Files Expected:** 8
- **Configuration Files Found:** 0 (0%)
- **Scripts with Configuration:** 0/8 (0%)

### Overall Alignment Health
- **Fully Aligned Scripts:** 0/8 (0%)
- **Scripts Passing Level-3:** 2/8 (25%)
- **Scripts Passing Level-4:** 0/8 (0%)
- **Critical Blocking Issues:** 14 total (6 Level-3 + 8 Level-4)

---
**Report Generated:** 2025-08-11T08:38:13  
**Source:** Comprehensive Alignment Validation Test Suite  
**Focus:** Level-3 (Specification ↔ Dependencies) and Level-4 (Builder ↔ Configuration)
