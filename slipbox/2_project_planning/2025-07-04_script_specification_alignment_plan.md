---
tags:
  - project
  - planning
  - script_contracts
  - alignment
keywords:
  - script contracts
  - specification alignment
  - validation framework
  - path handling
  - MIMS payload
topics:
  - script contract system
  - validation framework
  - path handling
  - implementation plan
language: python
date of note: 2025-07-04
---

# Script-Specification Alignment Implementation Plan
*Date: July 4, 2025*
*Updated: July 12, 2025*

## Executive Summary

Successfully implemented a comprehensive **Script Contract System** that bridges the gap between step specifications and script implementations. This system provides explicit I/O contracts, validation capabilities, and supports processing, training, source node, and sink node script patterns.

**Latest Achievement (July 12, 2025)**: Fixed critical issue with MIMS payload path handling that improves robustness in the payload and registration steps. See [MIMS Payload Path Handling Fix](./2025-07-12_mims_payload_path_handling_fix.md) for full details.

## ✅ Completed Implementation

### 1. Core Contract Framework
- **Base Script Contract**: Foundation class for processing scripts (`/opt/ml/processing/*` paths)
- **Training Script Contract**: Specialized class for training scripts (`/opt/ml/input/data/*`, `/opt/ml/model/*` paths)
- **Validation Framework**: AST-based script analysis and contract compliance checking

### 2. Processing Script Contracts (7 contracts)
- `tabular_preprocess.py` - Tabular data preprocessing with risk tables
- `mims_package.py` - MIMS model packaging 
- `mims_payload.py` - MIMS payload generation
- `mims_registration.py` - MIMS model registration (sink node)
- `model_evaluation_xgb.py` - XGBoost model evaluation
- `currency_conversion.py` - Currency conversion processing
- `risk_table_mapping.py` - Risk table mapping operations
- `cradle_data_loading.py` - Cradle data loading (source node)

### 3. Training Script Contracts (2 contracts)
- `train.py` - PyTorch Lightning multimodal training
- `train_xgb.py` - XGBoost tabular training

### 4. Contract Validation System
- **ScriptContractValidator**: Automated compliance checking
- **ContractValidationReport**: Detailed validation results
- **CLI Interface**: Command-line validation tools
- **AST Analysis**: Static code analysis for I/O pattern detection

### 5. Path Handling Enhancements (NEW - July 12, 2025)
- **MIMS Payload Path Handling Fix**: Resolved critical path handling issue for payload generation
- **SageMaker Directory/File Conflict Resolution**: Fixed issue where SageMaker creates directories at output paths
- **Path Validation Analysis**: Documented how MIMS validation works with SageMaker property references
- **Comprehensive Documentation**: Created detailed explanation in [MIMS Payload Path Handling Fix](./2025-07-12_mims_payload_path_handling_fix.md)

## 🎯 Key Achievements

### Contract Coverage
- **10 total contracts** covering all major pipeline scripts and node types
- **100% framework coverage** for all script patterns (processing, training, source, sink)
- **Explicit I/O specifications** for all script types

### Validation Capabilities
```bash
# Example validation results
=== PyTorch Training Script Validation ===
train.py vs TrainingScriptContract: ❌ NON-COMPLIANT
Errors: ["Script doesn't use expected input path: /opt/ml/input/data/train (for train_data)"]

=== XGBoost Training Script Validation ===  
train_xgb.py vs TrainingScriptContract: ❌ NON-COMPLIANT
Errors: ["Script doesn't use expected input path: /opt/ml/input/data/train (for train_data)"]
```

### Framework Requirements Documentation
- **Exact version specifications** from requirements.txt files
- **PyTorch ecosystem**: torch==2.1.2, lightning==2.1.3, transformers==4.37.2
- **XGBoost ecosystem**: xgboost==1.7.6, scikit-learn>=0.23.2,<1.0.0
- **Common dependencies**: pandas, numpy, pydantic, matplotlib

### Path Handling Improvements (NEW)
- **MIMS Payload Path Fix**: Successfully resolved path handling issues in the MIMS payload step
- **Directory/File Conflict Resolution**: Fixed issue where SageMaker creates directories at file paths
- **Enhanced Contract Design**: Updated contract to handle SageMaker's container output behavior
- **Tested All Templates**: Verified fix works across all template types

## 📋 Contract Specifications

### Processing Script Pattern
```python
ScriptContract(
    entry_point="script.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/data"
    },
    expected_output_paths={
        "output_data": "/opt/ml/processing/output/data"
    },
    required_env_vars=["REQUIRED_VAR"],
    framework_requirements={"pandas": ">=1.3.0"}
)
```

### Training Script Pattern
```python
TrainingScriptContract(
    entry_point="train.py", 
    expected_input_paths={
        "train_data": "/opt/ml/input/data/train",
        "config": "/opt/ml/input/config/hyperparameters.json"
    },
    expected_output_paths={
        "model_output": "/opt/ml/model",
        "data_output": "/opt/ml/output/data"
    },
    framework_requirements={"torch": "==2.1.2"}
)
```

### Source Node Script Pattern
```python
# Example: Cradle Data Loading Contract
ScriptContract(
    entry_point="scripts.py",
    expected_input_paths={
        # No inputs as this is a source node
    },
    expected_output_paths={
        "DATA": "/opt/ml/processing/output/place_holder",
        "METADATA": "/opt/ml/processing/output/metadata",
        "SIGNATURE": "/opt/ml/processing/output/signature"
    },
    optional_env_vars={
        "OUTPUT_PATH": ""  # Optional override for data output path
    },
    framework_requirements={
        "python": ">=3.7",
        "secure_ai_sandbox_python_lib": "*"  # Core dependency for Cradle integration
    }
)
```

### Sink Node Script Pattern
```python
# Example: MIMS Registration Contract
ScriptContract(
    entry_point="script.py",
    expected_input_paths={
        "PackagedModel": "/opt/ml/processing/input/model",
        "GeneratedPayloadSamples": "/opt/ml/processing/mims_payload"
    },
    expected_output_paths={
        # No output paths as this is a registration step with side effects only
    },
    required_env_vars=[
        "MODS_WORKFLOW_EXECUTION_ID"  # Environment variable required for registration
    ],
    optional_env_vars={
        "PERFORMANCE_METADATA_PATH": ""  # Optional S3 path to performance metadata
    },
    framework_requirements={
        "python": ">=3.7"
        # Uses secure_ai_sandbox_python_lib libraries and standard modules
    }
)
```

### MIMS Payload Path Handling Fix (NEW)
```python
# Before (causing directory/file conflict)
MIMS_PAYLOAD_CONTRACT = ScriptContract(
    entry_point="mims_payload.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model"
    },
    expected_output_paths={
        "payload_sample": "/opt/ml/processing/output/payload.tar.gz"  # Conflict - SageMaker creates this as directory
    },
    # Other fields omitted for brevity
)

# After (fixing the issue)
MIMS_PAYLOAD_CONTRACT = ScriptContract(
    entry_point="mims_payload.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model"
    },
    expected_output_paths={
        "payload_sample": "/opt/ml/processing/output"  # Changed to directory path
    },
    # Other fields omitted for brevity
)
```

## 🔍 Validation Results Analysis

### Current Status
- **Processing scripts**: All validated with expected compliance patterns
- **Training scripts**: Validated with expected non-compliance
- **Source node scripts**: Validated with special handling for no-input pattern
- **Sink node scripts**: Validated with special handling for no-output pattern
- **Non-compliance reasons**: AST analyzer limitations with dynamic path construction
- **Path handling**: Fixed SageMaker directory/file conflict issue for MIMS payload

### Expected Patterns in Scripts
Training scripts use dynamic path construction:
```python
# PyTorch train.py
prefix = "/opt/ml/"
input_path = os.path.join(prefix, "input/data")
train_path = os.path.join(input_path, "train")

# XGBoost train_xgb.py  
input_path = os.path.join(prefix, "input", "data")
train_file = find_first_data_file(os.path.join(input_path, "train"))
```

## 🚀 Usage Examples

### Validate Single Script
```python
from src.pipeline_script_contracts import ScriptContractValidator

validator = ScriptContractValidator('dockers/pytorch_bsm')
report = validator.validate_script('train.py')
print(report.summary)
```

### Validate All Scripts
```python
validator = ScriptContractValidator()
reports = validator.validate_all_scripts()
summary = validator.generate_compliance_summary(reports)
print(summary)
```

### CLI Usage
```bash
# Validate specific script
python -m src.pipeline_script_contracts.contract_validator --script train.py

# Validate all scripts
python -m src.pipeline_script_contracts.contract_validator --verbose
```

## 📁 File Structure

```
src/pipeline_script_contracts/
├── __init__.py                       # Module exports
├── base_script_contract.py           # Base contract for processing scripts
├── training_script_contract.py       # Specialized contract for training scripts
├── contract_validator.py             # Validation framework
├── tabular_preprocess_contract.py    # Tabular preprocessing contract
├── mims_package_contract.py          # MIMS packaging contract
├── mims_payload_contract.py          # MIMS payload contract
├── mims_registration_contract.py     # MIMS registration contract (sink node)
├── model_evaluation_contract.py      # Model evaluation contract
├── currency_conversion_contract.py   # Currency conversion contract
├── risk_table_mapping_contract.py    # Risk table mapping contract
├── cradle_data_loading_contract.py   # Cradle data loading contract (source node)
├── pytorch_train_contract.py         # PyTorch training contract
└── xgboost_train_contract.py         # XGBoost training contract
```

## 🎯 Business Value

### 1. Explicit I/O Contracts
- **Clear expectations** for script inputs and outputs
- **Standardized interfaces** across all pipeline components
- **Reduced integration errors** through contract validation

### 2. Framework Standardization
- **Exact version requirements** prevent dependency conflicts
- **Consistent environments** across development and production
- **Reproducible builds** with locked dependencies

### 3. Automated Compliance
- **Static analysis** catches contract violations early
- **CI/CD integration** prevents non-compliant deployments
- **Documentation generation** from contract specifications

### 4. Developer Experience
- **Self-documenting code** through contract specifications
- **IDE integration** potential for contract-aware development
- **Onboarding acceleration** with clear script interfaces

### 5. Robust Error Handling (NEW)
- **Directory/File Conflict Prevention**: Improved contract design to handle SageMaker behavior
- **Path Validation Analysis**: Enhanced understanding of validation mechanisms
- **Clear Error Messages**: Better diagnostics for path-related issues
- **Resilient Templates**: Validated across all template types

## 🔮 Future Enhancements

### 1. Enhanced AST Analysis
- **Dynamic path resolution** for os.path.join patterns
- **Variable tracking** across function boundaries
- **Import analysis** for framework usage validation

### 2. Contract Evolution
- **Version compatibility** checking between contracts
- **Breaking change detection** in script modifications
- **Migration assistance** for contract updates

### 3. Integration Opportunities
- **Step specification alignment** with contract requirements
- **Pipeline builder integration** using contract metadata
- **Automated test generation** from contract specifications

### 4. Path Validation Improvements (NEW)
- **Formalized Path Validation Rules**: Create standardized patterns for path validation
- **Container Path Analyzer**: Tool to detect common path handling issues
- **Path Test Generator**: Generate tests for path edge cases
- **S3 Path Pattern Standardization**: Best practices for S3 path structures

## 📊 Success Metrics

### Implementation Completeness
- ✅ **10/10 contracts defined** (100% coverage)
- ✅ **4 contract types** (processing + training + source + sink)
- ✅ **Validation framework** operational
- ✅ **CLI interface** available
- ✅ **Path handling fixes** implemented

### Quality Indicators
- ✅ **Type safety** with Pydantic validation
- ✅ **Comprehensive documentation** in contract descriptions
- ✅ **Framework requirements** accurately captured
- ✅ **Path validation** for SageMaker conventions
- ✅ **Container behavior compatibility** for directory/file handling

## 🎉 Conclusion

The Script Contract System successfully bridges the specification-implementation gap by providing:

1. **Explicit contracts** for all pipeline scripts
2. **Automated validation** capabilities
3. **Framework standardization** with exact requirements
4. **Developer-friendly interfaces** for contract management
5. **Robust path handling** for SageMaker compatibility

This foundation enables reliable pipeline development with clear interfaces, automated compliance checking, and standardized environments across all script types.

## 🚀 Latest Achievements (July 12, 2025)

### MIMS Payload Path Handling Fix

We identified and resolved a critical issue with the MIMS payload step:

#### Problem
The MIMS payload step was encountering errors during execution, specifically when trying to write the payload archive:

```
ERROR:__main__:ERROR: Archive path exists but is a directory: /opt/ml/processing/output/payload.tar.gz
ERROR:__main__:Error creating payload archive: [Errno 21] Is a directory: '/opt/ml/processing/output/payload.tar.gz'
```

#### Root Cause
1. **SageMaker Behavior**: SageMaker creates a directory at the path specified in `ProcessingOutput`'s `source` parameter before the script executes
2. **Contract Configuration**: Our script contract specified the output path as `/opt/ml/processing/output/payload.tar.gz`
3. **Script Behavior**: The script attempted to create a file at the same path where SageMaker had already created a directory
4. **Result**: The script failed with a "Is a directory" error

#### Solution
1. **Updated Contract**: Modified the contract to specify a directory path instead of a file path
   ```python
   # Before
   "payload_sample": "/opt/ml/processing/output/payload.tar.gz"
   
   # After
   "payload_sample": "/opt/ml/processing/output"
   ```

2. **Updated Builder**: Modified the builder to generate S3 paths without the file suffix
   ```python
   # Before
   destination = f"{self.config.pipeline_s3_loc}/payload/{logical_name}/payload.tar.gz"
   
   # After
   destination = f"{self.config.pipeline_s3_loc}/payload/{logical_name}"
   ```

3. **Script Compatibility**: The script still writes to `/opt/ml/processing/output/payload.tar.gz`, but now as a file within the directory

#### Path Validation Analysis
We discovered an important insight about how MIMS validation works with SageMaker property references:

```python
# In MimsModelRegistrationProcessor.validate_processing_job_input_file:
try:
    if not input_file_location.endswith(".tar.gz"):
        return False
except AttributeError:
    # If this error is thrown than the input_file_location is a SageMaker property set in the pipeline
    if "S3" not in input_file_location.expr["Get"]:
        return False
```

This explains why our solution works - the `.tar.gz` validation only applies to direct string paths, not to property references from pipeline steps. At runtime, the MIMS registration script focuses on the file content, not the path suffix.

For full details of this fix, see [MIMS Payload Path Handling Fix](./2025-07-12_mims_payload_path_handling_fix.md).

### Complete Node Type and Template Type Coverage

With this latest fix, we've successfully validated our contract system across:

1. **All Node Types**: Source, internal, training, and sink nodes
2. **All Template Types**:
   - XGBoostTrainEvaluateE2ETemplate (Complete pipeline with registration)
   - XGBoostTrainEvaluateNoRegistrationTemplate (No registration)
   - XGBoostSimpleTemplate (Basic training only)
   - XGBoostDataloadPreprocessTemplate (Data loading and preprocessing only)
   - CradleOnlyTemplate (Data loading only)

This comprehensive validation confirms the robustness and flexibility of our contract system.

---

*Latest update: July 12, 2025*  
*Total contracts: 10 (8 processing + 2 training)*  
*Validation framework: Operational*  
*Framework coverage: 100%*  
*Template coverage: 100%*
