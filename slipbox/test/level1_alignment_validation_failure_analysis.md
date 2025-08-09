---
tags:
  - test
  - validation
  - alignment
  - failure_analysis
  - level1
keywords:
  - alignment validation
  - script contract alignment
  - false positives
  - validation failure
  - static analysis
  - file operations detection
  - logical name extraction
topics:
  - validation framework
  - alignment testing
  - test failure analysis
  - script analysis
language: python
date of note: 2025-08-09
---

# Level 1 Alignment Validation Failure Analysis

## Executive Summary

The Level 1 script-to-contract alignment validation is producing **systematic false positives** across all 8 scripts in the test suite. All scripts are reporting as FAILING when they should be PASSING, indicating fundamental flaws in the validation logic rather than actual alignment issues.

**Related Design**: [Unified Alignment Tester Design](../1_design/unified_alignment_tester_design.md)

## Test Results Overview

**Validation Run**: 2025-08-09T00:12:36.501830

```
Total Scripts: 8
Passed Scripts: 0
Failed Scripts: 8
Error Scripts: 0
Overall Status: ALL FAILING
```

**Affected Scripts**:
- currency_conversion
- dummy_training
- mims_package
- mims_payload
- model_calibration
- model_evaluation_xgb
- risk_table_mapping
- tabular_preprocess

## Root Cause Analysis

After detailed analysis of the validation reports and source code, I've identified **three critical flaws** in the alignment validation system:

### 1. File Operations Detection Failure

**Problem**: The `ScriptAnalyzer.extract_file_operations()` method only detects explicit `open()` calls, but scripts use higher-level file operations.

**Evidence from dummy_training.py**:
- Uses `tarfile.open()` for reading tar files
- Uses `shutil.copy2()` for copying files  
- Uses `Path.mkdir()` for creating directories
- Uses `tarfile.extractall()` and `tarfile.add()` for tar operations

**Current Detection**: None of these operations are detected by the analyzer.

**Impact**: Validator incorrectly reports that scripts don't read/write the declared contract paths.

### 2. Incorrect Logical Name Extraction

**Problem**: The `extract_logical_name_from_path()` function in `alignment_utils.py` has a flawed algorithm.

**Current (Broken) Logic**:
```python
'/opt/ml/processing/input/model/model.tar.gz' → extracts 'model'
'/opt/ml/processing/input/config/hyperparameters.json' → extracts 'config'
```

**Contract Reality**:
- `pretrained_model_path` for `/opt/ml/processing/input/model/model.tar.gz`
- `hyperparameters_s3_uri` for `/opt/ml/processing/input/config/hyperparameters.json`

**Impact**: Validator incorrectly reports "config" and "model" as undeclared logical names.

### 3. Path Usage vs File Operations Mismatch

**Problem**: The validator checks path references separately from file operations without correlating them properly.

**Script Pattern**:
```python
# Script declares paths as constants (lines 33-35)
MODEL_INPUT_PATH = "/opt/ml/processing/input/model/model.tar.gz"
HYPERPARAMS_INPUT_PATH = "/opt/ml/processing/input/config/hyperparameters.json"
MODEL_OUTPUT_DIR = "/opt/ml/processing/output/model"

# Uses those constants in file operations throughout the code
model_path = Path(MODEL_INPUT_PATH)
hyperparams_path = Path(HYPERPARAMS_INPUT_PATH)
```

**Impact**: Analyzer treats path declarations and file operations as separate concerns, missing the connection.

## Detailed Example: dummy_training False Positives

### Reported Issues (All False Positives):

1. **WARNING**: Script uses logical name not in contract: config
   - **Reality**: Script correctly uses contract path `/opt/ml/processing/input/config/hyperparameters.json`
   - **Problem**: Logical name extraction incorrectly derives "config" from path

2. **WARNING**: Script uses logical name not in contract: model
   - **Reality**: Script correctly uses contract path `/opt/ml/processing/input/model/model.tar.gz`
   - **Problem**: Logical name extraction incorrectly derives "model" from path

3. **INFO**: Contract declares input not read by script: `/opt/ml/processing/input/config/hyperparameters.json`
   - **Reality**: Script DOES read this file via `hyperparams_path.exists()` and file operations
   - **Problem**: File operations detection misses higher-level operations

4. **INFO**: Contract declares input not read by script: `/opt/ml/processing/input/model/model.tar.gz`
   - **Reality**: Script DOES read this file via `tarfile.open()` operations
   - **Problem**: File operations detection misses tarfile operations

5. **WARNING**: Contract declares output not written by script: `/opt/ml/processing/output/model`
   - **Reality**: Script DOES write to this path via `create_tarfile()` function
   - **Problem**: File operations detection misses indirect file operations

### Verification of Actual Alignment

**Script Implementation** (`dummy_training.py`):
- ✅ Reads from: `/opt/ml/processing/input/model/model.tar.gz` (line 33, used in tarfile operations)
- ✅ Reads from: `/opt/ml/processing/input/config/hyperparameters.json` (line 34, used in file operations)
- ✅ Writes to: `/opt/ml/processing/output/model` (line 35, used in output operations)

**Contract Declaration** (`dummy_training_contract.py`):
- ✅ Input "pretrained_model_path": `/opt/ml/processing/input/model/model.tar.gz`
- ✅ Input "hyperparameters_s3_uri": `/opt/ml/processing/input/config/hyperparameters.json`
- ✅ Output "model_input": `/opt/ml/processing/output/model`

**Conclusion**: The script and contract ARE actually aligned - the validator is incorrectly detecting misalignment.

## Technical Implementation Issues

### ScriptAnalyzer Limitations

**File**: `src/cursus/validation/alignment/static_analysis/script_analyzer.py`

**Current `extract_file_operations()` method only detects**:
```python
def visit_Call(self, node):
    # Check for open() calls
    if isinstance(node.func, ast.Name) and node.func.id == 'open':
        # ... process open() calls only
```

**Missing Detection Patterns**:
- `tarfile.open()`, `tarfile.extractall()`, `tarfile.add()`
- `shutil.copy()`, `shutil.copy2()`, `shutil.move()`
- `Path.read_text()`, `Path.write_text()`, `Path.mkdir()`
- Variable-based file operations (when paths are stored in variables)

### Alignment Utils Flaws

**File**: `src/cursus/validation/alignment/alignment_utils.py`

**Broken `extract_logical_name_from_path()` function**:
```python
def extract_logical_name_from_path(path: str) -> Optional[str]:
    # ... extracts directory name instead of using contract mapping
    for pattern in patterns:
        if normalized_path.startswith(pattern):
            remainder = normalized_path[len(pattern):].strip('/')
            if remainder:
                # Return the first path component as logical name
                return remainder.split('/')[0]  # ← This is wrong!
```

**Problem**: Returns directory name ("config", "model") instead of contract logical name.

### Validation Logic Issues

**File**: `src/cursus/validation/alignment/script_contract_alignment.py`

**`_validate_file_operations()` method limitations**:
- Doesn't track variable assignments of paths
- Doesn't correlate path constants with their usage in file operations
- Doesn't handle indirect file operations through variables
- Treats path references and file operations as separate validation concerns

## Impact Assessment

### Immediate Impact
- **100% false positive rate** on Level 1 validation
- **All scripts incorrectly marked as failing** alignment validation
- **Development workflow disrupted** by unreliable validation results
- **Trust in validation system compromised**

### Downstream Impact
- Level 2-4 validations may be affected by incorrect Level 1 results
- Integration tests may fail due to validation dependencies
- CI/CD pipeline reliability compromised
- Developer productivity reduced by false alarms

## Recommended Fix Strategy

### Phase 1: Enhance File Operations Detection
**Target**: `src/cursus/validation/alignment/static_analysis/script_analyzer.py`

1. **Expand `extract_file_operations()`** to detect:
   - `tarfile.open()`, `tarfile.extractall()`, `tarfile.add()`
   - `shutil.copy()`, `shutil.copy2()`, `shutil.move()`
   - `Path.read_text()`, `Path.write_text()`, `Path.mkdir()`
   - Variable-based file operations

2. **Add variable tracking** to connect path constants to their usage

### Phase 2: Fix Logical Name Extraction
**Target**: `src/cursus/validation/alignment/alignment_utils.py`

1. **Replace broken `extract_logical_name_from_path()`** with contract-aware mapping
2. **Add path-to-logical-name resolution** using actual contract mappings

### Phase 3: Improve Path-Operation Correlation
**Target**: `src/cursus/validation/alignment/script_contract_alignment.py`

1. **Enhance `_validate_file_operations()`** to:
   - Track variable assignments of paths
   - Correlate path constants with file operations
   - Handle indirect operations through variables

### Phase 4: Add Contract-Aware Validation
1. **Use contract logical names as source of truth**
2. **Remove path-based logical name guessing**
3. **Implement contract-driven validation logic**

## Expected Outcome

After implementing these fixes:
- ✅ **Level 1: PASSING** for all correctly aligned scripts
- ✅ **No false positive warnings** about logical names
- ✅ **Proper detection** of all file operation patterns
- ✅ **Accurate alignment validation** based on actual script behavior

## Priority and Urgency

**Priority**: CRITICAL
**Urgency**: HIGH

**Rationale**: 
- Validation system is currently unusable due to 100% false positive rate
- Blocks development workflow and CI/CD reliability
- Undermines confidence in the entire alignment validation framework
- Must be fixed before any meaningful validation can occur

## Next Steps

1. **Immediate**: Implement Phase 1 file operations detection enhancements
2. **Short-term**: Fix logical name extraction and path correlation logic
3. **Medium-term**: Validate fixes against all 8 scripts in test suite
4. **Long-term**: Add regression tests to prevent similar issues

## Related Issues

- All Level 1 alignment validation failures are likely related to these core issues
- Similar patterns may exist in other validation levels
- Static analysis framework may need broader improvements beyond alignment validation

---

**Analysis Date**: 2025-08-09  
**Analyst**: System Analysis  
**Status**: Critical Issue Identified - Requires Immediate Fix  
**Related Design**: [Unified Alignment Tester Design](../1_design/unified_alignment_tester_design.md#level-1-script--contract-alignment)
