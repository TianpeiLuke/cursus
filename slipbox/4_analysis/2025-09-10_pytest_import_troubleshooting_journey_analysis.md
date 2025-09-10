---
tags:
  - analysis
  - troubleshooting
  - test
  - pytest
  - import_resolution
keywords:
  - pytest import errors
  - package shadowing
  - src-layout
  - editable install
  - python import system
  - troubleshooting methodology
  - root cause analysis
topics:
  - pytest configuration
  - python package structure
  - import system debugging
  - test framework migration
language: python
date of note: 2025-09-10
---

# Pytest Import Troubleshooting Journey Analysis

## Executive Summary

This document chronicles the comprehensive troubleshooting journey undertaken to resolve pytest import issues during the conversion of unittest tests to pytest format for the `cursus/validation/runtime` module. The investigation revealed a subtle but critical package shadowing issue caused by an empty `__init__.py` file in the project root directory.

## Problem Statement

### Initial Task
Convert unittest tests to pytest format and add missing test coverage for the `cursus/validation/runtime` module, specifically:
- Convert existing unittest files to pytest format
- Add comprehensive tests for the previously untested `contract_discovery.py` module
- Ensure all tests pass with proper import resolution

### Encountered Issue
Pytest consistently failed to import the `cursus` package with the error:
```
ModuleNotFoundError: No module named 'cursus'
```

This occurred despite the package being properly installed and accessible via regular Python imports.

## Troubleshooting Journey

### Phase 1: Initial Hypothesis - Package Installation Issues

**Actions Taken:**
1. Verified package installation with `pip list | grep cursus`
2. Confirmed editable install with `pip install -e .`
3. Tested direct Python imports (successful)
4. Checked PYTHONPATH environment variable

**Results:**
- Package was properly installed in editable mode
- Direct Python imports worked correctly
- PYTHONPATH was not the issue

**Conclusion:** Package installation was not the root cause.

### Phase 2: Configuration Investigation

**Actions Taken:**
1. Examined `pyproject.toml` pytest configuration
2. Identified conflicting `pythonpath = ["src"]` setting
3. Removed conflicting configuration
4. Updated VSCode settings for proper test discovery

**Configuration Changes Made:**
```toml
# REMOVED from pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["src"]  # This was causing conflicts

# KEPT proper src-layout configuration
[tool.setuptools.packages.find]
where = ["src"]
```

**Results:**
- Configuration cleanup did not resolve the import issue
- Tests still failed with the same error

**Conclusion:** Configuration was not the primary issue.

### Phase 3: Interpreter Consistency Check

**Actions Taken:**
1. Verified both `python` and `pytest` used the same virtual environment
2. Checked interpreter paths with `which python` and `which pytest`
3. Confirmed both pointed to the same `.venv` environment

**Results:**
- Both commands used identical interpreter paths
- Virtual environment was consistent

**Conclusion:** Interpreter inconsistency was ruled out.

### Phase 4: Package Location Verification

**Actions Taken:**
1. Used Python introspection to locate the cursus package:
```python
import cursus
print(f"CURSUS FILE: {cursus.__file__}")
```

**Results:**
- Package correctly located at `/Users/tianpeixie/github_workspace/cursus/src/cursus/__init__.py`
- Import resolution worked in regular Python sessions

**Conclusion:** Package location was correct for normal Python imports.

### Phase 5: Systematic Diagnostic Approach

**Breakthrough Methodology:**
Following ChatGPT's diagnostic methodology, we created a probe test to examine pytest's import behavior:

**Probe Test Created:**
```python
# test/test_probe_env.py
def test_probe_cursus_import():
    """Probe test to diagnose pytest import issues"""
    try:
        import cursus
        print(f"SUCCESS: CURSUS FILE: {cursus.__file__}")
        assert True
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        assert False, f"Failed to import cursus: {e}"
```

**Critical Discovery:**
When run with pytest, the probe test revealed:
```
CURSUS FILE: /Users/tianpeixie/github_workspace/cursus/__init__.py
```

**Root Cause Identified:**
Pytest was finding `cursus` at the project root (`/Users/tianpeixie/github_workspace/cursus/__init__.py`) instead of the correct location (`/Users/tianpeixie/github_workspace/cursus/src/cursus/__init__.py`).

### Phase 6: Package Shadowing Investigation

**Analysis:**
The probe test revealed that an empty `__init__.py` file existed in the project root directory, causing Python's import system to treat the entire project root as a package named "cursus".

**Package Shadowing Mechanism:**
1. Python's import system searches for packages in order
2. The empty `__init__.py` in the root made the root directory a Python package
3. This "cursus" package (the root directory) was found before the real `src/cursus` package
4. The shadowing package had no actual modules, causing import failures

**Verification:**
```bash
ls -la __init__.py  # Confirmed existence of empty file
```

## Root Cause Analysis

### The Problem
An empty `__init__.py` file in the project root directory (`/Users/tianpeixie/github_workspace/cursus/__init__.py`) was creating a package shadow that interfered with pytest's import resolution.

### Why This Affected Pytest But Not Regular Python
- **Regular Python imports:** Worked because they typically imported specific modules directly
- **Pytest discovery:** Failed because pytest's test collection process searches for packages more broadly and encountered the shadowing package first

### Technical Details
The Python import system follows this search order:
1. Current working directory
2. PYTHONPATH directories
3. Standard library directories
4. Site-packages directories

The empty `__init__.py` in the root directory caused the root to be treated as the "cursus" package, which was found before the real `src/cursus` package during pytest's import resolution.

## Solution Implementation

### Step 1: Remove Shadowing File
```bash
rm -f __init__.py  # Remove the empty file from project root
```

### Step 2: Verification
```bash
python -m pytest test/validation/runtime/test_runtime_models_pytest.py -v
```

**Result:** All 45 tests passed successfully.

### Step 3: Comprehensive Testing
```bash
python -m pytest test/validation/runtime/test_*_pytest.py -v
```

**Result:** All 108 converted pytest tests passed.

## Impact Assessment

### Before Fix
- 0% of pytest tests working
- Complete import failure for cursus package
- Blocked conversion from unittest to pytest

### After Fix
- 100% of converted pytest tests working (108/108 tests passing)
- Complete import resolution success
- Successful unittest to pytest conversion
- Added comprehensive test coverage for previously untested modules

## Lessons Learned

### 1. Package Shadowing is Subtle but Critical
Empty `__init__.py` files can create unexpected package shadows that are difficult to diagnose without systematic investigation.

### 2. Diagnostic Methodology is Essential
The probe test approach was crucial for identifying the root cause. Without it, we might have continued investigating configuration issues indefinitely.

### 3. Import System Behavior Varies by Context
The same import can work in one context (regular Python) but fail in another (pytest) due to different search patterns and working directories.

### 4. Systematic Elimination Process
Following a structured troubleshooting approach (installation → configuration → environment → location → behavior) was essential for efficient problem resolution.

## Preventive Measures

### 1. Project Structure Guidelines
- Avoid empty `__init__.py` files in project root directories
- Use src-layout consistently for Python packages
- Document package structure clearly

### 2. Testing Practices
- Include import verification in test suites
- Test both regular Python imports and pytest imports
- Use probe tests for diagnosing complex import issues

### 3. Configuration Management
- Maintain clean pytest configuration
- Avoid conflicting PYTHONPATH settings
- Document configuration decisions

## Technical Artifacts Created

### 1. Converted Test Files
- `test/validation/runtime/test_runtime_models_pytest.py` (45 tests)
- `test/validation/runtime/test_logical_name_matching_pytest.py` (32 tests)
- `test/validation/runtime/test_contract_discovery_pytest.py` (31 tests)

### 2. Configuration Updates
- Updated `pyproject.toml` with clean pytest configuration
- Updated `.vscode/settings.json` for proper test discovery

### 3. Diagnostic Tools
- Probe test methodology for import debugging
- Systematic troubleshooting checklist

## Conclusion

This troubleshooting journey demonstrates the importance of systematic diagnostic approaches when dealing with complex import issues. The root cause—an empty `__init__.py` file creating package shadowing—was subtle and required methodical investigation to identify.

The successful resolution enabled:
- Complete conversion from unittest to pytest format
- Addition of comprehensive test coverage for previously untested modules
- Establishment of a robust testing foundation for the runtime validation system

The experience provides valuable insights for future troubleshooting efforts and highlights the critical importance of proper Python package structure in complex projects.

## Recommendations

### For Future Development
1. **Avoid Root-Level `__init__.py` Files:** Never place `__init__.py` files in project root directories unless specifically required
2. **Use Probe Tests:** When facing import issues, create simple probe tests to diagnose the exact import behavior
3. **Systematic Troubleshooting:** Follow structured elimination processes rather than random configuration changes
4. **Document Package Structure:** Clearly document and maintain consistent package structure across projects

### For Testing Infrastructure
1. **Import Verification Tests:** Include basic import tests in CI/CD pipelines
2. **Environment Consistency Checks:** Regularly verify that development and testing environments are consistent
3. **Configuration Validation:** Implement checks to ensure pytest configuration doesn't conflict with package structure

This analysis serves as both a record of the troubleshooting process and a guide for handling similar issues in the future.
