---
title: "Circular Import Test Suite Documentation"
date: "2025-08-06"
type: "test_documentation"
related_docs:
  - "../../slipbox/test/circular_import_analysis_report.md"
  - "../../slipbox/test/circular_import_fix_summary.md"
tags:
  - "circular_imports"
  - "testing"
  - "test_suite"
  - "documentation"
---

# Circular Import Test for Cursus Package

## Related Documentation

- 📊 **[Circular Import Analysis Report](../../slipbox/test/circular_import_analysis_report.md)** - Detailed analysis of circular import issues found
- 📋 **[Circular Import Fix Summary Report](../../slipbox/test/circular_import_fix_summary.md)** - Complete fix implementation and results
- 📄 **[Latest Test Output](../../slipbox/test/circular_import_test_output_20250806_202223.txt)** - Most recent test execution results

This directory contains a comprehensive test suite to detect circular imports in the cursus package.

## Overview

Circular imports occur when two or more modules depend on each other directly or indirectly, creating a cycle in the import dependency graph. This can lead to:

- Import failures at runtime
- Unexpected behavior during module initialization
- Difficult-to-debug issues in complex applications

## Files

- `test_circular_imports.py` - Main test suite with comprehensive circular import detection
- `run_circular_import_test.py` - Simple runner script for easy execution
- `README.md` - This documentation file
- `__init__.py` - Package initialization file

## How to Run

### Method 1: Using the runner script (Recommended)

```bash
# From the project root directory
python test/circular_imports/run_circular_import_test.py
```

### Method 2: Running the test module directly

```bash
# From the project root directory
python -m test.circular_imports.test_circular_imports
```

### Method 3: Using unittest

```bash
# From the project root directory
python -m unittest test.circular_imports.test_circular_imports.TestCircularImports
```

## What the Test Does

The circular import test performs several checks:

### 1. Comprehensive Package Scan
- Discovers all Python modules in the `src/cursus` package
- Attempts to import each module systematically
- Tracks import chains to detect circular dependencies
- Reports detailed statistics on import success/failure

### 2. Core Module Testing
Tests critical core modules individually:
- `src.cursus.core.base.config_base`
- `src.cursus.core.base.builder_base`
- `src.cursus.core.base.specification_base`
- `src.cursus.core.base.contract_base`
- `src.cursus.core.base.hyperparameters_base`
- `src.cursus.core.base.enums`

### 3. API Module Testing
Tests API modules for circular dependencies:
- `src.cursus.api.dag.base_dag`
- `src.cursus.api.dag.edge_types`
- `src.cursus.api.dag.enhanced_dag`

### 4. Step Module Testing
Tests step-related modules:
- `src.cursus.steps.registry.builder_registry`
- `src.cursus.steps.registry.hyperparameter_registry`
- `src.cursus.steps.registry.step_names`

### 5. Import Order Independence
Tests that modules can be imported in different orders without issues, which helps detect subtle circular dependency problems.

## Test Output

The test provides detailed output including:

- **Total modules discovered**: Number of Python modules found in the package
- **Successful imports**: Modules that imported without issues
- **Failed imports**: Modules that failed to import (with error details)
- **Circular imports detected**: Specific circular import chains found
- **Import failure details**: Detailed error messages for failed imports
- **Circular import chains**: Visual representation of circular dependency paths

### Output Files

When you run the tests, they will generate detailed output files:

- **`slipbox/test/circular_import_test_output_YYYYMMDD_HHMMSS.txt`** - Complete test results with timestamps saved to the slipbox/test folder
- **Console output** - Real-time test progress and results displayed in the terminal

The output file includes:
- Complete test execution log
- Detailed error messages and stack traces
- Summary statistics
- Timestamp information for tracking test runs over time

### Example Output

```
================================================================================
TESTING CURSUS PACKAGE FOR CIRCULAR IMPORTS
================================================================================

Total modules discovered: 45
Successful imports: 42
Failed imports: 3
Circular imports detected: 0

FAILED IMPORTS (3):
  - src.cursus.steps.builders.builder_model_step_pytorch: Import error in src.cursus.steps.builders.builder_model_step_pytorch: No module named 'torch'
  - src.cursus.processing.bert_tokenize_processor: Import error in src.cursus.processing.bert_tokenize_processor: No module named 'transformers'
  - src.cursus.processing.gensim_tokenize_processor: Import error in src.cursus.processing.gensim_tokenize_processor: No module named 'gensim'

SUCCESSFUL IMPORTS (42):
  - src.cursus.api.dag.base_dag
  - src.cursus.api.dag.edge_types
  - src.cursus.api.dag.enhanced_dag
  - src.cursus.core.base.builder_base
  - src.cursus.core.base.config_base
  ... and 37 more
================================================================================
```

## Understanding Results

### ✅ Success Cases
- **No circular imports detected**: The package is free of circular import issues
- **Some import failures due to missing dependencies**: Normal and expected (e.g., optional dependencies like PyTorch, transformers)

### ❌ Failure Cases
- **Circular imports detected**: Shows the exact import chain causing the circular dependency
- **High failure rate**: More than 50% of modules failing to import might indicate systemic issues

## Integration with CI/CD

You can integrate this test into your CI/CD pipeline:

```yaml
# Example GitHub Actions step
- name: Test for circular imports
  run: python test/circular_imports/run_circular_import_test.py
```

The test will exit with code 0 on success and code 1 on failure, making it suitable for automated testing.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Some modules may fail to import due to missing optional dependencies (PyTorch, transformers, etc.). This is normal and doesn't indicate circular imports.

2. **Path Issues**: Ensure you're running the test from the project root directory.

3. **Module Not Found**: If the test can't find the cursus package, check that the `src/cursus` directory exists and contains the package files.

### Fixing Circular Imports

If circular imports are detected:

1. **Analyze the chain**: Look at the circular import chain to understand the dependency cycle
2. **Refactor imports**: Move imports inside functions, use late imports, or restructure the code
3. **Extract common code**: Move shared functionality to a separate module
4. **Use dependency injection**: Pass dependencies as parameters instead of importing them

## Advanced Usage

### Running Specific Tests

```bash
# Test only core modules
python -m unittest test.circular_imports.test_circular_imports.TestCircularImports.test_core_modules_import_successfully

# Test only API modules
python -m unittest test.circular_imports.test_circular_imports.TestCircularImports.test_api_modules_import_successfully

# Test import order independence
python -m unittest test.circular_imports.test_circular_imports.TestCircularImports.test_import_order_independence
```

### Customizing the Test

You can modify the `CircularImportDetector` class in `test_circular_imports.py` to:
- Change the maximum recursion depth
- Add custom module filtering
- Modify the circular import detection logic
- Add additional test cases

## Related Tests

This circular import test complements other existing tests in the package:
- `test/config_field/test_circular_reference_tracker.py` - Tests circular references in configuration objects
- `test/base/test_all_base.py` - Comprehensive tests for base classes

## Maintenance

- Run this test regularly, especially after adding new modules or changing import structures
- Update the test if the package structure changes significantly
- Consider adding new test cases for specific import patterns in your codebase
