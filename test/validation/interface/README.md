# Interface Validation Tests

This directory contains comprehensive tests for the Interface Standard Validator, which validates step builder interface compliance according to the standardization rules.

## Test Structure

The tests have been split into multiple focused test files, each containing one TestCase class:

### `test_interface_violation.py`
- **TestCase:** `TestInterfaceViolation`
- **Purpose:** Tests the `InterfaceViolation` data structure
- **Coverage:** 4 tests
- **Focus:** Violation creation, string representation, default values

### `test_validator_core.py`
- **TestCase:** `TestInterfaceStandardValidator`
- **Purpose:** Tests the main validator functionality
- **Coverage:** 17 tests
- **Focus:** 
  - Validator initialization
  - Individual validation methods (inheritance, methods, signatures, documentation, registry)
  - Violation accumulation and management
  - Edge cases (None class, good/bad builders)

### `test_validator_integration.py`
- **TestCase:** `TestInterfaceValidatorIntegration`
- **Purpose:** Integration tests for complete validation workflows
- **Coverage:** 3 tests
- **Focus:**
  - End-to-end validation workflows
  - Integration with real step builders from the codebase
  - Complete violation reporting

## Mock Classes

Each test file includes the necessary mock classes:

- **`MockGoodStepBuilder`:** Compliant step builder for positive testing
- **`MockBadStepBuilder`:** Non-compliant step builder for negative testing
- **`MockUndocumentedStepBuilder`:** Step builder with documentation issues

## Running Tests

### Individual Test Files
```bash
# Test violation data structure
python -m unittest test.validation.interface.test_interface_violation -v

# Test core validator functionality
python -m unittest test.validation.interface.test_validator_core -v

# Test integration scenarios
python -m unittest test.validation.interface.test_validator_integration -v
```

### All Interface Validation Tests
```bash
python -m unittest test.validation.interface.test_interface_violation test.validation.interface.test_validator_core test.validation.interface.test_validator_integration -v
```

## Test Coverage

Total: **24 tests** covering all aspects of interface validation:

- ✅ Interface violation data structure (4 tests)
- ✅ Inheritance compliance validation (2 tests)
- ✅ Required methods validation (2 tests)
- ✅ Method signature validation (2 tests)
- ✅ Documentation validation (4 tests)
- ✅ Registry compliance validation (2 tests)
- ✅ Violation management (2 tests)
- ✅ Edge cases and error handling (3 tests)
- ✅ Integration workflows (3 tests)

## Validation Categories Tested

1. **Inheritance Compliance**
   - Validates inheritance from `StepBuilderBase`
   - Checks method resolution order (MRO)

2. **Required Methods**
   - Ensures all required methods are implemented
   - Validates method callability

3. **Method Signatures**
   - Validates parameter signatures
   - Checks for required parameters and **kwargs

4. **Documentation Standards**
   - Method documentation validation
   - Class documentation validation
   - Return documentation validation

5. **Registry Compliance**
   - Naming convention validation
   - Registry compatibility checks

All tests pass successfully, confirming that the validation tools work correctly and can effectively identify both compliant and non-compliant step builders according to the established standards.
