---
tags:
  - code
  - validation
  - interface
  - standards
  - compliance
keywords:
  - InterfaceStandardValidator
  - InterfaceViolation
  - interface compliance
  - method validation
  - inheritance validation
  - signature validation
topics:
  - interface validation
  - compliance checking
  - standardization rules
language: python
date of note: 2025-09-07
---

# Interface Standard Validator

Validator for interface compliance and standardization rules in the Cursus pipeline framework.

## Overview

The Interface Standard Validator enforces interface patterns defined in the standardization rules document, including required method implementation, method signature compliance, inheritance compliance, and interface documentation standards.

This validator ensures that step builders implement required interfaces correctly and follow established patterns for consistency across the framework. It provides detailed violation reporting with suggestions for remediation.

## Classes and Methods

### Classes
- [`InterfaceStandardValidator`](#interfacestandardvalidator) - Main validator for interface compliance and standardization rules
- [`InterfaceViolation`](#interfaceviolation) - Represents an interface compliance violation

## API Reference

### InterfaceStandardValidator

_class_ cursus.validation.interface.interface_standard_validator.InterfaceStandardValidator()

Validator for interface compliance and standardization rules that enforces interface patterns for step builders.

```python
from cursus.validation.interface.interface_standard_validator import InterfaceStandardValidator

validator = InterfaceStandardValidator()
```

#### validate_step_builder_interface

validate_step_builder_interface(_builder_class_)

Validate complete interface compliance for a step builder class including inheritance, required methods, method signatures, and documentation.

**Parameters:**
- **builder_class** (_Type[StepBuilderBase]_) – Step builder class to validate

**Returns:**
- **List[InterfaceViolation]** – List of interface violations found

```python
from cursus.steps.builders.processing_step_builder import ProcessingStepBuilder

violations = validator.validate_step_builder_interface(ProcessingStepBuilder)
for violation in violations:
    print(f"Violation: {violation}")
```

#### validate_inheritance_compliance

validate_inheritance_compliance(_builder_class_)

Validate that the builder class properly inherits from StepBuilderBase and has correct method resolution order.

**Parameters:**
- **builder_class** (_Type[StepBuilderBase]_) – Step builder class to validate

**Returns:**
- **List[InterfaceViolation]** – List of inheritance violations found

```python
inheritance_violations = validator.validate_inheritance_compliance(ProcessingStepBuilder)
```

#### validate_required_methods

validate_required_methods(_builder_class_)

Validate that all required methods are implemented including validate_configuration, _get_inputs, _get_outputs, and create_step.

**Parameters:**
- **builder_class** (_Type[StepBuilderBase]_) – Step builder class to validate

**Returns:**
- **List[InterfaceViolation]** – List of method implementation violations found

```python
method_violations = validator.validate_required_methods(ProcessingStepBuilder)
```

#### validate_method_signatures

validate_method_signatures(_builder_class_)

Validate method signatures match expected patterns including parameter names, types, and return annotations.

**Parameters:**
- **builder_class** (_Type[StepBuilderBase]_) – Step builder class to validate

**Returns:**
- **List[InterfaceViolation]** – List of method signature violations found

```python
signature_violations = validator.validate_method_signatures(ProcessingStepBuilder)
```

#### validate_method_documentation

validate_method_documentation(_builder_class_)

Validate that required methods have proper documentation including docstrings, parameter documentation, and return value documentation.

**Parameters:**
- **builder_class** (_Type[StepBuilderBase]_) – Step builder class to validate

**Returns:**
- **List[InterfaceViolation]** – List of method documentation violations found

```python
doc_violations = validator.validate_method_documentation(ProcessingStepBuilder)
```

#### validate_class_documentation

validate_class_documentation(_builder_class_)

Validate that the class has proper documentation including class docstring, purpose description, and usage examples.

**Parameters:**
- **builder_class** (_Type[StepBuilderBase]_) – Step builder class to validate

**Returns:**
- **List[InterfaceViolation]** – List of class documentation violations found

```python
class_doc_violations = validator.validate_class_documentation(ProcessingStepBuilder)
```

#### validate_builder_registry_compliance

validate_builder_registry_compliance(_builder_class_)

Validate that the builder class is properly registered or registrable with correct naming conventions.

**Parameters:**
- **builder_class** (_Type[StepBuilderBase]_) – Step builder class to validate

**Returns:**
- **List[InterfaceViolation]** – List of registry compliance violations found

```python
registry_violations = validator.validate_builder_registry_compliance(ProcessingStepBuilder)
```

#### get_all_violations

get_all_violations()

Get all accumulated violations from previous validation runs.

**Returns:**
- **List[InterfaceViolation]** – List of all accumulated violations

```python
all_violations = validator.get_all_violations()
```

#### clear_violations

clear_violations()

Clear all accumulated violations to start fresh validation.

```python
validator.clear_violations()
```

### InterfaceViolation

_class_ cursus.validation.interface.interface_standard_validator.InterfaceViolation(_component_, _violation_type_, _message_, _expected=None_, _actual=None_, _suggestions=None_)

Represents an interface compliance violation with detailed information for remediation.

**Parameters:**
- **component** (_str_) – Component where violation occurred (class or method name)
- **violation_type** (_str_) – Type of violation (e.g., 'method_missing', 'signature_mismatch')
- **message** (_str_) – Human-readable description of the violation
- **expected** (_Optional[str]_) – Expected value or pattern
- **actual** (_Optional[str]_) – Actual value found
- **suggestions** (_Optional[List[str]]_) – List of suggestions for fixing the violation

```python
violation = InterfaceViolation(
    component="Method 'create_step' in ProcessingStepBuilder",
    violation_type="method_missing",
    message="Required method 'create_step' is not implemented",
    expected="def create_step(self, **kwargs) -> SageMakerStep",
    suggestions=["Implement create_step method", "Method should create the SageMaker step"]
)
print(str(violation))
```

## Related Documentation

- [Universal Test](../builders/universal_test.md) - Step builder testing framework
- [Standardization Rules](../../0_developer_guide/standardization_rules.md) - Interface standardization requirements
- [Builder Base](../../core/base/builder_base.md) - Base class for step builders
