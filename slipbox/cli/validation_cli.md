---
tags:
  - code
  - cli
  - validation
  - naming_standards
  - interface_compliance
keywords:
  - naming validation
  - interface validation
  - standardization rules
  - CLI validation tools
  - registry validation
  - compliance checking
topics:
  - naming standards
  - interface compliance
  - validation automation
  - standardization enforcement
language: python
date of note: 2024-12-07
---

# Validation CLI

Command-line tools for validating naming conventions and interface compliance according to standardization rules, serving as an automated enforcement mechanism for coding standards across the codebase.

## Overview

The Validation CLI provides command-line tools for validating naming conventions and interface compliance according to the standardization rules defined in the developer guide. It serves as an automated enforcement mechanism for coding standards across the codebase with comprehensive registry validation, file naming validation, step naming validation, logical naming validation, and interface compliance validation.

The CLI integrates with NamingStandardValidator for rule enforcement and violation detection, InterfaceStandardValidator for compliance checking and method validation, and the registry system for cross-reference validation and bulk operations.

## Commands and Functions

### registry

cursus validation registry [_options_]

Validate all registry entries for naming consistency and compliance.

**Options:**
- **--verbose** (_bool_) – Enable detailed violation reporting and diagnostic information

**Returns:**
- **int** – Exit code (0 for success, 1 for validation failures)

```bash
# Validate all registry entries
cursus validation registry

# Verbose output with detailed diagnostics
cursus validation registry --verbose
```

### file

cursus validation file _filename_ _file_type_ [_options_]

Validate specific file names against established naming conventions.

**Parameters:**
- **filename** (_str_) – Name of the file to validate
- **file_type** (_str_) – Type of file (builder, config, spec, contract)

**Options:**
- **--verbose** (_bool_) – Enable detailed violation reporting

**Returns:**
- **int** – Exit code (0 for success, 1 for validation failures)

```bash
# Validate builder file naming
cursus validation file builder_xgboost_training_step.py builder

# Validate config file naming
cursus validation file training_config.py config

# Validate spec file naming
cursus validation file training_spec.py spec

# Validate contract file naming
cursus validation file training_contract.py contract --verbose
```

### step

cursus validation step _step_name_ [_options_]

Validate canonical step names for PascalCase format and naming patterns.

**Parameters:**
- **step_name** (_str_) – Canonical step name to validate

**Options:**
- **--verbose** (_bool_) – Enable detailed violation reporting

**Returns:**
- **int** – Exit code (0 for success, 1 for validation failures)

```bash
# Validate canonical step names
cursus validation step XGBoostTraining

# Validate with detailed output
cursus validation step TabularPreprocessing --verbose
```

### logical

cursus validation logical _logical_name_ [_options_]

Validate logical names for snake_case format and descriptive clarity.

**Parameters:**
- **logical_name** (_str_) – Logical name to validate

**Options:**
- **--verbose** (_bool_) – Enable detailed violation reporting

**Returns:**
- **int** – Exit code (0 for success, 1 for validation failures)

```bash
# Validate logical names
cursus validation logical input_data

# Validate with detailed diagnostics
cursus validation logical processed_output --verbose
```

### interface

cursus validation interface _class_path_ [_options_]

Validate step builder interface compliance including method presence and signatures.

**Parameters:**
- **class_path** (_str_) – Full class path to validate (e.g., src.cursus.steps.builders.MyBuilder)

**Options:**
- **--verbose** (_bool_) – Enable detailed violation reporting and method analysis

**Returns:**
- **int** – Exit code (0 for success, 1 for validation failures)

```bash
# Validate interface compliance
cursus validation interface src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder

# Verbose interface validation
cursus validation interface src.cursus.steps.builders.MyBuilder --verbose
```

## Validation Scope

### Registry Validation

Registry validation performs comprehensive validation of all registered builders:

- **Builder Names** – Validates canonical step names in registry
- **Class Mappings** – Ensures registry-to-class name consistency
- **File Naming** – Validates builder file naming conventions
- **Documentation** – Checks registry entry documentation standards
- **Cross-Reference Validation** – Validates registry-to-implementation alignment
- **Bulk Validation** – Processes all entries in single operation

### File Name Validation

File name validation applies type-specific rules for different file types:

- **builder** – Step builder implementation files with pattern validation
- **config** – Configuration class files with convention enforcement
- **spec** – Specification definition files with extension validation
- **contract** – Script contract files with pattern matching

### Step Name Validation

Step name validation ensures canonical format compliance:

- **PascalCase Format** – Must use PascalCase convention
- **Descriptive Naming** – Must be descriptive and clear
- **No Abbreviations** – Avoids unclear abbreviations
- **Consistent Terminology** – Uses established terminology
- **Length Validation** – Validates appropriate name length
- **Reserved Words** – Checks against reserved word usage

### Logical Name Validation

Logical name validation ensures snake_case format compliance:

- **snake_case Format** – Validates snake_case convention
- **Descriptive Clarity** – Ensures clear, descriptive names
- **Consistency Checking** – Validates against established patterns
- **Reserved Name Checking** – Avoids system reserved names

### Interface Compliance Validation

Interface compliance validation checks step builder implementations:

- **Required Methods** – Validates presence of required methods (validate_configuration, create_step, etc.)
- **Method Signatures** – Checks method signatures and parameter types
- **Type Annotations** – Validates complete type hint usage
- **Documentation Standards** – Ensures proper docstring format and coverage
- **Inheritance Validation** – Validates proper base class usage
- **Error Handling** – Checks proper exception handling patterns

## Validation Engine Integration

### NamingStandardValidator Integration

The CLI integrates with NamingStandardValidator for:

- **Rule Enforcement** – Applies standardization rules consistently
- **Violation Detection** – Identifies naming violations with context
- **Suggestion Generation** – Provides improvement suggestions
- **Registry Access** – Accesses builder registry for validation

### InterfaceStandardValidator Integration

The CLI integrates with InterfaceStandardValidator for:

- **Compliance Checking** – Uses InterfaceStandardValidator class
- **Method Validation** – Validates required method presence
- **Signature Checking** – Validates method signatures and types
- **Documentation Validation** – Checks docstring compliance

### Dynamic Class Loading

The CLI supports dynamic class loading for interface validation:

- **Module Importing** – Dynamic module importing with error handling
- **Class Attribute Access** – Safe class attribute access and inspection
- **Interface Validation Execution** – Comprehensive interface validation
- **Result Processing** – Detailed result processing and reporting

## Output Format

### Standard Output Format

The CLI provides formatted violation reporting:

```
🔍 Validating registry entries...
❌ Found 3 naming violations:

📁 XGBoostTraining:
  • File name should follow pattern: builder_<canonical_name>_step.py
    💡 Suggestions: builder_xgboost_training_step.py

📁 TabularPreprocessing:
  • Logical name 'inputData' should use snake_case
    💡 Suggestions: input_data

✅ All interface compliance checks passed!
```

### Verbose Output Format

Verbose mode provides detailed diagnostic information:

```
🔍 Validating interface compliance for: XGBoostTrainingStepBuilder
❌ Found 2 interface violations:

📁 XGBoostTrainingStepBuilder:
  • [missing_method] Missing required method: validate_configuration
    📋 Expected: Method with signature validate_configuration(self) -> None
    📋 Actual: Method not found
    💡 Suggestions: Implement validate_configuration method

  • [type_hints] Method create_step missing return type annotation
    📋 Expected: -> Step
    📋 Actual: No return type annotation
    💡 Suggestions: Add return type annotation: -> Step
```

## Error Handling

### Exit Codes

- **0** – Success, all validations passed
- **1** – Validation failures detected
- **2** – Invalid arguments or usage
- **3** – Import or class loading error
- **4** – File or resource not found

### Error Categories

- **Import Error Management** – Graceful handling of missing imports with clear error messages
- **Module Resolution** – Clear error messages for import failures and path resolution
- **Class Loading** – Detailed error reporting for class loading issues
- **Validation Error Reporting** – Specific violations with context information and improvement suggestions

## Integration Points

### Alignment Validation System Integration

The validation CLI works in conjunction with the comprehensive alignment validation system:

#### Level 1: Script ↔ Contract Alignment
- Validates script arguments match contract specifications
- Ensures all required paths are properly declared
- Checks for unused or undeclared arguments

#### Level 2: Contract ↔ Specification Alignment
- Verifies contract fields align with step specifications
- Validates field types and constraints
- Ensures all required fields are present

#### Level 3: Specification ↔ Dependencies Alignment
- Checks dependency resolution and compatibility
- Validates specification requirements
- Ensures all dependencies are properly declared

#### Level 4: Builder ↔ Configuration Alignment
- Validates step builder configuration
- Ensures proper field mapping and resolution
- Checks for configuration consistency

### Registry System Integration

The CLI integrates with the registry system for:

- **Registry Access** – Accesses builder registry for validation
- **Cross-Reference Validation** – Validates registry-implementation alignment
- **Bulk Operations** – Processes all registry entries efficiently

## Usage Patterns

### Development Workflow

1. **Pre-commit Validation** – Validate changes before committing
2. **File Creation** – Validate new file names before creation
3. **Interface Implementation** – Validate interface compliance during development
4. **Registry Updates** – Validate registry entries after modifications

### Continuous Integration

```bash
# Validate all naming standards
cursus validation registry || exit 1

# Validate specific components
cursus validation interface <builder_class> || exit 1
```

### Code Review Process

```bash
# Validate new builder implementation
cursus validation interface <new_builder> --verbose
cursus validation file <new_file> builder
```

## Performance Considerations

### Validation Optimization

- **Lazy Loading** – Load validators only when needed
- **Caching** – Cache validation results for repeated operations
- **Parallel Processing** – Process multiple validations concurrently
- **Incremental Validation** – Validate only changed components

### Memory Management

- **Resource Cleanup** – Proper cleanup of loaded classes
- **Memory Monitoring** – Track memory usage during validation
- **Batch Processing** – Process large sets in batches
- **Garbage Collection** – Explicit garbage collection for large operations

## Related Documentation

- [Naming Standard Validator](../validation/naming_standard_validator.md) - Core naming validation logic
- [Interface Standard Validator](../validation/interface_standard_validator.md) - Interface compliance validation
- [Registry System](../registry/README.md) - Registry management and validation
- [Alignment Validation](../validation/alignment_validation.md) - Comprehensive alignment validation system
- [Standardization Rules](../../0_developer_guide/standardization_rules.md) - Coding standards and rules
