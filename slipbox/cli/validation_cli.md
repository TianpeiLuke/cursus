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
date of note: 2025-08-07
---

# Validation CLI Documentation

## Overview

The Validation CLI (`src/cursus/cli/validation_cli.py`) provides command-line tools for validating naming conventions and interface compliance according to the standardization rules defined in the developer guide. It serves as an automated enforcement mechanism for coding standards across the codebase.

## Architecture

### Command Structure

The CLI supports multiple validation modes:

```
validation_cli
├── registry                         # Validate all registry entries
├── file <filename> <type>          # Validate specific file names
├── step <step_name>                # Validate canonical step names
├── logical <logical_name>          # Validate logical names
└── interface <class_path>          # Validate interface compliance
```

### Core Components

#### 1. Naming Standard Validation
- **Registry validation**: Comprehensive registry entry validation
- **File naming**: Validates file naming conventions
- **Step naming**: Ensures canonical step name compliance
- **Logical naming**: Validates logical name standards

#### 2. Interface Standard Validation
- **Class compliance**: Validates step builder interface compliance
- **Method validation**: Ensures required method implementations
- **Type checking**: Validates type annotations and signatures
- **Documentation standards**: Checks docstring compliance

#### 3. Result Reporting System
- **Violation grouping**: Organizes violations by component
- **Detailed reporting**: Provides specific violation details
- **Suggestion system**: Offers improvement recommendations
- **Verbose mode**: Extended diagnostic information

## Key Features

### 1. Registry Validation
```bash
cursus validate registry
```
- **Complete registry scan**: Validates all registered builders
- **Naming consistency**: Ensures consistent naming across registry
- **Cross-reference validation**: Validates registry-to-implementation alignment
- **Bulk validation**: Processes all entries in single operation

#### Validation Scope
- **Builder names**: Canonical step names in registry
- **Class mappings**: Registry-to-class name consistency
- **File naming**: Builder file naming conventions
- **Documentation**: Registry entry documentation standards

### 2. File Name Validation
```bash
cursus validate file builder_xgboost_training_step.py builder
cursus validate file training_config.py config
cursus validate file training_spec.py spec
cursus validate file training_contract.py contract
```
- **Type-specific rules**: Different rules for different file types
- **Pattern matching**: Validates against established patterns
- **Convention enforcement**: Ensures consistent naming conventions
- **Extension validation**: Validates file extensions

#### Supported File Types
- **`builder`**: Step builder implementation files
- **`config`**: Configuration class files
- **`spec`**: Specification definition files
- **`contract`**: Script contract files

### 3. Step Name Validation
```bash
cursus validate step XGBoostTraining
cursus validate step TabularPreprocessing
```
- **Canonical format**: Validates PascalCase canonical names
- **Naming patterns**: Ensures consistent naming patterns
- **Reserved words**: Checks against reserved word usage
- **Length validation**: Validates appropriate name length

#### Validation Rules
- **PascalCase format**: Must use PascalCase convention
- **Descriptive naming**: Must be descriptive and clear
- **No abbreviations**: Avoids unclear abbreviations
- **Consistent terminology**: Uses established terminology

### 4. Logical Name Validation
```bash
cursus validate logical input_data
cursus validate logical processed_output
```
- **snake_case format**: Validates snake_case convention
- **Descriptive clarity**: Ensures clear, descriptive names
- **Consistency checking**: Validates against established patterns
- **Reserved name checking**: Avoids system reserved names

### 5. Interface Compliance Validation
```bash
cursus validate interface src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder
```
- **Method presence**: Validates required method implementations
- **Signature compliance**: Checks method signatures
- **Type annotations**: Validates type hint usage
- **Documentation standards**: Ensures proper docstring format
- **Inheritance validation**: Validates proper base class usage

#### Interface Requirements
- **Required methods**: `validate_configuration`, `create_step`, etc.
- **Method signatures**: Proper parameter types and return types
- **Type hints**: Complete type annotation coverage
- **Documentation**: Comprehensive docstring documentation
- **Error handling**: Proper exception handling patterns

## Implementation Details

### Validation Engine Integration
```python
def validate_registry(verbose: bool = False) -> int:
    """Validate all registry entries."""
    validator = NamingStandardValidator()
    violations = validator.validate_all_registry_entries()
    # Process and report violations
```

### Dynamic Class Loading
```python
def validate_interface(class_path: str, verbose: bool = False) -> int:
    """Validate step builder interface compliance."""
    # Dynamic module importing
    # Class attribute access
    # Interface validation execution
    # Result processing and reporting
```

### Violation Reporting System
```python
def print_violations(violations: List, verbose: bool = False) -> None:
    """Print violations in a formatted way."""
    # Violation type detection
    # Component-based grouping
    # Formatted output generation
    # Suggestion inclusion
```

## Usage Patterns

### Development Workflow
1. **Pre-commit validation**: Validate changes before committing
2. **File creation**: Validate new file names before creation
3. **Interface implementation**: Validate interface compliance during development
4. **Registry updates**: Validate registry entries after modifications

### Continuous Integration
```bash
# Validate all naming standards
python -m src.cursus.cli.validation_cli registry || exit 1

# Validate specific components
python -m src.cursus.cli.validation_cli interface <builder_class> || exit 1
```

### Code Review Process
```bash
# Validate new builder implementation
python -m src.cursus.cli.validation_cli interface <new_builder> --verbose
python -m src.cursus.cli.validation_cli file <new_file> builder
```

## Alignment Validation System Integration

### Comprehensive Script Alignment Validation
The validation CLI works in conjunction with the comprehensive alignment validation system located in `test/steps/scripts/alignment_validation/`. This system provides 4-level validation:

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

### Running Comprehensive Alignment Validation
```bash
# Run comprehensive validation for all scripts
cd test/steps/scripts/alignment_validation
python run_alignment_validation.py

# Run validation for individual scripts
python validate_currency_conversion.py
python validate_dummy_training.py
python validate_xgboost_training.py
```

### Generated Reports
The alignment validation system generates comprehensive reports:
- **JSON Reports**: Machine-readable format in `reports/json/`
- **HTML Reports**: Human-readable format in `reports/html/`
- **Summary Report**: Overall validation summary in `reports/validation_summary.json`

## Error Handling

### Import Error Management
- **Missing dependencies**: Graceful handling of missing imports
- **Module resolution**: Clear error messages for import failures
- **Class loading**: Detailed error reporting for class loading issues
- **Path resolution**: Helpful guidance for path-related errors

### Validation Error Reporting
- **Specific violations**: Detailed violation descriptions
- **Context information**: Relevant context for each violation
- **Improvement suggestions**: Actionable recommendations
- **Severity levels**: Different levels of violation severity

## Output Format

### Standard Output
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

### Verbose Output
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

## Integration Points

### 1. Naming Standard Validator
- **Direct integration**: Uses NamingStandardValidator class
- **Rule enforcement**: Applies standardization rules
- **Violation detection**: Identifies naming violations
- **Suggestion generation**: Provides improvement suggestions

### 2. Interface Standard Validator
- **Compliance checking**: Uses InterfaceStandardValidator class
- **Method validation**: Validates required method presence
- **Signature checking**: Validates method signatures
- **Documentation validation**: Checks docstring compliance

### 3. Registry System
- **Registry access**: Accesses builder registry for validation
- **Cross-reference validation**: Validates registry-implementation alignment
- **Bulk operations**: Processes all registry entries efficiently

## Extension Points

### Adding New File Types
```python
# Extend file type choices
file_parser.add_argument(
    "file_type",
    choices=["builder", "config", "spec", "contract", "new_type"],
    help="Type of file"
)
```

### Custom Validation Rules
- **Rule plugins**: Support for custom validation rules
- **Configuration-driven**: Rules defined in configuration files
- **Domain-specific**: Rules specific to particular domains
- **Extensible framework**: Easy addition of new validation types

### Output Formatters
- **JSON output**: Machine-readable output format
- **XML reports**: Structured report generation
- **Integration hooks**: Hooks for CI/CD integration
- **Custom formatters**: User-defined output formats

## Performance Considerations

### Validation Optimization
- **Lazy loading**: Load validators only when needed
- **Caching**: Cache validation results for repeated operations
- **Parallel processing**: Process multiple validations concurrently
- **Incremental validation**: Validate only changed components

### Memory Management
- **Resource cleanup**: Proper cleanup of loaded classes
- **Memory monitoring**: Track memory usage during validation
- **Batch processing**: Process large sets in batches
- **Garbage collection**: Explicit garbage collection for large operations

## Best Practices

### Validation Strategy
- **Early validation**: Validate early in development process
- **Comprehensive coverage**: Validate all relevant components
- **Consistent application**: Apply validation consistently
- **Automated enforcement**: Integrate into automated workflows

### Error Handling
- **Clear messages**: Provide clear, actionable error messages
- **Context preservation**: Maintain error context
- **Recovery guidance**: Offer recovery suggestions
- **Documentation links**: Link to relevant documentation

## Future Enhancements

### Planned Features
- **Configuration validation**: Validate configuration file formats
- **Cross-component validation**: Validate relationships between components
- **Custom rule definition**: User-defined validation rules
- **Integration testing**: Validate component integration

### Automation Improvements
- **Auto-fixing**: Automatic fixing of simple violations
- **Batch operations**: Bulk validation and fixing operations
- **Watch mode**: Continuous validation during development
- **IDE integration**: Integration with development environments

This validation CLI provides comprehensive automated enforcement of naming conventions and interface compliance standards, ensuring consistent code quality and adherence to established patterns across the entire codebase.
