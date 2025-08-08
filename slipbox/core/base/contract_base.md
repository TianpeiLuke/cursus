---
tags:
  - code
  - core
  - base
  - contracts
  - validation
keywords:
  - script contracts
  - validation framework
  - I/O specification
  - environment variables
  - CLI arguments
topics:
  - contract validation
  - script analysis
  - pipeline compliance
language: python
date of note: 2025-08-07
---

# Base Script Contract Classes

## Overview

The `contract_base.py` module defines the core ScriptContract class and validation framework for pipeline scripts. It provides a comprehensive system for defining, validating, and enforcing contracts between pipeline steps and their underlying script implementations.

## Purpose

This module provides:
- **Script Contracts**: Explicit definitions of script I/O, environment requirements, and CLI arguments
- **Validation Framework**: Tools to validate script implementations against their contracts
- **Script Analysis**: AST-based analysis of Python scripts to extract I/O patterns
- **Alignment Validation**: Verification that contracts align with step specifications

## Core Classes

### ValidationResult

A Pydantic model that represents the result of script contract validation.

```python
class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
```

#### Key Methods

- `success(message)`: Create a successful validation result
- `error(errors)`: Create a failed validation result
- `combine(results)`: Combine multiple validation results
- `add_error(error)`: Add an error and mark as invalid
- `add_warning(warning)`: Add a warning

#### Usage Examples

```python
# Success case
result = ValidationResult.success("All validations passed")

# Error case
result = ValidationResult.error(["Missing required input", "Invalid output path"])

# Combining results
combined = ValidationResult.combine([result1, result2, result3])
```

### AlignmentResult

Extends ValidationResult with specific fields for contract-specification alignment validation.

```python
class AlignmentResult(ValidationResult):
    missing_outputs: List[str] = Field(default_factory=list)
    missing_inputs: List[str] = Field(default_factory=list)
    extra_outputs: List[str] = Field(default_factory=list)
    extra_inputs: List[str] = Field(default_factory=list)
```

This specialized result type provides detailed information about misalignments between contracts and specifications.

### ScriptContract

The main contract class that defines explicit I/O, environment requirements, and CLI arguments for pipeline scripts.

```python
class ScriptContract(BaseModel):
    entry_point: str
    expected_input_paths: Dict[str, str]
    expected_output_paths: Dict[str, str]
    required_env_vars: List[str]
    optional_env_vars: Dict[str, str]
    expected_arguments: Dict[str, str]
    framework_requirements: Dict[str, str]
    description: str = ""
```

#### Field Descriptions

- **entry_point**: Script entry point filename (must be .py file)
- **expected_input_paths**: Mapping of logical names to expected input paths
- **expected_output_paths**: Mapping of logical names to expected output paths
- **required_env_vars**: List of required environment variables
- **optional_env_vars**: Optional environment variables with defaults
- **expected_arguments**: Mapping of argument names to container paths or values
- **framework_requirements**: Framework version requirements
- **description**: Human-readable description of the script

#### Validation Rules

1. **Entry Point**: Must be a Python file (.py extension)
2. **Input Paths**: Must start with `/opt/ml/processing/input` (except GeneratedPayloadSamples)
3. **Output Paths**: Must start with `/opt/ml/processing/output`
4. **Arguments**: Must follow CLI conventions (kebab-case, lowercase)

#### Usage Examples

```python
contract = ScriptContract(
    entry_point="train.py",
    expected_input_paths={
        "training_data": "/opt/ml/processing/input/train",
        "validation_data": "/opt/ml/processing/input/val"
    },
    expected_output_paths={
        "model_artifacts": "/opt/ml/processing/output/model"
    },
    required_env_vars=["MODEL_TYPE", "LEARNING_RATE"],
    optional_env_vars={"BATCH_SIZE": "32"},
    expected_arguments={
        "epochs": "10",
        "model-name": "my-model"
    },
    description="Training script for XGBoost model"
)
```

### ScriptAnalyzer

AST-based analyzer that extracts I/O patterns and environment variable usage from Python scripts.

```python
class ScriptAnalyzer:
    def __init__(self, script_path: str)
    def get_input_paths(self) -> Set[str]
    def get_output_paths(self) -> Set[str]
    def get_env_var_usage(self) -> Set[str]
    def get_argument_usage(self) -> Set[str]
```

#### Analysis Capabilities

1. **Input Path Detection**: Finds SageMaker input paths in string literals
2. **Output Path Detection**: Finds SageMaker output paths in string literals
3. **Environment Variable Usage**: Detects `os.environ`, `os.getenv` usage
4. **Argument Usage**: Finds `argparse.add_argument` calls

#### Implementation Details

The analyzer uses Python's AST module to parse scripts and extract patterns:

```python
# Environment variable detection
for node in ast.walk(self.ast_tree):
    # os.environ["VAR_NAME"]
    if (isinstance(node, ast.Subscript) and
        isinstance(node.value, ast.Attribute) and
        node.value.attr == "environ"):
        # Extract variable name
        
    # os.getenv("VAR_NAME")
    elif (isinstance(node, ast.Call) and
          isinstance(node.func, ast.Attribute) and
          node.func.attr == "getenv"):
        # Extract variable name
```

## Validation Framework

### Contract Implementation Validation

The `validate_implementation` method checks if a script complies with its contract:

```python
def validate_implementation(self, script_path: str) -> ValidationResult:
    """Validate that a script implementation matches this contract"""
    if not os.path.exists(script_path):
        return ValidationResult.error([f"Script file not found: {script_path}"])
    
    analyzer = ScriptAnalyzer(script_path)
    return self._validate_against_analysis(analyzer)
```

### Validation Checks

1. **Input Path Validation**: Ensures script uses all expected input paths
2. **Output Path Validation**: Ensures script uses all expected output paths
3. **Environment Variable Validation**: Checks for required environment variables
4. **Argument Validation**: Verifies expected CLI arguments are handled

### Error and Warning Categories

- **Errors**: Missing required inputs/outputs, missing required environment variables
- **Warnings**: Undeclared input paths, missing argument handling

## Design Patterns

### Contract-First Development

```python
# 1. Define the contract
contract = ScriptContract(
    entry_point="process.py",
    expected_input_paths={"data": "/opt/ml/processing/input/data"},
    expected_output_paths={"result": "/opt/ml/processing/output/result"},
    required_env_vars=["PROCESSING_MODE"]
)

# 2. Validate implementation
result = contract.validate_implementation("scripts/process.py")
if not result.is_valid:
    print(f"Contract violations: {result.errors}")
```

### Integration with Step Specifications

Contracts can be associated with step specifications for alignment validation:

```python
# In step specification
spec = StepSpecification(
    step_type="ProcessingStep",
    dependencies={"data": dep_spec},
    outputs={"result": output_spec},
    script_contract=contract
)

# Validate alignment
alignment_result = spec.validate_contract_alignment()
```

## Best Practices

### Contract Definition

1. **Explicit Paths**: Use full SageMaker paths for clarity
2. **Logical Names**: Use descriptive logical names for inputs/outputs
3. **Environment Variables**: Clearly separate required vs optional
4. **Documentation**: Provide clear descriptions

### Validation Strategy

1. **Early Validation**: Validate contracts during development
2. **Continuous Integration**: Include contract validation in CI/CD
3. **Error Handling**: Provide clear error messages
4. **Incremental Validation**: Validate changes incrementally

### Script Implementation

1. **Path Consistency**: Use exact paths specified in contract
2. **Environment Handling**: Handle all required environment variables
3. **Argument Parsing**: Implement all expected CLI arguments
4. **Error Handling**: Provide meaningful error messages

## Integration Points

### With Step Builders

Step builders use contracts to:
- Generate environment variables
- Construct CLI arguments
- Validate script compliance

### With Specifications

Specifications use contracts for:
- Alignment validation
- I/O path verification
- Dependency validation

### With Pipeline Validation

Pipeline validation uses contracts to:
- Ensure script compliance
- Validate data flow
- Check environment consistency

## Error Handling

The module provides comprehensive error handling:

1. **File Not Found**: Clear messages for missing scripts
2. **Parse Errors**: Graceful handling of malformed scripts
3. **Validation Errors**: Detailed error descriptions
4. **Type Errors**: Proper handling of type mismatches

## Future Extensions

The contract system is designed for extensibility:

1. **Additional Validation Rules**: Easy to add new validation patterns
2. **Language Support**: Framework for supporting non-Python scripts
3. **Advanced Analysis**: More sophisticated AST analysis
4. **Integration Testing**: Contract-based integration testing

This comprehensive contract system ensures reliable, validated pipeline scripts with clear interfaces and explicit requirements.
