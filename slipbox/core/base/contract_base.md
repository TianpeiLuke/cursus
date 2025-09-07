---
tags:
  - code
  - base
  - contract_base
  - script_contracts
  - validation_framework
keywords:
  - ScriptContract
  - ValidationResult
  - AlignmentResult
  - ScriptAnalyzer
  - script validation
  - contract compliance
  - I/O validation
topics:
  - script contracts
  - validation framework
  - contract compliance
language: python
date of note: 2024-12-07
---

# Contract Base

Base Script Contract Classes that define the core ScriptContract class and validation framework for pipeline scripts with comprehensive I/O validation and compliance checking.

## Overview

The contract base module provides a sophisticated framework for defining and validating script execution contracts in pipeline environments. The `ScriptContract` class establishes explicit contracts for script I/O, environment requirements, and CLI arguments, while the validation framework ensures compliance through static analysis and runtime validation.

The system supports advanced features including comprehensive I/O path validation for SageMaker environments, environment variable requirement specification and validation, CLI argument definition and compliance checking, static script analysis using AST parsing, and detailed validation reporting with errors and warnings.

## Classes and Methods

### Classes
- [`ScriptContract`](#scriptcontract) - Script execution contract defining I/O, environment, and argument requirements
- [`ValidationResult`](#validationresult) - Result of script contract validation with error and warning details
- [`AlignmentResult`](#alignmentresult) - Specialized validation result for contract-specification alignment
- [`ScriptAnalyzer`](#scriptanalyzer) - Static analyzer for Python scripts using AST parsing

## API Reference

### ScriptContract

_class_ cursus.core.base.contract_base.ScriptContract(_entry_point_, _expected_input_paths_, _expected_output_paths_, _required_env_vars_, _optional_env_vars={}_, _expected_arguments={}_, _framework_requirements={}_, _description=""_)

Script execution contract that defines explicit I/O, environment requirements, and CLI arguments. This class provides a comprehensive specification for script behavior and requirements in pipeline environments.

**Parameters:**
- **entry_point** (_str_) – Script entry point filename. Must be a Python file (.py).
- **expected_input_paths** (_Dict[str, str]_) – Mapping of logical names to expected input paths in SageMaker format.
- **expected_output_paths** (_Dict[str, str]_) – Mapping of logical names to expected output paths in SageMaker format.
- **required_env_vars** (_List[str]_) – List of required environment variables that must be present.
- **optional_env_vars** (_Dict[str, str]_) – Optional environment variables with default values. Defaults to empty dict.
- **expected_arguments** (_Dict[str, str]_) – Mapping of argument names to container paths or values. Defaults to empty dict.
- **framework_requirements** (_Dict[str, str]_) – Framework version requirements. Defaults to empty dict.
- **description** (_str_) – Human-readable description of the script. Defaults to empty string.

```python
from cursus.core.base.contract_base import ScriptContract

# Create processing script contract
processing_contract = ScriptContract(
    entry_point="preprocessing.py",
    expected_input_paths={
        "raw_data": "/opt/ml/processing/input/data",
        "config": "/opt/ml/processing/input/config"
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output/processed",
        "metrics": "/opt/ml/processing/output/metrics"
    },
    required_env_vars=["PROCESSING_MODE", "OUTPUT_FORMAT"],
    optional_env_vars={
        "DEBUG_MODE": "false",
        "LOG_LEVEL": "INFO"
    },
    expected_arguments={
        "batch-size": "32",
        "num-workers": "4"
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "scikit-learn": ">=1.0.0"
    },
    description="Data preprocessing script for ML pipeline"
)
```

#### validate_implementation

validate_implementation(_script_path_)

Validate that a script implementation matches this contract. This method performs static analysis of the script to ensure compliance with the contract requirements.

**Parameters:**
- **script_path** (_str_) – Path to the script file to validate.

**Returns:**
- **ValidationResult** – Result indicating whether the script complies with the contract, including detailed errors and warnings.

```python
# Validate script implementation
validation_result = processing_contract.validate_implementation("scripts/preprocessing.py")

if validation_result.is_valid:
    print("Script complies with contract")
else:
    print("Contract validation failed:")
    for error in validation_result.errors:
        print(f"  Error: {error}")
    for warning in validation_result.warnings:
        print(f"  Warning: {warning}")
```

### ValidationResult

_class_ cursus.core.base.contract_base.ValidationResult(_is_valid_, _errors=[]_, _warnings=[]_)

Result of script contract validation. This class encapsulates the outcome of validation operations with detailed error and warning information.

**Parameters:**
- **is_valid** (_bool_) – Whether the validation passed successfully.
- **errors** (_List[str]_) – List of validation errors that caused failure. Defaults to empty list.
- **warnings** (_List[str]_) – List of validation warnings that don't cause failure. Defaults to empty list.

```python
from cursus.core.base.contract_base import ValidationResult

# Create validation results
success_result = ValidationResult(is_valid=True)
failure_result = ValidationResult(
    is_valid=False,
    errors=["Missing required input path", "Invalid environment variable"],
    warnings=["Unused output path declared"]
)
```

#### success

_classmethod_ success(_message="Validation passed"_)

Create a successful validation result. This factory method creates a ValidationResult indicating successful validation.

**Parameters:**
- **message** (_str_) – Success message. Defaults to "Validation passed".

**Returns:**
- **ValidationResult** – Successful validation result.

```python
# Create success result
result = ValidationResult.success("All validations passed")
assert result.is_valid == True
```

#### error

_classmethod_ error(_errors_)

Create a failed validation result. This factory method creates a ValidationResult indicating validation failure with specific errors.

**Parameters:**
- **errors** (_Union[str, List[str]]_) – Error message(s) that caused validation failure.

**Returns:**
- **ValidationResult** – Failed validation result with specified errors.

```python
# Create error result
result = ValidationResult.error(["Missing input file", "Invalid configuration"])
assert result.is_valid == False
assert len(result.errors) == 2
```

#### combine

_classmethod_ combine(_results_)

Combine multiple validation results. This method aggregates multiple validation results into a single result.

**Parameters:**
- **results** (_List[ValidationResult]_) – List of validation results to combine.

**Returns:**
- **ValidationResult** – Combined validation result with aggregated errors and warnings.

```python
# Combine multiple validation results
result1 = ValidationResult.error(["Error 1"])
result2 = ValidationResult.success()
result3 = ValidationResult(is_valid=True, warnings=["Warning 1"])

combined = ValidationResult.combine([result1, result2, result3])
assert combined.is_valid == False  # Any error makes combined result invalid
assert len(combined.errors) == 1
assert len(combined.warnings) == 1
```

#### add_error

add_error(_error_)

Add an error to the result and mark as invalid. This method allows dynamic addition of errors to an existing validation result.

**Parameters:**
- **error** (_str_) – Error message to add.

```python
# Add error to existing result
result = ValidationResult.success()
result.add_error("New error discovered")
assert result.is_valid == False
```

#### add_warning

add_warning(_warning_)

Add a warning to the result. This method allows dynamic addition of warnings without affecting validity.

**Parameters:**
- **warning** (_str_) – Warning message to add.

```python
# Add warning to existing result
result = ValidationResult.success()
result.add_warning("Potential issue detected")
assert result.is_valid == True  # Warnings don't affect validity
```

### AlignmentResult

_class_ cursus.core.base.contract_base.AlignmentResult(_is_valid_, _errors=[]_, _warnings=[]_, _missing_outputs=[]_, _missing_inputs=[]_, _extra_outputs=[]_, _extra_inputs=[]_)

Result of contract-specification alignment validation. This specialized validation result provides detailed information about alignment between contracts and specifications, including missing and extra I/O paths.

**Parameters:**
- **is_valid** (_bool_) – Whether the alignment validation passed successfully.
- **errors** (_List[str]_) – List of alignment errors that caused failure. Defaults to empty list.
- **warnings** (_List[str]_) – List of alignment warnings that don't cause failure. Defaults to empty list.
- **missing_outputs** (_List[str]_) – List of output paths missing from the contract. Defaults to empty list.
- **missing_inputs** (_List[str]_) – List of input paths missing from the contract. Defaults to empty list.
- **extra_outputs** (_List[str]_) – List of extra output paths in the contract. Defaults to empty list.
- **extra_inputs** (_List[str]_) – List of extra input paths in the contract. Defaults to empty list.

```python
from cursus.core.base.contract_base import AlignmentResult

# Create alignment result with detailed I/O information
alignment_result = AlignmentResult(
    is_valid=False,
    errors=["Contract-specification mismatch"],
    missing_outputs=["model_artifacts"],
    extra_inputs=["unused_config"],
    warnings=["Optional input not used"]
)
```

#### success

_classmethod_ success(_message="Alignment validation passed"_)

Create a successful alignment result. This factory method creates an AlignmentResult indicating successful alignment validation.

**Parameters:**
- **message** (_str_) – Success message. Defaults to "Alignment validation passed".

**Returns:**
- **AlignmentResult** – Successful alignment result.

```python
# Create successful alignment result
result = AlignmentResult.success("Contract and specification aligned")
assert result.is_valid == True
```

#### error

_classmethod_ error(_errors_, _missing_outputs=None_, _missing_inputs=None_, _extra_outputs=None_, _extra_inputs=None_)

Create a failed alignment result. This factory method creates an AlignmentResult indicating alignment failure with specific errors and I/O details.

**Parameters:**
- **errors** (_Union[str, List[str]]_) – Error message(s) that caused alignment failure.
- **missing_outputs** (_List[str]_) – List of output paths missing from the contract. Defaults to None.
- **missing_inputs** (_List[str]_) – List of input paths missing from the contract. Defaults to None.
- **extra_outputs** (_List[str]_) – List of extra output paths in the contract. Defaults to None.
- **extra_inputs** (_List[str]_) – List of extra input paths in the contract. Defaults to None.

**Returns:**
- **AlignmentResult** – Failed alignment result with specified errors and I/O details.

```python
# Create alignment error with I/O details
result = AlignmentResult.error(
    errors=["Specification mismatch"],
    missing_outputs=["model_metrics"],
    extra_inputs=["deprecated_config"]
)
assert result.is_valid == False
assert "model_metrics" in result.missing_outputs
```

### ScriptAnalyzer

_class_ cursus.core.base.contract_base.ScriptAnalyzer(_script_path_)

Analyzes Python scripts to extract I/O patterns and environment variable usage. This class uses AST parsing to statically analyze script behavior and extract contract-relevant information.

**Parameters:**
- **script_path** (_str_) – Path to the Python script file to analyze.

```python
from cursus.core.base.contract_base import ScriptAnalyzer

# Analyze a script for contract compliance
analyzer = ScriptAnalyzer("scripts/preprocessing.py")

# Extract I/O patterns
input_paths = analyzer.get_input_paths()
output_paths = analyzer.get_output_paths()
env_vars = analyzer.get_env_var_usage()
arguments = analyzer.get_argument_usage()

print(f"Script uses {len(input_paths)} input paths")
print(f"Script uses {len(output_paths)} output paths")
print(f"Script accesses {len(env_vars)} environment variables")
print(f"Script handles {len(arguments)} command-line arguments")
```

#### ast_tree

_property_ ast_tree

Lazy load and parse the script AST. This property provides access to the parsed Abstract Syntax Tree of the script for analysis.

**Returns:**
- **ast.AST** – Parsed AST of the script.

```python
# Access the parsed AST
analyzer = ScriptAnalyzer("script.py")
tree = analyzer.ast_tree
print(f"AST has {len(list(ast.walk(tree)))} nodes")
```

#### get_input_paths

get_input_paths()

Extract input paths used by the script. This method analyzes the script to find SageMaker input path patterns.

**Returns:**
- **Set[str]** – Set of input paths found in the script.

```python
# Extract input paths from script
analyzer = ScriptAnalyzer("preprocessing.py")
input_paths = analyzer.get_input_paths()

for path in input_paths:
    print(f"Found input path: {path}")
```

#### get_output_paths

get_output_paths()

Extract output paths used by the script. This method analyzes the script to find SageMaker output path patterns.

**Returns:**
- **Set[str]** – Set of output paths found in the script.

```python
# Extract output paths from script
analyzer = ScriptAnalyzer("preprocessing.py")
output_paths = analyzer.get_output_paths()

for path in output_paths:
    print(f"Found output path: {path}")
```

#### get_env_var_usage

get_env_var_usage()

Extract environment variables accessed by the script. This method analyzes various patterns of environment variable access including `os.environ`, `os.environ.get()`, and `os.getenv()`.

**Returns:**
- **Set[str]** – Set of environment variable names accessed by the script.

```python
# Extract environment variable usage
analyzer = ScriptAnalyzer("preprocessing.py")
env_vars = analyzer.get_env_var_usage()

for var in env_vars:
    print(f"Script accesses environment variable: {var}")
```

#### get_argument_usage

get_argument_usage()

Extract command-line arguments used by the script. This method analyzes argparse patterns to identify CLI arguments handled by the script.

**Returns:**
- **Set[str]** – Set of argument names (without leading dashes) handled by the script.

```python
# Extract command-line argument usage
analyzer = ScriptAnalyzer("preprocessing.py")
arguments = analyzer.get_argument_usage()

for arg in arguments:
    print(f"Script handles argument: --{arg}")
```

## Usage Examples

### Complete Contract Validation Workflow

```python
from cursus.core.base.contract_base import ScriptContract, ScriptAnalyzer

# Define comprehensive script contract
contract = ScriptContract(
    entry_point="train.py",
    expected_input_paths={
        "training_data": "/opt/ml/processing/input/train",
        "validation_data": "/opt/ml/processing/input/validation",
        "hyperparameters": "/opt/ml/processing/input/config"
    },
    expected_output_paths={
        "model_artifacts": "/opt/ml/processing/output/model",
        "evaluation_metrics": "/opt/ml/processing/output/evaluation",
        "training_logs": "/opt/ml/processing/output/logs"
    },
    required_env_vars=[
        "SM_MODEL_DIR",
        "SM_CHANNEL_TRAINING",
        "SM_CHANNEL_VALIDATION"
    ],
    optional_env_vars={
        "SM_HP_EPOCHS": "10",
        "SM_HP_BATCH_SIZE": "32",
        "DEBUG_MODE": "false"
    },
    expected_arguments={
        "learning-rate": "0.001",
        "optimizer": "adam",
        "model-type": "xgboost"
    },
    framework_requirements={
        "xgboost": ">=1.6.0",
        "scikit-learn": ">=1.0.0",
        "pandas": ">=1.3.0"
    },
    description="Training script for XGBoost model with comprehensive validation"
)

# Validate script implementation
validation_result = contract.validate_implementation("scripts/train.py")

if validation_result.is_valid:
    print("✓ Script implementation complies with contract")
else:
    print("✗ Contract validation failed:")
    for error in validation_result.errors:
        print(f"  Error: {error}")
    
    if validation_result.warnings:
        print("Warnings:")
        for warning in validation_result.warnings:
            print(f"  Warning: {warning}")

# Detailed script analysis
analyzer = ScriptAnalyzer("scripts/train.py")
print(f"\nScript Analysis:")
print(f"Input paths: {analyzer.get_input_paths()}")
print(f"Output paths: {analyzer.get_output_paths()}")
print(f"Environment variables: {analyzer.get_env_var_usage()}")
print(f"CLI arguments: {analyzer.get_argument_usage()}")
```

### Validation Result Combination

```python
from cursus.core.base.contract_base import ValidationResult, AlignmentResult

# Combine multiple validation results
io_validation = ValidationResult.success("I/O validation passed")
env_validation = ValidationResult.error(["Missing required environment variable: SM_MODEL_DIR"])
arg_validation = ValidationResult(
    is_valid=True,
    warnings=["Unused argument: --debug-mode"]
)

# Combine all validation results
combined_result = ValidationResult.combine([
    io_validation,
    env_validation,
    arg_validation
])

print(f"Overall validation: {'PASSED' if combined_result.is_valid else 'FAILED'}")
print(f"Total errors: {len(combined_result.errors)}")
print(f"Total warnings: {len(combined_result.warnings)}")

# Create detailed alignment result
alignment_result = AlignmentResult.error(
    errors=["Contract-specification alignment failed"],
    missing_outputs=["model_metrics", "feature_importance"],
    extra_inputs=["deprecated_config"],
    missing_inputs=["validation_schema"]
)

print(f"\nAlignment Details:")
print(f"Missing outputs: {alignment_result.missing_outputs}")
print(f"Extra inputs: {alignment_result.extra_inputs}")
print(f"Missing inputs: {alignment_result.missing_inputs}")
```

## Related Components

- **[Specification Base](specification_base.md)** - Step specifications that work with script contracts
- **[Builder Base](builder_base.md)** - Step builders that validate against contracts
- **[Config Base](config_base.md)** - Configuration classes that support contract validation
- **[Enums](enums.md)** - Shared enumerations for dependency and node types
