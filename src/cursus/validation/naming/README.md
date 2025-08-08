# Naming Standard Validator

The `NamingStandardValidator` provides comprehensive validation for naming conventions as defined in the standardization rules document. It ensures consistency across all pipeline components and helps maintain code quality standards.

## Features

The validator checks the following naming conventions:

### 1. Canonical Step Names
- Must be in **PascalCase** (e.g., `CradleDataLoading`, `XGBoostTraining`)
- No underscores or special characters
- Should match registry entries

### 2. Config Class Names
- Must be in **PascalCase** with `Config` suffix
- Pattern: `{StepName}Config` (e.g., `XGBoostTrainingConfig`)
- Should align with canonical step names

### 3. Builder Class Names
- Must be in **PascalCase** with `StepBuilder` suffix
- Pattern: `{StepName}StepBuilder` (e.g., `XGBoostTrainingStepBuilder`)
- Should align with canonical step names

### 4. File Naming Patterns
- **Builder files**: `builder_{snake_case}_step.py`
- **Config files**: `config_{snake_case}_step.py`
- **Spec files**: `{snake_case}_spec.py`
- **Contract files**: `{snake_case}_contract.py`

### 5. Logical Names (Input/Output)
- Must be in **snake_case** (e.g., `input_data`, `model_artifacts`)
- No leading/trailing underscores
- No double underscores

### 6. Method and Field Names
- Public methods: **snake_case**
- Configuration fields: **snake_case**

## Usage

### Basic Usage

```python
from src.cursus.validation.naming import NamingStandardValidator

# Create validator instance
validator = NamingStandardValidator()

# Validate step specification
violations = validator.validate_step_specification(your_step_spec)

# Validate step builder class
violations = validator.validate_step_builder_class(YourStepBuilder)

# Validate config class
violations = validator.validate_config_class(YourConfigClass)

# Validate file naming
violations = validator.validate_file_naming("builder_xgboost_training_step.py", "builder")

# Validate registry entry
violations = validator.validate_registry_entry(step_name, registry_info)
```

### Registry Validation

```python
# Validate all registry entries
violations = validator.validate_all_registry_entries()

if violations:
    print("Registry naming violations found:")
    for violation in violations:
        print(f"  - {violation}")
```

### Import from Main Package

```python
# Import from main validation package
from src.cursus.validation import NamingStandardValidator

validator = NamingStandardValidator()
```

## Violation Types

The validator reports different types of violations:

- `pascal_case`: Name should be in PascalCase
- `snake_case`: Name should be in snake_case
- `config_suffix`: Config class missing 'Config' suffix
- `builder_suffix`: Builder class missing 'StepBuilder' suffix
- `file_naming`: File name doesn't match expected pattern
- `registry_inconsistency`: Name doesn't match registry entry
- `registry_missing`: Class not found in registry
- `spec_type_mismatch`: Spec type doesn't match canonical name

## Example Output

```
Registry naming violations found:
  - Registry Config Class: Config class name doesn't follow expected patterns for step 'Base' (Expected: One of: BaseConfig, Actual: BasePipelineConfig)
  - Builder Class 'StepBuilderBase': Builder class name must end with 'StepBuilder' (Expected: StepBuilderBaseStepBuilder, Actual: StepBuilderBase)
  - File 'XGBoostTrainingBuilder.py': File name doesn't match expected pattern for builder files (Expected: Pattern: ^builder_[a-z_]+_step\.py$, Actual: XGBoostTrainingBuilder.py) Suggestions: builder_xg_boost_training_builder_step.py
```

## Running Examples

To see the validator in action, run the example script:

```bash
cd /path/to/cursus
python -m src.cursus.validation.naming.example_usage
```

This will demonstrate all validation features with real examples from the codebase.

## Integration with Standardization Rules

This validator implements the naming conventions described in the standardization rules document (`slipbox/0_developer_guide/standardization_rules.md`). It ensures that all pipeline components follow consistent naming patterns for better maintainability and code quality.

## Error Handling

The validator provides detailed error messages with:
- Clear description of the violation
- Expected vs actual values
- Suggestions for corrections
- Component context information

This makes it easy to identify and fix naming convention violations across the codebase.
