---
tags:
  - test
  - validation
  - naming
  - standards
  - conventions
keywords:
  - naming validation
  - naming standards
  - naming conventions
  - identifier validation
  - code standards
topics:
  - naming validation
  - code standards
  - naming conventions
  - identifier compliance
language: python
date of note: 2025-08-18
---

# Naming Standard Validator

The Naming Standard Validator ensures that all identifiers (classes, methods, variables, files) within the Cursus framework follow established naming conventions and standards. It promotes consistency, readability, and maintainability across the codebase.

## Overview

The Naming Standard Validator is a specialized component of the validation framework that focuses on enforcing consistent naming patterns throughout the Cursus ecosystem. It validates naming conventions for various code elements and provides actionable feedback for improvements.

## Purpose

The validator serves several critical purposes:

1. **Consistency Enforcement**: Ensures uniform naming patterns across all components
2. **Readability Improvement**: Promotes clear, descriptive naming that enhances code readability
3. **Convention Compliance**: Enforces Python and framework-specific naming conventions
4. **Maintainability**: Facilitates easier code maintenance through consistent naming
5. **Documentation Standards**: Ensures names are self-documenting and meaningful

## Naming Convention Rules

### Class Names

#### Step Builder Classes
```python
# ✅ Correct naming
class TabularPreprocessingStepBuilder(StepBuilderBase):
    pass

class XGBoostTrainingStepBuilder(StepBuilderBase):
    pass

class ModelEvaluationStepBuilder(StepBuilderBase):
    pass

# ❌ Incorrect naming
class TabularPreprocessing(StepBuilderBase):  # Missing 'StepBuilder' suffix
    pass

class tabular_preprocessing_step_builder(StepBuilderBase):  # Wrong case
    pass

class TPStepBuilder(StepBuilderBase):  # Unclear abbreviation
    pass
```

**Rules for Step Builder Classes**:
- Must end with `StepBuilder`
- Use PascalCase (CapitalizedWords)
- Be descriptive of the step's purpose
- Avoid unclear abbreviations
- Maximum length: 50 characters

#### Other Classes
```python
# ✅ Correct naming
class ConfigurationManager:
    pass

class ValidationResult:
    pass

class AlignmentReport:
    pass

# ❌ Incorrect naming
class configManager:  # Wrong case
    pass

class ValidationRes:  # Unclear abbreviation
    pass

class alignment_report:  # Wrong case
    pass
```

**Rules for General Classes**:
- Use PascalCase
- Be descriptive and clear
- Avoid abbreviations unless widely understood
- Use noun phrases

### Method Names

#### Public Methods
```python
# ✅ Correct naming
def build_step(self, config):
    pass

def validate_configuration(self, config):
    pass

def get_step_name(self):
    pass

def calculate_alignment_score(self):
    pass

# ❌ Incorrect naming
def BuildStep(self, config):  # Wrong case
    pass

def validateConfig(self, config):  # camelCase not allowed
    pass

def get_name(self):  # Too generic
    pass

def calc_score(self):  # Unclear abbreviation
    pass
```

**Rules for Public Methods**:
- Use snake_case
- Start with a verb when appropriate
- Be descriptive of the action performed
- Avoid abbreviations
- Maximum length: 40 characters

#### Private Methods
```python
# ✅ Correct naming
def _prepare_internal_config(self):
    pass

def _validate_step_parameters(self):
    pass

def _calculate_dependency_score(self):
    pass

# ❌ Incorrect naming
def __prepare_config(self):  # Double underscore reserved for special methods
    pass

def _prep_config(self):  # Unclear abbreviation
    pass

def _PrepareConfig(self):  # Wrong case
    pass
```

**Rules for Private Methods**:
- Start with single underscore `_`
- Use snake_case
- Be descriptive despite being internal
- Avoid double underscores unless implementing special methods

#### Special Methods
```python
# ✅ Correct naming (Python special methods)
def __init__(self, config):
    pass

def __str__(self):
    pass

def __repr__(self):
    pass

def __eq__(self, other):
    pass

# ❌ Incorrect naming
def __custom_method__(self):  # Don't create custom dunder methods
    pass
```

### Variable Names

#### Instance Variables
```python
# ✅ Correct naming
def __init__(self):
    self.step_name = "processing"
    self.configuration_manager = ConfigManager()
    self.validation_results = []
    self.is_validated = False

# ❌ Incorrect naming
def __init__(self):
    self.stepName = "processing"  # camelCase not allowed
    self.cfg_mgr = ConfigManager()  # Unclear abbreviation
    self.results = []  # Too generic
    self.validated = None  # Unclear type/purpose
```

#### Local Variables
```python
# ✅ Correct naming
def process_data(self, input_data):
    processed_results = []
    validation_errors = []
    current_step_index = 0
    
    for data_item in input_data:
        item_result = self._process_single_item(data_item)
        processed_results.append(item_result)

# ❌ Incorrect naming
def process_data(self, input_data):
    results = []  # Too generic
    errs = []  # Unclear abbreviation
    i = 0  # Non-descriptive
    
    for item in input_data:  # 'item' is acceptable for short loops
        res = self._process_single_item(item)  # Unclear abbreviation
        results.append(res)
```

#### Constants
```python
# ✅ Correct naming
DEFAULT_TIMEOUT_SECONDS = 300
MAX_RETRY_ATTEMPTS = 3
VALIDATION_ERROR_THRESHOLD = 0.8
SUPPORTED_STEP_TYPES = ['Training', 'Processing', 'Transform']

# ❌ Incorrect naming
default_timeout = 300  # Wrong case
MAX_RETRIES = 3  # Less descriptive
THRESHOLD = 0.8  # Too generic
TYPES = ['Training', 'Processing']  # Too generic
```

### File and Module Names

#### Python Files
```python
# ✅ Correct naming
tabular_preprocessing_step.py
xgboost_training_builder.py
alignment_validation_utils.py
configuration_manager.py

# ❌ Incorrect naming
TabularPreprocessingStep.py  # Wrong case
xgboost_training.py  # Missing context
alignmentUtils.py  # camelCase not allowed
cfg_mgr.py  # Unclear abbreviation
```

**Rules for File Names**:
- Use snake_case
- Be descriptive of contents
- Include context when necessary
- Avoid abbreviations
- Use `.py` extension for Python files

#### Directory Names
```python
# ✅ Correct naming
validation/
alignment/
step_builders/
configuration/
test_utilities/

# ❌ Incorrect naming
Validation/  # Wrong case
align/  # Unclear abbreviation
stepBuilders/  # camelCase not allowed
cfg/  # Unclear abbreviation
```

## Validation Process

### 1. Identifier Discovery

The validator discovers all identifiers that need validation:

```python
# Discover identifiers in a module
identifiers = discover_module_identifiers(module)

# Categorize identifiers
classes = identifiers['classes']
methods = identifiers['methods']
variables = identifiers['variables']
constants = identifiers['constants']
```

### 2. Pattern Matching

Each identifier is validated against appropriate patterns:

```python
# Class name validation
def validate_class_name(class_name, class_type):
    if class_type == 'step_builder':
        if not class_name.endswith('StepBuilder'):
            return ValidationError("Step builder class must end with 'StepBuilder'")
        if not is_pascal_case(class_name):
            return ValidationError("Class name must use PascalCase")
    
    return ValidationSuccess()
```

### 3. Context-Aware Validation

The validator considers context when applying rules:

```python
# Method name validation with context
def validate_method_name(method_name, method_context):
    if method_context['visibility'] == 'private':
        if not method_name.startswith('_'):
            return ValidationError("Private method must start with underscore")
    
    if method_context['type'] == 'property_getter':
        if not method_name.startswith('get_'):
            return ValidationWarning("Getter method should start with 'get_'")
    
    return ValidationSuccess()
```

### 4. Abbreviation Detection

The validator detects and flags unclear abbreviations:

```python
# Common acceptable abbreviations
ACCEPTABLE_ABBREVIATIONS = {
    'config': 'configuration',
    'spec': 'specification',
    'eval': 'evaluation',
    'param': 'parameter',
    'arg': 'argument',
    'temp': 'temporary',
    'util': 'utility'
}

# Unacceptable abbreviations
UNACCEPTABLE_ABBREVIATIONS = {
    'cfg': 'configuration',
    'mgr': 'manager',
    'proc': 'process',
    'val': 'value',
    'res': 'result'
}
```

## Validation Categories

### Critical Violations

These violations must be fixed:

1. **Wrong Case Convention**: Using incorrect case for identifiers
2. **Missing Required Suffixes**: Step builders without 'StepBuilder' suffix
3. **Reserved Word Usage**: Using Python reserved words as identifiers
4. **Invalid Characters**: Using invalid characters in identifiers

### Warning Violations

These should be addressed but don't prevent operation:

1. **Unclear Abbreviations**: Using abbreviations that reduce readability
2. **Generic Names**: Using overly generic names like 'data', 'result'
3. **Length Violations**: Names that are too long or too short
4. **Inconsistent Patterns**: Deviating from established patterns

### Informational Messages

These provide guidance for improvement:

1. **Style Suggestions**: Alternative naming suggestions
2. **Best Practice Tips**: Recommendations for better naming
3. **Consistency Notes**: Highlighting inconsistencies with similar code

## Error Messages and Suggestions

### Class Name Errors

```python
# Error: Wrong case
"Class name 'tabularPreprocessingStepBuilder' should use PascalCase: 'TabularPreprocessingStepBuilder'"

# Error: Missing suffix
"Step builder class 'TabularPreprocessing' must end with 'StepBuilder'"

# Suggestion: Better naming
"Consider renaming 'TPStepBuilder' to 'TabularPreprocessingStepBuilder' for clarity"
```

### Method Name Errors

```python
# Error: Wrong case
"Method name 'BuildStep' should use snake_case: 'build_step'"

# Error: Unclear abbreviation
"Method name 'calc_score' uses unclear abbreviation. Consider 'calculate_score'"

# Warning: Generic name
"Method name 'process' is too generic. Consider 'process_validation_data'"
```

### Variable Name Errors

```python
# Error: Wrong case
"Variable name 'stepName' should use snake_case: 'step_name'"

# Warning: Abbreviation
"Variable name 'cfg' is unclear. Consider 'configuration' or 'config'"

# Info: Generic name
"Variable name 'data' is generic. Consider 'validation_data' or 'input_data'"
```

## Integration with Validation Framework

### Universal Test Integration

```python
class UniversalStepBuilderTest:
    def __init__(self, builder_class):
        self.naming_validator = NamingStandardValidator(builder_class)
    
    def run_level1_tests(self):
        # Include naming validation in Level 1 tests
        naming_results = self.naming_validator.validate_all_names()
        
        return {
            'naming_compliance': naming_results,
            # Other Level 1 tests...
        }
```

### Alignment Tester Integration

```python
class UnifiedAlignmentTester:
    def _run_level1_validation(self, target_scripts):
        # Include naming validation in alignment testing
        for script_name in target_scripts:
            naming_results = self.naming_validator.validate_script_names(script_name)
            # Include in alignment report...
```

## Usage Examples

### Basic Validation

```python
from cursus.validation.naming.naming_standard_validator import NamingStandardValidator

# Validate a specific class
validator = NamingStandardValidator()
results = validator.validate_class_names(CustomStepBuilder)

if results['passed']:
    print("✅ Naming validation passed")
else:
    print("❌ Naming violations found:")
    for violation in results['violations']:
        print(f"  - {violation['message']}")
```

### Module-Wide Validation

```python
# Validate entire module
import my_module

validator = NamingStandardValidator()
results = validator.validate_module_names(my_module)

# Print summary
print(f"Classes: {results['class_violations']} violations")
print(f"Methods: {results['method_violations']} violations")
print(f"Variables: {results['variable_violations']} violations")
```

### Custom Rules

```python
# Add custom naming rules
validator = NamingStandardValidator()
validator.add_custom_rule('step_builder_prefix', lambda name: name.startswith('Step'))
validator.add_abbreviation_exception('ml', 'machine learning')

results = validator.validate_class_names(CustomStepBuilder)
```

## Configuration Options

### Strictness Levels

```python
# Strict mode (all violations are errors)
validator = NamingStandardValidator(strict_mode=True)

# Lenient mode (some violations are warnings)
validator = NamingStandardValidator(strict_mode=False)

# Custom strictness
validator = NamingStandardValidator(
    class_name_strictness='error',
    method_name_strictness='warning',
    variable_name_strictness='info'
)
```

### Abbreviation Handling

```python
# Allow specific abbreviations
validator = NamingStandardValidator(
    allowed_abbreviations=['config', 'spec', 'eval'],
    forbidden_abbreviations=['cfg', 'mgr', 'proc']
)

# Custom abbreviation dictionary
custom_abbreviations = {
    'ml': 'machine learning',
    'ai': 'artificial intelligence',
    'nlp': 'natural language processing'
}
validator = NamingStandardValidator(abbreviations=custom_abbreviations)
```

### Context-Specific Rules

```python
# Different rules for different contexts
validator = NamingStandardValidator(
    step_builder_rules={
        'require_suffix': True,
        'max_length': 50,
        'allow_abbreviations': False
    },
    utility_class_rules={
        'require_suffix': False,
        'max_length': 30,
        'allow_abbreviations': True
    }
)
```

## Best Practices

### For Developers

1. **Be Descriptive**: Choose names that clearly describe purpose and function
2. **Avoid Abbreviations**: Use full words unless abbreviations are widely understood
3. **Follow Conventions**: Stick to established naming patterns consistently
4. **Consider Context**: Use appropriate names for the scope and context
5. **Review Regularly**: Regularly review and refactor names as code evolves

### For Code Reviewers

1. **Check Consistency**: Ensure naming follows established patterns
2. **Question Abbreviations**: Challenge unclear or unnecessary abbreviations
3. **Suggest Improvements**: Provide constructive suggestions for better names
4. **Consider Readability**: Evaluate names from a readability perspective
5. **Enforce Standards**: Consistently apply naming standards across the codebase

## Performance Considerations

The Naming Standard Validator is optimized for performance:

- **Pattern Caching**: Compiled regex patterns are cached for reuse
- **Incremental Validation**: Only validates changed identifiers when possible
- **Batch Processing**: Efficiently processes multiple identifiers together
- **Early Exit**: Stops validation early for critical violations when configured

## Future Enhancements

Planned improvements to the Naming Standard Validator:

1. **AI-Powered Suggestions**: Use machine learning to suggest better names
2. **Domain-Specific Rules**: Specialized rules for different domains (ML, data processing)
3. **IDE Integration**: Real-time naming validation in development environments
4. **Refactoring Support**: Automated renaming suggestions and implementations
5. **Internationalization**: Support for naming conventions in different languages

The Naming Standard Validator ensures that the Cursus framework maintains consistent, readable, and maintainable naming conventions across all components, contributing to overall code quality and developer productivity.
