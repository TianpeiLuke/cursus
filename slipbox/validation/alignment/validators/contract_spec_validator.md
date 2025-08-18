---
tags:
  - code
  - validation
  - alignment
  - contract
  - specification
  - validator
keywords:
  - contract specification validator
  - logical name validation
  - data type validation
  - input output alignment
  - contract validation
  - specification validation
  - alignment validation
  - validation logic
topics:
  - validation framework
  - contract specification
  - alignment validation
  - validation logic
language: python
date of note: 2025-08-18
---

# Contract Specification Validator

## Overview

The Contract Specification Validator provides core validation logic for Level 2 alignment between script contracts and step specifications. It handles data type validation, input/output alignment, and basic logical name validation for single specifications.

## Core Functionality

### ContractSpecValidator Class

The main validator class that implements contract-specification alignment validation:

```python
class ContractSpecValidator:
    """
    Handles core validation logic for contract-specification alignment.
    
    Provides methods for:
    - Data type consistency validation
    - Input/output alignment validation
    - Basic logical name validation (non-smart)
    """
```

**Validation Capabilities:**
- **Logical Name Validation**: Basic matching between contract and specification logical names
- **Data Type Validation**: Consistency checking for data types (extensible)
- **Input/Output Alignment**: Validation of input/output correspondence

## Validation Methods

### Logical Name Validation

```python
def validate_logical_names(self, contract: Dict[str, Any], specification: Dict[str, Any], 
                          contract_name: str, job_type: str = None) -> List[Dict[str, Any]]:
    """
    Validate that logical names match between contract and specification.
    
    This is the basic (non-smart) validation for single specifications.
    """
```

**Validation Process:**

#### 1. Extract Logical Names from Contract
```python
# Get logical names from contract
contract_inputs = set(contract.get('inputs', {}).keys())
contract_outputs = set(contract.get('outputs', {}).keys())
```

#### 2. Extract Logical Names from Specification
```python
# Get logical names from specification
spec_dependencies = set()
for dep in specification.get('dependencies', []):
    if 'logical_name' in dep:
        spec_dependencies.add(dep['logical_name'])

spec_outputs = set()
for output in specification.get('outputs', []):
    if 'logical_name' in output:
        spec_outputs.add(output['logical_name'])
```

#### 3. Validate Input Alignment
```python
# Check for contract inputs not in spec dependencies
missing_deps = contract_inputs - spec_dependencies
for logical_name in missing_deps:
    issues.append({
        'severity': 'ERROR',
        'category': 'logical_names',
        'message': f'Contract input {logical_name} not declared as specification dependency',
        'details': {'logical_name': logical_name, 'contract': contract_name},
        'recommendation': f'Add {logical_name} to specification dependencies'
    })
```

#### 4. Validate Output Alignment
```python
# Check for contract outputs not in spec outputs
missing_outputs = contract_outputs - spec_outputs
for logical_name in missing_outputs:
    issues.append({
        'severity': 'ERROR',
        'category': 'logical_names',
        'message': f'Contract output {logical_name} not declared as specification output',
        'details': {'logical_name': logical_name, 'contract': contract_name},
        'recommendation': f'Add {logical_name} to specification outputs'
    })
```

**Validation Features:**
- **Exact Matching**: Requires exact logical name matches
- **Bidirectional Validation**: Checks both inputs and outputs
- **Error Classification**: Reports missing dependencies and outputs as errors
- **Actionable Recommendations**: Provides specific fix suggestions

### Data Type Validation

```python
def validate_data_types(self, contract: Dict[str, Any], specification: Dict[str, Any], 
                       contract_name: str) -> List[Dict[str, Any]]:
    """
    Validate data type consistency between contract and specification.
    """
```

**Current Implementation:**
```python
# Note: Contract inputs/outputs are typically stored as simple path strings,
# while specifications have rich data type information.
# For now, we'll skip detailed data type validation since the contract
# format doesn't include explicit data type declarations.

# This could be enhanced in the future if contracts are extended
# to include data type information.

return issues
```

**Design Considerations:**
- **Contract Format Limitation**: Current contracts store paths as strings without type information
- **Specification Richness**: Specifications contain detailed data type information
- **Future Enhancement**: Extensible design for when contracts include type information
- **Graceful Handling**: Returns empty issues list rather than failing

**Future Enhancement Possibilities:**
```python
# Potential future implementation when contracts include type info
def validate_data_types(self, contract: Dict[str, Any], specification: Dict[str, Any], 
                       contract_name: str) -> List[Dict[str, Any]]:
    issues = []
    
    # Extract type information from contract (future)
    contract_input_types = contract.get('input_types', {})
    contract_output_types = contract.get('output_types', {})
    
    # Validate against specification types
    for dep in specification.get('dependencies', []):
        logical_name = dep.get('logical_name')
        spec_type = dep.get('data_type')
        contract_type = contract_input_types.get(logical_name)
        
        if contract_type and spec_type and contract_type != spec_type:
            issues.append({
                'severity': 'ERROR',
                'category': 'data_types',
                'message': f'Data type mismatch for {logical_name}: contract={contract_type}, spec={spec_type}',
                'recommendation': f'Align data types for {logical_name}'
            })
    
    return issues
```

### Input/Output Alignment Validation

```python
def validate_input_output_alignment(self, contract: Dict[str, Any], specification: Dict[str, Any], 
                                   contract_name: str) -> List[Dict[str, Any]]:
    """
    Validate input/output alignment between contract and specification.
    """
```

**Validation Process:**

#### 1. Dependency Alignment Check
```python
# Check for specification dependencies without corresponding contract inputs
spec_deps = {dep.get('logical_name') for dep in specification.get('dependencies', [])}
contract_inputs = set(contract.get('inputs', {}).keys())

unmatched_deps = spec_deps - contract_inputs
for logical_name in unmatched_deps:
    if logical_name:  # Skip None values
        issues.append({
            'severity': 'WARNING',
            'category': 'input_output_alignment',
            'message': f'Specification dependency {logical_name} has no corresponding contract input',
            'details': {'logical_name': logical_name, 'contract': contract_name},
            'recommendation': f'Add {logical_name} to contract inputs or remove from specification dependencies'
        })
```

#### 2. Output Alignment Check
```python
# Check for specification outputs without corresponding contract outputs
spec_outputs = {out.get('logical_name') for out in specification.get('outputs', [])}
contract_outputs = set(contract.get('outputs', {}).keys())

unmatched_outputs = spec_outputs - contract_outputs
for logical_name in unmatched_outputs:
    if logical_name:  # Skip None values
        issues.append({
            'severity': 'WARNING',
            'category': 'input_output_alignment',
            'message': f'Specification output {logical_name} has no corresponding contract output',
            'details': {'logical_name': logical_name, 'contract': contract_name},
            'recommendation': f'Add {logical_name} to contract outputs or remove from specification outputs'
        })
```

**Validation Features:**
- **Reverse Validation**: Checks specification → contract alignment
- **Warning Level**: Uses WARNING severity for alignment mismatches
- **Null Safety**: Skips None values to avoid false positives
- **Bidirectional Coverage**: Validates both dependencies and outputs

## Integration Patterns

### Usage with ContractSpecificationAlignmentTester

```python
# Initialize validator
validator = ContractSpecValidator()

# Perform validation checks
logical_issues = validator.validate_logical_names(contract, specification, contract_name)
type_issues = validator.validate_data_types(contract, specification, contract_name)
io_issues = validator.validate_input_output_alignment(contract, specification, contract_name)

# Combine all issues
all_issues = []
all_issues.extend(logical_issues)
all_issues.extend(type_issues)
all_issues.extend(io_issues)
```

### Smart Validation Integration

The validator works alongside smart validation components:

```python
# Basic validation (this validator)
basic_issues = validator.validate_logical_names(contract, specification, contract_name)

# Smart validation (SmartSpecificationSelector)
smart_issues = smart_selector.validate_logical_names_smart(contract, unified_spec, contract_name)

# Use smart validation for multi-variant scenarios, basic for single specifications
if has_multiple_variants:
    issues.extend(smart_issues)
else:
    issues.extend(basic_issues)
```

## Issue Classification

### Severity Levels

**ERROR Level Issues:**
- Missing contract inputs that are declared as specification dependencies
- Missing contract outputs that are declared as specification outputs
- Data type mismatches (when implemented)

**WARNING Level Issues:**
- Specification dependencies without corresponding contract inputs
- Specification outputs without corresponding contract outputs
- Potential alignment inconsistencies

### Issue Categories

**logical_names:**
- Contract-specification logical name mismatches
- Missing logical name declarations
- Inconsistent naming patterns

**input_output_alignment:**
- Input/output correspondence issues
- Unmatched dependencies or outputs
- Alignment inconsistencies

**data_types:**
- Data type consistency issues (future implementation)
- Type compatibility problems
- Format mismatches

## Validation Results Structure

### Issue Format

```python
{
    'severity': 'ERROR',
    'category': 'logical_names',
    'message': 'Contract input training_data not declared as specification dependency',
    'details': {
        'logical_name': 'training_data',
        'contract': 'xgboost_training_contract'
    },
    'recommendation': 'Add training_data to specification dependencies'
}
```

### Comprehensive Validation Result

```python
# Example validation result
validation_issues = [
    {
        'severity': 'ERROR',
        'category': 'logical_names',
        'message': 'Contract input training_data not declared as specification dependency',
        'details': {'logical_name': 'training_data', 'contract': 'xgboost_training'},
        'recommendation': 'Add training_data to specification dependencies'
    },
    {
        'severity': 'WARNING',
        'category': 'input_output_alignment',
        'message': 'Specification dependency validation_data has no corresponding contract input',
        'details': {'logical_name': 'validation_data', 'contract': 'xgboost_training'},
        'recommendation': 'Add validation_data to contract inputs or remove from specification dependencies'
    }
]
```

## Best Practices

### Contract Design

**Clear Logical Names:**
```python
# Good: Descriptive and consistent logical names
contract = {
    'inputs': {
        'training_data': '/path/to/training',
        'validation_data': '/path/to/validation'
    },
    'outputs': {
        'trained_model': '/path/to/model',
        'evaluation_metrics': '/path/to/metrics'
    }
}
```

### Specification Design

**Matching Logical Names:**
```python
# Good: Specification matches contract logical names exactly
specification = {
    'dependencies': [
        {'logical_name': 'training_data', 'data_type': 'tabular'},
        {'logical_name': 'validation_data', 'data_type': 'tabular'}
    ],
    'outputs': [
        {'logical_name': 'trained_model', 'output_type': 'model'},
        {'logical_name': 'evaluation_metrics', 'output_type': 'metrics'}
    ]
}
```

### Validation Usage

**Comprehensive Validation:**
```python
# Good: Use all validation methods
validator = ContractSpecValidator()

issues = []
issues.extend(validator.validate_logical_names(contract, spec, name))
issues.extend(validator.validate_data_types(contract, spec, name))
issues.extend(validator.validate_input_output_alignment(contract, spec, name))

# Process issues based on severity
errors = [issue for issue in issues if issue['severity'] == 'ERROR']
warnings = [issue for issue in issues if issue['severity'] == 'WARNING']
```

## Limitations and Future Enhancements

### Current Limitations

**Data Type Validation:**
- Limited by current contract format that doesn't include type information
- Cannot perform detailed type compatibility checking
- Relies on specification type information only

**Single Specification Focus:**
- Designed for basic single-specification validation
- Does not handle multi-variant specifications
- Complemented by smart validation for complex scenarios

### Future Enhancements

**Enhanced Data Type Support:**
- Contract format extension to include type information
- Detailed type compatibility checking
- Schema validation capabilities

**Advanced Matching:**
- Fuzzy logical name matching
- Semantic similarity detection
- Pattern-based name recognition

**Integration Improvements:**
- Better integration with smart validation
- Configurable validation strictness
- Custom validation rule support

The Contract Specification Validator provides essential basic validation capabilities for Level 2 alignment, serving as a foundation for more sophisticated validation approaches while maintaining simplicity and reliability for standard use cases.
