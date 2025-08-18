---
tags:
  - code
  - validation
  - alignment
  - contract
  - specification
keywords:
  - contract specification alignment
  - logical name validation
  - data type consistency
  - input output alignment
  - specification validation
  - contract validation
  - property path validation
  - smart specification selection
topics:
  - validation framework
  - alignment validation
  - contract specification
  - specification analysis
language: python
date of note: 2025-08-18
---

# Contract Specification Alignment

## Overview

The Contract Specification Alignment Tester validates Level 2 alignment between script contracts and step specifications. It ensures logical names, data types, and dependencies are consistent across the contract-specification boundary using smart specification selection and unified validation.

## Core Functionality

### ContractSpecificationAlignmentTester Class

The main class orchestrates comprehensive validation of contract-specification alignment:

```python
class ContractSpecificationAlignmentTester:
    """
    Tests alignment between script contracts and step specifications.
    
    Validates:
    - Logical names match between contract and specification
    - Data types are consistent
    - Input/output specifications align
    - Dependencies are properly declared
    """
```

### Component Architecture

The tester integrates multiple specialized components for robust validation:

**Core Components:**
- **FlexibleFileResolver**: Robust file discovery with fuzzy matching
- **SageMakerPropertyPathValidator**: Validates SageMaker property path references
- **ContractLoader**: Loads and parses contract files
- **SpecificationLoader**: Loads and processes specification files
- **SmartSpecificationSelector**: Handles multi-variant specification logic
- **ContractSpecValidator**: Performs alignment validation checks
- **ContractDiscoveryEngine**: Discovers contracts with corresponding scripts
- **SpecificationFileProcessor**: Processes specification file formats
- **ValidationOrchestrator**: Coordinates validation workflow

### Initialization and Setup

```python
def __init__(self, contracts_dir: str, specs_dir: str):
    """Initialize the contract-specification alignment tester."""
```

**Directory Structure:**
```python
base_directories = {
    'contracts': str(self.contracts_dir),
    'specs': str(self.specs_dir)
}
```

**Component Initialization:**
- File resolver for robust contract/spec discovery
- Property path validator for SageMaker step validation
- Loaders for contract and specification parsing
- Smart selector for multi-variant specification handling
- Validator for alignment checks

## Validation Process

### Comprehensive Contract Validation

```python
def validate_all_contracts(self, target_scripts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Validate alignment for all contracts or specified target scripts."""
```

**Process Flow:**
1. **Discovery Phase**: Identify contracts to validate
2. **Filtering**: Only validate contracts with corresponding scripts
3. **Individual Validation**: Process each contract separately
4. **Error Handling**: Capture and report validation failures
5. **Result Aggregation**: Compile comprehensive results

### Individual Contract Validation

```python
def validate_contract(self, script_or_contract_name: str) -> Dict[str, Any]:
    """Validate alignment for a specific contract using Smart Specification Selection."""
```

**Validation Steps:**

#### 1. Contract File Resolution
```python
# Use FlexibleFileResolver for robust file discovery
contract_file_path = self.file_resolver.find_contract_file(script_or_contract_name)
```

**Resolution Features:**
- Standard naming patterns: `{script_name}_contract.py`
- Fuzzy matching for similar names
- Known naming pattern recognition
- Comprehensive error reporting with search details

#### 2. Contract Loading
```python
contract = self.contract_loader.load_contract(contract_path, actual_contract_name)
```

**Loading Process:**
- Python file parsing and execution
- Contract structure validation
- Field extraction and normalization
- Error handling for syntax issues

#### 3. Specification Discovery
```python
spec_files = self.spec_loader.find_specifications_by_contract(actual_contract_name)
```

**Discovery Logic:**
- Find specifications referencing the contract
- Support multiple specification variants
- Handle job type variations
- Validate specification file structure

#### 4. Specification Loading
```python
specifications = {}
for spec_file, spec_info in spec_files.items():
    spec = self.spec_loader.load_specification(spec_file, spec_info)
    spec_key = spec_file.stem
    specifications[spec_key] = spec
```

**Multi-Specification Handling:**
- Load all related specifications
- Preserve job type information
- Handle specification variants
- Maintain specification metadata

#### 5. Smart Specification Selection
```python
# Create unified specification model
unified_spec = self.smart_spec_selector.create_unified_specification(specifications, actual_contract_name)
```

**Unified Specification Features:**
- **Primary Specification**: Main specification for validation
- **Variant Handling**: Manages job type variants (training, validation, testing)
- **Logical Name Unification**: Merges logical names across variants
- **Dependency Consolidation**: Combines dependencies from all variants

## Validation Checks

### Smart Logical Name Validation

```python
# Validate logical name alignment using smart multi-variant logic
logical_issues = self.smart_spec_selector.validate_logical_names_smart(contract, unified_spec, actual_contract_name)
```

**Smart Validation Features:**
- **Multi-Variant Logic**: Handles specifications with job type variants
- **Flexible Matching**: Accommodates naming variations across variants
- **Context-Aware Validation**: Considers specification context
- **Comprehensive Coverage**: Validates all logical name mappings

**Validation Process:**
1. Extract logical names from contract inputs/outputs
2. Extract logical names from unified specification
3. Perform cross-variant matching
4. Report mismatches with context

### Data Type Consistency Validation

```python
type_issues = self.validator.validate_data_types(contract, unified_spec['primary_spec'], actual_contract_name)
```

**Type Validation Features:**
- **Type Compatibility**: Ensures compatible data types
- **Format Consistency**: Validates data format specifications
- **Schema Alignment**: Checks schema compatibility
- **Type Coercion Rules**: Applies appropriate type conversion rules

### Input/Output Alignment Validation

```python
io_issues = self.validator.validate_input_output_alignment(contract, unified_spec['primary_spec'], actual_contract_name)
```

**I/O Validation Features:**
- **Input Mapping**: Validates contract inputs match specification requirements
- **Output Mapping**: Ensures contract outputs align with specification declarations
- **Cardinality Checks**: Validates input/output counts
- **Dependency Validation**: Checks dependency declarations

### Property Path Validation (Level 2 Enhancement)

```python
def _validate_property_paths(self, specification: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
    """Validate SageMaker Step Property Path References."""
```

**Property Path Features:**
- **SageMaker Step Type Validation**: Ensures property paths are valid for step type
- **Path Structure Validation**: Validates property path syntax
- **Reference Validation**: Checks property path references
- **Step Type Compatibility**: Ensures paths match SageMaker step capabilities

**Integration:**
```python
property_path_issues = self._validate_property_paths(unified_spec['primary_spec'], actual_contract_name)
```

## Smart Specification Selection

### Unified Specification Model

The SmartSpecificationSelector creates a unified view of multiple specification variants:

```python
unified_spec = {
    'primary_spec': primary_specification,
    'variants': {
        'training': training_spec,
        'validation': validation_spec,
        'testing': testing_spec
    },
    'unified_logical_names': merged_logical_names,
    'consolidated_dependencies': merged_dependencies
}
```

### Multi-Variant Logical Name Validation

**Smart Validation Logic:**
1. **Variant Detection**: Identify job type variants in specifications
2. **Name Extraction**: Extract logical names from all variants
3. **Cross-Variant Matching**: Match contract names against all variants
4. **Context-Aware Reporting**: Report issues with variant context

**Example Validation:**
```python
# Contract declares: input_data, output_model
# Training spec: input_training_data, output_trained_model
# Validation spec: input_validation_data, output_validated_model
# Smart matching recognizes the pattern and validates appropriately
```

## Error Handling and Diagnostics

### Missing File Diagnostics

**Contract File Not Found:**
```python
{
    'passed': False,
    'issues': [{
        'severity': 'CRITICAL',
        'category': 'missing_file',
        'message': f'Contract file not found for script: {script_or_contract_name}',
        'details': {
            'script': script_or_contract_name,
            'searched_patterns': [
                f'{script_or_contract_name}_contract.py',
                'Known naming patterns from FlexibleFileResolver'
            ]
        },
        'recommendation': f'Create contract file for {script_or_contract_name} or check naming patterns'
    }]
}
```

**Specification File Not Found:**
```python
{
    'passed': False,
    'issues': [{
        'severity': 'ERROR',
        'category': 'missing_specification',
        'message': f'No specification files found for {actual_contract_name}',
        'recommendation': f'Create specification files that reference {actual_contract_name}'
    }]
}
```

### Loading Error Diagnostics

**Contract Load Errors:**
```python
{
    'passed': False,
    'issues': [{
        'severity': 'CRITICAL',
        'category': 'contract_load_error',
        'message': f'Failed to load contract: {str(e)}',
        'recommendation': 'Fix Python syntax or contract structure in contract file'
    }]
}
```

**Specification Load Errors:**
```python
{
    'passed': False,
    'issues': [{
        'severity': 'CRITICAL',
        'category': 'spec_load_error',
        'message': f'Failed to load specification from {spec_file}: {str(e)}',
        'recommendation': 'Fix Python syntax or specification structure'
    }]
}
```

## Discovery and Resolution

### Contract Discovery

```python
def _discover_contracts_with_scripts(self) -> List[str]:
    """Discover contracts that have corresponding scripts by checking their entry_point field."""
```

**Discovery Features:**
- **Script Validation**: Only includes contracts with corresponding scripts
- **Entry Point Checking**: Validates contract entry_point field
- **Robust Discovery**: Uses ContractDiscoveryEngine for comprehensive discovery
- **Error Prevention**: Prevents validation of orphaned contracts

### File Resolution Strategies

**FlexibleFileResolver Integration:**
- **Pattern Matching**: Standard naming conventions
- **Fuzzy Matching**: Similar name detection
- **Normalization**: Handle naming variations
- **Comprehensive Search**: Multiple search strategies

**Resolution Priority:**
1. **Exact Match**: Direct file name matching
2. **Pattern Match**: Known naming patterns
3. **Fuzzy Match**: Similar name detection
4. **Fallback**: Alternative naming conventions

## Integration with Validation Framework

### Result Format

The tester returns comprehensive validation results:

```python
{
    'passed': bool,                           # Overall pass/fail status
    'issues': List[Dict[str, Any]],          # List of alignment issues
    'contract': Dict[str, Any],              # Loaded contract data
    'specifications': Dict[str, Any],         # All loaded specifications
    'unified_specification': Dict[str, Any]   # Unified specification model
}
```

### Issue Categories

- **missing_file**: Contract or specification files not found
- **contract_load_error**: Contract file loading failures
- **spec_load_error**: Specification file loading failures
- **logical_names**: Logical name alignment issues
- **data_types**: Data type consistency issues
- **input_output**: Input/output alignment issues
- **property_paths**: SageMaker property path validation issues

### Severity Levels

- **CRITICAL**: Prevents validation from completing
- **ERROR**: Alignment violations that should fail validation
- **WARNING**: Potential issues that may indicate problems
- **INFO**: Informational findings

## Best Practices

### Contract Design

**Clear Logical Names:**
```python
# Good: Descriptive logical names
contract = {
    'inputs': {'training_data': 'path/to/training'},
    'outputs': {'trained_model': 'path/to/model'}
}
```

**Consistent Naming:**
```python
# Good: Consistent naming between contract and specification
# Contract: input_training_data
# Specification: input_training_data (exact match)
```

### Specification Design

**Multi-Variant Support:**
```python
# Good: Clear job type variants
training_spec = {
    'job_type': 'training',
    'inputs': {'training_data': {...}},
    'outputs': {'trained_model': {...}}
}

validation_spec = {
    'job_type': 'validation', 
    'inputs': {'validation_data': {...}},
    'outputs': {'validation_results': {...}}
}
```

**Property Path Validation:**
```python
# Good: Valid SageMaker property paths
specification = {
    'outputs': {
        'model_artifacts': {
            'property_path': 'Steps.TrainingStep.ModelArtifacts.S3ModelArtifacts'
        }
    }
}
```

### File Organization

**Standard Naming Patterns:**
- Contract files: `{script_name}_contract.py`
- Specification files: `{spec_name}_spec.py`
- Consistent naming between related files

**Directory Structure:**
```
contracts/
├── xgboost_training_contract.py
├── pytorch_training_contract.py
└── model_evaluation_contract.py

specs/
├── xgboost_training_spec.py
├── pytorch_training_spec.py
└── model_evaluation_spec.py
```

The Contract Specification Alignment Tester provides essential Level 2 validation capabilities, ensuring consistency and compatibility between script contracts and step specifications through smart specification selection and comprehensive validation checks.
