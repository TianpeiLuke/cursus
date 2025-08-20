---
tags:
  - code
  - validation
  - alignment
  - contract_loading
  - module_loading
keywords:
  - contract loader
  - contract loading
  - module loading
  - import handling
  - contract parsing
  - dynamic loading
  - sys.path management
  - contract conversion
topics:
  - alignment validation
  - contract loading
  - module management
  - validation infrastructure
language: python
date of note: 2025-08-19
---

# Contract Loader

## Overview

The `ContractLoader` class handles loading and parsing of script contracts from Python files. It provides robust import handling and contract object extraction with sophisticated sys.path management to handle complex project structures and relative imports.

## Core Components

### ContractLoader Class

The main class responsible for loading script contracts from Python files.

#### Initialization

```python
def __init__(self, contracts_dir: str)
```

Initializes the contract loader with the directory containing contract files.

## Key Methods

### Contract Loading

```python
def load_contract(self, contract_path: Path, contract_name: str) -> Dict[str, Any]
```

Loads contract from Python file using robust sys.path management:

#### Sys.Path Management
- Temporarily adds project root, src root, and contract directory to sys.path
- Handles relative imports properly
- Cleans up sys.path modifications after loading
- Prevents import conflicts and path pollution

#### Module Loading Process
1. Creates module spec from file location
2. Sets module package for relative imports (`cursus.steps.contracts`)
3. Executes module to load contract objects
4. Finds contract object using multiple naming patterns
5. Converts contract object to dictionary format

#### Error Handling
- Comprehensive error handling for import failures
- Detailed error messages with context
- Graceful cleanup of sys.path modifications
- Proper exception propagation with debugging information

### Contract Object Discovery

```python
def _find_contract_object(self, module, contract_name: str)
```

Finds contract objects using multiple naming patterns:

#### Naming Pattern Support
- `{CONTRACT_NAME}_CONTRACT`: Standard uppercase pattern
- `{contract_name}_CONTRACT`: Mixed case pattern
- `{contract_name}_contract`: Lowercase pattern
- `XGBOOST_MODEL_EVAL_CONTRACT`: Specific legacy patterns
- `MODEL_EVALUATION_CONTRACT`: Legacy fallback patterns
- `CONTRACT`: Generic contract object
- `contract`: Lowercase generic pattern

#### Dynamic Discovery
- Scans module attributes for `_CONTRACT` suffix
- Removes duplicates while preserving order
- Validates contract objects by checking for `entry_point` attribute
- Supports various contract object structures

### Contract Conversion

```python
def _contract_to_dict(self, contract_obj, contract_name: str) -> Dict[str, Any]
```

Converts ScriptContract objects to standardized dictionary format:

#### Contract Dictionary Structure
```python
{
    'entry_point': 'script_name.py',
    'inputs': {
        'logical_name': {'path': '/path/to/input'}
    },
    'outputs': {
        'logical_name': {'path': '/path/to/output'}
    },
    'arguments': {
        'arg_name': {'default': value, 'required': bool}
    },
    'environment_variables': {
        'required': ['ENV_VAR1', 'ENV_VAR2'],
        'optional': {'ENV_VAR3': 'default_value'}
    },
    'description': 'Contract description',
    'framework_requirements': {...}
}
```

#### Attribute Mapping
- **entry_point**: Script filename from contract object
- **inputs**: Converted from `expected_input_paths`
- **outputs**: Converted from `expected_output_paths`
- **arguments**: Converted from `expected_arguments`
- **environment_variables**: Required and optional environment variables
- **description**: Contract description text
- **framework_requirements**: Framework-specific requirements

## Implementation Details

### Robust Import Handling

The loader uses sophisticated sys.path management:

```python
# Add paths temporarily for imports
project_root = str(contract_path.parent.parent.parent.parent)
src_root = str(contract_path.parent.parent.parent)
contract_dir = str(contract_path.parent)

paths_to_add = [project_root, src_root, contract_dir]
added_paths = []

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)
        added_paths.append(path)

try:
    # Load and execute module
    # ...
finally:
    # Clean up sys.path
    for path in added_paths:
        if path in sys.path:
            sys.path.remove(path)
```

### Module Loading Strategy

Uses `importlib.util` for dynamic module loading:

```python
# Create module spec from file
spec = importlib.util.spec_from_file_location(module_name, contract_path)
module = importlib.util.module_from_spec(spec)

# Set package for relative imports
module.__package__ = 'cursus.steps.contracts'

# Execute module
spec.loader.exec_module(module)
```

### Contract Object Validation

Validates contract objects by checking for required attributes:

```python
def _find_contract_object(self, module, contract_name: str):
    for name in possible_names:
        if hasattr(module, name):
            contract_obj = getattr(module, name)
            # Verify it's actually a contract object
            if hasattr(contract_obj, 'entry_point'):
                return contract_obj
    return None
```

## Usage Examples

### Basic Contract Loading

```python
# Initialize contract loader
loader = ContractLoader('src/cursus/steps/contracts')

# Load specific contract
contract_path = Path('src/cursus/steps/contracts/preprocessing_contract.py')
contract_dict = loader.load_contract(contract_path, 'preprocessing')

print(f"Entry point: {contract_dict['entry_point']}")
print(f"Inputs: {contract_dict['inputs']}")
print(f"Outputs: {contract_dict['outputs']}")
```

### Integration with Validation

```python
# Use in validation context
def validate_contract_alignment(script_name, contracts_dir):
    loader = ContractLoader(contracts_dir)
    
    try:
        contract_path = Path(contracts_dir) / f"{script_name}_contract.py"
        contract = loader.load_contract(contract_path, script_name)
        
        # Validate contract structure
        if not contract['entry_point']:
            print(f"Contract missing entry point: {script_name}")
        
        if not contract['inputs'] and not contract['outputs']:
            print(f"Contract has no inputs or outputs: {script_name}")
        
        return contract
        
    except Exception as e:
        print(f"Failed to load contract {script_name}: {e}")
        return None
```

### Batch Contract Loading

```python
# Load multiple contracts
def load_all_contracts(contracts_dir):
    loader = ContractLoader(contracts_dir)
    contracts = {}
    
    contracts_path = Path(contracts_dir)
    for contract_file in contracts_path.glob("*_contract.py"):
        contract_name = contract_file.stem.replace('_contract', '')
        
        try:
            contract = loader.load_contract(contract_file, contract_name)
            contracts[contract_name] = contract
        except Exception as e:
            print(f"Failed to load {contract_name}: {e}")
    
    return contracts
```

## Integration Points

### Contract Discovery Engine

Works with contract discovery for:
- Contract file location and mapping
- Script-contract relationship validation
- Contract existence verification
- Batch contract processing

### Validation Orchestrator

Provides contract loading services to orchestration:
- Dynamic contract loading for validation workflows
- Contract structure validation
- Error handling and recovery
- Integration with validation pipelines

### Script Contract Validator

Integrates with script-contract validation for:
- Contract object extraction and parsing
- Contract structure validation
- Input/output path validation
- Argument and environment variable validation

### Alignment Validation System

Supports alignment validation by:
- Loading contracts for alignment testing
- Converting contracts to validation-friendly formats
- Handling contract loading errors gracefully
- Enabling comprehensive contract analysis

## Benefits

### Robust Loading
- Handles complex project structures and import paths
- Supports multiple contract naming conventions
- Provides comprehensive error handling and recovery
- Manages sys.path modifications safely

### Flexible Contract Support
- Supports various contract object structures
- Handles legacy and modern contract patterns
- Provides fallback mechanisms for edge cases
- Enables extensible contract formats

### Clean Conversion
- Converts complex objects to simple dictionaries
- Preserves all contract metadata and structure
- Provides consistent output format
- Enables easy integration with validation systems

### Import Safety
- Proper sys.path management prevents conflicts
- Cleans up modifications after loading
- Handles relative imports correctly
- Prevents import path pollution

## Error Handling

The contract loader handles various error conditions:

### Import Errors
- Gracefully handles missing dependencies
- Provides detailed error messages with context
- Continues processing other contracts
- Offers debugging information for troubleshooting

### File System Errors
- Handles missing contract files
- Manages file permission issues
- Provides informative error messages
- Supports partial loading results

### Module Loading Errors
- Handles malformed contract files
- Manages import path issues
- Provides detailed error context
- Enables debugging of contract issues

### Contract Structure Errors
- Validates contract object structure
- Handles missing required attributes
- Provides fallback values where appropriate
- Enables graceful degradation

## Performance Considerations

### Efficient Loading
- Minimizes sys.path modifications
- Uses efficient module loading mechanisms
- Caches loaded modules where appropriate
- Optimizes import handling for repeated operations

### Memory Management
- Proper cleanup of sys.path modifications
- Efficient module reference management
- Memory-conscious handling of large contracts
- Garbage collection friendly loading process

### Scalability
- Handles large numbers of contract files
- Supports parallel loading operations
- Efficient batch processing capabilities
- Optimized for repeated loading operations

## Future Enhancements

### Planned Improvements
- Support for contract caching and memoization
- Enhanced error recovery and retry mechanisms
- Integration with external contract formats
- Advanced contract validation during loading
- Support for contract inheritance and composition
- Enhanced debugging and diagnostic capabilities
- Integration with IDE tooling for contract development
- Support for contract versioning and migration

## Conclusion

The `ContractLoader` provides essential functionality for loading and parsing script contracts in the alignment validation system. Its robust import handling and flexible contract support enable reliable contract loading while maintaining compatibility with various contract formats and project structures.
