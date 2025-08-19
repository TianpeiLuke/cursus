---
tags:
  - code
  - validation
  - alignment
  - contract_discovery
  - discovery_engine
keywords:
  - contract discovery
  - contract mapping
  - entry point mapping
  - contract references
  - script matching
  - contract loading
  - discovery engine
  - contract validation
topics:
  - alignment validation
  - contract discovery
  - file mapping
  - validation infrastructure
language: python
date of note: 2025-08-19
---

# Contract Discovery Engine

## Overview

The `ContractDiscoveryEngine` class handles discovery and mapping of contract files in the alignment validation system. It provides robust contract file discovery using multiple strategies including entry point mapping, specification file contract references, and naming convention patterns.

## Core Components

### ContractDiscoveryEngine Class

The main engine for discovering and mapping contract files to their corresponding scripts.

#### Initialization

```python
def __init__(self, contracts_dir: str)
```

Initializes the discovery engine with the directory containing script contracts.

## Key Methods

### Contract Discovery

```python
def discover_all_contracts(self) -> List[str]
```

Discovers all contract files in the contracts directory using naming convention patterns (`*_contract.py`).

```python
def discover_contracts_with_scripts(self) -> List[str]
```

Discovers contracts that have corresponding scripts by:
1. Loading each contract to extract its `entry_point` field
2. Verifying that the referenced script actually exists
3. Filtering out contracts without corresponding scripts
4. Providing informative logging for skipped contracts

### Entry Point Mapping

```python
def build_entry_point_mapping(self) -> Dict[str, str]
```

Builds a comprehensive mapping from entry_point values (script filenames) to contract filenames:
- Scans all contract files in the directory
- Extracts entry_point from each contract
- Creates bidirectional mapping for efficient lookups
- Caches results for performance

```python
def _extract_entry_point_from_contract(self, contract_path: Path) -> Optional[str]
```

Extracts the entry_point value from a contract file using dynamic module loading:
- Handles relative imports with proper sys.path management
- Supports multiple contract object naming patterns
- Provides robust error handling for malformed contracts

### Specification Integration

```python
def extract_script_contract_from_spec(self, spec_file: Path) -> Optional[str]
```

Extracts the script_contract field from specification files (primary method):
- Dynamically loads specification modules
- Handles complex import paths and package structures
- Extracts entry_point from script_contract objects
- Manages sys.path modifications safely

```python
def extract_contract_reference_from_spec(self, spec_file: Path) -> Optional[str]
```

Extracts contract references from specification files using regex patterns:
- Searches for import statements referencing contracts
- Supports multiple import pattern variations
- Provides fallback when dynamic loading fails

### Contract Matching

```python
def contracts_match(self, contract_from_spec: str, target_contract_name: str) -> bool
```

Determines if contracts match using flexible matching strategies:
- **Direct Match**: Exact string comparison
- **Extension Handling**: Handles `.py` extension variations
- **Prefix Matching**: Supports partial name matches
- **Bidirectional Matching**: Checks both directions for compatibility

## Implementation Details

### Dynamic Module Loading

The engine uses sophisticated dynamic module loading with proper path management:

```python
# Add paths temporarily for imports
paths_to_add = [project_root, src_root, contract_dir]
added_paths = []

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)
        added_paths.append(path)

try:
    # Load and execute module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = 'cursus.steps.contracts'
    spec.loader.exec_module(module)
finally:
    # Clean up sys.path
    for path in added_paths:
        if path in sys.path:
            sys.path.remove(path)
```

### Contract Object Detection

Supports multiple naming patterns for contract objects:

```python
possible_names = [
    f"{contract_name.upper()}_CONTRACT",
    f"{contract_name}_CONTRACT", 
    f"{contract_name}_contract",
    "MODEL_EVALUATION_CONTRACT",  # Specific patterns
    "CONTRACT",
    "contract"
]

# Also dynamically discover _CONTRACT suffixed attributes
for attr_name in dir(module):
    if attr_name.endswith('_CONTRACT') and not attr_name.startswith('_'):
        possible_names.append(attr_name)
```

### Error Handling and Logging

Provides comprehensive error handling with informative logging:

```python
# Informative logging for skipped contracts
print(f"ℹ️  Skipping contract '{contract_name}' - script '{script_name}' not found")
print(f"ℹ️  Skipping contract '{contract_name}' - no entry_point defined")
print(f"⚠️  Skipping contract '{contract_name}' - failed to load: {str(e)}")
```

## Usage Examples

### Basic Contract Discovery

```python
# Initialize discovery engine
discovery = ContractDiscoveryEngine('src/cursus/steps/contracts')

# Discover all contracts
all_contracts = discovery.discover_all_contracts()
print(f"Found contracts: {all_contracts}")

# Discover only contracts with corresponding scripts
valid_contracts = discovery.discover_contracts_with_scripts()
print(f"Valid contracts: {valid_contracts}")
```

### Entry Point Mapping

```python
# Build entry point mapping
mapping = discovery.build_entry_point_mapping()
print(f"Entry point mapping: {mapping}")

# Example mapping:
# {
#     'preprocessing.py': 'preprocessing_contract.py',
#     'training_xgb.py': 'training_contract.py',
#     'model_evaluation.py': 'model_evaluation_contract.py'
# }
```

### Specification Integration

```python
from pathlib import Path

# Extract contract from specification
spec_file = Path('src/cursus/steps/specifications/preprocessing_spec.py')
contract_ref = discovery.extract_script_contract_from_spec(spec_file)
print(f"Contract reference: {contract_ref}")

# Check contract matching
matches = discovery.contracts_match('preprocessing', 'preprocessing_contract')
print(f"Contracts match: {matches}")
```

### Integration with Validation

```python
# Use with unified alignment tester
from ..unified_alignment_tester import UnifiedAlignmentTester

tester = UnifiedAlignmentTester()
discovery = ContractDiscoveryEngine('src/cursus/steps/contracts')

# Get contracts that can be validated
valid_contracts = discovery.discover_contracts_with_scripts()

# Run validation only for contracts with scripts
for contract_name in valid_contracts:
    results = tester.run_alignment_validation(contract_name)
    print(f"Validation results for {contract_name}: {results}")
```

## Integration Points

### Unified Alignment Tester

Works closely with the `UnifiedAlignmentTester`:
- Provides list of valid contracts for validation
- Ensures only contracts with corresponding scripts are tested
- Prevents validation errors from missing scripts
- Supports efficient batch validation operations

### File Resolver

Complements the `FlexibleFileResolver`:
- Provides contract-specific discovery logic
- Handles complex contract-script relationships
- Supports dynamic contract loading and validation
- Enables comprehensive file mapping strategies

### Specification Loader

Integrates with specification loading:
- Extracts contract references from specifications
- Validates contract-specification relationships
- Supports job type variant handling
- Enables cross-component validation

## Benefits

### Robust Discovery
- Multiple discovery strategies for maximum coverage
- Handles various naming conventions and patterns
- Provides fallback mechanisms for edge cases
- Supports both static and dynamic analysis

### Intelligent Filtering
- Only discovers contracts with corresponding scripts
- Prevents validation errors from orphaned contracts
- Provides clear logging for skipped contracts
- Enables efficient validation workflows

### Flexible Matching
- Supports multiple contract matching strategies
- Handles naming variations and edge cases
- Provides bidirectional matching capabilities
- Enables robust contract-script relationships

### Performance Optimization
- Caches entry point mappings for efficiency
- Uses lazy loading for expensive operations
- Minimizes filesystem operations
- Provides efficient batch discovery operations

## Error Handling

The discovery engine handles various error conditions:

### Import Errors
- Gracefully handles missing dependencies
- Provides fallback discovery methods
- Logs import issues for debugging
- Continues processing other contracts

### File System Errors
- Handles missing directories gracefully
- Manages file permission issues
- Provides informative error messages
- Supports partial discovery results

### Module Loading Errors
- Handles malformed contract files
- Manages import path issues
- Provides detailed error context
- Enables debugging of contract issues

## Performance Considerations

### Caching Strategy
- Entry point mappings cached after first build
- Module loading results cached when possible
- Efficient lookup structures for repeated queries
- Minimal redundant filesystem operations

### Memory Management
- Temporary sys.path modifications cleaned up properly
- Module references managed to prevent memory leaks
- Efficient data structures for large contract sets
- Lazy loading of expensive discovery operations

### Scalability
- Handles large numbers of contract files efficiently
- Supports parallel discovery operations
- Optimized regex patterns for specification parsing
- Efficient batch processing capabilities

## Future Enhancements

### Planned Improvements
- Support for nested contract directory structures
- Enhanced contract metadata extraction
- Integration with version control systems
- Advanced contract dependency analysis
- Automated contract-script relationship validation
- Support for custom contract naming conventions
- Integration with external contract registries
- Enhanced debugging and diagnostic capabilities
