---
tags:
  - code
  - validation
  - alignment
  - file_resolution
  - discovery
keywords:
  - file resolver
  - dynamic discovery
  - component matching
  - flexible resolution
  - filesystem scanning
  - pattern matching
  - fuzzy matching
  - name normalization
topics:
  - alignment validation
  - file discovery
  - component resolution
  - pattern matching
language: python
date of note: 2025-08-19
---

# File Resolver

## Overview

The `FlexibleFileResolver` class provides intelligent file discovery and matching capabilities to find component files (contracts, specs, builders, configs) for scripts in the alignment validation system. It uses filesystem-driven discovery to eliminate hardcoded mappings that become stale over time.

## Core Components

### FlexibleFileResolver Class

The main resolver class that handles dynamic file resolution with multiple matching strategies.

#### Initialization

```python
def __init__(self, base_directories: Dict[str, str])
```

Initializes the resolver with base directories for different component types:
- `contracts`: Directory containing contract files
- `specs`: Directory containing specification files  
- `builders`: Directory containing builder files
- `configs`: Directory containing configuration files

#### File Discovery

The resolver automatically discovers files using these patterns:
- **Contracts**: `{name}_contract.py`
- **Specs**: `{name}_spec.py`
- **Builders**: `builder_{name}_step.py`
- **Configs**: `config_{name}_step.py`

### Matching Strategies

The resolver uses a three-tier matching strategy:

#### 1. Exact Match
Direct name matching between script name and component base name.

#### 2. Normalized Matching
Handles common name variations:
- `preprocess` ↔ `preprocessing`
- `eval` ↔ `evaluation`
- `xgb` ↔ `xgboost`

#### 3. Fuzzy Matching
Uses `difflib.SequenceMatcher` with 80% similarity threshold for approximate matches.

## Key Methods

### File Finding Methods

```python
def find_contract_file(self, script_name: str) -> Optional[str]
def find_spec_file(self, script_name: str) -> Optional[str]
def find_builder_file(self, script_name: str) -> Optional[str]
def find_config_file(self, script_name: str) -> Optional[str]
```

Find specific component files for a given script name.

```python
def find_all_component_files(self, script_name: str) -> Dict[str, Optional[str]]
```

Returns all component files for a script in a single call.

### Utility Methods

```python
def extract_base_name_from_spec(self, spec_path: Path) -> str
```

Extracts base name from specification file paths, handling job type variants like `preprocessing_training_spec.py` → `preprocessing`.

```python
def find_spec_constant_name(self, script_name: str, job_type: str = 'training') -> Optional[str]
```

Generates expected specification constant names based on discovered patterns.

```python
def refresh_cache(self)
```

Refreshes the file cache to pick up newly added files.

```python
def get_available_files_report(self) -> Dict[str, Dict[str, Any]]
```

Returns detailed report of all discovered files for debugging purposes.

## Implementation Details

### Directory Scanning

The `_scan_directory` method uses regex patterns to extract base names from actual filenames:

```python
patterns = {
    'contracts': r'^(.+)_contract\.py$',
    'specs': r'^(.+)_spec\.py$', 
    'builders': r'^builder_(.+)_step\.py$',
    'configs': r'^config_(.+)_step\.py$'
}
```

### Name Normalization

The `_normalize_name` method handles common variations:

```python
variations = {
    'preprocess': 'preprocessing',
    'eval': 'evaluation',
    'xgb': 'xgboost',
}
```

### Similarity Calculation

Uses `difflib.SequenceMatcher` for fuzzy matching with configurable threshold:

```python
def _calculate_similarity(self, str1: str, str2: str) -> float:
    return difflib.SequenceMatcher(None, str1, str2).ratio()
```

## Usage Examples

### Basic Usage

```python
# Initialize resolver
base_dirs = {
    'contracts': 'src/cursus/steps/contracts',
    'specs': 'src/cursus/steps/specifications',
    'builders': 'src/cursus/steps/builders',
    'configs': 'src/cursus/steps/configs'
}
resolver = FlexibleFileResolver(base_dirs)

# Find specific component
contract_file = resolver.find_contract_file('preprocessing')
spec_file = resolver.find_spec_file('preprocessing')

# Find all components
all_files = resolver.find_all_component_files('preprocessing')
```

### Job Type Handling

```python
# Extract base name from job type variant
spec_path = Path('preprocessing_training_spec.py')
base_name = resolver.extract_base_name_from_spec(spec_path)  # 'preprocessing'

# Generate constant name
constant_name = resolver.find_spec_constant_name('preprocessing', 'training')
# Returns: 'PREPROCESSING_TRAINING_SPEC'
```

### Debugging and Monitoring

```python
# Get discovery report
report = resolver.get_available_files_report()
print(f"Found {report['contracts']['count']} contract files")

# Refresh cache after adding new files
resolver.refresh_cache()
```

## Integration Points

### Alignment Validation System

The file resolver integrates with the alignment validation system to:
- Locate contract files for script-contract alignment validation
- Find specification files for contract-spec alignment validation
- Discover builder files for builder-configuration alignment validation
- Resolve configuration files for dependency validation

### Specification Loader

Works with the `SpecificationLoader` to:
- Find specification files dynamically
- Handle job type variants
- Generate appropriate constant names
- Support fallback resolution strategies

### Validation Orchestrator

Provides file resolution services to:
- Discover available components for validation
- Support batch validation operations
- Enable comprehensive validation coverage
- Handle missing component scenarios gracefully

## Benefits

### Dynamic Discovery
- Eliminates hardcoded file mappings
- Automatically discovers new files
- Reduces maintenance overhead
- Prevents stale configuration issues

### Flexible Matching
- Handles name variations gracefully
- Supports fuzzy matching for typos
- Normalizes common abbreviations
- Provides multiple fallback strategies

### Extensibility
- Easy to add new component types
- Configurable matching patterns
- Pluggable similarity algorithms
- Customizable normalization rules

## Error Handling

The resolver handles various error conditions:
- Missing directories (returns empty results)
- Invalid file patterns (skips malformed files)
- No matches found (returns None gracefully)
- Cache refresh failures (maintains existing cache)

## Performance Considerations

### Caching Strategy
- Files discovered once at initialization
- Cache refreshed only when explicitly requested
- Efficient lookup using dictionary structures
- Minimal filesystem operations during resolution

### Scalability
- Handles large numbers of component files
- Efficient regex-based pattern matching
- Fast similarity calculations using difflib
- Optimized directory scanning with glob patterns

## Future Enhancements

### Planned Improvements
- Support for nested directory structures
- Configurable similarity thresholds
- Advanced pattern matching with wildcards
- Integration with version control systems
- Automatic cache invalidation on file changes
- Support for multiple file extensions
- Enhanced debugging and logging capabilities
