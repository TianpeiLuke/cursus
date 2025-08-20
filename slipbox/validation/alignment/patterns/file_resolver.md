---
tags:
  - code
  - validation
  - alignment
  - patterns
  - file_resolution
keywords:
  - hybrid file resolver
  - file resolution strategies
  - production registry integration
  - fuzzy matching
  - fallback mechanisms
  - builder file resolution
  - config file resolution
topics:
  - validation framework
  - file resolution patterns
  - alignment validation
language: python
date of note: 2025-08-19
---

# Hybrid File Resolution Engine

## Overview

The `HybridFileResolver` class provides advanced file resolution strategies with production registry integration, fuzzy matching, and multiple fallback mechanisms. It combines several resolution approaches to reliably locate builder and configuration files in complex project structures.

## Architecture

### Core Components

1. **Multiple Resolution Strategies**: Combines standard naming conventions, production registry mapping, and flexible pattern matching
2. **Production Registry Integration**: Uses the step names registry for canonical name mapping
3. **Fallback Mechanisms**: Provides graceful degradation when primary strategies fail
4. **Diagnostic Capabilities**: Offers detailed resolution diagnostics for troubleshooting

### Resolution Priority

The resolver uses a hierarchical approach with multiple fallback strategies:

**Builder Files:**
1. Standard pattern: `builder_{builder_name}_step.py`
2. FlexibleFileResolver patterns (includes fuzzy matching)

**Config Files:**
1. Production registry mapping: script_name → canonical_name → config_name
2. Standard pattern: `config_{builder_name}_step.py`
3. FlexibleFileResolver patterns (includes fuzzy matching)

## Implementation Details

### Class Structure

```python
class HybridFileResolver:
    """
    Advanced file resolver with multiple resolution strategies.
    
    Combines:
    - Production registry mapping
    - Standard naming conventions
    - FlexibleFileResolver patterns
    - Fuzzy matching fallbacks
    """
```

### Key Methods

#### `find_builder_file(builder_name: str) -> Optional[str]`

Resolves builder files using hybrid strategy:

1. **Standard Pattern**: Checks for `builder_{builder_name}_step.py`
2. **Flexible Resolution**: Uses FlexibleFileResolver for known patterns and fuzzy matching
3. **Validation**: Ensures resolved path exists before returning

#### `find_config_file(builder_name: str) -> Optional[str]`

Resolves configuration files with production registry integration:

1. **Registry Mapping**: Converts script name to canonical name using production registry
2. **Config Name Derivation**: Maps canonical name to config file name
3. **Standard Fallback**: Falls back to standard naming convention
4. **Flexible Resolution**: Uses FlexibleFileResolver as final fallback

#### `_get_canonical_step_name(script_name: str) -> str`

Converts script names to canonical step names using production registry logic:

- Handles job type variants (training, validation, testing, calibration)
- Converts to PascalCase format for spec_type
- Uses production `get_step_name_from_spec_type` function
- Provides fallback when registry is unavailable

#### `_get_config_name_from_canonical(canonical_name: str) -> str`

Maps canonical step names to config file base names:

- Uses STEP_NAMES registry to find config class name
- Converts config class name to file name format
- Handles CamelCase to snake_case conversion
- Provides fallback when registry is unavailable

#### `get_resolution_diagnostics(builder_name: str) -> Dict[str, Any]`

Provides comprehensive diagnostics for resolution attempts:

- Builder file resolution details
- Config file resolution details
- Registry mapping information
- Available files report
- Final resolution results

### Registry Integration

The resolver integrates with the production step names registry:

```python
from ....steps.registry.step_names import get_step_name_from_spec_type, STEP_NAMES
```

This integration ensures consistency with the production system's mapping logic and enables accurate file resolution based on canonical step names.

### Fallback Mechanisms

The resolver implements robust fallback strategies:

1. **Registry Unavailable**: Falls back to local name conversion logic
2. **File Not Found**: Tries multiple naming patterns
3. **Import Errors**: Gracefully handles missing dependencies
4. **Path Resolution**: Uses flexible pattern matching as final fallback

## Usage Examples

### Basic Usage

```python
# Initialize with base directories
base_directories = {
    'builders': '/path/to/builders',
    'configs': '/path/to/configs'
}
resolver = HybridFileResolver(base_directories)

# Find builder file
builder_path = resolver.find_builder_file('mims_package')
# Returns: '/path/to/builders/builder_mims_package_step.py'

# Find config file
config_path = resolver.find_config_file('model_evaluation_xgb')
# Returns: '/path/to/configs/config_model_eval_step_xgboost.py'
```

### Diagnostic Usage

```python
# Get detailed resolution diagnostics
diagnostics = resolver.get_resolution_diagnostics('mims_package')

print(f"Builder resolution: {diagnostics['builder_resolution']}")
print(f"Config resolution: {diagnostics['config_resolution']}")
print(f"Available files: {diagnostics['available_files']}")
```

### Integration with Validation Framework

```python
# Used within alignment validation
class AlignmentValidator:
    def __init__(self, base_directories):
        self.file_resolver = HybridFileResolver(base_directories)
    
    def validate_step_files(self, step_name):
        builder_path = self.file_resolver.find_builder_file(step_name)
        config_path = self.file_resolver.find_config_file(step_name)
        
        if not builder_path:
            raise ValidationError(f"Builder file not found for {step_name}")
        if not config_path:
            raise ValidationError(f"Config file not found for {step_name}")
        
        return builder_path, config_path
```

## Integration Points

### FlexibleFileResolver Integration

The HybridFileResolver builds upon the FlexibleFileResolver:

- Inherits fuzzy matching capabilities
- Uses flexible pattern recognition
- Leverages available files reporting
- Extends with production registry integration

### Production Registry Integration

Integrates with the step names registry system:

- Uses canonical name mapping
- Leverages config class information
- Maintains consistency with production logic
- Handles job type variants appropriately

### Validation Framework Integration

Serves as a core component in the alignment validation system:

- Provides reliable file resolution for validation
- Supports diagnostic reporting for troubleshooting
- Enables consistent file location across validation levels
- Facilitates error reporting and debugging

## Error Handling

The resolver implements comprehensive error handling:

1. **Import Errors**: Gracefully handles missing registry modules
2. **File Not Found**: Returns None rather than raising exceptions
3. **Registry Errors**: Falls back to local logic when registry fails
4. **Path Errors**: Validates paths before returning results

## Performance Considerations

The resolver is optimized for validation workflows:

- **Lazy Loading**: Only imports registry modules when needed
- **Path Caching**: Leverages underlying FlexibleFileResolver caching
- **Early Returns**: Stops at first successful resolution
- **Minimal Overhead**: Efficient fallback chain execution

## Testing and Validation

The resolver supports comprehensive testing:

- **Diagnostic Mode**: Detailed resolution reporting
- **Fallback Testing**: Each strategy can be tested independently
- **Registry Mocking**: Can operate without production registry
- **Path Validation**: Ensures resolved paths are valid

## Future Enhancements

Potential improvements for the resolver:

1. **Caching Layer**: Add resolution result caching
2. **Pattern Learning**: Learn from successful resolutions
3. **Custom Strategies**: Allow pluggable resolution strategies
4. **Performance Metrics**: Track resolution success rates
5. **Configuration**: Make resolution strategies configurable

## Conclusion

The HybridFileResolver provides a robust, production-ready file resolution system that combines multiple strategies to reliably locate builder and configuration files. Its integration with the production registry system ensures consistency while its fallback mechanisms provide reliability in complex project structures.

The resolver serves as a critical component in the alignment validation framework, enabling accurate file location and supporting comprehensive diagnostic reporting for troubleshooting validation issues.
