---
tags:
  - code
  - validation
  - alignment
  - builder
  - configuration
keywords:
  - builder configuration alignment
  - configuration validation
  - step builder validation
  - config field validation
  - builder analysis
  - configuration schema
  - field alignment
  - required fields
topics:
  - validation framework
  - alignment validation
  - builder configuration
  - configuration analysis
language: python
date of note: 2025-08-18
---

# Builder Configuration Alignment

## Overview

The Builder Configuration Alignment Tester validates Level 4 alignment between step builders and their configuration requirements. It ensures builders properly handle configuration fields, validate required parameters, and maintain consistency with configuration schemas.

## Core Functionality

### BuilderConfigurationAlignmentTester Class

The main class orchestrates comprehensive validation of builder-configuration alignment:

```python
class BuilderConfigurationAlignmentTester:
    """
    Tests alignment between step builders and configuration requirements.
    
    Validates:
    - Configuration fields are properly handled
    - Required fields are validated
    - Default values are consistent
    - Configuration schema matches usage
    """
```

### Initialization and Setup

```python
def __init__(self, builders_dir: str, configs_dir: str):
    """Initialize the builder-configuration alignment tester."""
```

**Component Integration:**
- **ConfigurationAnalyzer**: Analyzes configuration files and schemas
- **BuilderCodeAnalyzer**: Parses builder code for configuration usage
- **PatternRecognizer**: Identifies acceptable architectural patterns
- **HybridFileResolver**: Resolves file paths with multiple strategies
- **FlexibleFileResolver**: Provides backward compatibility and fuzzy matching

**Directory Structure:**
```
base_directories = {
    'contracts': builders_dir/parent/contracts,
    'specs': builders_dir/parent/specs,
    'builders': builders_dir,
    'configs': configs_dir
}
```

## Validation Process

### Comprehensive Builder Validation

```python
def validate_all_builders(self, target_scripts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Validate alignment for all builders or specified target scripts."""
```

**Process Flow:**
1. **Discovery Phase**: Identify builders to validate
2. **Individual Validation**: Process each builder separately
3. **Error Handling**: Capture and report validation failures
4. **Result Aggregation**: Compile comprehensive results

### Individual Builder Validation

```python
def validate_builder(self, builder_name: str) -> Dict[str, Any]:
    """Validate alignment for a specific builder."""
```

**Validation Steps:**

#### 1. File Resolution
Uses hybrid approach with multiple fallback strategies:

```python
# Priority order for builder files:
# 1. Standard pattern: builder_{builder_name}_step.py
# 2. FlexibleFileResolver patterns (includes fuzzy matching)
builder_path_str = self._find_builder_file_hybrid(builder_name)

# Priority order for config files:
# 1. Production registry mapping: script_name -> canonical_name -> config_name
# 2. Standard pattern: config_{builder_name}_step.py
# 3. FlexibleFileResolver patterns (includes fuzzy matching)
config_path_str = self._find_config_file_hybrid(builder_name)
```

#### 2. Configuration Analysis
```python
config_analysis = self.config_analyzer.load_config_from_python(config_path, builder_name)
```

**Extracted Information:**
- Configuration class structure
- Field definitions and types
- Required vs optional fields
- Default values
- Validation rules

#### 3. Builder Code Analysis
```python
builder_analysis = self.builder_analyzer.analyze_builder_file(builder_path)
```

**Extracted Information:**
- Configuration field accesses
- Validation logic calls
- Default value assignments
- Import statements
- Class definitions

#### 4. Alignment Validation
Performs multiple validation checks:
- Configuration field handling
- Required field validation
- Configuration import validation

## Validation Checks

### Configuration Field Validation

```python
def _validate_configuration_fields(self, builder_analysis: Dict[str, Any], 
                                 config_analysis: Dict[str, Any], 
                                 builder_name: str) -> List[Dict[str, Any]]:
    """Validate that builder properly handles configuration fields."""
```

**Validation Logic:**

#### Undeclared Field Access Detection
```python
# Get configuration fields from analysis (includes inherited fields)
config_fields = set(config_analysis.get('fields', {}).keys())

# Get fields accessed in builder
accessed_fields = set()
for access in builder_analysis.get('config_accesses', []):
    accessed_fields.add(access['field_name'])

# Check for accessed fields not in configuration
undeclared_fields = accessed_fields - config_fields
```

**Issue Generation:**
```python
{
    'severity': 'ERROR',
    'category': 'configuration_fields',
    'message': f'Builder accesses undeclared configuration field: {field_name}',
    'details': {'field_name': field_name, 'builder': builder_name},
    'recommendation': f'Add {field_name} to configuration class or remove from builder'
}
```

#### Unaccessed Required Fields Detection
```python
# Check for required fields not accessed
unaccessed_required = required_fields - accessed_fields
```

**Issue Generation:**
```python
{
    'severity': 'WARNING',
    'category': 'configuration_fields',
    'message': f'Required configuration field not accessed in builder: {field_name}',
    'details': {'field_name': field_name, 'builder': builder_name},
    'recommendation': f'Access required field {field_name} in builder or make it optional'
}
```

### Pattern-Aware Filtering

```python
def _is_acceptable_pattern(self, field_name: str, builder_name: str, issue_type: str) -> bool:
    """Determine if a configuration field issue represents an acceptable architectural pattern."""
```

Uses the PatternRecognizer component to filter out acceptable architectural patterns:
- **Framework-specific patterns**: XGBoost vs PyTorch configuration differences
- **Step type patterns**: Processing vs Training configuration variations
- **Inheritance patterns**: Base class configuration fields
- **Optional patterns**: Fields that may be conditionally accessed

### Required Field Validation

```python
def _validate_required_fields(self, builder_analysis: Dict[str, Any], 
                            config_analysis: Dict[str, Any], 
                            builder_name: str) -> List[Dict[str, Any]]:
    """Validate that builder properly validates required fields."""
```

**Validation Process:**
1. Extract required fields from configuration analysis
2. Check for validation logic in builder code
3. Report missing validation for required fields

**Issue Generation:**
```python
{
    'severity': 'INFO',
    'category': 'required_field_validation',
    'message': f'Builder has required fields but no explicit validation logic detected',
    'details': {'required_fields': list(required_fields), 'builder': builder_name},
    'recommendation': 'Consider adding explicit validation logic for required configuration fields'
}
```

### Configuration Import Validation

```python
def _validate_config_import(self, builder_analysis: Dict[str, Any], 
                          config_analysis: Dict[str, Any], 
                          builder_name: str) -> List[Dict[str, Any]]:
    """Validate that builder properly imports and uses configuration."""
```

**Validation Logic:**
1. Extract configuration class name from analysis
2. Check for corresponding import in builder code
3. Verify proper usage of configuration class

## File Resolution Strategies

### Hybrid Builder File Resolution

```python
def _find_builder_file_hybrid(self, builder_name: str) -> Optional[str]:
    """Hybrid builder file resolution with multiple fallback strategies."""
```

**Resolution Priority:**
1. **Standard Pattern**: `builder_{builder_name}_step.py`
2. **FlexibleFileResolver**: Known patterns and fuzzy matching
3. **Return None**: If no file found

### Production Registry Integration

```python
def _find_config_file_hybrid(self, builder_name: str) -> Optional[str]:
    """Hybrid config file resolution with production registry integration."""
```

**Resolution Priority:**
1. **Production Registry Mapping**: 
   - `script_name` → `canonical_name` → `config_name`
   - Uses STEP_NAMES registry for accurate mapping
2. **Standard Pattern**: `config_{builder_name}_step.py`
3. **FlexibleFileResolver**: Known patterns and fuzzy matching
4. **Return None**: If no file found

#### Registry-Based Resolution Process

```python
def _get_canonical_step_name(self, script_name: str) -> str:
    """Convert script name to canonical step name using production registry logic."""
```

**Conversion Process:**
1. Parse script name components
2. Handle job type variants (training, validation, testing, calibration)
3. Convert to PascalCase spec_type format
4. Use `get_step_name_from_spec_type()` for canonical mapping

```python
def _get_config_name_from_canonical(self, canonical_name: str) -> str:
    """Get config file base name from canonical step name using production registry."""
```

**Mapping Process:**
1. Look up canonical name in STEP_NAMES registry
2. Extract config_class name
3. Convert CamelCase to snake_case
4. Generate config file name pattern

## Error Handling and Diagnostics

### Missing File Diagnostics

**Builder File Not Found:**
```python
{
    'passed': False,
    'issues': [{
        'severity': 'CRITICAL',
        'category': 'missing_file',
        'message': f'Builder file not found for {builder_name}',
        'details': {
            'searched_patterns': [
                f'builder_{builder_name}_step.py',
                'FlexibleFileResolver patterns',
                'Fuzzy matching'
            ],
            'search_directory': str(self.builders_dir)
        },
        'recommendation': f'Create builder file builder_{builder_name}_step.py'
    }]
}
```

**Configuration File Not Found:**
```python
{
    'passed': False,
    'issues': [{
        'severity': 'ERROR',
        'category': 'missing_configuration',
        'message': f'Configuration file not found for {builder_name}',
        'details': {
            'builder_name': builder_name,
            'search_directory': str(self.configs_dir),
            'available_config_files': config_report.get('discovered_files', []),
            'available_base_names': config_report.get('base_names', []),
            'total_configs_found': config_report.get('count', 0),
            'resolver_strategies': [
                'Exact match',
                'Normalized matching (preprocess↔preprocessing, eval↔evaluation, xgb↔xgboost)',
                'Fuzzy matching (80% similarity threshold)'
            ]
        },
        'recommendation': f'Check if config file exists with correct naming pattern, or create config_{builder_name}_step.py'
    }]
}
```

### Analysis Error Handling

**Configuration Load Errors:**
```python
{
    'passed': False,
    'issues': [{
        'severity': 'CRITICAL',
        'category': 'config_load_error',
        'message': f'Failed to load configuration: {str(e)}',
        'recommendation': 'Fix Python syntax or configuration structure in config file'
    }]
}
```

**Builder Analysis Errors:**
```python
{
    'passed': False,
    'issues': [{
        'severity': 'CRITICAL',
        'category': 'builder_analysis_error',
        'message': f'Failed to analyze builder: {str(e)}',
        'recommendation': 'Fix syntax errors in builder file'
    }]
}
```

## Builder Discovery

```python
def _discover_builders(self) -> List[str]:
    """Discover all builder files in the builders directory."""
```

**Discovery Process:**
1. Scan builders directory for `builder_*_step.py` files
2. Extract builder names from file patterns
3. Filter out system files (starting with `__`)
4. Return sorted list of builder names

**Name Extraction:**
```python
# From builder_example_step.py -> example
if stem.startswith('builder_') and stem.endswith('_step'):
    builder_name = stem[8:-5]  # Remove 'builder_' prefix and '_step' suffix
```

## Integration with Validation Framework

### Result Format

The tester returns standardized validation results:

```python
{
    'passed': bool,                    # Overall pass/fail status
    'issues': List[Dict[str, Any]],    # List of alignment issues
    'builder_analysis': Dict[str, Any], # Builder code analysis results
    'config_analysis': Dict[str, Any]   # Configuration analysis results
}
```

### Issue Severity Levels

- **CRITICAL**: Prevents validation from completing (syntax errors, missing files)
- **ERROR**: Alignment violations that should fail validation
- **WARNING**: Potential issues that may indicate problems
- **INFO**: Informational findings that may be acceptable

### Pass/Fail Determination

```python
# Determine overall pass/fail status
has_critical_or_error = any(
    issue['severity'] in ['CRITICAL', 'ERROR'] for issue in issues
)

return {
    'passed': not has_critical_or_error,
    # ... other fields
}
```

## Best Practices

### Configuration Design

**Clear Field Definitions:**
```python
# Good: Explicit field definition
class ProcessingStepConfig:
    required_fields = ['input_path', 'output_path']
    optional_fields = ['batch_size', 'num_workers']
```

**Consistent Naming:**
```python
# Good: Consistent naming between config and builder
# Config: input_data_path
# Builder: config.input_data_path
```

### Builder Implementation

**Proper Configuration Usage:**
```python
# Good: Access declared configuration fields
def build_step(self, config):
    input_path = config.input_path  # Declared in config
    batch_size = config.batch_size  # Declared in config
```

**Required Field Validation:**
```python
# Good: Validate required fields
def validate_config(self, config):
    if not config.input_path:
        raise ValueError("input_path is required")
```

### File Organization

**Standard Naming Patterns:**
- Builder files: `builder_{name}_step.py`
- Config files: `config_{name}_step.py`
- Consistent naming between builder and config

**Registry Integration:**
- Use production STEP_NAMES registry for canonical mapping
- Ensure config class names match registry expectations
- Handle job type variants consistently

The Builder Configuration Alignment Tester provides essential Level 4 validation capabilities, ensuring that step builders properly handle their configuration requirements and maintain consistency with configuration schemas.
