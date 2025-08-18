---
tags:
  - code
  - validation
  - alignment
  - configuration
  - analysis
keywords:
  - configuration analysis
  - Pydantic configuration
  - field detection
  - type annotation analysis
  - default value detection
  - required field classification
  - configuration schema
  - field analysis
topics:
  - validation framework
  - configuration analysis
  - field validation
  - schema analysis
language: python
date of note: 2025-08-18
---

# Configuration Analyzer

## Overview

The Configuration Analyzer provides comprehensive analysis of configuration classes, extracting field information, types, defaults, and requirements. It supports both Pydantic v1 and v2 configurations with robust field detection and handles complex inheritance hierarchies.

## Core Functionality

### ConfigurationAnalyzer Class

The main analyzer class that orchestrates configuration analysis:

```python
class ConfigurationAnalyzer:
    """
    Analyzes configuration classes to extract comprehensive field information.
    
    Supports:
    - Pydantic v1 and v2 configurations
    - Type annotation analysis
    - Default value detection
    - Required/optional field classification
    """
```

### Initialization

```python
def __init__(self, configs_dir: str):
    """
    Initialize the configuration analyzer.
    
    Args:
        configs_dir: Directory containing configuration files
    """
```

**Setup:**
- **Directory Management**: Store configuration directory path
- **Path Resolution**: Handle configuration file discovery
- **Import Context**: Prepare for dynamic module loading

## Configuration Loading

### Python File Loading

```python
def load_config_from_python(self, config_path: Path, builder_name: str) -> Dict[str, Any]:
    """
    Load configuration from Python file with robust import handling.
    
    Args:
        config_path: Path to the configuration file
        builder_name: Name of the builder (for class name inference)
        
    Returns:
        Configuration analysis dictionary
    """
```

**Loading Process:**

#### 1. Dynamic Module Import
```python
# Try to import the module directly
module_name = f"config_{builder_name}_step"

# Add both the configs directory and the project root to sys.path temporarily
configs_dir_str = str(self.configs_dir)
project_root_str = str(self.configs_dir.parent.parent.parent)
```

**Import Strategy:**
- **Path Management**: Temporarily add necessary paths to sys.path
- **Module Specification**: Create module spec from file location
- **Package Setup**: Configure module package for relative imports
- **Module Registration**: Register module in sys.modules with correct package structure

#### 2. Configuration Class Discovery

**Strategy 1: Registry-Based Discovery**
```python
# Use step registry to get the correct config class name
from src.cursus.steps.registry.step_names import get_canonical_name_from_file_name, get_config_class_name

# Get canonical step name from builder name (script name)
canonical_name = get_canonical_name_from_file_name(builder_name)

# Get the correct config class name from registry
registry_config_class_name = get_config_class_name(canonical_name)
```

**Strategy 2: Pattern Matching Fallback**
```python
possible_names = [
    f"{builder_name.title().replace('_', '')}Config",
    f"{''.join(word.capitalize() for word in builder_name.split('_'))}Config",
    f"{''.join(word.capitalize() for word in builder_name.split('_'))}StepConfig",
    f"CurrencyConversionConfig",  # Specific case
    f"DummyTrainingConfig",       # Specific case
    f"BatchTransformStepConfig",  # Specific case
    f"XGBoostModelEvalConfig"     # Specific case for xgboost_model_evaluation
]
```

**Discovery Features:**
- **Registry Integration**: Use production step registry for accurate class name mapping
- **Pattern Matching**: Fallback to common naming patterns
- **Specific Cases**: Handle known special cases
- **Error Handling**: Graceful degradation with informative error messages

#### 3. Error Handling and Fallback

```python
except Exception as e:
    # Return a simplified analysis if we can't load the module
    return {
        'class_name': f"{builder_name}Config",
        'fields': {},
        'required_fields': set(),
        'optional_fields': set(),
        'default_values': {},
        'load_error': str(e)
    }
```

## Configuration Class Analysis

### Comprehensive Field Analysis

```python
def analyze_config_class(self, config_class, class_name: str) -> Dict[str, Any]:
    """
    Analyze configuration class to extract comprehensive field information.
    
    Args:
        config_class: The configuration class to analyze
        class_name: Name of the configuration class
        
    Returns:
        Dictionary containing field analysis results
    """
```

**Analysis Categories:**
- **Field Definitions**: All field names and types
- **Required Fields**: Fields that must be provided
- **Optional Fields**: Fields with defaults or marked optional
- **Default Values**: Default values for optional fields

### Inheritance-Aware Field Detection

```python
# Get all annotations from the class hierarchy (including inherited fields)
all_annotations = {}

# Walk through the MRO (Method Resolution Order) to get all annotations
for cls in reversed(config_class.__mro__):
    if hasattr(cls, '__annotations__'):
        all_annotations.update(cls.__annotations__)
```

**Inheritance Features:**
- **MRO Traversal**: Walk Method Resolution Order for complete field discovery
- **Annotation Merging**: Combine annotations from all parent classes
- **Override Handling**: Properly handle field overrides in subclasses
- **Complete Coverage**: Ensure no inherited fields are missed

### Pydantic Version Support

#### Pydantic v2 Support
```python
# Check for Pydantic model fields (v2 style) - includes inherited fields
if hasattr(config_class, 'model_fields'):
    for field_name, field_info in config_class.model_fields.items():
        # Update the required status based on Pydantic field info
        if hasattr(field_info, 'is_required'):
            is_required = field_info.is_required()
```

#### Pydantic v1 Support
```python
# Check for Pydantic v1 style fields
elif hasattr(config_class, '__fields__'):
    for field_name, field_info in config_class.__fields__.items():
        # Update the required status based on Pydantic field info
        if hasattr(field_info, 'required'):
            is_required = field_info.required
```

**Version Compatibility:**
- **Automatic Detection**: Detect Pydantic version automatically
- **Field Info Extraction**: Extract field information using version-appropriate methods
- **Required Status**: Determine required/optional status correctly for each version
- **Default Values**: Extract default values using version-specific approaches

### Advanced Field Detection

#### Property Detection
```python
# Check for properties and other attributes (including inherited)
for attr_name in dir(config_class):
    if not attr_name.startswith('_'):
        attr_value = getattr(config_class, attr_name, None)
        
        # Check if it's a property
        if isinstance(attr_value, property):
            # Add property as an optional field if not already present
            if attr_name not in analysis['fields']:
                analysis['fields'][attr_name] = {
                    'type': 'property',
                    'required': False  # Properties are typically computed, so optional
                }
```

#### Default Value Detection
```python
# Check for default values (non-callable, non-descriptor attributes)
elif (attr_name in analysis['fields'] and 
      not callable(attr_value) and 
      not hasattr(attr_value, '__get__')):  # Skip descriptors
    analysis['default_values'][attr_name] = attr_value
    # If a field has a default value, it's optional
    if attr_name in analysis['required_fields']:
        analysis['required_fields'].remove(attr_name)
```

## Optional Field Detection

### Sophisticated Optional Detection

```python
def _is_optional_field(self, field_type, field_name: str, config_class) -> bool:
    """
    Determine if a field is optional based on its type annotation and Field definition.
    
    Supports both Pydantic v1 and v2 field detection with comprehensive type analysis.
    """
```

**Detection Priority:**

#### 1. Pydantic Field Info (Highest Priority)
```python
# First priority: Check Pydantic field info for definitive answer
if hasattr(config_class, 'model_fields'):
    # Pydantic v2 style
    field_info = config_class.model_fields.get(field_name)
    if field_info and hasattr(field_info, 'is_required'):
        return not field_info.is_required()
elif hasattr(config_class, '__fields__'):
    # Pydantic v1 style
    field_info = config_class.__fields__.get(field_name)
    if field_info and hasattr(field_info, 'required'):
        return not field_info.required
```

#### 2. Type Annotation Analysis
```python
# Second priority: Check for Optional[Type] or Union[Type, None] patterns
type_str = str(field_type)
if 'Optional[' in type_str or 'Union[' in type_str:
    if hasattr(typing, 'get_origin') and hasattr(typing, 'get_args'):
        origin = typing.get_origin(field_type)
        if origin is typing.Union:
            args = typing.get_args(field_type)
            # Check if None is one of the union types
            if type(None) in args:
                return True
```

#### 3. Default Value Detection
```python
# Third priority: Check if the field has a class-level default value
if hasattr(config_class, field_name):
    default_value = getattr(config_class, field_name)
    # If it's not a callable (method) and not a Field descriptor, it's a default
    if not callable(default_value):
        return True
```

**Detection Features:**
- **Multi-Level Analysis**: Use multiple detection strategies with priority order
- **Type System Integration**: Leverage Python's typing system for Optional detection
- **Union Type Handling**: Properly handle Union[Type, None] patterns
- **Default Value Recognition**: Detect class-level default values

## Schema Generation

### Standardized Schema Output

```python
def get_configuration_schema(self, config_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract configuration schema in a standardized format.
    
    Args:
        config_analysis: Result from analyze_config_class
        
    Returns:
        Standardized configuration schema
    """
```

**Schema Structure:**
```python
return {
    'configuration': {
        'required': config_analysis.get('required_fields', []),
        'optional': config_analysis.get('optional_fields', []),
        'fields': config_analysis.get('fields', {}),
        'defaults': config_analysis.get('default_values', {})
    }
}
```

## Analysis Results Structure

### Comprehensive Analysis Output

```python
analysis = {
    'class_name': 'ProcessingStepConfig',
    'fields': {
        'input_path': {
            'type': 'str',
            'required': True
        },
        'output_path': {
            'type': 'str', 
            'required': True
        },
        'batch_size': {
            'type': 'Optional[int]',
            'required': False
        }
    },
    'required_fields': ['input_path', 'output_path'],
    'optional_fields': ['batch_size'],
    'default_values': {
        'batch_size': 32
    }
}
```

### Error Handling Results

```python
# When loading fails
{
    'class_name': 'example_builderConfig',
    'fields': {},
    'required_fields': set(),
    'optional_fields': set(),
    'default_values': {},
    'load_error': 'Configuration class not found in /path/to/config.py'
}
```

## Integration with Alignment Validation

### Builder Configuration Alignment

The analyzer integrates with BuilderConfigurationAlignmentTester:

```python
# Load configuration using extracted component
config_analysis = self.config_analyzer.load_config_from_python(config_path, builder_name)

# Get configuration fields from analysis (now includes inherited fields)
config_fields = set(config_analysis.get('fields', {}).keys())
required_fields = set(config_analysis.get('required_fields', []))
```

### Field Validation Support

```python
# Validate configuration field handling
config_fields = set(config_analysis.get('fields', {}).keys())
required_fields_raw = config_analysis.get('required_fields', [])
if isinstance(required_fields_raw, list):
    required_fields = set(required_fields_raw)
else:
    required_fields = set(required_fields_raw)
```

## Error Handling and Robustness

### Import Error Handling

**Path Management:**
```python
try:
    # Add paths temporarily
    paths_to_add = []
    if configs_dir_str not in sys.path:
        sys.path.insert(0, configs_dir_str)
        paths_to_add.append(configs_dir_str)
finally:
    # Clean up sys.path
    for path in paths_to_add:
        if path in sys.path:
            sys.path.remove(path)
```

### Class Discovery Fallback

**Multiple Strategies:**
- **Registry-First**: Use production registry for accurate mapping
- **Pattern Matching**: Fall back to common naming patterns
- **Specific Cases**: Handle known special configurations
- **Graceful Degradation**: Return meaningful error information

### Analysis Error Recovery

**Defensive Programming:**
- **Exception Isolation**: Isolate failures to specific analysis steps
- **Partial Results**: Return partial analysis when possible
- **Error Documentation**: Include error information in results
- **Fallback Values**: Provide sensible defaults for failed analysis

## Best Practices

### Configuration Design

**Clear Field Definitions:**
```python
# Good: Explicit field types and requirements
class ProcessingStepConfig(BaseModel):
    input_path: str  # Required
    output_path: str  # Required
    batch_size: Optional[int] = 32  # Optional with default
```

**Inheritance Usage:**
```python
# Good: Proper inheritance with field overrides
class AdvancedProcessingConfig(ProcessingStepConfig):
    advanced_mode: bool = False
    optimization_level: Optional[int] = None
```

### Registry Integration

**Canonical Naming:**
```python
# Good: Use registry for consistent class naming
canonical_name = get_canonical_name_from_file_name(builder_name)
config_class_name = get_config_class_name(canonical_name)
```

**File Organization:**
```python
# Good: Standard naming patterns
# File: config_processing_step.py
# Class: ProcessingStepConfig
```

The Configuration Analyzer provides essential analysis capabilities for understanding configuration class structure and supporting comprehensive Level 4 alignment validation through detailed field analysis and robust configuration loading.
