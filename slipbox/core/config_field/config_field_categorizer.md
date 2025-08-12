---
tags:
  - code
  - core
  - config_field_categorizer
  - field_analysis
  - rule_based_categorization
keywords:
  - configuration field categorizer
  - field analysis
  - rule-based categorization
  - shared vs specific
  - special field detection
  - static field analysis
  - cross-type identification
  - declarative rules
topics:
  - configuration field categorization
  - field analysis algorithms
  - rule-based classification
  - configuration management
language: python
date of note: 2025-08-12
---

# Configuration Field Categorizer

## Overview

The `ConfigFieldCategorizer` is responsible for analyzing configuration fields across multiple configuration objects and categorizing them based on their characteristics. It implements a rule-based approach with explicit precedence for categorization decisions, following the **Declarative Over Imperative** principle.

## Class Definition

```python
class ConfigFieldCategorizer:
    """
    Responsible for categorizing configuration fields based on their characteristics.
    
    Analyzes field values and metadata across configs to determine proper placement.
    Uses explicit rules with clear precedence for categorization decisions,
    implementing the Declarative Over Imperative principle.
    
    Implements simplified categorization structure with just shared and specific sections.
    """
    
    def __init__(self, config_list: List[Any], processing_step_config_base_class: Optional[type] = None):
        """
        Initialize with list of config objects to categorize.
        
        Args:
            config_list: List of configuration objects to analyze
            processing_step_config_base_class: Base class for processing step configs
        """
```

## Key Design Principles

### 1. Declarative Over Imperative

The categorizer uses explicit rules with clear precedence rather than complex imperative logic:

```python
def _categorize_field(self, field_name: str) -> CategoryType:
    """
    Determine the category for a field based on simplified explicit rules.
    """
    info = self.field_info
    
    # Rule 1: Special fields always go to specific sections
    if info['is_special'][field_name]:
        return CategoryType.SPECIFIC
            
    # Rule 2: Fields that only appear in one config are specific
    if len(info['sources'][field_name]) <= 1:
        return CategoryType.SPECIFIC
            
    # Rule 3: Fields with different values across configs are specific
    if len(info['values'][field_name]) > 1:
        return CategoryType.SPECIFIC
            
    # Rule 4: Non-static fields are specific
    if not info['is_static'][field_name]:
        return CategoryType.SPECIFIC
            
    # Rule 5: Fields with identical values across all configs go to shared
    if len(info['sources'][field_name]) == len(self.config_list) and len(info['values'][field_name]) == 1:
        return CategoryType.SHARED
        
    # Default case: if we can't determine clearly, be safe and make it specific
    return CategoryType.SPECIFIC
```

### 2. Single Source of Truth

All field information is collected once and used consistently throughout the categorization process:

```python
def _collect_field_info(self) -> Dict[str, Any]:
    """
    Collect comprehensive information about all fields across configs.
    
    Implements the Single Source of Truth principle by gathering all information
    in one place for consistent categorization decisions.
    """
    field_info = {
        'values': defaultdict(set),             # field_name -> set of values (as JSON strings)
        'sources': defaultdict(list),           # field_name -> list of step names
        'processing_sources': defaultdict(list), # field_name -> list of processing step names
        'non_processing_sources': defaultdict(list), # field_name -> list of non-processing step names
        'is_static': defaultdict(bool),         # field_name -> bool (is this field likely static)
        'is_special': defaultdict(bool),        # field_name -> bool (is this a special field)
        'is_cross_type': defaultdict(bool),     # field_name -> bool (appears in both processing/non-processing)
        'raw_values': defaultdict(dict)         # field_name -> {step_name: actual value}
    }
```

### 3. Type-Safe Specifications

The categorizer uses enum-based category types instead of string literals:

```python
from .constants import CategoryType

# Returns CategoryType.SHARED or CategoryType.SPECIFIC
category = self._categorize_field(field_name)
```

## Core Functionality

### Field Information Collection

The categorizer collects comprehensive information about all fields:

```python
def _collect_field_info(self) -> Dict[str, Any]:
    """
    Collect comprehensive information about all fields across configs.
    """
    # For each configuration object:
    for config in self.config_list:
        serialized = serialize_config(config)
        step_name = self._extract_step_name(serialized, config)
        
        # For each field in the configuration:
        for field_name, value in serialized.items():
            if field_name == "_metadata":
                continue
                
            # Track raw value
            field_info['raw_values'][field_name][step_name] = value
            
            # Track serialized value for comparison
            value_str = self._serialize_value_for_comparison(value)
            field_info['values'][field_name].add(value_str)
            
            # Track sources and categorize by processing type
            field_info['sources'][field_name].append(step_name)
            self._categorize_by_processing_type(config, field_name, step_name, field_info)
            
            # Analyze field characteristics
            field_info['is_special'][field_name] = self._is_special_field(field_name, value, config)
            field_info['is_static'][field_name] = self._is_likely_static(field_name, value)
```

### Special Field Detection

Special fields are always placed in specific sections:

```python
def _is_special_field(self, field_name: str, value: Any, config: Any) -> bool:
    """
    Determine if a field should be treated as special.
    
    Special fields are always kept in specific sections.
    """
    # Check against known special fields
    if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        return True
        
    # Check if it's a Pydantic model
    if isinstance(value, BaseModel):
        return True
        
    # Check for fields with nested complex structures
    if isinstance(value, dict) and any(isinstance(v, (dict, list)) for v in value.values()):
        return True
        
    return False
```

### Static Field Analysis

Static fields are those that don't change at runtime:

```python
def _is_likely_static(self, field_name: str, value: Any, config=None) -> bool:
    """
    Determine if a field is likely static based on name and value.
    """
    # Fields in the exceptions list are considered static
    if field_name in NON_STATIC_FIELD_EXCEPTIONS:
        return True
        
    # Special fields are never static
    if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        return False
        
    # Pydantic models are never static
    if isinstance(value, BaseModel):
        return False
    
    # Check name patterns that suggest non-static fields
    if any(pattern in field_name for pattern in NON_STATIC_FIELD_PATTERNS):
        return False
        
    # Check complex values
    if isinstance(value, dict) and len(value) > 3:
        return False
    if isinstance(value, list) and len(value) > 5:
        return False
        
    # Default to static
    return True
```

## Categorization Rules

The categorizer applies five explicit rules in order of precedence:

### Rule 1: Special Fields → Specific

Special fields always go to specific sections regardless of other characteristics:

- Fields in `SPECIAL_FIELDS_TO_KEEP_SPECIFIC` constant
- Pydantic model instances
- Complex nested structures (dictionaries with nested objects)

### Rule 2: Single-Config Fields → Specific

Fields that appear in only one configuration are always specific:

```python
if len(info['sources'][field_name]) <= 1:
    return CategoryType.SPECIFIC
```

### Rule 3: Multi-Value Fields → Specific

Fields with different values across configurations are specific:

```python
if len(info['values'][field_name]) > 1:
    return CategoryType.SPECIFIC
```

### Rule 4: Non-Static Fields → Specific

Fields that are likely to change at runtime are specific:

```python
if not info['is_static'][field_name]:
    return CategoryType.SPECIFIC
```

### Rule 5: Identical Shared Fields → Shared

Fields with identical values across all configurations go to shared:

```python
if len(info['sources'][field_name]) == len(self.config_list) and len(info['values'][field_name]) == 1:
    return CategoryType.SHARED
```

### Default Rule: Safe Fallback → Specific

If categorization is unclear, fields are placed in specific sections for safety.

## Output Structure

The categorizer produces a simplified structure with two main sections:

```python
categorization = {
    'shared': {
        'field1': 'common_value1',
        'field2': 'common_value2'
    },
    'specific': {
        'step_name_1': {
            'field3': 'specific_value1',
            'field4': 'specific_value2'
        },
        'step_name_2': {
            'field5': 'specific_value3',
            'field6': 'specific_value4'
        }
    }
}
```

## Usage Examples

### Basic Categorization

```python
from src.cursus.core.config_fields.config_field_categorizer import ConfigFieldCategorizer

# Create list of configuration objects
config_list = [data_config, preprocessing_config, training_config]

# Create categorizer
categorizer = ConfigFieldCategorizer(config_list)

# Get categorization results
categorized_fields = categorizer.get_categorized_fields()

# Access shared fields
shared_fields = categorized_fields['shared']
print(f"Shared fields: {list(shared_fields.keys())}")

# Access specific fields
specific_fields = categorized_fields['specific']
for step_name, fields in specific_fields.items():
    print(f"Step {step_name}: {list(fields.keys())}")
```

### Field Category Lookup

```python
# Check category for a specific field
category = categorizer.get_category_for_field('processing_instance_count')
print(f"Field category: {category}")

# Check category for a field in a specific config
category = categorizer.get_category_for_field('image_uri', training_config)
print(f"Field category in training config: {category}")
```

### Statistics and Analysis

```python
# Print categorization statistics
categorizer.print_categorization_stats()

# Output:
# Field categorization statistics:
#   Shared: 15 (25.0%)
#   Specific: 45 (75.0%)
#   Total: 60
```

## Processing vs Non-Processing Configurations

The categorizer distinguishes between processing and non-processing configurations:

```python
def __init__(self, config_list: List[Any], processing_step_config_base_class: Optional[type] = None):
    # Determine the base class for processing steps
    self.processing_base_class = processing_step_config_base_class
    if self.processing_base_class is None:
        try:
            from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase
            self.processing_base_class = ProcessingStepConfigBase
        except ImportError:
            self.processing_base_class = object
    
    # Categorize configs
    self.processing_configs = [c for c in config_list 
                              if isinstance(c, self.processing_base_class)]
    self.non_processing_configs = [c for c in config_list 
                                  if not isinstance(c, self.processing_base_class)]
```

This distinction helps identify cross-type fields that appear in both processing and non-processing configurations.

## Cross-Type Field Detection

The categorizer identifies fields that appear in both processing and non-processing configurations:

```python
# Track processing/non-processing sources
if isinstance(config, self.processing_base_class):
    field_info['processing_sources'][field_name].append(step_name)
else:
    field_info['non_processing_sources'][field_name].append(step_name)

# Determine if cross-type
is_processing = bool(field_info['processing_sources'][field_name])
is_non_processing = bool(field_info['non_processing_sources'][field_name])
field_info['is_cross_type'][field_name] = is_processing and is_non_processing
```

## Error Handling and Validation

The categorizer includes comprehensive error handling:

### Serialization Errors

```python
try:
    value_str = json.dumps(value, sort_keys=True)
    field_info['values'][field_name].add(value_str)
except (TypeError, ValueError):
    # If not JSON serializable, use object ID as placeholder
    field_info['values'][field_name].add(f"__non_serializable_{id(value)}__")
```

### Missing Metadata

```python
if "_metadata" not in serialized:
    self.logger.warning(f"Config {config.__class__.__name__} does not have _metadata. "
                       "Using class name as step name.")
    step_name = config.__class__.__name__
else:
    step_name = serialized["_metadata"].get("step_name", config.__class__.__name__)
```

### Invalid Serialized Data

```python
if not isinstance(serialized, dict):
    self.logger.warning(f"Serialized config for {config.__class__.__name__} is not a dictionary, got {type(serialized)}")
    continue
```

## Performance Considerations

### Memory Efficiency

- Field information is collected once and reused
- JSON serialization is used for value comparison to handle complex objects
- Defaultdict is used to avoid key existence checks

### Processing Efficiency

- Early termination in categorization rules
- Batch processing of all configurations
- Optimized field comparison using serialized values

## Integration with Other Components

### Configuration Merger

The categorizer provides categorization results to the merger:

```python
# In ConfigMerger.__init__:
self.categorizer = ConfigFieldCategorizer(config_list, processing_step_config_base_class)

# In ConfigMerger.merge():
categorized = self.categorizer.get_categorized_fields()
```

### Type-Aware Serializer

The categorizer uses the serializer for consistent data handling:

```python
from .type_aware_config_serializer import serialize_config

serialized = serialize_config(config)
```

## Related Documentation

- [Configuration Merger](config_merger.md): Uses categorization results for merging
- [Configuration Constants](config_constants.md): Defines categorization rules and patterns
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md): Handles serialization
- [Configuration Fields Overview](README.md): System overview and integration
