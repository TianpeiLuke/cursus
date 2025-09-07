---
tags:
  - code
  - core
  - config_fields
  - field_categorization
  - configuration
keywords:
  - ConfigFieldCategorizer
  - field categorization
  - shared fields
  - specific fields
  - configuration analysis
  - declarative rules
topics:
  - configuration management
  - field categorization
  - rule-based processing
language: python
date of note: 2025-09-07
---

# Configuration Field Categorizer

Rule-based categorizer for organizing configuration fields across multiple configurations, implementing the Declarative Over Imperative principle with explicit rules.

## Overview

The `config_field_categorizer` module provides a rule-based categorizer for configuration fields, implementing the Declarative Over Imperative principle with explicit rules. The categorizer analyzes field values and metadata across multiple configuration objects to determine proper placement in either shared or specific sections.

The module implements a simplified categorization structure with just shared and specific sections, using explicit rules with clear precedence for categorization decisions. It analyzes field characteristics such as whether they are special fields, appear in multiple configurations, have different values across configurations, or are likely static vs. dynamic.

## Classes and Methods

### Classes
- [`ConfigFieldCategorizer`](#configfieldcategorizer) - Main categorizer for organizing configuration fields

## API Reference

### ConfigFieldCategorizer

_class_ cursus.core.config_fields.config_field_categorizer.ConfigFieldCategorizer(_config_list_, _processing_step_config_base_class=None_)

Responsible for categorizing configuration fields based on their characteristics. Analyzes field values and metadata across configs to determine proper placement. Uses explicit rules with clear precedence for categorization decisions, implementing the Declarative Over Imperative principle.

**Parameters:**
- **config_list** (_List[Any]_) – List of configuration objects to analyze
- **processing_step_config_base_class** (_Optional[type]_) – Base class for processing step configs

```python
from cursus.core.config_fields.config_field_categorizer import ConfigFieldCategorizer

# Create categorizer with list of config objects
configs = [config1, config2, config3]
categorizer = ConfigFieldCategorizer(configs)

# Get categorization results
categorized_fields = categorizer.get_categorized_fields()
```

#### get_category_for_field

get_category_for_field(_field_name_, _config=None_)

Get the category for a specific field, optionally in a specific config.

**Parameters:**
- **field_name** (_str_) – Name of the field
- **config** (_Optional[Any]_) – Optional config instance

**Returns:**
- **Optional[CategoryType]** – Category for the field or None if field not found

```python
from cursus.core.config_fields.constants import CategoryType

# Get category for a specific field
category = categorizer.get_category_for_field("instance_type")
if category == CategoryType.SHARED:
    print("Field is shared across all configs")
elif category == CategoryType.SPECIFIC:
    print("Field is specific to certain configs")
```

#### get_categorized_fields

get_categorized_fields()

Get the categorization result.

**Returns:**
- **Dict[str, Any]** – Field categorization with 'shared' and 'specific' sections

```python
# Get complete categorization results
categorization = categorizer.get_categorized_fields()

# Access shared fields
shared_fields = categorization['shared']
print(f"Shared fields: {list(shared_fields.keys())}")

# Access specific fields by step
specific_fields = categorization['specific']
for step_name, fields in specific_fields.items():
    print(f"Step {step_name} has {len(fields)} specific fields")
```

#### print_categorization_stats

print_categorization_stats()

Print statistics about field categorization for the simplified structure.

```python
# Print categorization statistics
categorizer.print_categorization_stats()
# Output:
# Field categorization statistics:
#   Shared: 15 (25.0%)
#   Specific: 45 (75.0%)
#   Total: 60
```

## Related Documentation

- [Configuration Constants](constants.md) - Defines CategoryType enum and field patterns used by categorizer
- [Configuration Merger](config_merger.md) - Primary consumer of categorization results
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md) - Used for serializing config objects during analysis
- [Configuration Fields Overview](README.md) - System overview and integration
