---
tags:
  - code
  - core
  - config_fields
  - tier_registry
  - three_tier_pattern
keywords:
  - ConfigFieldTierRegistry
  - tier classification
  - three-tier pattern
  - essential fields
  - system fields
  - derived fields
topics:
  - configuration management
  - three-tier architecture
  - field classification
language: python
date of note: 2025-09-07
---

# Configuration Field Tier Registry

Central registry for field tier classifications in the three-tier configuration architecture.

## Overview

The `tier_registry` module defines the ConfigFieldTierRegistry class which serves as the central registry for field tier classifications in the three-tier configuration architecture. The registry classifies configuration fields into three tiers: Essential User Inputs (Tier 1), System Inputs (Tier 2), and Derived Inputs (Tier 3).

This module implements the three-tier configuration pattern by providing a centralized mapping of field names to their appropriate tiers. The registry enables consistent field classification across the system and supports the proper handling of different field types during serialization, validation, and configuration management operations.

## Classes and Methods

### Classes
- [`ConfigFieldTierRegistry`](#configfieldtierregistry) - Registry for field tier classifications

## API Reference

### ConfigFieldTierRegistry

_class_ cursus.core.config_fields.tier_registry.ConfigFieldTierRegistry

Registry for field tier classifications. This class implements the registry for classifying configuration fields into tiers: 1. Essential User Inputs (Tier 1), 2. System Inputs (Tier 2), 3. Derived Inputs (Tier 3). The registry provides methods to get and set tier classifications for fields.

**Class Attribute:**
- **DEFAULT_TIER_REGISTRY** (_Dict[str, int]_) – Default tier classifications based on field analysis

```python
from cursus.core.config_fields.tier_registry import ConfigFieldTierRegistry

# Get tier for a field
tier = ConfigFieldTierRegistry.get_tier("region")
print(f"Field 'region' is in tier {tier}")  # Output: Field 'region' is in tier 1

# Register a new field
ConfigFieldTierRegistry.register_field("custom_field", 2)
```

#### get_tier

get_tier(_field_name_)

Get tier classification for a field.

**Parameters:**
- **field_name** (_str_) – The name of the field to get the tier for

**Returns:**
- **int** – Tier classification (1, 2, or 3)

```python
# Get tier classifications for different fields
essential_tier = ConfigFieldTierRegistry.get_tier("region")  # Returns 1
system_tier = ConfigFieldTierRegistry.get_tier("batch_size")  # Returns 2
derived_tier = ConfigFieldTierRegistry.get_tier("unknown_field")  # Returns 3 (default)

# Use in configuration logic
if ConfigFieldTierRegistry.get_tier(field_name) == 1:
    print(f"Field {field_name} is an essential user input")
```

#### register_field

register_field(_field_name_, _tier_)

Register a field with a specific tier.

**Parameters:**
- **field_name** (_str_) – The name of the field to register
- **tier** (_int_) – The tier to assign (1, 2, or 3)

**Raises:**
- **ValueError** – If tier is not 1, 2, or 3

```python
# Register individual fields
ConfigFieldTierRegistry.register_field("custom_essential_field", 1)
ConfigFieldTierRegistry.register_field("custom_system_field", 2)
ConfigFieldTierRegistry.register_field("custom_derived_field", 3)

# Invalid tier raises error
try:
    ConfigFieldTierRegistry.register_field("invalid_field", 4)
except ValueError as e:
    print(f"Error: {e}")  # Error: Tier must be 1, 2, or 3, got 4
```

#### register_fields

register_fields(_tier_mapping_)

Register multiple fields with their tiers.

**Parameters:**
- **tier_mapping** (_Dict[str, int]_) – Dictionary mapping field names to tier classifications

**Raises:**
- **ValueError** – If any tier is not 1, 2, or 3

```python
# Register multiple fields at once
new_fields = {
    "custom_region": 1,
    "custom_batch_size": 2,
    "custom_derived_value": 3
}
ConfigFieldTierRegistry.register_fields(new_fields)

# Verify registration
for field_name, expected_tier in new_fields.items():
    actual_tier = ConfigFieldTierRegistry.get_tier(field_name)
    assert actual_tier == expected_tier
```

#### get_fields_by_tier

get_fields_by_tier(_tier_)

Get all fields assigned to a specific tier.

**Parameters:**
- **tier** (_int_) – Tier classification (1, 2, or 3)

**Returns:**
- **Set[str]** – Set of field names assigned to the specified tier

**Raises:**
- **ValueError** – If tier is not 1, 2, or 3

```python
# Get all essential user input fields (Tier 1)
essential_fields = ConfigFieldTierRegistry.get_fields_by_tier(1)
print(f"Essential fields: {essential_fields}")
# Output: Essential fields: {'region', 'service_name', 'pipeline_version', ...}

# Get all system input fields (Tier 2)
system_fields = ConfigFieldTierRegistry.get_fields_by_tier(2)
print(f"System fields: {system_fields}")
# Output: System fields: {'batch_size', 'processing_instance_count', ...}

# Get fields by tier for validation
for tier in [1, 2, 3]:
    fields = ConfigFieldTierRegistry.get_fields_by_tier(tier)
    print(f"Tier {tier} has {len(fields)} fields")
```

#### reset_to_defaults

reset_to_defaults()

Reset the registry to default tier classifications. This method is primarily intended for testing purposes.

```python
# Reset registry (typically used in tests)
ConfigFieldTierRegistry.reset_to_defaults()

# Verify reset worked
tier = ConfigFieldTierRegistry.get_tier("region")
assert tier == 1  # Should be back to default
```

## Default Tier Classifications

The registry includes comprehensive default classifications for common configuration fields:

### Tier 1 - Essential User Inputs
Fields that users must provide or are core to the configuration:
- `region`, `service_name`, `pipeline_version`
- `training_start_datetime`, `training_end_datetime`
- `model_class`, `label_name`, `id_name`
- `author`, `model_owner`, `model_domain`

### Tier 2 - System Inputs
Fields with reasonable defaults that users may override:
- `batch_size`, `processing_instance_count`, `training_instance_type`
- `processing_framework_version`, `py_version`
- `test_val_ratio`, `calibration_method`
- `max_acceptable_error_rate`, `default_numeric_value`

### Tier 3 - Derived Inputs
Fields computed from other inputs (default for unregistered fields):
- Any field not explicitly registered defaults to Tier 3
- Computed paths, derived configurations, calculated values

## Related Documentation

- [Configuration Fields Overview](README.md) - System overview and integration
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md) - Uses tier information for serialization
- [Configuration Field Categorizer](config_field_categorizer.md) - May use tier information for categorization
