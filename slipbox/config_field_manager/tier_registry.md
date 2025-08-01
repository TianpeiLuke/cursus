# ConfigFieldTierRegistry Documentation

## Overview

The `ConfigFieldTierRegistry` is a core component of the three-tier configuration architecture that classifies all configuration fields into tiers based on their role and origin. This registry serves as the central reference point for determining how fields should be handled in the configuration process.

## Purpose

The registry implements a classification system that categorizes all 83 identified configuration fields into three distinct tiers:

1. **Tier 1 (Essential User Inputs)**: Fields that require direct user input, representing core business decisions (~23% of all fields)
2. **Tier 2 (System Inputs)**: Fields with standardized defaults that don't require user configuration
3. **Tier 3 (Derived Inputs)**: Fields that can be automatically generated from essential inputs and system defaults

This classification is crucial for the proper functioning of the simplified configuration system, as it determines which fields users need to provide, which can be defaulted, and which can be derived.

## Implementation

The `ConfigFieldTierRegistry` is implemented as a class with class methods, providing a static registry that can be accessed throughout the codebase:

```python
class ConfigFieldTierRegistry:
    # Default tier classifications based on field analysis
    DEFAULT_TIER_REGISTRY = {
        # Essential User Inputs (Tier 1)
        "region_list": 1,
        "region_selection": 1,
        "full_field_list": 1,
        # ... other essential inputs
        
        # System Inputs (Tier 2)
        "metric_choices": 2,
        "device": 2,
        "header": 2,
        # ... other system inputs
        
        # All other fields default to Tier 3 (derived)
    }
    
    @classmethod
    def get_tier(cls, field_name):
        """Get tier classification for a field"""
        return cls.DEFAULT_TIER_REGISTRY.get(field_name, 3)  # Default to Tier 3
        
    # ... other methods
```

## Key Features

1. **Default Tier Classifications**: Pre-populated registry with tier assignments based on field analysis
2. **Tier Lookup**: Methods to retrieve the tier classification for any field
3. **Registration API**: Methods to add or update tier classifications
4. **Field Filtering**: Ability to retrieve all fields belonging to a specific tier

## Field Classification Process

Fields were classified into tiers based on the following criteria:

### Tier 1 (Essential User Inputs)
- Represents core business decisions
- Cannot be reasonably defaulted or derived
- Varies significantly between use cases
- Examples: region selection, feature lists, date ranges, model parameters

### Tier 2 (System Inputs)
- Has standardized values that rarely change
- Represents system configuration rather than business logic
- Can be defaulted with reasonable values for most cases
- Examples: infrastructure settings, framework versions, batch sizes

### Tier 3 (Derived Inputs)
- Can be automatically generated from other fields
- Follows deterministic rules for derivation
- Represents technical implementation details
- Examples: derived dimensions, path constructions, SQL transformations

## Usage

The registry is designed to be used by other components of the three-tier architecture:

```python
# Check which tier a field belongs to
tier = ConfigFieldTierRegistry.get_tier("region_list")  # Returns 1

# Get all fields in Tier 1
essential_fields = ConfigFieldTierRegistry.get_fields_by_tier(1)

# Register a new field or update an existing one
ConfigFieldTierRegistry.register_field("new_field", 2)
```

## Benefits

1. **Centralized Field Management**: Single source of truth for field classifications
2. **Explicit Classification**: Clear documentation of field roles
3. **Adaptable System**: Easy to reclassify fields as needed
4. **Foundation for Automation**: Enables automatic processing based on field tier

## Future Enhancements

1. **Dynamic Classification**: Support for context-aware field classification
2. **Tier Migration**: Tools for safely migrating fields between tiers
3. **Metadata Enhancement**: Additional metadata for fields beyond just tier
4. **Validation Rules**: Tier-specific validation rules for fields

## Implementation Status

The `ConfigFieldTierRegistry` has been fully implemented as part of Phase 1 of the Essential Inputs Implementation Plan. It includes classifications for all 83 identified configuration fields and provides a complete API for tier management.
