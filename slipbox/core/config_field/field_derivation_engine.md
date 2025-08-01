# FieldDerivationEngine Documentation

## Overview

The `FieldDerivationEngine` is a critical component of the three-tier configuration architecture responsible for managing all derived inputs (Tier 3). This engine automatically generates configuration fields based on essential user inputs and system defaults, implementing the dependency relationships between fields.

## Purpose

The `FieldDerivationEngine` serves several key purposes in the configuration system:

1. Derive configuration fields from essential inputs and system defaults
2. Implement the dependency relationships between fields
3. Ensure consistency across derived values
4. Reduce the burden on users by automating field generation
5. Maintain compatibility with existing pipeline configurations

## Implementation

The `FieldDerivationEngine` is implemented as an instantiable class with method-based derivation rules:

```python
class FieldDerivationEngine:
    """Engine for deriving configuration fields from essential inputs and system defaults."""
    
    def __init__(self, logger=None):
        """Initialize the field derivation engine"""
        self.logger = logger
        self.derived_fields = set()
        self.derivation_graph = {}
        
    def derive_fields(self, config):
        """Derive all applicable fields for a configuration object"""
        # Implementation...
        
    def derive_input_dimensions(self, config):
        """Derive input dimensions from field lists"""
        # Implementation...
        
    # Additional derivation methods...
```

## Key Features

### 1. Dependency-Aware Processing

The `FieldDerivationEngine` implements a dependency-aware processing model:

- Fields are derived in the correct order based on their dependencies
- The system automatically handles dependency chains (derived field depends on another derived field)
- Circular dependencies are detected and reported

### 2. Explicit Derivation Rules

All derivation rules are implemented as separate methods with clear documentation:

```python
def derive_classification_type(self, config):
    """
    Derive classification type fields
    
    Applicable to: ModelHyperparameters, XGBoostHyperparameters
    
    Derives:
    - is_binary: Boolean indicating binary classification
    - num_classes: Number of classes
    - multiclass_categories: List of class values
    
    Depends on:
    - multiclass_categories: List of class values (if available)
    - is_binary: Boolean indicating binary classification (if available)
    - num_classes: Number of classes (if available)
    """
    # Implementation...
```

Each rule:
- Has a clear naming convention (`derive_<field_group>`)
- Documents its applicable configuration types
- Specifies the fields it derives
- Lists the fields it depends on
- Returns a dictionary of derived fields for tracking

### 3. Configuration Type Awareness

The engine respects configuration type boundaries:

- Derivation rules specify which configuration types they apply to
- Rules are only applied to appropriate configuration types
- Cross-configuration dependencies are handled through explicit references

### 4. Cross-Configuration Derivations

Some fields depend on values in other configurations. These are handled by dedicated logic:

```python
def _apply_cross_config_derivations(self, config, config_map):
    """Apply derivations that depend on other configurations"""
    # Implementation for cross-config derivations
```

## Field Derivation Rules

The `FieldDerivationEngine` includes comprehensive derivation rules organized into logical groups:

### 1. Classification and Model Structure

- `derive_input_dimensions`: Derives dimensions from field lists
- `derive_classification_type`: Derives classification type fields
- `derive_xgboost_specific`: Derives XGBoost-specific parameters

### 2. Field Lists and Schema Fields

- `derive_mds_field_list`: Derives MDS field list from component field lists
- `derive_output_schema`: Derives output schema from field list
- `derive_transform_sql`: Derives SQL transformation based on configuration

### 3. ETL and Data Source Fields

- `derive_etl_job_id`: Derives ETL job ID based on region
- `derive_edx_manifest`: Derives EDX manifest from provider, subject, dataset, and dates

### 4. Pipeline and Path Fields

- `derive_pipeline_fields`: Derives pipeline name, subdirectory, and S3 location
- `derive_aws_region`: Derives AWS region from region code

### 5. Model Registration Fields

- `derive_model_inference_variable_list`: Derives model inference input/output variable lists
- `derive_model_registration_objective`: Derives model registration objective

## Usage

The `FieldDerivationEngine` is designed to be used after applying system defaults (Tier 2):

```python
# Create config object with essential inputs and system defaults
config = ModelHyperparameters(
    tab_field_list=["field1", "field2", "field3"],
    is_binary=True
)
DefaultValuesProvider.apply_defaults(config)

# Initialize the derivation engine
engine = FieldDerivationEngine()

# Derive fields
engine.derive_fields(config)

# Config now has all applicable Tier 3 fields derived
assert config.input_tab_dim == 3  # Derived from tab_field_list length
assert config.num_classes == 2    # Derived from is_binary
```

For multiple configurations with cross-dependencies:

```python
# Apply derivations to multiple configurations
configs = [model_config, pipeline_config, evaluation_config]
engine.derive_fields_for_multiple(configs)
```

## Derivation Example: Classification Type

One of the most important derivation rules handles classification type fields:

```python
def derive_classification_type(self, config):
    derived = {}
    
    # Case 1: Derive from multiclass_categories
    if hasattr(config, "multiclass_categories") and config.multiclass_categories is not None:
        categories = config.multiclass_categories
        
        # Derive num_classes
        if not hasattr(config, "num_classes") or config.num_classes is None:
            config.num_classes = len(categories)
            derived["num_classes"] = config.num_classes
            
        # Derive is_binary
        if not hasattr(config, "is_binary") or config.is_binary is None:
            config.is_binary = (config.num_classes == 2)
            derived["is_binary"] = config.is_binary
            
    # Case 2: Derive from num_classes
    # ... additional cases
    
    return derived
```

This rule handles three cases:
1. If `multiclass_categories` is available, derive `num_classes` and `is_binary`
2. If `num_classes` is available, derive `is_binary` and potentially `multiclass_categories`
3. If `is_binary` is available, derive `num_classes` and `multiclass_categories`

This approach ensures that the classification type is consistent across all related fields.

## Integration with Three-Tier Architecture

Within the three-tier architecture, the `FieldDerivationEngine` implements Tier 3 (Derived Inputs). The typical workflow is:

1. Collect essential user inputs (Tier 1)
2. Create config objects from essential inputs
3. Apply system defaults (Tier 2) using `DefaultValuesProvider`
4. **Derive dependent fields (Tier 3)** using `FieldDerivationEngine`
5. Generate final configuration

## Logging and Diagnostics

The `FieldDerivationEngine` includes comprehensive logging to aid in debugging:

- Records which fields are derived for each configuration
- Reports any errors in derivation rules
- Can generate a visualization of the derivation graph

## Customization and Extension

The `FieldDerivationEngine` is designed for easy extension:

### 1. Add New Derivation Rules

To add new derivation rules, simply add new methods following the `derive_*` naming convention:

```python
def derive_new_field_group(self, config):
    """
    Derive new fields
    
    Applicable to: ConfigType1, ConfigType2
    
    Derives:
    - new_field_1: Description
    - new_field_2: Description
    
    Depends on:
    - dependency_1: Description
    - dependency_2: Description
    """
    derived = {}
    
    # Implementation
    
    return derived
```

### 2. Override Existing Rules

Existing derivation rules can be overridden by subclassing:

```python
class CustomFieldDerivationEngine(FieldDerivationEngine):
    def derive_classification_type(self, config):
        """Override with custom implementation"""
        # Custom implementation
```

## Benefits

1. **Reduced Configuration Burden**: Users don't need to specify fields that can be derived
2. **Consistent Derived Values**: Fields are derived using standardized rules
3. **Explicit Dependencies**: Clear documentation of field relationships
4. **Extensible System**: Easy to add new derivation rules as needed

## Future Enhancements

1. **Dependency Graph Visualization**: Visual representation of field dependencies
2. **Derivation Rule Testing**: Automated testing of individual derivation rules
3. **Performance Optimization**: Caching of derivation results for similar configurations
4. **Rule Generation**: Machine learning to generate new derivation rules from examples

## Implementation Status

The `FieldDerivationEngine` has been fully implemented as part of Phase 1 of the Essential Inputs Implementation Plan. It includes derivation rules for all identified Tier 3 fields and provides a complete API for field derivation.
