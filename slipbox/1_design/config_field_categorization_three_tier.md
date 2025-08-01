# Three-Tier Configuration Field Management

## Overview

This document outlines the design for integrating the **Three-Tier Configuration Architecture** with the existing Config Field Categorization system. This integration provides a structured approach for managing field classifications while maintaining compatibility with the existing serialization and configuration management infrastructure.

## Three-Tier Configuration Architecture

The Three-Tier Configuration Architecture classifies all configuration fields into three distinct categories:

### 1. Essential User Inputs (Tier 1)
Fields that represent core business decisions and require direct user input. These comprise approximately 23% of all fields and include:
- Feature selections (`full_field_list`, `cat_field_list`, `tab_field_list`)
- Region selection (`region_list`, `region_selection`)
- Date ranges (`training_start_datetime`, `training_end_datetime`, etc.)
- Target fields (`label_name`, `id_name`, `marketplace_id_col`)
- Core model parameters (`num_round`, `max_depth`, `min_child_weight`)

### 2. System Inputs (Tier 2)
Fields with standardized values that don't require user configuration. These fields have sensible defaults but can be modified by administrators:
- Infrastructure settings (`processing_instance_type_large`, etc.)
- Default hyperparameters (`device`, `batch_size`, `lr`, etc.)
- Framework settings (`py_version`, `processing_framework_version`)
- Processing entry points (script paths for different steps)

### 3. Derived Inputs (Tier 3)
Fields that can be automatically generated from essential inputs and system defaults:
- Field derivatives (`input_tab_dim`, `num_classes`, `is_binary`)
- SQL transformations (`schema_list`, `training_transform_sql`)
- Path constructions (`pipeline_subdirectory`, `pipeline_s3_loc`)
- Output specifications (`training_output_path`, `output_schema`)

## Integration with Config Field Categorization

### Extended Field Metadata Structure

The existing field information collection process will be extended to track the tier classification of each field:

```python
class FieldInfo:
    """Enhanced field information with tier classification"""
    def __init__(self, name, configs=None, values=None):
        self.name = name                # Field name
        self.configs = configs or []     # Configs containing this field
        self.values = values or {}       # Field values by config
        self.is_special = False          # Whether field is a special field
        self.is_static = True            # Whether field has static values
        self.is_cross_type = False       # Whether field appears in both processing/non-processing
        self.tier = None                 # Tier classification (1, 2, or 3)
        
    def classify_tier(self, tier_registry):
        """
        Classify field into a tier based on the registry
        
        Args:
            tier_registry: Registry mapping field names to tier classifications
        """
        if self.name in tier_registry:
            self.tier = tier_registry[self.name]
        else:
            # Default to Tier 3 (derived) if not specified
            self.tier = 3
```

### Tier Registry Implementation

A centralized tier registry will be maintained to classify fields:

```python
class ConfigFieldTierRegistry:
    """Registry for field tier classifications"""
    
    # Default tier classifications based on field analysis
    DEFAULT_TIER_REGISTRY = {
        # Essential User Inputs (Tier 1)
        "region_list": 1,
        "region_selection": 1,
        "full_field_list": 1,
        "cat_field_list": 1,
        "tab_field_list": 1,
        "label_name": 1,
        "id_name": 1,
        "marketplace_id_col": 1,
        "multiclass_categories": 1,
        "class_weights": 1,
        "model_class": 1,
        "num_round": 1,
        "max_depth": 1,
        "min_child_weight": 1,
        "service_name": 1,
        "pipeline_version": 1,
        "framework_version": 1,
        "current_date": 1,
        "source_dir": 1,
        # ... other essential inputs
        
        # System Inputs (Tier 2)
        "metric_choices": 2,
        "device": 2,
        "header": 2,
        "batch_size": 2,
        "lr": 2,
        "max_epochs": 2,
        "optimizer": 2,
        "py_version": 2,
        "processing_framework_version": 2,
        "processing_instance_type_large": 2,
        "processing_instance_type_small": 2,
        "processing_instance_count": 2,
        "processing_volume_size": 2,
        "test_val_ratio": 2,
        # ... other system inputs
        
        # All other fields default to Tier 3 (derived)
    }
    
    @classmethod
    def get_tier(cls, field_name):
        """Get tier classification for a field"""
        return cls.DEFAULT_TIER_REGISTRY.get(field_name, 3)  # Default to Tier 3
        
    @classmethod
    def register_field(cls, field_name, tier):
        """Register a field with a specific tier"""
        cls.DEFAULT_TIER_REGISTRY[field_name] = tier
        
    @classmethod
    def register_fields(cls, tier_mapping):
        """Register multiple fields with their tiers"""
        cls.DEFAULT_TIER_REGISTRY.update(tier_mapping)
```

### Enhanced Field Categorizer

The field categorization system will be extended to incorporate tier classifications:

```python
class ConfigFieldCategorizer:
    """Enhanced field categorizer with tier awareness"""
    
    def __init__(self, config_list, tier_registry=None):
        """
        Initialize with configs and optional tier registry
        
        Args:
            config_list: List of configuration objects
            tier_registry: Custom tier registry or None for default
        """
        self.config_list = config_list
        self.tier_registry = tier_registry or ConfigFieldTierRegistry.DEFAULT_TIER_REGISTRY
        self.field_info = {}  # Field info by name
        self.processing_configs = []
        self.non_processing_configs = []
        self._collect_field_information()
        
    def _collect_field_information(self):
        """Collect and classify field information from configs"""
        # ... existing field info collection logic
        
        # Add tier classification
        for name, info in self.field_info.items():
            info.classify_tier(self.tier_registry)
    
    def categorize_fields(self):
        """
        Categorize fields using enhanced rules that incorporate tier classification
        
        Returns:
            tuple: (shared_fields, specific_fields, processing_shared_fields, processing_specific_fields)
        """
        # ... existing categorization logic with tier-based enhancements
        
        # Special handling for essential user inputs (Tier 1)
        # These are always kept specific unless they have identical values across all configs
        for name, info in self.field_info.items():
            if info.tier == 1 and not self._has_identical_values_across_all_configs(info):
                self._ensure_field_is_specific(name, info)
                
        # System inputs (Tier 2) can be shared if they have identical values
        # This follows the existing logic for shared fields
        
        # Derived inputs (Tier 3) follow the normal categorization rules
        
        return (shared_fields, specific_fields, processing_shared_fields, processing_specific_fields)
```

### Hyperparameter Registry Extension

The existing hyperparameter registry will be extended to include tier classifications:

```python
# Extension to hyperparameter_registry.py
HYPERPARAMETER_REGISTRY = {
    "ModelHyperparameters": {
        "module_path": "src.processing.hyperparameters",
        "model_type": "any",
        "description": "Base hyperparameters class",
        # Add tier classification for fields
        "field_tiers": {
            # Essential user inputs
            "full_field_list": 1,
            "cat_field_list": 1,
            "tab_field_list": 1,
            "label_name": 1,
            "id_name": 1,
            "marketplace_id_col": 1,
            "multiclass_categories": 1,
            "class_weights": 1,
            
            # System inputs with fixed values
            "metric_choices": 2,
            "device": 2,
            "header": 2,
            "batch_size": 2,
            "lr": 2,
            "max_epochs": 2,
            "optimizer": 2,
            
            # Derived inputs
            "input_tab_dim": 3,
            "num_classes": 3,
            "is_binary": 3
        }
    },
    # Other hyperparameter classes...
}
```

## Implementing Default Values for System Inputs

A DefaultValuesProvider class will apply default values for system inputs:

```python
class DefaultValuesProvider:
    """Provides default values for system inputs (Tier 2)"""
    
    # Default values for system inputs
    DEFAULT_VALUES = {
        # Base Model Hyperparameters
        "metric_choices": lambda config: ['f1_score', 'auroc'] if getattr(config, 'is_binary', True) else ['accuracy', 'f1_score'],
        "device": -1,
        "header": "true",
        "batch_size": 32,
        "lr": 0.01,
        "max_epochs": 100,
        "optimizer": "adam",
        
        # Framework Settings
        "py_version": "py3",
        "processing_framework_version": "1.2-1",
        
        # Processing Resources
        "processing_instance_type_large": "ml.m5.4xlarge",
        "processing_instance_type_small": "ml.m5.xlarge",
        "processing_instance_count": 1,
        "processing_volume_size": 500,
        "test_val_ratio": 0.5,
        
        # Training Resources
        "training_instance_count": 1,
        "training_volume_size": 800,
        
        # Inference Resources
        "inference_instance_type": "ml.m5.4xlarge",
        
        # Processing Entry Points
        "processing_entry_point": lambda config: self._get_entry_point_by_config_type(config),
        "model_eval_processing_entry_point": "model_eval_xgb.py",
        "packaging_entry_point": "mims_package.py",
        
        # Payload Configuration
        "max_acceptable_error_rate": 0.2,
        "special_field_values": None
    }
    
    @classmethod
    def apply_defaults(cls, config):
        """
        Apply default values to a configuration object
        
        Args:
            config: Configuration object to apply defaults to
        """
        for field_name, default_value in cls.DEFAULT_VALUES.items():
            # Skip if field is already set
            if hasattr(config, field_name) and getattr(config, field_name) is not None:
                continue
                
            # Apply default (either value or callable)
            if callable(default_value):
                value = default_value(config)
            else:
                value = default_value
                
            # Set the default value
            setattr(config, field_name, value)
                
    @staticmethod
    def _get_entry_point_by_config_type(config):
        """Determine processing entry point based on config type"""
        if isinstance(config, TabularPreprocessingConfig):
            return "tabular_preprocess.py"
        elif isinstance(config, ModelCalibrationConfig):
            return "model_calibration.py"
        elif isinstance(config, PayloadConfig):
            return "mims_payload.py"
        else:
            return None
```

## Field Derivation Logic

A FieldDerivationEngine will implement derivation rules for Tier 3 fields:

```python
class FieldDerivationEngine:
    """Engine for deriving Tier 3 fields from other fields"""
    
    @staticmethod
    def derive_fields(config):
        """
        Derive fields for a configuration object
        
        Args:
            config: Configuration object to derive fields for
        """
        # Derive fields based on config type
        if hasattr(config, "tab_field_list") and hasattr(config, "cat_field_list"):
            # Derive input_tab_dim if not set
            if not hasattr(config, "input_tab_dim") or config.input_tab_dim is None:
                config.input_tab_dim = len(config.tab_field_list)
                
            # Derive num_classes from multiclass_categories
            if hasattr(config, "multiclass_categories") and \
               (not hasattr(config, "num_classes") or config.num_classes is None):
                config.num_classes = len(config.multiclass_categories)
                
            # Derive is_binary from num_classes
            if hasattr(config, "num_classes") and \
               (not hasattr(config, "is_binary") or config.is_binary is None):
                config.is_binary = (config.num_classes == 2)
                
        # MDS field list derivation
        if hasattr(config, "tab_field_list") and hasattr(config, "cat_field_list") and \
           (not hasattr(config, "mds_field_list") or config.mds_field_list is None):
            core_fields = ['objectId', 'transactionDate', 'Abuse.currency_exchange_rate_inline.exchangeRate', 'baseCurrency']
            config.mds_field_list = core_fields + config.tab_field_list + config.cat_field_list
            
        # Output schema derivation
        if hasattr(config, "mds_field_list") and \
           (not hasattr(config, "output_schema") or config.output_schema is None):
            config.output_schema = [{'field_name': field,'field_type':'STRING'} for field in config.mds_field_list]
            
        # Derive other fields based on specific relationships
        # ... additional derivation logic ...
```

## Configuration Processing Pipeline

The complete three-tier configuration processing pipeline combines these components:

```python
def process_configuration(essential_config):
    """
    Process configuration using the three-tier architecture
    
    Args:
        essential_config: Essential user inputs (Tier 1)
        
    Returns:
        dict: Complete configuration with all fields
    """
    # 1. Create the config objects from essential inputs
    config_objects = create_config_objects(essential_config)
    
    # 2. Apply system defaults (Tier 2)
    for config in config_objects:
        DefaultValuesProvider.apply_defaults(config)
        
    # 3. Derive dependent fields (Tier 3)
    for config in config_objects:
        FieldDerivationEngine.derive_fields(config)
        
    # 4. Merge and categorize fields
    categorizer = ConfigFieldCategorizer(config_objects)
    shared, specific, processing_shared, processing_specific = categorizer.categorize_fields()
    
    # 5. Build final configuration structure
    merged_config = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "config_types": {categorizer.generate_step_name(config): config.__class__.__name__ 
                            for config in config_objects}
        },
        "configuration": {
            "shared": shared,
            "specific": specific,
            "processing": {
                "processing_shared": processing_shared,
                "processing_specific": processing_specific
            }
        }
    }
    
    return merged_config
```

## Benefits of Three-Tier Integration

1. **Maintainable Field Management**
   - Clear classification of fields by purpose
   - Centralized registry of field tiers
   - Explicit rules for field handling

2. **Simplified User Experience**
   - Users only need to provide essential inputs (Tier 1)
   - System inputs (Tier 2) have sensible defaults
   - Derived inputs (Tier 3) are generated automatically

3. **Enhanced Administration**
   - System inputs can be modified by administrators without affecting the user interface
   - Default values can be updated in a single location
   - Field classifications can be refined over time

4. **Compatibility with Existing Infrastructure**
   - Works with the current serialization and deserialization systems
   - Maintains the same output format for configuration files
   - Reuses existing code for field categorization

## Implementation Steps

1. **Extend Hyperparameter Registry**
   - Add field tier classifications to hyperparameter classes
   - Update registry loader to include tier information

2. **Create Default Values Provider**
   - Implement defaults for all system inputs ([See detailed design](./default_values_provider_design.md))
   - Create application logic for defaults

3. **Implement Field Derivation Engine**
   - Define derivation rules for all Tier 3 fields ([See detailed design](./field_derivation_engine_design.md))
   - Create derivation application system

4. **Enhance Field Categorizer**
   - Update to incorporate tier classifications
   - Extend categorization rules

5. **Update Configuration Processing Pipeline**
   - Create integrated pipeline for three-tier processing
   - Ensure compatibility with existing infrastructure

## Related Technical Designs

For detailed technical designs of the key components in this architecture, refer to:

- [DefaultValuesProvider Design](./default_values_provider_design.md) - Complete design for the Tier 2 (System Inputs) component
- [FieldDerivationEngine Design](./field_derivation_engine_design.md) - Complete design for the Tier 3 (Derived Inputs) component
- [Essential Inputs Notebook Design](./essential_inputs_notebook_design_revised.md) - Revised design for the user interface layer of the three-tier architecture
- [Essential Inputs Field Dependency Analysis](./essential_inputs_field_dependency_analysis.md) - Detailed field analysis and categorization for the three-tier architecture

## Conclusion

The integration of the Three-Tier Configuration Architecture with the existing Config Field Categorization system provides a robust and maintainable approach to configuration management. By clearly classifying fields as essential user inputs, system inputs, or derived inputs, we create a system that is simpler for users, more maintainable for developers, and more flexible for administrators.

This design maintains compatibility with the existing infrastructure while enhancing it with a more structured approach to field classification and management. The centralized tier registry, default values provider, and field derivation engine work together to create a comprehensive system for configuration handling.
