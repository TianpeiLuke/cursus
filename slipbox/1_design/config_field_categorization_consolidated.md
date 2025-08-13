---
tags:
  - design
  - implementation
  - configuration
  - architecture
  - three-tier
keywords:
  - config field categorization
  - three-tier architecture
  - configuration management
  - type-aware serialization
  - essential inputs
  - system inputs
  - derived fields
  - field organization
  - serialization
  - deserialization
topics:
  - configuration management
  - field organization
  - three-tier design
  - serialization architecture
language: python
date of note: 2025-08-12
---

# Configuration Field Categorization Architecture

## Overview

The Configuration Field Categorization system provides a comprehensive framework for managing configuration fields in the pipeline infrastructure. It implements a sophisticated three-tier architecture that classifies fields by their purpose and lifecycle, while providing advanced serialization, deserialization, and organization capabilities.

This consolidated design document combines the core concepts from the refactored implementation, three-tier design, and original categorization approaches, focusing on the current implementation in `src/cursus/core/config_fields/`.

## Core Purpose

The Configuration Field Categorization system serves several critical purposes:

1. **Three-Tier Field Classification** - Categorize fields based on their purpose and relationship to user interaction
2. **Smart Serialization** - Intelligently serialize and deserialize complex configuration objects with type preservation
3. **Field Organization** - Organize fields to minimize redundancy and maintain clarity across configurations
4. **Type Preservation** - Maintain complex type information across serialization boundaries
5. **Circular Reference Management** - Detect and handle circular references in configuration graphs
6. **Storage Optimization** - Reduce redundancy by sharing common fields across configurations

## Three-Tier Classification Architecture

The foundation of our configuration system is the Three-Tier Classification Architecture, which categorizes all configuration fields into three distinct tiers based on their purpose and lifecycle:

### Tier 1: Essential User Inputs

Fields that represent core business decisions and require direct user input.

**Characteristics:**
- Explicitly provided by users
- No default values
- Public access
- Represent fundamental configuration decisions
- Subject to validation
- Examples: `region`, `author`, `bucket`, `pipeline_version`, `full_field_list`, `label_name`

**Implementation in Tier Registry:**
```python
# Essential User Inputs (Tier 1) from tier_registry.py
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
"author": 1,
"region": 1,
"bucket": 1,
```

### Tier 2: System Inputs with Defaults

Fields with standardized values that have sensible defaults but can be overridden when needed.

**Characteristics:**
- Have reasonable defaults
- Can be overridden by administrators
- Public access
- Represent system configuration settings
- Subject to validation
- Examples: `py_version`, `processing_framework_version`, `instance_type`, `batch_size`

**Implementation in Tier Registry:**
```python
# System Inputs (Tier 2) from tier_registry.py
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
"training_instance_count": 2,
"training_volume_size": 2,
"training_instance_type": 2,
"inference_instance_type": 2,
```

### Tier 3: Derived Fields

Fields that are calculated from Tier 1 and Tier 2 fields with clear derivation logic.

**Characteristics:**
- Calculated from other fields
- Private attributes with public read-only properties (in config classes)
- Not directly set by users or API
- Examples: `aws_region`, `pipeline_name`, `pipeline_s3_loc`, `input_tab_dim`, `num_classes`

**Implementation Pattern:**
```python
class BasePipelineConfig(BaseModel):
    # Private fields for derived values (Tier 3)
    _aws_region: Optional[str] = PrivateAttr(default=None)
    _pipeline_name: Optional[str] = PrivateAttr(default=None)
    
    # Public properties for derived fields
    @property
    def aws_region(self) -> str:
        """Get AWS region based on region code."""
        if self._aws_region is None:
            self._aws_region = self._REGION_MAPPING.get(self.region, "us-east-1")
        return self._aws_region
    
    @property
    def pipeline_name(self) -> str:
        """Get pipeline name derived from author, service_name, model_class, and region."""
        if self._pipeline_name is None:
            self._pipeline_name = f"{self.author}-{self.service_name}-{self.model_class}-{self.region}"
        return self._pipeline_name
```

## Field Categorization for Storage

When serializing and persisting configurations, fields are organized into logical categories to reduce redundancy and improve clarity. The current implementation uses a simplified two-category storage format:

### Storage Format

```json
{
  "metadata": {
    "created_at": "timestamp",
    "config_types": {
      "StepName1": "ConfigClass1",
      "StepName2": "ConfigClass2"
    },
    "field_sources": {
      "field1": ["StepName1", "StepName2"],
      "field2": ["StepName1"],
      "field3": ["StepName2"]
    }
  },
  "configuration": {
    "shared": {
      "common_field1": "common_value1",
      "common_field2": "common_value2"
    },
    "specific": {
      "StepName1": {
        "specific_field1": "specific_value1"
      },
      "StepName2": {
        "specific_field2": "specific_value2"
      }
    }
  }
}
```

### Storage Categorization Rules

The `ConfigFieldCategorizer` applies explicit rules with clear precedence:

1. **Field is special** → Place in `specific`
   - Special fields include those in the `SPECIAL_FIELDS_TO_KEEP_SPECIFIC` list
   - Pydantic models are considered special fields
   - Complex nested structures are considered special fields

2. **Field appears only in one config** → Place in `specific`
   - If a field exists in only one configuration instance, it belongs in that instance's specific section

3. **Field has different values across configs** → Place in `specific`
   - If a field has the same name but different values across multiple configs, each instance goes in specific

4. **Field is non-static** → Place in `specific`
   - Fields matching patterns in `NON_STATIC_FIELD_PATTERNS` are considered non-static
   - Non-static fields are kept specific to preserve their dynamic nature

5. **Field has identical value across all configs** → Place in `shared`
   - If a field has the same value across all configs, it belongs in shared

6. **Default case** → Place in `specific`
   - When in doubt, place in specific to ensure proper functioning

## Key Components

### 1. ConfigFieldCategorizer

The main categorizer that analyzes and categorizes fields based on their characteristics:

```python
class ConfigFieldCategorizer:
    """
    Responsible for categorizing configuration fields based on their characteristics.
    
    Analyzes field values and metadata across configs to determine proper placement.
    Uses explicit rules with clear precedence for categorization decisions,
    implementing the Declarative Over Imperative principle.
    """
    
    def __init__(self, config_list: List[Any], processing_step_config_base_class: Optional[type] = None):
        """Initialize with list of config objects to categorize."""
        self.config_list = config_list
        self.processing_base_class = processing_step_config_base_class or self._get_processing_base_class()
        
        # Categorize configs
        self.processing_configs = [c for c in config_list 
                                  if isinstance(c, self.processing_base_class)]
        self.non_processing_configs = [c for c in config_list 
                                      if not isinstance(c, self.processing_base_class)]
        
        # Collect field information and categorize
        self.field_info = self._collect_field_info()
        self.categorization = self._categorize_fields()
    
    def _categorize_field(self, field_name: str) -> CategoryType:
        """
        Determine the category for a field based on explicit rules.
        
        Returns:
            CategoryType: Category for the field (SHARED or SPECIFIC)
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

### 2. ConfigFieldTierRegistry

A centralized registry for field tier classifications:

```python
class ConfigFieldTierRegistry:
    """
    Registry for field tier classifications.

    This class implements the registry for classifying configuration fields into tiers:
    1. Essential User Inputs (Tier 1)
    2. System Inputs (Tier 2)
    3. Derived Inputs (Tier 3)
    """
    
    @classmethod
    def get_tier(cls, field_name: str) -> int:
        """Get tier classification for a field."""
        return cls.DEFAULT_TIER_REGISTRY.get(field_name, 3)  # Default to Tier 3
        
    @classmethod
    def register_field(cls, field_name: str, tier: int) -> None:
        """Register a field with a specific tier."""
        if tier not in [1, 2, 3]:
            raise ValueError(f"Tier must be 1, 2, or 3, got {tier}")
        cls.DEFAULT_TIER_REGISTRY[field_name] = tier
        
    @classmethod
    def get_fields_by_tier(cls, tier: int) -> Set[str]:
        """Get all fields assigned to a specific tier."""
        if tier not in [1, 2, 3]:
            raise ValueError(f"Tier must be 1, 2, or 3, got {tier}")
        return {field for field, t in cls.DEFAULT_TIER_REGISTRY.items() if t == tier}
```

### 3. TypeAwareConfigSerializer

The serializer that preserves type information during serialization:

```python
class TypeAwareConfigSerializer:
    """Serializer that preserves type information"""
    
    def __init__(self, config_classes=None):
        self.config_classes = config_classes or {}
        self.circular_reference_tracker = CircularReferenceTracker()
        
    def serialize(self, obj):
        """Serialize an object with type information"""
        # Handle primitive types, lists, dicts, Pydantic models, etc.
        # Preserve type information for complex objects
        
    def deserialize(self, data, expected_type=None):
        """Deserialize an object with type information"""
        # Reconstruct objects with proper types
        
    def generate_step_name(self, config: Any) -> str:
        """Generate a step name for a config, including job type variants."""
        # Implementation for step name generation with job type handling
```

### 4. ConfigMerger

The merger that combines multiple configurations into a unified structure:

```python
class ConfigMerger:
    """Merger for multiple configurations"""
    
    def __init__(self, config_list):
        self.config_list = config_list
        self.serializer = TypeAwareConfigSerializer()
        self.categorizer = ConfigFieldCategorizer(config_list)
        
    def merge(self):
        """Merge configurations into unified structure"""
        # Categorize fields
        shared_fields, specific_fields = self.categorizer.categorize_fields()
        
        # Create metadata with config types and field sources
        metadata = {
            "created_at": datetime.now().isoformat(),
            "config_types": {
                self._get_step_name(config): config.__class__.__name__
                for config in self.config_list
            },
            "field_sources": self._get_field_sources()
        }
        
        # Create final structure
        return {
            "metadata": metadata,
            "configuration": {
                "shared": shared_fields,
                "specific": specific_fields
            }
        }
```

### 5. CircularReferenceTracker

The tracker that detects and manages circular references:

```python
class CircularReferenceTracker:
    """Tracks object references to detect and handle circular references"""
    
    def __init__(self, max_depth=100):
        self.processing_stack = []
        self.object_id_to_path = {}
        self.current_path = []
        self.max_depth = max_depth
        
    def enter_object(self, obj_data):
        """Start tracking a new object, returns whether a circular reference was detected."""
        # Implementation for circular reference detection
        
    def exit_object(self):
        """Mark that processing of the current object is complete"""
        # Implementation for cleanup
```

### 6. Constants and Enums

Type-safe constants and enums for the system:

```python
# Special fields that should always be kept in specific sections
SPECIAL_FIELDS_TO_KEEP_SPECIFIC: Set[str] = {
    "image_uri", 
    "script_name",
    "output_path", 
    "input_path",
    "model_path",
    "hyperparameters",
    "instance_type",
    "job_name_prefix",
    "output_schema"
}

class CategoryType(Enum):
    """Enumeration of field category types for the simplified structure."""
    SHARED = auto()    # Fields shared across all configs
    SPECIFIC = auto()  # Fields specific to certain configs

class SerializationMode(Enum):
    """Enumeration of serialization modes."""
    PRESERVE_TYPES = auto()    # Preserve type information in serialized output
    SIMPLE_JSON = auto()       # Convert to plain JSON without type information
    CUSTOM_FIELDS = auto()     # Only preserve types for certain fields
```

## Public API

The public API provides simple, intuitive functions for users:

```python
def merge_and_save_configs(config_list: List[Any], output_file: str) -> Dict[str, Any]:
    """
    Merge and save multiple configs to a JSON file.
    
    Args:
        config_list: List of configuration objects
        output_file: Path to output JSON file
        
    Returns:
        Dict containing merged configuration
    """
    merger = ConfigMerger(config_list)
    return merger.save(output_file)

def load_configs(input_file: str, config_classes: Dict[str, Type] = None) -> Dict[str, Any]:
    """
    Load configurations from a JSON file.
    
    Args:
        input_file: Path to input JSON file
        config_classes: Optional mapping of class names to class types
        
    Returns:
        Dict mapping step names to instantiated config objects
    """
    # Implementation for loading and reconstructing configs
    # Uses TypeAwareConfigSerializer for proper type reconstruction
    # Handles shared/specific field merging with proper precedence
```

## Integration with Three-Tier Architecture

### Field Processing Pipeline

The complete three-tier configuration processing pipeline:

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
    shared, specific = categorizer.get_categorized_fields()
    
    # 5. Build final configuration structure
    merged_config = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "config_types": {categorizer.generate_step_name(config): config.__class__.__name__ 
                            for config in config_objects}
        },
        "configuration": {
            "shared": shared['shared'],
            "specific": shared['specific']
        }
    }
    
    return merged_config
```

### Tier-Aware Field Handling

The system can optionally use tier information for enhanced categorization:

```python
class TierAwareConfigFieldCategorizer(ConfigFieldCategorizer):
    """Enhanced field categorizer with tier awareness"""
    
    def __init__(self, config_list, tier_registry=None):
        super().__init__(config_list)
        self.tier_registry = tier_registry or ConfigFieldTierRegistry
        
    def _categorize_field_with_tier_info(self, field_name: str) -> CategoryType:
        """Categorize field using tier information as additional context"""
        # Get base categorization
        base_category = super()._categorize_field(field_name)
        
        # Get tier information
        tier = self.tier_registry.get_tier(field_name)
        
        # Essential user inputs (Tier 1) with different values should be specific
        if tier == 1 and not self._has_identical_values_across_all_configs(field_name):
            return CategoryType.SPECIFIC
            
        # System inputs (Tier 2) can be shared if they have identical values
        if tier == 2 and self._has_identical_values_across_all_configs(field_name):
            return CategoryType.SHARED
            
        # Derived inputs (Tier 3) follow normal categorization rules
        return base_category
```

## Benefits of the Consolidated Design

### 1. Comprehensive Field Management

- **Three-Tier Classification**: Clear separation of essential, system, and derived fields
- **Storage Optimization**: Intelligent sharing of common fields across configurations
- **Type Preservation**: Maintains complex type information across serialization boundaries
- **Circular Reference Handling**: Robust detection and management of circular references

### 2. Enhanced Maintainability

- **Clean Separation of Concerns**: Each component has a single, well-defined responsibility
- **Explicit Rules**: Categorization rules are clearly defined and easy to understand
- **Improved Testability**: Components can be tested in isolation
- **Better Debugging**: Proper logging and explicit error handling make issues easier to diagnose

### 3. Improved User Experience

- **Focused User Interface**: Users only need to specify essential inputs (Tier 1)
- **Default Handling**: System inputs (Tier 2) have sensible defaults
- **Automatic Derivation**: Derived values (Tier 3) are automatically calculated
- **Consistent API**: Simple public API for common operations

### 4. Robust Architecture

- **Type Safety**: Strong typing throughout the system prevents errors
- **Validation**: Comprehensive validation at multiple levels
- **Error Handling**: Graceful handling of edge cases and errors
- **Extensibility**: Easy to extend with new field types and categorization rules

## Implementation Status

The current implementation in `src/cursus/core/config_fields/` includes:

- ✅ **ConfigFieldCategorizer**: Complete implementation with explicit rules
- ✅ **ConfigFieldTierRegistry**: Complete tier registry with default classifications
- ✅ **TypeAwareConfigSerializer**: Advanced serialization with type preservation
- ✅ **CircularReferenceTracker**: Robust circular reference detection
- ✅ **ConfigMerger**: Configuration merging with field categorization
- ✅ **Constants and Enums**: Type-safe constants and categorization types
- ✅ **Public API Functions**: Simple interface for common operations

## Migration from Legacy Documents

This consolidated document replaces:

1. **config_field_categorization_refactored.md** - Incorporated refactored design concepts
2. **config_field_categorization_three_tier.md** - Integrated three-tier architecture
3. **config_field_categorization.md** - Combined original categorization concepts

The consolidated design maintains backward compatibility while providing a clearer, more comprehensive architecture that aligns with the actual implementation.

## Future Enhancements

Potential areas for future development:

1. **Enhanced Tier Integration**: Deeper integration of tier information in categorization decisions
2. **Performance Optimization**: Caching and optimization for large configuration sets
3. **Validation Framework**: Enhanced validation rules based on tier classifications
4. **Documentation Generation**: Automatic documentation generation from tier classifications
5. **Migration Tools**: Tools to help migrate existing configurations to the three-tier model

## References

- [Type-Aware Serializer](type_aware_serializer.md) - Details on the serialization system
- [Step Builder Registry](step_builder_registry_design.md) - How configuration types map to step builders
- [Config Registry](config_registry.md) - Registration system for configuration classes
- [Circular Reference Tracker](circular_reference_tracker.md) - Details on circular reference handling
- [Dynamic Template](dynamic_template.md) - How templates use the configuration system
- [Standardization Rules](standardization_rules.md) - Rules governing the configuration system
