---
tags:
  - design
  - implementation
  - configuration
  - architecture
keywords:
  - field categorization
  - three-tier architecture
  - configuration management
  - type-aware serialization
  - essential inputs
  - system inputs
  - derived fields
topics:
  - configuration management
  - field organization
  - property design
  - serialization
language: python
date of note: 2025-07-31
---

# Configuration Field Categorization Architecture

## Overview

The Configuration Field Categorization system provides a robust framework for managing configuration fields in the pipeline infrastructure. It implements a three-tier architecture that classifies fields by their purpose and lifecycle, while providing sophisticated serialization, deserialization, and organization capabilities. This design document consolidates the core concepts from various implementation approaches and focuses on the current three-tier design that's implemented in the codebase.

## Core Purpose

The Configuration Field Categorization system serves several critical purposes:

1. **Field Classification** - Categorize fields based on their purpose and relationship to user interaction
2. **Smart Serialization** - Intelligently serialize and deserialize complex configuration objects
3. **Field Organization** - Organize fields to minimize redundancy and maintain clarity
4. **Type Preservation** - Maintain complex type information across serialization boundaries
5. **Circular Reference Management** - Detect and handle circular references in configuration graphs

## Three-Tier Classification Architecture

The foundation of our configuration system is the Three-Tier Classification Architecture, which categorizes all configuration fields into three distinct tiers based on their purpose and lifecycle:

### Tier 1: Essential User Inputs

Fields that represent core business decisions and require direct user input.

**Characteristics:**
- Explicitly provided by users
- No default values
- Public access
- Represent fundamental configuration decisions
- Examples: `region`, `author`, `bucket`, `pipeline_version`

**Implementation:**
```python
class BasePipelineConfig(BaseModel):
    # Tier 1: Essential User Inputs
    region: str = Field(description="Region code (NA, EU, FE)")
    author: str = Field(description="Pipeline author/owner")
    service_name: str = Field(description="Service name for the pipeline")
    bucket: str = Field(description="S3 bucket name for pipeline artifacts")
    pipeline_version: str = Field(description="Version string for the SageMaker Pipeline")
```

### Tier 2: System Inputs with Defaults

Fields with standardized values that have sensible defaults but can be overridden when needed.

**Characteristics:**
- Have reasonable defaults
- Can be overridden
- Public access
- Represent system configuration settings
- Examples: `py_version`, `framework_version`, `instance_type`

**Implementation:**
```python
class BasePipelineConfig(BaseModel):
    # Tier 2: System Inputs with Defaults
    model_class: str = Field(default="xgboost", description="Model class (e.g., XGBoost, PyTorch)")
    py_version: str = Field(default="py310", description="Python version")
    framework_version: str = Field(default="2.1.0", description="Framework version")
    current_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Current date, typically used for versioning or pathing"
    )
```

### Tier 3: Derived Fields

Fields that are calculated from Tier 1 and Tier 2 fields with clear derivation logic.

**Characteristics:**
- Calculated from other fields
- Private attributes with public read-only properties
- Not directly set by users or API
- Examples: `aws_region`, `pipeline_name`, `pipeline_s3_loc`

**Implementation:**
```python
class BasePipelineConfig(BaseModel):
    # Private fields for derived values (Tier 3)
    _aws_region: Optional[str] = PrivateAttr(default=None)
    _pipeline_name: Optional[str] = PrivateAttr(default=None)
    _pipeline_description: Optional[str] = PrivateAttr(default=None)
    _pipeline_s3_loc: Optional[str] = PrivateAttr(default=None)
    
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
    
    # Initialize derived fields at creation time
    @model_validator(mode='after')
    def initialize_derived_fields(self) -> 'BasePipelineConfig':
        """Initialize all derived fields once after validation."""
        # Direct assignment to private fields avoids triggering validation loops
        self._aws_region = self._REGION_MAPPING.get(self.region, "us-east-1")
        self._pipeline_name = f"{self.author}-{self.service_name}-{self.model_class}-{self.region}"
        # Other field initializations...
        return self
```

## Field Categorization for Storage

When serializing and persisting configurations, fields are organized into logical categories to reduce redundancy and improve clarity:

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

1. **Field is special** → Place in `specific`
   - Special fields include those in the `SPECIAL_FIELDS_TO_KEEP_SPECIFIC` list
   - Pydantic models are considered special fields
   - Complex nested structures are considered special fields

2. **Field appears only in one config** → Place in `specific`
   - If a field exists in only one configuration instance, it belongs in that instance's specific section

3. **Field has different values across configs** → Place in `specific`
   - If a field has the same name but different values across multiple configs, each instance goes in specific

4. **Field has identical value across all configs** → Place in `shared`
   - If a field has the same value across all configs, it belongs in shared

5. **Default case** → Place in `specific`
   - When in doubt, place in specific to ensure proper functioning

## Key Components

### 1. ConfigFieldCategorizer

The ConfigFieldCategorizer analyzes and categorizes fields based on their characteristics:

```python
class ConfigFieldCategorizer:
    """Field categorizer with tier awareness"""
    
    def __init__(self, config_list, tier_registry=None):
        self.config_list = config_list
        self.tier_registry = tier_registry or ConfigFieldTierRegistry.DEFAULT_TIER_REGISTRY
        self.field_info = {}  # Field info by name
        self._collect_field_information()
        
    def _collect_field_information(self):
        """Collect and classify field information from configs"""
        for config in self.config_list:
            # Get all fields from config
            config_dict = config.model_dump()
            
            # Process each field
            for name, value in config_dict.items():
                # Skip special internal fields
                if name.startswith('_'):
                    continue
                    
                # Create or update field info
                if name not in self.field_info:
                    self.field_info[name] = FieldInfo(name)
                    
                # Add this config and value to field info
                self.field_info[name].configs.append(config)
                self.field_info[name].values[id(config)] = value
                
                # Check if field is special
                self.field_info[name].is_special = self._is_special_field(name, value)
                
                # Add tier classification
                self.field_info[name].classify_tier(self.tier_registry)
    
    def categorize_fields(self):
        """
        Categorize fields into shared and specific categories
        
        Returns:
            tuple: (shared_fields, specific_fields)
        """
        shared_fields = {}
        specific_fields = {config_id: {} for config_id in self._get_config_ids()}
        
        # Process each field
        for name, info in self.field_info.items():
            # Skip internal fields
            if name.startswith('_'):
                continue
                
            # Handle special fields and fields with different values
            if info.is_special or not self._has_identical_values(info):
                # Add to specific fields for each config
                for config in info.configs:
                    config_id = self._get_config_id(config)
                    specific_fields[config_id][name] = info.values[id(config)]
            else:
                # Add to shared fields
                shared_fields[name] = next(iter(info.values.values()))
                
        return shared_fields, specific_fields
```

### 2. TypeAwareSerializer

The TypeAwareSerializer preserves type information during serialization:

```python
class TypeAwareSerializer:
    """Serializer that preserves type information"""
    
    def __init__(self, config_classes=None):
        self.config_classes = config_classes or {}
        self.circular_reference_tracker = CircularReferenceTracker()
        
    def serialize(self, obj):
        """Serialize an object with type information"""
        # Handle None
        if obj is None:
            return None
            
        # Handle primitive types
        if isinstance(obj, (str, int, float, bool)):
            return obj
            
        # Handle lists
        if isinstance(obj, list):
            return [self.serialize(item) for item in obj]
            
        # Handle dictionaries
        if isinstance(obj, dict):
            return {key: self.serialize(value) for key, value in obj.items()}
            
        # Handle Pydantic models
        if isinstance(obj, BaseModel):
            # Check for circular references
            is_circular, _ = self.circular_reference_tracker.enter_object(obj)
            if is_circular:
                return None
                
            try:
                # Serialize to dict with type information
                result = {
                    "__model_type__": obj.__class__.__name__,
                    "__model_module__": obj.__class__.__module__
                }
                
                # Add all fields
                for field_name, field_value in obj.model_dump().items():
                    result[field_name] = self.serialize(field_value)
                    
                return result
            finally:
                self.circular_reference_tracker.exit_object()
                
        # Handle other types (datetime, Path, etc.)
        if hasattr(obj, "__dict__"):
            # Similar handling for other complex types
            # ...
            
        # Default fallback
        return str(obj)
    
    def deserialize(self, data, expected_type=None):
        """Deserialize an object with type information"""
        # Similar implementation for deserialization
        # ...
```

### 3. ConfigMerger

The ConfigMerger combines multiple configurations into a unified structure:

```python
class ConfigMerger:
    """Merger for multiple configurations"""
    
    def __init__(self, config_list):
        self.config_list = config_list
        self.serializer = TypeAwareSerializer()
        self.categorizer = ConfigFieldCategorizer(config_list)
        
    def merge(self):
        """Merge configurations into unified structure"""
        # Categorize fields
        shared_fields, specific_fields = self.categorizer.categorize_fields()
        
        # Get step names for each config
        config_ids = {}
        for config in self.config_list:
            step_name = self.serializer.generate_step_name(config)
            config_ids[id(config)] = step_name
            
        # Create specific fields by step name
        specific_by_step = {}
        for config_id, fields in specific_fields.items():
            if config_id in config_ids:
                step_name = config_ids[config_id]
                specific_by_step[step_name] = fields
            
        # Create metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "config_types": {
                config_ids[id(config)]: config.__class__.__name__
                for config in self.config_list
            }
        }
        
        # Create field sources tracking
        field_sources = self.categorizer.get_field_sources()
        if field_sources:
            metadata["field_sources"] = field_sources
            
        # Create final structure
        result = {
            "metadata": metadata,
            "configuration": {
                "shared": shared_fields,
                "specific": specific_by_step
            }
        }
        
        return result
    
    def save(self, output_file):
        """Save merged configuration to file"""
        merged = self.merge()
        
        with open(output_file, 'w') as f:
            json.dump(merged, f, indent=2)
            
        return merged
```

### 4. CircularReferenceTracker

The CircularReferenceTracker detects and manages circular references:

```python
class CircularReferenceTracker:
    """Tracks object references to detect and handle circular references"""
    
    def __init__(self, max_depth=100):
        self.processing_stack = []  # Stack of currently processing objects
        self.object_id_to_path = {}  # Maps object IDs to their path in the object graph
        self.current_path = []       # Current path in the object graph
        self.max_depth = max_depth
        
    def enter_object(self, obj_data):
        """
        Start tracking a new object, returns whether a circular reference was detected.
        
        Args:
            obj_data: The object to track
            
        Returns:
            tuple: (is_circular, error_message)
        """
        # Generate an ID for the object
        obj_id = self._generate_object_id(obj_data)
        
        # Check for circular reference
        if obj_id in self.object_id_to_path:
            error_msg = self._format_cycle_error(obj_data, obj_id)
            return True, error_msg
            
        # Check for maximum depth
        if len(self.current_path) >= self.max_depth:
            error_msg = self._format_depth_error()
            return True, error_msg
            
        # Update tracking
        node_info = {
            'id': obj_id,
            'type': self._get_type_name(obj_data),
            'identifier': self._get_identifier(obj_data)
        }
        
        self.processing_stack.append(node_info)
        self.current_path.append(node_info)
        self.object_id_to_path[obj_id] = list(self.current_path)
        
        return False, None
        
    def exit_object(self):
        """Mark that processing of the current object is complete"""
        if self.processing_stack:
            self.processing_stack.pop()
        if self.current_path:
            self.current_path.pop()
            
    def _generate_object_id(self, obj_data):
        """Generate a reliable ID for an object to detect circular refs"""
        if isinstance(obj_data, dict):
            type_name = obj_data.get('__model_type__')
            if type_name:
                # Create composite ID from type and identifiers
                id_parts = [type_name]
                for key in ['name', 'pipeline_name', 'id', 'step_name']:
                    if key in obj_data:
                        id_parts.append(f"{key}:{obj_data[key]}")
                return hash(tuple(id_parts))
        
        # Fallback to object ID
        return id(obj_data)
```

### 5. ConfigFieldTierRegistry

A registry for field tier classifications:

```python
class ConfigFieldTierRegistry:
    """Registry for field tier classifications"""
    
    # Default tier classifications based on field analysis
    DEFAULT_TIER_REGISTRY = {
        # Essential User Inputs (Tier 1)
        "region": 1,
        "author": 1,
        "bucket": 1,
        "pipeline_version": 1,
        # ... other essential inputs
        
        # System Inputs (Tier 2)
        "py_version": 2,
        "framework_version": 2,
        "processing_instance_type": 2,
        # ... other system inputs
        
        # All other fields default to Tier 3 (derived)
    }
    
    @classmethod
    def get_tier(cls, field_name):
        """Get tier classification for a field"""
        return cls.DEFAULT_TIER_REGISTRY.get(field_name, 3)  # Default to Tier 3
```

## Implementation in BasePipelineConfig

The BasePipelineConfig class implements the three-tier architecture:

```python
class BasePipelineConfig(BaseModel):
    """Base pipeline configuration with three-tier field structure"""
    
    # Tier 1: Essential User Inputs
    region: str = Field(description="Region code (NA, EU, FE)")
    author: str = Field(description="Pipeline author/owner")
    service_name: str = Field(description="Service name for the pipeline")
    bucket: str = Field(description="S3 bucket name for pipeline artifacts")
    pipeline_version: str = Field(description="Version string for the SageMaker Pipeline")
    
    # Tier 2: System Inputs with Defaults
    model_class: str = Field(default="xgboost", description="Model class (e.g., XGBoost, PyTorch)")
    py_version: str = Field(default="py310", description="Python version")
    framework_version: str = Field(default="2.1.0", description="Framework version")
    current_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Current date, typically used for versioning or pathing"
    )
    
    # Private fields for derived values (Tier 3)
    _aws_region: Optional[str] = PrivateAttr(default=None)
    _pipeline_name: Optional[str] = PrivateAttr(default=None)
    _pipeline_description: Optional[str] = PrivateAttr(default=None)
    _pipeline_s3_loc: Optional[str] = PrivateAttr(default=None)
    
    # Region mapping for derived aws_region
    _REGION_MAPPING: ClassVar[Dict[str, str]] = {
        "NA": "us-east-1",
        "EU": "eu-west-1",
        "FE": "us-west-2"
    }
    
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
    
    @property
    def pipeline_description(self) -> str:
        """Get pipeline description derived from service_name, model_class, and region."""
        if self._pipeline_description is None:
            self._pipeline_description = f"{self.service_name} {self.model_class} Model {self.region}"
        return self._pipeline_description
    
    @property
    def pipeline_s3_loc(self) -> str:
        """Get S3 location for pipeline artifacts."""
        if self._pipeline_s3_loc is None:
            pipeline_subdirectory = "MODS"
            pipeline_subsubdirectory = f"{self.pipeline_name}_{self.pipeline_version}"
            self._pipeline_s3_loc = f"s3://{self.bucket}/{pipeline_subdirectory}/{pipeline_subsubdirectory}"
        return self._pipeline_s3_loc
    
    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        # Add derived properties to output
        data["aws_region"] = self.aws_region
        data["pipeline_name"] = self.pipeline_name
        data["pipeline_description"] = self.pipeline_description
        data["pipeline_s3_loc"] = self.pipeline_s3_loc
        return data
    
    # Initialize derived fields at creation time
    @model_validator(mode='after')
    def initialize_derived_fields(self) -> 'BasePipelineConfig':
        """Initialize all derived fields once after validation."""
        # Direct assignment to private fields avoids triggering validation loops
        self._aws_region = self._REGION_MAPPING.get(self.region, "us-east-1")
        self._pipeline_name = f"{self.author}-{self.service_name}-{self.model_class}-{self.region}"
        self._pipeline_description = f"{self.service_name} {self.model_class} Model {self.region}"
        
        pipeline_subdirectory = "MODS"
        pipeline_subsubdirectory = f"{self._pipeline_name}_{self.pipeline_version}"
        self._pipeline_s3_loc = f"s3://{self.bucket}/{pipeline_subdirectory}/{pipeline_subsubdirectory}"
        
        return self
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
    # Use all registered config classes if none provided
    config_classes = config_classes or ConfigRegistry.get_all_classes()
    
    # Load JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract metadata and configuration
    metadata = data.get('metadata', {})
    config_types = metadata.get('config_types', {})
    config_data = data.get('configuration', {})
    
    # Get shared and specific data
    shared = config_data.get('shared', {})
    specific = config_data.get('specific', {})
    
    # Create result dict to store instantiated configs
    result = {}
    
    # Process each step in config_types
    for step_name, class_name in config_types.items():
        # Get the config class
        if class_name not in config_classes:
            logger.warning(f"Unknown config class: {class_name}")
            continue
        
        cls = config_classes[class_name]
        
        # Combine shared and specific data
        fields = {}
        fields.update(shared)  # Add shared fields (lower precedence)
        
        # Add specific fields (higher precedence)
        if step_name in specific:
            fields.update(specific[step_name])
        
        # Create the config instance
        try:
            # Deserialize any nested fields
            deserializer = TypeAwareSerializer(config_classes)
            for field_name, value in list(fields.items()):
                if isinstance(value, dict) and field_name in cls.model_fields:
                    field_type = cls.model_fields[field_name].annotation
                    fields[field_name] = deserializer.deserialize(value, field_type)
            
            # Create instance - derived fields will be initialized through model validators
            instance = cls(**fields)
            result[step_name] = instance
        except Exception as e:
            logger.error(f"Failed to instantiate {class_name}: {str(e)}")
    
    return result
```

## Integration with Other Components

### Integration with Step Builder Registry

The configuration field categorization system integrates with the [Step Builder Registry](step_builder_registry_design.md):

```python
def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
    """Auto-map DAG nodes to configurations using step builder registry."""
    config_map = {}
    
    for node_name in self._dag.get_node_names():
        # Find config for this node
        config_finder = ConfigFinder(self.configs)
        config = config_finder.find_config_for_node(node_name)
        
        if config is None:
            raise ValueError(f"No configuration found for DAG node: {node_name}")
            
        config_map[node_name] = config
        
    return config_map
```

### Integration with Config Registry

The configuration field categorization system uses the [Config Registry](config_registry.md) for class management:

```python
def load_configs(input_file: str, config_classes: Dict[str, Type] = None) -> Dict[str, Any]:
    # Use all registered config classes if none provided
    config_classes = config_classes or ConfigRegistry.get_all_classes()
    # ...
```

### Integration with Dynamic Template

The [Dynamic Template](dynamic_template.md) uses the configuration field categorization system:

```python
# In DynamicPipelineTemplate
def _detect_config_classes(self) -> Dict[str, Type[BasePipelineConfig]]:
    """Automatically detect required config classes based on config file."""
    from ..config_field_manager.utils import detect_config_classes_from_json
    detected_classes = detect_config_classes_from_json(self.config_path)
    return detected_classes
```

## Benefits of the Three-Tier Design

### 1. Clear Separation of Concerns

- **User API Separation**: Essential inputs are clearly distinguished from system inputs and derived values
- **Field Lifecycle Management**: Each field has a defined origin and lifecycle
- **Encapsulated Derivation Logic**: Derived fields are encapsulated in their respective classes

### 2. Enhanced Maintainability

- **Self-Contained Logic**: Each configuration class handles its own field derivation
- **Reduced Coupling**: No central derivation engine or complex dependency management
- **Explicit Field Classification**: Fields are explicitly classified by their tier

### 3. Improved User Experience

- **Focused User Interface**: Users only need to specify essential inputs
- **Default Handling**: System inputs have sensible defaults
- **Automatic Derivation**: Derived values are automatically calculated

### 4. Type Safety and Validation

- **Early Validation**: Essential inputs are validated at configuration creation
- **Type-Safe Derivation**: Derived fields maintain proper typing
- **Serialization Type Preservation**: Complex types are correctly reconstructed during deserialization

## Migration Considerations

When migrating existing code to use the three-tier architecture:

1. **Identify Field Tiers**: Analyze existing fields to classify them into the three tiers
2. **Convert Derived Fields**: Move derived fields to private attributes with properties
3. **Add Model Validators**: Implement model_validator to initialize derived fields
4. **Update Serialization**: Ensure TypeAwareSerializer handles all field types correctly
5. **Test Field Categorization**: Verify that fields are correctly categorized during serialization

## References

- [Type-Aware Serializer](type_aware_serializer.md) - Details on the serialization system
- [Step Builder Registry](step_builder_registry_design.md) - How configuration types map to step builders
- [Config Registry](config_registry.md) - Registration system for configuration classes
- [Circular Reference Tracker](circular_reference_tracker.md) - Details on circular reference handling
- [Dynamic Template](dynamic_template.md) - How templates use the configuration system
