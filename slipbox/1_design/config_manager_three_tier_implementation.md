---
tags:
  - design
  - implementation
  - configuration
  - three-tier
keywords:
  - three-tier configuration
  - field categorization
  - essential inputs
  - system inputs
  - derived fields
  - configuration hierarchy
  - pydantic models
topics:
  - configuration management
  - field organization
  - property design
  - serialization
language: python
date of note: 2025-07-31
---

# Three-Tier Configuration Management Implementation Design

## Overview

This document outlines the implementation of the Three-Tier Configuration Management system for the MODS pipeline. The system provides a clear separation between user-provided inputs, system defaults, and derived values, with a strong focus on encapsulation, type safety, and maintainability.

## Three-Tier Field Classification

All configuration fields are categorized into one of three tiers:

1. **Tier 1: Essential User Inputs**
   - Fields that users must explicitly provide
   - No default values
   - Public access
   - Example: `region`, `author`, `bucket`

2. **Tier 2: System Inputs with Defaults**
   - Fields with reasonable defaults that can be overridden
   - Public access
   - Example: `py_version`, `framework_version`

3. **Tier 3: Derived Fields**
   - Fields calculated from Tier 1 and Tier 2 fields
   - Implemented as private attributes with public read-only properties
   - Example: `aws_region`, `pipeline_name`, `pipeline_s3_loc`

## Implementation Components

The implementation consists of several components working together:

### 1. Self-Contained Configuration Classes

Each configuration class is responsible for its own field derivation logic using Pydantic's features:

```python
class BasePipelineConfig(BaseModel):
    # Tier 1: Essential User Inputs
    region: str = Field(description="Region code (NA, EU, FE)")
    author: str = Field(description="Pipeline author/owner")
    bucket: str = Field(description="S3 bucket for pipeline artifacts")
    
    # Tier 2: System Inputs with Defaults
    py_version: str = Field(default="py310", description="Python version")
    
    # Tier 3: Derived Fields (private with read-only properties)
    _aws_region: Optional[str] = PrivateAttr(default=None)
    _pipeline_name: Optional[str] = PrivateAttr(default=None)
    
    # Public properties for derived fields
    @property
    def aws_region(self) -> str:
        """Get AWS region based on region code."""
        if self._aws_region is None:
            region_mapping = {"NA": "us-east-1", "EU": "eu-west-1", "FE": "us-west-2"}
            self._aws_region = region_mapping.get(self.region, "us-east-1")
        return self._aws_region
    
    @property
    def pipeline_name(self) -> str:
        """Get pipeline name derived from author and service name."""
        if self._pipeline_name is None:
            self._pipeline_name = f"{self.author}-{self.service_name}-{self.model_class}-{self.region}"
        return self._pipeline_name
    
    # One-time initialization of all derived fields
    @model_validator(mode='after')
    def initialize_derived_fields(self) -> 'BasePipelineConfig':
        """Initialize all derived fields once after validation."""
        # Direct assignment to private fields avoids triggering validation loops
        self._aws_region = self._get_aws_region()
        self._pipeline_name = self._get_pipeline_name()
        return self
```

### 2. Automatic Field Categorization

A system that automatically categorizes fields based on their characteristics:

```python
def categorize_fields(self) -> Dict[str, List[str]]:
    """
    Categorize fields into three tiers:
    1. Tier 1: Essential User Inputs - public fields with no defaults
    2. Tier 2: System Inputs - public fields with defaults
    3. Tier 3: Derived Fields - properties that access private attributes
    """
    categories = {
        'essential': [],  # Tier 1
        'system': [],     # Tier 2
        'derived': []     # Tier 3
    }
    
    # Get model fields
    model_fields = self.__class__.model_fields
    
    # Categorize public fields into essential or system
    for field_name, field_info in model_fields.items():
        if field_name.startswith('_'):
            continue  # Skip private fields
            
        # Use is_required() to determine if a field is essential
        if field_info.is_required():
            categories['essential'].append(field_name)
        else:
            categories['system'].append(field_name)
    
    # Find derived properties (public properties that aren't in model_fields)
    for attr_name in dir(self):
        if (not attr_name.startswith('_') and 
            attr_name not in model_fields and
            isinstance(getattr(type(self), attr_name, None), property)):
            categories['derived'].append(attr_name)
    
    return categories
```

### 3. Tier-Aware Serialization

A serializer that preserves type information and handles the three-tier structure:

```python
def serialize_config(config: Any) -> Dict[str, Any]:
    """
    Serialize a configuration object with three-tier awareness.
    
    Args:
        config: Configuration object to serialize
        
    Returns:
        Dict containing serialized configuration with type metadata
    """
    serializer = TypeAwareConfigSerializer()
    result = serializer.serialize(config)
    
    # Ensure metadata with step name is present
    if "_metadata" not in result:
        result["_metadata"] = {
            "step_name": serializer.generate_step_name(config),
            "config_type": config.__class__.__name__,
        }
    
    return result
```

### 4. Tier-Aware Merger

A system for merging multiple configuration objects that respects the three-tier structure:

```python
def merge_and_save_configs(config_list: List[Any], output_file: str) -> Dict[str, Any]:
    """
    Merge and save multiple configs with three-tier awareness.
    
    Args:
        config_list: List of configuration objects
        output_file: Path to output JSON file
        
    Returns:
        Dict containing merged configuration
    """
    # Create merger that understands three-tier structure
    merger = ConfigMerger(config_list)
    
    # Get merged configuration with proper tier handling
    merged = merger.merge()
    
    # Create metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'config_types': {}
    }
    
    # Add mapping of step names to class names
    for cfg in config_list:
        step_name = generate_step_name(cfg)
        class_name = cfg.__class__.__name__
        metadata['config_types'][step_name] = class_name
    
    # Create output structure
    output = {
        'metadata': metadata,
        'configuration': merged
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    return merged
```

### 5. Tier-Aware Deserialization

A system for loading configurations that properly initializes the three-tier structure:

```python
def load_configs(input_file: str, config_classes: Dict[str, Type] = None) -> Dict[str, Any]:
    """
    Load configurations from a file with three-tier awareness.
    
    Args:
        input_file: Path to input JSON file
        config_classes: Optional mapping of class names to class types
        
    Returns:
        Dict mapping step names to instantiated config objects
    """
    # Use all registered config classes if none provided
    config_classes = config_classes or ConfigClassStore.get_all_classes()
    
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
            deserializer = TypeAwareConfigSerializer(config_classes)
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

## Design Patterns Used

### 1. Three-Tier Field Organization

**Pattern**: Explicit field categorization into three distinct tiers based on field characteristics.

**Implementation**:
- Tier 1 (Essential User Inputs): Fields with no defaults, required from users
- Tier 2 (System Inputs): Fields with defaults that can be overridden
- Tier 3 (Derived Fields): Private fields with read-only properties, derived from Tier 1 and 2

**Benefits**:
- Clear separation between user API and internal implementation
- Ensures derived fields cannot be accidentally overridden
- Makes documentation and usage intuitive

### 2. Self-Contained Derivation

**Pattern**: Each configuration class encapsulates its own derivation logic.

**Implementation**:
- Private attributes store derived values
- Public properties provide read-only access to derived values
- Model validator initializes all derived fields at once

**Benefits**:
- Eliminates the need for a central derivation engine
- Localizes derivation logic to the classes that need it
- Improves maintainability and testability

### 3. Automatic Field Classification

**Pattern**: Dynamic field classification based on field characteristics.

**Implementation**:
- Examine field definitions to determine tier (required vs optional)
- Find property methods to identify derived fields
- Create categorized lists for various operations

**Benefits**:
- Eliminates manual field list maintenance
- Automatically adapts to class changes
- Provides a uniform way to handle fields across tiers

### 4. Property-Based Derivation

**Pattern**: Use properties for derived values to avoid calculation loops.

**Implementation**:
- Private attributes store the actual values
- Public properties implement lazy computation and caching
- One-time initialization during object creation

**Benefits**:
- Avoids recalculating values unnecessarily
- Prevents infinite validation loops
- Maintains clean API while encapsulating implementation

### 5. Type-Aware Serialization

**Pattern**: Preserve type information during serialization for correct reconstruction.

**Implementation**:
- Include class type information in serialized data
- Handle nested objects and special types
- Use a registry to map class names to types

**Benefits**:
- Ensures correct reconstruction of complex objects
- Preserves relationships between objects
- Supports polymorphism in the configuration system

## Field Categorization and Serialization Strategy

### Field Categorization Rules

Fields are categorized into tiers based on the following rules:

1. **Tier 1 (Essential User Inputs)**:
   - Fields with no default values (`is_required()` returns True)
   - Public fields (not starting with `_`)
   - Required for object instantiation

2. **Tier 2 (System Inputs with Defaults)**:
   - Fields with default values (`is_required()` returns False)
   - Public fields (not starting with `_`)
   - Optional for object instantiation

3. **Tier 3 (Derived Fields)**:
   - Properties that are not in `model_fields`
   - Public properties (not starting with `_`)
   - Computed based on Tier 1 and Tier 2 fields
   - Not directly provided during instantiation

### Serialization Strategy

During serialization, fields are handled differently based on their tier:

1. **Tier 1 and Tier 2 Fields**:
   - Always included in serialized output
   - Placed in `shared` section if consistent across configs
   - Placed in `specific` section if different between configs

2. **Tier 3 Fields**:
   - Not included in serialized output
   - Will be recomputed during deserialization
   - May be included in `model_dump()` output for debugging

### Deserialization Strategy

During deserialization, the process respects the three-tier structure:

1. **Input Processing**:
   - Only Tier 1 and Tier 2 fields are passed to constructor
   - Complex nested objects are recursively deserialized

2. **Object Creation**:
   - Object is instantiated with processed fields
   - `model_validator(mode='after')` initializes all derived fields

3. **Field Precedence**:
   - Fields in `specific` section have higher precedence than `shared`
   - This ensures step-specific values override shared values

## Benefits Over Previous Approach

The three-tier approach provides several advantages over the previous centralized field derivation system:

1. **Enhanced Encapsulation**:
   - Derived fields are private with controlled access
   - Field derivation logic is encapsulated in each class

2. **Improved Maintainability**:
   - Each class is responsible for its own derivation logic
   - No complex central engine to maintain
   - Easier to understand field relationships

3. **Type Safety**:
   - Pydantic ensures type safety for all fields
   - Field annotations provide clear documentation
   - Type preservation during serialization/deserialization

4. **Better Testability**:
   - Classes can be tested in isolation
   - Derived fields can be verified directly
   - No dependencies on external derivation system

5. **Clearer API**:
   - Users only provide essential inputs
   - System defaults provide sensible fallbacks
   - Derived fields are computed consistently

## Migration Path

Analysis of the existing codebase shows that `BasePipelineConfig` already correctly implements the three-tier pattern. To extend this pattern to the rest of the system:

1. **Study the Existing BasePipelineConfig Implementation**:
   - Understand how it categorizes fields into the three tiers
   - Note how it uses private attributes with public properties for derived fields
   - Examine the `initialize_derived_fields` validator that prevents validation loops
   - Study the `categorize_fields` method for automatic tier detection

2. **Extend to Processing and Other Base Classes**:
   - Apply the same three-tier pattern to `ProcessingStepConfigBase` and others
   - Convert existing derived fields to the property pattern with private storage
   - Update field validators to be environment-aware (respect `MODS_SKIP_PATH_VALIDATION`)
   - Ensure proper model validator initialization

3. **Update Step-Specific Configurations**:
   - Apply the three-tier pattern to each step configuration
   - Move derived fields to private attributes with properties
   - Ensure proper inheritance from base classes
   - Implement tier-specific validation and initialization

4. **Update Serialization and Deserialization**:
   - Modify `merge_and_save_configs` to handle the three-tier structure
   - Update `load_configs` to respect three-tier initialization
   - Ensure proper handling of derived fields during serialization/deserialization
   - Handle backward compatibility with older configuration formats

5. **Test and Validate**:
   - Verify that configurations are properly serialized and deserialized
   - Ensure derived fields are correctly computed
   - Check that field precedence is maintained
   - Validate inheritance works correctly

## Example Implementation

### Base Configuration

```python
class BasePipelineConfig(BaseModel):
    """
    Base pipeline configuration with three-tier field structure.
    
    Tier 1: Essential User Inputs (required fields)
    Tier 2: System Inputs with Defaults (optional fields)
    Tier 3: Derived Fields (private fields with public properties)
    """
    
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

### Step-Specific Configuration

```python
class XGBoostTrainingConfig(BasePipelineConfig):
    """
    XGBoost Training configuration with three-tier field structure.
    
    Inherits the three-tier structure from BasePipelineConfig and adds
    training-specific fields.
    """
    
    # Tier 1: Essential User Inputs specific to training
    num_round: int = Field(description="Number of boosting rounds")
    max_depth: int = Field(description="Maximum tree depth")
    min_child_weight: int = Field(description="Minimum child weight")
    is_binary: bool = Field(description="Binary classification flag")
    
    # Tier 2: System Inputs with Defaults specific to training
    training_instance_type: str = Field(default="ml.m5.4xlarge", description="Training instance type")
    training_instance_count: int = Field(default=1, description="Number of training instances")
    training_volume_size: int = Field(default=800, description="Training volume size in GB")
    training_entry_point: str = Field(default="train_xgb.py", description="Training script entry point")
    
    # Private fields for derived values (Tier 3)
    _objective: Optional[str] = PrivateAttr(default=None)
    _eval_metric: Optional[List[str]] = PrivateAttr(default=None)
    _hyperparameter_file: Optional[str] = PrivateAttr(default=None)
    
    # Public properties for derived fields
    
    @property
    def objective(self) -> str:
        """Get XGBoost objective based on classification type."""
        if self._objective is None:
            self._objective = "binary:logistic" if self.is_binary else "multi:softmax"
        return self._objective
    
    @property
    def eval_metric(self) -> List[str]:
        """Get evaluation metrics based on classification type."""
        if self._eval_metric is None:
            self._eval_metric = ['logloss', 'auc'] if self.is_binary else ['mlogloss', 'merror']
        return self._eval_metric
    
    @property
    def hyperparameter_file(self) -> str:
        """Get hyperparameter file path."""
        if self._hyperparameter_file is None:
            self._hyperparameter_file = f"{self.pipeline_s3_loc}/hyperparameters/{self.region}_hyperparameters.json"
        return self._hyperparameter_file
    
    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        data["objective"] = self.objective
        data["eval_metric"] = self.eval_metric
        data["hyperparameter_file"] = self.hyperparameter_file
        return data
    
    # Initialize derived fields at creation time
    @model_validator(mode='after')
    def initialize_derived_fields(self) -> 'XGBoostTrainingConfig':
        """Initialize all derived fields once after validation."""
        # Call parent initializer first
        super().initialize_derived_fields()
        
        # Initialize training-specific derived fields
        self._objective = "binary:logistic" if self.is_binary else "multi:softmax"
        self._eval_metric = ['logloss', 'auc'] if self.is_binary else ['mlogloss', 'merror']
        self._hyperparameter_file = f"{self._pipeline_s3_loc}/hyperparameters/{self.region}_hyperparameters.json"
        
        return self
    
    def to_hyperparameter_dict(self) -> Dict[str, Any]:
        """Convert configuration to hyperparameter dictionary for training."""
        return {
            "num_round": self.num_round,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "objective": self.objective,
            "eval_metric": self.eval_metric
        }
```

### Config Field Manager

```python
class ConfigFieldManager:
    """Manager for handling configuration fields according to the three-tier model."""
    
    @staticmethod
    def categorize_fields(config: Any) -> Dict[str, List[str]]:
        """
        Categorize fields into three tiers:
        1. Tier 1: Essential User Inputs - public fields with no defaults
        2. Tier 2: System Inputs - public fields with defaults
        3. Tier 3: Derived Fields - properties that access private attributes
        """
        categories = {
            'essential': [],  # Tier 1
            'system': [],     # Tier 2
            'derived': []     # Tier 3
        }
        
        # Get model fields
        if not hasattr(config.__class__, 'model_fields'):
            return categories
            
        model_fields = config.__class__.model_fields
        
        # Categorize public fields into essential or system
        for field_name, field_info in model_fields.items():
            if field_name.startswith('_'):
                continue  # Skip private fields
                
            # Use is_required() to determine if a field is essential
            if field_info.is_required():
                categories['essential'].append(field_name)
            else:
                categories['system'].append(field_name)
        
        # Find derived properties (public properties that aren't in model_fields)
        for attr_name in dir(config):
            if (not attr_name.startswith('_') and 
                attr_name not in model_fields and
                isinstance(getattr(type(config), attr_name, None), property)):
                categories['derived'].append(attr_name)
        
        return categories
    
    @staticmethod
    def get_public_init_fields(config: Any) -> Dict[str, Any]:
        """
        Get a dictionary of fields suitable for initializing a child config.
        Only includes Tier 1 and Tier 2 fields (not derived fields).
        """
        # Get field categories
        categories = ConfigFieldManager.categorize_fields(config)
        
        # Initialize result dict
        init_fields = {}
        
        # Add all essential fields (Tier 1)
        for field_name in categories['essential']:
            init_fields[field_name] = getattr(config, field_name)
        
        # Add all system fields (Tier 2) that aren't None
        for field_name in categories['system']:
            value = getattr(config, field_name)
            if value is not None:  # Only include non-None values
                init_fields[field_name] = value
        
        return init_fields
```

### Config Factory

```python
class ConfigFactory:
    """Factory for creating configuration objects."""
    
    @staticmethod
    def create_config(config_class: Type, **kwargs) -> Any:
        """
        Create a configuration instance.
        
        Args:
            config_class: The configuration class to instantiate
            **kwargs: Fields to pass to the constructor
            
        Returns:
            Instantiated configuration object
        """
        # Pass only fields that are in the model_fields (Tier 1 & 2)
        if hasattr(config_class, 'model_fields'):
            init_kwargs = {
                k: v for k, v in kwargs.items() 
                if k in config_class.model_fields and not k.startswith('_')
            }
        else:
            init_kwargs = kwargs
            
        # Create the instance - derived fields will be initialized through validators
        return config_class(**init_kwargs)
        
    @staticmethod
    def from_base_config(cls: Type, base_config: Any, **kwargs) -> Any:
        """
        Create a new configuration instance from a base configuration.
        
        Args:
            cls: Class to instantiate
            base_config: Base configuration to inherit from
            **kwargs: Additional arguments specific to the derived class
            
        Returns:
            Instantiated configuration object
        """
        # Get public fields from parent
        parent_fields = ConfigFieldManager.get_public_init_fields(base_config)
        
        # Combine with additional fields (kwargs take precedence)
        config_dict = {**parent_fields, **kwargs}
        
        # Create new instance of the derived class
        return cls(**config_dict)
```

## Conclusion

The Three-Tier Configuration Management system provides a robust framework for handling configuration fields with clear separation of concerns and strong encapsulation. By categorizing fields into essential user inputs, system inputs with defaults, and derived fields, we create a clean API for users while maintaining internal consistency and type safety.

The self-contained derivation approach eliminates the need for a central derivation engine, making the system more maintainable and extensible. Each configuration class is responsible for its own field derivation, which improves cohesion and reduces coupling.

The tier-aware serialization and deserialization processes ensure that configurations are properly saved and loaded, with derived fields being consistently recomputed during instantiation.

Overall, this design addresses the scalability challenges of the previous approach while providing a more intuitive and maintainable solution for configuration management in the MODS pipeline.
