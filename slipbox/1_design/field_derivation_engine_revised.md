# Field Derivation Engine (Revised)

## Overview

This document presents the revised design for field derivation in the three-tier configuration architecture. Based on the actual implementation in `src/pipeline_steps/config_base.py`, the field derivation logic is now directly embedded within each configuration class rather than being handled by a separate engine. This self-contained approach provides stronger encapsulation, clearer dependencies, and simpler maintenance.

## Self-Contained Derivation Design

### Key Principles

1. **Self-Contained Logic**: Each configuration class handles its own field derivations
2. **Property-Based Access**: Derived values are accessed through read-only properties
3. **Private Implementation**: Derived fields are stored in private attributes
4. **Explicit Dependencies**: Dependencies between fields are clearly defined in property methods

### Implementation in Base Pipeline Config

The `BasePipelineConfig` class demonstrates this approach:

```python
class BasePipelineConfig(BaseModel):
    """Base configuration with shared pipeline attributes and self-contained derivation logic."""
    
    # Class variables using ClassVar for Pydantic
    _REGION_MAPPING: ClassVar[Dict[str, str]] = {
        "NA": "us-east-1",
        "EU": "eu-west-1",
        "FE": "us-west-2"
    }
    
    # For internal caching (completely private)
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    author: str = Field(
        description="Author or owner of the pipeline.")
    bucket: str = Field(
        description="S3 bucket name for pipeline artifacts and data.")
    role: str = Field(
        description="IAM role for pipeline execution.")
    region: str = Field(
        description="Custom region code (NA, EU, FE) for internal logic.")
    service_name: str = Field(
        description="Service name for the pipeline.")
    pipeline_version: str = Field(
        description="Version string for the SageMaker Pipeline.")
    
    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override
    model_class: str = Field(
        default='xgboost', 
        description="Model class (e.g., XGBoost, PyTorch).")
    current_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Current date, typically used for versioning or pathing.")
    
    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    _aws_region: Optional[str] = PrivateAttr(default=None)
    _pipeline_name: Optional[str] = PrivateAttr(default=None)
    _pipeline_description: Optional[str] = PrivateAttr(default=None)
    _pipeline_s3_loc: Optional[str] = PrivateAttr(default=None)
    
    # Public read-only properties for derived fields
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
            pipeline_subdirectory = f"{self.pipeline_name}_{self.pipeline_version}"
            self._pipeline_s3_loc = f"s3://{self.bucket}/MODS/{pipeline_subdirectory}"
        return self._pipeline_s3_loc
```

## Advantages over a Central Field Derivation Engine

### 1. Improved Encapsulation

The self-contained approach provides stronger encapsulation:
- Each configuration class fully owns its derived fields
- Implementation details are hidden through private attributes
- Public API is clean and consistent through property methods

### 2. Clearer Dependencies

Dependencies between fields are explicitly defined in property methods:
- Each property method clearly shows which other fields it depends on
- The dependency chain is easy to trace through method implementations
- Circular dependencies are easier to detect and prevent

### 3. Simplified Maintenance

Maintenance is simplified through class-based organization:
- Related derivation logic is grouped within a single class
- Adding or modifying derivations only affects the relevant class
- Testing is easier with well-defined class boundaries

### 4. Improved Type Safety

Type safety is enhanced through Pydantic integration:
- Private attributes have type annotations
- Property methods include return type annotations
- Pydantic handles validation of user-provided inputs

### 5. Automatic Initialization

The `model_validator` mode='after' ensures derived fields are initialized properly:

```python
@model_validator(mode='after')
def initialize_derived_fields(self) -> 'BasePipelineConfig':
    """Initialize all derived fields once after validation."""
    # Direct assignment to private fields avoids triggering validation
    self._aws_region = self._REGION_MAPPING.get(self.region, "us-east-1")
    self._pipeline_name = f"{self.author}-{self.service_name}-{self.model_class}-{self.region}"
    self._pipeline_description = f"{self.service_name} {self.model_class} Model {self.region}"
    
    pipeline_subdirectory = "MODS"
    pipeline_subsubdirectory = f"{self._pipeline_name}_{self.pipeline_version}"
    self._pipeline_s3_loc = f"s3://{self.bucket}/{pipeline_subdirectory}/{pipeline_subsubdirectory}"
    
    return self
```

This approach:
- Ensures all derived fields are initialized at object creation
- Avoids triggering property getters during validation
- Prevents potential validation loops

## Field Categorization

The configuration classes provide methods to automatically categorize fields:

```python
def categorize_fields(self) -> Dict[str, List[str]]:
    """
    Categorize all fields into three tiers:
    1. Tier 1: Essential User Inputs - public fields with no defaults (required)
    2. Tier 2: System Inputs - public fields with defaults (optional)
    3. Tier 3: Derived Fields - properties that access private attributes
    """
    categories = {
        'essential': [],  # Tier 1: Required, public
        'system': [],     # Tier 2: Optional (has default), public
        'derived': []     # Tier 3: Public properties
    }
    
    # Get model fields from the class
    model_fields = self.__class__.model_fields
    
    # Categorize public fields into essential (required) or system (with defaults)
    for field_name, field_info in model_fields.items():
        # Skip private fields
        if field_name.startswith('_'):
            continue
            
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

This method:
- Automatically detects essential fields (no defaults)
- Identifies system inputs (has defaults)
- Discovers derived fields (property methods)
- Works with inheritance by examining the class structure

## Parent-Child Configuration Relationships

The configuration system supports parent-child relationships through the `from_base_config` class method:

```python
@classmethod
def from_base_config(cls, base_config: 'BasePipelineConfig', **kwargs) -> 'BasePipelineConfig':
    """
    Create a new configuration instance from a base configuration.
    This is a virtual method that all derived classes can use to inherit from a parent config.
    
    Args:
        base_config: Parent BasePipelineConfig instance
        **kwargs: Additional arguments specific to the derived class
        
    Returns:
        A new instance of the derived class initialized with parent fields and additional kwargs
    """
    # Get public fields from parent
    parent_fields = base_config.get_public_init_fields()
    
    # Combine with additional fields (kwargs take precedence)
    config_dict = {**parent_fields, **kwargs}
    
    # Create new instance of the derived class (cls refers to the actual derived class)
    return cls(**config_dict)
```

This method:
- Extracts essential and system inputs from the parent
- Combines them with child-specific fields
- Creates a new instance of the child class
- Preserves polymorphic behavior

## Field Display and Introspection

The configuration system provides methods for displaying field information:

```python
def __str__(self) -> str:
    """
    Custom string representation that shows fields by category.
    This overrides the default __str__ method so that print(config) shows
    a nicely formatted representation with fields organized by tier.
    """
    # Use StringIO to build the string
    from io import StringIO
    output = StringIO()
    
    # Get class name
    print(f"=== {self.__class__.__name__} ===", file=output)
    
    # Get fields categorized by tier
    categories = self.categorize_fields()
    
    # Print each tier of fields
    # ... (implementation details omitted for brevity)
    
    return output.getvalue()
```

The `print_config()` method provides a more detailed view:

```python
def print_config(self) -> None:
    """
    Print complete configuration information organized by tiers.
    This method automatically categorizes fields by examining their characteristics:
    - Tier 1: Essential User Inputs (public fields without defaults)
    - Tier 2: System Inputs (public fields with defaults)
    - Tier 3: Derived Fields (properties that provide access to private fields)
    """
    print("\n===== CONFIGURATION =====")
    print(f"Class: {self.__class__.__name__}")
    
    # Get fields categorized by tier
    categories = self.categorize_fields()
    
    # Print each tier of fields
    # ... (implementation details omitted for brevity)
```

## Cross-Configuration Dependencies

For cases where configurations need to reference each other, the factory pattern is recommended:

```python
class ConfigFactory:
    """Factory for creating configuration objects with cross-dependencies."""
    
    def create_model_eval_config(self, base_config: BasePipelineConfig, 
                                training_config: XGBoostTrainingConfig) -> ModelEvaluationConfig:
        """
        Create a model evaluation configuration that depends on a training configuration.
        
        Args:
            base_config: Base pipeline configuration
            training_config: Training configuration to reference
            
        Returns:
            ModelEvaluationConfig instance with cross-references resolved
        """
        # Get fields from base config
        base_fields = base_config.get_public_init_fields()
        
        # Copy fields from training_config that eval_config needs
        eval_specific_fields = {
            "framework_version": training_config.framework_version,
            "hyperparameters": training_config.hyperparameters
        }
        
        # Create eval config with combined fields
        return ModelEvaluationConfig(**base_fields, **eval_specific_fields)
```

This pattern:
- Handles cross-configuration dependencies explicitly
- Avoids circular references during object creation
- Provides a clear pattern for resolving dependencies

## Integration with Configuration Serialization

The self-contained approach integrates seamlessly with configuration serialization:

```python
def model_dump(self, **kwargs) -> Dict[str, Any]:
    """Override model_dump to include derived properties."""
    data = super().model_dump(**kwargs)
    # Add derived properties to output
    data["aws_region"] = self.aws_region
    data["pipeline_name"] = self.pipeline_name
    data["pipeline_description"] = self.pipeline_description
    data["pipeline_s3_loc"] = self.pipeline_s3_loc
    return data
```

This method:
- Ensures derived fields are included in serialized output
- Maintains compatibility with existing serialization mechanisms
- Provides a complete representation of the configuration

## Conclusion

The field derivation approach implemented in the configuration classes represents a more maintainable and encapsulated design than a separate field derivation engine. By embedding derivation logic directly in the classes through property methods and private attributes, the system provides stronger encapsulation, clearer dependencies, and simpler maintenance.

This design approach aligns with the three-tier configuration architecture while providing a more cohesive implementation. Each configuration class takes full responsibility for its field derivations, resulting in a more modular, maintainable, and type-safe system.

Key advantages include:
1. Improved encapsulation through private attributes and property methods
2. Clearer dependencies explicitly defined in property implementations
3. Simplified maintenance with class-based organization
4. Enhanced type safety through Pydantic integration
5. Automatic field initialization at object creation
6. Support for parent-child relationships through factory methods
7. Comprehensive field categorization and display capabilities
8. Seamless integration with configuration serialization

This approach should be continued and extended for all new configuration classes to maintain consistency and leverage the benefits of self-contained field derivation.
