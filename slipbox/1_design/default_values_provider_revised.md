# Default Values Provider (Revised)

## Overview

This document presents the revised design for providing default values in the three-tier configuration architecture. Based on the actual implementation in `src/pipeline_steps/config_base.py`, system inputs (Tier 2) are handled directly within configuration classes as fields with default values rather than through a separate provider. This integrated approach simplifies the architecture while maintaining all intended functionality.

## Integrated Default Values Design

### Key Principles

1. **Field-Level Defaults**: Default values are defined directly in field declarations
2. **Self-Contained**: Each configuration class defines its own system input defaults
3. **Type-Safe Defaults**: Default values have proper type annotations
4. **Dynamic Defaults**: Support for both static and callable default factories

### Implementation in Base Pipeline Config

The `BasePipelineConfig` class demonstrates this approach:

```python
class BasePipelineConfig(BaseModel):
    """Base configuration with shared pipeline attributes and self-contained derivation logic."""
    
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
    
    framework_version: str = Field(
        default='2.1.0', 
        description="Default framework version (e.g., PyTorch).")
    
    py_version: str = Field(
        default='py310', 
        description="Default Python version.")
    
    source_dir: Optional[str] = Field(
        default=None, 
        description="Common source directory for scripts if applicable. Can be overridden by step configs.")
```

## Advantages over a Separate Default Values Provider

### 1. Simplified Architecture

The integrated approach provides a simpler architecture:
- No separate provider class to maintain
- No external dependency for configuration classes
- Clearer code organization with defaults defined alongside fields

### 2. Improved Type Safety

Type safety is enhanced through direct field definitions:
- Field types are enforced by Pydantic
- Default values must match the field type
- IDE auto-completion and type checking work correctly

### 3. Better Documentation

Documentation is more accessible:
- Default values are documented alongside field definitions
- Field descriptions explain the purpose and context of defaults
- The relationship between fields and their defaults is clear

### 4. Natural Validation

Validation occurs naturally through Pydantic:
- Default values are validated like any other field value
- Type constraints are enforced automatically
- Custom validators can be applied to default values

### 5. Inheritance Support

The approach works seamlessly with inheritance:
- Derived classes inherit default values from parent classes
- Defaults can be overridden in derived classes
- Parent-child relationships are preserved

## Default Value Handling

### Static Defaults

Simple static defaults are defined directly in the field:

```python
model_class: str = Field(
    default='xgboost', 
    description="Model class (e.g., XGBoost, PyTorch).")
```

### Dynamic Defaults with default_factory

For dynamic defaults, the `default_factory` parameter is used:

```python
current_date: str = Field(
    default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
    description="Current date, typically used for versioning or pathing.")
```

### Context-Aware Defaults

For more complex context-aware defaults, methods within the class can be used:

```python
@field_validator('processing_entry_point', mode='before')
def _set_default_entry_point(cls, v, info: ValidationInfo):
    """Set default entry point based on configuration type."""
    if v is not None:
        return v
        
    # Determine entry point based on class name
    config_type = info.data.get('__class__.__name__', '')
    entry_point_map = {
        "TabularPreprocessingConfig": "tabular_preprocess.py",
        "ModelCalibrationConfig": "model_calibration.py",
        "PayloadConfig": "mims_payload.py",
        "XGBoostTrainingConfig": "train_xgb.py"
    }
    return entry_point_map.get(config_type, "processing.py")
```

## Field Categorization

The configuration classes automatically categorize fields based on whether they have defaults:

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
    
    # Find derived properties
    for attr_name in dir(self):
        if (not attr_name.startswith('_') and 
            attr_name not in model_fields and
            isinstance(getattr(type(self), attr_name, None), property)):
            categories['derived'].append(attr_name)
    
    return categories
```

This method:
- Automatically identifies fields with defaults as system inputs (Tier 2)
- Identifies fields without defaults as essential user inputs (Tier 1)
- Finds property methods that provide derived fields (Tier 3)

## Parent-Child Default Inheritance

When creating child configurations from parent configurations, the defaults are preserved:

```python
def get_public_init_fields(self) -> Dict[str, Any]:
    """
    Get a dictionary of public fields suitable for initializing a child config.
    Only includes fields that should be passed to child class constructors.
    Both essential user inputs and system inputs with defaults or user-overridden values
    are included to ensure all user customizations are properly propagated to derived classes.
    """
    # Use categorize_fields to get essential and system fields
    categories = self.categorize_fields()
    
    # Initialize result dict
    init_fields = {}
    
    # Add all essential fields (Tier 1)
    for field_name in categories['essential']:
        init_fields[field_name] = getattr(self, field_name)
    
    # Add all system fields (Tier 2) that aren't None
    for field_name in categories['system']:
        value = getattr(self, field_name)
        if value is not None:  # Only include non-None values
            init_fields[field_name] = value
    
    return init_fields
```

This method:
- Extracts both essential inputs and system inputs with their current values
- Ensures that user-overridden default values are preserved
- Provides a clean way to pass configuration from parent to child

## Configuration Factory Pattern

For more complex default value handling, especially when dealing with cross-configuration dependencies, a factory pattern is recommended:

```python
class ConfigFactory:
    """Factory for creating configuration objects with sensible defaults."""
    
    def create_training_config(self, base_config: BasePipelineConfig, 
                              data_config: DataConfig = None) -> TrainingConfig:
        """
        Create a training configuration with appropriate defaults.
        
        Args:
            base_config: Base pipeline configuration
            data_config: Optional data configuration for deriving data-dependent defaults
            
        Returns:
            TrainingConfig instance with appropriate defaults
        """
        # Get fields from base config
        base_fields = base_config.get_public_init_fields()
        
        # Set up training-specific defaults
        training_defaults = {
            "training_instance_type": "ml.m5.4xlarge",
            "training_instance_count": 1,
            "training_volume_size": 800,
            "training_entry_point": "train_xgb.py"
        }
        
        # If data_config is provided, derive additional defaults
        if data_config:
            # Example: Set different instance type based on data size
            if hasattr(data_config, 'data_size_gb') and data_config.data_size_gb > 100:
                training_defaults["training_instance_type"] = "ml.m5.8xlarge"
                training_defaults["training_volume_size"] = 1600
        
        # Create config with combined fields
        return TrainingConfig(**base_fields, **training_defaults)
```

This pattern:
- Provides a centralized place for complex default logic
- Handles cross-configuration dependencies explicitly
- Enables context-aware default generation

## Default Value Categories

The integrated default values approach covers several categories of system inputs:

### Infrastructure Settings

```python
# Example infrastructure defaults
processing_instance_type_large: str = Field(
    default="ml.m5.4xlarge",
    description="Large processing instance type")
    
processing_instance_type_small: str = Field(
    default="ml.m5.xlarge", 
    description="Small processing instance type")
    
processing_instance_count: int = Field(
    default=1,
    description="Number of processing instances")
    
processing_volume_size: int = Field(
    default=500,
    description="Storage volume size in GB")
```

### Framework Settings

```python
# Example framework defaults
framework_version: str = Field(
    default="2.1.0",
    description="Framework version for training")
    
py_version: str = Field(
    default="py310",
    description="Python version")
```

### Algorithm Parameters

```python
# Example algorithm defaults
metric_choices: List[str] = Field(
    default_factory=lambda: ["f1_score", "auroc"],
    description="Evaluation metrics")
    
eval_metric: List[str] = Field(
    default_factory=lambda: ["logloss", "auc"],
    description="XGBoost evaluation metrics")
```

### Process Configuration

```python
# Example process configuration defaults
test_val_ratio: float = Field(
    default=0.5,
    description="Test/validation split ratio")
    
calibration_method: str = Field(
    default="gam",
    description="Calibration algorithm")
```

## Custom Display and Introspection

The configuration system includes methods for displaying field information, including default values:

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
    
    # Print Tier 2 fields (system inputs with defaults)
    print("\n----- System Inputs with Defaults (Tier 2) -----")
    for field_name in sorted(categories['system']):
        value = getattr(self, field_name)
        if value is not None:  # Skip None values for cleaner output
            print(f"{field_name.title()}: {value}")
```

This method:
- Shows system inputs separately from essential inputs
- Clearly identifies fields that have defaults
- Provides a view of the current values (default or user-overridden)

## Conclusion

The integrated default values approach implemented in the configuration classes provides a simpler, more maintainable design than a separate default values provider. By defining default values directly in field declarations, the system achieves better type safety, clearer documentation, and simpler architecture.

This design approach aligns with the three-tier configuration architecture while providing a more cohesive implementation. Each configuration class takes full responsibility for its field definitions and default values, resulting in a more modular, maintainable, and type-safe system.

Key advantages include:
1. Simplified architecture with fewer components
2. Improved type safety through Pydantic field definitions
3. Better documentation with defaults alongside field descriptions
4. Natural validation through Pydantic's validation system
5. Seamless integration with inheritance hierarchies
6. Support for both static and dynamic defaults
7. Clear categorization of fields based on default presence
8. Parent-child default value propagation

This approach should be continued and extended for all new configuration classes to maintain consistency and leverage the benefits of integrated default values.
