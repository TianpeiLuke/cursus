---
tags:
  - code
  - core
  - base
  - hyperparameters
  - three_tier_design
keywords:
  - model hyperparameters
  - three-tier configuration
  - field categorization
  - derived fields
  - pydantic model
topics:
  - hyperparameter management
  - configuration design
  - model training
language: python
date of note: 2025-08-07
---

# Model Hyperparameters Base Class

## Overview

The `hyperparameters_base.py` module implements the base model hyperparameters class for training tasks using a sophisticated three-tier design pattern. This class organizes hyperparameter fields into logical categories and provides self-contained derivation logic through private fields and read-only properties.

## Purpose

This module provides:
- **Three-Tier Field Organization**: Essential user inputs, system inputs with defaults, and derived fields
- **Self-Contained Derivation**: Private attributes with public properties for computed values
- **Field Categorization**: Automatic categorization of fields by their characteristics
- **Inheritance Support**: Virtual methods for creating derived hyperparameter classes
- **Serialization Support**: SageMaker-compatible serialization methods

## Three-Tier Design Architecture

### Tier 1: Essential User Inputs
Fields that users must explicitly provide - no defaults available.

```python
# Field lists
full_field_list: List[str] = Field(description="Full list of original field names.")
cat_field_list: List[str] = Field(description="Categorical fields using original names.")
tab_field_list: List[str] = Field(description="Tabular/numeric fields using original names.")

# Identifier and label fields
id_name: str = Field(description="ID field name.")
label_name: str = Field(description="Label field name.")

# Classification parameters
multiclass_categories: List[Union[int, str]] = Field(description="List of unique category labels.")
```

### Tier 2: System Inputs with Defaults
Fields with reasonable defaults that users can override.

```python
# Model and Training Parameters
model_class: str = Field(default='base_model', description="Model class name.")
device: int = Field(default=-1, description="Device ID for training (-1 for CPU).")
lr: float = Field(default=3e-05, description="Learning rate.")
batch_size: int = Field(default=2, gt=0, le=256, description="Batch size for training.")
max_epochs: int = Field(default=3, gt=0, le=10, description="Maximum epochs for training.")

# Optional overrides
class_weights: Optional[List[float]] = Field(default=None, description="Class weights for loss function.")
```

### Tier 3: Derived Fields
Fields calculated from other fields, stored as private attributes with public properties.

```python
# Private attributes for derived values
_input_tab_dim: Optional[int] = PrivateAttr(default=None)
_is_binary: Optional[bool] = PrivateAttr(default=None)
_num_classes: Optional[int] = PrivateAttr(default=None)

# Public read-only properties
@property
def input_tab_dim(self) -> int:
    """Get input tabular dimension derived from tab_field_list."""
    if self._input_tab_dim is None:
        self._input_tab_dim = len(self.tab_field_list)
    return self._input_tab_dim

@property
def num_classes(self) -> int:
    """Get number of classes derived from multiclass_categories."""
    if self._num_classes is None:
        self._num_classes = len(self.multiclass_categories)
    return self._num_classes

@property
def is_binary(self) -> bool:
    """Determine if this is a binary classification task based on num_classes."""
    if self._is_binary is None:
        self._is_binary = (self.num_classes == 2)
    return self._is_binary
```

## Key Features

### Field Categorization

The `categorize_fields()` method automatically categorizes all fields:

```python
def categorize_fields(self) -> Dict[str, List[str]]:
    """
    Categorize all fields into three tiers:
    1. Tier 1: Essential User Inputs - fields with no defaults (required)
    2. Tier 2: System Inputs - fields with defaults (optional)
    3. Tier 3: Derived Fields - properties that access private attributes
    """
    categories = {
        'essential': [],  # Tier 1: Required, public
        'system': [],     # Tier 2: Optional (has default), public
        'derived': []     # Tier 3: Public properties
    }
    
    model_fields = self.__class__.model_fields
    
    # Categorize public fields
    for field_name, field_info in model_fields.items():
        if field_name.startswith('_'):
            continue
            
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

### Custom String Representation

The class provides a custom `__str__` method that organizes output by tier:

```python
def __str__(self) -> str:
    """Custom string representation that shows fields by category."""
    output = StringIO()
    
    print(f"=== {self.__class__.__name__} ===", file=output)
    
    categories = self.categorize_fields()
    
    # Print Tier 1 fields (essential user inputs)
    if categories['essential']:
        print("\n- Essential User Inputs -", file=output)
        for field_name in sorted(categories['essential']):
            print(f"{field_name}: {getattr(self, field_name)}", file=output)
    
    # Print Tier 2 fields (system inputs with defaults)
    if categories['system']:
        print("\n- System Inputs -", file=output)
        for field_name in sorted(categories['system']):
            value = getattr(self, field_name)
            if value is not None:
                print(f"{field_name}: {value}", file=output)
    
    # Print Tier 3 fields (derived properties)
    if categories['derived']:
        print("\n- Derived Fields -", file=output)
        for field_name in sorted(categories['derived']):
            try:
                value = getattr(self, field_name)
                if not callable(value):
                    print(f"{field_name}: {value}", file=output)
            except Exception:
                pass
    
    return output.getvalue()
```

### Validation and Initialization

The class includes comprehensive validation:

```python
@model_validator(mode='after')
def validate_dimensions(self) -> 'ModelHyperparameters':
    """Validate model dimensions and configurations"""
    # Initialize derived fields
    self._input_tab_dim = len(self.tab_field_list)
    self._num_classes = len(self.multiclass_categories)
    self._is_binary = (self._num_classes == 2)
    
    # Set default class_weights if not provided
    if self.class_weights is None:
        self.class_weights = [1.0] * self._num_classes
    
    # Validate class weights length
    if len(self.class_weights) != self._num_classes:
        raise ValueError(f"class_weights length ({len(self.class_weights)}) must match multiclass_categories length ({self._num_classes}).")
    
    # Validate binary classification consistency
    if self._is_binary and self._num_classes != 2:
        raise ValueError("For binary classification, multiclass_categories length must be 2.")
        
    return self
```

## Inheritance and Composition

### Virtual Constructor Pattern

The class provides a virtual constructor for creating derived classes:

```python
@classmethod
def from_base_hyperparam(cls, base_hyperparam: 'ModelHyperparameters', **kwargs) -> 'ModelHyperparameters':
    """
    Create a new hyperparameter instance from a base hyperparameter.
    This is a virtual method that all derived classes can use to inherit from a parent config.
    """
    # Get public fields from parent
    parent_fields = base_hyperparam.get_public_init_fields()
    
    # Combine with additional fields (kwargs take precedence)
    config_dict = {**parent_fields, **kwargs}
    
    # Create new instance of the derived class
    return cls(**config_dict)
```

### Public Field Extraction

The `get_public_init_fields()` method extracts fields suitable for child initialization:

```python
def get_public_init_fields(self) -> Dict[str, Any]:
    """
    Get a dictionary of public fields suitable for initializing a child hyperparameter.
    Only includes fields that should be passed to child class constructors.
    """
    categories = self.categorize_fields()
    init_fields = {}
    
    # Add all essential fields (Tier 1)
    for field_name in categories['essential']:
        init_fields[field_name] = getattr(self, field_name)
    
    # Add all system fields (Tier 2) that aren't None
    for field_name in categories['system']:
        value = getattr(self, field_name)
        if value is not None:
            init_fields[field_name] = value
    
    return init_fields
```

## Serialization Support

### SageMaker Serialization

The class provides SageMaker-compatible serialization:

```python
def serialize_config(self) -> Dict[str, str]:
    """Serialize configuration for SageMaker."""
    # Start with the full model configuration
    config = self.get_config()
    
    # Add derived fields (these won't be in model_dump)
    config["input_tab_dim"] = self.input_tab_dim
    config["is_binary"] = self.is_binary
    config["num_classes"] = self.num_classes
    
    # Serialize all values to strings for SageMaker
    return {
        k: json.dumps(v) if isinstance(v, (list, dict, bool)) else str(v)
        for k, v in config.items()
    }
```

### Configuration Access

```python
def get_config(self) -> Dict[str, Any]:
    """Get the complete configuration dictionary."""
    return self.model_dump()
```

## Usage Patterns

### Basic Usage

```python
# Create hyperparameters with essential inputs
hyperparams = ModelHyperparameters(
    full_field_list=["feature1", "feature2", "feature3", "target"],
    cat_field_list=["feature1"],
    tab_field_list=["feature2", "feature3"],
    id_name="id",
    label_name="target",
    multiclass_categories=["class_a", "class_b"]
)

# Access derived fields
print(f"Input dimension: {hyperparams.input_tab_dim}")  # 2
print(f"Number of classes: {hyperparams.num_classes}")  # 2
print(f"Is binary: {hyperparams.is_binary}")  # True
```

### Inheritance Pattern

```python
class XGBoostHyperparameters(ModelHyperparameters):
    # Additional XGBoost-specific fields
    n_estimators: int = Field(default=100, description="Number of boosting rounds")
    max_depth: int = Field(default=6, description="Maximum tree depth")
    
    @classmethod
    def from_base_hyperparam(cls, base_hyperparam: ModelHyperparameters, **kwargs):
        """Create XGBoost hyperparameters from base hyperparameters."""
        return super().from_base_hyperparam(base_hyperparam, **kwargs)

# Usage
base_params = ModelHyperparameters(...)
xgb_params = XGBoostHyperparameters.from_base_hyperparam(
    base_params,
    n_estimators=200,
    max_depth=8
)
```

### Field Categorization Usage

```python
# Examine field organization
categories = hyperparams.categorize_fields()

print("Essential fields:", categories['essential'])
print("System fields:", categories['system'])  
print("Derived fields:", categories['derived'])

# Print organized view
hyperparams.print_hyperparam()
```

## Design Benefits

### 1. Clear Separation of Concerns
- **Essential fields**: User must provide
- **System fields**: Reasonable defaults available
- **Derived fields**: Computed automatically

### 2. Self-Contained Logic
- No external dependencies for field derivation
- Private attributes prevent external modification
- Public properties provide controlled access

### 3. Inheritance-Friendly
- Virtual constructor pattern enables easy inheritance
- Field categorization works across inheritance hierarchy
- Public field extraction supports composition

### 4. Validation and Type Safety
- Pydantic validation ensures data integrity
- Custom validators enforce business rules
- Type hints provide IDE support

### 5. Serialization Support
- SageMaker-compatible string serialization
- Complete configuration export
- Derived fields included in serialization

## Best Practices

### Field Definition

1. **Essential Fields**: Only include truly required fields
2. **System Fields**: Provide sensible defaults
3. **Derived Fields**: Use private attributes with properties
4. **Validation**: Add custom validators for business rules

### Inheritance

1. **Virtual Constructor**: Always call parent's `from_base_hyperparam`
2. **Field Categories**: Ensure new fields fit the three-tier model
3. **Validation**: Add class-specific validation as needed

### Usage

1. **Field Access**: Use properties for derived fields
2. **Serialization**: Use `serialize_config()` for SageMaker
3. **Debugging**: Use `print_hyperparam()` for inspection
4. **Composition**: Use `get_public_init_fields()` for child creation

## Integration Points

### With Configuration Classes

Hyperparameters are typically embedded in configuration classes:

```python
class TrainingConfig(BasePipelineConfig):
    hyperparameters: ModelHyperparameters = Field(...)
    
    def get_training_args(self) -> Dict[str, str]:
        return self.hyperparameters.serialize_config()
```

### With Step Builders

Step builders use hyperparameters for:
- Environment variable generation
- Training job configuration
- Model parameter validation

### With Training Scripts

Training scripts receive hyperparameters as:
- Environment variables (serialized)
- Command-line arguments
- Configuration files

## Error Handling

The class provides comprehensive error handling:

1. **Validation Errors**: Clear messages for invalid field combinations
2. **Type Errors**: Pydantic handles type validation
3. **Business Logic Errors**: Custom validators enforce constraints
4. **Serialization Errors**: Graceful handling of non-serializable values

## Performance Considerations

1. **Lazy Evaluation**: Derived fields computed on first access
2. **Caching**: Private attributes cache computed values
3. **Memory Efficiency**: Only stores essential data
4. **Serialization**: Efficient string conversion for SageMaker

This three-tier hyperparameter design provides a robust, extensible foundation for managing model training parameters across the entire cursus framework.
