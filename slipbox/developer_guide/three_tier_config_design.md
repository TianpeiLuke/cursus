# Three-Tier Config Design Implementation Guide

**Version**: 1.0  
**Date**: July 30, 2025  
**Author**: MODS Development Team

## Overview

This guide explains how to implement the Three-Tier Configuration Design pattern in pipeline components. The three-tier design provides clear separation between different types of configuration fields, improving maintainability, encapsulation, and user experience.

## Configuration Field Tiers

The Three-Tier Classification divides configuration fields into three categories:

1. **Tier 1 (Essential Fields)**: Required inputs explicitly provided by users
   - Must be provided by the user
   - No default values allowed
   - Public access

2. **Tier 2 (System Fields)**: Default values that can be overridden by users
   - Have sensible default values
   - Can be overridden by users
   - Public access

3. **Tier 3 (Derived Fields)**: Values calculated from other fields
   - Private fields with leading underscores
   - Values calculated from Tier 1 and Tier 2 fields
   - Accessed through read-only properties
   - Not directly settable by users

## Implementation Guide

### Base Structure Using Pydantic

All configuration classes should use Pydantic for validation and field management:

```python
from pydantic import BaseModel, Field, PrivateAttr
from typing import Dict, List, Optional, Any, ClassVar

class BasePipelineConfig(BaseModel):
    # Configuration fields here
    ...
```

### Implementing Tier 1 (Essential Fields)

Essential fields are required inputs with no defaults:

```python
# Tier 1: Essential fields (required user inputs)
region: str = Field(..., description="AWS region code (NA, EU, FE)")
author: str = Field(..., description="Pipeline author/owner")
service_name: str = Field(..., description="Service name for pipeline")
```

Key characteristics:
- Use `Field(...)` to indicate a required field with no default
- Always add a description to document the field's purpose
- Use appropriate type annotations for validation

### Implementing Tier 2 (System Fields)

System fields have default values but can be overridden:

```python
# Tier 2: System fields (with defaults, can be overridden)
instance_type: str = Field(default="ml.m5.4xlarge", description="Training instance type")
instance_count: int = Field(default=1, description="Number of training instances")
py_version: str = Field(default="py3", description="Python version")
volume_size_gb: int = Field(default=30, description="EBS volume size in GB")
```

Key characteristics:
- Always provide a sensible default value
- Include a description
- Use appropriate type annotations

### Implementing Tier 3 (Derived Fields)

Derived fields are private with public property access:

```python
# Tier 3: Derived fields (private with property access)
_pipeline_name: Optional[str] = Field(default=None, exclude=True)
_aws_region: Optional[str] = Field(default=None, exclude=True)

# Non-serializable internal state
_cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

# Public property for accessing derived values
@property
def pipeline_name(self) -> str:
    """Get pipeline name derived from author, service and region."""
    if self._pipeline_name is None:
        self._pipeline_name = f"{self.author}-{self.service_name}-{self.region}"
    return self._pipeline_name

@property
def aws_region(self) -> str:
    """Get AWS region for the region code."""
    if self._aws_region is None:
        region_mapping = {"NA": "us-east-1", "EU": "eu-west-1", "FE": "us-west-2"}
        self._aws_region = region_mapping.get(self.region, "us-east-1")
    return self._aws_region
```

Key characteristics:
- Private fields start with underscore `_`
- Use `Field(default=None, exclude=True)` to exclude from serialization
- Implement public properties with meaningful docstrings
- Use lazy initialization in properties (calculate only when needed)
- For purely internal state that should never be serialized, use `PrivateAttr`

### Including Derived Fields in Serialization

To include derived fields when serializing a config:

```python
def model_dump(self, **kwargs) -> Dict[str, Any]:
    """Override model_dump to include derived properties."""
    data = super().model_dump(**kwargs)
    
    # Add derived properties to output
    data["aws_region"] = self.aws_region
    data["pipeline_name"] = self.pipeline_name
    
    return data
```

### Model Validation for Derived Fields

For one-time initialization of derived fields, use a model validator:

```python
from pydantic import model_validator

@model_validator(mode='after')
def initialize_derived_fields(self) -> 'YourConfigClass':
    """Initialize derived fields at creation time."""
    # Access properties to trigger initialization
    _ = self.aws_region
    _ = self.pipeline_name
    return self
```

## Complete Example

Here's a complete example of a configuration class using the Three-Tier design:

```python
from typing import Dict, Any, Optional, ClassVar, List
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from datetime import datetime

class TrainingStepConfig(BaseModel):
    """
    Configuration for training steps using the Three-Tier design.
    
    Tier 1: Essential fields (required user inputs)
    Tier 2: System fields (with defaults, can be overridden)
    Tier 3: Derived fields (private with property access)
    """
    
    # Tier 1: Essential user inputs
    region: str = Field(..., description="AWS region code (NA, EU, FE)")
    pipeline_s3_loc: str = Field(..., description="S3 location for pipeline artifacts")
    num_round: int = Field(..., description="Number of boosting rounds")
    max_depth: int = Field(..., description="Maximum tree depth")
    is_binary: bool = Field(..., description="Binary classification flag")
    
    # Tier 2: System inputs with defaults
    training_instance_type: str = Field(default="ml.m5.4xlarge", description="Training instance type")
    training_instance_count: int = Field(default=1, description="Number of training instances")
    training_volume_size: int = Field(default=800, description="Training volume size in GB")
    framework_version: str = Field(default="1.5-1", description="XGBoost framework version")
    py_version: str = Field(default="py3", description="Python version")
    
    # Tier 3: Derived fields (private with property access)
    _objective: Optional[str] = Field(default=None, exclude=True)
    _eval_metric: Optional[List[str]] = Field(default=None, exclude=True)
    _hyperparameter_file: Optional[str] = Field(default=None, exclude=True)
    
    # Non-serializable internal state
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    # Internal mapping as class variable
    _region_mapping: ClassVar[Dict[str, str]] = {
        "NA": "us-east-1", 
        "EU": "eu-west-1", 
        "FE": "us-west-2"
    }
    
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
            self._hyperparameter_file = f"{self.pipeline_s3_loc}/hyperparameters/params.json"
        return self._hyperparameter_file
    
    @property
    def aws_region(self) -> str:
        """Get AWS region from the region code."""
        return self._region_mapping.get(self.region, "us-east-1")
    
    # Optional: Model validator to initialize all derived fields at once
    @model_validator(mode='after')
    def initialize_derived_fields(self) -> 'TrainingStepConfig':
        """Initialize all derived fields."""
        # Access properties to trigger initialization
        _ = self.objective
        _ = self.eval_metric
        _ = self.hyperparameter_file
        return self
    
    # Include derived properties in serialization
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        # Add derived properties to output
        data["objective"] = self.objective
        data["eval_metric"] = self.eval_metric
        data["hyperparameter_file"] = self.hyperparameter_file
        data["aws_region"] = self.aws_region
        return data
    
    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..pipeline_script_contracts.xgboost_training_contract import XGBOOST_TRAINING_CONTRACT
        return XGBOOST_TRAINING_CONTRACT
    
    def to_hyperparameter_dict(self) -> Dict[str, Any]:
        """Convert configuration to hyperparameter dictionary."""
        return {
            "num_round": self.num_round,
            "max_depth": self.max_depth,
            "objective": self.objective,
            "eval_metric": self.eval_metric
        }
```

## Creating Config Objects

When creating a configuration object, only provide the essential fields (Tier 1) and any system fields (Tier 2) you want to override:

```python
# Create config with essential fields and some overridden system fields
config = TrainingStepConfig(
    # Tier 1: Essential fields (required)
    region="NA",
    pipeline_s3_loc="s3://my-bucket/pipeline",
    num_round=300,
    max_depth=10,
    is_binary=True,
    
    # Tier 2: Override some system defaults
    training_instance_type="ml.m5.12xlarge",
    training_volume_size=1000
)

# Access derived properties - these are computed automatically
print(f"Objective: {config.objective}")  # binary:logistic
print(f"Eval metrics: {config.eval_metric}")  # ['logloss', 'auc']
print(f"AWS region: {config.aws_region}")  # us-east-1
```

## Config Inheritance and Composition

### Inheritance Approach

When extending a base configuration:

```python
class SpecializedTrainingConfig(TrainingStepConfig):
    """Specialized training configuration with additional fields."""
    
    # Add specialized essential fields
    special_param: str = Field(..., description="Special parameter")
    
    # Add specialized system fields
    special_system_param: int = Field(default=42, description="Special system parameter")
    
    # Add specialized derived fields
    _special_derived: Optional[str] = Field(default=None, exclude=True)
    
    @property
    def special_derived(self) -> str:
        """Get specialized derived value."""
        if self._special_derived is None:
            self._special_derived = f"{self.special_param}_{self.num_round}"
        return self._special_derived
    
    # Override model_dump to include new derived fields
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["special_derived"] = self.special_derived
        return data
```

### Composition Approach

For more complex configurations, use composition:

```python
class PipelineConfig(BaseModel):
    """Top-level pipeline configuration using composition."""
    
    # Components
    base: BasePipelineConfig
    training: TrainingStepConfig
    evaluation: Optional[EvaluationConfig] = None
    
    def create_config_list(self) -> List[Any]:
        """Create list of configurations for pipeline assembly."""
        configs = []
        
        # Add base config
        configs.append(self.base)
        
        # Create and add training config with base fields
        training_fields = self.base.model_dump()
        training_fields.update(self.training.model_dump())
        configs.append(TrainingStepConfig(**training_fields))
        
        # Add evaluation config if present
        if self.evaluation:
            eval_fields = self.base.model_dump()
            eval_fields.update(self.evaluation.model_dump())
            configs.append(EvaluationConfig(**eval_fields))
        
        return configs
```

## Best Practices

### Field Classification

1. **Be Judicious with Essential Fields**: Only make fields essential (Tier 1) if they absolutely must be provided by users with no reasonable defaults.

2. **Favor System Fields**: Whenever possible, use system fields (Tier 2) with sensible defaults rather than essential fields.

3. **Encapsulate Derivation Logic**: Keep all derivation logic within the property methods for derived fields (Tier 3).

### Property Implementation

1. **Use Lazy Initialization**: Only calculate derived values when first requested, then cache them.

2. **Document with Docstrings**: Always provide clear docstrings for property methods explaining how values are derived.

3. **Handle Edge Cases**: Consider all possible edge cases in property implementations, with appropriate error handling.

### Inheritance and Composition

1. **Follow Liskov Substitution**: Derived classes should be substitutable for their base classes without altering program correctness.

2. **Avoid Validation Loops**: Be careful with property methods that might trigger validation loops.

3. **Use Factory Methods**: For complex object creation, use factory methods or a separate factory class.

### Serialization

1. **Override model_dump**: Always override `model_dump` to include derived properties in serialized output.

2. **Be Consistent**: Ensure consistent behavior between object creation from serialized data and fresh object creation.

## Common Pitfalls

### 1. Validation Loops

**Problem**: Property methods that trigger validators can cause infinite loops.

**Solution**: Use private fields with `exclude=True` and avoid triggering validation in property methods.

### 2. Circular Dependencies

**Problem**: Derived properties that depend on each other can cause circular dependencies.

**Solution**: Break circular dependencies or use a single property method that calculates multiple values.

### 3. Missing Serialization

**Problem**: Derived properties aren't included in serialized output by default.

**Solution**: Override `model_dump` to include derived properties.

### 4. Inconsistent Behavior

**Problem**: Different behavior when creating objects from scratch vs. from serialized data.

**Solution**: Use model validators to ensure consistent derived values regardless of creation method.

### 5. Inefficient Calculations

**Problem**: Repeatedly calculating expensive derived properties.

**Solution**: Implement caching in property methods to calculate values only once.

## Related Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Config Tiered Design Document](../pipeline_design/config_tiered_design.md)
- [Step Builder Implementation Guide](./step_builder.md)
