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

### Base Structure Using Pydantic v2

All configuration classes should use Pydantic v2 for validation and field management:

```python
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict, field_serializer
from typing import Dict, List, Optional, Any, ClassVar

class BasePipelineConfig(BaseModel):
    # Configuration fields here
    ...
    
    # Pydantic v2 model configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
        protected_namespaces=(),
    )
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

### Implementing Portable Path Support (Tier 3 Enhancement)

For configuration classes that handle file paths, implement portable path support as Tier 3 derived fields to enable universal deployment compatibility:

```python
import inspect
from pathlib import Path

class ProcessingStepConfigBase(BasePipelineConfig):
    """Base configuration with portable path support."""
    
    # Tier 2: System fields for paths
    source_dir: Optional[str] = Field(default=None, description="Source directory for scripts")
    processing_source_dir: Optional[str] = Field(default=None, description="Processing-specific source directory")
    processing_entry_point: str = Field(default="script.py", description="Processing script entry point")
    
    # Tier 3: Portable path derived fields (private with property access)
    _portable_source_dir: Optional[str] = PrivateAttr(default=None)
    _portable_processing_source_dir: Optional[str] = PrivateAttr(default=None)
    _portable_script_path: Optional[str] = PrivateAttr(default=None)
    
    @property
    def portable_source_dir(self) -> Optional[str]:
        """Get source directory as relative path for portability."""
        if self.source_dir is None:
            return None
            
        if self._portable_source_dir is None:
            self._portable_source_dir = self._convert_to_relative_path(self.source_dir)
        
        return self._portable_source_dir
    
    @property
    def portable_processing_source_dir(self) -> Optional[str]:
        """Get processing source directory as relative path for portability."""
        if self.processing_source_dir is None:
            return None
            
        if self._portable_processing_source_dir is None:
            self._portable_processing_source_dir = self._convert_to_relative_path(self.processing_source_dir)
        
        return self._portable_processing_source_dir
    
    @property
    def portable_effective_source_dir(self) -> Optional[str]:
        """Get effective source directory as relative path for step builders to use."""
        return self.portable_processing_source_dir or self.portable_source_dir
    
    def get_portable_script_path(self, default_path: Optional[str] = None) -> Optional[str]:
        """Get script path as relative path for portability."""
        if self._portable_script_path is None:
            # Get the absolute script path first
            absolute_script_path = self.get_script_path(default_path)
            if absolute_script_path:
                self._portable_script_path = self._convert_to_relative_path(absolute_script_path)
            else:
                self._portable_script_path = None
        
        return self._portable_script_path
    
    def _convert_to_relative_path(self, path: str) -> str:
        """Convert absolute path to relative path based on config/builder relationship."""
        if not path or not Path(path).is_absolute():
            return path  # Already relative, keep as-is
        
        try:
            # Directory structure analysis:
            # Config location: src/cursus/steps/configs/config_*.py
            # Builder location: src/cursus/steps/builders/builder_*.py
            # Target: Make path relative to builders directory
            
            config_file = Path(inspect.getfile(self.__class__))
            config_dir = config_file.parent      # .../steps/configs/
            steps_dir = config_dir.parent        # .../steps/
            builders_dir = steps_dir / "builders" # .../steps/builders/
            
            # Convert absolute path to be relative from builders directory
            abs_path = Path(path)
            relative_path = abs_path.relative_to(builders_dir)
            
            return str(relative_path)
            
        except (ValueError, OSError):
            # Fallback to common parent approach
            return self._convert_via_common_parent(path)
    
    def _convert_via_common_parent(self, path: str) -> str:
        """Fallback conversion using common parent directory."""
        try:
            config_file = Path(inspect.getfile(self.__class__))
            config_dir = config_file.parent
            abs_path = Path(path)
            
            # Find common parent and create relative path
            common_parent = self._find_common_parent(abs_path, config_dir)
            if common_parent:
                config_to_common = config_dir.relative_to(common_parent)
                common_to_target = abs_path.relative_to(common_parent)
                
                up_levels = len(config_to_common.parts)
                relative_parts = ['..'] * up_levels + list(common_to_target.parts)
                
                return str(Path(*relative_parts))
        
        except (ValueError, OSError):
            pass
        
        # Final fallback: return original path
        return path
    
    def _find_common_parent(self, path1: Path, path2: Path) -> Optional[Path]:
        """Find common parent directory of two paths."""
        try:
            parts1 = path1.parts
            parts2 = path2.parts
            
            common_parts = []
            for p1, p2 in zip(parts1, parts2):
                if p1 == p2:
                    common_parts.append(p1)
                else:
                    break
            
            if common_parts:
                return Path(*common_parts)
        except Exception:
            pass
        
        return None

    # Enhanced model_dump to include portable paths in serialization
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include both original and portable paths."""
        data = super().model_dump(**kwargs)
        
        # Add portable paths as additional fields - keep original paths intact
        if self.portable_source_dir is not None:
            data["portable_source_dir"] = self.portable_source_dir
        
        if self.portable_processing_source_dir is not None:
            data["portable_processing_source_dir"] = self.portable_processing_source_dir
        
        portable_script = self.get_portable_script_path()
        if portable_script is not None:
            data["portable_script_path"] = portable_script
        
        return data
```

#### Key Benefits of Portable Path Support

1. **Universal Deployment Compatibility**: Same configuration files work across development, PyPI, Docker, Lambda environments
2. **Zero Breaking Changes**: Original absolute paths preserved, portable paths added as derived fields
3. **Automatic Fallback**: Step builders use portable paths with automatic fallback to absolute paths
4. **Enhanced Serialization**: Configuration files include both original and portable paths

#### Usage in Step Builders

Step builders can use portable paths with automatic fallback:

```python
# In step builder create_step() method
def create_step(self, **kwargs) -> ProcessingStep:
    # Use portable path with automatic fallback
    script_path = self.config.get_portable_script_path() or self.config.get_script_path()
    source_dir = self.config.portable_effective_source_dir or self.config.get_effective_source_dir()
    
    # Log which path type is being used
    self.log_info("Using script path: %s (portable: %s)", 
                 script_path, 
                 "yes" if self.config.get_portable_script_path() else "no")
    
    return ProcessingStep(
        name=step_name,
        processor=processor,
        code=script_path,
        # ... other parameters
    )
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
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ConfigDict, field_serializer
from datetime import datetime
from pathlib import Path

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
    _objective: Optional[str] = PrivateAttr(default=None)
    _eval_metric: Optional[List[str]] = PrivateAttr(default=None)
    _hyperparameter_file: Optional[str] = PrivateAttr(default=None)
    
    # Non-serializable internal state
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    # Internal mapping as class variable
    _region_mapping: ClassVar[Dict[str, str]] = {
        "NA": "us-east-1", 
        "EU": "eu-west-1", 
        "FE": "us-west-2"
    }
    
    # Pydantic v2 model configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
        protected_namespaces=(),
    )
    
    # Custom serializer for Path fields (if any)
    @field_serializer('pipeline_s3_loc', when_used='json')
    def serialize_path_fields(self, value: Optional[str]) -> Optional[str]:
        """Serialize Path objects to strings"""
        if value is None:
            return None
        return str(value)
    
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
