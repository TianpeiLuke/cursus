---
tags:
  - code
  - core
  - base
  - configuration
  - three-tier
keywords:
  - base configuration
  - pipeline config
  - pydantic model
  - three-tier architecture
  - field derivation
  - self-contained design
  - essential inputs
  - system inputs
  - derived fields
topics:
  - base configuration class
  - three-tier configuration
  - field categorization
  - configuration inheritance
language: python
date of note: 2025-08-08
---

# Base Pipeline Configuration Documentation

## Overview

The `config_base.py` module implements the foundational configuration class for all pipeline steps using the **Three-Tier Configuration Architecture**. This self-contained design pattern ensures each configuration class is responsible for its own field derivations through private fields and read-only properties, following the established three-tier classification system.

## Three-Tier Configuration Architecture

The system categorizes all configuration fields into three distinct tiers based on their purpose and lifecycle:

### Tier 1: Essential User Inputs
- **Purpose**: Required fields that users must explicitly provide
- **Characteristics**: No default values, public access, required for object instantiation
- **Examples**: `author`, `bucket`, `role`, `region`, `service_name`, `pipeline_version`
- **Implementation**: `Field(description="...")` with no default value

### Tier 2: System Inputs with Defaults  
- **Purpose**: Fields with reasonable defaults that can be overridden by users
- **Characteristics**: Have default values, public access, optional for object instantiation
- **Examples**: `model_class`, `current_date`, `framework_version`, `py_version`, `source_dir`
- **Implementation**: `Field(default=value, description="...")` or `Field(default_factory=func, description="...")`

### Tier 3: Derived Fields
- **Purpose**: Fields calculated from Tier 1 and Tier 2 fields
- **Characteristics**: Private attributes with public read-only properties, not directly settable
- **Examples**: `aws_region`, `pipeline_name`, `pipeline_description`, `pipeline_s3_loc`
- **Implementation**: `PrivateAttr(default=None)` with `@property` accessors

## Core Implementation

### BasePipelineConfig Class

The main class that serves as the foundation for all pipeline step configurations:

```python
class BasePipelineConfig(BaseModel):
    """Base configuration with shared pipeline attributes and self-contained derivation logic."""
    
    # Class variables using ClassVar for Pydantic
    _REGION_MAPPING: ClassVar[Dict[str, str]] = {
        "NA": "us-east-1",
        "EU": "eu-west-1", 
        "FE": "us-west-2"
    }
    
    _STEP_NAMES: ClassVar[Dict[str, str]] = {}  # Lazy loaded
    
    # Internal caching (completely private)
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
```

### Field Implementation Patterns

#### Tier 1: Essential User Inputs
```python
# ===== Essential User Inputs (Tier 1) =====
# These are fields that users must explicitly provide

author: str = Field(description="Author or owner of the pipeline.")
bucket: str = Field(description="S3 bucket name for pipeline artifacts and data.")
role: str = Field(description="IAM role for pipeline execution.")
region: str = Field(description="Custom region code (NA, EU, FE) for internal logic.")
service_name: str = Field(description="Service name for the pipeline.")
pipeline_version: str = Field(description="Version string for the SageMaker Pipeline.")
```

#### Tier 2: System Inputs with Defaults
```python
# ===== System Inputs with Defaults (Tier 2) =====
# These are fields with reasonable defaults that users can override

model_class: str = Field(default='xgboost', description="Model class (e.g., XGBoost, PyTorch).")
current_date: str = Field(
    default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
    description="Current date, typically used for versioning or pathing."
)
framework_version: str = Field(default='2.1.0', description="Default framework version (e.g., PyTorch).")
py_version: str = Field(default='py310', description="Default Python version.")
source_dir: Optional[str] = Field(default=None, description="Common source directory for scripts if applicable.")
```

#### Tier 3: Derived Fields
```python
# ===== Derived Fields (Tier 3) =====
# These are fields calculated from other fields, stored in private attributes
# with public read-only properties for access

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
```

## Key Features

### Self-Contained Design Pattern
- **Encapsulation**: Each configuration class manages its own field derivations
- **Private Attributes**: Calculated values stored using `PrivateAttr(default=None)`
- **Public Properties**: Read-only access through properties with lazy evaluation
- **No External Dependencies**: Eliminates need for external field derivation engines

### Region Mapping System
Built-in mapping from custom region codes to AWS regions:
```python
_REGION_MAPPING: ClassVar[Dict[str, str]] = {
    "NA": "us-east-1",
    "EU": "eu-west-1",
    "FE": "us-west-2"
}
```

### Lazy Evaluation Pattern
Properties use lazy initialization with caching:
```python
@property
def pipeline_s3_loc(self) -> str:
    """Get S3 location for pipeline artifacts."""
    if self._pipeline_s3_loc is None:
        pipeline_subdirectory = "MODS"
        pipeline_subsubdirectory = f"{self.pipeline_name}_{self.pipeline_version}"
        self._pipeline_s3_loc = f"s3://{self.bucket}/{pipeline_subdirectory}/{pipeline_subsubdirectory}"
    return self._pipeline_s3_loc
```

## Validation and Initialization

### Field Validators
```python
@field_validator('region')
@classmethod
def _validate_custom_region(cls, v: str) -> str:
    """Validate region code."""
    valid_regions = ['NA', 'EU', 'FE']
    if v not in valid_regions:
        raise ValueError(f"Invalid custom region code: {v}. Must be one of {valid_regions}")
    return v

@field_validator('source_dir', check_fields=False)
@classmethod
def _validate_source_dir_exists(cls, v: Optional[str]) -> Optional[str]:
    """Validate that source_dir exists if it's a local path."""
    if v is not None and not v.startswith('s3://'):  # Only validate local paths
        if not Path(v).exists():
            logger.warning(f"Local source directory does not exist: {v}")
            raise ValueError(f"Local source directory does not exist: {v}")
        if not Path(v).is_dir():
            logger.warning(f"Local source_dir is not a directory: {v}")
            raise ValueError(f"Local source_dir is not a directory: {v}")
    return v
```

### Model Validator for One-Time Initialization
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

## Utility Methods

### Field Categorization
```python
def categorize_fields(self) -> Dict[str, List[str]]:
    """
    Categorize all fields into three tiers:
    1. Tier 1: Essential User Inputs - public fields with no defaults (required)
    2. Tier 2: System Inputs - public fields with defaults (optional)
    3. Tier 3: Derived Fields - properties that access private attributes
    
    Returns:
        Dict with keys 'essential', 'system', and 'derived' mapping to lists of field names
    """
    categories = {
        'essential': [],  # Tier 1: Required, public
        'system': [],     # Tier 2: Optional (has default), public
        'derived': []     # Tier 3: Public properties
    }
    
    # Get model fields from the class (not instance) to avoid deprecation warnings
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

### Display and Debugging
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
    
    # Print Tier 1 fields (essential user inputs)
    print("\n----- Essential User Inputs (Tier 1) -----")
    for field_name in sorted(categories['essential']):
        print(f"{field_name.title()}: {getattr(self, field_name)}")
    
    # Print Tier 2 fields (system inputs with defaults)
    print("\n----- System Inputs with Defaults (Tier 2) -----")
    for field_name in sorted(categories['system']):
        value = getattr(self, field_name)
        if value is not None:  # Skip None values for cleaner output
            print(f"{field_name.title()}: {value}")
    
    # Print Tier 3 fields (derived properties)
    print("\n----- Derived Fields (Tier 3) -----")
    for field_name in sorted(categories['derived']):
        try:
            value = getattr(self, field_name)
            if not callable(value):  # Skip methods
                print(f"{field_name.title()}: {value}")
        except Exception as e:
            print(f"{field_name.title()}: <Error: {e}>")
    
    print("\n===================================\n")
```

### Enhanced Serialization
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

### Custom String Representation
```python
def __str__(self) -> str:
    """
    Custom string representation that shows fields by category.
    This overrides the default __str__ method so that print(config) shows
    a nicely formatted representation with fields organized by tier.
    """
    from io import StringIO
    output = StringIO()
    
    # Get class name
    print(f"=== {self.__class__.__name__} ===", file=output)
    
    # Get fields categorized by tier
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
            if value is not None:  # Skip None values for cleaner output
                print(f"{field_name}: {value}", file=output)
    
    # Print Tier 3 fields (derived properties)
    if categories['derived']:
        print("\n- Derived Fields -", file=output)
        for field_name in sorted(categories['derived']):
            try:
                value = getattr(self, field_name)
                if not callable(value):  # Skip methods
                    print(f"{field_name}: {value}", file=output)
            except Exception:
                # Skip properties that cause errors
                pass
    
    return output.getvalue()
```

## Script Contract Integration

### Dynamic Contract Loading
```python
def get_script_contract(self) -> Optional['ScriptContract']:
    """
    Get script contract for this configuration.
    
    This base implementation returns None. Derived classes should override
    this method to return their specific script contract.
    """
    # Check for hardcoded script_contract first (for backward compatibility)
    if hasattr(self, '_script_contract'):
        return self._script_contract
        
    # Otherwise attempt to load based on class and job_type
    try:
        class_name = self.__class__.__name__.replace('Config', '')
        
        # Try with job_type if available
        if hasattr(self, 'job_type') and self.job_type:
            module_name = f"...steps.contracts.{class_name.lower()}_{self.job_type.lower()}_contract"
            contract_name = f"{class_name.upper()}_{self.job_type.upper()}_CONTRACT"
            
            try:
                contract_module = __import__(module_name, fromlist=[''])
                if hasattr(contract_module, contract_name):
                    return getattr(contract_module, contract_name)
            except (ImportError, AttributeError):
                pass
        
        # Try without job_type
        module_name = f"...steps.contracts.{class_name.lower()}_contract"
        contract_name = f"{class_name.upper()}_CONTRACT"
        
        try:
            contract_module = __import__(module_name, fromlist=[''])
            if hasattr(contract_module, contract_name):
                return getattr(contract_module, contract_name)
        except (ImportError, AttributeError):
            pass
            
    except Exception as e:
        logger.debug(f"Error loading script contract: {e}")
        
    return None

@property
def script_contract(self) -> Optional['ScriptContract']:
    """Property accessor for script contract."""
    return self.get_script_contract()

def get_script_path(self, default_path: str = None) -> str:
    """Get script path, preferring contract-defined path if available."""
    # Try to get from contract
    contract = self.get_script_contract()
    if contract and hasattr(contract, 'script_path'):
        return contract.script_path
        
    # Fall back to default or hardcoded path
    if hasattr(self, 'script_path'):
        return self.script_path
        
    return default_path
```

## Configuration Inheritance

### Base Configuration Creation
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

def get_public_init_fields(self) -> Dict[str, Any]:
    """
    Get a dictionary of public fields suitable for initializing a child config.
    Only includes fields that should be passed to child class constructors.
    Both essential user inputs and system inputs with defaults or user-overridden values
    are included to ensure all user customizations are properly propagated to derived classes.
    
    Returns:
        Dict[str, Any]: Dictionary of field names to values for child initialization
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

## Step Registry Integration

### Lazy Loading Pattern
```python
@classmethod
def get_step_name(cls, config_class_name: str) -> str:
    """Get the step name for a configuration class."""
    step_names = cls._get_step_registry()
    return step_names.get(config_class_name, config_class_name)

@classmethod
def get_config_class_name(cls, step_name: str) -> str:
    """Get the configuration class name from a step name."""
    step_names = cls._get_step_registry()
    reverse_mapping = {v: k for k, v in step_names.items()}
    return reverse_mapping.get(step_name, step_name)

@classmethod
def _get_step_registry(cls) -> Dict[str, str]:
    """Lazy load step registry to avoid circular imports."""
    if not cls._STEP_NAMES:
        try:
            from ...steps.registry.step_names import CONFIG_STEP_REGISTRY
            cls._STEP_NAMES = CONFIG_STEP_REGISTRY
        except ImportError:
            logger.warning("Could not import step registry, using empty registry")
            cls._STEP_NAMES = {}
    return cls._STEP_NAMES
```

## Usage Patterns

### Basic Configuration Creation
```python
# Create configuration with essential fields (Tier 1) and some system overrides (Tier 2)
config = BasePipelineConfig(
    # Tier 1: Essential fields (required)
    author="user",
    bucket="my-bucket", 
    role="arn:aws:iam::123456789012:role/MyRole",
    region="NA",
    service_name="my-service",
    pipeline_version="1.0.0",
    
    # Tier 2: Override some system defaults
    model_class="pytorch",
    framework_version="1.8.0"
)

# Derived fields (Tier 3) are automatically available
print(config.aws_region)        # "us-east-1"
print(config.pipeline_name)     # "user-my-service-pytorch-NA"
print(config.pipeline_s3_loc)   # "s3://my-bucket/MODS/user-my-service-pytorch-NA_1.0.0"
```

### Configuration Inheritance
```python
# Create derived configuration from base
child_config = ChildConfig.from_base_config(
    base_config,
    additional_field="value",
    override_field="new_value"
)

# Inspect field categorization
categories = config.categorize_fields()
print(f"Essential fields: {categories['essential']}")
print(f"System fields: {categories['system']}")
print(f"Derived fields: {categories['derived']}")
```

### Configuration Display and Debugging
```python
# Organized display by tiers
config.print_config()

# Custom string representation
print(config)  # Shows fields organized by tier

# Export with derived fields included
config_dict = config.model_dump()
```

## Design Benefits

### Encapsulation and Maintainability
- **Self-Contained Logic**: Each configuration class manages its own derivations
- **Clear Separation**: Three-tier system provides clear boundaries between field types
- **Type Safety**: Pydantic ensures comprehensive type validation
- **Consistent API**: Uniform interface across all configuration classes

### Performance and Efficiency
- **Lazy Evaluation**: Derived fields calculated only when accessed
- **Caching**: Results cached to prevent repeated calculations
- **Memory Efficiency**: Minimal overhead for unused derived fields
- **Fast Initialization**: One-time initialization with model validator

### Extensibility and Flexibility
- **Easy Extension**: Simple pattern for adding new derived fields
- **Inheritance Support**: Clean inheritance with field propagation
- **Contract Integration**: Dynamic script contract loading
- **Registry Integration**: Seamless step name resolution

## Dependencies and Requirements

### Core Dependencies
- **pydantic**: Model definition, validation, and serialization
- **pathlib**: Path handling for source directory validation
- **json**: Configuration serialization support
- **datetime**: Date field defaults and timestamps
- **logging**: Error reporting and debug information

### Optional Dependencies
- **Script Contract Classes**: TYPE_CHECKING imports for type hints
- **Step Registry Module**: Lazy-loaded for step name resolution
- **Child Configuration Classes**: Extended configurations inheriting from base

This base configuration class provides a robust, scalable foundation for all pipeline configurations, ensuring consistency, type safety, and maintainability across the entire system while following the proven three-tier architecture pattern established in the developer guide.
