---
tags:
  - design
  - configuration
  - tiered_system
  - field_classification
keywords:
  - three-tier design
  - configuration management
  - field classification
  - encapsulation
  - derived fields
topics:
  - configuration system
  - field categorization
  - self-contained design
language: python
date of note: 2025-07-31
---

# Three-Tier Field Classification Design for Configs and Hyperparameters

## Overview

This document describes a unified design for both configuration management and hyperparameters in the MODS pipeline, addressing scalability issues with previous approaches. The key insight is to move from centralized field derivation systems to a self-contained design where each class (configuration or hyperparameters) is responsible for its own field derivations.

The design principles apply uniformly to both:
1. **Configuration Classes** - Settings for pipeline components and steps
2. **Hyperparameter Classes** - Settings for model training and behavior

## Current Challenges

The current system has several limitations that affect both configurations and hyperparameters:

1. **Centralized Derivation Logic**: As the `FieldDerivationEngine` grows to accommodate more step types, it becomes unwieldy and difficult to maintain. Similarly, hyperparameter derivation logic is scattered.

2. **Separation of Concerns**: Logic for deriving values is separated from the classes that use them, creating maintenance challenges.

3. **Lack of Encapsulation**: Fields are exposed publicly, regardless of whether they should be part of the public API.

4. **Complex Factory Pattern**: Creation logic needs to understand internal details of each configuration type.

5. **Scalability Issues**: Adding new components requires updates to multiple parts of the codebase.

6. **Inconsistent Field Classification**: No clear distinction between essential inputs, system defaults, and derived fields across both configurations and hyperparameters.

## Proposed Solution: Self-Contained Configurations

Instead of a centralized derivation engine, each configuration class should encapsulate its own derivation logic using Pydantic's powerful features:

### Key Design Principles

1. **Self-Contained Derivation**: Each configuration class handles its own field derivations.
2. **Public vs. Private API**: Clear separation between essential user inputs and internal implementation details.
3. **Computed Properties**: Use Pydantic's computed fields for dynamic values.
4. **Composition over Inheritance**: Build configurations through composition rather than deep inheritance.
5. **Smart Validators**: Use model-level validators to derive fields during instantiation.

### Design Elements

#### 1. Essential vs. Derived Fields with Access Control

Fields in each configuration class will be categorized as:

- **Essential Fields (Tier 1)**: Explicitly required from the user, public access
- **System Fields (Tier 2)**: Default values that can be overridden, public access
- **Derived Fields (Tier 3)**: Computed from other fields, private with read-only property access

This approach enforces proper encapsulation, preventing users from accidentally overriding derived values.

#### 2. Computed Properties

For derived values that should be part of the public API:

```python
class TrainingConfig(BaseModel):
    # Essential user inputs (explicit)
    learning_rate: float
    num_boost_rounds: int
    
    # Derived fields via properties (computed)
    @property
    def framework_parameters(self) -> Dict[str, Any]:
        """Return parameters formatted for the training framework."""
        return {
            "eta": self.learning_rate,
            "num_round": self.num_boost_rounds,
            "objective": "binary:logistic" if self.is_binary else "multi:softmax"
        }
```

#### 3. Private Fields with Property Access

For implementation details and derived values that should be hidden with read-only access:

```python
class DataConfig(BaseModel):
    # Essential user inputs (Tier 1) - public fields
    region: str
    training_start_date: datetime
    training_end_date: datetime
    
    # Internal cache (completely private)
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    # Derived fields (Tier 3) - private with read-only property access
    # Two approaches for implementation:
    
    # Approach 1: Using PrivateAttr for values that shouldn't be serialized
    @property
    def date_range_string(self) -> str:
        """Format date range for display purposes."""
        if "date_range_string" not in self._cache:
            self._cache["date_range_string"] = f"{self.training_start_date.strftime('%Y%m%d')}-{self.training_end_date.strftime('%Y%m%d')}"
        return self._cache["date_range_string"]
    
    # Approach 2: Using private fields with properties for values that need to be serialized
    _etl_job_id: Optional[str] = Field(default=None, exclude=True)
    
    @property
    def etl_job_id(self) -> str:
        """Get ETL job ID for region."""
        if self._etl_job_id is None:
            region_mapping = {"NA": "24292902", "EU": "24292941", "FE": "25782074"}
            self._etl_job_id = region_mapping.get(self.region, "24292902")
        return self._etl_job_id
```

#### 4. Model Validators for Initializing Private Fields

Use model validators to initialize private fields during object creation:

```python
class BaseConfig(BaseModel):
    # Essential user inputs (Tier 1)
    region: str  # NA, EU, FE
    author: str
    service_name: str
    
    # Private derived fields (Tier 3)
    _aws_region: Optional[str] = Field(default=None, exclude=True)
    _pipeline_name: Optional[str] = Field(default=None, exclude=True)
    _pipeline_description: Optional[str] = Field(default=None, exclude=True)
    
    # Public read-only properties for derived fields
    @property
    def aws_region(self) -> str:
        """Get AWS region for the region code."""
        if self._aws_region is None:
            region_mapping = {"NA": "us-east-1", "EU": "eu-west-1", "FE": "us-west-2"}
            self._aws_region = region_mapping.get(self.region, "us-east-1")
        return self._aws_region
    
    @property
    def pipeline_name(self) -> str:
        """Get pipeline name derived from author, service and region."""
        if self._pipeline_name is None:
            self._pipeline_name = f"{self.author}-{self.service_name}-XGBoostModel-{self.region}"
        return self._pipeline_name
    
    @property
    def pipeline_description(self) -> str:
        """Get pipeline description derived from service and region."""
        if self._pipeline_description is None:
            self._pipeline_description = f"{self.service_name} XGBoost Model {self.region}"
        return self._pipeline_description
            
    # Optional: Model validator to initialize all derived fields at once
    @model_validator(mode='after')
    def initialize_derived_fields(self) -> 'BaseConfig':
        """Initialize all derived fields."""
        # Access properties to trigger initialization
        _ = self.aws_region
        _ = self.pipeline_name
        _ = self.pipeline_description
        return self
```

#### 5. Config Composition and Factory

Build more complex configurations through composition while avoiding validation loop issues:

```python
class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""
    
    # Configuration components
    base: BaseConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: Optional[EvaluationConfig] = None
    registration: Optional[RegistrationConfig] = None
    
    def create_config_list(self) -> List[Any]:
        """Create list of step configurations for pipeline assembly."""
        # Use ConfigFactory to safely create derived configurations
        factory = ConfigFactory()
        configs = [self.base]
        
        # Create data loading config using the factory
        data_load_config = factory.create_data_load_config(
            base_config=self.base,
            data_config=self.data
        )
        configs.append(data_load_config)
        
        # Create training config using the factory
        training_config = factory.create_training_config(
            base_config=self.base,
            training_config=self.training
        )
        configs.append(training_config)
        
        # And so on for other steps...
        
        return configs
```

The factory pattern helps prevent validation loops during composition by separating object creation from the model classes themselves:

```python
class ConfigFactory:
    """Factory for creating configuration objects safely."""
    
    def create_data_load_config(
        self, 
        base_config: BaseConfig, 
        data_config: DataConfig
    ) -> CradleDataLoadConfig:
        """
        Create a data load configuration safely without triggering validation loops.
        
        This approach prepares all values before creating the Pydantic model,
        which avoids validation loops during object creation.
        """
        # Extract values from base config without triggering validation
        base_values = base_config.model_dump()
        
        # Pre-compute derived values from data config
        data_sources_spec = data_config.create_data_sources_spec()
        
        # Create config with all values prepared in advance
        return CradleDataLoadConfig(
            **base_values,
            data_sources_spec=data_sources_spec,
            # Other data loading parameters derived from data_config
        )
```

This factory approach serves several purposes:

1. **Prevents Validation Loops**: By preparing all values before creating the model instance, it avoids the back-and-forth between validators and computed properties that can lead to infinite loops.

2. **Improves Separation of Concerns**: The factory handles the complexity of creating valid configurations, while each model class focuses on validation and behavior.

3. **Centralizes Creation Logic**: Complex creation rules are centralized in one place, making them easier to understand and modify.

4. **Reduces Duplication**: Common patterns for creating configurations can be extracted into factory methods.

## Implementation Plan

### 1. Base Configuration Classes with Private Derived Fields

Start by refactoring the base configuration classes to use the self-contained approach with proper encapsulation:

```python
class BasePipelineConfig(BaseModel):
    """Base configuration with self-contained derivation logic and encapsulated fields."""
    
    # Essential user inputs (Tier 1)
    region: str = Field(..., description="Region code (NA, EU, FE)")
    author: str = Field(..., description="Pipeline author/owner")
    service_name: str = Field(..., description="Service name for pipeline")
    pipeline_version: str = Field(..., description="Pipeline version")
    bucket: str = Field(..., description="S3 bucket for pipeline artifacts")
    
    # System inputs with defaults (Tier 2)
    py_version: str = Field(default="py3", description="Python version")
    current_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Current date"
    )
    
    # Internal state (completely private)
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    # Private derived fields (Tier 3) - hidden from constructor signature
    _aws_region: Optional[str] = Field(default=None, exclude=True)
    _pipeline_name: Optional[str] = Field(default=None, exclude=True)
    _pipeline_description: Optional[str] = Field(default=None, exclude=True)
    _pipeline_s3_loc: Optional[str] = Field(default=None, exclude=True)
    
    # Internal mapping (private class variable)
    _region_mapping: ClassVar[Dict[str, str]] = {
        "NA": "us-east-1", 
        "EU": "eu-west-1", 
        "FE": "us-west-2"
    }
    
    # Public read-only properties for derived fields
    @property
    def aws_region(self) -> str:
        """Get AWS region for the region code."""
        if self._aws_region is None:
            self._aws_region = self._region_mapping.get(self.region, "us-east-1")
        return self._aws_region
            
    @property
    def pipeline_name(self) -> str:
        """Get pipeline name derived from author, service and region."""
        if self._pipeline_name is None:
            self._pipeline_name = f"{self.author}-{self.service_name}-XGBoostModel-{self.region}"
        return self._pipeline_name
            
    @property
    def pipeline_description(self) -> str:
        """Get pipeline description derived from service and region."""
        if self._pipeline_description is None:
            self._pipeline_description = f"{self.service_name} XGBoost Model {self.region}"
        return self._pipeline_description
            
    @property
    def pipeline_s3_loc(self) -> str:
        """Get S3 location for pipeline artifacts."""
        if self._pipeline_s3_loc is None:
            pipeline_subdirectory = f"{self.pipeline_name}_{self.pipeline_version}"
            self._pipeline_s3_loc = f"s3://{self.bucket}/MODS/{pipeline_subdirectory}"
        return self._pipeline_s3_loc
    
    # Custom model_dump method to include derived properties in serialization
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

### 2. Step-Specific Configuration Classes

Each step would have its own configuration class with specific derivation logic and proper encapsulation:

```python
class XGBoostTrainingConfig(BasePipelineConfig):
    """Training configuration with self-contained derivation logic and encapsulated fields."""
    
    # Essential user inputs specific to training (Tier 1)
    num_round: int = Field(..., description="Number of boosting rounds")
    max_depth: int = Field(..., description="Maximum tree depth")
    min_child_weight: int = Field(..., description="Minimum child weight")
    is_binary: bool = Field(..., description="Binary classification flag")
    
    # System inputs with defaults (Tier 2)
    training_instance_type: str = Field(default="ml.m5.4xlarge", description="Training instance type")
    training_instance_count: int = Field(default=1, description="Number of training instances")
    training_volume_size: int = Field(default=800, description="Training volume size in GB")
    training_entry_point: str = Field(default="train_xgb.py", description="Training script entry point")
    
    # Private derived fields (Tier 3)
    _hyperparameter_file: Optional[str] = Field(default=None, exclude=True)
    _objective: Optional[str] = Field(default=None, exclude=True)
    _eval_metric: Optional[List[str]] = Field(default=None, exclude=True)
    
    # Public read-only properties for derived fields
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

### 3. Step Integration API

Provide an interface method to generate step configuration for pipeline assembly:

```python
class XGBoostTrainingConfig(BasePipelineConfig):
    # ... fields and methods as above ...
    
    def create_training_step(self, dependencies: List[Any] = None) -> Dict[str, Any]:
        """
        Create training step configuration for pipeline assembly.
        
        This method generates a complete step configuration with all
        necessary fields derived and ready for the pipeline to use.
        
        Args:
            dependencies: List of step dependencies
            
        Returns:
            Complete step configuration dictionary
        """
        return {
            "name": f"XGBoostTraining-{self.region}",
            "estimator_config": {
                "instance_type": self.training_instance_type,
                "instance_count": self.training_instance_count,
                "volume_size": self.training_volume_size,
                "framework_version": self.framework_version,
                "py_version": self.py_version,
                "entry_point": self.training_entry_point,
                "source_dir": self.source_dir,
                "role": self.role,
                "hyperparameters": self.to_hyperparameter_dict()
            },
            "dependencies": dependencies or []
        }
```

### 4. Data Config Example

The data loading configuration would follow a similar pattern:

```python
class CradleDataLoadConfig(BasePipelineConfig):
    """Data loading configuration with self-contained derivation logic."""
    
    # Essential user inputs
    training_start_datetime: str = Field(..., description="Training data start date")
    training_end_datetime: str = Field(..., description="Training data end date")
    tag_edx_provider: str = Field(..., description="EDX provider name")
    tag_edx_subject: str = Field(..., description="EDX subject name") 
    tag_edx_dataset: str = Field(..., description="EDX dataset name")
    
    # System inputs with defaults
    job_type: str = Field(default="training", description="Job type (training/calibration)")
    cluster_type: str = Field(default="STANDARD", description="Cradle cluster type")
    cradle_account: str = Field(default="Buyer-Abuse-RnD-Dev", description="Cradle account name")
    
    # Derived fields
    etl_job_id: Optional[str] = Field(default=None, description="ETL job ID")
    data_sources_spec: Optional[Dict[str, Any]] = Field(default=None, description="Data sources spec")
    transform_spec: Optional[Dict[str, Any]] = Field(default=None, description="Transform spec")
    output_spec: Optional[Dict[str, Any]] = Field(default=None, description="Output spec")
    
    # Mapping for ETL job IDs by region
    _etl_job_mapping: ClassVar[Dict[str, str]] = {
        'NA': '24292902',
        'EU': '24292941',
        'FE': '25782074',
    }
    
    @model_validator(mode='after')
    def derive_data_fields(self) -> 'CradleDataLoadConfig':
        """Derive data loading specific fields."""
        # Call parent derivation
        super().derive_fields()
        
        # Derive ETL job ID
        if self.etl_job_id is None:
            self.etl_job_id = self._etl_job_mapping.get(self.region, '24292902')
            
        # Derive data sources spec
        if self.data_sources_spec is None:
            self.data_sources_spec = self._create_data_sources_spec()
            
        # Derive transform spec
        if self.transform_spec is None:
            self.transform_spec = self._create_transform_spec()
            
        # Derive output spec
        if self.output_spec is None:
            self.output_spec = self._create_output_spec()
            
        return self
        
    def _create_data_sources_spec(self) -> Dict[str, Any]:
        """Create data sources specification."""
        # Implementation details...
        pass
        
    def _create_transform_spec(self) -> Dict[str, Any]:
        """Create transform specification."""
        # Implementation details...
        pass
        
    def _create_output_spec(self) -> Dict[str, Any]:
        """Create output specification."""
        # Implementation details...
        pass
```

## User Interface

The user interface is significantly simplified:

```python
from datetime import datetime
from my_pipeline.configs import BasePipelineConfig, XGBoostTrainingConfig, CradleDataLoadConfig, PipelineConfig

# Create base configuration
base_config = BasePipelineConfig(
    region="NA",
    author="data-scientist",
    service_name="AtoZ",
    pipeline_version="0.1.0",
    bucket="my-bucket"
)

# Create training configuration
training_config = XGBoostTrainingConfig(
    # Include base fields
    **base_config.model_dump(),
    
    # Training-specific fields
    num_round=300,
    max_depth=10,
    min_child_weight=1,
    is_binary=True
)

# All derived fields are automatically populated
print(f"Pipeline name: {training_config.pipeline_name}")
print(f"AWS region: {training_config.aws_region}")
print(f"Objective: {training_config.objective}")
print(f"Eval metrics: {training_config.eval_metric}")

# Or use the composition approach
pipeline_config = PipelineConfig(
    base=BasePipelineConfig(
        region="NA",
        author="data-scientist",
        service_name="AtoZ",
        pipeline_version="0.1.0",
        bucket="my-bucket"
    ),
    data=DataConfig(
        training_start_date=datetime(2025, 1, 1),
        training_end_date=datetime(2025, 4, 17)
    ),
    training=TrainingConfig(
        num_round=300,
        max_depth=10,
        min_child_weight=1,
        is_binary=True
    )
)

# Generate configuration list for merge_and_save_configs
config_list = pipeline_config.create_config_list()
```

## Field Classification and Policy

The configuration system follows a strict policy regarding fields and their characteristics:

### Field Classification Policy

1. **Essential User Inputs (Tier 1)**:
   - Must be explicitly provided by users
   - No default values allowed
   - Subject to field validation
   - Public access

2. **System Inputs (Tier 2)**:
   - Have reasonable default values
   - Can be overridden by users
   - Subject to field validation
   - Public access
   - User-overridden values must be propagated to derived classes

3. **Derived Fields (Tier 3)**:
   - Private fields with leading underscores
   - Values assigned within the config class
   - Implemented using PrivateAttr() in Pydantic v2
   - No field validation applies (as they're private)
   - Accessed through read-only properties
   - Calculated based on Tier 1 and Tier 2 fields

### Field Implementation Guidelines

- **Essential User Inputs**: Defined as regular Pydantic fields with no default value
  ```python
  author: str = Field(description="Author or owner of the pipeline.")
  ```

- **System Inputs**: Defined as regular Pydantic fields with default values
  ```python
  model_class: str = Field(
      default='xgboost', 
      description="Model class (e.g., XGBoost, PyTorch).")
  ```

- **Derived Fields**: Defined as private attributes with PrivateAttr()
  ```python
  _pipeline_name: Optional[str] = PrivateAttr(default=None)
  
  @property
  def pipeline_name(self) -> str:
      """Get pipeline name derived from author, service_name, model_class, and region."""
      if self._pipeline_name is None:
          self._pipeline_name = f"{self.author}-{self.service_name}-{self.model_class}-{self.region}"
      return self._pipeline_name
  ```

## Field Classification and Inheritance

### Automatic Field Classification

A key feature of our design is automatic field classification. Rather than manually categorizing fields, we've implemented a system that automatically detects the tier of each field based on its characteristics:

```python
def categorize_fields(self) -> Dict[str, List[str]]:
    """
    Categorize all fields into three tiers:
    1. Tier 1: Essential User Inputs - public fields with no defaults
    2. Tier 2: System Inputs - public fields with defaults
    3. Tier 3: Derived Fields - properties that access private attributes
    """
    categories = {
        'essential': [],  # Tier 1: Essential user inputs (no defaults)
        'system': [],     # Tier 2: System inputs (with defaults)
        'derived': []     # Tier 3: Derived fields (properties)
    }
    
    # Get model fields
    model_fields = self.model_fields
    
    # Categorize public fields based on whether they have defaults
    for field_name, field_info in model_fields.items():
        if field_name.startswith('_'):
            continue  # Skip private fields
            
        has_default = (field_info.default is not None or 
                      field_info.default_factory is not None)
        
        if has_default:
            categories['system'].append(field_name)
        else:
            categories['essential'].append(field_name)
    
    # Find derived properties (public properties not in model_fields)
    for attr_name in dir(self):
        if (not attr_name.startswith('_') and 
            attr_name not in model_fields and
            isinstance(getattr(type(self), attr_name, None), property)):
            categories['derived'].append(attr_name)
    
    return categories
```

This approach:
- Eliminates the need for manually maintaining lists of fields
- Correctly handles fields from all classes in the inheritance hierarchy
- Automatically adapts when fields are added or changed
- Works properly with polymorphism and inheritance

### Configuration Display

To make it easy to view configuration details, we've implemented a custom string representation:

```python
def __str__(self) -> str:
    """Custom string representation with fields organized by tier."""
    from io import StringIO
    output = StringIO()
    
    print(f"=== {self.__class__.__name__} ===", file=output)
    
    # Get fields categorized by tier
    categories = self.categorize_fields()
    
    # Print each tier of fields
    if categories['essential']:
        print("\n- Essential User Inputs -", file=output)
        for field_name in sorted(categories['essential']):
            print(f"{field_name}: {getattr(self, field_name)}", file=output)
    
    # ... similar blocks for system and derived fields ...
    
    return output.getvalue()
```

This means that simply using `print(config)` will display a nicely formatted representation of the configuration object with all fields organized by tier.

### Parent-to-Child Configuration

To efficiently create child configurations from parent configurations without duplicating fields, we implement a pattern that allows passing parent fields to children:

1. **Dynamic Field Extraction**: The base class provides a method that uses field categorization to extract fields for initialization:
   ```python
   def get_public_init_fields(self) -> Dict[str, Any]:
       """Get fields suitable for initializing a child config."""
       # Use categorize_fields to get essential and system fields
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

2. **Virtual Factory Method**: BasePipelineConfig provides a virtual class method that all derived classes inherit:
   ```python
   @classmethod
   def from_base_config(cls, base_config: BasePipelineConfig, **kwargs) -> 'BasePipelineConfig':
       """Create a new configuration instance from a base configuration."""
       parent_fields = base_config.get_public_init_fields()
       config_dict = {**parent_fields, **kwargs}
       return cls(**config_dict)
   ```

   This means any class that inherits from BasePipelineConfig automatically has this factory method
   available without having to implement it. The `cls` refers to the actual derived class that is
   being instantiated, allowing proper polymorphic behavior.

3. **Method Overriding in Derived Classes**: When a derived class adds its own fields, it should override the get_public_init_fields method to include those fields:
   ```python
   def get_public_init_fields(self) -> Dict[str, Any]:
       """Override to include derived class fields."""
       # Get base class fields first
       base_fields = super().get_public_init_fields()
       
       # Add derived class specific fields using categorize_fields
       categories = self.categorize_fields()
       derived_fields = {}
       
       # Add fields that aren't already in base_fields
       for field_type in ['essential', 'system']:
           for field_name in categories[field_type]:
               if field_name not in base_fields:
                   value = getattr(self, field_name)
                   if value is not None:
                       derived_fields[field_name] = value
       
       # Combine (derived fields take precedence if overlap)
       return {**base_fields, **derived_fields}
   ```

This approach ensures:
- No duplication of configuration values
- Both essential inputs and system inputs (default or user-overridden) are properly propagated
- Child configs can override parent values when needed
- Derived fields are properly initialized in both parent and child classes
- Each level in the inheritance hierarchy adds its own fields to the initialization chain

## Benefits

1. **Maintainability**: Each configuration class is self-contained and responsible for its own field derivations.

2. **Enhanced Encapsulation**: Clear separation between essential user inputs and derived fields, with private fields and read-only access for derived values.

3. **Scalability**: Adding new step types doesn't require updates to a central derivation engine.

4. **Discoverability**: It's clear which fields are required vs. derived from the class definition and property methods.

5. **Flexibility**: Easy to extend and customize without breaking other parts of the system.

6. **Type Safety**: Full type checking
