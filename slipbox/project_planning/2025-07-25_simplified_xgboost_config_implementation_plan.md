# Enhanced Self-Contained XGBoost Configuration Implementation Plan

## Overview

This document outlines a revised plan for implementing a minimal viable product (MVP) that streamlines the configuration process for XGBoost pipeline creation using a self-contained configuration design with enhanced encapsulation. The key insight is to move from a centralized field derivation system to a self-contained approach where each configuration class is responsible for its own field derivations, with derived fields being private and accessible only through read-only properties.

## Background

The current configuration process (as seen in `template_config_xgb_eval_v2.ipynb`) is verbose and requires users to manually specify many configuration values. The original simplified plan used a centralized derivation engine, but as pipeline steps grow, this approach becomes unwieldy. The self-contained design with private derived fields addresses scalability and encapsulation issues while maintaining the three-tier categorization of inputs:

1. Essential User Inputs (Tier 1) - explicitly provided by users, public access
2. System Inputs (Tier 2) - default values that can be overridden, public access
3. Derived Inputs (Tier 3) - calculated from other fields, private with read-only property access

## Goals

1. Create a simplified user input interface that captures only essential information
2. Implement self-contained configuration classes that handle their own defaults and derivations
3. Enforce proper encapsulation by making derived fields private with read-only property access
4. Eliminate the need for a centralized field derivation engine
5. Maintain compatibility with the existing `merge_and_save_configs` utility
6. Generate equivalent output configuration structure as the current system
7. Prevent accidental overriding of derived values by users

## Technical Components

### 1. Self-Contained Configuration Classes with Private Derived Fields

Refactor configuration classes to include their own derivation logic with proper encapsulation:

```python
class BasePipelineConfig(BaseModel):
    """Base configuration with self-contained derivation logic and encapsulated fields."""
    
    # Essential user inputs (Tier 1) - public fields
    region: str = Field(..., description="Region code (NA, EU, FE)")
    author: str = Field(..., description="Pipeline author/owner")
    service_name: str = Field(..., description="Service name for pipeline")
    pipeline_version: str = Field(..., description="Pipeline version")
    bucket: str = Field(..., description="S3 bucket for pipeline artifacts")
    
    # System inputs with defaults (Tier 2) - public fields
    py_version: str = Field(default="py3", description="Python version")
    current_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Current date"
    )
    
    # Internal state (completely private)
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    # Derived fields (Tier 3) - private fields with read-only properties
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

### 2. Step-Specific Configuration Classes with Private Derived Fields

Create specialized configuration classes for each step type with their own derivation logic and proper encapsulation:

```python
class XGBoostTrainingConfig(BasePipelineConfig):
    """Training configuration with self-contained derivation logic and encapsulated fields."""
    
    # Essential user inputs specific to training (Tier 1) - public fields
    num_round: int = Field(..., description="Number of boosting rounds")
    max_depth: int = Field(..., description="Maximum tree depth")
    min_child_weight: int = Field(..., description="Minimum child weight")
    is_binary: bool = Field(..., description="Binary classification flag")
    
    # System inputs with defaults (Tier 2) - public fields
    training_instance_type: str = Field(default="ml.m5.4xlarge", description="Training instance type")
    training_instance_count: int = Field(default=1, description="Number of training instances")
    training_volume_size: int = Field(default=800, description="Training volume size in GB")
    training_entry_point: str = Field(default="train_xgb.py", description="Training script entry point")
    
    # Derived fields (Tier 3) - private fields
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
    
    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        data["objective"] = self.objective
        data["eval_metric"] = self.eval_metric
        return data
```

### 3. Top-Level Pipeline Configuration

Create a composition-based top-level configuration:

```python
class XGBoostPipelineConfig(BaseModel):
    """Top-level pipeline configuration using composition."""
    
    # Component configurations
    base: BasePipelineConfig
    data: DataConfig
    training: TrainingParamsConfig
    evaluation: Optional[EvaluationParamsConfig] = None
    registration: Optional[RegistrationParamsConfig] = None
    
    def create_config_list(self) -> List[BasePipelineConfig]:
        """Create list of configuration objects for the pipeline."""
        # Start with base config
        configs = []
        
        # Create data loading configurations
        training_data_config = CradleDataLoadConfig(
            **self.base.model_dump(),  # Include base fields
            job_type="training",
            training_start_datetime=self.data.training_start_date.isoformat(),
            training_end_datetime=self.data.training_end_date.isoformat(),
            # ... other data-specific fields
        )
        configs.append(training_data_config)
        
        # Similarly for other steps...
        # Each config handles its own derivations via model_validator
        
        return configs
```

### 4. Essential User Input Interface

Create a simplified interface for collecting essential inputs:

```python
class DateRangePeriod(BaseModel):
    start_date: datetime
    end_date: datetime

class DataConfig(BaseModel):
    region: str
    training_period: DateRangePeriod
    calibration_period: DateRangePeriod
    full_field_list: List[str]
    cat_field_list: List[str]
    tab_field_list: List[str]
    
class TrainingParamsConfig(BaseModel):
    num_round: int = 300
    max_depth: int = 10
    min_child_weight: int = 1
    is_binary: bool = True
    label_name: str = "is_abuse"
    id_name: str = "order_id"
    marketplace_id_col: str = "marketplace_id"
    
class RegistrationParamsConfig(BaseModel):
    model_owner: str
    model_registration_domain: str = "AtoZ"
    expected_tps: int = 2
    max_latency_ms: int = 800
```

### 5. Interactive Notebook

Create a simplified notebook that guides users through the essential inputs and generates the full configuration:

```python
import json
from datetime import datetime
from src.config_models import (
    BasePipelineConfig, DataConfig, TrainingParamsConfig, 
    RegistrationParamsConfig, XGBoostPipelineConfig, DateRangePeriod
)
from src.pipeline_steps.utils import merge_and_save_configs

# Step 1: Define base configuration
base_config = BasePipelineConfig(
    region="NA",
    author="data-scientist",
    service_name="AtoZ",
    pipeline_version="0.1.0",
    bucket="my-ml-bucket"
)

# Step 2: Define data configuration
data_config = DataConfig(
    region="NA",
    training_period=DateRangePeriod(
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 4, 17)
    ),
    calibration_period=DateRangePeriod(
        start_date=datetime(2025, 4, 17),
        end_date=datetime(2025, 4, 28)
    ),
    full_field_list=[...],  # User provides this
    cat_field_list=[...],   # User provides this
    tab_field_list=[...]    # User provides this
)

# Step 3: Define training parameters
training_config = TrainingParamsConfig(
    num_round=300,
    max_depth=10,
    min_child_weight=1,
    is_binary=True
)

# Step 4: Define registration parameters
registration_config = RegistrationParamsConfig(
    model_owner="amzn1.abacus.team.djmdvixm5abr3p75c5ca"
)

# Step 5: Create pipeline configuration
pipeline_config = XGBoostPipelineConfig(
    base=base_config,
    data=data_config,
    training=training_config,
    registration=registration_config
)

# Step 6: Generate configuration list for pipeline assembly
config_list = pipeline_config.create_config_list()

# Step 7: Save configuration
config_dir = f'pipeline_config/config_{base_config.region}_xgboost_v2'
config_file_name = f'config_{base_config.region}_xgboost.json'
merged_config = merge_and_save_configs(config_list, f"{config_dir}/{config_file_name}")
```

## Implementation Plan

### Phase 1: Self-Contained Base Classes (Week 1)

1. Implement `BasePipelineConfig` with self-contained derivation logic
2. Create essential parameter models (DataConfig, TrainingParamsConfig, etc.)
3. Implement the first step-specific configuration class (e.g., XGBoostTrainingConfig)
4. Add unit tests for each class

### Phase 2: Complete Configuration Classes (Week 2)

1. Implement remaining step configuration classes with self-contained derivation
2. Create the top-level `XGBoostPipelineConfig` class for composition
3. Implement the `create_config_list()` method to generate pipeline configurations
4. Add integration tests to verify configuration behavior

### Phase 3: User Interface and Documentation (Week 3)

1. Create simplified notebook for user input
2. Implement validation and error handling
3. Add comprehensive documentation
4. Verify compatibility with `merge_and_save_configs` utility

## Implementation Notes

### Key Benefits of Enhanced Self-Contained Design

1. **Strict Encapsulation**: Each configuration class handles its own defaults and derivations, with derived fields being private
2. **Read-Only Access**: Derived fields can only be accessed through read-only properties, preventing accidental overrides
3. **Scalability**: Adding new step types doesn't require modifying a central component
4. **Maintainability**: Changes to derivation logic are localized to relevant classes
5. **Type Safety**: Full type checking with Pydantic's validation system
6. **Self-Documentation**: Property methods document how derived values are calculated
7. **Clear Contract**: Explicit distinction between fields users can set vs. derived values they can only read

### Integration with Existing System

The enhanced self-contained design changes the roles of existing components:

1. **DefaultValuesProvider**: No longer needed - defaults are specified in field definitions
2. **FieldDerivationEngine**: No longer needed - derivations are handled by property methods
3. **ConfigFactory**: Transformed into a simpler factory that helps prevent validation loops when composing configurations

The new design provides:
1. **Proper Encapsulation**: Clear separation between user inputs and derived values
2. **Value Protection**: Prevents accidental overriding of derived fields
3. **Serialization Control**: Custom `model_dump` methods ensure proper serialization of private fields
4. **Validation Loop Prevention**: Factory pattern helps avoid infinite validation loops during config creation

The output remains compatible with the existing `merge_and_save_configs` utility, ensuring seamless integration with the current workflow.

### Refined ConfigFactory Role

Rather than being eliminated, the ConfigFactory plays a critical role in preventing validation loops:

```python
class ConfigFactory:
    """
    Factory to create configuration objects safely without triggering validation loops.
    
    This factory prepares all derived values before creating model instances,
    which prevents the back-and-forth between validators and properties that
    can lead to infinite loops during composition.
    """
    
    def create_data_load_config(
        self, 
        base_config: BasePipelineConfig, 
        data_config: DataConfig
    ) -> CradleDataLoadConfig:
        """Create a data load configuration safely."""
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
    
    def create_training_config(
        self,
        base_config: BasePipelineConfig,
        training_params: TrainingParamsConfig
    ) -> XGBoostTrainingConfig:
        """Create a training configuration safely."""
        # Extract values from base config
        base_values = base_config.model_dump()
        
        # Pre-compute derived values
        hyperparameters = {
            "num_round": training_params.num_round,
            "max_depth": training_params.max_depth,
            "min_child_weight": training_params.min_child_weight,
            "objective": "binary:logistic" if training_params.is_binary else "multi:softmax",
            "eval_metric": ['logloss', 'auc'] if training_params.is_binary else ['mlogloss', 'merror']
        }
        
        # Create config with prepared values
        return XGBoostTrainingConfig(
            **base_values,
            num_round=training_params.num_round,
            max_depth=training_params.max_depth,
            min_child_weight=training_params.min_child_weight,
            is_binary=training_params.is_binary,
            # Setting pre-computed values directly to private fields would require
            # custom initialization logic in the model class
        )
```

This approach provides several benefits:

1. **Prevents Validation Loops**: By preparing all values before creating the model instance
2. **Improves Separation of Concerns**: Models focus on validation and behavior, factory handles creation
3. **Centralizes Creation Logic**: Complex creation rules are in one place
4. **Reduces Duplication**: Common patterns for creating configurations are extracted into factory methods

## Testing Strategy

1. **Unit Tests**: Test each configuration class in isolation, verifying that private fields are properly initialized
2. **Property Tests**: Ensure properties correctly calculate and cache derived values
3. **Integration Tests**: Verify that composed configurations work together correctly
4. **Migration Tests**: Ensure new configurations produce equivalent results to the old approach
5. **Serialization Tests**: Verify compatibility with merge_and_save_configs, ensuring private fields are properly handled
6. **Access Control Tests**: Verify that derived values can't be directly modified
7. **Validation Loop Tests**: Verify that the factory pattern successfully prevents validation loops
8. **Composition Tests**: Test the creation of complex configurations through the factory

## Future Enhancements

1. **Field Documentation**: Add detailed descriptions to all fields and property methods for self-documenting code
2. **Schema Generation**: Generate JSON schema documentation from Pydantic models, clearly marking read-only properties
3. **Configuration Templates**: Provide pre-configured templates for common scenarios
4. **Visualization Tools**: Create visual representation of configuration relationships, distinguishing between user inputs and derived values
5. **Web Interface**: Develop a web form for generating configurations, with clear visual cues for derived read-only fields
6. **IDE Integration**: Create editor tooling that shows which fields are derived vs. settable
7. **Configuration Validators**: Add cross-field validation to ensure consistency between related fields
8. **Enhanced Factory Patterns**: Develop more sophisticated factory patterns to handle complex configuration composition
9. **Validation Loop Detection**: Add tools to detect and prevent potential validation loops during development

## Conclusion

The enhanced self-contained configuration design offers significant advantages over both the centralized derivation approach and the basic self-contained approach. By encapsulating defaults and derivation logic within each configuration class, making derived fields private with read-only property access, and utilizing a factory pattern to prevent validation loops, we create a more maintainable, scalable, and robust system that enforces proper design principles.

This approach provides a simplified user experience while preventing common errors like accidentally overriding derived values or triggering infinite validation loops during configuration composition. The clear separation between essential user inputs (Tier 1), system defaults (Tier 2), and derived values (Tier 3) creates a clean API that guides users toward correct usage.

The refined factory pattern plays an important role in this design by safely handling the creation and composition of configuration objects, ensuring that validation loops are prevented while maintaining the benefits of self-contained configuration classes.

The implementation plan provides a clear path to migrate from the current system to this new design while maintaining compatibility with existing tools, enhancing the overall robustness of the configuration system.
