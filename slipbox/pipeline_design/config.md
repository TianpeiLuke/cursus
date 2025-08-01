# Config

## What is the Purpose of Config?

Configs serve as the **centralized configuration management layer** that provides hierarchical, validated, and environment-specific configuration for pipeline components. After the spec-driven design refactoring, configs are now integrated with script contracts and step specifications, representing the evolution from scattered configuration parameters to a unified, intelligent configuration system with contract enforcement.

## Core Purpose

Configs provide **centralized configuration management** that:

1. **Hierarchical Configuration** - Support inheritance and composition patterns
2. **Environment-Specific Overrides** - Enable dev/staging/prod configurations
3. **Validation and Type Safety** - Ensure configuration correctness at creation time with Pydantic
4. **Template and Defaults** - Provide sensible defaults with customization points
5. **Integration Bridge** - Connect high-level settings to low-level implementation details
6. **Contract Enforcement** - Ensure alignment between configurations and script contracts

## Key Features

### 1. Hierarchical Configuration with Pydantic

Configs support inheritance and composition for maximum reusability, now using Pydantic models:

```python
from pydantic import BaseModel, Field, model_validator, field_validator

# Base configuration with common settings
class BasePipelineConfig(BaseModel):
    """Base configuration for all pipeline steps"""
    
    # Shared basic info
    bucket: str = Field(description="S3 bucket name for pipeline artifacts and data.")
    current_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Current date, typically used for versioning or pathing."
    )
    region: str = Field(default='NA', description="Custom region code (NA, EU, FE) for internal logic.")
    aws_region: Optional[str] = Field(default=None, description="Derived AWS region (e.g., us-east-1).")
    author: str = Field(description="Author or owner of the pipeline.")

    # Overall pipeline identification
    pipeline_name: str = Field(description="Name of the SageMaker Pipeline.")
    pipeline_description: str = Field(description="Description for the SageMaker Pipeline.")
    pipeline_version: str = Field(description="Version string for the SageMaker Pipeline.")
    pipeline_s3_loc: str = Field(
        description="Root S3 location for storing pipeline definition and step artifacts.",
        pattern=r'^s3://[a-zA-Z0-9.-][a-zA-Z0-9.-]*(?:/[a-zA-Z0-9.-][a-zA-Z0-9._-]*)*$'
    )

    # Common framework/scripting info
    framework_version: str = Field(default='2.1.0', description="Default framework version.")
    py_version: str = Field(default='py310', description="Default Python version.")
    source_dir: Optional[str] = Field(default=None, description="Common source directory for scripts.")
    
    class Config:
        validate_assignment = True

# Specialized configuration inheriting from base
class TrainingStepConfig(BasePipelineConfig):
    """Configuration for training steps"""
    
    # Training-specific parameters
    training_instance_type: str = Field(description="EC2 instance type for training")
    training_instance_count: int = Field(default=1, description="Number of instances for training")
    training_volume_size: int = Field(default=30, description="Size of the EBS volume in GB")
    enable_caching: bool = Field(default=True, description="Enable step caching")
    
    @field_validator('training_instance_type')
    @classmethod
    def validate_instance_type(cls, v: str) -> str:
        valid_prefixes = ['ml.m5', 'ml.c5', 'ml.p3', 'ml.g4dn']
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(f"Instance type must start with one of {valid_prefixes}")
        return v

# Specific implementation configuration
class XGBoostTrainingConfig(TrainingStepConfig):
    """Configuration for XGBoost training steps"""
    
    # XGBoost-specific parameters
    training_entry_point: str = Field(default="train_xgb.py", description="Training script path")
    hyperparameters: XGBoostModelHyperparameters = Field(
        default_factory=XGBoostModelHyperparameters,
        description="XGBoost hyperparameters"
    )
    hyperparameters_s3_uri: Optional[str] = Field(default=None, description="S3 path for hyperparameters")
```

### 2. Environment-Specific Overrides

Support different configurations for different environments:

```python
def get_config_for_environment(base_config: BasePipelineConfig, environment: str) -> BasePipelineConfig:
    """Apply environment-specific overrides using Pydantic model copying"""
    
    # Create a copy to modify
    config_dict = base_config.model_dump()
    
    if environment == "dev":
        # Development environment overrides
        config_dict.update({
            "training_instance_type": "ml.t3.medium",
            "training_volume_size": 20,
            "use_spot_instances": True,
            "max_runtime_in_seconds": 1800
        })
    elif environment == "prod":
        # Production environment overrides
        config_dict.update({
            "training_instance_type": "ml.m5.2xlarge",
            "training_volume_size": 100,
            "use_spot_instances": False,
            "max_runtime_in_seconds": 7200
        })
    
    # Create a new config instance with updated values
    return type(base_config)(**config_dict)
```

### 3. Validation and Type Safety with Pydantic

Configs provide validation at creation time to catch errors early:

```python
class XGBoostTrainingConfig(TrainingStepConfig):
    """Configuration with comprehensive Pydantic validation"""
    
    # Validate instance type
    @field_validator('training_instance_type')
    @classmethod
    def validate_training_instance(cls, v: str) -> str:
        valid_instance_types = ["ml.t3.medium", "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge"]
        if v not in valid_instance_types:
            raise ValueError(f"Invalid instance_type: {v}. Must be one of: {valid_instance_types}")
        return v
    
    # Validate source_dir exists
    @field_validator('source_dir', check_fields=False)
    @classmethod
    def validate_source_dir_exists(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith('s3://'):
            if not Path(v).exists():
                raise ValueError(f"Local source directory does not exist: {v}")
        return v
    
    # Model validator for interdependent fields
    @model_validator(mode='after')
    def validate_dependencies(self) -> 'XGBoostTrainingConfig':
        """Validate interdependent fields after all values are set"""
        # Ensure hyperparameters are appropriate for objective
        if self.hyperparameters.objective == "binary:logistic" and self.hyperparameters.num_class > 1:
            raise ValueError("num_class should be 1 for binary classification")
        return self
```

### 4. Template and Defaults

Provide sensible defaults with clear customization points:

```python
# Factory functions for common configuration patterns
def create_quick_prototype_config(region: str = "NA", author: str = "default-author") -> XGBoostTrainingConfig:
    """Fast, cheap configuration for prototyping"""
    return XGBoostTrainingConfig(
        region=region,
        author=author,
        training_instance_type="ml.t3.medium",
        training_instance_count=1,
        training_volume_size=30,
        hyperparameters=XGBoostModelHyperparameters(
            max_depth=3,
            eta=0.3,
            num_round=100
        )
    )

def create_production_config(region: str, author: str) -> XGBoostTrainingConfig:
    """Robust configuration for production workloads"""
    return XGBoostTrainingConfig(
        region=region,
        author=author,
        training_instance_type="ml.m5.2xlarge",
        training_instance_count=1,
        training_volume_size=100,
        hyperparameters=XGBoostModelHyperparameters(
            max_depth=6,
            eta=0.1,
            num_round=1000,
            early_stopping_rounds=50
        )
    )
```

### 5. Integration Bridge

Connect high-level settings to low-level implementation details:

```python
class XGBoostTrainingConfig(TrainingStepConfig):
    """Configuration that bridges high-level intent to implementation"""
    
    # Performance tier options
    performance_tier: str = Field(default="balanced", description="Performance tier setting")
    
    @model_validator(mode='after')
    def apply_performance_tier(self) -> 'XGBoostTrainingConfig':
        """Apply settings based on performance tier if not explicitly overridden"""
        if not hasattr(self, '_performance_applied') or not self._performance_applied:
            self._performance_applied = True
            
            if self.performance_tier == "cost_optimized":
                if not hasattr(self, '_explicit_instance_type'):
                    self.training_instance_type = "ml.t3.medium"
                if not hasattr(self, '_explicit_hyperparams'):
                    self.hyperparameters.max_depth = 3
                    self.hyperparameters.eta = 0.3
                    self.hyperparameters.num_round = 100
                
            elif self.performance_tier == "balanced":
                if not hasattr(self, '_explicit_instance_type'):
                    self.training_instance_type = "ml.m5.large"
                if not hasattr(self, '_explicit_hyperparams'):
                    self.hyperparameters.max_depth = 6
                    self.hyperparameters.eta = 0.2
                    self.hyperparameters.num_round = 500
                
            elif self.performance_tier == "performance_optimized":
                if not hasattr(self, '_explicit_instance_type'):
                    self.training_instance_type = "ml.m5.2xlarge" 
                if not hasattr(self, '_explicit_hyperparams'):
                    self.hyperparameters.max_depth = 8
                    self.hyperparameters.eta = 0.1
                    self.hyperparameters.num_round = 1000
        
        return self
```

### 6. Script Contract Integration

Configs now integrate with script contracts to ensure alignment:

```python
class BasePipelineConfig(BaseModel):
    """Base configuration with script contract integration"""
    
    def get_script_contract(self) -> Optional['ScriptContract']:
        """
        Get script contract for this configuration.
        
        This base implementation attempts to dynamically load the script contract
        based on naming conventions.
        
        Returns:
            Script contract instance or None if not available
        """
        # Check for hardcoded script_contract first (for backward compatibility)
        if hasattr(self, '_script_contract'):
            return self._script_contract
            
        # Otherwise attempt to load based on class name
        try:
            class_name = self.__class__.__name__.replace('Config', '')
            
            # Try with job_type if available
            if hasattr(self, 'job_type') and self.job_type:
                module_name = f"..pipeline_script_contracts.{class_name.lower()}_{self.job_type.lower()}_contract"
                contract_name = f"{class_name.upper()}_{self.job_type.upper()}_CONTRACT"
                
                try:
                    contract_module = __import__(module_name, fromlist=[''])
                    if hasattr(contract_module, contract_name):
                        return getattr(contract_module, contract_name)
                except (ImportError, AttributeError):
                    pass
            
            # Try without job_type
            module_name = f"..pipeline_script_contracts.{class_name.lower()}_contract"
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
        """
        Property accessor for script contract.
        
        Returns:
            Script contract instance or None if not available
        """
        return self.get_script_contract()
        
    def get_script_path(self, default_path: str = None) -> str:
        """
        Get script path, preferring contract-defined path if available.
        
        Args:
            default_path: Default script path to use if not found in contract
            
        Returns:
            Script path
        """
        # Try to get from contract
        contract = self.get_script_contract()
        if contract and hasattr(contract, 'script_path'):
            return contract.script_path
            
        # Fall back to default or hardcoded path
        if hasattr(self, 'script_path'):
            return self.script_path
            
        return default_path
```

## Integration with Other Components

### With Step Builders

[Step Builders](step_builder.md) use dependency injection pattern with configs:

```python
class XGBoostTrainingStepBuilder(StepBuilderBase):
    def __init__(self, config: XGBoostTrainingConfig, spec=None, **kwargs):
        # Initialize with both config and specification
        if not spec:
            # Load default specification if not provided
            from ..pipeline_step_specs.xgboost_training_spec import XGBOOST_TRAINING_SPEC
            spec = XGBOOST_TRAINING_SPEC
            
        super().__init__(config=config, spec=spec, **kwargs)
        self.config = config
        
        # Use script contract from config if available
        if not self.contract and hasattr(self.config, 'script_contract'):
            self.contract = self.config.script_contract
```

### With Script Contracts

Configs now directly link with script contracts for validation and integration:

```python
class XGBoostTrainingConfig(TrainingStepConfig):
    # This will automatically load the contract from pipeline_script_contracts.xgboost_train_contract
    
    @model_validator(mode='after')
    def validate_against_contract(self) -> 'XGBoostTrainingConfig':
        """Validate configuration against script contract if available"""
        contract = self.script_contract
        if contract:
            # Ensure entry_point matches contract
            if hasattr(contract, 'entry_point') and self.training_entry_point != contract.entry_point:
                logger.warning(
                    f"Entry point '{self.training_entry_point}' doesn't match contract entry point '{contract.entry_point}'. "
                    f"Using contract entry point."
                )
                self.training_entry_point = contract.entry_point
                
            # Ensure environment variables required by contract are set
            if hasattr(contract, 'required_env_vars') and contract.required_env_vars:
                env = getattr(self, 'env', {}) or {}
                for req_var in contract.required_env_vars:
                    if req_var not in env:
                        raise ValueError(f"Environment variable '{req_var}' required by contract but missing in config")
        
        return self
```

### With Step Specifications

Configs now work with step specifications for validation:

```python
class XGBoostTrainingConfig(TrainingStepConfig):
    # Configuration values
    
    def validate_against_specification(self, spec: 'StepSpecification') -> None:
        """
        Validate configuration against a step specification.
        
        Args:
            spec: Step specification to validate against
            
        Raises:
            ValueError: If configuration doesn't meet specification requirements
        """
        if not spec:
            return
            
        # Check if contract is compatible with specification
        if spec.script_contract and self.script_contract:
            if spec.script_contract != self.script_contract:
                # Detailed contract compatibility check
                incompatibilities = []
                for input_name, input_path in spec.script_contract.expected_input_paths.items():
                    if input_name in self.script_contract.expected_input_paths:
                        if self.script_contract.expected_input_paths[input_name] != input_path:
                            incompatibilities.append(
                                f"Input path mismatch for '{input_name}': "
                                f"spec={input_path}, config={self.script_contract.expected_input_paths[input_name]}"
                            )
                
                if incompatibilities:
                    raise ValueError(
                        f"Configuration script contract incompatible with specification:\n" +
                        "\n".join(incompatibilities)
                    )
```

## Strategic Value

Configs provide:

1. **Configuration Centralization**: Single source of truth for all settings
2. **Contract Integration**: Ensure alignment with script contracts
3. **Specification Validation**: Validate against step specifications
4. **Validation and Safety**: Early error detection through Pydantic
5. **Template Reusability**: Common patterns can be shared and customized
6. **Integration Simplification**: Bridge between high-level intent and implementation
7. **Maintainability**: Changes to configuration logic isolated and manageable

## Example Usage with Specifications and Contracts

```python
# Create config for XGBoost training
config = XGBoostTrainingConfig(
    region="NA",
    author="data-scientist",
    bucket="my-ml-bucket",
    pipeline_name="fraud-detection",
    pipeline_description="Fraud Detection ML Pipeline",
    pipeline_version="1.0.0",
    training_instance_type="ml.m5.xlarge",
    hyperparameters=XGBoostModelHyperparameters(
        max_depth=6,
        eta=0.3,
        objective="binary:logistic"
    )
)

# Config automatically links with appropriate script contract
assert config.script_contract.entry_point == "train_xgb.py"

# Create builder with config
from ..pipeline_step_specs.xgboost_training_spec import XGBOOST_TRAINING_SPEC
builder = XGBoostTrainingStepBuilder(config=config, spec=XGBOOST_TRAINING_SPEC)

# Contract is available from both config and spec
assert builder.contract.entry_point == "train_xgb.py"
```

## JSON Configuration Loading

```python
def load_config_from_json(json_path: str, config_class: Type[BasePipelineConfig]) -> BasePipelineConfig:
    """
    Load configuration from JSON file.
    
    Args:
        json_path: Path to JSON config file
        config_class: Configuration class to instantiate
        
    Returns:
        Configuration instance
    """
    with open(json_path, 'r') as f:
        config_data = json.load(f)
    
    # Use Pydantic model_validate for proper validation
    return config_class.model_validate(config_data)

# Usage
config = load_config_from_json("configs/xgboost_training.json", XGBoostTrainingConfig)
```

Configs form the **configuration management foundation** that enables flexible, validated, and environment-aware configuration of pipeline components while maintaining alignment with script contracts and step specifications.
