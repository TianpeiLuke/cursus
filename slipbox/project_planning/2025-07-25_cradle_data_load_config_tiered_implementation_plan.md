# Tiered Field Classification Implementation Plan for Cradle Data Load Configuration

**Date:** 2025-07-25  
**Author:** Luke Xie  
**Status:** Draft  

## Background

We recently implemented a three-tier field classification system in our base configuration and hyperparameter classes. This has successfully provided a clearer separation between user inputs, system defaults, and derived values. The implementation is documented in [config_tiered_design.md](../pipeline_design/config_tiered_design.md).

Now we need to extend this approach to our Cradle data loading configuration, which presents unique challenges due to its nested structure.

## Goals and Objectives

1. Implement three-tier field classification across all Cradle data loading configuration classes
2. Ensure proper inheritance between base configuration and derived classes
3. Maintain backward compatibility with existing pipeline configurations
4. Improve field documentation and organization
5. Reduce redundancy in derived field calculations

## Current Structure Overview

The Cradle data loading configuration follows this nested hierarchy:

```
CradleDataLoadConfig
├── DataSourcesSpecificationConfig
│   └── List[DataSourceConfig]
│       ├── MdsDataSourceConfig
│       ├── EdxDataSourceConfig
│       └── AndesDataSourceConfig
├── TransformSpecificationConfig
│   └── JobSplitOptionsConfig
├── OutputSpecificationConfig
└── CradleJobSpecificationConfig
```

This presents challenges for field categorization because:
1. The nested nature means derived fields can depend on values at different levels of the hierarchy
2. Each config class needs its own tier implementation
3. We need to ensure proper field inheritance through the hierarchy

## Tiered Field Classification

For each configuration class, we'll implement the three-tier field classification as defined in [config_tiered_design.md](../pipeline_design/config_tiered_design.md):

1. **Tier 1: Essential User Inputs** - Fields that users must explicitly provide
2. **Tier 2: System Inputs with Defaults** - Fields with reasonable defaults that can be overridden
3. **Tier 3: Derived Fields** - Fields calculated from other fields via private attributes with properties

## Implementation Details

### Common Base Class For All Components

To avoid code duplication and ensure consistent implementation, a new base class `BaseCradleComponentConfig` was created with the following features:

1. Common Pydantic model configuration
2. Standard implementation of `categorize_fields()` for automatic field categorization
3. Standard implementation of `get_public_init_fields()` for proper inheritance support

All Cradle configuration classes (except the top-level `CradleDataLoadConfig` which extends `BasePipelineConfig`) will inherit from this base class to ensure consistent behavior.

```python
class BaseCradleComponentConfig(BaseModel):
    """
    Base class for Cradle configuration components with three-tier field classification support.
    
    Implements common functionality for categorizing fields and supporting inheritance.
    
    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """
    # Model configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    def categorize_fields(self) -> Dict[str, List[str]]:
        """
        Categorize all fields into three tiers:
        1. Tier 1: Essential User Inputs - fields with no defaults (required)
        2. Tier 2: System Inputs - fields with defaults (optional)
        3. Tier 3: Derived Fields - properties that access private attributes
        
        Returns:
            Dict with keys 'essential', 'system', and 'derived' mapping to lists of field names
        """
        # Implementation details...
    
    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Get fields suitable for initializing a child config.
        Only includes fields that should be passed to child class constructors.
        
        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Implementation details...
```

### Component-Specific Implementation

Each class will be updated to:

1. Inherit from `BaseCradleComponentConfig`
2. Organize fields with explicit tier annotations
3. Use private attributes with read-only properties for derived fields
4. Implement `model_validator` methods to initialize derived fields where needed
5. Override model_config when needed for specific behaviors

### Class-Specific Implementation Details

#### 1. MdsDataSourceConfig

**Tier 1: Essential User Inputs**
- `service_name` - Name of the MDS service
- `region` - Region code for MDS
- `output_schema` - Schema definition

**Tier 2: System Inputs with Defaults**
- `org_id` - Default 0 for regional MDS bucket
- `use_hourly_edx_data_set` - Default False

**Tier 3: Derived Fields**
- None identified yet

#### 2. EdxDataSourceConfig

**Tier 1: Essential User Inputs**
- `edx_provider` - Provider portion of the EDX manifest ARN
- `edx_subject` - Subject portion of the EDX manifest ARN
- `edx_dataset` - Dataset portion of the EDX manifest ARN
- `edx_manifest_key` - Manifest key in format "[...]" that completes the ARN
- `schema_overrides` - Schema definition for EDX

**Tier 2: System Inputs with Defaults**
- None identified yet

**Tier 3: Derived Fields**
- `edx_manifest` - Full ARN derived from provider, subject, dataset, and key
  ```python
  _edx_manifest = f'arn:amazon:edx:iad::manifest/{edx_provider}/{edx_subject}/{edx_dataset}/{edx_manifest_key}'
  ```

#### 3. AndesDataSourceConfig

**Tier 1: Essential User Inputs**
- `provider` - Andes provider ID 
- `table_name` - Name of the Andes table

**Tier 2: System Inputs with Defaults**
- `andes3_enabled` - Default True

**Tier 3: Derived Fields**
- None identified yet

#### 4. DataSourceConfig

**Tier 1: Essential User Inputs**
- `data_source_name` - Logical name for the data source
- `data_source_type` - Type of data source ('MDS', 'EDX', or 'ANDES')

**Tier 2: System Inputs with Defaults**
- None identified yet

**Tier 3: Derived Fields**
- None identified yet, but may need to handle the specific data source properties objects

#### 5. DataSourcesSpecificationConfig

**Tier 1: Essential User Inputs**
- `start_date` - Start timestamp
- `end_date` - End timestamp
- `data_sources` - List of data source configs

**Tier 2: System Inputs with Defaults**
- None identified yet

**Tier 3: Derived Fields**
- None identified yet

#### 6. JobSplitOptionsConfig

**Tier 1: Essential User Inputs**
- `merge_sql` - SQL to run after merging split results (if split_job=True)

**Tier 2: System Inputs with Defaults**
- `split_job` - Default False
- `days_per_split` - Default 7

**Tier 3: Derived Fields**
- None identified yet

#### 7. TransformSpecificationConfig

**Tier 1: Essential User Inputs**
- `transform_sql` - SQL transformation

**Tier 2: System Inputs with Defaults**
- `job_split_options` - Default options for job splitting

**Tier 3: Derived Fields**
- None identified yet

#### 8. OutputSpecificationConfig

**Tier 1: Essential User Inputs**
- `output_schema` - List of column names
- `job_type` - Type of job (training, validation, testing, calibration)

**Tier 2: System Inputs with Defaults**
- `output_format` - Default "PARQUET"
- `output_save_mode` - Default "ERRORIFEXISTS"
- `output_file_count` - Default 0
- `keep_dot_in_output_schema` - Default False
- `include_header_in_s3_output` - Default True

**Tier 3: Derived Fields**
- `output_path` - Derived from pipeline_s3_loc and job_type using this formula:
  ```python
  _output_path = f"{config.pipeline_s3_loc}/data-load/{job_type}"
  ```

#### 9. CradleJobSpecificationConfig

**Tier 1: Essential User Inputs**
- `cradle_account` - Cradle account name

**Tier 2: System Inputs with Defaults**
- `cluster_type` - Default "STANDARD"
- `extra_spark_job_arguments` - Default empty string
- `job_retry_count` - Default 1

**Tier 3: Derived Fields**
- None identified yet

#### 10. CradleDataLoadConfig (Top-Level)

**Tier 1: Essential User Inputs**
- `job_type` - Type of job (training, validation, etc.)
- `cradle_job_spec` - Cradle job specification
- `data_sources_spec` - Data sources specification
- `transform_spec` - Transform specification 
- Will inherit essential user inputs from BasePipelineConfig (role, region, etc.)

**Tier 2: System Inputs with Defaults**
- `s3_input_override` - Default None

**Tier 3: Derived Fields**
- `output_spec.output_path` - Derived from BasePipelineConfig's pipeline_s3_loc and job_type

## Sample Implementation for EdxDataSourceConfig

```python
class EdxDataSourceConfig(BaseModel):
    """
    Configuration for EDX data source with three-tier field classification.
    
    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """
    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    
    edx_provider: str = Field(
        description="Provider portion of the EDX manifest ARN"
    )
    
    edx_subject: str = Field(
        description="Subject portion of the EDX manifest ARN"
    )
    
    edx_dataset: str = Field(
        description="Dataset portion of the EDX manifest ARN"
    )
    
    edx_manifest_key: str = Field(
        description="Manifest key in format '[\"xxx\",...]' that completes the ARN"
    )
    
    schema_overrides: List[Dict[str, Any]] = Field(
        description="List of dicts overriding the EDX schema, e.g. "
                    "[{'field_name':'order_id','field_type':'STRING'}, ...]"
    )
    
    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields
    
    _edx_manifest: Optional[str] = PrivateAttr(default=None)
    
    # Model configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    @property
    def edx_manifest(self) -> str:
        """Get EDX manifest ARN derived from provider, subject, dataset and key."""
        if self._edx_manifest is None:
            self._edx_manifest = (
                f"arn:amazon:edx:iad::manifest/"
                f"{self.edx_provider}/{self.edx_subject}/{self.edx_dataset}/{self.edx_manifest_key}"
            )
        return self._edx_manifest
    
    @field_validator("edx_manifest_key")
    @classmethod
    def validate_manifest_key_format(cls, v: str) -> str:
        """Validate that edx_manifest_key is in the format '[...]'"""
        if not (v.startswith('[') and v.endswith(']')):
            raise ValueError(
                f"edx_manifest_key must be in format '[\"xxx\",...]', got '{v}'"
            )
        return v
    
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> 'EdxDataSourceConfig':
        """Initialize derived fields after validation."""
        # Initialize the manifest using the key
        self._edx_manifest = (
            f"arn:amazon:edx:iad::manifest/"
            f"{self.edx_provider}/{self.edx_subject}/{self.edx_dataset}/{self.edx_manifest_key}"
        )
        return self
    
    def categorize_fields(self) -> Dict[str, List[str]]:
        """
        Categorize all fields into three tiers:
        1. Tier 1: Essential User Inputs - fields with no defaults (required)
        2. Tier 2: System Inputs - fields with defaults (optional)
        3. Tier 3: Derived Fields - properties that access private attributes
        
        Returns:
            Dict with keys 'essential', 'system', and 'derived' mapping to lists of field names
        """
        # Initialize categories
        categories = {
            'essential': [],  # Tier 1: Required, public
            'system': [],     # Tier 2: Optional (has default), public
            'derived': []     # Tier 3: Public properties
        }
        
        # Get model fields
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
    
    def get_public_init_fields(self) -> Dict[str, Any]:
        """Get fields suitable for initializing a child config."""
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

## Special Implementation for CradleDataLoadConfig

The top-level `CradleDataLoadConfig` needs special attention as it inherits from `BasePipelineConfig` (not `BaseCradleComponentConfig`) and contains nested configuration objects:

```python
class CradleDataLoadConfig(BasePipelineConfig):
    """
    Top-level configuration for Cradle data loading steps with three-tier field classification.
    
    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """
    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    
    job_type: str = Field(
        description="One of ['training','validation','testing','calibration'] to indicate which dataset this job is pulling"
    )
    
    data_sources_spec: DataSourcesSpecificationConfig = Field(
        description="Full data sources specification (start/end dates plus list of sources)."
    )
    
    transform_spec: TransformSpecificationConfig = Field(
        description="Transform specification: SQL + job split options."
    )
    
    output_spec: OutputSpecificationConfig = Field(
        description="Output specification: schema, path, format, save mode, etc."
    )
    
    cradle_job_spec: CradleJobSpecificationConfig = Field(
        description="Cradle job specification: cluster type, account, retry count, etc."
    )
    
    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override
    
    s3_input_override: Optional[str] = Field(
        default=None,
        description="If set, skip Cradle data pull and use this S3 prefix directly"
    )
    
    # ===== Derived Fields (Tier 3) =====
    # None currently, but could add derived fields if needed
    
    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v
    
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> 'CradleDataLoadConfig':
        """Initialize all derived fields once after validation."""
        # Initialize base class derived fields first
        super().initialize_derived_fields()
        
        # Initialize any additional derived fields for this class here
        
        return self
    
    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Get a dictionary of public fields suitable for initializing a child config.
        Only includes fields that should be passed to child class constructors.
        
        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()
        
        # Get fields from this class using categorize_fields
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
        
        # Combine (base fields and derived fields, with derived taking precedence)
        return {**base_fields, **init_fields}
```

## Testing Strategy

1. **Unit Tests**:
   - Update existing tests for each configuration class
   - Add new tests for field categorization
   - Test inheritance and composition with nested configurations
   - Verify derived field calculations

2. **Integration Tests**:
   - Test the full CradleDataLoadConfig with nested configurations
   - Verify backward compatibility with existing config JSON files
   - Test serialization/deserialization

## Implementation Timeline

1. **Day 1**:
   - Create `BaseCradleComponentConfig` class
   - Update base data source configurations to inherit from it:
     - `MdsDataSourceConfig`
     - `EdxDataSourceConfig`
     - `AndesDataSourceConfig`
   
2. **Day 2**:
   - Update intermediate configurations to use the base class:
     - `DataSourceConfig`
     - `JobSplitOptionsConfig` 
     - `DataSourcesSpecificationConfig`
     - `TransformSpecificationConfig`
   
3. **Day 3**:
   - Update output and job specifications:
     - `OutputSpecificationConfig` (with derived output_path)
     - `CradleJobSpecificationConfig`
   - Update top-level `CradleDataLoadConfig`
   
4. **Day 4**:
   - Add tests for all configuration classes
   - Fix issues and refine implementation
   - Update documentation
   - Test with real-world pipelines

## Conclusion

Implementing the three-tier field classification system across the nested Cradle data loading configurations will:

1. Clarify which fields are essential user inputs vs. system defaults vs. derived fields
2. Provide a common base class for all configuration components
2. Improve field inheritance and composition
3. Reduce redundancy in field calculations
4. Make the configuration system more maintainable

This implementation follows the design principles established in [config_tiered_design.md](../pipeline_design/config_tiered_design.md) and will create a consistent approach across all configuration classes in the system.
