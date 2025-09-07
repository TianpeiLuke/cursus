---
tags:
  - code
  - core
  - config_fields
  - factory_functions
  - cradle_integration
keywords:
  - cradle config factory
  - configuration generation
  - factory functions
  - data loading configuration
  - MDS EDX integration
topics:
  - configuration factory
  - data loading
  - cradle integration
language: python
date of note: 2025-09-07
---

# Cradle Configuration Factory

## Overview

The `cradle_config_factory.py` module provides helper functions for creating complex `CradleDataLoadConfig` objects with minimal user inputs. It implements factory patterns to simplify the creation of nested configuration structures by deriving complex configurations from essential user inputs, following the **Simplified Configuration Creation** principle.

## Purpose

This module provides factory functions that:
- Simplify creation of complex CradleDataLoadConfig objects
- Generate nested configurations from essential user inputs
- Handle MDS and EDX data source configuration automatically
- Support both training and calibration job types
- Provide intelligent defaults for common use cases
- Generate SQL transformations automatically

## Module Constants

### Default Schema Definitions

```python
# Default values
DEFAULT_TAG_SCHEMA = [
    'order_id',
    'marketplace_id',
    'tag_date',
    'is_abuse',
    'abuse_type',
    'concession_type'
]

DEFAULT_MDS_BASE_FIELDS = [
    'objectId', 
    'transactionDate'
]
```

These constants provide sensible defaults for common data loading scenarios while allowing customization when needed.

## Core Factory Functions

### Primary Factory Function

The main factory function for creating CradleDataLoadConfig objects:

```python
def create_cradle_data_load_config(
    # Base configuration (for inheritance)
    base_config: BasePipelineConfig,
    
    # Job configuration
    job_type: str,  # 'training' or 'calibration'
    
    # MDS field list (direct fields to include)
    mds_field_list: List[str],
    
    # Data timeframe
    start_date: str,
    end_date: str,
    
    # EDX data source
    tag_edx_provider: str,
    tag_edx_subject: str,
    tag_edx_dataset: str,
    etl_job_id: str,
    edx_manifest_comment: Optional[str] = None,  # Optional comment for EDX manifest key
    
    # MDS data source (if not in base_config)
    service_name: Optional[str] = None,
    
    # Infrastructure configuration
    cradle_account: str = "Buyer-Abuse-RnD-Dev",
    org_id: int = 0,  # Default organization ID for regional MDS bucket
    
    # Optional overrides with reasonable defaults
    cluster_type: str = "STANDARD",
    output_format: str = "PARQUET",
    output_save_mode: str = "ERRORIFEXISTS",
    split_job: bool = False,
    days_per_split: int = 7,
    merge_sql: Optional[str] = None,
    s3_input_override: Optional[str] = None,
    transform_sql: Optional[str] = None,  # Auto-generated if not provided
    tag_schema: Optional[List[str]] = None,  # Default provided if not specified
    use_dedup_sql: Optional[bool] = None,  # Whether to use dedup SQL format (default: same as split_job)
    
    # Join configuration
    mds_join_key: str = 'objectId',
    edx_join_key: str = 'order_id',
    join_type: str = 'JOIN'
) -> CradleDataLoadConfig:
    """
    Create a CradleDataLoadConfig with minimal required inputs.
    
    This helper function simplifies the creation of a CradleDataLoadConfig
    by handling the generation of nested configurations from essential user inputs.
    """
```

### Dual Configuration Factory

Factory function for creating both training and calibration configurations:

```python
def create_training_and_calibration_configs(
    # Base config for inheritance
    base_config: BasePipelineConfig,
    
    # MDS field list
    mds_field_list: List[str],
    
    # EDX data source
    tag_edx_provider: str,
    tag_edx_subject: str,
    tag_edx_dataset: str,
    etl_job_id: str,
    
    # Data timeframes
    training_start_date: str,
    training_end_date: str,
    calibration_start_date: str,
    calibration_end_date: str,
    
    # Additional configuration options...
) -> Dict[str, CradleDataLoadConfig]:
    """
    Create both training and calibration CradleDataLoadConfig objects with consistent settings.
    
    Returns:
        Dict[str, CradleDataLoadConfig]: Dictionary with 'training' and 'calibration' configs
    """
```

## Helper Functions

### Region Mapping

```python
def _map_region_to_aws_region(region: str) -> str:
    """
    Map marketplace region to AWS region.
    
    Args:
        region (str): Marketplace region ('NA', 'EU', 'FE')
        
    Returns:
        str: AWS region name
    """
    region_mapping = {
        "NA": "us-east-1",
        "EU": "eu-west-1",
        "FE": "us-west-2"
    }
    
    if region not in region_mapping:
        raise ValueError(f"Invalid region: {region}. Must be one of {list(region_mapping.keys())}")
        
    return region_mapping[region]
```

### Schema Generation

```python
def _create_field_schema(fields: List[str]) -> List[Dict[str, str]]:
    """
    Convert a list of field names to schema dictionaries.
    
    Args:
        fields (List[str]): List of field names
        
    Returns:
        List[Dict[str, str]]: List of schema dictionaries
    """
    return [{'field_name': field, 'field_type': 'STRING'} for field in fields]
```

### EDX Manifest Generation

```python
def _format_edx_manifest_key(
    etl_job_id: str,
    start_date: str,
    end_date: str,
    comment: Optional[str] = None
) -> str:
    """
    Format an EDX manifest key with date components and optional comment.
    
    This function supports two formats:
    1. With comment: ["etl_job_id",start_dateZ,end_dateZ,"comment"]
    2. Without comment: ["etl_job_id",start_dateZ,end_dateZ]
    
    Args:
        etl_job_id (str): ETL job ID
        start_date (str): Start date string
        end_date (str): End date string
        comment (Optional[str]): Optional comment or region code (None for no comment)
        
    Returns:
        str: Properly formatted EDX manifest key
    """
    # Ensure the date strings do not already have 'Z' appended
    start_date_clean = start_date.rstrip('Z')
    end_date_clean = end_date.rstrip('Z')
    
    # Format depends on whether comment is provided
    if comment:
        return f'["{etl_job_id}",{start_date_clean}Z,{end_date_clean}Z,"{comment}"]'
    else:
        return f'["{etl_job_id}",{start_date_clean}Z,{end_date_clean}Z]'

def _create_edx_manifest(
    provider: str,
    subject: str,
    dataset: str,
    etl_job_id: str,
    start_date: str,
    end_date: str,
    comment: Optional[str] = None
) -> str:
    """
    Create an EDX manifest ARN with date components.
    
    Returns:
        str: Properly formatted EDX manifest ARN
    """
    # Format the manifest key using the helper function
    manifest_key = _format_edx_manifest_key(etl_job_id, start_date, end_date, comment)
    
    return (
        f'arn:amazon:edx:iad::manifest/'
        f'{provider}/{subject}/{dataset}/'
        f'{manifest_key}'
    )
```

## SQL Generation Engine

### Advanced Transform SQL Generation

The module includes a sophisticated SQL generation engine:

```python
def _generate_transform_sql(
    mds_source_name: str,
    edx_source_name: str,
    mds_field_list: List[str],
    tag_schema: List[str],
    mds_join_key: str = 'objectId',
    edx_join_key: str = 'order_id',
    join_type: str = 'JOIN',
    use_dedup_sql: bool = False
) -> str:
    """
    Generate a SQL query to join MDS and EDX data with configurable join keys.
    
    This function ensures there are no duplicate fields in the SELECT clause
    by checking for fields that appear in both MDS and tag schema.
    Field names are compared case-insensitively to prevent duplications
    where the only difference is case (e.g., "OrderId" and "orderid").
    
    Two SQL formats are supported:
    1. Standard format (default): Direct join with source prefixes for each column
    2. Deduplication format: Uses a subquery with ROW_NUMBER() to ensure only one 
       record per objectId/order_id pair, and lists fields without source prefixes
    
    Special handling for join keys:
    - Both join keys (from MDS and EDX) are explicitly included in the SQL
    - This ensures the join operation works correctly even with case differences
    - The keys are aliased to avoid ambiguity and collisions
    """
```

#### Standard SQL Format

For simple joins without deduplication:

```sql
SELECT
mds_source.objectId as mds_source_objectId,
edx_source.order_id as edx_source_order_id,
mds_source.transactionDate,
mds_source.field1,
edx_source.marketplace_id,
edx_source.is_abuse
FROM RAW_MDS_NA
JOIN TAGS 
ON RAW_MDS_NA.objectId=TAGS.order_id
```

#### Deduplication SQL Format

For joins with row deduplication:

```sql
SELECT
    field1,
    is_abuse,
    marketplace_id,
    objectId,
    order_id,
    transactionDate
FROM (
    SELECT
        RAW_MDS_NA.field1,
        TAGS.is_abuse,
        TAGS.marketplace_id,
        RAW_MDS_NA.objectId,
        TAGS.order_id,
        RAW_MDS_NA.transactionDate,
        ROW_NUMBER() OVER (PARTITION BY RAW_MDS_NA.objectId, TAGS.order_id ORDER BY RAW_MDS_NA.transactionDate DESC) as row_num
    FROM RAW_MDS_NA
    JOIN TAGS ON RAW_MDS_NA.objectId = TAGS.order_id
)
WHERE row_num = 1
```

### Field Deduplication Logic

```python
def _get_all_fields(
    mds_fields: List[str],
    tag_fields: List[str]
) -> List[str]:
    """
    Get a combined list of all fields from MDS and EDX sources.
    
    This function handles case-insensitivity to avoid duplicate columns in SQL SELECT
    statements where the only difference is case (e.g., "OrderId" and "orderid").
    When duplicates with different cases are found, the first occurrence is kept.
    
    Args:
        mds_fields (List[str]): List of MDS fields
        tag_fields (List[str]): List of tag fields
        
    Returns:
        List[str]: Combined and deduplicated list of fields
    """
    # Track lowercase field names to detect duplicates
    seen_lowercase = {}
    deduplicated_fields = []
    
    # Process all fields, keeping only the first occurrence when case-insensitive duplicates exist
    for field in mds_fields + tag_fields:
        field_lower = field.lower()
        if field_lower not in seen_lowercase:
            seen_lowercase[field_lower] = True
            deduplicated_fields.append(field)
    
    return sorted(deduplicated_fields)
```

## Usage Examples

### Basic Configuration Creation

```python
from src.cursus.core.config_fields.cradle_config_factory import create_cradle_data_load_config
from src.cursus.core.base.config_base import BasePipelineConfig

# Create base configuration
base_config = BasePipelineConfig(
    author="data-scientist",
    bucket="my-ml-bucket",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    region="NA",
    service_name="buyer-abuse",
    pipeline_version="1.0.0"
)

# Create training data loading configuration
training_config = create_cradle_data_load_config(
    base_config=base_config,
    job_type="training",
    mds_field_list=[
        "orderId",
        "customerId", 
        "transactionAmount",
        "paymentMethod"
    ],
    start_date="2025-01-01T00:00:00",
    end_date="2025-01-31T23:59:59",
    tag_edx_provider="abuse-detection",
    tag_edx_subject="buyer-abuse",
    tag_edx_dataset="training-labels",
    etl_job_id="training-job-2025-01"
)

print(f"Created training config: {training_config}")
print(f"Generated SQL: {training_config.transform_spec.transform_sql}")
```

### Advanced Configuration with Custom Settings

```python
# Create configuration with custom SQL and splitting
advanced_config = create_cradle_data_load_config(
    base_config=base_config,
    job_type="training",
    mds_field_list=["orderId", "customerId", "transactionAmount"],
    start_date="2025-01-01T00:00:00",
    end_date="2025-03-31T23:59:59",
    tag_edx_provider="abuse-detection",
    tag_edx_subject="buyer-abuse", 
    tag_edx_dataset="training-labels",
    etl_job_id="training-job-q1-2025",
    
    # Advanced options
    split_job=True,
    days_per_split=14,
    use_dedup_sql=True,
    merge_sql="SELECT * FROM INPUT WHERE transactionAmount > 0",
    join_type="LEFT JOIN",
    mds_join_key="orderId",
    edx_join_key="order_id",
    
    # Custom tag schema
    tag_schema=[
        "order_id",
        "marketplace_id", 
        "is_fraudulent",
        "fraud_type",
        "confidence_score"
    ]
)

print(f"Split job enabled: {advanced_config.transform_spec.job_split_options.split_job}")
print(f"Days per split: {advanced_config.transform_spec.job_split_options.days_per_split}")
```

### Creating Both Training and Calibration Configs

```python
from src.cursus.core.config_fields.cradle_config_factory import create_training_and_calibration_configs

# Create both configurations with consistent settings
configs = create_training_and_calibration_configs(
    base_config=base_config,
    mds_field_list=["orderId", "customerId", "transactionAmount"],
    
    # EDX configuration
    tag_edx_provider="abuse-detection",
    tag_edx_subject="buyer-abuse",
    tag_edx_dataset="labels",
    etl_job_id="ml-pipeline-2025-q1",
    
    # Training data timeframe
    training_start_date="2025-01-01T00:00:00",
    training_end_date="2025-02-28T23:59:59",
    
    # Calibration data timeframe  
    calibration_start_date="2025-03-01T00:00:00",
    calibration_end_date="2025-03-31T23:59:59",
    
    # Shared configuration
    split_job=True,
    days_per_split=7,
    output_format="PARQUET"
)

training_config = configs["training"]
calibration_config = configs["calibration"]

print(f"Training config job type: {training_config.job_type}")
print(f"Calibration config job type: {calibration_config.job_type}")
print(f"Both configs use same MDS fields: {training_config.data_sources_spec.data_sources[0].mds_data_source_properties.output_schema}")
```

### Integration with Pipeline Templates

```python
from src.cursus.core.assembler.pipeline_template_base import PipelineTemplateBase

class CradleDataPipelineTemplate(PipelineTemplateBase):
    
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        """Create configuration map using the factory."""
        
        # Create base configuration from template settings
        base_config = BasePipelineConfig(
            author=self.author,
            bucket=self.bucket,
            role=self.role,
            region=self.region,
            service_name=self.service_name,
            pipeline_version=self.pipeline_version
        )
        
        # Create data loading configurations using factory
        data_configs = create_training_and_calibration_configs(
            base_config=base_config,
            mds_field_list=self.mds_fields,
            tag_edx_provider=self.edx_provider,
            tag_edx_subject=self.edx_subject,
            tag_edx_dataset=self.edx_dataset,
            etl_job_id=self.etl_job_id,
            training_start_date=self.training_start,
            training_end_date=self.training_end,
            calibration_start_date=self.calibration_start,
            calibration_end_date=self.calibration_end
        )
        
        return {
            "data_loading_training": data_configs["training"],
            "data_loading_calibration": data_configs["calibration"]
        }
```

## Configuration Generation Process

The factory function follows a systematic process to generate complex configurations:

### 1. Input Validation and Defaults

```python
# Use default tag schema if not provided
if tag_schema is None:
    tag_schema = DEFAULT_TAG_SCHEMA
    
# Get service_name from base_config if not provided
if service_name is None:
    service_name = base_config.service_name
    
# Get the region from base_config
region = base_config.region

# Set path validation env var if needed
if "MODS_SKIP_PATH_VALIDATION" not in os.environ:
    os.environ["MODS_SKIP_PATH_VALIDATION"] = "true"

# If split_job is True, ensure merge_sql is provided
if split_job and merge_sql is None:
    merge_sql = "SELECT * FROM INPUT"  # Default merge SQL
    
# Set use_dedup_sql default if not provided
if use_dedup_sql is None:
    use_dedup_sql = split_job  # Default to using dedup SQL when split_job is True
```

### 2. MDS Data Source Configuration

```python
# Create complete MDS field list by combining base fields with provided fields
complete_mds_field_list = list(set(DEFAULT_MDS_BASE_FIELDS + mds_field_list))
mds_field_list = sorted(mds_field_list)

# Create MDS schema
mds_output_schema = _create_field_schema(complete_mds_field_list)

# Create MDS data source inner config
mds_data_source_inner_config = MdsDataSourceConfig(
    service_name=service_name,
    region=region,
    output_schema=mds_output_schema,
    org_id=org_id  # Use the provided org_id parameter
)
```

### 3. EDX Data Source Configuration

```python
# Create EDX manifest key with proper Z suffixes for timestamps
# Use edx_manifest_comment as-is (including None) - don't default to region
edx_manifest_key = _format_edx_manifest_key(
    etl_job_id=etl_job_id,
    start_date=start_date,
    end_date=end_date,
    comment=edx_manifest_comment
)

# Create EDX schema overrides
edx_schema_overrides = _create_field_schema(tag_schema)

# Create EDX data source inner config
edx_source_inner_config = EdxDataSourceConfig(
    edx_provider=tag_edx_provider,
    edx_subject=tag_edx_subject,
    edx_dataset=tag_edx_dataset,
    edx_manifest_key=edx_manifest_key,
    schema_overrides=edx_schema_overrides
)
```

### 4. Transform SQL Generation

```python
# Generate transform SQL if not provided
if transform_sql is None:
    transform_sql = _generate_transform_sql(
        mds_source_name=mds_data_source.data_source_name,
        edx_source_name=edx_data_source.data_source_name,
        mds_field_list=complete_mds_field_list,
        tag_schema=tag_schema,
        mds_join_key=mds_join_key,
        edx_join_key=edx_join_key,
        join_type=join_type,
        use_dedup_sql=use_dedup_sql
    )
```

### 5. Final Configuration Assembly

```python
# Use from_base_config to inherit from the base configuration
# This ensures all base fields (region, role, etc.) are properly inherited
# while also respecting the three-tier design pattern
cradle_data_load_config = CradleDataLoadConfig.from_base_config(
    base_config,
    
    # Add step-specific fields
    job_type=job_type,
    data_sources_spec=data_sources_spec,
    transform_spec=transform_spec,
    output_spec=output_spec,
    cradle_job_spec=cradle_job_spec,
    s3_input_override=s3_input_override
)
```

## Error Handling and Validation

### Region Validation

```python
def _map_region_to_aws_region(region: str) -> str:
    """Map marketplace region to AWS region."""
    region_mapping = {
        "NA": "us-east-1",
        "EU": "eu-west-1",
        "FE": "us-west-2"
    }
    
    if region not in region_mapping:
        raise ValueError(f"Invalid region: {region}. Must be one of {list(region_mapping.keys())}")
        
    return region_mapping[region]
```

### Input Validation

```python
# Validate required parameters
if not mds_field_list:
    raise ValueError("mds_field_list cannot be empty")

if not start_date or not end_date:
    raise ValueError("start_date and end_date are required")

if job_type not in ["training", "calibration"]:
    raise ValueError("job_type must be 'training' or 'calibration'")
```

### SQL Generation Safety

The SQL generation includes safety measures:

```python
# Ensure no duplicate fields in SELECT clause
seen_lowercase = {}
deduplicated_fields = []

for field in mds_fields + tag_fields:
    field_lower = field.lower()
    if field_lower not in seen_lowercase:
        seen_lowercase[field_lower] = True
        deduplicated_fields.append(field)
```

## Performance Considerations

### Field Deduplication

The factory uses efficient algorithms for field deduplication:

```python
# O(n) deduplication using set-based tracking
complete_mds_field_list = list(set(DEFAULT_MDS_BASE_FIELDS + mds_field_list))
```

### SQL Generation Optimization

The SQL generation is optimized for readability and performance:

```python
# Sort fields for consistent output
outer_field_list = ',\n    '.join(sorted(outer_select_fields))
inner_field_list = ',\n        '.join(sorted(inner_select_fields))
```

### Memory Efficiency

The factory functions avoid unnecessary object creation:

```python
# Reuse existing configurations when possible
if service_name is None:
    service_name = base_config.service_name  # Reuse from base config
```

## Design Patterns

### Factory Pattern

The module implements the factory pattern for complex object creation:

```python
def create_cradle_data_load_config(...) -> CradleDataLoadConfig:
    """Factory function that encapsulates complex object creation logic."""
    # Complex creation logic hidden from users
    return fully_configured_object
```

### Builder Pattern Elements

The factory incorporates builder pattern elements:

```python
# Step-by-step construction
data_sources_spec = DataSourcesSpecificationConfig(...)
transform_spec = TransformSpecificationConfig(...)
output_spec = OutputSpecificationConfig(...)
cradle_job_spec = CradleJobSpecificationConfig(...)

# Final assembly
config = CradleDataLoadConfig.from_base_config(base_config, ...)
```

### Template Method Pattern

The dual configuration factory uses template method pattern:

```python
def create_training_and_calibration_configs(...):
    """Template method that creates both configurations using the same process."""
    
    # Create training config using template
    training_config = create_cradle_data_load_config(
        base_config=base_config,
        job_type="training",
        # ... other parameters
    )
    
    # Create calibration config using same template
    calibration_config = create_cradle_data_load_config(
        base_config=base_config,
        job_type="calibration", 
        # ... other parameters
    )
    
    return {"training": training_config, "calibration": calibration_config}
```

## Integration Points

### Base Configuration Integration

The factory integrates with the base configuration system:

```python
# Inherit from base configuration using three-tier pattern
cradle_data_load_config = CradleDataLoadConfig.from_base_config(
    base_config,
    # Additional step-specific fields
)
```

### Pipeline Template Integration

The factory is designed for use in pipeline templates:

```python
def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
    """Pipeline template method using the factory."""
    configs = create_training_and_calibration_configs(...)
    return {
        "data_loading_training": configs["training"],
        "data_loading_calibration": configs["calibration"]
    }
```

### Configuration Field System Integration

The factory works with the configuration field management system:

```python
# Uses configuration constants and patterns
from .constants import DEFAULT_TAG_SCHEMA, DEFAULT_MDS_BASE_FIELDS

# Integrates with type-aware serialization
# Generated configurations are fully serializable
```

## Related Documentation

### Core Dependencies
- [Configuration Base](../base/config_base.md): Base configuration class used for inheritance
- [Configuration Constants](constants.md): May use constants for default values
- [Configuration Class Store](config_class_store.md): Generated configurations can be registered

### Integration Points
- [Pipeline Template Base](../assembler/pipeline_template_base.md): Uses factory functions for configuration creation
- [Configuration Merger](../config_field/config_merger.md): Generated configurations can be merged
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md): Generated configurations are serializable

### Cradle Integration
- **CradleDataLoadConfig**: The main configuration class created by the factory
- **MDS/EDX Data Sources**: External data source configurations
- **Cradle Job Specifications**: Infrastructure and job configuration

### System Overview
- [Configuration Fields Overview](README.md): System overview and integration
