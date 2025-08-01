# Cradle Data Load Config Helper Design

## Overview

This design document outlines the development of a helper function that simplifies the creation of `CradleDataLoadConfig` objects. The goal is to reduce complexity by abstracting away the nested structure and interrelated fields, requiring only essential user inputs while automatically handling derived fields.

## Problem Statement

The current process for creating a `CradleDataLoadConfig` is complex and error-prone:

1. **Deeply nested structure**: The configuration requires multiple levels of nested objects (e.g., `DataSourcesSpecificationConfig`, `DataSourceConfig`, `MdsDataSourceConfig`, `EdxDataSourceConfig`).

2. **Complex interdependencies**: Several fields across different nested objects are derived from common sources:
   - `output_schema` in `MdsDataSourceConfig`
   - `schema_overrides` in `EdxDataSourceConfig` 
   - `transform_sql` in `TransformSpecificationConfig`
   - `output_schema` in `OutputSpecificationConfig`

3. **Repetitive boilerplate code**: The notebook template contains substantial boilerplate code to set up each configuration component.

4. **Field dependencies**: Many configuration fields are derived from hyperparameters, particularly `full_field_list`, `tab_field_list`, and `cat_field_list`.

## Design Goals

1. **Simplify API**: Provide a function that accepts only essential user inputs.

2. **Maintain consistency**: Ensure derived fields remain consistent with their sources.

3. **Handle field derivation**: Automatically generate field schemas, SQL queries, and output specifications.

4. **Support both training and calibration**: Allow easy creation of both training and calibration data load configurations.

5. **Apply reasonable defaults**: Use system defaults for non-essential fields.

## API Design

```python
def create_cradle_data_load_config(
    # Base pipeline essentials
    role: str,
    region: str,
    pipeline_s3_loc: str,
    
    # Job configuration
    job_type: str,  # 'training' or 'calibration'
    
    # MDS field list (direct fields to include)
    mds_field_list: List[str],
    
    # Data timeframe
    start_date: str,
    end_date: str,
    
    # MDS data source
    service_name: str,
    
    # EDX data source
    tag_edx_provider: str,
    tag_edx_subject: str,
    tag_edx_dataset: str,
    etl_job_id: str,
    
    # Infrastructure configuration
    cradle_account: str = "Buyer-Abuse-RnD-Dev",
    aws_region: Optional[str] = None,
    
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
    
    # Join configuration
    mds_join_key: str = 'objectId',  # MDS field to use for joining
    edx_join_key: str = 'order_id',  # EDX field to use for joining
    join_type: str = 'JOIN'  # SQL join type ('JOIN', 'LEFT JOIN', etc.)
) -> CradleDataLoadConfig:
    """
    Create a CradleDataLoadConfig with minimal required inputs.
    
    This helper function simplifies the creation of a CradleDataLoadConfig
    by handling the generation of nested configurations from essential user inputs.
    
    Parameters:
        role (str): IAM role to use for the pipeline
        region (str): Marketplace region ('NA', 'EU', 'FE')
        pipeline_s3_loc (str): S3 location for pipeline artifacts
        
        job_type (str): Type of job ('training' or 'calibration')
        
        mds_field_list (List[str]): List of fields to include from MDS
        
        start_date (str): Start date for data pull (format: YYYY-MM-DDT00:00:00)
        end_date (str): End date for data pull (format: YYYY-MM-DDT00:00:00)
        
        service_name (str): Name of the MDS service
        
        tag_edx_provider (str): EDX provider for tags
        tag_edx_subject (str): EDX subject for tags
        tag_edx_dataset (str): EDX dataset for tags
        etl_job_id (str): ETL job ID for the EDX manifest
        
        cradle_account (str): Cradle account name (default: "Buyer-Abuse-RnD-Dev")
        aws_region (str, optional): AWS region, derived from region if not provided
        
        cluster_type (str): Cradle cluster type (default: "STANDARD")
        output_format (str): Output format (default: "PARQUET")
        output_save_mode (str): Output save mode (default: "ERRORIFEXISTS")
        split_job (bool): Whether to split the job (default: False)
        days_per_split (int): Days per split if splitting (default: 7)
        merge_sql (str, optional): SQL to merge split results, required if split_job=True
        s3_input_override (str, optional): S3 input override
        transform_sql (str, optional): Custom transform SQL, auto-generated if not provided
        tag_schema (List[str], optional): Schema for tag data, default provided if not specified
    
    Returns:
        CradleDataLoadConfig: A fully configured CradleDataLoadConfig object
    """
```

## Implementation Details

### Field Generation

1. **MDS output_schema**:
   - Generate from `DEFAULT_MDS_BASE_FIELDS + mds_field_list`
   - Convert each field to `{'field_name': field, 'field_type': 'STRING'}`

2. **EDX schema_overrides**:
   - Default tag schema if not provided: `['order_id', 'marketplace_id', 'tag_date', 'is_abuse', 'abuse_type', 'concession_type']`
   - Convert to `[{'field_name': field, 'field_type': 'STRING'}]`

3. **EDX manifest**:
   - Construct from template using provided parameters:
   ```
   f'arn:amazon:edx:iad::manifest/{tag_edx_provider}/{tag_edx_subject}/{tag_edx_dataset}/["{etl_job_id}",{start_date}Z,{end_date}Z,"{region}"]'
   ```

4. **Transform SQL**:
   - If provided, use as-is
   - If not provided, generate SQL that joins MDS and EDX sources using specified join keys and type:
   ```
   SELECT [fields] FROM mds_source JOIN_TYPE edx_source ON mds_source.mds_join_key=edx_source.edx_join_key
   ```
   - Default join: `JOIN mds_source.objectId=edx_source.order_id`
   - Configurable to use different fields and join types (e.g., LEFT JOIN, INNER JOIN)
   - Handles duplicate fields between sources (keeping MDS version by default)

5. **Output schema**:
   - Collect all fields from both MDS and EDX sources

### Component Construction

1. Build `MdsDataSourceConfig` from MDS fields
2. Build `EdxDataSourceConfig` from EDX fields
3. Create `DataSourceConfig` objects for both MDS and EDX
4. Create `DataSourcesSpecificationConfig` with data sources and timeframe
5. Create `JobSplitOptionsConfig` from split parameters
6. Build `TransformSpecificationConfig` with transform SQL
7. Build `OutputSpecificationConfig` with output fields
8. Create `CradleJobSpecificationConfig` with cluster settings
9. Assemble final `CradleDataLoadConfig`

## Usage Examples

### Basic Usage for Training Data

```python
training_config = create_cradle_data_load_config(
    # Base pipeline info
    role="ML-Developer",
    region="NA",
    pipeline_s3_loc="s3://my-bucket/pipeline",
    
    # Job type
    job_type="training",
    
    # MDS fields to include
    mds_field_list=hyperparams.mds_field_list,
    
    # Data timeframe
    start_date="2025-01-01T00:00:00",
    end_date="2025-04-17T00:00:00",
    
    # MDS data source
    service_name="AtoZ",
    
    # EDX data source
    tag_edx_provider="trms-abuse-analytics",
    tag_edx_subject="qingyuye-notr-exp",
    tag_edx_dataset="atoz-tag",
    etl_job_id="24292902"
)
```

### Creating Both Training and Calibration Configs

```python
def create_training_and_calibration_configs(
    hyperparams,
    role,
    region,
    pipeline_s3_loc,
    service_name,
    tag_edx_info,
    training_dates,
    calibration_dates
):
    """Create both training and calibration configs with consistent settings."""
    
    # Create training config
    training_config = create_cradle_data_load_config(
        role=role,
        region=region,
        pipeline_s3_loc=pipeline_s3_loc,
        job_type="training",
        mds_field_list=hyperparams.mds_field_list,
        start_date=training_dates["start"],
        end_date=training_dates["end"],
        service_name=service_name,
        tag_edx_provider=tag_edx_info["provider"],
        tag_edx_subject=tag_edx_info["subject"],
        tag_edx_dataset=tag_edx_info["dataset"],
        etl_job_id=tag_edx_info["etl_job_id"]
    )
    
    # Create calibration config with the same settings but different date range
    calibration_config = create_cradle_data_load_config(
        role=role,
        region=region,
        pipeline_s3_loc=pipeline_s3_loc,
        job_type="calibration",
        mds_field_list=hyperparams.mds_field_list,
        start_date=calibration_dates["start"],
        end_date=calibration_dates["end"],
        service_name=service_name,
        tag_edx_provider=tag_edx_info["provider"],
        tag_edx_subject=tag_edx_info["subject"],
        tag_edx_dataset=tag_edx_info["dataset"],
        etl_job_id=tag_edx_info["etl_job_id"]
    )
    
    return training_config, calibration_config
```

## Implementation Plan

1. Create a new file `src/config_field_manager/cradle_config_factory.py`

2. Implement the `create_cradle_data_load_config` function as the main API

3. Add utility functions for:
   - Converting field lists to schema definitions
   - Generating standard SQL templates
   - Constructing manifest ARNs

4. Add tests in `test/config_field_manager/test_cradle_config_factory.py`

5. Add example usage in documentation and notebook templates

## Benefits

1. **Reduces complexity**: Simplifies creating Cradle configurations from 100+ lines to ~20 lines of code

2. **Prevents errors**: Ensures proper field derivation and consistency between nested objects

3. **Maintains flexibility**: Allows overriding defaults when needed while providing sensible defaults

4. **Improves maintainability**: Centralizes the complex logic for creating configurations

5. **Better documentation**: Makes the configuration requirements explicit and well-documented

## Next Steps

1. Implement the helper function as per this design

2. Create comprehensive tests to verify all aspects of the functionality

3. Update example notebooks to use the new helper function

4. Consider extending the pattern to other complex configuration classes
