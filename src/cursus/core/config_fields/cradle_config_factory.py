"""
Helper functions for creating CradleDataLoadConfig objects with minimal inputs.

This module provides utilities to simplify the creation of complex CradleDataLoadConfig
objects by deriving nested configurations from essential user inputs.

Author: Luke Xie
Date: July 26, 2025
"""

from typing import List, Dict, Any, Optional, Union
import uuid
import os
from pathlib import Path

# Import base classes to ensure essential classes are always available
try:
    from ..base.config_base import BasePipelineConfig
except ImportError:
    # For testing environments or when running outside the main project
    BasePipelineConfig = None

from ...steps.configs.config_cradle_data_loading_step import (
    CradleDataLoadConfig,
    MdsDataSourceConfig,
    EdxDataSourceConfig,
    DataSourceConfig,
    DataSourcesSpecificationConfig,
    JobSplitOptionsConfig,
    TransformSpecificationConfig,
    OutputSpecificationConfig,
    CradleJobSpecificationConfig
)

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


def _create_field_schema(fields: List[str]) -> List[Dict[str, str]]:
    """
    Convert a list of field names to schema dictionaries.
    
    Args:
        fields (List[str]): List of field names
        
    Returns:
        List[Dict[str, str]]: List of schema dictionaries
    """
    return [{'field_name': field, 'field_type': 'STRING'} for field in fields]


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
    
    Args:
        provider (str): EDX provider name
        subject (str): EDX subject
        dataset (str): EDX dataset name
        etl_job_id (str): ETL job ID
        start_date (str): Start date string
        end_date (str): End date string
        comment (Optional[str]): Optional comment or region code
        
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


def _create_edx_manifest_from_key(
    provider: str,
    subject: str,
    dataset: str,
    manifest_key: str
) -> str:
    """
    Create an EDX manifest ARN from a provided manifest key.
    
    Args:
        provider (str): EDX provider name
        subject (str): EDX subject
        dataset (str): EDX dataset name
        manifest_key (str): The complete manifest key portion (e.g., '["xxx",...]')
        
    Returns:
        str: Properly formatted EDX manifest ARN
    """
    return (
        f'arn:amazon:edx:iad::manifest/'
        f'{provider}/{subject}/{dataset}/{manifest_key}'
    )


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
    
    Args:
        mds_source_name (str): Logical name for MDS source
        edx_source_name (str): Logical name for EDX source
        mds_field_list (List[str]): List of fields from MDS
        tag_schema (List[str]): List of fields from EDX tags
        mds_join_key (str): Join key field name from MDS
        edx_join_key (str): Join key field name from EDX
        join_type (str): SQL join type (JOIN, LEFT JOIN, etc.)
        use_dedup_sql (bool): Whether to use the deduplication SQL format with subquery
        
    Returns:
        str: SQL query string
    """
    if not use_dedup_sql:
        # Standard format (original behavior)
        # Build the select column list
        select_variable_text_list = []
        
        # Track lowercase field names to detect duplicates
        added_fields = {}
        
        # Get lowercase versions of join keys for case-insensitive comparison
        mds_join_key_lower = mds_join_key.lower()
        edx_join_key_lower = edx_join_key.lower()
        
        # Always include both join keys to ensure the JOIN operation works
        # First add the MDS join key (replacing dots if needed)
        mds_join_field_dot_replaced = mds_join_key.replace('.', '__DOT__')
        select_variable_text_list.append(f'{mds_source_name}.{mds_join_field_dot_replaced} as {mds_source_name}_{mds_join_key}')
        added_fields[mds_join_key_lower] = True
        
        # Then add the EDX join key (we'll always include this from EDX)
        select_variable_text_list.append(f'{edx_source_name}.{edx_join_key} as {edx_source_name}_{edx_join_key}')
        # Mark both join keys as added (using lowercase for case-insensitive tracking)
        added_fields[edx_join_key_lower] = True
        
        # Add MDS fields, replacing dots with __DOT__
        for field in mds_field_list:
            # Skip the join key as we've already added it
            field_lower = field.lower()
            if field_lower == mds_join_key_lower:
                continue
                
            field_dot_replaced = field.replace('.', '__DOT__')
            
            # Add to our select list and track that we've seen this field
            select_variable_text_list.append(f'{mds_source_name}.{field_dot_replaced}')
            added_fields[field_lower] = True
        
        # Add tag fields, skipping any that have already been added from MDS
        # and also skipping the EDX join key which we've already explicitly added
        for var in tag_schema:
            var_lower = var.lower()
            
            # Skip fields we've already added (including join keys)
            if var_lower in added_fields:
                continue
            
            select_variable_text_list.append(f'{edx_source_name}.{var}')
            added_fields[var_lower] = True
        
        # Join into a comma-separated list
        schema_list = ',\n'.join(select_variable_text_list)
        
        # Create the final SQL
        transform_sql = f"""
SELECT
{schema_list}
FROM {mds_source_name}
{join_type} {edx_source_name} 
ON {mds_source_name}.{mds_join_key}={edx_source_name}.{edx_join_key}
"""
    else:
        # Deduplication format with subquery and ROW_NUMBER()
        # Track all unique fields by lowercase name for outer query
        all_fields = []
        added_fields_lower = {}
        
        # Get lowercase versions of join keys for case-insensitive comparison
        mds_join_key_lower = mds_join_key.lower()
        edx_join_key_lower = edx_join_key.lower()
        
        # First collect MDS fields
        for field in mds_field_list:
            field_lower = field.lower()
            # Include all fields (even join key) as we'll need them in the subquery
            if field_lower not in added_fields_lower:
                field_dot_replaced = field.replace('.', '__DOT__')
                all_fields.append((field_dot_replaced, f"{mds_source_name}.{field_dot_replaced}", field_lower))
                added_fields_lower[field_lower] = True
        
        # Then collect tag fields
        for var in tag_schema:
            var_lower = var.lower()
            if var_lower not in added_fields_lower:
                all_fields.append((var, f"{edx_source_name}.{var}", var_lower))
                added_fields_lower[var_lower] = True
        
        # Build the outer query field list (without source prefixes)
        outer_select_fields = []
        for field_name, _, _ in all_fields:
            outer_select_fields.append(field_name)
        
        # Build the inner query field list (with source prefixes)
        inner_select_fields = []
        for _, field_with_prefix, _ in all_fields:
            inner_select_fields.append(field_with_prefix)
            
        # Join into comma-separated lists
        outer_field_list = ',\n    '.join(sorted(outer_select_fields))
        inner_field_list = ',\n        '.join(sorted(inner_select_fields))
        
        # Create the deduplication SQL with subquery
        transform_sql = f"""
SELECT
    {outer_field_list}
FROM (
    SELECT
        {inner_field_list},
        ROW_NUMBER() OVER (PARTITION BY {mds_source_name}.{mds_join_key}, {edx_source_name}.{edx_join_key} ORDER BY {mds_source_name}.transactionDate DESC) as row_num
    FROM {mds_source_name}
    {join_type} {edx_source_name} ON {mds_source_name}.{mds_join_key} = {edx_source_name}.{edx_join_key}
)
WHERE row_num = 1
"""
    
    return transform_sql


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
        current_date (str, optional): Current date string for metadata
        
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
    # 1. Derive values and set defaults
    
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
    
    # 2. Create MDS Data Source Config
    
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
    
    # 3. Create EDX Data Source Config
    
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
    
    # 4. Create Data Source Configs
    
    # MDS data source
    mds_data_source = DataSourceConfig(
        data_source_name=f"RAW_MDS_{region}",
        data_source_type="MDS",
        mds_data_source_properties=mds_data_source_inner_config
    )
    
    # EDX data source
    edx_data_source = DataSourceConfig(
        data_source_name="TAGS",
        data_source_type="EDX",
        edx_data_source_properties=edx_source_inner_config
    )
    
    # 5. Create Data Sources Specification
    
    data_sources_spec = DataSourcesSpecificationConfig(
        start_date=start_date,
        end_date=end_date,
        data_sources=[mds_data_source, edx_data_source]
    )
    
    # 6. Create Job Split Options
    
    job_split_options = JobSplitOptionsConfig(
        split_job=split_job,
        days_per_split=days_per_split,
        merge_sql=merge_sql
    )
    
    # 7. Create Transform Specification
    
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
    
    transform_spec = TransformSpecificationConfig(
        transform_sql=transform_sql,
        job_split_options=job_split_options
    )
    
    # 8. Create Output Specification
    
    # Combine all fields from both sources
    output_fields = _get_all_fields(complete_mds_field_list, tag_schema)
    
    # Create the output spec - let job_type drive output_path derivation
    output_spec = OutputSpecificationConfig(
        output_schema=output_fields,
        job_type=job_type,  # Required field that will determine the output path
        output_format=output_format,
        output_save_mode=output_save_mode,
        keep_dot_in_output_schema=False,
        include_header_in_s3_output=True,
        pipeline_s3_loc=base_config.pipeline_s3_loc  # Pass the pipeline_s3_loc for output_path derivation
    )
    
    # 9. Create Cradle Job Specification
    
    cradle_job_spec = CradleJobSpecificationConfig(
        cluster_type=cluster_type,
        cradle_account=cradle_account,
        job_retry_count=4  # Default to 4 retries
    )
    
    # 10. Create the final CradleDataLoadConfig using from_base_config
    
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
    
    return cradle_data_load_config


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
    
    # MDS data source (if not in base_config)
    service_name: Optional[str] = None,
    
    # EDX manifest configuration
    edx_manifest_comment: Optional[str] = None,  # Optional comment for EDX manifest key
    
    # Optional shared configuration
    cradle_account: str = "Buyer-Abuse-RnD-Dev",
    cluster_type: str = "STANDARD",
    output_format: str = "PARQUET",
    output_save_mode: str = "ERRORIFEXISTS",
    split_job: bool = False,
    days_per_split: int = 7,
    merge_sql: Optional[str] = None,
    transform_sql: Optional[str] = None,
    tag_schema: Optional[List[str]] = None
) -> Dict[str, CradleDataLoadConfig]:
    """
    Create both training and calibration CradleDataLoadConfig objects with consistent settings.
    
    Args:
        role (str): IAM role to use for the pipeline
        region (str): Marketplace region ('NA', 'EU', 'FE')
        pipeline_s3_loc (str): S3 location for pipeline artifacts
        
        mds_field_list (List[str]): List of fields to include from MDS
        
        service_name (str): Name of the MDS service
        tag_edx_provider (str): EDX provider for tags
        tag_edx_subject (str): EDX subject for tags
        tag_edx_dataset (str): EDX dataset for tags
        etl_job_id (str): ETL job ID for the EDX manifest
        
        training_start_date (str): Training data start date
        training_end_date (str): Training data end date
        calibration_start_date (str): Calibration data start date
        calibration_end_date (str): Calibration data end date
        
        cradle_account (str): Cradle account name (default: "Buyer-Abuse-RnD-Dev")
        aws_region (str, optional): AWS region, derived from region if not provided
        current_date (str, optional): Current date string for metadata
        cluster_type (str): Cradle cluster type (default: "STANDARD")
        output_format (str): Output format (default: "PARQUET")
        output_save_mode (str): Output save mode (default: "ERRORIFEXISTS")
        split_job (bool): Whether to split the job (default: False)
        days_per_split (int): Days per split if splitting (default: 7)
        merge_sql (str, optional): SQL to merge split results
        transform_sql (str, optional): Custom transform SQL
        tag_schema (List[str], optional): Schema for tag data
        
    Returns:
        Dict[str, CradleDataLoadConfig]: Dictionary with 'training' and 'calibration' configs
    """
    # Create training config using the base_config
    training_config = create_cradle_data_load_config(
        base_config=base_config,
        job_type="training",
        mds_field_list=mds_field_list,
        start_date=training_start_date,
        end_date=training_end_date,
        service_name=service_name,
        tag_edx_provider=tag_edx_provider,
        tag_edx_subject=tag_edx_subject,
        tag_edx_dataset=tag_edx_dataset,
        etl_job_id=etl_job_id,
        edx_manifest_comment=edx_manifest_comment,
        cradle_account=cradle_account,
        cluster_type=cluster_type,
        output_format=output_format,
        output_save_mode=output_save_mode,
        split_job=split_job,
        days_per_split=days_per_split,
        merge_sql=merge_sql,
        transform_sql=transform_sql,
        tag_schema=tag_schema
    )
    
    # Create calibration config using the base_config
    calibration_config = create_cradle_data_load_config(
        base_config=base_config,
        job_type="calibration",
        mds_field_list=mds_field_list,
        start_date=calibration_start_date,
        end_date=calibration_end_date,
        service_name=service_name,
        tag_edx_provider=tag_edx_provider,
        tag_edx_subject=tag_edx_subject,
        tag_edx_dataset=tag_edx_dataset,
        etl_job_id=etl_job_id,
        edx_manifest_comment=edx_manifest_comment,
        cradle_account=cradle_account,
        cluster_type=cluster_type,
        output_format=output_format,
        output_save_mode=output_save_mode,
        split_job=split_job,
        days_per_split=days_per_split,
        merge_sql=merge_sql,
        transform_sql=transform_sql,
        tag_schema=tag_schema
    )
    
    return {
        "training": training_config,
        "calibration": calibration_config
    }
