"""
Comprehensive field definitions for specialized configurations.

This module provides detailed field definitions for configuration classes that require
custom field layouts and enhanced UI components beyond standard Pydantic field discovery.
"""

from typing import List, Dict, Any


def get_cradle_data_loading_fields() -> List[Dict[str, Any]]:
    """
    Get comprehensive field definition for CradleDataLoadingConfig single-page form.
    
    This function provides the complete field structure for the Cradle Data Loading
    configuration, organized by tiers and sections for optimal user experience.
    
    Based on analysis of src/cursus/steps/configs/config_cradle_data_loading_step.py,
    this creates field definitions for the complete 5-level hierarchical structure:
    
    LEVEL 1: CradleDataLoadingConfig (Root)
    LEVEL 3: Specification Components (DataSourcesSpecificationConfig, etc.)
    LEVEL 4: DataSourceConfig (wrapper)
    LEVEL 5: Leaf Components (MdsDataSourceConfig, EdxDataSourceConfig, AndesDataSourceConfig)
    
    Returns:
        List of field definitions with comprehensive metadata
    """
    return [
        # ========================================
        # INHERITED FIELDS (Tier 3) - Auto-filled from parent configs
        # ========================================
        {"name": "author", "type": "text", "tier": "inherited", "required": True,
         "description": "Author of the pipeline configuration"},
        {"name": "bucket", "type": "text", "tier": "inherited", "required": True,
         "description": "S3 bucket for pipeline artifacts and outputs"},
        {"name": "role", "type": "text", "tier": "inherited", "required": True,
         "description": "IAM role ARN for pipeline execution"},
        {"name": "region", "type": "dropdown", "options": ["NA", "EU", "FE"], "tier": "inherited",
         "default": "NA", "description": "Geographic region for data processing"},
        {"name": "service_name", "type": "text", "tier": "inherited", "required": True,
         "description": "Service name for the pipeline"},
        {"name": "pipeline_version", "type": "text", "tier": "inherited", "required": True,
         "default": "1.0.0", "description": "Version of the pipeline configuration"},
        {"name": "project_root_folder", "type": "text", "tier": "inherited", "required": True,
         "description": "Root folder path for project artifacts"},
        
        # ========================================
        # DATA SOURCES FIELDS (Tier 1 - Essential)
        # ========================================
        {"name": "start_date", "type": "datetime", "tier": "essential", "required": True,
         "placeholder": "YYYY-MM-DDTHH:MM:SS", "default": "2025-01-01T00:00:00",
         "description": "Start date for data loading (inclusive)"},
        {"name": "end_date", "type": "datetime", "tier": "essential", "required": True,
         "placeholder": "YYYY-MM-DDTHH:MM:SS", "default": "2025-04-17T00:00:00",
         "description": "End date for data loading (exclusive)"},
        {"name": "data_source_name", "type": "text", "tier": "essential", "required": True,
         "default": "RAW_MDS_NA", "description": "Unique name identifier for the data source"},
        {"name": "data_source_type", "type": "dropdown", "options": ["MDS", "EDX", "ANDES"], 
         "tier": "essential", "default": "MDS", "required": True,
         "description": "Type of data source to configure"},
        
        # ========================================
        # MDS-SPECIFIC FIELDS (Tier 1 - Essential, conditional on data_source_type=="MDS")
        # ========================================
        {"name": "mds_service", "type": "text", "tier": "essential", "conditional": "data_source_type==MDS",
         "default": "AtoZ", "required": True, "description": "MDS service name (e.g., AtoZ, PDA)"},
        {"name": "mds_region", "type": "dropdown", "options": ["NA", "EU", "FE"], 
         "tier": "essential", "conditional": "data_source_type==MDS", "default": "NA", "required": True,
         "description": "MDS region for data source"},
        {"name": "mds_output_schema", "type": "tag_list", "tier": "essential", "conditional": "data_source_type==MDS",
         "default": ["objectId", "transactionDate"], "required": True,
         "description": "List of field names to include in MDS output schema"},
        {"name": "mds_org_id", "type": "number", "tier": "system", "conditional": "data_source_type==MDS",
         "default": 0, "description": "Organization ID (integer) for MDS. Default 0 for regional MDS bucket"},
        {"name": "mds_use_hourly", "type": "checkbox", "tier": "system", "conditional": "data_source_type==MDS",
         "default": False, "description": "Whether to use the hourly EDX dataset flag in MDS"},
        
        # ========================================
        # EDX-SPECIFIC FIELDS (Tier 1 - Essential, conditional on data_source_type=="EDX")
        # ========================================
        {"name": "edx_provider", "type": "text", "tier": "essential", "conditional": "data_source_type==EDX",
         "required": True, "description": "Provider portion of the EDX manifest ARN"},
        {"name": "edx_subject", "type": "text", "tier": "essential", "conditional": "data_source_type==EDX",
         "required": True, "description": "Subject portion of the EDX manifest ARN"},
        {"name": "edx_dataset", "type": "text", "tier": "essential", "conditional": "data_source_type==EDX",
         "required": True, "description": "Dataset portion of the EDX manifest ARN"},
        {"name": "edx_manifest_key", "type": "text", "tier": "essential", "conditional": "data_source_type==EDX",
         "placeholder": '["xxx",...]', "required": True,
         "description": "Manifest key in format '[\"xxx\",...] that completes the ARN"},
        {"name": "edx_schema_overrides", "type": "tag_list", "tier": "essential", "conditional": "data_source_type==EDX",
         "default": [], "description": "List of dicts overriding the EDX schema"},
        
        # ========================================
        # ANDES-SPECIFIC FIELDS (Tier 1 - Essential, conditional on data_source_type=="ANDES")
        # ========================================
        {"name": "andes_provider", "type": "text", "tier": "essential", "conditional": "data_source_type==ANDES",
         "required": True, "description": "Andes provider ID (32-digit UUID or 'booker')"},
        {"name": "andes_table_name", "type": "text", "tier": "essential", "conditional": "data_source_type==ANDES",
         "required": True, "description": "Name of the Andes table to query"},
        {"name": "andes3_enabled", "type": "checkbox", "tier": "system", "conditional": "data_source_type==ANDES",
         "default": True, "description": "Whether the table uses Andes 3.0 with latest version"},
        
        # ========================================
        # TRANSFORM FIELDS (Tier 1 - Essential)
        # ========================================
        {"name": "transform_sql", "type": "code_editor", "language": "sql", "tier": "essential", "required": True,
         "height": "200px", "default": "SELECT * FROM input_data",
         "description": "SQL transformation query to process the input data"},
        {"name": "split_job", "type": "checkbox", "tier": "system", "default": False,
         "description": "Enable job splitting for large datasets to improve performance"},
        {"name": "days_per_split", "type": "number", "tier": "system", "default": 7,
         "conditional": "split_job==True", "description": "Number of days per split when job splitting is enabled"},
        {"name": "merge_sql", "type": "textarea", "tier": "essential", "default": "SELECT * FROM INPUT",
         "conditional": "split_job==True", "required": True,
         "description": "SQL query for merging split job results"},
        
        # ========================================
        # OUTPUT FIELDS (Tier 2 - System)
        # ========================================
        {"name": "output_schema", "type": "tag_list", "tier": "essential", "required": True,
         "default": ["objectId", "transactionDate", "is_abuse"],
         "description": "List of field names to include in the final output schema"},
        {"name": "output_format", "type": "dropdown", "tier": "system", "default": "PARQUET",
         "options": ["PARQUET", "CSV", "JSON", "ION", "UNESCAPED_TSV"],
         "description": "Output file format for the processed data"},
        {"name": "output_save_mode", "type": "dropdown", "tier": "system", "default": "ERRORIFEXISTS",
         "options": ["ERRORIFEXISTS", "OVERWRITE", "APPEND", "IGNORE"],
         "description": "Save mode behavior when output already exists"},
        {"name": "output_file_count", "type": "number", "tier": "system", "default": 0,
         "description": "Number of output files to create (0 = auto-split based on data size)"},
        {"name": "keep_dot_in_output_schema", "type": "checkbox", "tier": "system", "default": False,
         "description": "Keep dots in output schema field names (affects column naming)"},
        {"name": "include_header_in_s3_output", "type": "checkbox", "tier": "system", "default": True,
         "description": "Include header row in S3 output files"},
        
        # ========================================
        # JOB CONFIGURATION FIELDS (Tier 2 - System)
        # ========================================
        {"name": "cradle_account", "type": "text", "tier": "essential", "required": True,
         "default": "Buyer-Abuse-RnD-Dev", "description": "Cradle account name for job execution"},
        {"name": "cluster_type", "type": "dropdown", "tier": "system", "default": "STANDARD",
         "options": ["STANDARD", "SMALL", "MEDIUM", "LARGE"],
         "description": "Cluster size type for job execution"},
        {"name": "job_retry_count", "type": "number", "tier": "system", "default": 1,
         "description": "Number of retries for failed jobs"},
        {"name": "extra_spark_job_arguments", "type": "textarea", "tier": "system", "default": "",
         "description": "Additional Spark job arguments (advanced users only)"},
        
        # ========================================
        # JOB TYPE FIELD (Tier 1 - Essential)
        # ========================================
        {"name": "job_type", "type": "radio", "tier": "essential", "required": True,
         "options": ["training", "validation", "testing", "calibration"], "default": "training",
         "description": "Type of job to execute (affects output paths and processing)"},
        
        # ========================================
        # ADVANCED SYSTEM FIELDS (Tier 2 - System)
        # ========================================
        {"name": "s3_input_override", "type": "text", "tier": "system", "default": None,
         "description": "If set, skip Cradle data pull and use this S3 prefix directly (advanced)"}
    ]


def get_field_sections() -> List[Dict[str, Any]]:
    """
    Get field section definitions for organizing the single-page form.
    
    Returns:
        List of section definitions with styling and organization metadata
    """
    return [
        {
            "title": "💾 Inherited Configuration (Tier 3)",
            "description": "Configuration inherited from parent pipeline steps",
            "bg_gradient": "linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)",
            "border_color": "#9ca3af",
            "tier": "inherited",
            "collapsible": True,
            "collapsed_by_default": True
        },
        {
            "title": "🔥 Data Sources Configuration (Tier 1)",
            "description": "Configure time range and data sources for your job",
            "bg_gradient": "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
            "border_color": "#f59e0b",
            "tier": "essential",
            "collapsible": False
        },
        {
            "title": "⚙️ Transform Configuration (Tier 1)",
            "description": "Configure SQL transformation and job splitting options",
            "bg_gradient": "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
            "border_color": "#f59e0b",
            "tier": "essential",
            "collapsible": False
        },
        {
            "title": "📊 Output Configuration (Tier 2)",
            "description": "Configure output schema and format options",
            "bg_gradient": "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)",
            "border_color": "#3b82f6",
            "tier": "system",
            "collapsible": True
        },
        {
            "title": "🎛️ Job Configuration (Tier 2)",
            "description": "Configure cluster and job execution settings",
            "bg_gradient": "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)",
            "border_color": "#3b82f6",
            "tier": "system",
            "collapsible": True
        },
        {
            "title": "🎯 Job Type Selection (Tier 1)",
            "description": "Select the job type for this configuration",
            "bg_gradient": "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
            "border_color": "#f59e0b",
            "tier": "essential",
            "collapsible": False
        }
    ]


def get_field_validation_rules() -> Dict[str, Dict[str, Any]]:
    """
    Get validation rules for Cradle Data Loading Config fields.
    
    Returns:
        Dictionary mapping field names to validation rules
    """
    return {
        "start_date": {
            "format": "YYYY-MM-DDTHH:MM:SS",
            "required": True,
            "validation_message": "Start date must be in format YYYY-MM-DDTHH:MM:SS"
        },
        "end_date": {
            "format": "YYYY-MM-DDTHH:MM:SS",
            "required": True,
            "validation_message": "End date must be in format YYYY-MM-DDTHH:MM:SS",
            "depends_on": "start_date",
            "validation_rule": "must_be_after_start_date"
        },
        "data_source_name": {
            "required": True,
            "min_length": 1,
            "validation_message": "Data source name cannot be empty"
        },
        "data_source_type": {
            "required": True,
            "options": ["MDS", "EDX", "ANDES"],
            "validation_message": "Must select a valid data source type"
        },
        "transform_sql": {
            "required": True,
            "min_length": 10,
            "validation_message": "Transform SQL must be at least 10 characters"
        },
        "output_schema": {
            "required": True,
            "min_items": 1,
            "validation_message": "At least one output field is required"
        },
        "cradle_account": {
            "required": True,
            "min_length": 1,
            "validation_message": "Cradle account cannot be empty"
        },
        "job_type": {
            "required": True,
            "options": ["training", "validation", "testing", "calibration"],
            "validation_message": "Must select a valid job type"
        }
    }


def get_conditional_field_rules() -> Dict[str, List[str]]:
    """
    Get conditional field display rules.
    
    Returns:
        Dictionary mapping condition values to lists of dependent field names
    """
    return {
        "data_source_type==MDS": [
            "mds_service", "mds_region", "mds_output_schema", "mds_org_id", "mds_use_hourly"
        ],
        "data_source_type==EDX": [
            "edx_provider", "edx_subject", "edx_dataset", "edx_manifest_key", "edx_schema_overrides"
        ],
        "data_source_type==ANDES": [
            "andes_provider", "andes_table_name", "andes3_enabled"
        ],
        "split_job==True": [
            "days_per_split", "merge_sql"
        ]
    }


def get_field_defaults_by_context() -> Dict[str, Dict[str, Any]]:
    """
    Get context-specific field defaults.
    
    Returns:
        Dictionary mapping context types to field defaults
    """
    return {
        "training": {
            "job_type": "training",
            "output_schema": ["objectId", "transactionDate", "is_abuse"],
            "transform_sql": "SELECT objectId, transactionDate, is_abuse FROM input_data WHERE is_abuse IS NOT NULL"
        },
        "validation": {
            "job_type": "validation",
            "output_schema": ["objectId", "transactionDate", "prediction_score"],
            "transform_sql": "SELECT objectId, transactionDate, prediction_score FROM input_data"
        },
        "testing": {
            "job_type": "testing",
            "output_schema": ["objectId", "transactionDate", "test_result"],
            "transform_sql": "SELECT objectId, transactionDate, test_result FROM input_data"
        },
        "calibration": {
            "job_type": "calibration",
            "output_schema": ["objectId", "transactionDate", "calibrated_score"],
            "transform_sql": "SELECT objectId, transactionDate, calibrated_score FROM input_data"
        }
    }
