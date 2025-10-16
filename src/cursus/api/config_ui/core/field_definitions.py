"""
Field definitions for configuration UI - Discovery-Based Approach with Sub-Config Organization

This module uses dynamic discovery instead of hardcoded field definitions and organizes
fields by major sub-config blocks for optimal user experience.
"""

from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


def get_cradle_fields_by_sub_config(config_core=None, _recursion_guard=None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get Cradle Data Loading fields organized by major sub-config blocks.
    
    This function organizes fields into four logical blocks that match the actual
    CradleDataLoadingConfig structure, enabling the widget to create distinct
    sections for each major specification component.
    
    Args:
        config_core: Optional UniversalConfigCore instance for discovery
        
    Returns:
        Dict with keys for each sub-config block:
        - 'inherited': Fields from BasePipelineConfig
        - 'data_sources_spec': Time range + dynamic data sources
        - 'transform_spec': SQL transformation + job splitting
        - 'output_spec': Output schema, format, and file options
        - 'cradle_job_spec': Cluster, account, and execution settings
        - 'root': Top-level fields like job_type
    """
    # Import factory field extractor directly
    from ...factory import extract_field_requirements
    
    if config_core is None:
        from .core import UniversalConfigCore
        config_core = UniversalConfigCore()
    
    # Get all discovered config classes
    all_config_classes = config_core.discover_config_classes()
    
    # Get the main config class and sub-config classes
    cradle_config_class = all_config_classes.get("CradleDataLoadingConfig")
    data_sources_spec_class = all_config_classes.get("DataSourcesSpecificationConfig")
    transform_spec_class = all_config_classes.get("TransformSpecificationConfig")
    output_spec_class = all_config_classes.get("OutputSpecificationConfig")
    cradle_job_spec_class = all_config_classes.get("CradleJobSpecificationConfig")
    
    if not cradle_config_class:
        logger.warning("CradleDataLoadingConfig not found in discovery, using fallback")
        return get_cradle_fields_by_sub_config_fallback()
    
    # Initialize the result structure
    field_blocks = {
        "inherited": [],
        "data_sources_spec": [],
        "transform_spec": [],
        "output_spec": [],
        "cradle_job_spec": [],
        "root": []
    }
    
    # Get inherited fields from BasePipelineConfig
    base_config_class = all_config_classes.get("BasePipelineConfig")
    if base_config_class:
        inherited_fields = config_core._get_form_fields(base_config_class, _recursion_guard)
        for field in inherited_fields:
            field["section"] = "inherited"
            field["tier"] = "inherited"
            field_blocks["inherited"].append(field)
    
    # Get data sources specification fields
    if data_sources_spec_class:
        ds_fields = config_core._get_form_fields(data_sources_spec_class, _recursion_guard)
        for field in ds_fields:
            # CRITICAL FIX: Skip the original data_sources field - we'll replace it with dynamic version
            if field["name"] == "data_sources":
                logger.info(f"Skipping original data_sources field (type: {field.get('type')}) - will be replaced with dynamic version")
                continue
                
            field["section"] = "data_sources_spec"
            # Set tier based on field requirements
            field["tier"] = "essential" if field.get("required", False) else "system"
            field_blocks["data_sources_spec"].append(field)
    
    # Add the special dynamic data sources field (this replaces the original data_sources field)
    dynamic_data_sources_field = {
        "name": "data_sources",
        "type": "dynamic_data_sources",
        "section": "data_sources_spec",
        "tier": "essential",
        "required": True,
        "description": "Configure one or more data sources for your job"
    }
    field_blocks["data_sources_spec"].append(dynamic_data_sources_field)
    
    # Remove any duplicate data_sources fields from ALL sections to avoid conflicts
    for section_name, fields in field_blocks.items():
        original_count = len(fields)
        field_blocks[section_name] = [f for f in fields if not (f.get("name") == "data_sources" and f.get("type") != "dynamic_data_sources")]
        removed_count = original_count - len(field_blocks[section_name])
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate data_sources fields from {section_name} section")
    
    # Get transform specification fields
    if transform_spec_class:
        transform_fields = config_core._get_form_fields(transform_spec_class, _recursion_guard)
        for field in transform_fields:
            field["section"] = "transform_spec"
            field["tier"] = "essential" if field.get("required", False) else "system"
            
            # CRITICAL FIX: Override transform_sql to be code_editor type with larger window
            if field["name"] == "transform_sql":
                field["type"] = "code_editor"
                field["language"] = "sql"
                field["height"] = "300px"
                field["description"] = "SQL transformation query to process the input data"
            
            field_blocks["transform_spec"].append(field)
    
    # Get output specification fields
    if output_spec_class:
        output_fields = config_core._get_form_fields(output_spec_class, _recursion_guard)
        for field in output_fields:
            field["section"] = "output_spec"
            field["tier"] = "essential" if field.get("required", False) else "system"
            field_blocks["output_spec"].append(field)
    
    # Get cradle job specification fields
    if cradle_job_spec_class:
        job_fields = config_core._get_form_fields(cradle_job_spec_class, _recursion_guard)
        for field in job_fields:
            field["section"] = "cradle_job_spec"
            field["tier"] = "essential" if field.get("required", False) else "system"
            field_blocks["cradle_job_spec"].append(field)
    
    # FIELD PARTITIONING: Get root-level fields using exclusion-based filtering
    if cradle_config_class:
        # Step 1: Build exclusion list of all fields that belong to sub-configs or are inherited
        excluded_field_names = set()
        
        # Add all inherited field names to exclusion list
        for inherited_field in field_blocks["inherited"]:
            excluded_field_names.add(inherited_field["name"])
        
        # Add all sub-config field names to exclusion list
        for section_name in ["data_sources_spec", "transform_spec", "output_spec", "cradle_job_spec"]:
            for field in field_blocks[section_name]:
                excluded_field_names.add(field["name"])
        
        # Add sub-config object names themselves to exclusion list
        excluded_field_names.update(["data_sources_spec", "transform_spec", "output_spec", "cradle_job_spec"])
        
        logger.info(f"Field partitioning: excluding {len(excluded_field_names)} fields from root section: {sorted(excluded_field_names)}")
        
        # Step 2: Get all fields from main config and filter out excluded ones
        main_fields = config_core._get_form_fields(cradle_config_class, _recursion_guard)
        for field in main_fields:
            field_name = field["name"]
            # CLEAN PARTITIONING: Only include fields that are truly root-level
            if field_name not in excluded_field_names:
                field["section"] = "root"
                field["tier"] = "essential" if field.get("required", False) else "system"
                field_blocks["root"].append(field)
                logger.info(f"Added root-level field: {field_name}")
            else:
                logger.debug(f"Excluded field from root section: {field_name} (belongs to sub-config or inherited)")
    
    logger.info(f"Generated field blocks using discovery-based approach:")
    for block_name, fields in field_blocks.items():
        logger.info(f"  {block_name}: {len(fields)} fields")
    
    return field_blocks


def get_cradle_fields_by_sub_config_fallback() -> Dict[str, List[Dict[str, Any]]]:
    """
    Fallback field organization if discovery fails.
    
    Returns:
        Minimal field blocks to ensure the system still works
    """
    logger.warning("Using fallback field blocks for CradleDataLoadingConfig")
    
    return {
        "inherited": [
            {"name": "author", "type": "text", "tier": "inherited", "required": True, "section": "inherited"},
            {"name": "bucket", "type": "text", "tier": "inherited", "required": True, "section": "inherited"},
            {"name": "role", "type": "text", "tier": "inherited", "required": True, "section": "inherited"},
        ],
        "data_sources_spec": [
            {"name": "start_date", "type": "datetime", "section": "data_sources_spec", "tier": "essential", "required": True},
            {"name": "end_date", "type": "datetime", "section": "data_sources_spec", "tier": "essential", "required": True},
            {"name": "data_sources", "type": "dynamic_data_sources", "section": "data_sources_spec", "tier": "essential", "required": True},
        ],
        "transform_spec": [
            {"name": "transform_sql", "type": "code_editor", "language": "sql", "section": "transform_spec", "tier": "essential", "required": True},
        ],
        "output_spec": [
            {"name": "output_schema", "type": "tag_list", "section": "output_spec", "tier": "essential", "required": True},
            {"name": "output_format", "type": "dropdown", "section": "output_spec", "tier": "system", "default": "PARQUET", "options": ["PARQUET", "CSV", "JSON"]},
        ],
        "cradle_job_spec": [
            {"name": "cradle_account", "type": "text", "section": "cradle_job_spec", "tier": "essential", "required": True}
        ],
        "root": [
            {"name": "job_type", "type": "radio", "section": "root", "tier": "essential", "required": True, 
             "options": ["training", "validation", "testing", "calibration"]},
        ]
    }


def get_sub_config_section_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for each sub-config section for widget styling and organization.
    
    Returns:
        Dict mapping section names to their display metadata
    """
    return {
        "inherited": {
            "title": "💾 Inherited Configuration",
            "description": "Configuration inherited from parent pipeline steps",
            "bg_gradient": "linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)",
            "border_color": "#9ca3af",
            "tier": "inherited",
            "collapsible": True,
            "collapsed_by_default": True,
            "icon": "💾"
        },
        "data_sources_spec": {
            "title": "📊 Data Sources Specification",
            "description": "Configure time range and data sources for your job",
            "bg_gradient": "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
            "border_color": "#f59e0b",
            "tier": "essential",
            "collapsible": False,
            "icon": "📊"
        },
        "transform_spec": {
            "title": "⚙️ Transform Specification",
            "description": "Configure SQL transformation and job splitting options",
            "bg_gradient": "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
            "border_color": "#f59e0b",
            "tier": "essential",
            "collapsible": False,
            "icon": "⚙️"
        },
        "output_spec": {
            "title": "📤 Output Specification",
            "description": "Configure output schema, format, and file options",
            "bg_gradient": "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)",
            "border_color": "#3b82f6",
            "tier": "system",
            "collapsible": True,
            "icon": "📤"
        },
        "cradle_job_spec": {
            "title": "🎛️ Cradle Job Specification",
            "description": "Configure cluster, account, and execution settings",
            "bg_gradient": "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)",
            "border_color": "#3b82f6",
            "tier": "system",
            "collapsible": True,
            "icon": "🎛️"
        },
        "root": {
            "title": "🎯 Job Configuration",
            "description": "Select job type and advanced options",
            "bg_gradient": "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
            "border_color": "#f59e0b",
            "tier": "essential",
            "collapsible": False,
            "icon": "🎯"
        }
    }


# DEPRECATED: Legacy functions kept for backward compatibility
def get_cradle_data_loading_fields() -> List[Dict[str, Any]]:
    """
    DEPRECATED: Use get_cradle_fields_by_sub_config() instead.
    
    This function is kept for backward compatibility but will be removed.
    """
    logger.warning("get_cradle_data_loading_fields() is deprecated, use get_cradle_fields_by_sub_config()")
    
    # Convert new format to old format for backward compatibility
    field_blocks = get_cradle_fields_by_sub_config()
    all_fields = []
    
    for block_name, fields in field_blocks.items():
        all_fields.extend(fields)
    
    return all_fields


def get_legacy_cradle_data_loading_fields() -> List[Dict[str, Any]]:
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
