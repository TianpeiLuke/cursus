"""Utilities module for Cradle Data Load Config UI."""

from ...factory import (
    extract_field_requirements as extract_field_schema,
)

def get_data_source_variant_schemas():
    """
    Get schemas for all data source variants using factory field extraction.
    
    Returns:
        Dict mapping data source type to schema information
    """
    try:
        # Import the data source config classes
        from ....steps.configs.config_cradle_data_loading_step import (
            MdsDataSourceConfig,
            EdxDataSourceConfig,
            AndesDataSourceConfig
        )
        
        return {
            "MDS": {"fields": extract_field_schema(MdsDataSourceConfig)},
            "EDX": {"fields": extract_field_schema(EdxDataSourceConfig)},
            "ANDES": {"fields": extract_field_schema(AndesDataSourceConfig)}
        }
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting data source variant schemas: {e}")
        return {}

def get_all_config_schemas():
    """
    Get schemas for all configuration classes using factory field extraction.
    
    Returns:
        Dict mapping config class name to schema information
    """
    try:
        from ....steps.configs.config_cradle_data_loading_step import (
            MdsDataSourceConfig,
            EdxDataSourceConfig,
            AndesDataSourceConfig,
            DataSourceConfig,
            DataSourcesSpecificationConfig,
            TransformSpecificationConfig,
            JobSplitOptionsConfig,
            OutputSpecificationConfig,
            CradleJobSpecificationConfig,
            CradleDataLoadingConfig
        )
        
        config_classes = {
            "MdsDataSourceConfig": MdsDataSourceConfig,
            "EdxDataSourceConfig": EdxDataSourceConfig,
            "AndesDataSourceConfig": AndesDataSourceConfig,
            "DataSourceConfig": DataSourceConfig,
            "DataSourcesSpecificationConfig": DataSourcesSpecificationConfig,
            "JobSplitOptionsConfig": JobSplitOptionsConfig,
            "TransformSpecificationConfig": TransformSpecificationConfig,
            "OutputSpecificationConfig": OutputSpecificationConfig,
            "CradleJobSpecificationConfig": CradleJobSpecificationConfig,
            "CradleDataLoadingConfig": CradleDataLoadingConfig
        }
        
        schemas = {}
        for name, config_class in config_classes.items():
            try:
                schemas[name] = {"fields": extract_field_schema(config_class)}
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error extracting schema for {name}: {e}")
                schemas[name] = {"fields": []}
        
        return schemas
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting all config schemas: {e}")
        return {}

def get_field_defaults():
    """
    Get default values for common configuration fields.
    
    Returns:
        Dict containing default values organized by section
    """
    return {
        "mds": {
            "region": "NA",
            "org_id": 0,
            "use_hourly_edx_data_set": False
        },
        "edx": {
            "edx_manifest_key": '[""]'
        },
        "andes": {
            "andes3_enabled": True
        },
        "jobSplitOptions": {
            "split_job": False,
            "days_per_split": 7,
            "merge_sql": ""
        },
        "output": {
            "output_format": "PARQUET",
            "output_save_mode": "ERRORIFEXISTS",
            "output_file_count": 0,
            "keep_dot_in_output_schema": False,
            "include_header_in_s3_output": True
        },
        "cradleJob": {
            "cluster_type": "STANDARD",
            "extra_spark_job_arguments": "",
            "job_retry_count": 1
        }
    }

def get_field_validation_rules():
    """
    Get validation rules for common fields.
    
    Returns:
        Dict containing validation rules organized by field
    """
    return {
        "region": {
            "enum": ["NA", "EU", "FE"],
            "message": "Region must be one of: NA, EU, FE"
        },
        "data_source_type": {
            "enum": ["MDS", "EDX", "ANDES"],
            "message": "Data source type must be one of: MDS, EDX, ANDES"
        },
        "output_format": {
            "enum": ["CSV", "UNESCAPED_TSV", "JSON", "ION", "PARQUET"],
            "message": "Output format must be one of: CSV, UNESCAPED_TSV, JSON, ION, PARQUET"
        },
        "output_save_mode": {
            "enum": ["ERRORIFEXISTS", "OVERWRITE", "APPEND", "IGNORE"],
            "message": "Save mode must be one of: ERRORIFEXISTS, OVERWRITE, APPEND, IGNORE"
        },
        "cluster_type": {
            "enum": ["STANDARD", "SMALL", "MEDIUM", "LARGE"],
            "message": "Cluster type must be one of: STANDARD, SMALL, MEDIUM, LARGE"
        },
        "job_type": {
            "enum": ["training", "validation", "testing", "calibration"],
            "message": "Job type must be one of: training, validation, testing, calibration"
        },
        "datetime": {
            "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
            "message": "Date must be in format YYYY-MM-DDTHH:MM:SS"
        }
    }

__all__ = [
    "extract_field_schema",
    "get_data_source_variant_schemas", 
    "get_all_config_schemas",
    "get_field_defaults",
    "get_field_validation_rules"
]
