"""
Field extraction utilities for Cradle Data Load Config UI

This module provides utilities to extract field schema information from
Pydantic configuration classes for dynamic form generation.
"""

from typing import Dict, List, Any, Type, get_origin, get_args
import logging
from pydantic import BaseModel
from pydantic.fields import FieldInfo

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

logger = logging.getLogger(__name__)


def extract_field_schema(config_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Extract field schema from Pydantic config class for UI generation.
    
    Args:
        config_class: Pydantic model class to extract schema from
        
    Returns:
        Dict containing field information and categories
    """
    schema = {
        "fields": {},
        "categories": {}
    }
    
    try:
        # Get field categories using the three-tier system
        if hasattr(config_class, 'categorize_fields'):
            # Create a dummy instance to call categorize_fields
            try:
                # Try to create instance with minimal required fields
                dummy_instance = _create_dummy_instance(config_class)
                categories = dummy_instance.categorize_fields()
                schema["categories"] = categories
            except Exception as e:
                logger.warning(f"Could not create dummy instance for {config_class.__name__}: {e}")
                schema["categories"] = {"essential": [], "system": [], "derived": []}
        else:
            schema["categories"] = {"essential": [], "system": [], "derived": []}
        
        # Extract field information from Pydantic model
        for field_name, field_info in config_class.model_fields.items():
            schema["fields"][field_name] = {
                "type": _get_field_type_string(field_info.annotation),
                "required": field_info.is_required(),
                "default": _get_field_default(field_info),
                "description": field_info.description,
                "validation": _extract_validation_rules(field_info)
            }
    
    except Exception as e:
        logger.error(f"Error extracting schema for {config_class.__name__}: {e}")
        # Return minimal schema on error
        schema = {
            "fields": {},
            "categories": {"essential": [], "system": [], "derived": []}
        }
    
    return schema


def get_data_source_variant_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Get schemas for all data source variants.
    
    Returns:
        Dict mapping data source type to schema information
    """
    try:
        return {
            "MDS": extract_field_schema(MdsDataSourceConfig),
            "EDX": extract_field_schema(EdxDataSourceConfig),
            "ANDES": extract_field_schema(AndesDataSourceConfig)
        }
    except Exception as e:
        logger.error(f"Error getting data source variant schemas: {e}")
        return {}


def get_all_config_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Get schemas for all configuration classes.
    
    Returns:
        Dict mapping config class name to schema information
    """
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
            schemas[name] = extract_field_schema(config_class)
        except Exception as e:
            logger.error(f"Error extracting schema for {name}: {e}")
            schemas[name] = {
                "fields": {},
                "categories": {"essential": [], "system": [], "derived": []}
            }
    
    return schemas


def _create_dummy_instance(config_class: Type[BaseModel]) -> BaseModel:
    """
    Create a dummy instance of a config class with minimal required fields.
    
    Args:
        config_class: Pydantic model class
        
    Returns:
        Dummy instance of the class
    """
    # Get required fields and provide minimal values
    dummy_data = {}
    
    for field_name, field_info in config_class.model_fields.items():
        if field_info.is_required():
            field_type = field_info.annotation
            dummy_data[field_name] = _get_dummy_value_for_type(field_type)
    
    return config_class(**dummy_data)


def _get_dummy_value_for_type(field_type: Type) -> Any:
    """
    Get a dummy value for a given type.
    
    Args:
        field_type: Type to get dummy value for
        
    Returns:
        Dummy value of the appropriate type
    """
    # Handle common types
    if field_type == str:
        return "dummy"
    elif field_type == int:
        return 0
    elif field_type == float:
        return 0.0
    elif field_type == bool:
        return False
    elif field_type == list or get_origin(field_type) == list:
        return []
    elif field_type == dict or get_origin(field_type) == dict:
        return {}
    else:
        # For complex types, try to return None or empty value
        return None


def _get_field_type_string(field_type: Type) -> str:
    """
    Convert a field type to a string representation.
    
    Args:
        field_type: Field type annotation
        
    Returns:
        String representation of the type
    """
    try:
        # Handle generic types (List, Dict, Optional, etc.)
        origin = get_origin(field_type)
        if origin is not None:
            args = get_args(field_type)
            if origin == list:
                if args:
                    return f"List[{_get_field_type_string(args[0])}]"
                else:
                    return "List"
            elif origin == dict:
                if len(args) >= 2:
                    return f"Dict[{_get_field_type_string(args[0])}, {_get_field_type_string(args[1])}]"
                else:
                    return "Dict"
            elif origin == type(None) or str(origin) == "typing.Union":
                # Handle Optional types (Union[X, None])
                non_none_args = [arg for arg in args if arg != type(None)]
                if len(non_none_args) == 1:
                    return f"Optional[{_get_field_type_string(non_none_args[0])}]"
                else:
                    return f"Union[{', '.join(_get_field_type_string(arg) for arg in args)}]"
        
        # Handle basic types
        if hasattr(field_type, '__name__'):
            return field_type.__name__
        else:
            return str(field_type)
            
    except Exception as e:
        logger.warning(f"Could not determine type string for {field_type}: {e}")
        return "Any"


def _get_field_default(field_info: FieldInfo) -> Any:
    """
    Get the default value for a field.
    
    Args:
        field_info: Pydantic field information
        
    Returns:
        Default value or None if no default
    """
    try:
        if field_info.default is not None:
            return field_info.default
        elif field_info.default_factory is not None:
            # Try to call the default factory
            try:
                return field_info.default_factory()
            except Exception:
                # If factory fails, return None
                return None
        else:
            return None
    except Exception as e:
        logger.warning(f"Could not get default value: {e}")
        return None


def _extract_validation_rules(field_info: FieldInfo) -> Dict[str, Any]:
    """
    Extract validation rules from field info.
    
    Args:
        field_info: Pydantic field information
        
    Returns:
        Dict containing validation rules
    """
    validation_rules = {}
    
    try:
        # Extract constraints from field info
        if hasattr(field_info, 'constraints'):
            constraints = field_info.constraints
            if constraints:
                for constraint in constraints:
                    if hasattr(constraint, 'gt'):
                        validation_rules['min'] = constraint.gt
                    elif hasattr(constraint, 'ge'):
                        validation_rules['min'] = constraint.ge
                    elif hasattr(constraint, 'lt'):
                        validation_rules['max'] = constraint.lt
                    elif hasattr(constraint, 'le'):
                        validation_rules['max'] = constraint.le
                    elif hasattr(constraint, 'min_length'):
                        validation_rules['minLength'] = constraint.min_length
                    elif hasattr(constraint, 'max_length'):
                        validation_rules['maxLength'] = constraint.max_length
                    elif hasattr(constraint, 'pattern'):
                        validation_rules['pattern'] = constraint.pattern
        
        # Add required flag
        validation_rules['required'] = field_info.is_required()
        
    except Exception as e:
        logger.warning(f"Could not extract validation rules: {e}")
    
    return validation_rules


def get_field_defaults() -> Dict[str, Any]:
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


def get_field_validation_rules() -> Dict[str, Dict[str, Any]]:
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
