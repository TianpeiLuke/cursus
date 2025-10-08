"""
Configuration loader for Cradle Data Load Config

This module provides functions to load and save CradleDataLoadingConfig objects
from/to JSON files, handling the nested configuration structure properly.
"""

import json
from typing import Dict, Any, List
from pathlib import Path

from ....steps.configs.config_cradle_data_loading_step import (
    CradleDataLoadingConfig,
    DataSourcesSpecificationConfig,
    DataSourceConfig,
    MdsDataSourceConfig,
    EdxDataSourceConfig,
    AndesDataSourceConfig,
    TransformSpecificationConfig,
    JobSplitOptionsConfig,
    OutputSpecificationConfig,
    CradleJobSpecificationConfig
)


def load_cradle_config_from_json(file_path: str) -> CradleDataLoadingConfig:
    """
    Load a CradleDataLoadingConfig from a JSON file.
    
    This function properly handles the nested configuration structure by
    reconstructing all the nested config objects from their dictionaries.
    
    Args:
        file_path: Path to the JSON file containing the configuration
        
    Returns:
        CradleDataLoadingConfig: The loaded configuration object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON is invalid or missing required fields
        
    Example:
        ```python
        # Load configuration from JSON file
        config = load_cradle_config_from_json('cradle_config_training.json')
        config_list.append(config)
        ```
    """
    try:
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        # Reconstruct nested configuration objects
        return _reconstruct_cradle_config(config_data)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {str(e)}")


def save_cradle_config_to_json(config: CradleDataLoadingConfig, file_path: str) -> None:
    """
    Save a CradleDataLoadingConfig to a JSON file.
    
    Args:
        config: The configuration object to save
        file_path: Path where to save the JSON file
        
    Example:
        ```python
        # Save configuration to JSON file
        save_cradle_config_to_json(my_config, 'cradle_config_training.json')
        ```
    """
    try:
        config_data = config.model_dump()
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    except Exception as e:
        raise ValueError(f"Error saving configuration: {str(e)}")


def _reconstruct_cradle_config(config_data: Dict[str, Any]) -> CradleDataLoadingConfig:
    """
    Reconstruct a CradleDataLoadingConfig from a dictionary, properly handling
    all nested configuration objects.
    
    Args:
        config_data: Dictionary containing the configuration data
        
    Returns:
        CradleDataLoadingConfig: The reconstructed configuration object
    """
    # Reconstruct data sources specification
    data_sources_spec_data = config_data.get('data_sources_spec', {})
    data_sources_spec = _reconstruct_data_sources_spec(data_sources_spec_data)
    
    # Reconstruct transform specification
    transform_spec_data = config_data.get('transform_spec', {})
    transform_spec = _reconstruct_transform_spec(transform_spec_data)
    
    # Reconstruct output specification
    output_spec_data = config_data.get('output_spec', {})
    output_spec = _reconstruct_output_spec(output_spec_data)
    
    # Reconstruct cradle job specification
    cradle_job_spec_data = config_data.get('cradle_job_spec', {})
    cradle_job_spec = _reconstruct_cradle_job_spec(cradle_job_spec_data)
    
    # Create the main config with reconstructed nested objects
    config_dict = config_data.copy()
    config_dict['data_sources_spec'] = data_sources_spec
    config_dict['transform_spec'] = transform_spec
    config_dict['output_spec'] = output_spec
    config_dict['cradle_job_spec'] = cradle_job_spec
    
    return CradleDataLoadingConfig(**config_dict)


def _reconstruct_data_sources_spec(data: Dict[str, Any]) -> DataSourcesSpecificationConfig:
    """Reconstruct DataSourcesSpecificationConfig from dictionary."""
    # Reconstruct data sources list
    data_sources_list = []
    for ds_data in data.get('data_sources', []):
        data_source = _reconstruct_data_source(ds_data)
        data_sources_list.append(data_source)
    
    return DataSourcesSpecificationConfig(
        start_date=data.get('start_date'),
        end_date=data.get('end_date'),
        data_sources=data_sources_list
    )


def _reconstruct_data_source(data: Dict[str, Any]) -> DataSourceConfig:
    """Reconstruct DataSourceConfig from dictionary."""
    data_source_type = data.get('data_source_type')
    
    # Reconstruct the appropriate properties based on type
    mds_props = None
    edx_props = None
    andes_props = None
    
    if data_source_type == 'MDS' and 'mds_data_source_properties' in data:
        mds_props = MdsDataSourceConfig(**data['mds_data_source_properties'])
    elif data_source_type == 'EDX' and 'edx_data_source_properties' in data:
        edx_props = EdxDataSourceConfig(**data['edx_data_source_properties'])
    elif data_source_type == 'ANDES' and 'andes_data_source_properties' in data:
        andes_props = AndesDataSourceConfig(**data['andes_data_source_properties'])
    
    return DataSourceConfig(
        data_source_name=data.get('data_source_name'),
        data_source_type=data_source_type,
        mds_data_source_properties=mds_props,
        edx_data_source_properties=edx_props,
        andes_data_source_properties=andes_props
    )


def _reconstruct_transform_spec(data: Dict[str, Any]) -> TransformSpecificationConfig:
    """Reconstruct TransformSpecificationConfig from dictionary."""
    # Reconstruct job split options
    job_split_data = data.get('job_split_options', {})
    job_split_options = JobSplitOptionsConfig(**job_split_data)
    
    return TransformSpecificationConfig(
        transform_sql=data.get('transform_sql'),
        job_split_options=job_split_options
    )


def _reconstruct_output_spec(data: Dict[str, Any]) -> OutputSpecificationConfig:
    """Reconstruct OutputSpecificationConfig from dictionary."""
    return OutputSpecificationConfig(**data)


def _reconstruct_cradle_job_spec(data: Dict[str, Any]) -> CradleJobSpecificationConfig:
    """Reconstruct CradleJobSpecificationConfig from dictionary."""
    return CradleJobSpecificationConfig(**data)


def create_config_from_ui_data(ui_data: Dict[str, Any]) -> CradleDataLoadingConfig:
    """
    Create a CradleDataLoadingConfig from UI data format.
    
    This function handles the conversion from the UI's data format
    to the proper nested configuration structure.
    
    Args:
        ui_data: Dictionary containing UI form data
        
    Returns:
        CradleDataLoadingConfig: The created configuration object
    """
    # This is essentially the same as the validation service's build_final_config
    # but can be used independently
    from ..services.validation_service import ValidationService
    
    validation_service = ValidationService()
    return validation_service.build_final_config(ui_data)
