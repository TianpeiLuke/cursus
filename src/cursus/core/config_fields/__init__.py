"""
Configuration Field Manager Package.

This package provides robust tools for managing configuration fields, including:
- Field categorization for configuration organization
- Type-aware serialization and deserialization
- Configuration class registration
- Configuration merging and loading
- Three-tier configuration architecture components

Primary API functions:
- merge_and_save_configs: Merge and save multiple config objects to a unified JSON file
- load_configs: Load config objects from a saved JSON file
- serialize_config: Convert a config object to a JSON-serializable dict with type metadata
- deserialize_config: Convert a serialized dict back to a config object

New Three-Tier Architecture Components:
- ConfigFieldTierRegistry: Registry for field tier classifications (Tier 1, 2, 3)
- DefaultValuesProvider: Provider for default values (Tier 2)
- FieldDerivationEngine: Engine for deriving field values (Tier 3)
- Essential Input Models: Pydantic models for Data, Model, and Registration configurations

Usage:
    ```python
    from ..config_field_manager import merge_and_save_configs, load_configs, ConfigClassStore    
    # Register config classes for type-aware serialization
    @ConfigClassStore.register
    class MyConfig:
        ...
        
    # Merge and save configs
    configs = [MyConfig(...), AnotherConfig(...)]
    merge_and_save_configs(configs, "output.json")
    
    # Load configs
    loaded_configs = load_configs("output.json")
    
    # Using the three-tier architecture
    from ..config_field_manager import (        ConfigFieldTierRegistry, DefaultValuesProvider, 
        FieldDerivationEngine, DataConfig, ModelConfig, RegistrationConfig
    )
    
    # Apply defaults and derive fields
    DefaultValuesProvider.apply_defaults(config)
    field_engine = FieldDerivationEngine()
    field_engine.derive_fields(config)
    ```
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Type, Union, Tuple, Set
from pathlib import Path

from .config_merger import ConfigMerger
from .config_class_store import ConfigClassStore
from .type_aware_config_serializer import (
    TypeAwareConfigSerializer, 
    serialize_config as _serialize_config,
    deserialize_config as _deserialize_config
)
from .config_field_categorizer import ConfigFieldCategorizer
from .circular_reference_tracker import CircularReferenceTracker
# Three-tier architecture components
from .tier_registry import ConfigFieldTierRegistry
# Import below modules when they are available
# from .default_values_provider import DefaultValuesProvider
# from .field_derivation_engine import FieldDerivationEngine
# from .essential_input_models import (
#     DataConfig,
#     ModelConfig,
#     RegistrationConfig,
#     EssentialInputs
# )


__all__ = [
    # Original exports
    'merge_and_save_configs', 
    'load_configs',
    'serialize_config',
    'deserialize_config',
    'ConfigClassStore',  # Export for use as a decorator
    'register_config_class',  # Convenient alias for the decorator
    'CircularReferenceTracker',  # For advanced circular reference handling
    
    # Three-tier architecture components
    'ConfigFieldTierRegistry',
    # The following modules are not currently available
    # 'DefaultValuesProvider',
    # 'FieldDerivationEngine',
    # 'DataConfig',
    # 'ModelConfig',
    # 'RegistrationConfig',
    # 'EssentialInputs'
]


# Create logger
logger = logging.getLogger(__name__)


def merge_and_save_configs(
    config_list: List[Any], 
    output_file: str, 
    processing_step_config_base_class: Optional[type] = None
) -> Dict[str, Any]:
    """
    Merge and save multiple configs to a single JSON file.
    
    This function uses the ConfigFieldCategorizer to analyze fields across all configurations,
    organizing them into shared and specific sections based on values and usage patterns.
    
    The output follows the simplified structure:
    ```
    {
      "metadata": {
        "created_at": "ISO timestamp",
        "config_types": {
          "StepName1": "ConfigClassName1",
          "StepName2": "ConfigClassName2",
          ...
        }
      },
      "configuration": {
        "shared": {
          "common_field1": "common_value1",
          ...
        },
        "specific": {
          "StepName1": {
            "specific_field1": "specific_value1",
            ...
          },
          "StepName2": {
            "specific_field2": "specific_value2",
            ...
          }
        }
      }
    }
    ```
    
    Args:
        config_list: List of configuration objects to merge and save
        output_file: Path to the output JSON file
        processing_step_config_base_class: Optional base class to identify processing step configs
        
    Returns:
        dict: The merged configuration
        
    Raises:
        ValueError: If config_list is empty or contains invalid configs
        IOError: If there's an issue writing to the output file
        TypeError: If configs are not serializable
    """
    # Validate inputs
    if not config_list:
        raise ValueError("Config list cannot be empty")
        
    try:
        # Create directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create merger and save configs
        logger.info(f"Merging and saving {len(config_list)} configs to {output_file}")
        merger = ConfigMerger(config_list, processing_step_config_base_class)
        merged = merger.save(output_file)
        logger.info(f"Successfully saved merged configs to {output_file}")
        
        return merged
    except Exception as e:
        logger.error(f"Error merging and saving configs: {str(e)}")
        raise


def load_configs(
    input_file: str, 
    config_classes: Optional[Dict[str, Type]] = None
) -> Dict[str, Any]:
    """
    Load multiple configs from a JSON file.
    
    This function loads configurations from a JSON file that was previously saved using
    merge_and_save_configs. It reconstructs the configuration objects based on the
    type information stored in the file, using the simplified structure with shared
    and specific fields.
    
    Args:
        input_file: Path to the input JSON file
        config_classes: Optional dictionary mapping class names to class types
                       If not provided, all classes registered with ConfigClassStore will be used
        
    Returns:
        dict: A dictionary with the following structure:
            {
                "shared": {shared_field1: value1, ...},
                "specific": {
                    "StepName1": {specific_field1: value1, ...},
                    "StepName2": {specific_field2: value2, ...},
                    ...
                }
            }
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        json.JSONDecodeError: If the input file is not valid JSON
        KeyError: If required keys are missing from the file
        TypeError: If deserialization fails due to type mismatches
    """
    # Validate input file
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    try:
        # Get config classes from store or use provided ones
        all_config_classes = config_classes or ConfigClassStore.get_all_classes()
        
        if not all_config_classes:
            logger.warning("No config classes provided or registered with ConfigClassStore")
            
        # Use the ConfigMerger's load method which handles deserialization
        logger.info(f"Loading configs from {input_file}")
        loaded_configs = ConfigMerger.load(input_file, all_config_classes)
        logger.info(f"Successfully loaded configs from {input_file} with {len(loaded_configs.get('specific', {}))} specific configs")
        
        return loaded_configs
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {str(e)}")
        raise
    except KeyError as e:
        logger.error(f"Missing required key in input file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading configs: {str(e)}")
        raise


def serialize_config(config: Any) -> Dict[str, Any]:
    """
    Serialize a configuration object to a JSON-serializable dictionary.
    
    This function serializes a configuration object, preserving its type information
    and special fields. It embeds metadata including the step name derived from
    attributes like 'job_type', 'data_type', and 'mode'.
    
    Args:
        config: The configuration object to serialize
        
    Returns:
        dict: A serialized representation of the config
        
    Raises:
        TypeError: If the config is not serializable
    """
    try:
        return _serialize_config(config)
    except Exception as e:
        logger.error(f"Error serializing config: {str(e)}")
        raise TypeError(f"Failed to serialize config of type {type(config).__name__}: {str(e)}")


def deserialize_config(data: Dict[str, Any], config_classes: Optional[Dict[str, Type]] = None) -> Any:
    """
    Deserialize a dictionary back into a configuration object.
    
    This function deserializes a dictionary into a configuration object based on
    type information embedded in the dictionary. If the dictionary contains the
    __model_type__ and __model_module__ fields, it will attempt to reconstruct
    the original object type.
    
    Args:
        data: The serialized dictionary
        config_classes: Optional dictionary mapping class names to class types
                       If not provided, all classes registered with ConfigClassStore will be used
        
    Returns:
        Any: The deserialized configuration object
        
    Raises:
        TypeError: If the data cannot be deserialized to the specified type
    """
    # Get config classes from store or use provided ones
    all_config_classes = config_classes or ConfigClassStore.get_all_classes()
    
    try:
        serializer = TypeAwareConfigSerializer(all_config_classes)
        return serializer.deserialize(data)
    except Exception as e:
        logger.error(f"Error deserializing config: {str(e)}")
        raise TypeError(f"Failed to deserialize config: {str(e)}")


# Convenient alias for the ConfigClassStore.register decorator
def register_config_class(cls):
    """
    Register a configuration class with the ConfigClassStore.
    
    This is a convenient alias for ConfigClassStore.register decorator.
    
    Args:
        cls: The class to register
        
    Returns:
        The class, allowing this to be used as a decorator
    """
    return ConfigClassStore.register(cls)
