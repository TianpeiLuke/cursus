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

from .unified_config_manager import UnifiedConfigManager
from .type_aware_config_serializer import (
    TypeAwareConfigSerializer,
    serialize_config as _serialize_config,
    deserialize_config as _deserialize_config,
)
from .step_catalog_aware_categorizer import StepCatalogAwareConfigFieldCategorizer
from .inheritance_aware_field_generator import (
    InheritanceAwareFieldGenerator,
    get_inheritance_aware_field_generator,
    get_inheritance_aware_form_fields,
)
# CircularReferenceTracker eliminated - functionality moved to UnifiedConfigManager's simple tracker
# TierRegistry eliminated - functionality moved to UnifiedConfigManager

# Import step catalog adapters for config class functionality
try:
    from ...step_catalog.adapters.config_class_detector import (
        ConfigClassStoreAdapter as ConfigClassStore,
        ConfigClassDetectorAdapter as ConfigClassDetector,
        detect_config_classes_from_json,
        build_complete_config_classes,
    )
except ImportError:
    # Fallback for environments where step catalog is not available
    class ConfigClassStore:
        """Fallback ConfigClassStore for environments without step catalog."""
        _classes = {}
        
        @classmethod
        def register(cls, config_class):
            cls._classes[config_class.__name__] = config_class
            return config_class
        
        @classmethod
        def get_all_classes(cls):
            return cls._classes.copy()
    
    # Fallback ConfigClassDetector
    class ConfigClassDetector:
        """Fallback ConfigClassDetector for environments without step catalog."""
        MODEL_TYPE_FIELD = "__model_type__"
        METADATA_FIELD = "metadata"
        CONFIG_TYPES_FIELD = "config_types"
        CONFIGURATION_FIELD = "configuration"
        SPECIFIC_FIELD = "specific"
        
        @classmethod
        def detect_from_json(cls, config_file_path: str):
            return ConfigClassStore.get_all_classes()
        
        @classmethod
        def from_config_store(cls, config_file_path: str):
            return cls.detect_from_json(config_file_path)
        
        @classmethod
        def _extract_class_names(cls, data, logger):
            return set()
    
    def detect_config_classes_from_json(config_file_path: str):
        """Fallback function for detecting config classes from JSON."""
        return ConfigClassDetector.detect_from_json(config_file_path)
    
    def build_complete_config_classes():
        """Fallback function for building complete config classes."""
        return ConfigClassStore.get_all_classes()

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
    # Primary API functions - PRESERVED
    "merge_and_save_configs",
    "load_configs",
    "serialize_config",
    "deserialize_config",
    # Unified config management - MAIN COMPONENT
    "UnifiedConfigManager",  # Primary config management component
    # Config class management
    "ConfigClassStore",  # Export for use as a decorator
    "register_config_class",  # Convenient alias for the decorator
    # Enhanced categorization
    "StepCatalogAwareConfigFieldCategorizer",  # Enhanced field categorizer
    # Inheritance-aware field generation - NEW CONSOLIDATED COMPONENT
    "InheritanceAwareFieldGenerator",  # Centralized inheritance-aware field generation
    "get_inheritance_aware_field_generator",  # Factory function for field generator
    "get_inheritance_aware_form_fields",  # Convenience function for direct usage
    # Config class detection functionality
    "ConfigClassDetector",
    "detect_config_classes_from_json",
    "build_complete_config_classes",
    # NOTE: The following components have been eliminated:
    # - ConfigMerger (replaced with UnifiedConfigManager)
    # - CircularReferenceTracker (replaced with UnifiedConfigManager's simple tracker)
    # - ConfigFieldTierRegistry (replaced with config class methods)
    # The following modules are not currently available:
    # - DefaultValuesProvider
    # - FieldDerivationEngine
    # - DataConfig, ModelConfig, RegistrationConfig, EssentialInputs
]


# Create logger
logger = logging.getLogger(__name__)


def merge_and_save_configs(
    config_list: List[Any],
    output_file: str,
    processing_step_config_base_class: Optional[type] = None,
    workspace_dirs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Merge and save multiple configs to a single JSON file.
    
    Uses UnifiedConfigManager for streamlined processing with workspace awareness.

    Args:
        config_list: List of configuration objects to merge and save
        output_file: Path to the output JSON file
        processing_step_config_base_class: Optional base class to identify processing step configs
        workspace_dirs: Optional list of workspace directories for step catalog integration

    Returns:
        dict: The categorized configuration structure

    Raises:
        ValueError: If config_list is empty or contains invalid configs
        IOError: If there's an issue writing to the output file
        TypeError: If configs are not serializable
    """
    # Validate inputs
    if not config_list:
        raise ValueError("Config list cannot be empty")

    try:
        # Use UnifiedConfigManager with workspace awareness
        from .unified_config_manager import UnifiedConfigManager
        
        # Pass workspace_dirs directly to UnifiedConfigManager
        manager = UnifiedConfigManager(workspace_dirs=workspace_dirs)
        
        # Save configs using UnifiedConfigManager
        logger.info(f"Merging and saving {len(config_list)} configs to {output_file}")
        merged = manager.save(config_list, output_file, processing_step_config_base_class)
        
        logger.info(f"Successfully saved merged configs to {output_file}")
        return merged
        
    except Exception as e:
        logger.error(f"Error merging and saving configs: {str(e)}")
        raise


def load_configs(
    input_file: str, 
    config_classes: Optional[Dict[str, Type]] = None,
    workspace_dirs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load multiple configs from a JSON file.
    
    Uses UnifiedConfigManager for streamlined processing with workspace awareness.

    Args:
        input_file: Path to the input JSON file
        config_classes: Optional dictionary mapping class names to class types
                       If not provided, UnifiedConfigManager discovery will be used
        workspace_dirs: Optional list of workspace directories for step catalog integration

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
    try:
        # Use UnifiedConfigManager with workspace awareness
        from .unified_config_manager import UnifiedConfigManager
        
        # Pass workspace_dirs directly to UnifiedConfigManager
        manager = UnifiedConfigManager(workspace_dirs=workspace_dirs)
        
        # Load configs using UnifiedConfigManager
        logger.info(f"Loading configs from {input_file}")
        loaded_configs = manager.load(input_file, config_classes)

        logger.info(
            f"Successfully loaded configs from {input_file} with {len(loaded_configs.get('specific', {}))} specific configs"
        )

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


def _get_enhanced_config_classes() -> Dict[str, Type]:
    """
    Get config classes using enhanced discovery with step catalog integration.
        
    Returns:
        Dictionary mapping class names to class types
    """
    try:
        # Try to use unified config manager for enhanced discovery
        from .unified_config_manager import get_unified_config_manager
        manager = get_unified_config_manager()
        config_classes = manager.get_config_classes()
        
        if config_classes:
            logger.info(f"Enhanced discovery found {len(config_classes)} config classes")
            return config_classes
            
    except ImportError:
        logger.debug("UnifiedConfigManager not available")
    except Exception as e:
        logger.debug(f"Enhanced discovery failed: {e}")
    
    # Fallback to step catalog integration from utils
    try:
        from ...steps.configs.utils import build_complete_config_classes
        config_classes = build_complete_config_classes()
        
        if config_classes:
            logger.info(f"Step catalog discovery found {len(config_classes)} config classes")
            return config_classes
            
    except ImportError:
        logger.debug("Step catalog utils not available")
    except Exception as e:
        logger.debug(f"Step catalog discovery failed: {e}")
    
    # Final fallback to ConfigClassStore
    return ConfigClassStore.get_all_classes()


def _get_basic_config_classes() -> Dict[str, Type]:
    """
    Get basic config classes as final fallback.
    
    Returns:
        Dictionary with basic config classes
    """
    try:
        from ...core.base.config_base import BasePipelineConfig
        from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase
        from ...core.base.hyperparameters_base import ModelHyperparameters
        
        return {
            "BasePipelineConfig": BasePipelineConfig,
            "ProcessingStepConfigBase": ProcessingStepConfigBase,
            "ModelHyperparameters": ModelHyperparameters,
        }
    except ImportError as e:
        logger.error(f"Could not import basic config classes: {e}")
        return {}


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
        raise TypeError(
            f"Failed to serialize config of type {type(config).__name__}: {str(e)}"
        )


def deserialize_config(
    data: Dict[str, Any], config_classes: Optional[Dict[str, Type]] = None
) -> Any:
    """
    Deserialize a dictionary back into a configuration object.

    This function deserializes a dictionary into a configuration object based on
    type information embedded in the dictionary. If the dictionary contains the
    __model_type__ field, it will attempt to reconstruct
    the original object type using the step catalog system.

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
def register_config_class(cls: Any) -> Any:
    """
    Register a configuration class with the ConfigClassStore.

    This is a convenient alias for ConfigClassStore.register decorator.

    Args:
        cls: The class to register

    Returns:
        The class, allowing this to be used as a decorator
    """
    return ConfigClassStore.register(cls)
