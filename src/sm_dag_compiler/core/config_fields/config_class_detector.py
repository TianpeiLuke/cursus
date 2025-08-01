"""
Config Class Detector for configuration objects.

This module provides utilities for detecting required configuration classes
from JSON configuration files, implementing efficient class loading.
"""

import json
import logging
from typing import Dict, Type, Set, Optional, Any
from pathlib import Path

from .config_class_store import ConfigClassStore, build_complete_config_classes

# Import base classes to ensure essential classes are always available
try:
    from ...steps.configs.config_base import BasePipelineConfig
except ImportError:
    # For testing environments or when running outside the main project
    BasePipelineConfig = None
    

class ConfigClassDetector:
    """
    Utility class for detecting required configuration classes from JSON files.
    
    This class implements the Type Detection and Validation principle by analyzing
    configuration files to determine which configuration classes are required,
    rather than loading all possible classes.
    """
    
    # Constants for JSON field names
    MODEL_TYPE_FIELD = "__model_type__"
    METADATA_FIELD = "metadata"
    CONFIG_TYPES_FIELD = "config_types"
    CONFIGURATION_FIELD = "configuration"
    SPECIFIC_FIELD = "specific"
    
    # Essential base classes that should always be included
    ESSENTIAL_CLASSES = ["BasePipelineConfig", "ProcessingStepConfigBase"]
    
    @staticmethod
    def detect_from_json(config_path: str) -> Dict[str, Type]:
        """
        Detect required config classes from a configuration JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            Dictionary mapping config class names to config classes
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Verify the file exists
            if not Path(config_path).is_file():
                logger.error(f"Configuration file not found: {config_path}")
                return build_complete_config_classes()
                
            # Read and parse the JSON file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract required class names
            required_class_names = ConfigClassDetector._extract_class_names(config_data, logger)
            
            if not required_class_names:
                logger.warning("No config class names found in configuration file")
                # Fallback to loading all classes
                return build_complete_config_classes()
                
            logger.info(f"Detected {len(required_class_names)} required config classes in configuration file")
            
            # Get only the required config classes from the complete set
            all_available_classes = build_complete_config_classes()
            required_classes = {}
            
            # Only keep classes that are actually used in the config file
            for class_name, class_type in all_available_classes.items():
                if class_name in required_class_names:
                    required_classes[class_name] = class_type
                    
            # Always include essential base classes
            for essential_class in ConfigClassDetector.ESSENTIAL_CLASSES:
                if essential_class not in required_classes and essential_class in all_available_classes:
                    required_classes[essential_class] = all_available_classes[essential_class]
                    
            # Report on any missing classes that couldn't be loaded
            missing_classes = required_class_names - set(required_classes.keys())
            if missing_classes:
                logger.warning(f"Could not load {len(missing_classes)} required classes: {missing_classes}")
                
            logger.info(f"Successfully loaded {len(required_classes)} of {len(required_class_names)} required classes")
            
            return required_classes
            
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading or parsing configuration file: {e}")
            logger.warning("Falling back to loading all available config classes")
            # Fallback to loading all classes if we can't parse the file
            return build_complete_config_classes()
    
    @staticmethod
    def _extract_class_names(config_data: Dict[str, Any], logger: logging.Logger) -> Set[str]:
        """
        Extract config class names from configuration data.
        
        Args:
            config_data: Parsed configuration data
            logger: Logger instance
            
        Returns:
            Set of required class names
        """
        required_class_names = set()
        
        # Extract from metadata.config_types
        if (ConfigClassDetector.METADATA_FIELD in config_data and 
            ConfigClassDetector.CONFIG_TYPES_FIELD in config_data[ConfigClassDetector.METADATA_FIELD]):
            config_types = config_data[ConfigClassDetector.METADATA_FIELD][ConfigClassDetector.CONFIG_TYPES_FIELD]
            required_class_names.update(config_types.values())
            logger.debug(f"Found {len(config_types)} config class names in metadata")
        
        # Extract from configuration.specific.__model_type__ fields
        if (ConfigClassDetector.CONFIGURATION_FIELD in config_data and 
            ConfigClassDetector.SPECIFIC_FIELD in config_data[ConfigClassDetector.CONFIGURATION_FIELD]):
            specific_configs = config_data[ConfigClassDetector.CONFIGURATION_FIELD][ConfigClassDetector.SPECIFIC_FIELD]
            model_type_count = 0
            
            for step_name, step_config in specific_configs.items():
                if isinstance(step_config, dict) and ConfigClassDetector.MODEL_TYPE_FIELD in step_config:
                    model_type = step_config[ConfigClassDetector.MODEL_TYPE_FIELD]
                    required_class_names.add(model_type)
                    model_type_count += 1
                    
            if model_type_count > 0:
                logger.debug(f"Found {model_type_count} model type fields in specific configurations")
        
        return required_class_names
    
    @classmethod
    def from_config_store(cls, config_path: str) -> Dict[str, Type]:
        """
        Alternative implementation that uses only ConfigClassStore.
        
        This method doesn't rely on build_complete_config_classes() and is
        designed for future use when all classes are properly registered
        with ConfigClassStore.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            Dictionary mapping config class names to config classes
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Read and parse the JSON file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract required class names
            required_class_names = cls._extract_class_names(config_data, logger)
            
            if not required_class_names:
                logger.warning("No config class names found in configuration file")
                # Return all registered classes from ConfigClassStore
                return {name: ConfigClassStore.get_class(name) 
                        for name in ConfigClassStore.registered_names()}
            
            # Get classes from ConfigClassStore
            required_classes = {}
            for class_name in required_class_names:
                class_type = ConfigClassStore.get_class(class_name)
                if class_type:
                    required_classes[class_name] = class_type
            
            # Always include essential base classes
            for essential_class in cls.ESSENTIAL_CLASSES:
                if essential_class not in required_classes:
                    class_type = ConfigClassStore.get_class(essential_class)
                    if class_type:
                        required_classes[essential_class] = class_type
            
            return required_classes
            
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading or parsing configuration file: {e}")
            # Return all registered classes from ConfigClassStore
            return {name: ConfigClassStore.get_class(name) 
                    for name in ConfigClassStore.registered_names()}


def detect_config_classes_from_json(config_path: str) -> Dict[str, Type]:
    """
    Detect required config classes from a configuration JSON file.
    
    This helper function analyzes the configuration file to determine which
    configuration classes are actually used, rather than loading all possible classes.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Dictionary mapping config class names to config classes
    """
    return ConfigClassDetector.detect_from_json(config_path)
