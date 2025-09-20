"""
Config class detector adapters for backward compatibility.

This module provides adapters that maintain existing config class detection APIs
during the migration from legacy discovery systems to the unified StepCatalog system.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, Type

from ..step_catalog import StepCatalog

logger = logging.getLogger(__name__)


class ConfigClassDetectorAdapter:
    """
    Adapter maintaining backward compatibility with ConfigClassDetector.
    
    Replaces: src/cursus/core/config_fields/config_class_detector.py
    
    MODERN APPROACH: Uses step catalog's superior AST-based config discovery
    instead of legacy JSON parsing. This provides more accurate and comprehensive
    config class detection.
    """
    
    # Constants for backward compatibility (minimal legacy support)
    MODEL_TYPE_FIELD = "__model_type__"
    METADATA_FIELD = "metadata"
    CONFIG_TYPES_FIELD = "config_types"
    CONFIGURATION_FIELD = "configuration"
    SPECIFIC_FIELD = "specific"
    
    def __init__(self, workspace_root: Optional[Path] = None):
        """Initialize with unified catalog."""
        # PORTABLE: Use package-only discovery by default (works in all deployment scenarios)
        if workspace_root is None:
            self.catalog = StepCatalog(workspace_dirs=None)
        else:
            self.catalog = StepCatalog(workspace_dirs=[workspace_root])
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def detect_from_json(config_path: str) -> Dict[str, Any]:
        """
        MODERN APPROACH: Use step catalog's AST-based discovery instead of JSON parsing.
        
        This method now uses the superior AST-based discovery from the unified step catalog
        rather than the legacy JSON parsing approach. This provides more accurate and 
        comprehensive config class detection by analyzing actual source code.
        
        Real usage pattern (from dynamic_template.py):
        detected_classes = detect_config_classes_from_json(self.config_path)
        
        Args:
            config_path: Path to configuration file (used to determine workspace root)
            
        Returns:
            Dictionary mapping config class names to config class types
        """
        try:
            # PORTABLE: Use package-only discovery (works in all deployment scenarios)
            catalog = StepCatalog(workspace_dirs=None)
            
            # Get complete config classes (AST discovery + ConfigClassStore integration)
            config_classes = catalog.build_complete_config_classes()
            
            logger = logging.getLogger(__name__)
            logger.info(f"Discovered {len(config_classes)} config classes via unified step catalog (AST-based)")
            
            return config_classes
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error detecting config classes via step catalog: {e}")
            
            # No fallback needed - return empty dict if step catalog fails
            logger.warning("Step catalog discovery failed, returning empty dict")
            return {}
    
    @staticmethod
    def _extract_class_names(config_data: Dict[str, Any], logger) -> set:
        """
        LEGACY COMPATIBILITY: Minimal JSON parsing for backward compatibility only.
        
        This method is maintained for any legacy code that might still use it,
        but new code should use the step catalog's AST-based discovery instead.
        
        NOTE: Analysis shows this is only used in tests, not in production code.
        """
        class_names = set()
        
        try:
            # Extract from metadata.config_types (legacy pattern)
            if ConfigClassDetectorAdapter.METADATA_FIELD in config_data:
                metadata = config_data[ConfigClassDetectorAdapter.METADATA_FIELD]
                if isinstance(metadata, dict) and ConfigClassDetectorAdapter.CONFIG_TYPES_FIELD in metadata:
                    config_types = metadata[ConfigClassDetectorAdapter.CONFIG_TYPES_FIELD]
                    if isinstance(config_types, dict):
                        class_names.update(config_types.values())
            
            # Extract from configuration.specific sections (legacy pattern)
            if ConfigClassDetectorAdapter.CONFIGURATION_FIELD in config_data:
                configuration = config_data[ConfigClassDetectorAdapter.CONFIGURATION_FIELD]
                if isinstance(configuration, dict) and ConfigClassDetectorAdapter.SPECIFIC_FIELD in configuration:
                    specific = configuration[ConfigClassDetectorAdapter.SPECIFIC_FIELD]
                    if isinstance(specific, dict):
                        for step_config in specific.values():
                            if isinstance(step_config, dict) and ConfigClassDetectorAdapter.MODEL_TYPE_FIELD in step_config:
                                class_names.add(step_config[ConfigClassDetectorAdapter.MODEL_TYPE_FIELD])
            
            logger.debug(f"Legacy JSON parsing extracted {len(class_names)} class names")
            return class_names
            
        except Exception as e:
            logger.error(f"Error in legacy JSON parsing: {e}")
            return set()
    
    @classmethod
    def from_config_store(cls, config_path: str) -> Dict[str, Any]:
        """
        MODERN APPROACH: Use step catalog's integrated discovery.
        
        This method uses the step catalog's build_complete_config_classes which
        integrates both AST-based discovery and ConfigClassStore registration,
        providing the most comprehensive config class detection.
        """
        return cls.detect_from_json(config_path)


class ConfigClassStoreAdapter:
    """
    Adapter maintaining backward compatibility with ConfigClassStore.
    
    Replaces: src/cursus/core/config_fields/config_class_store.py (partial)
    """
    
    # Single registry instance - implementing Single Source of Truth
    _registry: Dict[str, Any] = {}
    _logger = logging.getLogger(__name__)

    @classmethod
    def register(cls, config_class: Optional[Any] = None) -> Any:
        """Legacy method: register a config class."""
        def _register(cls_to_register: Any) -> Any:
            cls_name = cls_to_register.__name__
            if (
                cls_name in cls._registry
                and cls._registry[cls_name] != cls_to_register
            ):
                cls._logger.warning(
                    f"Class {cls_name} is already registered and is being overwritten."
                )
            cls._registry[cls_name] = cls_to_register
            cls._logger.debug(f"Registered class: {cls_name}")
            return cls_to_register

        if config_class is not None:
            return _register(config_class)
        return _register

    @classmethod
    def get_class(cls, class_name: str) -> Optional[Any]:
        """Legacy method: get a registered class by name."""
        class_obj = cls._registry.get(class_name)
        if class_obj is None:
            cls._logger.debug(f"Class not found in registry: {class_name}")
        return class_obj

    @classmethod
    def get_all_classes(cls) -> Dict[str, Any]:
        """Legacy method: get all registered classes."""
        return cls._registry.copy()

    @classmethod
    def register_many(cls, *config_classes: Any) -> None:
        """Legacy method: register multiple config classes at once."""
        for config_class in config_classes:
            cls.register(config_class)

    @classmethod
    def clear(cls) -> None:
        """Legacy method: clear the registry."""
        cls._registry.clear()
        cls._logger.debug("Cleared config class registry")

    @classmethod
    def registered_names(cls) -> set:
        """Legacy method: get all registered class names."""
        return set(cls._registry.keys())


# Legacy functions for backward compatibility
def build_complete_config_classes(project_id: Optional[str] = None) -> Dict[str, Any]:
    """Legacy function: build complete config classes using catalog."""
    try:
        # PORTABLE: Use package-only discovery (works in all deployment scenarios)
        catalog = StepCatalog(workspace_dirs=None)
        
        # Use catalog's build_complete_config_classes method
        config_classes = catalog.build_complete_config_classes(project_id)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Built {len(config_classes)} complete config classes via unified catalog")
        
        return config_classes
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error building complete config classes: {e}")
        
        # Fallback to registered classes
        return ConfigClassStoreAdapter.get_all_classes()


def detect_config_classes_from_json(config_path: str) -> Dict[str, Any]:
    """Legacy function: detect config classes using catalog."""
    return ConfigClassDetectorAdapter.detect_from_json(config_path)
