"""
Configuration class store for serialization and deserialization.

This module provides a centralized store for configuration classes used by the
serializer and deserializer, following the Single Source of Truth principle.
"""

import logging
from typing import Dict, Type, Optional, List, Callable, TypeVar, cast, Any, Set

# Type variable for generic class registration
T = TypeVar('T')

class ConfigClassStore:
    """
    Registry of configuration classes for serialization and deserialization.
    
    Maintains a centralized registry of config classes that can be easily extended.
    Implements the Single Source of Truth principle by providing a single place
    to register and retrieve config classes.
    """
    
    # Single registry instance - implementing Single Source of Truth
    _registry: Dict[str, Type] = {}
    _logger = logging.getLogger(__name__)
    
    @classmethod
    def register(cls, config_class: Optional[Type[T]] = None) -> Callable[[Type[T]], Type[T]]:
        """
        Register a config class.
        
        Can be used as a decorator:
        
        @ConfigClassStore.register
        class MyConfig(BasePipelineConfig):
            ...
        
        Args:
            config_class: Optional class to register directly
            
        Returns:
            Decorator function that registers the class or the class itself if provided
        """
        def _register(cls_to_register: Type[T]) -> Type[T]:
            cls_name = cls_to_register.__name__
            if cls_name in ConfigClassStore._registry and ConfigClassStore._registry[cls_name] != cls_to_register:
                cls._logger.warning(
                    f"Class {cls_name} is already registered and is being overwritten. "
                    f"This may cause issues if the classes are different."
                )
            ConfigClassStore._registry[cls_name] = cls_to_register
            cls._logger.debug(f"Registered class: {cls_name}")
            return cls_to_register
            
        if config_class is not None:
            # Used directly as a function
            return _register(config_class)
            
        # Used as a decorator
        return _register
        
    @classmethod
    def get_class(cls, class_name: str) -> Optional[Type]:
        """
        Get a registered class by name.
        
        Args:
            class_name: Name of the class
            
        Returns:
            The class or None if not found
        """
        class_obj = cls._registry.get(class_name)
        if class_obj is None:
            cls._logger.debug(f"Class not found in registry: {class_name}")
        return class_obj
        
    @classmethod
    def get_all_classes(cls) -> Dict[str, Type]:
        """
        Get all registered classes.
        
        Returns:
            dict: Mapping of class names to classes
        """
        return cls._registry.copy()
        
    @classmethod
    def register_many(cls, *config_classes: Type) -> None:
        """
        Register multiple config classes at once.
        
        Args:
            *config_classes: Classes to register
        """
        for config_class in config_classes:
            cls.register(config_class)
            
    @classmethod
    def clear(cls) -> None:
        """
        Clear the registry.
        
        This is useful for testing or when you need to reset the registry.
        """
        cls._registry.clear()
        cls._logger.debug("Cleared config class registry")
        
    @classmethod
    def registered_names(cls) -> Set[str]:
        """
        Get all registered class names.
        
        Returns:
            set: Set of registered class names
        """
        return set(cls._registry.keys())


def build_complete_config_classes() -> Dict[str, Type]:
    """
    Build a complete mapping of config classes from all available sources.
    
    This function scans for all available config classes in the system,
    including those from third-party modules, and registers them.
    
    Returns:
        dict: Mapping of class names to class objects
    """
    # Start with registered classes
    config_classes = ConfigClassStore.get_all_classes()
    
    # TODO: Add logic to scan for classes in pipeline_steps, etc.
    # This is a placeholder for future implementation
    
    return config_classes
