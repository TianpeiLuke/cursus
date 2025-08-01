"""
Registry manager for coordinating multiple isolated specification registries.

This module provides centralized management of multiple registry instances,
ensuring complete isolation between different contexts (pipelines, environments, etc.).
"""

from typing import Dict, List, Optional
import logging
from .specification_registry import SpecificationRegistry

logger = logging.getLogger(__name__)


class RegistryManager:
    """Manager for context-scoped registries with complete isolation."""
    
    def __init__(self):
        """Initialize the registry manager."""
        self._registries: Dict[str, SpecificationRegistry] = {}
        logger.info("Initialized registry manager")
    
    def get_registry(self, context_name: str = "default", create_if_missing: bool = True) -> Optional[SpecificationRegistry]:
        """
        Get the registry for a specific context.
        
        Args:
            context_name: Name of the context (e.g., pipeline name, environment)
            create_if_missing: Whether to create a new registry if one doesn't exist
            
        Returns:
            Context-specific registry or None if not found and create_if_missing is False
        """
        if context_name not in self._registries and create_if_missing:
            self._registries[context_name] = SpecificationRegistry(context_name)
            logger.info(f"Created new registry for context '{context_name}'")
        
        return self._registries.get(context_name)
    
    def list_contexts(self) -> List[str]:
        """
        Get list of all registered context names.
        
        Returns:
            List of context names with registries
        """
        return list(self._registries.keys())
    
    def clear_context(self, context_name: str) -> bool:
        """
        Clear the registry for a specific context.
        
        Args:
            context_name: Name of the context to clear
            
        Returns:
            True if the registry was cleared, False if it didn't exist
        """
        if context_name in self._registries:
            del self._registries[context_name]
            logger.info(f"Cleared registry for context '{context_name}'")
            return True
        return False
    
    def clear_all_contexts(self):
        """Clear all registries."""
        context_count = len(self._registries)
        self._registries.clear()
        logger.info(f"Cleared all {context_count} registries")
    
    def get_context_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for all contexts.
        
        Returns:
            Dictionary mapping context names to their statistics
        """
        stats = {}
        for context_name, registry in self._registries.items():
            stats[context_name] = {
                "step_count": len(registry.list_step_names()),
                "step_type_count": len(registry.list_step_types())
            }
        return stats
    
    def __repr__(self) -> str:
        """String representation of the registry manager."""
        return f"RegistryManager(contexts={len(self._registries)})"




def get_registry(manager: RegistryManager, context_name: str = "default") -> SpecificationRegistry:
    """
    Get the registry for a specific context.
    
    Args:
        manager: Registry manager instance
        context_name: Name of the context (e.g., pipeline name, environment)
        
    Returns:
        Context-specific registry
    """
    return manager.get_registry(context_name)


def list_contexts(manager: RegistryManager) -> List[str]:
    """
    Get list of all registered context names.
    
    Args:
        manager: Registry manager instance
        
    Returns:
        List of context names with registries
    """
    return manager.list_contexts()


def clear_context(manager: RegistryManager, context_name: str) -> bool:
    """
    Clear the registry for a specific context.
    
    Args:
        manager: Registry manager instance
        context_name: Name of the context to clear
        
    Returns:
        True if the registry was cleared, False if it didn't exist
    """
    return manager.clear_context(context_name)


def get_context_stats(manager: RegistryManager) -> Dict[str, Dict[str, int]]:
    """
    Get statistics for all contexts.
    
    Args:
        manager: Registry manager instance
        
    Returns:
        Dictionary mapping context names to their statistics
    """
    return manager.get_context_stats()


# Backward compatibility functions
def get_pipeline_registry(manager: RegistryManager, pipeline_name: str) -> SpecificationRegistry:
    """
    Get registry for a pipeline (backward compatibility).
    
    Args:
        manager: Registry manager instance
        pipeline_name: Name of the pipeline
        
    Returns:
        Pipeline-specific registry
    """
    return get_registry(manager, pipeline_name)


def get_default_registry(manager: RegistryManager) -> SpecificationRegistry:
    """
    Get the default registry (backward compatibility).
    
    Args:
        manager: Registry manager instance
        
    Returns:
        Default registry
    """
    return get_registry(manager, "default")


__all__ = [
    'RegistryManager',
    'get_registry',
    'get_pipeline_registry',
    'get_default_registry',
    'integrate_with_pipeline_builder',
    'list_contexts',
    'clear_context',
    'get_context_stats'
]


# Integration with PipelineBuilderTemplate
def integrate_with_pipeline_builder(pipeline_builder_cls, manager: RegistryManager = None):
    """
    Decorator to integrate context-scoped registries with a pipeline builder class.
    
    This decorator modifies a pipeline builder class to use context-scoped registries.
    
    Args:
        pipeline_builder_cls: Pipeline builder class to modify
        manager: Registry manager instance (if None, a new instance will be created)
        
    Returns:
        Modified pipeline builder class
    """
    original_init = pipeline_builder_cls.__init__
    
    def new_init(self, *args, **kwargs):
        # Call original __init__
        original_init(self, *args, **kwargs)
        
        # Get or create registry manager
        self.registry_manager = manager or RegistryManager()
        
        # Get context name from base_config
        context_name = 'default_pipeline'
        if hasattr(self, 'base_config'):
            try:
                if hasattr(self.base_config, 'pipeline_name') and self.base_config.pipeline_name:
                    context_name = self.base_config.pipeline_name
            except (AttributeError, TypeError):
                pass
        
        # Create context-specific registry
        self.registry = self.registry_manager.get_registry(context_name)
        logger.info(f"Pipeline builder using registry for context '{context_name}'")
    
    # Replace __init__ method
    pipeline_builder_cls.__init__ = new_init
    
    return pipeline_builder_cls
