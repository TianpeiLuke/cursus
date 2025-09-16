"""
Registry manager for coordinating multiple isolated specification registries.

This module provides centralized management of multiple registry instances,
ensuring complete isolation between different contexts (pipelines, environments, etc.).
"""

from typing import Dict, List, Optional, Any
import logging
from .specification_registry import SpecificationRegistry

logger = logging.getLogger(__name__)


class RegistryManager:
    """
    Manager for context-scoped registries with complete isolation and workspace awareness.

    This enhanced registry manager now supports:
    1. Workspace-aware registry resolution
    2. Integration with hybrid registry system
    3. Backward compatibility with existing code
    4. Thread-local workspace context management
    """

    def __init__(self, workspace_context: Optional[str] = None):
        """
        Initialize the registry manager with optional workspace context.

        Args:
            workspace_context: Optional workspace context for registry isolation
        """
        self._registries: Dict[str, SpecificationRegistry] = {}
        self._workspace_context = workspace_context
        self._hybrid_manager: Optional[Any] = None
        logger.info(
            f"Initialized registry manager with workspace context: {workspace_context}"
        )

    def _get_hybrid_manager(self) -> Optional[Any]:
        """Get or create hybrid registry manager."""
        if self._hybrid_manager is None:
            try:
                from ...registry.hybrid.manager import HybridRegistryManager

                self._hybrid_manager = HybridRegistryManager()
                logger.debug("Created hybrid registry manager")
            except ImportError:
                logger.debug("Hybrid registry not available")
                self._hybrid_manager = None
        return self._hybrid_manager

    def _get_workspace_aware_context_name(self, context_name: str) -> str:
        """
        Get workspace-aware context name by combining workspace context with local context.

        Args:
            context_name: Local context name

        Returns:
            Workspace-aware context name
        """
        if self._workspace_context:
            return f"{self._workspace_context}::{context_name}"
        return context_name

    def get_registry(
        self, context_name: str = "default", create_if_missing: bool = True
    ) -> SpecificationRegistry:
        """
        Get the registry for a specific context with workspace awareness.

        This enhanced method now supports:
        1. Workspace-aware context naming
        2. Integration with hybrid registry system
        3. Automatic registry population from hybrid sources
        4. Backward compatibility with existing code

        Args:
            context_name: Name of the context (e.g., pipeline name, environment)
            create_if_missing: Whether to create a new registry if one doesn't exist

        Returns:
            Context-specific registry or None if not found and create_if_missing is False
        """
        # Get workspace-aware context name
        workspace_aware_context = self._get_workspace_aware_context_name(context_name)

        if workspace_aware_context not in self._registries and create_if_missing:
            # Create new registry
            registry = SpecificationRegistry(workspace_aware_context)

            # Try to populate from hybrid registry if available
            hybrid_manager = self._get_hybrid_manager()
            if hybrid_manager:
                try:
                    # Get step definitions from hybrid registry for this workspace context
                    if self._workspace_context:
                        step_definitions = hybrid_manager.get_all_step_definitions(
                            self._workspace_context
                        )
                    else:
                        step_definitions = hybrid_manager.get_all_step_definitions()

                    # Convert step definitions to specifications and register them
                    for step_name, step_def in step_definitions.items():
                        # Create a minimal specification from step definition
                        # This is a simplified conversion - in practice you might want more sophisticated mapping
                        registry.register(step_name, step_def)

                    logger.debug(
                        f"Populated registry '{workspace_aware_context}' with {len(step_definitions)} specifications from hybrid registry"
                    )

                except Exception as e:
                    logger.debug(f"Could not populate registry from hybrid source: {e}")

            self._registries[workspace_aware_context] = registry
            logger.info(
                f"Created new workspace-aware registry for context '{workspace_aware_context}'"
            )

        if workspace_aware_context not in self._registries:
            raise ValueError(f"Registry not found for context '{workspace_aware_context}' and create_if_missing is False")
        # Direct access: key is guaranteed to exist due to the check above
        return self._registries[workspace_aware_context]

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

    def clear_all_contexts(self) -> None:
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
                "step_type_count": len(registry.list_step_types()),
            }
        return stats

    def __repr__(self) -> str:
        """String representation of the registry manager."""
        return f"RegistryManager(contexts={len(self._registries)})"


def get_registry(
    manager: Optional[RegistryManager] = None, context_name: str = "default"
) -> SpecificationRegistry:
    """
    Get the registry for a specific context.

    Args:
        manager: Registry manager instance
        context_name: Name of the context (e.g., pipeline name, environment)

    Returns:
        Context-specific registry
    """
    if manager is None:
        manager = RegistryManager()
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
def get_pipeline_registry(
    manager: RegistryManager, pipeline_name: str
) -> SpecificationRegistry:
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
    "RegistryManager",
    "get_registry",
    "get_pipeline_registry",
    "get_default_registry",
    "integrate_with_pipeline_builder",
    "list_contexts",
    "clear_context",
    "get_context_stats",
]


# Integration with PipelineBuilderTemplate
def integrate_with_pipeline_builder(
    pipeline_builder_cls: Any, manager: Optional[RegistryManager] = None
) -> Any:
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

    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Call original __init__
        original_init(self, *args, **kwargs)

        # Get or create registry manager
        self.registry_manager = manager or RegistryManager()

        # Get context name from base_config
        context_name = "default_pipeline"
        if hasattr(self, "base_config"):
            try:
                if (
                    hasattr(self.base_config, "pipeline_name")
                    and self.base_config.pipeline_name
                ):
                    context_name = self.base_config.pipeline_name
            except (AttributeError, TypeError):
                pass

        # Create context-specific registry
        self.registry = self.registry_manager.get_registry(context_name)
        logger.info(f"Pipeline builder using registry for context '{context_name}'")

    # Replace __init__ method
    pipeline_builder_cls.__init__ = new_init

    return pipeline_builder_cls
