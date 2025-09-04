"""
Streamlined Hybrid Registry Backward Compatibility Layer

This module provides backward compatibility for the hybrid registry system,
ensuring existing code continues to work with minimal overhead.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any
from .manager import UnifiedRegistryManager

logger = logging.getLogger(__name__)


class BackwardCompatibilityAdapter:
    """
    Streamlined compatibility adapter that provides legacy registry interface.
    
    This adapter consolidates all compatibility functionality into a single class
    with simple parameter-based workspace context management.
    """
    
    def __init__(self, registry_manager: UnifiedRegistryManager):
        self.registry_manager = registry_manager
        self._workspace_context: Optional[str] = None
    
    def set_workspace_context(self, workspace_id: str) -> None:
        """Set workspace context for subsequent operations."""
        self._workspace_context = workspace_id
    
    def clear_workspace_context(self) -> None:
        """Clear workspace context."""
        self._workspace_context = None
    
    def get_step_names(self, workspace_id: str = None) -> Dict[str, Dict[str, Any]]:
        """Get step names in legacy STEP_NAMES format."""
        effective_workspace = workspace_id or self._workspace_context
        definitions = self.registry_manager.get_all_step_definitions(effective_workspace)
        return {name: defn.to_legacy_format() for name, defn in definitions.items()}
    
    def get_builder_step_names(self, workspace_id: str = None) -> Dict[str, str]:
        """Get builder step names in legacy BUILDER_STEP_NAMES format."""
        step_names = self.get_step_names(workspace_id)
        return {name: info["builder_step_name"] for name, info in step_names.items()}
    
    def get_config_step_registry(self, workspace_id: str = None) -> Dict[str, str]:
        """Get config step registry in legacy CONFIG_STEP_REGISTRY format."""
        step_names = self.get_step_names(workspace_id)
        return {info["config_class"]: name for name, info in step_names.items()}
    
    def get_spec_step_types(self, workspace_id: str = None) -> Dict[str, str]:
        """Get spec step types in legacy SPEC_STEP_TYPES format."""
        step_names = self.get_step_names(workspace_id)
        return {name: info["spec_type"] for name, info in step_names.items()}
    
    def get_step_definition(self, step_name: str, workspace_id: str = None) -> Optional[Dict[str, Any]]:
        """Get single step definition in legacy format."""
        effective_workspace = workspace_id or self._workspace_context
        definition = self.registry_manager.get_step_definition(step_name, effective_workspace)
        return definition.to_legacy_format() if definition else None
    
    def has_step(self, step_name: str, workspace_id: str = None) -> bool:
        """Check if step exists."""
        effective_workspace = workspace_id or self._workspace_context
        return self.registry_manager.get_step_definition(step_name, effective_workspace) is not None
    
    def list_all_steps(self, workspace_id: str = None) -> List[str]:
        """List all step names."""
        step_names = self.get_step_names(workspace_id)
        return list(step_names.keys())


# Global compatibility adapter instance
_global_compatibility_adapter: Optional[BackwardCompatibilityAdapter] = None


def get_compatibility_adapter() -> Optional[BackwardCompatibilityAdapter]:
    """Get the global compatibility adapter instance."""
    return _global_compatibility_adapter


def set_compatibility_adapter(adapter: BackwardCompatibilityAdapter) -> None:
    """Set the global compatibility adapter instance."""
    global _global_compatibility_adapter
    _global_compatibility_adapter = adapter


def create_compatibility_adapter(registry_manager: UnifiedRegistryManager) -> BackwardCompatibilityAdapter:
    """Factory function to create a compatibility adapter."""
    adapter = BackwardCompatibilityAdapter(registry_manager)
    set_compatibility_adapter(adapter)
    return adapter


# Legacy API functions for backward compatibility
def get_step_names(workspace_id: str = None) -> Dict[str, Dict[str, Any]]:
    """Global function to get STEP_NAMES for backward compatibility."""
    adapter = get_compatibility_adapter()
    if adapter:
        return adapter.get_step_names(workspace_id)
    return {}


def get_builder_step_names(workspace_id: str = None) -> Dict[str, str]:
    """Global function to get BUILDER_STEP_NAMES for backward compatibility."""
    adapter = get_compatibility_adapter()
    if adapter:
        return adapter.get_builder_step_names(workspace_id)
    return {}


def get_config_step_registry(workspace_id: str = None) -> Dict[str, str]:
    """Global function to get CONFIG_STEP_REGISTRY for backward compatibility."""
    adapter = get_compatibility_adapter()
    if adapter:
        return adapter.get_config_step_registry(workspace_id)
    return {}


def get_spec_step_types(workspace_id: str = None) -> Dict[str, str]:
    """Global function to get SPEC_STEP_TYPES for backward compatibility."""
    adapter = get_compatibility_adapter()
    if adapter:
        return adapter.get_spec_step_types(workspace_id)
    return {}


def set_workspace_context(workspace_id: str) -> None:
    """Set workspace context globally."""
    adapter = get_compatibility_adapter()
    if adapter:
        adapter.set_workspace_context(workspace_id)


def clear_workspace_context() -> None:
    """Clear workspace context globally."""
    adapter = get_compatibility_adapter()
    if adapter:
        adapter.clear_workspace_context()


# Simplified context manager for workspace switching
class workspace_context:
    """Simple context manager for temporary workspace context."""
    
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.previous_context = None
    
    def __enter__(self):
        adapter = get_compatibility_adapter()
        if adapter:
            self.previous_context = adapter._workspace_context
            adapter.set_workspace_context(self.workspace_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        adapter = get_compatibility_adapter()
        if adapter:
            if self.previous_context:
                adapter.set_workspace_context(self.previous_context)
            else:
                adapter.clear_workspace_context()
