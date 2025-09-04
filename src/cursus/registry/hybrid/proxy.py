"""
Context-Aware Registry Proxy for thread-local workspace context management.

This module provides thread-local workspace context management using contextvars,
context managers for temporary workspace switching, and global instance coordination.
"""

import contextvars
import threading
from typing import Optional, ContextManager, Dict, Any
from contextlib import contextmanager

from .manager import HybridRegistryManager
from .compatibility import EnhancedBackwardCompatibilityLayer

# Thread-local workspace context using contextvars
_workspace_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('workspace_id', default=None)

# Global instances for coordination
_global_registry_manager: Optional[HybridRegistryManager] = None
_global_compatibility_layer: Optional[EnhancedBackwardCompatibilityLayer] = None
_instance_lock = threading.Lock()


class ContextAwareRegistryProxy:
    """
    Context-aware proxy for registry operations with automatic workspace context management.
    
    This proxy provides clean, automatic workspace context management without requiring
    manual workspace_id parameter passing to every registry function call.
    """
    
    def __init__(self, registry_manager: HybridRegistryManager):
        self.registry_manager = registry_manager
        self.compatibility_layer = EnhancedBackwardCompatibilityLayer(registry_manager)
    
    def get_step_names(self) -> Dict[str, Dict[str, Any]]:
        """Get STEP_NAMES with automatic workspace context."""
        workspace_id = get_workspace_context()
        return self.compatibility_layer.get_step_names(workspace_id)
    
    def get_builder_step_names(self) -> Dict[str, str]:
        """Get BUILDER_STEP_NAMES with automatic workspace context."""
        workspace_id = get_workspace_context()
        return self.compatibility_layer.get_builder_step_names(workspace_id)
    
    def get_config_step_registry(self) -> Dict[str, str]:
        """Get CONFIG_STEP_REGISTRY with automatic workspace context."""
        workspace_id = get_workspace_context()
        return self.compatibility_layer.get_config_step_registry(workspace_id)
    
    def get_spec_step_types(self) -> Dict[str, str]:
        """Get SPEC_STEP_TYPES with automatic workspace context."""
        workspace_id = get_workspace_context()
        return self.compatibility_layer.get_spec_step_types(workspace_id)
    
    def get_step_definition(self, step_name: str) -> Optional['HybridStepDefinition']:
        """Get step definition with automatic workspace context."""
        workspace_id = get_workspace_context()
        return self.registry_manager.get_step_definition(step_name, workspace_id)
    
    def get_step_definition_with_resolution(self, step_name: str, 
                                          preferred_framework: Optional[str] = None,
                                          environment_tags: Optional[list] = None) -> Optional['HybridStepDefinition']:
        """Get step definition with intelligent resolution and automatic workspace context."""
        workspace_id = get_workspace_context()
        return self.registry_manager.get_step_definition_with_resolution(
            step_name=step_name,
            workspace_id=workspace_id,
            preferred_framework=preferred_framework,
            environment_tags=environment_tags or []
        )
    
    def get_all_step_definitions(self) -> Dict[str, 'HybridStepDefinition']:
        """Get all step definitions with automatic workspace context."""
        workspace_id = get_workspace_context()
        return self.registry_manager.get_all_step_definitions(workspace_id)


# Thread-local workspace context management functions

def set_workspace_context(workspace_id: str) -> None:
    """
    Set current workspace context for thread-local registry operations.
    
    Args:
        workspace_id: Workspace identifier to set as current context
    """
    _workspace_context.set(workspace_id)
    
    # Update global compatibility layer if it exists
    compatibility_layer = get_enhanced_compatibility()
    if compatibility_layer:
        compatibility_layer.set_workspace_context(workspace_id)


def get_workspace_context() -> Optional[str]:
    """
    Get current workspace context.
    
    Returns:
        Current workspace identifier or None if no context is set
    """
    return _workspace_context.get()


def clear_workspace_context() -> None:
    """Clear current workspace context."""
    _workspace_context.set(None)
    
    # Update global compatibility layer if it exists
    compatibility_layer = get_enhanced_compatibility()
    if compatibility_layer:
        compatibility_layer.clear_workspace_context()


@contextmanager
def workspace_context(workspace_id: str) -> ContextManager[None]:
    """
    Context manager for temporary workspace context.
    
    Args:
        workspace_id: Workspace identifier for temporary context
        
    Example:
        with workspace_context("developer_1"):
            config_class = get_config_class_name("MyCustomStep")
        # Context automatically restored
    """
    old_context = get_workspace_context()
    try:
        set_workspace_context(workspace_id)
        yield
    finally:
        if old_context:
            set_workspace_context(old_context)
        else:
            clear_workspace_context()


# Global instance coordination functions

def get_global_registry_manager() -> HybridRegistryManager:
    """
    Get global registry manager instance with thread-safe initialization.
    
    Returns:
        Global HybridRegistryManager instance
    """
    global _global_registry_manager
    
    if _global_registry_manager is None:
        with _instance_lock:
            # Double-check locking pattern
            if _global_registry_manager is None:
                _global_registry_manager = HybridRegistryManager()
    
    return _global_registry_manager


def get_enhanced_compatibility() -> EnhancedBackwardCompatibilityLayer:
    """
    Get enhanced compatibility layer instance with thread-safe initialization.
    
    Returns:
        Global EnhancedBackwardCompatibilityLayer instance
    """
    global _global_compatibility_layer
    
    if _global_compatibility_layer is None:
        with _instance_lock:
            # Double-check locking pattern
            if _global_compatibility_layer is None:
                registry_manager = get_global_registry_manager()
                _global_compatibility_layer = EnhancedBackwardCompatibilityLayer(registry_manager)
    
    return _global_compatibility_layer


def get_context_aware_proxy() -> ContextAwareRegistryProxy:
    """
    Get context-aware registry proxy instance.
    
    Returns:
        ContextAwareRegistryProxy instance with automatic workspace context
    """
    registry_manager = get_global_registry_manager()
    return ContextAwareRegistryProxy(registry_manager)


def reset_global_instances() -> None:
    """
    Reset global instances (primarily for testing).
    
    Warning: This should only be used in test scenarios.
    """
    global _global_registry_manager, _global_compatibility_layer
    
    with _instance_lock:
        _global_registry_manager = None
        _global_compatibility_layer = None
    
    clear_workspace_context()


# Environment variable integration

def get_workspace_from_environment() -> Optional[str]:
    """
    Get workspace context from environment variable.
    
    Returns:
        Workspace ID from CURSUS_WORKSPACE_ID environment variable or None
    """
    import os
    return os.environ.get('CURSUS_WORKSPACE_ID')


def auto_set_workspace_from_environment() -> bool:
    """
    Automatically set workspace context from environment variable if not already set.
    
    Returns:
        True if workspace context was set from environment, False otherwise
    """
    current_context = get_workspace_context()
    if current_context is None:
        env_workspace = get_workspace_from_environment()
        if env_workspace:
            set_workspace_context(env_workspace)
            return True
    return False


# Context validation and debugging

def validate_workspace_context() -> Dict[str, Any]:
    """
    Validate current workspace context and provide debugging information.
    
    Returns:
        Dictionary with context validation results and debugging info
    """
    current_context = get_workspace_context()
    env_context = get_workspace_from_environment()
    
    validation_result = {
        'current_context': current_context,
        'environment_context': env_context,
        'context_source': None,
        'is_valid': False,
        'available_workspaces': [],
        'recommendations': []
    }
    
    # Determine context source
    if current_context:
        if current_context == env_context:
            validation_result['context_source'] = 'environment_variable'
        else:
            validation_result['context_source'] = 'explicit_call'
    elif env_context:
        validation_result['context_source'] = 'environment_available'
        validation_result['recommendations'].append(
            f"Call auto_set_workspace_from_environment() to use environment context: {env_context}"
        )
    else:
        validation_result['context_source'] = 'none'
        validation_result['recommendations'].append(
            "Set workspace context with set_workspace_context() or CURSUS_WORKSPACE_ID environment variable"
        )
    
    # Check if current context is valid
    if current_context:
        try:
            registry_manager = get_global_registry_manager()
            if current_context in registry_manager._local_registries:
                validation_result['is_valid'] = True
            else:
                validation_result['recommendations'].append(
                    f"Workspace '{current_context}' not found. Initialize with: "
                    f"python -m cursus.cli.registry init-workspace {current_context}"
                )
        except Exception as e:
            validation_result['recommendations'].append(f"Registry validation failed: {e}")
    
    # Get available workspaces
    try:
        registry_manager = get_global_registry_manager()
        validation_result['available_workspaces'] = list(registry_manager._local_registries.keys())
    except Exception:
        validation_result['available_workspaces'] = []
    
    return validation_result


def debug_workspace_context() -> str:
    """
    Get detailed debugging information about workspace context.
    
    Returns:
        Formatted debugging information string
    """
    validation = validate_workspace_context()
    
    debug_info = []
    debug_info.append("=== Workspace Context Debug Information ===")
    debug_info.append(f"Current Context: {validation['current_context'] or 'None'}")
    debug_info.append(f"Environment Context: {validation['environment_context'] or 'None'}")
    debug_info.append(f"Context Source: {validation['context_source']}")
    debug_info.append(f"Is Valid: {validation['is_valid']}")
    
    if validation['available_workspaces']:
        debug_info.append(f"Available Workspaces: {', '.join(validation['available_workspaces'])}")
    else:
        debug_info.append("Available Workspaces: None found")
    
    if validation['recommendations']:
        debug_info.append("\nRecommendations:")
        for i, rec in enumerate(validation['recommendations'], 1):
            debug_info.append(f"  {i}. {rec}")
    
    return '\n'.join(debug_info)


# Workspace context decorators for advanced usage

def with_workspace_context(workspace_id: str):
    """
    Decorator to run function with specific workspace context.
    
    Args:
        workspace_id: Workspace identifier to use for function execution
        
    Example:
        @with_workspace_context("developer_1")
        def my_function():
            return get_config_class_name("MyCustomStep")
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with workspace_context(workspace_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def auto_workspace_context(func):
    """
    Decorator to automatically set workspace context from environment if not set.
    
    Example:
        @auto_workspace_context
        def my_function():
            return get_config_class_name("SomeStep")  # Uses env context if available
    """
    def wrapper(*args, **kwargs):
        auto_set_workspace_from_environment()
        return func(*args, **kwargs)
    return wrapper


# Module-level convenience functions

def ensure_workspace_context(workspace_id: Optional[str] = None) -> str:
    """
    Ensure workspace context is set, using provided ID, environment, or raising error.
    
    Args:
        workspace_id: Optional workspace ID to use if no context is set
        
    Returns:
        Active workspace context ID
        
    Raises:
        ValueError: If no workspace context can be determined
    """
    current_context = get_workspace_context()
    
    if current_context:
        return current_context
    
    if workspace_id:
        set_workspace_context(workspace_id)
        return workspace_id
    
    env_context = get_workspace_from_environment()
    if env_context:
        set_workspace_context(env_context)
        return env_context
    
    raise ValueError(
        "No workspace context available. Set context with:\n"
        "  1. set_workspace_context('workspace_id')\n"
        "  2. export CURSUS_WORKSPACE_ID=workspace_id\n"
        "  3. Pass workspace_id parameter to ensure_workspace_context()"
    )


def get_effective_workspace_context() -> Optional[str]:
    """
    Get effective workspace context, checking thread-local context first, then environment.
    
    Returns:
        Effective workspace context or None if no context available
    """
    # Check thread-local context first
    current_context = get_workspace_context()
    if current_context:
        return current_context
    
    # Check environment variable as fallback
    return get_workspace_from_environment()


# Integration with existing compatibility layer

def update_compatibility_layer_context() -> None:
    """Update global compatibility layer with current workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    current_context = get_workspace_context()
    
    if current_context:
        compatibility_layer.set_workspace_context(current_context)
    else:
        compatibility_layer.clear_workspace_context()


# Context synchronization utilities

def sync_all_contexts(workspace_id: Optional[str] = None) -> None:
    """
    Synchronize all context-aware components with specified or current workspace context.
    
    Args:
        workspace_id: Workspace ID to sync to, or None to use current context
    """
    target_workspace = workspace_id or get_workspace_context()
    
    if target_workspace:
        set_workspace_context(target_workspace)
    else:
        clear_workspace_context()
    
    # Ensure compatibility layer is synchronized
    update_compatibility_layer_context()


def get_context_status() -> Dict[str, Any]:
    """
    Get comprehensive status of all context-aware components.
    
    Returns:
        Dictionary with status of all context-aware components
    """
    current_context = get_workspace_context()
    env_context = get_workspace_from_environment()
    
    status = {
        'thread_local_context': current_context,
        'environment_context': env_context,
        'effective_context': get_effective_workspace_context(),
        'contexts_synchronized': True,
        'global_instances_initialized': False,
        'compatibility_layer_context': None
    }
    
    # Check global instances
    try:
        registry_manager = get_global_registry_manager()
        compatibility_layer = get_enhanced_compatibility()
        status['global_instances_initialized'] = True
        
        # Check compatibility layer context
        if hasattr(compatibility_layer, '_current_workspace_context'):
            status['compatibility_layer_context'] = compatibility_layer._current_workspace_context
            status['contexts_synchronized'] = (
                status['compatibility_layer_context'] == current_context
            )
    except Exception as e:
        status['initialization_error'] = str(e)
    
    return status
