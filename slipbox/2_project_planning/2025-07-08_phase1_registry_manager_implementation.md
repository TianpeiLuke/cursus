# Phase 1: Registry Manager Implementation

This document provides a detailed implementation plan for removing the global singleton from the `registry_manager.py` file and updating the related convenience functions.

## Current Implementation

The current implementation in `registry_manager.py` has a global singleton instance:

```python
# Global registry manager instance
registry_manager = RegistryManager()
```

And convenience functions that use this global instance:

```python
def get_registry(context_name: str = "default") -> SpecificationRegistry:
    """
    Get the registry for a specific context.
    
    This is a convenience function that uses the global registry manager.
    
    Args:
        context_name: Name of the context (e.g., pipeline name, environment)
        
    Returns:
        Context-specific registry
    """
    return registry_manager.get_registry(context_name)


def list_contexts() -> List[str]:
    """
    Get list of all registered context names.
    
    This is a convenience function that uses the global registry manager.
    
    Returns:
        List of context names with registries
    """
    return registry_manager.list_contexts()


def clear_context(context_name: str) -> bool:
    """
    Clear the registry for a specific context.
    
    This is a convenience function that uses the global registry manager.
    
    Args:
        context_name: Name of the context to clear
        
    Returns:
        True if the registry was cleared, False if it didn't exist
    """
    return registry_manager.clear_context(context_name)


def get_context_stats() -> Dict[str, Dict[str, int]]:
    """
    Get statistics for all contexts.
    
    This is a convenience function that uses the global registry manager.
    
    Returns:
        Dictionary mapping context names to their statistics
    """
    return registry_manager.get_context_stats()


# Backward compatibility functions
def get_pipeline_registry(pipeline_name: str) -> SpecificationRegistry:
    """
    Get registry for a pipeline (backward compatibility).
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        Pipeline-specific registry
    """
    return get_registry(pipeline_name)


def get_default_registry() -> SpecificationRegistry:
    """
    Get the default registry (backward compatibility).
    
    Returns:
        Default registry
    """
    return get_registry("default")
```

## Proposed Changes

### 1. Remove Global Instance

Remove the global registry manager instance:

```python
# REMOVE THIS LINE
registry_manager = RegistryManager()
```

### 2. Update Convenience Functions

Update all convenience functions to accept a manager instance:

```python
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
```

### 3. Update the `integrate_with_pipeline_builder` Function

The `integrate_with_pipeline_builder` function currently uses the global registry manager:

```python
def integrate_with_pipeline_builder(pipeline_builder_cls):
    """
    Decorator to integrate context-scoped registries with a pipeline builder class.
    
    This decorator modifies a pipeline builder class to use context-scoped registries.
    
    Args:
        pipeline_builder_cls: Pipeline builder class to modify
        
    Returns:
        Modified pipeline builder class
    """
    original_init = pipeline_builder_cls.__init__
    
    def new_init(self, *args, **kwargs):
        # Call original __init__
        original_init(self, *args, **kwargs)
        
        # Get context name from base_config
        context_name = 'default_pipeline'
        if hasattr(self, 'base_config'):
            try:
                if hasattr(self.base_config, 'pipeline_name') and self.base_config.pipeline_name:
                    context_name = self.base_config.pipeline_name
            except (AttributeError, TypeError):
                pass
        
        # Create context-specific registry
        self.registry = get_registry(context_name)
        logger.info(f"Pipeline builder using registry for context '{context_name}'")
    
    # Replace __init__ method
    pipeline_builder_cls.__init__ = new_init
    
    return pipeline_builder_cls
```

Update it to accept a registry manager instance:

```python
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
```

### 4. Update `__all__` List

Update the `__all__` list to reflect the changes:

```python
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
```

## Impact Analysis

### Affected Files

1. **Test Files**:
   - `test_registry_manager.py`
   - `test_global_state_isolation.py`
   - Any other tests that use the global registry manager

2. **Source Files**:
   - Any files that import and use the convenience functions
   - Any files that directly use the global registry manager

### Required Changes in Other Files

1. **In Test Files**:
   - Create a registry manager instance in each test or test fixture
   - Pass the instance to convenience functions

2. **In Source Files**:
   - Create a registry manager instance or pass an existing one
   - Update function calls to pass the instance

## Implementation Steps

1. Create a new branch for the changes
2. Update `registry_manager.py` as described above
3. Fix the immediate compilation errors in dependent files
4. Update tests to use the new approach
5. Run tests and fix any remaining issues
6. Document the changes
7. Submit a pull request

## Backward Compatibility Considerations

To maintain backward compatibility during the transition, we could:

1. Add a deprecation warning when the global instance is used
2. Provide a transitional period where both approaches are supported
3. Create a compatibility layer that maintains the old API but uses the new implementation internally

However, for a clean break, it's recommended to update all usages at once to avoid confusion and ensure consistent usage throughout the codebase.
