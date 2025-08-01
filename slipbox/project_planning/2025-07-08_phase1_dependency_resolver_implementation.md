# Phase 1: Dependency Resolver Implementation

This document provides a detailed implementation plan for removing the global singleton from the `dependency_resolver.py` file and updating related code.

## Current Implementation

The current implementation in `dependency_resolver.py` has a global singleton instance:

```python
# Global resolver instance
global_resolver = UnifiedDependencyResolver()
```

The `UnifiedDependencyResolver` class is used for intelligent dependency resolution between pipeline steps. It currently creates its own `SemanticMatcher` instance and accepts an optional `SpecificationRegistry`:

```python
class UnifiedDependencyResolver:
    """Intelligent dependency resolver using declarative specifications."""
    
    def __init__(self, registry: Optional[SpecificationRegistry] = None):
        """
        Initialize the dependency resolver.
        
        Args:
            registry: Optional specification registry. If None, creates a new one.
        """
        self.registry = registry or SpecificationRegistry()
        self.semantic_matcher = SemanticMatcher()
        self._resolution_cache: Dict[str, Dict[str, PropertyReference]] = {}
```

## Proposed Changes

### 1. Remove Global Instance

Remove the global dependency resolver instance:

```python
# REMOVE THIS LINE
global_resolver = UnifiedDependencyResolver()
```

### 2. Update `UnifiedDependencyResolver` Class

Update the `UnifiedDependencyResolver` class to accept both a `SpecificationRegistry` and a `SemanticMatcher` instance:

```python
class UnifiedDependencyResolver:
    """Intelligent dependency resolver using declarative specifications."""
    
    def __init__(self, registry: SpecificationRegistry, semantic_matcher: SemanticMatcher):
        """
        Initialize the dependency resolver.
        
        Args:
            registry: Specification registry
            semantic_matcher: Semantic matcher for name similarity calculations
        """
        self.registry = registry
        self.semantic_matcher = semantic_matcher
        self._resolution_cache: Dict[str, Dict[str, PropertyReference]] = {}
```

### 3. Add Factory Function

Add a factory function to simplify the creation of properly configured instances:

```python
def create_dependency_resolver(registry: Optional[SpecificationRegistry] = None,
                             semantic_matcher: Optional[SemanticMatcher] = None) -> UnifiedDependencyResolver:
    """
    Create a properly configured dependency resolver.
    
    Args:
        registry: Optional specification registry. If None, creates a new one.
        semantic_matcher: Optional semantic matcher. If None, creates a new one.
        
    Returns:
        Configured UnifiedDependencyResolver instance
    """
    registry = registry or SpecificationRegistry()
    semantic_matcher = semantic_matcher or SemanticMatcher()
    return UnifiedDependencyResolver(registry, semantic_matcher)
```

### 4. Update `__all__` List

Update the `__all__` list to reflect the changes:

```python
__all__ = [
    'UnifiedDependencyResolver',
    'DependencyResolutionError',
    'create_dependency_resolver'
]
```

## Impact Analysis

### Affected Files

1. **Direct Dependencies**:
   - Any files that import and use the global `global_resolver` instance
   - Any files that create instances of `UnifiedDependencyResolver`

2. **Indirect Dependencies**:
   - Step builders that use the dependency resolver
   - Test files that use the dependency resolver

### Required Changes in Other Files

1. **In Files Using `global_resolver`**:
   - Create a new `UnifiedDependencyResolver` instance
   - Use the factory function `create_dependency_resolver()`

2. **In Files Creating `UnifiedDependencyResolver` Instances**:
   - Update constructor calls to provide both a registry and a semantic matcher
   - Or use the factory function `create_dependency_resolver()`

3. **In Step Builders**:
   - Update to accept a dependency resolver instance
   - Pass the instance to methods that need it

## Implementation Steps

1. Create a new branch for the changes
2. Update `dependency_resolver.py` to remove the global instance
3. Update the `UnifiedDependencyResolver` constructor
4. Add the factory function
5. Fix any files that directly use the global instance
6. Run tests and fix any issues
7. Document the changes
8. Submit a pull request

## Example Code Changes

### In `dependency_resolver.py`:

```python
# BEFORE
class UnifiedDependencyResolver:
    def __init__(self, registry: Optional[SpecificationRegistry] = None):
        self.registry = registry or SpecificationRegistry()
        self.semantic_matcher = SemanticMatcher()
        self._resolution_cache: Dict[str, Dict[str, PropertyReference]] = {}

# Global resolver instance
global_resolver = UnifiedDependencyResolver()


# AFTER
class UnifiedDependencyResolver:
    def __init__(self, registry: SpecificationRegistry, semantic_matcher: SemanticMatcher):
        self.registry = registry
        self.semantic_matcher = semantic_matcher
        self._resolution_cache: Dict[str, Dict[str, PropertyReference]] = {}

def create_dependency_resolver(registry: Optional[SpecificationRegistry] = None,
                             semantic_matcher: Optional[SemanticMatcher] = None) -> UnifiedDependencyResolver:
    """
    Create a properly configured dependency resolver.
    
    Args:
        registry: Optional specification registry. If None, creates a new one.
        semantic_matcher: Optional semantic matcher. If None, creates a new one.
        
    Returns:
        Configured UnifiedDependencyResolver instance
    """
    registry = registry or SpecificationRegistry()
    semantic_matcher = semantic_matcher or SemanticMatcher()
    return UnifiedDependencyResolver(registry, semantic_matcher)

# Remove global resolver instance
# global_resolver = UnifiedDependencyResolver()
```

### In Files Using `global_resolver`:

```python
# BEFORE
from ..pipeline_deps.dependency_resolver import global_resolver

# Use global resolver
resolved = global_resolver.resolve_step_dependencies(step_name, available_steps)


# AFTER
from ..pipeline_deps.dependency_resolver import create_dependency_resolver

# Create a resolver instance
resolver = create_dependency_resolver()

# Use the instance
resolved = resolver.resolve_step_dependencies(step_name, available_steps)
```

### In Step Builders:

```python
# BEFORE
class StepBuilderBase(ABC):
    def __init__(self, config, spec=None, sagemaker_session=None, role=None, notebook_root=None):
        # ...
        
    def extract_inputs_from_dependencies(self, dependency_steps: List[Step]) -> Dict[str, Any]:
        # ...
        resolver = UnifiedDependencyResolver()
        # ...


# AFTER
class StepBuilderBase(ABC):
    def __init__(self, config, spec=None, sagemaker_session=None, role=None, notebook_root=None, 
                 dependency_resolver=None):
        # ...
        self.dependency_resolver = dependency_resolver
        
    def extract_inputs_from_dependencies(self, dependency_steps: List[Step]) -> Dict[str, Any]:
        # ...
        resolver = self.dependency_resolver or create_dependency_resolver()
        # ...
```

## Backward Compatibility Considerations

To maintain backward compatibility during the transition, we could:

1. Add default parameters to the `UnifiedDependencyResolver` constructor:

```python
def __init__(self, registry: Optional[SpecificationRegistry] = None, 
           semantic_matcher: Optional[SemanticMatcher] = None):
    self.registry = registry or SpecificationRegistry()
    self.semantic_matcher = semantic_matcher or SemanticMatcher()
```

2. Add a deprecation warning when the defaults are used:

```python
def __init__(self, registry: Optional[SpecificationRegistry] = None, 
           semantic_matcher: Optional[SemanticMatcher] = None):
    if registry is None or semantic_matcher is None:
        import warnings
        warnings.warn(
            "Creating registry or semantic_matcher instances automatically is deprecated. "
            "Please provide instances explicitly.",
            DeprecationWarning, stacklevel=2
        )
    self.registry = registry or SpecificationRegistry()
    self.semantic_matcher = semantic_matcher or SemanticMatcher()
```

However, for a clean break, it's recommended to update all usages at once to avoid confusion and ensure consistent usage throughout the codebase.

## Integration with Other Changes

This change should be coordinated with the removal of global singletons from:

1. `registry_manager.py` - Removing the global `registry_manager` instance
2. `semantic_matcher.py` - Removing the global `semantic_matcher` instance

The dependency resolver depends on both of these components, so the changes should be made in a coordinated way to ensure consistency.
