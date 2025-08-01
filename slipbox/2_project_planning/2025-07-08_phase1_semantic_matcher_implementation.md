# Phase 1: Semantic Matcher Implementation

This document provides a detailed implementation plan for removing the global singleton from the `semantic_matcher.py` file and updating related code.

## Current Implementation

The current implementation in `semantic_matcher.py` has a global singleton instance:

```python
# Global semantic matcher instance
semantic_matcher = SemanticMatcher()
```

The `SemanticMatcher` class is used for calculating semantic similarity between names, which is crucial for intelligent dependency resolution. The global instance is used in the `UnifiedDependencyResolver` class:

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
        self.semantic_matcher = SemanticMatcher()  # Creates a new instance instead of using the global one
        self._resolution_cache: Dict[str, Dict[str, PropertyReference]] = {}
```

## Proposed Changes

### 1. Remove Global Instance

Remove the global semantic matcher instance:

```python
# REMOVE THIS LINE
semantic_matcher = SemanticMatcher()
```

### 2. Update `UnifiedDependencyResolver` Class

Update the `UnifiedDependencyResolver` class to accept a `SemanticMatcher` instance:

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

### 3. Update `__all__` List

Update the `__all__` list to reflect the changes:

```python
__all__ = [
    'SemanticMatcher'
]
```

## Impact Analysis

### Affected Files

1. **Direct Dependencies**:
   - `dependency_resolver.py`: Uses `SemanticMatcher` in `UnifiedDependencyResolver`

2. **Indirect Dependencies**:
   - Any files that create instances of `UnifiedDependencyResolver`
   - Any files that directly use the global `semantic_matcher` instance

### Required Changes in Other Files

1. **In `dependency_resolver.py`**:
   - Update the `UnifiedDependencyResolver` constructor to accept a `SemanticMatcher` instance
   - Remove the creation of a new `SemanticMatcher` instance in the constructor
   - Update the global `global_resolver` instance creation (if it's kept during the transition)

2. **In Files Using `UnifiedDependencyResolver`**:
   - Create a `SemanticMatcher` instance
   - Pass it to the `UnifiedDependencyResolver` constructor

## Implementation Steps

1. Create a new branch for the changes
2. Update `semantic_matcher.py` to remove the global instance
3. Update `dependency_resolver.py` to accept a `SemanticMatcher` instance
4. Fix any other files that directly use the global instance
5. Run tests and fix any issues
6. Document the changes
7. Submit a pull request

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

# Remove global resolver instance or update it to use explicit instances
# global_resolver = UnifiedDependencyResolver()
```

### In Files Using `UnifiedDependencyResolver`:

```python
# BEFORE
resolver = UnifiedDependencyResolver(registry)

# AFTER
semantic_matcher = SemanticMatcher()
resolver = UnifiedDependencyResolver(registry, semantic_matcher)
```

## Factory Function Approach

To simplify the creation of properly configured instances, we can add a factory function:

```python
def create_dependency_resolver(registry: Optional[SpecificationRegistry] = None) -> UnifiedDependencyResolver:
    """
    Create a properly configured dependency resolver.
    
    Args:
        registry: Optional specification registry. If None, creates a new one.
        
    Returns:
        Configured UnifiedDependencyResolver instance
    """
    registry = registry or SpecificationRegistry()
    semantic_matcher = SemanticMatcher()
    return UnifiedDependencyResolver(registry, semantic_matcher)
```

This factory function can be added to `dependency_resolver.py` to make it easier to create properly configured instances.

## Backward Compatibility Considerations

To maintain backward compatibility during the transition, we could:

1. Add a default parameter to the `UnifiedDependencyResolver` constructor:

```python
def __init__(self, registry: SpecificationRegistry, semantic_matcher: Optional[SemanticMatcher] = None):
    self.registry = registry
    self.semantic_matcher = semantic_matcher or SemanticMatcher()
```

2. Add a deprecation warning when the default is used:

```python
def __init__(self, registry: SpecificationRegistry, semantic_matcher: Optional[SemanticMatcher] = None):
    self.registry = registry
    if semantic_matcher is None:
        import warnings
        warnings.warn(
            "Creating a SemanticMatcher instance automatically is deprecated. "
            "Please provide a SemanticMatcher instance explicitly.",
            DeprecationWarning, stacklevel=2
        )
        semantic_matcher = SemanticMatcher()
    self.semantic_matcher = semantic_matcher
```

However, for a clean break, it's recommended to update all usages at once to avoid confusion and ensure consistent usage throughout the codebase.
