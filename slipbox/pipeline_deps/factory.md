# Factory Module

## Overview
The Factory module provides convenience functions for creating and managing pipeline dependency components. It simplifies the creation and wiring of the core components of the dependency resolution system, ensuring they are properly connected and configured.

## Core Functionality

### Key Features
- **Component Creation**: Centralized creation of all dependency resolution components
- **Thread Management**: Thread-local storage for component isolation in multi-threaded environments
- **Context Management**: Context managers for scoped dependency resolution
- **Proper Wiring**: Ensures all components are correctly connected to each other

## Key Functions

### create_pipeline_components
Creates all necessary pipeline components with proper dependencies.

```python
def create_pipeline_components(context_name=None):
    """
    Create all necessary pipeline components with proper dependencies.
    
    Args:
        context_name: Optional name for the registry context
        
    Returns:
        Dictionary of components: semantic_matcher, registry_manager, registry, resolver
    """
```

### get_thread_components
Gets thread-specific component instances, creating them if they don't exist.

```python
def get_thread_components():
    """
    Get thread-specific component instances.
    
    Returns:
        Dictionary of thread-local components
    """
```

### dependency_resolution_context
Context manager for creating a scoped dependency resolution context.

```python
@contextmanager
def dependency_resolution_context(clear_on_exit=True):
    """
    Create a scoped dependency resolution context.
    
    Args:
        clear_on_exit: Whether to clear caches when exiting the context
        
    Yields:
        Dictionary of components
    """
```

## Usage Examples

### Basic Component Creation
```python
from src.pipeline_deps.factory import create_pipeline_components

# Create all components with default context
components = create_pipeline_components()

# Access individual components
semantic_matcher = components["semantic_matcher"]
registry_manager = components["registry_manager"]
registry = components["registry"]
resolver = components["resolver"]

# Use components
registry.register("data_loading", data_loading_spec)
dependencies = resolver.resolve_step_dependencies("training", ["data_loading", "training"])
```

### Named Context Creation
```python
from src.pipeline_deps.factory import create_pipeline_components

# Create components with named context
components = create_pipeline_components("my_pipeline")

# The registry will be scoped to "my_pipeline" context
registry = components["registry"]
registry.register("data_loading", data_loading_spec)
```

### Thread-Local Components
```python
from src.pipeline_deps.factory import get_thread_components
import threading

def worker_function():
    # Each thread gets its own isolated components
    components = get_thread_components()
    registry = components["registry"]
    
    # Operations in this thread won't affect other threads
    registry.register("thread_specific_step", step_spec)

# Create worker threads
threads = []
for i in range(3):
    thread = threading.Thread(target=worker_function)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
```

### Using Context Manager
```python
from src.pipeline_deps.factory import dependency_resolution_context

def process_pipeline():
    # Create scoped context with automatic cleanup
    with dependency_resolution_context() as components:
        registry = components["registry"]
        resolver = components["resolver"]
        
        # Register specifications
        registry.register("data_loading", data_loading_spec)
        registry.register("training", training_spec)
        
        # Resolve dependencies
        dependencies = resolver.resolve_step_dependencies("training", ["data_loading", "training"])
        
        # Use dependencies
        # ...
    
    # When exiting the context:
    # - Resolver cache is cleared
    # - Registry contexts are cleared
```

### Context Manager with Cache Preservation
```python
from src.pipeline_deps.factory import dependency_resolution_context

# Keep caches when exiting context (useful for reusing results)
with dependency_resolution_context(clear_on_exit=False) as components:
    registry = components["registry"]
    resolver = components["resolver"]
    
    # Register and resolve
    # ...

# Caches are preserved after context exit
```

## Component Relationships

The factory creates and wires together these components:

1. **SemanticMatcher**: Provides semantic similarity calculation for dependency matching
2. **RegistryManager**: Manages multiple isolated specification registries
3. **SpecificationRegistry**: Stores and retrieves step specifications
4. **UnifiedDependencyResolver**: Resolves dependencies between steps

```
┌─────────────────┐
│ RegistryManager │
└────────┬────────┘
         │ creates
         ▼
┌─────────────────┐     uses     ┌─────────────────┐
│SpecificationReg.│◄────────────►│DependencyResolver│
└─────────────────┘              └────────┬─────────┘
                                         │ uses
                                         ▼
                                  ┌─────────────────┐
                                  │ SemanticMatcher │
                                  └─────────────────┘
```

## Thread Safety

The factory provides several features to ensure thread safety:

1. **Thread-local storage**: Component instances are stored in thread-local variables
2. **Context isolation**: Each thread gets its own isolated set of components
3. **Explicit context management**: Context managers control component lifecycle

This allows for safe concurrent use in multi-threaded environments like web servers or asynchronous processing systems.

## Integration with Pipeline Builder

The factory is typically used during pipeline initialization:

```python
from src.pipeline_builder.template import PipelineBuilderTemplate
from src.pipeline_deps.factory import create_pipeline_components

def create_pipeline(pipeline_name: str):
    # Create dependency resolution components
    components = create_pipeline_components(pipeline_name)
    registry = components["registry"]
    resolver = components["resolver"]
    
    # Configure registry
    registry.register("data_loading", data_loading_spec)
    registry.register("preprocessing", preprocessing_spec)
    registry.register("training", training_spec)
    
    # Create pipeline template with resolver
    template = PipelineBuilderTemplate(
        dag=dag,
        config_map=config_map,
        step_builder_map=step_builder_map,
        registry=registry,
        dependency_resolver=resolver
    )
    
    # Generate pipeline
    return template.generate_pipeline(pipeline_name)
```

## Best Practices

### 1. Component Creation
- Use `create_pipeline_components` for consistent component creation
- Specify context names for better isolation and debugging
- Create fresh components for each pipeline to avoid cross-contamination

### 2. Thread Management
- Use `get_thread_components` in multi-threaded environments
- Avoid sharing components across threads without proper synchronization
- Clear thread-local storage when threads are recycled

### 3. Context Management
- Use `dependency_resolution_context` for automatic cleanup
- Set `clear_on_exit=False` when you need to preserve caches
- Create short-lived contexts for better resource management

### 4. Performance Considerations
- Component creation is lightweight but not free
- Reuse components when processing multiple related pipelines
- Clear caches when memory usage becomes a concern
