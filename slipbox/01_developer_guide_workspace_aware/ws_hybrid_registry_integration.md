# Hybrid Registry Integration for Isolated Projects

**Version**: 1.0.0  
**Date**: September 5, 2025  
**Author**: Tianpei Xie

## Overview

This guide provides comprehensive instructions for integrating with the hybrid registry system in isolated project environments. The hybrid registry enables seamless access to both shared (`src/cursus`) and project-specific (`src/cursus_dev`) components with automatic fallback resolution and workspace-aware caching.

## Hybrid Registry Architecture

The hybrid registry system provides a unified interface for component access across workspaces:

```
┌─────────────────────────────────────────────────────────────┐
│                 UnifiedRegistryManager                      │
├─────────────────────────────────────────────────────────────┤
│  Workspace Context: "your_project"                         │
│                                                             │
│  Resolution Priority:                                       │
│  1. Project-specific components (src/cursus_dev)            │
│  2. Shared components (src/cursus) - fallback              │
│  3. Default implementations - final fallback               │
│                                                             │
│  Features:                                                  │
│  • Workspace-aware caching                                 │
│  • Automatic fallback resolution                           │
│  • Context management                                       │
│  • Cross-workspace validation                              │
└─────────────────────────────────────────────────────────────┘
```

## Basic Registry Usage

### Getting Registry Instance

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get singleton registry instance
registry = UnifiedRegistryManager()

# Set project workspace context
registry.set_workspace_context("your_project")

# Verify current workspace
current_workspace = registry.get_current_workspace()
print(f"Current workspace: {current_workspace}")
```

### Component Registration

#### Project-Specific Component Registration

```python
# File: src/cursus_dev/steps/builders/builder_custom_step.py

from cursus.registry.hybrid.manager import UnifiedRegistryManager
from cursus.core.step_builder import StepBuilderBase

class CustomStepBuilder(StepBuilderBase):
    """Project-specific step builder."""
    
    def build_step(self, config):
        # Project-specific implementation
        pass

# Register with project workspace context
registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")
registry.register_step_builder("custom_step", CustomStepBuilder)
```

#### Using Registration Decorators

```python
# File: src/cursus_dev/steps/builders/builder_custom_step.py

from cursus.registry.decorators import register_builder
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Set workspace context before using decorator
registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")

@register_builder("custom_step")
class CustomStepBuilder(StepBuilderBase):
    """Project-specific step builder registered with decorator."""
    
    def build_step(self, config):
        # Implementation here
        pass
```

### Component Resolution

#### Accessing Project-Specific Components

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry with project context
registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")

# Access project-specific component (first priority)
custom_builder = registry.get_step_builder("custom_step")
print(f"Retrieved: {custom_builder.__class__.__name__}")
```

#### Accessing Shared Components

```python
# Access shared component (automatic fallback)
shared_builder = registry.get_step_builder("preprocessing_step")
print(f"Retrieved shared: {shared_builder.__class__.__name__}")

# List all accessible components
all_components = registry.list_step_builders()
print(f"Available step builders: {all_components}")
```

#### Resolution Priority Example

```python
# Example showing resolution priority
registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")

# If both project and shared versions exist, project takes priority
builder = registry.get_step_builder("data_processing")

# Check which version was resolved
if hasattr(builder, '_workspace_origin'):
    print(f"Resolved from workspace: {builder._workspace_origin}")
else:
    print("Resolved from shared components")
```

## Advanced Registry Patterns

### Context Management Patterns

#### Pattern 1: Temporary Context Switching

```python
def access_shared_component(component_name):
    """Safely access shared component with context switching."""
    registry = UnifiedRegistryManager()
    
    # Save current workspace context
    original_workspace = registry.get_current_workspace()
    
    try:
        # Switch to main workspace for shared access
        registry.set_workspace_context("main")
        component = registry.get_component(component_name)
        return component
    finally:
        # Always restore original context
        registry.set_workspace_context(original_workspace)

# Usage
shared_validator = access_shared_component("validation_utils")
```

#### Pattern 2: Context Manager

```python
from contextlib import contextmanager

@contextmanager
def workspace_context(workspace_name):
    """Context manager for safe workspace switching."""
    registry = UnifiedRegistryManager()
    original_workspace = registry.get_current_workspace()
    
    try:
        registry.set_workspace_context(workspace_name)
        yield registry
    finally:
        registry.set_workspace_context(original_workspace)

# Usage
with workspace_context("main") as main_registry:
    shared_component = main_registry.get_component("shared_utility")

with workspace_context("your_project") as project_registry:
    project_component = project_registry.get_component("custom_utility")
```

#### Pattern 3: Multi-Workspace Access

```python
class MultiWorkspaceAccessor:
    """Helper class for accessing components across workspaces."""
    
    def __init__(self, primary_workspace):
        self.registry = UnifiedRegistryManager()
        self.primary_workspace = primary_workspace
        self.registry.set_workspace_context(primary_workspace)
    
    def get_project_component(self, component_name):
        """Get component from current project workspace."""
        return self.registry.get_component(component_name)
    
    def get_shared_component(self, component_name):
        """Get component from shared workspace."""
        with workspace_context("main") as main_registry:
            return main_registry.get_component(component_name)
    
    def get_component_from_workspace(self, component_name, workspace):
        """Get component from specific workspace."""
        with workspace_context(workspace) as target_registry:
            return target_registry.get_component(component_name)

# Usage
accessor = MultiWorkspaceAccessor("your_project")
project_step = accessor.get_project_component("custom_step")
shared_step = accessor.get_shared_component("preprocessing_step")
other_project_step = accessor.get_component_from_workspace("special_step", "project_alpha")
```

### Component Discovery and Introspection

#### Discovering Available Components

```python
def discover_components():
    """Discover components across workspaces."""
    registry = UnifiedRegistryManager()
    
    # Discover project-specific components
    registry.set_workspace_context("your_project")
    project_components = {
        'step_builders': registry.list_step_builders(),
        'components': registry.list_components(),
        'workspace': 'your_project'
    }
    
    # Discover shared components
    registry.set_workspace_context("main")
    shared_components = {
        'step_builders': registry.list_step_builders(),
        'components': registry.list_components(),
        'workspace': 'main'
    }
    
    return {
        'project': project_components,
        'shared': shared_components
    }

# Usage
components = discover_components()
print("Project components:", components['project']['step_builders'])
print("Shared components:", components['shared']['step_builders'])
```

#### Component Metadata and Inspection

```python
def inspect_component(component_name, workspace=None):
    """Inspect component metadata and capabilities."""
    registry = UnifiedRegistryManager()
    
    if workspace:
        registry.set_workspace_context(workspace)
    
    try:
        component = registry.get_step_builder(component_name)
        
        metadata = {
            'name': component_name,
            'class': component.__class__.__name__,
            'module': component.__class__.__module__,
            'workspace': registry.get_current_workspace(),
            'methods': [method for method in dir(component) if not method.startswith('_')],
            'docstring': component.__class__.__doc__
        }
        
        return metadata
    except Exception as e:
        return {'error': str(e), 'component': component_name}

# Usage
project_metadata = inspect_component("custom_step", "your_project")
shared_metadata = inspect_component("preprocessing_step", "main")
```

## Registry Configuration

### Workspace-Specific Configuration

```python
# File: src/cursus_dev/__init__.py

from cursus.registry.hybrid.manager import UnifiedRegistryManager

def configure_project_registry():
    """Configure registry for project-specific needs."""
    registry = UnifiedRegistryManager()
    
    # Set workspace context
    registry.set_workspace_context("your_project")
    
    # Configure fallback behavior
    registry.configure_fallback(
        enable_main_fallback=True,          # Enable fallback to main workspace
        enable_shared_access=True,          # Allow access to shared components
        cache_shared_components=True,       # Cache shared components for performance
        strict_workspace_isolation=False,   # Allow cross-workspace access
        validation_mode="strict"            # Strict validation for component compatibility
    )
    
    # Configure caching
    registry.configure_caching(
        enable_component_cache=True,        # Enable component caching
        cache_timeout=3600,                 # Cache timeout in seconds
        max_cache_size=1000                 # Maximum cached components
    )
    
    return registry

# Initialize project registry
PROJECT_REGISTRY = configure_project_registry()
```

### Dynamic Configuration

```python
def setup_dynamic_registry_config(project_name, config_overrides=None):
    """Set up registry with dynamic configuration."""
    registry = UnifiedRegistryManager()
    registry.set_workspace_context(project_name)
    
    # Default configuration
    default_config = {
        'fallback': {
            'enable_main_fallback': True,
            'enable_shared_access': True,
            'cache_shared_components': True
        },
        'validation': {
            'strict_mode': True,
            'cross_workspace_validation': True
        },
        'caching': {
            'enable_component_cache': True,
            'cache_timeout': 3600,
            'max_cache_size': 1000
        }
    }
    
    # Apply configuration overrides
    if config_overrides:
        for section, settings in config_overrides.items():
            if section in default_config:
                default_config[section].update(settings)
    
    # Apply configuration
    registry.configure_fallback(**default_config['fallback'])
    registry.configure_validation(**default_config['validation'])
    registry.configure_caching(**default_config['caching'])
    
    return registry

# Usage
custom_config = {
    'fallback': {'strict_workspace_isolation': True},
    'validation': {'strict_mode': False}
}
registry = setup_dynamic_registry_config("your_project", custom_config)
```

## Component Lifecycle Management

### Component Registration Lifecycle

```python
class ProjectComponentManager:
    """Manage component lifecycle in project workspace."""
    
    def __init__(self, project_name):
        self.project_name = project_name
        self.registry = UnifiedRegistryManager()
        self.registry.set_workspace_context(project_name)
        self.registered_components = {}
    
    def register_component(self, name, component_class, metadata=None):
        """Register component with lifecycle tracking."""
        try:
            self.registry.register_step_builder(name, component_class)
            
            self.registered_components[name] = {
                'class': component_class,
                'metadata': metadata or {},
                'registered_at': datetime.now(),
                'workspace': self.project_name
            }
            
            print(f"Registered {name} in workspace {self.project_name}")
            return True
            
        except Exception as e:
            print(f"Failed to register {name}: {str(e)}")
            return False
    
    def unregister_component(self, name):
        """Unregister component and clean up."""
        try:
            self.registry.unregister_step_builder(name)
            if name in self.registered_components:
                del self.registered_components[name]
            print(f"Unregistered {name} from workspace {self.project_name}")
            return True
        except Exception as e:
            print(f"Failed to unregister {name}: {str(e)}")
            return False
    
    def list_registered_components(self):
        """List all components registered by this manager."""
        return list(self.registered_components.keys())
    
    def get_component_info(self, name):
        """Get detailed information about registered component."""
        return self.registered_components.get(name)

# Usage
manager = ProjectComponentManager("your_project")
manager.register_component("custom_step", CustomStepBuilder, 
                         metadata={"version": "1.0", "author": "Your Name"})
```

### Component Validation and Health Checks

```python
def validate_registry_health(workspace_name):
    """Validate registry health and component integrity."""
    registry = UnifiedRegistryManager()
    registry.set_workspace_context(workspace_name)
    
    health_report = {
        'workspace': workspace_name,
        'timestamp': datetime.now(),
        'components': {},
        'issues': [],
        'overall_health': 'unknown'
    }
    
    try:
        # Check component availability
        step_builders = registry.list_step_builders()
        
        for builder_name in step_builders:
            try:
                builder = registry.get_step_builder(builder_name)
                health_report['components'][builder_name] = {
                    'status': 'healthy',
                    'class': builder.__class__.__name__,
                    'module': builder.__class__.__module__
                }
            except Exception as e:
                health_report['components'][builder_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_report['issues'].append(f"Component {builder_name}: {str(e)}")
        
        # Check fallback functionality
        with workspace_context("main") as main_registry:
            shared_components = main_registry.list_step_builders()
            fallback_accessible = len(shared_components) > 0
            
        if not fallback_accessible:
            health_report['issues'].append("Shared component fallback not accessible")
        
        # Determine overall health
        error_count = len(health_report['issues'])
        if error_count == 0:
            health_report['overall_health'] = 'healthy'
        elif error_count < 3:
            health_report['overall_health'] = 'warning'
        else:
            health_report['overall_health'] = 'critical'
            
    except Exception as e:
        health_report['overall_health'] = 'critical'
        health_report['issues'].append(f"Registry access error: {str(e)}")
    
    return health_report

# Usage
health = validate_registry_health("your_project")
print(f"Registry health: {health['overall_health']}")
if health['issues']:
    print("Issues found:")
    for issue in health['issues']:
        print(f"  - {issue}")
```

## Performance Optimization

### Caching Strategies

```python
class OptimizedRegistryAccess:
    """Optimized registry access with intelligent caching."""
    
    def __init__(self, workspace_name):
        self.workspace_name = workspace_name
        self.registry = UnifiedRegistryManager()
        self.registry.set_workspace_context(workspace_name)
        self.local_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def get_component_cached(self, component_name, cache_timeout=300):
        """Get component with local caching."""
        cache_key = f"{self.workspace_name}:{component_name}"
        current_time = time.time()
        
        # Check local cache first
        if cache_key in self.local_cache:
            cached_item = self.local_cache[cache_key]
            if current_time - cached_item['timestamp'] < cache_timeout:
                self.cache_stats['hits'] += 1
                return cached_item['component']
        
        # Cache miss - fetch from registry
        self.cache_stats['misses'] += 1
        try:
            component = self.registry.get_step_builder(component_name)
            
            # Cache the result
            self.local_cache[cache_key] = {
                'component': component,
                'timestamp': current_time
            }
            
            return component
        except Exception as e:
            print(f"Failed to get component {component_name}: {str(e)}")
            return None
    
    def preload_components(self, component_names):
        """Preload components into cache."""
        for name in component_names:
            self.get_component_cached(name)
    
    def clear_cache(self):
        """Clear local cache."""
        self.local_cache.clear()
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def get_cache_stats(self):
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'cached_items': len(self.local_cache)
        }

# Usage
optimizer = OptimizedRegistryAccess("your_project")

# Preload frequently used components
optimizer.preload_components(["custom_step", "data_processor", "validator"])

# Use cached access
component = optimizer.get_component_cached("custom_step")

# Check performance
stats = optimizer.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

### Batch Operations

```python
def batch_component_operations(workspace_name, operations):
    """Perform batch operations on registry components."""
    registry = UnifiedRegistryManager()
    registry.set_workspace_context(workspace_name)
    
    results = []
    
    # Group operations by type for efficiency
    registrations = [op for op in operations if op['type'] == 'register']
    retrievals = [op for op in operations if op['type'] == 'get']
    
    # Batch registrations
    for op in registrations:
        try:
            registry.register_step_builder(op['name'], op['component'])
            results.append({'operation': op, 'status': 'success'})
        except Exception as e:
            results.append({'operation': op, 'status': 'error', 'error': str(e)})
    
    # Batch retrievals
    for op in retrievals:
        try:
            component = registry.get_step_builder(op['name'])
            results.append({'operation': op, 'status': 'success', 'component': component})
        except Exception as e:
            results.append({'operation': op, 'status': 'error', 'error': str(e)})
    
    return results

# Usage
operations = [
    {'type': 'register', 'name': 'step1', 'component': Step1Builder},
    {'type': 'register', 'name': 'step2', 'component': Step2Builder},
    {'type': 'get', 'name': 'shared_step'},
    {'type': 'get', 'name': 'step1'}
]

results = batch_component_operations("your_project", operations)
for result in results:
    print(f"Operation {result['operation']['type']} {result['operation']['name']}: {result['status']}")
```

## Error Handling and Debugging

### Registry Error Handling

```python
class RegistryErrorHandler:
    """Comprehensive error handling for registry operations."""
    
    def __init__(self, workspace_name):
        self.workspace_name = workspace_name
        self.registry = UnifiedRegistryManager()
        self.registry.set_workspace_context(workspace_name)
        self.error_log = []
    
    def safe_get_component(self, component_name, fallback_workspaces=None):
        """Safely get component with fallback options."""
        fallback_workspaces = fallback_workspaces or ["main"]
        
        # Try current workspace first
        try:
            component = self.registry.get_step_builder(component_name)
            return {'component': component, 'workspace': self.workspace_name, 'status': 'success'}
        except Exception as e:
            self.error_log.append({
                'workspace': self.workspace_name,
                'component': component_name,
                'error': str(e),
                'timestamp': datetime.now()
            })
        
        # Try fallback workspaces
        for fallback_workspace in fallback_workspaces:
            try:
                with workspace_context(fallback_workspace) as fallback_registry:
                    component = fallback_registry.get_step_builder(component_name)
                    return {
                        'component': component, 
                        'workspace': fallback_workspace, 
                        'status': 'fallback_success'
                    }
            except Exception as e:
                self.error_log.append({
                    'workspace': fallback_workspace,
                    'component': component_name,
                    'error': str(e),
                    'timestamp': datetime.now()
                })
        
        return {'component': None, 'workspace': None, 'status': 'failed'}
    
    def get_error_summary(self):
        """Get summary of encountered errors."""
        error_summary = {}
        for error in self.error_log:
            key = f"{error['workspace']}:{error['component']}"
            if key not in error_summary:
                error_summary[key] = []
            error_summary[key].append(error['error'])
        
        return error_summary

# Usage
error_handler = RegistryErrorHandler("your_project")
result = error_handler.safe_get_component("potentially_missing_step")

if result['status'] == 'success':
    print(f"Component found in {result['workspace']}")
elif result['status'] == 'fallback_success':
    print(f"Component found in fallback workspace {result['workspace']}")
else:
    print("Component not found in any workspace")
    print("Errors:", error_handler.get_error_summary())
```

### Debugging Tools

```python
def debug_registry_state(workspace_name):
    """Debug registry state and component resolution."""
    registry = UnifiedRegistryManager()
    registry.set_workspace_context(workspace_name)
    
    debug_info = {
        'workspace': workspace_name,
        'registry_state': {},
        'component_resolution': {},
        'workspace_isolation': {},
        'performance_metrics': {}
    }
    
    # Registry state
    debug_info['registry_state'] = {
        'current_workspace': registry.get_current_workspace(),
        'available_components': registry.list_step_builders(),
        'cache_status': registry.get_cache_status() if hasattr(registry, 'get_cache_status') else 'N/A'
    }
    
    # Component resolution testing
    test_components = ['preprocessing_step', 'custom_step', 'nonexistent_step']
    for component_name in test_components:
        try:
            start_time = time.time()
            component = registry.get_step_builder(component_name)
            resolution_time = time.time() - start_time
            
            debug_info['component_resolution'][component_name] = {
                'status': 'found',
                'class': component.__class__.__name__,
                'module': component.__class__.__module__,
                'resolution_time': resolution_time
            }
        except Exception as e:
            debug_info['component_resolution'][component_name] = {
                'status': 'not_found',
                'error': str(e)
            }
    
    # Workspace isolation testing
    with workspace_context("main") as main_registry:
        main_components = main_registry.list_step_builders()
        
    debug_info['workspace_isolation'] = {
        'project_only_components': set(debug_info['registry_state']['available_components']) - set(main_components),
        'shared_components': set(main_components),
        'accessible_from_project': len(debug_info['registry_state']['available_components'])
    }
    
    return debug_info

# Usage
debug_info = debug_registry_state("your_project")
print("Registry Debug Information:")
print(f"Current workspace: {debug_info['registry_state']['current_workspace']}")
print(f"Available components: {len(debug_info['registry_state']['available_components'])}")
print(f"Project-only components: {debug_info['workspace_isolation']['project_only_components']}")
```

## Best Practices

### 1. Workspace Context Management

```python
# ✅ Good: Always use context managers for temporary switches
with workspace_context("main") as main_registry:
    shared_component = main_registry.get_component("shared_utility")

# ❌ Bad: Manual context switching without proper restoration
registry.set_workspace_context("main")
shared_component = registry.get_component("shared_utility")
# Forgot to restore original context!
```

### 2. Component Registration

```python
# ✅ Good: Set workspace context before registration
registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")

@register_builder("custom_step")
class CustomStepBuilder(StepBuilderBase):
    pass

# ❌ Bad: Registration without explicit workspace context
@register_builder("custom_step")  # Will use default context
class CustomStepBuilder(StepBuilderBase):
    pass
```

### 3. Error Handling

```python
# ✅ Good: Comprehensive error handling with fallbacks
def get_component_safely(component_name):
    try:
        return registry.get_step_builder(component_name)
    except ComponentNotFoundError:
        # Try fallback workspace
        with workspace_context("main") as main_registry:
            return main_registry.get_step_builder(component_name)
    except Exception as e:
        logger.error(f"Failed to get component {component_name}: {str(e)}")
        return None

# ❌ Bad: No error handling
component = registry.get_step_builder("might_not_exist")  # Could raise exception
```

### 4. Performance Considerations

```python
# ✅ Good: Cache frequently accessed components
class ComponentCache:
    def __init__(self):
        self._cache = {}
    
    def get_component(self, name):
        if name not in self._cache:
            self._cache[name] = registry.get_step_builder(name)
        return self._cache[name]

# ❌ Bad: Repeated registry lookups
for i in range(100):
    component = registry.get_step_builder("same_component")  # Inefficient
```

## Troubleshooting Common Issues

### Issue 1: Component Not Found in Project Workspace

```python
# Problem: Component registered in wrong workspace
# Solution: Verify workspace context during registration

def diagnose_component_registration(component_name):
    """Diagnose component registration issues."""
    registry = UnifiedRegistryManager()
    
    # Check all workspaces
    workspaces = ["your_project", "main", "project_alpha", "project_beta"]
    found_in = []
    
    for workspace in workspaces:
        try:
            registry.set_workspace_context(workspace)
            component = registry.get_step_builder(component_name)
            found_in.append(workspace)
        except:
            pass
    
    if found_in:
        print(f"Component '{component_name}' found in workspaces: {found_in}")
    else:
        print(f"Component '{component_name}' not found in any workspace")
    
    return found_in
```

### Issue 2: Fallback Not Working

```python
# Problem: Fallback to shared components not working
# Solution: Check fallback configuration

def check_fallback_configuration():
    """Check if fallback is properly configured."""
    registry = UnifiedRegistryManager()
    registry.set_workspace_context("your_project")
    
    # Test fallback access
    try:
        # Try to access a known shared component
        shared_component = registry.get_step_builder("preprocessing_step")
        print("Fallback working: Successfully accessed shared component")
        return True
    except Exception as e:
        print(f"Fallback not working: {str(e)}")
        
        # Check configuration
        config = registry.get_configuration()
        print(f"Fallback enabled: {config.get('enable_main_fallback', 'Unknown')}")
        return False
```

### Issue 3: Performance Issues

```python
# Problem: Slow component resolution
# Solution: Implement caching and batch operations

def profile_registry_performance():
    """Profile registry performance."""
    registry = UnifiedRegistryManager()
    registry.set_workspace_context("your_project")
    
    components = registry.list_step_builders()
    
    # Time individual lookups
    lookup_times = {}
    for component_name in components[:5]:  # Test first 5 components
        start_time = time.time()
        component = registry.get_step_builder(component_name)
        lookup_times[component_name] = time.time() - start_time
    
    avg_lookup_time = sum(lookup_times.values()) / len(lookup_times)
    print(f"Average component lookup time: {avg_lookup_time:.4f} seconds")
    
    if avg_lookup_time > 0.1:  # 100ms threshold
        print("Performance issue detected. Consider implementing caching.")
    
    return lookup_times
```

## Related Documentation

- [Workspace Setup Guide](ws_workspace_setup_guide.md) - Initial project setup and configuration
- [Shared Code Access Patterns](ws_shared_code_access_patterns.md) - Patterns for accessing shared components
- [Testing in Isolated Projects](ws_testing_in_isolated_projects.md) - Testing strategies with hybrid registry
- [Workspace CLI Reference](ws_workspace_cli_reference.md) - CLI tools for registry management
- [Troubleshooting Workspace Issues](ws_troubleshooting_workspace_issues.md) - Common problems and solutions

### Main Developer Guide References
- [Step Builder Registry Guide](../0_developer_guide/step_builder_registry_guide.md) - Main workspace registry usage
- [Step Builder Registry Usage](../0_developer_guide/step_builder_registry_usage.md) - Registry usage patterns
