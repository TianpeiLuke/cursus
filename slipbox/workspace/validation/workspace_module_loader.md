---
tags:
  - code
  - workspace
  - validation
  - module-loader
  - dynamic-loading
keywords:
  - WorkspaceModuleLoader
  - dynamic module loading
  - Python path management
  - workspace isolation
  - module caching
topics:
  - workspace validation
  - module loading
  - Python path management
language: python
date of note: 2024-12-07
---

# Workspace Module Loader

Workspace-aware dynamic module loader with Python path management for multi-developer workspaces with isolation and caching capabilities.

## Overview

The Workspace Module Loader provides workspace-aware dynamic module loading with Python path management for multi-developer workspaces. This module handles module discovery, loading, and isolation while maintaining backward compatibility with single workspace mode and providing context managers for safe path manipulation.

The loader extends existing dynamic module loading capabilities, manages Python sys.path for workspace isolation, supports developer workspace module discovery, and provides comprehensive module caching and invalidation. It integrates seamlessly with the workspace validation system to provide isolated module loading environments.

Key features include workspace-specific module path management, dynamic module loading with fallback to shared workspace, Python path isolation and cleanup, module caching and invalidation, and developer workspace discovery.

## Classes and Methods

### Classes
- [`WorkspaceModuleLoader`](#workspacemoduleloader) - Workspace-aware module loader with Python path management

### Methods
- [`workspace_path_context`](#workspace_path_context) - Context manager for workspace-aware Python path manipulation
- [`load_builder_class`](#load_builder_class) - Load step builder class with workspace-aware module loading
- [`load_contract_class`](#load_contract_class) - Load script contract class with workspace-aware module loading
- [`load_module_from_file`](#load_module_from_file) - Load module from specific file path
- [`discover_workspace_modules`](#discover_workspace_modules) - Discover available modules in workspace
- [`clear_cache`](#clear_cache) - Clear module cache
- [`invalidate_cache_for_step`](#invalidate_cache_for_step) - Invalidate cache entries for specific step
- [`get_workspace_info`](#get_workspace_info) - Get workspace configuration information
- [`switch_developer`](#switch_developer) - Switch to different developer workspace

## API Reference

### WorkspaceModuleLoader

_class_ cursus.workspace.validation.workspace_module_loader.WorkspaceModuleLoader(_workspace_root=None_, _developer_id=None_, _enable_shared_fallback=True_, _cache_modules=True_)

Workspace-aware module loader that provides dynamic module loading with Python path management for multi-developer workspaces.

**Parameters:**
- **workspace_root** (_Optional[Union[str, Path]]_) – Root directory containing developer workspaces.
- **developer_id** (_Optional[str]_) – Specific developer workspace to target.
- **enable_shared_fallback** (_bool_) – Whether to fallback to shared workspace, defaults to True.
- **cache_modules** (_bool_) – Whether to cache loaded modules, defaults to True.

```python
from cursus.workspace.validation.workspace_module_loader import WorkspaceModuleLoader
from pathlib import Path

# Create workspace-aware module loader
loader = WorkspaceModuleLoader(
    workspace_root="/path/to/workspace",
    developer_id="alice",
    enable_shared_fallback=True,
    cache_modules=True
)

# Create loader without workspace mode (single workspace compatibility)
single_loader = WorkspaceModuleLoader()

print("Workspace mode:", loader.workspace_mode)
print("Developer ID:", loader.developer_id)
print("Cache enabled:", loader.cache_modules)
```

#### workspace_path_context

workspace_path_context(_include_developer=True_, _include_shared=None_)

Context manager for workspace-aware Python path manipulation with automatic cleanup.

**Parameters:**
- **include_developer** (_bool_) – Whether to include developer workspace paths, defaults to True.
- **include_shared** (_Optional[bool]_) – Whether to include shared workspace paths, defaults to enable_shared_fallback setting.

**Returns:**
- **ContextManager[List[str]]** – Context manager yielding list of paths added to sys.path.

```python
# Use context manager for safe path manipulation
with loader.workspace_path_context() as added_paths:
    print("Added paths:", added_paths)
    
    # Import modules with workspace paths available
    import cursus_dev.steps.builders.my_builder
    
    # Paths are automatically cleaned up when exiting context

# Use with specific path inclusion
with loader.workspace_path_context(include_developer=True, include_shared=False) as paths:
    # Only developer paths are included
    print("Developer-only paths:", paths)

# Use for isolated module loading
with loader.workspace_path_context(include_developer=False, include_shared=True) as paths:
    # Only shared workspace paths are included
    print("Shared-only paths:", paths)
```

#### load_builder_class

load_builder_class(_step_name_, _builder_module_name=None_, _builder_class_name=None_)

Load step builder class with workspace-aware module loading and automatic fallback.

**Parameters:**
- **step_name** (_str_) – Name of the step to load builder for.
- **builder_module_name** (_Optional[str]_) – Specific module name to load, defaults to step_name + "_builder".
- **builder_class_name** (_Optional[str]_) – Specific class name to load, auto-generated from step_name if None.

**Returns:**
- **Optional[Type]** – Builder class if found, None otherwise.

```python
# Load builder class by step name
builder_class = loader.load_builder_class("data_preprocessing")

if builder_class:
    print(f"Loaded builder: {builder_class.__name__}")
    
    # Create builder instance
    builder_instance = builder_class()
    print(f"Builder type: {type(builder_instance)}")
    
    # Check builder capabilities
    if hasattr(builder_class, 'step_type'):
        print(f"Step type: {builder_class.step_type}")
else:
    print("Builder not found")

# Load with specific module and class names
custom_builder = loader.load_builder_class(
    step_name="custom_process",
    builder_module_name="advanced_processor",
    builder_class_name="AdvancedProcessorBuilder"
)

# Load from different naming patterns
# Automatically tries: custom_process_builder -> CustomProcessBuilder
auto_builder = loader.load_builder_class("custom_process")
```

#### load_contract_class

load_contract_class(_step_name_, _contract_module_name=None_, _contract_class_name=None_)

Load script contract class with workspace-aware module loading and caching.

**Parameters:**
- **step_name** (_str_) – Name of the step to load contract for.
- **contract_module_name** (_Optional[str]_) – Specific module name to load, defaults to step_name + "_contract".
- **contract_class_name** (_Optional[str]_) – Specific class name to load, auto-generated from step_name if None.

**Returns:**
- **Optional[Type]** – Contract class if found, None otherwise.

```python
# Load contract class by step name
contract_class = loader.load_contract_class("model_training")

if contract_class:
    print(f"Loaded contract: {contract_class.__name__}")
    
    # Create contract instance
    contract_instance = contract_class()
    
    # Inspect contract fields
    if hasattr(contract_instance, '__dict__'):
        print("Contract fields:", list(contract_instance.__dict__.keys()))
    
    # Check if it's a Pydantic model
    if hasattr(contract_class, '__fields__'):
        print("Pydantic fields:", list(contract_class.__fields__.keys()))
else:
    print("Contract not found")

# Load with specific naming
specific_contract = loader.load_contract_class(
    step_name="validation",
    contract_module_name="model_validator_contract",
    contract_class_name="ModelValidatorContract"
)
```

#### load_module_from_file

load_module_from_file(_file_path_, _module_name=None_)

Load module from specific file path with caching and error handling.

**Parameters:**
- **file_path** (_Union[str, Path]_) – Path to the Python file to load.
- **module_name** (_Optional[str]_) – Name to assign to the module, defaults to file stem.

**Returns:**
- **Optional[Any]** – Loaded module if successful, None otherwise.

```python
from pathlib import Path

# Load module from specific file
module_path = Path("/path/to/workspace/developers/alice/src/cursus_dev/steps/builders/custom_builder.py")
module = loader.load_module_from_file(module_path)

if module:
    print(f"Loaded module: {module.__name__}")
    
    # Access module contents
    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            attr = getattr(module, attr_name)
            print(f"  {attr_name}: {type(attr)}")
    
    # Find classes in module
    classes = [getattr(module, name) for name in dir(module) 
               if isinstance(getattr(module, name), type)]
    print(f"Classes found: {[cls.__name__ for cls in classes]}")

# Load with custom module name
custom_module = loader.load_module_from_file(
    "/path/to/script.py",
    module_name="my_custom_module"
)

# Load and execute immediately
script_path = Path("workspace/shared/src/cursus_dev/steps/scripts/utility.py")
utility_module = loader.load_module_from_file(script_path)
if utility_module and hasattr(utility_module, 'main'):
    utility_module.main()
```

#### discover_workspace_modules

discover_workspace_modules(_module_type="builders"_)

Discover available modules in workspace with comprehensive module type support.

**Parameters:**
- **module_type** (_str_) – Type of modules to discover ('builders', 'contracts', 'specs', 'scripts', 'configs'), defaults to 'builders'.

**Returns:**
- **Dict[str, List[str]]** – Dictionary mapping workspace identifier to list of module names.

```python
# Discover builder modules
builders = loader.discover_workspace_modules("builders")

print("Builder modules by workspace:")
for workspace, module_list in builders.items():
    print(f"  {workspace}: {module_list}")

# Discover contract modules
contracts = loader.discover_workspace_modules("contracts")

# Discover all module types
module_types = ["builders", "contracts", "specs", "scripts", "configs"]
all_modules = {}

for module_type in module_types:
    modules = loader.discover_workspace_modules(module_type)
    all_modules[module_type] = modules
    
    total_count = sum(len(module_list) for module_list in modules.values())
    print(f"{module_type}: {total_count} modules across {len(modules)} workspaces")

# Get detailed breakdown
for module_type, workspaces in all_modules.items():
    print(f"\n{module_type.upper()}:")
    for workspace, modules in workspaces.items():
        print(f"  {workspace}: {len(modules)} modules")
        for module_name in modules[:3]:  # Show first 3
            print(f"    - {module_name}")
        if len(modules) > 3:
            print(f"    ... and {len(modules) - 3} more")
```

#### clear_cache

clear_cache()

Clear all cached modules and path information.

```python
# Clear all caches
loader.clear_cache()
print("Module cache cleared")

# Force fresh loading
builder_class = loader.load_builder_class("data_processing")
print("Loaded fresh builder class")

# Check cache status
info = loader.get_workspace_info()
print(f"Cached modules: {info['cached_modules']}")
```

#### invalidate_cache_for_step

invalidate_cache_for_step(_step_name_)

Invalidate cache entries for a specific step to force reloading.

**Parameters:**
- **step_name** (_str_) – Name of the step to invalidate cache for.

```python
# Invalidate cache for specific step
loader.invalidate_cache_for_step("data_preprocessing")
print("Cache invalidated for data_preprocessing")

# Reload will now fetch fresh modules
builder_class = loader.load_builder_class("data_preprocessing")
contract_class = loader.load_contract_class("data_preprocessing")

# Useful after modifying step files during development
def reload_step_after_changes(loader, step_name):
    # Invalidate cache
    loader.invalidate_cache_for_step(step_name)
    
    # Reload components
    builder = loader.load_builder_class(step_name)
    contract = loader.load_contract_class(step_name)
    
    print(f"Reloaded {step_name}:")
    print(f"  Builder: {'✓' if builder else '✗'}")
    print(f"  Contract: {'✓' if contract else '✗'}")
    
    return builder, contract
```

#### get_workspace_info

get_workspace_info()

Get comprehensive information about current workspace configuration and status.

**Returns:**
- **Dict[str, Any]** – Dictionary containing workspace configuration, paths, and cache status.

```python
# Get workspace configuration info
info = loader.get_workspace_info()

print("Workspace Module Loader Info:")
print(f"  Mode: {info['workspace_mode']}")
print(f"  Root: {info['workspace_root']}")
print(f"  Developer: {info['developer_id']}")
print(f"  Shared fallback: {info['enable_shared_fallback']}")
print(f"  Cache enabled: {info['cache_modules']}")
print(f"  Cached modules: {info['cached_modules']}")

print("Python paths:")
print(f"  Developer paths: {info['developer_paths']}")
print(f"  Shared paths: {info['shared_paths']}")

# Use info for diagnostics
if not info['workspace_mode']:
    print("Operating in single workspace mode")
elif not info['developer_paths']:
    print("Warning: No developer paths configured")
```

#### switch_developer

switch_developer(_developer_id_)

Switch to a different developer workspace and rebuild all paths and caches.

**Parameters:**
- **developer_id** (_str_) – Developer identifier to switch to.

**Raises:**
- **ValueError** – If not in workspace mode or developer workspace not found.

```python
# Switch to different developer workspace
try:
    loader.switch_developer("bob")
    print("Switched to Bob's workspace")
    
    # Verify switch
    info = loader.get_workspace_info()
    print(f"Current developer: {info['developer_id']}")
    print(f"Developer paths: {info['developer_paths']}")
    
    # Load modules from new workspace
    bob_builder = loader.load_builder_class("bob_special_processor")
    if bob_builder:
        print(f"Found Bob's builder: {bob_builder.__name__}")
    
except ValueError as e:
    print(f"Switch failed: {e}")

# Switch back to original developer
loader.switch_developer("alice")
print("Switched back to Alice's workspace")
```

## Module Loading Patterns

### Safe Module Loading with Context

```python
# Safe module loading with automatic path cleanup
def safe_load_workspace_module(loader, step_name, module_type="builders"):
    try:
        with loader.workspace_path_context() as paths:
            print(f"Loading {step_name} with paths: {paths}")
            
            if module_type == "builders":
                module_class = loader.load_builder_class(step_name)
            elif module_type == "contracts":
                module_class = loader.load_contract_class(step_name)
            else:
                raise ValueError(f"Unsupported module type: {module_type}")
            
            if module_class:
                print(f"Successfully loaded {module_class.__name__}")
                return module_class
            else:
                print(f"Module {step_name} not found")
                return None
                
    except Exception as e:
        print(f"Error loading {step_name}: {e}")
        return None

# Usage
builder = safe_load_workspace_module(loader, "data_processing", "builders")
contract = safe_load_workspace_module(loader, "data_processing", "contracts")
```

### Multi-Developer Module Discovery

```python
# Comprehensive multi-developer module discovery
def discover_all_workspace_modules(loader):
    module_types = ["builders", "contracts", "specs", "scripts", "configs"]
    discovery_results = {}
    
    for module_type in module_types:
        modules = loader.discover_workspace_modules(module_type)
        discovery_results[module_type] = modules
        
        print(f"\n{module_type.upper()} MODULES:")
        for workspace, module_list in modules.items():
            print(f"  {workspace}: {len(module_list)} modules")
            
            # Try to load each module
            for module_name in module_list:
                try:
                    if module_type == "builders":
                        module_class = loader.load_builder_class(module_name)
                    elif module_type == "contracts":
                        module_class = loader.load_contract_class(module_name)
                    else:
                        # For other types, try to load from file
                        continue
                    
                    status = "✓" if module_class else "✗"
                    print(f"    {status} {module_name}")
                    
                except Exception as e:
                    print(f"    ✗ {module_name} (error: {e})")
    
    return discovery_results
```

### Development Workflow Integration

```python
# Development workflow with hot reloading
class DevelopmentWorkflow:
    def __init__(self, workspace_root, developer_id):
        self.loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id=developer_id,
            cache_modules=True
        )
        self.watched_steps = set()
    
    def watch_step(self, step_name):
        """Add step to watch list for hot reloading."""
        self.watched_steps.add(step_name)
        print(f"Now watching {step_name} for changes")
    
    def reload_step(self, step_name):
        """Reload a step after changes."""
        if step_name not in self.watched_steps:
            print(f"Step {step_name} not in watch list")
            return False
        
        # Invalidate cache
        self.loader.invalidate_cache_for_step(step_name)
        
        # Reload components
        builder = self.loader.load_builder_class(step_name)
        contract = self.loader.load_contract_class(step_name)
        
        print(f"Reloaded {step_name}:")
        print(f"  Builder: {'✓' if builder else '✗'}")
        print(f"  Contract: {'✓' if contract else '✗'}")
        
        return builder is not None or contract is not None
    
    def reload_all_watched(self):
        """Reload all watched steps."""
        results = {}
        for step_name in self.watched_steps:
            results[step_name] = self.reload_step(step_name)
        return results
    
    def switch_developer_context(self, new_developer_id):
        """Switch to different developer context."""
        try:
            self.loader.switch_developer(new_developer_id)
            print(f"Switched to {new_developer_id}'s workspace")
            
            # Reload all watched steps in new context
            self.reload_all_watched()
            
        except ValueError as e:
            print(f"Failed to switch developer: {e}")

# Usage
workflow = DevelopmentWorkflow("/path/to/workspace", "alice")
workflow.watch_step("data_preprocessing")
workflow.watch_step("model_training")

# After making changes to files
workflow.reload_step("data_preprocessing")

# Switch to different developer
workflow.switch_developer_context("bob")
```

### Module Loading with Error Recovery

```python
# Robust module loading with error recovery
def robust_module_loading(loader, step_name, fallback_developers=None):
    """
    Load module with fallback to other developers if primary fails.
    """
    if fallback_developers is None:
        fallback_developers = []
    
    # Try primary developer first
    try:
        builder = loader.load_builder_class(step_name)
        if builder:
            return builder, loader.developer_id
    except Exception as e:
        print(f"Failed to load from {loader.developer_id}: {e}")
    
    # Try fallback developers
    original_developer = loader.developer_id
    
    for fallback_dev in fallback_developers:
        try:
            loader.switch_developer(fallback_dev)
            builder = loader.load_builder_class(step_name)
            if builder:
                print(f"Found {step_name} in {fallback_dev}'s workspace")
                return builder, fallback_dev
        except Exception as e:
            print(f"Failed to load from {fallback_dev}: {e}")
            continue
    
    # Restore original developer
    try:
        loader.switch_developer(original_developer)
    except:
        pass
    
    return None, None

# Usage
builder, found_in = robust_module_loading(
    loader, 
    "advanced_processor", 
    fallback_developers=["bob", "charlie", "shared"]
)

if builder:
    print(f"Successfully loaded from {found_in}")
else:
    print("Module not found in any workspace")
```

### Performance Monitoring

```python
# Monitor module loading performance
def monitor_loading_performance(loader, steps_to_test):
    import time
    
    results = {
        'cache_hits': 0,
        'cache_misses': 0,
        'load_times': {},
        'total_time': 0
    }
    
    start_total = time.time()
    
    for step_name in steps_to_test:
        # First load (cache miss)
        loader.invalidate_cache_for_step(step_name)
        
        start_time = time.time()
        builder = loader.load_builder_class(step_name)
        first_load_time = time.time() - start_time
        
        # Second load (cache hit)
        start_time = time.time()
        builder_cached = loader.load_builder_class(step_name)
        second_load_time = time.time() - start_time
        
        results['load_times'][step_name] = {
            'cache_miss': first_load_time,
            'cache_hit': second_load_time,
            'speedup': first_load_time / second_load_time if second_load_time > 0 else float('inf')
        }
        
        if builder:
            results['cache_hits'] += 1
        else:
            results['cache_misses'] += 1
    
    results['total_time'] = time.time() - start_total
    
    # Print results
    print("Module Loading Performance:")
    print(f"  Total time: {results['total_time']:.3f}s")
    print(f"  Successful loads: {results['cache_hits']}")
    print(f"  Failed loads: {results['cache_misses']}")
    
    print("\nPer-step performance:")
    for step_name, times in results['load_times'].items():
        print(f"  {step_name}:")
        print(f"    Cache miss: {times['cache_miss']:.3f}s")
        print(f"    Cache hit: {times['cache_hit']:.3f}s")
        print(f"    Speedup: {times['speedup']:.1f}x")
    
    return results
```

## Related Documentation

- [Developer Workspace File Resolver](workspace_file_resolver.md) - File resolution for workspace components
- [Workspace Test Manager](workspace_test_manager.md) - Testing framework for workspace components
- [Cross Workspace Validator](cross_workspace_validator.md) - Cross-workspace validation capabilities
- [Workspace Manager](workspace_manager.md) - Workspace management and coordination
- [Unified Validation Core](unified_validation_core.md) - Core validation functionality
