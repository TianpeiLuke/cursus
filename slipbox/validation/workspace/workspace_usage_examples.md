---
tags:
  - validation
  - workspace
  - examples
  - usage
  - multi_developer
keywords:
  - workspace validation
  - developer workspace
  - file resolution
  - module loading
  - workspace management
topics:
  - workspace validation examples
  - multi-developer workspace usage
  - validation framework integration
language: python
date of note: 2025-08-28
---

# Workspace Validation Usage Examples

This document provides comprehensive usage examples for the Cursus workspace validation system, demonstrating how to use the multi-developer workspace functionality for validation, file resolution, and module loading.

## Table of Contents

1. [Basic Workspace Setup](#basic-workspace-setup)
2. [File Resolution Examples](#file-resolution-examples)
3. [Module Loading Examples](#module-loading-examples)
4. [Workspace Management Examples](#workspace-management-examples)
5. [Configuration Management](#configuration-management)
6. [Integration Patterns](#integration-patterns)
7. [Error Handling](#error-handling)
8. [Advanced Usage](#advanced-usage)

## Basic Workspace Setup

### Workspace Directory Structure

First, let's understand the expected workspace structure based on the real Cursus codebase:

```
developer_workspaces/
├── developers/
│   ├── developer_1/
│   │   └── src/cursus_dev/steps/
│   │       ├── __init__.py
│   │       ├── builders/
│   │       │   ├── __init__.py
│   │       │   ├── builder_custom_feature_engineering_step.py
│   │       │   ├── builder_neural_network_training_step.py
│   │       │   ├── builder_data_validation_step.py
│   │       │   └── s3_utils.py
│   │       ├── configs/
│   │       │   ├── __init__.py
│   │       │   ├── config_custom_feature_engineering_step.py
│   │       │   ├── config_neural_network_training_step.py
│   │       │   ├── config_data_validation_step.py
│   │       │   └── utils.py
│   │       ├── contracts/
│   │       │   ├── __init__.py
│   │       │   ├── custom_feature_engineering_contract.py
│   │       │   ├── neural_network_training_contract.py
│   │       │   ├── data_validation_contract.py
│   │       │   └── contract_validator.py
│   │       ├── scripts/
│   │       │   ├── __init__.py
│   │       │   ├── custom_feature_engineering.py
│   │       │   ├── neural_network_training.py
│   │       │   └── data_validation.py
│   │       ├── specs/
│   │       │   ├── __init__.py
│   │       │   ├── custom_feature_engineering_spec.py
│   │       │   ├── neural_network_training_spec.py
│   │       │   ├── data_validation_training_spec.py
│   │       │   └── data_validation_testing_spec.py
│   │       ├── hyperparams/
│   │       │   ├── __init__.py
│   │       │   ├── hyperparameters_neural_network.py
│   │       │   └── hyperparameters_custom_model.py
│   │       └── registry/
│   │           ├── __init__.py
│   │           ├── builder_registry.py
│   │           ├── exceptions.py
│   │           ├── step_names.py
│   │           └── step_type_test_variants.py
│   └── developer_2/
│       └── src/cursus_dev/steps/
│           ├── __init__.py
│           ├── builders/
│           ├── configs/
│           ├── contracts/
│           ├── scripts/
│           ├── specs/
│           ├── hyperparams/
│           └── registry/
└── shared/
    └── src/cursus_dev/steps/
        ├── __init__.py
        ├── builders/
        ├── configs/
        ├── contracts/
        ├── scripts/
        ├── specs/
        ├── hyperparams/
        └── registry/
```

**Key Structure Notes:**
- **builders/**: Contains step builder classes following the pattern `builder_[step_name]_step.py` (e.g., `builder_neural_network_training_step.py`)
- **configs/**: Contains configuration classes following the pattern `config_[step_name]_step.py` (e.g., `config_neural_network_training_step.py`)
- **contracts/**: Contains script contracts following the pattern `[step_name]_contract.py` (e.g., `neural_network_training_contract.py`)
- **scripts/**: Contains actual execution scripts following the pattern `[step_name].py` (e.g., `neural_network_training.py`)
- **specs/**: Contains step specifications with variant support (e.g., `neural_network_training_spec.py`, `data_validation_training_spec.py`)
- **hyperparams/**: Contains hyperparameter classes following the pattern `hyperparameters_[model_type].py` (e.g., `hyperparameters_neural_network.py`)
- **registry/**: Contains registration and metadata utilities including `builder_registry.py`, `step_names.py`, and `step_type_test_variants.py`
- Each directory includes `__init__.py` for proper Python module structure
- Utility files like `s3_utils.py`, `utils.py`, `contract_validator.py` provide shared functionality
- Specs support multiple variants per step type (training, testing, validation, calibration)
```

### Basic Import and Setup

```python
from cursus.validation.workspace import (
    WorkspaceManager,
    DeveloperWorkspaceFileResolver,
    WorkspaceModuleLoader,
    WorkspaceConfig
)

# Initialize workspace manager
workspace_root = "/path/to/developer_workspaces"
manager = WorkspaceManager(workspace_root)

# Discover available workspaces
workspace_info = manager.discover_workspaces()
print(f"Found {workspace_info.total_developers} developer workspaces")
print(f"Available developers: {[dev.developer_id for dev in workspace_info.developers]}")
```

## File Resolution Examples

### Basic File Resolution

```python
# Create file resolver for a specific developer
file_resolver = manager.get_file_resolver("developer_1")

# Find contract file
contract_file = file_resolver.find_contract_file("my_custom_step")
print(f"Contract file: {contract_file}")

# Find spec file
spec_file = file_resolver.find_spec_file("my_custom_step")
print(f"Spec file: {spec_file}")

# Find builder file
builder_file = file_resolver.find_builder_file("my_custom_step")
print(f"Builder file: {builder_file}")

# Find script file
script_file = file_resolver.find_script_file("my_custom_step")
print(f"Script file: {script_file}")

# Find config file
config_file = file_resolver.find_config_file("my_custom_step")
print(f"Config file: {config_file}")
```

### Direct File Resolver Usage

```python
# Create file resolver directly
file_resolver = DeveloperWorkspaceFileResolver(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1",
    enable_shared_fallback=True
)

# Get workspace information
workspace_info = file_resolver.get_workspace_info()
print(f"Workspace mode: {workspace_info['workspace_mode']}")
print(f"Developer workspace exists: {workspace_info['developer_workspace_exists']}")
print(f"Shared workspace exists: {workspace_info['shared_workspace_exists']}")

# List available developers
developers = file_resolver.list_available_developers()
print(f"Available developers: {developers}")

# Switch to different developer
file_resolver.switch_developer("developer_2")
print(f"Switched to: {file_resolver.developer_id}")
```

### Fallback Behavior

```python
# File resolution with shared workspace fallback
file_resolver = DeveloperWorkspaceFileResolver(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1",
    enable_shared_fallback=True
)

# This will first look in developer_1's workspace, then fall back to shared
contract_file = file_resolver.find_contract_file("shared_component")
print(f"Found contract (possibly from shared): {contract_file}")

# Disable fallback for strict workspace isolation
file_resolver_strict = DeveloperWorkspaceFileResolver(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1",
    enable_shared_fallback=False
)

# This will only look in developer_1's workspace
contract_file_strict = file_resolver_strict.find_contract_file("shared_component")
print(f"Found contract (developer only): {contract_file_strict}")  # May be None
```

## Module Loading Examples

### Basic Module Loading

```python
# Create module loader for a specific developer
module_loader = manager.get_module_loader("developer_1")

# Load builder class
builder_class = module_loader.load_builder_class("my_custom_step")
if builder_class:
    print(f"Loaded builder class: {builder_class.__name__}")
    builder_instance = builder_class()
else:
    print("Builder class not found")

# Load contract class
contract_class = module_loader.load_contract_class("my_custom_step")
if contract_class:
    print(f"Loaded contract class: {contract_class.__name__}")
    contract_instance = contract_class()
else:
    print("Contract class not found")
```

### Direct Module Loader Usage

```python
# Create module loader directly
module_loader = WorkspaceModuleLoader(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1",
    enable_shared_fallback=True,
    cache_modules=True
)

# Load module from specific file
module_file = "/path/to/developer_workspaces/developers/developer_1/src/cursus_dev/steps/builders/my_builder.py"
module = module_loader.load_module_from_file(module_file, "my_builder")
if module:
    print(f"Loaded module: {module}")
    # Access classes from the module
    if hasattr(module, 'MyCustomBuilder'):
        builder_class = module.MyCustomBuilder
        print(f"Found builder class: {builder_class}")
```

### Workspace Path Context Management

```python
# Use context manager for safe Python path manipulation
module_loader = WorkspaceModuleLoader(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1"
)

# Context manager ensures sys.path is properly restored
with module_loader.workspace_path_context() as added_paths:
    print(f"Added paths to sys.path: {added_paths}")
    
    # Import modules directly while context is active
    try:
        import cursus_dev.steps.builders.my_custom_builder as my_builder
        builder_class = my_builder.MyCustomBuilder
        print(f"Successfully imported: {builder_class}")
    except ImportError as e:
        print(f"Import failed: {e}")

# sys.path is automatically restored here
print("Context exited, sys.path restored")
```

### Module Discovery

```python
# Discover available modules in workspace
module_loader = WorkspaceModuleLoader(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1"
)

# Discover builders
builders = module_loader.discover_workspace_modules("builders")
print(f"Available builders: {builders}")

# Discover contracts
contracts = module_loader.discover_workspace_modules("contracts")
print(f"Available contracts: {contracts}")

# Example output:
# Available builders: {
#     'developer:developer_1': ['my_custom_builder', 'advanced_processing_builder'],
#     'shared': ['shared_builder', 'common_builder']
# }
```

### Module Caching

```python
# Module loader with caching enabled
module_loader = WorkspaceModuleLoader(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1",
    cache_modules=True
)

# First load - will import and cache
builder_class1 = module_loader.load_builder_class("my_custom_step")
print("First load completed")

# Second load - will use cache
builder_class2 = module_loader.load_builder_class("my_custom_step")
print("Second load completed (from cache)")

# Clear cache for specific step
module_loader.invalidate_cache_for_step("my_custom_step")
print("Cache invalidated for my_custom_step")

# Clear all cache
module_loader.clear_cache()
print("All cache cleared")
```

## Workspace Management Examples

### Workspace Discovery and Validation

```python
# Create workspace manager
manager = WorkspaceManager("/path/to/developer_workspaces")

# Discover workspaces
workspace_info = manager.discover_workspaces()
print(f"Workspace root: {workspace_info.workspace_root}")
print(f"Has shared workspace: {workspace_info.has_shared}")
print(f"Total developers: {workspace_info.total_developers}")
print(f"Total modules: {workspace_info.total_modules}")

# Validate workspace structure
is_valid, issues = manager.validate_workspace_structure()
if is_valid:
    print("Workspace structure is valid")
else:
    print("Workspace structure issues:")
    for issue in issues:
        print(f"  - {issue}")

# Strict validation
is_valid_strict, issues_strict = manager.validate_workspace_structure(strict=True)
if not is_valid_strict:
    print("Strict validation issues:")
    for issue in issues_strict:
        print(f"  - {issue}")
```

### Creating New Workspaces

```python
# Create new developer workspace
manager = WorkspaceManager()

# Create workspace with full directory structure
new_workspace = manager.create_developer_workspace(
    "new_developer",
    workspace_root="/path/to/developer_workspaces",
    create_structure=True
)
print(f"Created workspace at: {new_workspace}")

# Create shared workspace
shared_workspace = manager.create_shared_workspace(
    workspace_root="/path/to/developer_workspaces",
    create_structure=True
)
print(f"Created shared workspace at: {shared_workspace}")
```

### Workspace Summary

```python
# Get comprehensive workspace summary
manager = WorkspaceManager("/path/to/developer_workspaces")
summary = manager.get_workspace_summary()

print(f"Workspace Summary:")
print(f"  Root: {summary['workspace_root']}")
print(f"  Has shared: {summary['has_shared']}")
print(f"  Total developers: {summary['total_developers']}")
print(f"  Total modules: {summary['total_modules']}")

print("\nDeveloper Details:")
for dev in summary['developers']:
    print(f"  {dev['developer_id']}:")
    print(f"    Modules: {dev['module_count']}")
    print(f"    Has builders: {dev['has_builders']}")
    print(f"    Has contracts: {dev['has_contracts']}")
    print(f"    Has specs: {dev['has_specs']}")
    print(f"    Has scripts: {dev['has_scripts']}")
    print(f"    Has configs: {dev['has_configs']}")
```

## Configuration Management

### Creating and Using Workspace Configuration

```python
# Create workspace configuration
config = WorkspaceConfig(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1",
    enable_shared_fallback=True,
    cache_modules=True,
    auto_create_structure=False,
    validation_settings={
        "strict_validation": False,
        "require_all_module_types": False,
        "validate_imports": True
    }
)

# Initialize manager with configuration
manager = WorkspaceManager()
manager.config = config

# Use configuration defaults
file_resolver = manager.get_file_resolver()  # Uses config.developer_id
module_loader = manager.get_module_loader()  # Uses config settings

print(f"Using developer: {file_resolver.developer_id}")
print(f"Shared fallback enabled: {file_resolver.enable_shared_fallback}")
print(f"Module caching enabled: {module_loader.cache_modules}")
```

### Saving and Loading Configuration

```python
# Save configuration to JSON
config = WorkspaceConfig(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1",
    enable_shared_fallback=True,
    cache_modules=True
)

manager = WorkspaceManager()
manager.save_config("/path/to/workspace.json", config)
print("Configuration saved to JSON")

# Load configuration from JSON
manager = WorkspaceManager(config_file="/path/to/workspace.json")
print(f"Loaded config - Developer: {manager.config.developer_id}")

# Save configuration to YAML
manager.save_config("/path/to/workspace.yaml", config)
print("Configuration saved to YAML")

# Load configuration from YAML
manager = WorkspaceManager(config_file="/path/to/workspace.yaml")
print(f"Loaded config from YAML - Developer: {manager.config.developer_id}")
```

### Configuration Templates

```python
from cursus.validation.workspace import get_config_template

# Get basic configuration template
basic_config = get_config_template("basic")
basic_config["workspace_root"] = "/my/workspace/path"
basic_config["developer_id"] = "my_developer"

print("Basic config template:")
print(basic_config)

# Get multi-developer configuration template
multi_dev_config = get_config_template("multi_developer")
multi_dev_config["workspace_root"] = "/my/workspace/path"

print("Multi-developer config template:")
print(multi_dev_config)
```

## Integration Patterns

### Integration with Existing Validation Code

```python
# Example: Integrating workspace-aware file resolution with existing validation
from cursus.validation.workspace import WorkspaceManager

def validate_step_with_workspace(step_name, developer_id, workspace_root):
    """Validate a step using workspace-aware components."""
    
    # Initialize workspace manager
    manager = WorkspaceManager(workspace_root)
    
    # Get workspace-aware components
    file_resolver = manager.get_file_resolver(developer_id)
    module_loader = manager.get_module_loader(developer_id)
    
    # Find step components
    contract_file = file_resolver.find_contract_file(step_name)
    spec_file = file_resolver.find_spec_file(step_name)
    builder_file = file_resolver.find_builder_file(step_name)
    script_file = file_resolver.find_script_file(step_name)
    
    print(f"Validating step '{step_name}' for developer '{developer_id}':")
    print(f"  Contract: {contract_file}")
    print(f"  Spec: {spec_file}")
    print(f"  Builder: {builder_file}")
    print(f"  Script: {script_file}")
    
    # Load and validate builder class
    builder_class = module_loader.load_builder_class(step_name)
    if builder_class:
        print(f"  Builder class loaded: {builder_class.__name__}")
        # Perform validation logic here
        return True
    else:
        print(f"  Error: Could not load builder class for {step_name}")
        return False

# Usage
result = validate_step_with_workspace(
    "my_custom_step", 
    "developer_1", 
    "/path/to/developer_workspaces"
)
```

### Batch Processing Multiple Workspaces

```python
def validate_all_workspaces(workspace_root):
    """Validate all developer workspaces."""
    
    manager = WorkspaceManager(workspace_root)
    workspace_info = manager.discover_workspaces()
    
    results = {}
    
    for developer in workspace_info.developers:
        developer_id = developer.developer_id
        print(f"\nValidating workspace for {developer_id}...")
        
        try:
            # Get workspace-specific components
            file_resolver = manager.get_file_resolver(developer_id)
            module_loader = manager.get_module_loader(developer_id)
            
            # Discover available modules
            builders = module_loader.discover_workspace_modules("builders")
            dev_builders = builders.get(f"developer:{developer_id}", [])
            
            validation_results = []
            
            for builder_name in dev_builders:
                # Extract step name from builder name (remove _builder suffix)
                step_name = builder_name.replace("_builder", "")
                
                # Validate step components exist
                has_contract = file_resolver.find_contract_file(step_name) is not None
                has_spec = file_resolver.find_spec_file(step_name) is not None
                has_script = file_resolver.find_script_file(step_name) is not None
                
                # Try to load builder class
                builder_class = module_loader.load_builder_class(step_name)
                has_builder = builder_class is not None
                
                validation_results.append({
                    'step_name': step_name,
                    'has_contract': has_contract,
                    'has_spec': has_spec,
                    'has_script': has_script,
                    'has_builder': has_builder,
                    'is_complete': all([has_contract, has_spec, has_script, has_builder])
                })
            
            results[developer_id] = validation_results
            
        except Exception as e:
            print(f"Error validating {developer_id}: {e}")
            results[developer_id] = {'error': str(e)}
    
    return results

# Usage
results = validate_all_workspaces("/path/to/developer_workspaces")
for developer_id, validation_results in results.items():
    if isinstance(validation_results, dict) and 'error' in validation_results:
        print(f"{developer_id}: ERROR - {validation_results['error']}")
    else:
        complete_steps = [r for r in validation_results if r['is_complete']]
        print(f"{developer_id}: {len(complete_steps)}/{len(validation_results)} steps complete")
```

### Workspace Switching

```python
# Example: Processing multiple developers with the same components
def process_step_across_developers(step_name, workspace_root, developers):
    """Process the same step across multiple developer workspaces."""
    
    manager = WorkspaceManager(workspace_root)
    results = {}
    
    for developer_id in developers:
        print(f"\nProcessing {step_name} for {developer_id}...")
        
        # Get components for this developer
        file_resolver = manager.get_file_resolver(developer_id)
        module_loader = manager.get_module_loader(developer_id)
        
        # Find and load components
        builder_class = module_loader.load_builder_class(step_name)
        contract_file = file_resolver.find_contract_file(step_name)
        spec_file = file_resolver.find_spec_file(step_name)
        
        results[developer_id] = {
            'builder_available': builder_class is not None,
            'contract_file': contract_file,
            'spec_file': spec_file,
            'builder_class': builder_class.__name__ if builder_class else None
        }
    
    return results

# Usage
developers = ["developer_1", "developer_2", "developer_3"]
results = process_step_across_developers(
    "custom_processing_step", 
    "/path/to/developer_workspaces", 
    developers
)

for dev_id, result in results.items():
    print(f"{dev_id}: Builder={result['builder_available']}, "
          f"Contract={result['contract_file'] is not None}")
```

## Error Handling

### Handling Missing Workspaces

```python
from cursus.validation.workspace import WorkspaceManager

try:
    # This will raise ValueError if workspace doesn't exist
    manager = WorkspaceManager("/nonexistent/workspace/path")
except ValueError as e:
    print(f"Workspace error: {e}")
    
    # Create workspace structure if needed
    manager = WorkspaceManager()
    workspace_path = manager.create_developer_workspace(
        "new_developer",
        workspace_root="/path/to/new/workspaces",
        create_structure=True
    )
    print(f"Created new workspace at: {workspace_path}")
```

### Handling Module Loading Errors

```python
module_loader = WorkspaceModuleLoader(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1"
)

# Handle missing builder class
try:
    builder_class = module_loader.load_builder_class("nonexistent_step")
    if builder_class is None:
        print("Builder class not found - this is expected behavior")
except Exception as e:
    print(f"Unexpected error loading builder: {e}")

# Handle module loading from invalid file
try:
    module = module_loader.load_module_from_file("/nonexistent/file.py")
    if module is None:
        print("Module file not found - this is expected behavior")
except Exception as e:
    print(f"Error loading module from file: {e}")
```

### Validation Error Handling

```python
def safe_workspace_validation(workspace_root):
    """Safely validate workspace with comprehensive error handling."""
    
    try:
        manager = WorkspaceManager(workspace_root)
        
        # Validate workspace structure
        is_valid, issues = manager.validate_workspace_structure()
        
        if not is_valid:
            print("Workspace validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        # Discover workspaces
        workspace_info = manager.discover_workspaces()
        
        if workspace_info.total_developers == 0:
            print("Warning: No developer workspaces found")
            return False
        
        print(f"Successfully validated workspace with {workspace_info.total_developers} developers")
        return True
        
    except ValueError as e:
        print(f"Workspace configuration error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during validation: {e}")
        return False

# Usage
if safe_workspace_validation("/path/to/developer_workspaces"):
    print("Workspace is ready for use")
else:
    print("Workspace validation failed")
```

## Advanced Usage

### Custom File Resolution Patterns

```python
# Extend DeveloperWorkspaceFileResolver for custom behavior
class CustomWorkspaceFileResolver(DeveloperWorkspaceFileResolver):
    """Custom file resolver with additional search patterns."""
    
    def find_custom_file_type(self, step_name, file_type):
        """Find custom file types not covered by standard resolver."""
        
        if not self.workspace_mode:
            return None
        
        # Custom search logic
        custom_dir = getattr(self, f'{file_type}_dir', None)
        if custom_dir:
            return self._find_file_in_directory(
                custom_dir, step_name, None, ['.py', '.json', '.yaml']
            )
        
        return None

# Usage
custom_resolver = CustomWorkspaceFileResolver(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1"
)

# Use custom resolution
custom_file = custom_resolver.find_custom_file_type("my_step", "custom_type")
```

### Workspace-Aware Context Managers

```python
from contextlib import contextmanager

@contextmanager
def workspace_context(workspace_root, developer_id):
    """Context manager for workspace operations."""
    
    manager = WorkspaceManager(workspace_root)
    file_resolver = manager.get_file_resolver(developer_id)
    module_loader = manager.get_module_loader(developer_id)
    
    try:
        yield {
            'manager': manager,
            'file_resolver': file_resolver,
            'module_loader': module_loader
        }
    finally:
        # Cleanup operations
        module_loader.clear_cache()
        print(f"Workspace context closed for {developer_id}")

# Usage
with workspace_context("/path/to/workspaces", "developer_1") as ctx:
    # Use workspace components
    builder_class = ctx['module_loader'].load_builder_class("my_step")
    contract_file = ctx['file_resolver'].find_contract_file("my_step")
    
    print(f"Builder: {builder_class}")
    print(f"Contract: {contract_file}")

# Cache is automatically cleared when context exits
```

### Performance Optimization

```python
# Pre-warm module cache for better performance
def preload_workspace_modules(workspace_root, developer_id, step_names):
    """Pre-load modules for better performance."""
    
    module_loader = WorkspaceModuleLoader(
        workspace_root=workspace_root,
        developer_id=developer_id,
        cache_modules=True
    )
    
    loaded_count = 0
    
    for step_name in step_names:
        try:
            builder_class = module_loader.load_builder_class(step_name)
            contract_class = module_loader.load_contract_class(step_name)
            
            if builder_class:
                loaded_count += 1
            if contract_class:
                loaded_count += 1
                
        except Exception as e:
            print(f"Warning: Could not preload {step_name}: {e}")
    
    print(f"Preloaded {loaded_count} modules for {developer_id}")
    return module_loader

# Usage
step_names = ["step1", "step2", "step3", "step4"]
module_loader = preload_workspace_modules(
    "/path/to/workspaces", 
    "developer_1", 
    step_names
)

# Subsequent loads will be faster due to caching
for step_name in step_names:
    builder_class = module_loader.load_builder_class(step_name)  # Fast cache hit
```

### Workspace Monitoring

```python
import time
from pathlib import Path

def monitor_workspace_changes(workspace_root, developer_id, check_interval=5):
    """Monitor workspace for file changes."""
    
    manager = WorkspaceManager(workspace_root)
    file_resolver = manager.get_file_resolver(developer_id)
    
    # Get initial state
    workspace_path = Path(workspace_root) / "developers" / developer_id
    last_modified = workspace_path.stat().st_mtime if workspace_path.exists() else 0
    
    print(f"Monitoring workspace for {developer_id}...")
    
    while True:
        try:
            current_modified = workspace_path.stat().st_mtime
            
            if current_modified > last_modified:
                print(f"Workspace change detected for {developer_id}")
                
                # Refresh workspace info
                workspace_info = manager.discover_workspaces()
                dev_info = next(
                    (dev for dev in workspace_info.developers if dev.developer_id == developer_id),
                    None
                )
                
                if dev_info:
                    print(f"  Module count: {dev_info.module_count}")
                    print(f"  Has builders: {dev_info.has_builders}")
                    print(f"  Has contracts: {dev_info.has_contracts}")
                
                last_modified = current_modified
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("Monitoring stopped")
            break
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(check_interval)

# Usage (run in background or separate process)
# monitor_workspace_changes("/path/to/workspaces", "developer_1")
```

### Integration with Testing Frameworks

```python
import unittest
from cursus.validation.workspace import WorkspaceManager

class WorkspaceTestCase(unittest.TestCase):
    """Base test case for workspace-aware testing."""
    
    def setUp(self):
        """Set up workspace test environment."""
        self.workspace_root = "/path/to/test/workspaces"
        self.manager = WorkspaceManager(self.workspace_root)
        self.developer_id = "test_developer"
        
        # Ensure test workspace exists
        try:
            self.manager.create_developer_workspace(
                self.developer_id,
                create_structure=True
            )
        except ValueError:
            # Workspace already exists
            pass
    
    def get_workspace_components(self):
        """Get workspace components for testing."""
        return {
            'file_resolver': self.manager.get_file_resolver(self.developer_id),
            'module_loader': self.manager.get_module_loader(self.developer_id)
        }
    
    def test_step_validation(self):
        """Test step validation in workspace context."""
        components = self.get_workspace_components()
        
        # Test file resolution
        contract_file = components['file_resolver'].find_contract_file("test_step")
        self.assertIsNotNone(contract_file, "Contract file should be found")
        
        # Test module loading
        builder_class = components['module_loader'].load_builder_class("test_step")
        self.assertIsNotNone(builder_class, "Builder class should be loaded")

# Usage
if __name__ == '__main__':
    unittest.main()
```

## Best Practices

### 1. Workspace Organization

```python
# Good: Organize workspaces by developer/team
workspace_structure = {
    "developers": {
        "team_a_dev1": "Individual developer workspace",
        "team_a_dev2": "Individual developer workspace", 
        "team_b_dev1": "Individual developer workspace"
    },
    "shared": "Common components and utilities"
}

# Good: Use consistent naming conventions
step_naming = {
    "builder": "my_custom_step_builder.py",
    "contract": "my_custom_step_contract.py",
    "spec": "my_custom_step_spec.json",
    "script": "my_custom_step_script.py"
}
```

### 2. Error Handling

```python
# Good: Always handle workspace errors gracefully
def safe_workspace_operation(workspace_root, developer_id, operation):
    """Safely perform workspace operations with error handling."""
    
    try:
        manager = WorkspaceManager(workspace_root)
        components = {
            'file_resolver': manager.get_file_resolver(developer_id),
            'module_loader': manager.get_module_loader(developer_id)
        }
        
        return operation(components)
        
    except ValueError as e:
        print(f"Workspace configuration error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### 3. Performance Considerations

```python
# Good: Use caching for repeated operations
module_loader = WorkspaceModuleLoader(
    workspace_root=workspace_root,
    developer_id=developer_id,
    cache_modules=True  # Enable caching
)

# Good: Use context managers for sys.path manipulation
with module_loader.workspace_path_context():
    # Perform imports here
    pass
# sys.path automatically restored

# Good: Clear cache when workspace changes
if workspace_changed:
    module_loader.clear_cache()
```

### 4. Configuration Management

```python
# Good: Use configuration files for workspace settings
config = WorkspaceConfig(
    workspace_root="/path/to/workspaces",
    developer_id="default_developer",
    enable_shared_fallback=True,
    cache_modules=True,
    validation_settings={
        "strict_validation": False,
        "validate_imports": True
    }
)

# Save configuration for reuse
manager = WorkspaceManager()
manager.save_config("workspace.json", config)
```

This comprehensive guide covers all major aspects of using the Cursus workspace validation system. The examples progress from basic usage to advanced patterns, providing a complete reference for developers working with multi-developer workspace environments.
