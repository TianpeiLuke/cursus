# Registry API Reference

Complete reference for the Cursus Registry System API, covering all functions, classes, and methods for step registration, discovery, and management.

## Table of Contents

1. [Core Registry Functions](#core-registry-functions)
2. [Step Names Registry](#step-names-registry)
3. [Builder Registry](#builder-registry)
4. [Unified Registry Manager](#unified-registry-manager)
5. [Workspace-Aware Registry](#workspace-aware-registry)
6. [Validation and Enforcement](#validation-and-enforcement)
7. [Export and Import](#export-and-import)
8. [Utility Functions](#utility-functions)
9. [Error Handling](#error-handling)
10. [Advanced Usage](#advanced-usage)

---

## Core Registry Functions

### `get_step_names()`

Returns the complete step names registry with workspace awareness.

```python
from cursus.registry import get_step_names

# Get all registered steps
step_names = get_step_names()
print(f"Total steps: {len(step_names)}")

# Access specific step
processing_step = step_names.get('processing')
if processing_step:
    print(f"Config class: {processing_step['config_class']}")
    print(f"Builder: {processing_step['builder_step_name']}")
```

**Returns:** `Dict[str, Dict[str, Any]]` - Complete step registry

**Workspace Context:** Automatically includes workspace-specific steps when in workspace context.

### `get_available_steps()`

Get list of available step names with optional filtering.

```python
from cursus.registry import get_available_steps

# Get all step names
all_steps = get_available_steps()
print(f"Available steps: {all_steps}")

# Filter by step type
ml_steps = get_available_steps(step_type='ml')
processing_steps = get_available_steps(step_type='processing')
```

**Parameters:**
- `step_type` (str, optional): Filter by step type
- `include_deprecated` (bool, default=False): Include deprecated steps

**Returns:** `List[str]` - List of step names

### `register_step()`

Register a new step in the registry with validation.

```python
from cursus.registry import register_step

# Register new step
step_definition = {
    'config_class': 'MyCustomConfig',
    'builder_step_name': 'my_custom_step',
    'spec_type': 'custom',
    'sagemaker_step_type': 'ProcessingStep',
    'description': 'Custom processing step for specialized tasks'
}

success = register_step('my_custom', step_definition)
if success:
    print("Step registered successfully")
```

**Parameters:**
- `step_name` (str): Unique step identifier
- `step_definition` (dict): Step configuration dictionary
- `validate` (bool, default=True): Enable validation
- `permanent` (bool, default=False): Make registration permanent

**Returns:** `bool` - Registration success status

---

## Step Names Registry

### `StepNamesRegistry`

Core registry class for managing step definitions.

```python
from cursus.registry.step_names import StepNamesRegistry

# Initialize registry
registry = StepNamesRegistry()

# Get step definition
step_def = registry.get_step('processing')
print(f"Step definition: {step_def}")

# Check if step exists
if registry.has_step('custom_step'):
    print("Custom step is registered")

# List all steps
all_steps = registry.list_steps()
print(f"Registered steps: {all_steps}")
```

#### Methods

##### `get_step(step_name: str) -> Optional[Dict[str, Any]]`

Retrieve step definition by name.

```python
# Get step with error handling
step = registry.get_step('processing')
if step:
    config_class = step['config_class']
    builder_name = step['builder_step_name']
```

##### `has_step(step_name: str) -> bool`

Check if step is registered.

```python
# Check step existence
if registry.has_step('my_custom_step'):
    print("Step exists")
else:
    print("Step not found")
```

##### `list_steps(category: Optional[str] = None) -> List[str]`

List registered steps with optional category filtering.

```python
# List all steps
all_steps = registry.list_steps()

# List by category
ml_steps = registry.list_steps(category='ml')
```

##### `add_step(step_name: str, definition: Dict[str, Any]) -> bool`

Add new step to registry.

```python
# Add step with validation
definition = {
    'config_class': 'CustomConfig',
    'builder_step_name': 'custom_builder',
    'spec_type': 'custom',
    'sagemaker_step_type': 'ProcessingStep',
    'description': 'Custom step description'
}

success = registry.add_step('custom_step', definition)
```

---

## Builder Registry

### `StepBuilderRegistry`

Registry for step builders with automatic discovery.

```python
from cursus.registry.builder_registry import StepBuilderRegistry

# Initialize builder registry
builder_registry = StepBuilderRegistry()

# Register builder
builder_registry.register_builder('my_step', MyStepBuilder)

# Get builder
builder_class = builder_registry.get_builder('my_step')
if builder_class:
    builder = builder_class()
```

#### Methods

##### `register_builder(name: str, builder_class: Type) -> None`

Register a step builder class.

```python
from cursus.steps.base import BaseStepBuilder

class CustomStepBuilder(BaseStepBuilder):
    def build_step(self, config):
        # Implementation
        pass

# Register the builder
builder_registry.register_builder('custom', CustomStepBuilder)
```

##### `get_builder(name: str) -> Optional[Type]`

Retrieve builder class by name.

```python
# Get and instantiate builder
BuilderClass = builder_registry.get_builder('processing')
if BuilderClass:
    builder = BuilderClass()
    step = builder.build_step(config)
```

##### `list_builders() -> List[str]`

List all registered builders.

```python
# Get all builder names
builders = builder_registry.list_builders()
print(f"Available builders: {builders}")
```

##### `discover_builders(package_path: str) -> int`

Automatically discover and register builders from package.

```python
# Discover builders in custom package
count = builder_registry.discover_builders('my_package.steps')
print(f"Discovered {count} builders")
```

---

## Unified Registry Manager

### `UnifiedRegistryManager`

Hybrid registry manager with workspace awareness.

```python
from cursus.registry.unified import UnifiedRegistryManager

# Initialize unified manager
manager = UnifiedRegistryManager()

# Get step with workspace context
step = manager.get_step('processing', workspace_context=True)

# Register step with workspace awareness
manager.register_step('custom', definition, workspace_scope='local')
```

#### Methods

##### `get_step(name: str, workspace_context: bool = True) -> Optional[Dict]`

Get step with optional workspace context.

```python
# Get step with workspace awareness
step = manager.get_step('processing', workspace_context=True)

# Get step without workspace context
step = manager.get_step('processing', workspace_context=False)
```

##### `register_step(name: str, definition: Dict, workspace_scope: str = 'global') -> bool`

Register step with workspace scope control.

```python
# Register globally
manager.register_step('global_step', definition, workspace_scope='global')

# Register locally to workspace
manager.register_step('local_step', definition, workspace_scope='local')

# Register with project scope
manager.register_step('project_step', definition, workspace_scope='project')
```

##### `sync_registries() -> bool`

Synchronize workspace and global registries.

```python
# Sync registries
success = manager.sync_registries()
if success:
    print("Registries synchronized")
```

---

## Workspace-Aware Registry

### `WorkspaceComponentRegistry`

The primary workspace-aware registry for component discovery and management across developer workspaces.

```python
from cursus.workspace.core.registry import WorkspaceComponentRegistry

# Initialize workspace component registry
workspace_registry = WorkspaceComponentRegistry('/path/to/workspace')

# Discover all components
components = workspace_registry.discover_components()
print(f"Total components: {components['summary']['total_components']}")

# Find builder for specific developer
builder_class = workspace_registry.find_builder_class('processing', 'developer_1')
```

#### Methods

##### `discover_components(developer_id: str = None) -> Dict[str, Any]`

Discover components in workspace(s) with optional developer filtering.

```python
# Discover all components
all_components = workspace_registry.discover_components()

# Discover components for specific developer
dev_components = workspace_registry.discover_components('developer_1')

# Access discovered components
builders = all_components['builders']
configs = all_components['configs']
contracts = all_components['contracts']
specs = all_components['specs']
scripts = all_components['scripts']

# Check summary
summary = all_components['summary']
print(f"Developers: {summary['developers']}")
print(f"Step types: {summary['step_types']}")
```

##### `find_builder_class(step_name: str, developer_id: str = None) -> Optional[Type]`

Find builder class with workspace awareness and core registry fallback.

```python
# Find builder for specific developer
builder_class = workspace_registry.find_builder_class('processing', 'developer_1')

# Find builder across all developers
builder_class = workspace_registry.find_builder_class('processing')

# Use the builder
if builder_class:
    builder = builder_class()
    step = builder.build_step(config)
```

##### `find_config_class(step_name: str, developer_id: str = None) -> Optional[Type]`

Find config class with workspace awareness.

```python
# Find config for specific developer
config_class = workspace_registry.find_config_class('processing', 'developer_1')

# Find config across all developers
config_class = workspace_registry.find_config_class('processing')

# Use the config
if config_class:
    config = config_class()
```

##### `get_workspace_summary() -> Dict[str, Any]`

Get comprehensive summary of workspace components.

```python
# Get workspace summary
summary = workspace_registry.get_workspace_summary()

print(f"Workspace root: {summary['workspace_root']}")
print(f"Total components: {summary['total_components']}")
print(f"Developers: {summary['developers']}")
print(f"Step types: {summary['step_types']}")

# Component counts by type
counts = summary['component_counts']
print(f"Builders: {counts['builders']}")
print(f"Configs: {counts['configs']}")
print(f"Contracts: {counts['contracts']}")
print(f"Specs: {counts['specs']}")
print(f"Scripts: {counts['scripts']}")
```

##### `validate_component_availability(workspace_config) -> Dict[str, Any]`

Validate component availability for pipeline assembly.

```python
# Validate components for pipeline
validation_result = workspace_registry.validate_component_availability(workspace_config)

if validation_result['valid']:
    print("All components available")
    for component in validation_result['available_components']:
        print(f"✓ {component['step_name']} ({component['developer_id']})")
else:
    print("Missing components:")
    for missing in validation_result['missing_components']:
        print(f"✗ {missing['step_name']} ({missing['developer_id']})")

# Check warnings
for warning in validation_result['warnings']:
    print(f"⚠ {warning}")
```

##### `clear_cache() -> None`

Clear component discovery cache.

```python
# Clear cache to force fresh discovery
workspace_registry.clear_cache()
```

### Workspace Context Functions

Functions for managing workspace-specific registry behavior.

```python
from cursus.registry.workspace import (
    get_workspace_context,
    set_workspace_context,
    clear_workspace_context
)

# Get current workspace context
context = get_workspace_context()
print(f"Current workspace: {context}")

# Set workspace context
set_workspace_context('/path/to/workspace')

# Clear workspace context
clear_workspace_context()
```

---

## Validation and Enforcement

### `RegistryValidator`

Validation system for registry entries.

```python
from cursus.registry.validation import RegistryValidator

# Initialize validator
validator = RegistryValidator()

# Validate step definition
definition = {
    'config_class': 'MyConfig',
    'builder_step_name': 'my_builder',
    'spec_type': 'custom',
    'sagemaker_step_type': 'ProcessingStep',
    'description': 'Custom step'
}

is_valid, errors = validator.validate_step_definition(definition)
if not is_valid:
    print(f"Validation errors: {errors}")
```

#### Methods

##### `validate_step_definition(definition: Dict) -> Tuple[bool, List[str]]`

Validate step definition structure and content.

```python
# Validate definition
is_valid, errors = validator.validate_step_definition(definition)
if is_valid:
    print("Definition is valid")
else:
    for error in errors:
        print(f"Error: {error}")
```

##### `validate_step_name(name: str) -> Tuple[bool, str]`

Validate step name format and uniqueness.

```python
# Validate step name
is_valid, message = validator.validate_step_name('my_custom_step')
if not is_valid:
    print(f"Name validation failed: {message}")
```

##### `enforce_standards(definition: Dict) -> Dict`

Enforce standardization rules on definition.

```python
# Enforce standards
standardized = validator.enforce_standards(definition)
print(f"Standardized definition: {standardized}")
```

---

## Export and Import

### JSON Export Functions

Export registry data to JSON format.

```python
from cursus.registry.export import (
    export_registry_to_json,
    import_registry_from_json,
    export_step_to_json
)

# Export entire registry
json_data = export_registry_to_json()
with open('registry_export.json', 'w') as f:
    json.dump(json_data, f, indent=2)

# Export specific step
step_json = export_step_to_json('processing')
print(f"Step JSON: {step_json}")

# Import from JSON
with open('registry_import.json', 'r') as f:
    import_data = json.load(f)
success = import_registry_from_json(import_data)
```

### `RegistryExporter`

Advanced export functionality.

```python
from cursus.registry.export import RegistryExporter

# Initialize exporter
exporter = RegistryExporter()

# Export with filtering
filtered_export = exporter.export_filtered(
    step_types=['ml', 'processing'],
    include_deprecated=False
)

# Export with metadata
export_with_meta = exporter.export_with_metadata(
    include_timestamps=True,
    include_workspace_info=True
)
```

#### Methods

##### `export_filtered(step_types: List[str] = None, **kwargs) -> Dict`

Export registry with filtering options.

```python
# Export specific step types
ml_export = exporter.export_filtered(step_types=['ml'])

# Export non-deprecated steps
current_export = exporter.export_filtered(include_deprecated=False)
```

##### `export_with_metadata(include_timestamps: bool = True, **kwargs) -> Dict`

Export with additional metadata.

```python
# Export with full metadata
full_export = exporter.export_with_metadata(
    include_timestamps=True,
    include_workspace_info=True,
    include_validation_info=True
)
```

---

## Utility Functions

### Registry Inspection

```python
from cursus.registry.utils import (
    inspect_registry,
    get_registry_stats,
    find_step_dependencies,
    validate_registry_integrity
)

# Inspect registry
inspection = inspect_registry()
print(f"Registry inspection: {inspection}")

# Get statistics
stats = get_registry_stats()
print(f"Total steps: {stats['total_steps']}")
print(f"Step types: {stats['step_types']}")

# Find dependencies
deps = find_step_dependencies('processing')
print(f"Dependencies: {deps}")

# Validate integrity
is_valid, issues = validate_registry_integrity()
if not is_valid:
    print(f"Registry issues: {issues}")
```

### Registry Debugging

```python
from cursus.registry.debug import (
    debug_step_resolution,
    trace_registry_access,
    profile_registry_performance
)

# Debug step resolution
debug_info = debug_step_resolution('processing')
print(f"Resolution path: {debug_info['resolution_path']}")

# Trace registry access
with trace_registry_access() as tracer:
    step = get_step_names()['processing']
    access_log = tracer.get_log()

# Profile performance
with profile_registry_performance() as profiler:
    # Registry operations
    steps = get_available_steps()
    performance_data = profiler.get_results()
```

---

## Error Handling

### Registry Exceptions

```python
from cursus.registry.exceptions import (
    RegistryError,
    StepNotFoundError,
    InvalidStepDefinitionError,
    RegistryValidationError,
    WorkspaceRegistryError
)

try:
    step = get_step_names()['nonexistent_step']
except StepNotFoundError as e:
    print(f"Step not found: {e}")

try:
    register_step('invalid', {})  # Invalid definition
except InvalidStepDefinitionError as e:
    print(f"Invalid definition: {e}")

try:
    # Workspace operation
    ws_manager.register_workspace_step('test', definition)
except WorkspaceRegistryError as e:
    print(f"Workspace error: {e}")
```

### Error Recovery

```python
from cursus.registry.recovery import (
    recover_registry,
    backup_registry,
    restore_registry
)

# Backup registry
backup_path = backup_registry()
print(f"Registry backed up to: {backup_path}")

# Recover from corruption
try:
    recovered = recover_registry()
    if recovered:
        print("Registry recovered successfully")
except RegistryError as e:
    print(f"Recovery failed: {e}")

# Restore from backup
restore_success = restore_registry(backup_path)
```

---

## Advanced Usage

### Custom Registry Backends

```python
from cursus.registry.backends import BaseRegistryBackend

class CustomRegistryBackend(BaseRegistryBackend):
    def get_step(self, name: str) -> Optional[Dict]:
        # Custom implementation
        pass
    
    def register_step(self, name: str, definition: Dict) -> bool:
        # Custom implementation
        pass

# Use custom backend
from cursus.registry import set_registry_backend
set_registry_backend(CustomRegistryBackend())
```

### Registry Plugins

```python
from cursus.registry.plugins import RegistryPlugin

class ValidationPlugin(RegistryPlugin):
    def on_step_register(self, name: str, definition: Dict):
        # Custom validation logic
        pass
    
    def on_step_access(self, name: str):
        # Access logging
        pass

# Register plugin
from cursus.registry import register_plugin
register_plugin(ValidationPlugin())
```

### Batch Operations

```python
from cursus.registry.batch import BatchRegistryOperations

# Initialize batch operations
batch = BatchRegistryOperations()

# Add multiple operations
batch.add_register_operation('step1', definition1)
batch.add_register_operation('step2', definition2)
batch.add_update_operation('existing_step', updated_definition)

# Execute batch
results = batch.execute()
for operation, success in results.items():
    print(f"Operation {operation}: {'Success' if success else 'Failed'}")
```

### Registry Monitoring

```python
from cursus.registry.monitoring import RegistryMonitor

# Initialize monitor
monitor = RegistryMonitor()

# Start monitoring
monitor.start()

# Get metrics
metrics = monitor.get_metrics()
print(f"Registry access count: {metrics['access_count']}")
print(f"Registration count: {metrics['registration_count']}")

# Stop monitoring
monitor.stop()
```

---

## Configuration

### Registry Configuration

```python
from cursus.registry.config import RegistryConfig

# Configure registry behavior
config = RegistryConfig(
    enable_workspace_awareness=True,
    auto_discovery=True,
    validation_level='strict',
    cache_enabled=True,
    backup_enabled=True
)

# Apply configuration
from cursus.registry import configure_registry
configure_registry(config)
```

### Environment Variables

Registry behavior can be controlled via environment variables:

```bash
# Enable debug mode
export CURSUS_REGISTRY_DEBUG=true

# Set workspace path
export CURSUS_WORKSPACE_PATH=/path/to/workspace

# Configure validation level
export CURSUS_REGISTRY_VALIDATION=strict

# Enable caching
export CURSUS_REGISTRY_CACHE=true
```

---

## Best Practices

### 1. Step Registration

```python
# Good: Complete definition with validation
definition = {
    'config_class': 'ProcessingConfig',
    'builder_step_name': 'processing_step_builder',
    'spec_type': 'processing',
    'sagemaker_step_type': 'ProcessingStep',
    'description': 'Data processing step with validation'
}

success = register_step('processing', definition, validate=True)
```

### 2. Error Handling

```python
# Good: Comprehensive error handling
try:
    step = get_step_names()['my_step']
    builder_class = get_builder(step['builder_step_name'])
    builder = builder_class()
except StepNotFoundError:
    logger.error("Step not found in registry")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

### 3. Workspace Awareness

```python
# Good: Explicit workspace context
with workspace_context('/path/to/workspace'):
    workspace_steps = get_available_steps()
    # Work with workspace-specific steps
```

### 4. Performance Optimization

```python
# Good: Cache frequently accessed steps
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_step(step_name: str):
    return get_step_names().get(step_name)
```

---

## Migration Guide

### From Legacy Registry

```python
# Legacy approach
from cursus.registry.step_names_original import STEP_NAMES
step_def = STEP_NAMES['processing']

# New approach
from cursus.registry import get_step_names
step_names = get_step_names()
step_def = step_names['processing']
```

### Workspace Migration

```python
# Migrate to workspace-aware registry
from cursus.registry.migration import migrate_to_workspace

# Migrate existing registry
migration_result = migrate_to_workspace(
    workspace_path='/path/to/workspace',
    preserve_global=True
)

if migration_result.success:
    print("Migration completed successfully")
else:
    print(f"Migration issues: {migration_result.issues}")
```

---

## Resources

- [Registry Quick Start Tutorial](./registry_quick_start.md)
- [Workspace Integration Guide](../workspace/workspace_api_reference.md)
- [Step Builder Development Guide](../../0_developer_guide/step_builder_registry_guide.md)
- [Registry Design Documentation](../../1_design/hybrid_registry_standardization_enforcement_design.md)

---

*Last updated: September 2025*
*Version: 1.0.0*
