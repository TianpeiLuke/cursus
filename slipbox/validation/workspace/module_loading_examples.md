# Module Loading Examples

This document provides examples of using the workspace-aware module loading system.

## Basic Module Loading

### Using WorkspaceModuleLoader

```python
from cursus.validation.workspace.module_loader import WorkspaceModuleLoader

# Initialize loader for a specific developer workspace
loader = WorkspaceModuleLoader(
    workspace_root="/path/to/workspaces",
    developer_id="developer_1"
)

# Load a builder module
builder_module = loader.load_builder_module("xgboost_trainer")
# Access the builder class
XGBoostTrainer = builder_module.XGBoostTrainer

# Load a contract module
contract_module = loader.load_contract_module("training_contract")
TrainingContract = contract_module.TrainingContract

# Load a script module
script_module = loader.load_script_module("train_model")
# Access script functions
train_function = script_module.train_model
```

### Context Manager Usage

```python
# Use context manager for automatic cleanup
with loader.workspace_context():
    # Load modules within isolated context
    builder_module = loader.load_builder_module("custom_trainer")
    CustomTrainer = builder_module.CustomTrainer
    
    # Create instance and use
    trainer = CustomTrainer(config={})
    result = trainer.build_step()
    
# Context automatically cleaned up, sys.path restored
```

## Advanced Module Loading

### Loading with Custom Import Paths

```python
# Load module with specific import requirements
loader = WorkspaceModuleLoader(
    workspace_root="/path/to/workspaces",
    developer_id="developer_1",
    additional_paths=["/custom/lib", "/shared/utils"]
)

# Load module that depends on custom libraries
advanced_module = loader.load_builder_module("advanced_trainer")
```

### Dynamic Class Discovery

```python
def discover_builder_classes(loader, module_name):
    """Discover all builder classes in a module"""
    module = loader.load_builder_module(module_name)
    
    builder_classes = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type) and 
            hasattr(attr, 'build_step') and 
            attr_name != 'BaseStepBuilder'):
            builder_classes.append(attr)
    
    return builder_classes

# Usage
with loader.workspace_context():
    builders = discover_builder_classes(loader, "multi_trainer")
    print(f"Found {len(builders)} builder classes")
    
    for builder_class in builders:
        print(f"Builder: {builder_class.__name__}")
```

### Module Reloading

```python
# Load module initially
builder_module = loader.load_builder_module("dynamic_trainer")
original_class = builder_module.DynamicTrainer

# Reload module (useful during development)
reloaded_module = loader.reload_module("builders", "dynamic_trainer")
updated_class = reloaded_module.DynamicTrainer

print(f"Classes are same: {original_class is updated_class}")  # False
```

## Error Handling

### Handling Import Errors

```python
from cursus.validation.workspace.exceptions import WorkspaceModuleLoadError

try:
    module = loader.load_builder_module("problematic_builder")
except WorkspaceModuleLoadError as e:
    print(f"Failed to load module: {e}")
    print(f"Error details: {e.original_error}")
    
    # Try fallback module
    try:
        module = loader.load_builder_module("default_builder")
        print("Using fallback builder")
    except WorkspaceModuleLoadError:
        print("No fallback available")
```

### Validation Before Loading

```python
def safe_module_load(loader, module_type, module_name):
    """Safely load module with validation"""
    
    # Check if file exists first
    file_method = getattr(loader.file_resolver, f"resolve_{module_type}_file")
    try:
        module_path = file_method(module_name)
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Module file not found: {module_path}")
    except Exception as e:
        print(f"File resolution failed: {e}")
        return None
    
    # Attempt to load module
    try:
        load_method = getattr(loader, f"load_{module_type}_module")
        return load_method(module_name)
    except WorkspaceModuleLoadError as e:
        print(f"Module loading failed: {e}")
        return None

# Usage
module = safe_module_load(loader, "builder", "experimental_trainer")
if module:
    print("Module loaded successfully")
else:
    print("Module loading failed")
```

## Integration with Validation

### Loading Modules for Validation

```python
from cursus.validation.workspace.workspace_builder_test import WorkspaceUniversalStepBuilderTest

# Initialize builder tester
tester = WorkspaceUniversalStepBuilderTest(
    workspace_root="/path/to/workspaces"
)

# Switch to developer workspace
tester.switch_developer("developer_1")

# Test uses module loader internally
test_result = tester.run_workspace_builder_test("xgboost_trainer")

print(f"Builder test passed: {test_result.passed}")
if not test_result.passed:
    print(f"Errors: {test_result.errors}")
```

### Custom Module Validation

```python
def validate_builder_module(loader, module_name):
    """Validate a builder module meets requirements"""
    
    try:
        with loader.workspace_context():
            module = loader.load_builder_module(module_name)
            
            # Check required attributes
            required_attrs = ['build_step', '__init__']
            missing_attrs = []
            
            for attr in required_attrs:
                if not hasattr(module, attr):
                    missing_attrs.append(attr)
            
            if missing_attrs:
                return False, f"Missing attributes: {missing_attrs}"
            
            # Check if class can be instantiated
            try:
                builder_class = getattr(module, module_name.title().replace('_', ''))
                instance = builder_class(config={})
                return True, "Module validation passed"
            except Exception as e:
                return False, f"Instantiation failed: {e}"
                
    except WorkspaceModuleLoadError as e:
        return False, f"Module loading failed: {e}"

# Usage
is_valid, message = validate_builder_module(loader, "custom_trainer")
print(f"Validation result: {message}")
```

## Multi-Developer Module Loading

### Switching Between Developer Workspaces

```python
# Load modules from different developers
developers = ["developer_1", "developer_2", "developer_3"]
loaded_modules = {}

for dev_id in developers:
    loader.switch_developer(dev_id)
    
    try:
        with loader.workspace_context():
            module = loader.load_builder_module("common_trainer")
            loaded_modules[dev_id] = module
            print(f"Loaded module from {dev_id}")
    except WorkspaceModuleLoadError:
        print(f"No common_trainer found for {dev_id}")

print(f"Loaded modules from {len(loaded_modules)} developers")
```

### Cross-Developer Module Comparison

```python
def compare_builder_implementations(loader, builder_name, developers):
    """Compare builder implementations across developers"""
    
    implementations = {}
    
    for dev_id in developers:
        loader.switch_developer(dev_id)
        
        try:
            with loader.workspace_context():
                module = loader.load_builder_module(builder_name)
                builder_class = getattr(module, builder_name.title().replace('_', ''))
                
                # Get method signatures
                methods = {}
                for method_name in dir(builder_class):
                    if not method_name.startswith('_'):
                        method = getattr(builder_class, method_name)
                        if callable(method):
                            methods[method_name] = str(inspect.signature(method))
                
                implementations[dev_id] = {
                    'class_name': builder_class.__name__,
                    'methods': methods,
                    'docstring': builder_class.__doc__
                }
                
        except Exception as e:
            implementations[dev_id] = {'error': str(e)}
    
    return implementations

# Usage
import inspect
comparison = compare_builder_implementations(
    loader, 
    "xgboost_trainer", 
    ["developer_1", "developer_2"]
)

for dev_id, info in comparison.items():
    print(f"\n{dev_id}:")
    if 'error' in info:
        print(f"  Error: {info['error']}")
    else:
        print(f"  Class: {info['class_name']}")
        print(f"  Methods: {list(info['methods'].keys())}")
```

## Best Practices

### 1. Always Use Context Managers

```python
# Good: Use context manager for isolation
with loader.workspace_context():
    module = loader.load_builder_module("trainer")
    # Use module
    result = module.SomeClass().method()

# Bad: Direct loading without context
# module = loader.load_builder_module("trainer")  # May pollute sys.path
```

### 2. Handle Import Dependencies

```python
def load_with_dependencies(loader, module_name, dependencies=None):
    """Load module with its dependencies"""
    
    dependencies = dependencies or []
    loaded_deps = {}
    
    with loader.workspace_context():
        # Load dependencies first
        for dep in dependencies:
            try:
                loaded_deps[dep] = loader.load_builder_module(dep)
            except WorkspaceModuleLoadError as e:
                print(f"Warning: Could not load dependency {dep}: {e}")
        
        # Load main module
        main_module = loader.load_builder_module(module_name)
        
        return main_module, loaded_deps

# Usage
main_module, deps = load_with_dependencies(
    loader, 
    "complex_trainer",
    dependencies=["base_trainer", "utils_module"]
)
```

### 3. Cache Loaded Modules

```python
class CachedModuleLoader:
    """Module loader with caching capability"""
    
    def __init__(self, workspace_root, developer_id):
        self.loader = WorkspaceModuleLoader(workspace_root, developer_id)
        self.cache = {}
    
    def load_cached_module(self, module_type, module_name):
        """Load module with caching"""
        cache_key = f"{self.loader.developer_id}:{module_type}:{module_name}"
        
        if cache_key not in self.cache:
            load_method = getattr(self.loader, f"load_{module_type}_module")
            with self.loader.workspace_context():
                self.cache[cache_key] = load_method(module_name)
        
        return self.cache[cache_key]
    
    def clear_cache(self):
        """Clear module cache"""
        self.cache.clear()

# Usage
cached_loader = CachedModuleLoader("/path/to/workspaces", "developer_1")
module1 = cached_loader.load_cached_module("builder", "trainer")  # Loads from file
module2 = cached_loader.load_cached_module("builder", "trainer")  # Returns cached
```

### 4. Module Health Checks

```python
def health_check_module(loader, module_type, module_name):
    """Perform health check on a module"""
    
    health_report = {
        'module_name': module_name,
        'loadable': False,
        'has_required_classes': False,
        'instantiable': False,
        'errors': []
    }
    
    try:
        # Test loading
        with loader.workspace_context():
            module = getattr(loader, f"load_{module_type}_module")(module_name)
            health_report['loadable'] = True
            
            # Test for required classes
            expected_class = module_name.title().replace('_', '')
            if hasattr(module, expected_class):
                health_report['has_required_classes'] = True
                
                # Test instantiation
                try:
                    cls = getattr(module, expected_class)
                    instance = cls(config={})
                    health_report['instantiable'] = True
                except Exception as e:
                    health_report['errors'].append(f"Instantiation error: {e}")
            else:
                health_report['errors'].append(f"Missing class: {expected_class}")
                
    except Exception as e:
        health_report['errors'].append(f"Loading error: {e}")
    
    return health_report

# Usage
report = health_check_module(loader, "builder", "xgboost_trainer")
print(f"Module health: {report}")
