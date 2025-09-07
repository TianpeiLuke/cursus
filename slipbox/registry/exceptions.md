---
tags:
  - code
  - registry
  - exceptions
  - error_handling
  - registry_errors
keywords:
  - RegistryError
  - RegistryLoadError
  - exception handling
  - error messages
  - registry exceptions
topics:
  - exception handling
  - registry errors
  - error management
language: python
date of note: 2024-12-07
---

# Registry Exceptions

Exception classes for the Pipeline Registry that provide clear, actionable error messages for registry-related operations and failures.

## Overview

The registry exceptions module defines custom exception classes used throughout the Pipeline Registry system to provide clear, actionable error messages. These exceptions are designed to help developers quickly identify and resolve registry-related issues, whether they involve missing step builders, failed registry loading, or configuration problems.

The exception classes include detailed context information such as unresolvable step types, available alternatives, registry paths, and helpful suggestions for resolution. This comprehensive error reporting enables efficient debugging and troubleshooting of registry-related issues.

## Classes and Methods

### Exception Classes
- [`RegistryError`](#registryerror) - Base exception for step builder registry errors
- [`RegistryLoadError`](#registryloaderror) - Exception for registry loading failures

## API Reference

### RegistryError

_class_ cursus.registry.exceptions.RegistryError(_message_, _unresolvable_types=None_, _available_builders=None_)

Raised when step builder registry errors occur. This exception provides detailed context about registry failures including unresolvable step types and available alternatives.

**Parameters:**
- **message** (_str_) – Primary error message describing the registry failure.
- **unresolvable_types** (_Optional[List[str]]_) – List of step types that could not be resolved. Defaults to None.
- **available_builders** (_Optional[List[str]]_) – List of available builder types for reference. Defaults to None.

```python
from cursus.registry.exceptions import RegistryError

# Basic registry error
try:
    builder = registry.get_builder_for_step_type("NonExistentStep")
except RegistryError as e:
    print(f"Registry error: {e}")
    print(f"Unresolvable types: {e.unresolvable_types}")
    print(f"Available builders: {e.available_builders}")

# Create custom registry error
error = RegistryError(
    "Failed to resolve step builders",
    unresolvable_types=["CustomStep", "AnotherStep"],
    available_builders=["XGBoostTraining", "CradleDataLoading"]
)
```

#### Attributes

**unresolvable_types** (_List[str]_) – List of step types that could not be resolved during the operation.

**available_builders** (_List[str]_) – List of available builder types that can be used as alternatives.

#### \_\_str\_\_

\_\_str\_\_()

Return a comprehensive string representation of the error including context information.

**Returns:**
- **str** – Formatted error message with unresolvable types and available builders.

```python
from cursus.registry.exceptions import RegistryError

# Create error with context
error = RegistryError(
    "Step builder not found",
    unresolvable_types=["InvalidStep"],
    available_builders=["XGBoostTraining", "CradleDataLoading", "Package"]
)

# Print comprehensive error message
print(str(error))
# Output:
# Step builder not found
# Unresolvable step types: ['InvalidStep']
# Available builders: ['XGBoostTraining', 'CradleDataLoading', 'Package']
```

### RegistryLoadError

_class_ cursus.registry.exceptions.RegistryLoadError(_message_, _registry_path=None_, _suggestions=None_)

Raised when registry loading fails. This specialized exception provides context about registry loading failures including file paths and helpful suggestions for resolution.

**Parameters:**
- **message** (_str_) – Primary error message describing the loading failure.
- **registry_path** (_Optional[str]_) – Path to the registry file that failed to load. Defaults to None.
- **suggestions** (_Optional[List[str]]_) – List of suggestions for resolving the loading issue. Defaults to None.

```python
from cursus.registry.exceptions import RegistryLoadError

# Handle registry loading error
try:
    registry_manager.load_workspace_registry("invalid_workspace")
except RegistryLoadError as e:
    print(f"Loading failed: {e}")
    print(f"Registry path: {e.registry_path}")
    print(f"Suggestions: {e.suggestions}")

# Create custom loading error
error = RegistryLoadError(
    "Failed to load workspace registry",
    registry_path="/path/to/workspace/registry.py",
    suggestions=[
        "Check if the registry file exists",
        "Verify file permissions",
        "Validate registry file syntax"
    ]
)
```

#### Attributes

**registry_path** (_Optional[str]_) – Path to the registry file that failed to load.

**suggestions** (_List[str]_) – List of helpful suggestions for resolving the loading issue.

#### \_\_str\_\_

\_\_str\_\_()

Return a comprehensive string representation of the loading error including path and suggestions.

**Returns:**
- **str** – Formatted error message with registry path and resolution suggestions.

```python
from cursus.registry.exceptions import RegistryLoadError

# Create loading error with context
error = RegistryLoadError(
    "Registry file not found",
    registry_path="/workspace/developer_1/registry.py",
    suggestions=[
        "Create the missing registry file",
        "Check workspace configuration",
        "Verify workspace path exists"
    ]
)

# Print comprehensive error message
print(str(error))
# Output:
# Registry file not found
# Registry path: /workspace/developer_1/registry.py
# Suggestions: Create the missing registry file, Check workspace configuration, Verify workspace path exists
```

## Usage Examples

### Basic Error Handling

```python
from cursus.registry import get_global_registry
from cursus.registry.exceptions import RegistryError, RegistryLoadError

# Get registry instance
registry = get_global_registry()

# Handle step builder resolution errors
def safe_get_builder(step_type):
    try:
        return registry.get_builder_for_step_type(step_type)
    except RegistryError as e:
        print(f"Failed to get builder for '{step_type}':")
        print(f"  Error: {e}")
        
        if e.unresolvable_types:
            print(f"  Unresolvable types: {e.unresolvable_types}")
        
        if e.available_builders:
            print(f"  Available alternatives: {e.available_builders[:5]}")  # Show first 5
            
        return None

# Test with various step types
test_types = ["XGBoostTraining", "InvalidStep", "NonExistentBuilder"]
for step_type in test_types:
    builder = safe_get_builder(step_type)
    if builder:
        print(f"✓ {step_type}: {builder.__name__}")
    else:
        print(f"✗ {step_type}: No builder found")
```

### Configuration Resolution Error Handling

```python
from cursus.registry import get_global_registry
from cursus.registry.exceptions import RegistryError

registry = get_global_registry()

def safe_get_builder_for_config(config, node_name=None):
    """Safely get builder for configuration with comprehensive error handling."""
    try:
        return registry.get_builder_for_config(config, node_name)
    except RegistryError as e:
        config_name = config.__class__.__name__
        print(f"Failed to resolve builder for config '{config_name}':")
        print(f"  Error: {e}")
        
        # Show context information
        if e.unresolvable_types:
            print(f"  Unresolvable types: {e.unresolvable_types}")
        
        if e.available_builders:
            print(f"  Available builders ({len(e.available_builders)}):")
            for builder in e.available_builders[:10]:  # Show first 10
                print(f"    - {builder}")
        
        # Suggest similar builders
        config_base = config_name.replace('Config', '').replace('Step', '')
        similar_builders = [b for b in (e.available_builders or []) if config_base.lower() in b.lower()]
        if similar_builders:
            print(f"  Similar builders: {similar_builders}")
        
        return None

# Test configuration resolution
test_configs = [
    XGBoostTrainingConfig(),
    type('UnknownConfig', (), {})(),  # Create unknown config
    CradleDataLoadingConfig(job_type="training")
]

for config in test_configs:
    builder = safe_get_builder_for_config(config)
    config_name = config.__class__.__name__
    if builder:
        print(f"✓ {config_name}: {builder.__name__}")
    else:
        print(f"✗ {config_name}: Resolution failed")
```

### Registry Loading Error Handling

```python
from cursus.registry.exceptions import RegistryLoadError
from cursus.registry.hybrid.manager import UnifiedRegistryManager

def safe_load_workspace_registry(workspace_id, workspace_path):
    """Safely load workspace registry with comprehensive error handling."""
    try:
        manager = UnifiedRegistryManager()
        manager.add_workspace_registry(workspace_id, workspace_path)
        print(f"✓ Successfully loaded workspace '{workspace_id}'")
        return True
        
    except RegistryLoadError as e:
        print(f"Failed to load workspace '{workspace_id}':")
        print(f"  Error: {e}")
        
        if e.registry_path:
            print(f"  Registry path: {e.registry_path}")
        
        if e.suggestions:
            print(f"  Suggestions:")
            for suggestion in e.suggestions:
                print(f"    - {suggestion}")
        
        return False
    
    except Exception as e:
        print(f"Unexpected error loading workspace '{workspace_id}': {e}")
        return False

# Test workspace loading
test_workspaces = [
    ("developer_1", "/valid/workspace/path"),
    ("invalid_workspace", "/nonexistent/path"),
    ("malformed_workspace", "/path/with/bad/registry")
]

for workspace_id, path in test_workspaces:
    success = safe_load_workspace_registry(workspace_id, path)
    print(f"Workspace '{workspace_id}': {'Loaded' if success else 'Failed'}")
```

### Advanced Error Context Analysis

```python
from cursus.registry.exceptions import RegistryError, RegistryLoadError

def analyze_registry_error(error):
    """Analyze registry error and provide detailed diagnostics."""
    print(f"Registry Error Analysis:")
    print(f"=" * 30)
    print(f"Error Type: {type(error).__name__}")
    print(f"Message: {error}")
    
    if isinstance(error, RegistryError):
        print(f"\nRegistry Error Details:")
        print(f"  Unresolvable types: {len(error.unresolvable_types or [])}")
        print(f"  Available builders: {len(error.available_builders or [])}")
        
        # Analyze unresolvable types
        if error.unresolvable_types:
            print(f"\n  Unresolvable Types Analysis:")
            for step_type in error.unresolvable_types:
                print(f"    - {step_type}")
                # Check for common naming issues
                if step_type.endswith('Config'):
                    print(f"      → Suggestion: Remove 'Config' suffix")
                if '_' in step_type:
                    print(f"      → Suggestion: Try PascalCase format")
        
        # Analyze available alternatives
        if error.available_builders:
            print(f"\n  Available Alternatives:")
            for builder in error.available_builders[:5]:
                print(f"    - {builder}")
            if len(error.available_builders) > 5:
                print(f"    ... and {len(error.available_builders) - 5} more")
    
    elif isinstance(error, RegistryLoadError):
        print(f"\nRegistry Load Error Details:")
        print(f"  Registry path: {error.registry_path}")
        print(f"  Suggestions: {len(error.suggestions or [])}")
        
        if error.suggestions:
            print(f"\n  Resolution Suggestions:")
            for i, suggestion in enumerate(error.suggestions, 1):
                print(f"    {i}. {suggestion}")

# Example error analysis
try:
    registry = get_global_registry()
    builder = registry.get_builder_for_step_type("NonExistentStep")
except RegistryError as e:
    analyze_registry_error(e)

try:
    manager = UnifiedRegistryManager()
    manager.add_workspace_registry("bad_workspace", "/invalid/path")
except RegistryLoadError as e:
    analyze_registry_error(e)
```

### Error Recovery Strategies

```python
from cursus.registry.exceptions import RegistryError, RegistryLoadError

class RegistryErrorRecovery:
    """Helper class for implementing registry error recovery strategies."""
    
    def __init__(self, registry):
        self.registry = registry
        self.fallback_builders = {}
    
    def get_builder_with_fallback(self, step_type, fallback_type=None):
        """Get builder with automatic fallback strategy."""
        try:
            return self.registry.get_builder_for_step_type(step_type)
        except RegistryError as e:
            print(f"Primary resolution failed for '{step_type}': {e}")
            
            # Strategy 1: Try fallback type if provided
            if fallback_type:
                try:
                    print(f"Trying fallback type: {fallback_type}")
                    return self.registry.get_builder_for_step_type(fallback_type)
                except RegistryError:
                    print(f"Fallback type '{fallback_type}' also failed")
            
            # Strategy 2: Try similar names
            if e.available_builders:
                similar = self._find_similar_builders(step_type, e.available_builders)
                if similar:
                    print(f"Trying similar builder: {similar[0]}")
                    try:
                        return self.registry.get_builder_for_step_type(similar[0])
                    except RegistryError:
                        print(f"Similar builder '{similar[0]}' failed")
            
            # Strategy 3: Use registered fallback
            if step_type in self.fallback_builders:
                fallback = self.fallback_builders[step_type]
                print(f"Using registered fallback: {fallback}")
                return self.registry.get_builder_for_step_type(fallback)
            
            # Strategy 4: Use generic processing builder
            try:
                print("Trying generic Processing builder")
                return self.registry.get_builder_for_step_type("Processing")
            except RegistryError:
                print("Generic Processing builder not available")
            
            # All strategies failed
            raise RegistryError(
                f"All recovery strategies failed for step type: {step_type}",
                unresolvable_types=[step_type],
                available_builders=e.available_builders
            )
    
    def _find_similar_builders(self, target, available):
        """Find builders with similar names."""
        target_lower = target.lower()
        similar = []
        
        for builder in available:
            builder_lower = builder.lower()
            # Check for partial matches
            if target_lower in builder_lower or builder_lower in target_lower:
                similar.append(builder)
        
        return sorted(similar, key=lambda x: len(x))  # Prefer shorter names
    
    def register_fallback(self, step_type, fallback_type):
        """Register a fallback builder for a step type."""
        self.fallback_builders[step_type] = fallback_type
        print(f"Registered fallback: {step_type} → {fallback_type}")

# Example usage
recovery = RegistryErrorRecovery(get_global_registry())

# Register fallbacks
recovery.register_fallback("CustomStep", "Processing")
recovery.register_fallback("UnknownStep", "XGBoostTraining")

# Test recovery strategies
test_steps = [
    ("XGBoostTraining", None),           # Should work normally
    ("CustomStep", "Processing"),        # Should use fallback
    ("NonExistent", "XGBoostTraining"),  # Should use provided fallback
    ("CompletelyUnknown", None)          # Should try all strategies
]

for step_type, fallback in test_steps:
    try:
        builder = recovery.get_builder_with_fallback(step_type, fallback)
        print(f"✓ {step_type}: {builder.__name__}")
    except RegistryError as e:
        print(f"✗ {step_type}: All recovery strategies failed")
        print(f"  Final error: {e}")
    print("-" * 40)
```

## Error Prevention Best Practices

### Validation Before Use
```python
from cursus.registry import validate_step_name, get_all_step_names
from cursus.registry.exceptions import RegistryError

def validate_step_types_before_use(step_types):
    """Validate step types before attempting to use them."""
    valid_types = []
    invalid_types = []
    
    all_available = get_all_step_names()
    
    for step_type in step_types:
        if validate_step_name(step_type):
            valid_types.append(step_type)
        else:
            invalid_types.append(step_type)
    
    if invalid_types:
        print(f"Invalid step types found: {invalid_types}")
        print(f"Available types: {all_available[:10]}...")  # Show first 10
        
        # Suggest corrections
        for invalid_type in invalid_types:
            suggestions = [t for t in all_available if invalid_type.lower() in t.lower()]
            if suggestions:
                print(f"  Suggestions for '{invalid_type}': {suggestions[:3]}")
    
    return valid_types, invalid_types

# Example validation
pipeline_steps = [
    "XGBoostTraining",
    "InvalidStep", 
    "CradleDataLoading",
    "NonExistentBuilder"
]

valid, invalid = validate_step_types_before_use(pipeline_steps)
print(f"Valid steps: {valid}")
print(f"Invalid steps: {invalid}")
```

## Related Components

- **[Registry Module](__init__.md)** - Main registry module that uses these exceptions
- **[Builder Registry](builder_registry.md)** - Builder registry that raises these exceptions
- **[Step Names](step_names.md)** - Step names registry that may trigger exceptions
- **[Hybrid Manager](hybrid/manager.md)** - Unified registry manager that handles exceptions
