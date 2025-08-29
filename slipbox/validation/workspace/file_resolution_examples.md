# File Resolution Examples

This document provides examples of using the workspace-aware file resolution system.

## Basic File Resolution

### Using DeveloperWorkspaceFileResolver

```python
from cursus.validation.workspace.file_resolver import DeveloperWorkspaceFileResolver

# Initialize resolver for a specific developer workspace
resolver = DeveloperWorkspaceFileResolver(
    workspace_root="/path/to/workspaces",
    developer_id="developer_1"
)

# Resolve different types of files
builder_file = resolver.resolve_builder_file("xgboost_trainer")
# Returns: /path/to/workspaces/developer_1/builders/xgboost_trainer.py

contract_file = resolver.resolve_contract_file("training_contract")
# Returns: /path/to/workspaces/developer_1/contracts/training_contract.py

script_file = resolver.resolve_script_file("train_model")
# Returns: /path/to/workspaces/developer_1/scripts/train_model.py

spec_file = resolver.resolve_spec_file("model_training")
# Returns: /path/to/workspaces/developer_1/specs/model_training.json

config_file = resolver.resolve_config_file("xgboost_config")
# Returns: /path/to/workspaces/developer_1/configs/xgboost_config.json
```

### Switching Between Developers

```python
# Switch to different developer workspace
resolver.switch_developer("developer_2")

# Now resolves files from developer_2's workspace
builder_file = resolver.resolve_builder_file("custom_trainer")
# Returns: /path/to/workspaces/developer_2/builders/custom_trainer.py
```

## Advanced File Resolution

### Custom File Extensions

```python
# Resolve files with specific extensions
yaml_config = resolver.resolve_config_file("model_config", extension=".yaml")
# Returns: /path/to/workspaces/developer_1/configs/model_config.yaml

json_spec = resolver.resolve_spec_file("pipeline_spec", extension=".json")
# Returns: /path/to/workspaces/developer_1/specs/pipeline_spec.json
```

### File Existence Checking

```python
# Check if files exist before using them
if resolver.file_exists("builders", "xgboost_trainer.py"):
    builder_path = resolver.resolve_builder_file("xgboost_trainer")
    print(f"Builder found at: {builder_path}")
else:
    print("Builder not found")

# List all files in a category
builder_files = resolver.list_files("builders")
print(f"Available builders: {builder_files}")
```

### Batch File Resolution

```python
# Resolve multiple files at once
file_names = ["trainer_1", "trainer_2", "trainer_3"]
builder_paths = [
    resolver.resolve_builder_file(name) 
    for name in file_names 
    if resolver.file_exists("builders", f"{name}.py")
]

print(f"Found {len(builder_paths)} builders")
```

## Integration with Validation

### Using Resolver in Validation Context

```python
from cursus.validation.workspace.workspace_alignment_tester import WorkspaceUnifiedAlignmentTester

# Initialize tester with workspace support
tester = WorkspaceUnifiedAlignmentTester(
    workspace_root="/path/to/workspaces"
)

# Switch to specific developer and validate
tester.switch_developer("developer_1")

# The tester uses the resolver internally
validation_result = tester.run_workspace_validation(
    builder_name="xgboost_trainer",
    levels=[1, 2, 3, 4]
)

print(f"Validation passed: {validation_result.passed}")
```

### Custom Resolution Logic

```python
class CustomWorkspaceResolver(DeveloperWorkspaceFileResolver):
    """Custom resolver with additional logic"""
    
    def resolve_template_file(self, template_name: str) -> str:
        """Resolve template files from custom directory"""
        return self._resolve_file("templates", template_name, ".jinja2")
    
    def resolve_data_file(self, data_name: str) -> str:
        """Resolve data files from custom directory"""
        return self._resolve_file("data", data_name, ".csv")

# Use custom resolver
custom_resolver = CustomWorkspaceResolver(
    workspace_root="/path/to/workspaces",
    developer_id="developer_1"
)

template_path = custom_resolver.resolve_template_file("model_template")
data_path = custom_resolver.resolve_data_file("training_data")
```

## Error Handling

### Handling Missing Files

```python
from cursus.validation.workspace.exceptions import WorkspaceFileNotFoundError

try:
    builder_path = resolver.resolve_builder_file("nonexistent_builder")
except WorkspaceFileNotFoundError as e:
    print(f"Builder not found: {e}")
    # Handle gracefully - maybe use default builder
    builder_path = resolver.resolve_builder_file("default_builder")
```

### Validation with Error Recovery

```python
def safe_file_resolution(resolver, file_type, file_name):
    """Safely resolve files with fallback options"""
    try:
        return getattr(resolver, f"resolve_{file_type}_file")(file_name)
    except WorkspaceFileNotFoundError:
        # Try with common suffixes
        for suffix in ["_v1", "_default", "_base"]:
            try:
                return getattr(resolver, f"resolve_{file_type}_file")(f"{file_name}{suffix}")
            except WorkspaceFileNotFoundError:
                continue
        raise WorkspaceFileNotFoundError(f"No {file_type} file found for {file_name}")

# Usage
try:
    builder_path = safe_file_resolution(resolver, "builder", "custom_trainer")
    print(f"Found builder: {builder_path}")
except WorkspaceFileNotFoundError as e:
    print(f"Could not resolve builder: {e}")
```

## Best Practices

### 1. Initialize Once, Use Many Times

```python
# Good: Initialize resolver once
resolver = DeveloperWorkspaceFileResolver(workspace_root, developer_id)

# Use for multiple resolutions
files = {
    'builder': resolver.resolve_builder_file("trainer"),
    'contract': resolver.resolve_contract_file("contract"),
    'script': resolver.resolve_script_file("script"),
    'spec': resolver.resolve_spec_file("spec"),
    'config': resolver.resolve_config_file("config")
}
```

### 2. Validate File Existence

```python
# Always check existence for critical files
critical_files = ["main_builder", "core_contract", "primary_script"]

for file_name in critical_files:
    if not resolver.file_exists("builders", f"{file_name}.py"):
        raise ValueError(f"Critical file missing: {file_name}")
```

### 3. Use Context Managers for Developer Switching

```python
class DeveloperContext:
    """Context manager for temporary developer switching"""
    
    def __init__(self, resolver, developer_id):
        self.resolver = resolver
        self.new_developer = developer_id
        self.original_developer = None
    
    def __enter__(self):
        self.original_developer = self.resolver.developer_id
        self.resolver.switch_developer(self.new_developer)
        return self.resolver
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.resolver.switch_developer(self.original_developer)

# Usage
with DeveloperContext(resolver, "developer_2") as temp_resolver:
    # Temporarily work with developer_2's files
    builder_path = temp_resolver.resolve_builder_file("special_builder")
    # Automatically switches back to original developer
```

### 4. Batch Operations

```python
def resolve_all_files_for_component(resolver, component_name):
    """Resolve all related files for a component"""
    files = {}
    
    file_types = ["builder", "contract", "script", "spec", "config"]
    for file_type in file_types:
        try:
            method = getattr(resolver, f"resolve_{file_type}_file")
            files[file_type] = method(component_name)
        except WorkspaceFileNotFoundError:
            files[file_type] = None
    
    return files

# Usage
component_files = resolve_all_files_for_component(resolver, "xgboost_trainer")
print(f"Found files: {[k for k, v in component_files.items() if v is not None]}")
