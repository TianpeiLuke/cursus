---
tags:
  - code
  - workspace
  - api_reference
  - documentation
  - implementation
keywords:
  - workspace API
  - developer workspaces
  - workspace management
  - component discovery
  - workspace validation
  - multi-developer collaboration
topics:
  - workspace API reference
  - workspace management
  - developer collaboration
  - component discovery
language: python
date of note: 2025-09-02
---

# Workspace API Reference

## Overview

The Workspace API provides a unified interface for managing developer workspaces, enabling isolated development environments and cross-workspace collaboration. This reference documents the actual implemented functionality with practical examples.

## Core API Classes

### WorkspaceAPI

The main entry point for all workspace operations.

```python
from cursus.workspace import WorkspaceAPI

# Initialize with default workspace root
api = WorkspaceAPI()

# Initialize with custom workspace root
api = WorkspaceAPI(base_path="custom/workspace/path")
```

## Core Operations

### setup_developer_workspace()

Creates a new developer workspace with optional template.

**Signature:**
```python
def setup_developer_workspace(
    self,
    developer_id: str,
    template: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> WorkspaceSetupResult
```

**Parameters:**
- `developer_id`: Unique identifier for the developer
- `template`: Optional template name (e.g., "ml_pipeline", "basic", "data_processing")
- `config_overrides`: Optional configuration overrides

**Returns:** `WorkspaceSetupResult` with success status and details

**Example:**
```python
api = WorkspaceAPI()
result = api.setup_developer_workspace(
    developer_id="john_doe",
    template="ml_pipeline"
)

if result.success:
    print(f"Workspace created at: {result.workspace_path}")
    print(f"Developer ID: {result.developer_id}")
else:
    print(f"Setup failed: {result.message}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
```

### validate_workspace()

Validates workspace isolation and configuration.

**Signature:**
```python
def validate_workspace(
    self,
    workspace_path: Union[str, Path]
) -> ValidationReport
```

**Parameters:**
- `workspace_path`: Path to the workspace to validate

**Returns:** `ValidationReport` with validation status and details

**Example:**
```python
report = api.validate_workspace("development/projects/john_doe")

if report.status == WorkspaceStatus.HEALTHY:
    print("✅ Workspace is ready for development")
    print(f"Components found: {len(report.components)}")
else:
    print(f"❌ Validation failed: {report.status}")
    for violation in report.violations:
        print(f"  - {violation}")
```

### discover_workspace_components()

Discovers all components across developer workspaces.

**Signature:**
```python
def discover_workspace_components(
    self,
    developer_id: Optional[str] = None,
    component_type: Optional[str] = None
) -> Dict[str, Any]
```

**Parameters:**
- `developer_id`: Optional filter by specific developer
- `component_type`: Optional filter by component type ("builders", "configs", etc.)

**Returns:** Dictionary of discovered components

**Example:**
```python
# Discover all components
all_components = api.discover_workspace_components()

print("Available workspace components:")
for dev_id, components in all_components.items():
    print(f"\n{dev_id}:")
    for comp_type, files in components.items():
        print(f"  {comp_type}: {len(files)} files")

# Discover components for specific developer
john_components = api.discover_workspace_components(developer_id="john_doe")

# Discover only builders
builders = api.discover_workspace_components(component_type="builders")
```

### promote_workspace_component()

Promotes a workspace component to shared core or integration staging.

**Signature:**
```python
def promote_workspace_component(
    self,
    developer_id: str,
    component_path: str,
    target: str = "staging"
) -> PromotionResult
```

**Parameters:**
- `developer_id`: Developer who owns the component
- `component_path`: Relative path to component within workspace
- `target`: Promotion target ("staging" or "core")

**Returns:** `PromotionResult` with promotion status

**Example:**
```python
result = api.promote_workspace_component(
    developer_id="john_doe",
    component_path="src/cursus_dev/steps/builders/builder_custom_step.py",
    target="staging"
)

if result.success:
    print(f"Component promoted to: {result.target_path}")
else:
    print(f"Promotion failed: {result.message}")
```

### list_workspaces()

Lists all available workspaces.

**Signature:**
```python
def list_workspaces(self) -> List[WorkspaceInfo]
```

**Parameters:**
- None

**Returns:** List of `WorkspaceInfo` objects

**Example:**
```python
workspaces = api.list_workspaces()

for info in workspaces:
    print(f"Developer: {info.developer_id}")
    print(f"Path: {info.path}")
    print(f"Status: {info.status}")
    print(f"Created: {info.created_at or 'Unknown'}")
    print(f"Size: {info.size_bytes or 0} bytes")
    print("---")

# Find specific workspace
john_workspace = next((w for w in workspaces if w.developer_id == "john_doe"), None)
if john_workspace:
    print(f"Found John's workspace at: {john_workspace.path}")
else:
    print("John's workspace not found")
```

### cleanup_workspace()

Cleans up workspace resources and temporary files.

**Signature:**
```python
def cleanup_workspace(
    self,
    developer_id: str,
    deep_clean: bool = False
) -> CleanupResult
```

**Parameters:**
- `developer_id`: Developer identifier
- `deep_clean`: Whether to perform deep cleanup (removes all generated files)

**Returns:** `CleanupResult` with cleanup status

**Example:**
```python
result = api.cleanup_workspace("john_doe", deep_clean=True)

if result.success:
    print(f"Cleaned up {result.files_removed} files")
    print(f"Freed {result.space_freed} bytes")
else:
    print(f"Cleanup failed: {result.message}")
```

## Data Models

### WorkspaceSetupResult

Result of workspace setup operation.

```python
class WorkspaceSetupResult(BaseModel):
    success: bool
    workspace_path: Path
    developer_id: str = Field(..., min_length=1)
    message: str
    warnings: List[str] = Field(default_factory=list)
    template_used: Optional[str] = None
    created_components: List[str] = Field(default_factory=list)
```

### ValidationReport

Result of workspace validation.

```python
class ValidationReport(BaseModel):
    status: WorkspaceStatus
    workspace_path: Path
    violations: List[str] = Field(default_factory=list)
    components: Dict[str, int] = Field(default_factory=dict)
    validation_time: datetime = Field(default_factory=datetime.now)
    details: Dict[str, Any] = Field(default_factory=dict)
```

### WorkspaceInfo

Detailed workspace information.

```python
class WorkspaceInfo(BaseModel):
    developer_id: str
    workspace_path: Path
    status: WorkspaceStatus
    created_at: datetime
    last_modified: datetime
    component_count: int
    is_valid: bool
    template_used: Optional[str] = None
    size_bytes: int
```

## Error Handling

### Common Exceptions

**WorkspaceNotFoundError**
```python
try:
    info = api.get_workspace_info("nonexistent_dev")
except WorkspaceNotFoundError as e:
    print(f"Workspace not found: {e}")
```

**WorkspaceValidationError**
```python
try:
    result = api.setup_developer_workspace("invalid-id!")
except WorkspaceValidationError as e:
    print(f"Validation failed: {e}")
```

**ComponentNotFoundError**
```python
try:
    result = api.promote_workspace_component(
        developer_id="john_doe",
        component_path="nonexistent/component.py"
    )
except ComponentNotFoundError as e:
    print(f"Component not found: {e}")
```

### Error Handling Best Practices

```python
from cursus.workspace import WorkspaceAPI, WorkspaceError

api = WorkspaceAPI()

try:
    # Workspace operations
    result = api.setup_developer_workspace("new_developer")
    
    if result.success:
        # Validate the workspace
        report = api.validate_workspace(result.workspace_path)
        
        if report.status == WorkspaceStatus.HEALTHY:
            print("✅ Workspace ready for development")
        else:
            print("⚠️ Workspace has issues:")
            for violation in report.violations:
                print(f"  - {violation}")
    else:
        print(f"❌ Setup failed: {result.message}")
        
except WorkspaceError as e:
    print(f"Workspace operation failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Usage

### Batch Operations

```python
# Set up multiple workspaces
developers = ["alice", "bob", "charlie"]
results = []

for dev_id in developers:
    try:
        result = api.setup_developer_workspace(
            developer_id=dev_id,
            template="standard_ml_pipeline"
        )
        results.append((dev_id, result))
    except Exception as e:
        print(f"Failed to set up workspace for {dev_id}: {e}")

# Report results
successful = [r for r in results if r[1].success]
print(f"Successfully created {len(successful)} workspaces")
```

### Component Discovery and Analysis

```python
# Analyze workspace components across all developers
components = api.discover_workspace_components()

# Count components by type
component_stats = {}
for dev_id, dev_components in components.items():
    for comp_type, files in dev_components.items():
        if comp_type not in component_stats:
            component_stats[comp_type] = 0
        component_stats[comp_type] += len(files)

print("Component Statistics:")
for comp_type, count in component_stats.items():
    print(f"  {comp_type}: {count} files")

# Find developers with specific components
xgboost_developers = []
for dev_id, dev_components in components.items():
    builders = dev_components.get('builders', {})
    for builder_name in builders.keys():
        if 'xgboost' in builder_name.lower():
            xgboost_developers.append(dev_id)
            break

print(f"Developers with XGBoost components: {xgboost_developers}")
```

### Workspace Health Monitoring

```python
# Monitor workspace health across all developers
def monitor_workspace_health():
    workspaces = api.list_workspaces()
    health_report = {}
    
    for info in workspaces:
        try:
            report = api.validate_workspace(info.path)
            
            health_report[info.developer_id] = {
                'status': report.status,
                'component_count': len(report.issues) if hasattr(report, 'issues') else 0,
                'violations': len(report.issues) if hasattr(report, 'issues') else 0,
                'last_modified': info.last_modified or 'Unknown'
            }
        except Exception as e:
            health_report[info.developer_id] = {'error': str(e)}
    
    return health_report

# Run health check
health = monitor_workspace_health()
for dev_id, status in health.items():
    if 'error' in status:
        print(f"❌ {dev_id}: {status['error']}")
    elif status['status'] == WorkspaceStatus.HEALTHY:
        print(f"✅ {dev_id}: healthy workspace")
    else:
        print(f"⚠️ {dev_id}: {status['violations']} violations")
```

## Integration Examples

### With Pipeline Assembly

```python
from cursus.workspace import WorkspaceAPI
from cursus.workspace.core import WorkspacePipelineAssembler
from cursus.core.dag import PipelineDAG

# Set up workspace API
api = WorkspaceAPI()

# Create a pipeline using workspace components
dag = PipelineDAG()
dag.add_node("data_loading")
dag.add_node("preprocessing") 
dag.add_node("training")
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "training")

# Define workspace step configurations
workspace_config = {
    "data_loading": WorkspaceStepDefinition(
        developer_id="data_team_alice",
        step_name="data_loading",
        step_type="CradleDataLoading",
        config_data={"dataset": "customer_data", "format": "parquet"}
    ),
    "preprocessing": WorkspaceStepDefinition(
        developer_id="ml_team_bob",
        step_name="preprocessing", 
        step_type="TabularPreprocessing",
        config_data={"features": ["age", "income"], "scaling": "standard"}
    ),
    "training": WorkspaceStepDefinition(
        developer_id="ml_team_charlie",
        step_name="training",
        step_type="XGBoostTraining", 
        config_data={"max_depth": 6, "learning_rate": 0.1}
    )
}

# Create workspace pipeline assembler
assembler = WorkspacePipelineAssembler(
    dag=dag,
    workspace_config_map=workspace_config,
    workspace_root="development/projects"
)

# Validate components before assembly
validation = assembler.validate_workspace_components()
if validation['overall_valid']:
    # Generate pipeline
    pipeline = assembler.generate_pipeline("MultiDeveloperPipeline")
    print(f"✅ Pipeline created: {pipeline.name}")
else:
    print("❌ Component validation failed")
    for issue in validation['component_issues']:
        print(f"  - {issue}")
```

## Troubleshooting

### Common Issues

**Issue: Workspace validation fails with "Invalid workspace structure"**
```python
# Check workspace structure
workspaces = api.list_workspaces()
info = next((w for w in workspaces if w.developer_id == "developer_id"), None)

if info:
    print(f"Workspace path: {info.path}")
    
    # Validate specific paths
    required_paths = [
        "src/cursus_dev",
        "src/cursus_dev/steps",
        "src/cursus_dev/steps/builders",
        "src/cursus_dev/steps/configs"
    ]
    
    for path in required_paths:
        full_path = info.path / path
        if not full_path.exists():
            print(f"❌ Missing: {full_path}")
        else:
            print(f"✅ Found: {full_path}")
```

**Issue: Component discovery returns empty results**
```python
# Debug component discovery
components = api.discover_workspace_components(developer_id="problem_dev")

if not components:
    print("No components found. Checking workspace...")
    workspaces = api.list_workspaces()
    info = next((w for w in workspaces if w.developer_id == "problem_dev"), None)
    
    if not info:
        print("❌ Workspace doesn't exist")
    elif info.status != WorkspaceStatus.HEALTHY:
        print("❌ Workspace is invalid")
        # Try to validate and get specific errors
        report = api.validate_workspace(info.path)
        for violation in report.issues:
            print(f"  - {violation}")
    else:
        print("✅ Workspace exists and is valid")
        print("Check if component files follow naming conventions:")
        print("  - Builders: builder_<type>_step.py")
        print("  - Configs: config_<type>_step.py")
```

**Issue: Component promotion fails**
```python
# Debug component promotion
try:
    result = api.promote_workspace_component(
        developer_id="dev_id",
        component_path="problematic/component.py"
    )
except Exception as e:
    print(f"Promotion failed: {e}")
    
    # Check if component exists
    workspaces = api.list_workspaces()
    info = next((w for w in workspaces if w.developer_id == "dev_id"), None)
    if info:
        component_full_path = info.path / "problematic/component.py"
        if component_full_path.exists():
            print("✅ Component file exists")
            # Check file permissions
            if component_full_path.is_file():
                print("✅ Is a file")
            else:
                print("❌ Not a regular file")
        else:
            print("❌ Component file not found")
```

### Performance Tips

1. **Cache component discovery results** when possible
2. **Use specific developer_id filters** to reduce discovery scope
3. **Batch workspace operations** instead of individual calls
4. **Clean up workspaces regularly** to maintain performance

### Best Practices

1. **Always validate workspaces** after setup
2. **Handle exceptions gracefully** in production code
3. **Use meaningful developer IDs** for easier management
4. **Monitor workspace health** regularly
5. **Follow component naming conventions** for reliable discovery

## API Reference Summary

| Method | Purpose | Returns |
|--------|---------|---------|
| `setup_developer_workspace()` | Create new workspace | `WorkspaceSetupResult` |
| `validate_workspace()` | Validate workspace | `ValidationReport` |
| `discover_workspace_components()` | Find components | `Dict[str, Any]` |
| `promote_workspace_component()` | Promote component | `PromotionResult` |
| `list_workspaces()` | List all workspaces | `List[WorkspaceInfo]` |
| `cleanup_workspace()` | Clean workspace | `CleanupResult` |

For additional examples and advanced usage patterns, see the [Workspace Quick Start Guide](workspace_quick_start.md).
