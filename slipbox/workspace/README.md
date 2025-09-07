---
tags:
  - entry_point
  - code
  - workspace
  - management
  - isolation
  - collaboration
keywords:
  - WorkspaceAPI
  - WorkspaceManager
  - workspace isolation
  - cross-workspace dependencies
  - developer workspaces
  - workspace templates
  - workspace validation
topics:
  - workspace management
  - developer isolation
  - collaborative development
  - workspace templates
language: python
date of note: 2024-12-07
---

# Workspace Management

Comprehensive workspace management system providing isolated development environments, cross-workspace collaboration, and unified workspace operations for the Cursus pipeline system.

## Overview

The Workspace Management module provides a complete solution for managing isolated developer workspaces, enabling collaborative pipeline development while maintaining strict isolation boundaries. The system supports workspace lifecycle management, template-based workspace creation, cross-workspace dependency resolution, and comprehensive validation and monitoring capabilities.

The module implements a hierarchical architecture with a high-level unified API, specialized core managers for different functional areas, and comprehensive validation systems. It supports multiple workspace types including developer workspaces, shared workspaces, and test environments, with full integration into the Cursus pipeline ecosystem.

## Classes and Methods

### Core API Classes
- [`WorkspaceAPI`](#workspaceapi) - Unified high-level API for workspace operations
- [`WorkspaceSetupResult`](#workspacesetupresult) - Result of workspace setup operations
- [`ValidationReport`](#validationreport) - Workspace validation report with issues and recommendations
- [`PromotionResult`](#promotionresult) - Result of workspace artifact promotion
- [`HealthReport`](#healthreport) - Overall workspace system health report
- [`CleanupReport`](#cleanupreport) - Result of workspace cleanup operations
- [`WorkspaceInfo`](#workspaceinfo) - Information about a workspace

### Template System Classes
- [`WorkspaceTemplate`](#workspacetemplate) - Workspace template configuration
- [`TemplateManager`](#templatemanager) - Manages workspace templates
- [`TemplateType`](#templatetype) - Enumeration of available template types

### Core Management Classes
- [`WorkspaceManager`](#workspacemanager) - Centralized workspace management with functional separation
- [`WorkspaceContext`](#workspacecontext) - Context information for a workspace
- [`WorkspaceConfig`](#workspaceconfig) - Configuration for workspace management

### Utility Functions
- [`get_default_config`](#get_default_config) - Get default configuration for workspace API

## API Reference

### WorkspaceAPI

_class_ cursus.workspace.api.WorkspaceAPI(_base_path=None_)

Unified high-level API for workspace-aware system operations, providing a simplified interface to the workspace system and abstracting the complexity of underlying managers.

**Parameters:**
- **base_path** (_Optional[Union[str, Path]]_) – Base path for workspace operations (defaults to "development")

```python
from cursus.workspace import WorkspaceAPI

# Initialize the unified API
api = WorkspaceAPI()

# Developer operations
result = api.setup_developer_workspace("developer_1", "ml_template")
pipeline = api.build_cross_workspace_pipeline(pipeline_spec)
report = api.validate_workspace_components("developer_1")
```

#### setup_developer_workspace

setup_developer_workspace(_developer_id_, _template=None_, _config_overrides=None_)

Set up a new developer workspace with optional template and configuration overrides.

**Parameters:**
- **developer_id** (_str_) – Unique identifier for the developer
- **template** (_Optional[str]_) – Optional template to use for workspace setup
- **config_overrides** (_Optional[Dict[str, Any]]_) – Optional configuration overrides

**Returns:**
- **WorkspaceSetupResult** – Setup result with workspace details and any warnings

```python
# Create basic developer workspace
result = api.setup_developer_workspace("alice")

# Create ML workspace with template
result = api.setup_developer_workspace(
    "bob", 
    template="ml_pipeline",
    config_overrides={"enable_gpu": True}
)
```

#### validate_workspace

validate_workspace(_workspace_path_)

Validate a workspace for compliance and isolation violations.

**Parameters:**
- **workspace_path** (_Union[str, Path]_) – Path to the workspace to validate

**Returns:**
- **ValidationReport** – Validation results with status, issues, and recommendations

```python
# Validate workspace
report = api.validate_workspace("/path/to/workspace")
if report.status == WorkspaceStatus.ERROR:
    print(f"Validation errors: {report.issues}")
```

#### list_workspaces

list_workspaces()

List all available workspaces with their current status and information.

**Returns:**
- **List[WorkspaceInfo]** – List of workspace information objects

```python
workspaces = api.list_workspaces()
for workspace in workspaces:
    print(f"{workspace.developer_id}: {workspace.status}")
```

#### promote_workspace_artifacts

promote_workspace_artifacts(_workspace_path_, _target_environment="staging"_)

Promote artifacts from a workspace to target environment.

**Parameters:**
- **workspace_path** (_Union[str, Path]_) – Path to the source workspace
- **target_environment** (_str_) – Target environment (staging, production, etc.)

**Returns:**
- **PromotionResult** – Promotion results with promoted artifacts list

```python
# Promote to staging
result = api.promote_workspace_artifacts("/path/to/workspace", "staging")
print(f"Promoted {len(result.artifacts_promoted)} artifacts")
```

#### get_system_health

get_system_health()

Get overall system health report including all workspace validation results.

**Returns:**
- **HealthReport** – System-wide health information with workspace reports

```python
health = api.get_system_health()
print(f"Overall status: {health.overall_status}")
for report in health.workspace_reports:
    print(f"Workspace {report.workspace_path}: {report.status}")
```

#### cleanup_workspaces

cleanup_workspaces(_inactive_days=30_, _dry_run=True_)

Clean up inactive workspaces based on inactivity threshold.

**Parameters:**
- **inactive_days** (_int_) – Number of days of inactivity before cleanup
- **dry_run** (_bool_) – If True, only report what would be cleaned

**Returns:**
- **CleanupReport** – Cleanup results with cleaned workspaces and errors

```python
# Dry run cleanup
report = api.cleanup_workspaces(inactive_days=60, dry_run=True)
print(f"Would clean {len(report.cleaned_workspaces)} workspaces")

# Actual cleanup
report = api.cleanup_workspaces(inactive_days=60, dry_run=False)
```

### WorkspaceSetupResult

_class_ cursus.workspace.api.WorkspaceSetupResult(_success_, _workspace_path_, _developer_id_, _message_, _warnings=[]_)

Result of workspace setup operation with success status and details.

**Parameters:**
- **success** (_bool_) – Whether setup was successful
- **workspace_path** (_Path_) – Path to the created workspace
- **developer_id** (_str_) – Unique identifier for the developer
- **message** (_str_) – Setup result message
- **warnings** (_List[str]_) – List of warnings encountered during setup

```python
result = api.setup_developer_workspace("developer_1")
if result.success:
    print(f"Workspace created at: {result.workspace_path}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
```

### ValidationReport

_class_ cursus.workspace.api.ValidationReport(_workspace_path_, _status_, _issues=[]_, _recommendations=[]_, _isolation_violations=[]_)

Workspace validation report with status, issues, and recommendations.

**Parameters:**
- **workspace_path** (_Path_) – Path to the validated workspace
- **status** (_WorkspaceStatus_) – Validation status (HEALTHY, WARNING, ERROR, UNKNOWN)
- **issues** (_List[str]_) – List of validation issues found
- **recommendations** (_List[str]_) – List of recommendations for fixing issues
- **isolation_violations** (_List[Dict[str, Any]]_) – Detailed isolation violation information

```python
report = api.validate_workspace("workspace_path")
if report.status == WorkspaceStatus.WARNING:
    for issue in report.issues:
        print(f"Issue: {issue}")
    for rec in report.recommendations:
        print(f"Recommendation: {rec}")
```

### WorkspaceTemplate

_class_ cursus.workspace.templates.WorkspaceTemplate(_name_, _type_, _description=""_, _directories=[]_, _files={}_, _dependencies=[]_, _config_overrides={}_, _created_at=None_, _version="1.0.0"_)

Workspace template configuration defining structure, files, and dependencies.

**Parameters:**
- **name** (_str_) – Template name
- **type** (_TemplateType_) – Type of template (BASIC, ML_PIPELINE, DATA_PROCESSING, CUSTOM)
- **description** (_str_) – Template description
- **directories** (_List[str]_) – Directories to create
- **files** (_Dict[str, str]_) – Files to create with content
- **dependencies** (_List[str]_) – Required dependencies
- **config_overrides** (_Dict[str, Any]_) – Default configuration
- **created_at** (_Optional[str]_) – Template creation timestamp
- **version** (_str_) – Template version

```python
from cursus.workspace.templates import WorkspaceTemplate, TemplateType

# Create custom template
template = WorkspaceTemplate(
    name="custom_ml",
    type=TemplateType.ML_PIPELINE,
    description="Custom ML pipeline template",
    directories=["data", "models", "notebooks"],
    files={"README.md": "# Custom ML Workspace"},
    dependencies=["pandas", "scikit-learn"]
)
```

### TemplateManager

_class_ cursus.workspace.templates.TemplateManager(_templates_dir=None_)

Manages workspace templates including built-in and custom templates.

**Parameters:**
- **templates_dir** (_Optional[Path]_) – Directory containing template definitions

```python
from cursus.workspace.templates import TemplateManager

# Initialize template manager
manager = TemplateManager()

# List available templates
templates = manager.list_templates()
for template in templates:
    print(f"{template.name}: {template.description}")
```

#### get_template

get_template(_name_)

Get a template by name from the template directory.

**Parameters:**
- **name** (_str_) – Template name

**Returns:**
- **Optional[WorkspaceTemplate]** – Template if found, None otherwise

```python
template = manager.get_template("ml_pipeline")
if template:
    print(f"Template: {template.description}")
```

#### list_templates

list_templates()

List all available templates in the template directory.

**Returns:**
- **List[WorkspaceTemplate]** – List of available templates

```python
templates = manager.list_templates()
print(f"Found {len(templates)} templates")
```

#### create_template

create_template(_template_)

Create a new template and save it to the template directory.

**Parameters:**
- **template** (_WorkspaceTemplate_) – Template to create

**Returns:**
- **bool** – True if successful, False otherwise

```python
success = manager.create_template(custom_template)
if success:
    print("Template created successfully")
```

#### apply_template

apply_template(_template_name_, _workspace_path_)

Apply a template to a workspace directory.

**Parameters:**
- **template_name** (_str_) – Name of template to apply
- **workspace_path** (_Path_) – Path to workspace directory

**Returns:**
- **bool** – True if successful, False otherwise

```python
success = manager.apply_template("ml_pipeline", Path("workspace"))
if success:
    print("Template applied successfully")
```

### WorkspaceManager

_class_ cursus.workspace.core.manager.WorkspaceManager(_workspace_root=None_, _config_file=None_, _auto_discover=True_)

Centralized workspace management with functional separation through specialized managers.

**Parameters:**
- **workspace_root** (_Optional[Union[str, Path]]_) – Root directory for workspaces
- **config_file** (_Optional[Union[str, Path]]_) – Path to workspace configuration file
- **auto_discover** (_bool_) – Whether to automatically discover workspaces

```python
from cursus.workspace.core.manager import WorkspaceManager

# Initialize workspace manager
manager = WorkspaceManager("/path/to/workspaces")

# Create workspace
context = manager.create_workspace("developer_1", template="basic")
```

#### create_workspace

create_workspace(_developer_id_, _workspace_type="developer"_, _template=None_, _**kwargs_)

Create a new workspace with specified type and optional template.

**Parameters:**
- **developer_id** (_str_) – Developer identifier for the workspace
- **workspace_type** (_str_) – Type of workspace ("developer", "shared", "test")
- **template** (_str_) – Optional template to use for workspace creation
- **kwargs** – Additional arguments passed to lifecycle manager

**Returns:**
- **WorkspaceContext** – Context for the created workspace

```python
# Create developer workspace
context = manager.create_workspace("alice", "developer", "ml_pipeline")
print(f"Created workspace: {context.workspace_id}")
```

#### discover_components

discover_components(_workspace_ids=None_, _developer_id=None_)

Discover components across workspaces with optional filtering.

**Parameters:**
- **workspace_ids** (_Optional[List[str]]_) – Optional list of workspace IDs to search
- **developer_id** (_Optional[str]_) – Optional specific developer ID to search

**Returns:**
- **Dict[str, Any]** – Dictionary containing discovered components

```python
# Discover all components
components = manager.discover_components()

# Discover for specific developer
components = manager.discover_components(developer_id="alice")
```

#### validate_workspace_structure

validate_workspace_structure(_workspace_root=None_, _strict=False_)

Validate workspace structure for compliance and best practices.

**Parameters:**
- **workspace_root** (_Optional[Union[str, Path]]_) – Root directory to validate
- **strict** (_bool_) – Whether to apply strict validation rules

**Returns:**
- **Tuple[bool, List[str]]** – Tuple of (is_valid, list_of_issues)

```python
is_valid, issues = manager.validate_workspace_structure(strict=True)
if not is_valid:
    for issue in issues:
        print(f"Issue: {issue}")
```

### get_default_config

get_default_config()

Get default configuration for workspace API with standard settings.

**Returns:**
- **Dict[str, Any]** – Default configuration dictionary

```python
from cursus.workspace import get_default_config

config = get_default_config()
print(f"Default workspace root: {config['workspace_root']}")
```

## Usage Examples

### Basic Workspace Setup

```python
from cursus.workspace import WorkspaceAPI

# Initialize API
api = WorkspaceAPI()

# Set up developer workspace
result = api.setup_developer_workspace("alice", template="basic")
if result.success:
    print(f"Workspace created at: {result.workspace_path}")
    
    # Validate the new workspace
    report = api.validate_workspace(result.workspace_path)
    print(f"Validation status: {report.status}")
```

### ML Pipeline Workspace

```python
# Create ML pipeline workspace
result = api.setup_developer_workspace(
    "data_scientist_1", 
    template="ml_pipeline",
    config_overrides={
        "enable_gpu": True,
        "data_path": "/shared/datasets"
    }
)

# List all workspaces
workspaces = api.list_workspaces()
ml_workspaces = [w for w in workspaces if "ml" in w.path.name]
print(f"Found {len(ml_workspaces)} ML workspaces")
```

### Template Management

```python
from cursus.workspace.templates import TemplateManager, WorkspaceTemplate, TemplateType

# Initialize template manager
manager = TemplateManager()

# Create custom template
custom_template = WorkspaceTemplate(
    name="data_science",
    type=TemplateType.ML_PIPELINE,
    description="Data science workspace with Jupyter and visualization tools",
    directories=["data", "notebooks", "models", "reports"],
    files={
        "README.md": "# Data Science Workspace",
        "requirements.txt": "pandas\nnumpy\njupyter\nmatplotlib\nseaborn"
    },
    dependencies=["pandas", "numpy", "jupyter", "matplotlib", "seaborn"]
)

# Save template
manager.create_template(custom_template)

# Apply to workspace
manager.apply_template("data_science", Path("workspace"))
```

### Cross-Workspace Operations

```python
# Discover components across workspaces
components = api.workspace_manager.discover_components()
print(f"Found components: {list(components.keys())}")

# Resolve cross-workspace dependencies
pipeline_def = {
    "steps": [
        {"name": "data_prep", "workspace": "alice"},
        {"name": "training", "workspace": "bob", "depends_on": ["data_prep"]}
    ]
}

resolved = api.workspace_manager.resolve_cross_workspace_dependencies(pipeline_def)
print(f"Resolved dependencies: {resolved}")
```

### System Health Monitoring

```python
# Get system health
health = api.get_system_health()
print(f"Overall status: {health.overall_status}")

# Check individual workspace health
for report in health.workspace_reports:
    if report.status != WorkspaceStatus.HEALTHY:
        print(f"Workspace {report.workspace_path} has issues:")
        for issue in report.issues:
            print(f"  - {issue}")
        for rec in report.recommendations:
            print(f"  Recommendation: {rec}")
```

### Workspace Cleanup

```python
# Dry run cleanup to see what would be cleaned
cleanup_report = api.cleanup_workspaces(inactive_days=30, dry_run=True)
print(f"Would clean {len(cleanup_report.cleaned_workspaces)} workspaces")

# Actual cleanup
if len(cleanup_report.cleaned_workspaces) > 0:
    cleanup_report = api.cleanup_workspaces(inactive_days=30, dry_run=False)
    print(f"Cleaned {len(cleanup_report.cleaned_workspaces)} workspaces")
```

## Integration Points

### Pipeline System Integration
The workspace system integrates with the Cursus pipeline system for component discovery, dependency resolution, and pipeline execution across workspace boundaries.

### Validation Framework Integration
Comprehensive integration with the validation framework for workspace structure validation, isolation checking, and compliance monitoring.

### CLI Integration
Full integration with the Cursus CLI for workspace management commands, validation operations, and administrative tasks.

## Related Documentation

- [Workspace Core](core/README.md) - Core workspace management functionality
- [Workspace Validation](validation/README.md) - Workspace validation and testing systems
- [Workspace Quality](quality/README.md) - Quality monitoring and user experience validation
- [API Documentation](api.md) - High-level workspace API
- [Templates Documentation](templates.md) - Workspace template system
- [CLI Integration](../cli/workspace_cli.md) - Command-line interface for workspace operations
