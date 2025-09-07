---
tags:
  - code
  - workspace
  - api
  - management
  - high-level
keywords:
  - WorkspaceAPI
  - WorkspaceSetupResult
  - ValidationReport
  - PromotionResult
  - HealthReport
  - CleanupReport
  - WorkspaceInfo
  - WorkspaceStatus
topics:
  - workspace API
  - workspace management
  - developer operations
  - workspace validation
language: python
date of note: 2024-12-07
---

# Workspace API

High-level unified API for workspace-aware system operations, providing a simplified interface to the workspace system and abstracting the complexity of underlying managers.

## Overview

The Workspace API module provides a developer-friendly interface to the workspace-aware system, consolidating functionality from the Phase 1-3 consolidated architecture. It offers simplified methods for common workspace operations including setup, validation, promotion, health monitoring, and cleanup operations.

The API is built on top of specialized managers for lifecycle, isolation, discovery, and integration operations, providing a unified entry point for all workspace management tasks. It supports multiple workspace types, template-based creation, cross-workspace dependency resolution, and comprehensive validation and monitoring capabilities.

## Classes and Methods

### Classes
- [`WorkspaceAPI`](#workspaceapi) - Unified high-level API for workspace operations
- [`WorkspaceStatus`](#workspacestatus) - Enumeration of workspace status values
- [`WorkspaceSetupResult`](#workspacesetupresult) - Result of workspace setup operations
- [`ValidationReport`](#validationreport) - Workspace validation report with issues and recommendations
- [`PromotionResult`](#promotionresult) - Result of workspace artifact promotion
- [`HealthReport`](#healthreport) - Overall workspace system health report
- [`CleanupReport`](#cleanupreport) - Result of workspace cleanup operations
- [`WorkspaceInfo`](#workspaceinfo) - Information about a workspace

## API Reference

### WorkspaceAPI

_class_ cursus.workspace.api.WorkspaceAPI(_base_path=None_)

Unified high-level API for workspace-aware system operations, abstracting the complexity of underlying managers and providing developer-friendly methods for common operations.

**Parameters:**
- **base_path** (_Optional[Union[str, Path]]_) – Base path for workspace operations (defaults to "development")

```python
from cursus.workspace.api import WorkspaceAPI

# Initialize with default path
api = WorkspaceAPI()

# Initialize with custom path
api = WorkspaceAPI("/custom/workspace/root")
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
# Basic workspace setup
result = api.setup_developer_workspace("alice")

# Setup with ML template
result = api.setup_developer_workspace(
    "bob", 
    template="ml_pipeline",
    config_overrides={"enable_gpu": True, "data_path": "/shared/data"}
)

if result.success:
    print(f"Workspace created at: {result.workspace_path}")
else:
    print(f"Setup failed: {result.message}")
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

print(f"Validation status: {report.status}")
if report.issues:
    print("Issues found:")
    for issue in report.issues:
        print(f"  - {issue}")

if report.recommendations:
    print("Recommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
```

#### list_workspaces

list_workspaces()

List all available workspaces with their current status and information.

**Returns:**
- **List[WorkspaceInfo]** – List of workspace information objects

```python
workspaces = api.list_workspaces()
print(f"Found {len(workspaces)} workspaces:")

for workspace in workspaces:
    print(f"  {workspace.developer_id}: {workspace.status} ({workspace.path})")
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

if result.success:
    print(f"Promoted {len(result.artifacts_promoted)} artifacts:")
    for artifact in result.artifacts_promoted:
        print(f"  - {artifact}")
else:
    print(f"Promotion failed: {result.message}")
```

#### get_system_health

get_system_health()

Get overall system health report including all workspace validation results.

**Returns:**
- **HealthReport** – System-wide health information with workspace reports

```python
health = api.get_system_health()
print(f"Overall system status: {health.overall_status}")

# Check for issues
if health.system_issues:
    print("System issues:")
    for issue in health.system_issues:
        print(f"  - {issue}")

# Review workspace health
for report in health.workspace_reports:
    if report.status != WorkspaceStatus.HEALTHY:
        print(f"Workspace {report.workspace_path} needs attention: {report.status}")
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
# Dry run to see what would be cleaned
report = api.cleanup_workspaces(inactive_days=60, dry_run=True)
print(f"Would clean {len(report.cleaned_workspaces)} workspaces")

# Actual cleanup if needed
if len(report.cleaned_workspaces) > 0:
    confirm = input("Proceed with cleanup? (y/N): ")
    if confirm.lower() == 'y':
        report = api.cleanup_workspaces(inactive_days=60, dry_run=False)
        print(f"Cleaned {len(report.cleaned_workspaces)} workspaces")
```

### WorkspaceStatus

_class_ cursus.workspace.api.WorkspaceStatus(_Enum_)

Enumeration defining workspace status values for health monitoring and validation.

**Values:**
- **HEALTHY** – Workspace is functioning normally
- **WARNING** – Workspace has minor issues that should be addressed
- **ERROR** – Workspace has critical issues requiring immediate attention
- **UNKNOWN** – Workspace status cannot be determined

```python
from cursus.workspace.api import WorkspaceStatus

# Check workspace status
if report.status == WorkspaceStatus.ERROR:
    print("Critical issues found!")
elif report.status == WorkspaceStatus.WARNING:
    print("Minor issues detected")
elif report.status == WorkspaceStatus.HEALTHY:
    print("Workspace is healthy")
```

### WorkspaceSetupResult

_class_ cursus.workspace.api.WorkspaceSetupResult(_success_, _workspace_path_, _developer_id_, _message_, _warnings=[]_)

Result of workspace setup operation with success status, location details, and any warnings encountered.

**Parameters:**
- **success** (_bool_) – Whether setup was successful
- **workspace_path** (_Path_) – Path to the created workspace
- **developer_id** (_str_) – Unique identifier for the developer
- **message** (_str_) – Setup result message
- **warnings** (_List[str]_) – List of warnings encountered during setup

```python
result = api.setup_developer_workspace("developer_1")

if result.success:
    print(f"Success: {result.message}")
    print(f"Workspace location: {result.workspace_path}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
else:
    print(f"Setup failed: {result.message}")
```

### ValidationReport

_class_ cursus.workspace.api.ValidationReport(_workspace_path_, _status_, _issues=[]_, _recommendations=[]_, _isolation_violations=[]_)

Workspace validation report containing status, issues found, recommendations for fixes, and detailed isolation violation information.

**Parameters:**
- **workspace_path** (_Path_) – Path to the validated workspace
- **status** (_WorkspaceStatus_) – Validation status (HEALTHY, WARNING, ERROR, UNKNOWN)
- **issues** (_List[str]_) – List of validation issues found
- **recommendations** (_List[str]_) – List of recommendations for fixing issues
- **isolation_violations** (_List[Dict[str, Any]]_) – Detailed isolation violation information

```python
report = api.validate_workspace("workspace_path")

print(f"Validation status: {report.status}")
print(f"Workspace: {report.workspace_path}")

if report.issues:
    print("\nIssues found:")
    for i, issue in enumerate(report.issues, 1):
        print(f"  {i}. {issue}")

if report.recommendations:
    print("\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")

# Check detailed isolation violations
if report.isolation_violations:
    print(f"\nFound {len(report.isolation_violations)} isolation violations")
    for violation in report.isolation_violations:
        print(f"  - {violation.get('message', 'Unknown violation')}")
```

### PromotionResult

_class_ cursus.workspace.api.PromotionResult(_success_, _source_workspace_, _target_environment_, _message_, _artifacts_promoted=[]_)

Result of workspace promotion operation with success status, source/target information, and list of promoted artifacts.

**Parameters:**
- **success** (_bool_) – Whether promotion was successful
- **source_workspace** (_Path_) – Path to the source workspace
- **target_environment** (_str_) – Target environment name
- **message** (_str_) – Promotion result message
- **artifacts_promoted** (_List[str]_) – List of promoted artifact names

```python
result = api.promote_workspace_artifacts("/workspace", "production")

if result.success:
    print(f"Promotion successful: {result.message}")
    print(f"Source: {result.source_workspace}")
    print(f"Target: {result.target_environment}")
    print(f"Artifacts promoted: {len(result.artifacts_promoted)}")
    
    for artifact in result.artifacts_promoted:
        print(f"  - {artifact}")
else:
    print(f"Promotion failed: {result.message}")
```

### HealthReport

_class_ cursus.workspace.api.HealthReport(_overall_status_, _workspace_reports=[]_, _system_issues=[]_, _recommendations=[]_)

Overall workspace system health report containing system-wide status, individual workspace reports, and system-level recommendations.

**Parameters:**
- **overall_status** (_WorkspaceStatus_) – Overall system health status
- **workspace_reports** (_List[ValidationReport]_) – Individual workspace validation reports
- **system_issues** (_List[str]_) – System-level issues found
- **recommendations** (_List[str]_) – System-level recommendations

```python
health = api.get_system_health()

print(f"System Health: {health.overall_status}")

if health.system_issues:
    print("\nSystem Issues:")
    for issue in health.system_issues:
        print(f"  - {issue}")

if health.recommendations:
    print("\nSystem Recommendations:")
    for rec in health.recommendations:
        print(f"  - {rec}")

# Review individual workspaces
print(f"\nWorkspace Details ({len(health.workspace_reports)} workspaces):")
for report in health.workspace_reports:
    status_icon = "✓" if report.status == WorkspaceStatus.HEALTHY else "⚠" if report.status == WorkspaceStatus.WARNING else "✗"
    print(f"  {status_icon} {report.workspace_path}: {report.status}")
```

### CleanupReport

_class_ cursus.workspace.api.CleanupReport(_success_, _cleaned_workspaces=[]_, _errors=[]_, _space_freed=None_)

Result of workspace cleanup operation with success status, list of cleaned workspaces, errors encountered, and space freed.

**Parameters:**
- **success** (_bool_) – Whether cleanup operation was successful
- **cleaned_workspaces** (_List[Path]_) – List of workspace paths that were cleaned
- **errors** (_List[str]_) – List of errors encountered during cleanup
- **space_freed** (_Optional[int]_) – Space freed in bytes (if available)

```python
report = api.cleanup_workspaces(inactive_days=30, dry_run=False)

if report.success:
    print(f"Cleanup completed successfully")
    print(f"Cleaned workspaces: {len(report.cleaned_workspaces)}")
    
    for workspace in report.cleaned_workspaces:
        print(f"  - {workspace}")
    
    if report.space_freed:
        space_mb = report.space_freed / (1024 * 1024)
        print(f"Space freed: {space_mb:.2f} MB")
else:
    print("Cleanup encountered errors:")
    for error in report.errors:
        print(f"  - {error}")
```

### WorkspaceInfo

_class_ cursus.workspace.api.WorkspaceInfo(_path_, _developer_id_, _status_, _created_at=None_, _last_modified=None_, _size_bytes=None_, _active_pipelines=[]_)

Information about a workspace including path, developer, status, timestamps, size, and active pipelines.

**Parameters:**
- **path** (_Path_) – Path to the workspace
- **developer_id** (_str_) – Unique identifier for the developer
- **status** (_WorkspaceStatus_) – Current workspace status
- **created_at** (_Optional[str]_) – ISO format creation timestamp
- **last_modified** (_Optional[str]_) – ISO format last modification timestamp
- **size_bytes** (_Optional[int]_) – Workspace size in bytes
- **active_pipelines** (_List[str]_) – List of active pipeline names

```python
workspaces = api.list_workspaces()

for workspace in workspaces:
    print(f"Workspace: {workspace.path}")
    print(f"  Developer: {workspace.developer_id}")
    print(f"  Status: {workspace.status}")
    
    if workspace.created_at:
        print(f"  Created: {workspace.created_at}")
    
    if workspace.size_bytes:
        size_mb = workspace.size_bytes / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
    
    if workspace.active_pipelines:
        print(f"  Active pipelines: {', '.join(workspace.active_pipelines)}")
```

## Usage Examples

### Complete Workspace Setup Workflow

```python
from cursus.workspace.api import WorkspaceAPI, WorkspaceStatus

# Initialize API
api = WorkspaceAPI()

# Set up new developer workspace
print("Setting up workspace...")
result = api.setup_developer_workspace(
    "data_scientist_1", 
    template="ml_pipeline",
    config_overrides={
        "enable_gpu": True,
        "data_path": "/shared/datasets",
        "model_registry": "mlflow"
    }
)

if result.success:
    print(f"✓ Workspace created: {result.workspace_path}")
    
    # Validate the new workspace
    print("Validating workspace...")
    report = api.validate_workspace(result.workspace_path)
    
    if report.status == WorkspaceStatus.HEALTHY:
        print("✓ Workspace validation passed")
    else:
        print(f"⚠ Validation issues: {report.status}")
        for issue in report.issues:
            print(f"  - {issue}")
else:
    print(f"✗ Setup failed: {result.message}")
```

### System Health Monitoring

```python
# Get comprehensive system health
health = api.get_system_health()

print(f"System Status: {health.overall_status}")
print(f"Total Workspaces: {len(health.workspace_reports)}")

# Count workspaces by status
status_counts = {}
for report in health.workspace_reports:
    status_counts[report.status] = status_counts.get(report.status, 0) + 1

for status, count in status_counts.items():
    print(f"  {status}: {count}")

# Show problematic workspaces
problem_workspaces = [
    report for report in health.workspace_reports 
    if report.status in [WorkspaceStatus.ERROR, WorkspaceStatus.WARNING]
]

if problem_workspaces:
    print(f"\nWorkspaces needing attention ({len(problem_workspaces)}):")
    for report in problem_workspaces:
        print(f"  {report.workspace_path}: {report.status}")
        for issue in report.issues[:3]:  # Show first 3 issues
            print(f"    - {issue}")
```

### Batch Operations

```python
# List all workspaces and perform batch validation
workspaces = api.list_workspaces()
print(f"Found {len(workspaces)} workspaces")

validation_results = []
for workspace in workspaces:
    print(f"Validating {workspace.developer_id}...")
    report = api.validate_workspace(workspace.path)
    validation_results.append((workspace, report))

# Summary of validation results
healthy_count = sum(1 for _, report in validation_results if report.status == WorkspaceStatus.HEALTHY)
warning_count = sum(1 for _, report in validation_results if report.status == WorkspaceStatus.WARNING)
error_count = sum(1 for _, report in validation_results if report.status == WorkspaceStatus.ERROR)

print(f"\nValidation Summary:")
print(f"  Healthy: {healthy_count}")
print(f"  Warnings: {warning_count}")
print(f"  Errors: {error_count}")
```

### Workspace Lifecycle Management

```python
# Create workspace
result = api.setup_developer_workspace("temp_developer", "basic")
if result.success:
    workspace_path = result.workspace_path
    
    # Use workspace for development...
    print(f"Workspace ready at: {workspace_path}")
    
    # Validate before promotion
    report = api.validate_workspace(workspace_path)
    if report.status == WorkspaceStatus.HEALTHY:
        # Promote artifacts
        promotion = api.promote_workspace_artifacts(workspace_path, "staging")
        if promotion.success:
            print(f"Promoted {len(promotion.artifacts_promoted)} artifacts")
        
        # Clean up if no longer needed
        cleanup = api.cleanup_workspaces(inactive_days=0, dry_run=False)
        print(f"Cleanup completed: {cleanup.success}")
```

## Error Handling

### Robust Error Handling Pattern

```python
from cursus.workspace.api import WorkspaceAPI, WorkspaceStatus

api = WorkspaceAPI()

try:
    # Workspace setup with error handling
    result = api.setup_developer_workspace("new_developer")
    
    if not result.success:
        print(f"Setup failed: {result.message}")
        return
    
    # Validation with error handling
    report = api.validate_workspace(result.workspace_path)
    
    if report.status == WorkspaceStatus.ERROR:
        print("Critical validation errors found:")
        for issue in report.issues:
            print(f"  - {issue}")
        
        print("Recommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
        return
    
    # Promotion with error handling
    promotion = api.promote_workspace_artifacts(result.workspace_path)
    
    if not promotion.success:
        print(f"Promotion failed: {promotion.message}")
        return
    
    print("Workspace lifecycle completed successfully")
    
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log error details for debugging
    import logging
    logging.exception("Workspace operation failed")
```

## Integration Points

### Pipeline System Integration
The Workspace API integrates with the Cursus pipeline system for cross-workspace dependency resolution and pipeline execution coordination.

### CLI Integration
Full integration with the Cursus CLI through workspace management commands, providing command-line access to all API functionality.

### Validation Framework Integration
Comprehensive integration with the validation framework for workspace structure validation, isolation checking, and compliance monitoring.

## Related Documentation

- [Workspace Templates](templates.md) - Template system for workspace creation
- [Workspace Core](core/README.md) - Core workspace management functionality
- [Workspace Validation](validation/README.md) - Workspace validation and testing systems
- [CLI Integration](../cli/workspace_cli.md) - Command-line interface for workspace operations
- [Main Workspace Documentation](README.md) - Overview of workspace management system
