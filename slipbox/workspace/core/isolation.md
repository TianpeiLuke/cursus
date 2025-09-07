---
tags:
  - code
  - workspace
  - isolation
  - boundaries
  - validation
keywords:
  - WorkspaceIsolationManager
  - IsolationViolation
  - workspace boundaries
  - path isolation
  - namespace isolation
topics:
  - workspace management
  - isolation enforcement
  - boundary validation
language: python
date of note: 2024-12-07
---

# Workspace Isolation Manager

Workspace isolation and sandboxing utilities for ensuring proper workspace boundaries and preventing cross-workspace interference.

## Overview

The Workspace Isolation Manager provides comprehensive workspace boundary validation and enforcement, path isolation and access control, namespace isolation management, and isolation violation detection. This module ensures that workspaces maintain proper isolation while enabling controlled sharing through the shared workspace mechanism.

The isolation system validates workspace boundaries, enforces path access controls, manages namespace isolation, detects various types of isolation violations, and provides workspace health monitoring. It supports both strict and permissive validation modes depending on the use case.

Key features include workspace boundary validation, path isolation enforcement, namespace isolation management, violation detection and reporting, and workspace health monitoring and validation.

## Classes and Methods

### Classes
- [`IsolationViolation`](#isolationviolation) - Represents a workspace isolation violation with metadata
- [`WorkspaceIsolationManager`](#workspaceisolationmanager) - Workspace isolation utilities and enforcement

### Methods
- [`validate_workspace_boundaries`](#validate_workspace_boundaries) - Validate workspace boundaries and isolation
- [`enforce_path_isolation`](#enforce_path_isolation) - Enforce path isolation for workspace access
- [`manage_namespace_isolation`](#manage_namespace_isolation) - Manage namespace isolation for components
- [`detect_isolation_violations`](#detect_isolation_violations) - Detect isolation violations in workspace
- [`validate_workspace_structure`](#validate_workspace_structure) - Validate workspace structure
- [`get_workspace_health`](#get_workspace_health) - Get health information for workspace
- [`get_validation_summary`](#get_validation_summary) - Get summary of validation activities
- [`get_statistics`](#get_statistics) - Get isolation management statistics

## API Reference

### IsolationViolation

_class_ cursus.workspace.core.isolation.IsolationViolation(_violation_type_, _workspace_id_, _description_, _severity="medium"_, _details={}_, _detected_at=None_, _detected_path=None_, _recommendation=""_)

Represents a workspace isolation violation with comprehensive metadata and tracking information.

**Parameters:**
- **violation_type** (_str_) – Type of violation ('path_access', 'namespace_conflict', 'environment', 'dependency', 'resource').
- **workspace_id** (_str_) – Workspace identifier where violation occurred.
- **description** (_str_) – Description of the violation.
- **severity** (_str_) – Severity level ('low', 'medium', 'high', 'critical'), defaults to 'medium'.
- **details** (_Dict[str, Any]_) – Additional violation details dictionary, defaults to empty dict.
- **detected_at** (_Optional[datetime]_) – When violation was detected, defaults to current time.
- **detected_path** (_Optional[str]_) – Path where violation was detected.
- **recommendation** (_str_) – Recommendation for fixing the violation, defaults to empty string.

```python
from cursus.workspace.core.isolation import IsolationViolation
from datetime import datetime

# Create isolation violation
violation = IsolationViolation(
    violation_type="path_access",
    workspace_id="alice",
    description="Attempted access to path outside workspace boundaries",
    severity="high",
    details={
        "attempted_path": "/external/path",
        "workspace_path": "/workspace/alice"
    },
    detected_path="/workspace/alice/src/script.py",
    recommendation="Use relative paths within workspace boundaries"
)

print("Violation type:", violation.violation_type)
print("Severity:", violation.severity)
```

#### to_dict

to_dict()

Convert isolation violation to dictionary representation.

**Returns:**
- **Dict[str, Any]** – Dictionary containing all violation information including metadata and recommendations.

```python
# Convert violation to dictionary
violation_dict = violation.to_dict()

print("Violation details:", violation_dict['details'])
print("Detected at:", violation_dict['detected_at'])
print("Recommendation:", violation_dict['recommendation'])
```

### WorkspaceIsolationManager

_class_ cursus.workspace.core.isolation.WorkspaceIsolationManager(_workspace_manager_)

Workspace isolation utilities providing boundary validation, access control, and violation detection.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Parent WorkspaceManager instance for integration.

```python
from cursus.workspace.core.isolation import WorkspaceIsolationManager
from cursus.workspace.core.manager import WorkspaceManager

# Create isolation manager
workspace_manager = WorkspaceManager("/path/to/workspace")
isolation_manager = WorkspaceIsolationManager(workspace_manager)

# Validate workspace boundaries
validation = isolation_manager.validate_workspace_boundaries("alice")
print("Boundaries valid:", validation['valid'])
```

#### validate_workspace_boundaries

validate_workspace_boundaries(_workspace_id_)

Validate workspace boundaries and isolation with comprehensive checks.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to validate.

**Returns:**
- **Dict[str, Any]** – Validation result with boundary check results, violations, warnings, and performed checks.

```python
# Validate workspace boundaries
validation_result = isolation_manager.validate_workspace_boundaries("alice")

if validation_result['valid']:
    print("Workspace boundaries are valid")
    print("Checks performed:", validation_result['checks_performed'])
else:
    print("Boundary violations:", validation_result['violations'])
    print("Warnings:", validation_result['warnings'])

# Review specific checks
for check in validation_result['checks_performed']:
    print(f"✓ {check} check completed")
```

#### enforce_path_isolation

enforce_path_isolation(_workspace_path_, _access_path_)

Enforce path isolation for workspace access with boundary validation.

**Parameters:**
- **workspace_path** (_str_) – Workspace root path.
- **access_path** (_str_) – Path being accessed.

**Returns:**
- **bool** – True if access is allowed, False if access violates isolation boundaries.

```python
# Check path access permissions
workspace_path = "/workspace/alice"
access_path = "/workspace/alice/src/data.py"

is_allowed = isolation_manager.enforce_path_isolation(workspace_path, access_path)

if is_allowed:
    print("Path access allowed")
else:
    print("Path access denied - isolation violation")

# Test external path access
external_path = "/external/system/file.py"
external_allowed = isolation_manager.enforce_path_isolation(workspace_path, external_path)
print("External access allowed:", external_allowed)
```

#### manage_namespace_isolation

manage_namespace_isolation(_workspace_id_, _component_name_)

Manage namespace isolation for workspace components with automatic namespacing.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier.
- **component_name** (_str_) – Component name to namespace.

**Returns:**
- **str** – Namespaced component name for isolation.

```python
# Create namespaced component names
alice_component = isolation_manager.manage_namespace_isolation("alice", "data_processor")
print("Alice's component:", alice_component)  # "alice:data_processor"

bob_component = isolation_manager.manage_namespace_isolation("bob", "data_processor")
print("Bob's component:", bob_component)  # "bob:data_processor"

# Shared components don't get namespaced
shared_component = isolation_manager.manage_namespace_isolation("shared", "common_utils")
print("Shared component:", shared_component)  # "common_utils"
```

#### detect_isolation_violations

detect_isolation_violations(_workspace_id_)

Detect isolation violations in workspace with comprehensive analysis.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to analyze for violations.

**Returns:**
- **List[IsolationViolation]** – List of detected isolation violations with details and recommendations.

```python
# Detect isolation violations
violations = isolation_manager.detect_isolation_violations("alice")

print(f"Found {len(violations)} violations")

for violation in violations:
    print(f"Type: {violation.violation_type}")
    print(f"Severity: {violation.severity}")
    print(f"Description: {violation.description}")
    if violation.recommendation:
        print(f"Recommendation: {violation.recommendation}")
    print("---")

# Filter by severity
critical_violations = [v for v in violations if v.severity == "critical"]
if critical_violations:
    print(f"Critical violations requiring immediate attention: {len(critical_violations)}")
```

#### validate_workspace_structure

validate_workspace_structure(_workspace_root=None_, _strict=False_)

Validate workspace structure with configurable strictness levels.

**Parameters:**
- **workspace_root** (_Optional[Union[str, Path]]_) – Root directory to validate, uses manager's root if None.
- **strict** (_bool_) – Whether to apply strict validation rules, defaults to False.

**Returns:**
- **Tuple[bool, List[str]]** – Tuple of (is_valid, list_of_issues) with validation results.

```python
from pathlib import Path

# Validate workspace structure
is_valid, issues = isolation_manager.validate_workspace_structure(
    workspace_root="/path/to/workspace",
    strict=True
)

if is_valid:
    print("Workspace structure is valid")
else:
    print("Workspace structure issues:")
    for issue in issues:
        print(f"  - {issue}")

# Validate with permissive rules
is_valid_permissive, issues_permissive = isolation_manager.validate_workspace_structure(
    strict=False
)
print(f"Permissive validation: {'PASS' if is_valid_permissive else 'FAIL'}")
```

#### get_workspace_health

get_workspace_health(_workspace_id_)

Get comprehensive health information for a workspace.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to check health for.

**Returns:**
- **Dict[str, Any]** – Health information with status, issues, warnings, and health score.

```python
# Get workspace health information
health_info = isolation_manager.get_workspace_health("alice")

print(f"Workspace healthy: {health_info['healthy']}")
print(f"Health score: {health_info['health_score']}/100")
print(f"Last checked: {health_info['last_checked']}")

if health_info['issues']:
    print("Health issues:")
    for issue in health_info['issues']:
        print(f"  - {issue}")

if health_info['warnings']:
    print("Health warnings:")
    for warning in health_info['warnings']:
        print(f"  - {warning}")

# Monitor health over time
if health_info['health_score'] < 70:
    print("⚠️  Workspace requires attention")
elif health_info['health_score'] < 90:
    print("⚡ Workspace has minor issues")
else:
    print("✅ Workspace is healthy")
```

#### get_validation_summary

get_validation_summary()

Get summary of validation activities and violation tracking.

**Returns:**
- **Dict[str, Any]** – Summary of violations by type, severity distribution, and recent violations.

```python
# Get validation activity summary
summary = isolation_manager.get_validation_summary()

print("Total violations:", summary['total_violations'])
print("Violation types:", summary['violation_types'])
print("Severity distribution:", summary['severity_distribution'])

# Review recent violations
print("\nRecent violations:")
for violation in summary['recent_violations']:
    print(f"  {violation['workspace_id']}: {violation['violation_type']} ({violation['severity']})")
```

#### get_statistics

get_statistics()

Get comprehensive isolation management statistics.

**Returns:**
- **Dict[str, Any]** – Statistics including isolation checks, workspace health, and violation summaries.

```python
# Get comprehensive isolation statistics
stats = isolation_manager.get_statistics()

print("Isolation checks:", stats['isolation_checks'])
print("Total violations:", stats['isolation_checks']['total_violations'])
print("Active workspaces:", stats['isolation_checks']['active_workspaces'])
print("Healthy workspaces:", stats['isolation_checks']['healthy_workspaces'])

# Detailed violation summary
violation_summary = stats['violation_summary']
print("Violation summary:", violation_summary)
```

## Isolation Workflow

### Complete Workspace Validation

```python
# Complete workspace validation workflow
def validate_workspace_completely(isolation_manager, workspace_id):
    print(f"Validating workspace: {workspace_id}")
    
    # 1. Validate boundaries
    boundary_validation = isolation_manager.validate_workspace_boundaries(workspace_id)
    print(f"Boundary validation: {'PASS' if boundary_validation['valid'] else 'FAIL'}")
    
    # 2. Detect violations
    violations = isolation_manager.detect_isolation_violations(workspace_id)
    print(f"Violations detected: {len(violations)}")
    
    # 3. Check workspace health
    health = isolation_manager.get_workspace_health(workspace_id)
    print(f"Health score: {health['health_score']}/100")
    
    # 4. Validate structure
    structure_valid, structure_issues = isolation_manager.validate_workspace_structure()
    print(f"Structure validation: {'PASS' if structure_valid else 'FAIL'}")
    
    # Overall assessment
    overall_healthy = (
        boundary_validation['valid'] and
        len(violations) == 0 and
        health['healthy'] and
        structure_valid
    )
    
    print(f"Overall workspace status: {'HEALTHY' if overall_healthy else 'NEEDS ATTENTION'}")
    return overall_healthy
```

### Isolation Enforcement

```python
# Enforce isolation during component access
def safe_component_access(isolation_manager, workspace_id, component_path):
    workspace_path = f"/workspace/{workspace_id}"
    
    # Check path isolation
    if not isolation_manager.enforce_path_isolation(workspace_path, component_path):
        print(f"Access denied: {component_path} violates isolation boundaries")
        return None
    
    # Create namespaced component name
    component_name = Path(component_path).stem
    namespaced_name = isolation_manager.manage_namespace_isolation(workspace_id, component_name)
    
    print(f"Access granted: {component_path} -> {namespaced_name}")
    return namespaced_name
```

### Health Monitoring

```python
# Monitor workspace health over time
def monitor_workspace_health(isolation_manager, workspace_ids):
    health_report = {}
    
    for workspace_id in workspace_ids:
        health = isolation_manager.get_workspace_health(workspace_id)
        health_report[workspace_id] = health
        
        # Alert on critical issues
        if not health['healthy']:
            print(f"🚨 ALERT: Workspace {workspace_id} is unhealthy!")
            print(f"   Health score: {health['health_score']}/100")
            print(f"   Issues: {len(health['issues'])}")
        
        # Detect violations
        violations = isolation_manager.detect_isolation_violations(workspace_id)
        critical_violations = [v for v in violations if v.severity == "critical"]
        
        if critical_violations:
            print(f"🔥 CRITICAL: {len(critical_violations)} critical violations in {workspace_id}")
    
    return health_report
```

### Violation Response

```python
# Respond to isolation violations
def handle_isolation_violations(isolation_manager, workspace_id):
    violations = isolation_manager.detect_isolation_violations(workspace_id)
    
    if not violations:
        print(f"✅ No violations detected in {workspace_id}")
        return
    
    # Group by severity
    by_severity = {}
    for violation in violations:
        severity = violation.severity
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(violation)
    
    # Handle critical violations first
    if "critical" in by_severity:
        print(f"🔥 Handling {len(by_severity['critical'])} critical violations")
        for violation in by_severity["critical"]:
            print(f"   {violation.description}")
            if violation.recommendation:
                print(f"   → {violation.recommendation}")
    
    # Report other violations
    for severity in ["high", "medium", "low"]:
        if severity in by_severity:
            print(f"⚠️  {len(by_severity[severity])} {severity} severity violations")
```

## Related Documentation

- [Workspace Manager](manager.md) - Consolidated workspace management system
- [Workspace Discovery Manager](discovery.md) - Component discovery and validation
- [Workspace Integration Manager](integration.md) - Integration staging and promotion
- [Workspace Configuration](config.md) - Pipeline and step configuration models
- [Workspace API](../api.md) - High-level workspace API interface
