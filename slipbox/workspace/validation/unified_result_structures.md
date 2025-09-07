---
tags:
  - code
  - workspace
  - validation
  - data-structures
  - results
keywords:
  - BaseValidationResult
  - UnifiedValidationResult
  - ValidationSummary
  - WorkspaceValidationEntry
  - ValidationResultBuilder
topics:
  - validation results
  - data structures
  - workspace validation
  - result standardization
language: python
date of note: 2024-12-07
---

# Unified Result Structures

Standardized data structures for workspace validation results that work identically for single and multi-workspace scenarios with consistent reporting and extensible result structures.

## Overview

The Unified Result Structures module provides standardized data models for workspace validation results, ensuring consistent handling regardless of whether validating a single workspace (count=1) or multiple workspaces (count=N). The module implements unified result structures, consistent summary statistics, standardized error handling, and extensible result formats for future enhancements.

The system supports base validation result classes with common fields, specialized result types for different validation scenarios, unified summary structures that work for any workspace count, builder patterns for result construction, and backward compatibility with existing result formats.

## Classes and Methods

### Core Result Classes
- [`BaseValidationResult`](#basevalidationresult) - Base class for all validation results with common fields
- [`UnifiedValidationResult`](#unifiedvalidationresult) - Standardized result structure for all validation scenarios
- [`ValidationSummary`](#validationsummary) - Unified summary that works for count=1 or count=N workspaces
- [`WorkspaceValidationEntry`](#workspacevalidationentry) - Validation result for a single workspace entry

### Specialized Result Classes
- [`WorkspaceValidationResult`](#workspacevalidationresult) - Validation result for workspace validation
- [`AlignmentTestResult`](#alignmenttestresult) - Result for alignment testing validation
- [`BuilderTestResult`](#buildertestresult) - Result for builder testing validation
- [`IsolationTestResult`](#isolationtestresult) - Result for isolation testing validation

### Builder and Utility Classes
- [`ValidationResultBuilder`](#validationresultbuilder) - Builder class for creating unified validation results

### Utility Functions
- [`create_single_workspace_result`](#create_single_workspace_result) - Convenience function for single workspace results
- [`create_empty_result`](#create_empty_result) - Create empty validation result for error cases

## API Reference

### BaseValidationResult

_class_ cursus.workspace.validation.unified_result_structures.BaseValidationResult(_**kwargs_)

Base class for all validation results with common fields and standardized interface for validation reporting.

**Parameters:**
- **success** (_bool_) – Whether the validation was successful
- **timestamp** (_datetime_) – When the validation was performed (defaults to now)
- **workspace_path** (_Path_) – Path to the workspace that was validated
- **messages** (_List[str]_) – Informational messages from validation
- **warnings** (_List[str]_) – Warning messages from validation
- **errors** (_List[str]_) – Error messages from validation

```python
from cursus.workspace.validation.unified_result_structures import BaseValidationResult
from pathlib import Path

# Create base validation result
result = BaseValidationResult(
    success=True,
    workspace_path=Path("/workspaces/alice"),
    messages=["Validation completed successfully"],
    warnings=["Minor configuration issue detected"],
    errors=[]
)

print(f"Success: {result.success}")
print(f"Has warnings: {result.has_warnings}")
print(f"Message count: {result.message_count}")
```

#### Properties

##### has_warnings

_property_ has_warnings

Check if there are any warnings.

**Returns:**
- **bool** – True if warnings exist

##### has_errors

_property_ has_errors

Check if there are any errors.

**Returns:**
- **bool** – True if errors exist

##### message_count

_property_ message_count

Get total number of messages.

**Returns:**
- **int** – Total count of messages, warnings, and errors

#### Methods

##### add_message

add_message(_message_)

Add an informational message.

**Parameters:**
- **message** (_str_) – Message to add

```python
result.add_message("Configuration file loaded successfully")
result.add_message("Found 15 pipeline definitions")
```

##### add_warning

add_warning(_warning_)

Add a warning message.

**Parameters:**
- **warning** (_str_) – Warning to add

```python
result.add_warning("Deprecated configuration option detected")
result.add_warning("Missing optional dependency")
```

##### add_error

add_error(_error_)

Add an error message.

**Parameters:**
- **error** (_str_) – Error to add

```python
result.add_error("Required configuration file not found")
result.add_error("Invalid workspace structure")
```

### UnifiedValidationResult

_class_ cursus.workspace.validation.unified_result_structures.UnifiedValidationResult(_**kwargs_)

Standardized result structure for all validation scenarios that works identically whether validating a single workspace (count=1) or multiple workspaces (count=N).

**Parameters:**
- **workspace_root** (_str_) – Root directory of the workspace(s)
- **workspace_type** (_str_) – Type of workspace structure (single or multi)
- **validation_start_time** (_str_) – ISO timestamp when validation started
- **validation_end_time** (_Optional[str]_) – ISO timestamp when validation completed
- **validation_duration_seconds** (_Optional[float]_) – Total validation duration in seconds
- **workspaces** (_Dict[str, WorkspaceValidationEntry]_) – Validation results for each workspace
- **summary** (_ValidationSummary_) – Summary statistics for all workspaces
- **recommendations** (_List[str]_) – Recommendations based on validation results
- **global_error** (_Optional[str]_) – Global error that prevented validation

```python
from cursus.workspace.validation.unified_result_structures import (
    UnifiedValidationResult, ValidationSummary, WorkspaceValidationEntry
)

# Create unified validation result
result = UnifiedValidationResult(
    workspace_root="/workspaces",
    workspace_type="multi",
    validation_start_time="2024-12-07T10:00:00Z",
    workspaces={
        "alice": WorkspaceValidationEntry(
            workspace_id="alice",
            workspace_type="developer",
            workspace_path="/workspaces/alice",
            success=True
        )
    },
    summary=ValidationSummary(
        total_workspaces=1,
        successful_workspaces=1,
        failed_workspaces=0,
        success_rate=1.0
    )
)

print(f"Workspace count: {result.workspace_count}")
print(f"Overall success: {result.overall_success}")
```

#### Properties

##### is_single_workspace

_property_ is_single_workspace

Check if this represents a single workspace validation.

**Returns:**
- **bool** – True if single workspace

##### is_multi_workspace

_property_ is_multi_workspace

Check if this represents a multi-workspace validation.

**Returns:**
- **bool** – True if multi-workspace

##### workspace_count

_property_ workspace_count

Get the number of workspaces validated.

**Returns:**
- **int** – Number of workspaces

##### workspace_ids

_property_ workspace_ids

Get list of workspace identifiers.

**Returns:**
- **List[str]** – List of workspace IDs

##### overall_success

_property_ overall_success

Check if all workspaces passed validation.

**Returns:**
- **bool** – True if all successful and no global errors

##### has_failures

_property_ has_failures

Check if any workspaces failed validation.

**Returns:**
- **bool** – True if any failures or global errors

#### Methods

##### get_workspace_result

get_workspace_result(_workspace_id_)

Get validation result for a specific workspace.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier

**Returns:**
- **Optional[WorkspaceValidationResult]** – Workspace result or None

```python
# Get specific workspace result
alice_result = result.get_workspace_result("alice")
if alice_result:
    print(f"Alice's workspace: {'✓ PASS' if alice_result.success else '✗ FAIL'}")
```

##### get_failed_workspaces

get_failed_workspaces()

Get list of workspace IDs that failed validation.

**Returns:**
- **List[str]** – List of failed workspace IDs

```python
# Check for failed workspaces
failed = result.get_failed_workspaces()
if failed:
    print(f"Failed workspaces: {', '.join(failed)}")
```

##### get_successful_workspaces

get_successful_workspaces()

Get list of workspace IDs that passed validation.

**Returns:**
- **List[str]** – List of successful workspace IDs

```python
# Check successful workspaces
successful = result.get_successful_workspaces()
print(f"Successful workspaces: {', '.join(successful)}")
```

##### get_validation_duration

get_validation_duration()

Get total validation duration, calculating if not set.

**Returns:**
- **Optional[float]** – Duration in seconds or None

```python
# Get validation duration
duration = result.get_validation_duration()
if duration:
    print(f"Validation took {duration:.2f} seconds")
```

##### add_recommendation

add_recommendation(_recommendation_)

Add a recommendation to the results.

**Parameters:**
- **recommendation** (_str_) – Recommendation to add

```python
result.add_recommendation("Consider upgrading workspace configuration")
result.add_recommendation("Review isolation boundaries")
```

##### finalize_validation

finalize_validation(_end_time=None_)

Finalize validation results with end time and duration.

**Parameters:**
- **end_time** (_Optional[datetime]_) – End time (defaults to now)

```python
from datetime import datetime

# Finalize validation
result.finalize_validation(datetime.now())
print(f"Validation completed in {result.validation_duration_seconds:.2f}s")
```

### ValidationSummary

_class_ cursus.workspace.validation.unified_result_structures.ValidationSummary(_**kwargs_)

Unified summary that works for count=1 or count=N workspaces, providing consistent statistics regardless of workspace count.

**Parameters:**
- **total_workspaces** (_int_) – Total number of workspaces validated
- **successful_workspaces** (_int_) – Number of workspaces that passed validation
- **failed_workspaces** (_int_) – Number of workspaces that failed validation
- **success_rate** (_float_) – Success rate as decimal (0.0 to 1.0)

```python
from cursus.workspace.validation.unified_result_structures import ValidationSummary

# Create validation summary
summary = ValidationSummary(
    total_workspaces=5,
    successful_workspaces=4,
    failed_workspaces=1,
    success_rate=0.8
)

print(f"Success rate: {summary.success_percentage:.1f}%")
print(f"All successful: {summary.all_successful}")
```

#### Properties

##### success_percentage

_property_ success_percentage

Get success rate as percentage (0-100).

**Returns:**
- **float** – Success rate as percentage

##### all_successful

_property_ all_successful

Check if all workspaces passed validation.

**Returns:**
- **bool** – True if all successful

##### any_failed

_property_ any_failed

Check if any workspaces failed validation.

**Returns:**
- **bool** – True if any failed

### WorkspaceValidationEntry

_class_ cursus.workspace.validation.unified_result_structures.WorkspaceValidationEntry(_**kwargs_)

Validation result for a single workspace entry, used for both single workspace scenarios (count=1) and individual workspace results in multi-workspace scenarios.

**Parameters:**
- **workspace_id** (_str_) – Unique identifier for the workspace
- **workspace_type** (_str_) – Type of workspace (single, developer, shared)
- **workspace_path** (_str_) – File system path to the workspace
- **success** (_bool_) – Overall validation success for this workspace
- **validation_start_time** (_Optional[str]_) – ISO timestamp when validation started
- **validation_end_time** (_Optional[str]_) – ISO timestamp when validation completed
- **validation_duration_seconds** (_Optional[float]_) – Duration of validation in seconds
- **alignment_results** (_Optional[Dict[str, Any]]_) – Results from alignment validation
- **builder_results** (_Optional[Dict[str, Any]]_) – Results from builder validation
- **error** (_Optional[str]_) – Error message if validation failed
- **warnings** (_List[str]_) – Warning messages from validation
- **developer_info** (_Optional[Dict[str, Any]]_) – Information about the developer workspace

```python
from cursus.workspace.validation.unified_result_structures import WorkspaceValidationEntry

# Create workspace validation entry
entry = WorkspaceValidationEntry(
    workspace_id="alice",
    workspace_type="developer",
    workspace_path="/workspaces/alice",
    success=True,
    alignment_results={"score": 0.95, "checks_passed": 18},
    builder_results={"total_builders": 5, "successful": 5},
    warnings=["Minor configuration issue"]
)

print(f"Validation types run: {entry.validation_types_run}")
print(f"Has alignment results: {entry.has_alignment_results}")
```

#### Properties

##### has_alignment_results

_property_ has_alignment_results

Check if alignment validation was performed.

**Returns:**
- **bool** – True if alignment results exist

##### has_builder_results

_property_ has_builder_results

Check if builder validation was performed.

**Returns:**
- **bool** – True if builder results exist

##### validation_types_run

_property_ validation_types_run

Get list of validation types that were executed.

**Returns:**
- **List[str]** – List of validation types

#### Methods

##### get_validation_duration

get_validation_duration()

Get validation duration, calculating if not set.

**Returns:**
- **Optional[float]** – Duration in seconds or None

```python
# Get validation duration for workspace
duration = entry.get_validation_duration()
if duration:
    print(f"Workspace validation took {duration:.2f} seconds")
```

### ValidationResultBuilder

_class_ cursus.workspace.validation.unified_result_structures.ValidationResultBuilder(_workspace_root_, _workspace_type_, _start_time=None_)

Builder class for creating unified validation results, simplifying result creation and ensuring consistency across different validation scenarios.

**Parameters:**
- **workspace_root** (_str_) – Root directory of workspace(s)
- **workspace_type** (_str_) – Type of workspace structure (single or multi)
- **start_time** (_Optional[datetime]_) – Validation start time (defaults to now)

```python
from cursus.workspace.validation.unified_result_structures import ValidationResultBuilder
from datetime import datetime

# Create validation result builder
builder = ValidationResultBuilder(
    workspace_root="/workspaces",
    workspace_type="multi",
    start_time=datetime.now()
)

# Add workspace results
builder.add_workspace_result(
    workspace_id="alice",
    workspace_type="developer",
    workspace_path="/workspaces/alice",
    success=True,
    alignment_results={"score": 0.95}
)

# Add recommendations
builder.add_recommendation("Consider upgrading configuration")

# Build final result
result = builder.build()
```

#### Methods

##### add_workspace_result

add_workspace_result(_workspace_id_, _workspace_type_, _workspace_path_, _success_, _**kwargs_)

Add a workspace validation result.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier
- **workspace_type** (_str_) – Type of workspace
- **workspace_path** (_str_) – Path to workspace
- **success** (_bool_) – Whether validation succeeded
- **alignment_results** (_Optional[Dict[str, Any]]_) – Alignment validation results
- **builder_results** (_Optional[Dict[str, Any]]_) – Builder validation results
- **error** (_Optional[str]_) – Error message if failed
- **warnings** (_Optional[List[str]]_) – Warning messages
- **developer_info** (_Optional[Dict[str, Any]]_) – Developer information
- **validation_start_time** (_Optional[datetime]_) – Validation start time
- **validation_end_time** (_Optional[datetime]_) – Validation end time

**Returns:**
- **ValidationResultBuilder** – Self for method chaining

```python
# Add multiple workspace results
builder.add_workspace_result(
    workspace_id="alice",
    workspace_type="developer", 
    workspace_path="/workspaces/alice",
    success=True,
    alignment_results={"score": 0.95, "checks_passed": 18}
).add_workspace_result(
    workspace_id="bob",
    workspace_type="developer",
    workspace_path="/workspaces/bob", 
    success=False,
    error="Configuration validation failed",
    warnings=["Deprecated settings found"]
)
```

##### add_recommendation

add_recommendation(_recommendation_)

Add a recommendation.

**Parameters:**
- **recommendation** (_str_) – Recommendation to add

**Returns:**
- **ValidationResultBuilder** – Self for method chaining

```python
# Add multiple recommendations
builder.add_recommendation("Upgrade workspace configuration").add_recommendation("Review isolation boundaries")
```

##### set_global_error

set_global_error(_error_)

Set a global error that prevented validation.

**Parameters:**
- **error** (_str_) – Global error message

**Returns:**
- **ValidationResultBuilder** – Self for method chaining

```python
# Set global error
builder.set_global_error("Workspace root directory not found")
```

##### build

build(_end_time=None_)

Build the final validation result.

**Parameters:**
- **end_time** (_Optional[datetime]_) – End time (defaults to now)

**Returns:**
- **UnifiedValidationResult** – Final validation result

```python
# Build final result
result = builder.build()
print(f"Validation completed: {result.overall_success}")
```

### Specialized Result Classes

### WorkspaceValidationResult

_class_ cursus.workspace.validation.unified_result_structures.WorkspaceValidationResult(_**kwargs_)

Validation result for workspace validation, inheriting common fields from BaseValidationResult.

**Additional Parameters:**
- **violations** (_List[Dict[str, Any]]_) – List of validation violations found
- **isolation_score** (_Optional[float]_) – Workspace isolation score (0.0 to 1.0)

```python
from cursus.workspace.validation.unified_result_structures import WorkspaceValidationResult
from pathlib import Path

# Create workspace validation result
result = WorkspaceValidationResult(
    success=True,
    workspace_path=Path("/workspaces/alice"),
    violations=[],
    isolation_score=0.95
)

print(f"Has violations: {result.has_violations}")
print(f"Isolation score: {result.isolation_score}")
```

#### Properties

##### has_violations

_property_ has_violations

Check if there are any violations.

**Returns:**
- **bool** – True if violations exist

##### violation_count

_property_ violation_count

Get number of violations.

**Returns:**
- **int** – Number of violations

#### Methods

##### add_violation

add_violation(_violation_)

Add a validation violation.

**Parameters:**
- **violation** (_Dict[str, Any]_) – Violation details

```python
result.add_violation({
    "type": "isolation_breach",
    "severity": "high",
    "message": "Cross-workspace dependency detected",
    "recommendation": "Review workspace boundaries"
})
```

### AlignmentTestResult

_class_ cursus.workspace.validation.unified_result_structures.AlignmentTestResult(_**kwargs_)

Result for alignment testing validation, inheriting common fields from BaseValidationResult.

**Additional Parameters:**
- **alignment_score** (_float_) – Alignment score (0.0 to 1.0)
- **failed_checks** (_List[str]_) – List of failed alignment checks
- **level_results** (_Dict[str, Dict[str, Any]]_) – Results for each validation level

```python
from cursus.workspace.validation.unified_result_structures import AlignmentTestResult
from pathlib import Path

# Create alignment test result
result = AlignmentTestResult(
    success=True,
    workspace_path=Path("/workspaces/alice"),
    alignment_score=0.92,
    failed_checks=[],
    level_results={
        "level1": {"score": 0.95, "checks_passed": 10},
        "level2": {"score": 0.89, "checks_passed": 8}
    }
)

print(f"Alignment score: {result.alignment_score}")
print(f"Failed checks: {result.failed_check_count}")
```

### BuilderTestResult

_class_ cursus.workspace.validation.unified_result_structures.BuilderTestResult(_**kwargs_)

Result for builder testing validation, inheriting common fields from BaseValidationResult.

**Additional Parameters:**
- **test_results** (_Dict[str, Any]_) – Detailed test results for each builder
- **total_builders** (_int_) – Total number of builders tested
- **successful_tests** (_int_) – Number of successful builder tests
- **failed_tests** (_int_) – Number of failed builder tests

```python
from cursus.workspace.validation.unified_result_structures import BuilderTestResult
from pathlib import Path

# Create builder test result
result = BuilderTestResult(
    success=True,
    workspace_path=Path("/workspaces/alice"),
    test_results={
        "data_prep_builder": {"success": True, "duration": 1.2},
        "training_builder": {"success": True, "duration": 2.1}
    },
    total_builders=2,
    successful_tests=2,
    failed_tests=0
)

print(f"Success rate: {result.success_rate:.2f}")
print(f"Has test failures: {result.has_test_failures}")
```

### IsolationTestResult

_class_ cursus.workspace.validation.unified_result_structures.IsolationTestResult(_**kwargs_)

Result for isolation testing validation, inheriting common fields from BaseValidationResult.

**Additional Parameters:**
- **isolation_violations** (_List[str]_) – List of isolation violations
- **boundary_checks** (_Dict[str, bool]_) – Results of boundary checks
- **recommendations** (_List[str]_) – Recommendations for fixing issues

```python
from cursus.workspace.validation.unified_result_structures import IsolationTestResult
from pathlib import Path

# Create isolation test result
result = IsolationTestResult(
    success=True,
    workspace_path=Path("/workspaces/alice"),
    isolation_violations=[],
    boundary_checks={
        "file_system_isolation": True,
        "python_path_isolation": True,
        "config_isolation": True
    },
    recommendations=[]
)

print(f"Is isolated: {result.is_isolated}")
print(f"Violation count: {result.violation_count}")
```

## Utility Functions

### create_single_workspace_result

create_single_workspace_result(_workspace_root_, _**kwargs_)

Convenience function to create a single workspace validation result.

**Parameters:**
- **workspace_root** (_str_) – Root directory of workspace
- **workspace_id** (_str_) – Workspace identifier (defaults to "default")
- **workspace_type** (_str_) – Type of workspace (defaults to "single")
- **workspace_path** (_Optional[str]_) – Path to workspace (defaults to workspace_root)
- **success** (_bool_) – Whether validation succeeded (defaults to True)
- **alignment_results** (_Optional[Dict[str, Any]]_) – Alignment validation results
- **builder_results** (_Optional[Dict[str, Any]]_) – Builder validation results
- **error** (_Optional[str]_) – Error message if failed
- **recommendations** (_Optional[List[str]]_) – Recommendations
- **start_time** (_Optional[datetime]_) – Validation start time
- **end_time** (_Optional[datetime]_) – Validation end time

**Returns:**
- **UnifiedValidationResult** – Single workspace validation result

```python
from cursus.workspace.validation.unified_result_structures import create_single_workspace_result

# Create single workspace result
result = create_single_workspace_result(
    workspace_root="/workspaces/alice",
    workspace_id="alice",
    success=True,
    alignment_results={"score": 0.95, "checks_passed": 18},
    recommendations=["Consider upgrading configuration"]
)

print(f"Single workspace result: {result.overall_success}")
print(f"Workspace count: {result.workspace_count}")
```

### create_empty_result

create_empty_result(_workspace_root_, _workspace_type="unknown"_, _error="No workspaces found"_, _start_time=None_)

Create an empty validation result for cases where no workspaces are found.

**Parameters:**
- **workspace_root** (_str_) – Root directory
- **workspace_type** (_str_) – Type of workspace (defaults to "unknown")
- **error** (_str_) – Error message (defaults to "No workspaces found")
- **start_time** (_Optional[datetime]_) – Start time

**Returns:**
- **UnifiedValidationResult** – Empty validation result with error

```python
from cursus.workspace.validation.unified_result_structures import create_empty_result

# Create empty result for error case
result = create_empty_result(
    workspace_root="/invalid/path",
    error="Workspace directory does not exist"
)

print(f"Has global error: {result.global_error is not None}")
print(f"Recommendations: {len(result.recommendations)}")
```

## Usage Examples

### Creating Validation Results with Builder Pattern

```python
from cursus.workspace.validation.unified_result_structures import ValidationResultBuilder
from datetime import datetime

# Create comprehensive validation result
def create_multi_workspace_validation():
    """Create validation result for multiple workspaces."""
    
    builder = ValidationResultBuilder(
        workspace_root="/workspaces",
        workspace_type="multi",
        start_time=datetime.now()
    )
    
    # Add successful workspace
    builder.add_workspace_result(
        workspace_id="alice",
        workspace_type="developer",
        workspace_path="/workspaces/alice",
        success=True,
        alignment_results={
            "score": 0.95,
            "checks_passed": 18,
            "checks_failed": 1,
            "level_results": {
                "level1": {"score": 0.98, "checks": 10},
                "level2": {"score": 0.92, "checks": 8}
            }
        },
        builder_results={
            "total_builders": 5,
            "successful": 5,
            "failed": 0,
            "test_results": {
                "data_prep_builder": {"success": True, "duration": 1.2},
                "training_builder": {"success": True, "duration": 2.1}
            }
        },
        warnings=["Minor configuration issue in pipeline.yaml"]
    )
    
    # Add failed workspace
    builder.add_workspace_result(
        workspace_id="bob",
        workspace_type="developer",
        workspace_path="/workspaces/bob",
        success=False,
        error="Configuration validation failed",
        alignment_results={
            "score": 0.65,
            "checks_passed": 12,
            "checks_failed": 6
        },
        warnings=["Deprecated settings found", "Missing optional dependencies"]
    )
    
    # Add recommendations
    builder.add_recommendation("Upgrade workspace configurations to latest version")
    builder.add_recommendation("Review failed alignment checks in Bob's workspace")
    builder.add_recommendation("Consider implementing automated configuration updates")
    
    # Build final result
    result = builder.build()
    
    print(f"Validation Summary:")
    print(f"  Total workspaces: {result.summary.total_workspaces}")
    print(f"  Success rate: {result.summary.success_percentage:.1f}%")
    print(f"  Overall success: {result.overall_success}")
    
    return result

# Create the validation result
validation_result = create_multi_workspace_validation()
```

### Working with Specialized Result Types

```python
from cursus.workspace.validation.unified_result_structures import (
    WorkspaceValidationResult, AlignmentTestResult, BuilderTestResult, IsolationTestResult
)
from pathlib import Path

# Create workspace validation result with violations
workspace_result = WorkspaceValidationResult(
    success=False,
    workspace_path=Path("/workspaces/alice"),
    violations=[
        {
            "type": "isolation_breach",
            "severity": "high",
            "message": "Cross-workspace dependency detected",
            "file": "/workspaces/alice/config.yaml",
            "line": 15,
            "recommendation": "Remove reference to ../bob/shared_config.yaml"
        },
        {
            "type": "configuration_error",
            "severity": "medium",
            "message": "Invalid pipeline step configuration",
            "file": "/workspaces/alice/pipeline.yaml",
            "line": 42,
            "recommendation": "Update step configuration to match schema"
        }
    ],
    isolation_score=0.75
)

print(f"Workspace validation:")
print(f"  Success: {workspace_result.success}")
print(f"  Violations: {workspace_result.violation_count}")
print(f"  Isolation score: {workspace_result.isolation_score}")

# Create alignment test result
alignment_result = AlignmentTestResult(
    success=True,
    workspace_path=Path("/workspaces/alice"),
    alignment_score=0.92,
    failed_checks=["level3_dependency_check"],
    level_results={
        "level1": {
            "score": 0.98,
            "checks_passed": 10,
            "checks_failed": 0,
            "details": "All script contracts validated successfully"
        },
        "level2": {
            "score": 0.95,
            "checks_passed": 18,
            "checks_failed": 1,
            "details": "Minor specification alignment issue"
        },
        "level3": {
            "score": 0.83,
            "checks_passed": 15,
            "checks_failed": 3,
            "details": "Dependency validation issues detected"
        }
    }
)

print(f"\nAlignment test:")
print(f"  Score: {alignment_result.alignment_score}")
print(f"  Failed checks: {alignment_result.failed_check_count}")

# Create builder test result
builder_result = BuilderTestResult(
    success=True,
    workspace_path=Path("/workspaces/alice"),
    test_results={
        "data_prep_builder": {
            "success": True,
            "duration": 1.2,
            "tests_run": 15,
            "tests_passed": 15
        },
        "training_builder": {
            "success": True,
            "duration": 2.1,
            "tests_run": 22,
            "tests_passed": 22
        },
        "evaluation_builder": {
            "success": False,
            "duration": 0.8,
            "tests_run": 10,
            "tests_passed": 8,
            "error": "Configuration validation failed"
        }
    },
    total_builders=3,
    successful_tests=2,
    failed_tests=1
)

# Update counts based on test results
builder_result.update_counts()

print(f"\nBuilder test:")
print(f"  Success rate: {builder_result.success_rate:.2f}")
print(f"  Has failures: {builder_result.has_test_failures}")

# Create isolation test result
isolation_result = IsolationTestResult(
    success=True,
    workspace_path=Path("/workspaces/alice"),
    isolation_violations=[],
    boundary_checks={
        "file_system_isolation": True,
        "python_path_isolation": True,
        "config_isolation": True,
        "dependency_isolation": True,
        "output_isolation": True
    },
    recommendations=[]
)

print(f"\nIsolation test:")
print(f"  Is isolated: {isolation_result.is_isolated}")
print(f"  Boundary checks passed: {sum(isolation_result.boundary_checks.values())}/{len(isolation_result.boundary_checks)}")
```

### Result Analysis and Reporting

```python
def analyze_validation_results(result: UnifiedValidationResult):
    """Analyze validation results and generate insights."""
    
    print("=== Validation Analysis Report ===")
    print(f"Workspace Root: {result.workspace_root}")
    print(f"Validation Type: {result.workspace_type}")
    print(f"Duration: {result.get_validation_duration():.2f} seconds")
    print()
    
    # Overall summary
    print("Overall Summary:")
    print(f"  Total Workspaces: {result.summary.total_workspaces}")
    print(f"  Success Rate: {result.summary.success_percentage:.1f}%")
    print(f"  Status: {'✓ PASS' if result.overall_success else '✗ FAIL'}")
    print()
    
    # Workspace details
    print("Workspace Details:")
    for workspace_id, workspace_result in result.workspaces.items():
        status = "✓ PASS" if workspace_result.success else "✗ FAIL"
        print(f"  {workspace_id}: {status}")
        
        if workspace_result.has_alignment_results:
            score = workspace_result.alignment_results.get('score', 'N/A')
            print(f"    Alignment Score: {score}")
        
        if workspace_result.has_builder_results:
            builders = workspace_result.builder_results
            total = builders.get('total_builders', 0)
            successful = builders.get('successful', 0)
            print(f"    Builder Tests: {successful}/{total}")
        
        if workspace_result.warnings:
            print(f"    Warnings: {len(workspace_result.warnings)}")
            for warning in workspace_result.warnings[:3]:  # Show first 3
                print(f"      - {warning}")
        
        if workspace_result.error:
            print(f"    Error: {workspace_result.error}")
        
        print()
    
    # Failed workspaces analysis
    failed_workspaces = result.get_failed_workspaces()
    if failed_workspaces:
        print("Failed Workspaces Analysis:")
        for workspace_id in failed_workspaces:
            workspace_result = result.workspaces[workspace_id]
            print(f"  {workspace_id}:")
            print(f"    Error: {workspace_result.error}")
            
            if workspace_result.has_alignment_results:
                alignment = workspace_result.alignment_results
                failed_checks = alignment.get('checks_failed', 0)
                if failed_checks > 0:
                    print(f"    Failed Alignment Checks: {failed_checks}")
            
            if workspace_result.warnings:
                print(f"    Warnings: {len(workspace_result.warnings)}")
        print()
    
    # Recommendations
    if result.recommendations:
        print("Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
        print()
    
    # Global issues
    if result.global_error:
        print(f"Global Error: {result.global_error}")
        print()

# Example usage
validation_result = create_multi_workspace_validation()
analyze_validation_results(validation_result)
```

### Converting Between Result Types

```python
def convert_legacy_result_to_unified(legacy_result: dict) -> UnifiedValidationResult:
    """Convert legacy validation result to unified format."""
    
    builder = ValidationResultBuilder(
        workspace_root=legacy_result.get('workspace_root', '/unknown'),
        workspace_type=legacy_result.get('type', 'unknown'),
        start_time=datetime.fromisoformat(legacy_result.get('start_time', datetime.now().isoformat()))
    )
    
    # Convert workspace results
    for workspace_id, workspace_data in legacy_result.get('workspaces', {}).items():
        builder.add_workspace_result(
            workspace_id=workspace_id,
            workspace_type=workspace_data.get('type', 'unknown'),
            workspace_path=workspace_data.get('path', ''),
            success=workspace_data.get('success', False),
            alignment_results=workspace_data.get('alignment'),
            builder_results=workspace_data.get('builders'),
            error=workspace_data.get('error'),
            warnings=workspace_data.get('warnings', [])
        )
    
    # Add recommendations
    for rec in legacy_result.get('recommendations', []):
        builder.add_recommendation(rec)
    
    # Set global error if present
    if 'global_error' in legacy_result:
        builder.set_global_error(legacy_result['global_error'])
    
    return builder.build()

# Example legacy result conversion
legacy_result = {
    'workspace_root': '/workspaces',
    'type': 'multi',
    'start_time': '2024-12-07T10:00:00Z',
    'workspaces': {
        'alice': {
            'type': 'developer',
            'path': '/workspaces/alice',
            'success': True,
            'alignment': {'score': 0.95},
            'warnings': ['Minor issue']
        }
    },
    'recommendations': ['Upgrade configuration']
}

unified_result = convert_legacy_result_to_unified(legacy_result)
print(f"Converted result: {unified_result.overall_success}")
```

## Integration Points

### Validation Framework Integration
The unified result structures integrate with the complete workspace validation framework, providing consistent data models across all validation components.

### Reporting System Integration
Results integrate with unified reporting generators for consistent output formatting and analysis capabilities.

### CLI Integration
Result structures support CLI output formatting and machine-readable export formats for automation and monitoring.

### Legacy System Support
Backward compatibility adapters ensure existing validation systems can migrate to unified result structures without breaking changes.

## Related Documentation

- [Cross Workspace Validator](cross_workspace_validator.md) - Cross-workspace dependency validation using unified results
- [Workspace Test Manager](workspace_test_manager.md) - Test management with unified result reporting
- [Unified Report Generator](unified_report_generator.md) - Report generation from unified result structures
- [Base Validation Result](base_validation_result.md) - Simple re-export module for BaseValidationResult
- [Main Workspace Validation](README.md) - Overview of complete workspace validation system
