# Workspace Validation Examples

This document provides examples of using the workspace-aware validation system.

## Basic Validation

### Alignment Validation

```python
from cursus.validation.workspace.workspace_alignment_tester import WorkspaceUnifiedAlignmentTester

# Initialize alignment tester
tester = WorkspaceUnifiedAlignmentTester(
    workspace_root="/path/to/workspaces"
)

# Switch to specific developer workspace
tester.switch_developer("developer_1")

# Run validation for all levels
result = tester.run_workspace_validation(
    builder_name="xgboost_trainer",
    levels=[1, 2, 3, 4]
)

print(f"Validation passed: {result.passed}")
print(f"Total tests: {result.total_tests}")
print(f"Failed tests: {result.failed_tests}")

if not result.passed:
    for error in result.errors:
        print(f"Error: {error}")
```

### Builder Validation

```python
from cursus.validation.workspace.workspace_builder_test import WorkspaceUniversalStepBuilderTest

# Initialize builder tester
builder_tester = WorkspaceUniversalStepBuilderTest(
    workspace_root="/path/to/workspaces"
)

# Switch to developer workspace
builder_tester.switch_developer("developer_1")

# Test specific builder
test_result = builder_tester.run_workspace_builder_test("custom_trainer")

print(f"Builder test passed: {test_result.passed}")
if not test_result.passed:
    print("Test failures:")
    for error in test_result.errors:
        print(f"  - {error}")
```

## Comprehensive Validation

### Using Workspace Orchestrator

```python
from cursus.validation.workspace.workspace_orchestrator import WorkspaceValidationOrchestrator

# Initialize orchestrator
orchestrator = WorkspaceValidationOrchestrator(
    workspace_root="/path/to/workspaces"
)

# Validate single workspace
workspace_result = orchestrator.validate_workspace(
    developer_id="developer_1",
    components=["xgboost_trainer", "custom_processor"],
    validation_types=["alignment", "builder"],
    levels=[1, 2, 3, 4]
)

print(f"Workspace validation summary:")
print(f"  Developer: {workspace_result.developer_id}")
print(f"  Total components: {len(workspace_result.component_results)}")
print(f"  Passed: {workspace_result.overall_passed}")

# Check individual component results
for component, result in workspace_result.component_results.items():
    print(f"  {component}: {'PASS' if result.passed else 'FAIL'}")
```

### Multi-Workspace Validation

```python
# Validate all workspaces
all_results = orchestrator.validate_all_workspaces(
    validation_types=["alignment", "builder"],
    levels=[1, 2, 3, 4],
    parallel=True,
    max_workers=4
)

print(f"Validated {len(all_results)} workspaces")

# Summary report
passed_workspaces = sum(1 for r in all_results if r.overall_passed)
print(f"Passed: {passed_workspaces}/{len(all_results)}")

# Detailed results
for result in all_results:
    status = "PASS" if result.overall_passed else "FAIL"
    print(f"{result.developer_id}: {status}")
    
    if not result.overall_passed:
        failed_components = [
            comp for comp, res in result.component_results.items() 
            if not res.passed
        ]
        print(f"  Failed components: {failed_components}")
```

## Advanced Validation Scenarios

### Custom Validation Pipeline

```python
def custom_validation_pipeline(orchestrator, developer_id, components):
    """Custom validation pipeline with specific requirements"""
    
    validation_steps = [
        ("Level 1 - Script Contract", [1]),
        ("Level 2 - Contract-Spec", [2]),
        ("Level 3 - Spec Dependencies", [3]),
        ("Level 4 - Builder Config", [4]),
        ("Builder Tests", "builder"),
        ("Full Integration", [1, 2, 3, 4])
    ]
    
    results = {}
    
    for step_name, validation_config in validation_steps:
        print(f"Running {step_name}...")
        
        if isinstance(validation_config, list):
            # Alignment validation
            result = orchestrator.validate_workspace(
                developer_id=developer_id,
                components=components,
                validation_types=["alignment"],
                levels=validation_config
            )
        else:
            # Builder validation
            result = orchestrator.validate_workspace(
                developer_id=developer_id,
                components=components,
                validation_types=["builder"]
            )
        
        results[step_name] = result
        status = "PASS" if result.overall_passed else "FAIL"
        print(f"  {step_name}: {status}")
        
        # Stop on first failure for critical steps
        if not result.overall_passed and step_name.startswith("Level"):
            print(f"Critical step failed: {step_name}")
            break
    
    return results

# Usage
pipeline_results = custom_validation_pipeline(
    orchestrator, 
    "developer_1", 
    ["xgboost_trainer", "data_processor"]
)
```

### Conditional Validation

```python
def conditional_validation(orchestrator, developer_id):
    """Validate based on workspace contents"""
    
    # Discover available components
    workspace_manager = orchestrator.workspace_manager
    workspace_info = workspace_manager.get_workspace_info(developer_id)
    
    available_builders = workspace_info.get('builders', [])
    available_contracts = workspace_info.get('contracts', [])
    
    print(f"Found {len(available_builders)} builders, {len(available_contracts)} contracts")
    
    # Validate only if both builders and contracts exist
    if available_builders and available_contracts:
        # Full validation
        result = orchestrator.validate_workspace(
            developer_id=developer_id,
            components=available_builders[:3],  # Limit to first 3
            validation_types=["alignment", "builder"],
            levels=[1, 2, 3, 4]
        )
    elif available_builders:
        # Builder-only validation
        result = orchestrator.validate_workspace(
            developer_id=developer_id,
            components=available_builders,
            validation_types=["builder"]
        )
    else:
        print("No components to validate")
        return None
    
    return result

# Usage
conditional_result = conditional_validation(orchestrator, "developer_2")
if conditional_result:
    print(f"Conditional validation: {'PASS' if conditional_result.overall_passed else 'FAIL'}")
```

## Error Handling and Debugging

### Detailed Error Analysis

```python
def analyze_validation_errors(validation_result):
    """Analyze and categorize validation errors"""
    
    error_categories = {
        'import_errors': [],
        'alignment_errors': [],
        'builder_errors': [],
        'config_errors': [],
        'other_errors': []
    }
    
    for component, result in validation_result.component_results.items():
        if not result.passed:
            for error in result.errors:
                error_str = str(error).lower()
                
                if 'import' in error_str or 'module' in error_str:
                    error_categories['import_errors'].append((component, error))
                elif 'alignment' in error_str:
                    error_categories['alignment_errors'].append((component, error))
                elif 'builder' in error_str:
                    error_categories['builder_errors'].append((component, error))
                elif 'config' in error_str:
                    error_categories['config_errors'].append((component, error))
                else:
                    error_categories['other_errors'].append((component, error))
    
    # Print categorized errors
    for category, errors in error_categories.items():
        if errors:
            print(f"\n{category.upper()}:")
            for component, error in errors:
                print(f"  {component}: {error}")
    
    return error_categories

# Usage
if not workspace_result.overall_passed:
    error_analysis = analyze_validation_errors(workspace_result)
```

### Retry Logic for Flaky Tests

```python
def robust_validation(orchestrator, developer_id, components, max_retries=3):
    """Validation with retry logic for flaky tests"""
    
    for attempt in range(max_retries):
        print(f"Validation attempt {attempt + 1}/{max_retries}")
        
        result = orchestrator.validate_workspace(
            developer_id=developer_id,
            components=components,
            validation_types=["alignment", "builder"],
            levels=[1, 2, 3, 4]
        )
        
        if result.overall_passed:
            print("Validation passed!")
            return result
        
        # Analyze failures
        failed_components = [
            comp for comp, res in result.component_results.items() 
            if not res.passed
        ]
        
        print(f"Failed components: {failed_components}")
        
        # Retry only failed components on next attempt
        if attempt < max_retries - 1:
            components = failed_components
            print(f"Retrying with {len(components)} components...")
    
    print("Validation failed after all retries")
    return result

# Usage
robust_result = robust_validation(
    orchestrator, 
    "developer_1", 
    ["xgboost_trainer", "data_processor", "model_evaluator"]
)
```

## Performance Optimization

### Parallel Validation

```python
# Enable parallel validation for better performance
fast_results = orchestrator.validate_all_workspaces(
    validation_types=["alignment", "builder"],
    levels=[1, 2, 3, 4],
    parallel=True,
    max_workers=8  # Adjust based on system capabilities
)

print(f"Parallel validation completed for {len(fast_results)} workspaces")
```

### Selective Validation

```python
def selective_validation(orchestrator, developer_id, quick_mode=False):
    """Selective validation based on mode"""
    
    if quick_mode:
        # Quick validation - only critical levels
        result = orchestrator.validate_workspace(
            developer_id=developer_id,
            validation_types=["alignment"],
            levels=[1, 2],  # Only script-contract and contract-spec
            components=None  # Auto-discover
        )
        print("Quick validation completed")
    else:
        # Full validation
        result = orchestrator.validate_workspace(
            developer_id=developer_id,
            validation_types=["alignment", "builder"],
            levels=[1, 2, 3, 4],
            components=None
        )
        print("Full validation completed")
    
    return result

# Usage
quick_result = selective_validation(orchestrator, "developer_1", quick_mode=True)
full_result = selective_validation(orchestrator, "developer_1", quick_mode=False)
```

## Reporting and Monitoring

### Generate Validation Reports

```python
def generate_validation_report(orchestrator, output_file="validation_report.json"):
    """Generate comprehensive validation report"""
    
    # Validate all workspaces
    all_results = orchestrator.validate_all_workspaces(
        validation_types=["alignment", "builder"],
        levels=[1, 2, 3, 4]
    )
    
    # Generate report
    report = orchestrator.generate_validation_report(
        all_results,
        include_details=True,
        include_recommendations=True
    )
    
    # Save to file
    import json
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Validation report saved to {output_file}")
    
    # Print summary
    print(f"\nValidation Summary:")
    print(f"Total workspaces: {report['summary']['total_workspaces']}")
    print(f"Passed workspaces: {report['summary']['passed_workspaces']}")
    print(f"Success rate: {report['summary']['success_rate']:.1%}")
    
    return report

# Usage
report = generate_validation_report(orchestrator)
```

### Continuous Monitoring

```python
import time
from datetime import datetime

def continuous_validation_monitor(orchestrator, interval_minutes=30):
    """Continuously monitor workspace validation status"""
    
    print(f"Starting continuous validation monitor (interval: {interval_minutes} minutes)")
    
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Running validation cycle...")
        
        try:
            results = orchestrator.validate_all_workspaces(
                validation_types=["alignment"],
                levels=[1, 2],  # Quick check
                parallel=True
            )
            
            passed = sum(1 for r in results if r.overall_passed)
            total = len(results)
            
            print(f"Validation cycle complete: {passed}/{total} workspaces passed")
            
            # Alert on failures
            if passed < total:
                failed_workspaces = [r.developer_id for r in results if not r.overall_passed]
                print(f"ALERT: Failed workspaces: {failed_workspaces}")
            
        except Exception as e:
            print(f"Validation cycle failed: {e}")
        
        # Wait for next cycle
        time.sleep(interval_minutes * 60)

# Usage (run in background or separate process)
# continuous_validation_monitor(orchestrator, interval_minutes=15)
```

## Best Practices

### 1. Incremental Validation

```python
# Start with basic validation, then add complexity
levels_progression = [
    [1],        # Script-contract only
    [1, 2],     # Add contract-spec
    [1, 2, 3],  # Add spec dependencies
    [1, 2, 3, 4] # Full validation
]

for i, levels in enumerate(levels_progression, 1):
    print(f"Validation phase {i}: levels {levels}")
    result = orchestrator.validate_workspace(
        developer_id="developer_1",
        validation_types=["alignment"],
        levels=levels
    )
    
    if not result.overall_passed:
        print(f"Failed at phase {i}")
        break
    else:
        print(f"Phase {i} passed")
```

### 2. Component Isolation

```python
# Test components individually before batch testing
components = ["trainer_1", "trainer_2", "processor_1"]

individual_results = {}
for component in components:
    result = orchestrator.validate_workspace(
        developer_id="developer_1",
        components=[component],  # Single component
        validation_types=["alignment", "builder"],
        levels=[1, 2, 3, 4]
    )
    individual_results[component] = result.overall_passed

print("Individual component results:")
for component, passed in individual_results.items():
    print(f"  {component}: {'PASS' if passed else 'FAIL'}")

# Only test passing components together
passing_components = [c for c, p in individual_results.items() if p]
if len(passing_components) > 1:
    batch_result = orchestrator.validate_workspace(
        developer_id="developer_1",
        components=passing_components,
        validation_types=["alignment", "builder"],
        levels=[1, 2, 3, 4]
    )
    print(f"Batch validation: {'PASS' if batch_result.overall_passed else 'FAIL'}")
```

### 3. Environment-Specific Validation

```python
def environment_aware_validation(orchestrator, developer_id, environment="development"):
    """Adjust validation based on environment"""
    
    if environment == "development":
        # Relaxed validation for development
        validation_types = ["builder"]  # Skip alignment for speed
        levels = [1, 2]  # Basic levels only
    elif environment == "staging":
        # Standard validation for staging
        validation_types = ["alignment", "builder"]
        levels = [1, 2, 3]
    elif environment == "production":
        # Strict validation for production
        validation_types = ["alignment", "builder"]
        levels = [1, 2, 3, 4]
    else:
        raise ValueError(f"Unknown environment: {environment}")
    
    result = orchestrator.validate_workspace(
        developer_id=developer_id,
        validation_types=validation_types,
        levels=levels
    )
    
    print(f"Environment '{environment}' validation: {'PASS' if result.overall_passed else 'FAIL'}")
    return result

# Usage
dev_result = environment_aware_validation(orchestrator, "developer_1", "development")
prod_result = environment_aware_validation(orchestrator, "developer_1", "production")
