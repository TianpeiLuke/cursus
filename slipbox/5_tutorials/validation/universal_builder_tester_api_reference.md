---
tags:
  - test
  - validation
  - builders
  - api_reference
  - documentation
keywords:
  - universal step builder tester API
  - step builder validation API
  - builder testing framework API
  - step builder compliance API
  - sagemaker step validation API
  - workspace builder testing API
topics:
  - universal step builder testing API
  - builder validation API reference
  - step builder testing methods
  - workspace-aware builder testing API
language: python
date of note: 2025-09-06
---

# Universal Step Builder Tester API Reference

## Overview

The Universal Step Builder Tester API provides comprehensive validation of step builder implementations across all architectural levels. This reference documents the complete API with practical examples and usage patterns.

## Core API Classes

### UniversalStepBuilderTest

The main test suite for validating step builder implementations.

```python
from cursus.validation.builders.universal_test import UniversalStepBuilderTest

# Initialize with builder class
tester = UniversalStepBuilderTest(
    builder_class=YourStepBuilder,
    config=None,  # Optional config
    spec=None,    # Optional specification
    contract=None,  # Optional contract
    step_name=None,  # Optional step name
    verbose=False,  # Enable detailed output
    enable_scoring=True,  # Enable quality scoring
    enable_structured_reporting=False  # Enable structured reports
)
```

## Core Operations

### run_all_tests()

Runs comprehensive tests across all levels with optional scoring and reporting.

**Signature:**
```python
def run_all_tests(
    self,
    include_scoring: bool = None,
    include_structured_report: bool = None
) -> Union[Dict[str, Dict[str, Any]], Dict[str, Any]]
```

**Parameters:**
- `include_scoring`: Whether to calculate quality scores (overrides instance setting)
- `include_structured_report`: Whether to generate structured report (overrides instance setting)

**Returns:** Test results dictionary (format depends on scoring/reporting settings)

**Example:**
```python
# Basic test execution
results = tester.run_all_tests()

# With scoring enabled
results = tester.run_all_tests(include_scoring=True)

# Check results format
if 'test_results' in results:
    # Enhanced format with scoring/reporting
    test_results = results['test_results']
    scoring = results.get('scoring', {})
    structured_report = results.get('structured_report', {})
    
    # Calculate summary
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results.values() if r.get('passed', False))
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Pass rate: {pass_rate:.1f}%")
    
    if scoring:
        overall_score = scoring.get('overall', {}).get('score', 0)
        print(f"Quality score: {overall_score:.1f}/100")
else:
    # Legacy format (raw test results)
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r.get('passed', False))
    print(f"Pass rate: {(passed_tests/total_tests*100):.1f}%")
```

### run_all_tests_legacy()

Returns raw test results for backward compatibility.

**Signature:**
```python
def run_all_tests_legacy(self) -> Dict[str, Dict[str, Any]]
```

**Returns:** Raw test results dictionary

**Example:**
```python
# Get raw test results without scoring/reporting
raw_results = tester.run_all_tests_legacy()

for test_name, result in raw_results.items():
    status = "✅" if result.get('passed', False) else "❌"
    print(f"{status} {test_name}")
    if not result.get('passed', False):
        print(f"    Error: {result.get('error', 'Unknown error')}")
```

### run_all_tests_with_scoring()

Convenience method to run tests with scoring enabled.

**Signature:**
```python
def run_all_tests_with_scoring(self) -> Dict[str, Any]
```

**Returns:** Dictionary containing test results and scoring data

**Example:**
```python
# Run tests with scoring
results = tester.run_all_tests_with_scoring()

# Access scoring information
scoring = results['scoring']
overall_score = scoring['overall']['score']
overall_rating = scoring['overall']['rating']

print(f"Overall Score: {overall_score:.1f}/100 ({overall_rating})")

# Level-specific scores
levels = scoring['levels']
for level_name, level_data in levels.items():
    score = level_data['score']
    passed = level_data['passed']
    total = level_data['total']
    print(f"{level_name}: {score:.1f}/100 ({passed}/{total} tests)")
```

### run_all_tests_with_full_report()

Convenience method to run tests with both scoring and structured reporting.

**Signature:**
```python
def run_all_tests_with_full_report(self) -> Dict[str, Any]
```

**Returns:** Dictionary containing test results, scoring, and structured report

**Example:**
```python
# Run tests with full reporting
results = tester.run_all_tests_with_full_report()

# Access all components
test_results = results['test_results']
scoring = results['scoring']
structured_report = results['structured_report']

# Builder information
builder_info = structured_report['builder_info']
print(f"Builder: {builder_info['builder_class']}")
print(f"Step Name: {builder_info['builder_name']}")
print(f"SageMaker Type: {builder_info['sagemaker_step_type']}")

# Summary statistics
summary = structured_report['summary']
print(f"Total Tests: {summary['total_tests']}")
print(f"Pass Rate: {summary['pass_rate']:.1f}%")
print(f"Overall Score: {summary.get('overall_score', 'N/A')}")
```

### export_results_to_json()

Exports test results with scoring to JSON format.

**Signature:**
```python
def export_results_to_json(self, output_path: Optional[str] = None) -> str
```

**Parameters:**
- `output_path`: Optional path to save the JSON file

**Returns:** JSON string of the results

**Example:**
```python
# Export to file
json_content = tester.export_results_to_json('builder_test_report.json')

# Export to string only
json_content = tester.export_results_to_json()

# Parse the JSON content
import json
report_data = json.loads(json_content)

# Access report sections
if 'structured_report' in report_data:
    builder_info = report_data['structured_report']['builder_info']
    print(f"Tested builder: {builder_info['builder_class']}")
```

## Class Methods

### test_all_builders_by_type()

Tests all builders for a specific SageMaker step type using registry discovery.

**Signature:**
```python
@classmethod
def test_all_builders_by_type(
    cls,
    sagemaker_step_type: str,
    verbose: bool = False,
    enable_scoring: bool = True
) -> Dict[str, Any]
```

**Parameters:**
- `sagemaker_step_type`: The SageMaker step type to test (e.g., 'Training', 'Processing')
- `verbose`: Whether to print verbose output
- `enable_scoring`: Whether to calculate and include quality scores

**Returns:** Dictionary containing test results for all builders of the specified type

**Example:**
```python
# Test all Processing builders
processing_results = UniversalStepBuilderTest.test_all_builders_by_type(
    sagemaker_step_type="Processing",
    verbose=True,
    enable_scoring=True
)

if 'error' not in processing_results:
    print(f"Tested {len(processing_results)} Processing builders")
    
    for step_name, result in processing_results.items():
        if 'error' in result:
            print(f"❌ {step_name}: {result['error']}")
        else:
            if 'scoring' in result:
                score = result['scoring']['overall']['score']
                rating = result['scoring']['overall']['rating']
                print(f"✅ {step_name}: {score:.1f}/100 ({rating})")
            else:
                # Legacy format
                test_results = result.get('test_results', result)
                passed = sum(1 for r in test_results.values() if r.get('passed', False))
                total = len(test_results)
                print(f"✅ {step_name}: {passed}/{total} tests passed")
else:
    print(f"❌ Batch testing failed: {processing_results['error']}")

# Test all Training builders
training_results = UniversalStepBuilderTest.test_all_builders_by_type(
    sagemaker_step_type="Training",
    enable_scoring=True
)
```

### generate_registry_discovery_report()

Generates a comprehensive report of step builder discovery status.

**Signature:**
```python
@classmethod
def generate_registry_discovery_report(cls) -> Dict[str, Any]
```

**Returns:** Dictionary containing discovery report

**Example:**
```python
# Generate discovery report
discovery_report = UniversalStepBuilderTest.generate_registry_discovery_report()

if 'error' not in discovery_report:
    print("📊 Registry Discovery Report:")
    print(f"  Total step types: {discovery_report.get('total_step_types', 0)}")
    print(f"  Available builders: {discovery_report.get('available_builders', 0)}")
    print(f"  Missing builders: {discovery_report.get('missing_builders', 0)}")
    
    # Step type coverage
    coverage = discovery_report.get('step_type_coverage', {})
    print(f"\n🔍 Step Type Coverage:")
    for step_type, info in coverage.items():
        status = "✅" if info.get('builder_available', False) else "❌"
        builder_class = info.get('builder_class', 'Not Available')
        print(f"  {status} {step_type}: {builder_class}")
    
    # Missing builders
    missing = discovery_report.get('missing_builders_list', [])
    if missing:
        print(f"\n⚠️ Missing Builders:")
        for step_type in missing:
            print(f"  • {step_type}")
else:
    print(f"❌ Discovery failed: {discovery_report['error']}")
```

### validate_builder_availability()

Validates that a step builder is available and can be loaded.

**Signature:**
```python
@classmethod
def validate_builder_availability(cls, step_name: str) -> Dict[str, Any]
```

**Parameters:**
- `step_name`: The step name to validate

**Returns:** Dictionary containing validation results

**Example:**
```python
# Validate specific builders
builders_to_check = ["tabular_preprocessing", "xgboost_training", "model_evaluation"]

for step_name in builders_to_check:
    validation = UniversalStepBuilderTest.validate_builder_availability(step_name)
    
    if validation.get('available', False):
        builder_class = validation.get('builder_class', 'Unknown')
        module_path = validation.get('module_path', 'Unknown')
        print(f"✅ {step_name}: Available")
        print(f"    Class: {builder_class}")
        print(f"    Module: {module_path}")
    else:
        reason = validation.get('reason', 'Unknown reason')
        print(f"❌ {step_name}: Not available - {reason}")
        
        # Additional error details
        if 'error_details' in validation:
            print(f"    Details: {validation['error_details']}")
```

## Workspace-Aware API

### WorkspaceUniversalStepBuilderTest

Workspace-aware version of UniversalStepBuilderTest for multi-developer environments.

```python
from cursus.workspace.validation.workspace_builder_test import WorkspaceUniversalStepBuilderTest

# Initialize workspace-aware tester
workspace_tester = WorkspaceUniversalStepBuilderTest(
    workspace_root="development/projects",
    developer_id="your_developer_id",
    builder_file_path="src/cursus_dev/steps/builders/builder_custom_step.py",
    enable_shared_fallback=True
)
```

### run_workspace_builder_test()

Runs builder test with workspace-specific context.

**Signature:**
```python
def run_workspace_builder_test(
    self,
    test_config: Optional[Dict[str, Any]] = None,
    workspace_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `test_config`: Test configuration parameters
- `workspace_context`: Additional workspace context for testing

**Returns:** Comprehensive test results with workspace context

**Example:**
```python
# Run workspace builder test
results = workspace_tester.run_workspace_builder_test()

if results['success']:
    print("✅ Workspace builder test completed successfully")
    
    # Workspace metadata
    metadata = results['workspace_metadata']
    print(f"Developer: {metadata['developer_id']}")
    print(f"Builder Class: {metadata['builder_class_name']}")
    print(f"Shared Fallback: {metadata['enable_shared_fallback']}")
    
    # Workspace statistics
    stats = results['workspace_statistics']
    print(f"Builder loaded from workspace: {stats['builder_loaded_from_workspace']}")
    print(f"Shared fallback used: {stats['shared_fallback_used']}")
    
    # Component availability
    components = stats['workspace_components_available']
    for comp_type, available in components.items():
        status = "✅" if available else "❌"
        print(f"{status} {comp_type}")
    
    # Workspace validation
    if 'workspace_validation' in results:
        validation = results['workspace_validation']
        print(f"Builder class valid: {'✅' if validation['builder_class_valid'] else '❌'}")
        
        # Integration issues
        issues = validation.get('integration_issues', [])
        if issues:
            print(f"Integration issues ({len(issues)}):")
            for issue in issues:
                print(f"  • {issue['type']}: {issue['description']}")
        
        # Recommendations
        recommendations = validation.get('recommendations', [])
        for rec in recommendations:
            print(f"💡 {rec}")
else:
    print(f"❌ Workspace builder test failed: {results.get('error')}")
```

### get_workspace_info()

Gets information about current workspace configuration.

**Signature:**
```python
def get_workspace_info(self) -> Dict[str, Any]
```

**Returns:** Dictionary with workspace configuration details

**Example:**
```python
# Get workspace information
workspace_info = workspace_tester.get_workspace_info()

print("🏢 Workspace Configuration:")
print(f"  Developer ID: {workspace_info['developer_id']}")
print(f"  Workspace Root: {workspace_info['workspace_root']}")
print(f"  Builder File: {workspace_info['builder_file_path']}")
print(f"  Shared Fallback: {workspace_info['enable_shared_fallback']}")
print(f"  Builder Class: {workspace_info['builder_class_name']}")

# Available developers
available_devs = workspace_info['available_developers']
print(f"  Available Developers: {available_devs}")

# Workspace manager info
manager_info = workspace_info['workspace_manager_info']
print(f"  Total Workspaces: {len(manager_info.get('developers', {}))}")
```

### switch_developer()

Switches to a different developer workspace.

**Signature:**
```python
def switch_developer(
    self,
    developer_id: str,
    builder_file_path: Optional[str] = None
) -> None
```

**Parameters:**
- `developer_id`: Target developer workspace ID
- `builder_file_path`: Optional new builder file path

**Example:**
```python
# Switch to different developer workspace
try:
    workspace_tester.switch_developer(
        "alice_developer",
        "src/cursus_dev/steps/builders/builder_alice_custom_step.py"
    )
    print("✅ Switched to Alice's workspace")
    
    # Run test in new workspace
    results = workspace_tester.run_workspace_builder_test()
    success = "✅" if results['success'] else "❌"
    print(f"{success} Alice's builder test: {results.get('success', False)}")
    
except ValueError as e:
    print(f"❌ Failed to switch workspace: {e}")
    
    # List available developers
    workspace_info = workspace_tester.get_workspace_info()
    available = workspace_info['available_developers']
    print(f"Available developers: {available}")
```

### test_all_workspace_builders()

Discovers and tests all builders in a workspace.

**Signature:**
```python
@classmethod
def test_all_workspace_builders(
    cls,
    workspace_root: Union[str, Path],
    developer_id: str,
    test_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**
- `workspace_root`: Root directory containing developer workspaces
- `developer_id`: Specific developer workspace to target
- `test_config`: Test configuration parameters
- `**kwargs`: Additional arguments passed to individual tests

**Returns:** Comprehensive test results for all workspace builders

**Example:**
```python
# Test all builders in a workspace
workspace_results = WorkspaceUniversalStepBuilderTest.test_all_workspace_builders(
    workspace_root="development/projects",
    developer_id="your_developer_id",
    test_config={'enable_scoring': True}
)

if workspace_results['success']:
    print(f"✅ Tested {workspace_results['tested_builders']} workspace builders")
    print(f"Success rate: {workspace_results['successful_tests']}/{workspace_results['tested_builders']}")
    
    # Individual results
    results = workspace_results['results']
    for builder_name, result in results.items():
        if result.get('success', False):
            print(f"  ✅ {builder_name}: Passed")
        else:
            print(f"  ❌ {builder_name}: {result.get('error', 'Failed')}")
    
    # Summary analysis
    summary = workspace_results.get('summary', {})
    if summary:
        success_rate = summary['overall_success_rate']
        print(f"Overall success rate: {success_rate:.1%}")
        
        # Common issues
        common_issues = summary.get('common_issues', [])
        if common_issues:
            print("Common issues:")
            for issue in common_issues:
                print(f"  • {issue['type']}: {issue['count']} builders")
        
        # Recommendations
        recommendations = summary.get('recommendations', [])
        for rec in recommendations:
            print(f"💡 {rec}")
else:
    print(f"❌ Workspace testing failed: {workspace_results.get('error')}")
```

## Data Models

### Test Result Structure

Individual test results follow this structure:

```python
{
    "passed": bool,           # Whether the test passed
    "error": str,            # Error message if failed (optional)
    "details": Dict[str, Any] # Additional test details (optional)
}
```

### Enhanced Results Structure

When scoring/reporting is enabled:

```python
{
    "test_results": Dict[str, Dict[str, Any]],  # Raw test results
    "scoring": {                                # Quality scoring (optional)
        "overall": {
            "score": float,      # Overall score (0-100)
            "rating": str,       # Rating (Excellent, Good, Fair, Poor)
            "total_tests": int,  # Total number of tests
            "passed_tests": int  # Number of passed tests
        },
        "levels": {
            "level1_interface": {
                "score": float,
                "passed": int,
                "total": int,
                "weight": float
            },
            # ... other levels
        }
    },
    "structured_report": {                      # Structured report (optional)
        "builder_info": {
            "builder_name": str,
            "builder_class": str,
            "sagemaker_step_type": str
        },
        "test_results": {
            "level1_interface": Dict[str, Any],
            "level2_specification": Dict[str, Any],
            "level3_step_creation": Dict[str, Any],
            "level4_integration": Dict[str, Any],
            "step_type_specific": Dict[str, Any]
        },
        "summary": {
            "total_tests": int,
            "passed_tests": int,
            "pass_rate": float,
            "overall_score": float,    # If scoring enabled
            "score_rating": str        # If scoring enabled
        }
    }
}
```

### Workspace Results Structure

Workspace-aware test results:

```python
{
    "success": bool,
    "workspace_metadata": {
        "developer_id": str,
        "workspace_root": str,
        "builder_file_path": str,
        "enable_shared_fallback": bool,
        "builder_class_name": str,
        "workspace_info": Dict[str, Any]
    },
    "workspace_statistics": {
        "builder_loaded_from_workspace": bool,
        "builder_class_name": str,
        "builder_module_path": str,
        "workspace_components_available": {
            "contracts": bool,
            "specs": bool,
            "configs": bool
        },
        "shared_fallback_used": bool
    },
    "workspace_validation": {
        "builder_class_valid": bool,
        "workspace_dependencies_available": Dict[str, Any],
        "integration_issues": List[Dict[str, Any]],
        "recommendations": List[str]
    }
}
```

## Error Handling

### Common Exceptions

**ImportError**
```python
try:
    from cursus.steps.builders.builder_nonexistent_step import NonexistentStepBuilder
    tester = UniversalStepBuilderTest(NonexistentStepBuilder)
except ImportError as e:
    print(f"Builder import failed: {e}")
    print("💡 Check if the builder module exists")
    print("💡 Verify the builder class name is correct")
```

**ValidationError**
```python
from cursus.validation.builders.universal_test import UniversalStepBuilderTest

try:
    # Invalid builder class (not inheriting from correct base)
    class InvalidBuilder:
        pass
    
    tester = UniversalStepBuilderTest(InvalidBuilder)
    results = tester.run_all_tests()
except Exception as e:
    print(f"Validation failed: {e}")
    print("💡 Ensure builder inherits from StepBuilderBase")
    print("💡 Check builder implementation follows expected patterns")
```

**WorkspaceError**
```python
from cursus.workspace.validation.workspace_builder_test import WorkspaceUniversalStepBuilderTest

try:
    workspace_tester = WorkspaceUniversalStepBuilderTest(
        workspace_root="nonexistent/workspace",
        developer_id="unknown_developer",
        builder_file_path="invalid/path.py"
    )
except Exception as e:
    print(f"Workspace setup failed: {e}")
    print("💡 Check if workspace directory exists")
    print("💡 Verify developer workspace is initialized")
    print("💡 Ensure builder file path is correct")
```

### Error Handling Best Practices

```python
def robust_builder_testing(builder_class):
    """Example of robust builder testing with error handling."""
    
    try:
        # Initialize tester
        tester = UniversalStepBuilderTest(
            builder_class=builder_class,
            enable_scoring=True,
            verbose=True
        )
        
        # Run tests with error handling
        results = tester.run_all_tests()
        
        # Check results format
        if 'test_results' in results:
            test_results = results['test_results']
            scoring = results.get('scoring', {})
        else:
            test_results = results
            scoring = {}
        
        # Calculate summary
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results.values() if r.get('passed', False))
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        if pass_rate >= 90:
            print(f"✅ Builder testing passed: {pass_rate:.1f}%")
            
            if scoring:
                score = scoring.get('overall', {}).get('score', 0)
                print(f"Quality score: {score:.1f}/100")
            
            return True
        else:
            print(f"⚠️ Builder testing issues: {pass_rate:.1f}% pass rate")
            
            # Show failed tests
            failed_tests = [name for name, result in test_results.items() 
                          if not result.get('passed', False)]
            
            print("Failed tests:")
            for test_name in failed_tests[:5]:  # Show first 5
                error = test_results[test_name].get('error', 'Unknown error')
                print(f"  • {test_name}: {error}")
            
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Check if builder class can be imported")
        return False
    except Exception as e:
        print(f"❌ Testing error: {e}")
        print("💡 Check builder implementation and dependencies")
        return False

# Use robust testing
from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
success = robust_builder_testing(TabularPreprocessingStepBuilder)
```

## Advanced Usage

### Custom Test Configuration

```python
# Test with custom configuration
from types import SimpleNamespace

# Create custom config
custom_config = SimpleNamespace()
custom_config.region = 'NA'
custom_config.pipeline_name = 'custom-pipeline'
custom_config.job_type = 'training'

# Test with explicit components
from cursus.steps.specs.tabular_preprocessing_training_spec import TABULAR_PREPROCESSING_TRAINING_SPEC

tester = UniversalStepBuilderTest(
    builder_class=TabularPreprocessingStepBuilder,
    config=custom_config,
    spec=TABULAR_PREPROCESSING_TRAINING_SPEC,
    step_name='CustomPreprocessingStep',
    enable_scoring=True,
    enable_structured_reporting=True
)

results = tester.run_all_tests()
```

### Batch Testing with Analysis

```python
def comprehensive_builder_analysis():
    """Comprehensive analysis of all builders."""
    
    # Test all step types
    step_types = ["Processing", "Training", "Transform", "CreateModel", "RegisterModel"]
    all_results = {}
    
    for step_type in step_types:
        print(f"\n🔍 Testing {step_type} builders...")
        
        type_results = UniversalStepBuilderTest.test_all_builders_by_type(
            sagemaker_step_type=step_type,
            enable_scoring=True
        )
        
        if 'error' not in type_results:
            all_results[step_type] = type_results
            print(f"  ✅ Tested {len(type_results)} {step_type} builders")
        else:
            print(f"  ❌ {step_type} testing failed: {type_results['error']}")
    
    # Analyze results across all types
    total_builders = sum(len(results) for results in all_results.values())
    successful_builders = 0
    high_quality_builders = 0
    
    print(f"\n📊 Comprehensive Analysis:")
    print(f"Total builders tested: {total_builders}")
    
    for step_type, type_results in all_results.items():
        type_successful = 0
        type_high_quality = 0
        
        for step_name, result in type_results.items():
            if 'error' not in result:
                # Check test success
                if 'test_results' in result:
                    test_results = result['test_results']
                    scoring = result.get('scoring', {})
                else:
                    test_results = result
                    scoring = {}
                
                passed = sum(1 for r in test_results.values() if r.get('passed', False))
                total = len(test_results)
                pass_rate = (passed / total * 100) if total > 0 else 0
                
                if pass_rate >= 80:
                    successful_builders += 1
                    type_successful += 1
                
                if scoring:
                    score = scoring.get('overall', {}).get('score', 0)
                    if score >= 85:
                        high_quality_builders += 1
                        type_high_quality += 1
        
        print(f"  {step_type}: {type_successful}/{len(type_results)} successful")
        if type_high_quality > 0:
            print(f"    High quality: {type_high_quality}/{len(type_results)}")
    
    # Overall statistics
    success_rate = (successful_builders / total_builders * 100) if total_builders > 0 else 0
    quality_rate = (high_quality_builders / total_builders * 100) if total_builders > 0 else 0
    
    print(f"\n📈 Overall Statistics:")
    print(f"Success rate: {success_rate:.1f}% ({successful_builders}/{total_builders})")
    print(f"High quality rate: {quality_rate:.1f}% ({high_quality_builders}/{total_builders})")
    
    return all_results

# Run comprehensive analysis
analysis_results = comprehensive_builder_analysis()
```

### Integration with CI/CD

```python
def ci_cd_builder_validation():
    """Builder validation suitable for CI/CD pipelines."""
    
    import sys
    import os
    
    # Get builders to test from environment or default list
    builders_to_test = os.getenv('BUILDERS_TO_TEST', 'Processing,Training').split(',')
    
    overall_success = True
    results_summary = {}
    
    for step_type in builders_to_test:
        print(f"=== Testing {step_type} Builders ===")
        
        type_results = UniversalStepBuilderTest.test_all_builders_by_type(
            sagemaker_step_type=step_type.strip(),
            enable_scoring=True
        )
        
        if 'error' not in type_results:
            type_success = True
            type_summary = {
                'total': len(type_results),
                'passed': 0,
                'failed': 0,
                'scores': []
            }
            
            for step_name, result in type_results.items():
                if 'error' in result:
                    print(f"❌ {step_name}: {result['error']}")
                    type_summary['failed'] += 1
                    type_success = False
                else:
                    # Check test results
                    if 'test_results' in result:
                        test_results = result['test_results']
                        scoring = result.get('scoring', {})
                    else:
                        test_results = result
                        scoring = {}
                    
                    passed = sum(1 for r in test_results.values() if r.get('passed', False))
                    total = len(test_results)
                    pass_rate = (passed / total * 100) if total > 0 else 0
                    
                    if pass_rate >= 80:  # Require 80% pass rate for CI/CD
                        print(f"✅ {step_name}: {pass_rate:.1f}% pass rate")
                        type_summary['passed'] += 1
                        
                        if scoring:
                            score = scoring.get('overall', {}).get('score', 0)
                            type_summary['scores'].append(score)
                            print(f"    Quality Score: {score:.1f}/100")
                    else:
                        print(f"❌ {step_name}: {pass_rate:.1f}% pass rate (below 80% threshold)")
                        type_summary['failed'] += 1
                        type_success = False
            
            results_summary[step_type] = type_summary
            
            if not type_success:
                overall_success = False
                
        else:
            print(f"❌ {step_type} testing failed: {type_results['error']}")
            overall_success = False
    
    # Print final summary
    print(f"\n=== CI/CD BUILDER VALIDATION SUMMARY ===")
    for step_type, summary in results_summary.items():
        print(f"{step_type}: {summary['passed']}/{summary['total']} builders passed")
        if summary['scores']:
            avg_score = sum(summary['scores']) / len(summary['scores'])
            print(f"  Average Quality Score: {avg_score:.1f}/100")
    
    # Set exit code
    if overall_success:
        print("✅ CI/CD builder validation PASSED")
        return 0
    else:
        print("❌ CI/CD builder validation FAILED")
        return 1

# Use in CI/CD pipeline
if __name__ == "__main__":
    exit_code = ci_cd_builder_validation()
    sys.exit(exit_code)
```

## Performance Considerations

### Optimizing Test Performance

```python
def optimized_builder_testing():
    """Optimized testing for large numbers of builders."""
    
    # Test builders in parallel by type
    import concurrent.futures
    from threading import Lock
    
    step_types = ["Processing", "Training", "Transform"]
    results_lock = Lock()
    all_results = {}
    
    def test_step_type(step_type):
        """Test all builders of a specific type."""
        try:
            type_results = UniversalStepBuilderTest.test_all_builders_by_type(
                sagemaker_step_type=step_type,
                verbose=False,  # Reduce output for parallel execution
                enable_scoring=True
            )
            
            with results_lock:
                all_results[step_type] = type_results
                
            return step_type, len(type_results) if 'error' not in type_results else 0
            
        except Exception as e:
            with results_lock:
                all_results[step_type] = {'error': str(e)}
            return step_type, 0
    
    # Execute tests in parallel
    print("🚀 Running optimized parallel builder testing...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(test_step_type, step_type) for step_type in step_types]
        
        for future in concurrent.futures.as_completed(futures):
            step_type, count = future.result()
            print(f"✅ {step_type}: {count} builders tested")
    
    # Analyze combined results
    total_builders = 0
    successful_builders = 0
    
    for step_type, type_results in all_results.items():
        if 'error' not in type_results:
            total_builders += len(type_results)
            
            for step_name, result in type_results.items():
                if 'error' not in result:
                    if 'test_results' in result:
                        test_results = result['test_results']
                    else:
                        test_results = result
                    
                    passed = sum(1 for r in test_results.values() if r.get('passed', False))
                    total = len(test_results)
                    pass_rate = (passed / total * 100) if total > 0 else 0
                    
                    if pass_rate >= 80:
                        successful_builders += 1
    
    success_rate = (successful_builders / total_builders * 100) if total_builders > 0 else 0
    print(f"\n📊 Optimized Testing Results:")
    print(f"Total builders: {total_builders}")
    print(f"Success rate: {success_rate:.1f}% ({successful_builders}/{total_builders})")
    
    return all_results

# Run optimized testing
optimized_results = optimized_builder_testing()
```

## API Reference Summary

### Core Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `run_all_tests()` | Comprehensive testing with optional scoring/reporting | `Union[Dict, Dict]` |
| `run_all_tests_legacy()` | Raw test results for backward compatibility | `Dict[str, Dict[str, Any]]` |
| `run_all_tests_with_scoring()` | Tests with scoring enabled | `Dict[str, Any]` |
| `run_all_tests_with_full_report()` | Tests with scoring and structured reporting | `Dict[str, Any]` |
| `export_results_to_json()` | Export results to JSON format | `str` |

### Class Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `test_all_builders_by_type()` | Test all builders of specific SageMaker type | `Dict[str, Any]` |
| `generate_registry_discovery_report()` | Generate builder discovery report | `Dict[str, Any]` |
| `validate_builder_availability()` | Validate specific builder availability | `Dict[str, Any]` |

### Workspace-Aware Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `run_workspace_builder_test()` | Workspace-aware builder testing | `Dict[str, Any]` |
| `get_workspace_info()` | Workspace configuration info | `Dict[str, Any]` |
| `switch_developer()` | Switch developer workspace | `None` |
| `test_all_workspace_builders()` | Test all builders in workspace | `Dict[str, Any]` |

### Scoring Levels

The scoring system evaluates builders across multiple levels:

- **Level 1 (Interface)**: Weight 1.0x - Basic inheritance and method compliance
- **Level 2 (Specification)**: Weight 1.2x - Contract and specification alignment  
- **Level 3 (Step Creation)**: Weight 1.5x - Step creation and configuration capabilities
- **Level 4 (Integration)**: Weight 1.3x - End-to-end integration validation
- **SageMaker Type Specific**: Weight 1.1x - Framework-specific compliance

### Quality Ratings

- **Excellent**: 90-100 points
- **Good**: 75-89 points  
- **Fair**: 60-74 points
- **Poor**: Below 60 points

## Best Practices

1. **Use Scoring**: Enable scoring to track quality improvements over time
2. **Structured Reports**: Use structured reporting for detailed analysis
3. **Batch Testing**: Use class methods for testing multiple builders efficiently
4. **Workspace Isolation**: Use workspace-aware testing for multi-developer projects
5. **Error Handling**: Always implement robust error handling in production code
6. **Performance**: Use optimized testing patterns for large codebases
7. **CI/CD Integration**: Implement automated testing in development pipelines

For additional examples and usage patterns, see the [Universal Builder Tester Quick Start Guide](universal_builder_tester_quick_start.md).
