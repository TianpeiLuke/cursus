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
  - streamlined builder testing API
  - alignment integration API
  - builder testing framework API
topics:
  - universal step builder testing API
  - builder validation API reference
  - streamlined builder testing methods
  - alignment system integration API
language: python
date of note: 2025-10-05
---

# Universal Step Builder Tester API Reference

## Overview

The Universal Step Builder Tester API provides streamlined validation of step builder implementations through alignment system integration. This reference documents the complete API based on the actual refactored implementation that eliminates 60-70% redundancy while preserving unique builder testing capabilities.

## Core API Classes

### UniversalStepBuilderTest

The main test suite for validating step builder implementations using the streamlined approach.

```python
from cursus.validation.builders import UniversalStepBuilderTest

# Initialize with workspace-aware discovery (recommended)
tester = UniversalStepBuilderTest(
    workspace_dirs=["development/projects/project_alpha"],  # Optional workspace directories
    verbose=False,  # Enable detailed output
    enable_scoring=True,  # Enable quality scoring
    enable_structured_reporting=False  # Enable structured reports
)

# Initialize with legacy single-builder mode (backward compatibility)
tester = UniversalStepBuilderTest.from_builder_class(
    YourStepBuilder,
    workspace_dirs=None,  # Optional workspace directories
    verbose=False,
    enable_scoring=True
)
```

**Constructor Parameters:**
- `workspace_dirs`: Optional list of workspace directories for step discovery. If None, only discovers package internal steps.
- `verbose`: Enable detailed output during validation
- `enable_scoring`: Enable quality scoring calculations
- `enable_structured_reporting`: Enable structured report generation

## Core Operations

### run_validation_for_step()

Runs comprehensive validation for a specific step using the streamlined approach.

**Signature:**
```python
def run_validation_for_step(self, step_name: str) -> Dict[str, Any]
```

**Parameters:**
- `step_name`: Name of the step to validate

**Returns:** Dictionary containing comprehensive validation results with component breakdown

**Example:**
```python
# Run validation for a specific step
result = tester.run_validation_for_step("tabular_preprocessing")

print(f"Step: {result['step_name']}")
print(f"Builder Class: {result['builder_class']}")
print(f"Overall Status: {result['overall_status']}")

# Check validation components
components = result.get("components", {})
for component_name, component_result in components.items():
    status = component_result.get("status", "UNKNOWN")
    print(f"  {component_name}: {status}")
    
    # Show any errors
    if "error" in component_result:
        print(f"    Error: {component_result['error']}")

# Check scoring if available
if "scoring" in result:
    scoring = result["scoring"]
    overall_score = scoring.get("overall", {}).get("score", 0)
    overall_rating = scoring.get("overall", {}).get("rating", "Unknown")
    print(f"Quality Score: {overall_score:.1f}/100 ({overall_rating})")
```

### run_full_validation()

Runs validation for all discovered steps using the streamlined approach.

**Signature:**
```python
def run_full_validation(self) -> Dict[str, Any]
```

**Returns:** Dictionary containing validation results for all discovered steps

**Example:**
```python
# Run comprehensive validation for all steps
full_results = tester.run_full_validation()

print(f"Validation Type: {full_results['validation_type']}")
print(f"Total Steps: {full_results['total_steps']}")

# Check summary statistics
summary = full_results.get("summary", {})
if summary:
    print(f"Passed Steps: {summary['passed_steps']}")
    print(f"Failed Steps: {summary['failed_steps']}")
    print(f"Pass Rate: {summary['pass_rate']:.1f}%")

# Check individual step results
step_results = full_results.get("step_results", {})
for step_name, result in step_results.items():
    overall_status = result.get("overall_status", "UNKNOWN")
    builder_class = result.get("builder_class", "Unknown")
    print(f"  {step_name} ({builder_class}): {overall_status}")
    
    # Show scoring if available
    if "scoring" in result:
        score = result["scoring"].get("overall", {}).get("score", 0)
        print(f"    Quality Score: {score:.1f}/100")
```

### run_all_tests()

Runs all tests with optional scoring and structured reporting (enhanced method).

**Signature:**
```python
def run_all_tests(
    self,
    include_scoring: bool = None,
    include_structured_report: bool = None
) -> Dict[str, Any]
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

# With full reporting
results = tester.run_all_tests(
    include_scoring=True,
    include_structured_report=True
)

# Check results format
if 'test_results' in results:
    # Enhanced format with scoring/reporting
    test_results = results['test_results']
    scoring = results.get('scoring', {})
    structured_report = results.get('structured_report', {})
    
    print(f"Enhanced results with scoring: {scoring is not None}")
    print(f"Structured report available: {structured_report is not None}")
else:
    # Legacy format (raw test results)
    print(f"Legacy format: {len(results)} test results")
```

### run_all_tests_legacy()

Returns raw test results for backward compatibility.

**Signature:**
```python
def run_all_tests_legacy(self) -> Dict[str, Dict[str, Any]]
```

**Returns:** Raw test results dictionary without scoring or structured reporting

**Example:**
```python
# Get raw test results without enhancements
raw_results = tester.run_all_tests_legacy()

for test_name, result in raw_results.items():
    passed = result.get('passed', False)
    status = "✅" if passed else "❌"
    print(f"{status} {test_name}")
    
    if not passed and 'error' in result:
        print(f"    Error: {result['error']}")
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

# Component-specific scores
components = scoring.get('components', {})
for comp_name, comp_data in components.items():
    score = comp_data.get('score', 0)
    weight = comp_data.get('weight', 1.0)
    print(f"{comp_name}: {score:.1f}/100 (weight: {weight})")
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

# Builder information from structured report
builder_info = structured_report['builder_info']
print(f"Builder: {builder_info['builder_class']}")
print(f"Step Name: {builder_info['builder_name']}")
print(f"SageMaker Type: {builder_info['sagemaker_step_type']}")

# Summary statistics
summary = structured_report['summary']
print(f"Total Tests: {summary['total_tests']}")
print(f"Pass Rate: {summary['pass_rate']:.1f}%")
if 'overall_score' in summary:
    print(f"Overall Score: {summary['overall_score']:.1f}/100")
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
print("✅ Results exported to builder_test_report.json")

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

Tests all builders for a specific SageMaker step type using the streamlined approach.

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
            overall_status = result.get("overall_status", "UNKNOWN")
            print(f"✅ {step_name}: {overall_status}")
            
            # Show scoring if available
            if "scoring" in result:
                score = result["scoring"].get("overall", {}).get("score", 0)
                rating = result["scoring"].get("overall", {}).get("rating", "Unknown")
                print(f"    Quality: {score:.1f}/100 ({rating})")
else:
    print(f"❌ Batch testing failed: {processing_results['error']}")
```

### from_builder_class()

Creates a tester instance for a specific builder class (backward compatibility).

**Signature:**
```python
@classmethod
def from_builder_class(
    cls,
    builder_class: Type,
    workspace_dirs: Optional[List[str]] = None,
    **kwargs
) -> 'UniversalStepBuilderTest'
```

**Parameters:**
- `builder_class`: The step builder class to test
- `workspace_dirs`: Optional workspace directories
- `**kwargs`: Additional configuration options

**Returns:** UniversalStepBuilderTest instance configured for single builder testing

**Example:**
```python
from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder

# Create tester for specific builder class
tester = UniversalStepBuilderTest.from_builder_class(
    TabularPreprocessingStepBuilder,
    workspace_dirs=["development/projects"],
    verbose=True,
    enable_scoring=True
)

print(f"Testing builder: {tester.builder_class.__name__}")
print(f"Single builder mode: {tester.single_builder_mode}")

# Run tests for this specific builder
results = tester.run_all_tests_with_scoring()
```

## Discovery and Utility Methods

### _discover_all_steps()

Discovers all steps using step catalog - consolidated discovery method.

**Signature:**
```python
def _discover_all_steps(self) -> List[str]
```

**Returns:** List of discovered step names

**Example:**
```python
# Discover all available steps
discovered_steps = tester._discover_all_steps()

print(f"📁 Discovered {len(discovered_steps)} steps:")
for step in discovered_steps[:5]:  # Show first 5
    print(f"  • {step}")

if len(discovered_steps) > 5:
    print(f"  ... and {len(discovered_steps) - 5} more")

# Use discovered steps for validation
if discovered_steps:
    sample_step = discovered_steps[0]
    result = tester.run_validation_for_step(sample_step)
    print(f"Sample validation: {result.get('overall_status', 'UNKNOWN')}")
```

### _get_builder_class_from_catalog()

Gets builder class from step catalog or registry.

**Signature:**
```python
def _get_builder_class_from_catalog(self, step_name: str) -> Optional[Type]
```

**Parameters:**
- `step_name`: Name of the step

**Returns:** Builder class or None if not found

**Example:**
```python
# Get builder class for a step
step_name = "tabular_preprocessing"
builder_class = tester._get_builder_class_from_catalog(step_name)

if builder_class:
    print(f"✅ Builder class loaded: {builder_class.__name__}")
    print(f"Module: {builder_class.__module__}")
    
    # Check if it's a valid builder
    from cursus.core.base.builder_base import StepBuilderBase
    is_valid = issubclass(builder_class, StepBuilderBase)
    print(f"Valid builder: {is_valid}")
else:
    print(f"❌ No builder class found for {step_name}")
```

## Reporting and Scoring

### generate_report()

Generates comprehensive report for a step using the streamlined reporting system.

**Signature:**
```python
def generate_report(self, step_name: str) -> Union[StreamlinedBuilderTestReport, Dict[str, Any]]
```

**Parameters:**
- `step_name`: Name of the step to generate report for

**Returns:** StreamlinedBuilderTestReport object or validation results dictionary

**Example:**
```python
# Generate detailed report for a step
step_name = "tabular_preprocessing"
report = tester.generate_report(step_name)

if hasattr(report, 'print_summary'):
    # StreamlinedBuilderTestReport object
    print("📋 Streamlined Report:")
    report.print_summary()
    
    # Export to JSON
    json_content = report.export_to_json()
    with open(f"{step_name}_report.json", "w") as f:
        f.write(json_content)
    print(f"✅ Report exported to {step_name}_report.json")
    
    # Check report properties
    print(f"Overall Status: {report.get_overall_status()}")
    print(f"Quality Score: {report.get_quality_score():.1f}/100")
    print(f"Quality Rating: {report.get_quality_rating()}")
    print(f"Is Passing: {report.is_passing()}")
    
    # Get critical issues
    critical_issues = report.get_critical_issues()
    if critical_issues:
        print(f"Critical Issues: {len(critical_issues)}")
        for issue in critical_issues[:3]:  # Show first 3
            print(f"  • {issue}")
else:
    # Fallback to validation results
    print("📋 Validation Results:")
    overall_status = report.get("overall_status", "UNKNOWN")
    builder_class = report.get("builder_class", "Unknown")
    print(f"Step: {step_name}")
    print(f"Builder: {builder_class}")
    print(f"Status: {overall_status}")
```

## Legacy Compatibility Methods

### validate_specific_script()

Legacy method for validating a specific script (maintained for backward compatibility).

**Signature:**
```python
def validate_specific_script(
    self,
    step_name: str,
    skip_levels: Optional[set] = None
) -> Dict[str, Any]
```

**Parameters:**
- `step_name`: Name of the step to validate
- `skip_levels`: Optional set of validation levels to skip (deprecated - ignored in new system)

**Returns:** Dictionary containing validation results

**Example:**
```python
# Legacy API usage (automatically uses new streamlined system)
result = tester.validate_specific_script("tabular_preprocessing")

print(f"Step: {result['step_name']}")
print(f"Overall Status: {result['overall_status']}")

# Note: skip_levels parameter is deprecated and ignored
result_with_skip = tester.validate_specific_script(
    "tabular_preprocessing", 
    skip_levels={3, 4}  # This is ignored - streamlined system determines validation
)
```

### discover_scripts()

Legacy method for discovering scripts (maintained for backward compatibility).

**Signature:**
```python
def discover_scripts(self) -> List[str]
```

**Returns:** List of discovered script names

**Example:**
```python
# Legacy discovery method (uses new streamlined discovery internally)
scripts = tester.discover_scripts()

print(f"📁 Discovered scripts: {len(scripts)}")
for script in scripts[:5]:  # Show first 5
    print(f"  • {script}")
```

### get_validation_summary()

Legacy method for getting validation summary (enhanced with streamlined metrics).

**Signature:**
```python
def get_validation_summary(self) -> Dict[str, Any]
```

**Returns:** Dictionary containing validation summary

**Example:**
```python
# Get enhanced validation summary
summary = tester.get_validation_summary()

print(f"Total Steps: {summary.get('total_steps', 0)}")
print(f"Passed Steps: {summary.get('passed_steps', 0)}")
print(f"Failed Steps: {summary.get('failed_steps', 0)}")
print(f"Pass Rate: {summary.get('pass_rate', 0):.2f}%")
```

### print_summary()

Legacy method for printing validation summary to console.

**Signature:**
```python
def print_summary(self) -> None
```

**Example:**
```python
# Print enhanced validation summary
tester.print_summary()

# Output example:
# ============================================================
# REFACTORED VALIDATION SUMMARY
# ============================================================
# Total Steps: 21
# Passed: 15
# Failed: 3
# Issues: 2
# Pass Rate: 75.00%
# Refactored: True (60-70% redundancy eliminated)
# ============================================================
```

## Streamlined Validation Components

The streamlined system validates builders through four main components:

### 1. Alignment Validation

Leverages the proven alignment system for core validation (replaces old Levels 1-2).

```python
# Component result structure
{
    "status": "COMPLETED",
    "validation_approach": "alignment_system",
    "results": {
        "overall_status": "PASSED",
        "validation_results": {
            "level_1": {...},
            "level_2": {...}
        }
    },
    "levels_covered": ["interface_compliance", "specification_alignment"]
}
```

### 2. Integration Testing

Preserves unique integration capabilities (unique to builders).

```python
# Component result structure
{
    "status": "COMPLETED",
    "checks": {
        "dependency_resolution": {
            "passed": True,
            "found_methods": ["_get_inputs", "_get_outputs"]
        },
        "cache_configuration": {
            "passed": True,
            "found_methods": []
        },
        "step_instantiation": {
            "passed": True,
            "checks": {
                "config_class_exists": {"passed": True},
                "config_import": {"passed": True},
                "input_output_methods": {"passed": True},
                "sagemaker_methods": {"passed": True}
            }
        }
    },
    "integration_type": "capability_validation"
}
```

### 3. Step Creation Capability

Validates step creation availability (simplified Level 3).

```python
# Component result structure
{
    "status": "COMPLETED",
    "capability_validated": True,
    "checks": {
        "config_availability": {
            "available": True,
            "config_class": "TabularPreprocessingConfig"
        },
        "method_availability": {
            "has_required_methods": True,
            "found_required": ["create_step", "validate_configuration"],
            "missing_required": [],
            "found_optional": ["__init__"]
        },
        "field_requirements": {
            "requirements_identifiable": True,
            "total_fields": 25,
            "essential_fields": ["role", "region", "bucket"]
        }
    },
    "note": "Availability testing - no actual instantiation performed"
}
```

### 4. Step Type Specific Validation

Framework-specific compliance checks based on detected step type.

```python
# Component result structure
{
    "status": "COMPLETED",
    "results": {
        "step_type": "Processing",
        "step_type_tests": {
            "processor_methods": {
                "passed": True,
                "found_methods": ["_create_processor"],
                "expected_methods": ["_create_processor", "_get_processor"]
            },
            "io_methods": {
                "passed": True,
                "found_methods": ["_get_inputs", "_get_outputs"],
                "expected_methods": ["_get_inputs", "_get_outputs"]
            }
        }
    }
}
```

## Scoring System

### StreamlinedStepBuilderScorer

The scoring system evaluates builders across validation components with weighted scoring.

```python
from cursus.validation.builders.reporting.scoring import StreamlinedStepBuilderScorer

# Create scorer from validation results
validation_results = tester.run_validation_for_step("tabular_preprocessing")
scorer = StreamlinedStepBuilderScorer(validation_results)

# Generate scoring report
scoring_report = scorer.generate_report()

print(f"Overall Score: {scoring_report['overall']['score']:.1f}/100")
print(f"Overall Rating: {scoring_report['overall']['rating']}")

# Component scores
components = scoring_report['components']
for comp_name, comp_data in components.items():
    score = comp_data['score']
    weight = comp_data['weight']
    print(f"{comp_name}: {score:.1f}/100 (weight: {weight})")
```

### Scoring Weights

The streamlined scoring system uses the following component weights:

- **Alignment Validation**: 2.0x (most important - core validation)
- **Integration Testing**: 1.5x (unique builder capabilities)
- **Step Creation**: 1.0x (basic step creation capability)

### Quality Ratings

- **Excellent**: 90-100 points
- **Good**: 80-89 points
- **Satisfactory**: 70-79 points
- **Needs Work**: 60-69 points
- **Poor**: 0-59 points

## Reporting System

### StreamlinedBuilderTestReport

Enhanced reporting with alignment system integration.

```python
from cursus.validation.builders.reporting import StreamlinedBuilderTestReport

# Create report
report = StreamlinedBuilderTestReport(
    builder_name="tabular_preprocessing",
    builder_class="TabularPreprocessingStepBuilder",
    sagemaker_step_type="Processing"
)

# Add validation results
report.add_alignment_results(alignment_results)
report.add_integration_results(integration_results)
report.add_scoring_data(scoring_data)

# Use report methods
print(f"Overall Status: {report.get_overall_status()}")
print(f"Quality Score: {report.get_quality_score():.1f}/100")
print(f"Is Passing: {report.is_passing()}")

# Export report
json_content = report.export_to_json()
report.save_to_file(Path("report.json"))
report.print_summary()
```

### StreamlinedBuilderTestReporter

Automated testing and reporting for builders.

```python
from cursus.validation.builders.reporting import StreamlinedBuilderTestReporter

# Initialize reporter
reporter = StreamlinedBuilderTestReporter(
    output_dir=Path("test_reports")
)

# Test and report single builder
report = reporter.test_and_report_builder(
    builder_class=TabularPreprocessingStepBuilder,
    step_name="tabular_preprocessing"
)

# Test all builders of a specific type
reports = reporter.test_step_type_builders("Processing")

print(f"Tested {len(reports)} Processing builders")
for step_name, report in reports.items():
    print(f"  {step_name}: {'✅' if report.is_passing() else '❌'}")
```

## Error Handling

### Common Exceptions

The streamlined system handles various error conditions gracefully:

```python
def robust_builder_testing(step_name):
    """Example of robust builder testing with comprehensive error handling."""
    
    try:
        # Initialize tester
        tester = UniversalStepBuilderTest(
            verbose=True,
            enable_scoring=True
        )
        
        # Check if step exists
        discovered_steps = tester._discover_all_steps()
        if step_name not in discovered_steps:
            print(f"⚠️ Step {step_name} not found in discovered steps")
            return False
        
        # Run validation
        result = tester.run_validation_for_step(step_name)
        
        # Check results
        overall_status = result.get("overall_status", "UNKNOWN")
        
        if overall_status == "PASSED":
            print(f"✅ {step_name} validation passed")
            
            # Show quality score if available
            if "scoring" in result:
                score = result["scoring"].get("overall", {}).get("score", 0)
                print(f"Quality score: {score:.1f}/100")
            
            return True
        else:
            print(f"❌ {step_name} validation failed: {overall_status}")
            
            # Show component issues
            components = result.get("components", {})
            for comp_name, comp_result in components.items():
                if comp_result.get("status") == "ERROR":
                    error = comp_result.get("error", "Unknown error")
                    print(f"  {comp_name}: {error}")
            
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Check if builder class can be imported")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        print("💡 Check builder implementation and dependencies")
        return False

# Use robust testing
success = robust_builder_testing("tabular_preprocessing")
```

### Configuration Validation

```python
def validate_tester_configuration():
    """Validate tester configuration and dependencies."""
    
    try:
        tester = UniversalStepBuilderTest(verbose=True)
        
        print("🔍 Configuration Validation:")
        print(f"  Step catalog available: {tester.step_catalog_available}")
        print(f"  Alignment system available: {tester.alignment_available}")
        
        # Test step discovery
        discovered_steps = tester._discover_all_steps()
        print(f"  Steps discovered: {len(discovered_steps)}")
        
        if tester.step_catalog_available:
            try:
                all_steps = tester.step_catalog.list_available_steps()
                print(f"  Step catalog steps: {len(all_steps)}")
            except Exception as e:
                print(f"  Step catalog error: {e}")
        
        if tester.alignment_available:
            print("  ✅ Alignment system integration ready")
        else:
            print("  ⚠️ Alignment system not available - using fallback")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

# Validate configuration
config_valid = validate_tester_configuration()
```

## Advanced Usage

### Custom Validation Workflows

```python
def custom_validation_workflow():
    """Custom validation workflow with specific requirements."""
    
    # Initialize with specific configuration
    tester = UniversalStepBuilderTest(
        workspace_dirs=["development/projects"],
        verbose=False,
        enable_scoring=True
    )
    
    # Define validation criteria
    required_score = 85.0
    required_pass_rate = 90.0
    
    # Run validation for specific steps
    critical_steps = ["tabular_preprocessing", "xgboost_training", "model_evaluation"]
    
    results = {}
    overall_success = True
    
    for step_name in critical_steps:
        try:
            result = tester.run_validation_for_step(step_name)
            results[step_name] = result
            
            overall_status = result.get("overall_status", "UNKNOWN")
            
            # Check pass criteria
            if overall_status != "PASSED":
                print(f"❌ {step_name}: Failed validation ({overall_status})")
                overall_success = False
                continue
            
            # Check quality score
            if "scoring" in result:
                score = result["scoring"].get("overall", {}).get("score", 0)
                if score < required_score:
                    print(f"⚠️ {step_name}: Low quality score ({score:.1f} < {required_score})")
                    overall_success = False
                else:
                    print(f"✅ {step_name}: Passed ({score:.1f}/100)")
            else:
                print(f"✅ {step_name}: Passed (no scoring)")
                
        except Exception as e:
            print(f"❌ {step_name}: Error - {e}")
            overall_success = False
    
    # Generate summary
    if overall_success:
        print(f"\n✅ Custom validation workflow passed!")
        print(f"All {len(critical_steps)} critical steps meet requirements")
    else:
        print(f"\n❌ Custom validation workflow failed")
        print(f"Review failed steps and address issues")
    
    return results, overall_success

# Run custom workflow
results, success = custom_validation_workflow()
```

### Integration with CI/CD

```python
def ci_cd_builder_validation():
    """Builder validation suitable for CI/CD pipelines."""
    
    import sys
    import os
    
    # Get configuration from environment
    step_types = os.getenv('STEP_TYPES_TO_TEST', 'Processing,Training').split(',')
    min_score = float(os.getenv('MIN_QUALITY_SCORE', '80.0'))
    
    overall_success = True
    results_summary = {}
    
    for step_type in step_types:
        print(f"=== Testing {step_type.strip()} Builders ===")
        
        try:
            type_results = UniversalStepBuilderTest.test_all_builders_by_type(
                sagemaker_step_type=step_type.strip(),
                enable_scoring=True
            )
            
            if 'error' not in type_results:
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
                        overall_success = False
                    else:
                        overall_status = result.get("overall_status", "UNKNOWN")
                        
                        if overall_status == "PASSED":
                            # Check quality score
                            if "scoring" in result:
                                score = result["scoring"].get("overall", {}).get("score", 0)
                                type_summary['scores'].append(score)
                                
                                if score >= min_score:
                                    print(f"✅ {step_name}: {score:.1f}/100")
                                    type_summary['passed'] += 1
                                else:
                                    print(f"⚠️ {step_name}: {score:.1f}/100 (below {min_score})")
                                    type_summary['failed'] += 1
                                    overall_success = False
                            else:
                                print(f"✅ {step_name}: Passed (no scoring)")
                                type_summary['passed'] += 1
                        else:
                            print(f"❌ {step_name}: {overall_status}")
                            type_summary['failed'] += 1
                            overall_success = False
                
                results_summary[step_type.strip()] = type_summary
                
            else:
                print(f"❌ {step_type} testing failed: {type_results['error']}")
                overall_success = False
                
        except Exception as e:
            print(f"❌ {step_type} testing error: {e}")
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

### Optimized Testing Patterns

```python
def optimized_builder_testing():
    """Optimized testing patterns for large codebases."""
    
    # Use batch testing for efficiency
    step_types = ["Processing", "Training", "CreateModel"]
    all_results = {}
    
    print("🚀 Running optimized batch testing...")
    
    for step_type in step_types:
        print(f"\n🔍 Testing {step_type} builders...")
        
        try:
            # Batch test all builders of this type
            type_results = UniversalStepBuilderTest.test_all_builders_by_type(
                sagemaker_step_type=step_type,
                verbose=False,  # Reduce output for batch processing
                enable_scoring=True
            )
            
            if 'error' not in type_results:
                all_results[step_type] = type_results
                
                # Quick summary
                total = len(type_results)
                passed = sum(1 for r in type_results.values() 
                           if r.get("overall_status") == "PASSED")
                
                print(f"  ✅ {step_type}: {passed}/{total} builders passed")
                
                # Average quality score
                scores = []
                for result in type_results.values():
                    if "scoring" in result:
                        score = result["scoring"].get("overall", {}).get("score", 0)
                        if score > 0:
                            scores.append(score)
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"      Average quality: {avg_score:.1f}/100")
            else:
                print(f"  ❌ {step_type}: {type_results['error']}")
                
        except Exception as e:
            print(f"  ❌ {step_type}: Error - {e}")
    
    # Overall summary
    total_builders = sum(len(results) for results in all_results.values())
    total_passed = sum(
        sum(1 for r in results.values() if r.get("overall_status") == "PASSED")
        for results in all_results.values()
    )
    
    success_rate = (total_passed / total_builders * 100) if total_builders > 0 else 0
    
    print(f"\n📊 Optimized Testing Summary:")
    print(f"  Total builders: {total_builders}")
    print(f"  Success rate: {success_rate:.1f}% ({total_passed}/{total_builders})")
    print(f"  Performance: Streamlined approach (60-70% faster)")
    
    return all_results

# Run optimized testing
optimized_results = optimized_builder_testing()
```

## API Reference Summary

### Core Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `run_validation_for_step()` | Streamlined validation for single step | `Dict[str, Any]` |
| `run_full_validation()` | Validation for all discovered steps | `Dict[str, Any]` |
| `run_all_tests()` | Enhanced testing with optional scoring/reporting | `Dict[str, Any]` |
| `run_all_tests_legacy()` | Raw test results for backward compatibility | `Dict[str, Dict[str, Any]]` |
| `run_all_tests_with_scoring()` | Tests with scoring enabled | `Dict[str, Any]` |
| `run_all_tests_with_full_report()` | Tests with scoring and structured reporting | `Dict[str, Any]` |
| `export_results_to_json()` | Export results to JSON format | `str` |

### Class Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `test_all_builders_by_type()` | Test all builders of specific SageMaker type | `Dict[str, Any]` |
| `from_builder_class()` | Create tester for specific builder class | `UniversalStepBuilderTest` |

### Discovery and Utility Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `_discover_all_steps()` | Discover all steps using step catalog | `List[str]` |
| `_get_builder_class_from_catalog()` | Get builder class from catalog | `Optional[Type]` |
| `generate_report()` | Generate comprehensive report | `Union[Report, Dict]` |

### Legacy Compatibility Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `validate_specific_script()` | Legacy single script validation | `Dict[str, Any]` |
| `discover_scripts()` | Legacy script discovery | `List[str]` |
| `get_validation_summary()` | Legacy validation summary | `Dict[str, Any]` |
| `print_summary()` | Print validation summary to console | `None` |

### Validation Components

The streamlined system validates through four main components:

1. **Alignment Validation** (Weight: 2.0x) - Core validation using alignment system
2. **Integration Testing** (Weight: 1.5x) - Unique builder integration capabilities  
3. **Step Creation** (Weight: 1.0x) - Step creation availability testing
4. **Step Type Validation** - Framework-specific compliance checks

### Key Benefits

- **60-70% code reduction** through alignment system integration
- **50% faster execution** by eliminating duplicate validation
- **Single maintenance point** for core validation logic
- **Proven validation foundation** with 100% test pass rate
- **Backward compatibility** for seamless migration
- **Enhanced reporting** with quality scoring and insights

## Best Practices

1. **Use Streamlined Approach**: Prefer the new workspace-aware initialization over legacy single-builder mode
2. **Enable Scoring**: Use scoring to track quality improvements over time
3. **Batch Testing**: Use class methods for testing multiple builders efficiently
4. **Error Handling**: Always implement robust error handling in production code
5. **Performance**: Leverage the streamlined system's performance optimizations
6. **CI/CD Integration**: Implement automated testing in development pipelines
7. **Quality Monitoring**: Use quality scores and ratings for continuous improvement

For additional examples and usage patterns, see the [Universal Builder Tester Quick Start Guide](universal_builder_tester_quick_start.md).
