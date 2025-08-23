---
tags:
  - design
  - testing
  - script_functionality
  - usage_examples
  - api_design
keywords:
  - usage examples design
  - API design
  - testing examples
  - notebook examples
  - CLI examples
topics:
  - testing framework
  - usage examples
  - API design
  - user interface
language: python
date of note: 2025-08-21
---

# Pipeline Script Functionality Testing - Usage Examples and API Design

## Overview

This document provides comprehensive usage examples and API design specifications for the Pipeline Script Functionality Testing System. It demonstrates how users can leverage the system across different interfaces (Jupyter notebooks, CLI, Python API) for various testing scenarios.

## API Design Philosophy

### Design Principles

#### 1. Simplicity First
- **One-liner APIs** for common tasks
- **Progressive complexity** - simple tasks are simple, complex tasks are possible
- **Sensible defaults** that work for most use cases
- **Clear, intuitive method names**

#### 2. Consistency Across Interfaces
- **Unified parameter naming** across CLI, Python API, and Jupyter
- **Consistent return types** and data structures
- **Harmonized configuration** options
- **Shared terminology** and concepts

#### 3. Discoverability
- **Auto-completion support** in Jupyter and IDEs
- **Built-in help and documentation**
- **Example-driven documentation**
- **Progressive disclosure** of advanced features

## Core API Structure

### Main API Classes

```python
# Core API Classes
from cursus.validation.script_functionality import (
    PipelineScriptExecutor,      # Core execution engine
    PipelineTestingNotebook,     # Jupyter-optimized interface
    ScriptFunctionalityTester,   # High-level testing interface
    TestResultAnalyzer,          # Result analysis utilities
    TestConfigBuilder            # Configuration builder
)

# One-liner convenience functions
from cursus.validation.script_functionality import (
    quick_test_script,           # Test single script
    quick_test_pipeline,         # Test complete pipeline
    deep_dive_analysis,          # Deep analysis with S3 data
    compare_executions,          # Compare pipeline executions
    batch_test_scripts           # Test multiple scripts
)
```

### Configuration API

```python
# Configuration builders for different scenarios
from cursus.validation.script_functionality.config import (
    IsolationTestConfig,         # Isolation testing configuration
    PipelineTestConfig,          # Pipeline testing configuration
    DeepDiveTestConfig,          # Deep dive analysis configuration
    PerformanceTestConfig,       # Performance testing configuration
    ComparisonTestConfig         # Execution comparison configuration
)
```

## Jupyter Notebook Usage Examples

### 1. Quick Script Testing

#### Basic Script Testing
```python
# Import the testing system
from cursus.validation.script_functionality import quick_test_script

# Test a single script with default settings
result = quick_test_script("currency_conversion")
# Automatically displays rich HTML summary with:
# - Execution status and timing
# - Memory usage
# - Data quality metrics
# - Error details (if any)
# - Recommendations
```

#### Advanced Script Testing
```python
# Test with multiple scenarios and custom configuration
result = quick_test_script(
    script_name="currency_conversion",
    scenarios=["standard", "edge_cases", "large_volume"],
    data_source="synthetic",
    data_size="medium",
    timeout=300,
    memory_limit="1GB"
)

# Display different views of results
result.show_summary()           # Concise summary
result.show_details()           # Detailed results
result.visualize_performance()  # Performance charts
result.show_recommendations()   # Improvement suggestions
```

#### Parameterized Testing
```python
# Test script with different parameter combinations
from cursus.validation.script_functionality import PipelineTestingNotebook

tester = PipelineTestingNotebook()

# Define parameter variations
parameter_sets = [
    {"learning_rate": 0.01, "max_depth": 3},
    {"learning_rate": 0.1, "max_depth": 5},
    {"learning_rate": 0.2, "max_depth": 7}
]

results = []
for params in parameter_sets:
    result = tester.test_script_with_parameters(
        script_name="xgboost_training",
        parameters=params,
        data_source="synthetic"
    )
    results.append(result)

# Compare results across parameter sets
comparison = tester.compare_parameter_results(results)
comparison.visualize_parameter_impact()
```

### 2. Pipeline Testing

#### Basic Pipeline Testing
```python
from cursus.validation.script_functionality import quick_test_pipeline

# Test complete pipeline end-to-end
result = quick_test_pipeline("xgb_training_simple")
# Automatically displays:
# - Pipeline execution flow diagram
# - Step-by-step results
# - Data flow validation
# - Performance metrics
# - Overall success/failure status
```

#### Advanced Pipeline Testing
```python
# Test pipeline with custom configuration
result = quick_test_pipeline(
    pipeline_name="xgb_training_simple",
    data_source="synthetic",
    validation_level="strict",
    execution_mode="sequential",
    continue_on_failure=False,
    save_intermediate_results=True
)

# Explore different aspects of results
result.show_execution_flow()        # Interactive flow diagram
result.show_data_quality_evolution() # Data quality through pipeline
result.analyze_bottlenecks()        # Performance bottlenecks
result.show_step_details("xgboost_training")  # Specific step details
```

#### Pipeline Comparison
```python
# Compare different pipeline configurations
config1 = {"data_size": "small", "validation_level": "lenient"}
config2 = {"data_size": "large", "validation_level": "strict"}

result1 = quick_test_pipeline("xgb_training_simple", **config1)
result2 = quick_test_pipeline("xgb_training_simple", **config2)

# Compare results
comparison = tester.compare_pipeline_results([result1, result2])
comparison.show_side_by_side()      # Side-by-side comparison
comparison.show_performance_diff()  # Performance differences
comparison.highlight_key_differences()  # Key differences summary
```

### 3. Interactive Debugging

#### Step-by-Step Execution
```python
from cursus.validation.script_functionality import PipelineTestingNotebook

tester = PipelineTestingNotebook()

# Start interactive debugging session
debug_session = tester.interactive_debug(
    pipeline_name="xgb_training_simple",
    data_source="synthetic"
)

# Set breakpoints
debug_session.set_breakpoint("currency_conversion")
debug_session.set_breakpoint("xgboost_training", condition="data_size > 10000")

# Execute to first breakpoint
debug_session.run_to_breakpoint()
# Displays current execution state and available actions
```

#### Data Inspection at Breakpoints
```python
# At breakpoint, inspect intermediate data
data_inspector = debug_session.inspect_data()

# Interactive data exploration widgets appear
data_inspector.show_data_summary()    # Statistical summary
data_inspector.show_sample_data(100)  # Sample of data
data_inspector.show_data_quality()    # Quality metrics
data_inspector.export_data("csv")     # Export for external analysis

# Modify parameters before continuing
debug_session.modify_parameters(
    learning_rate=0.05,
    max_depth=4
)

# Continue execution
debug_session.continue_execution()
```

### 4. Deep Dive Analysis

#### Real S3 Data Analysis
```python
from cursus.validation.script_functionality import deep_dive_analysis

# Analyze pipeline with real S3 data
result = deep_dive_analysis(
    pipeline_name="xgb_training_simple",
    s3_execution_arn="arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345",
    analysis_scope="full",
    focus_areas=["performance", "data_quality", "error_patterns"]
)

# Rich analysis displays
result.show_performance_analysis()    # Performance vs synthetic data
result.show_data_quality_report()     # Real data quality analysis
result.show_optimization_recommendations()  # Specific optimizations
result.show_production_readiness()    # Production readiness assessment
```

#### Comparative Analysis
```python
# Compare production executions
comparison = tester.compare_s3_executions(
    baseline_arn="arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345",
    comparison_arn="arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12346",
    comparison_metrics=["execution_time", "data_quality", "resource_usage"]
)

comparison.show_regression_analysis()  # Performance regressions
comparison.show_data_drift_analysis()  # Data drift detection
comparison.identify_root_causes()     # Root cause analysis
```

### 5. Batch Testing and Analysis

#### Batch Script Testing
```python
from cursus.validation.script_functionality import batch_test_scripts

# Test multiple scripts in parallel
scripts_to_test = [
    "currency_conversion",
    "tabular_preprocessing", 
    "xgboost_training",
    "model_evaluation"
]

batch_results = batch_test_scripts(
    script_names=scripts_to_test,
    scenarios=["standard", "edge_cases"],
    parallel_execution=True,
    max_workers=4
)

# Analyze batch results
batch_results.show_summary_dashboard()  # Overall dashboard
batch_results.identify_problematic_scripts()  # Scripts with issues
batch_results.show_performance_comparison()   # Performance comparison
batch_results.generate_test_report()    # Comprehensive report
```

## CLI Usage Examples

### 1. Basic CLI Commands

#### Script Testing
```bash
# Test single script
cursus script-functionality test-script currency_conversion

# Test with specific scenarios
cursus script-functionality test-script currency_conversion \
    --scenarios standard,edge_cases,performance \
    --data-source synthetic \
    --data-size large \
    --output-dir ./test_results

# Test with timeout and memory limits
cursus script-functionality test-script xgboost_training \
    --timeout 600 \
    --memory-limit 2GB \
    --save-intermediate-results
```

#### Pipeline Testing
```bash
# Test complete pipeline
cursus script-functionality test-pipeline xgb_training_simple

# Test with custom configuration
cursus script-functionality test-pipeline xgb_training_simple \
    --data-source synthetic \
    --validation-level strict \
    --execution-mode sequential \
    --output-dir ./pipeline_results \
    --generate-report

# Test with real S3 data
cursus script-functionality test-pipeline xgb_training_simple \
    --s3-execution-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345 \
    --analysis-scope sample \
    --sample-size 10000
```

### 2. Advanced CLI Usage

#### Batch Testing
```bash
# Test multiple scripts
cursus script-functionality batch-test \
    --scripts currency_conversion,tabular_preprocessing,xgboost_training \
    --scenarios standard,edge_cases \
    --parallel \
    --max-workers 4 \
    --output-dir ./batch_results

# Test all scripts in pipeline
cursus script-functionality test-all-pipeline-scripts xgb_training_simple \
    --exclude model_registration \
    --scenarios standard \
    --generate-summary-report
```

#### Performance Analysis
```bash
# Performance benchmarking
cursus script-functionality benchmark-script xgboost_training \
    --data-volumes small,medium,large \
    --iterations 5 \
    --output-format json \
    --save-performance-data

# Memory profiling
cursus script-functionality profile-memory currency_conversion \
    --data-size large \
    --detailed-analysis \
    --export-profile ./memory_profile.json
```

#### Deep Dive Analysis
```bash
# Deep dive with S3 data
cursus script-functionality deep-dive-analysis \
    --pipeline xgb_training_simple \
    --s3-execution-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345 \
    --focus-areas performance,data_quality \
    --generate-recommendations \
    --output-dir ./deep_dive_results

# Compare executions
cursus script-functionality compare-executions \
    --baseline-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345 \
    --comparison-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12346 \
    --metrics execution_time,data_quality,resource_usage \
    --generate-comparison-report
```

### 3. Configuration and Customization

#### Configuration Files
```bash
# Use configuration file
cursus script-functionality test-pipeline xgb_training_simple \
    --config ./test_config.yaml \
    --override data_source=synthetic

# Generate configuration template
cursus script-functionality generate-config \
    --template pipeline-test \
    --output ./pipeline_test_config.yaml

# Validate configuration
cursus script-functionality validate-config ./test_config.yaml
```

#### Custom Test Scenarios
```bash
# Create custom test scenario
cursus script-functionality create-scenario \
    --name custom_large_data \
    --base-scenario standard \
    --data-size large \
    --memory-limit 4GB \
    --timeout 1800

# List available scenarios
cursus script-functionality list-scenarios

# Test with custom scenario
cursus script-functionality test-script currency_conversion \
    --scenarios custom_large_data
```

## Python API Usage Examples

### 1. Programmatic Testing

#### Basic API Usage
```python
from cursus.validation.script_functionality import ScriptFunctionalityTester

# Initialize tester
tester = ScriptFunctionalityTester(workspace_dir="./testing_workspace")

# Test single script
result = tester.test_script(
    script_name="currency_conversion",
    scenarios=["standard", "edge_cases"],
    data_source="synthetic"
)

# Process results programmatically
if result.is_successful():
    print(f"Script test passed in {result.execution_time:.2f} seconds")
    print(f"Memory usage: {result.peak_memory_mb} MB")
else:
    print(f"Script test failed: {result.error_message}")
    for recommendation in result.recommendations:
        print(f"- {recommendation}")
```

#### Advanced Configuration
```python
from cursus.validation.script_functionality.config import IsolationTestConfig

# Create detailed test configuration
config = IsolationTestConfig(
    scenarios=["standard", "edge_cases", "performance"],
    data_source="synthetic",
    data_size="large",
    timeout_seconds=600,
    memory_limit_mb=2048,
    save_intermediate_results=True,
    enable_performance_profiling=True,
    quality_gates={
        "execution_time_max": 300,
        "memory_usage_max": 1024,
        "success_rate_min": 0.95
    }
)

# Execute test with configuration
result = tester.test_script_with_config("xgboost_training", config)
```

### 2. Integration with Testing Frameworks

#### pytest Integration
```python
import pytest
from cursus.validation.script_functionality import ScriptFunctionalityTester

class TestPipelineScripts:
    @classmethod
    def setup_class(cls):
        cls.tester = ScriptFunctionalityTester()
    
    @pytest.mark.parametrize("script_name", [
        "currency_conversion",
        "tabular_preprocessing",
        "xgboost_training"
    ])
    def test_script_functionality(self, script_name):
        """Test that all pipeline scripts execute successfully"""
        result = self.tester.test_script(
            script_name=script_name,
            scenarios=["standard"],
            data_source="synthetic"
        )
        
        assert result.is_successful(), f"Script {script_name} failed: {result.error_message}"
        assert result.execution_time < 300, f"Script {script_name} too slow: {result.execution_time}s"
        assert result.peak_memory_mb < 1024, f"Script {script_name} uses too much memory: {result.peak_memory_mb}MB"
    
    def test_pipeline_end_to_end(self):
        """Test complete pipeline execution"""
        result = self.tester.test_pipeline(
            pipeline_name="xgb_training_simple",
            data_source="synthetic"
        )
        
        assert result.is_successful(), f"Pipeline failed: {result.error_message}"
        assert result.data_flow_validation.is_valid(), "Data flow validation failed"
        assert len(result.failed_steps) == 0, f"Failed steps: {result.failed_steps}"
```

#### unittest Integration
```python
import unittest
from cursus.validation.script_functionality import ScriptFunctionalityTester

class PipelineScriptTests(unittest.TestCase):
    def setUp(self):
        self.tester = ScriptFunctionalityTester()
    
    def test_currency_conversion_standard_scenario(self):
        """Test currency conversion with standard data"""
        result = self.tester.test_script(
            script_name="currency_conversion",
            scenarios=["standard"],
            data_source="synthetic"
        )
        
        self.assertTrue(result.is_successful())
        self.assertLess(result.execution_time, 60)
        self.assertIsNotNone(result.output_data)
    
    def test_pipeline_data_flow(self):
        """Test data flow compatibility across pipeline"""
        result = self.tester.test_pipeline(
            pipeline_name="xgb_training_simple",
            data_source="synthetic",
            validation_level="strict"
        )
        
        self.assertTrue(result.data_flow_validation.is_valid())
        for step_name, step_result in result.step_results.items():
            self.assertTrue(step_result.is_successful(), 
                          f"Step {step_name} failed: {step_result.error_message}")
```

### 3. Custom Test Scenarios

#### Creating Custom Test Data
```python
from cursus.validation.script_functionality.data import SyntheticDataGenerator

# Create custom data generator
data_generator = SyntheticDataGenerator()

# Define custom data scenario
custom_scenario = {
    "name": "high_volume_currency_data",
    "description": "High volume currency conversion test",
    "data_config": {
        "num_records": 100000,
        "currency_pairs": ["USD-EUR", "EUR-GBP", "GBP-JPY", "JPY-USD"],
        "date_range": ("2020-01-01", "2024-12-31"),
        "include_edge_cases": True,
        "missing_data_percentage": 0.05
    }
}

# Generate custom test data
test_data = data_generator.generate_scenario_data(custom_scenario)

# Test script with custom data
result = tester.test_script_with_data(
    script_name="currency_conversion",
    test_data=test_data,
    scenario_name="high_volume_currency_data"
)
```

#### Custom Validation Rules
```python
from cursus.validation.script_functionality.validation import CustomValidationRule

# Define custom validation rule
class DataQualityRule(CustomValidationRule):
    def __init__(self, min_quality_score=0.9):
        self.min_quality_score = min_quality_score
    
    def validate(self, test_result):
        if test_result.data_quality_score < self.min_quality_score:
            return ValidationResult(
                passed=False,
                message=f"Data quality score {test_result.data_quality_score} below threshold {self.min_quality_score}"
            )
        return ValidationResult(passed=True)

# Apply custom validation
tester.add_validation_rule("data_quality", DataQualityRule(min_quality_score=0.95))

# Test with custom validation
result = tester.test_script("currency_conversion", scenarios=["standard"])
```

## Configuration Examples

### 1. YAML Configuration Files

#### Basic Test Configuration
```yaml
# test_config.yaml
test_configuration:
  workspace_dir: "./pipeline_testing"
  default_data_source: "synthetic"
  default_scenarios: ["standard", "edge_cases"]
  
  isolation_testing:
    timeout_seconds: 300
    memory_limit_mb: 1024
    save_intermediate_results: true
    
  pipeline_testing:
    execution_mode: "sequential"
    validation_level: "strict"
    continue_on_failure: false
    
  performance_testing:
    enable_profiling: true
    benchmark_iterations: 3
    resource_monitoring: true
    
  quality_gates:
    execution_time_max: 300
    memory_usage_max: 1024
    success_rate_min: 0.95
    data_quality_min: 0.9
```

#### Advanced Configuration with Custom Scenarios
```yaml
# advanced_test_config.yaml
test_configuration:
  workspace_dir: "./advanced_testing"
  
  custom_scenarios:
    high_volume:
      description: "High volume data testing"
      data_size: "large"
      num_records: 100000
      timeout_seconds: 600
      memory_limit_mb: 2048
      
    edge_cases_extended:
      description: "Extended edge case testing"
      base_scenario: "edge_cases"
      additional_edge_cases:
        - "empty_datasets"
        - "malformed_data"
        - "extreme_values"
      timeout_seconds: 450
      
  script_specific_config:
    xgboost_training:
      scenarios: ["standard", "high_volume"]
      memory_limit_mb: 4096
      timeout_seconds: 1200
      
    currency_conversion:
      scenarios: ["standard", "edge_cases_extended"]
      enable_currency_validation: true
      
  s3_integration:
    default_bucket: "ml-pipeline-test-data"
    cache_dir: "/tmp/pipeline_cache"
    cache_ttl_hours: 24
    
  reporting:
    generate_html_reports: true
    include_performance_charts: true
    export_raw_data: false
    report_template: "detailed"
```

### 2. Environment-Specific Configuration

#### Development Environment
```yaml
# dev_config.yaml
environment: "development"

test_configuration:
  workspace_dir: "./dev_testing"
  default_data_source: "synthetic"
  
  isolation_testing:
    timeout_seconds: 120  # Shorter timeouts for dev
    memory_limit_mb: 512
    
  pipeline_testing:
    execution_mode: "sequential"
    validation_level: "lenient"  # More lenient for dev
    
  performance_testing:
    enable_profiling: false  # Disable for faster dev cycles
    
  quality_gates:
    execution_time_max: 180
    success_rate_min: 0.8  # Lower threshold for dev
```

#### Production Environment
```yaml
# prod_config.yaml
environment: "production"

test_configuration:
  workspace_dir: "/opt/pipeline_testing"
  default_data_source: "s3"
  
  isolation_testing:
    timeout_seconds: 600
    memory_limit_mb: 2048
    
  pipeline_testing:
    execution_mode: "parallel"
    validation_level: "strict"
    
  performance_testing:
    enable_profiling: true
    benchmark_iterations: 5
    
  quality_gates:
    execution_time_max: 300
    memory_usage_max: 1024
    success_rate_min: 0.98  # High threshold for prod
    data_quality_min: 0.95
    
  s3_integration:
    bucket: "prod-ml-pipeline-data"
    use_production_data: true
    sample_size: 50000
```

## Error Handling Examples

### 1. Graceful Error Handling

#### Script Execution Errors
```python
from cursus.validation.script_functionality import ScriptFunctionalityTester
from cursus.validation.script_functionality.exceptions import (
    ScriptExecutionError,
    DataCompatibilityError,
    ConfigurationError
)

tester = ScriptFunctionalityTester()

try:
    result = tester.test_script("problematic_script", scenarios=["standard"])
    
    if not result.is_successful():
        # Handle different types of failures
        if result.error_type == "IMPORT_ERROR":
            print("Script import failed - check dependencies")
            print(f"Missing modules: {result.missing_dependencies}")
            
        elif result.error_type == "DATA_ERROR":
            print("Data compatibility issue detected")
            print(f"Data validation errors: {result.data_validation_errors}")
            
        elif result.error_type == "RESOURCE_ERROR":
            print("Resource constraint exceeded")
            print(f"Peak memory usage: {result.peak_memory_mb} MB")
            print(f"Execution time: {result.execution_time} seconds")
            
        # Show recommendations
        for recommendation in result.recommendations:
            print(f"Recommendation: {recommendation}")
            
except ScriptExecutionError as e:
    print(f"Script execution failed: {e}")
    print(f"Error details: {e.details}")
    
except DataCompatibilityError as e:
    print(f"Data compatibility issue: {e}")
    print(f"Incompatible fields: {e.incompatible_fields}")
    
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    print(f"Invalid configuration keys: {e.invalid_keys}")
```

#### Pipeline Execution Error Recovery
```python
# Test pipeline with error recovery
result = tester.test_pipeline(
    pipeline_name="xgb_training_simple",
    data_source="synthetic",
    continue_on_failure=True,  # Continue even if some steps fail
    error_recovery_strategy="skip_failed_steps"
)

# Analyze partial results
if result.has_failures():
    print(f"Pipeline completed with {len(result.failed_steps)} failed steps")
    
    for step_name in result.failed_steps:
        step_result = result.step_results[step_name]
        print(f"Step {step_name} failed: {step_result.error_message}")
        
        # Show recovery suggestions
        for suggestion in step_result.recovery_suggestions:
            print(f"  - {suggestion}")
    
    # Show successful steps
    successful_steps = [name for name, result in result.step_results.items() 
                       if result.is_successful()]
    print(f"Successful steps: {successful_steps}")
```

### 2. Debugging and Troubleshooting

#### Detailed Error Analysis
```python
# Enable detailed error analysis
tester = ScriptFunctionalityTester(
    enable_detailed_errors=True,
    capture_stack_traces=True,
    save_error_context=True
)

result = tester.test_script("failing_script", scenarios=["standard"])

if not result.is_successful():
    # Get detailed error information
    error_analysis = result.get_detailed_error_analysis()
    
    print("Error Analysis:")
    print(f"Error type: {error_analysis.error_type}")
    print(f"Error location: {error_analysis.error_location}")
    print(f"Stack trace: {error_analysis.stack_trace}")
    
    # Context information
    print(f"Input data summary: {error_analysis.input_data_summary}")
    print(f"Environment variables: {error_analysis.environment_variables}")
    print(f"Configuration: {error_analysis.configuration}")
    
    # Suggested fixes
    print("Suggested fixes:")
    for fix in error_analysis.suggested_fixes:
        print(f"  - {fix.description}")
        print(f"    Command: {fix.command}")
        print(f"    Confidence: {fix.confidence}")
```

## Performance Optimization Examples

### 1. Parallel Execution

#### Parallel Script Testing
```python
from cursus.validation.script_functionality import batch_test_scripts

# Test multiple scripts in parallel
scripts = ["currency_conversion", "tabular_preprocessing", "xgboost_training"]

results = batch_test_scripts(
    script_names=scripts,
    scenarios=["standard", "edge_cases"],
    parallel_execution=True,
    max_workers=4,
    timeout_per_script=300
)

# Analyze parallel execution results
print(f"Total execution time: {results.total_execution_time:.2f} seconds")
print(f"Time saved vs sequential: {results.time_saved:.2f} seconds")
print(f"Parallel efficiency: {results.parallel_efficiency:.1%}")
```

#### Parallel Pipeline Testing
```python
# Test pipeline with parallel step execution where possible
result = tester.test_pipeline(
    pipeline_name="complex_ml_pipeline",
    execution_mode="parallel",
    max_parallel_steps=3,
    data_source="synthetic"
)

# Analyze parallel execution
execution_plan = result.execution_plan
print(f"Parallel groups: {len(execution_plan.parallel_groups)}")
for i, group in enumerate(execution_plan.parallel_groups):
    print(f"Group {i+1}: {group.step_names} (estimated time: {group.estimated_time}s)")
```

### 2. Resource Optimization

#### Memory-Optimized Testing
```python
# Configure memory-optimized testing
memory_config = {
    "enable_memory_monitoring": True,
    "memory_limit_mb": 1024,
    "enable_garbage_collection": True,
    "stream_large_datasets": True,
    "cleanup_intermediate_data": True
}

result = tester.test_script(
    script_name="memory_intensive_script",
    scenarios=["large_volume"],
    **memory_config
)

# Analyze memory usage
memory_profile = result.memory_profile
print(f"Peak memory usage: {memory_profile.peak_usage_mb} MB")
print(f"Memory efficiency: {memory_profile.efficiency_score:.2f}")
print(f"Garbage collection events: {memory_profile.gc_events}")
```

## Integration Examples

### 1. CI/CD Integration

#### GitHub Actions Integration
```yaml
# .github/workflows/pipeline_testing.yml
name: Pipeline Script Functionality Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-pipeline-scripts:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
        
    - name: Run script functionality tests
      run: |
        cursus script-functionality batch-test \
          --scripts currency_conversion,tabular_preprocessing,xgboost_training \
          --scenarios standard,edge_cases \
          --output-format junit \
          --output-dir ./test-results \
          --generate-coverage-report
        
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: test-results/
        
    - name: Publish test results
      uses: dorny/test-reporter@v1
      if: always()
      with:
        name: Pipeline Script Tests
        path: test-results/*.xml
        reporter: java-junit
```

#### Jenkins Integration
```groovy
// Jenkinsfile
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh
