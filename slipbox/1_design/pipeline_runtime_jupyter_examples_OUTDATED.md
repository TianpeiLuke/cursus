---
tags:
  - design
  - testing
  - jupyter
  - notebook_examples
  - interactive_testing
keywords:
  - Jupyter notebook
  - interactive testing
  - notebook examples
  - visualization
topics:
  - Jupyter integration
  - interactive testing
  - notebook examples
language: python
date of note: 2025-08-21
---

# Pipeline Runtime Testing - Jupyter Notebook Examples

## Overview

This document provides comprehensive Jupyter notebook usage examples for the Pipeline Runtime Testing System. It demonstrates interactive testing, debugging, and analysis capabilities optimized for notebook environments.

## Jupyter Notebook Usage Examples

### 1. Quick Script Testing

#### Basic Script Testing
```python
# Import the testing system
from cursus.validation.runtime import quick_test_script

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
# Test XGBoost Training script with different hyperparameter combinations
from cursus.validation.runtime import PipelineTestingNotebook

tester = PipelineTestingNotebook()

# Define XGBoost hyperparameter variations based on actual contract
parameter_sets = [
    {
        "eta": 0.01, "max_depth": 3, "num_round": 100,
        "is_binary": True, "early_stopping_rounds": 10
    },
    {
        "eta": 0.1, "max_depth": 5, "num_round": 200,
        "is_binary": True, "early_stopping_rounds": 15
    },
    {
        "eta": 0.2, "max_depth": 7, "num_round": 150,
        "is_binary": True, "early_stopping_rounds": 20
    }
]

results = []
for params in parameter_sets:
    result = tester.test_script_with_parameters(
        script_name="XGBoostTraining",  # Using canonical step name
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
from cursus.validation.runtime import quick_test_pipeline

# Test XGBoost Training -> Model Evaluation pipeline end-to-end
result = quick_test_pipeline("xgb_training_with_eval")
# Automatically displays:
# - Pipeline execution flow diagram showing XGBoostTraining -> XGBoostModelEval
# - Step-by-step results with model artifacts and evaluation metrics
# - Data flow validation between training outputs and evaluation inputs
# - Performance metrics for both training and evaluation phases
# - Overall success/failure status
```

#### Advanced Pipeline Testing
```python
# Test XGBoost pipeline with custom configuration
result = quick_test_pipeline(
    pipeline_name="xgb_training_with_eval",
    data_source="synthetic",
    validation_level="strict",
    execution_mode="sequential",
    continue_on_failure=False,
    save_intermediate_results=True
)

# Explore different aspects of results
result.show_execution_flow()        # Interactive flow diagram showing training->evaluation
result.show_data_quality_evolution() # Data quality through preprocessing->training->evaluation
result.analyze_bottlenecks()        # Performance bottlenecks (likely in XGBoost training)
result.show_step_details("XGBoostTraining")  # Training step details with model artifacts
result.show_step_details("XGBoostModelEval") # Evaluation step details with metrics
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
from cursus.validation.runtime import PipelineTestingNotebook

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
from cursus.validation.runtime import deep_dive_analysis

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
from cursus.validation.runtime import batch_test_scripts

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

## Interactive Widgets and Visualizations

### 1. Parameter Exploration Widgets

#### Hyperparameter Tuning Widget
```python
from cursus.validation.runtime.widgets import ParameterExplorationWidget

# Create interactive parameter exploration widget
param_widget = ParameterExplorationWidget(
    script_name="XGBoostTraining",
    parameters={
        "eta": {"type": "slider", "min": 0.01, "max": 0.3, "step": 0.01, "default": 0.1},
        "max_depth": {"type": "slider", "min": 3, "max": 10, "step": 1, "default": 6},
        "num_round": {"type": "slider", "min": 50, "max": 500, "step": 50, "default": 100},
        "early_stopping_rounds": {"type": "slider", "min": 5, "max": 50, "step": 5, "default": 10}
    }
)

# Display widget
param_widget.display()

# Widget automatically runs tests when parameters change
# Results are displayed in real-time with performance charts
```

#### Data Size Impact Widget
```python
from cursus.validation.runtime.widgets import DataSizeImpactWidget

# Explore impact of data size on performance
size_widget = DataSizeImpactWidget(
    script_name="currency_conversion",
    data_sizes=["small", "medium", "large", "xlarge"],
    metrics=["execution_time", "memory_usage", "data_quality"]
)

size_widget.display()
# Interactive charts showing performance scaling
```

### 2. Result Visualization Widgets

#### Performance Comparison Dashboard
```python
# Create interactive performance dashboard
from cursus.validation.runtime.widgets import PerformanceDashboard

dashboard = PerformanceDashboard()

# Add multiple test results
dashboard.add_result("XGBoost Training - Small Data", xgb_small_result)
dashboard.add_result("XGBoost Training - Large Data", xgb_large_result)
dashboard.add_result("Currency Conversion", currency_result)

# Display interactive dashboard
dashboard.display()
# Features:
# - Interactive charts with zoom/pan
# - Metric selection dropdowns
# - Side-by-side comparisons
# - Export capabilities
```

#### Data Flow Visualization
```python
from cursus.validation.runtime.widgets import DataFlowVisualizer

# Visualize data flow through pipeline
flow_viz = DataFlowVisualizer(pipeline_result)
flow_viz.display()

# Interactive features:
# - Click on steps to see details
# - Hover for data statistics
# - Zoom into specific pipeline sections
# - Export flow diagrams
```

### 3. Debugging and Analysis Widgets

#### Error Analysis Widget
```python
from cursus.validation.runtime.widgets import ErrorAnalysisWidget

# Analyze errors across multiple test runs
error_widget = ErrorAnalysisWidget()
error_widget.add_results([result1, result2, result3, result4])
error_widget.display()

# Features:
# - Error categorization
# - Frequency analysis
# - Root cause suggestions
# - Fix recommendations
```

#### Memory Profiling Widget
```python
from cursus.validation.runtime.widgets import MemoryProfileWidget

# Interactive memory profiling
memory_widget = MemoryProfileWidget(test_result)
memory_widget.display()

# Features:
# - Memory usage timeline
# - Peak usage identification
# - Memory leak detection
# - Optimization suggestions
```

## Custom Test Scenarios for Jupyter

### 1. Creating Custom Test Data for XGBoost Pipeline

```python
from cursus.validation.runtime.data import BaseSyntheticDataGenerator, DefaultSyntheticDataGenerator

# Create default data generator
data_generator = DefaultSyntheticDataGenerator()

# Define custom XGBoost training data scenario based on actual contract
xgboost_training_scenario = {
    "name": "xgboost_binary_classification",
    "description": "Binary classification data for XGBoost training",
    "data_config": {
        "num_records": 50000,
        "features": {
            "numerical_features": ["age", "income", "credit_score", "debt_ratio"],
            "categorical_features": ["region", "product_type", "customer_segment"],
            "target_feature": "default_flag"  # Binary classification target
        },
        "data_structure": {
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15
        },
        "output_paths": {
            "input_path": "/opt/ml/input/data",  # Training data with train/val/test subdirs
            "hyperparameters_s3_uri": "/opt/ml/input/data/config/hyperparameters.json"
        },
        "hyperparameters": {
            "is_binary": True,
            "eta": 0.1,
            "max_depth": 6,
            "num_round": 100,
            "early_stopping_rounds": 10,
            "tab_field_list": ["age", "income", "credit_score", "debt_ratio"],
            "cat_field_list": ["region", "product_type", "customer_segment"],
            "label_name": "default_flag",
            "id_name": "customer_id"
        }
    }
}

# Generate custom test data for XGBoost Training
training_data = data_generator.generate_scenario_data(xgboost_training_scenario)

# Test XGBoost Training script with custom data
training_result = tester.test_script_with_data(
    script_name="XGBoostTraining",
    test_data=training_data,
    scenario_name="xgboost_binary_classification"
)

# Now test the data flow: XGBoost Training -> Model Evaluation
if training_result.is_successful():
    # Extract model artifacts from training output
    model_artifacts = training_result.outputs["model_output"]  # /opt/ml/model
    evaluation_data = training_result.outputs["evaluation_output"]  # /opt/ml/output/data
    
    # Define evaluation scenario using training outputs
    evaluation_scenario = {
        "name": "xgboost_model_evaluation",
        "description": "Model evaluation using XGBoost training outputs",
        "data_config": {
            "input_paths": {
                "model_input": model_artifacts,  # Model artifacts from training
                "processed_data": evaluation_data  # Evaluation data from training
            },
            "environment_vars": {
                "ID_FIELD": "customer_id",
                "LABEL_FIELD": "default_flag"
            },
            "expected_outputs": {
                "eval_output": "/opt/ml/processing/output/eval/eval_predictions.csv",
                "metrics_output": "/opt/ml/processing/output/metrics/metrics.json"
            }
        }
    }
    
    # Test XGBoost Model Evaluation with training outputs
    evaluation_result = tester.test_script_with_data(
        script_name="XGBoostModelEval",
        test_data=evaluation_scenario,
        scenario_name="xgboost_model_evaluation"
    )
    
    # Validate the complete data flow
    if evaluation_result.is_successful():
        print("✅ Complete XGBoost Training -> Evaluation pipeline validated!")
        print(f"Training time: {training_result.execution_time:.2f}s")
        print(f"Evaluation time: {evaluation_result.execution_time:.2f}s")
        print(f"Model artifacts: {model_artifacts}")
        print(f"Evaluation metrics: {evaluation_result.outputs['metrics_output']}")
```

### 2. Interactive Data Exploration

#### Data Quality Analysis
```python
# Interactive data quality exploration
from cursus.validation.runtime.widgets import DataQualityExplorer

quality_explorer = DataQualityExplorer(training_data)
quality_explorer.display()

# Features:
# - Statistical summaries
# - Missing value analysis
# - Distribution visualizations
# - Correlation matrices
# - Outlier detection
```

#### Feature Engineering Validation
```python
# Validate feature engineering steps
from cursus.validation.runtime.widgets import FeatureEngineeringValidator

feature_validator = FeatureEngineeringValidator()
feature_validator.add_before_after_data(raw_data, processed_data)
feature_validator.display()

# Features:
# - Before/after comparisons
# - Feature importance changes
# - Data distribution shifts
# - Quality metric changes
```

## Notebook Templates

### 1. Script Testing Template

```python
# Pipeline Runtime Testing Template
# =================================

# 1. Setup
from cursus.validation.runtime import PipelineTestingNotebook
tester = PipelineTestingNotebook()

# 2. Define test parameters
SCRIPT_NAME = "your_script_name"
SCENARIOS = ["standard", "edge_cases"]
DATA_SOURCE = "synthetic"

# 3. Run basic test
result = tester.test_script(
    script_name=SCRIPT_NAME,
    scenarios=SCENARIOS,
    data_source=DATA_SOURCE
)

# 4. Display results
result.show_summary()
result.visualize_performance()

# 5. Analyze results
if not result.is_successful():
    result.show_error_analysis()
    result.show_recommendations()

# 6. Export results (optional)
result.export_report("./test_results")
```

### 2. Pipeline Testing Template

```python
# Pipeline Testing Template
# =========================

# 1. Setup
from cursus.validation.runtime import quick_test_pipeline
from cursus.validation.runtime.widgets import DataFlowVisualizer

# 2. Define pipeline parameters
PIPELINE_NAME = "your_pipeline_name"
VALIDATION_LEVEL = "strict"

# 3. Run pipeline test
result = quick_test_pipeline(
    pipeline_name=PIPELINE_NAME,
    validation_level=VALIDATION_LEVEL,
    save_intermediate_results=True
)

# 4. Visualize pipeline execution
flow_viz = DataFlowVisualizer(result)
flow_viz.display()

# 5. Analyze step-by-step results
for step_name, step_result in result.step_results.items():
    print(f"\n=== {step_name} ===")
    step_result.show_summary()
    
    if not step_result.is_successful():
        step_result.show_error_details()

# 6. Data flow validation
if not result.data_flow_validation.is_valid():
    print("Data flow validation failed:")
    for error in result.data_flow_validation.errors:
        print(f"  - {error}")
```

### 3. Comparative Analysis Template

```python
# Comparative Analysis Template
# =============================

# 1. Setup
from cursus.validation.runtime import batch_test_scripts
from cursus.validation.runtime.widgets import PerformanceDashboard

# 2. Define comparison scenarios
scenarios = [
    {"name": "baseline", "data_size": "medium", "timeout": 300},
    {"name": "optimized", "data_size": "medium", "timeout": 300, "memory_limit": "2GB"},
    {"name": "large_scale", "data_size": "large", "timeout": 600, "memory_limit": "4GB"}
]

# 3. Run comparative tests
results = {}
for scenario in scenarios:
    result = tester.test_script(
        script_name="your_script",
        **scenario
    )
    results[scenario["name"]] = result

# 4. Create comparison dashboard
dashboard = PerformanceDashboard()
for name, result in results.items():
    dashboard.add_result(name, result)

dashboard.display()

# 5. Statistical comparison
comparison = tester.compare_results(list(results.values()))
comparison.show_statistical_analysis()
comparison.identify_significant_differences()
```

This comprehensive Jupyter notebook guide provides interactive examples and templates for effective pipeline runtime testing in notebook environments.
