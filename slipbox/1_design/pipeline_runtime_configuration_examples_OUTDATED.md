---
tags:
  - design
  - testing
  - configuration
  - yaml
  - examples
keywords:
  - configuration examples
  - YAML configuration
  - test scenarios
  - environment setup
topics:
  - configuration management
  - test scenarios
  - environment configuration
language: yaml
date of note: 2025-08-21
---

# Pipeline Runtime Testing - Configuration Examples

## Overview

This document provides comprehensive configuration examples for the Pipeline Runtime Testing System. It demonstrates various configuration patterns, environment-specific setups, and advanced customization options.

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

#### CI/CD Environment
```yaml
# ci_config.yaml
environment: "ci_cd"

test_configuration:
  workspace_dir: "/tmp/ci_testing"
  default_data_source: "synthetic"
  
  isolation_testing:
    timeout_seconds: 300
    memory_limit_mb: 1024
    save_intermediate_results: false  # Save space in CI
    
  pipeline_testing:
    execution_mode: "parallel"
    validation_level: "standard"
    continue_on_failure: true  # Continue to get full test results
    
  performance_testing:
    enable_profiling: false  # Skip profiling in CI for speed
    benchmark_iterations: 1
    
  quality_gates:
    execution_time_max: 240  # Stricter time limits for CI
    memory_usage_max: 1024
    success_rate_min: 0.95
    
  reporting:
    generate_html_reports: false  # Skip HTML reports in CI
    export_raw_data: true  # Export for analysis
    report_format: "junit"  # CI-friendly format
```

### 3. Script-Specific Configurations

#### XGBoost Training Configuration
```yaml
# xgboost_training_config.yaml
script_name: "XGBoostTraining"

test_configuration:
  scenarios:
    - name: "binary_classification"
      description: "Binary classification with balanced data"
      data_config:
        num_records: 50000
        target_distribution: [0.5, 0.5]
        features:
          numerical: ["age", "income", "credit_score", "debt_ratio"]
          categorical: ["region", "product_type", "customer_segment"]
        hyperparameters:
          is_binary: true
          eta: 0.1
          max_depth: 6
          num_round: 100
          early_stopping_rounds: 10
    
    - name: "imbalanced_classification"
      description: "Binary classification with imbalanced data"
      data_config:
        num_records: 75000
        target_distribution: [0.9, 0.1]  # Imbalanced
        features:
          numerical: ["age", "income", "credit_score", "debt_ratio"]
          categorical: ["region", "product_type", "customer_segment"]
        hyperparameters:
          is_binary: true
          eta: 0.05  # Lower learning rate for imbalanced data
          max_depth: 8
          num_round: 200
          early_stopping_rounds: 15
    
    - name: "large_scale"
      description: "Large scale training test"
      data_config:
        num_records: 200000
        features:
          numerical: ["age", "income", "credit_score", "debt_ratio", "employment_years"]
          categorical: ["region", "product_type", "customer_segment", "education"]
        hyperparameters:
          is_binary: true
          eta: 0.1
          max_depth: 6
          num_round: 150
          early_stopping_rounds: 10
      
  resource_limits:
    timeout_seconds: 1800  # 30 minutes for large scale
    memory_limit_mb: 4096
    
  validation_rules:
    - name: "model_quality"
      type: "custom"
      parameters:
        min_auc: 0.7
        max_training_time: 1200
    - name: "memory_efficiency"
      type: "resource"
      parameters:
        max_memory_mb: 3072
```

#### Currency Conversion Configuration
```yaml
# currency_conversion_config.yaml
script_name: "currency_conversion"

test_configuration:
  scenarios:
    - name: "standard_currencies"
      description: "Standard currency conversion test"
      data_config:
        currencies: ["USD", "EUR", "GBP", "JPY", "CAD"]
        num_transactions: 10000
        date_range: "2023-01-01,2023-12-31"
        
    - name: "exotic_currencies"
      description: "Test with exotic currencies"
      data_config:
        currencies: ["USD", "EUR", "THB", "ZAR", "BRL", "INR"]
        num_transactions: 5000
        date_range: "2023-06-01,2023-12-31"
        
    - name: "historical_rates"
      description: "Historical exchange rate testing"
      data_config:
        currencies: ["USD", "EUR", "GBP"]
        num_transactions: 15000
        date_range: "2020-01-01,2023-12-31"
        include_weekends: false
        
    - name: "edge_cases"
      description: "Edge case testing"
      data_config:
        currencies: ["USD", "EUR", "GBP", "JPY"]
        num_transactions: 1000
        include_edge_cases:
          - zero_amounts
          - negative_amounts
          - very_large_amounts
          - invalid_dates
          - missing_rates
  
  validation_rules:
    - name: "conversion_accuracy"
      type: "custom"
      parameters:
        max_error_rate: 0.001
        precision_digits: 4
    - name: "performance"
      type: "performance"
      parameters:
        max_execution_time: 60
        max_memory_mb: 512
```

### 4. Pipeline-Specific Configurations

#### XGBoost Training Pipeline Configuration
```yaml
# xgb_pipeline_config.yaml
pipeline_name: "xgb_training_with_eval"

test_configuration:
  pipeline_steps:
    - step_name: "data_preprocessing"
      scenarios: ["standard", "large_volume"]
      timeout_seconds: 300
      
    - step_name: "XGBoostTraining"
      scenarios: ["binary_classification", "imbalanced_classification"]
      timeout_seconds: 1200
      memory_limit_mb: 4096
      
    - step_name: "XGBoostModelEval"
      scenarios: ["standard_evaluation"]
      timeout_seconds: 600
      depends_on: ["XGBoostTraining"]
  
  data_flow_validation:
    strict_mode: true
    validate_schemas: true
    validate_data_quality: true
    
  execution_strategy:
    mode: "sequential"  # Training must complete before evaluation
    continue_on_failure: false
    save_checkpoints: true
    
  quality_gates:
    pipeline_success_rate: 0.95
    max_total_execution_time: 2400  # 40 minutes
    data_flow_validation_required: true
```

#### Complete ML Pipeline Configuration
```yaml
# complete_ml_pipeline_config.yaml
pipeline_name: "complete_ml_pipeline"

test_configuration:
  pipeline_steps:
    - step_name: "data_ingestion"
      scenarios: ["standard", "large_volume"]
      timeout_seconds: 600
      parallel_group: 1
      
    - step_name: "data_validation"
      scenarios: ["standard", "edge_cases"]
      timeout_seconds: 300
      parallel_group: 1
      depends_on: ["data_ingestion"]
      
    - step_name: "feature_engineering"
      scenarios: ["standard", "advanced_features"]
      timeout_seconds: 900
      parallel_group: 2
      depends_on: ["data_validation"]
      
    - step_name: "model_training"
      scenarios: ["xgboost", "random_forest"]
      timeout_seconds: 1800
      memory_limit_mb: 8192
      parallel_group: 3
      depends_on: ["feature_engineering"]
      
    - step_name: "model_evaluation"
      scenarios: ["comprehensive_eval"]
      timeout_seconds: 600
      parallel_group: 4
      depends_on: ["model_training"]
      
    - step_name: "model_registration"
      scenarios: ["standard"]
      timeout_seconds: 300
      parallel_group: 4
      depends_on: ["model_evaluation"]
  
  execution_strategy:
    mode: "parallel"
    max_parallel_groups: 2
    continue_on_failure: false
    retry_failed_steps: true
    max_retries: 2
    
  resource_management:
    total_memory_limit_mb: 16384
    total_cpu_cores: 8
    disk_space_limit_gb: 100
    
  monitoring:
    enable_real_time_monitoring: true
    alert_on_resource_threshold: 0.8
    log_level: "INFO"
```

### 5. Performance Testing Configurations

#### Benchmark Configuration
```yaml
# benchmark_config.yaml
benchmark_configuration:
  scripts_to_benchmark:
    - script_name: "xgboost_training"
      data_volumes: ["small", "medium", "large", "xlarge"]
      iterations: 5
      warmup_iterations: 2
      
    - script_name: "currency_conversion"
      data_volumes: ["medium", "large"]
      iterations: 10
      warmup_iterations: 1
      
    - script_name: "tabular_preprocessing"
      data_volumes: ["small", "medium", "large"]
      iterations: 3
      warmup_iterations: 1
  
  data_volume_definitions:
    small:
      num_records: 1000
      memory_limit_mb: 256
      
    medium:
      num_records: 10000
      memory_limit_mb: 512
      
    large:
      num_records: 100000
      memory_limit_mb: 2048
      
    xlarge:
      num_records: 500000
      memory_limit_mb: 8192
  
  performance_metrics:
    - execution_time
    - peak_memory_usage
    - cpu_utilization
    - disk_io
    - network_io
    
  baseline_comparison:
    enabled: true
    baseline_file: "./baselines/performance_baseline.json"
    alert_threshold: 0.2  # 20% performance degradation
    
  reporting:
    generate_charts: true
    export_raw_data: true
    include_statistical_analysis: true
```

#### Load Testing Configuration
```yaml
# load_test_config.yaml
load_test_configuration:
  test_scenarios:
    - name: "concurrent_script_execution"
      description: "Test multiple scripts running concurrently"
      concurrent_scripts: 4
      scripts:
        - "currency_conversion"
        - "tabular_preprocessing"
        - "xgboost_training"
        - "model_evaluation"
      duration_minutes: 30
      
    - name: "pipeline_stress_test"
      description: "Stress test complete pipeline"
      concurrent_pipelines: 2
      pipeline_name: "xgb_training_simple"
      duration_minutes: 60
      
    - name: "memory_pressure_test"
      description: "Test under memory pressure"
      memory_limit_mb: 1024  # Constrained memory
      scripts: ["xgboost_training"]
      data_size: "large"
      iterations: 10
  
  resource_monitoring:
    monitor_interval_seconds: 30
    metrics:
      - cpu_usage
      - memory_usage
      - disk_usage
      - network_usage
      - process_count
      
  failure_conditions:
    max_failure_rate: 0.1  # 10% failure rate
    max_response_time: 1800  # 30 minutes
    min_success_rate: 0.9
    
  recovery_testing:
    simulate_failures: true
    failure_types:
      - "memory_exhaustion"
      - "disk_full"
      - "network_timeout"
    recovery_time_limit: 300  # 5 minutes
```

### 6. Data Generation Configurations

#### Synthetic Data Configuration
```yaml
# synthetic_data_config.yaml
synthetic_data_configuration:
  default_generators:
    tabular_data:
      generator_class: "TabularDataGenerator"
      default_params:
        num_records: 10000
        num_features: 20
        categorical_ratio: 0.3
        missing_value_ratio: 0.05
        
    time_series:
      generator_class: "TimeSeriesDataGenerator"
      default_params:
        num_points: 1000
        seasonality: true
        trend: true
        noise_level: 0.1
        
    financial_data:
      generator_class: "FinancialDataGenerator"
      default_params:
        num_transactions: 50000
        currencies: ["USD", "EUR", "GBP", "JPY"]
        date_range: "2023-01-01,2023-12-31"
  
  custom_scenarios:
    xgboost_binary_classification:
      generator: "tabular_data"
      params:
        num_records: 50000
        features:
          numerical: 
            - name: "age"
              distribution: "normal"
              mean: 35
              std: 10
              min: 18
              max: 80
            - name: "income"
              distribution: "lognormal"
              mean: 50000
              std: 20000
              min: 20000
              max: 200000
            - name: "credit_score"
              distribution: "normal"
              mean: 650
              std: 100
              min: 300
              max: 850
          categorical:
            - name: "region"
              values: ["North", "South", "East", "West"]
              probabilities: [0.3, 0.25, 0.25, 0.2]
            - name: "product_type"
              values: ["A", "B", "C"]
              probabilities: [0.5, 0.3, 0.2]
          target:
            name: "default_flag"
            type: "binary"
            correlation_features: ["credit_score", "income"]
            
    currency_conversion_data:
      generator: "financial_data"
      params:
        num_transactions: 25000
        currencies: ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
        transaction_types: ["spot", "forward", "swap"]
        amount_range: [100, 1000000]
        date_range: "2023-01-01,2023-12-31"
        include_weekends: false
        volatility_level: "medium"
```

### 7. Error Handling and Recovery Configurations

#### Error Handling Configuration
```yaml
# error_handling_config.yaml
error_handling_configuration:
  retry_policies:
    default:
      max_retries: 3
      retry_delay_seconds: 5
      exponential_backoff: true
      retryable_errors:
        - "TIMEOUT"
        - "RESOURCE_UNAVAILABLE"
        - "NETWORK_ERROR"
        - "TEMPORARY_FAILURE"
        
    memory_errors:
      max_retries: 2
      retry_delay_seconds: 30
      cleanup_before_retry: true
      retryable_errors:
        - "OUT_OF_MEMORY"
        - "MEMORY_ALLOCATION_FAILED"
        
    data_errors:
      max_retries: 1
      regenerate_data: true
      retryable_errors:
        - "DATA_CORRUPTION"
        - "SCHEMA_MISMATCH"
  
  recovery_strategies:
    timeout_recovery:
      clear_caches: true
      reduce_data_size: true
      increase_timeout: 1.5  # 50% increase
      
    memory_recovery:
      force_garbage_collection: true
      reduce_batch_size: true
      enable_streaming: true
      
    resource_recovery:
      wait_for_resources: true
      max_wait_time: 300
      fallback_to_smaller_config: true
  
  error_reporting:
    capture_stack_traces: true
    save_error_context: true
    generate_error_reports: true
    alert_on_critical_errors: true
    
  graceful_degradation:
    enable_fallback_modes: true
    fallback_configurations:
      reduced_functionality:
        disable_profiling: true
        reduce_validation_level: "lenient"
        skip_non_essential_tests: true
      minimal_mode:
        run_only_critical_tests: true
        disable_reporting: true
        use_smallest_data_size: true
```

### 8. Integration Configurations

#### CI/CD Integration Configuration
```yaml
# ci_cd_integration_config.yaml
ci_cd_integration:
  github_actions:
    trigger_events:
      - push
      - pull_request
      - schedule  # Daily runs
      
    test_matrix:
      python_versions: ["3.8", "3.9", "3.10"]
      os_versions: ["ubuntu-latest", "windows-latest"]
      test_suites:
        - name: "smoke_tests"
          scripts: ["currency_conversion", "tabular_preprocessing"]
          scenarios: ["standard"]
          timeout_minutes: 10
          
        - name: "integration_tests"
          pipelines: ["xgb_training_simple"]
          scenarios: ["standard", "edge_cases"]
          timeout_minutes: 30
          
        - name: "performance_tests"
          scripts: ["xgboost_training"]
          scenarios: ["performance"]
          timeout_minutes: 60
          run_on: ["ubuntu-latest"]  # Only on Linux for performance
  
  jenkins:
    pipeline_stages:
      - name: "unit_tests"
        parallel: true
        scripts: ["currency_conversion", "tabular_preprocessing"]
        
      - name: "integration_tests"
        depends_on: ["unit_tests"]
        pipelines: ["xgb_training_simple", "complete_ml_pipeline"]
        
      - name: "performance_tests"
        depends_on: ["integration_tests"]
        trigger: "main_branch_only"
        scripts: ["xgboost_training"]
        benchmark: true
        
    notifications:
      on_failure:
        email: ["team@company.com"]
        slack: "#ml-pipeline-alerts"
      on_success:
        slack: "#ml-pipeline-status"
        
  docker_integration:
    base_images:
      - "python:3.9-slim"
      - "python:3.10-slim"
      
    test_environments:
      - name: "minimal"
        image: "python:3.9-slim"
        memory_limit: "1g"
        cpu_limit: "1"
        
      - name: "standard"
        image: "python:3.9"
        memory_limit: "4g"
        cpu_limit: "2"
        
      - name: "performance"
        image: "python:3.10"
        memory_limit: "8g"
        cpu_limit: "4"
```

This comprehensive configuration guide provides examples for various testing scenarios, environments, and integration patterns to support effective pipeline script testing across different contexts.
