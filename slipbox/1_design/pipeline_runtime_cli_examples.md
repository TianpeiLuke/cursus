---
tags:
  - design
  - testing
  - cli
  - command_line
  - usage_examples
keywords:
  - CLI usage
  - command line
  - testing commands
  - batch processing
topics:
  - CLI interface
  - command line testing
  - batch operations
language: bash
date of note: 2025-08-21
---

# Pipeline Runtime Testing - CLI Usage Examples

## Overview

This document provides comprehensive CLI usage examples for the Pipeline Runtime Testing System. It demonstrates command-line testing, batch operations, and automation capabilities for various testing scenarios.

## CLI Usage Examples

### 1. Basic CLI Commands

#### Script Testing
```bash
# Test single script
cursus runtime test-script currency_conversion

# Test with specific scenarios
cursus runtime test-script currency_conversion \
    --scenarios standard,edge_cases,performance \
    --data-source synthetic \
    --data-size large \
    --output-dir ./test_results

# Test with timeout and memory limits
cursus runtime test-script xgboost_training \
    --timeout 600 \
    --memory-limit 2GB \
    --save-intermediate-results
```

#### Pipeline Testing
```bash
# Test complete pipeline
cursus runtime test-pipeline xgb_training_simple

# Test with custom configuration
cursus runtime test-pipeline xgb_training_simple \
    --data-source synthetic \
    --validation-level strict \
    --execution-mode sequential \
    --output-dir ./pipeline_results \
    --generate-report

# Test with real S3 data
cursus runtime test-pipeline xgb_training_simple \
    --s3-execution-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345 \
    --analysis-scope sample \
    --sample-size 10000
```

### 2. Advanced CLI Usage

#### Batch Testing
```bash
# Test multiple scripts
cursus runtime batch-test \
    --scripts currency_conversion,tabular_preprocessing,xgboost_training \
    --scenarios standard,edge_cases \
    --parallel \
    --max-workers 4 \
    --output-dir ./batch_results

# Test all scripts in pipeline
cursus runtime test-all-pipeline-scripts xgb_training_simple \
    --exclude model_registration \
    --scenarios standard \
    --generate-summary-report
```

#### Performance Analysis
```bash
# Performance benchmarking
cursus runtime benchmark-script xgboost_training \
    --data-volumes small,medium,large \
    --iterations 5 \
    --output-format json \
    --save-performance-data

# Memory profiling
cursus runtime profile-memory currency_conversion \
    --data-size large \
    --detailed-analysis \
    --export-profile ./memory_profile.json
```

#### Deep Dive Analysis
```bash
# Deep dive with S3 data
cursus runtime deep-dive-analysis \
    --pipeline xgb_training_simple \
    --s3-execution-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345 \
    --focus-areas performance,data_quality \
    --generate-recommendations \
    --output-dir ./deep_dive_results

# Compare executions
cursus runtime compare-executions \
    --baseline-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345 \
    --comparison-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12346 \
    --metrics execution_time,data_quality,resource_usage \
    --generate-comparison-report
```

### 3. Configuration and Customization

#### Configuration Files
```bash
# Use configuration file
cursus runtime test-pipeline xgb_training_simple \
    --config ./test_config.yaml \
    --override data_source=synthetic

# Generate configuration template
cursus runtime generate-config \
    --template pipeline-test \
    --output ./pipeline_test_config.yaml

# Validate configuration
cursus runtime validate-config ./test_config.yaml
```

#### Custom Test Scenarios
```bash
# Create custom test scenario
cursus runtime create-scenario \
    --name custom_large_data \
    --base-scenario standard \
    --data-size large \
    --memory-limit 4GB \
    --timeout 1800

# List available scenarios
cursus runtime list-scenarios

# Test with custom scenario
cursus runtime test-script currency_conversion \
    --scenarios custom_large_data
```

## Detailed Command Reference

### 1. Script Testing Commands

#### `test-script` Command
```bash
cursus runtime test-script [SCRIPT_NAME] [OPTIONS]

# Required Arguments:
#   SCRIPT_NAME    Name of the script to test

# Options:
#   --scenarios TEXT           Comma-separated list of test scenarios
#   --data-source TEXT         Data source: synthetic, s3, local
#   --data-size TEXT           Data size: small, medium, large, xlarge
#   --timeout INTEGER          Timeout in seconds (default: 300)
#   --memory-limit TEXT        Memory limit (e.g., 1GB, 2048MB)
#   --output-dir PATH          Output directory for results
#   --output-format TEXT       Output format: json, yaml, html, text
#   --save-intermediate-results Save intermediate data and results
#   --enable-profiling         Enable performance profiling
#   --verbose                  Enable verbose output
#   --quiet                    Suppress non-essential output
#   --help                     Show help message

# Examples:
cursus runtime test-script currency_conversion \
    --scenarios standard,edge_cases \
    --data-source synthetic \
    --data-size medium \
    --timeout 300 \
    --output-dir ./results \
    --verbose

cursus runtime test-script xgboost_training \
    --scenarios performance \
    --data-size large \
    --memory-limit 4GB \
    --timeout 1200 \
    --enable-profiling \
    --save-intermediate-results
```

#### `batch-test` Command
```bash
cursus runtime batch-test [OPTIONS]

# Options:
#   --scripts TEXT             Comma-separated list of script names
#   --scenarios TEXT           Comma-separated list of scenarios
#   --parallel                 Enable parallel execution
#   --max-workers INTEGER      Maximum number of parallel workers
#   --timeout-per-script INTEGER  Timeout per script in seconds
#   --output-dir PATH          Output directory for batch results
#   --generate-summary-report  Generate summary report
#   --continue-on-failure      Continue testing even if some scripts fail
#   --export-raw-data          Export raw test data
#   --help                     Show help message

# Examples:
cursus runtime batch-test \
    --scripts currency_conversion,tabular_preprocessing,xgboost_training \
    --scenarios standard,edge_cases \
    --parallel \
    --max-workers 4 \
    --timeout-per-script 600 \
    --output-dir ./batch_results \
    --generate-summary-report

cursus runtime batch-test \
    --scripts all \
    --scenarios standard \
    --continue-on-failure \
    --export-raw-data
```

### 2. Pipeline Testing Commands

#### `test-pipeline` Command
```bash
cursus runtime test-pipeline [PIPELINE_NAME] [OPTIONS]

# Required Arguments:
#   PIPELINE_NAME    Name of the pipeline to test

# Options:
#   --data-source TEXT         Data source: synthetic, s3, local
#   --validation-level TEXT    Validation level: lenient, standard, strict
#   --execution-mode TEXT      Execution mode: sequential, parallel
#   --continue-on-failure      Continue execution on step failures
#   --max-parallel-steps INTEGER  Maximum parallel steps
#   --save-intermediate-results Save intermediate results
#   --output-dir PATH          Output directory for results
#   --generate-report          Generate HTML report
#   --s3-execution-arn TEXT    S3 execution ARN for real data analysis
#   --analysis-scope TEXT      Analysis scope: sample, full
#   --sample-size INTEGER      Sample size for analysis
#   --help                     Show help message

# Examples:
cursus runtime test-pipeline xgb_training_simple \
    --data-source synthetic \
    --validation-level strict \
    --execution-mode sequential \
    --save-intermediate-results \
    --generate-report \
    --output-dir ./pipeline_results

cursus runtime test-pipeline xgb_training_with_eval \
    --s3-execution-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345 \
    --analysis-scope sample \
    --sample-size 50000 \
    --output-dir ./s3_analysis
```

#### `test-all-pipeline-scripts` Command
```bash
cursus runtime test-all-pipeline-scripts [PIPELINE_NAME] [OPTIONS]

# Required Arguments:
#   PIPELINE_NAME    Name of the pipeline

# Options:
#   --include TEXT             Comma-separated list of scripts to include
#   --exclude TEXT             Comma-separated list of scripts to exclude
#   --scenarios TEXT           Test scenarios to run
#   --parallel                 Enable parallel testing
#   --generate-summary-report  Generate summary report
#   --output-dir PATH          Output directory
#   --help                     Show help message

# Examples:
cursus runtime test-all-pipeline-scripts xgb_training_simple \
    --exclude model_registration,data_validation \
    --scenarios standard,edge_cases \
    --parallel \
    --generate-summary-report

cursus runtime test-all-pipeline-scripts complex_ml_pipeline \
    --include preprocessing,training,evaluation \
    --scenarios performance \
    --output-dir ./comprehensive_test
```

### 3. Analysis and Comparison Commands

#### `benchmark-script` Command
```bash
cursus runtime benchmark-script [SCRIPT_NAME] [OPTIONS]

# Required Arguments:
#   SCRIPT_NAME    Name of the script to benchmark

# Options:
#   --data-volumes TEXT        Comma-separated data volumes to test
#   --iterations INTEGER       Number of benchmark iterations
#   --warmup-iterations INTEGER  Number of warmup iterations
#   --output-format TEXT       Output format: json, csv, html
#   --save-performance-data    Save detailed performance data
#   --compare-with-baseline    Compare with baseline performance
#   --baseline-file PATH       Baseline performance file
#   --help                     Show help message

# Examples:
cursus runtime benchmark-script xgboost_training \
    --data-volumes small,medium,large,xlarge \
    --iterations 5 \
    --warmup-iterations 2 \
    --output-format json \
    --save-performance-data \
    --output-dir ./benchmarks

cursus runtime benchmark-script currency_conversion \
    --data-volumes medium,large \
    --iterations 10 \
    --compare-with-baseline \
    --baseline-file ./baseline_performance.json
```

#### `compare-executions` Command
```bash
cursus runtime compare-executions [OPTIONS]

# Options:
#   --baseline-arn TEXT        Baseline execution ARN
#   --comparison-arn TEXT      Comparison execution ARN
#   --baseline-file PATH       Baseline test results file
#   --comparison-file PATH     Comparison test results file
#   --metrics TEXT             Comma-separated metrics to compare
#   --generate-comparison-report  Generate detailed comparison report
#   --output-dir PATH          Output directory
#   --statistical-analysis     Include statistical analysis
#   --help                     Show help message

# Examples:
cursus runtime compare-executions \
    --baseline-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345 \
    --comparison-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12346 \
    --metrics execution_time,memory_usage,data_quality \
    --generate-comparison-report \
    --statistical-analysis

cursus runtime compare-executions \
    --baseline-file ./baseline_results.json \
    --comparison-file ./new_results.json \
    --metrics all \
    --output-dir ./comparison_analysis
```

#### `deep-dive-analysis` Command
```bash
cursus runtime deep-dive-analysis [OPTIONS]

# Options:
#   --pipeline TEXT            Pipeline name
#   --script TEXT              Script name (for single script analysis)
#   --s3-execution-arn TEXT    S3 execution ARN
#   --focus-areas TEXT         Comma-separated focus areas
#   --analysis-scope TEXT      Analysis scope: sample, full
#   --sample-size INTEGER      Sample size for analysis
#   --generate-recommendations Generate optimization recommendations
#   --include-visualizations   Include performance visualizations
#   --output-dir PATH          Output directory
#   --help                     Show help message

# Examples:
cursus runtime deep-dive-analysis \
    --pipeline xgb_training_simple \
    --s3-execution-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345 \
    --focus-areas performance,data_quality,error_patterns \
    --analysis-scope full \
    --generate-recommendations \
    --include-visualizations

cursus runtime deep-dive-analysis \
    --script xgboost_training \
    --s3-execution-arn arn:aws:sagemaker:us-east-1:123456789012:pipeline/prod/execution/12345 \
    --focus-areas memory_usage,execution_time \
    --sample-size 25000
```

### 4. Configuration Management Commands

#### `generate-config` Command
```bash
cursus runtime generate-config [OPTIONS]

# Options:
#   --template TEXT            Configuration template type
#   --output PATH              Output configuration file path
#   --script TEXT              Script name for script-specific config
#   --pipeline TEXT            Pipeline name for pipeline-specific config
#   --include-examples         Include example configurations
#   --help                     Show help message

# Available Templates:
#   - basic-test: Basic testing configuration
#   - pipeline-test: Pipeline testing configuration
#   - performance-test: Performance testing configuration
#   - batch-test: Batch testing configuration
#   - deep-dive: Deep dive analysis configuration

# Examples:
cursus runtime generate-config \
    --template pipeline-test \
    --pipeline xgb_training_simple \
    --output ./xgb_test_config.yaml \
    --include-examples

cursus runtime generate-config \
    --template performance-test \
    --script xgboost_training \
    --output ./performance_config.yaml
```

#### `validate-config` Command
```bash
cursus runtime validate-config [CONFIG_FILE] [OPTIONS]

# Required Arguments:
#   CONFIG_FILE    Path to configuration file

# Options:
#   --strict                   Enable strict validation
#   --show-warnings           Show validation warnings
#   --fix-issues              Attempt to fix common issues
#   --output-fixed PATH       Output path for fixed configuration
#   --help                    Show help message

# Examples:
cursus runtime validate-config ./test_config.yaml \
    --strict \
    --show-warnings

cursus runtime validate-config ./test_config.yaml \
    --fix-issues \
    --output-fixed ./test_config_fixed.yaml
```

#### `create-scenario` Command
```bash
cursus runtime create-scenario [OPTIONS]

# Options:
#   --name TEXT                Scenario name
#   --base-scenario TEXT       Base scenario to extend
#   --data-size TEXT           Data size for scenario
#   --memory-limit TEXT        Memory limit
#   --timeout INTEGER          Timeout in seconds
#   --description TEXT         Scenario description
#   --save-global              Save as global scenario
#   --output-dir PATH          Output directory for scenario definition
#   --help                     Show help message

# Examples:
cursus runtime create-scenario \
    --name high_performance_test \
    --base-scenario standard \
    --data-size xlarge \
    --memory-limit 8GB \
    --timeout 3600 \
    --description "High performance testing with large datasets" \
    --save-global

cursus runtime create-scenario \
    --name memory_constrained \
    --base-scenario edge_cases \
    --memory-limit 512MB \
    --timeout 1800 \
    --description "Testing under memory constraints"
```

### 5. Utility Commands

#### `list-scenarios` Command
```bash
cursus runtime list-scenarios [OPTIONS]

# Options:
#   --show-details            Show detailed scenario information
#   --filter-by-type TEXT     Filter by scenario type
#   --output-format TEXT      Output format: table, json, yaml
#   --help                    Show help message

# Examples:
cursus runtime list-scenarios --show-details
cursus runtime list-scenarios --filter-by-type performance --output-format json
```

#### `list-scripts` Command
```bash
cursus runtime list-scripts [OPTIONS]

# Options:
#   --pipeline TEXT           Filter by pipeline
#   --show-dependencies       Show script dependencies
#   --output-format TEXT      Output format: table, json, yaml
#   --help                    Show help message

# Examples:
cursus runtime list-scripts --pipeline xgb_training_simple --show-dependencies
cursus runtime list-scripts --output-format json
```

#### `profile-memory` Command
```bash
cursus runtime profile-memory [SCRIPT_NAME] [OPTIONS]

# Required Arguments:
#   SCRIPT_NAME    Name of the script to profile

# Options:
#   --data-size TEXT           Data size for profiling
#   --detailed-analysis        Enable detailed memory analysis
#   --track-allocations        Track memory allocations
#   --export-profile PATH      Export profile data
#   --generate-report          Generate memory profiling report
#   --help                     Show help message

# Examples:
cursus runtime profile-memory xgboost_training \
    --data-size large \
    --detailed-analysis \
    --track-allocations \
    --export-profile ./memory_profile.json \
    --generate-report

cursus runtime profile-memory currency_conversion \
    --data-size medium \
    --export-profile ./currency_memory.json
```

## Automation and CI/CD Integration

### 1. GitHub Actions Integration

#### Basic Testing Workflow
```yaml
# .github/workflows/pipeline_testing.yml
name: Pipeline Runtime Tests

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
        
    - name: Run runtime tests
      run: |
        cursus runtime batch-test \
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
        name: Pipeline Runtime Tests
        path: test-results/*.xml
        reporter: java-junit
```

#### Performance Regression Testing
```yaml
# .github/workflows/performance_regression.yml
name: Performance Regression Tests

on:
  pull_request:
    branches: [ main ]

jobs:
  performance-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0  # Fetch full history for baseline comparison
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: pip install -e .
        
    - name: Run baseline performance tests
      run: |
        git checkout main
        cursus runtime benchmark-script xgboost_training \
          --data-volumes medium,large \
          --iterations 3 \
          --output-format json \
          --save-performance-data \
          --output-dir ./baseline_results
        
    - name: Run current performance tests
      run: |
        git checkout ${{ github.head_ref }}
        cursus runtime benchmark-script xgboost_training \
          --data-volumes medium,large \
          --iterations 3 \
          --output-format json \
          --save-performance-data \
          --output-dir ./current_results
        
    - name: Compare performance
      run: |
        cursus runtime compare-executions \
          --baseline-file ./baseline_results/benchmark_results.json \
          --comparison-file ./current_results/benchmark_results.json \
          --metrics execution_time,memory_usage \
          --generate-comparison-report \
          --statistical-analysis \
          --output-dir ./performance_comparison
        
    - name: Upload performance comparison
      uses: actions/upload-artifact@v3
      with:
        name: performance-comparison
        path: performance_comparison/
```

### 2. Jenkins Integration

#### Basic Pipeline
```groovy
// Jenkinsfile
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -e .'
            }
        }
        
        stage('Runtime Tests') {
            parallel {
                stage('Core Scripts') {
                    steps {
                        sh '''
                            cursus runtime batch-test \
                                --scripts currency_conversion,tabular_preprocessing \
                                --scenarios standard,edge_cases \
                                --output-dir ./core_results \
                                --generate-summary-report
                        '''
                    }
                }
                
                stage('ML Scripts') {
                    steps {
                        sh '''
                            cursus runtime batch-test \
                                --scripts xgboost_training,model_evaluation \
                                --scenarios standard,performance \
                                --output-dir ./ml_results \
                                --generate-summary-report
                        '''
                    }
                }
            }
        }
        
        stage('Pipeline Integration Tests') {
            steps {
                sh '''
                    cursus runtime test-pipeline xgb_training_simple \
                        --validation-level strict \
                        --generate-report \
                        --output-dir ./pipeline_results
                '''
            }
        }
        
        stage('Performance Analysis') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    cursus runtime benchmark-script xgboost_training \
                        --data-volumes medium,large \
                        --iterations 5 \
                        --save-performance-data \
                        --output-dir ./performance_results
                '''
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: '*_results/**/*', fingerprint: true
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'pipeline_results',
                reportFiles: 'report.html',
                reportName: 'Pipeline Test Report'
            ])
        }
        
        failure {
            emailext (
                subject: "Pipeline Runtime Tests Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Pipeline runtime tests failed. Check the build logs for details.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

### 3. Scheduled Testing

#### Cron-based Testing Script
```bash
#!/bin/bash
# scheduled_testing.sh

# Daily comprehensive testing
if [ "$(date +%H)" = "02" ]; then
    echo "Running daily comprehensive tests..."
    
    # Test all scripts with multiple scenarios
    cursus runtime batch-test \
        --scripts all \
        --scenarios standard,edge_cases,performance \
        --parallel \
        --max-workers 6 \
        --output-dir ./daily_results/$(date +%Y-%m-%d) \
        --generate-summary-report
    
    # Deep dive analysis with production data
    cursus runtime deep-dive-analysis \
        --pipeline xgb_training_simple \
        --s3-execution-arn $(get_latest_production_arn) \
        --focus-areas performance,data_quality \
        --generate-recommendations \
        --output-dir ./daily_results/$(date +%Y-%m-%d)/deep_dive
fi

# Weekly performance benchmarking
if [ "$(date +%u)" = "1" ] && [ "$(date +%H)" = "03" ]; then
    echo "Running weekly performance benchmarks..."
    
    cursus runtime benchmark-script xgboost_training \
        --data-volumes small,medium,large,xlarge \
        --iterations 10 \
        --output-format json \
        --save-performance-data \
        --output-dir ./weekly_benchmarks/$(date +%Y-W%V)
    
    # Compare with previous week
    if [ -d "./weekly_benchmarks/$(date -d '7 days ago' +%Y-W%V)" ]; then
        cursus runtime compare-executions \
            --baseline-file ./weekly_benchmarks/$(date -d '7 days ago' +%Y-W%V)/benchmark_results.json \
            --comparison-file ./weekly_benchmarks/$(date +%Y-W%V)/benchmark_results.json \
            --metrics all \
            --generate-comparison-report \
            --statistical-analysis \
            --output-dir ./weekly_benchmarks/$(date +%Y-W%V)/comparison
    fi
fi
```

This comprehensive CLI guide provides detailed command references and automation examples for effective pipeline runtime testing from the command line.
