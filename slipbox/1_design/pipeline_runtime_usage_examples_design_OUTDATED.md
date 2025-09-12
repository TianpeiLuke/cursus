---
tags:
  - design
  - testing
  - runtime
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

# Pipeline Runtime Testing - Usage Examples Master Index

## Overview

This document serves as the master index for comprehensive usage examples and API design specifications for the Pipeline Runtime Testing System. The content has been organized into focused, specialized documents for better navigation and maintainability.

## Document Structure

The usage examples and API documentation have been split into the following specialized documents:

### 📋 [API Design](./pipeline_runtime_api_design.md)
**Core API structure, design principles, and class specifications**
- API design philosophy and principles
- Core API classes and interfaces
- Configuration system architecture
- Result objects and error handling
- Extension points and customization

### 📓 [Jupyter Notebook Examples](./pipeline_runtime_jupyter_examples.md)
**Interactive testing examples optimized for notebook environments**
- Quick script and pipeline testing
- Interactive debugging and data inspection
- Parameterized testing with widgets
- Deep dive analysis with visualizations
- Custom test scenarios and templates

### 💻 [CLI Usage Examples](./pipeline_runtime_cli_examples.md)
**Command-line interface examples and automation**
- Basic and advanced CLI commands
- Batch testing and performance analysis
- Configuration management
- CI/CD integration examples
- Automation scripts and scheduled testing

### 🐍 [Python API Examples](./pipeline_runtime_python_api_examples.md)
**Programmatic usage and framework integration**
- Programmatic testing patterns
- Integration with pytest and unittest
- Advanced batch processing
- Performance analysis and profiling
- Error handling and recovery strategies

### ⚙️ [Configuration Examples](./pipeline_runtime_configuration_examples.md)
**Configuration patterns and environment setups**
- YAML configuration files
- Environment-specific configurations
- Script and pipeline-specific settings
- Performance and load testing configurations
- Error handling and integration configurations

## Quick Start Guide

For users new to the system, we recommend following this learning path:

1. **Start with API Design** - Understand the core concepts and design principles
2. **Try Jupyter Examples** - Get hands-on experience with interactive testing
3. **Explore CLI Usage** - Learn command-line automation and batch operations
4. **Dive into Python API** - Integrate testing into your development workflow
5. **Customize with Configuration** - Tailor the system to your specific needs

## Common Usage Patterns

### Quick Testing
```python
# One-liner script testing
from cursus.validation.runtime import quick_test_script
result = quick_test_script("currency_conversion")
```

### Pipeline Validation
```python
# End-to-end pipeline testing
from cursus.validation.runtime import quick_test_pipeline
result = quick_test_pipeline("xgb_training_with_eval")
```

### Batch Operations
```bash
# CLI batch testing
cursus runtime batch-test \
    --scripts currency_conversion,xgboost_training \
    --scenarios standard,edge_cases \
    --parallel
```

### Custom Configuration
```yaml
# YAML configuration
test_configuration:
  workspace_dir: "./pipeline_testing"
  default_data_source: "synthetic"
  quality_gates:
    execution_time_max: 300
    success_rate_min: 0.95
```

## Core API Overview

### Main API Classes
```python
from cursus.validation.runtime import (
    PipelineScriptExecutor,      # Core execution engine
    PipelineTestingNotebook,     # Jupyter-optimized interface
    RuntimeTester,               # High-level testing interface
    TestResultAnalyzer,          # Result analysis utilities
    TestConfigBuilder            # Configuration builder
)
```

### One-liner Functions
```python
from cursus.validation.runtime import (
    quick_test_script,           # Test single script
    quick_test_pipeline,         # Test complete pipeline
    deep_dive_analysis,          # Deep analysis with S3 data
    compare_executions,          # Compare pipeline executions
    batch_test_scripts           # Test multiple scripts
)
```

## Interface Overview

### Jupyter Notebooks
- **Interactive testing** with rich HTML displays
- **Parameterized testing** with hyperparameter exploration
- **Interactive debugging** with breakpoints and data inspection
- **Visualization widgets** for performance analysis
- **Deep dive analysis** with S3 data integration

*→ See [Jupyter Notebook Examples](./pipeline_runtime_jupyter_examples.md) for detailed examples*

### Command Line Interface
- **Basic commands** for script and pipeline testing
- **Batch operations** for testing multiple scripts
- **Performance analysis** and benchmarking
- **Configuration management** and validation
- **CI/CD integration** examples

*→ See [CLI Usage Examples](./pipeline_runtime_cli_examples.md) for comprehensive command reference*

### Python API
- **Programmatic testing** for automation
- **Testing framework integration** (pytest, unittest)
- **Advanced batch processing** with parallel execution
- **Custom validation rules** and data generators
- **Error handling and recovery** strategies

*→ See [Python API Examples](./pipeline_runtime_python_api_examples.md) for advanced usage patterns*

## Configuration Overview

### Environment Types
- **Development** - Fast iteration with lenient validation
- **Production** - Strict validation with comprehensive testing
- **CI/CD** - Optimized for automated testing pipelines

### Configuration Patterns
- **YAML files** for declarative configuration
- **Script-specific** settings for targeted testing
- **Pipeline-specific** configurations for end-to-end testing
- **Performance testing** configurations for benchmarking

*→ See [Configuration Examples](./pipeline_runtime_configuration_examples.md) for detailed configuration patterns*

## Getting Started

### 1. Basic Script Testing
```python
# Test a single script
result = quick_test_script("currency_conversion")
```

### 2. Pipeline Testing
```python
# Test complete pipeline
result = quick_test_pipeline("xgb_training_with_eval")
```

### 3. CLI Testing
```bash
# Command line testing
cursus runtime test-script currency_conversion
```

### 4. Configuration
```yaml
# Basic configuration
test_configuration:
  workspace_dir: "./testing"
  default_data_source: "synthetic"
```

## Next Steps

- **New users**: Start with [API Design](./pipeline_runtime_api_design.md) to understand core concepts
- **Jupyter users**: Explore [Jupyter Notebook Examples](./pipeline_runtime_jupyter_examples.md) for interactive testing
- **CLI users**: Check [CLI Usage Examples](./pipeline_runtime_cli_examples.md) for command-line automation
- **Developers**: Review [Python API Examples](./pipeline_runtime_python_api_examples.md) for programmatic integration
- **DevOps**: See [Configuration Examples](./pipeline_runtime_configuration_examples.md) for environment setup

For questions or advanced use cases, refer to the specialized documents linked above.
