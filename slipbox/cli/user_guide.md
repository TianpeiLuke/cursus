---
tags:
  - cli
  - user_guide
  - validation
  - alignment
  - development_tools
keywords:
  - CLI user guide
  - validation tools
  - alignment validation
  - development workflow
  - command line interface
topics:
  - CLI usage
  - validation workflows
  - development tools
  - quality assurance
language: python
date of note: 2025-08-14
---

# Cursus CLI Tools User Guide

## Overview

The Cursus CLI provides a comprehensive set of command-line tools for validating code quality, naming conventions, interface compliance, and architectural alignment across the entire pipeline system. This guide covers all available CLI tools and their practical usage in development workflows.

## Available CLI Tools

### 1. Validation CLI (`validation_cli.py`)
- **Purpose**: Validate naming conventions and interface compliance
- **Location**: `src/cursus/cli/validation_cli.py`
- **Usage**: Standards enforcement and code quality validation

### 2. Alignment CLI (`alignment_cli.py`)
- **Purpose**: Validate architectural alignment across components
- **Location**: `src/cursus/cli/alignment_cli.py`
- **Usage**: Cross-component alignment validation

### 3. Builder Test CLI (`builder_test_cli.py`)
- **Purpose**: Test step builder implementations
- **Location**: `src/cursus/cli/builder_test_cli.py`
- **Usage**: Builder functionality and compliance testing

## Quick Start Guide

### Installation and Setup

```bash
# Navigate to project root
cd /path/to/cursus

# Ensure Python path includes the project
export PYTHONPATH="${PYTHONPATH}:/path/to/cursus"

# Test CLI availability
python -m src.cursus.cli.validation_cli --help
```

### Basic Usage Pattern

All CLI tools follow a consistent pattern:
```bash
python -m src.cursus.cli.<tool_name> <command> [options] [arguments]
```

## Validation CLI - Detailed Usage

### 1. Registry Validation

**Purpose**: Validate all registry entries for naming consistency and compliance.

```bash
# Basic registry validation
python -m src.cursus.cli.validation_cli registry

# Verbose output with detailed suggestions
python -m src.cursus.cli.validation_cli registry --verbose
```

**Example Output**:
```
🔍 Validating registry entries...
❌ Found 6 naming violations:

📁 Registry Config Class:
  • Registry Config Class: Config class name doesn't follow expected patterns for step 'Base'
    💡 Suggestions: Expected: One of: BaseConfig, Actual: BasePipelineConfig

✅ All naming conventions checks passed!
```

**When to Use**:
- Before committing changes to registry
- After adding new step builders
- During code reviews
- In CI/CD pipelines

### 2. File Name Validation

**Purpose**: Validate individual file names against naming conventions.

```bash
# Validate builder file names
python -m src.cursus.cli.validation_cli file builder_xgboost_training_step.py builder

# Validate config file names
python -m src.cursus.cli.validation_cli file xgboost_training_config.py config

# Validate specification file names
python -m src.cursus.cli.validation_cli file xgboost_training_spec.py spec

# Validate contract file names
python -m src.cursus.cli.validation_cli file xgboost_training_contract.py contract
```

**Supported File Types**:
- `builder`: Step builder implementation files
- `config`: Configuration class files
- `spec`: Specification definition files
- `contract`: Script contract files

**When to Use**:
- Before creating new files
- When renaming existing files
- During file organization
- In automated file validation scripts

### 3. Step Name Validation

**Purpose**: Validate canonical step names for consistency.

```bash
# Validate step names
python -m src.cursus.cli.validation_cli step XGBoostTraining
python -m src.cursus.cli.validation_cli step TabularPreprocessing
python -m src.cursus.cli.validation_cli step CurrencyConversion
```

**Validation Rules**:
- Must use PascalCase format
- Must be descriptive and clear
- Should avoid unclear abbreviations
- Must use consistent terminology

**When to Use**:
- When defining new step types
- During step name standardization
- Before registry updates
- In naming consistency reviews

### 4. Logical Name Validation

**Purpose**: Validate logical names used in specifications and contracts.

```bash
# Validate logical names
python -m src.cursus.cli.validation_cli logical input_data
python -m src.cursus.cli.validation_cli logical processed_output
python -m src.cursus.cli.validation_cli logical model_artifacts
```

**Validation Rules**:
- Must use snake_case format
- Must be descriptive and clear
- Should avoid system reserved names
- Must follow established patterns

**When to Use**:
- When defining input/output logical names
- During specification creation
- In contract definition
- For dependency mapping validation

### 5. Interface Compliance Validation

**Purpose**: Validate step builder interface compliance and implementation standards.

```bash
# Validate step builder interfaces
python -m src.cursus.cli.validation_cli interface src.cursus.steps.builders.builder_xgboost_training_step.XGBoostTrainingStepBuilder

# Verbose output with detailed diagnostics
python -m src.cursus.cli.validation_cli interface src.cursus.steps.builders.builder_xgboost_training_step.XGBoostTrainingStepBuilder --verbose
```

**Example Output**:
```
🔍 Validating interface compliance for: XGBoostTrainingStepBuilder
❌ Found 3 interface violations:

📁 Method '_get_inputs' in XGBoostTrainingStepBuilder:
  • [signature_return_type] Return type annotation may not match expected pattern

📁 Method 'validate_configuration' in XGBoostTrainingStepBuilder:
  • [documentation_missing_return] Method has return type but no return documentation
```

**Validation Scope**:
- Required method implementations
- Method signature compliance
- Type annotation validation
- Documentation standards
- Inheritance validation

**When to Use**:
- After implementing new step builders
- During code reviews
- Before merging builder changes
- In automated testing pipelines

## Alignment Validation System

### Comprehensive Script Alignment Validation

**Purpose**: Validate alignment across all architectural levels for pipeline scripts.

```bash
# Navigate to alignment validation directory
cd test/steps/scripts/alignment_validation

# Run comprehensive validation for all scripts
python run_alignment_validation.py

# Run validation for individual scripts
python validate_currency_conversion.py
python validate_dummy_training.py
python validate_xgboost_training.py
```

### 4-Level Validation System

#### Level 1: Script ↔ Contract Alignment
- **Purpose**: Ensure scripts match their contracts
- **Validates**: Argument alignment, path usage, environment variables
- **Example Issues**: Missing arguments, incorrect path references

#### Level 2: Contract ↔ Specification Alignment
- **Purpose**: Ensure contracts align with specifications
- **Validates**: Field types, constraints, required fields
- **Example Issues**: Missing fields, type mismatches

#### Level 3: Specification ↔ Dependencies Alignment
- **Purpose**: Ensure dependency resolution works correctly
- **Validates**: Dependency compatibility, specification requirements
- **Example Issues**: Unresolvable dependencies, missing sources

#### Level 4: Builder ↔ Configuration Alignment
- **Purpose**: Ensure builders use configurations correctly
- **Validates**: Field mapping, configuration consistency
- **Example Issues**: Undeclared configuration fields, missing imports

### Generated Reports

The alignment validation system generates comprehensive reports:

```bash
# Report locations after running validation
ls test/steps/scripts/alignment_validation/reports/

# JSON reports (machine-readable)
ls test/steps/scripts/alignment_validation/reports/json/
# currency_conversion_alignment_report.json
# dummy_training_alignment_report.json
# ...

# HTML reports (human-readable)
ls test/steps/scripts/alignment_validation/reports/html/
# currency_conversion_alignment_report.html
# dummy_training_alignment_report.html
# ...

# Overall summary
cat test/steps/scripts/alignment_validation/reports/validation_summary.json
```

**Example Summary**:
```json
{
  "total_scripts": 9,
  "passed_scripts": 9,
  "failed_scripts": 0,
  "error_scripts": 0,
  "validation_timestamp": "2025-08-14T10:24:23.301726",
  "script_results": {
    "currency_conversion": {
      "status": "PASSING",
      "timestamp": "2025-08-14T10:24:23.329670"
    }
  }
}
```

## Development Workflows

### Pre-Commit Validation Workflow

```bash
#!/bin/bash
# pre-commit-validation.sh

echo "🔍 Running pre-commit validation..."

# 1. Validate registry entries
echo "Validating registry..."
python -m src.cursus.cli.validation_cli registry || exit 1

# 2. Validate any new builder interfaces
echo "Validating builder interfaces..."
for builder in $(find src/cursus/steps/builders -name "*.py" -type f); do
    if [[ $builder == *"builder_"* ]]; then
        class_path=$(echo $builder | sed 's|/|.|g' | sed 's|\.py||' | sed 's|^src\.|src.|')
        builder_class=$(basename $builder .py | sed 's/builder_//' | sed 's/_step//' | awk '{print toupper(substr($0,1,1)) substr($0,2)}')StepBuilder
        python -m src.cursus.cli.validation_cli interface "${class_path}.${builder_class}" || exit 1
    fi
done

# 3. Run alignment validation
echo "Running alignment validation..."
cd test/steps/scripts/alignment_validation
python run_alignment_validation.py || exit 1

echo "✅ All validations passed!"
```

### New Component Creation Workflow

```bash
#!/bin/bash
# create-new-step.sh

STEP_NAME=$1
if [ -z "$STEP_NAME" ]; then
    echo "Usage: $0 <StepName>"
    exit 1
fi

echo "🚀 Creating new step: $STEP_NAME"

# 1. Validate step name
echo "Validating step name..."
python -m src.cursus.cli.validation_cli step $STEP_NAME || exit 1

# 2. Create files with proper naming
SNAKE_NAME=$(echo $STEP_NAME | sed 's/\([A-Z]\)/_\L\1/g' | sed 's/^_//')

# Validate file names before creation
echo "Validating file names..."
python -m src.cursus.cli.validation_cli file "builder_${SNAKE_NAME}_step.py" builder || exit 1
python -m src.cursus.cli.validation_cli file "${SNAKE_NAME}_config.py" config || exit 1
python -m src.cursus.cli.validation_cli file "${SNAKE_NAME}_spec.py" spec || exit 1
python -m src.cursus.cli.validation_cli file "${SNAKE_NAME}_contract.py" contract || exit 1

# 3. Create the files (implementation details omitted)
echo "Creating files..."
# ... file creation logic ...

echo "✅ Step $STEP_NAME created successfully!"
```

### Code Review Validation Workflow

```bash
#!/bin/bash
# code-review-validation.sh

echo "🔍 Running code review validation..."

# 1. Validate all changed files
git diff --name-only HEAD~1 | while read file; do
    if [[ $file == src/cursus/steps/builders/* ]]; then
        echo "Validating builder file: $file"
        filename=$(basename $file)
        python -m src.cursus.cli.validation_cli file $filename builder || exit 1
    elif [[ $file == src/cursus/steps/configs/* ]]; then
        echo "Validating config file: $file"
        filename=$(basename $file)
        python -m src.cursus.cli.validation_cli file $filename config || exit 1
    fi
done

# 2. Validate interfaces for changed builders
git diff --name-only HEAD~1 | grep "builder_.*\.py" | while read file; do
    class_path=$(echo $file | sed 's|/|.|g' | sed 's|\.py||' | sed 's|^src\.|src.|')
    builder_class=$(basename $file .py | sed 's/builder_//' | sed 's/_step//' | awk '{print toupper(substr($0,1,1)) substr($0,2)}')StepBuilder
    echo "Validating interface: ${class_path}.${builder_class}"
    python -m src.cursus.cli.validation_cli interface "${class_path}.${builder_class}" --verbose || exit 1
done

# 3. Run alignment validation if scripts changed
if git diff --name-only HEAD~1 | grep -q "src/cursus/steps/scripts/"; then
    echo "Scripts changed, running alignment validation..."
    cd test/steps/scripts/alignment_validation
    python run_alignment_validation.py || exit 1
fi

echo "✅ Code review validation completed!"
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Cursus Validation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  validation:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Validate Registry
      run: |
        python -m src.cursus.cli.validation_cli registry
    
    - name: Validate Builder Interfaces
      run: |
        # Add specific builder validations here
        python -m src.cursus.cli.validation_cli interface src.cursus.steps.builders.builder_xgboost_training_step.XGBoostTrainingStepBuilder
    
    - name: Run Alignment Validation
      run: |
        cd test/steps/scripts/alignment_validation
        python run_alignment_validation.py
    
    - name: Upload Validation Reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: validation-reports
        path: test/steps/scripts/alignment_validation/reports/
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    
    stages {
        stage('Validation') {
            parallel {
                stage('Registry Validation') {
                    steps {
                        sh 'python -m src.cursus.cli.validation_cli registry'
                    }
                }
                
                stage('Interface Validation') {
                    steps {
                        script {
                            def builders = sh(
                                script: "find src/cursus/steps/builders -name 'builder_*.py' -type f",
                                returnStdout: true
                            ).trim().split('\n')
                            
                            builders.each { builder ->
                                def className = builder.replaceAll(/.*\//, '').replaceAll(/\.py$/, '')
                                def classPath = builder.replaceAll(/\//, '.').replaceAll(/\.py$/, '').replaceAll(/^src\./, 'src.')
                                sh "python -m src.cursus.cli.validation_cli interface ${classPath}.${className}"
                            }
                        }
                    }
                }
                
                stage('Alignment Validation') {
                    steps {
                        dir('test/steps/scripts/alignment_validation') {
                            sh 'python run_alignment_validation.py'
                        }
                    }
                }
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'test/steps/scripts/alignment_validation/reports/**/*', fingerprint: true
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'test/steps/scripts/alignment_validation/reports/html',
                reportFiles: '*.html',
                reportName: 'Alignment Validation Report'
            ])
        }
    }
}
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` when running CLI commands

**Solution**:
```bash
# Ensure Python path includes project root
export PYTHONPATH="${PYTHONPATH}:/path/to/cursus"

# Or run from project root
cd /path/to/cursus
python -m src.cursus.cli.validation_cli registry
```

#### 2. Class Path Issues

**Problem**: Cannot find class for interface validation

**Solution**:
```bash
# Use correct class path format
python -m src.cursus.cli.validation_cli interface src.cursus.steps.builders.builder_xgboost_training_step.XGBoostTrainingStepBuilder

# Not: cursus.steps.builders...
# Not: /src/cursus/steps/builders...
```

#### 3. Permission Issues

**Problem**: Cannot write validation reports

**Solution**:
```bash
# Ensure write permissions for reports directory
chmod -R 755 test/steps/scripts/alignment_validation/reports/

# Or run with appropriate permissions
sudo python run_alignment_validation.py
```

#### 4. Missing Dependencies

**Problem**: Validation fails due to missing dependencies

**Solution**:
```bash
# Install all required dependencies
pip install -r requirements.txt

# Install development dependencies if needed
pip install -r requirements-dev.txt
```

### Debugging Tips

#### 1. Use Verbose Mode
```bash
# Get detailed output for debugging
python -m src.cursus.cli.validation_cli registry --verbose
python -m src.cursus.cli.validation_cli interface <class_path> --verbose
```

#### 2. Check Log Files
```bash
# Check alignment validation logs
tail -f test/steps/scripts/alignment_validation/reports/validation_summary.json

# Check individual script logs
cat test/steps/scripts/alignment_validation/reports/json/currency_conversion_alignment_report.json
```

#### 3. Validate Step by Step
```bash
# Test individual components
python -m src.cursus.cli.validation_cli step XGBoostTraining
python -m src.cursus.cli.validation_cli file builder_xgboost_training_step.py builder
python -m src.cursus.cli.validation_cli logical input_data
```

## Best Practices

### 1. Regular Validation
- Run validation before every commit
- Include validation in CI/CD pipelines
- Validate after major refactoring
- Regular registry validation

### 2. Comprehensive Coverage
- Validate all new components
- Test interface compliance for all builders
- Run alignment validation for script changes
- Validate naming conventions consistently

### 3. Report Analysis
- Review HTML reports for detailed analysis
- Track validation trends over time
- Address violations promptly
- Use reports for code quality metrics

### 4. Automation
- Automate validation in development workflows
- Set up pre-commit hooks
- Integrate with IDE/editor
- Use validation in code review process

## Advanced Usage

### Custom Validation Scripts

```python
#!/usr/bin/env python3
"""Custom validation script example."""

import subprocess
import sys
from pathlib import Path

def run_validation(command):
    """Run validation command and return result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    """Run custom validation suite."""
    validations = [
        "python -m src.cursus.cli.validation_cli registry",
        "python -m src.cursus.cli.validation_cli step XGBoostTraining",
        "python -m src.cursus.cli.validation_cli logical input_data"
    ]
    
    results = []
    for validation in validations:
        success, output = run_validation(validation)
        results.append((validation, success, output))
        
        if success:
            print(f"✅ {validation}")
        else:
            print(f"❌ {validation}")
            print(f"   {output}")
    
    # Summary
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n📊 Validation Summary: {passed}/{total} passed")
    
    if passed < total:
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Batch Validation

```bash
#!/bin/bash
# batch-validation.sh

# Validate all builders in a directory
find src/cursus/steps/builders -name "builder_*.py" | while read file; do
    filename=$(basename $file)
    echo "Validating: $filename"
    python -m src.cursus.cli.validation_cli file $filename builder
done

# Validate all step names from registry
python -c "
from src.cursus.steps.registry.step_names import STEP_NAMES
import subprocess
for step_name in STEP_NAMES.keys():
    if step_name not in ['Base', 'Processing']:
        result = subprocess.run(['python', '-m', 'src.cursus.cli.validation_cli', 'step', step_name])
        if result.returncode != 0:
            print(f'Failed: {step_name}')
"
```

## Conclusion

The Cursus CLI tools provide comprehensive validation capabilities for maintaining code quality, architectural alignment, and naming consistency across the entire pipeline system. By integrating these tools into your development workflow, you can ensure high-quality, maintainable code that adheres to established standards and patterns.

For additional help or questions:
- Check the individual CLI tool documentation
- Review the standardization rules in the developer guide
- Examine the alignment validation system documentation
- Consult the troubleshooting section for common issues

Remember to run validations regularly and address violations promptly to maintain code quality and system integrity.
