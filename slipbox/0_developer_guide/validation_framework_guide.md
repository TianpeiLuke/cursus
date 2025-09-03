# Validation Framework Guide

This guide provides comprehensive instructions for using the validation frameworks in the Cursus pipeline system. The validation frameworks ensure that your step implementations are correct, aligned, and follow best practices.

## Overview

The Cursus validation system consists of two complementary frameworks:

1. **Unified Alignment Tester** (`cursus/validation/alignment`) - Validates 4-tier alignment between components
2. **Universal Step Builder Test** (`cursus/validation/builders`) - Performs 4-level builder testing

Both frameworks must pass before integrating new steps into the pipeline system.

## Unified Alignment Tester

The Unified Alignment Tester validates the alignment between your step's components across four tiers:

### 4-Tier Validation Levels

1. **Level 1: Script ↔ Contract Alignment**
   - Validates that script paths match contract definitions
   - Ensures input/output paths are correctly defined
   - Checks environment variable usage

2. **Level 2: Contract ↔ Specification Alignment**
   - Validates logical name consistency between contract and specification
   - Ensures dependency definitions match contract expectations
   - Checks output specification alignment

3. **Level 3: Specification ↔ Dependencies Alignment**
   - Validates dependency compatibility with upstream steps
   - Ensures semantic keyword consistency
   - Checks data type compatibility

4. **Level 4: Builder ↔ Configuration Alignment**
   - Validates builder configuration integration
   - Ensures proper specification usage in builders
   - Checks property path consistency

### Usage Options

#### Option A: CLI Commands (Recommended)

```bash
# Validate a specific script with detailed output and scoring
python -m cursus.cli.alignment_cli validate your_step_name --verbose --show-scoring

# Validate a specific alignment level only
python -m cursus.cli.alignment_cli validate-level your_step_name 1 --verbose

# Generate comprehensive visualization and scoring reports
python -m cursus.cli.alignment_cli visualize your_step_name --output-dir ./validation_reports --verbose

# Run validation for all scripts with reports
python -m cursus.cli.alignment_cli validate-all --output-dir ./reports --format both --verbose
```

**CLI Command Options:**
- `--verbose`: Show detailed validation information
- `--show-scoring`: Display scoring metrics and analysis
- `--output-dir`: Specify directory for reports and visualizations
- `--format both`: Generate both JSON and visual reports

#### Option B: Using Test Scripts

```bash
# Create individual validation script (following existing patterns)
python test/steps/scripts/alignment_validation/validate_your_step_name.py

# Run comprehensive alignment validation for all scripts
python test/steps/scripts/alignment_validation/run_alignment_validation.py
```

#### Option C: Direct Python Usage

Create a validation script following this pattern:

```python
#!/usr/bin/env python3
"""
Alignment validation for your step.
Based on pattern from validate_tabular_preprocessing.py
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

def main():
    """Run alignment validation for your step."""
    print("🔍 Your Step Alignment Validation")
    print("=" * 60)
    
    # Initialize the tester with directory paths
    tester = UnifiedAlignmentTester(
        scripts_dir=str(project_root / "src" / "cursus" / "steps" / "scripts"),
        contracts_dir=str(project_root / "src" / "cursus" / "steps" / "contracts"),
        specs_dir=str(project_root / "src" / "cursus" / "steps" / "specs"),
        builders_dir=str(project_root / "src" / "cursus" / "steps" / "builders"),
        configs_dir=str(project_root / "src" / "cursus" / "steps" / "configs")
    )
    
    # Run validation for your specific script
    script_name = "your_step_name"  # Replace with your actual script name
    
    try:
        results = tester.validate_specific_script(script_name)
        
        # Print results
        status = results.get('overall_status', 'UNKNOWN')
        status_emoji = '✅' if status == 'PASSING' else '❌'
        print(f"{status_emoji} Overall Status: {status}")
        
        # Print level-by-level results
        for level_num, level_name in enumerate([
            "Script ↔ Contract",
            "Contract ↔ Specification", 
            "Specification ↔ Dependencies",
            "Builder ↔ Configuration"
        ], 1):
            level_key = f"level{level_num}"
            level_result = results.get(level_key, {})
            level_passed = level_result.get('passed', False)
            level_issues = level_result.get('issues', [])
            
            status_emoji = '✅' if level_passed else '❌'
            print(f"\n{status_emoji} Level {level_num}: {level_name}")
            print(f"   Status: {'PASS' if level_passed else 'FAIL'}")
            print(f"   Issues: {len(level_issues)}")
            
            # Print issues with details
            for issue in level_issues:
                severity = issue.get('severity', 'ERROR')
                message = issue.get('message', 'No message')
                recommendation = issue.get('recommendation', '')
                
                print(f"   • {severity}: {message}")
                if recommendation:
                    print(f"     💡 Recommendation: {recommendation}")
        
        return 0 if status == 'PASSING' else 1
        
    except Exception as e:
        print(f"❌ ERROR during validation: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
```

## Universal Step Builder Test

The Universal Step Builder Test performs comprehensive testing of your step builder implementation across four levels:

### 4-Level Testing Framework

1. **Level 1: Interface Testing**
   - Validates builder interface compliance
   - Checks required method implementations
   - Ensures proper inheritance from StepBuilderBase

2. **Level 2: Specification Testing**
   - Validates spec-driven functionality
   - Ensures proper specification integration
   - Checks specification-based input/output generation

3. **Level 3: Path Mapping Testing**
   - Validates input/output path correctness
   - Ensures proper SageMaker path mapping
   - Checks container path consistency

4. **Level 4: Integration Testing**
   - Performs end-to-end step creation testing
   - Validates complete builder workflow
   - Ensures proper SageMaker step generation

### Usage Options

#### Option A: CLI Commands (Recommended)

```bash
# Run all tests for your builder with scoring
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_step.YourStepBuilder --scoring --verbose

# Run specific level tests
python -m cursus.cli.builder_test_cli level 1 src.cursus.steps.builders.builder_your_step.YourStepBuilder --verbose

# Test all builders of your step type (e.g., Processing)
python -m cursus.cli.builder_test_cli test-by-type Processing --verbose --scoring

# Export results to JSON and generate charts
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_step.YourStepBuilder --export-json ./reports/builder_test_results.json --export-chart --output-dir ./reports
```

**CLI Command Options:**
- `--scoring`: Enable scoring metrics and analysis
- `--verbose`: Show detailed test information
- `--export-json`: Export results to JSON file
- `--export-chart`: Generate visual charts
- `--output-dir`: Specify directory for outputs

#### Option B: Using Test Scripts by Step Type

```bash
# Run Processing-specific tests (if your step is a Processing step)
python test/steps/builders/run_processing_tests.py

# Run Training-specific tests (if your step is a Training step)
python test/steps/builders/run_training_tests.py

# Run Transform-specific tests (if your step is a Transform step)
python test/steps/builders/run_transform_tests.py

# Run CreateModel-specific tests (if your step is a CreateModel step)
python test/steps/builders/run_createmodel_tests.py

# Run RegisterModel-specific tests (if your step is a RegisterModel step)
python test/steps/builders/run_registermodel_tests.py
```

#### Option C: Direct Python Usage

Create a builder test script following this pattern:

```python
#!/usr/bin/env python3
"""
Builder validation for your step.
Based on pattern from test_processing_step_builders.py
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cursus.validation.builders.universal_test import UniversalStepBuilderTest

def main():
    """Run builder validation for your step."""
    print("🔧 Your Step Builder Validation")
    print("=" * 60)
    
    # Import your builder class
    from cursus.steps.builders.builder_your_step import YourStepBuilder
    
    try:
        # Initialize the tester with enhanced features
        tester = UniversalStepBuilderTest(
            YourStepBuilder, 
            verbose=True,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        # Run all tests
        results = tester.run_all_tests()
        
        # Extract test results from enhanced format
        test_results = results.get('test_results', results) if isinstance(results, dict) and 'test_results' in results else results
        
        # Print results
        passed_tests = sum(1 for result in test_results.values() 
                          if isinstance(result, dict) and result.get("passed", False))
        total_tests = len([r for r in test_results.values() if isinstance(r, dict)])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📊 Builder Test Results: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        
        # Show failed tests
        failed_tests = {k: v for k, v in test_results.items() 
                       if isinstance(v, dict) and not v.get("passed", True)}
        
        if failed_tests:
            print("\n❌ Failed Tests:")
            for test_name, result in failed_tests.items():
                print(f"  • {test_name}: {result.get('error', 'Unknown error')}")
        else:
            print("\n✅ All builder tests passed!")
        
        # Print scoring information if available
        scoring = results.get('scoring', {})
        if scoring:
            print(f"\n📈 Scoring Information:")
            for metric, value in scoring.items():
                print(f"  • {metric}: {value}")
        
        return 0 if pass_rate == 100 else 1
        
    except Exception as e:
        print(f"❌ ERROR during builder validation: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())
```

## Step Type-Specific Validation

The validation frameworks automatically apply step type-specific validation variants based on your `sagemaker_step_type` field in the registry:

### Processing Steps
- Standard processing validation patterns
- Input/output path validation for processing containers
- Environment variable validation for processing jobs

### Training Steps
- Training-specific validation with hyperparameter checks
- Model artifact path validation
- Training job configuration validation

### Transform Steps
- Transform-specific validation patterns
- Batch transform input/output validation
- Transform job configuration checks

### CreateModel Steps
- Model creation validation patterns
- Model artifact and container validation
- Endpoint configuration checks

### RegisterModel Steps
- Model registration validation patterns
- Model package and registry validation
- Custom registration step validation (e.g., MimsModelRegistrationProcessingStep)

## Recommended Validation Workflow

### 1. Initial Validation Run

Start with CLI commands for quick feedback:

```bash
# Quick alignment check
python -m cursus.cli.alignment_cli validate your_step_name --verbose

# Quick builder test
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_step.YourStepBuilder --verbose
```

### 2. Detailed Analysis

If issues are found, run with scoring and reports:

```bash
# Detailed alignment analysis with scoring
python -m cursus.cli.alignment_cli validate your_step_name --verbose --show-scoring

# Detailed builder analysis with scoring and exports
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_step.YourStepBuilder --scoring --verbose --export-json ./reports/results.json
```

### 3. Comprehensive Validation

Before final integration, run comprehensive tests:

```bash
# Full alignment validation with visualization
python -m cursus.cli.alignment_cli visualize your_step_name --output-dir ./validation_reports --verbose

# Full builder validation with charts
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_step.YourStepBuilder --export-chart --output-dir ./reports --scoring
```

### 4. Step Type-Specific Testing

Run step type-specific tests based on your SageMaker step type:

```bash
# For Processing steps
python test/steps/builders/run_processing_tests.py

# For Training steps  
python test/steps/builders/run_training_tests.py

# For Transform steps
python test/steps/builders/run_transform_tests.py

# For CreateModel steps
python test/steps/builders/run_createmodel_tests.py

# For RegisterModel steps
python test/steps/builders/run_registermodel_tests.py
```

## Understanding Validation Results

### Alignment Tester Results

The alignment tester provides results in this format:

```
🔍 Step Alignment Validation
============================================================
✅ Overall Status: PASSING

✅ Level 1: Script ↔ Contract
   Status: PASS
   Issues: 0

❌ Level 2: Contract ↔ Specification
   Status: FAIL
   Issues: 2
   • ERROR: Logical name mismatch: 'input_data' in contract vs 'data_input' in spec
     💡 Recommendation: Update contract logical name to match specification

✅ Level 3: Specification ↔ Dependencies
   Status: PASS
   Issues: 0

✅ Level 4: Builder ↔ Configuration
   Status: PASS
   Issues: 0
```

### Builder Test Results

The builder tester provides results in this format:

```
🔧 Step Builder Validation
============================================================
📊 Builder Test Results: 15/16 tests passed (93.8%)

❌ Failed Tests:
  • test_specification_integration: Specification not properly integrated in builder

✅ All other tests passed!

📈 Scoring Information:
  • interface_compliance_score: 100.0
  • specification_integration_score: 87.5
  • path_mapping_accuracy: 100.0
  • integration_test_score: 93.8
```

## Common Issues and Solutions

### Alignment Issues

**Issue**: Logical name mismatch between contract and specification
```
ERROR: Logical name 'input_data' in contract doesn't match 'data_input' in specification
```
**Solution**: Ensure logical names are consistent across contract and specification files.

**Issue**: Property path inconsistency
```
ERROR: Property path format doesn't follow standard pattern
```
**Solution**: Use the standard property path format: `properties.ProcessingOutputConfig.Outputs['output_name'].S3Output.S3Uri`

**Issue**: Missing dependency compatibility
```
ERROR: Dependency 'input_data' not compatible with upstream step outputs
```
**Solution**: Check that your dependency specifications match the outputs of upstream steps.

### Builder Test Issues

**Issue**: Interface compliance failure
```
ERROR: Builder missing required method '_get_inputs'
```
**Solution**: Ensure your builder implements all required methods from StepBuilderBase.

**Issue**: Specification integration failure
```
ERROR: Builder not using specification for input/output generation
```
**Solution**: Use `_get_spec_driven_processor_inputs()` and `_get_spec_driven_processor_outputs()` methods.

**Issue**: Path mapping failure
```
ERROR: Input path mapping doesn't match contract expectations
```
**Solution**: Ensure your builder's path mapping aligns with the script contract definitions.

## Advanced Usage

### Custom Validation Scripts

You can create custom validation scripts for specific scenarios:

```python
#!/usr/bin/env python3
"""
Custom validation script for specific step requirements.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from cursus.validation.builders.universal_test import UniversalStepBuilderTest

def run_comprehensive_validation(step_name: str, builder_class):
    """Run both alignment and builder validation."""
    print(f"🔍 Comprehensive Validation for {step_name}")
    print("=" * 80)
    
    # Run alignment validation
    print("\n1. Running Alignment Validation...")
    alignment_tester = UnifiedAlignmentTester(
        scripts_dir="src/cursus/steps/scripts",
        contracts_dir="src/cursus/steps/contracts",
        specs_dir="src/cursus/steps/specs",
        builders_dir="src/cursus/steps/builders",
        configs_dir="src/cursus/steps/configs"
    )
    
    alignment_results = alignment_tester.validate_specific_script(step_name)
    alignment_passed = alignment_results.get('overall_status') == 'PASSING'
    
    # Run builder validation
    print("\n2. Running Builder Validation...")
    builder_tester = UniversalStepBuilderTest(
        builder_class,
        verbose=True,
        enable_scoring=True
    )
    
    builder_results = builder_tester.run_all_tests()
    test_results = builder_results.get('test_results', builder_results)
    passed_tests = sum(1 for result in test_results.values() 
                      if isinstance(result, dict) and result.get("passed", False))
    total_tests = len([r for r in test_results.values() if isinstance(r, dict)])
    builder_passed = passed_tests == total_tests
    
    # Summary
    print(f"\n📋 Validation Summary for {step_name}")
    print("=" * 80)
    print(f"✅ Alignment Validation: {'PASS' if alignment_passed else 'FAIL'}")
    print(f"✅ Builder Validation: {'PASS' if builder_passed else 'FAIL'}")
    
    overall_passed = alignment_passed and builder_passed
    print(f"\n🎯 Overall Result: {'✅ READY FOR INTEGRATION' if overall_passed else '❌ NEEDS FIXES'}")
    
    return 0 if overall_passed else 1

# Usage example:
# from cursus.steps.builders.builder_your_step import YourStepBuilder
# run_comprehensive_validation("your_step", YourStepBuilder)
```

### Batch Validation

For validating multiple steps at once:

```bash
# Validate all alignment for multiple steps
python -m cursus.cli.alignment_cli validate-all --output-dir ./reports --format both

# Test all builders of a specific type
python -m cursus.cli.builder_test_cli test-by-type Processing --verbose --scoring --output-dir ./reports
```

## Integration with Development Workflow

### Pre-Commit Validation

Add validation checks to your development workflow:

```bash
#!/bin/bash
# pre-commit-validation.sh

echo "🔍 Running pre-commit validation..."

# Run alignment validation for changed steps
python -m cursus.cli.alignment_cli validate-all --format json --output-dir ./validation_reports

# Run builder tests for changed builders
python -m cursus.cli.builder_test_cli test-by-type Processing --export-json ./validation_reports/builder_results.json

echo "✅ Pre-commit validation complete"
```

### Continuous Integration

Integrate validation into your CI pipeline:

```yaml
# .github/workflows/validation.yml
name: Step Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run alignment validation
        run: python -m cursus.cli.alignment_cli validate-all --format json
      - name: Run builder validation
        run: python -m cursus.cli.builder_test_cli test-by-type Processing --export-json ./results.json
```

## Best Practices

### 1. Validation-Driven Development

- Run validation tests early and often during development
- Fix alignment issues before proceeding to builder implementation
- Use validation feedback to guide implementation decisions

### 2. Incremental Validation

- Validate each component as you create it
- Start with Level 1 alignment and work your way up
- Fix issues at each level before proceeding

### 3. Comprehensive Testing

- Always run both alignment and builder validation
- Use step type-specific tests for your SageMaker step type
- Generate reports for documentation and review

### 4. Error Resolution

- Read validation error messages carefully
- Follow the provided recommendations
- Use verbose output for detailed debugging information

### 5. Documentation

- Document any custom validation requirements
- Include validation results in your step documentation
- Share validation reports with team members for review

## Troubleshooting

### Common CLI Issues

**Issue**: Module not found errors
```bash
ModuleNotFoundError: No module named 'cursus.cli.alignment_cli'
```
**Solution**: Ensure you're running from the project root and have installed dependencies.

**Issue**: Builder class not found
```bash
ImportError: cannot import name 'YourStepBuilder'
```
**Solution**: Check that your builder class is properly registered with `@register_builder` decorator.

### Common Validation Failures

**Issue**: Script contract mismatch
**Solution**: Ensure your script uses the exact paths defined in the contract.

**Issue**: Specification dependency errors
**Solution**: Verify that your dependencies match upstream step outputs and use correct logical names.

**Issue**: Builder integration failures
**Solution**: Ensure your builder properly inherits from StepBuilderBase and implements required methods.

## Technical Design References

For detailed technical design information about the validation frameworks:

- [SageMaker Step Type Aware Unified Alignment Tester Design](../1_design/sagemaker_step_type_aware_unified_alignment_tester_design.md) - Complete technical design for the alignment validation framework including step type-specific validation variants, scoring algorithms, and architectural patterns
- [SageMaker Step Type Universal Builder Tester Design](../1_design/sagemaker_step_type_universal_builder_tester_design.md) - Complete technical design for the builder testing framework including 4-level testing methodology, step type-specific test variants, and integration patterns

## Related Documentation

- [Adding New Pipeline Step](adding_new_pipeline_step.md) - Quick start guide
- [Step Creation Process](creation_process.md) - Detailed step creation process
- [Step Builder Guide](step_builder.md) - Builder implementation patterns
- [Validation Checklist](validation_checklist.md) - Pre-integration checklist
