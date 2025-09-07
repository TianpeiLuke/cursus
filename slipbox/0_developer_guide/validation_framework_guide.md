# Validation Framework Guide

This guide provides comprehensive instructions for using the validation frameworks in the Cursus pipeline system. The validation frameworks ensure that your step implementations are correct, aligned, and follow best practices.

## Overview

The Cursus validation system consists of three complementary frameworks with both legacy and workspace-aware implementations:

### Legacy Validation (Single Workspace)
1. **Unified Alignment Tester** (`cursus/validation/alignment`) - Validates 4-tier alignment between components
2. **Universal Step Builder Test** (`cursus/validation/builders`) - Performs 4-level builder testing
3. **Script Runtime Tester** (`cursus/validation/runtime`) - Validates actual script execution and data flow

### Workspace-Aware Validation (Multi-Developer Support)
1. **Workspace Unified Alignment Tester** (`cursus/workspace/validation`) - Workspace-aware 4-tier alignment validation
2. **Workspace Universal Step Builder Test** (`cursus/workspace/validation`) - Workspace-aware 4-level builder testing
3. **Workspace Script Runtime Tester** (`cursus/workspace/validation`) - Workspace-aware script execution validation

**Recommendation**: Use the workspace-aware validation frameworks for all new development, as they support both isolated developer workspaces and shared workspace fallback.

All three frameworks must pass before integrating new steps into the pipeline system.

## Workspace-Aware Validation (Recommended)

### Workspace Unified Alignment Tester

The Workspace Unified Alignment Tester extends the legacy alignment tester with full workspace awareness, supporting both isolated developer workspaces and shared workspace fallback.

#### Key Features
- **Multi-Developer Support**: Validates steps in isolated developer workspaces (`development/projects/*/src/cursus_dev/`)
- **Shared Fallback**: Automatically falls back to shared workspace (`src/cursus/steps/`) when components aren't found in developer workspace
- **Cross-Workspace Validation**: Validates dependencies across different workspaces
- **Workspace-Specific Reporting**: Enhanced error messages and statistics with workspace context

#### Usage

```python
#!/usr/bin/env python3
"""
Workspace-aware alignment validation for your step.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cursus.workspace.validation.workspace_alignment_tester import WorkspaceUnifiedAlignmentTester

def main():
    """Run workspace-aware alignment validation for your step."""
    print("🔍 Workspace-Aware Step Alignment Validation")
    print("=" * 60)
    
    # Initialize the workspace-aware tester
    tester = WorkspaceUnifiedAlignmentTester(
        workspace_root=project_root,
        developer_id="your_developer_id",  # e.g., "developer_1"
        enable_shared_fallback=True
    )
    
    # Run workspace validation
    try:
        results = tester.run_workspace_validation(
            target_scripts=["your_step_name"],  # Replace with your script name
            workspace_context={
                "validation_purpose": "development",
                "step_type": "processing"  # or training, transform, etc.
            }
        )
        
        # Print results
        success = results.get('success', False)
        status_emoji = '✅' if success else '❌'
        print(f"{status_emoji} Overall Status: {'PASSING' if success else 'FAILING'}")
        
        # Print workspace metadata
        workspace_metadata = results.get('workspace_metadata', {})
        print(f"\n🏢 Workspace Context:")
        print(f"   Developer ID: {workspace_metadata.get('developer_id', 'Unknown')}")
        print(f"   Shared Fallback: {workspace_metadata.get('enable_shared_fallback', False)}")
        
        # Print validation results for each script
        script_results = results.get('results', {})
        for script_name, script_result in script_results.items():
            print(f"\n📋 Results for {script_name}:")
            
            for level_name, level_result in script_result.items():
                if isinstance(level_result, dict) and 'passed' in level_result:
                    level_passed = level_result.get('passed', False)
                    level_details = level_result.get('details', {})
                    
                    status_emoji = '✅' if level_passed else '❌'
                    print(f"   {status_emoji} {level_name}: {'PASS' if level_passed else 'FAIL'}")
                    
                    if not level_passed and level_details:
                        print(f"      Issues: {level_details}")
        
        # Print cross-workspace validation if available
        cross_workspace = results.get('cross_workspace_validation', {})
        if cross_workspace.get('enabled', False):
            print(f"\n🔗 Cross-Workspace Validation:")
            shared_components = cross_workspace.get('shared_components_used', {})
            if shared_components:
                print(f"   Shared Components Used: {len(shared_components)} scripts")
            
            recommendations = cross_workspace.get('recommendations', [])
            if recommendations:
                print(f"   Recommendations:")
                for rec in recommendations:
                    print(f"     • {rec}")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ ERROR during workspace validation: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
```

### Workspace Universal Step Builder Test

The Workspace Universal Step Builder Test extends the legacy builder test with workspace awareness and dynamic builder loading.

#### Key Features
- **Dynamic Builder Loading**: Loads builder classes from workspace directories using `WorkspaceModuleLoader`
- **Workspace Integration Validation**: Validates builder dependencies and integration within workspace context
- **Multi-Workspace Testing**: Can test all builders across different developer workspaces
- **Workspace-Specific Error Reporting**: Enhanced error messages with workspace context

#### Usage

```python
#!/usr/bin/env python3
"""
Workspace-aware builder validation for your step.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cursus.workspace.validation.workspace_builder_test import WorkspaceUniversalStepBuilderTest

def main():
    """Run workspace-aware builder validation for your step."""
    print("🔧 Workspace-Aware Step Builder Validation")
    print("=" * 60)
    
    # Initialize the workspace-aware builder test
    tester = WorkspaceUniversalStepBuilderTest(
        workspace_root=project_root,
        developer_id="your_developer_id",  # e.g., "developer_1"
        builder_file_path="builders/builder_your_step.py",  # Relative to workspace
        enable_shared_fallback=True
    )
    
    # Run workspace builder test
    try:
        results = tester.run_workspace_builder_test(
            test_config={
                "enable_integration_tests": True,
                "validate_dependencies": True
            },
            workspace_context={
                "test_purpose": "development",
                "step_type": "processing"
            }
        )
        
        # Print results
        success = results.get('success', False)
        status_emoji = '✅' if success else '❌'
        print(f"{status_emoji} Overall Status: {'PASSING' if success else 'FAILING'}")
        
        # Print workspace metadata
        workspace_metadata = results.get('workspace_metadata', {})
        print(f"\n🏢 Workspace Context:")
        print(f"   Developer ID: {workspace_metadata.get('developer_id', 'Unknown')}")
        print(f"   Builder Class: {workspace_metadata.get('builder_class_name', 'Unknown')}")
        print(f"   Shared Fallback: {workspace_metadata.get('enable_shared_fallback', False)}")
        
        # Print workspace validation results
        workspace_validation = results.get('workspace_validation', {})
        if workspace_validation:
            builder_valid = workspace_validation.get('builder_class_valid', False)
            print(f"\n🔧 Builder Integration:")
            print(f"   Builder Class Valid: {'✅' if builder_valid else '❌'}")
            
            dependencies = workspace_validation.get('workspace_dependencies_available', {})
            if dependencies:
                print(f"   Dependencies:")
                for dep_type, dep_info in dependencies.items():
                    available = dep_info.get('available', False)
                    from_shared = dep_info.get('from_shared', False)
                    status = '✅' if available else '❌'
                    source = ' (shared)' if from_shared else ' (workspace)'
                    print(f"     {status} {dep_type}{source if available else ''}")
            
            issues = workspace_validation.get('integration_issues', [])
            if issues:
                print(f"   Integration Issues:")
                for issue in issues:
                    severity = issue.get('severity', 'INFO')
                    description = issue.get('description', 'No description')
                    print(f"     • {severity}: {description}")
            
            recommendations = workspace_validation.get('recommendations', [])
            if recommendations:
                print(f"   Recommendations:")
                for rec in recommendations:
                    print(f"     • {rec}")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ ERROR during workspace builder validation: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
```

#### Testing All Workspace Builders

You can test all builders in a workspace at once:

```python
#!/usr/bin/env python3
"""
Test all builders in a developer workspace.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cursus.workspace.validation.workspace_builder_test import WorkspaceUniversalStepBuilderTest

def main():
    """Test all builders in the workspace."""
    print("🔧 Testing All Workspace Builders")
    print("=" * 60)
    
    try:
        # Test all builders for a developer
        results = WorkspaceUniversalStepBuilderTest.test_all_workspace_builders(
            workspace_root=project_root,
            developer_id="your_developer_id",  # e.g., "developer_1"
            test_config={
                "enable_integration_tests": True,
                "validate_dependencies": True
            }
        )
        
        # Print summary
        success = results.get('success', False)
        total_builders = results.get('total_builders', 0)
        successful_tests = results.get('successful_tests', 0)
        failed_tests = results.get('failed_tests', 0)
        
        print(f"📊 Test Summary:")
        print(f"   Total Builders: {total_builders}")
        print(f"   Successful Tests: {successful_tests}")
        print(f"   Failed Tests: {failed_tests}")
        print(f"   Success Rate: {(successful_tests/total_builders*100):.1f}%" if total_builders > 0 else "   Success Rate: N/A")
        
        # Print individual results
        individual_results = results.get('results', {})
        if individual_results:
            print(f"\n📋 Individual Results:")
            for builder_name, builder_result in individual_results.items():
                builder_success = builder_result.get('success', False)
                status_emoji = '✅' if builder_success else '❌'
                print(f"   {status_emoji} {builder_name}")
                
                if not builder_success and 'error' in builder_result:
                    print(f"      Error: {builder_result['error']}")
        
        # Print summary recommendations
        summary = results.get('summary', {})
        if summary and 'recommendations' in summary:
            print(f"\n💡 Overall Recommendations:")
            for rec in summary['recommendations']:
                print(f"   • {rec}")
        
        return 0 if success and failed_tests == 0 else 1
        
    except Exception as e:
        print(f"❌ ERROR during workspace builder testing: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
```

## Legacy Validation (Single Workspace)

### Unified Alignment Tester

The legacy Unified Alignment Tester validates the alignment between your step's components across four tiers in a single workspace:

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

## Script Runtime Testing

The Script Runtime Tester validates actual script execution and data flow between pipeline steps. This is the third essential validation framework that ensures your scripts can execute successfully and transfer data correctly along the pipeline DAG.

### Key Features

- **Script Functionality Validation**: Verifies that individual scripts can execute without import/syntax errors
- **Data Transfer Consistency**: Ensures data output by one script is compatible with the input expectations of the next script
- **End-to-End Pipeline Flow**: Tests that the entire pipeline can execute successfully with data flowing correctly between steps
- **Dependency-Agnostic Testing**: Focuses on script execution and data compatibility, not step-to-step dependency resolution

### 3-Mode Runtime Validation

The runtime testing system provides three complementary validation modes:

1. **Individual Script Testing** (`test_script`) - Test each script can execute independently
2. **Data Compatibility Testing** (`test_data_compatibility`) - Test data flow between connected scripts  
3. **Pipeline Flow Testing** (`test_pipeline_flow`) - Test complete end-to-end pipeline execution

### Usage Options

#### Option A: CLI Commands (Recommended)

```bash
# Test single script functionality
cursus runtime test-script your_script_name --workspace-dir ./test_workspace --verbose

# Test data compatibility between two scripts
cursus runtime test-compatibility script_a script_b --workspace-dir ./test_workspace --verbose

# Test complete pipeline flow
cursus runtime test-pipeline pipeline_config.json --workspace-dir ./test_workspace --verbose

# Test with JSON output for CI/CD integration
cursus runtime test-script your_script_name --output-format json --workspace-dir ./test_workspace
```

**CLI Command Options:**
- `--workspace-dir`: Specify test workspace directory
- `--verbose`: Show detailed execution information
- `--output-format json`: Generate JSON output for automation
- `--timeout`: Set script execution timeout (default: 300 seconds)

#### Option B: Direct Python Usage

Create a runtime testing script following this pattern:

```python
#!/usr/bin/env python3
"""
Runtime testing for your step.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_models import RuntimeTestingConfiguration

def main():
    """Run runtime testing for your step."""
    print("🚀 Script Runtime Testing")
    print("=" * 60)
    
    # Configure runtime testing
    config = RuntimeTestingConfiguration(
        workspace_dir="./test_workspace",
        timeout_seconds=300,
        enable_logging=True,
        log_level="INFO",
        cleanup_after_test=True,
        preserve_outputs=False
    )
    
    tester = RuntimeTester(config)
    
    # Test individual script
    script_name = "your_script_name"  # Replace with your actual script name
    
    try:
        print(f"\n1️⃣ Testing Script: {script_name}")
        result = tester.test_script(script_name)
        
        if result.success:
            print(f"  ✅ PASS ({result.execution_time:.3f}s)")
            print(f"  Has main function: {'Yes' if result.has_main_function else 'No'}")
        else:
            print(f"  ❌ FAIL: {result.error_message}")
            if not result.has_main_function:
                print("    💡 Add main(input_paths, output_paths, environ_vars, job_args) function")
        
        return 0 if result.success else 1
        
    except Exception as e:
        print(f"❌ ERROR during runtime testing: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
```

#### Option C: Pipeline Flow Testing with DAG

For testing complete pipeline flows:

```python
#!/usr/bin/env python3
"""
Pipeline flow runtime testing.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_models import RuntimeTestingConfiguration
from cursus.api.dag.base_dag import PipelineDAG

def main():
    """Run pipeline flow runtime testing."""
    print("🚀 Pipeline Flow Runtime Testing")
    print("=" * 60)
    
    # Configure runtime testing
    config = RuntimeTestingConfiguration(
        workspace_dir="./test_workspace",
        timeout_seconds=600,  # Longer timeout for pipeline testing
        enable_logging=True,
        log_level="INFO"
    )
    
    tester = RuntimeTester(config)
    
    # Define test pipeline
    pipeline_config = {
        "steps": {
            "data_preprocessing": {"script": "tabular_preprocessing.py"},
            "model_training": {"script": "xgboost_training.py"},
            "model_evaluation": {"script": "model_evaluation.py"}
        }
    }
    
    try:
        print(f"\n🔄 Testing Pipeline Flow...")
        results = tester.test_pipeline_flow(pipeline_config)
        
        if results["pipeline_success"]:
            print("  ✅ PIPELINE FLOW PASSED")
        else:
            print("  ❌ PIPELINE FLOW FAILED")
            for error in results["errors"]:
                print(f"    - {error}")
        
        # Print individual script results
        print(f"\n📝 Individual Script Results:")
        for script_name, result in results["script_results"].items():
            status = "✅" if result.success else "❌"
            print(f"  {status} {script_name}: {'PASS' if result.success else 'FAIL'}")
            if not result.success:
                print(f"    Error: {result.error_message}")
        
        # Print data flow results
        print(f"\n🔗 Data Flow Results:")
        for flow_name, result in results["data_flow_results"].items():
            status = "✅" if result.compatible else "❌"
            print(f"  {status} {flow_name}: {'PASS' if result.compatible else 'FAIL'}")
            if result.compatibility_issues:
                for issue in result.compatibility_issues:
                    print(f"    Issue: {issue}")
        
        return 0 if results["pipeline_success"] else 1
        
    except Exception as e:
        print(f"❌ ERROR during pipeline flow testing: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
```

### Runtime Testing Results

The runtime tester provides results in these formats:

#### Script Test Results
```
🚀 Script Runtime Testing
1️⃣ Testing Script: tabular_preprocessing
  ✅ PASS (2.345s)
  Has main function: Yes
```

#### Data Compatibility Results
```
🔗 Data Compatibility Testing
Testing: tabular_preprocessing -> xgboost_training
  ✅ COMPATIBLE
  Data formats: CSV -> CSV
```

#### Pipeline Flow Results
```
🚀 Pipeline Flow Runtime Testing
🔄 Testing Pipeline Flow...
  ✅ PIPELINE FLOW PASSED

📝 Individual Script Results:
  ✅ data_preprocessing: PASS
  ✅ model_training: PASS
  ✅ model_evaluation: PASS

🔗 Data Flow Results:
  ✅ data_preprocessing->model_training: PASS
  ✅ model_training->model_evaluation: PASS
```

### Common Runtime Testing Issues

**Issue**: Script missing main function
```
❌ FAIL: Script missing main() function
💡 Add main(input_paths, output_paths, environ_vars, job_args) function
```
**Solution**: Ensure your script implements the standardized main function interface from the script development guide.

**Issue**: Script execution timeout
```
❌ FAIL: Script execution timed out after 300 seconds
```
**Solution**: Check for infinite loops or increase timeout in RuntimeTestingConfiguration.

**Issue**: Data compatibility failure
```
❌ INCOMPATIBLE: Script B failed with script A output: KeyError: 'required_column'
```
**Solution**: Ensure output data from script A contains all columns expected by script B.

**Issue**: Import errors
```
❌ FAIL: ModuleNotFoundError: No module named 'custom_module'
```
**Solution**: Check that all required dependencies are installed and importable.

### Integration with Existing Validation

Runtime testing integrates seamlessly with the existing validation frameworks:

```python
def run_comprehensive_validation(step_name: str, builder_class, pipeline_dag=None):
    """Run all three validation frameworks for complete coverage."""
    
    # 1. Alignment validation (existing)
    alignment_results = run_alignment_validation(step_name)
    
    # 2. Builder validation (existing)  
    builder_results = run_builder_validation(step_name, builder_class)
    
    # 3. Runtime validation (NEW)
    runtime_results = run_runtime_validation(step_name, pipeline_dag)
    
    # Overall validation status
    overall_passed = all([
        alignment_results.get('overall_status') == 'PASSING',
        builder_results.get('overall_passed', False),
        runtime_results.get('success', False)
    ])
    
    print(f"\n🎯 Comprehensive Validation Summary:")
    print(f"✅ Alignment Validation: {'PASS' if alignment_results.get('overall_status') == 'PASSING' else 'FAIL'}")
    print(f"✅ Builder Validation: {'PASS' if builder_results.get('overall_passed', False) else 'FAIL'}")
    print(f"✅ Runtime Validation: {'PASS' if runtime_results.get('success', False) else 'FAIL'}")
    print(f"\n🏆 Overall Result: {'✅ READY FOR INTEGRATION' if overall_passed else '❌ NEEDS FIXES'}")
    
    return {
        'alignment': alignment_results,
        'builder': builder_results, 
        'runtime': runtime_results,
        'overall_passed': overall_passed
    }
```

## Recommended Validation Workflow

### 1. Initial Validation Run

Start with CLI commands for quick feedback across all three validation frameworks:

```bash
# Quick alignment check
python -m cursus.cli.alignment_cli validate your_step_name --verbose

# Quick builder test
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_step.YourStepBuilder --verbose

# Quick runtime test
cursus runtime test-script your_script_name --workspace-dir ./test_workspace --verbose
```

### 2. Detailed Analysis

If issues are found, run with scoring and reports:

```bash
# Detailed alignment analysis with scoring
python -m cursus.cli.alignment_cli validate your_step_name --verbose --show-scoring

# Detailed builder analysis with scoring and exports
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_step.YourStepBuilder --scoring --verbose --export-json ./reports/results.json

# Detailed runtime testing with JSON output
cursus runtime test-script your_script_name --output-format json --workspace-dir ./test_workspace
```

### 3. Comprehensive Validation

Before final integration, run comprehensive tests across all frameworks:

```bash
# Full alignment validation with visualization
python -m cursus.cli.alignment_cli visualize your_step_name --output-dir ./validation_reports --verbose

# Full builder validation with charts
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_step.YourStepBuilder --export-chart --output-dir ./reports --scoring

# Full runtime validation with pipeline flow testing
cursus runtime test-pipeline pipeline_config.json --workspace-dir ./test_workspace --verbose
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

### 5. Data Flow Validation (NEW)

Test data compatibility between connected scripts:

```bash
# Test data compatibility between adjacent pipeline steps
cursus runtime test-compatibility script_a script_b --workspace-dir ./test_workspace --verbose

# Test complete pipeline data flow
cursus runtime test-pipeline pipeline_config.json --workspace-dir ./test_workspace --verbose
```

### 6. Complete Validation Workflow

For comprehensive validation, run all three frameworks in sequence:

```python
#!/usr/bin/env python3
"""
Complete validation workflow for pipeline steps.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from cursus.validation.builders.universal_test import UniversalStepBuilderTest
from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_models import RuntimeTestingConfiguration

def run_complete_validation_workflow(step_name: str, builder_class, script_name: str):
    """Run complete validation workflow with all three frameworks."""
    
    print("🎯 Complete Validation Workflow")
    print("=" * 80)
    
    validation_results = {
        'alignment': None,
        'builder': None,
        'runtime': None,
        'overall_passed': False
    }
    
    # 1. Alignment Validation
    print("\n1️⃣ Running Alignment Validation...")
    try:
        alignment_tester = UnifiedAlignmentTester(
            scripts_dir=str(project_root / "src" / "cursus" / "steps" / "scripts"),
            contracts_dir=str(project_root / "src" / "cursus" / "steps" / "contracts"),
            specs_dir=str(project_root / "src" / "cursus" / "steps" / "specs"),
            builders_dir=str(project_root / "src" / "cursus" / "steps" / "builders"),
            configs_dir=str(project_root / "src" / "cursus" / "steps" / "configs")
        )
        
        alignment_results = alignment_tester.validate_specific_script(step_name)
        alignment_passed = alignment_results.get('overall_status') == 'PASSING'
        validation_results['alignment'] = alignment_results
        
        print(f"   {'✅' if alignment_passed else '❌'} Alignment Validation: {'PASS' if alignment_passed else 'FAIL'}")
        
    except Exception as e:
        print(f"   ❌ Alignment Validation: ERROR - {e}")
        alignment_passed = False
    
    # 2. Builder Validation
    print("\n2️⃣ Running Builder Validation...")
    try:
        builder_tester = UniversalStepBuilderTest(
            builder_class,
            verbose=False,
            enable_scoring=True
        )
        
        builder_results = builder_tester.run_all_tests()
        test_results = builder_results.get('test_results', builder_results)
        passed_tests = sum(1 for result in test_results.values() 
                          if isinstance(result, dict) and result.get("passed", False))
        total_tests = len([r for r in test_results.values() if isinstance(r, dict)])
        builder_passed = passed_tests == total_tests
        validation_results['builder'] = builder_results
        
        print(f"   {'✅' if builder_passed else '❌'} Builder Validation: {'PASS' if builder_passed else 'FAIL'} ({passed_tests}/{total_tests})")
        
    except Exception as e:
        print(f"   ❌ Builder Validation: ERROR - {e}")
        builder_passed = False
    
    # 3. Runtime Validation
    print("\n3️⃣ Running Runtime Validation...")
    try:
        config = RuntimeTestingConfiguration(
            workspace_dir="./test_workspace",
            timeout_seconds=300,
            enable_logging=False,  # Quiet for workflow
            cleanup_after_test=True
        )
        
        runtime_tester = RuntimeTester(config)
        runtime_result = runtime_tester.test_script(script_name)
        runtime_passed = runtime_result.success
        validation_results['runtime'] = runtime_result
        
        print(f"   {'✅' if runtime_passed else '❌'} Runtime Validation: {'PASS' if runtime_passed else 'FAIL'}")
        if not runtime_passed:
            print(f"      Error: {runtime_result.error_message}")
        
    except Exception as e:
        print(f"   ❌ Runtime Validation: ERROR - {e}")
        runtime_passed = False
    
    # Overall Results
    overall_passed = alignment_passed and builder_passed and runtime_passed
    validation_results['overall_passed'] = overall_passed
    
    print(f"\n🏆 Complete Validation Summary:")
    print("=" * 80)
    print(f"✅ Alignment Validation: {'PASS' if alignment_passed else 'FAIL'}")
    print(f"✅ Builder Validation: {'PASS' if builder_passed else 'FAIL'}")
    print(f"✅ Runtime Validation: {'PASS' if runtime_passed else 'FAIL'}")
    print(f"\n🎯 Overall Result: {'✅ READY FOR INTEGRATION' if overall_passed else '❌ NEEDS FIXES'}")
    
    if not overall_passed:
        print(f"\n💡 Next Steps:")
        if not alignment_passed:
            print(f"   • Fix alignment issues between script, contract, specification, and builder")
        if not builder_passed:
            print(f"   • Fix builder implementation and integration issues")
        if not runtime_passed:
            print(f"   • Fix script execution issues and ensure main function compliance")
    
    return validation_results

# Usage example:
# from cursus.steps.builders.builder_your_step import YourStepBuilder
# results = run_complete_validation_workflow("your_step", YourStepBuilder, "your_script")
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

### Core Development Guides
- [Adding New Pipeline Step](adding_new_pipeline_step.md) - Main developer guide with overview and quick start
- [Step Creation Process](creation_process.md) - Detailed step-by-step creation process
- [Prerequisites](prerequisites.md) - Required information before starting development

### Component-Specific Guides
- [Step Builder Guide](step_builder.md) - Detailed step builder implementation patterns
- [Script Contract Development](script_contract.md) - Script contract creation guide
- [Step Specification Development](step_specification.md) - Step specification creation guide
- [Three-Tier Config Design](three_tier_config_design.md) - Configuration design patterns
- [Step Builder Registry Guide](step_builder_registry_guide.md) - Registry usage and auto-discovery
- [Hyperparameter Class Guide](hyperparameter_class.md) - Adding hyperparameter classes for training steps

### Rules and Standards
- [Design Principles](design_principles.md) - Core architectural principles
- [Best Practices](best_practices.md) - Recommended development practices
- [Standardization Rules](standardization_rules.md) - Coding and naming conventions
- [Alignment Rules](alignment_rules.md) - Component alignment requirements
- [Common Pitfalls](common_pitfalls.md) - Common mistakes to avoid

### Validation and Testing
- [Validation Checklist](validation_checklist.md) - Pre-integration validation checklist
- [Example](example.md) - Complete step implementation example

### Reference Materials
- [SageMaker Property Path Reference Database](sagemaker_property_path_reference_database.md) - Property path reference guide
- [Config Field Manager Guide](config_field_manager_guide.md) - Configuration field management
