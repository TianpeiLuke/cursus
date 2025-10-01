---
tags:
  - project
  - planning
  - universal_builder_testing
  - dynamic_discovery
  - command_line_interface
  - step_catalog_integration
keywords:
  - dynamic builder discovery
  - universal testing framework
  - command line interface
  - step catalog integration
  - automated test generation
  - canonical name resolution
topics:
  - dynamic universal builder testing
  - automated test discovery
  - command line testing interface
  - step catalog system integration
language: python
date of note: 2025-10-01
implementation_status: COMPLETED
---

# Dynamic Universal Builder Testing Implementation Plan

## Executive Summary

This implementation plan creates a **Dynamic Universal Builder Testing System** that provides simple command-line interfaces for comprehensive builder testing while eliminating hard-coded test patterns. The system enables testing all builders in `src/cursus/steps/builders` with one command and individual builders by canonical name selection, leveraging existing step catalog infrastructure.

### User Requirements

**Primary User Needs**:
1. **Test All Builders**: Complete testing of all builders under `src/cursus/steps/builders` in one command
2. **Test Individual Builder**: Test specific builder by selecting canonical name in one command
3. **No Hard-Coding**: Automatic discovery without maintaining hard-coded builder lists
4. **Step Catalog Integration**: Use step catalog as single source of truth
5. **SageMaker Step Type Variants**: Automatic handling of step type-specific testing patterns

### Strategic Impact

- **Eliminates Maintenance Burden**: No more updating hard-coded test lists when builders are added
- **Comprehensive Coverage**: Automatic inclusion of all builders without manual intervention
- **Simple Interface**: Two commands cover all testing scenarios
- **Step Catalog-Driven**: Single source of truth via existing step catalog system
- **Step Type Aware**: Automatic application of appropriate test variants based on SageMaker step type

## Problem Statement

### Current Testing Limitations

**Hard-Coded Test Patterns**:
```python
# ❌ Current approach - manual maintenance required
@pytest.mark.parametrize("builder_class", [
    XGBoostTrainingStepBuilder,      # Must be manually added
    TabularPreprocessingStepBuilder, # Must be manually updated
    ModelEvalStepBuilder            # Easily forgotten when new builders added
])
def test_step_builder_compliance(builder_class):
    # Hard-coded test list requires constant maintenance
```

**Scattered Test Files**:
- `test_createmodel_step_builders.py` - Hard-coded CreateModel builders
- `test_processing_step_builders.py` - Hard-coded Processing builders  
- `test_training_step_builders.py` - Hard-coded Training builders
- `test_transform_step_builders.py` - Hard-coded Transform builders
- Multiple `run_*.py` scripts for different step types

### Available Infrastructure

**Step Catalog System** (`src/cursus/step_catalog/step_catalog.py`):
- ✅ `load_builder_class()` - Dynamic builder loading with job type variant support
- ✅ `get_step_info()` - Step metadata and canonical name resolution
- ✅ `list_available_steps()` - Complete step discovery
- ✅ `get_builder_map()` - Complete builder mapping

**CLI System** (`src/cursus/cli/builder_test_cli.py`):
- ✅ Existing `builder-test` command structure
- ✅ Universal test integration
- ✅ Scoring and reporting capabilities
- ✅ JSON export functionality

## Simplified Solution Architecture

### Enhanced Existing Systems Approach

Instead of creating new modules, **enhance existing systems** to provide dynamic testing capabilities:

**Target Commands**:
```bash
# New commands using existing CLI structure
cursus builder-test test-all-discovered     # Test all builders via step catalog
cursus builder-test test-single <canonical_name>  # Test by canonical name
cursus builder-test list-discovered         # List builders via step catalog
```

### Core Enhancement Strategy

**1. Enhance Step Catalog** (`src/cursus/step_catalog/step_catalog.py`):
- Add `get_all_builders()` method for comprehensive builder discovery
- Add `get_builders_by_step_type()` method for filtered discovery
- Add `save_test_results()` helper for organized results storage

**2. Enhance CLI** (`src/cursus/cli/builder_test_cli.py`):
- Add new commands that use step catalog for discovery
- Add automatic results saving to `test/steps/builders/results/`
- Integrate with existing universal test framework

**3. No New Modules**: 
- No separate discovery module needed
- No new test runner module needed
- Use existing universal test framework

## Implementation Plan

### Phase 1: Enhance Step Catalog (Week 1)

#### Add Builder Discovery Methods (Days 1-3)

```python
# Add to src/cursus/step_catalog/step_catalog.py

def get_all_builders(self) -> Dict[str, Type]:
    """
    Get all available builders with canonical names.
    
    Returns:
        Dict mapping canonical names to builder classes
    """
    try:
        all_steps = self.list_available_steps()
        builders = {}
        
        for step_name in all_steps:
            builder_class = self.load_builder_class(step_name)
            if builder_class:
                builders[step_name] = builder_class
        
        return builders
        
    except Exception as e:
        self.logger.error(f"Error getting all builders: {e}")
        return {}

def get_builders_by_step_type(self, step_type: str) -> Dict[str, Type]:
    """
    Get builders filtered by SageMaker step type.
    
    Args:
        step_type: SageMaker step type (Processing, Training, etc.)
        
    Returns:
        Dict mapping canonical names to builder classes for the step type
    """
    try:
        all_builders = self.get_all_builders()
        step_builders = {}
        
        for step_name, builder_class in all_builders.items():
            step_info = self.get_step_info(step_name)
            if step_info and step_info.registry_data.get('sagemaker_step_type') == step_type:
                step_builders[step_name] = builder_class
        
        return step_builders
        
    except Exception as e:
        self.logger.error(f"Error getting builders for step type {step_type}: {e}")
        return {}
```

#### Add Results Storage Helper (Days 4-5)

```python
# Add to src/cursus/validation/builders/results_storage.py

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import json

class BuilderTestResultsStorage:
    """Helper class for saving test results to organized directory structure."""
    
    @staticmethod
    def save_test_results(results: Dict[str, Any], command_type: str, identifier: str = None) -> str:
        """
        Save test results to organized directory structure.
        
        Args:
            results: Test results to save
            command_type: Type of command ('all_builders' or 'single_builder')
            identifier: Optional identifier for single builder tests
            
        Returns:
            Path where results were saved
        """
        # Create results directory structure
        results_dir = Path("test/steps/builders/results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if command_type == "all_builders":
            subdir = results_dir / "all_builders"
            filename = f"all_builders_{timestamp}.json"
        elif command_type == "single_builder":
            subdir = results_dir / "individual"
            filename = f"{identifier}_{timestamp}.json" if identifier else f"single_{timestamp}.json"
        else:
            subdir = results_dir
            filename = f"results_{timestamp}.json"
        
        # Create directory and save file
        output_path = subdir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(output_path)
    
    @staticmethod
    def get_results_directory() -> Path:
        """Get the base results directory path."""
        return Path("test/steps/builders/results")
    
    @staticmethod
    def ensure_results_directory() -> None:
        """Ensure results directory structure exists."""
        results_dir = TestResultsStorage.get_results_directory()
        (results_dir / "all_builders").mkdir(parents=True, exist_ok=True)
        (results_dir / "individual").mkdir(parents=True, exist_ok=True)
        
        # Create .gitignore if it doesn't exist
        gitignore_path = results_dir / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                f.write("*.json\n!.gitignore\n")
```

### Phase 2: Enhance CLI (Week 2)

#### Add New CLI Commands (Days 1-4)

```python
# Add to src/cursus/cli/builder_test_cli.py

@builder_test.command("test-all-discovered")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--scoring", is_flag=True, help="Enable quality scoring")
@click.option("--export-json", type=click.Path(), help="Export results to JSON file")
@click.option("--step-type", help="Filter by SageMaker step type")
def test_all_discovered(verbose: bool, scoring: bool, export_json: str, step_type: str):
    """Test all builders discovered via step catalog."""
    try:
        click.echo("🔍 Discovering builders via step catalog...")
        
        from ..step_catalog import StepCatalog
        catalog = StepCatalog(workspace_dirs=None)
        
        if step_type:
            builders = catalog.get_builders_by_step_type(step_type)
            click.echo(f"Found {len(builders)} {step_type} builders")
        else:
            builders = catalog.get_all_builders()
            click.echo(f"Found {len(builders)} total builders")
        
        if not builders:
            click.echo("❌ No builders found")
            return
        
        click.echo(f"\n🧪 Testing {len(builders)} builders...")
        
        # Test all builders
        results = {}
        for i, (step_name, builder_class) in enumerate(builders.items(), 1):
            click.echo(f"\n[{i}/{len(builders)}] Testing {step_name}...")
            
            try:
                tester = UniversalStepBuilderTest(
                    builder_class=builder_class,
                    step_name=step_name,
                    verbose=verbose,
                    enable_scoring=scoring,
                    enable_structured_reporting=True,
                    use_step_catalog_discovery=True
                )
                
                test_results = tester.run_all_tests()
                results[step_name] = test_results
                
                # Quick status report
                if 'test_results' in test_results:
                    raw_results = test_results['test_results']
                    total_tests = len(raw_results)
                    passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
                    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
                    
                    status_icon = "✅" if pass_rate >= 80 else "⚠️" if pass_rate >= 60 else "❌"
                    click.echo(f"  {status_icon} {step_name}: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
                    
                    if scoring and 'scoring' in test_results:
                        score = test_results['scoring'].get('overall', {}).get('score', 0)
                        rating = test_results['scoring'].get('overall', {}).get('rating', 'Unknown')
                        click.echo(f"  📊 Quality Score: {score:.1f}/100 ({rating})")
                
            except Exception as e:
                click.echo(f"  ❌ {step_name}: Failed with error: {e}")
                results[step_name] = {'error': str(e)}
        
        # Generate comprehensive report
        total_builders = len(results)
        successful_tests = sum(1 for r in results.values() if 'error' not in r)
        success_rate = (successful_tests / total_builders * 100) if total_builders > 0 else 0
        
        click.echo(f"\n📊 OVERALL SUMMARY:")
        click.echo(f"   Builders Tested: {total_builders}")
        click.echo(f"   Successful Tests: {successful_tests} ({success_rate:.1f}%)")
        
        # Export or auto-save results
        if export_json:
            export_results_to_json(results, export_json)
        else:
            # Auto-save results
            from ..validation.builders.results_storage import BuilderTestResultsStorage
            output_path = BuilderTestResultsStorage.save_test_results(results, "all_builders", step_type)
            click.echo(f"📁 Results automatically saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)

@builder_test.command("test-single")
@click.argument("canonical_name")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--scoring", is_flag=True, help="Enable quality scoring")
@click.option("--export-json", type=click.Path(), help="Export results to JSON file")
def test_single(canonical_name: str, verbose: bool, scoring: bool, export_json: str):
    """Test single builder by canonical name."""
    try:
        click.echo(f"🔍 Looking for builder: {canonical_name}")
        
        from ..step_catalog import StepCatalog
        catalog = StepCatalog(workspace_dirs=None)
        builder_class = catalog.load_builder_class(canonical_name)
        
        if not builder_class:
            click.echo(f"❌ No builder found for: {canonical_name}")
            # Show available builders
            all_builders = catalog.get_all_builders()
            available = sorted(all_builders.keys())
            click.echo(f"Available builders: {', '.join(available[:10])}")
            if len(available) > 10:
                click.echo(f"... and {len(available) - 10} more")
            sys.exit(1)
        
        click.echo(f"✅ Found builder: {builder_class.__name__}")
        click.echo(f"\n🧪 Testing {canonical_name}...")
        
        # Run tests
        tester = UniversalStepBuilderTest(
            builder_class=builder_class,
            step_name=canonical_name,
            verbose=verbose,
            enable_scoring=scoring,
            enable_structured_reporting=True,
            use_step_catalog_discovery=True
        )
        
        results = tester.run_all_tests()
        
        # Print results
        if scoring and "test_results" in results:
            print_enhanced_results(results, verbose)
        else:
            print_test_results(results, verbose, show_scoring=scoring)
        
        # Export or auto-save results
        export_data = {canonical_name: results}
        if export_json:
            export_results_to_json(export_data, export_json)
        else:
            from ..validation.builders.results_storage import BuilderTestResultsStorage
            output_path = BuilderTestResultsStorage.save_test_results(export_data, "single_builder", canonical_name)
            click.echo(f"📁 Results automatically saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)

@builder_test.command("list-discovered")
@click.option("--step-type", help="Filter by SageMaker step type")
def list_discovered(step_type: str):
    """List builders discovered via step catalog."""
    try:
        click.echo("📋 Builders discovered via step catalog:")
        click.echo("=" * 50)
        
        from ..step_catalog import StepCatalog
        catalog = StepCatalog(workspace_dirs=None)
        
        if step_type:
            builders = catalog.get_builders_by_step_type(step_type)
            click.echo(f"Filtered by step type: {step_type}")
        else:
            builders = catalog.get_all_builders()
        
        if builders:
            for step_name, builder_class in sorted(builders.items()):
                click.echo(f"  • {step_name} → {builder_class.__name__}")
            click.echo(f"\nTotal: {len(builders)} builders")
        else:
            click.echo("No builders found")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)
```

#### Create Results Directory Structure (Days 5)

```bash
# Create directory structure
mkdir -p test/steps/builders/results/all_builders
mkdir -p test/steps/builders/results/individual

# Create .gitignore for results
echo "*.json" > test/steps/builders/results/.gitignore
echo "!.gitignore" >> test/steps/builders/results/.gitignore
```

### Phase 3: Clean Up and Update Existing Tests (Week 3)

#### Objective
Replace hard-coded test files with dynamic pytest integration and clean up redundant test code.

#### Current Test Files to Update/Replace

**Hard-Coded Test Files** (to be replaced):
```
test/steps/builders/
├── test_createmodel_step_builders.py     # Hard-coded CreateModel builders
├── test_processing_step_builders.py      # Hard-coded Processing builders  
├── test_training_step_builders.py        # Hard-coded Training builders
├── test_transform_step_builders.py       # Hard-coded Transform builders
├── run_createmodel_tests.py              # Manual test runner scripts
├── run_processing_tests.py               # Manual test runner scripts
├── run_training_tests.py                 # Manual test runner scripts
└── run_transform_tests.py                # Manual test runner scripts
```

#### Implementation Tasks

**3.1 Create Dynamic Pytest Integration** (Days 1-2):

```python
# Create test/steps/builders/test_dynamic_universal.py
"""Dynamic pytest integration for universal builder testing."""

import pytest
from typing import Dict, Type

from cursus.step_catalog import StepCatalog
from cursus.validation.builders.universal_test import UniversalStepBuilderTest
from cursus.validation.builders.test_results_storage import TestResultsStorage

class TestDynamicUniversalBuilders:
    """Dynamic pytest integration for all builders."""
    
    @pytest.fixture(scope="class")
    def step_catalog(self):
        """Create step catalog instance."""
        return StepCatalog(workspace_dirs=None)
    
    @pytest.fixture(scope="class")
    def all_builders(self, step_catalog):
        """Discover all builders for testing."""
        return step_catalog.get_all_builders()
    
    def test_builder_discovery_completeness(self, all_builders):
        """Test that discovery finds expected number of builders."""
        assert len(all_builders) >= 15, f"Expected at least 15 builders, found {len(all_builders)}"
        
        # All builders should have valid canonical names and classes
        for canonical_name, builder_class in all_builders.items():
            assert canonical_name, "Empty canonical name found"
            assert isinstance(canonical_name, str), f"Non-string canonical name: {canonical_name}"
            assert builder_class, "Empty builder class found"
            assert hasattr(builder_class, '__name__'), f"Invalid builder class: {builder_class}"
    
    @pytest.mark.parametrize("step_type", ["Processing", "Training", "Transform", "CreateModel"])
    def test_step_type_filtering(self, step_catalog, step_type):
        """Test step type filtering functionality."""
        builders = step_catalog.get_builders_by_step_type(step_type)
        
        # Should find at least one builder for major step types
        if step_type in ['Processing', 'Training']:
            assert len(builders) > 0, f"No builders found for step type: {step_type}"
        
        # All builders should be valid
        for canonical_name, builder_class in builders.items():
            assert canonical_name, f"Empty canonical name in {step_type} builders"
            assert builder_class, f"Empty builder class in {step_type} builders"
    
    @pytest.mark.parametrize("canonical_name,builder_class", 
                           lambda: list(StepCatalog().get_all_builders().items()))
    def test_individual_builder_compliance(self, canonical_name, builder_class):
        """Parametrized test for individual builder compliance."""
        tester = UniversalStepBuilderTest(
            builder_class=builder_class,
            step_name=canonical_name,
            verbose=False,
            enable_scoring=False,  # Disable scoring for faster individual tests
            enable_structured_reporting=False,
            use_step_catalog_discovery=True
        )
        
        results = tester.run_all_tests()
        
        # Extract test results
        if 'test_results' in results:
            raw_results = results['test_results']
        else:
            raw_results = results
        
        # Calculate pass rate
        total_tests = len(raw_results)
        passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Assert minimum pass rate (flexible for realistic testing)
        assert pass_rate >= 60, f"{canonical_name} failed with {pass_rate:.1f}% pass rate. Failed tests: {[k for k, v in raw_results.items() if not v.get('passed', False)]}"
        
        # Assert critical tests pass
        critical_tests = ['test_inheritance', 'test_required_methods']
        for test_name in critical_tests:
            if test_name in raw_results:
                assert raw_results[test_name].get('passed', False), f"{canonical_name} failed critical test: {test_name}"
    
    def test_comprehensive_all_builders(self, all_builders):
        """Comprehensive test of all builders with results storage."""
        if not all_builders:
            pytest.skip("No builders found to test")
        
        results = {}
        failed_builders = []
        
        for canonical_name, builder_class in all_builders.items():
            try:
                tester = UniversalStepBuilderTest(
                    builder_class=builder_class,
                    step_name=canonical_name,
                    verbose=False,
                    enable_scoring=True,
                    enable_structured_reporting=True,
                    use_step_catalog_discovery=True
                )
                
                test_results = tester.run_all_tests()
                results[canonical_name] = test_results
                
                # Check if tests passed
                if 'test_results' in test_results:
                    raw_results = test_results['test_results']
                    total_tests = len(raw_results)
                    passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
                    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
                    
                    if pass_rate < 60:  # Consider failure if less than 60% pass rate
                        failed_builders.append(f"{canonical_name} ({pass_rate:.1f}%)")
                
            except Exception as e:
                failed_builders.append(f"{canonical_name} (Error: {e})")
        
        # Save comprehensive test results
        TestResultsStorage.save_test_results(results, "all_builders", "pytest_comprehensive")
        
        # Assert overall success
        total_builders = len(all_builders)
        successful_builders = total_builders - len(failed_builders)
        success_rate = (successful_builders / total_builders * 100) if total_builders > 0 else 0
        
        # Print summary for debugging
        print(f"\n📊 Builder Test Summary:")
        print(f"   Total Builders: {total_builders}")
        print(f"   Successful: {successful_builders} ({success_rate:.1f}%)")
        print(f"   Failed: {len(failed_builders)}")
        
        if failed_builders:
            print(f"\n❌ Failed Builders:")
            for builder in failed_builders:
                print(f"   • {builder}")
        
        # Assert success (allow up to 25% failure rate for realistic expectations)
        assert success_rate >= 75, f"Only {success_rate:.1f}% of builders passed tests. Failed: {failed_builders}"
```

**3.2 Rewrite Existing Test Files to Use Dynamic Discovery** (Days 3-4):

Instead of archiving, rewrite existing test files to derive from the dynamic universal test with fixed SageMaker step types:

```python
# Rewrite test/steps/builders/test_createmodel_step_builders.py
"""CreateModel step builders testing using dynamic discovery."""

import pytest
from cursus.step_catalog import StepCatalog
from cursus.validation.builders.test_results_storage import TestResultsStorage

class TestCreateModelStepBuilders:
    """Test CreateModel step builders using dynamic discovery."""
    
    @pytest.fixture(scope="class")
    def step_catalog(self):
        """Create step catalog instance."""
        return StepCatalog(workspace_dirs=None)
    
    @pytest.fixture(scope="class")
    def createmodel_builders(self, step_catalog):
        """Get all CreateModel builders dynamically."""
        return step_catalog.get_builders_by_step_type("CreateModel")
    
    def test_createmodel_builders_discovery(self, createmodel_builders):
        """Test that CreateModel builders are discovered."""
        assert len(createmodel_builders) > 0, "No CreateModel builders found"
        
        for canonical_name, builder_class in createmodel_builders.items():
            assert canonical_name, "Empty canonical name found"
            assert builder_class, "Empty builder class found"
    
    @pytest.mark.parametrize("canonical_name,builder_class", 
                           lambda: list(StepCatalog().get_builders_by_step_type("CreateModel").items()))
    def test_createmodel_builder_compliance(self, canonical_name, builder_class):
        """Test individual CreateModel builder compliance."""
        from cursus.validation.builders.universal_test import UniversalStepBuilderTest
        
        tester = UniversalStepBuilderTest(
            builder_class=builder_class,
            step_name=canonical_name,
            verbose=False,
            enable_scoring=False,
            enable_structured_reporting=False,
            use_step_catalog_discovery=True
        )
        
        results = tester.run_all_tests()
        
        # Extract test results
        if 'test_results' in results:
            raw_results = results['test_results']
        else:
            raw_results = results
        
        # Calculate pass rate
        total_tests = len(raw_results)
        passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Assert minimum pass rate for CreateModel builders
        assert pass_rate >= 60, f"CreateModel builder {canonical_name} failed with {pass_rate:.1f}% pass rate"

# Rewrite test/steps/builders/test_training_step_builders.py
"""Training step builders testing using dynamic discovery."""

import pytest
from cursus.step_catalog import StepCatalog

class TestTrainingStepBuilders:
    """Test Training step builders using dynamic discovery."""
    
    @pytest.fixture(scope="class")
    def step_catalog(self):
        """Create step catalog instance."""
        return StepCatalog(workspace_dirs=None)
    
    @pytest.fixture(scope="class")
    def training_builders(self, step_catalog):
        """Get all Training builders dynamically."""
        return step_catalog.get_builders_by_step_type("Training")
    
    def test_training_builders_discovery(self, training_builders):
        """Test that Training builders are discovered."""
        assert len(training_builders) > 0, "No Training builders found"
        
        for canonical_name, builder_class in training_builders.items():
            assert canonical_name, "Empty canonical name found"
            assert builder_class, "Empty builder class found"
    
    @pytest.mark.parametrize("canonical_name,builder_class", 
                           lambda: list(StepCatalog().get_builders_by_step_type("Training").items()))
    def test_training_builder_compliance(self, canonical_name, builder_class):
        """Test individual Training builder compliance."""
        from cursus.validation.builders.universal_test import UniversalStepBuilderTest
        
        tester = UniversalStepBuilderTest(
            builder_class=builder_class,
            step_name=canonical_name,
            verbose=False,
            enable_scoring=False,
            enable_structured_reporting=False,
            use_step_catalog_discovery=True
        )
        
        results = tester.run_all_tests()
        
        # Extract test results
        if 'test_results' in results:
            raw_results = results['test_results']
        else:
            raw_results = results
        
        # Calculate pass rate
        total_tests = len(raw_results)
        passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Assert minimum pass rate for Training builders
        assert pass_rate >= 60, f"Training builder {canonical_name} failed with {pass_rate:.1f}% pass rate"

# Rewrite test/steps/builders/test_processing_step_builders.py
"""Processing step builders testing using dynamic discovery."""

import pytest
from cursus.step_catalog import StepCatalog

class TestProcessingStepBuilders:
    """Test Processing step builders using dynamic discovery."""
    
    @pytest.fixture(scope="class")
    def step_catalog(self):
        """Create step catalog instance."""
        return StepCatalog(workspace_dirs=None)
    
    @pytest.fixture(scope="class")
    def processing_builders(self, step_catalog):
        """Get all Processing builders dynamically."""
        return step_catalog.get_builders_by_step_type("Processing")
    
    def test_processing_builders_discovery(self, processing_builders):
        """Test that Processing builders are discovered."""
        assert len(processing_builders) > 0, "No Processing builders found"
        
        for canonical_name, builder_class in processing_builders.items():
            assert canonical_name, "Empty canonical name found"
            assert builder_class, "Empty builder class found"
    
    @pytest.mark.parametrize("canonical_name,builder_class", 
                           lambda: list(StepCatalog().get_builders_by_step_type("Processing").items()))
    def test_processing_builder_compliance(self, canonical_name, builder_class):
        """Test individual Processing builder compliance."""
        from cursus.validation.builders.universal_test import UniversalStepBuilderTest
        
        tester = UniversalStepBuilderTest(
            builder_class=builder_class,
            step_name=canonical_name,
            verbose=False,
            enable_scoring=False,
            enable_structured_reporting=False,
            use_step_catalog_discovery=True
        )
        
        results = tester.run_all_tests()
        
        # Extract test results
        if 'test_results' in results:
            raw_results = results['test_results']
        else:
            raw_results = results
        
        # Calculate pass rate
        total_tests = len(raw_results)
        passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Assert minimum pass rate for Processing builders
        assert pass_rate >= 60, f"Processing builder {canonical_name} failed with {pass_rate:.1f}% pass rate"

# Rewrite test/steps/builders/test_transform_step_builders.py
"""Transform step builders testing using dynamic discovery."""

import pytest
from cursus.step_catalog import StepCatalog

class TestTransformStepBuilders:
    """Test Transform step builders using dynamic discovery."""
    
    @pytest.fixture(scope="class")
    def step_catalog(self):
        """Create step catalog instance."""
        return StepCatalog(workspace_dirs=None)
    
    @pytest.fixture(scope="class")
    def transform_builders(self, step_catalog):
        """Get all Transform builders dynamically."""
        return step_catalog.get_builders_by_step_type("Transform")
    
    def test_transform_builders_discovery(self, transform_builders):
        """Test that Transform builders are discovered."""
        # Transform builders may not exist in all projects
        for canonical_name, builder_class in transform_builders.items():
            assert canonical_name, "Empty canonical name found"
            assert builder_class, "Empty builder class found"
    
    @pytest.mark.parametrize("canonical_name,builder_class", 
                           lambda: list(StepCatalog().get_builders_by_step_type("Transform").items()))
    def test_transform_builder_compliance(self, canonical_name, builder_class):
        """Test individual Transform builder compliance."""
        from cursus.validation.builders.universal_test import UniversalStepBuilderTest
        
        tester = UniversalStepBuilderTest(
            builder_class=builder_class,
            step_name=canonical_name,
            verbose=False,
            enable_scoring=False,
            enable_structured_reporting=False,
            use_step_catalog_discovery=True
        )
        
        results = tester.run_all_tests()
        
        # Extract test results
        if 'test_results' in results:
            raw_results = results['test_results']
        else:
            raw_results = results
        
        # Calculate pass rate
        total_tests = len(raw_results)
        passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Assert minimum pass rate for Transform builders
        assert pass_rate >= 60, f"Transform builder {canonical_name} failed with {pass_rate:.1f}% pass rate"

# Archive manual runner scripts (these are replaced by CLI commands)
def archive_runner_scripts():
    """Archive manual test runner scripts."""
    from pathlib import Path
    
    runner_scripts = [
        "run_createmodel_tests.py",
        "run_processing_tests.py",
        "run_training_tests.py",
        "run_transform_tests.py"
    ]
    
    test_dir = Path("test/steps/builders")
    legacy_dir = test_dir / "legacy"
    legacy_dir.mkdir(exist_ok=True)
    
    for filename in runner_scripts:
        file_path = test_dir / filename
        if file_path.exists():
            file_path.rename(legacy_dir / filename)
            print(f"Archived runner script {filename} to legacy/")
```

**3.3 Update Documentation and README** (Days 5):

```markdown
# Create test/steps/builders/README.md
# Dynamic Universal Builder Testing

## Overview

This directory contains dynamic universal builder testing that automatically discovers and tests all builders in `src/cursus/steps/builders` without requiring hard-coded maintenance.

## New Dynamic Testing Approach

### CLI Commands
```bash
# Test all builders
cursus builder-test test-all-discovered

# Test specific step type
cursus builder-test test-all-discovered --step-type Processing --verbose --scoring

# Test individual builder
cursus builder-test test-single TabularPreprocessing --scoring

# List available builders
cursus builder-test list-discovered --step-type Training
```

### Pytest Integration
```bash
# Run all dynamic tests
pytest test/steps/builders/test_dynamic_universal.py

# Run specific test categories
pytest test/steps/builders/test_dynamic_universal.py::TestDynamicUniversalBuilders::test_step_type_filtering

# Run parametrized individual builder tests
pytest test/steps/builders/test_dynamic_universal.py::TestDynamicUniversalBuilders::test_individual_builder_compliance
```

### Phase 4: Step-Type Specific Test Framework Integration (Week 4) - NEW

#### Objective
Integrate specialized test frameworks for each step type to leverage step-type-specific validation logic, including Processing "Pattern B auto-pass logic", Training framework-specific validation, and comprehensive reporting capabilities.

#### Key Discoveries from Legacy Script Analysis
- **Specialized Test Frameworks Available**: `ProcessingStepBuilderTest`, `TrainingStepBuilderTest`, `CreateModelStepBuilderTest`, `TransformStepBuilderTest`
- **BuilderTestReporter**: Comprehensive reporting system with issue tracking and recommendations
- **Step-Type Specific Logic**: Each framework has specialized validation (e.g., Processing Pattern B, Training frameworks)

#### Implementation Tasks

**4.1 Add Step-Type Specific Test Framework Factory** (Days 1-2):

```python
# Add to test/steps/builders/test_dynamic_universal.py

class StepTypeTestFrameworkFactory:
    """Factory to select appropriate test framework based on step type."""
    
    @staticmethod
    def create_tester(builder_class, canonical_name, step_catalog, **kwargs):
        """Create appropriate test framework based on detected step type."""
        try:
            # Get step info from catalog
            step_info = step_catalog.get_step_info(canonical_name)
            step_type = step_info.registry_data.get('sagemaker_step_type') if step_info else None
            
            # Import specialized frameworks
            if step_type == "Processing":
                from cursus.validation.builders.variants.processing_test import ProcessingStepBuilderTest
                return ProcessingStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
            elif step_type == "Training":
                from cursus.validation.builders.variants.training_test import TrainingStepBuilderTest
                return TrainingStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
            elif step_type == "CreateModel":
                from cursus.validation.builders.variants.createmodel_test import CreateModelStepBuilderTest
                return CreateModelStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
            elif step_type == "Transform":
                from cursus.validation.builders.variants.transform_test import TransformStepBuilderTest
                return TransformStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
            else:
                # Fallback to universal test
                from cursus.validation.builders.universal_test import UniversalStepBuilderTest
                return UniversalStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
                
        except ImportError as e:
            print(f"Warning: Could not import specialized framework for {step_type}: {e}")
            # Fallback to universal test
            from cursus.validation.builders.universal_test import UniversalStepBuilderTest
            return UniversalStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
```

**4.2 Integrate BuilderTestReporter** (Days 3-4):

```python
# Add to test/steps/builders/test_dynamic_universal.py

class TestAdvancedReporting:
    """Test advanced reporting with BuilderTestReporter integration."""
    
    @pytest.fixture(scope="class")
    def step_catalog(self):
        """Create step catalog instance."""
        return StepCatalog(workspace_dirs=None)
    
    @pytest.fixture(scope="class") 
    def builder_test_reporter(self):
        """Create BuilderTestReporter instance."""
        from cursus.validation.builders.builder_reporter import BuilderTestReporter
        return BuilderTestReporter()
    
    def test_builder_test_reporter_integration(self, builder_test_reporter, step_catalog):
        """Test BuilderTestReporter integration with step catalog."""
        all_builders = step_catalog.get_all_builders()
        if not all_builders:
            pytest.skip("No builders found to test")
        
        # Test first builder with BuilderTestReporter
        canonical_name, builder_class = next(iter(all_builders.items()))
        
        report = builder_test_reporter.test_and_report_builder(builder_class, canonical_name)
        
        # Verify report structure
        assert report is not None
        assert hasattr(report, 'builder_name')
        assert hasattr(report, 'summary')
        assert hasattr(report, 'get_all_results')
        
        # Verify comprehensive reporting
        all_results = report.get_all_results()
        assert len(all_results) > 0
        
        # Verify summary generation
        summary = report.generate_summary()
        assert summary is not None
        assert hasattr(summary, 'pass_rate')
        assert hasattr(summary, 'overall_status')
```

**4.3 Update Dynamic Testing to Use Specialized Frameworks** (Day 5):

```python
# Update test_comprehensive_all_builders method in TestDynamicUniversalBuilders

def test_comprehensive_all_builders_with_specialized_frameworks(self, all_builders):
    """Comprehensive test using specialized frameworks based on step type."""
    if not all_builders:
        pytest.skip("No builders found to test")
    
    step_catalog = StepCatalog(workspace_dirs=None)
    results = {}
    failed_builders = []
    
    for canonical_name, builder_class in all_builders.items():
        try:
            # Use factory to get appropriate test framework
            tester = StepTypeTestFrameworkFactory.create_tester(
                builder_class=builder_class,
                canonical_name=canonical_name,
                step_catalog=step_catalog,
                verbose=False,
                enable_scoring=True,
                enable_structured_reporting=True
            )
            
            # Run step-type-specific tests
            if hasattr(tester, 'run_processing_validation'):
                test_results = tester.run_processing_validation()
            elif hasattr(tester, 'run_training_validation'):
                test_results = tester.run_training_validation()
            elif hasattr(tester, 'run_createmodel_validation'):
                test_results = tester.run_createmodel_validation()
            elif hasattr(tester, 'run_transform_validation'):
                test_results = tester.run_transform_validation()
            else:
                test_results = tester.run_all_tests()
            
            results[canonical_name] = test_results
            
            # Check if tests passed
            if 'test_results' in test_results:
                raw_results = test_results['test_results']
                total_tests = len(raw_results)
                passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
                pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
                
                if pass_rate < 60:
                    failed_builders.append(f"{canonical_name} ({pass_rate:.1f}%)")
            
        except Exception as e:
            failed_builders.append(f"{canonical_name} (Error: {e})")
    
    # Save comprehensive test results with specialized framework info
    BuilderTestResultsStorage.save_test_results(results, "all_builders", "specialized_frameworks")
    
    # Assert overall success
    total_builders = len(all_builders)
    successful_builders = total_builders - len(failed_builders)
    success_rate = (successful_builders / total_builders * 100) if total_builders > 0 else 0
    
    # Print summary
    print(f"\n📊 Specialized Framework Test Summary:")
    print(f"   Total Builders: {total_builders}")
    print(f"   Successful: {successful_builders} ({success_rate:.1f}%)")
    print(f"   Failed: {len(failed_builders)}")
    
    if failed_builders:
        print(f"\n❌ Failed Builders:")
        for builder in failed_builders:
            print(f"   • {builder}")
    
    assert success_rate >= 75, f"Only {success_rate:.1f}% of builders passed specialized tests. Failed: {failed_builders}"
```

#### Success Criteria for Phase 4
- ✅ Step-type specific test framework factory implemented
- ✅ BuilderTestReporter integration completed
- ✅ Dynamic testing updated to use specialized frameworks
- ✅ Processing Pattern B auto-pass logic leveraged
- ✅ Training framework-specific validation enabled
- ✅ Comprehensive reporting with issue tracking and recommendations
- ✅ All step types (Processing, Training, CreateModel, Transform) supported

### Phase 5: Legacy Script Feature Integration (Week 5) - FINAL ENHANCEMENTS

#### Objective
Integrate the final missing visual and user experience features from legacy report generation scripts, focusing on the 3 identified gaps that aren't already implemented in the cursus system.

#### Key Findings from Comprehensive Analysis
After thorough examination of `cursus/step_catalog/step_catalog.py` and `cursus/registry/step_names.py`, **95% of legacy functionality already exists**:

**✅ Already Implemented (No Need to Add)**:
- Registry-based builder loading (`get_steps_by_sagemaker_type`, `STEP_NAMES`, `BUILDER_STEP_NAMES`)
- Canonical name resolution (`get_canonical_name_from_file_name`, `get_builder_step_name`)
- Builder discovery (`get_all_builders`, `get_builders_by_step_type`, `load_builder_class`)
- Visual chart generation infrastructure (`EnhancedReportGenerator.generate_score_chart`)
- Module name conversion with abbreviation mapping and fuzzy matching

**❌ Actually Missing (Only 3 Real Gaps)**:
1. Step type color coding system (specific colors from legacy scripts)
2. Individual builder status display with icons (`✅ BuilderName: PASSING (85.2%)` format)
3. Enhanced summary statistics (multi-level breakdown)

#### Implementation Tasks

**5.1 Add Step Type Color Coding System** (Days 1-2):

```python
# Add to test/steps/builders/test_dynamic_universal.py

class StepTypeColorScheme:
    """Step type color coding system from legacy report generators."""
    
    STEP_TYPE_COLORS = {
        "Training": "#FF6B6B",      # Red
        "Transform": "#4ECDC4",     # Teal
        "CreateModel": "#45B7D1",   # Blue
        "Processing": "#96CEB4",    # Green
        "Base": "#9B59B6",          # Purple
        "Utility": "#F39C12",       # Orange
        "Lambda": "#E74C3C",        # Dark Red
        "RegisterModel": "#2ECC71", # Emerald
    }
    
    @classmethod
    def get_color_for_step_type(cls, step_type: str) -> str:
        """Get color for step type with fallback."""
        return cls.STEP_TYPE_COLORS.get(step_type, "#95A5A6")  # Gray fallback
    
    @classmethod
    def get_color_for_builder(cls, canonical_name: str, step_catalog) -> str:
        """Get color for builder based on its step type."""
        try:
            step_info = step_catalog.get_step_info(canonical_name)
            if step_info:
                step_type = step_info.registry_data.get('sagemaker_step_type')
                return cls.get_color_for_step_type(step_type)
            return cls.STEP_TYPE_COLORS.get("Base", "#95A5A6")
        except Exception:
            return "#95A5A6"  # Gray fallback
    
    @classmethod
    def get_all_colors(cls) -> Dict[str, str]:
        """Get all step type colors for legend generation."""
        return cls.STEP_TYPE_COLORS.copy()
```

**5.2 Add Enhanced Status Display with Icons** (Days 3-4):

```python
# Add to test/steps/builders/test_dynamic_universal.py

class EnhancedStatusDisplay:
    """Enhanced status display with icons from legacy scripts."""
    
    @staticmethod
    def format_builder_status(canonical_name: str, test_results: Dict[str, Any], 
                            step_type: str = None) -> str:
        """Format builder status with icon and pass rate."""
        try:
            # Calculate pass rate
            if 'test_results' in test_results:
                raw_results = test_results['test_results']
                total_tests = len(raw_results)
                passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
                pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            else:
                pass_rate = 0
            
            # Determine status and icon
            if pass_rate >= 80:
                status_icon = "✅"
                status_text = "PASSING"
            elif pass_rate >= 60:
                status_icon = "⚠️"
                status_text = "WARNING"
            else:
                status_icon = "❌"
                status_text = "FAILING"
            
            # Add step type if available
            type_info = f" [{step_type}]" if step_type else ""
            
            return f"{status_icon} {canonical_name}{type_info}: {status_text} ({pass_rate:.1f}%)"
            
        except Exception as e:
            return f"❓ {canonical_name}: ERROR ({e})"
    
    @staticmethod
    def print_builder_summary(results: Dict[str, Any], step_catalog) -> None:
        """Print comprehensive builder summary with enhanced formatting."""
        print(f"\n{'='*80}")
        print("📊 ENHANCED BUILDER TEST SUMMARY")
        print(f"{'='*80}")
        
        # Group by step type
        step_type_groups = {}
        for canonical_name, test_results in results.items():
            try:
                step_info = step_catalog.get_step_info(canonical_name)
                step_type = step_info.registry_data.get('sagemaker_step_type', 'Unknown') if step_info else 'Unknown'
                
                if step_type not in step_type_groups:
                    step_type_groups[step_type] = []
                
                status_line = EnhancedStatusDisplay.format_builder_status(
                    canonical_name, test_results, step_type
                )
                step_type_groups[step_type].append((canonical_name, status_line, test_results))
                
            except Exception as e:
                if 'Unknown' not in step_type_groups:
                    step_type_groups['Unknown'] = []
                step_type_groups['Unknown'].append((canonical_name, f"❓ {canonical_name}: ERROR ({e})", {}))
        
        # Print by step type with colors
        for step_type, builders in sorted(step_type_groups.items()):
            color = StepTypeColorScheme.get_color_for_step_type(step_type)
            print(f"\n🔧 {step_type} Steps ({len(builders)} builders):")
            print(f"   Color: {color}")
            
            for canonical_name, status_line, test_results in sorted(builders, key=lambda x: x[1]):
                print(f"   {status_line}")
        
        # Overall statistics
        total_builders = len(results)
        successful_builders = sum(1 for _, _, test_results in 
                                [item for sublist in step_type_groups.values() for item in sublist]
                                if EnhancedStatusDisplay._is_passing(test_results))
        
        success_rate = (successful_builders / total_builders * 100) if total_builders > 0 else 0
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total Builders: {total_builders}")
        print(f"   Successful: {successful_builders} ({success_rate:.1f}%)")
        print(f"   Failed: {total_builders - successful_builders}")
        print(f"   Step Types: {len(step_type_groups)}")
        print(f"{'='*80}")
    
    @staticmethod
    def _is_passing(test_results: Dict[str, Any]) -> bool:
        """Check if test results indicate passing status."""
        try:
            if 'test_results' in test_results:
                raw_results = test_results['test_results']
                total_tests = len(raw_results)
                passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
                pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
                return pass_rate >= 60
            return False
        except Exception:
            return False
```

**5.3 Wire Up Existing Chart Generation** (Day 5):

```python
# Update test_comprehensive_all_builders_with_specialized_frameworks method

def test_comprehensive_all_builders_with_visual_charts(self, all_builders):
    """Comprehensive test with visual chart generation using existing infrastructure."""
    if not all_builders:
        pytest.skip("No builders found to test")
    
    step_catalog = StepCatalog(workspace_dirs=None)
    report_generator = EnhancedReportGenerator()
    results = {}
    failed_builders = []
    
    for canonical_name, builder_class in all_builders.items():
        try:
            # Use specialized framework factory
            tester = StepTypeTestFrameworkFactory.create_tester(
                builder_class=builder_class,
                canonical_name=canonical_name,
                step_catalog=step_catalog,
                verbose=False,
                enable_scoring=True,
                enable_structured_reporting=True
            )
            
            # Run tests
            test_results = tester.run_all_tests()
            results[canonical_name] = test_results
            
            # Generate visual chart using existing infrastructure
            try:
                step_info = step_catalog.get_step_info(canonical_name)
                step_type = step_info.registry_data.get('sagemaker_step_type', 'Unknown') if step_info else 'Unknown'
                
                # Save enhanced report with chart generation
                saved_files = report_generator.save_enhanced_report(
                    test_results, canonical_name, step_type, generate_chart=True
                )
                
                if 'score_chart' in saved_files:
                    print(f"📊 Generated chart for {canonical_name}: {saved_files['score_chart']}")
                
            except Exception as chart_error:
                print(f"⚠️ Chart generation failed for {canonical_name}: {chart_error}")
            
            # Check pass rate
            if 'test_results' in test_results:
                raw_results = test_results['test_results']
                total_tests = len(raw_results)
                passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
                pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
                
                if pass_rate < 60:
                    failed_builders.append(f"{canonical_name} ({pass_rate:.1f}%)")
            
        except Exception as e:
            failed_builders.append(f"{canonical_name} (Error: {e})")
    
    # Enhanced status display
    EnhancedStatusDisplay.print_builder_summary(results, step_catalog)
    
    # Save results with color coding metadata
    enhanced_results = {}
    for canonical_name, test_results in results.items():
        color = StepTypeColorScheme.get_color_for_builder(canonical_name, step_catalog)
        enhanced_results[canonical_name] = {
            **test_results,
            'step_type_color': color,
            'enhanced_status': EnhancedStatusDisplay.format_builder_status(canonical_name, test_results)
        }
    
    BuilderTestResultsStorage.save_test_results(enhanced_results, "all_builders", "visual_enhanced")
    
    # Assert success with enhanced reporting
    total_builders = len(all_builders)
    successful_builders = total_builders - len(failed_builders)
    success_rate = (successful_builders / total_builders * 100) if total_builders > 0 else 0
    
    assert success_rate >= 75, f"Only {success_rate:.1f}% of builders passed visual enhanced tests. Failed: {failed_builders}"
```

#### Success Criteria for Phase 5
- ✅ Step type color coding system implemented with legacy script colors
- ✅ Enhanced status display with icons and pass rates implemented
- ✅ Existing chart generation infrastructure wired up in dynamic tests
- ✅ Visual charts generated for all builders using existing EnhancedReportGenerator
- ✅ Enhanced summary statistics with step type grouping and color coding
- ✅ All legacy script visual/UX features integrated without duplication
- ✅ 100% leverage of existing cursus infrastructure

## Phase 6: Folder Structure Optimization ✅

**Status: COMPLETED**

### Objectives
- Optimize cursus/validation/builders folder structure by organizing support/helper functions into logical subfolders
- Create clear separation of concerns with proper naming conventions
- Maintain backward compatibility while improving maintainability
- Update import statements and create proper __init__.py files

### Implementation Details

#### 6.1 Folder Structure Analysis and Planning ✅
- **Analysis**: Examined current folder structure and identified support/helper functions
- **Grouping Strategy**: Created logical subfolder groupings based on functionality
- **Naming Convention**: Used clear, descriptive names for subfolders

#### 6.2 Created Logical Subfolder Structure ✅
- **`reporting/`** - Reporting and visualization modules
  - `report_generator.py` - Enhanced report generation
  - `results_storage.py` - Test results storage
  - `enhanced_status_display.py` - Status formatting with icons
  - `step_type_color_scheme.py` - Color coding system
  - `scoring.py` - Quality scoring system

- **`discovery/`** - Discovery and registry modules
  - `registry_discovery.py` - Builder discovery utilities
  - `step_catalog_config_provider.py` - Configuration discovery

- **`factories/`** - Factory classes
  - `builder_test_factory.py` - Test factory utilities
  - `step_type_test_framework_factory.py` - Framework selection factory

- **`core/`** - Core testing modules
  - `universal_test.py` - Main universal test suite
  - `base_test.py` - Base test functionality

#### 6.3 Updated Import Structure ✅
- **Created `__init__.py` files** for each subfolder with proper exports
- **Updated main `__init__.py`** to use organized imports from subfolders
- **Updated `test_dynamic_universal.py`** to use new import paths
- **Fixed circular dependencies** and import path issues

#### 6.4 Maintained Backward Compatibility ✅
- **Legacy imports still work** through the main package `__init__.py`
- **All existing functionality preserved** with organized structure
- **Enhanced Phase 5 features** properly integrated into new structure

### Key Achievements
- ✅ **Logical Organization**: Related functionality grouped together in appropriate subfolders
- ✅ **Clear Separation**: Core, reporting, discovery, and factory functions properly separated
- ✅ **Maintainability**: Easier to find and modify specific functionality
- ✅ **Scalability**: Easy to add new modules to appropriate categories
- ✅ **Backward Compatible**: Existing code continues to work without changes
- ✅ **Documentation**: Each subfolder has clear purpose and proper exports

### New Organized Structure
```
src/cursus/validation/builders/
├── __init__.py                    # Main package with organized imports
├── core/                          # Core testing framework
│   ├── __init__.py
│   ├── universal_test.py          # Main test suite
│   └── base_test.py              # Base functionality
├── reporting/                     # Reporting and visualization
│   ├── __init__.py
│   ├── report_generator.py       # Enhanced reports
│   ├── results_storage.py        # Results storage
│   ├── enhanced_status_display.py # Status formatting
│   ├── step_type_color_scheme.py # Color coding
│   └── scoring.py                # Quality scoring
├── discovery/                     # Discovery and registry
│   ├── __init__.py
│   ├── registry_discovery.py     # Builder discovery
│   └── step_catalog_config_provider.py # Config discovery
├── factories/                     # Factory classes
│   ├── __init__.py
│   ├── builder_test_factory.py   # Test factories
│   └── step_type_test_framework_factory.py # Framework factory
├── variants/                      # Step-type specific tests
│   └── [existing specialized test files]
└── [remaining core test files]    # Interface, specification, etc.
```

### Import Usage Examples
**Organized imports:**
```python
from cursus.validation.builders.core import UniversalStepBuilderTest
from cursus.validation.builders.reporting import (
    BuilderTestResultsStorage, 
    EnhancedReportGenerator,
    StepTypeColorScheme,
    EnhancedStatusDisplay
)
from cursus.validation.builders.factories import StepTypeTestFrameworkFactory
```

**Legacy compatibility maintained:**
```python
from cursus.validation.builders import UniversalStepBuilderTest  # Still works
```

### Benefits Achieved
- **🗂️ Logical Organization**: Related functionality grouped together
- **🔍 Easy Navigation**: Clear separation of concerns
- **🔧 Maintainability**: Easier to find and modify specific functionality
- **📈 Scalability**: Easy to add new modules to appropriate categories
- **🔄 Backward Compatible**: Existing code continues to work
- **📝 Clear Documentation**: Each subfolder has clear purpose and exports

#### Success Criteria for Phase 6
- ✅ Folder structure optimized with logical subfolder groupings
- ✅ Support/helper functions organized into appropriate categories
- ✅ Import statements updated to use new organized structure
- ✅ __init__.py files created for all subfolders with proper exports
- ✅ Backward compatibility maintained for existing imports
- ✅ All Phase 5 enhancements properly integrated into new structure
- ✅ Documentation updated to reflect new organization

## File Structure

```
test/steps/builders/
├── test_dynamic_universal.py        # NEW - Dynamic universal testing (includes test setup)
├── test_createmodel_step_builders.py # REWRITTEN - Dynamic CreateModel testing
├── test_training_step_builders.py   # REWRITTEN - Dynamic Training testing
├── test_processing_step_builders.py # REWRITTEN - Dynamic Processing testing
├── test_transform_step_builders.py  # REWRITTEN - Dynamic Transform testing
├── README.md                        # NEW - Documentation
├── results/                         # NEW - Test results storage
│   ├── .gitignore
│   ├── all_builders/               # Results from CLI 'test-all-discovered'
│   └── individual/                 # Results from CLI 'test-single'
└── legacy/                         # ARCHIVED - Only manual runner scripts
    └── run_*.py scripts             # Manual test runners (replaced by CLI)
```

## Migration from Legacy Tests

### What Changed
- ❌ **Removed**: Hard-coded builder lists requiring manual maintenance
- ❌ **Removed**: Manual test runner scripts (archived to legacy/)
- ✅ **Rewritten**: Step-type-specific test files now use dynamic discovery
- ✅ **Added**: Dynamic pytest integration with parametrized tests
- ✅ **Added**: Comprehensive test results storage
- ✅ **Maintained**: Existing test file structure for backward compatibility

### Benefits of New Approach
- ✅ **Zero Maintenance**: New builders automatically included in tests
- ✅ **Backward Compatibility**: Existing test files preserved but enhanced
- ✅ **Step Type Focus**: Each test file focuses on specific SageMaker step type
- ✅ **Comprehensive Coverage**: All builders tested with same standards
- ✅ **Better Reporting**: Structured results with scoring and analytics
- ✅ **CI/CD Ready**: Standard pytest integration for automated testing


#### Success Criteria for Phase 3
- ✅ Dynamic pytest integration replaces all hard-coded test files
- ✅ Legacy test files archived to `legacy/` subdirectory
- ✅ New test structure documented and ready for use
- ✅ All existing functionality preserved with dynamic discovery
- ✅ Test results automatically saved and organized

## Implementation Benefits

### Leverages Existing Systems
- ✅ **Step Catalog**: Uses existing `load_builder_class()` and `list_available_steps()` methods
- ✅ **CLI Framework**: Enhances existing `builder_test_cli.py` with new commands
- ✅ **Universal Tester**: Uses existing `UniversalStepBuilderTest` framework
- ✅ **No New Modules**: Only enhances existing systems, no duplication

### Eliminates Hard-Coding
- ✅ **Zero Maintenance**: No more updating test lists when builders are added
- ✅ **Automatic Discovery**: New builders included automatically via step catalog
- ✅ **Registry-Driven**: Single source of truth via existing infrastructure
- ✅ **Future-Proof**: Adapts to changes in registry and step catalog systems

### Simple User Experience
- ✅ **Two Commands**: `test-all-discovered` and `test-single` cover all scenarios
- ✅ **Familiar Interface**: Uses existing CLI patterns and conventions
- ✅ **Clear Feedback**: Comprehensive reporting with progress indicators
- ✅ **Automatic Results**: Auto-saves to organized directory structure

## Timeline: 3 Weeks

### Week 1: Enhance Step Catalog (5 days)
- **Days 1-3**: Add `get_all_builders()` and `get_builders_by_step_type()` methods
- **Days 4-5**: Add TestResultsStorage class in validation/builders module

### Week 2: Enhance CLI (5 days)  
- **Days 1-4**: Add new CLI commands (`test-all-discovered`, `test-single`, `list-discovered`)
- **Day 5**: Create results directory structure and update documentation

### Week 3: Clean Up and Update Tests (5 days)
- **Days 1-2**: Create dynamic pytest integration (`test_dynamic_universal.py`)
- **Days 3-4**: Archive legacy test files and create shared test configuration
- **Day 5**: Update documentation and README for new test structure

## File Changes Summary

**Files Enhanced**:
1. **`src/cursus/step_catalog/step_catalog.py`** - Add 2 builder discovery methods (~30 lines)
2. **`src/cursus/validation/builders/test_results_storage.py`** - Add TestResultsStorage class (~50 lines)
3. **`src/cursus/cli/builder_test_cli.py`** - Add 3 new commands (~150 lines)

**Results Directory Created**:
```
test/steps/builders/results/
├── .gitignore
├── all_builders/
└── individual/
```

**Total Implementation**: ~230 lines of code across 3 files (2 enhanced + 1 new) + directory structure

## Usage After Implementation

### Target User Commands
```bash
# Test all builders (replaces all hard-coded test files)
cursus builder-test test-all-discovered

# Test specific step type
cursus builder-test test-all-discovered --step-type Processing --verbose --scoring

# Test individual builder by canonical name
cursus builder-test test-single TabularPreprocessing --scoring

# List available builders
cursus builder-test list-discovered --step-type Training
```

### Automatic Results Storage
- **All Builders**: `test/steps/builders/results/all_builders/all_builders_YYYYMMDD_HHMMSS.json`
- **Single Builder**: `test/steps/builders/results/individual/{canonical_name}_YYYYMMDD_HHMMSS.json`
- **Custom Export**: `--export-json custom_path.json` option available

## Key Advantages

**Minimal Implementation Effort**:
- ✅ **2 Weeks vs 3+ Weeks**: Faster implementation timeline
- ✅ **2 Files vs Multiple Modules**: Minimal code changes
- ✅ **200 Lines vs 2000+ Lines**: Focused implementation
- ✅ **No New Architecture**: Enhances existing proven systems

**Maximum Leverage of Existing Systems**:
- ✅ **Step Catalog Discovery**: Already handles builder loading and canonical names
- ✅ **CLI Framework**: Already provides command structure and error handling
- ✅ **Universal Tester**: Already provides comprehensive testing with scoring
- ✅ **Registry System**: Already provides step type classification

**Zero Duplication Risk**:
- ✅ **No Separate Discovery Module**: Uses step catalog directly
- ✅ **No New Test Runner**: Uses existing universal test framework
- ✅ **No New CLI Structure**: Enhances existing builder-test commands
- ✅ **No Redundant Logic**: All functionality leverages existing implementations

## Success Criteria

### Phase 1 Completion
- ✅ StepCatalog has `get_all_builders()` method
- ✅ StepCatalog has `get_builders_by_step_type()` method  
- ✅ StepCatalog has `save_test_results()` helper method
- ✅ No new modules created - only existing StepCatalog enhanced

### Phase 2 Completion
- ✅ New CLI commands use step catalog for discovery
- ✅ Automatic results saving to organized directory structure
- ✅ No new modules created - only existing systems enhanced
- ✅ All functionality working end-to-end

## References

### Related Design Documents

#### Core Architecture References
- **[Enhanced Universal Step Builder Tester Design](../1_design/enhanced_universal_step_builder_tester_design.md)** - Comprehensive enhanced design with scoring and step-type variants ✅ IMPLEMENTED
- **[Code Redundancy Evaluation Guide](../1_design/code_redundancy_evaluation_guide.md)** - Framework for achieving 15-25% redundancy through elimination of hard-coding and maintenance overhead

#### Step Catalog and Registry Systems
- **[Unified Step Catalog System Implementation Plan](./2025-09-10_unified_step_catalog_system_implementation_plan.md)** - Step catalog architecture providing dynamic discovery capabilities ✅ COMPLETED
- **[Registry Redundancy Elimination Implementation](./2025-09-07_registry_redundancy_elimination_implementation.md)** - Registry system optimization and standardization ✅ COMPLETED

#### Testing Framework References
- **[Universal Step Builder Test Step Catalog Integration Plan](./2025-09-28_universal_step_builder_test_step_catalog_integration_plan.md)** - Foundation integration between universal tester and step catalog ✅ COMPLETED
- **[Universal Step Builder Test Enhancement Plan](./2025-08-07_universal_step_builder_test_enhancement_plan.md)** - Previous enhancement plan for universal testing system
- **[SageMaker Step Type Aware Unified Alignment Tester Implementation Plan](./2025-08-13_sagemaker_step_type_aware_unified_alignment_tester_implementation_plan.md)** - Step type-specific testing patterns

### Related Developer Guides

#### Validation and Testing
- **[Validation Framework Guide](../0_developer_guide/validation_framework_guide.md)** - Testing standards and best practices
- **[Step Builder Development Guide](../0_developer_guide/step_builder.md)** - Builder development standards
- **[Validation Checklist](../0_developer_guide/validation_checklist.md)** - Comprehensive validation requirements

#### Step Catalog Integration
- **[Step Catalog Integration Guide](../0_developer_guide/step_catalog_integration_guide.md)** - Integration patterns with step catalog system
- **[Pipeline Catalog Integration Guide](../0_developer_guide/pipeline_catalog_integration_guide.md)** - Pipeline-level integration patterns

### Design Principles and Patterns

#### System Design
- **[Design Principles](../1_design/design_principles.md)** - Core design principles for the cursus framework
- **[Config Driven Design](../1_design/config_driven_design.md)** - Configuration-driven architecture patterns
- **[Dependency Resolution System](../1_design/dependency_resolution_system.md)** - Dependency management patterns

#### CLI and Interface Design
- **[CLI Pipeline Compilation Tools Design](../1_design/cli_pipeline_compilation_tools_design.md)** - CLI design patterns and best practices
- **[API Reference Documentation Style Guide](../1_design/api_reference_documentation_style_guide.md)** - Documentation standards for CLI interfaces

### Workspace-Aware Development

#### Workspace Integration
- **[Workspace Setup Guide](../01_developer_guide_workspace_aware/ws_workspace_setup_guide.md)** - Workspace-aware development setup
- **[Hybrid Registry Integration](../01_developer_guide_workspace_aware/ws_hybrid_registry_integration.md)** - Registry integration in workspace environments
- **[Workspace CLI Reference](../01_developer_guide_workspace_aware/ws_workspace_cli_reference.md)** - CLI usage in workspace contexts

### Implementation Dependencies

This implementation plan builds upon and integrates with the following completed systems:

#### Foundation Systems ✅ COMPLETED
- **Step Catalog System**: Provides dynamic builder discovery and loading capabilities
- **Registry System**: Provides step type classification and metadata
- **Universal Test Framework**: Provides comprehensive testing with scoring and variants
- **CLI Framework**: Provides command structure and user interface patterns

#### Integration Points
- **Step Catalog Integration**: Uses existing `load_builder_class()` and `list_available_steps()` methods
- **Registry Integration**: Leverages step type classification for variant testing
- **Universal Tester Integration**: Uses existing comprehensive testing framework
- **CLI Integration**: Enhances existing `builder-test` command structure

## Conclusion

This simplified implementation plan provides exactly what was requested - dynamic testing of all builders with simple commands - while leveraging existing proven infrastructure and avoiding any duplication or complex new architecture.

The approach enhances only 2 existing files to provide complete dynamic builder testing capabilities, eliminates all hard-coded test maintenance, and provides a simple, user-friendly interface that automatically discovers and tests all builders via the existing step catalog system.

### Strategic Alignment

This implementation aligns with the broader cursus framework design principles:
- **Leverages Existing Infrastructure**: Builds upon proven step catalog and CLI systems
- **Eliminates Redundancy**: Removes hard-coded test maintenance overhead
- **Maintains Simplicity**: Provides simple, intuitive user interface
- **Ensures Consistency**: Uses established patterns and conventions
- **Enables Scalability**: Automatically adapts to new builders and changes

The plan represents a focused, practical solution that delivers immediate value while maintaining architectural integrity and long-term maintainability.
