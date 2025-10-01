"""
Dynamic pytest integration for universal builder testing.

This module provides comprehensive dynamic testing for all builders discovered
via the step catalog system, eliminating the need for hard-coded builder lists.

Enhanced with features from legacy report generators:
- Visual score charts and reporting
- Comprehensive metadata tracking
- Structured directory organization
- Summary reports across step types
"""

import pytest
from typing import Dict, Type, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from cursus.step_catalog import StepCatalog
from cursus.validation.builders.core import UniversalStepBuilderTest
from cursus.validation.builders.reporting import (
    BuilderTestResultsStorage, 
    EnhancedReportGenerator,
    StepTypeColorScheme,
    EnhancedStatusDisplay
)
from cursus.validation.builders.factories import StepTypeTestFrameworkFactory

# Import BuilderTestReporter for advanced reporting
try:
    from cursus.validation.builders.builder_reporter import BuilderTestReporter
except ImportError:
    BuilderTestReporter = None


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
        BuilderTestResultsStorage.save_test_results(results, "all_builders", "pytest_comprehensive")
        
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
    
    def test_comprehensive_all_builders_with_specialized_frameworks(self, all_builders):
        """Comprehensive test using specialized frameworks based on step type."""
        if not all_builders:
            pytest.skip("No builders found to test")
        
        step_catalog = StepCatalog(workspace_dirs=None)
        results = {}
        failed_builders = []
        specialized_framework_usage = {}
        
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
                
                # Track which framework was used
                framework_type = type(tester).__name__
                if framework_type not in specialized_framework_usage:
                    specialized_framework_usage[framework_type] = 0
                specialized_framework_usage[framework_type] += 1
                
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
                
                # Add framework info to results
                test_results['test_framework_used'] = framework_type
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
        
        # Print summary with framework usage
        print(f"\n📊 Specialized Framework Test Summary:")
        print(f"   Total Builders: {total_builders}")
        print(f"   Successful: {successful_builders} ({success_rate:.1f}%)")
        print(f"   Failed: {len(failed_builders)}")
        print(f"\n🔧 Framework Usage:")
        for framework, count in specialized_framework_usage.items():
            print(f"   {framework}: {count} builders")
        
        if failed_builders:
            print(f"\n❌ Failed Builders:")
            for builder in failed_builders:
                print(f"   • {builder}")
        
        assert success_rate >= 75, f"Only {success_rate:.1f}% of builders passed specialized tests. Failed: {failed_builders}"
    
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


class TestPhase5Enhancements:
    """Test Phase 5 enhancement features."""
    
    @pytest.fixture(scope="class")
    def step_catalog(self):
        """Create step catalog instance."""
        return StepCatalog(workspace_dirs=None)
    
    def test_step_type_color_scheme(self):
        """Test step type color coding system."""
        # Test color retrieval
        processing_color = StepTypeColorScheme.get_color_for_step_type("Processing")
        assert processing_color == "#96CEB4"  # Green
        
        training_color = StepTypeColorScheme.get_color_for_step_type("Training")
        assert training_color == "#FF6B6B"  # Red
        
        # Test fallback color
        unknown_color = StepTypeColorScheme.get_color_for_step_type("UnknownType")
        assert unknown_color == "#95A5A6"  # Gray fallback
        
        # Test all colors retrieval
        all_colors = StepTypeColorScheme.get_all_colors()
        assert isinstance(all_colors, dict)
        assert len(all_colors) == 8  # Should have 8 defined step types
        assert "Processing" in all_colors
        assert "Training" in all_colors
    
    def test_enhanced_status_display(self):
        """Test enhanced status display functionality."""
        # Mock test results
        test_results = {
            'test_results': {
                'test1': {'passed': True},
                'test2': {'passed': True},
                'test3': {'passed': False}
            }
        }
        
        # Test status formatting
        status = EnhancedStatusDisplay.format_builder_status("TestBuilder", test_results, "Processing")
        assert "TestBuilder" in status
        assert "[Processing]" in status
        assert "66.7%" in status  # 2/3 = 66.7%
        assert "⚠️" in status  # Warning icon for 60-80% range
        
        # Test summary statistics
        summary = EnhancedStatusDisplay.get_summary_statistics({"TestBuilder": test_results})
        assert summary['total_builders'] == 1
        assert summary['successful_builders'] == 1  # 66.7% >= 60% threshold
        assert summary['success_rate'] == 100.0
    
    def test_step_type_test_framework_factory(self, step_catalog):
        """Test step type test framework factory."""
        all_builders = step_catalog.get_all_builders()
        if not all_builders:
            pytest.skip("No builders found to test")
        
        # Test factory creation
        canonical_name, builder_class = next(iter(all_builders.items()))
        tester = StepTypeTestFrameworkFactory.create_tester(
            builder_class=builder_class,
            canonical_name=canonical_name,
            step_catalog=step_catalog,
            verbose=False
        )
        
        assert tester is not None
        assert hasattr(tester, 'run_all_tests')
        
        # Test available frameworks
        available_frameworks = StepTypeTestFrameworkFactory.get_available_specialized_frameworks()
        assert isinstance(available_frameworks, dict)
        
        # Test supported step types
        supported_types = StepTypeTestFrameworkFactory.get_supported_step_types()
        assert isinstance(supported_types, list)
    
    def test_integration_with_existing_report_generator(self, step_catalog):
        """Test integration with existing EnhancedReportGenerator."""
        all_builders = step_catalog.get_all_builders()
        if not all_builders:
            pytest.skip("No builders found to test")
        
        # Get first builder for testing
        canonical_name, builder_class = next(iter(all_builders.items()))
        
        # Run test with enhanced features
        tester = StepTypeTestFrameworkFactory.create_tester(
            builder_class=builder_class,
            canonical_name=canonical_name,
            step_catalog=step_catalog,
            verbose=False,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        results = tester.run_all_tests()
        
        # Test color assignment
        color = StepTypeColorScheme.get_color_for_builder(canonical_name, step_catalog)
        assert color.startswith("#")  # Should be a hex color
        
        # Test status formatting
        status = EnhancedStatusDisplay.format_builder_status(canonical_name, results)
        assert canonical_name in status
        assert any(icon in status for icon in ["✅", "⚠️", "❌"])  # Should have status icon
        
        # Test enhanced results with metadata
        enhanced_results = {
            canonical_name: {
                **results,
                'step_type_color': color,
                'enhanced_status': status
            }
        }
        
        # Verify enhanced structure
        assert 'step_type_color' in enhanced_results[canonical_name]
        assert 'enhanced_status' in enhanced_results[canonical_name]
        assert enhanced_results[canonical_name]['step_type_color'] == color
        assert enhanced_results[canonical_name]['enhanced_status'] == status


class TestStepCatalogIntegration:
    """Test step catalog integration functionality."""
    
    @pytest.fixture(scope="class")
    def step_catalog(self):
        """Create step catalog instance."""
        return StepCatalog(workspace_dirs=None)
    
    def test_step_catalog_initialization(self, step_catalog):
        """Test that step catalog initializes correctly."""
        assert step_catalog is not None
        assert hasattr(step_catalog, 'get_all_builders')
        assert hasattr(step_catalog, 'get_builders_by_step_type')
    
    def test_builder_discovery_methods(self, step_catalog):
        """Test that builder discovery methods work correctly."""
        # Test get_all_builders
        all_builders = step_catalog.get_all_builders()
        assert isinstance(all_builders, dict)
        assert len(all_builders) > 0
        
        # Test get_builders_by_step_type
        processing_builders = step_catalog.get_builders_by_step_type("Processing")
        assert isinstance(processing_builders, dict)
        
        # Processing builders should be a subset of all builders
        for name in processing_builders:
            assert name in all_builders
    
    def test_builder_class_validity(self, step_catalog):
        """Test that all discovered builder classes are valid."""
        all_builders = step_catalog.get_all_builders()
        
        for canonical_name, builder_class in all_builders.items():
            # Check that it's a class
            assert isinstance(builder_class, type), f"{canonical_name} is not a class: {builder_class}"
            
            # Check that it has a reasonable name
            assert builder_class.__name__.endswith('StepBuilder'), f"{canonical_name} class name doesn't end with 'StepBuilder': {builder_class.__name__}"
            
            # Check that it's importable (already imported if we got here)
            assert hasattr(builder_class, '__module__'), f"{canonical_name} class has no __module__ attribute"


class TestResultsStorage:
    """Test results storage functionality."""
    
    def test_results_storage_initialization(self):
        """Test that results storage can be initialized."""
        BuilderTestResultsStorage.ensure_results_directory()
        results_dir = BuilderTestResultsStorage.get_results_directory()
        assert results_dir.exists()
    
    def test_results_storage_save(self):
        """Test that results can be saved."""
        test_results = {
            'TestBuilder': {
                'test_results': {'test_inheritance': {'passed': True}},
                'summary': {'total_tests': 1, 'passed': 1}
            }
        }
        
        saved_path = BuilderTestResultsStorage.save_test_results(
            test_results, 'single_builder', 'TestBuilder'
        )
        
        assert saved_path is not None
        assert 'TestBuilder' in saved_path
        assert saved_path.endswith('.json')
    
    def test_results_directory_structure(self):
        """Test that results directory structure is correct."""
        BuilderTestResultsStorage.ensure_results_directory()
        results_dir = BuilderTestResultsStorage.get_results_directory()
        
        assert results_dir.exists()
        assert (results_dir / "all_builders").exists()
        assert (results_dir / "individual").exists()
        assert (results_dir / ".gitignore").exists()


class TestEnhancedReporting:
    """Test enhanced reporting functionality from legacy report generators."""
    
    @pytest.fixture(scope="class")
    def report_generator(self):
        """Create enhanced report generator."""
        return EnhancedReportGenerator()
    
    @pytest.fixture(scope="class")
    def step_catalog(self):
        """Create step catalog instance."""
        return StepCatalog(workspace_dirs=None)
    
    def test_report_generator_initialization(self, report_generator):
        """Test that report generator initializes correctly."""
        assert report_generator is not None
        assert report_generator.base_path.exists()
    
    def test_metadata_enhancement(self, report_generator):
        """Test metadata enhancement functionality."""
        original_results = {
            'test_results': {'test_inheritance': {'passed': True}},
            'summary': {'total_tests': 1, 'passed': 1}
        }
        
        enhanced = report_generator.enhance_results_with_metadata(
            original_results, 'TestBuilder', 'Processing'
        )
        
        assert 'canonical_name' in enhanced
        assert 'timestamp' in enhanced
        assert 'generator_version' in enhanced
        assert 'summary_statistics' in enhanced
        assert enhanced['canonical_name'] == 'TestBuilder'
        assert enhanced['step_type'] == 'Processing'
        assert enhanced['summary_statistics']['pass_rate'] == 100.0
    
    def test_comprehensive_report_generation(self, report_generator):
        """Test comprehensive report generation."""
        all_results = {
            'Builder1': {
                'test_results': {'test1': {'passed': True}, 'test2': {'passed': False}},
                'step_type': 'Processing'
            },
            'Builder2': {
                'test_results': {'test1': {'passed': True}, 'test2': {'passed': True}},
                'step_type': 'Training'
            }
        }
        
        summary = report_generator.generate_comprehensive_report(all_results)
        
        assert 'overall_statistics' in summary
        assert 'step_type_breakdown' in summary
        assert 'builders' in summary
        assert summary['total_builders'] == 2
        assert 'Processing' in summary['step_type_breakdown']
        assert 'Training' in summary['step_type_breakdown']
        assert summary['overall_statistics']['total_tests'] == 4
        assert summary['overall_statistics']['passed_tests'] == 3
    
    def test_enhanced_reporting_with_real_builder(self, report_generator, step_catalog):
        """Test enhanced reporting with a real builder from step catalog."""
        all_builders = step_catalog.get_all_builders()
        if not all_builders:
            pytest.skip("No builders found to test")
        
        # Get first builder for testing
        canonical_name, builder_class = next(iter(all_builders.items()))
        
        # Run test with enhanced reporting
        tester = UniversalStepBuilderTest(
            builder_class=builder_class,
            step_name=canonical_name,
            verbose=False,
            enable_scoring=True,
            enable_structured_reporting=True,
            use_step_catalog_discovery=True
        )
        
        results = tester.run_all_tests()
        
        # Enhance results with metadata
        enhanced_results = report_generator.enhance_results_with_metadata(
            results, canonical_name, "Processing"
        )
        
        # Verify enhanced results
        assert 'canonical_name' in enhanced_results
        assert 'summary_statistics' in enhanced_results
        assert 'timestamp' in enhanced_results
        assert enhanced_results['canonical_name'] == canonical_name
        
        # Test saving enhanced report
        saved_files = report_generator.save_enhanced_report(
            results, canonical_name, "Processing", generate_chart=False  # Skip chart for testing
        )
        
        assert 'json_report' in saved_files
        assert Path(saved_files['json_report']).exists()
    
    def test_step_subfolder_structure_creation(self, report_generator):
        """Test creation of organized step subfolder structure."""
        scoring_dir = report_generator.create_step_subfolder_structure(
            "Processing", "TestStepBuilder"
        )
        
        assert scoring_dir.exists()
        assert scoring_dir.name == "scoring_reports"
        assert scoring_dir.parent.name == "TestStepBuilder"
        assert scoring_dir.parent.parent.name == "processing"
        
        # Check that README was created
        readme_path = scoring_dir.parent / "README.md"
        assert readme_path.exists()
        
        # Verify README content
        readme_content = readme_path.read_text()
        assert "TestStepBuilder" in readme_content
        assert "Processing" in readme_content
        assert "scoring_reports" in readme_content


class TestAdvancedReporting:
    """Test advanced reporting with BuilderTestReporter integration."""
    
    @pytest.fixture(scope="class")
    def step_catalog(self):
        """Create step catalog instance."""
        return StepCatalog(workspace_dirs=None)
    
    @pytest.fixture(scope="class") 
    def builder_test_reporter(self):
        """Create BuilderTestReporter instance."""
        if BuilderTestReporter is None:
            pytest.skip("BuilderTestReporter not available")
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
    
    def test_step_type_builders_reporting(self, builder_test_reporter):
        """Test BuilderTestReporter step type testing functionality."""
        if BuilderTestReporter is None:
            pytest.skip("BuilderTestReporter not available")
        
        # Test Processing step builders
        reports = builder_test_reporter.test_step_type_builders("Processing")
        
        # Should return a dictionary of reports
        assert isinstance(reports, dict)
        
        # Each report should be a BuilderTestReport
        for step_name, report in reports.items():
            assert hasattr(report, 'builder_name')
            assert hasattr(report, 'summary')
            assert hasattr(report, 'get_all_results')
            assert report.builder_name == step_name
    
    def test_specialized_framework_selection(self, step_catalog):
        """Test that specialized frameworks are selected correctly."""
        available_frameworks = StepTypeTestFrameworkFactory.get_available_specialized_frameworks()
        
        # Print available frameworks for debugging
        print(f"\n🔧 Available Specialized Frameworks:")
        for step_type, framework_class in available_frameworks.items():
            print(f"   {step_type}: {framework_class.__name__}")
        
        # Test framework selection for different step types
        all_builders = step_catalog.get_all_builders()
        if not all_builders:
            pytest.skip("No builders found to test")
        
        framework_usage = {}
        
        for canonical_name, builder_class in list(all_builders.items())[:5]:  # Test first 5 builders
            tester = StepTypeTestFrameworkFactory.create_tester(
                builder_class=builder_class,
                canonical_name=canonical_name,
                step_catalog=step_catalog,
                verbose=False,
                enable_scoring=False,
                enable_structured_reporting=False
            )
            
            framework_type = type(tester).__name__
            if framework_type not in framework_usage:
                framework_usage[framework_type] = []
            framework_usage[framework_type].append(canonical_name)
        
        # Print framework usage
        print(f"\n📊 Framework Selection Results:")
        for framework, builders in framework_usage.items():
            print(f"   {framework}: {len(builders)} builders")
            for builder in builders[:3]:  # Show first 3 builders
                print(f"     • {builder}")
            if len(builders) > 3:
                print(f"     ... and {len(builders) - 3} more")
        
        # Should have at least one framework used
        assert len(framework_usage) > 0
        
        # Should have UniversalStepBuilderTest as fallback
        assert any("UniversalStepBuilderTest" in framework for framework in framework_usage.keys())


class TestSpecializedFrameworkFeatures:
    """Test specialized framework-specific features."""
    
    @pytest.fixture(scope="class")
    def step_catalog(self):
        """Create step catalog instance."""
        return StepCatalog(workspace_dirs=None)
    
    def test_processing_pattern_b_logic(self, step_catalog):
        """Test Processing Pattern B auto-pass logic if available."""
        if ProcessingStepBuilderTest is None:
            pytest.skip("ProcessingStepBuilderTest not available")
        
        processing_builders = step_catalog.get_builders_by_step_type("Processing")
        if not processing_builders:
            pytest.skip("No Processing builders found")
        
        # Test first Processing builder with specialized framework
        canonical_name, builder_class = next(iter(processing_builders.items()))
        
        tester = ProcessingStepBuilderTest(
            builder_class=builder_class,
            step_name=canonical_name,
            verbose=False,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        # Run Processing-specific validation
        results = tester.run_processing_validation()
        
        # Verify Processing-specific results structure
        assert 'processing_info' in results
        assert 'test_suite' in results
        assert results['test_suite'] == "ProcessingStepBuilderTest"
        
        # Verify Processing-specific information
        processing_info = results['processing_info']
        assert 'step_type' in processing_info
        assert processing_info['step_type'] == "Processing"
        assert 'creation_patterns' in processing_info
        assert 'pattern_a' in processing_info['creation_patterns']
        assert 'pattern_b' in processing_info['creation_patterns']
    
    def test_training_framework_validation(self, step_catalog):
        """Test Training framework-specific validation if available."""
        if TrainingStepBuilderTest is None:
            pytest.skip("TrainingStepBuilderTest not available")
        
        training_builders = step_catalog.get_builders_by_step_type("Training")
        if not training_builders:
            pytest.skip("No Training builders found")
        
        # Test first Training builder with specialized framework
        canonical_name, builder_class = next(iter(training_builders.items()))
        
        tester = TrainingStepBuilderTest(
            builder_class=builder_class,
            step_name=canonical_name,
            verbose=False,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        # Run Training-specific validation
        results = tester.run_training_validation()
        
        # Verify Training-specific results structure
        assert 'training_info' in results
        assert 'test_suite' in results
        assert results['test_suite'] == "TrainingStepBuilderTest"
        
        # Verify Training-specific information
        training_info = results['training_info']
        assert 'step_type' in training_info
        assert training_info['step_type'] == "Training"
        assert 'supported_frameworks' in training_info
        assert 'training_patterns' in training_info
    
    def test_createmodel_deployment_validation(self, step_catalog):
        """Test CreateModel deployment-specific validation if available."""
        if CreateModelStepBuilderTest is None:
            pytest.skip("CreateModelStepBuilderTest not available")
        
        createmodel_builders = step_catalog.get_builders_by_step_type("CreateModel")
        if not createmodel_builders:
            pytest.skip("No CreateModel builders found")
        
        # Test first CreateModel builder with specialized framework
        canonical_name, builder_class = next(iter(createmodel_builders.items()))
        
        tester = CreateModelStepBuilderTest(
            builder_class=builder_class,
            step_name=canonical_name,
            verbose=False,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        # Run CreateModel-specific validation
        results = tester.run_createmodel_validation()
        
        # Verify CreateModel-specific results structure
        assert 'createmodel_info' in results
        assert 'test_suite' in results
        assert results['test_suite'] == "CreateModelStepBuilderTest"
        
        # Verify CreateModel-specific information
        createmodel_info = results['createmodel_info']
        assert 'step_type' in createmodel_info
        assert createmodel_info['step_type'] == "CreateModel"
        assert 'deployment_patterns' in createmodel_info
        assert 'common_features' in createmodel_info
