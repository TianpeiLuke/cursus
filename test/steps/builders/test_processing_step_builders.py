"""
Processing step builders testing using dynamic discovery.

This module provides dynamic testing for Processing step builders discovered
via the step catalog system, eliminating hard-coded builder lists.
"""

import pytest
from cursus.step_catalog import StepCatalog
from cursus.validation.builders.universal_test import UniversalStepBuilderTest
from cursus.validation.builders.results_storage import BuilderTestResultsStorage


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
            assert builder_class.__name__.endswith('StepBuilder'), f"Invalid builder class name: {builder_class.__name__}"
    
    @pytest.mark.parametrize("canonical_name,builder_class", 
                           lambda: list(StepCatalog().get_builders_by_step_type("Processing").items()))
    def test_processing_builder_compliance(self, canonical_name, builder_class):
        """Test individual Processing builder compliance."""
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
        
        # Assert critical tests pass
        critical_tests = ['test_inheritance', 'test_required_methods']
        for test_name in critical_tests:
            if test_name in raw_results:
                assert raw_results[test_name].get('passed', False), f"{canonical_name} failed critical test: {test_name}"
    
    def test_processing_specific_requirements(self, processing_builders):
        """Test Processing-specific requirements for all builders."""
        if not processing_builders:
            pytest.skip("No Processing builders found")
        
        results = {}
        
        for canonical_name, builder_class in processing_builders.items():
            builder_results = {}
            
            # Test processor creation methods
            processor_methods = ["_create_processor", "_get_processor", "create_step"]
            found_methods = [m for m in processor_methods if hasattr(builder_class, m)]
            builder_results['processor_methods'] = {
                'passed': len(found_methods) >= 1,
                'found_methods': found_methods,
                'expected_methods': processor_methods
            }
            
            # Test I/O methods
            io_methods = ["_get_inputs", "_get_outputs"]
            missing_io_methods = [m for m in io_methods if not hasattr(builder_class, m)]
            builder_results['io_methods'] = {
                'passed': len(missing_io_methods) == 0,
                'missing_methods': missing_io_methods,
                'required_methods': io_methods
            }
            
            # Test processing code handling
            code_methods = ["_get_code", "_get_source_dir", "_process_code"]
            found_code_methods = [m for m in code_methods if hasattr(builder_class, m)]
            builder_results['code_methods'] = {
                'passed': len(found_code_methods) >= 1,
                'found_methods': found_code_methods,
                'expected_methods': code_methods
            }
            
            # Test environment variables method
            builder_results['environment_variables'] = {
                'passed': hasattr(builder_class, '_get_environment_variables'),
                'has_method': hasattr(builder_class, '_get_environment_variables')
            }
            
            # Test processing job arguments
            job_arg_methods = ["_get_job_arguments", "_process_arguments"]
            found_job_methods = [m for m in job_arg_methods if hasattr(builder_class, m)]
            builder_results['job_arguments'] = {
                'passed': len(found_job_methods) >= 1,
                'found_methods': found_job_methods,
                'expected_methods': job_arg_methods
            }
            
            results[canonical_name] = builder_results
        
        # Save results
        BuilderTestResultsStorage.save_test_results(results, "single_builder", "processing_specific")
        
        # Assert that all builders have basic Processing requirements
        for canonical_name, builder_results in results.items():
            assert builder_results['processor_methods']['passed'], f"{canonical_name} missing processor creation methods"
            assert builder_results['io_methods']['passed'], f"{canonical_name} missing required I/O methods"


class TestProcessingBuilderIntegration:
    """Integration tests for Processing builders with step catalog."""
    
    def test_processing_step_type_classification(self):
        """Test that Processing builders are correctly classified."""
        catalog = StepCatalog(workspace_dirs=None)
        processing_builders = catalog.get_builders_by_step_type("Processing")
        all_builders = catalog.get_all_builders()
        
        # All Processing builders should be in the complete builders list
        for name in processing_builders:
            assert name in all_builders, f"Processing builder {name} not found in all builders"
        
        # Verify step type classification
        for canonical_name, builder_class in processing_builders.items():
            step_info = catalog.get_step_info(canonical_name)
            if step_info and step_info.registry_data:
                step_type = step_info.registry_data.get('sagemaker_step_type')
                assert step_type == 'Processing', f"{canonical_name} has incorrect step type: {step_type}"
    
    def test_processing_builder_loading(self):
        """Test that Processing builders can be loaded individually."""
        catalog = StepCatalog(workspace_dirs=None)
        processing_builders = catalog.get_builders_by_step_type("Processing")
        
        for canonical_name in processing_builders:
            builder_class = catalog.load_builder_class(canonical_name)
            assert builder_class is not None, f"Failed to load Processing builder: {canonical_name}"
            assert builder_class.__name__.endswith('StepBuilder'), f"Invalid builder class: {builder_class.__name__}"
    
    def test_processing_registry_data(self, processing_builders):
        """Test that Processing builders have proper registry data."""
        if not processing_builders:
            pytest.skip("No Processing builders found")
        
        catalog = StepCatalog(workspace_dirs=None)
        
        for canonical_name, builder_class in processing_builders.items():
            step_info = catalog.get_step_info(canonical_name)
            
            # Should have step info from registry
            assert step_info is not None, f"No step info found for {canonical_name}"
            
            # Should have registry data
            assert step_info.registry_data is not None, f"No registry data found for {canonical_name}"
            
            # Should be classified as Processing step type
            step_type = step_info.registry_data.get('sagemaker_step_type')
            assert step_type == 'Processing', f"{canonical_name} has incorrect step type in registry: {step_type}"
    
    def test_processing_builder_count(self, processing_builders):
        """Test that we have a reasonable number of Processing builders."""
        # Processing is typically the most common step type
        assert len(processing_builders) >= 5, f"Expected at least 5 Processing builders, found {len(processing_builders)}"
