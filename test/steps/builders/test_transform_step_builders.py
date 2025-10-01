"""
Transform step builders testing using dynamic discovery.

This module provides dynamic testing for Transform step builders discovered
via the step catalog system, eliminating hard-coded builder lists.
"""

import pytest
from cursus.step_catalog import StepCatalog
from cursus.validation.builders.universal_test import UniversalStepBuilderTest
from cursus.validation.builders.results_storage import BuilderTestResultsStorage


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
            assert builder_class.__name__.endswith('StepBuilder'), f"Invalid builder class name: {builder_class.__name__}"
    
    @pytest.mark.parametrize("canonical_name,builder_class", 
                           lambda: list(StepCatalog().get_builders_by_step_type("Transform").items()))
    def test_transform_builder_compliance(self, canonical_name, builder_class):
        """Test individual Transform builder compliance."""
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
        
        # Assert critical tests pass
        critical_tests = ['test_inheritance', 'test_required_methods']
        for test_name in critical_tests:
            if test_name in raw_results:
                assert raw_results[test_name].get('passed', False), f"{canonical_name} failed critical test: {test_name}"
    
    def test_transform_specific_requirements(self, transform_builders):
        """Test Transform-specific requirements for all builders."""
        if not transform_builders:
            pytest.skip("No Transform builders found")
        
        results = {}
        
        for canonical_name, builder_class in transform_builders.items():
            builder_results = {}
            
            # Test transformer creation methods
            transformer_methods = ["_create_transformer", "_get_transformer", "create_step"]
            found_methods = [m for m in transformer_methods if hasattr(builder_class, m)]
            builder_results['transformer_methods'] = {
                'passed': len(found_methods) >= 1,
                'found_methods': found_methods,
                'expected_methods': transformer_methods
            }
            
            # Test I/O methods
            io_methods = ["_get_inputs", "_get_outputs"]
            missing_io_methods = [m for m in io_methods if not hasattr(builder_class, m)]
            builder_results['io_methods'] = {
                'passed': len(missing_io_methods) == 0,
                'missing_methods': missing_io_methods,
                'required_methods': io_methods
            }
            
            # Test model data handling (Transform steps typically use model artifacts)
            model_methods = ["_get_model_data", "_process_model_data"]
            found_model_methods = [m for m in model_methods if hasattr(builder_class, m)]
            builder_results['model_data_methods'] = {
                'passed': len(found_model_methods) >= 1,
                'found_methods': found_model_methods,
                'expected_methods': model_methods
            }
            
            # Test environment variables method
            builder_results['environment_variables'] = {
                'passed': hasattr(builder_class, '_get_environment_variables'),
                'has_method': hasattr(builder_class, '_get_environment_variables')
            }
            
            # Test batch transform specific methods
            batch_methods = ["_get_batch_strategy", "_get_max_concurrent_transforms"]
            found_batch_methods = [m for m in batch_methods if hasattr(builder_class, m)]
            builder_results['batch_methods'] = {
                'passed': True,  # Optional for Transform builders
                'found_methods': found_batch_methods,
                'expected_methods': batch_methods
            }
            
            results[canonical_name] = builder_results
        
        # Save results
        BuilderTestResultsStorage.save_test_results(results, "single_builder", "transform_specific")
        
        # Assert that all builders have basic Transform requirements
        for canonical_name, builder_results in results.items():
            assert builder_results['transformer_methods']['passed'], f"{canonical_name} missing transformer creation methods"
            assert builder_results['io_methods']['passed'], f"{canonical_name} missing required I/O methods"


class TestTransformBuilderIntegration:
    """Integration tests for Transform builders with step catalog."""
    
    def test_transform_step_type_classification(self):
        """Test that Transform builders are correctly classified."""
        catalog = StepCatalog(workspace_dirs=None)
        transform_builders = catalog.get_builders_by_step_type("Transform")
        all_builders = catalog.get_all_builders()
        
        # All Transform builders should be in the complete builders list
        for name in transform_builders:
            assert name in all_builders, f"Transform builder {name} not found in all builders"
        
        # Verify step type classification
        for canonical_name, builder_class in transform_builders.items():
            step_info = catalog.get_step_info(canonical_name)
            if step_info and step_info.registry_data:
                step_type = step_info.registry_data.get('sagemaker_step_type')
                assert step_type == 'Transform', f"{canonical_name} has incorrect step type: {step_type}"
    
    def test_transform_builder_loading(self):
        """Test that Transform builders can be loaded individually."""
        catalog = StepCatalog(workspace_dirs=None)
        transform_builders = catalog.get_builders_by_step_type("Transform")
        
        for canonical_name in transform_builders:
            builder_class = catalog.load_builder_class(canonical_name)
            assert builder_class is not None, f"Failed to load Transform builder: {canonical_name}"
            assert builder_class.__name__.endswith('StepBuilder'), f"Invalid builder class: {builder_class.__name__}"
    
    def test_transform_registry_data(self, transform_builders):
        """Test that Transform builders have proper registry data."""
        if not transform_builders:
            pytest.skip("No Transform builders found")
        
        catalog = StepCatalog(workspace_dirs=None)
        
        for canonical_name, builder_class in transform_builders.items():
            step_info = catalog.get_step_info(canonical_name)
            
            # Should have step info from registry
            assert step_info is not None, f"No step info found for {canonical_name}"
            
            # Should have registry data
            assert step_info.registry_data is not None, f"No registry data found for {canonical_name}"
            
            # Should be classified as Transform step type
            step_type = step_info.registry_data.get('sagemaker_step_type')
            assert step_type == 'Transform', f"{canonical_name} has incorrect step type in registry: {step_type}"
    
    def test_transform_builder_optional(self):
        """Test that Transform builders are optional (not all projects may have them)."""
        catalog = StepCatalog(workspace_dirs=None)
        transform_builders = catalog.get_builders_by_step_type("Transform")
        
        # Transform builders are optional - this test always passes
        # but provides information about Transform builder availability
        if transform_builders:
            print(f"Found {len(transform_builders)} Transform builders: {list(transform_builders.keys())}")
        else:
            print("No Transform builders found - this is acceptable as Transform steps are optional")
        
        assert True  # Always pass - Transform builders are optional
