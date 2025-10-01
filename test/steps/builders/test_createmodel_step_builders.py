"""
CreateModel step builders testing using dynamic discovery.

This module provides dynamic testing for CreateModel step builders discovered
via the step catalog system, eliminating hard-coded builder lists.
"""

import pytest
from cursus.step_catalog import StepCatalog
from cursus.validation.builders.universal_test import UniversalStepBuilderTest
from cursus.validation.builders.results_storage import BuilderTestResultsStorage


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
        # CreateModel builders may not exist in all projects
        for canonical_name, builder_class in createmodel_builders.items():
            assert canonical_name, "Empty canonical name found"
            assert builder_class, "Empty builder class found"
            assert builder_class.__name__.endswith('StepBuilder'), f"Invalid builder class name: {builder_class.__name__}"
    
    @pytest.mark.parametrize("canonical_name,builder_class", 
                           lambda: list(StepCatalog().get_builders_by_step_type("CreateModel").items()))
    def test_createmodel_builder_compliance(self, canonical_name, builder_class):
        """Test individual CreateModel builder compliance."""
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
        
        # Assert critical tests pass
        critical_tests = ['test_inheritance', 'test_required_methods']
        for test_name in critical_tests:
            if test_name in raw_results:
                assert raw_results[test_name].get('passed', False), f"{canonical_name} failed critical test: {test_name}"
    
    def test_createmodel_specific_requirements(self, createmodel_builders):
        """Test CreateModel-specific requirements for all builders."""
        if not createmodel_builders:
            pytest.skip("No CreateModel builders found")
        
        results = {}
        
        for canonical_name, builder_class in createmodel_builders.items():
            builder_results = {}
            
            # Test model creation methods
            model_methods = ["_create_model", "_get_model", "create_step"]
            found_methods = [m for m in model_methods if hasattr(builder_class, m)]
            builder_results['model_methods'] = {
                'passed': len(found_methods) >= 1,
                'found_methods': found_methods,
                'expected_methods': model_methods
            }
            
            # Test I/O methods
            io_methods = ["_get_inputs", "_get_outputs"]
            missing_io_methods = [m for m in io_methods if not hasattr(builder_class, m)]
            builder_results['io_methods'] = {
                'passed': len(missing_io_methods) == 0,
                'missing_methods': missing_io_methods,
                'required_methods': io_methods
            }
            
            # Test environment variables method
            builder_results['environment_variables'] = {
                'passed': hasattr(builder_class, '_get_environment_variables'),
                'has_method': hasattr(builder_class, '_get_environment_variables')
            }
            
            results[canonical_name] = builder_results
        
        # Save results
        BuilderTestResultsStorage.save_test_results(results, "single_builder", "createmodel_specific")
        
        # Assert that all builders have basic CreateModel requirements
        for canonical_name, builder_results in results.items():
            assert builder_results['model_methods']['passed'], f"{canonical_name} missing model creation methods"
            assert builder_results['io_methods']['passed'], f"{canonical_name} missing required I/O methods"


class TestCreateModelBuilderIntegration:
    """Integration tests for CreateModel builders with step catalog."""
    
    def test_createmodel_step_type_classification(self):
        """Test that CreateModel builders are correctly classified."""
        catalog = StepCatalog(workspace_dirs=None)
        createmodel_builders = catalog.get_builders_by_step_type("CreateModel")
        all_builders = catalog.get_all_builders()
        
        # All CreateModel builders should be in the complete builders list
        for name in createmodel_builders:
            assert name in all_builders, f"CreateModel builder {name} not found in all builders"
        
        # Verify step type classification
        for canonical_name, builder_class in createmodel_builders.items():
            step_info = catalog.get_step_info(canonical_name)
            if step_info and step_info.registry_data:
                step_type = step_info.registry_data.get('sagemaker_step_type')
                assert step_type == 'CreateModel', f"{canonical_name} has incorrect step type: {step_type}"
    
    def test_createmodel_builder_loading(self):
        """Test that CreateModel builders can be loaded individually."""
        catalog = StepCatalog(workspace_dirs=None)
        createmodel_builders = catalog.get_builders_by_step_type("CreateModel")
        
        for canonical_name in createmodel_builders:
            builder_class = catalog.load_builder_class(canonical_name)
            assert builder_class is not None, f"Failed to load CreateModel builder: {canonical_name}"
            assert builder_class.__name__.endswith('StepBuilder'), f"Invalid builder class: {builder_class.__name__}"
