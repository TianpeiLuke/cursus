"""
Training step builders testing using dynamic discovery.

This module provides dynamic testing for Training step builders discovered
via the step catalog system, eliminating hard-coded builder lists.
"""

import pytest
from cursus.step_catalog import StepCatalog
from cursus.validation.builders.universal_test import UniversalStepBuilderTest
from cursus.validation.builders.results_storage import BuilderTestResultsStorage


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
            assert builder_class.__name__.endswith('StepBuilder'), f"Invalid builder class name: {builder_class.__name__}"
    
    @pytest.mark.parametrize("canonical_name,builder_class", 
                           lambda: list(StepCatalog().get_builders_by_step_type("Training").items()))
    def test_training_builder_compliance(self, canonical_name, builder_class):
        """Test individual Training builder compliance."""
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
        
        # Assert critical tests pass
        critical_tests = ['test_inheritance', 'test_required_methods']
        for test_name in critical_tests:
            if test_name in raw_results:
                assert raw_results[test_name].get('passed', False), f"{canonical_name} failed critical test: {test_name}"
    
    def test_training_specific_requirements(self, training_builders):
        """Test Training-specific requirements for all builders."""
        if not training_builders:
            pytest.skip("No Training builders found")
        
        results = {}
        
        for canonical_name, builder_class in training_builders.items():
            builder_results = {}
            
            # Test estimator creation methods
            estimator_methods = ["_create_estimator", "_get_estimator", "create_step"]
            found_methods = [m for m in estimator_methods if hasattr(builder_class, m)]
            builder_results['estimator_methods'] = {
                'passed': len(found_methods) >= 1,
                'found_methods': found_methods,
                'expected_methods': estimator_methods
            }
            
            # Test I/O methods
            io_methods = ["_get_inputs", "_get_outputs"]
            missing_io_methods = [m for m in io_methods if not hasattr(builder_class, m)]
            builder_results['io_methods'] = {
                'passed': len(missing_io_methods) == 0,
                'missing_methods': missing_io_methods,
                'required_methods': io_methods
            }
            
            # Test hyperparameter handling
            hyperparameter_methods = ["_get_hyperparameters", "_process_hyperparameters"]
            found_hp_methods = [m for m in hyperparameter_methods if hasattr(builder_class, m)]
            builder_results['hyperparameter_methods'] = {
                'passed': len(found_hp_methods) >= 1,
                'found_methods': found_hp_methods,
                'expected_methods': hyperparameter_methods
            }
            
            # Test environment variables method
            builder_results['environment_variables'] = {
                'passed': hasattr(builder_class, '_get_environment_variables'),
                'has_method': hasattr(builder_class, '_get_environment_variables')
            }
            
            results[canonical_name] = builder_results
        
        # Save results
        BuilderTestResultsStorage.save_test_results(results, "single_builder", "training_specific")
        
        # Assert that all builders have basic Training requirements
        for canonical_name, builder_results in results.items():
            assert builder_results['estimator_methods']['passed'], f"{canonical_name} missing estimator creation methods"
            assert builder_results['io_methods']['passed'], f"{canonical_name} missing required I/O methods"


class TestTrainingBuilderIntegration:
    """Integration tests for Training builders with step catalog."""
    
    def test_training_step_type_classification(self):
        """Test that Training builders are correctly classified."""
        catalog = StepCatalog(workspace_dirs=None)
        training_builders = catalog.get_builders_by_step_type("Training")
        all_builders = catalog.get_all_builders()
        
        # All Training builders should be in the complete builders list
        for name in training_builders:
            assert name in all_builders, f"Training builder {name} not found in all builders"
        
        # Verify step type classification
        for canonical_name, builder_class in training_builders.items():
            step_info = catalog.get_step_info(canonical_name)
            if step_info and step_info.registry_data:
                step_type = step_info.registry_data.get('sagemaker_step_type')
                assert step_type == 'Training', f"{canonical_name} has incorrect step type: {step_type}"
    
    def test_training_builder_loading(self):
        """Test that Training builders can be loaded individually."""
        catalog = StepCatalog(workspace_dirs=None)
        training_builders = catalog.get_builders_by_step_type("Training")
        
        for canonical_name in training_builders:
            builder_class = catalog.load_builder_class(canonical_name)
            assert builder_class is not None, f"Failed to load Training builder: {canonical_name}"
            assert builder_class.__name__.endswith('StepBuilder'), f"Invalid builder class: {builder_class.__name__}"
    
    def test_training_registry_data(self, training_builders):
        """Test that Training builders have proper registry data."""
        if not training_builders:
            pytest.skip("No Training builders found")
        
        catalog = StepCatalog(workspace_dirs=None)
        
        for canonical_name, builder_class in training_builders.items():
            step_info = catalog.get_step_info(canonical_name)
            
            # Should have step info from registry
            assert step_info is not None, f"No step info found for {canonical_name}"
            
            # Should have registry data
            assert step_info.registry_data is not None, f"No registry data found for {canonical_name}"
            
            # Should be classified as Training step type
            step_type = step_info.registry_data.get('sagemaker_step_type')
            assert step_type == 'Training', f"{canonical_name} has incorrect step type in registry: {step_type}"
            
            # Training steps should have framework information in registry if available
            framework_info = step_info.registry_data.get('framework', step_info.registry_data.get('ml_framework'))
            # Framework info is optional but if present should be meaningful
            if framework_info:
                assert isinstance(framework_info, str) and len(framework_info) > 0, f"{canonical_name} has invalid framework info: {framework_info}"
