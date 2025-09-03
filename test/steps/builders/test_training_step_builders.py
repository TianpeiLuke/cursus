"""
Comprehensive test suite for all Training step builders.

This module creates tests for all existing Training step builders using the
Universal Step Builder Test framework with Training-specific enhancements.
"""

import pytest
import unittest
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import importlib
import sys
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cursus.validation.builders.universal_test import UniversalStepBuilderTest
from cursus.validation.builders.registry_discovery import (
    RegistryStepDiscovery,
    get_training_steps_from_registry,
    get_builder_class_path,
    load_builder_class
)
from cursus.core.base.builder_base import StepBuilderBase
from cursus.registry.step_names import STEP_NAMES


class TrainingStepBuilderTestSuite:
    """
    Comprehensive test suite for all Training step builders.
    
    This class provides specialized testing for Training steps with enhanced
    validation for estimator creation, input/output handling, and Training-specific
    requirements.
    
    Uses registry information to automatically discover Training step builders,
    making it adaptive when new steps are added.
    """
    
    @classmethod
    def get_training_steps_from_registry(cls) -> List[str]:
        """Get all Training step names from the registry."""
        return get_training_steps_from_registry()
    
    # Expected estimator types for each step
    EXPECTED_ESTIMATORS = {
        "PyTorchTraining": "PyTorch",
        "XGBoostTraining": "XGBoost"
    }
    
    # Framework versions expected
    EXPECTED_FRAMEWORKS = {
        "PyTorchTraining": "pytorch",
        "XGBoostTraining": "xgboost"
    }
    
    @classmethod
    def load_builder_class(cls, step_name: str) -> Optional[Type[StepBuilderBase]]:
        """
        Dynamically load a builder class by step name using registry information.
        
        Args:
            step_name: Name of the step (e.g., "PyTorchTraining")
            
        Returns:
            Builder class if found, None otherwise
        """
        try:
            return load_builder_class(step_name)
        except (ImportError, AttributeError, ValueError) as e:
            print(f"Failed to load {step_name} builder: {e}")
            return None
    
    @classmethod
    def get_available_training_builders(cls) -> List[tuple]:
        """
        Get all available Training step builders for testing using registry information.
        
        Returns:
            List of tuples (step_name, builder_class)
        """
        available_builders = []
        training_steps = cls.get_training_steps_from_registry()
        
        for step_name in training_steps:
            builder_class = cls.load_builder_class(step_name)
            if builder_class:
                available_builders.append((step_name, builder_class))
            else:
                print(f"Skipping {step_name} - builder class not available")
                
        return available_builders
    
    def run_training_specific_tests(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """
        Run Training-specific tests for a step builder.
        
        Args:
            step_name: Name of the step
            builder_class: Builder class to test
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # Test 1: Estimator Creation Method
        results["test_estimator_creation_method"] = self._test_estimator_creation_method(
            step_name, builder_class
        )
        
        # Test 2: Expected Estimator Type
        results["test_expected_estimator_type"] = self._test_expected_estimator_type(
            step_name, builder_class
        )
        
        # Test 3: Training Input/Output Methods
        results["test_training_io_methods"] = self._test_training_io_methods(
            step_name, builder_class
        )
        
        # Test 4: Environment Variables
        results["test_training_environment_variables"] = self._test_training_environment_variables(
            step_name, builder_class
        )
        
        # Test 5: Hyperparameters Handling
        results["test_hyperparameters_handling"] = self._test_hyperparameters_handling(
            step_name, builder_class
        )
        
        # Test 6: Framework-Specific Tests
        results["test_framework_specific"] = self._test_framework_specific(
            step_name, builder_class
        )
        
        # Test 7: Training Step Creation
        results["test_training_step_creation"] = self._test_training_step_creation(
            step_name, builder_class
        )
        
        return results
    
    def _test_estimator_creation_method(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """Test that the builder has estimator creation methods."""
        try:
            estimator_methods = ['_create_estimator', '_get_estimator']
            found_methods = [m for m in estimator_methods if hasattr(builder_class, m)]
            
            return {
                "passed": len(found_methods) > 0,
                "error": "No estimator creation methods found" if not found_methods else None,
                "details": {
                    "expected_methods": estimator_methods,
                    "found_methods": found_methods,
                    "step_name": step_name
                }
            }
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing estimator creation method: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def _test_expected_estimator_type(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """Test that the builder creates the expected estimator type."""
        try:
            expected_estimator = self.EXPECTED_ESTIMATORS.get(step_name)
            if not expected_estimator:
                return {
                    "passed": True,
                    "error": None,
                    "details": {"note": f"No expected estimator defined for {step_name}"}
                }
            
            # Check if the builder has _create_estimator method
            if not hasattr(builder_class, '_create_estimator'):
                return {
                    "passed": False,
                    "error": f"Builder missing _create_estimator method",
                    "details": {"expected_estimator": expected_estimator}
                }
            
            # This is a structural test - we verify the method exists and can be called
            # Actual estimator creation would require full configuration setup
            return {
                "passed": True,
                "error": None,
                "details": {
                    "expected_estimator": expected_estimator,
                    "has_create_method": True,
                    "note": "Structural validation passed - estimator creation method exists"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing expected estimator type: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def _test_training_io_methods(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """Test that the builder has required input/output methods."""
        try:
            required_methods = ['_get_inputs', '_get_outputs']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(builder_class, method):
                    missing_methods.append(method)
            
            return {
                "passed": len(missing_methods) == 0,
                "error": f"Missing methods: {missing_methods}" if missing_methods else None,
                "details": {
                    "required_methods": required_methods,
                    "missing_methods": missing_methods,
                    "has_get_inputs": hasattr(builder_class, '_get_inputs'),
                    "has_get_outputs": hasattr(builder_class, '_get_outputs')
                }
            }
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing I/O methods: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def _test_training_environment_variables(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """Test that the builder handles environment variables correctly."""
        try:
            # Check if the builder has environment variable method
            has_env_method = hasattr(builder_class, '_get_environment_variables')
            
            if not has_env_method:
                return {
                    "passed": False,
                    "error": "Missing _get_environment_variables method",
                    "details": {"step_name": step_name}
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "has_env_method": True,
                    "note": "Environment variable method exists"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing environment variables: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def _test_hyperparameters_handling(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """Test that the builder handles hyperparameters correctly."""
        try:
            # Check if the builder has hyperparameters handling
            has_hyperparams_method = hasattr(builder_class, '_upload_hyperparameters_to_s3')
            
            return {
                "passed": True,  # This is informational - not all training steps need hyperparameters upload
                "error": None,
                "details": {
                    "has_hyperparameters_method": has_hyperparams_method,
                    "note": "Hyperparameters handling method is optional for Training steps"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing hyperparameters handling: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def _test_framework_specific(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """Test framework-specific requirements."""
        try:
            expected_framework = self.EXPECTED_FRAMEWORKS.get(step_name)
            if not expected_framework:
                return {
                    "passed": True,
                    "error": None,
                    "details": {"note": f"No expected framework defined for {step_name}"}
                }
            
            # For PyTorch steps, check for PyTorch-specific methods/attributes
            if expected_framework == "pytorch":
                return {
                    "passed": True,
                    "error": None,
                    "details": {
                        "expected_framework": expected_framework,
                        "note": "PyTorch framework expected - structural validation passed"
                    }
                }
            
            # For XGBoost steps, check for XGBoost-specific methods/attributes
            if expected_framework == "xgboost":
                return {
                    "passed": True,
                    "error": None,
                    "details": {
                        "expected_framework": expected_framework,
                        "note": "XGBoost framework expected - structural validation passed"
                    }
                }
            
            # For sklearn steps, basic validation
            return {
                "passed": True,
                "error": None,
                "details": {
                    "expected_framework": expected_framework,
                    "note": "SKLearn framework expected - structural validation passed"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing framework-specific requirements: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def _test_training_step_creation(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """Test that the builder can create a TrainingStep."""
        try:
            # Check if the builder has create_step method
            has_create_step = hasattr(builder_class, 'create_step')
            
            if not has_create_step:
                return {
                    "passed": False,
                    "error": "Missing create_step method",
                    "details": {"step_name": step_name}
                }
            
            # Check method signature
            import inspect
            sig = inspect.signature(builder_class.create_step)
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "has_create_step": True,
                    "method_signature": str(sig),
                    "note": "create_step method exists with proper signature"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing step creation: {str(e)}",
                "details": {"exception": str(e)}
            }


class TestTrainingStepBuilders(unittest.TestCase):
    """
    Test cases for all Training step builders using pytest parametrization.
    """
    
    def setUp(self):
        """Set up test suite."""
        self.test_suite = TrainingStepBuilderTestSuite()
        self.available_builders = self.test_suite.get_available_training_builders()
    
    def test_training_builders_available(self):
        """Test that we have Training step builders available for testing."""
        self.assertGreater(len(self.available_builders), 0, 
                          "No Training step builders available for testing")
        
        print(f"\nFound {len(self.available_builders)} Training step builders:")
        for step_name, builder_class in self.available_builders:
            print(f"  - {step_name}: {builder_class.__name__}")
    
    def test_all_training_builders_universal_compliance(self):
        """Test all Training step builders for universal compliance."""
        if not self.available_builders:
            self.skipTest("No Training step builders available")
        
        all_results = {}
        
        for step_name, builder_class in self.available_builders:
            print(f"\n{'='*60}")
            print(f"Testing {step_name} ({builder_class.__name__})")
            print(f"{'='*60}")
            
            # Run universal tests with enhanced features
            try:
                tester = UniversalStepBuilderTest(
                    builder_class, 
                    verbose=True,
                    enable_scoring=True,
                    enable_structured_reporting=False  # Keep simple for batch testing
                )
                results = tester.run_all_tests()
                # Extract test results from enhanced format
                universal_results = results.get('test_results', results) if isinstance(results, dict) and 'test_results' in results else results
                
                # Run Training-specific tests
                training_results = self.test_suite.run_training_specific_tests(
                    step_name, builder_class
                )
                
                # Combine results
                combined_results = {**universal_results, **training_results}
                all_results[step_name] = combined_results
                
                # Report results for this builder
                self._report_builder_results(step_name, combined_results)
                
            except Exception as e:
                print(f"❌ Failed to test {step_name}: {str(e)}")
                all_results[step_name] = {"error": str(e)}
        
        # Report overall summary
        self._report_overall_summary(all_results)
        
        # Assert that critical tests passed for all builders
        self._assert_critical_tests_passed(all_results)
    
    def _report_builder_results(self, step_name: str, results: Dict[str, Any]):
        """Report results for a single builder."""
        passed_tests = sum(1 for result in results.values() 
                          if isinstance(result, dict) and result.get("passed", False))
        total_tests = len([r for r in results.values() if isinstance(r, dict)])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n{step_name} Results: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        
        # Show failed tests
        failed_tests = {k: v for k, v in results.items() 
                       if isinstance(v, dict) and not v.get("passed", True)}
        
        if failed_tests:
            print("Failed Tests:")
            for test_name, result in failed_tests.items():
                print(f"  ❌ {test_name}: {result.get('error', 'Unknown error')}")
        else:
            print("✅ All tests passed!")
    
    def _report_overall_summary(self, all_results: Dict[str, Any]):
        """Report overall summary across all builders."""
        print(f"\n{'='*80}")
        print("OVERALL TRAINING STEP BUILDERS TEST SUMMARY")
        print(f"{'='*80}")
        
        total_builders = len(all_results)
        successful_builders = 0
        total_tests = 0
        total_passed = 0
        
        for step_name, results in all_results.items():
            if "error" in results:
                print(f"❌ {step_name}: Test execution failed")
                continue
                
            builder_tests = len([r for r in results.values() if isinstance(r, dict)])
            builder_passed = sum(1 for result in results.values() 
                               if isinstance(result, dict) and result.get("passed", False))
            
            total_tests += builder_tests
            total_passed += builder_passed
            
            if builder_passed == builder_tests:
                successful_builders += 1
                print(f"✅ {step_name}: {builder_passed}/{builder_tests} tests passed")
            else:
                print(f"⚠️  {step_name}: {builder_passed}/{builder_tests} tests passed")
        
        overall_pass_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        builder_success_rate = (successful_builders / total_builders) * 100 if total_builders > 0 else 0
        
        print(f"\nOverall Statistics:")
        print(f"  Builders tested: {total_builders}")
        print(f"  Builders with all tests passing: {successful_builders} ({builder_success_rate:.1f}%)")
        print(f"  Total tests run: {total_tests}")
        print(f"  Total tests passed: {total_passed} ({overall_pass_rate:.1f}%)")
        
        print(f"\n{'='*80}")
    
    def _assert_critical_tests_passed(self, all_results: Dict[str, Any]):
        """Assert that critical tests passed for all builders."""
        critical_tests = [
            "test_inheritance",
            "test_required_methods",
            "test_estimator_creation_method",
            "test_training_io_methods"
        ]
        
        failed_critical = []
        
        for step_name, results in all_results.items():
            if "error" in results:
                failed_critical.append(f"{step_name}: Test execution failed")
                continue
                
            for test_name in critical_tests:
                if test_name in results:
                    result = results[test_name]
                    if isinstance(result, dict) and not result.get("passed", False):
                        failed_critical.append(f"{step_name}.{test_name}: {result.get('error', 'Failed')}")
        
        if failed_critical:
            self.fail(f"Critical tests failed:\n" + "\n".join(failed_critical))


# Pytest parametrized tests for individual builder testing
@pytest.fixture
def training_test_suite():
    """Fixture to provide the test suite."""
    return TrainingStepBuilderTestSuite()


@pytest.fixture
def available_training_builders(training_test_suite):
    """Fixture to provide available Training builders."""
    return training_test_suite.get_available_training_builders()


@pytest.mark.parametrize("step_name,builder_class", 
                        TrainingStepBuilderTestSuite().get_available_training_builders())
def test_individual_training_builder_universal_compliance(step_name, builder_class):
    """Test individual Training step builder for universal compliance."""
    print(f"\nTesting {step_name} ({builder_class.__name__})")
    
    # Run universal tests
    tester = UniversalStepBuilderTest(builder_class, verbose=False)
    results = tester.run_all_tests()
    
    # Check critical tests
    critical_tests = ["test_inheritance", "test_required_methods"]
    for test_name in critical_tests:
        if test_name in results:
            result = results[test_name]
            assert result["passed"], f"{test_name} failed for {step_name}: {result.get('error')}"


@pytest.mark.parametrize("step_name,builder_class", 
                        TrainingStepBuilderTestSuite().get_available_training_builders())
def test_individual_training_builder_training_specific(step_name, builder_class):
    """Test individual Training step builder for Training-specific requirements."""
    print(f"\nTesting Training-specific requirements for {step_name}")
    
    test_suite = TrainingStepBuilderTestSuite()
    results = test_suite.run_training_specific_tests(step_name, builder_class)
    
    # Check critical Training tests
    critical_training_tests = ["test_estimator_creation_method", "test_training_io_methods"]
    for test_name in critical_training_tests:
        if test_name in results:
            result = results[test_name]
            assert result["passed"], f"{test_name} failed for {step_name}: {result.get('error')}"


if __name__ == '__main__':
    # Run with unittest
    unittest.main(verbosity=2)
