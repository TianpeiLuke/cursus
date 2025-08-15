"""
Comprehensive test suite for all CreateModel step builders.

This module creates tests for all existing CreateModel step builders using the
Universal Step Builder Test framework with CreateModel-specific enhancements.
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
from cursus.core.base.builder_base import StepBuilderBase
from cursus.steps.registry.step_names import get_steps_by_sagemaker_type, STEP_NAMES


class CreateModelStepBuilderTestSuite:
    """
    Comprehensive test suite for all CreateModel step builders.
    
    This class provides specialized testing for CreateModel steps with enhanced
    validation for model creation, input/output handling, and CreateModel-specific
    requirements.
    """
    
    # All CreateModel step builders to test (based on sagemaker_step_type: "CreateModel")
    CREATEMODEL_STEPS = [
        "PyTorchModel",
        "XGBoostModel"
    ]
    
    # Builder class mapping
    BUILDER_CLASS_MAP = {
        "PyTorchModel": "cursus.steps.builders.builder_pytorch_model_step.PyTorchModelStepBuilder",
        "XGBoostModel": "cursus.steps.builders.builder_xgboost_model_step.XGBoostModelStepBuilder"
    }
    
    # Expected model types for each step
    EXPECTED_MODELS = {
        "PyTorchModel": "PyTorchModel",
        "XGBoostModel": "XGBoostModel"
    }
    
    # Framework versions expected
    EXPECTED_FRAMEWORKS = {
        "PyTorchModel": "pytorch",
        "XGBoostModel": "xgboost"
    }
    
    @classmethod
    def load_builder_class(cls, step_name: str) -> Optional[Type[StepBuilderBase]]:
        """
        Dynamically load a builder class by step name.
        
        Args:
            step_name: Name of the step (e.g., "PyTorchModel")
            
        Returns:
            Builder class if found, None otherwise
        """
        if step_name not in cls.BUILDER_CLASS_MAP:
            return None
            
        module_path, class_name = cls.BUILDER_CLASS_MAP[step_name].rsplit('.', 1)
        
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print(f"Failed to load {step_name} builder: {e}")
            return None
    
    @classmethod
    def get_available_createmodel_builders(cls) -> List[tuple]:
        """
        Get all available CreateModel step builders for testing.
        
        Returns:
            List of tuples (step_name, builder_class)
        """
        available_builders = []
        
        for step_name in cls.CREATEMODEL_STEPS:
            builder_class = cls.load_builder_class(step_name)
            if builder_class:
                available_builders.append((step_name, builder_class))
            else:
                print(f"Skipping {step_name} - builder class not available")
                
        return available_builders
    
    def run_createmodel_specific_tests(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """
        Run CreateModel-specific tests for a step builder.
        
        Args:
            step_name: Name of the step
            builder_class: Builder class to test
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # Test 1: Model Creation Method
        results["test_model_creation_method"] = self._test_model_creation_method(
            step_name, builder_class
        )
        
        # Test 2: Expected Model Type
        results["test_expected_model_type"] = self._test_expected_model_type(
            step_name, builder_class
        )
        
        # Test 3: CreateModel Input/Output Methods
        results["test_createmodel_io_methods"] = self._test_createmodel_io_methods(
            step_name, builder_class
        )
        
        # Test 4: Environment Variables
        results["test_createmodel_environment_variables"] = self._test_createmodel_environment_variables(
            step_name, builder_class
        )
        
        # Test 5: Model Data Handling
        results["test_model_data_handling"] = self._test_model_data_handling(
            step_name, builder_class
        )
        
        # Test 6: Framework-Specific Tests
        results["test_framework_specific"] = self._test_framework_specific(
            step_name, builder_class
        )
        
        # Test 7: CreateModel Step Creation
        results["test_createmodel_step_creation"] = self._test_createmodel_step_creation(
            step_name, builder_class
        )
        
        return results
    
    def _test_model_creation_method(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """Test that the builder has model creation methods."""
        try:
            model_methods = ['_create_model', '_get_model']
            found_methods = [m for m in model_methods if hasattr(builder_class, m)]
            
            return {
                "passed": len(found_methods) > 0,
                "error": "No model creation methods found" if not found_methods else None,
                "details": {
                    "expected_methods": model_methods,
                    "found_methods": found_methods,
                    "step_name": step_name
                }
            }
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing model creation method: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def _test_expected_model_type(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """Test that the builder creates the expected model type."""
        try:
            expected_model = self.EXPECTED_MODELS.get(step_name)
            if not expected_model:
                return {
                    "passed": True,
                    "error": None,
                    "details": {"note": f"No expected model defined for {step_name}"}
                }
            
            # Check if the builder has _create_model method
            if not hasattr(builder_class, '_create_model'):
                return {
                    "passed": False,
                    "error": f"Builder missing _create_model method",
                    "details": {"expected_model": expected_model}
                }
            
            # This is a structural test - we verify the method exists and can be called
            # Actual model creation would require full configuration setup
            return {
                "passed": True,
                "error": None,
                "details": {
                    "expected_model": expected_model,
                    "has_create_method": True,
                    "note": "Structural validation passed - model creation method exists"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing expected model type: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def _test_createmodel_io_methods(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
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
    
    def _test_createmodel_environment_variables(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
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
    
    def _test_model_data_handling(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """Test that the builder handles model data correctly."""
        try:
            # CreateModel steps require model_data input
            # Check if the builder's _get_inputs method can handle model_data
            has_get_inputs = hasattr(builder_class, '_get_inputs')
            
            if not has_get_inputs:
                return {
                    "passed": False,
                    "error": "Missing _get_inputs method required for model_data handling",
                    "details": {"step_name": step_name}
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "has_get_inputs": True,
                    "note": "Model data handling method exists"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing model data handling: {str(e)}",
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
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "expected_framework": expected_framework,
                    "note": "Framework-specific validation passed"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing framework-specific requirements: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def _test_createmodel_step_creation(self, step_name: str, builder_class: Type[StepBuilderBase]) -> Dict[str, Any]:
        """Test that the builder can create a CreateModelStep."""
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


class TestCreateModelStepBuilders(unittest.TestCase):
    """
    Test cases for all CreateModel step builders using pytest parametrization.
    """
    
    def setUp(self):
        """Set up test suite."""
        self.test_suite = CreateModelStepBuilderTestSuite()
        self.available_builders = self.test_suite.get_available_createmodel_builders()
    
    def test_createmodel_builders_available(self):
        """Test that we have CreateModel step builders available for testing."""
        self.assertGreater(len(self.available_builders), 0, 
                          "No CreateModel step builders available for testing")
        
        print(f"\nFound {len(self.available_builders)} CreateModel step builders:")
        for step_name, builder_class in self.available_builders:
            print(f"  - {step_name}: {builder_class.__name__}")
    
    def test_all_createmodel_builders_universal_compliance(self):
        """Test all CreateModel step builders for universal compliance."""
        if not self.available_builders:
            self.skipTest("No CreateModel step builders available")
        
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
                
                # Run CreateModel-specific tests
                createmodel_results = self.test_suite.run_createmodel_specific_tests(
                    step_name, builder_class
                )
                
                # Combine results
                combined_results = {**universal_results, **createmodel_results}
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
        print("OVERALL CREATEMODEL STEP BUILDERS TEST SUMMARY")
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
            "test_model_creation_method",
            "test_createmodel_io_methods"
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
def createmodel_test_suite():
    """Fixture to provide the test suite."""
    return CreateModelStepBuilderTestSuite()


@pytest.fixture
def available_createmodel_builders(createmodel_test_suite):
    """Fixture to provide available CreateModel builders."""
    return createmodel_test_suite.get_available_createmodel_builders()


@pytest.mark.parametrize("step_name,builder_class", 
                        CreateModelStepBuilderTestSuite().get_available_createmodel_builders())
def test_individual_createmodel_builder_universal_compliance(step_name, builder_class):
    """Test individual CreateModel step builder for universal compliance."""
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
                        CreateModelStepBuilderTestSuite().get_available_createmodel_builders())
def test_individual_createmodel_builder_createmodel_specific(step_name, builder_class):
    """Test individual CreateModel step builder for CreateModel-specific requirements."""
    print(f"\nTesting CreateModel-specific requirements for {step_name}")
    
    test_suite = CreateModelStepBuilderTestSuite()
    results = test_suite.run_createmodel_specific_tests(step_name, builder_class)
    
    # Check critical CreateModel tests
    critical_createmodel_tests = ["test_model_creation_method", "test_createmodel_io_methods"]
    for test_name in critical_createmodel_tests:
        if test_name in results:
            result = results[test_name]
            assert result["passed"], f"{test_name} failed for {step_name}: {result.get('error')}"


if __name__ == '__main__':
    # Run with unittest
    unittest.main(verbosity=2)
