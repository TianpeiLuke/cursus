"""
Universal Step Builder Test Suite.

This module combines all test levels into a single comprehensive test suite
that evaluates step builders against all architectural requirements.
"""

import unittest
from types import SimpleNamespace
from typing import Dict, List, Any, Optional, Union, Type

# Import base classes for type hints
from ...core.base.builder_base import StepBuilderBase
from ...core.base.specification_base import StepSpecification
from ...core.base.contract_base import ScriptContract
from ...core.base.config_base import BaseModel as ConfigBase

# Import test levels
from .interface_tests import InterfaceTests
from .specification_tests import SpecificationTests
from .path_mapping_tests import PathMappingTests
from .integration_tests import IntegrationTests
from .sagemaker_step_type_validator import SageMakerStepTypeValidator
from .base_test import StepName


class UniversalStepBuilderTest:
    """
    Universal test suite for validating step builder implementation compliance.
    
    This test combines all test levels to provide a comprehensive validation
    of step builder implementations. Tests are grouped by architectural level
    to provide clearer feedback and easier debugging.
    
    Usage:
        # Test a specific builder
        tester = UniversalStepBuilderTest(XGBoostTrainingStepBuilder)
        tester.run_all_tests()
        
        # Or register with pytest
        @pytest.mark.parametrize("builder_class", [
            XGBoostTrainingStepBuilder,
            TabularPreprocessingStepBuilder,
            ModelEvalStepBuilder
        ])
        def test_step_builder_compliance(builder_class):
            tester = UniversalStepBuilderTest(builder_class)
            tester.run_all_tests()
    """
    
    def __init__(
        self, 
        builder_class: Type[StepBuilderBase],
        config: Optional[ConfigBase] = None,
        spec: Optional[StepSpecification] = None,
        contract: Optional[ScriptContract] = None,
        step_name: Optional[Union[str, StepName]] = None,
        verbose: bool = False
    ):
        """
        Initialize with explicit components.
        
        Args:
            builder_class: The step builder class to test
            config: Optional config to use (will create mock if not provided)
            spec: Optional step specification (will extract from builder if not provided)
            contract: Optional script contract (will extract from builder if not provided)
            step_name: Optional step name (will extract from class name if not provided)
            verbose: Whether to print verbose output
        """
        self.builder_class = builder_class
        self.config = config
        self.spec = spec
        self.contract = contract
        self.step_name = step_name
        self.verbose = verbose
        
        # Create test suites for each level
        self.interface_tests = InterfaceTests(
            builder_class=builder_class,
            config=config,
            spec=spec,
            contract=contract,
            step_name=step_name,
            verbose=verbose
        )
        
        self.specification_tests = SpecificationTests(
            builder_class=builder_class,
            config=config,
            spec=spec,
            contract=contract,
            step_name=step_name,
            verbose=verbose
        )
        
        self.path_mapping_tests = PathMappingTests(
            builder_class=builder_class,
            config=config,
            spec=spec,
            contract=contract,
            step_name=step_name,
            verbose=verbose
        )
        
        self.integration_tests = IntegrationTests(
            builder_class=builder_class,
            config=config,
            spec=spec,
            contract=contract,
            step_name=step_name,
            verbose=verbose
        )
        
        # Create SageMaker step type validator
        self.sagemaker_validator = SageMakerStepTypeValidator(builder_class)
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all tests across all levels.
        
        Returns:
            Dictionary mapping test names to their results
        """
        # Run tests for each level
        level1_results = self.interface_tests.run_all_tests()
        level2_results = self.specification_tests.run_all_tests()
        level3_results = self.path_mapping_tests.run_all_tests()
        level4_results = self.integration_tests.run_all_tests()
        
        # Run SageMaker step type validation
        sagemaker_results = self.run_step_type_specific_tests()
        
        # Combine results
        all_results = {}
        all_results.update(level1_results)
        all_results.update(level2_results)
        all_results.update(level3_results)
        all_results.update(level4_results)
        all_results.update(sagemaker_results)
        
        # Print consolidated results if verbose
        if self.verbose:
            self._report_consolidated_results(all_results)
        
        return all_results
    
    def run_step_type_specific_tests(self) -> Dict[str, Any]:
        """Run tests specific to the SageMaker step type."""
        results = {}
        
        try:
            # Get step type information
            step_type_info = self.sagemaker_validator.get_step_type_info()
            
            # Test step type detection
            results["test_step_type_detection"] = {
                "passed": step_type_info["detected_step_name"] is not None,
                "error": None if step_type_info["detected_step_name"] else "Could not detect step name from builder class",
                "details": step_type_info
            }
            
            # Test step type classification
            results["test_step_type_classification"] = {
                "passed": step_type_info["sagemaker_step_type"] is not None,
                "error": None if step_type_info["sagemaker_step_type"] else "No SageMaker step type classification found",
                "details": {
                    "sagemaker_step_type": step_type_info["sagemaker_step_type"],
                    "is_valid": step_type_info["is_valid_step_type"]
                }
            }
            
            # Run step type compliance validation
            violations = self.sagemaker_validator.validate_step_type_compliance()
            
            # Convert violations to test results
            error_violations = [v for v in violations if v.level.name == "ERROR"]
            warning_violations = [v for v in violations if v.level.name == "WARNING"]
            info_violations = [v for v in violations if v.level.name == "INFO"]
            
            results["test_step_type_compliance"] = {
                "passed": len(error_violations) == 0,
                "error": f"{len(error_violations)} critical violations found" if error_violations else None,
                "details": {
                    "error_count": len(error_violations),
                    "warning_count": len(warning_violations),
                    "info_count": len(info_violations),
                    "violations": [
                        {
                            "level": v.level.name,
                            "category": v.category,
                            "message": v.message,
                            "details": v.details
                        }
                        for v in violations
                    ]
                }
            }
            
            # Add specific tests based on step type
            if step_type_info["sagemaker_step_type"] == "Processing":
                results.update(self._run_processing_tests())
            elif step_type_info["sagemaker_step_type"] == "Training":
                results.update(self._run_training_tests())
            elif step_type_info["sagemaker_step_type"] == "Transform":
                results.update(self._run_transform_tests())
            elif step_type_info["sagemaker_step_type"] == "CreateModel":
                results.update(self._run_create_model_tests())
            elif step_type_info["sagemaker_step_type"] == "RegisterModel":
                results.update(self._run_register_model_tests())
                
        except Exception as e:
            results["test_step_type_validation"] = {
                "passed": False,
                "error": f"Step type validation failed: {str(e)}",
                "details": {"exception": str(e)}
            }
        
        return results
    
    def _run_processing_tests(self) -> Dict[str, Any]:
        """Run Processing-specific tests."""
        results = {}
        
        # Test processor creation methods
        processor_methods = ['_create_processor', '_get_processor']
        found_methods = [m for m in processor_methods if hasattr(self.builder_class, m)]
        
        results["test_processing_processor_methods"] = {
            "passed": len(found_methods) > 0,
            "error": "No processor creation methods found" if not found_methods else None,
            "details": {
                "expected_methods": processor_methods,
                "found_methods": found_methods
            }
        }
        
        # Test input/output methods
        results["test_processing_io_methods"] = {
            "passed": hasattr(self.builder_class, '_get_inputs') and hasattr(self.builder_class, '_get_outputs'),
            "error": "Missing _get_inputs or _get_outputs methods" if not (hasattr(self.builder_class, '_get_inputs') and hasattr(self.builder_class, '_get_outputs')) else None,
            "details": {
                "has_get_inputs": hasattr(self.builder_class, '_get_inputs'),
                "has_get_outputs": hasattr(self.builder_class, '_get_outputs')
            }
        }
        
        return results
    
    def _run_training_tests(self) -> Dict[str, Any]:
        """Run Training-specific tests."""
        results = {}
        
        # Test estimator creation methods
        estimator_methods = ['_create_estimator', '_get_estimator']
        found_methods = [m for m in estimator_methods if hasattr(self.builder_class, m)]
        
        results["test_training_estimator_methods"] = {
            "passed": len(found_methods) > 0,
            "error": "No estimator creation methods found" if not found_methods else None,
            "details": {
                "expected_methods": estimator_methods,
                "found_methods": found_methods
            }
        }
        
        # Test hyperparameter methods
        hyperparameter_methods = ['_prepare_hyperparameters_file', '_get_hyperparameters']
        found_hp_methods = [m for m in hyperparameter_methods if hasattr(self.builder_class, m)]
        
        results["test_training_hyperparameter_methods"] = {
            "passed": True,  # This is informational, not required
            "error": None,
            "details": {
                "expected_methods": hyperparameter_methods,
                "found_methods": found_hp_methods,
                "note": "Hyperparameter methods are recommended but not required"
            }
        }
        
        return results
    
    def _run_transform_tests(self) -> Dict[str, Any]:
        """Run Transform-specific tests."""
        results = {}
        
        # Test transformer creation methods
        transformer_methods = ['_create_transformer', '_get_transformer']
        found_methods = [m for m in transformer_methods if hasattr(self.builder_class, m)]
        
        results["test_transform_transformer_methods"] = {
            "passed": len(found_methods) > 0,
            "error": "No transformer creation methods found" if not found_methods else None,
            "details": {
                "expected_methods": transformer_methods,
                "found_methods": found_methods
            }
        }
        
        return results
    
    def _run_create_model_tests(self) -> Dict[str, Any]:
        """Run CreateModel-specific tests."""
        results = {}
        
        # Test model creation methods
        model_methods = ['_create_model', '_get_model']
        found_methods = [m for m in model_methods if hasattr(self.builder_class, m)]
        
        results["test_create_model_methods"] = {
            "passed": len(found_methods) > 0,
            "error": "No model creation methods found" if not found_methods else None,
            "details": {
                "expected_methods": model_methods,
                "found_methods": found_methods
            }
        }
        
        return results
    
    def _run_register_model_tests(self) -> Dict[str, Any]:
        """Run RegisterModel-specific tests."""
        results = {}
        
        # Test model package methods
        package_methods = ['_create_model_package', '_get_model_package_args']
        found_methods = [m for m in package_methods if hasattr(self.builder_class, m)]
        
        results["test_register_model_package_methods"] = {
            "passed": True,  # This is informational, not required
            "error": None,
            "details": {
                "expected_methods": package_methods,
                "found_methods": found_methods,
                "note": "Model package methods are recommended but not required"
            }
        }
        
        return results
    
    def _report_consolidated_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Report consolidated results across all test levels."""
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result["passed"])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Group results by level
        level1_results = {k: v for k, v in results.items() if k.startswith('test_') and hasattr(InterfaceTests, k)}
        level2_results = {k: v for k, v in results.items() if k.startswith('test_') and hasattr(SpecificationTests, k)}
        level3_results = {k: v for k, v in results.items() if k.startswith('test_') and hasattr(PathMappingTests, k)}
        level4_results = {k: v for k, v in results.items() if k.startswith('test_') and hasattr(IntegrationTests, k)}
        
        # Calculate level statistics
        def get_level_stats(level_results):
            level_total = len(level_results)
            level_passed = sum(1 for result in level_results.values() if result["passed"])
            level_rate = (level_passed / level_total) * 100 if level_total > 0 else 0
            return level_total, level_passed, level_rate
        
        l1_total, l1_passed, l1_rate = get_level_stats(level1_results)
        l2_total, l2_passed, l2_rate = get_level_stats(level2_results)
        l3_total, l3_passed, l3_rate = get_level_stats(level3_results)
        l4_total, l4_passed, l4_rate = get_level_stats(level4_results)
        
        # Print summary header
        print("\n" + "=" * 80)
        print(f"UNIVERSAL STEP BUILDER TEST RESULTS FOR {self.builder_class.__name__}")
        print("=" * 80)
        
        # Print overall summary
        print(f"\nOVERALL: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        
        # Print level summaries
        print(f"\nLevel 1 (Interface): {l1_passed}/{l1_total} tests passed ({l1_rate:.1f}%)")
        print(f"Level 2 (Specification): {l2_passed}/{l2_total} tests passed ({l2_rate:.1f}%)")
        print(f"Level 3 (Path Mapping): {l3_passed}/{l3_total} tests passed ({l3_rate:.1f}%)")
        print(f"Level 4 (Integration): {l4_passed}/{l4_total} tests passed ({l4_rate:.1f}%)")
        
        # Print failed tests if any
        failed_tests = {k: v for k, v in results.items() if not v["passed"]}
        if failed_tests:
            print("\nFailed Tests:")
            for test_name, result in failed_tests.items():
                print(f"❌ {test_name}: {result['error']}")
        
        print("\n" + "=" * 80)


class TestUniversalStepBuilder(unittest.TestCase):
    """
    Test cases for the UniversalStepBuilderTest class itself.
    
    These tests verify that the universal test suite works correctly
    by applying it to known step builders.
    """
    
    def test_with_xgboost_training_builder(self):
        """Test UniversalStepBuilderTest with XGBoostTrainingStepBuilder."""
        try:
            # Import the builder class
            from ...steps.builders.builder_training_step_xgboost import XGBoostTrainingStepBuilder
            
            # Create tester
            tester = UniversalStepBuilderTest(XGBoostTrainingStepBuilder)
            
            # Run all tests
            results = tester.run_all_tests()
            
            # Check that key tests passed
            self.assertTrue(results["test_inheritance"]["passed"])
            self.assertTrue(results["test_required_methods"]["passed"])
        except ImportError:
            self.skipTest("XGBoostTrainingStepBuilder not available")
    
    def test_with_tabular_preprocessing_builder(self):
        """Test UniversalStepBuilderTest with TabularPreprocessingStepBuilder."""
        try:
            # Import the builder class
            from ...steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
            
            # Create tester
            tester = UniversalStepBuilderTest(TabularPreprocessingStepBuilder)
            
            # Run all tests
            results = tester.run_all_tests()
            
            # Check that key tests passed
            self.assertTrue(results["test_inheritance"]["passed"])
            self.assertTrue(results["test_required_methods"]["passed"])
        except ImportError:
            self.skipTest("TabularPreprocessingStepBuilder not available")

    def test_with_explicit_components(self):
        """Test UniversalStepBuilderTest with explicitly provided components."""
        try:
            # Import the builder class
            from ...steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
            from ...steps.specs.preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC
            
            # Create a custom configuration
            config = SimpleNamespace()
            config.region = 'NA'
            config.pipeline_name = 'test-pipeline'
            config.job_type = 'training'
            
            # Create tester with explicit components
            tester = UniversalStepBuilderTest(
                TabularPreprocessingStepBuilder,
                config=config,
                spec=PREPROCESSING_TRAINING_SPEC,
                step_name='CustomPreprocessingStep'
            )
            
            # Run all tests
            results = tester.run_all_tests()
            
            # Check that key tests passed
            self.assertTrue(results["test_inheritance"]["passed"])
        except ImportError:
            self.skipTest("TabularPreprocessingStepBuilder or PREPROCESSING_TRAINING_SPEC not available")


if __name__ == '__main__':
    unittest.main()
