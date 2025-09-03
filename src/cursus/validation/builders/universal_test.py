"""
Universal Step Builder Test Suite.

This module combines all test levels into a single comprehensive test suite
that evaluates step builders against all architectural requirements with
integrated scoring and reporting capabilities.
"""

import unittest
from types import SimpleNamespace
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path
import json

# Import base classes for type hints
from ...core.base.builder_base import StepBuilderBase
from ...core.base.specification_base import StepSpecification
from ...core.base.contract_base import ScriptContract
from ...core.base.config_base import BaseModel as ConfigBase

# Import test levels
from .interface_tests import InterfaceTests
from .specification_tests import SpecificationTests
from .step_creation_tests import StepCreationTests
from .integration_tests import IntegrationTests
from .sagemaker_step_type_validator import SageMakerStepTypeValidator
from .base_test import StepName

# Import scoring and reporting
from .scoring import StepBuilderScorer, LEVEL_WEIGHTS, RATING_LEVELS
from ...registry.step_names import STEP_NAMES

# Import registry discovery utilities
from .registry_discovery import RegistryStepDiscovery


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
        verbose: bool = False,
        enable_scoring: bool = True,
        enable_structured_reporting: bool = False
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
            enable_scoring: Whether to calculate and include quality scores
            enable_structured_reporting: Whether to generate structured reports
        """
        self.builder_class = builder_class
        self.config = config
        self.spec = spec
        self.contract = contract
        self.step_name = step_name
        self.verbose = verbose
        self.enable_scoring = enable_scoring
        self.enable_structured_reporting = enable_structured_reporting
        
        # Infer step name if not provided
        if not self.step_name:
            self.step_name = self._infer_step_name()
        
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
        
        # Use processing-specific test variant for Processing step builders
        if self._is_processing_step_builder():
            from .variants.processing_step_creation_tests import ProcessingStepCreationTests
            self.step_creation_tests = ProcessingStepCreationTests(
                builder_class=builder_class,
                config=config,
                spec=spec,
                contract=contract,
                step_name=step_name,
                verbose=verbose
            )
        else:
            self.step_creation_tests = StepCreationTests(
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
    
    def _is_processing_step_builder(self) -> bool:
        """Check if this is a Processing step builder."""
        try:
            step_type_info = self.sagemaker_validator.get_step_type_info()
            return step_type_info.get("sagemaker_step_type") == "Processing"
        except Exception:
            # Fallback: check if builder class name suggests it's a processing builder
            class_name = self.builder_class.__name__.lower()
            processing_indicators = [
                'processing', 'preprocess', 'eval', 'calibration', 
                'package', 'payload', 'currency', 'tabular'
            ]
            return any(indicator in class_name for indicator in processing_indicators)
    
    def run_all_tests(self, include_scoring: bool = None, 
                      include_structured_report: bool = None) -> Dict[str, Any]:
        """
        Run all tests across all levels with optional scoring and structured reporting.
        
        Args:
            include_scoring: Whether to calculate and include quality scores (overrides instance setting)
            include_structured_report: Whether to generate structured report (overrides instance setting)
        
        Returns:
            Dictionary containing test results and optional scoring/reporting data
        """
        # Use method parameters or fall back to instance settings
        calc_scoring = include_scoring if include_scoring is not None else self.enable_scoring
        gen_report = include_structured_report if include_structured_report is not None else self.enable_structured_reporting
        
        # Run tests for each level
        level1_results = self.interface_tests.run_all_tests()
        level2_results = self.specification_tests.run_all_tests()
        level3_results = self.step_creation_tests.run_all_tests()
        level4_results = self.integration_tests.run_all_tests()
        
        # Run SageMaker step type validation
        sagemaker_results = self.run_step_type_specific_tests()
        
        # Combine raw test results
        raw_results = {}
        raw_results.update(level1_results)
        raw_results.update(level2_results)
        raw_results.update(level3_results)
        raw_results.update(level4_results)
        raw_results.update(sagemaker_results)
        
        # Prepare return data
        result_data = {
            'test_results': raw_results
        }
        
        # Add scoring if enabled
        if calc_scoring:
            try:
                scorer = StepBuilderScorer(raw_results)
                score_report = scorer.generate_report()
                result_data['scoring'] = score_report
                
                # Enhanced console output with scoring
                if self.verbose:
                    self._report_consolidated_results_with_scoring(raw_results, score_report)
            except Exception as e:
                print(f"⚠️  Scoring calculation failed: {e}")
                result_data['scoring_error'] = str(e)
                if self.verbose:
                    self._report_consolidated_results(raw_results)
        else:
            # Standard console output without scoring
            if self.verbose:
                self._report_consolidated_results(raw_results)
        
        # Add structured report if enabled
        if gen_report:
            try:
                structured_report = self._generate_structured_report(raw_results, result_data.get('scoring'))
                result_data['structured_report'] = structured_report
            except Exception as e:
                print(f"⚠️  Structured report generation failed: {e}")
                result_data['report_error'] = str(e)
        
        # For backward compatibility, if no enhanced features are enabled, return just the raw results
        if not calc_scoring and not gen_report:
            return raw_results
        
        return result_data
    
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
    
    def _infer_step_name(self) -> str:
        """Infer step name from builder class name."""
        class_name = self.builder_class.__name__
        
        # Remove "StepBuilder" suffix
        if class_name.endswith("StepBuilder"):
            step_name = class_name[:-11]  # Remove "StepBuilder"
        else:
            step_name = class_name
        
        # Look for matching step name in registry
        for name, info in STEP_NAMES.items():
            if info.get("builder_step_name", "").replace("StepBuilder", "") == step_name:
                return name
        
        return step_name
    
    def _report_consolidated_results_with_scoring(self, results: Dict[str, Dict[str, Any]], 
                                                 score_report: Dict[str, Any]) -> None:
        """Report consolidated results with integrated scoring information."""
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result["passed"])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Get scoring data
        overall_score = score_report.get('overall', {}).get('score', 0.0)
        overall_rating = score_report.get('overall', {}).get('rating', 'Unknown')
        level_scores = score_report.get('levels', {})
        
        # Print summary header
        print("\n" + "=" * 80)
        print(f"UNIVERSAL STEP BUILDER TEST RESULTS FOR {self.builder_class.__name__}")
        print("=" * 80)
        
        # Print overall summary with scoring
        status_icon = "✅" if passed_tests == total_tests else "❌"
        print(f"\n{status_icon} OVERALL: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        print(f"📊 QUALITY SCORE: {overall_score:.1f}/100 - {overall_rating}")
        
        # Print level summaries with scores
        print(f"\nLevel Performance:")
        for level_name in ["level1_interface", "level2_specification", "level3_step_creation", "level4_integration"]:
            if level_name in level_scores:
                level_data = level_scores[level_name]
                display_name = level_name.replace("level", "Level ").replace("_", " ").title()
                score = level_data.get('score', 0)
                passed = level_data.get('passed', 0)
                total = level_data.get('total', 0)
                rate = (passed / total * 100) if total > 0 else 0
                print(f"  {display_name}: {score:.1f}/100 ({passed}/{total} tests, {rate:.1f}%)")
        
        # Print failed tests if any
        failed_tests = {k: v for k, v in results.items() if not v["passed"]}
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test_name, result in failed_tests.items():
                print(f"  • {test_name}: {result['error']}")
        
        # Print score breakdown
        if len(level_scores) > 0:
            print(f"\n📈 Score Breakdown:")
            for level_name, level_data in level_scores.items():
                display_name = level_name.replace("level", "L").replace("_", " ").title()
                weight = LEVEL_WEIGHTS.get(level_name, 1.0)
                score = level_data.get('score', 0)
                print(f"  {display_name}: {score:.1f}/100 (weight: {weight}x)")
        
        print("\n" + "=" * 80)
    
    def _generate_structured_report(self, raw_results: Dict[str, Dict[str, Any]], 
                                   scoring_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a structured report following the alignment validation pattern."""
        
        # Get step information
        step_info = STEP_NAMES.get(str(self.step_name), {})
        sagemaker_step_type = step_info.get("sagemaker_step_type", "Unknown")
        
        # Create structured report
        structured_report = {
            'builder_info': {
                'builder_name': str(self.step_name),
                'builder_class': self.builder_class.__name__,
                'sagemaker_step_type': sagemaker_step_type
            },
            'test_results': {
                'level1_interface': self._extract_level_results(raw_results, 'level1'),
                'level2_specification': self._extract_level_results(raw_results, 'level2'),
                'level3_step_creation': self._extract_level_results(raw_results, 'level3'),
                'level4_integration': self._extract_level_results(raw_results, 'level4'),
                'step_type_specific': self._extract_step_type_results(raw_results)
            },
            'summary': {
                'total_tests': len(raw_results),
                'passed_tests': sum(1 for r in raw_results.values() if r.get('passed', False)),
                'pass_rate': (sum(1 for r in raw_results.values() if r.get('passed', False)) / len(raw_results) * 100) if raw_results else 0
            }
        }
        
        # Add scoring data if available
        if scoring_data:
            structured_report['scoring'] = scoring_data
            structured_report['summary']['overall_score'] = scoring_data.get('overall', {}).get('score', 0.0)
            structured_report['summary']['score_rating'] = scoring_data.get('overall', {}).get('rating', 'Unknown')
        
        return structured_report
    
    def _extract_level_results(self, raw_results: Dict[str, Dict[str, Any]], level: str) -> Dict[str, Any]:
        """Extract results for a specific test level."""
        level_results = {}
        
        # Define test patterns for each level
        level_patterns = {
            'level1': ['test_inheritance', 'test_required_methods', 'test_error_handling', 'test_generic'],
            'level2': ['test_specification', 'test_contract', 'test_environment', 'test_job'],
            'level3': ['test_input', 'test_output', 'test_path', 'test_property'],
            'level4': ['test_dependency', 'test_step_creation', 'test_integration']
        }
        
        patterns = level_patterns.get(level, [])
        
        for test_name, result in raw_results.items():
            if any(pattern in test_name.lower() for pattern in patterns):
                level_results[test_name] = result
        
        return level_results
    
    def _extract_step_type_results(self, raw_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract step type-specific test results."""
        step_type_results = {}
        
        step_type_patterns = ['test_step_type', 'test_processing', 'test_training', 'test_transform', 'test_create_model', 'test_register']
        
        for test_name, result in raw_results.items():
            if any(pattern in test_name.lower() for pattern in step_type_patterns):
                step_type_results[test_name] = result
        
        return step_type_results
    
    def run_all_tests_legacy(self) -> Dict[str, Dict[str, Any]]:
        """
        Legacy method that returns raw results for backward compatibility.
        
        This method maintains the original behavior of run_all_tests() before
        the scoring and structured reporting enhancements were added.
        """
        return self.run_all_tests(include_scoring=False, include_structured_report=False)
    
    def run_all_tests_with_scoring(self) -> Dict[str, Any]:
        """
        Convenience method to run tests with scoring enabled.
        
        Returns:
            Dictionary containing test results and scoring data
        """
        return self.run_all_tests(include_scoring=True, include_structured_report=False)
    
    def run_all_tests_with_full_report(self) -> Dict[str, Any]:
        """
        Convenience method to run tests with both scoring and structured reporting.
        
        Returns:
            Dictionary containing test results, scoring, and structured report
        """
        return self.run_all_tests(include_scoring=True, include_structured_report=True)
    
    def export_results_to_json(self, output_path: Optional[str] = None) -> str:
        """
        Export test results with scoring to JSON format.
        
        Args:
            output_path: Optional path to save the JSON file
            
        Returns:
            JSON string of the results
        """
        results = self.run_all_tests_with_full_report()
        json_content = json.dumps(results, indent=2, default=str)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(json_content)
            print(f"✅ Results exported to: {output_path}")
        
        return json_content
    
    @classmethod
    def test_all_builders_by_type(cls, sagemaker_step_type: str, 
                                  verbose: bool = False,
                                  enable_scoring: bool = True) -> Dict[str, Any]:
        """
        Test all builders for a specific SageMaker step type using registry discovery.
        
        Args:
            sagemaker_step_type: The SageMaker step type to test (e.g., 'Training', 'Transform')
            verbose: Whether to print verbose output
            enable_scoring: Whether to calculate and include quality scores
            
        Returns:
            Dictionary containing test results for all builders of the specified type
        """
        results = {}
        
        try:
            # Get all builder classes for the step type
            builder_classes = RegistryStepDiscovery.get_all_builder_classes_by_type(sagemaker_step_type)
            
            for step_name, builder_class in builder_classes.items():
                if verbose:
                    print(f"\n🔍 Testing {step_name} ({builder_class.__name__})...")
                
                try:
                    # Create tester for this builder
                    tester = cls(
                        builder_class=builder_class,
                        step_name=step_name,
                        verbose=verbose,
                        enable_scoring=enable_scoring,
                        enable_structured_reporting=True
                    )
                    
                    # Run tests
                    test_results = tester.run_all_tests()
                    results[step_name] = test_results
                    
                    if verbose:
                        if enable_scoring and 'scoring' in test_results:
                            score = test_results['scoring'].get('overall', {}).get('score', 0)
                            rating = test_results['scoring'].get('overall', {}).get('rating', 'Unknown')
                            print(f"✅ {step_name}: Score {score:.1f}/100 ({rating})")
                        else:
                            passed = test_results.get('test_results', {})
                            total_tests = len(passed)
                            passed_tests = sum(1 for r in passed.values() if r.get('passed', False))
                            print(f"✅ {step_name}: {passed_tests}/{total_tests} tests passed")
                            
                except Exception as e:
                    results[step_name] = {
                        'error': f"Failed to test {step_name}: {str(e)}",
                        'builder_class': builder_class.__name__
                    }
                    if verbose:
                        print(f"❌ {step_name}: {str(e)}")
        
        except Exception as e:
            return {'error': f"Failed to discover builders for type '{sagemaker_step_type}': {str(e)}"}
        
        return results
    
    @classmethod
    def generate_registry_discovery_report(cls) -> Dict[str, Any]:
        """
        Generate a comprehensive report of step builder discovery status.
        
        Returns:
            Dictionary containing discovery report
        """
        return RegistryStepDiscovery.generate_discovery_report()
    
    @classmethod
    def validate_builder_availability(cls, step_name: str) -> Dict[str, Any]:
        """
        Validate that a step builder is available and can be loaded.
        
        Args:
            step_name: The step name to validate
            
        Returns:
            Dictionary containing validation results
        """
        return RegistryStepDiscovery.validate_step_builder_availability(step_name)
    
    def _report_consolidated_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Report consolidated results across all test levels."""
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result["passed"])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Group results by level
        level1_results = {k: v for k, v in results.items() if k.startswith('test_') and hasattr(InterfaceTests, k)}
        level2_results = {k: v for k, v in results.items() if k.startswith('test_') and hasattr(SpecificationTests, k)}
        level3_results = {k: v for k, v in results.items() if k.startswith('test_') and hasattr(StepCreationTests, k)}
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
        print(f"Level 3 (Step Creation): {l3_passed}/{l3_total} tests passed ({l3_rate:.1f}%)")
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
            from ...steps.builders.builder_xgboost_training_step import XGBoostTrainingStepBuilder
            
            # Create tester with scoring disabled for backward compatibility
            tester = UniversalStepBuilderTest(XGBoostTrainingStepBuilder, enable_scoring=False)
            
            # Run all tests (returns raw results when scoring is disabled)
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
            
            # Create tester with scoring disabled for backward compatibility
            tester = UniversalStepBuilderTest(TabularPreprocessingStepBuilder, enable_scoring=False)
            
            # Run all tests (returns raw results when scoring is disabled)
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
            from ...steps.specs.tabular_preprocessing_training_spec import TABULAR_PREPROCESSING_TRAINING_SPEC
            
            # Create a custom configuration
            config = SimpleNamespace()
            config.region = 'NA'
            config.pipeline_name = 'test-pipeline'
            config.job_type = 'training'
            
            # Create tester with explicit components and scoring disabled for backward compatibility
            tester = UniversalStepBuilderTest(
                TabularPreprocessingStepBuilder,
                config=config,
                spec=TABULAR_PREPROCESSING_TRAINING_SPEC,
                step_name='CustomPreprocessingStep',
                enable_scoring=False
            )
            
            # Run all tests (returns raw results when scoring is disabled)
            results = tester.run_all_tests()
            
            # Check that key tests passed
            self.assertTrue(results["test_inheritance"]["passed"])
        except ImportError:
            self.skipTest("TabularPreprocessingStepBuilder or TABULAR_PREPROCESSING_TRAINING_SPEC not available")
    
    def test_scoring_integration(self):
        """Test the new scoring integration functionality."""
        try:
            # Import the builder class
            from ...steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
            
            # Create tester with scoring enabled
            tester = UniversalStepBuilderTest(TabularPreprocessingStepBuilder, enable_scoring=True, verbose=False)
            
            # Run all tests with scoring
            results = tester.run_all_tests()
            
            # Check that the enhanced result structure is returned
            self.assertIn('test_results', results)
            self.assertIn('scoring', results)
            
            # Check scoring structure
            scoring = results['scoring']
            self.assertIn('overall', scoring)
            self.assertIn('levels', scoring)
            self.assertIn('score', scoring['overall'])
            self.assertIn('rating', scoring['overall'])
            
            # Check that raw test results are still accessible
            test_results = results['test_results']
            self.assertIn('test_inheritance', test_results)
            self.assertTrue(test_results['test_inheritance']['passed'])
            
        except ImportError:
            self.skipTest("TabularPreprocessingStepBuilder not available")
    
    def test_structured_reporting(self):
        """Test the structured reporting functionality."""
        try:
            # Import the builder class
            from ...steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
            
            # Create tester with both scoring and structured reporting enabled
            tester = UniversalStepBuilderTest(
                TabularPreprocessingStepBuilder, 
                enable_scoring=True, 
                enable_structured_reporting=True,
                verbose=False
            )
            
            # Run all tests with full reporting
            results = tester.run_all_tests()
            
            # Check that all components are present
            self.assertIn('test_results', results)
            self.assertIn('scoring', results)
            self.assertIn('structured_report', results)
            
            # Check structured report structure
            report = results['structured_report']
            self.assertIn('builder_info', report)
            self.assertIn('test_results', report)
            self.assertIn('summary', report)
            
            # Check builder info
            builder_info = report['builder_info']
            self.assertIn('builder_name', builder_info)
            self.assertIn('builder_class', builder_info)
            self.assertIn('sagemaker_step_type', builder_info)
            
        except ImportError:
            self.skipTest("TabularPreprocessingStepBuilder not available")
    
    def test_convenience_methods(self):
        """Test the convenience methods for different testing modes."""
        try:
            # Import the builder class
            from ...steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
            
            # Create tester
            tester = UniversalStepBuilderTest(TabularPreprocessingStepBuilder, verbose=False)
            
            # Test legacy method
            legacy_results = tester.run_all_tests_legacy()
            self.assertIsInstance(legacy_results, dict)
            self.assertIn('test_inheritance', legacy_results)
            
            # Test scoring method
            scoring_results = tester.run_all_tests_with_scoring()
            self.assertIn('test_results', scoring_results)
            self.assertIn('scoring', scoring_results)
            
            # Test full report method
            full_results = tester.run_all_tests_with_full_report()
            self.assertIn('test_results', full_results)
            self.assertIn('scoring', full_results)
            self.assertIn('structured_report', full_results)
            
        except ImportError:
            self.skipTest("TabularPreprocessingStepBuilder not available")


if __name__ == '__main__':
    unittest.main()
