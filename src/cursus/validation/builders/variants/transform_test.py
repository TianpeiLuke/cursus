"""
Transform Step Builder Validation Test Suite.

This module provides comprehensive validation for Transform step builders using
a modular 4-level testing approach:

Level 1: Interface Tests - Basic interface and inheritance validation
Level 2: Specification Tests - Specification and contract compliance
Level 3: Step Creation Tests - Step creation and configuration validation  
Level 4: Integration Tests - End-to-end step creation and system integration

The tests are designed to validate Transform-specific patterns including:
- Batch inference and processing
- Model integration and artifact handling
- Data format and content type handling
- Performance optimization and resource allocation
- Multi-framework transform support
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock
import logging

from .transform_interface_tests import TransformInterfaceTests
from .transform_specification_tests import TransformSpecificationTests
from .transform_integration_tests import TransformIntegrationTests
from ..universal_test import UniversalStepBuilderTest

logger = logging.getLogger(__name__)


class TransformStepBuilderTest(UniversalStepBuilderTest):
    """
    Comprehensive Transform step builder validation test suite.

    This class orchestrates all 4 levels of Transform-specific validation tests
    using the modular test structure. It extends the UniversalStepBuilderTest
    to provide Transform-specific testing capabilities.
    """

    def __init__(
        self,
        builder_class,
        config=None,
        spec=None,
        contract=None,
        step_name=None,
        verbose: bool = False,
        test_reporter=None,
        step_info: Optional[Dict[str, Any]] = None,
        enable_scoring: bool = False,
        enable_structured_reporting: bool = False,
        **kwargs
    ):
        """
        Initialize Transform step builder test suite.

        Args:
            builder_class: The Transform step builder class to test
            config: Optional config to use (will create via step catalog if not provided)
            spec: Optional step specification
            contract: Optional script contract
            step_name: Optional step name
            verbose: Whether to print verbose output
            test_reporter: Optional function to report test results
            step_info: Optional step information dictionary
            enable_scoring: Whether to enable scoring functionality
            enable_structured_reporting: Whether to enable structured reporting
            **kwargs: Additional arguments for subclasses
        """
        # Set Transform-specific step info
        if step_info is None:
            step_info = {
                "sagemaker_step_type": "Transform",
                "step_category": "inference",
                "supported_frameworks": ["pytorch", "xgboost", "tensorflow", "sklearn"],
                "transform_patterns": [
                    "Batch Inference",
                    "Model Integration",
                    "Data Format Handling",
                    "Performance Optimization",
                ],
                "common_features": [
                    "batch_processing",
                    "model_integration",
                    "data_format_handling",
                    "performance_optimization",
                ],
                "special_features": [
                    "multi_framework_support",
                    "content_type_handling",
                    "batch_size_optimization",
                    "inference_acceleration",
                ],
            }

        # Store step_info for transform-specific use
        self.step_info = step_info

        # Initialize parent class with new signature
        super().__init__(
            builder_class=builder_class,
            config=config,
            spec=spec,
            contract=contract,
            step_name=step_name,
            verbose=verbose,
            test_reporter=test_reporter,
            **kwargs
        )

        # Initialize Transform-specific test levels
        self._initialize_transform_test_levels()

    def _initialize_transform_test_levels(self) -> None:
        """Initialize Transform-specific test level instances."""
        # Level 1: Transform Interface Tests
        self.level1_tester = TransformInterfaceTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Level 2: Transform Specification Tests
        self.level2_tester = TransformSpecificationTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Level 3: Transform Step Creation Tests (using base step creation tests)
        from ..step_creation_tests import StepCreationTests
        self.level3_tester = StepCreationTests(
            builder_class=self.builder_class,
            config=self._provided_config,
            spec=self._provided_spec,
            contract=self._provided_contract,
            step_name=self._provided_step_name,
            verbose=self.verbose,
            test_reporter=self.test_reporter,
        )

        # Level 4: Transform Integration Tests
        self.level4_tester = TransformIntegrationTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Override the base test levels with Transform-specific ones
        self.test_levels = {
            1: self.level1_tester,
            2: self.level2_tester,
            3: self.level3_tester,
            4: self.level4_tester,
        }

    def get_transform_specific_info(self) -> Dict[str, Any]:
        """Get Transform-specific information for reporting."""
        return {
            "step_type": "Transform",
            "supported_frameworks": ["pytorch", "xgboost", "tensorflow", "sklearn"],
            "transform_patterns": {
                "batch_inference": "Large-scale batch inference processing",
                "model_integration": "Integration with trained models for inference",
                "data_format_handling": "Support for multiple data formats and content types",
                "performance_optimization": "Batch size and resource optimization",
            },
            "common_features": {
                "batch_processing": "Efficient batch processing capabilities",
                "model_integration": "Seamless model artifact integration",
                "data_format_handling": "Multiple input/output format support",
                "performance_optimization": "Resource and throughput optimization",
            },
            "special_features": {
                "multi_framework_support": "Support for multiple ML frameworks",
                "content_type_handling": "Advanced content type processing",
                "batch_size_optimization": "Automatic batch size optimization",
                "inference_acceleration": "Hardware acceleration for inference",
            },
        }

    def run_transform_validation(
        self, levels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run Transform-specific validation tests.

        Args:
            levels: Optional list of test levels to run (1-4). If None, runs all levels.

        Returns:
            Dictionary containing test results and Transform-specific information
        """
        if levels is None:
            levels = [1, 2, 3, 4]

        # Run the standard validation
        results = self.run_all_tests()

        # Add Transform-specific information
        if isinstance(results, dict):
            results["transform_info"] = self.get_transform_specific_info()
            results["test_suite"] = "TransformStepBuilderTest"

            # Add level-specific summaries
            if "test_results" in results:
                test_results = results["test_results"]

                # Level 1 summary
                if 1 in levels and "level_1" in test_results:
                    level1_results = test_results["level_1"]
                    level1_results["summary"] = (
                        "Transform interface and inheritance validation"
                    )
                    level1_results["focus"] = (
                        "Transformer creation methods, batch processing configuration, model integration"
                    )

                # Level 2 summary
                if 2 in levels and "level_2" in test_results:
                    level2_results = test_results["level_2"]
                    level2_results["summary"] = (
                        "Transform specification and contract compliance"
                    )
                    level2_results["focus"] = (
                        "Batch processing specification, data format handling, performance configuration"
                    )

                # Level 3 summary
                if 3 in levels and "level_3" in test_results:
                    level3_results = test_results["level_3"]
                    level3_results["summary"] = (
                        "Transform step creation and configuration validation"
                    )
                    level3_results["focus"] = (
                        "TransformJob creation, input/output mapping, model artifact integration"
                    )

                # Level 4 summary
                if 4 in levels and "level_4" in test_results:
                    level4_results = test_results["level_4"]
                    level4_results["summary"] = (
                        "Transform integration and end-to-end workflow"
                    )
                    level4_results["focus"] = (
                        "Complete transform workflow, batch processing integration, performance optimization"
                    )

        return results

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all Transform validation tests across all 4 levels.

        Returns:
            Dict containing comprehensive test results
        """
        logger.info("Running complete Transform validation test suite")

        results = {
            "step_type": self.step_type,
            "builder_type": type(self.builder_instance).__name__,
            "test_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "overall_passed": True,
            },
            "level_results": {},
        }

        # Run Level 1: Interface Tests
        logger.info("Running Level 1: Transform Interface Tests")
        level1_results = self.interface_tests.run_all_tests()
        results["level_results"]["level_1_interface"] = level1_results
        self._update_summary(results["test_summary"], level1_results)

        # Run Level 2: Specification Tests
        logger.info("Running Level 2: Transform Specification Tests")
        level2_results = self.specification_tests.run_all_tests()
        results["level_results"]["level_2_specification"] = level2_results
        self._update_summary(results["test_summary"], level2_results)

        # Run Level 3: Path Mapping Tests
        logger.info("Running Level 3: Transform Path Mapping Tests")
        level3_results = self.path_mapping_tests.run_all_tests()
        results["level_results"]["level_3_path_mapping"] = level3_results
        self._update_summary(results["test_summary"], level3_results)

        # Run Level 4: Integration Tests
        logger.info("Running Level 4: Transform Integration Tests")
        level4_results = self.integration_tests.run_all_tests()
        results["level_results"]["level_4_integration"] = level4_results
        self._update_summary(results["test_summary"], level4_results)

        # Finalize summary
        results["test_summary"]["overall_passed"] = (
            results["test_summary"]["failed_tests"] == 0
        )

        logger.info(
            f"Transform validation complete. "
            f"Passed: {results['test_summary']['passed_tests']}, "
            f"Failed: {results['test_summary']['failed_tests']}"
        )

        return results

    def run_interface_tests(self) -> Dict[str, Any]:
        """Run only Level 1 Transform interface tests."""
        logger.info("Running Transform Interface Tests (Level 1)")
        return self.interface_tests.run_all_tests()

    def run_specification_tests(self) -> Dict[str, Any]:
        """Run only Level 2 Transform specification tests."""
        logger.info("Running Transform Specification Tests (Level 2)")
        return self.specification_tests.run_all_tests()

    def run_path_mapping_tests(self) -> Dict[str, Any]:
        """Run only Level 3 Transform path mapping tests."""
        logger.info("Running Transform Path Mapping Tests (Level 3)")
        return self.path_mapping_tests.run_all_tests()

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run only Level 4 Transform integration tests."""
        logger.info("Running Transform Integration Tests (Level 4)")
        return self.integration_tests.run_all_tests()

    def run_framework_specific_tests(self, framework: str) -> Dict[str, Any]:
        """
        Run Transform tests specific to a particular ML framework.

        Args:
            framework: The ML framework to test ('pytorch', 'xgboost', 'tensorflow', 'sklearn')

        Returns:
            Dict containing framework-specific test results
        """
        logger.info(f"Running Transform tests for framework: {framework}")

        results = {
            "step_type": self.step_type,
            "framework": framework,
            "builder_type": type(self.builder_instance).__name__,
            "framework_tests": {},
        }

        # Run framework-specific interface tests
        if hasattr(self.interface_tests, f"test_{framework}_specific_methods"):
            framework_interface = getattr(
                self.interface_tests, f"test_{framework}_specific_methods"
            )()
            results["framework_tests"]["interface"] = framework_interface

        # Run framework-specific specification tests
        if hasattr(self.specification_tests, f"test_{framework}_configuration"):
            framework_spec = getattr(
                self.specification_tests, f"test_{framework}_configuration"
            )()
            results["framework_tests"]["specification"] = framework_spec

        # Run framework-specific path mapping tests
        if hasattr(self.path_mapping_tests, f"test_{framework}_transform_paths"):
            framework_paths = getattr(
                self.path_mapping_tests, f"test_{framework}_transform_paths"
            )()
            results["framework_tests"]["path_mapping"] = framework_paths

        # Run framework-specific integration tests
        if hasattr(self.integration_tests, f"test_{framework}_transform_workflow"):
            framework_integration = getattr(
                self.integration_tests, f"test_{framework}_transform_workflow"
            )()
            results["framework_tests"]["integration"] = framework_integration

        return results

    def run_batch_processing_tests(self) -> Dict[str, Any]:
        """
        Run Transform batch processing validation tests.

        Returns:
            Dict containing batch processing test results
        """
        logger.info("Running Transform batch processing tests")

        results = {
            "step_type": self.step_type,
            "test_type": "batch_processing",
            "builder_type": type(self.builder_instance).__name__,
            "batch_tests": {},
        }

        # Run batch processing specification tests
        if hasattr(
            self.specification_tests, "test_batch_processing_specification_compliance"
        ):
            batch_spec = (
                self.specification_tests.test_batch_processing_specification_compliance()
            )
            results["batch_tests"]["specification"] = batch_spec

        # Run batch input/output configuration tests
        if hasattr(self.path_mapping_tests, "test_transform_input_object_creation"):
            batch_io = self.path_mapping_tests.test_transform_input_object_creation()
            results["batch_tests"]["input_output"] = batch_io

        # Run batch processing integration
        if hasattr(self.integration_tests, "test_batch_processing_integration"):
            batch_integration = (
                self.integration_tests.test_batch_processing_integration()
            )
            results["batch_tests"]["integration"] = batch_integration

        # Run batch transform optimization
        if hasattr(self.integration_tests, "test_batch_transform_optimization"):
            batch_optimization = (
                self.integration_tests.test_batch_transform_optimization()
            )
            results["batch_tests"]["optimization"] = batch_optimization

        return results

    def run_model_integration_tests(self) -> Dict[str, Any]:
        """
        Run Transform model integration tests.

        Returns:
            Dict containing model integration test results
        """
        logger.info("Running Transform model integration tests")

        results = {
            "step_type": self.step_type,
            "test_type": "model_integration",
            "builder_type": type(self.builder_instance).__name__,
            "model_tests": {},
        }

        # Run model integration specification tests
        if hasattr(self.specification_tests, "test_model_integration_specification"):
            model_spec = self.specification_tests.test_model_integration_specification()
            results["model_tests"]["specification"] = model_spec

        # Run model artifact path handling tests
        if hasattr(self.path_mapping_tests, "test_model_artifact_path_handling"):
            model_paths = self.path_mapping_tests.test_model_artifact_path_handling()
            results["model_tests"]["path_handling"] = model_paths

        # Run model integration workflow tests
        if hasattr(self.integration_tests, "test_model_integration_workflow"):
            model_workflow = self.integration_tests.test_model_integration_workflow()
            results["model_tests"]["workflow"] = model_workflow

        # Run model dependency resolution tests
        if hasattr(self.integration_tests, "test_model_dependency_resolution"):
            model_dependencies = (
                self.integration_tests.test_model_dependency_resolution()
            )
            results["model_tests"]["dependencies"] = model_dependencies

        return results

    def run_data_format_tests(self) -> Dict[str, Any]:
        """
        Run Transform data format and content type tests.

        Returns:
            Dict containing data format test results
        """
        logger.info("Running Transform data format tests")

        results = {
            "step_type": self.step_type,
            "test_type": "data_formats",
            "builder_type": type(self.builder_instance).__name__,
            "format_tests": {},
        }

        # Run data format specification tests
        if hasattr(self.specification_tests, "test_data_format_specification"):
            format_spec = self.specification_tests.test_data_format_specification()
            results["format_tests"]["specification"] = format_spec

        # Run content type and format handling tests
        if hasattr(self.path_mapping_tests, "test_content_type_and_format_handling"):
            content_handling = (
                self.path_mapping_tests.test_content_type_and_format_handling()
            )
            results["format_tests"]["content_handling"] = content_handling

        # Run data format integration tests
        if hasattr(self.integration_tests, "test_data_format_integration"):
            format_integration = self.integration_tests.test_data_format_integration()
            results["format_tests"]["integration"] = format_integration

        return results

    def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run Transform performance optimization tests.

        Returns:
            Dict containing performance test results
        """
        logger.info("Running Transform performance tests")

        results = {
            "step_type": self.step_type,
            "test_type": "performance",
            "builder_type": type(self.builder_instance).__name__,
            "performance_tests": {},
        }

        # Run batch size optimization tests
        if hasattr(
            self.specification_tests, "test_batch_size_optimization_specification"
        ):
            batch_optimization = (
                self.specification_tests.test_batch_size_optimization_specification()
            )
            results["performance_tests"]["batch_optimization"] = batch_optimization

        # Run resource allocation tests
        if hasattr(self.specification_tests, "test_resource_allocation_specification"):
            resource_test = (
                self.specification_tests.test_resource_allocation_specification()
            )
            results["performance_tests"]["resource_allocation"] = resource_test

        # Run transform performance integration
        if hasattr(self.integration_tests, "test_transform_performance_optimization"):
            performance_integration = (
                self.integration_tests.test_transform_performance_optimization()
            )
            results["performance_tests"]["integration"] = performance_integration

        return results

    def generate_transform_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive Transform validation report.

        Args:
            test_results: Results from running Transform tests

        Returns:
            Dict containing formatted report
        """
        logger.info("Generating Transform validation report")

        report = {
            "report_type": "Transform Validation Report",
            "step_type": self.step_type,
            "builder_type": type(self.builder_instance).__name__,
            "timestamp": self._get_timestamp(),
            "summary": self._generate_summary(test_results),
            "detailed_results": test_results,
            "recommendations": self._generate_recommendations(test_results),
            "framework_analysis": self._analyze_framework_compatibility(test_results),
            "batch_processing_readiness": self._assess_batch_processing_readiness(
                test_results
            ),
        }

        return report

    def get_transform_test_coverage(self) -> Dict[str, Any]:
        """
        Get Transform test coverage information.

        Returns:
            Dict containing test coverage details
        """
        coverage = {
            "step_type": self.step_type,
            "coverage_analysis": {
                "level_1_interface": {
                    "total_tests": len(
                        self.interface_tests.get_step_type_specific_tests()
                    ),
                    "test_categories": [
                        "transformer_creation_methods",
                        "batch_processing_configuration",
                        "model_integration_methods",
                        "framework_specific_methods",
                    ],
                },
                "level_2_specification": {
                    "total_tests": len(
                        self.specification_tests.get_step_type_specific_tests()
                    ),
                    "test_categories": [
                        "batch_processing_specification",
                        "model_integration_specification",
                        "data_format_specification",
                        "framework_specific_specifications",
                    ],
                },
                "level_3_path_mapping": {
                    "total_tests": len(
                        self.path_mapping_tests.get_step_type_specific_tests()
                    ),
                    "test_categories": [
                        "transform_input_object_creation",
                        "model_artifact_path_handling",
                        "content_type_and_format_handling",
                        "transform_output_path_mapping",
                    ],
                },
                "level_4_integration": {
                    "total_tests": len(
                        self.integration_tests.get_step_type_specific_tests()
                    ),
                    "test_categories": [
                        "complete_transform_step_creation",
                        "model_integration_workflow",
                        "batch_processing_integration",
                        "framework_specific_transform_workflow",
                    ],
                },
            },
            "framework_support": [
                "pytorch",
                "xgboost",
                "tensorflow",
                "sklearn",
                "custom",
            ],
            "transform_patterns": [
                "batch_inference",
                "model_integration",
                "data_format_handling",
                "performance_optimization",
            ],
        }

        total_tests = sum(
            level["total_tests"] for level in coverage["coverage_analysis"].values()
        )
        coverage["total_test_count"] = total_tests

        return coverage

    # Helper methods

    def _update_summary(
        self, summary: Dict[str, Any], level_results: Dict[str, Any]
    ) -> None:
        """Update test summary with level results."""
        if "test_results" in level_results:
            for test_result in level_results["test_results"]:
                summary["total_tests"] += 1
                if test_result.get("passed", False):
                    summary["passed_tests"] += 1
                else:
                    summary["failed_tests"] += 1

    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test results summary."""
        if "test_summary" in test_results:
            return test_results["test_summary"]

        return {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "overall_passed": True,
        }

    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Analyze failed tests and generate recommendations
        if "level_results" in test_results:
            for level_name, level_result in test_results["level_results"].items():
                if "test_results" in level_result:
                    for test in level_result["test_results"]:
                        if not test.get("passed", True) and "errors" in test:
                            for error in test["errors"]:
                                if "transformer" in error.lower():
                                    recommendations.append(
                                        "Review transformer configuration and batch processing setup"
                                    )
                                elif "model" in error.lower():
                                    recommendations.append(
                                        "Validate model integration and artifact accessibility"
                                    )
                                elif "batch" in error.lower():
                                    recommendations.append(
                                        "Check batch processing configuration and data format handling"
                                    )
                                elif "transform" in error.lower():
                                    recommendations.append(
                                        "Verify transform job configuration and resource allocation"
                                    )

        # Add general recommendations
        if not recommendations:
            recommendations.append("All Transform validation tests passed successfully")

        return list(set(recommendations))  # Remove duplicates

    def _analyze_framework_compatibility(
        self, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze framework compatibility from test results."""
        framework_analysis = {
            "detected_framework": "unknown",
            "compatibility_status": "unknown",
            "framework_specific_issues": [],
        }

        # Try to detect framework from test results
        if "level_results" in test_results:
            for level_result in test_results["level_results"].values():
                if "test_results" in level_result:
                    for test in level_result["test_results"]:
                        if "details" in test and "framework" in test["details"]:
                            framework_analysis["detected_framework"] = test["details"][
                                "framework"
                            ]
                            break

        return framework_analysis

    def _assess_batch_processing_readiness(
        self, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess batch processing readiness from test results."""
        readiness = {
            "ready_for_batch_processing": True,
            "readiness_score": 100,
            "blocking_issues": [],
            "warnings": [],
        }

        # Analyze test results for batch processing readiness
        if "test_summary" in test_results:
            if test_results["test_summary"]["failed_tests"] > 0:
                readiness["ready_for_batch_processing"] = False
                readiness["readiness_score"] = max(
                    0, 100 - (test_results["test_summary"]["failed_tests"] * 10)
                )

        return readiness

    def _get_timestamp(self) -> str:
        """Get current timestamp for reporting."""
        from datetime import datetime

        return datetime.now().isoformat()


# Convenience functions for Transform testing


def run_transform_validation(
    builder_instance, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run complete Transform validation.

    Args:
        builder_instance: Transform step builder instance
        config: Optional configuration dictionary

    Returns:
        Dict containing complete validation results
    """
    if config is None:
        config = {}

    test_orchestrator = TransformTest(builder_instance, config)
    return test_orchestrator.run_all_tests()


def run_transform_framework_tests(
    builder_instance, framework: str, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run framework-specific Transform tests.

    Args:
        builder_instance: Transform step builder instance
        framework: ML framework to test
        config: Optional configuration dictionary

    Returns:
        Dict containing framework-specific test results
    """
    if config is None:
        config = {}

    test_orchestrator = TransformTest(builder_instance, config)
    return test_orchestrator.run_framework_specific_tests(framework)


def run_transform_batch_processing_tests(
    builder_instance, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run Transform batch processing tests.

    Args:
        builder_instance: Transform step builder instance
        config: Optional configuration dictionary

    Returns:
        Dict containing batch processing test results
    """
    if config is None:
        config = {}

    test_orchestrator = TransformTest(builder_instance, config)
    return test_orchestrator.run_batch_processing_tests()


def run_transform_model_integration_tests(
    builder_instance, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run Transform model integration tests.

    Args:
        builder_instance: Transform step builder instance
        config: Optional configuration dictionary

    Returns:
        Dict containing model integration test results
    """
    if config is None:
        config = {}

    test_orchestrator = TransformTest(builder_instance, config)
    return test_orchestrator.run_model_integration_tests()


def generate_transform_report(
    builder_instance, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate Transform validation report.

    Args:
        builder_instance: Transform step builder instance
        config: Optional configuration dictionary

    Returns:
        Dict containing comprehensive validation report
    """
    if config is None:
        config = {}

    test_orchestrator = TransformTest(builder_instance, config)
    test_results = test_orchestrator.run_all_tests()
    return test_orchestrator.generate_transform_report(test_results)
