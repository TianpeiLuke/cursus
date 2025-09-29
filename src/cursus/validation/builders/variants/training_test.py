"""
Training Step Builder Validation Test Suite.

This module provides comprehensive validation for Training step builders using
a modular 4-level testing approach:

Level 1: Interface Tests - Basic interface and inheritance validation
Level 2: Specification Tests - Specification and contract compliance
Level 3: Step Creation Tests - Step creation and configuration validation  
Level 4: Integration Tests - End-to-end step creation and system integration

The tests are designed to validate Training-specific patterns including:
- PyTorch vs XGBoost vs TensorFlow vs SKLearn training
- Hyperparameter optimization and tuning
- Distributed training configurations
- Data channel mapping and validation
- Model artifact handling
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock
import logging

from .training_interface_tests import TrainingInterfaceTests
from .training_specification_tests import TrainingSpecificationTests
from .training_integration_tests import TrainingIntegrationTests
from ..universal_test import UniversalStepBuilderTest

logger = logging.getLogger(__name__)


class TrainingStepBuilderTest(UniversalStepBuilderTest):
    """
    Comprehensive Training step builder validation test suite.

    This class orchestrates all 4 levels of Training-specific validation tests
    using the modular test structure. It extends the UniversalStepBuilderTest
    to provide Training-specific testing capabilities.
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
        Initialize Training step builder test suite.

        Args:
            builder_class: The Training step builder class to test
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
        # Set Training-specific step info
        if step_info is None:
            step_info = {
                "sagemaker_step_type": "Training",
                "step_category": "training",
                "supported_frameworks": ["pytorch", "xgboost", "tensorflow", "sklearn"],
                "training_patterns": [
                    "Single Instance Training",
                    "Distributed Training",
                    "Hyperparameter Tuning",
                    "Multi-Framework Support",
                ],
                "common_features": [
                    "hyperparameter_optimization",
                    "distributed_training",
                    "data_channels",
                    "model_artifacts",
                ],
                "special_features": [
                    "spot_instance_support",
                    "checkpointing",
                    "early_stopping",
                    "custom_metrics",
                ],
            }

        # Store step_info for training-specific use
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

        # Initialize Training-specific test levels
        self._initialize_training_test_levels()

    def _initialize_training_test_levels(self) -> None:
        """Initialize Training-specific test level instances."""
        # Level 1: Training Interface Tests
        self.level1_tester = TrainingInterfaceTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Level 2: Training Specification Tests
        self.level2_tester = TrainingSpecificationTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Level 3: Training Integration Tests (using base step creation tests)
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

        # Level 4: Training Integration Tests
        self.level4_tester = TrainingIntegrationTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Override the base test levels with Training-specific ones
        self.test_levels = {
            1: self.level1_tester,
            2: self.level2_tester,
            3: self.level3_tester,
            4: self.level4_tester,
        }

    def get_training_specific_info(self) -> Dict[str, Any]:
        """Get Training-specific information for reporting."""
        return {
            "step_type": "Training",
            "supported_frameworks": ["pytorch", "xgboost", "tensorflow", "sklearn"],
            "training_patterns": {
                "single_instance": "Standard single-instance training",
                "distributed": "Multi-node distributed training",
                "hyperparameter_tuning": "Automated hyperparameter optimization",
                "multi_framework": "Support for multiple ML frameworks",
            },
            "common_features": {
                "hyperparameter_optimization": "Built-in hyperparameter tuning support",
                "distributed_training": "Multi-node training capabilities",
                "data_channels": "Multiple data input channels",
                "model_artifacts": "Automatic model artifact management",
            },
            "special_features": {
                "spot_instance_support": "Cost-effective spot instance training",
                "checkpointing": "Training checkpoint management",
                "early_stopping": "Automatic early stopping based on metrics",
                "custom_metrics": "Custom training metrics tracking",
            },
        }

    def run_training_validation(
        self, levels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run Training-specific validation tests.

        Args:
            levels: Optional list of test levels to run (1-4). If None, runs all levels.

        Returns:
            Dictionary containing test results and Training-specific information
        """
        if levels is None:
            levels = [1, 2, 3, 4]

        # Run the standard validation
        results = self.run_all_tests()

        # Add Training-specific information
        if isinstance(results, dict):
            results["training_info"] = self.get_training_specific_info()
            results["test_suite"] = "TrainingStepBuilderTest"

            # Add level-specific summaries
            if "test_results" in results:
                test_results = results["test_results"]

                # Level 1 summary
                if 1 in levels and "level_1" in test_results:
                    level1_results = test_results["level_1"]
                    level1_results["summary"] = (
                        "Training interface and inheritance validation"
                    )
                    level1_results["focus"] = (
                        "Estimator creation methods, framework-specific attributes, training configuration"
                    )

                # Level 2 summary
                if 2 in levels and "level_2" in test_results:
                    level2_results = test_results["level_2"]
                    level2_results["summary"] = (
                        "Training specification and contract compliance"
                    )
                    level2_results["focus"] = (
                        "Framework configuration, hyperparameter specification, data channels"
                    )

                # Level 3 summary
                if 3 in levels and "level_3" in test_results:
                    level3_results = test_results["level_3"]
                    level3_results["summary"] = (
                        "Training step creation and configuration validation"
                    )
                    level3_results["focus"] = (
                        "TrainingJob creation, data channel mapping, model artifact paths"
                    )

                # Level 4 summary
                if 4 in levels and "level_4" in test_results:
                    level4_results = test_results["level_4"]
                    level4_results["summary"] = (
                        "Training integration and end-to-end workflow"
                    )
                    level4_results["focus"] = (
                        "Complete training workflow, framework integration, dependency resolution"
                    )

        return results

    def run_interface_tests(self) -> Dict[str, Any]:
        """Run only Level 1 Training interface tests."""
        logger.info("Running Training Interface Tests (Level 1)")
        return self.interface_tests.run_all_tests()

    def run_specification_tests(self) -> Dict[str, Any]:
        """Run only Level 2 Training specification tests."""
        logger.info("Running Training Specification Tests (Level 2)")
        return self.specification_tests.run_all_tests()

    def run_path_mapping_tests(self) -> Dict[str, Any]:
        """Run only Level 3 Training path mapping tests."""
        logger.info("Running Training Path Mapping Tests (Level 3)")
        return self.path_mapping_tests.run_all_tests()

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run only Level 4 Training integration tests."""
        logger.info("Running Training Integration Tests (Level 4)")
        return self.integration_tests.run_all_tests()

    def run_framework_specific_tests(self, framework: str) -> Dict[str, Any]:
        """
        Run Training tests specific to a particular ML framework.

        Args:
            framework: The ML framework to test ('pytorch', 'xgboost', 'tensorflow', 'sklearn')

        Returns:
            Dict containing framework-specific test results
        """
        logger.info(f"Running Training tests for framework: {framework}")

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
        if hasattr(self.path_mapping_tests, f"test_{framework}_training_paths"):
            framework_paths = getattr(
                self.path_mapping_tests, f"test_{framework}_training_paths"
            )()
            results["framework_tests"]["path_mapping"] = framework_paths

        # Run framework-specific integration tests
        if hasattr(self.integration_tests, f"test_{framework}_training_workflow"):
            framework_integration = getattr(
                self.integration_tests, f"test_{framework}_training_workflow"
            )()
            results["framework_tests"]["integration"] = framework_integration

        return results

    def run_hyperparameter_optimization_tests(self) -> Dict[str, Any]:
        """
        Run Training hyperparameter optimization tests.

        Returns:
            Dict containing hyperparameter optimization test results
        """
        logger.info("Running Training hyperparameter optimization tests")

        results = {
            "step_type": self.step_type,
            "test_type": "hyperparameter_optimization",
            "builder_type": type(self.builder_instance).__name__,
            "hyperparameter_tests": {},
        }

        # Run hyperparameter handling tests
        if hasattr(self.interface_tests, "test_hyperparameter_handling_methods"):
            hyperparam_interface = (
                self.interface_tests.test_hyperparameter_handling_methods()
            )
            results["hyperparameter_tests"]["interface"] = hyperparam_interface

        # Run hyperparameter specification tests
        if hasattr(
            self.specification_tests, "test_hyperparameter_specification_compliance"
        ):
            hyperparam_spec = (
                self.specification_tests.test_hyperparameter_specification_compliance()
            )
            results["hyperparameter_tests"]["specification"] = hyperparam_spec

        # Run hyperparameter optimization integration
        if hasattr(
            self.integration_tests, "test_hyperparameter_optimization_integration"
        ):
            hyperparam_integration = (
                self.integration_tests.test_hyperparameter_optimization_integration()
            )
            results["hyperparameter_tests"]["integration"] = hyperparam_integration

        return results

    def run_distributed_training_tests(self) -> Dict[str, Any]:
        """
        Run Training distributed training tests.

        Returns:
            Dict containing distributed training test results
        """
        logger.info("Running Training distributed training tests")

        results = {
            "step_type": self.step_type,
            "test_type": "distributed_training",
            "builder_type": type(self.builder_instance).__name__,
            "distributed_tests": {},
        }

        # Run distributed training specification tests
        if hasattr(self.specification_tests, "test_distributed_training_specification"):
            distributed_spec = (
                self.specification_tests.test_distributed_training_specification()
            )
            results["distributed_tests"]["specification"] = distributed_spec

        # Run distributed training integration
        if hasattr(self.integration_tests, "test_distributed_training_integration"):
            distributed_integration = (
                self.integration_tests.test_distributed_training_integration()
            )
            results["distributed_tests"]["integration"] = distributed_integration

        return results

    def run_data_channel_tests(self) -> Dict[str, Any]:
        """
        Run Training data channel validation tests.

        Returns:
            Dict containing data channel test results
        """
        logger.info("Running Training data channel tests")

        results = {
            "step_type": self.step_type,
            "test_type": "data_channels",
            "builder_type": type(self.builder_instance).__name__,
            "data_channel_tests": {},
        }

        # Run data channel specification tests
        if hasattr(self.specification_tests, "test_data_channel_specification"):
            channel_spec = self.specification_tests.test_data_channel_specification()
            results["data_channel_tests"]["specification"] = channel_spec

        # Run data channel path mapping tests
        if hasattr(self.path_mapping_tests, "test_data_channel_mapping_strategies"):
            channel_paths = (
                self.path_mapping_tests.test_data_channel_mapping_strategies()
            )
            results["data_channel_tests"]["path_mapping"] = channel_paths

        # Run data channel integration tests
        if hasattr(self.integration_tests, "test_data_channel_integration"):
            channel_integration = self.integration_tests.test_data_channel_integration()
            results["data_channel_tests"]["integration"] = channel_integration

        return results

    def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run Training performance optimization tests.

        Returns:
            Dict containing performance test results
        """
        logger.info("Running Training performance tests")

        results = {
            "step_type": self.step_type,
            "test_type": "performance",
            "builder_type": type(self.builder_instance).__name__,
            "performance_tests": {},
        }

        # Run performance optimization tests
        if hasattr(self.integration_tests, "test_training_performance_optimization"):
            performance_test = (
                self.integration_tests.test_training_performance_optimization()
            )
            results["performance_tests"]["optimization"] = performance_test

        # Run resource allocation tests
        if hasattr(self.specification_tests, "test_resource_allocation_specification"):
            resource_test = (
                self.specification_tests.test_resource_allocation_specification()
            )
            results["performance_tests"]["resource_allocation"] = resource_test

        return results

    def generate_training_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive Training validation report.

        Args:
            test_results: Results from running Training tests

        Returns:
            Dict containing formatted report
        """
        logger.info("Generating Training validation report")

        report = {
            "report_type": "Training Validation Report",
            "step_type": self.step_type,
            "builder_type": type(self.builder_instance).__name__,
            "timestamp": self._get_timestamp(),
            "summary": self._generate_summary(test_results),
            "detailed_results": test_results,
            "recommendations": self._generate_recommendations(test_results),
            "framework_analysis": self._analyze_framework_compatibility(test_results),
            "training_readiness": self._assess_training_readiness(test_results),
        }

        return report

    def get_training_test_coverage(self) -> Dict[str, Any]:
        """
        Get Training test coverage information.

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
                        "estimator_creation_methods",
                        "framework_specific_methods",
                        "hyperparameter_handling",
                        "training_configuration",
                    ],
                },
                "level_2_specification": {
                    "total_tests": len(
                        self.specification_tests.get_step_type_specific_tests()
                    ),
                    "test_categories": [
                        "framework_configuration",
                        "hyperparameter_specification",
                        "data_channel_specification",
                        "resource_allocation",
                    ],
                },
                "level_3_path_mapping": {
                    "total_tests": len(
                        self.path_mapping_tests.get_step_type_specific_tests()
                    ),
                    "test_categories": [
                        "training_input_paths",
                        "data_channel_mapping",
                        "model_artifact_paths",
                        "training_property_paths",
                    ],
                },
                "level_4_integration": {
                    "total_tests": len(
                        self.integration_tests.get_step_type_specific_tests()
                    ),
                    "test_categories": [
                        "complete_step_creation",
                        "framework_training_workflows",
                        "hyperparameter_optimization",
                        "distributed_training",
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
            "training_patterns": [
                "single_instance_training",
                "distributed_training",
                "hyperparameter_tuning",
                "multi_framework_support",
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
                                if "estimator" in error.lower():
                                    recommendations.append(
                                        "Review estimator configuration and framework compatibility"
                                    )
                                elif "hyperparameter" in error.lower():
                                    recommendations.append(
                                        "Validate hyperparameter specification and tuning configuration"
                                    )
                                elif "data" in error.lower():
                                    recommendations.append(
                                        "Check data channel configuration and input paths"
                                    )
                                elif "training" in error.lower():
                                    recommendations.append(
                                        "Verify training job configuration and resource allocation"
                                    )

        # Add general recommendations
        if not recommendations:
            recommendations.append("All Training validation tests passed successfully")

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

    def _assess_training_readiness(
        self, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess training readiness from test results."""
        readiness = {
            "ready_for_training": True,
            "readiness_score": 100,
            "blocking_issues": [],
            "warnings": [],
        }

        # Analyze test results for training readiness
        if "test_summary" in test_results:
            if test_results["test_summary"]["failed_tests"] > 0:
                readiness["ready_for_training"] = False
                readiness["readiness_score"] = max(
                    0, 100 - (test_results["test_summary"]["failed_tests"] * 10)
                )

        return readiness

    def _get_timestamp(self) -> str:
        """Get current timestamp for reporting."""
        from datetime import datetime

        return datetime.now().isoformat()


# Convenience functions for Training testing


def run_training_validation(
    builder_instance, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run complete Training validation.

    Args:
        builder_instance: Training step builder instance
        config: Optional configuration dictionary

    Returns:
        Dict containing complete validation results
    """
    if config is None:
        config = {}

    test_orchestrator = TrainingStepBuilderTest(builder_instance, config)
    return test_orchestrator.run_all_tests()


def run_training_framework_tests(
    builder_instance, framework: str, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run framework-specific Training tests.

    Args:
        builder_instance: Training step builder instance
        framework: ML framework to test
        config: Optional configuration dictionary

    Returns:
        Dict containing framework-specific test results
    """
    if config is None:
        config = {}

    test_orchestrator = TrainingStepBuilderTest(builder_instance, config)
    return test_orchestrator.run_framework_specific_tests(framework)


def generate_training_report(
    builder_instance, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate Training validation report.

    Args:
        builder_instance: Training step builder instance
        config: Optional configuration dictionary

    Returns:
        Dict containing comprehensive validation report
    """
    if config is None:
        config = {}

    test_orchestrator = TrainingStepBuilderTest(builder_instance, config)
    test_results = test_orchestrator.run_all_tests()
    return test_orchestrator.generate_training_report(test_results)
