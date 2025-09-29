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
