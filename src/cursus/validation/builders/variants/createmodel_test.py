"""
CreateModel Step Builder Validation Test Suite.

This module provides comprehensive validation for CreateModel step builders using
a modular 4-level testing approach:

Level 1: Interface Tests - Basic interface and inheritance validation
Level 2: Specification Tests - Specification and contract compliance
Level 3: Step Creation Tests - Step creation and configuration validation  
Level 4: Integration Tests - End-to-end step creation and system integration

The tests are designed to validate CreateModel-specific patterns including:
- Model artifact handling and deployment
- Container configuration and optimization
- Framework-specific deployment patterns
- Model integration workflows
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock
import logging

from .createmodel_interface_tests import CreateModelInterfaceTests
from .createmodel_specification_tests import CreateModelSpecificationTests
from .createmodel_integration_tests import CreateModelIntegrationTests
from ..universal_test import UniversalStepBuilderTest

logger = logging.getLogger(__name__)


class CreateModelStepBuilderTest(UniversalStepBuilderTest):
    """
    Comprehensive CreateModel step builder validation test suite.

    This class orchestrates all 4 levels of CreateModel-specific validation tests
    using the modular test structure. It extends the UniversalStepBuilderTest
    to provide CreateModel-specific testing capabilities.
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
        Initialize CreateModel step builder test suite.

        Args:
            builder_class: The CreateModel step builder class to test
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
        # Set CreateModel-specific step info
        if step_info is None:
            step_info = {
                "sagemaker_step_type": "CreateModel",
                "step_category": "deployment",
                "supported_frameworks": ["pytorch", "xgboost", "tensorflow", "sklearn"],
                "deployment_patterns": [
                    "Single Container Deployment",
                    "Multi-Container Deployment",
                    "Model Registry Integration",
                    "Endpoint Deployment",
                ],
                "common_features": [
                    "model_artifact_handling",
                    "container_configuration",
                    "inference_preparation",
                    "deployment_optimization",
                ],
                "special_features": [
                    "multi_container_support",
                    "model_registry_integration",
                    "inference_optimization",
                    "production_deployment",
                ],
            }

        # Store step_info for createmodel-specific use
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

        # Initialize CreateModel-specific test levels
        self._initialize_createmodel_test_levels()

    def _initialize_createmodel_test_levels(self) -> None:
        """Initialize CreateModel-specific test level instances."""
        # Level 1: CreateModel Interface Tests
        self.level1_tester = CreateModelInterfaceTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Level 2: CreateModel Specification Tests
        self.level2_tester = CreateModelSpecificationTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Level 3: CreateModel Step Creation Tests (using base step creation tests)
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

        # Level 4: CreateModel Integration Tests
        self.level4_tester = CreateModelIntegrationTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Override the base test levels with CreateModel-specific ones
        self.test_levels = {
            1: self.level1_tester,
            2: self.level2_tester,
            3: self.level3_tester,
            4: self.level4_tester,
        }

    def get_createmodel_specific_info(self) -> Dict[str, Any]:
        """Get CreateModel-specific information for reporting."""
        return {
            "step_type": "CreateModel",
            "supported_frameworks": ["pytorch", "xgboost", "tensorflow", "sklearn"],
            "deployment_patterns": {
                "single_container": "Single container model deployment",
                "multi_container": "Multi-container model deployment",
                "model_registry": "Model registry integration deployment",
                "endpoint_deployment": "Real-time inference endpoint deployment",
            },
            "common_features": {
                "model_artifact_handling": "Model artifact management and validation",
                "container_configuration": "Container image and environment setup",
                "inference_preparation": "Inference code and environment preparation",
                "deployment_optimization": "Performance and resource optimization",
            },
            "special_features": {
                "multi_container_support": "Support for multi-container deployments",
                "model_registry_integration": "Integration with SageMaker Model Registry",
                "inference_optimization": "Inference performance optimization",
                "production_deployment": "Production-ready deployment configuration",
            },
        }

    def run_createmodel_validation(
        self, levels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run CreateModel-specific validation tests.

        Args:
            levels: Optional list of test levels to run (1-4). If None, runs all levels.

        Returns:
            Dictionary containing test results and CreateModel-specific information
        """
        if levels is None:
            levels = [1, 2, 3, 4]

        # Run the standard validation
        results = self.run_all_tests()

        # Add CreateModel-specific information
        if isinstance(results, dict):
            results["createmodel_info"] = self.get_createmodel_specific_info()
            results["test_suite"] = "CreateModelStepBuilderTest"

            # Add level-specific summaries
            if "test_results" in results:
                test_results = results["test_results"]

                # Level 1 summary
                if 1 in levels and "level_1" in test_results:
                    level1_results = test_results["level_1"]
                    level1_results["summary"] = (
                        "CreateModel interface and inheritance validation"
                    )
                    level1_results["focus"] = (
                        "Model creation methods, container configuration, framework-specific patterns"
                    )

                # Level 2 summary
                if 2 in levels and "level_2" in test_results:
                    level2_results = test_results["level_2"]
                    level2_results["summary"] = (
                        "CreateModel specification and contract compliance"
                    )
                    level2_results["focus"] = (
                        "Container specification, framework configuration, deployment specifications"
                    )

                # Level 3 summary
                if 3 in levels and "level_3" in test_results:
                    level3_results = test_results["level_3"]
                    level3_results["summary"] = (
                        "CreateModel step creation and configuration validation"
                    )
                    level3_results["focus"] = (
                        "CreateModelStep creation, model artifact integration, container setup"
                    )

                # Level 4 summary
                if 4 in levels and "level_4" in test_results:
                    level4_results = test_results["level_4"]
                    level4_results["summary"] = (
                        "CreateModel integration and end-to-end deployment workflow"
                    )
                    level4_results["focus"] = (
                        "Complete deployment workflow, framework-specific deployment, model integration"
                    )

        return results


# Convenience functions for CreateModel testing


def run_createmodel_validation(
    builder_instance, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run complete CreateModel validation.

    Args:
        builder_instance: CreateModel step builder instance
        config: Optional configuration dictionary

    Returns:
        Dict containing complete validation results
    """
    if config is None:
        config = {}

    test_orchestrator = CreateModelStepBuilderTest(builder_instance, config)
    return test_orchestrator.run_all_tests()


def run_createmodel_framework_tests(
    builder_instance, framework: str, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run framework-specific CreateModel tests.

    Args:
        builder_instance: CreateModel step builder instance
        framework: ML framework to test
        config: Optional configuration dictionary

    Returns:
        Dict containing framework-specific test results
    """
    if config is None:
        config = {}

    test_orchestrator = CreateModelStepBuilderTest(builder_instance, config)
    return test_orchestrator.run_createmodel_validation()


def generate_createmodel_report(
    builder_instance, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate CreateModel validation report.

    Args:
        builder_instance: CreateModel step builder instance
        config: Optional configuration dictionary

    Returns:
        Dict containing comprehensive validation report
    """
    if config is None:
        config = {}

    test_orchestrator = CreateModelStepBuilderTest(builder_instance, config)
    test_results = test_orchestrator.run_all_tests()
    return test_results
