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
    
    Note: Framework-specific functionality is now integrated into the universal test framework.
    Use TrainingStepBuilderTest.run_training_validation() for comprehensive validation.

    Args:
        builder_instance: Training step builder instance
        framework: ML framework to test
        config: Optional configuration dictionary

    Returns:
        Dict containing framework-specific test results (integrated into universal test results)
    """
    if config is None:
        config = {}

    test_orchestrator = TrainingStepBuilderTest(builder_instance, config)
    results = test_orchestrator.run_training_validation()
    
    # Extract framework-related information from universal test results
    return {
        "framework_validation": "Integrated into universal test framework",
        "framework": framework,
        "full_results": results,
        "note": "Use run_training_validation() for complete Training-specific validation"
    }


def generate_training_report(
    builder_instance, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate Training validation report.
    
    Note: Reporting functionality is now integrated into the universal test framework.
    Use TrainingStepBuilderTest.run_training_validation() for comprehensive reporting.

    Args:
        builder_instance: Training step builder instance
        config: Optional configuration dictionary

    Returns:
        Dict containing comprehensive validation report (integrated into universal test results)
    """
    if config is None:
        config = {}

    test_orchestrator = TrainingStepBuilderTest(builder_instance, config)
    results = test_orchestrator.run_training_validation()
    
    # Extract reporting information from universal test results
    return {
        "training_report": "Integrated into universal test framework",
        "full_results": results,
        "note": "Use run_training_validation() for complete Training-specific validation and reporting"
    }
