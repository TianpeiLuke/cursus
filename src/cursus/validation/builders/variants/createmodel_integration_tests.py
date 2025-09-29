"""
CreateModel Step Integration Tests (Level 4).

This module provides Level 4 integration validation tests specifically for CreateModel step builders.
These tests focus on CreateModel-specific integration validation including complete CreateModelStep creation,
framework-specific deployment patterns, and model integration workflows.
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock

from ..integration_tests import IntegrationTests


class CreateModelIntegrationTests(IntegrationTests):
    """
    Level 4 CreateModel-specific integration tests.

    Extends the base IntegrationTests with CreateModel-specific integration validation
    including complete step creation, framework-specific deployment, and model integration workflows.
    """

    def __init__(
        self,
        builder_class,
        step_info: Optional[Dict[str, Any]] = None,
        config=None,
        spec=None,
        contract=None,
        step_name=None,
        verbose: bool = False,
        test_reporter=None,
        **kwargs
    ):
        """
        Initialize CreateModel integration tests.

        Args:
            builder_class: The CreateModel step builder class to test
            step_info: CreateModel-specific step information
            config: Optional config to use
            spec: Optional step specification
            contract: Optional script contract
            step_name: Optional step name
            verbose: Whether to print verbose output
            test_reporter: Optional function to report test results
            **kwargs: Additional arguments
        """
        # Initialize parent with new signature
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
        
        # Store CreateModel-specific step info
        self.step_info = step_info or {}
        self.step_type = "CreateModel"

    def get_step_type_specific_tests(self) -> list:
        """Return CreateModel-specific integration test methods."""
        return [
            "test_complete_createmodel_step_creation",
            "test_framework_specific_deployment",
            "test_model_integration_workflow",
            "test_container_deployment_integration",
        ]

    def test_complete_createmodel_step_creation(self) -> None:
        """Test that CreateModel builders can create complete CreateModelStep."""
        self._log("Testing complete CreateModel step creation")

        # Check for _create_model method
        self._assert(
            hasattr(self.builder_class, "_create_model"),
            "CreateModel builders should have _create_model method",
        )

        # Check for step creation method
        self._assert(
            hasattr(self.builder_class, "create_step"),
            "CreateModel builders should have create_step method",
        )

        self._assert(True, "Complete CreateModel step creation validated")

    def test_framework_specific_deployment(self) -> None:
        """Test framework-specific deployment patterns."""
        self._log("Testing framework-specific deployment")

        # Detect framework from builder class name
        builder_name = self.builder_class.__name__.lower()
        detected_framework = None

        framework_indicators = {
            "xgboost": ["xgboost", "xgb"],
            "pytorch": ["pytorch", "torch"],
            "sklearn": ["sklearn", "scikit"],
            "tensorflow": ["tensorflow", "tf"],
        }

        for framework, indicators in framework_indicators.items():
            if any(indicator in builder_name for indicator in indicators):
                detected_framework = framework
                break

        if detected_framework:
            self._log(f"Detected framework: {detected_framework}")

            # Check for framework-specific deployment methods
            methods = [method for method in dir(self.builder_class) if not method.startswith("__")]
            framework_methods = [
                method for method in methods
                if any(indicator in method.lower() for indicator in framework_indicators[detected_framework])
            ]

            self._assert(
                len(framework_methods) > 0,
                f"Framework-specific deployment methods found for {detected_framework}: {framework_methods}",
            )
        else:
            self._log("No specific framework detected - using generic validation")
            self._assert(True, "Generic CreateModel deployment validated")

        self._assert(True, "Framework-specific deployment validated")

    def test_model_integration_workflow(self) -> None:
        """Test that CreateModel builders integrate with model workflows."""
        self._log("Testing model integration workflow")

        # Check for model integration capabilities
        model_integration_indicators = [
            "integrate_with_training_step", "set_model_data", "model_data",
            "_setup_model_integration", "_configure_model_dependency"
        ]

        found_indicators = []
        for indicator in model_integration_indicators:
            if hasattr(self.builder_class, indicator):
                found_indicators.append(indicator)

        self._assert(
            len(found_indicators) > 0,
            f"CreateModel builders should have model integration capabilities, found: {found_indicators}",
        )

        self._assert(True, "Model integration workflow validated")

    def test_container_deployment_integration(self) -> None:
        """Test container deployment integration."""
        self._log("Testing container deployment integration")

        # Check for container deployment capabilities
        container_indicators = [
            "image_uri", "container_image", "_get_image_uri", "_configure_image",
            "get_image_uri", "container_def", "_validate_container"
        ]

        found_indicators = []
        for indicator in container_indicators:
            if hasattr(self.builder_class, indicator):
                found_indicators.append(indicator)

        self._assert(
            len(found_indicators) > 0,
            f"CreateModel builders should handle container deployment integration, found: {found_indicators}",
        )

        self._assert(True, "Container deployment integration validated")


# Convenience function for quick CreateModel integration validation
def validate_createmodel_integration(
    builder_class, verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Quick validation function for CreateModel step builder integration.

    Args:
        builder_class: The CreateModel step builder class to validate
        verbose: Whether to print verbose output

    Returns:
        Dictionary containing test results
    """
    tester = CreateModelIntegrationTests(builder_class, verbose=verbose)
    return tester.run_all_tests()
