"""
CreateModel Step Interface Tests (Level 1).

This module provides Level 1 interface validation tests specifically for CreateModel step builders.
These tests focus on CreateModel-specific interface patterns including model creation methods,
framework-specific configuration, and container image setup.
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

from ..interface_tests import InterfaceTests


class CreateModelInterfaceTests(InterfaceTests):
    """
    Level 1 CreateModel-specific interface tests.

    Extends the base InterfaceTests with CreateModel-specific interface validation
    including model creation methods, framework-specific patterns, and container configuration.
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
        Initialize CreateModel interface tests.

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
        """Return CreateModel-specific interface test methods."""
        return [
            "test_model_creation_method",
            "test_framework_specific_methods",
            "test_container_image_configuration",
            "test_model_integration_methods",
        ]

    def test_model_creation_method(self) -> None:
        """Test that CreateModel builders implement model creation method."""
        self._log("Testing model creation method")

        # Check for _create_model method
        self._assert(
            hasattr(self.builder_class, "_create_model"),
            "CreateModel builders should have _create_model method",
        )

        self._assert(True, "Model creation method validated")

    def test_framework_specific_methods(self) -> None:
        """Test that CreateModel builders implement framework-specific methods."""
        self._log("Testing framework-specific methods")

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

            # Check for framework-specific methods
            methods = [method for method in dir(self.builder_class) if not method.startswith("__")]
            framework_methods = [
                method for method in methods
                if any(indicator in method.lower() for indicator in framework_indicators[detected_framework])
            ]

            self._assert(
                len(framework_methods) > 0,
                f"Framework-specific methods found for {detected_framework}: {framework_methods}",
            )
        else:
            self._log("No specific framework detected - using generic validation")
            self._assert(True, "Generic CreateModel builder validated")

        self._assert(True, "Framework-specific methods validated")

    def test_container_image_configuration(self) -> None:
        """Test that CreateModel builders configure container images correctly."""
        self._log("Testing container image configuration")

        # Check for image configuration methods or attributes
        image_indicators = [
            "image_uri", "container_image", "_get_image_uri", "_configure_image",
            "get_image_uri", "container_def"
        ]

        found_indicators = []
        for indicator in image_indicators:
            if hasattr(self.builder_class, indicator):
                found_indicators.append(indicator)

        self._assert(
            len(found_indicators) > 0,
            f"CreateModel builders should handle container image configuration, found: {found_indicators}",
        )

        self._assert(True, "Container image configuration validated")

    def test_model_integration_methods(self) -> None:
        """Test that CreateModel builders implement model integration methods."""
        self._log("Testing model integration methods")

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

        self._assert(True, "Model integration methods validated")


# Convenience function for quick CreateModel interface validation
def validate_createmodel_interface(
    builder_class, verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Quick validation function for CreateModel step builder interfaces.

    Args:
        builder_class: The CreateModel step builder class to validate
        verbose: Whether to print verbose output

    Returns:
        Dictionary containing test results
    """
    tester = CreateModelInterfaceTests(builder_class, verbose=verbose)
    return tester.run_all_tests()
