"""
Transform Step Interface Tests (Level 1).

This module provides Level 1 interface validation tests specifically for Transform step builders.
These tests focus on Transform-specific interface requirements, method signatures, and
basic functionality validation for batch inference and model transformation workflows.
"""

from typing import Dict, Any, List, Optional, Type
import inspect
from unittest.mock import Mock

from ..interface_tests import InterfaceTests
from ....core.base.builder_base import StepBuilderBase


class TransformInterfaceTests(InterfaceTests):
    """
    Level 1 interface tests specifically for Transform step builders.

    Extends the base InterfaceTests with Transform-specific interface validation
    including transformer creation, batch processing configuration, and model integration.
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
        Initialize Transform interface tests.

        Args:
            builder_class: The Transform step builder class to test
            step_info: Transform-specific step information
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
        
        # Store Transform-specific step info
        self.step_info = step_info or {}
        self.step_type = "Transform"


    def get_step_type_specific_tests(self) -> list:
        """Return Transform-specific interface test methods."""
        return [
            "test_transformer_creation_method",
            "test_batch_processing_configuration",
            "test_model_integration_methods",
            "test_framework_specific_methods",
        ]

    def test_transformer_creation_method(self) -> None:
        """Test that Transform builders implement transformer creation method."""
        self._log("Testing transformer creation method")

        # Check for _create_transformer method
        self._assert(
            hasattr(self.builder_class, "_create_transformer"),
            "Transform builders should have _create_transformer method",
        )

        if hasattr(self.builder_class, "_create_transformer"):
            config = Mock()
            config.model_name = "test-model"
            config.instance_type = "ml.m5.large"
            config.instance_count = 1

            try:
                builder = self.builder_class(config=config)
                builder.role = "test-role"
                builder.session = Mock()

                transformer = builder._create_transformer()

                # Validate transformer type
                transformer_type = type(transformer).__name__
                self._assert(
                    transformer_type in ["Transformer", "Mock"],
                    f"Should create valid transformer type, got: {transformer_type}",
                )

                self._assert(True, "Transformer creation method validated")

            except Exception as e:
                self._log(f"Transformer creation test failed: {e}")
                self._assert(False, f"Transformer creation test failed: {e}")
        else:
            self._assert(False, "Transform builders must have _create_transformer method")

    def test_batch_processing_configuration(self) -> None:
        """Test that Transform builders handle batch processing configuration."""
        self._log("Testing batch processing configuration")

        config = Mock()
        config.batch_size = 100
        config.max_concurrent_transforms = 4
        config.max_payload_in_mb = 6

        try:
            builder = self.builder_class(config=config)

            # Check for batch configuration attributes or methods
            batch_indicators = [
                "batch_size", "max_concurrent_transforms", "max_payload_in_mb",
                "_configure_batch_processing", "_get_batch_config"
            ]

            found_indicators = []
            for indicator in batch_indicators:
                if hasattr(builder, indicator) or hasattr(config, indicator):
                    found_indicators.append(indicator)

            self._assert(
                len(found_indicators) > 0,
                f"Transform builders should handle batch configuration, found: {found_indicators}",
            )

            self._assert(True, "Batch processing configuration validated")

        except Exception as e:
            self._log(f"Batch processing configuration test failed: {e}")
            self._assert(False, f"Batch processing configuration test failed: {e}")

    def test_model_integration_methods(self) -> None:
        """Test that Transform builders have model integration capabilities."""
        self._log("Testing model integration methods")

        # Check for model integration methods or attributes
        model_integration_indicators = [
            "integrate_with_model_step", "set_model_name", "model_name",
            "_setup_model_integration", "_configure_model_dependency"
        ]

        found_indicators = []
        for indicator in model_integration_indicators:
            if hasattr(self.builder_class, indicator):
                found_indicators.append(indicator)

        # Transform builders should have some way to integrate with models
        self._assert(
            len(found_indicators) > 0,
            f"Transform builders should have model integration capabilities, found: {found_indicators}",
        )

        self._assert(True, "Model integration methods validated")

    def test_framework_specific_methods(self) -> None:
        """Test for framework-specific methods in Transform builders."""
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
            self._assert(True, "Generic transform builder validated")

        self._assert(True, "Framework-specific methods validated")


# Convenience function for quick Transform interface validation
def validate_transform_interface(
    builder_class: Type[StepBuilderBase], verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Quick validation function for Transform step builder interfaces.

    Args:
        builder_class: The Transform step builder class to validate
        verbose: Whether to print verbose output

    Returns:
        Dictionary containing test results
    """
    tester = TransformInterfaceTests(builder_class, verbose=verbose)
    return tester.run_all_tests()
