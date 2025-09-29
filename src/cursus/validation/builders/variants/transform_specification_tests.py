"""
Transform Step Specification Tests (Level 2).

This module provides Level 2 specification validation tests specifically for Transform step builders.
These tests focus on Transform-specific specification compliance, contract alignment, and
batch processing configuration validation for model inference workflows.
"""

from typing import Dict, Any, List, Optional, Type
import os

from ..specification_tests import SpecificationTests
from ....core.base.builder_base import StepBuilderBase


class TransformSpecificationTests(SpecificationTests):
    """
    Level 2 specification tests specifically for Transform step builders.

    Extends the base SpecificationTests with Transform-specific specification validation
    including batch processing configuration, model integration, and transform job specifications.
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
        Initialize Transform specification tests.

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
        """Return Transform-specific specification test methods."""
        return [
            "test_batch_processing_specification",
            "test_model_integration_specification",
            "test_transform_input_specification",
            "test_transform_output_specification",
            "test_framework_specific_specifications",
        ]

    def test_batch_processing_specification(self) -> None:
        """Test that Transform builders handle batch processing specification."""
        self._log("Testing batch processing specification compliance")

        # Check for batch processing configuration
        batch_indicators = [
            "batch_size", "max_concurrent_transforms", "max_payload_in_mb",
            "_configure_batch_processing", "_get_batch_config"
        ]

        found_indicators = []
        for indicator in batch_indicators:
            if hasattr(self.builder_class, indicator):
                found_indicators.append(indicator)

        self._assert(
            len(found_indicators) > 0,
            f"Transform builders should handle batch processing specification, found: {found_indicators}",
        )

        self._assert(True, "Batch processing specification validated")

    def test_model_integration_specification(self) -> None:
        """Test that Transform builders handle model integration specification."""
        self._log("Testing model integration specification")

        # Check for model integration capabilities
        model_indicators = [
            "model_name", "integrate_with_model_step", "set_model_name",
            "_setup_model_integration", "_configure_model_dependency"
        ]

        found_indicators = []
        for indicator in model_indicators:
            if hasattr(self.builder_class, indicator):
                found_indicators.append(indicator)

        self._assert(
            len(found_indicators) > 0,
            f"Transform builders should handle model integration specification, found: {found_indicators}",
        )

        self._assert(True, "Model integration specification validated")

    def test_transform_input_specification(self) -> None:
        """Test that Transform builders handle transform input specification."""
        self._log("Testing transform input specification compliance")

        # Check for input configuration
        input_indicators = [
            "_get_inputs", "_prepare_transform_input", "_configure_input_data",
            "input_data", "data_source", "content_type"
        ]

        found_indicators = []
        for indicator in input_indicators:
            if hasattr(self.builder_class, indicator):
                found_indicators.append(indicator)

        self._assert(
            len(found_indicators) > 0,
            f"Transform builders should handle input specification, found: {found_indicators}",
        )

        self._assert(True, "Transform input specification validated")

    def test_transform_output_specification(self) -> None:
        """Test that Transform builders handle transform output specification."""
        self._log("Testing transform output specification compliance")

        # Check for output configuration
        output_indicators = [
            "_get_outputs", "_configure_transform_output", "_setup_output_config",
            "output_path", "accept_type", "assemble_with"
        ]

        found_indicators = []
        for indicator in output_indicators:
            if hasattr(self.builder_class, indicator):
                found_indicators.append(indicator)

        self._assert(
            len(found_indicators) > 0,
            f"Transform builders should handle output specification, found: {found_indicators}",
        )

        self._assert(True, "Transform output specification validated")

    def test_framework_specific_specifications(self) -> None:
        """Test that Transform builders handle framework-specific specifications."""
        self._log("Testing framework-specific specifications")

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

            # Check for framework-specific attributes or methods
            methods = [method for method in dir(self.builder_class) if not method.startswith("__")]
            framework_methods = [
                method for method in methods
                if any(indicator in method.lower() for indicator in framework_indicators[detected_framework])
            ]

            self._assert(
                len(framework_methods) > 0,
                f"Framework-specific specifications found for {detected_framework}: {framework_methods}",
            )
        else:
            self._log("No specific framework detected - using generic validation")
            self._assert(True, "Generic transform specification validated")

        self._assert(True, "Framework-specific specifications validated")


# Convenience function for quick Transform specification validation
def validate_transform_specification(
    builder_class: Type[StepBuilderBase], verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Quick validation function for Transform step builder specifications.

    Args:
        builder_class: The Transform step builder class to validate
        verbose: Whether to print verbose output

    Returns:
        Dictionary containing test results
    """
    tester = TransformSpecificationTests(builder_class, verbose=verbose)
    return tester.run_all_tests()
