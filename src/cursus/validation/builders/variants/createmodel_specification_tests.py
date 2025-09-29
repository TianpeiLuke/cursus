"""
CreateModel Step Specification Tests (Level 2).

This module provides Level 2 specification validation tests specifically for CreateModel step builders.
These tests focus on CreateModel-specific specification compliance including container configuration,
framework-specific patterns, and model deployment specifications.
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

from ..specification_tests import SpecificationTests


class CreateModelSpecificationTests(SpecificationTests):
    """
    Level 2 CreateModel-specific specification tests.

    Extends the base SpecificationTests with CreateModel-specific specification validation
    including container configuration, framework-specific patterns, and deployment specifications.
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
        Initialize CreateModel specification tests.

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
        """Return CreateModel-specific specification test methods."""
        return [
            "test_container_configuration_specification",
            "test_framework_specific_configuration",
            "test_model_artifact_specification",
            "test_inference_environment_specification",
            "test_deployment_configuration_specification",
        ]

    def test_container_configuration_specification(self) -> None:
        """Test that CreateModel builders handle container configuration specification."""
        self._log("Testing container configuration specification")

        # Check for container configuration handling
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
            f"CreateModel builders should handle container configuration specification, found: {found_indicators}",
        )

        self._assert(True, "Container configuration specification validated")

    def test_framework_specific_configuration(self) -> None:
        """Test that CreateModel builders handle framework-specific configuration."""
        self._log("Testing framework-specific configuration")

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

            # Check for framework-specific configuration methods
            methods = [method for method in dir(self.builder_class) if not method.startswith("__")]
            framework_methods = [
                method for method in methods
                if any(indicator in method.lower() for indicator in framework_indicators[detected_framework])
            ]

            self._assert(
                len(framework_methods) > 0,
                f"Framework-specific configuration methods found for {detected_framework}: {framework_methods}",
            )
        else:
            self._log("No specific framework detected - using generic validation")
            self._assert(True, "Generic CreateModel configuration validated")

        self._assert(True, "Framework-specific configuration validated")

    def test_model_artifact_specification(self) -> None:
        """Test that CreateModel builders handle model artifact specification."""
        self._log("Testing model artifact specification")

        # Check for model artifact handling
        artifact_indicators = [
            "model_data", "model_artifacts", "_validate_model_artifacts",
            "_setup_model_data", "set_model_data", "_configure_model_artifacts"
        ]

        found_indicators = []
        for indicator in artifact_indicators:
            if hasattr(self.builder_class, indicator):
                found_indicators.append(indicator)

        self._assert(
            len(found_indicators) > 0,
            f"CreateModel builders should handle model artifact specification, found: {found_indicators}",
        )

        self._assert(True, "Model artifact specification validated")

    def test_inference_environment_specification(self) -> None:
        """Test that CreateModel builders handle inference environment specification."""
        self._log("Testing inference environment specification")

        # Check for inference environment handling
        env_indicators = [
            "_get_environment_variables", "environment_variables", "env_vars",
            "_setup_inference_env", "_configure_environment", "inference_env"
        ]

        found_indicators = []
        for indicator in env_indicators:
            if hasattr(self.builder_class, indicator):
                found_indicators.append(indicator)

        self._assert(
            len(found_indicators) > 0,
            f"CreateModel builders should handle inference environment specification, found: {found_indicators}",
        )

        self._assert(True, "Inference environment specification validated")

    def test_deployment_configuration_specification(self) -> None:
        """Test that CreateModel builders handle deployment configuration specification."""
        self._log("Testing deployment configuration specification")

        # Check for deployment configuration handling
        deployment_indicators = [
            "prepare_for_registration", "prepare_for_batch_transform", 
            "integrate_with_training_step", "_configure_deployment",
            "deployment_config", "_setup_deployment"
        ]

        found_indicators = []
        for indicator in deployment_indicators:
            if hasattr(self.builder_class, indicator):
                found_indicators.append(indicator)

        self._assert(
            len(found_indicators) > 0,
            f"CreateModel builders should handle deployment configuration specification, found: {found_indicators}",
        )

        self._assert(True, "Deployment configuration specification validated")


# Convenience function for quick CreateModel specification validation
def validate_createmodel_specification(
    builder_class, verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Quick validation function for CreateModel step builder specifications.

    Args:
        builder_class: The CreateModel step builder class to validate
        verbose: Whether to print verbose output

    Returns:
        Dictionary containing test results
    """
    tester = CreateModelSpecificationTests(builder_class, verbose=verbose)
    return tester.run_all_tests()
