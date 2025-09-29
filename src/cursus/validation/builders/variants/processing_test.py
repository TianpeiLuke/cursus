"""
Processing Step Builder Validation Test Suite.

This module provides comprehensive validation for Processing step builders using
a modular 4-level testing approach:

Level 1: Interface Tests - Basic interface and inheritance validation
Level 2: Specification Tests - Specification and contract compliance
Level 3: Path Mapping Tests - Input/output path mapping validation  
Level 4: Integration Tests - End-to-end step creation and system integration

The tests are designed to validate Processing-specific patterns including:
- SKLearnProcessor vs XGBoostProcessor usage
- Pattern A (direct ProcessingStep) vs Pattern B (processor.run + step_args)
- Job type-based specification loading
- Container path mapping from contracts
- Environment variable handling
- Special input patterns (local paths, file uploads)
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock

from .processing_interface_tests import ProcessingInterfaceTests
from .processing_specification_tests import ProcessingSpecificationTests
from .processing_step_creation_tests import ProcessingStepCreationTests
from .processing_integration_tests import ProcessingIntegrationTests
from ..universal_test import UniversalStepBuilderTest


class ProcessingStepBuilderTest(UniversalStepBuilderTest):
    """
    Comprehensive Processing step builder validation test suite.

    This class orchestrates all 4 levels of Processing-specific validation tests
    using the modular test structure. It extends the UniversalStepBuilderTest
    to provide Processing-specific testing capabilities.
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
        Initialize Processing step builder test suite.

        Args:
            builder_class: The Processing step builder class to test
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
        # Set Processing-specific step info
        if step_info is None:
            step_info = {
                "sagemaker_step_type": "Processing",
                "step_category": "processing",
                "processor_types": ["SKLearnProcessor", "XGBoostProcessor"],
                "creation_patterns": [
                    "Pattern A (Direct ProcessingStep)",
                    "Pattern B (processor.run + step_args)",
                ],
                "common_job_types": [
                    "training",
                    "validation",
                    "testing",
                    "calibration",
                ],
                "special_features": [
                    "local_path_override",
                    "file_upload",
                    "s3_path_validation",
                ],
            }

        # Store step_info for processing-specific use
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

        # Initialize Processing-specific test levels
        self._initialize_processing_test_levels()

    def _initialize_processing_test_levels(self) -> None:
        """Initialize Processing-specific test level instances."""
        # Level 1: Processing Interface Tests
        self.level1_tester = ProcessingInterfaceTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Level 2: Processing Specification Tests
        self.level2_tester = ProcessingSpecificationTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Level 3: Processing Step Creation Tests
        self.level3_tester = ProcessingStepCreationTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Level 4: Processing Integration Tests
        self.level4_tester = ProcessingIntegrationTests(
            builder_class=self.builder_class, step_info=self.step_info
        )

        # Override the base test levels with Processing-specific ones
        self.test_levels = {
            1: self.level1_tester,
            2: self.level2_tester,
            3: self.level3_tester,
            4: self.level4_tester,
        }

    def get_processing_specific_info(self) -> Dict[str, Any]:
        """Get Processing-specific information for reporting."""
        return {
            "step_type": "Processing",
            "supported_processors": ["SKLearnProcessor", "XGBoostProcessor"],
            "creation_patterns": {
                "pattern_a": "Direct ProcessingStep creation (SKLearnProcessor)",
                "pattern_b": "processor.run() + step_args (XGBoostProcessor)",
            },
            "job_type_support": "Multi-job-type (training/validation/testing/calibration)",
            "special_features": {
                "local_path_override": "Package step pattern for inference scripts",
                "file_upload": "DummyTraining step pattern for model/config upload",
                "s3_path_validation": "S3 URI normalization and validation",
                "environment_variables": "JSON serialization for complex configurations",
            },
            "container_paths": {
                "input_base": "/opt/ml/processing/input",
                "output_base": "/opt/ml/processing/output",
                "code_base": "/opt/ml/processing/input/code",
            },
        }

    def run_processing_validation(
        self, levels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run Processing-specific validation tests.

        Args:
            levels: Optional list of test levels to run (1-4). If None, runs all levels.

        Returns:
            Dictionary containing test results and Processing-specific information
        """
        if levels is None:
            levels = [1, 2, 3, 4]

        # Run the standard validation
        results = self.run_all_tests()

        # Add Processing-specific information
        if isinstance(results, dict):
            results["processing_info"] = self.get_processing_specific_info()
            results["test_suite"] = "ProcessingStepBuilderTest"

            # Add level-specific summaries
            if "test_results" in results:
                test_results = results["test_results"]

                # Level 1 summary
                if 1 in levels and "level_1" in test_results:
                    level1_results = test_results["level_1"]
                    level1_results["summary"] = (
                        "Processing interface and inheritance validation"
                    )
                    level1_results["focus"] = (
                        "Processor creation methods, framework-specific attributes, step creation patterns"
                    )

                # Level 2 summary
                if 2 in levels and "level_2" in test_results:
                    level2_results = test_results["level_2"]
                    level2_results["summary"] = (
                        "Processing specification and contract compliance"
                    )
                    level2_results["focus"] = (
                        "Job type specification loading, environment variables, processor configuration"
                    )

                # Level 3 summary
                if 3 in levels and "level_3" in test_results:
                    level3_results = test_results["level_3"]
                    level3_results["summary"] = (
                        "Processing path mapping and property path validation"
                    )
                    level3_results["focus"] = (
                        "ProcessingInput/Output creation, container paths, S3 validation, special patterns"
                    )

                # Level 4 summary
                if 4 in levels and "level_4" in test_results:
                    level4_results = test_results["level_4"]
                    level4_results["summary"] = (
                        "Processing integration and end-to-end workflow"
                    )
                    level4_results["focus"] = (
                        "Complete step creation, Pattern A/B validation, dependency resolution"
                    )

        return results



# Convenience function for quick Processing builder validation
def validate_processing_builder(
    builder_class,
    enable_scoring: bool = False,
    enable_structured_reporting: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to validate a Processing step builder.

    Args:
        builder_class: The Processing step builder class to validate
        enable_scoring: Whether to enable scoring functionality
        enable_structured_reporting: Whether to enable structured reporting

    Returns:
        Comprehensive validation results
    """
    tester = ProcessingStepBuilderTest(
        builder_class=builder_class,
        enable_scoring=enable_scoring,
        enable_structured_reporting=enable_structured_reporting,
    )

    return tester.run_processing_validation()


# Convenience function for processor type validation
def validate_processor_type(
    builder_class, expected_processor: str = None
) -> Dict[str, Any]:
    """
    Convenience function to validate processor type for a Processing builder.
    
    Note: This functionality is now integrated into the universal test framework.
    Use ProcessingStepBuilderTest.run_processing_validation() for comprehensive validation.

    Args:
        builder_class: The Processing step builder class to validate
        expected_processor: Expected processor type ('SKLearnProcessor' or 'XGBoostProcessor')

    Returns:
        Processor type validation results (integrated into universal test results)
    """
    tester = ProcessingStepBuilderTest(builder_class=builder_class)
    results = tester.run_processing_validation()
    
    # Extract processor-related information from universal test results
    return {
        "processor_validation": "Integrated into universal test framework",
        "full_results": results,
        "note": "Use run_processing_validation() for complete Processing-specific validation"
    }


# Convenience function for job type support validation
def validate_job_type_support(
    builder_class, job_types: List[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate job type support for a Processing builder.
    
    Note: This functionality is now integrated into the universal test framework.
    Use ProcessingStepBuilderTest.run_processing_validation() for comprehensive validation.

    Args:
        builder_class: The Processing step builder class to validate
        job_types: List of job types to test

    Returns:
        Job type support validation results (integrated into universal test results)
    """
    tester = ProcessingStepBuilderTest(builder_class=builder_class)
    results = tester.run_processing_validation()
    
    # Extract job type-related information from universal test results
    return {
        "job_type_validation": "Integrated into universal test framework",
        "full_results": results,
        "note": "Use run_processing_validation() for complete Processing-specific validation"
    }
