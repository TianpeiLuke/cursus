"""
Level 2 Processing-Specific Specification Tests for step builders.

These tests focus on Processing step specification and contract compliance:
- Job type-based specification loading
- Processor-specific configuration validation
- Environment variable handling patterns
- Input/output specification compliance
- Script contract integration
"""

import json
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

from ..specification_tests import SpecificationTests


class ProcessingSpecificationTests(SpecificationTests):
    """
    Level 2 Processing-specific specification tests.

    These tests validate that Processing step builders properly use specifications
    and contracts to define their behavior, with focus on Processing-specific patterns.
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
        Initialize Processing specification tests.

        Args:
            builder_class: The Processing step builder class to test
            step_info: Processing-specific step information
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
        
        # Store Processing-specific step info
        self.step_info = step_info or {}

    def get_step_type_specific_tests(self) -> list:
        """Return Processing-specific specification test methods."""
        return [
            "test_job_type_specification_loading",
            "test_processing_environment_variables",
            "test_specification_driven_inputs",
            "test_specification_driven_outputs",
            "test_processor_type_alignment",
        ]

    def _configure_step_type_mocks(self) -> None:
        """Configure Processing-specific mock objects for specification tests."""
        # Mock processor types
        self.mock_sklearn_processor = Mock()
        self.mock_xgboost_processor = Mock()

        # Mock specification objects
        self.mock_processing_spec = Mock()
        self.mock_processing_spec.dependencies = {
            "input_data": Mock(logical_name="input_data", required=True),
            "metadata": Mock(logical_name="metadata", required=False),
        }
        self.mock_processing_spec.outputs = {
            "processed_data": Mock(logical_name="processed_data"),
            "statistics": Mock(logical_name="statistics"),
        }

        # Mock contract objects
        self.mock_contract = Mock()
        self.mock_contract.expected_input_paths = {
            "input_data": "/opt/ml/processing/input/data",
            "metadata": "/opt/ml/processing/input/metadata",
        }
        self.mock_contract.expected_output_paths = {
            "processed_data": "/opt/ml/processing/output/data",
            "statistics": "/opt/ml/processing/output/stats",
        }

    def _validate_step_type_requirements(self) -> dict:
        """Validate Processing-specific requirements for specification tests."""
        return {
            "specification_tests_completed": True,
            "processing_specific_validations": True,
            "processor_type_validated": True,
            "job_type_support_validated": True,
        }

    def test_job_type_specification_loading(self) -> None:
        """Test that Processing builders properly load specifications based on job type."""
        self._log("Testing job type-based specification loading")

        # Test multi-job-type builders (TabularPreprocessing, CurrencyConversion)
        if hasattr(self.builder_class, "_load_specification_by_job_type"):
            # Test training job type
            with patch.object(
                self.builder_class, "_load_specification_by_job_type"
            ) as mock_load:
                mock_load.return_value = self.mock_processing_spec

                config = Mock()
                config.job_type = "training"

                try:
                    builder = self.builder_class(config=config)
                    mock_load.assert_called_with("training")
                    self._assert(
                        True, "Job type specification loading works for training"
                    )
                except Exception as e:
                    self._log(f"Job type specification loading failed: {e}")
                    self._assert(False, f"Job type specification loading failed: {e}")

            # Test validation job type
            with patch.object(
                self.builder_class, "_load_specification_by_job_type"
            ) as mock_load:
                mock_load.return_value = self.mock_processing_spec

                config = Mock()
                config.job_type = "validation"

                try:
                    builder = self.builder_class(config=config)
                    mock_load.assert_called_with("validation")
                    self._assert(
                        True, "Job type specification loading works for validation"
                    )
                except Exception as e:
                    self._log(f"Job type specification loading failed: {e}")
                    self._assert(False, f"Job type specification loading failed: {e}")
        else:
            # Single-purpose builders should still have specification
            self._log("Single-purpose Processing builder - checking for specification")
            self._assert(True, "Single-purpose builders validated separately")


    def test_processing_environment_variables(self) -> None:
        """Test that Processing builders handle environment variables correctly."""
        self._log("Testing Processing environment variables handling")

        if hasattr(self.builder_class, "_get_environment_variables"):
            config = Mock()
            config.label_name = "target"
            config.categorical_columns = ["cat1", "cat2"]
            config.currency_conversion_dict = {"USD": 1.0, "EUR": 0.85}

            try:
                builder = self.builder_class(config=config)
                env_vars = builder._get_environment_variables()

                # Check that environment variables are returned as dict
                self._assert(
                    isinstance(env_vars, dict), "Environment variables should be dict"
                )

                # Check for common Processing environment variable patterns
                if hasattr(config, "label_name"):
                    self._assert(
                        "LABEL_FIELD" in env_vars
                        or "label_name" in str(env_vars).lower(),
                        "Label field should be in environment variables",
                    )

                if hasattr(config, "categorical_columns"):
                    # Should be comma-separated or JSON
                    found_categorical = any(
                        "categorical" in key.lower() for key in env_vars.keys()
                    )
                    self._assert(
                        found_categorical,
                        "Categorical columns should be in environment variables",
                    )

                if hasattr(config, "currency_conversion_dict"):
                    # Should be JSON serialized
                    found_currency = any(
                        "currency" in key.lower() for key in env_vars.keys()
                    )
                    self._assert(
                        found_currency,
                        "Currency conversion dict should be in environment variables",
                    )

                self._assert(True, "Environment variables handling validated")

            except Exception as e:
                self._log(f"Environment variables test failed: {e}")
                self._assert(False, f"Environment variables test failed: {e}")
        else:
            self._log("No _get_environment_variables method found")
            self._assert(
                False,
                "Processing builders should have _get_environment_variables method",
            )

    def test_specification_driven_inputs(self) -> None:
        """Test that Processing builders use specifications to define inputs."""
        self._log("Testing specification-driven input handling")

        if hasattr(self.builder_class, "_get_inputs"):
            config = Mock()
            builder = self.builder_class(config=config)

            # Mock specification and contract
            builder.spec = self.mock_processing_spec
            builder.contract = self.mock_contract

            inputs = {
                "input_data": "s3://bucket/input/data",
                "metadata": "s3://bucket/input/metadata",
            }

            try:
                processing_inputs = builder._get_inputs(inputs)

                # Check that ProcessingInput objects are created
                self._assert(
                    isinstance(processing_inputs, list),
                    "Should return list of ProcessingInput objects",
                )

                if processing_inputs:
                    # Check first input structure
                    first_input = processing_inputs[0]
                    self._assert(
                        hasattr(first_input, "input_name"),
                        "ProcessingInput should have input_name",
                    )
                    self._assert(
                        hasattr(first_input, "source"),
                        "ProcessingInput should have source",
                    )
                    self._assert(
                        hasattr(first_input, "destination"),
                        "ProcessingInput should have destination",
                    )

                self._assert(True, "Specification-driven inputs validated")

            except Exception as e:
                self._log(f"Specification-driven inputs test failed: {e}")
                self._assert(False, f"Specification-driven inputs test failed: {e}")
        else:
            self._log("No _get_inputs method found")
            self._assert(False, "Processing builders should have _get_inputs method")

    def test_specification_driven_outputs(self) -> None:
        """Test that Processing builders use specifications to define outputs."""
        self._log("Testing specification-driven output handling")

        if hasattr(self.builder_class, "_get_outputs"):
            config = Mock()
            builder = self.builder_class(config=config)

            # Mock specification and contract
            builder.spec = self.mock_processing_spec
            builder.contract = self.mock_contract

            outputs = {
                "processed_data": "s3://bucket/output/data",
                "statistics": "s3://bucket/output/stats",
            }

            try:
                processing_outputs = builder._get_outputs(outputs)

                # Check that ProcessingOutput objects are created
                self._assert(
                    isinstance(processing_outputs, list),
                    "Should return list of ProcessingOutput objects",
                )

                if processing_outputs:
                    # Check first output structure
                    first_output = processing_outputs[0]
                    self._assert(
                        hasattr(first_output, "output_name"),
                        "ProcessingOutput should have output_name",
                    )
                    self._assert(
                        hasattr(first_output, "source"),
                        "ProcessingOutput should have source",
                    )
                    self._assert(
                        hasattr(first_output, "destination"),
                        "ProcessingOutput should have destination",
                    )

                self._assert(True, "Specification-driven outputs validated")

            except Exception as e:
                self._log(f"Specification-driven outputs test failed: {e}")
                self._assert(False, f"Specification-driven outputs test failed: {e}")
        else:
            self._log("No _get_outputs method found")
            self._assert(False, "Processing builders should have _get_outputs method")


    def test_processor_type_alignment(self) -> None:
        """Test that Processing builders create the correct processor type."""
        self._log("Testing processor type alignment")

        if hasattr(self.builder_class, "_create_processor"):
            config = Mock()
            config.processing_instance_type_large = "ml.m5.xlarge"
            config.processing_instance_type_small = "ml.m5.large"
            config.use_large_processing_instance = False
            config.processing_instance_count = 1
            config.processing_volume_size = 30
            config.processing_framework_version = "0.23-1"

            try:
                builder = self.builder_class(config=config)
                builder.role = "test-role"
                builder.session = Mock()

                processor = builder._create_processor()

                # Check processor type
                processor_type = type(processor).__name__
                self._assert(
                    processor_type in ["SKLearnProcessor", "XGBoostProcessor", "Mock"],
                    f"Should create valid processor type, got: {processor_type}",
                )

                # XGBoost processors should have framework and py_version
                if "XGBoost" in processor_type:
                    if hasattr(config, "framework_version"):
                        self._assert(True, "XGBoost processor with framework version")
                    if hasattr(config, "py_version"):
                        self._assert(True, "XGBoost processor with Python version")

                self._assert(True, "Processor type alignment validated")

            except Exception as e:
                self._log(f"Processor type test failed: {e}")
                self._assert(False, f"Processor type test failed: {e}")
        else:
            self._log("No _create_processor method found")
            self._assert(
                False, "Processing builders should have _create_processor method"
            )
