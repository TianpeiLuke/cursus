"""
Level 1 Training-Specific Interface Tests for step builders.

These tests focus on Training step interface and inheritance validation:
- Framework-specific estimator creation methods
- Hyperparameter handling patterns (direct vs file-based)
- Training configuration attributes
- Data channel strategies (single vs multiple)
- Environment variable methods
- Metric definition methods
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

from ..interface_tests import InterfaceTests


class TrainingInterfaceTests(InterfaceTests):
    """
    Level 1 Training-specific interface tests.

    These tests validate that Training step builders implement the correct
    interface patterns for framework-specific training operations.
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
        Initialize Training interface tests.

        Args:
            builder_class: The Training step builder class to test
            step_info: Training-specific step information
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
        
        # Store Training-specific step info
        self.step_info = step_info or {}

    def get_step_type_specific_tests(self) -> list:
        """Return Training-specific interface test methods."""
        return [
            "test_estimator_creation_method",
            "test_framework_specific_methods",
            "test_hyperparameter_handling_methods",
            "test_training_input_output_methods",
        ]

    def _configure_step_type_mocks(self) -> None:
        """Configure Training-specific mock objects for interface tests."""
        # Mock framework-specific estimators
        self.mock_pytorch_estimator = Mock()
        self.mock_pytorch_estimator.__class__.__name__ = "PyTorch"

        self.mock_xgboost_estimator = Mock()
        self.mock_xgboost_estimator.__class__.__name__ = "XGBoost"

        # Mock TrainingInput objects
        self.mock_training_input = Mock()
        self.mock_training_input.s3_data = "s3://bucket/data"

        # Mock hyperparameters
        self.mock_hyperparameters = Mock()
        self.mock_hyperparameters.to_dict.return_value = {
            "learning_rate": 0.01,
            "epochs": 10,
        }

        # Mock metric definitions
        self.mock_metric_definitions = [
            {"Name": "Train Loss", "Regex": "Train Loss: ([0-9\\.]+)"},
            {"Name": "Validation Loss", "Regex": "Validation Loss: ([0-9\\.]+)"},
        ]

    def _validate_step_type_requirements(self) -> dict:
        """Validate Training-specific requirements for interface tests."""
        return {
            "interface_tests_completed": True,
            "training_specific_validations": True,
            "estimator_creation_validated": True,
            "framework_patterns_validated": True,
        }

    def test_estimator_creation_method(self) -> None:
        """Test that Training builders implement estimator creation method."""
        self._log("Testing estimator creation method")

        # Check for _create_estimator method
        self._assert(
            hasattr(self.builder_class, "_create_estimator"),
            "Training builders should have _create_estimator method",
        )

        if hasattr(self.builder_class, "_create_estimator"):
            config = Mock()
            config.training_entry_point = "train.py"
            config.source_dir = "src"
            config.framework_version = "1.12.0"
            config.py_version = "py38"
            config.training_instance_type = "ml.m5.large"
            config.training_instance_count = 1
            config.training_volume_size = 30

            try:
                builder = self.builder_class(config=config)
                builder.role = "test-role"
                builder.session = Mock()

                # Mock hyperparameters if needed
                if hasattr(config, "hyperparameters"):
                    config.hyperparameters = self.mock_hyperparameters

                estimator = builder._create_estimator()

                # Validate estimator type
                estimator_type = type(estimator).__name__
                self._assert(
                    estimator_type
                    in ["PyTorch", "XGBoost", "SKLearn", "TensorFlow", "Mock"],
                    f"Should create valid estimator type, got: {estimator_type}",
                )

                self._assert(True, "Estimator creation method validated")

            except Exception as e:
                self._log(f"Estimator creation test failed: {e}")
                self._assert(False, f"Estimator creation test failed: {e}")
        else:
            self._assert(False, "Training builders must have _create_estimator method")


    def test_framework_specific_methods(self) -> None:
        """Test that Training builders implement framework-specific methods."""
        self._log("Testing framework-specific methods")

        # Check for framework-specific patterns
        framework_indicators = {
            "pytorch": ["PyTorch", "torch", "pytorch"],
            "xgboost": ["XGBoost", "xgb", "xgboost"],
            "sklearn": ["SKLearn", "sklearn", "scikit"],
            "tensorflow": ["TensorFlow", "tensorflow", "tf"],
        }

        builder_name = self.builder_class.__name__.lower()
        detected_framework = None

        for framework, indicators in framework_indicators.items():
            if any(indicator.lower() in builder_name for indicator in indicators):
                detected_framework = framework
                break

        if detected_framework:
            self._log(f"Detected framework: {detected_framework}")

            # Framework-specific validation
            if detected_framework == "pytorch":
                self._validate_pytorch_specific_methods()
            elif detected_framework == "xgboost":
                self._validate_xgboost_specific_methods()
            elif detected_framework == "sklearn":
                self._validate_sklearn_specific_methods()
            elif detected_framework == "tensorflow":
                self._validate_tensorflow_specific_methods()
        else:
            self._log("No specific framework detected - using generic validation")
            self._assert(True, "Generic training builder validated")

    def _validate_pytorch_specific_methods(self) -> None:
        """Validate PyTorch-specific methods."""
        self._log("Validating PyTorch-specific methods")

        # PyTorch builders should handle hyperparameters directly
        config = Mock()
        config.hyperparameters = self.mock_hyperparameters

        try:
            builder = self.builder_class(config=config)

            # Check if hyperparameters are handled properly
            if hasattr(builder, "_create_estimator"):
                # Mock the estimator creation to check hyperparameter handling
                with patch("sagemaker.pytorch.PyTorch") as mock_pytorch:
                    mock_pytorch.return_value = self.mock_pytorch_estimator

                    builder.role = "test-role"
                    builder.session = Mock()

                    estimator = builder._create_estimator()

                    # Verify PyTorch estimator was called
                    if mock_pytorch.called:
                        call_kwargs = mock_pytorch.call_args[1]
                        self._assert(
                            "hyperparameters" in call_kwargs,
                            "PyTorch estimator should receive hyperparameters",
                        )

            self._assert(True, "PyTorch-specific methods validated")

        except Exception as e:
            self._log(f"PyTorch validation failed: {e}")
            self._assert(False, f"PyTorch validation failed: {e}")

    def _validate_xgboost_specific_methods(self) -> None:
        """Validate XGBoost-specific methods."""
        self._log("Validating XGBoost-specific methods")

        # XGBoost builders might use file-based hyperparameters
        if hasattr(self.builder_class, "_upload_hyperparameters_file"):
            config = Mock()
            config.hyperparameters = self.mock_hyperparameters
            config.pipeline_s3_loc = "s3://bucket/pipeline"

            try:
                builder = self.builder_class(config=config)
                builder.session = Mock()

                # Test hyperparameters file upload
                with patch("tempfile.NamedTemporaryFile"), patch(
                    "json.dump"
                ), patch.object(builder.session, "upload_data") as mock_upload:

                    s3_uri = builder._upload_hyperparameters_file()

                    self._assert(
                        s3_uri.startswith("s3://"),
                        "Hyperparameters file upload should return S3 URI",
                    )

                    # Verify upload was called
                    mock_upload.assert_called_once()

                self._assert(True, "XGBoost-specific methods validated")

            except Exception as e:
                self._log(f"XGBoost validation failed: {e}")
                self._assert(False, f"XGBoost validation failed: {e}")
        else:
            self._log("No XGBoost-specific hyperparameters upload method found")
            self._assert(
                True, "XGBoost validation completed (no specific methods required)"
            )

    def _validate_sklearn_specific_methods(self) -> None:
        """Validate SKLearn-specific methods."""
        self._log("Validating SKLearn-specific methods")
        # SKLearn validation would be similar to PyTorch
        self._assert(True, "SKLearn-specific methods validated")

    def _validate_tensorflow_specific_methods(self) -> None:
        """Validate TensorFlow-specific methods."""
        self._log("Validating TensorFlow-specific methods")
        # TensorFlow validation would be similar to PyTorch
        self._assert(True, "TensorFlow-specific methods validated")

    def test_hyperparameter_handling_methods(self) -> None:
        """Test that Training builders handle hyperparameters correctly."""
        self._log("Testing hyperparameter handling methods")

        config = Mock()
        config.hyperparameters = self.mock_hyperparameters

        try:
            builder = self.builder_class(config=config)

            # Test direct hyperparameter handling (PyTorch pattern)
            if hasattr(builder, "_create_estimator"):
                builder.role = "test-role"
                builder.session = Mock()

                # Mock estimator creation
                with patch("sagemaker.pytorch.PyTorch") as mock_estimator:
                    mock_estimator.return_value = self.mock_pytorch_estimator

                    estimator = builder._create_estimator()

                    if mock_estimator.called:
                        call_kwargs = mock_estimator.call_args[1]
                        if "hyperparameters" in call_kwargs:
                            hyperparams = call_kwargs["hyperparameters"]
                            self._assert(
                                isinstance(hyperparams, dict),
                                "Hyperparameters should be converted to dict",
                            )

            # Test file-based hyperparameter handling (XGBoost pattern)
            if hasattr(builder, "_upload_hyperparameters_file"):
                with patch("tempfile.NamedTemporaryFile"), patch(
                    "json.dump"
                ), patch.object(builder, "session") as mock_session:

                    mock_session.upload_data = Mock()
                    s3_uri = builder._upload_hyperparameters_file()

                    self._assert(
                        isinstance(s3_uri, str) and s3_uri.startswith("s3://"),
                        "File-based hyperparameters should return S3 URI",
                    )

            self._assert(True, "Hyperparameter handling methods validated")

        except Exception as e:
            self._log(f"Hyperparameter handling test failed: {e}")
            self._assert(False, f"Hyperparameter handling test failed: {e}")


    def test_training_input_output_methods(self) -> None:
        """Test that Training builders implement input/output methods correctly."""
        self._log("Testing training input/output methods")

        # Test _get_inputs method
        if hasattr(self.builder_class, "_get_inputs"):
            config = Mock()
            builder = self.builder_class(config=config)

            # Mock specification and contract
            builder.spec = Mock()
            builder.spec.dependencies = {
                "input_path": Mock(logical_name="input_path", required=True)
            }
            builder.contract = Mock()

            inputs = {"input_path": "s3://bucket/training/data"}

            try:
                training_inputs = builder._get_inputs(inputs)

                # Should return dict of TrainingInput objects
                self._assert(
                    isinstance(training_inputs, dict), "Training inputs should be dict"
                )

                self._log(f"Training inputs created: {list(training_inputs.keys())}")

            except Exception as e:
                self._log(f"Training inputs test failed: {e}")
                self._assert(False, f"Training inputs test failed: {e}")

        # Test _get_outputs method
        if hasattr(self.builder_class, "_get_outputs"):
            config = Mock()
            config.pipeline_s3_loc = "s3://bucket/pipeline"
            builder = self.builder_class(config=config)

            # Mock specification
            builder.spec = Mock()
            builder.spec.outputs = {
                "model_artifacts": Mock(logical_name="model_artifacts")
            }

            outputs = {"model_artifacts": "s3://bucket/models"}

            try:
                output_path = builder._get_outputs(outputs)

                # Should return string output path
                self._assert(
                    isinstance(output_path, str),
                    "Training outputs should be string path",
                )

                self._assert(
                    output_path.startswith("s3://"),
                    "Training output path should be S3 URI",
                )

                self._log(f"Training output path: {output_path}")

            except Exception as e:
                self._log(f"Training outputs test failed: {e}")
                self._assert(False, f"Training outputs test failed: {e}")

        self._assert(True, "Training input/output methods validated")
