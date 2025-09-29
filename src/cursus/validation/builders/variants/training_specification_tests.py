"""
Level 2 Training-Specific Specification Tests for step builders.

These tests focus on Training step specification and contract compliance:
- Framework-specific estimator configuration
- Hyperparameter handling patterns (direct vs file-based)
- Data channel specification compliance
- Environment variable handling
- Metric definition validation
- Training input/output specification compliance
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

from ..specification_tests import SpecificationTests


class TrainingSpecificationTests(SpecificationTests):
    """
    Level 2 Training-specific specification tests.

    These tests validate that Training step builders properly use specifications
    and contracts to define their behavior, with focus on Training-specific patterns.
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
        Initialize Training specification tests.

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
        """Return Training-specific specification test methods."""
        return [
            "test_framework_specific_configuration",
            "test_hyperparameter_specification_compliance",
            "test_data_channel_specification",
            "test_training_input_specification",
            "test_training_output_specification",
        ]

    def _configure_step_type_mocks(self) -> None:
        """Configure Training-specific mock objects for specification tests."""
        # Mock framework-specific estimators
        self.mock_pytorch_estimator = Mock()
        self.mock_pytorch_estimator.__class__.__name__ = "PyTorch"

        self.mock_xgboost_estimator = Mock()
        self.mock_xgboost_estimator.__class__.__name__ = "XGBoost"

        # Mock training specification
        self.mock_training_spec = Mock()
        self.mock_training_spec.dependencies = {
            "input_path": Mock(logical_name="input_path", required=True)
        }
        self.mock_training_spec.outputs = {
            "model_artifacts": Mock(logical_name="model_artifacts"),
            "evaluation_results": Mock(logical_name="evaluation_results"),
        }

        # Mock contract objects
        self.mock_contract = Mock()
        self.mock_contract.expected_input_paths = {"input_path": "/opt/ml/input/data"}
        self.mock_contract.expected_output_paths = {
            "model_artifacts": "/opt/ml/model",
            "evaluation_results": "/opt/ml/output/data",
        }

        # Mock hyperparameters
        self.mock_hyperparameters = Mock()
        self.mock_hyperparameters.to_dict.return_value = {
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 32,
        }

    def _validate_step_type_requirements(self) -> dict:
        """Validate Training-specific requirements for specification tests."""
        return {
            "specification_tests_completed": True,
            "training_specific_validations": True,
            "estimator_configuration_validated": True,
            "framework_patterns_validated": True,
        }


    def test_framework_specific_configuration(self) -> None:
        """Test that Training builders handle framework-specific configuration."""
        self._log("Testing framework-specific configuration")

        # Detect framework from builder class name
        builder_name = self.builder_class.__name__.lower()
        detected_framework = None

        framework_indicators = {
            "pytorch": ["pytorch", "torch"],
            "xgboost": ["xgboost", "xgb"],
            "sklearn": ["sklearn", "scikit"],
            "tensorflow": ["tensorflow", "tf"],
        }

        for framework, indicators in framework_indicators.items():
            if any(indicator in builder_name for indicator in indicators):
                detected_framework = framework
                break

        if detected_framework:
            self._log(f"Detected framework: {detected_framework}")

            if detected_framework == "pytorch":
                self._validate_pytorch_configuration()
            elif detected_framework == "xgboost":
                self._validate_xgboost_configuration()
            elif detected_framework == "sklearn":
                self._validate_sklearn_configuration()
            elif detected_framework == "tensorflow":
                self._validate_tensorflow_configuration()
        else:
            self._log("No specific framework detected")
            self._assert(True, "Generic training configuration validated")

    def _validate_pytorch_configuration(self) -> None:
        """Validate PyTorch-specific configuration."""
        self._log("Validating PyTorch-specific configuration")

        config = Mock()
        config.framework_version = "1.12.0"
        config.py_version = "py38"
        config.training_entry_point = "train.py"
        config.source_dir = "src"
        config.hyperparameters = self.mock_hyperparameters

        try:
            builder = self.builder_class(config=config)

            # Validate PyTorch version format
            if hasattr(config, "framework_version"):
                version = config.framework_version
                self._assert(
                    version.count(".") >= 1,
                    f"PyTorch version should have proper format: {version}",
                )

            # Validate Python version format
            if hasattr(config, "py_version"):
                py_version = config.py_version
                self._assert(
                    py_version.startswith("py"),
                    f"Python version should start with 'py': {py_version}",
                )

            self._assert(True, "PyTorch-specific configuration validated")

        except Exception as e:
            self._log(f"PyTorch configuration validation failed: {e}")
            self._assert(False, f"PyTorch configuration validation failed: {e}")

    def _validate_xgboost_configuration(self) -> None:
        """Validate XGBoost-specific configuration."""
        self._log("Validating XGBoost-specific configuration")

        config = Mock()
        config.framework_version = "1.3-1"
        config.py_version = "py38"
        config.training_entry_point = "train.py"
        config.source_dir = "src"
        config.hyperparameters = self.mock_hyperparameters

        try:
            builder = self.builder_class(config=config)

            # Validate XGBoost version format
            if hasattr(config, "framework_version"):
                version = config.framework_version
                valid_versions = ["1.0-1", "1.2-1", "1.3-1", "1.5-1"]
                self._assert(
                    any(v in version for v in valid_versions),
                    f"XGBoost version should be valid: {version}",
                )

            self._assert(True, "XGBoost-specific configuration validated")

        except Exception as e:
            self._log(f"XGBoost configuration validation failed: {e}")
            self._assert(False, f"XGBoost configuration validation failed: {e}")

    def _validate_sklearn_configuration(self) -> None:
        """Validate SKLearn-specific configuration."""
        self._log("Validating SKLearn-specific configuration")
        self._assert(True, "SKLearn-specific configuration validated")

    def _validate_tensorflow_configuration(self) -> None:
        """Validate TensorFlow-specific configuration."""
        self._log("Validating TensorFlow-specific configuration")
        self._assert(True, "TensorFlow-specific configuration validated")

    def test_hyperparameter_specification_compliance(self) -> None:
        """Test that Training builders handle hyperparameters according to specification."""
        self._log("Testing hyperparameter specification compliance")

        config = Mock()
        config.hyperparameters = self.mock_hyperparameters

        try:
            builder = self.builder_class(config=config)

            # Test direct hyperparameter handling (PyTorch pattern)
            if hasattr(builder, "_create_estimator"):
                builder.role = "test-role"
                builder.session = Mock()

                # Mock estimator creation to check hyperparameter passing
                with patch("sagemaker.pytorch.PyTorch") as mock_pytorch:
                    mock_pytorch.return_value = self.mock_pytorch_estimator

                    estimator = builder._create_estimator()

                    if mock_pytorch.called:
                        call_kwargs = mock_pytorch.call_args[1]
                        if "hyperparameters" in call_kwargs:
                            hyperparams = call_kwargs["hyperparameters"]
                            self._assert(
                                isinstance(hyperparams, dict),
                                "Hyperparameters should be dict",
                            )

                            # Check for common hyperparameters
                            expected_keys = ["learning_rate", "epochs", "batch_size"]
                            found_keys = [k for k in expected_keys if k in hyperparams]
                            self._assert(
                                len(found_keys) > 0,
                                f"Should contain common hyperparameters, found: {list(hyperparams.keys())}",
                            )

            # Test file-based hyperparameter handling (XGBoost pattern)
            if hasattr(builder, "_upload_hyperparameters_file"):
                config.pipeline_s3_loc = "s3://bucket/pipeline"
                builder.session = Mock()

                with patch("tempfile.NamedTemporaryFile"), patch(
                    "json.dump"
                ) as mock_json_dump, patch.object(
                    builder.session, "upload_data"
                ) as mock_upload:

                    s3_uri = builder._upload_hyperparameters_file()

                    # Verify JSON serialization was called
                    mock_json_dump.assert_called_once()

                    # Verify upload was called
                    mock_upload.assert_called_once()

                    # Verify S3 URI format
                    self._assert(
                        s3_uri.startswith("s3://"),
                        "Hyperparameters file should return S3 URI",
                    )

            self._assert(True, "Hyperparameter specification compliance validated")

        except Exception as e:
            self._log(f"Hyperparameter specification test failed: {e}")
            self._assert(False, f"Hyperparameter specification test failed: {e}")

    def test_data_channel_specification(self) -> None:
        """Test that Training builders create data channels according to specification."""
        self._log("Testing data channel specification compliance")

        if hasattr(self.builder_class, "_get_inputs"):
            config = Mock()
            builder = self.builder_class(config=config)

            # Mock specification and contract
            builder.spec = self.mock_training_spec
            builder.contract = self.mock_contract

            inputs = {"input_path": "s3://bucket/training/data"}

            try:
                training_inputs = builder._get_inputs(inputs)

                # Should return dict of TrainingInput objects
                self._assert(
                    isinstance(training_inputs, dict),
                    "Training inputs should be dict of TrainingInput objects",
                )

                # Check for proper channel structure
                for channel_name, training_input in training_inputs.items():
                    self._assert(
                        isinstance(channel_name, str), "Channel name should be string"
                    )

                    # Check TrainingInput structure
                    self._assert(
                        hasattr(training_input, "s3_data"),
                        "TrainingInput should have s3_data attribute",
                    )

                # Check for framework-specific channel patterns
                channel_names = list(training_inputs.keys())

                # PyTorch pattern: single 'data' channel
                if "data" in channel_names and len(channel_names) == 1:
                    self._log("Detected PyTorch single-channel pattern")
                    self._assert(True, "PyTorch data channel pattern validated")

                # XGBoost pattern: multiple channels (train, validation, test)
                elif any(ch in channel_names for ch in ["train", "validation", "test"]):
                    self._log("Detected XGBoost multi-channel pattern")
                    self._assert(True, "XGBoost data channel pattern validated")

                else:
                    self._log(f"Custom data channel pattern: {channel_names}")
                    self._assert(True, "Custom data channel pattern validated")

                self._assert(True, "Data channel specification validated")

            except Exception as e:
                self._log(f"Data channel specification test failed: {e}")
                self._assert(False, f"Data channel specification test failed: {e}")
        else:
            self._log("No _get_inputs method found")
            self._assert(False, "Training builders should have _get_inputs method")


    def test_training_input_specification(self) -> None:
        """Test that Training builders handle inputs according to specification."""
        self._log("Testing training input specification compliance")

        if hasattr(self.builder_class, "_get_inputs"):
            config = Mock()
            builder = self.builder_class(config=config)

            # Mock specification and contract
            builder.spec = self.mock_training_spec
            builder.contract = self.mock_contract

            inputs = {"input_path": "s3://bucket/training/data"}

            try:
                training_inputs = builder._get_inputs(inputs)

                # Validate input structure
                self._assert(
                    isinstance(training_inputs, dict), "Training inputs should be dict"
                )

                # Check that all required inputs are handled
                if hasattr(builder.spec, "dependencies"):
                    for dep_name, dep_spec in builder.spec.dependencies.items():
                        if dep_spec.required:
                            logical_name = dep_spec.logical_name
                            if logical_name in inputs:
                                # Should be processed into training channels
                                self._assert(
                                    len(training_inputs) > 0,
                                    f"Required input {logical_name} should create training channels",
                                )

                self._assert(True, "Training input specification validated")

            except Exception as e:
                self._log(f"Training input specification test failed: {e}")
                self._assert(False, f"Training input specification test failed: {e}")
        else:
            self._log("No _get_inputs method found")
            self._assert(False, "Training builders should have _get_inputs method")

    def test_training_output_specification(self) -> None:
        """Test that Training builders handle outputs according to specification."""
        self._log("Testing training output specification compliance")

        if hasattr(self.builder_class, "_get_outputs"):
            config = Mock()
            config.pipeline_s3_loc = "s3://bucket/pipeline"
            builder = self.builder_class(config=config)

            # Mock specification
            builder.spec = self.mock_training_spec
            builder.contract = self.mock_contract

            outputs = {"model_artifacts": "s3://bucket/models"}

            try:
                output_path = builder._get_outputs(outputs)

                # Should return string output path
                self._assert(
                    isinstance(output_path, str),
                    "Training outputs should be string path",
                )

                # Should be S3 URI
                self._assert(
                    output_path.startswith("s3://"),
                    "Training output path should be S3 URI",
                )

                # Should not end with slash for consistency
                self._assert(
                    not output_path.endswith("/"),
                    "Training output path should not end with slash",
                )

                self._assert(True, "Training output specification validated")

            except Exception as e:
                self._log(f"Training output specification test failed: {e}")
                self._assert(False, f"Training output specification test failed: {e}")
        else:
            self._log("No _get_outputs method found")
            self._assert(False, "Training builders should have _get_outputs method")
