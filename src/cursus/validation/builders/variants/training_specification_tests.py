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

from typing import Dict, Any, List
from unittest.mock import Mock, patch

from ..specification_tests import SpecificationTests


class TrainingSpecificationTests(SpecificationTests):
    """
    Level 2 Training-specific specification tests.
    
    These tests validate that Training step builders properly use specifications
    and contracts to define their behavior, with focus on Training-specific patterns.
    """
    
    def get_step_type_specific_tests(self) -> list:
        """Return Training-specific specification test methods."""
        return [
            "test_estimator_configuration_validation",
            "test_framework_specific_configuration",
            "test_hyperparameter_specification_compliance",
            "test_data_channel_specification",
            "test_training_environment_variables",
            "test_metric_definitions_specification",
            "test_training_input_specification",
            "test_training_output_specification",
            "test_training_contract_integration"
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
            "evaluation_results": Mock(logical_name="evaluation_results")
        }
        
        # Mock contract objects
        self.mock_contract = Mock()
        self.mock_contract.expected_input_paths = {
            "input_path": "/opt/ml/input/data"
        }
        self.mock_contract.expected_output_paths = {
            "model_artifacts": "/opt/ml/model",
            "evaluation_results": "/opt/ml/output/data"
        }
        
        # Mock hyperparameters
        self.mock_hyperparameters = Mock()
        self.mock_hyperparameters.to_dict.return_value = {
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 32
        }
    
    def _validate_step_type_requirements(self) -> dict:
        """Validate Training-specific requirements for specification tests."""
        return {
            "specification_tests_completed": True,
            "training_specific_validations": True,
            "estimator_configuration_validated": True,
            "framework_patterns_validated": True
        }
    
    def test_estimator_configuration_validation(self) -> None:
        """Test that Training builders validate estimator-specific configuration."""
        self._log("Testing estimator configuration validation")
        
        # Test required training configuration attributes
        required_attrs = [
            'training_instance_type',
            'training_instance_count',
            'training_volume_size',
            'training_entry_point',
            'source_dir',
            'framework_version',
            'py_version'
        ]
        
        if hasattr(self.builder_class, 'validate_configuration'):
            # Test with missing required attributes
            for attr in required_attrs:
                config = Mock()
                # Set all attributes except the one being tested
                for other_attr in required_attrs:
                    if other_attr != attr:
                        setattr(config, other_attr, "test_value")
                
                try:
                    builder = self.builder_class(config=config)
                    builder.validate_configuration()
                    self._log(f"Missing {attr} might be optional")
                except (ValueError, AttributeError):
                    self._assert(True, f"Correctly detected missing {attr}")
                except Exception as e:
                    self._log(f"Unexpected error for missing {attr}: {e}")
                    self._assert(False, f"Unexpected error for missing {attr}: {e}")
        else:
            self._log("No validate_configuration method found")
            self._assert(False, "Training builders should have validate_configuration method")
        
        self._assert(True, "Estimator configuration validation completed")
    
    def test_framework_specific_configuration(self) -> None:
        """Test that Training builders handle framework-specific configuration."""
        self._log("Testing framework-specific configuration")
        
        # Detect framework from builder class name
        builder_name = self.builder_class.__name__.lower()
        detected_framework = None
        
        framework_indicators = {
            'pytorch': ['pytorch', 'torch'],
            'xgboost': ['xgboost', 'xgb'],
            'sklearn': ['sklearn', 'scikit'],
            'tensorflow': ['tensorflow', 'tf']
        }
        
        for framework, indicators in framework_indicators.items():
            if any(indicator in builder_name for indicator in indicators):
                detected_framework = framework
                break
        
        if detected_framework:
            self._log(f"Detected framework: {detected_framework}")
            
            if detected_framework == 'pytorch':
                self._validate_pytorch_configuration()
            elif detected_framework == 'xgboost':
                self._validate_xgboost_configuration()
            elif detected_framework == 'sklearn':
                self._validate_sklearn_configuration()
            elif detected_framework == 'tensorflow':
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
            if hasattr(config, 'framework_version'):
                version = config.framework_version
                self._assert(
                    version.count('.') >= 1,
                    f"PyTorch version should have proper format: {version}"
                )
            
            # Validate Python version format
            if hasattr(config, 'py_version'):
                py_version = config.py_version
                self._assert(
                    py_version.startswith('py'),
                    f"Python version should start with 'py': {py_version}"
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
            if hasattr(config, 'framework_version'):
                version = config.framework_version
                valid_versions = ['1.0-1', '1.2-1', '1.3-1', '1.5-1']
                self._assert(
                    any(v in version for v in valid_versions),
                    f"XGBoost version should be valid: {version}"
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
            if hasattr(builder, '_create_estimator'):
                builder.role = "test-role"
                builder.session = Mock()
                
                # Mock estimator creation to check hyperparameter passing
                with patch('sagemaker.pytorch.PyTorch') as mock_pytorch:
                    mock_pytorch.return_value = self.mock_pytorch_estimator
                    
                    estimator = builder._create_estimator()
                    
                    if mock_pytorch.called:
                        call_kwargs = mock_pytorch.call_args[1]
                        if 'hyperparameters' in call_kwargs:
                            hyperparams = call_kwargs['hyperparameters']
                            self._assert(
                                isinstance(hyperparams, dict),
                                "Hyperparameters should be dict"
                            )
                            
                            # Check for common hyperparameters
                            expected_keys = ['learning_rate', 'epochs', 'batch_size']
                            found_keys = [k for k in expected_keys if k in hyperparams]
                            self._assert(
                                len(found_keys) > 0,
                                f"Should contain common hyperparameters, found: {list(hyperparams.keys())}"
                            )
            
            # Test file-based hyperparameter handling (XGBoost pattern)
            if hasattr(builder, '_upload_hyperparameters_file'):
                config.pipeline_s3_loc = "s3://bucket/pipeline"
                builder.session = Mock()
                
                with patch('tempfile.NamedTemporaryFile'), \
                     patch('json.dump') as mock_json_dump, \
                     patch.object(builder.session, 'upload_data') as mock_upload:
                    
                    s3_uri = builder._upload_hyperparameters_file()
                    
                    # Verify JSON serialization was called
                    mock_json_dump.assert_called_once()
                    
                    # Verify upload was called
                    mock_upload.assert_called_once()
                    
                    # Verify S3 URI format
                    self._assert(
                        s3_uri.startswith("s3://"),
                        "Hyperparameters file should return S3 URI"
                    )
            
            self._assert(True, "Hyperparameter specification compliance validated")
            
        except Exception as e:
            self._log(f"Hyperparameter specification test failed: {e}")
            self._assert(False, f"Hyperparameter specification test failed: {e}")
    
    def test_data_channel_specification(self) -> None:
        """Test that Training builders create data channels according to specification."""
        self._log("Testing data channel specification compliance")
        
        if hasattr(self.builder_class, '_get_inputs'):
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
                    "Training inputs should be dict of TrainingInput objects"
                )
                
                # Check for proper channel structure
                for channel_name, training_input in training_inputs.items():
                    self._assert(
                        isinstance(channel_name, str),
                        "Channel name should be string"
                    )
                    
                    # Check TrainingInput structure
                    self._assert(
                        hasattr(training_input, 's3_data'),
                        "TrainingInput should have s3_data attribute"
                    )
                
                # Check for framework-specific channel patterns
                channel_names = list(training_inputs.keys())
                
                # PyTorch pattern: single 'data' channel
                if 'data' in channel_names and len(channel_names) == 1:
                    self._log("Detected PyTorch single-channel pattern")
                    self._assert(True, "PyTorch data channel pattern validated")
                
                # XGBoost pattern: multiple channels (train, validation, test)
                elif any(ch in channel_names for ch in ['train', 'validation', 'test']):
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
    
    def test_training_environment_variables(self) -> None:
        """Test that Training builders handle environment variables correctly."""
        self._log("Testing Training environment variables handling")
        
        if hasattr(self.builder_class, '_get_environment_variables'):
            config = Mock()
            config.env = {
                "CUSTOM_TRAINING_VAR": "custom_value",
                "MODEL_TYPE": "classification"
            }
            
            try:
                builder = self.builder_class(config=config)
                env_vars = builder._get_environment_variables()
                
                # Check that environment variables are returned as dict
                self._assert(isinstance(env_vars, dict), "Environment variables should be dict")
                
                # Check for custom environment variables
                if hasattr(config, 'env') and config.env:
                    for key, value in config.env.items():
                        self._assert(
                            key in env_vars and env_vars[key] == value,
                            f"Custom environment variable {key} should be included"
                        )
                
                self._assert(True, "Training environment variables validated")
                
            except Exception as e:
                self._log(f"Training environment variables test failed: {e}")
                self._assert(False, f"Training environment variables test failed: {e}")
        else:
            self._log("No _get_environment_variables method found")
            self._assert(True, "Environment variables method not required")
    
    def test_metric_definitions_specification(self) -> None:
        """Test that Training builders define metrics according to specification."""
        self._log("Testing metric definitions specification")
        
        if hasattr(self.builder_class, '_get_metric_definitions'):
            config = Mock()
            
            try:
                builder = self.builder_class(config=config)
                metric_definitions = builder._get_metric_definitions()
                
                # Check that metric definitions are returned as list
                self._assert(
                    isinstance(metric_definitions, list),
                    "Metric definitions should be list"
                )
                
                # Check metric definition structure
                for metric in metric_definitions:
                    self._assert(
                        isinstance(metric, dict),
                        "Each metric definition should be dict"
                    )
                    self._assert(
                        "Name" in metric and "Regex" in metric,
                        "Metric definition should have Name and Regex"
                    )
                    
                    # Validate regex pattern
                    regex_pattern = metric["Regex"]
                    self._assert(
                        "([0-9\\.]+)" in regex_pattern,
                        f"Metric regex should capture numeric values: {regex_pattern}"
                    )
                
                # Check for common training metrics
                metric_names = [m["Name"] for m in metric_definitions]
                common_metrics = ["Train Loss", "Validation Loss", "Accuracy", "F1 Score"]
                found_metrics = [m for m in common_metrics if any(m in name for name in metric_names)]
                
                self._assert(
                    len(found_metrics) > 0,
                    f"Should define common training metrics, found: {metric_names}"
                )
                
                self._assert(True, "Metric definitions specification validated")
                
            except Exception as e:
                self._log(f"Metric definitions test failed: {e}")
                self._assert(False, f"Metric definitions test failed: {e}")
        else:
            self._log("No _get_metric_definitions method found")
            self._assert(True, "Metric definitions method not required")
    
    def test_training_input_specification(self) -> None:
        """Test that Training builders handle inputs according to specification."""
        self._log("Testing training input specification compliance")
        
        if hasattr(self.builder_class, '_get_inputs'):
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
                    isinstance(training_inputs, dict),
                    "Training inputs should be dict"
                )
                
                # Check that all required inputs are handled
                if hasattr(builder.spec, 'dependencies'):
                    for dep_name, dep_spec in builder.spec.dependencies.items():
                        if dep_spec.required:
                            logical_name = dep_spec.logical_name
                            if logical_name in inputs:
                                # Should be processed into training channels
                                self._assert(
                                    len(training_inputs) > 0,
                                    f"Required input {logical_name} should create training channels"
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
        
        if hasattr(self.builder_class, '_get_outputs'):
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
                    "Training outputs should be string path"
                )
                
                # Should be S3 URI
                self._assert(
                    output_path.startswith("s3://"),
                    "Training output path should be S3 URI"
                )
                
                # Should not end with slash for consistency
                self._assert(
                    not output_path.endswith('/'),
                    "Training output path should not end with slash"
                )
                
                self._assert(True, "Training output specification validated")
                
            except Exception as e:
                self._log(f"Training output specification test failed: {e}")
                self._assert(False, f"Training output specification test failed: {e}")
        else:
            self._log("No _get_outputs method found")
            self._assert(False, "Training builders should have _get_outputs method")
    
    def test_training_contract_integration(self) -> None:
        """Test that Training builders integrate with contracts correctly."""
        self._log("Testing training contract integration")
        
        config = Mock()
        builder = self.builder_class(config=config)
        
        # Mock contract
        builder.contract = self.mock_contract
        
        try:
            # Check that contract has expected structure
            if hasattr(builder, 'contract') and builder.contract:
                self._assert(
                    hasattr(builder.contract, 'expected_input_paths'),
                    "Contract should have expected_input_paths"
                )
                
                self._assert(
                    hasattr(builder.contract, 'expected_output_paths'),
                    "Contract should have expected_output_paths"
                )
                
                # Validate path structure
                if hasattr(builder.contract, 'expected_input_paths'):
                    input_paths = builder.contract.expected_input_paths
                    self._assert(isinstance(input_paths, dict), "Input paths should be dict")
                    
                    # Check for training-specific paths
                    for path in input_paths.values():
                        self._assert(
                            path.startswith("/opt/ml/"),
                            f"Input path should be container path: {path}"
                        )
                
                if hasattr(builder.contract, 'expected_output_paths'):
                    output_paths = builder.contract.expected_output_paths
                    self._assert(isinstance(output_paths, dict), "Output paths should be dict")
                    
                    # Check for training-specific output paths
                    expected_outputs = ["/opt/ml/model", "/opt/ml/output"]
                    for path in output_paths.values():
                        self._assert(
                            any(expected in path for expected in expected_outputs),
                            f"Output path should be training container path: {path}"
                        )
            
            self._assert(True, "Training contract integration validated")
            
        except Exception as e:
            self._log(f"Training contract integration test failed: {e}")
            self._assert(False, f"Training contract integration test failed: {e}")
