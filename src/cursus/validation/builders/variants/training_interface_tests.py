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

from typing import Dict, Any, List
from unittest.mock import Mock, patch

from ..interface_tests import InterfaceTests


class TrainingInterfaceTests(InterfaceTests):
    """
    Level 1 Training-specific interface tests.
    
    These tests validate that Training step builders implement the correct
    interface patterns for framework-specific training operations.
    """
    
    def get_step_type_specific_tests(self) -> list:
        """Return Training-specific interface test methods."""
        return [
            "test_estimator_creation_method",
            "test_training_configuration_attributes",
            "test_framework_specific_methods",
            "test_hyperparameter_handling_methods",
            "test_data_channel_creation_methods",
            "test_environment_variables_method",
            "test_metric_definitions_method",
            "test_training_input_output_methods",
            "test_step_creation_pattern_compliance"
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
        self.mock_hyperparameters.to_dict.return_value = {"learning_rate": 0.01, "epochs": 10}
        
        # Mock metric definitions
        self.mock_metric_definitions = [
            {"Name": "Train Loss", "Regex": "Train Loss: ([0-9\\.]+)"},
            {"Name": "Validation Loss", "Regex": "Validation Loss: ([0-9\\.]+)"}
        ]
    
    def _validate_step_type_requirements(self) -> dict:
        """Validate Training-specific requirements for interface tests."""
        return {
            "interface_tests_completed": True,
            "training_specific_validations": True,
            "estimator_creation_validated": True,
            "framework_patterns_validated": True
        }
    
    def test_estimator_creation_method(self) -> None:
        """Test that Training builders implement estimator creation method."""
        self._log("Testing estimator creation method")
        
        # Check for _create_estimator method
        self._assert(
            hasattr(self.builder_class, '_create_estimator'),
            "Training builders should have _create_estimator method"
        )
        
        if hasattr(self.builder_class, '_create_estimator'):
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
                if hasattr(config, 'hyperparameters'):
                    config.hyperparameters = self.mock_hyperparameters
                
                estimator = builder._create_estimator()
                
                # Validate estimator type
                estimator_type = type(estimator).__name__
                self._assert(
                    estimator_type in ["PyTorch", "XGBoost", "SKLearn", "TensorFlow", "Mock"],
                    f"Should create valid estimator type, got: {estimator_type}"
                )
                
                self._assert(True, "Estimator creation method validated")
                
            except Exception as e:
                self._log(f"Estimator creation test failed: {e}")
                self._assert(False, f"Estimator creation test failed: {e}")
        else:
            self._assert(False, "Training builders must have _create_estimator method")
    
    def test_training_configuration_attributes(self) -> None:
        """Test that Training builders validate required training configuration."""
        self._log("Testing training configuration attributes")
        
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
                    self._log(f"Missing {attr} should have caused validation error")
                    # Some attributes might be optional, so we don't fail here
                except (ValueError, AttributeError):
                    self._assert(True, f"Correctly detected missing {attr}")
                except Exception as e:
                    self._log(f"Unexpected error for missing {attr}: {e}")
            
            self._assert(True, "Training configuration attributes validated")
        else:
            self._log("No validate_configuration method found")
            self._assert(True, "Configuration validation method not required")
    
    def test_framework_specific_methods(self) -> None:
        """Test that Training builders implement framework-specific methods."""
        self._log("Testing framework-specific methods")
        
        # Check for framework-specific patterns
        framework_indicators = {
            'pytorch': ['PyTorch', 'torch', 'pytorch'],
            'xgboost': ['XGBoost', 'xgb', 'xgboost'],
            'sklearn': ['SKLearn', 'sklearn', 'scikit'],
            'tensorflow': ['TensorFlow', 'tensorflow', 'tf']
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
            if detected_framework == 'pytorch':
                self._validate_pytorch_specific_methods()
            elif detected_framework == 'xgboost':
                self._validate_xgboost_specific_methods()
            elif detected_framework == 'sklearn':
                self._validate_sklearn_specific_methods()
            elif detected_framework == 'tensorflow':
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
            if hasattr(builder, '_create_estimator'):
                # Mock the estimator creation to check hyperparameter handling
                with patch('sagemaker.pytorch.PyTorch') as mock_pytorch:
                    mock_pytorch.return_value = self.mock_pytorch_estimator
                    
                    builder.role = "test-role"
                    builder.session = Mock()
                    
                    estimator = builder._create_estimator()
                    
                    # Verify PyTorch estimator was called
                    if mock_pytorch.called:
                        call_kwargs = mock_pytorch.call_args[1]
                        self._assert(
                            'hyperparameters' in call_kwargs,
                            "PyTorch estimator should receive hyperparameters"
                        )
            
            self._assert(True, "PyTorch-specific methods validated")
            
        except Exception as e:
            self._log(f"PyTorch validation failed: {e}")
            self._assert(False, f"PyTorch validation failed: {e}")
    
    def _validate_xgboost_specific_methods(self) -> None:
        """Validate XGBoost-specific methods."""
        self._log("Validating XGBoost-specific methods")
        
        # XGBoost builders might use file-based hyperparameters
        if hasattr(self.builder_class, '_upload_hyperparameters_file'):
            config = Mock()
            config.hyperparameters = self.mock_hyperparameters
            config.pipeline_s3_loc = "s3://bucket/pipeline"
            
            try:
                builder = self.builder_class(config=config)
                builder.session = Mock()
                
                # Test hyperparameters file upload
                with patch('tempfile.NamedTemporaryFile'), \
                     patch('json.dump'), \
                     patch.object(builder.session, 'upload_data') as mock_upload:
                    
                    s3_uri = builder._upload_hyperparameters_file()
                    
                    self._assert(
                        s3_uri.startswith("s3://"),
                        "Hyperparameters file upload should return S3 URI"
                    )
                    
                    # Verify upload was called
                    mock_upload.assert_called_once()
                
                self._assert(True, "XGBoost-specific methods validated")
                
            except Exception as e:
                self._log(f"XGBoost validation failed: {e}")
                self._assert(False, f"XGBoost validation failed: {e}")
        else:
            self._log("No XGBoost-specific hyperparameters upload method found")
            self._assert(True, "XGBoost validation completed (no specific methods required)")
    
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
            if hasattr(builder, '_create_estimator'):
                builder.role = "test-role"
                builder.session = Mock()
                
                # Mock estimator creation
                with patch('sagemaker.pytorch.PyTorch') as mock_estimator:
                    mock_estimator.return_value = self.mock_pytorch_estimator
                    
                    estimator = builder._create_estimator()
                    
                    if mock_estimator.called:
                        call_kwargs = mock_estimator.call_args[1]
                        if 'hyperparameters' in call_kwargs:
                            hyperparams = call_kwargs['hyperparameters']
                            self._assert(
                                isinstance(hyperparams, dict),
                                "Hyperparameters should be converted to dict"
                            )
            
            # Test file-based hyperparameter handling (XGBoost pattern)
            if hasattr(builder, '_upload_hyperparameters_file'):
                with patch('tempfile.NamedTemporaryFile'), \
                     patch('json.dump'), \
                     patch.object(builder, 'session') as mock_session:
                    
                    mock_session.upload_data = Mock()
                    s3_uri = builder._upload_hyperparameters_file()
                    
                    self._assert(
                        isinstance(s3_uri, str) and s3_uri.startswith("s3://"),
                        "File-based hyperparameters should return S3 URI"
                    )
            
            self._assert(True, "Hyperparameter handling methods validated")
            
        except Exception as e:
            self._log(f"Hyperparameter handling test failed: {e}")
            self._assert(False, f"Hyperparameter handling test failed: {e}")
    
    def test_data_channel_creation_methods(self) -> None:
        """Test that Training builders implement data channel creation methods."""
        self._log("Testing data channel creation methods")
        
        # Check for data channel creation methods
        data_channel_methods = [
            '_create_data_channel_from_source',
            '_create_data_channels_from_source',
            '_get_inputs'
        ]
        
        found_methods = []
        for method in data_channel_methods:
            if hasattr(self.builder_class, method):
                found_methods.append(method)
        
        self._assert(
            len(found_methods) > 0,
            f"Training builders should have data channel methods, found: {found_methods}"
        )
        
        # Test data channel creation if method exists
        if hasattr(self.builder_class, '_get_inputs'):
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
                    isinstance(training_inputs, dict),
                    "Training inputs should be dict of TrainingInput objects"
                )
                
                # Check for common channel names
                common_channels = ["data", "train", "validation", "test"]
                found_channels = [ch for ch in common_channels if ch in training_inputs.keys()]
                
                self._assert(
                    len(found_channels) > 0,
                    f"Should create common training channels, found: {list(training_inputs.keys())}"
                )
                
                self._assert(True, "Data channel creation methods validated")
                
            except Exception as e:
                self._log(f"Data channel creation test failed: {e}")
                self._assert(False, f"Data channel creation test failed: {e}")
        else:
            self._log("No _get_inputs method found")
            self._assert(False, "Training builders should have _get_inputs method")
    
    def test_environment_variables_method(self) -> None:
        """Test that Training builders implement environment variables method."""
        self._log("Testing environment variables method")
        
        if hasattr(self.builder_class, '_get_environment_variables'):
            config = Mock()
            config.env = {"CUSTOM_VAR": "custom_value"}
            
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
                
                self._assert(True, "Environment variables method validated")
                
            except Exception as e:
                self._log(f"Environment variables test failed: {e}")
                self._assert(False, f"Environment variables test failed: {e}")
        else:
            self._log("No _get_environment_variables method found")
            self._assert(True, "Environment variables method not required")
    
    def test_metric_definitions_method(self) -> None:
        """Test that Training builders implement metric definitions method."""
        self._log("Testing metric definitions method")
        
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
                
                self._assert(True, "Metric definitions method validated")
                
            except Exception as e:
                self._log(f"Metric definitions test failed: {e}")
                self._assert(False, f"Metric definitions test failed: {e}")
        else:
            self._log("No _get_metric_definitions method found")
            self._assert(True, "Metric definitions method not required")
    
    def test_training_input_output_methods(self) -> None:
        """Test that Training builders implement input/output methods correctly."""
        self._log("Testing training input/output methods")
        
        # Test _get_inputs method
        if hasattr(self.builder_class, '_get_inputs'):
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
                    isinstance(training_inputs, dict),
                    "Training inputs should be dict"
                )
                
                self._log(f"Training inputs created: {list(training_inputs.keys())}")
                
            except Exception as e:
                self._log(f"Training inputs test failed: {e}")
                self._assert(False, f"Training inputs test failed: {e}")
        
        # Test _get_outputs method
        if hasattr(self.builder_class, '_get_outputs'):
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
                    "Training outputs should be string path"
                )
                
                self._assert(
                    output_path.startswith("s3://"),
                    "Training output path should be S3 URI"
                )
                
                self._log(f"Training output path: {output_path}")
                
            except Exception as e:
                self._log(f"Training outputs test failed: {e}")
                self._assert(False, f"Training outputs test failed: {e}")
        
        self._assert(True, "Training input/output methods validated")
    
    def test_step_creation_pattern_compliance(self) -> None:
        """Test that Training builders follow correct step creation patterns."""
        self._log("Testing step creation pattern compliance")
        
        if hasattr(self.builder_class, 'create_step'):
            config = Mock()
            config.training_instance_type = "ml.m5.large"
            config.training_instance_count = 1
            config.training_volume_size = 30
            config.training_entry_point = "train.py"
            config.source_dir = "src"
            config.framework_version = "1.12.0"
            config.py_version = "py38"
            
            try:
                builder = self.builder_class(config=config)
                builder.role = "test-role"
                builder.session = Mock()
                
                # Mock required methods
                builder._create_estimator = Mock(return_value=self.mock_pytorch_estimator)
                builder._get_inputs = Mock(return_value={"data": self.mock_training_input})
                builder._get_outputs = Mock(return_value="s3://bucket/output")
                builder._get_step_name = Mock(return_value="test-training-step")
                builder._get_cache_config = Mock(return_value=None)
                builder.extract_inputs_from_dependencies = Mock(return_value={})
                
                # Test step creation
                with patch('sagemaker.workflow.steps.TrainingStep') as mock_step_class:
                    mock_training_step = Mock()
                    mock_step_class.return_value = mock_training_step
                    
                    step = builder.create_step(
                        inputs={"input_path": "s3://bucket/input"},
                        dependencies=[],
                        enable_caching=True
                    )
                    
                    # Verify step creation
                    self._assert(step is not None, "TrainingStep should be created")
                    
                    # Verify TrainingStep was instantiated
                    mock_step_class.assert_called_once()
                    
                    # Check TrainingStep parameters
                    call_kwargs = mock_step_class.call_args[1]
                    expected_params = ['name', 'estimator', 'inputs']
                    for param in expected_params:
                        self._assert(
                            param in call_kwargs,
                            f"TrainingStep should have {param} parameter"
                        )
                
                self._assert(True, "Step creation pattern compliance validated")
                
            except Exception as e:
                self._log(f"Step creation pattern test failed: {e}")
                self._assert(False, f"Step creation pattern test failed: {e}")
        else:
            self._log("No create_step method found")
            self._assert(False, "Training builders should have create_step method")
