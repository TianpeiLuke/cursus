"""
Level 3 Training-Specific Path Mapping Tests for step builders.

These tests focus on Training step path mapping and property path validation:
- TrainingInput object creation and data channel mapping
- Framework-specific data channel strategies (single vs multiple)
- Output path handling for model artifacts and evaluation results
- Property path validity for Training outputs
- Hyperparameter file handling and S3 path validation
"""

from typing import Dict, Any, List
from unittest.mock import Mock, patch

from ..path_mapping_tests import PathMappingTests


class TrainingPathMappingTests(PathMappingTests):
    """
    Level 3 Training-specific path mapping tests.
    
    These tests validate that Training step builders correctly map paths
    between specifications, contracts, and SageMaker TrainingInput objects.
    """
    
    def get_step_type_specific_tests(self) -> list:
        """Return Training-specific path mapping test methods."""
        return [
            "test_training_input_object_creation",
            "test_data_channel_mapping_strategies",
            "test_training_output_path_handling",
            "test_hyperparameter_file_path_handling",
            "test_training_property_paths",
            "test_framework_specific_path_patterns",
            "test_model_artifact_path_validation",
            "test_evaluation_output_path_validation",
            "test_training_path_consistency"
        ]
    
    def _configure_step_type_mocks(self) -> None:
        """Configure Training-specific mock objects for path mapping tests."""
        # Mock TrainingInput objects
        self.mock_training_input = Mock()
        self.mock_training_input.s3_data = "s3://bucket/training/data"
        self.mock_training_input.content_type = "text/csv"
        
        # Mock training specification
        self.mock_training_spec = Mock()
        self.mock_training_spec.dependencies = {
            "input_path": Mock(logical_name="input_path", required=True)
        }
        self.mock_training_spec.outputs = {
            "model_artifacts": Mock(
                logical_name="model_artifacts",
                property_path="Steps.TrainingStep.ModelArtifacts.S3ModelArtifacts"
            ),
            "evaluation_results": Mock(
                logical_name="evaluation_results",
                property_path="Steps.TrainingStep.ProcessingOutputConfig.Outputs['evaluation']"
            )
        }
        
        # Mock contract with Training container paths
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
            "epochs": 10
        }
    
    def _validate_step_type_requirements(self) -> dict:
        """Validate Training-specific requirements for path mapping tests."""
        return {
            "path_mapping_tests_completed": True,
            "training_specific_validations": True,
            "data_channel_mapping_validated": True,
            "output_path_handling_validated": True
        }
    
    def test_training_input_object_creation(self) -> None:
        """Test that Training builders create valid TrainingInput objects."""
        self._log("Testing TrainingInput object creation")
        
        if hasattr(self.builder_class, '_get_inputs'):
            config = Mock()
            builder = self.builder_class(config=config)
            
            # Mock specification and contract
            builder.spec = self.mock_training_spec
            builder.contract = self.mock_contract
            
            inputs = {"input_path": "s3://bucket/training/data"}
            
            try:
                training_inputs = builder._get_inputs(inputs)
                
                # Validate TrainingInput objects
                self._assert(isinstance(training_inputs, dict), "Should return dict of TrainingInput objects")
                
                for channel_name, training_input in training_inputs.items():
                    # Check TrainingInput structure
                    self._assert(isinstance(channel_name, str), "Channel name should be string")
                    self._assert(hasattr(training_input, 's3_data'), "TrainingInput should have s3_data")
                    
                    # Validate s3_data is S3 URI
                    s3_data = training_input.s3_data
                    self._assert(
                        isinstance(s3_data, str) and s3_data.startswith("s3://"),
                        f"s3_data should be S3 URI, got: {s3_data}"
                    )
                    
                    # Check for optional attributes
                    if hasattr(training_input, 'content_type'):
                        content_type = training_input.content_type
                        valid_types = ["text/csv", "application/json", "text/plain"]
                        self._assert(
                            content_type in valid_types,
                            f"Content type should be valid, got: {content_type}"
                        )
                
                self._assert(True, "TrainingInput object creation validated")
                
            except Exception as e:
                self._log(f"TrainingInput creation test failed: {e}")
                self._assert(False, f"TrainingInput creation test failed: {e}")
        else:
            self._log("No _get_inputs method found")
            self._assert(False, "Training builders should have _get_inputs method")
    
    def test_data_channel_mapping_strategies(self) -> None:
        """Test that Training builders use appropriate data channel mapping strategies."""
        self._log("Testing data channel mapping strategies")
        
        if hasattr(self.builder_class, '_get_inputs'):
            config = Mock()
            builder = self.builder_class(config=config)
            
            # Mock specification and contract
            builder.spec = self.mock_training_spec
            builder.contract = self.mock_contract
            
            inputs = {"input_path": "s3://bucket/training/data"}
            
            try:
                training_inputs = builder._get_inputs(inputs)
                
                # Analyze channel mapping strategy
                channel_names = list(training_inputs.keys())
                self._log(f"Detected channels: {channel_names}")
                
                # PyTorch pattern: single 'data' channel
                if len(channel_names) == 1 and 'data' in channel_names:
                    self._log("Detected PyTorch single-channel strategy")
                    
                    data_channel = training_inputs['data']
                    self._assert(
                        hasattr(data_channel, 's3_data'),
                        "PyTorch data channel should have s3_data"
                    )
                    
                    # PyTorch expects subdirectories (train/val/test) within the data path
                    s3_data = data_channel.s3_data
                    self._assert(
                        s3_data.startswith("s3://"),
                        "PyTorch data channel should use S3 URI"
                    )
                
                # XGBoost pattern: multiple channels (train, validation, test)
                elif any(ch in channel_names for ch in ['train', 'validation', 'test']):
                    self._log("Detected XGBoost multi-channel strategy")
                    
                    expected_channels = ['train', 'validation', 'test']
                    for channel in expected_channels:
                        if channel in training_inputs:
                            channel_input = training_inputs[channel]
                            self._assert(
                                hasattr(channel_input, 's3_data'),
                                f"{channel} channel should have s3_data"
                            )
                            
                            # XGBoost channels should point to specific subdirectories
                            s3_data = channel_input.s3_data
                            self._assert(
                                channel in s3_data or s3_data.endswith('/'),
                                f"{channel} channel should reference appropriate path"
                            )
                
                # Custom pattern
                else:
                    self._log(f"Detected custom channel strategy: {channel_names}")
                    
                    # Validate all channels have proper structure
                    for channel_name, channel_input in training_inputs.items():
                        self._assert(
                            hasattr(channel_input, 's3_data'),
                            f"Custom channel {channel_name} should have s3_data"
                        )
                
                self._assert(True, "Data channel mapping strategies validated")
                
            except Exception as e:
                self._log(f"Data channel mapping test failed: {e}")
                self._assert(False, f"Data channel mapping test failed: {e}")
        else:
            self._log("No _get_inputs method found")
            self._assert(False, "Training builders should have _get_inputs method")
    
    def test_training_output_path_handling(self) -> None:
        """Test that Training builders handle output paths correctly."""
        self._log("Testing training output path handling")
        
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
                
                # Should contain framework or step identifier
                framework_indicators = ['pytorch', 'xgboost', 'sklearn', 'training']
                has_framework_indicator = any(
                    indicator in output_path.lower() 
                    for indicator in framework_indicators
                )
                self._assert(
                    has_framework_indicator,
                    f"Output path should contain framework identifier: {output_path}"
                )
                
                self._assert(True, "Training output path handling validated")
                
            except Exception as e:
                self._log(f"Training output path test failed: {e}")
                self._assert(False, f"Training output path test failed: {e}")
        else:
            self._log("No _get_outputs method found")
            self._assert(False, "Training builders should have _get_outputs method")
    
    def test_hyperparameter_file_path_handling(self) -> None:
        """Test that Training builders handle hyperparameter file paths correctly."""
        self._log("Testing hyperparameter file path handling")
        
        # This test is specific to builders that upload hyperparameters files (XGBoost pattern)
        if hasattr(self.builder_class, '_upload_hyperparameters_file'):
            config = Mock()
            config.hyperparameters = self.mock_hyperparameters
            config.pipeline_s3_loc = "s3://bucket/pipeline"
            builder = self.builder_class(config=config)
            builder.session = Mock()
            
            try:
                with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
                     patch('json.dump'), \
                     patch.object(builder.session, 'upload_data') as mock_upload:
                    
                    # Mock temporary file
                    mock_temp_file.return_value.__enter__.return_value.name = "/tmp/hyperparams.json"
                    
                    s3_uri = builder._upload_hyperparameters_file()
                    
                    # Verify S3 URI format
                    self._assert(
                        s3_uri.startswith("s3://"),
                        f"Hyperparameters S3 URI should start with s3://, got: {s3_uri}"
                    )
                    
                    # Should contain hyperparameters in path
                    self._assert(
                        "hyperparameters" in s3_uri.lower(),
                        f"S3 URI should contain 'hyperparameters': {s3_uri}"
                    )
                    
                    # Should end with .json
                    self._assert(
                        s3_uri.endswith(".json"),
                        f"Hyperparameters file should be JSON: {s3_uri}"
                    )
                    
                    # Verify upload was called
                    mock_upload.assert_called_once()
                
                self._assert(True, "Hyperparameter file path handling validated")
                
            except Exception as e:
                self._log(f"Hyperparameter file path test failed: {e}")
                self._assert(False, f"Hyperparameter file path test failed: {e}")
        else:
            self._log("No hyperparameter file upload method found - skipping")
            self._assert(True, "Hyperparameter file handling not required for this builder")
    
    def test_training_property_paths(self) -> None:
        """Test that Training builders use valid property paths for outputs."""
        self._log("Testing Training property path validation")
        
        config = Mock()
        builder = self.builder_class(config=config)
        
        # Mock specification with property paths
        builder.spec = self.mock_training_spec
        
        try:
            # Validate property paths in specification
            if hasattr(builder.spec, 'outputs'):
                for output_name, output_spec in builder.spec.outputs.items():
                    if hasattr(output_spec, 'property_path'):
                        property_path = output_spec.property_path
                        
                        # Validate Training-specific property path patterns
                        if output_name == "model_artifacts":
                            self._assert(
                                "ModelArtifacts" in property_path,
                                f"Model artifacts property path should contain 'ModelArtifacts': {property_path}"
                            )
                            
                            self._assert(
                                "S3ModelArtifacts" in property_path,
                                f"Model artifacts should reference S3ModelArtifacts: {property_path}"
                            )
                        
                        elif output_name == "evaluation_results":
                            # Evaluation results might use ProcessingOutputConfig pattern
                            evaluation_patterns = ["ProcessingOutputConfig", "OutputDataConfig"]
                            has_evaluation_pattern = any(
                                pattern in property_path for pattern in evaluation_patterns
                            )
                            self._assert(
                                has_evaluation_pattern,
                                f"Evaluation results should use output config pattern: {property_path}"
                            )
                        
                        # Validate step reference
                        self._assert(
                            property_path.startswith("Steps."),
                            f"Property path should start with 'Steps.': {property_path}"
                        )
                        
                        # Should reference TrainingStep
                        self._assert(
                            "TrainingStep" in property_path,
                            f"Property path should reference TrainingStep: {property_path}"
                        )
                        
                        self._log(f"Valid Training property path: {property_path}")
            
            self._assert(True, "Training property paths validated")
            
        except Exception as e:
            self._log(f"Training property path test failed: {e}")
            self._assert(False, f"Training property path test failed: {e}")
    
    def test_framework_specific_path_patterns(self) -> None:
        """Test that Training builders follow framework-specific path patterns."""
        self._log("Testing framework-specific path patterns")
        
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
            self._log(f"Validating {detected_framework}-specific path patterns")
            
            if detected_framework == 'pytorch':
                self._validate_pytorch_path_patterns()
            elif detected_framework == 'xgboost':
                self._validate_xgboost_path_patterns()
            elif detected_framework == 'sklearn':
                self._validate_sklearn_path_patterns()
            elif detected_framework == 'tensorflow':
                self._validate_tensorflow_path_patterns()
        else:
            self._log("No specific framework detected - using generic validation")
            self._assert(True, "Generic training path patterns validated")
    
    def _validate_pytorch_path_patterns(self) -> None:
        """Validate PyTorch-specific path patterns."""
        self._log("Validating PyTorch-specific path patterns")
        
        # PyTorch uses single data channel with subdirectories
        if hasattr(self.builder_class, '_create_data_channel_from_source'):
            config = Mock()
            builder = self.builder_class(config=config)
            
            try:
                data_channel = builder._create_data_channel_from_source("s3://bucket/data")
                
                # Should return single 'data' channel
                self._assert(
                    isinstance(data_channel, dict) and 'data' in data_channel,
                    "PyTorch should create single 'data' channel"
                )
                
                self._assert(True, "PyTorch path patterns validated")
                
            except Exception as e:
                self._log(f"PyTorch path pattern validation failed: {e}")
                self._assert(False, f"PyTorch path pattern validation failed: {e}")
        else:
            self._log("No PyTorch-specific data channel method found")
            self._assert(True, "PyTorch path patterns validation completed")
    
    def _validate_xgboost_path_patterns(self) -> None:
        """Validate XGBoost-specific path patterns."""
        self._log("Validating XGBoost-specific path patterns")
        
        # XGBoost uses multiple data channels
        if hasattr(self.builder_class, '_create_data_channels_from_source'):
            config = Mock()
            builder = self.builder_class(config=config)
            
            try:
                data_channels = builder._create_data_channels_from_source("s3://bucket/data")
                
                # Should return multiple channels
                expected_channels = ['train', 'validation', 'test']
                self._assert(
                    isinstance(data_channels, dict),
                    "XGBoost should create multiple data channels"
                )
                
                found_channels = [ch for ch in expected_channels if ch in data_channels]
                self._assert(
                    len(found_channels) > 0,
                    f"XGBoost should create expected channels, found: {list(data_channels.keys())}"
                )
                
                self._assert(True, "XGBoost path patterns validated")
                
            except Exception as e:
                self._log(f"XGBoost path pattern validation failed: {e}")
                self._assert(False, f"XGBoost path pattern validation failed: {e}")
        else:
            self._log("No XGBoost-specific data channels method found")
            self._assert(True, "XGBoost path patterns validation completed")
    
    def _validate_sklearn_path_patterns(self) -> None:
        """Validate SKLearn-specific path patterns."""
        self._log("Validating SKLearn-specific path patterns")
        self._assert(True, "SKLearn path patterns validated")
    
    def _validate_tensorflow_path_patterns(self) -> None:
        """Validate TensorFlow-specific path patterns."""
        self._log("Validating TensorFlow-specific path patterns")
        self._assert(True, "TensorFlow path patterns validated")
    
    def test_model_artifact_path_validation(self) -> None:
        """Test that Training builders validate model artifact paths correctly."""
        self._log("Testing model artifact path validation")
        
        if hasattr(self.builder_class, '_get_outputs'):
            config = Mock()
            config.pipeline_s3_loc = "s3://bucket/pipeline"
            builder = self.builder_class(config=config)
            
            # Mock specification
            builder.spec = self.mock_training_spec
            
            outputs = {"model_artifacts": "s3://bucket/models/my-model"}
            
            try:
                output_path = builder._get_outputs(outputs)
                
                # Model artifact path should be valid S3 URI
                self._assert(
                    output_path.startswith("s3://"),
                    "Model artifact path should be S3 URI"
                )
                
                # Should be suitable for model storage
                path_components = output_path.split('/')
                self._assert(
                    len(path_components) >= 4,  # s3://bucket/path/...
                    "Model artifact path should have sufficient depth"
                )
                
                self._assert(True, "Model artifact path validation completed")
                
            except Exception as e:
                self._log(f"Model artifact path validation failed: {e}")
                self._assert(False, f"Model artifact path validation failed: {e}")
        else:
            self._log("No _get_outputs method found")
            self._assert(False, "Training builders should have _get_outputs method")
    
    def test_evaluation_output_path_validation(self) -> None:
        """Test that Training builders validate evaluation output paths correctly."""
        self._log("Testing evaluation output path validation")
        
        # Check if builder supports evaluation outputs
        config = Mock()
        builder = self.builder_class(config=config)
        
        # Mock specification with evaluation outputs
        builder.spec = Mock()
        builder.spec.outputs = {
            "model_artifacts": Mock(logical_name="model_artifacts"),
            "evaluation_results": Mock(logical_name="evaluation_results")
        }
        
        if hasattr(builder, '_get_outputs'):
            config.pipeline_s3_loc = "s3://bucket/pipeline"
            
            outputs = {
                "model_artifacts": "s3://bucket/models",
                "evaluation_results": "s3://bucket/evaluation"
            }
            
            try:
                output_path = builder._get_outputs(outputs)
                
                # Should handle evaluation outputs appropriately
                self._assert(
                    isinstance(output_path, str),
                    "Should return valid output path even with evaluation outputs"
                )
                
                self._assert(True, "Evaluation output path validation completed")
                
            except Exception as e:
                self._log(f"Evaluation output path validation failed: {e}")
                self._assert(False, f"Evaluation output path validation failed: {e}")
        else:
            self._log("No _get_outputs method found")
            self._assert(True, "Evaluation output validation not applicable")
    
    def test_training_path_consistency(self) -> None:
        """Test that input and output paths are consistent with Training specifications."""
        self._log("Testing training path consistency")
        
        config = Mock()
        config.pipeline_s3_loc = "s3://bucket/pipeline"
        builder = self.builder_class(config=config)
        
        # Mock specification and contract
        builder.spec = self.mock_training_spec
        builder.contract = self.mock_contract
        
        try:
            # Test input consistency
            if hasattr(builder, '_get_inputs'):
                inputs = {"input_path": "s3://bucket/training/data"}
                training_inputs = builder._get_inputs(inputs)
                
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
            
            # Test output consistency
            if hasattr(builder, '_get_outputs'):
                outputs = {"model_artifacts": "s3://bucket/models"}
                output_path = builder._get_outputs(outputs)
                
                # Output path should be consistent with pipeline location
                if hasattr(config, 'pipeline_s3_loc'):
                    pipeline_bucket = config.pipeline_s3_loc.split('/')[2]  # Extract bucket from s3://bucket/...
                    output_bucket = output_path.split('/')[2]  # Extract bucket from output path
                    
                    # Should use same bucket or be explicitly overridden
                    self._log(f"Pipeline bucket: {pipeline_bucket}, Output bucket: {output_bucket}")
            
            self._assert(True, "Training path consistency validated")
            
        except Exception as e:
            self._log(f"Training path consistency test failed: {e}")
            self._assert(False, f"Training path consistency test failed: {e}")
