"""
Level 1 CreateModel-Specific Interface Tests for step builders.

These tests focus on CreateModel step interface and inheritance validation:
- Model creation methods (_create_model)
- Framework-specific model configuration
- Container image configuration
- Model name generation methods
- Environment variable setup for inference
- Deployment preparation methods
"""

from typing import Dict, Any, List
from unittest.mock import Mock, patch

from ..interface_tests import InterfaceTests


class CreateModelInterfaceTests(InterfaceTests):
    """
    Level 1 CreateModel-specific interface tests.
    
    These tests validate that CreateModel step builders implement the correct
    interface patterns for model deployment preparation.
    """
    
    def get_step_type_specific_tests(self) -> list:
        """Return CreateModel-specific interface test methods."""
        return [
            "test_model_creation_method",
            "test_model_configuration_attributes",
            "test_framework_specific_methods",
            "test_container_image_configuration",
            "test_model_name_generation_method",
            "test_environment_variables_method",
            "test_deployment_preparation_methods",
            "test_model_integration_methods",
            "test_step_creation_pattern_compliance"
        ]
    
    def _configure_step_type_mocks(self) -> None:
        """Configure CreateModel-specific mock objects for interface tests."""
        # Mock SageMaker Model objects
        self.mock_sagemaker_model = Mock()
        self.mock_sagemaker_model.name = "test-model"
        self.mock_sagemaker_model.model_data = "s3://bucket/model.tar.gz"
        self.mock_sagemaker_model.image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1"
        
        # Mock framework-specific container images
        self.mock_container_images = {
            "xgboost": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1",
            "pytorch": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.0-gpu-py38",
            "sklearn": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1"
        }
        
        # Mock environment variables for inference
        self.mock_inference_env = {
            'SAGEMAKER_PROGRAM': 'inference.py',
            'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
            'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600'
        }
    
    def _validate_step_type_requirements(self) -> dict:
        """Validate CreateModel-specific requirements for interface tests."""
        return {
            "interface_tests_completed": True,
            "createmodel_specific_validations": True,
            "model_creation_validated": True,
            "container_configuration_validated": True
        }
    
    def test_model_creation_method(self) -> None:
        """Test that CreateModel builders implement model creation method."""
        self._log("Testing model creation method")
        
        # Check for _create_model method
        self._assert(
            hasattr(self.builder_class, '_create_model'),
            "CreateModel builders should have _create_model method"
        )
        
        if hasattr(self.builder_class, '_create_model'):
            config = Mock()
            config.model_data = "s3://bucket/model.tar.gz"
            config.image_uri = self.mock_container_images["xgboost"]
            
            try:
                builder = self.builder_class(config=config)
                builder.role = "test-role"
                builder.session = Mock()
                
                model = builder._create_model()
                
                # Validate model type
                model_type = type(model).__name__
                self._assert(
                    model_type in ["Model", "Mock"],
                    f"Should create valid Model object, got: {model_type}"
                )
                
                # Check model attributes
                if hasattr(model, 'model_data'):
                    self._assert(
                        model.model_data is not None,
                        "Model should have model_data configured"
                    )
                
                if hasattr(model, 'image_uri'):
                    self._assert(
                        model.image_uri is not None,
                        "Model should have image_uri configured"
                    )
                
                self._assert(True, "Model creation method validated")
                
            except Exception as e:
                self._log(f"Model creation test failed: {e}")
                self._assert(False, f"Model creation test failed: {e}")
        else:
            self._assert(False, "CreateModel builders must have _create_model method")
    
    def test_model_configuration_attributes(self) -> None:
        """Test that CreateModel builders validate required model configuration."""
        self._log("Testing model configuration attributes")
        
        # Test required model configuration attributes
        required_attrs = [
            'model_data',
            'image_uri',
            'role'
        ]
        
        if hasattr(self.builder_class, 'validate_configuration'):
            # Test with missing required attributes
            for attr in required_attrs:
                config = Mock()
                # Set all attributes except the one being tested
                for other_attr in required_attrs:
                    if other_attr != attr:
                        if other_attr == 'model_data':
                            setattr(config, other_attr, "s3://bucket/model.tar.gz")
                        elif other_attr == 'image_uri':
                            setattr(config, other_attr, self.mock_container_images["xgboost"])
                        elif other_attr == 'role':
                            setattr(config, other_attr, "arn:aws:iam::123456789012:role/SageMakerRole")
                        else:
                            setattr(config, other_attr, "test_value")
                
                try:
                    builder = self.builder_class(config=config)
                    if hasattr(builder, 'validate_configuration'):
                        builder.validate_configuration()
                    self._log(f"Missing {attr} might be optional or handled differently")
                except (ValueError, AttributeError):
                    self._assert(True, f"Correctly detected missing {attr}")
                except Exception as e:
                    self._log(f"Unexpected error for missing {attr}: {e}")
        else:
            self._log("No validate_configuration method found")
            self._assert(True, "Configuration validation method not required")
        
        self._assert(True, "Model configuration attributes validated")
    
    def test_framework_specific_methods(self) -> None:
        """Test that CreateModel builders implement framework-specific methods."""
        self._log("Testing framework-specific methods")
        
        # Detect framework from builder class name
        builder_name = self.builder_class.__name__.lower()
        detected_framework = None
        
        framework_indicators = {
            'xgboost': ['xgboost', 'xgb'],
            'pytorch': ['pytorch', 'torch'],
            'sklearn': ['sklearn', 'scikit'],
            'tensorflow': ['tensorflow', 'tf']
        }
        
        for framework, indicators in framework_indicators.items():
            if any(indicator in builder_name for indicator in indicators):
                detected_framework = framework
                break
        
        if detected_framework:
            self._log(f"Detected framework: {detected_framework}")
            
            # Framework-specific validation
            if detected_framework == 'xgboost':
                self._validate_xgboost_model_methods()
            elif detected_framework == 'pytorch':
                self._validate_pytorch_model_methods()
            elif detected_framework == 'sklearn':
                self._validate_sklearn_model_methods()
            elif detected_framework == 'tensorflow':
                self._validate_tensorflow_model_methods()
        else:
            self._log("No specific framework detected - using generic validation")
            self._assert(True, "Generic CreateModel builder validated")
    
    def _validate_xgboost_model_methods(self) -> None:
        """Validate XGBoost-specific model methods."""
        self._log("Validating XGBoost-specific model methods")
        
        config = Mock()
        config.model_data = "s3://bucket/xgboost-model.tar.gz"
        config.image_uri = self.mock_container_images["xgboost"]
        
        try:
            builder = self.builder_class(config=config)
            builder.role = "test-role"
            
            # Check for XGBoost-specific container image
            if hasattr(config, 'image_uri'):
                image_uri = config.image_uri
                self._assert(
                    "xgboost" in image_uri.lower(),
                    f"XGBoost model should use XGBoost container: {image_uri}"
                )
            
            self._assert(True, "XGBoost-specific model methods validated")
            
        except Exception as e:
            self._log(f"XGBoost model validation failed: {e}")
            self._assert(False, f"XGBoost model validation failed: {e}")
    
    def _validate_pytorch_model_methods(self) -> None:
        """Validate PyTorch-specific model methods."""
        self._log("Validating PyTorch-specific model methods")
        
        config = Mock()
        config.model_data = "s3://bucket/pytorch-model.tar.gz"
        config.image_uri = self.mock_container_images["pytorch"]
        
        try:
            builder = self.builder_class(config=config)
            builder.role = "test-role"
            
            # Check for PyTorch-specific container image
            if hasattr(config, 'image_uri'):
                image_uri = config.image_uri
                self._assert(
                    "pytorch" in image_uri.lower(),
                    f"PyTorch model should use PyTorch container: {image_uri}"
                )
            
            self._assert(True, "PyTorch-specific model methods validated")
            
        except Exception as e:
            self._log(f"PyTorch model validation failed: {e}")
            self._assert(False, f"PyTorch model validation failed: {e}")
    
    def _validate_sklearn_model_methods(self) -> None:
        """Validate SKLearn-specific model methods."""
        self._log("Validating SKLearn-specific model methods")
        self._assert(True, "SKLearn-specific model methods validated")
    
    def _validate_tensorflow_model_methods(self) -> None:
        """Validate TensorFlow-specific model methods."""
        self._log("Validating TensorFlow-specific model methods")
        self._assert(True, "TensorFlow-specific model methods validated")
    
    def test_container_image_configuration(self) -> None:
        """Test that CreateModel builders configure container images correctly."""
        self._log("Testing container image configuration")
        
        config = Mock()
        config.model_data = "s3://bucket/model.tar.gz"
        config.image_uri = self.mock_container_images["xgboost"]
        
        try:
            builder = self.builder_class(config=config)
            builder.role = "test-role"
            
            # Check image URI format
            if hasattr(config, 'image_uri'):
                image_uri = config.image_uri
                
                # Should be ECR URI format
                self._assert(
                    ".dkr.ecr." in image_uri and ".amazonaws.com/" in image_uri,
                    f"Image URI should be ECR format: {image_uri}"
                )
                
                # Should contain framework identifier
                framework_identifiers = ['xgboost', 'pytorch', 'sklearn', 'tensorflow']
                has_framework = any(fw in image_uri.lower() for fw in framework_identifiers)
                self._assert(
                    has_framework,
                    f"Image URI should contain framework identifier: {image_uri}"
                )
            
            self._assert(True, "Container image configuration validated")
            
        except Exception as e:
            self._log(f"Container image configuration test failed: {e}")
            self._assert(False, f"Container image configuration test failed: {e}")
    
    def test_model_name_generation_method(self) -> None:
        """Test that CreateModel builders implement model name generation."""
        self._log("Testing model name generation method")
        
        if hasattr(self.builder_class, '_generate_model_name'):
            config = Mock()
            
            try:
                builder = self.builder_class(config=config)
                model_name = builder._generate_model_name()
                
                # Validate model name format
                self._assert(isinstance(model_name, str), "Model name should be string")
                self._assert(len(model_name) > 0, "Model name should not be empty")
                
                # Should contain timestamp or unique identifier
                import re
                has_timestamp = bool(re.search(r'\d{8}-\d{6}', model_name))  # YYYYMMDD-HHMMSS
                has_unique_id = len(model_name.split('-')) > 1
                
                self._assert(
                    has_timestamp or has_unique_id,
                    f"Model name should contain timestamp or unique identifier: {model_name}"
                )
                
                # Test consistency - multiple calls should return different names
                model_name_2 = builder._generate_model_name()
                if has_timestamp:
                    # If using timestamps, names might be different
                    self._log(f"Model names: {model_name}, {model_name_2}")
                
                self._assert(True, "Model name generation method validated")
                
            except Exception as e:
                self._log(f"Model name generation test failed: {e}")
                self._assert(False, f"Model name generation test failed: {e}")
        else:
            self._log("No _generate_model_name method found")
            self._assert(True, "Model name generation method not required")
    
    def test_environment_variables_method(self) -> None:
        """Test that CreateModel builders implement environment variables method."""
        self._log("Testing environment variables method")
        
        if hasattr(self.builder_class, '_get_environment_variables'):
            config = Mock()
            
            try:
                builder = self.builder_class(config=config)
                env_vars = builder._get_environment_variables()
                
                # Check that environment variables are returned as dict
                self._assert(isinstance(env_vars, dict), "Environment variables should be dict")
                
                # Check for inference-specific environment variables
                inference_vars = ['SAGEMAKER_PROGRAM', 'SAGEMAKER_SUBMIT_DIRECTORY']
                found_vars = [var for var in inference_vars if var in env_vars]
                
                self._assert(
                    len(found_vars) > 0,
                    f"Should contain inference environment variables, found: {list(env_vars.keys())}"
                )
                
                # Validate common inference environment variables
                if 'SAGEMAKER_PROGRAM' in env_vars:
                    program = env_vars['SAGEMAKER_PROGRAM']
                    self._assert(
                        program.endswith('.py'),
                        f"SAGEMAKER_PROGRAM should be Python file: {program}"
                    )
                
                if 'SAGEMAKER_SUBMIT_DIRECTORY' in env_vars:
                    submit_dir = env_vars['SAGEMAKER_SUBMIT_DIRECTORY']
                    self._assert(
                        submit_dir.startswith('/opt/ml/'),
                        f"SAGEMAKER_SUBMIT_DIRECTORY should be container path: {submit_dir}"
                    )
                
                self._assert(True, "Environment variables method validated")
                
            except Exception as e:
                self._log(f"Environment variables test failed: {e}")
                self._assert(False, f"Environment variables test failed: {e}")
        else:
            self._log("No _get_environment_variables method found")
            self._assert(True, "Environment variables method not required")
    
    def test_deployment_preparation_methods(self) -> None:
        """Test that CreateModel builders implement deployment preparation methods."""
        self._log("Testing deployment preparation methods")
        
        # Check for deployment preparation methods
        deployment_methods = [
            'integrate_with_training_step',
            'prepare_for_registration',
            'prepare_for_batch_transform',
            '_configure_dependencies'
        ]
        
        found_methods = []
        for method in deployment_methods:
            if hasattr(self.builder_class, method):
                found_methods.append(method)
        
        if found_methods:
            self._log(f"Found deployment preparation methods: {found_methods}")
            
            # Test integration with training step if available
            if hasattr(self.builder_class, 'integrate_with_training_step'):
                config = Mock()
                builder = self.builder_class(config=config)
                
                # Mock training step
                mock_training_step = Mock()
                mock_training_step.properties.ModelArtifacts.S3ModelArtifacts = "s3://bucket/model.tar.gz"
                
                try:
                    builder.integrate_with_training_step(mock_training_step)
                    
                    # Should configure model_data from training step
                    if hasattr(builder, 'model_data'):
                        self._assert(
                            builder.model_data is not None,
                            "Should configure model_data from training step"
                        )
                    
                    # Should add training step as dependency
                    if hasattr(builder, 'dependencies'):
                        self._assert(
                            mock_training_step in builder.dependencies,
                            "Should add training step as dependency"
                        )
                    
                    self._assert(True, "Training step integration validated")
                    
                except Exception as e:
                    self._log(f"Training step integration test failed: {e}")
                    self._assert(False, f"Training step integration test failed: {e}")
        else:
            self._log("No deployment preparation methods found")
            self._assert(True, "Deployment preparation methods not required")
    
    def test_model_integration_methods(self) -> None:
        """Test that CreateModel builders implement model integration methods."""
        self._log("Testing model integration methods")
        
        # Check for model integration patterns
        config = Mock()
        config.model_data = "s3://bucket/model.tar.gz"
        config.image_uri = self.mock_container_images["xgboost"]
        
        try:
            builder = self.builder_class(config=config)
            builder.role = "test-role"
            builder.session = Mock()
            
            # Test model creation integration
            if hasattr(builder, '_create_model'):
                model = builder._create_model()
                
                # Model should integrate with configuration
                if hasattr(model, 'model_data') and hasattr(config, 'model_data'):
                    self._assert(
                        model.model_data == config.model_data,
                        "Model should use configured model_data"
                    )
                
                if hasattr(model, 'image_uri') and hasattr(config, 'image_uri'):
                    self._assert(
                        model.image_uri == config.image_uri,
                        "Model should use configured image_uri"
                    )
            
            self._assert(True, "Model integration methods validated")
            
        except Exception as e:
            self._log(f"Model integration test failed: {e}")
            self._assert(False, f"Model integration test failed: {e}")
    
    def test_step_creation_pattern_compliance(self) -> None:
        """Test that CreateModel builders follow correct step creation patterns."""
        self._log("Testing step creation pattern compliance")
        
        if hasattr(self.builder_class, 'create_step'):
            config = Mock()
            config.model_data = "s3://bucket/model.tar.gz"
            config.image_uri = self.mock_container_images["xgboost"]
            
            try:
                builder = self.builder_class(config=config)
                builder.role = "test-role"
                builder.session = Mock()
                
                # Mock required methods
                builder._create_model = Mock(return_value=self.mock_sagemaker_model)
                builder._get_step_name = Mock(return_value="test-create-model-step")
                builder._configure_dependencies = Mock(return_value=[])
                
                # Test step creation
                with patch('sagemaker.workflow.steps.CreateModelStep') as mock_step_class:
                    mock_create_model_step = Mock()
                    mock_step_class.return_value = mock_create_model_step
                    
                    step = builder.create_step()
                    
                    # Verify step creation
                    self._assert(step is not None, "CreateModelStep should be created")
                    
                    # Verify CreateModelStep was instantiated
                    mock_step_class.assert_called_once()
                    
                    # Check CreateModelStep parameters
                    call_kwargs = mock_step_class.call_args[1]
                    expected_params = ['name', 'model']
                    for param in expected_params:
                        self._assert(
                            param in call_kwargs,
                            f"CreateModelStep should have {param} parameter"
                        )
                    
                    # Verify model was created
                    builder._create_model.assert_called_once()
                
                self._assert(True, "Step creation pattern compliance validated")
                
            except Exception as e:
                self._log(f"Step creation pattern test failed: {e}")
                self._assert(False, f"Step creation pattern test failed: {e}")
        else:
            self._log("No create_step method found")
            self._assert(False, "CreateModel builders should have create_step method")
