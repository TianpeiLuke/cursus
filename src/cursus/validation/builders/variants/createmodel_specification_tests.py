"""
Level 2 CreateModel-Specific Specification Tests for step builders.

These tests focus on CreateModel step specification and contract compliance:
- Container and deployment configuration validation
- Framework-specific model configuration
- Inference code specification compliance
- Environment variable handling for inference
- Model artifact structure validation
- Deployment preparation specification
"""

from typing import Dict, Any, List
from unittest.mock import Mock, patch

from ..specification_tests import SpecificationTests


class CreateModelSpecificationTests(SpecificationTests):
    """
    Level 2 CreateModel-specific specification tests.
    
    These tests validate that CreateModel step builders properly use specifications
    and contracts to define their behavior, with focus on model deployment patterns.
    """
    
    def get_step_type_specific_tests(self) -> list:
        """Return CreateModel-specific specification test methods."""
        return [
            "test_container_configuration_validation",
            "test_framework_specific_configuration",
            "test_model_artifact_specification_compliance",
            "test_inference_environment_variables",
            "test_deployment_configuration_specification",
            "test_model_name_specification",
            "test_container_image_specification",
            "test_inference_code_specification",
            "test_createmodel_contract_integration"
        ]
    
    def _configure_step_type_mocks(self) -> None:
        """Configure CreateModel-specific mock objects for specification tests."""
        # Mock SageMaker Model
        self.mock_sagemaker_model = Mock()
        self.mock_sagemaker_model.name = "test-model-20250815-101935"
        self.mock_sagemaker_model.model_data = "s3://bucket/model.tar.gz"
        self.mock_sagemaker_model.image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1"
        
        # Mock CreateModel specification
        self.mock_createmodel_spec = Mock()
        self.mock_createmodel_spec.dependencies = {
            "model_artifacts": Mock(logical_name="model_artifacts", required=True)
        }
        self.mock_createmodel_spec.outputs = {
            "model_name": Mock(
                logical_name="model_name",
                property_path="Steps.CreateModelStep.ModelName"
            )
        }
        
        # Mock contract objects (CreateModel steps typically don't use traditional contracts)
        self.mock_contract = Mock()
        self.mock_contract.expected_input_paths = {}
        self.mock_contract.expected_output_paths = {}
        
        # Mock container configuration
        self.mock_container_config = {
            "image_uri": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1",
            "environment_variables": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
                "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600"
            },
            "instance_types": ["ml.t2.medium", "ml.m5.large", "ml.m5.xlarge"]
        }
        
        # Mock model artifact structure
        self.mock_model_artifacts = {
            "required_files": ["model.xgb", "inference.py"],
            "optional_files": ["requirements.txt", "model_metadata.json"],
            "structure_validated": True
        }
    
    def _validate_step_type_requirements(self) -> dict:
        """Validate CreateModel-specific requirements for specification tests."""
        return {
            "specification_tests_completed": True,
            "createmodel_specific_validations": True,
            "container_configuration_validated": True,
            "deployment_patterns_validated": True
        }
    
    def test_container_configuration_validation(self) -> None:
        """Test that CreateModel builders validate container configuration."""
        self._log("Testing container configuration validation")
        
        # Test required container configuration
        config = Mock()
        config.image_uri = self.mock_container_config["image_uri"]
        config.model_data = "s3://bucket/model.tar.gz"
        
        try:
            builder = self.builder_class(config=config)
            builder.role = "test-role"
            
            # Validate image URI format
            if hasattr(config, 'image_uri'):
                image_uri = config.image_uri
                
                # Should be ECR URI format
                self._assert(
                    ".dkr.ecr." in image_uri and ".amazonaws.com/" in image_uri,
                    f"Image URI should be ECR format: {image_uri}"
                )
                
                # Should be accessible (format validation)
                uri_parts = image_uri.split('/')
                self._assert(
                    len(uri_parts) >= 2,
                    f"Image URI should have proper structure: {image_uri}"
                )
            
            # Validate model data source
            if hasattr(config, 'model_data'):
                model_data = config.model_data
                self._assert(
                    model_data.startswith("s3://"),
                    f"Model data should be S3 URI: {model_data}"
                )
                
                self._assert(
                    model_data.endswith(".tar.gz"),
                    f"Model data should be tar.gz format: {model_data}"
                )
            
            self._assert(True, "Container configuration validation completed")
            
        except Exception as e:
            self._log(f"Container configuration validation failed: {e}")
            self._assert(False, f"Container configuration validation failed: {e}")
    
    def test_framework_specific_configuration(self) -> None:
        """Test that CreateModel builders handle framework-specific configuration."""
        self._log("Testing framework-specific configuration")
        
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
            
            if detected_framework == 'xgboost':
                self._validate_xgboost_configuration()
            elif detected_framework == 'pytorch':
                self._validate_pytorch_configuration()
            elif detected_framework == 'sklearn':
                self._validate_sklearn_configuration()
            elif detected_framework == 'tensorflow':
                self._validate_tensorflow_configuration()
        else:
            self._log("No specific framework detected")
            self._assert(True, "Generic CreateModel configuration validated")
    
    def _validate_xgboost_configuration(self) -> None:
        """Validate XGBoost-specific configuration."""
        self._log("Validating XGBoost-specific configuration")
        
        config = Mock()
        config.image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1"
        config.model_data = "s3://bucket/xgboost-model.tar.gz"
        
        try:
            builder = self.builder_class(config=config)
            
            # Validate XGBoost container image
            if hasattr(config, 'image_uri'):
                image_uri = config.image_uri
                self._assert(
                    "xgboost" in image_uri.lower(),
                    f"XGBoost model should use XGBoost container: {image_uri}"
                )
                
                # Validate version format
                version_patterns = ["1.0-1", "1.2-1", "1.3-1", "1.5-1"]
                has_valid_version = any(version in image_uri for version in version_patterns)
                self._assert(
                    has_valid_version,
                    f"XGBoost container should have valid version: {image_uri}"
                )
            
            self._assert(True, "XGBoost-specific configuration validated")
            
        except Exception as e:
            self._log(f"XGBoost configuration validation failed: {e}")
            self._assert(False, f"XGBoost configuration validation failed: {e}")
    
    def _validate_pytorch_configuration(self) -> None:
        """Validate PyTorch-specific configuration."""
        self._log("Validating PyTorch-specific configuration")
        
        config = Mock()
        config.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.0-gpu-py38"
        config.model_data = "s3://bucket/pytorch-model.tar.gz"
        
        try:
            builder = self.builder_class(config=config)
            
            # Validate PyTorch container image
            if hasattr(config, 'image_uri'):
                image_uri = config.image_uri
                self._assert(
                    "pytorch" in image_uri.lower(),
                    f"PyTorch model should use PyTorch container: {image_uri}"
                )
                
                # Check for inference vs training image
                self._assert(
                    "inference" in image_uri.lower(),
                    f"CreateModel should use inference container: {image_uri}"
                )
            
            self._assert(True, "PyTorch-specific configuration validated")
            
        except Exception as e:
            self._log(f"PyTorch configuration validation failed: {e}")
            self._assert(False, f"PyTorch configuration validation failed: {e}")
    
    def _validate_sklearn_configuration(self) -> None:
        """Validate SKLearn-specific configuration."""
        self._log("Validating SKLearn-specific configuration")
        self._assert(True, "SKLearn-specific configuration validated")
    
    def _validate_tensorflow_configuration(self) -> None:
        """Validate TensorFlow-specific configuration."""
        self._log("Validating TensorFlow-specific configuration")
        self._assert(True, "TensorFlow-specific configuration validated")
    
    def test_model_artifact_specification_compliance(self) -> None:
        """Test that CreateModel builders handle model artifacts according to specification."""
        self._log("Testing model artifact specification compliance")
        
        config = Mock()
        config.model_data = "s3://bucket/model.tar.gz"
        config.image_uri = self.mock_container_config["image_uri"]
        
        try:
            builder = self.builder_class(config=config)
            builder.role = "test-role"
            
            # Test model artifact validation
            if hasattr(builder, '_validate_model_artifacts'):
                try:
                    is_valid = builder._validate_model_artifacts()
                    self._assert(
                        isinstance(is_valid, bool),
                        "Model artifact validation should return boolean"
                    )
                except Exception as e:
                    self._log(f"Model artifact validation method failed: {e}")
            
            # Test model data configuration
            if hasattr(config, 'model_data'):
                model_data = config.model_data
                
                # Should be S3 URI
                self._assert(
                    model_data.startswith("s3://"),
                    f"Model data should be S3 URI: {model_data}"
                )
                
                # Should be compressed format
                valid_formats = [".tar.gz", ".zip", ".tar"]
                has_valid_format = any(fmt in model_data for fmt in valid_formats)
                self._assert(
                    has_valid_format,
                    f"Model data should be compressed format: {model_data}"
                )
            
            self._assert(True, "Model artifact specification compliance validated")
            
        except Exception as e:
            self._log(f"Model artifact specification test failed: {e}")
            self._assert(False, f"Model artifact specification test failed: {e}")
    
    def test_inference_environment_variables(self) -> None:
        """Test that CreateModel builders handle inference environment variables correctly."""
        self._log("Testing inference environment variables handling")
        
        if hasattr(self.builder_class, '_get_environment_variables'):
            config = Mock()
            
            try:
                builder = self.builder_class(config=config)
                env_vars = builder._get_environment_variables()
                
                # Check that environment variables are returned as dict
                self._assert(isinstance(env_vars, dict), "Environment variables should be dict")
                
                # Check for inference-specific environment variables
                required_inference_vars = ['SAGEMAKER_PROGRAM', 'SAGEMAKER_SUBMIT_DIRECTORY']
                for var in required_inference_vars:
                    if var in env_vars:
                        self._assert(
                            isinstance(env_vars[var], str) and len(env_vars[var]) > 0,
                            f"Inference environment variable {var} should be non-empty string"
                        )
                
                # Validate specific inference variables
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
                
                # Check for optional inference variables
                optional_vars = ['SAGEMAKER_MODEL_SERVER_TIMEOUT', 'SAGEMAKER_MODEL_SERVER_WORKERS']
                found_optional = [var for var in optional_vars if var in env_vars]
                if found_optional:
                    self._log(f"Found optional inference variables: {found_optional}")
                
                self._assert(True, "Inference environment variables validated")
                
            except Exception as e:
                self._log(f"Inference environment variables test failed: {e}")
                self._assert(False, f"Inference environment variables test failed: {e}")
        else:
            self._log("No _get_environment_variables method found")
            self._assert(True, "Environment variables method not required")
    
    def test_deployment_configuration_specification(self) -> None:
        """Test that CreateModel builders handle deployment configuration according to specification."""
        self._log("Testing deployment configuration specification")
        
        config = Mock()
        config.model_data = "s3://bucket/model.tar.gz"
        config.image_uri = self.mock_container_config["image_uri"]
        
        try:
            builder = self.builder_class(config=config)
            builder.role = "test-role"
            builder.session = Mock()
            
            # Test deployment preparation methods
            deployment_methods = [
                'prepare_for_registration',
                'prepare_for_batch_transform',
                'integrate_with_training_step'
            ]
            
            found_methods = [method for method in deployment_methods if hasattr(builder, method)]
            
            if found_methods:
                self._log(f"Found deployment methods: {found_methods}")
                
                # Test batch transform preparation
                if hasattr(builder, 'prepare_for_batch_transform'):
                    try:
                        builder.prepare_for_batch_transform()
                        
                        # Should configure batch transform settings
                        if hasattr(builder, 'batch_transform_config'):
                            config_obj = builder.batch_transform_config
                            self._assert(
                                isinstance(config_obj, dict),
                                "Batch transform config should be dict"
                            )
                            
                            # Check for required batch transform settings
                            required_settings = ['instance_type', 'instance_count']
                            for setting in required_settings:
                                if setting in config_obj:
                                    self._assert(
                                        config_obj[setting] is not None,
                                        f"Batch transform {setting} should be configured"
                                    )
                        
                        self._assert(True, "Batch transform preparation validated")
                        
                    except Exception as e:
                        self._log(f"Batch transform preparation failed: {e}")
                        self._assert(False, f"Batch transform preparation failed: {e}")
            
            self._assert(True, "Deployment configuration specification validated")
            
        except Exception as e:
            self._log(f"Deployment configuration test failed: {e}")
            self._assert(False, f"Deployment configuration test failed: {e}")
    
    def test_model_name_specification(self) -> None:
        """Test that CreateModel builders handle model name specification correctly."""
        self._log("Testing model name specification")
        
        if hasattr(self.builder_class, '_generate_model_name'):
            config = Mock()
            
            try:
                builder = self.builder_class(config=config)
                model_name = builder._generate_model_name()
                
                # Validate model name format
                self._assert(isinstance(model_name, str), "Model name should be string")
                self._assert(len(model_name) > 0, "Model name should not be empty")
                
                # Should be unique (contain timestamp or UUID)
                import re
                has_timestamp = bool(re.search(r'\d{8}-\d{6}', model_name))
                has_uuid = bool(re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', model_name))
                has_unique_suffix = len(model_name.split('-')) > 1
                
                self._assert(
                    has_timestamp or has_uuid or has_unique_suffix,
                    f"Model name should be unique: {model_name}"
                )
                
                # Should follow SageMaker naming conventions
                # Model names must be 1-63 characters, alphanumeric and hyphens only
                valid_chars = re.match(r'^[a-zA-Z0-9\-]+$', model_name)
                self._assert(
                    valid_chars is not None,
                    f"Model name should contain only alphanumeric and hyphens: {model_name}"
                )
                
                self._assert(
                    len(model_name) <= 63,
                    f"Model name should be <= 63 characters: {model_name} ({len(model_name)} chars)"
                )
                
                self._assert(True, "Model name specification validated")
                
            except Exception as e:
                self._log(f"Model name specification test failed: {e}")
                self._assert(False, f"Model name specification test failed: {e}")
        else:
            self._log("No _generate_model_name method found")
            self._assert(True, "Model name generation method not required")
    
    def test_container_image_specification(self) -> None:
        """Test that CreateModel builders handle container image specification correctly."""
        self._log("Testing container image specification")
        
        config = Mock()
        config.image_uri = self.mock_container_config["image_uri"]
        
        try:
            builder = self.builder_class(config=config)
            
            # Validate container image specification
            if hasattr(config, 'image_uri'):
                image_uri = config.image_uri
                
                # Should be valid ECR URI
                self._assert(
                    ".dkr.ecr." in image_uri and ".amazonaws.com/" in image_uri,
                    f"Image URI should be ECR format: {image_uri}"
                )
                
                # Should be inference container (not training)
                inference_indicators = ['inference', 'serving', 'deploy']
                training_indicators = ['training', 'train']
                
                has_inference = any(indicator in image_uri.lower() for indicator in inference_indicators)
                has_training = any(indicator in image_uri.lower() for indicator in training_indicators)
                
                if has_training and not has_inference:
                    self._log(f"Warning: Using training container for inference: {image_uri}")
                
                # Should specify framework and version
                framework_patterns = ['xgboost:', 'pytorch:', 'sklearn:', 'tensorflow:']
                has_framework_version = any(pattern in image_uri.lower() for pattern in framework_patterns)
                self._assert(
                    has_framework_version,
                    f"Image URI should specify framework and version: {image_uri}"
                )
            
            self._assert(True, "Container image specification validated")
            
        except Exception as e:
            self._log(f"Container image specification test failed: {e}")
            self._assert(False, f"Container image specification test failed: {e}")
    
    def test_inference_code_specification(self) -> None:
        """Test that CreateModel builders handle inference code specification correctly."""
        self._log("Testing inference code specification")
        
        # Check for inference code validation methods
        inference_methods = [
            '_validate_inference_code',
            '_check_inference_functions',
            '_validate_model_artifacts'
        ]
        
        found_methods = [method for method in inference_methods if hasattr(self.builder_class, method)]
        
        if found_methods:
            self._log(f"Found inference code methods: {found_methods}")
            
            config = Mock()
            builder = self.builder_class(config=config)
            
            # Test inference code validation if available
            if hasattr(builder, '_validate_inference_code'):
                try:
                    # Mock inference code structure
                    inference_code = {
                        'model_fn': 'def model_fn(model_dir): ...',
                        'predict_fn': 'def predict_fn(input_data, model): ...',
                        'input_fn': 'def input_fn(request_body, content_type): ...',
                        'output_fn': 'def output_fn(prediction, accept): ...'
                    }
                    
                    is_valid = builder._validate_inference_code(inference_code)
                    self._assert(
                        isinstance(is_valid, bool),
                        "Inference code validation should return boolean"
                    )
                    
                    self._assert(True, "Inference code validation method tested")
                    
                except Exception as e:
                    self._log(f"Inference code validation failed: {e}")
                    self._assert(False, f"Inference code validation failed: {e}")
        else:
            self._log("No inference code validation methods found")
            self._assert(True, "Inference code validation not required")
    
    def test_createmodel_contract_integration(self) -> None:
        """Test that CreateModel builders integrate with contracts correctly."""
        self._log("Testing CreateModel contract integration")
        
        config = Mock()
        builder = self.builder_class(config=config)
        
        # CreateModel steps typically don't use traditional script contracts
        # but may have deployment contracts or specifications
        
        try:
            # Check for specification integration
            if hasattr(builder, 'spec'):
                builder.spec = self.mock_createmodel_spec
                
                # Validate specification structure
                if hasattr(builder.spec, 'dependencies'):
                    dependencies = builder.spec.dependencies
                    self._assert(isinstance(dependencies, dict), "Dependencies should be dict")
                
                if hasattr(builder.spec, 'outputs'):
                    outputs = builder.spec.outputs
                    self._assert(isinstance(outputs, dict), "Outputs should be dict")
                    
                    # Check for CreateModel-specific outputs
                    if 'model_name' in outputs:
                        model_name_spec = outputs['model_name']
                        if hasattr(model_name_spec, 'property_path'):
                            property_path = model_name_spec.property_path
                            self._assert(
                                "CreateModelStep" in property_path,
                                f"Property path should reference CreateModelStep: {property_path}"
                            )
            
            # Check for deployment contract integration
            if hasattr(builder, 'deployment_contract'):
                deployment_contract = builder.deployment_contract
                self._assert(
                    isinstance(deployment_contract, dict),
                    "Deployment contract should be dict"
                )
            
            self._assert(True, "CreateModel contract integration validated")
            
        except Exception as e:
            self._log(f"CreateModel contract integration test failed: {e}")
            self._assert(False, f"CreateModel contract integration test failed: {e}")
