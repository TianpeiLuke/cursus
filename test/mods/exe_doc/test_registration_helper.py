"""
Tests for Registration Helper.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from cursus.mods.exe_doc.registration_helper import RegistrationHelper
from cursus.mods.exe_doc.base import ExecutionDocumentGenerationError


class TestRegistrationHelper:
    """Tests for RegistrationHelper class."""
    
    def test_init_success(self):
        """Test successful initialization."""
        helper = RegistrationHelper()
        assert helper is not None
        assert helper.logger is not None
    
    def test_can_handle_step_registration_config(self):
        """Test can_handle_step with registration configuration."""
        helper = RegistrationHelper()
        
        # Mock config with registration-like class name
        mock_config = Mock()
        mock_config.__class__.__name__ = "RegistrationConfig"
        
        assert helper.can_handle_step("test_step", mock_config) is True
    
    def test_can_handle_step_registration_step_name(self):
        """Test can_handle_step with registration step name."""
        helper = RegistrationHelper()
        
        # Mock config with non-registration class name but registration step name
        mock_config = Mock()
        mock_config.__class__.__name__ = "SomeOtherConfig"
        
        assert helper.can_handle_step("model_registration", mock_config) is True
        assert helper.can_handle_step("register_model", mock_config) is True
        assert helper.can_handle_step("Registration_us_east_1", mock_config) is True
    
    def test_can_handle_step_payload_config_excluded(self):
        """Test can_handle_step excludes payload configurations."""
        helper = RegistrationHelper()
        
        # Mock config with payload in name (should be excluded)
        mock_config = Mock()
        mock_config.__class__.__name__ = "RegistrationPayloadConfig"
        
        assert helper.can_handle_step("test_step", mock_config) is False
    
    def test_can_handle_step_non_registration_config(self):
        """Test can_handle_step with non-registration configuration."""
        helper = RegistrationHelper()
        
        # Mock config with non-registration class name and step name
        mock_config = Mock()
        mock_config.__class__.__name__ = "XGBoostTrainingConfig"
        
        assert helper.can_handle_step("training_step", mock_config) is False
    
    @patch('cursus.mods.exe_doc.registration_helper.SAGEMAKER_AVAILABLE', True)
    def test_extract_step_config_success(self):
        """Test successful step config extraction."""
        helper = RegistrationHelper()
        
        # Create mock config
        mock_config = self._create_mock_registration_config()
        
        # Mock the helper methods
        with patch.object(helper, '_get_image_uri', return_value="test-image-uri") as mock_get_uri:
            with patch.object(helper, '_create_execution_doc_config', return_value={"test": "config"}) as mock_create_config:
                
                result = helper.extract_step_config("registration_step", mock_config)
                
                assert result == {"test": "config"}
                mock_get_uri.assert_called_once_with(mock_config)
                # Check that _create_execution_doc_config was called with the new signature
                mock_create_config.assert_called_once_with("test-image-uri", {"registration": mock_config})
    
    def test_extract_step_config_failure(self):
        """Test step config extraction when _get_image_uri fails."""
        helper = RegistrationHelper()
        
        mock_config = Mock()
        
        with patch.object(helper, '_get_image_uri', side_effect=ValueError("Image URI failed")):
            with pytest.raises(ExecutionDocumentGenerationError, match="Registration configuration extraction failed"):
                helper.extract_step_config("registration_step", mock_config)
    
    @patch('cursus.mods.exe_doc.registration_helper.SAGEMAKER_AVAILABLE', False)
    def test_get_image_uri_sagemaker_not_available(self):
        """Test _get_image_uri when SageMaker is not available."""
        helper = RegistrationHelper()
        mock_config = Mock()
        
        result = helper._get_image_uri(mock_config)
        
        assert result == "image-uri-placeholder"
    
    @patch('cursus.mods.exe_doc.registration_helper.SAGEMAKER_AVAILABLE', True)
    def test_get_image_uri_missing_attributes(self):
        """Test _get_image_uri with missing required attributes."""
        helper = RegistrationHelper()
        
        # Mock config missing required attributes
        mock_config = Mock()
        mock_config.framework = "pytorch"
        # Missing other required attributes
        
        result = helper._get_image_uri(mock_config)
        
        assert result == "image-uri-placeholder"
    
    @patch('cursus.mods.exe_doc.registration_helper.SAGEMAKER_AVAILABLE', True)
    @patch('cursus.mods.exe_doc.registration_helper.retrieve_image_uri')
    def test_get_image_uri_success(self, mock_retrieve):
        """Test successful _get_image_uri."""
        helper = RegistrationHelper()
        
        # Create comprehensive mock config
        mock_config = self._create_mock_registration_config()
        mock_retrieve.return_value = "123456789012.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.8.1-gpu-py3"
        
        result = helper._get_image_uri(mock_config)
        
        assert result == "123456789012.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.8.1-gpu-py3"
        mock_retrieve.assert_called_once_with(
            framework="pytorch",
            region="us-east-1",
            version="1.8.1",
            py_version="py3",
            instance_type="ml.m5.large",
            image_scope="inference",
        )
    
    @patch('cursus.mods.exe_doc.registration_helper.SAGEMAKER_AVAILABLE', True)
    @patch('cursus.mods.exe_doc.registration_helper.retrieve_image_uri')
    def test_get_image_uri_retrieve_failure(self, mock_retrieve):
        """Test _get_image_uri when retrieve fails."""
        helper = RegistrationHelper()
        
        mock_config = self._create_mock_registration_config()
        mock_retrieve.side_effect = Exception("Retrieve failed")
        
        result = helper._get_image_uri(mock_config)
        
        assert result == "image-uri-placeholder"
    
    def test_create_execution_doc_config_basic(self):
        """Test basic _create_execution_doc_config."""
        helper = RegistrationHelper()
        
        # Create a custom class to avoid Mock's automatic attribute creation
        class BasicRegistrationConfig:
            def __init__(self):
                self.__class__.__name__ = "RegistrationConfig"
                self.model_domain = "computer_vision"
                self.model_objective = "image_classification"
                self.aws_region = "us-east-1"
                self.region = "us-west-2"
                self.model_owner = "test-owner"
        
        mock_config = BasicRegistrationConfig()
        
        # Use new signature: (image_uri, configs_dict)
        configs_dict = {"registration": mock_config}
        result = helper._create_execution_doc_config("test-image-uri", configs_dict)
        
        expected = {
            "source_model_inference_image_arn": "test-image-uri",
            "model_domain": "computer_vision",
            "model_objective": "image_classification",
            "source_model_region": "us-east-1",  # Mapped from aws_region
            "model_registration_region": "us-west-2",  # Mapped from region
            "model_owner": "test-owner",
        }
        
        assert result == expected
    
    def test_create_execution_doc_config_with_entry_point(self):
        """Test _create_execution_doc_config with inference entry point."""
        helper = RegistrationHelper()
        
        mock_config = Mock()
        mock_config.__class__.__name__ = "RegistrationConfig"  # Set proper class name
        mock_config.model_domain = "nlp"
        mock_config.model_objective = "text_classification"
        mock_config.inference_entry_point = "inference.py"
        mock_config.aws_region = "us-east-1"
        
        # Use new signature: (image_uri, configs_dict)
        configs_dict = {"registration": mock_config}
        result = helper._create_execution_doc_config("test-image-uri", configs_dict)
        
        assert "source_model_environment_variable_map" in result
        env_vars = result["source_model_environment_variable_map"]
        assert env_vars["SAGEMAKER_CONTAINER_LOG_LEVEL"] == "20"
        assert env_vars["SAGEMAKER_PROGRAM"] == "inference.py"
        assert env_vars["SAGEMAKER_REGION"] == "us-east-1"
        assert env_vars["SAGEMAKER_SUBMIT_DIRECTORY"] == "/opt/ml/model/code"
    
    def test_create_execution_doc_config_with_all_fields(self):
        """Test _create_execution_doc_config with all possible fields."""
        helper = RegistrationHelper()
        
        mock_config = self._create_comprehensive_mock_config()
        
        # Use new signature: (image_uri, configs_dict)
        configs_dict = {"registration": mock_config}
        result = helper._create_execution_doc_config("test-image-uri", configs_dict)
        
        # Check all fields are present
        assert result["source_model_inference_image_arn"] == "test-image-uri"
        assert result["model_domain"] == "computer_vision"
        assert result["model_objective"] == "image_classification"
        assert result["source_model_inference_content_types"] == ["application/json"]
        assert result["source_model_inference_response_types"] == ["application/json"]
        assert result["source_model_inference_input_variable_list"] == ["input"]
        assert result["source_model_inference_output_variable_list"] == ["output"]
        assert result["source_model_region"] == "us-east-1"
        assert result["model_registration_region"] == "us-west-2"
        assert result["model_owner"] == "test-owner"
    
    def test_create_execution_doc_config_with_related_configs(self):
        """Test create_execution_doc_config_with_related_configs."""
        helper = RegistrationHelper()
        
        # Create mock configs
        registration_config = self._create_mock_registration_config()
        payload_config = self._create_mock_payload_config()
        package_config = self._create_mock_package_config()
        
        with patch.object(helper, '_get_image_uri', return_value="test-image-uri"):
            result = helper.create_execution_doc_config_with_related_configs(
                registration_config, payload_config, package_config
            )
        
        # Check basic config is present
        assert result["source_model_inference_image_arn"] == "test-image-uri"
        assert result["model_domain"] == "computer_vision"
        
        # Check load testing info is added
        assert "load_testing_info_map" in result
        load_testing = result["load_testing_info_map"]
        assert load_testing["sample_payload_s3_key"] == "test-payload.json"
        assert load_testing["expected_tps"] == 100
        assert load_testing["max_latency_in_millisecond"] == 1000
        assert load_testing["max_acceptable_error_rate"] == 0.01
        assert load_testing["instance_type_list"] == ["ml.m5.large"]
    
    def test_create_execution_doc_config_with_related_configs_no_payload(self):
        """Test create_execution_doc_config_with_related_configs without payload config."""
        helper = RegistrationHelper()
        
        registration_config = self._create_mock_registration_config()
        
        with patch.object(helper, '_get_image_uri', return_value="test-image-uri"):
            result = helper.create_execution_doc_config_with_related_configs(
                registration_config, None, None
            )
        
        # Check basic config is present
        assert result["source_model_inference_image_arn"] == "test-image-uri"
        assert result["model_domain"] == "computer_vision"
        
        # Check load testing info is NOT added
        assert "load_testing_info_map" not in result
    
    def test_find_registration_step_patterns(self):
        """Test find_registration_step_patterns."""
        helper = RegistrationHelper()
        
        step_names = [
            "training_step",
            "ModelRegistration-us-east-1",
            "model_registration",
            "some_other_step",
            "registration_final",
        ]
        
        result = helper.find_registration_step_patterns(step_names, region="us-east-1")
        
        expected = [
            "ModelRegistration-us-east-1",
            "model_registration", 
            "registration_final",
        ]
        
        assert set(result) == set(expected)
    
    def test_find_registration_step_patterns_no_region(self):
        """Test find_registration_step_patterns without region."""
        helper = RegistrationHelper()
        
        step_names = [
            "training_step",
            "Registration",
            "register_model",
            "some_other_step",
        ]
        
        result = helper.find_registration_step_patterns(step_names)
        
        expected = ["Registration", "register_model"]
        
        assert set(result) == set(expected)
    
    def test_find_registration_step_patterns_empty(self):
        """Test find_registration_step_patterns with no matches."""
        helper = RegistrationHelper()
        
        step_names = ["training_step", "evaluation_step", "preprocessing_step"]
        
        result = helper.find_registration_step_patterns(step_names, region="us-east-1")
        
        assert result == []
    
    def test_validate_registration_config_success(self):
        """Test successful validate_registration_config."""
        helper = RegistrationHelper()
        
        mock_config = Mock()
        mock_config.model_domain = "computer_vision"
        mock_config.model_objective = "image_classification"
        mock_config.region = "us-east-1"
        
        result = helper.validate_registration_config(mock_config)
        
        assert result is True
    
    def test_validate_registration_config_missing_fields(self):
        """Test validate_registration_config with missing fields."""
        helper = RegistrationHelper()
        
        # Create a custom object that only has specific attributes
        class PartialConfig:
            def __init__(self):
                self.model_domain = "computer_vision"
                # Missing model_objective and region
        
        mock_config = PartialConfig()
        
        result = helper.validate_registration_config(mock_config)
        
        assert result is False
    
    def test_validate_registration_config_empty_config(self):
        """Test validate_registration_config with empty config."""
        helper = RegistrationHelper()
        
        # Create a custom object with no attributes
        class EmptyConfig:
            pass
        
        mock_config = EmptyConfig()
        
        result = helper.validate_registration_config(mock_config)
        
        assert result is False
    
    def _create_mock_registration_config(self):
        """Create a mock registration configuration for testing."""
        mock_config = Mock()
        mock_config.__class__.__name__ = "RegistrationConfig"  # Set proper class name
        mock_config.framework = "pytorch"
        mock_config.aws_region = "us-east-1"
        mock_config.framework_version = "1.8.1"
        mock_config.py_version = "py3"
        mock_config.inference_instance_type = "ml.m5.large"
        mock_config.model_domain = "computer_vision"
        mock_config.model_objective = "image_classification"
        mock_config.region = "us-west-2"
        mock_config.model_owner = "test-owner"
        mock_config.bucket = "test-bucket"
        return mock_config
    
    def _create_comprehensive_mock_config(self):
        """Create a comprehensive mock config with all fields."""
        mock_config = self._create_mock_registration_config()
        mock_config.source_model_inference_content_types = ["application/json"]
        mock_config.source_model_inference_response_types = ["application/json"]
        mock_config.source_model_inference_input_variable_list = ["input"]
        mock_config.source_model_inference_output_variable_list = ["output"]
        mock_config.model_registration_region = "us-west-2"
        mock_config.source_model_region = "us-east-1"
        mock_config.inference_entry_point = "inference.py"
        return mock_config
    
    def _create_mock_payload_config(self):
        """Create a mock payload configuration for testing."""
        mock_config = Mock()
        mock_config.__class__.__name__ = "PayloadConfig"  # Set proper class name
        mock_config.sample_payload_s3_key = "test-payload.json"
        mock_config.expected_tps = 100
        mock_config.max_latency_in_millisecond = 1000
        mock_config.max_acceptable_error_rate = 0.01
        return mock_config
    
    def _create_mock_package_config(self):
        """Create a mock package configuration for testing."""
        mock_config = Mock()
        mock_config.__class__.__name__ = "PackageConfig"  # Set proper class name
        mock_config.get_instance_type = Mock(return_value="ml.m5.large")
        return mock_config
