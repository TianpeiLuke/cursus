"""
Tests for execution document generation utilities.
"""

import pytest
from unittest.mock import Mock, patch

from cursus.mods.exe_doc.utils import (
    determine_step_type,
    validate_execution_document_structure,
    create_execution_document_template,
    merge_execution_documents,
)


class TestDetermineStepType:
    """Tests for determine_step_type function."""
    
    @patch('cursus.registry.step_names.get_config_step_registry')
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_cradle_step_type_from_registry(self, mock_get_sagemaker_type, mock_get_config_registry):
        """Test step type determination using registry system."""
        config = Mock()
        config.__class__.__name__ = "CradleDataLoadingConfig"
        
        # Mock registry responses
        mock_get_config_registry.return_value = {"CradleDataLoadingConfig": "CradleDataLoading"}
        mock_get_sagemaker_type.return_value = "CradleDataLoading"
        
        step_types = determine_step_type("data_loading", config)
        
        assert "PROCESSING_STEP" in step_types
        assert "CradleDataLoading" in step_types
    
    @patch('cursus.registry.step_names.get_config_step_registry')
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_registration_step_type_from_registry(self, mock_get_sagemaker_type, mock_get_config_registry):
        """Test step type determination for registration steps using registry."""
        config = Mock()
        config.__class__.__name__ = "RegistrationConfig"
        
        # Mock registry responses
        mock_get_config_registry.return_value = {"RegistrationConfig": "Registration"}
        mock_get_sagemaker_type.return_value = "MimsModelRegistrationProcessing"
        
        step_types = determine_step_type("model_reg", config)
        
        assert "PROCESSING_STEP" in step_types
        assert "ModelRegistration" in step_types
    
    @patch('cursus.registry.step_names.get_config_step_registry')
    @patch('cursus.registry.step_names.get_canonical_name_from_file_name')
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_step_name_resolution_fallback(self, mock_get_sagemaker_type, mock_get_canonical_name, mock_get_config_registry):
        """Test step name resolution using file name fallback."""
        config = Mock()
        config.__class__.__name__ = "SomeConfig"
        
        # Mock registry responses - config not found, but file name resolution works
        mock_get_config_registry.return_value = {}
        mock_get_canonical_name.return_value = "XGBoostTraining"
        mock_get_sagemaker_type.return_value = "Training"
        
        step_types = determine_step_type("xgboost_training", config)
        
        assert "PROCESSING_STEP" in step_types
        assert "Training" in step_types
    
    @patch('cursus.registry.step_names.get_config_step_registry')
    def test_fallback_to_legacy_logic(self, mock_get_config_registry):
        """Test fallback to legacy logic when registry fails."""
        config = Mock()
        config.__class__.__name__ = "CradleDataLoadingConfig"
        
        # Mock registry failure
        mock_get_config_registry.side_effect = Exception("Registry failed")
        
        step_types = determine_step_type("cradle_data_loading", config)
        
        assert "PROCESSING_STEP" in step_types
        assert "CradleDataLoading" in step_types
    
    def test_fallback_logic_directly(self):
        """Test the fallback logic function directly."""
        from cursus.mods.exe_doc.utils import _determine_step_type_fallback
        
        config = Mock()
        config.__class__.__name__ = "RegistrationConfig"
        
        step_types = _determine_step_type_fallback("model_registration", config)
        
        assert "PROCESSING_STEP" in step_types
        assert "ModelRegistration" in step_types
    
    def test_fallback_default_step_type(self):
        """Test default step type for unknown steps in fallback."""
        from cursus.mods.exe_doc.utils import _determine_step_type_fallback
        
        config = Mock()
        config.__class__.__name__ = "UnknownConfig"
        
        step_types = _determine_step_type_fallback("unknown_step", config)
        
        assert step_types == ["PROCESSING_STEP"]


class TestValidateExecutionDocumentStructure:
    """Tests for validate_execution_document_structure function."""
    
    def test_valid_execution_document(self):
        """Test validation of valid execution document."""
        doc = {
            "PIPELINE_STEP_CONFIGS": {
                "step1": {"STEP_TYPE": ["PROCESSING_STEP"], "STEP_CONFIG": {}}
            }
        }
        
        assert validate_execution_document_structure(doc) is True
    
    def test_invalid_document_not_dict(self):
        """Test validation fails for non-dictionary input."""
        assert validate_execution_document_structure("not a dict") is False
    
    def test_invalid_document_missing_key(self):
        """Test validation fails for missing PIPELINE_STEP_CONFIGS key."""
        doc = {"OTHER_KEY": {}}
        
        assert validate_execution_document_structure(doc) is False
    
    def test_invalid_document_wrong_value_type(self):
        """Test validation fails for wrong value type."""
        doc = {"PIPELINE_STEP_CONFIGS": "not a dict"}
        
        assert validate_execution_document_structure(doc) is False


class TestCreateExecutionDocumentTemplate:
    """Tests for create_execution_document_template function."""
    
    def test_create_template_single_step(self):
        """Test template creation for single step."""
        template = create_execution_document_template(["step1"])
        
        expected = {
            "PIPELINE_STEP_CONFIGS": {
                "step1": {
                    "STEP_TYPE": ["PROCESSING_STEP"],
                    "STEP_CONFIG": {}
                }
            }
        }
        
        assert template == expected
    
    def test_create_template_multiple_steps(self):
        """Test template creation for multiple steps."""
        template = create_execution_document_template(["step1", "step2"])
        
        assert "step1" in template["PIPELINE_STEP_CONFIGS"]
        assert "step2" in template["PIPELINE_STEP_CONFIGS"]
        assert len(template["PIPELINE_STEP_CONFIGS"]) == 2
    
    def test_create_template_empty_list(self):
        """Test template creation for empty step list."""
        template = create_execution_document_template([])
        
        expected = {"PIPELINE_STEP_CONFIGS": {}}
        
        assert template == expected


class TestMergeExecutionDocuments:
    """Tests for merge_execution_documents function."""
    
    def test_merge_documents_no_overlap(self):
        """Test merging documents with no overlapping steps."""
        base_doc = {
            "PIPELINE_STEP_CONFIGS": {
                "step1": {"STEP_TYPE": ["PROCESSING_STEP"], "STEP_CONFIG": {"key1": "value1"}}
            }
        }
        
        additional_doc = {
            "PIPELINE_STEP_CONFIGS": {
                "step2": {"STEP_TYPE": ["PROCESSING_STEP"], "STEP_CONFIG": {"key2": "value2"}}
            }
        }
        
        merged = merge_execution_documents(base_doc, additional_doc)
        
        assert "step1" in merged["PIPELINE_STEP_CONFIGS"]
        assert "step2" in merged["PIPELINE_STEP_CONFIGS"]
        assert len(merged["PIPELINE_STEP_CONFIGS"]) == 2
    
    def test_merge_documents_with_overlap(self):
        """Test merging documents with overlapping steps."""
        base_doc = {
            "PIPELINE_STEP_CONFIGS": {
                "step1": {"STEP_TYPE": ["PROCESSING_STEP"], "STEP_CONFIG": {"key1": "value1"}}
            }
        }
        
        additional_doc = {
            "PIPELINE_STEP_CONFIGS": {
                "step1": {"STEP_CONFIG": {"key2": "value2"}}
            }
        }
        
        merged = merge_execution_documents(base_doc, additional_doc)
        
        step1_config = merged["PIPELINE_STEP_CONFIGS"]["step1"]
        assert step1_config["STEP_TYPE"] == ["PROCESSING_STEP"]
        assert step1_config["STEP_CONFIG"]["key1"] == "value1"
        assert step1_config["STEP_CONFIG"]["key2"] == "value2"
    
    def test_merge_documents_invalid_base(self):
        """Test merging fails with invalid base document."""
        base_doc = {"INVALID": "structure"}
        additional_doc = {"PIPELINE_STEP_CONFIGS": {}}
        
        with pytest.raises(ValueError, match="Invalid base execution document structure"):
            merge_execution_documents(base_doc, additional_doc)
    
    def test_merge_documents_invalid_additional(self):
        """Test merging fails with invalid additional document."""
        base_doc = {"PIPELINE_STEP_CONFIGS": {}}
        additional_doc = {"INVALID": "structure"}
        
        with pytest.raises(ValueError, match="Invalid additional execution document structure"):
            merge_execution_documents(base_doc, additional_doc)
