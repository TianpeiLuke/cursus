"""
Unit tests for step_catalog.models module.

Tests the data models used in the unified step catalog system:
- FileMetadata
- StepInfo  
- StepSearchResult
"""

import pytest
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from cursus.step_catalog.models import FileMetadata, StepInfo, StepSearchResult


class TestFileMetadata:
    """Test cases for FileMetadata model."""
    
    def test_file_metadata_creation(self):
        """Test basic FileMetadata creation."""
        path = Path("/test/path/script.py")
        modified_time = datetime.now()
        
        metadata = FileMetadata(
            path=path,
            file_type="script",
            modified_time=modified_time
        )
        
        assert metadata.path == path
        assert metadata.file_type == "script"
        assert metadata.modified_time == modified_time
    
    def test_file_metadata_immutable(self):
        """Test that FileMetadata is frozen/immutable."""
        metadata = FileMetadata(
            path=Path("/test/path/script.py"),
            file_type="script",
            modified_time=datetime.now()
        )
        
        # Should not be able to modify frozen model
        with pytest.raises(Exception):  # ValidationError or similar
            metadata.file_type = "contract"
    
    def test_file_metadata_validation(self):
        """Test FileMetadata field validation."""
        # Valid file types
        valid_types = ["script", "contract", "spec", "builder", "config"]
        
        for file_type in valid_types:
            metadata = FileMetadata(
                path=Path(f"/test/{file_type}.py"),
                file_type=file_type,
                modified_time=datetime.now()
            )
            assert metadata.file_type == file_type


class TestStepInfo:
    """Test cases for StepInfo model."""
    
    def test_step_info_creation(self):
        """Test basic StepInfo creation."""
        step_info = StepInfo(
            step_name="test_step",
            workspace_id="core",
            registry_data={"config_class": "TestConfig"},
            file_components={}
        )
        
        assert step_info.step_name == "test_step"
        assert step_info.workspace_id == "core"
        assert step_info.registry_data == {"config_class": "TestConfig"}
        assert step_info.file_components == {}
    
    def test_step_info_with_file_components(self):
        """Test StepInfo with file components."""
        script_metadata = FileMetadata(
            path=Path("/test/script.py"),
            file_type="script",
            modified_time=datetime.now()
        )
        
        step_info = StepInfo(
            step_name="test_step",
            workspace_id="core",
            file_components={"script": script_metadata}
        )
        
        assert "script" in step_info.file_components
        assert step_info.file_components["script"] == script_metadata
    
    def test_step_info_properties(self):
        """Test StepInfo property methods."""
        registry_data = {
            "config_class": "TestConfig",
            "sagemaker_step_type": "Processing",
            "builder_step_name": "TestBuilder",
            "description": "Test step description"
        }
        
        step_info = StepInfo(
            step_name="test_step",
            workspace_id="core",
            registry_data=registry_data
        )
        
        assert step_info.config_class == "TestConfig"
        assert step_info.sagemaker_step_type == "Processing"
        assert step_info.builder_step_name == "TestBuilder"
        assert step_info.description == "Test step description"
    
    def test_step_info_properties_with_none_values(self):
        """Test StepInfo properties handle None values safely."""
        registry_data = {
            "config_class": None,
            "sagemaker_step_type": None,
            "builder_step_name": None,
            "description": None
        }
        
        step_info = StepInfo(
            step_name="test_step",
            workspace_id="core",
            registry_data=registry_data
        )
        
        # Should return empty strings for None values
        assert step_info.config_class == ""
        assert step_info.sagemaker_step_type == ""
        assert step_info.builder_step_name == ""
        assert step_info.description == ""
    
    def test_step_info_properties_with_missing_keys(self):
        """Test StepInfo properties handle missing keys gracefully."""
        step_info = StepInfo(
            step_name="test_step",
            workspace_id="core",
            registry_data={}  # Empty registry data
        )
        
        # Should return empty strings for missing keys
        assert step_info.config_class == ""
        assert step_info.sagemaker_step_type == ""
        assert step_info.builder_step_name == ""
        assert step_info.description == ""
    
    def test_step_info_defaults(self):
        """Test StepInfo default values."""
        step_info = StepInfo(
            step_name="test_step",
            workspace_id="core"
        )
        
        assert step_info.registry_data == {}
        assert step_info.file_components == {}


class TestStepSearchResult:
    """Test cases for StepSearchResult model."""
    
    def test_search_result_creation(self):
        """Test basic StepSearchResult creation."""
        result = StepSearchResult(
            step_name="test_step",
            workspace_id="core",
            match_score=0.95,
            match_reason="name_match",
            components_available=["script", "contract"]
        )
        
        assert result.step_name == "test_step"
        assert result.workspace_id == "core"
        assert result.match_score == 0.95
        assert result.match_reason == "name_match"
        assert result.components_available == ["script", "contract"]
    
    def test_search_result_immutable(self):
        """Test that StepSearchResult is frozen/immutable."""
        result = StepSearchResult(
            step_name="test_step",
            workspace_id="core",
            match_score=0.95,
            match_reason="name_match"
        )
        
        # Should not be able to modify frozen model
        with pytest.raises(Exception):  # ValidationError or similar
            result.match_score = 0.5
    
    def test_search_result_defaults(self):
        """Test StepSearchResult default values."""
        result = StepSearchResult(
            step_name="test_step",
            workspace_id="core",
            match_score=0.95,
            match_reason="name_match"
        )
        
        assert result.components_available == []
    
    def test_search_result_score_validation(self):
        """Test match score validation."""
        # Valid scores (0.0 to 1.0)
        valid_scores = [0.0, 0.5, 0.95, 1.0]
        
        for score in valid_scores:
            result = StepSearchResult(
                step_name="test_step",
                workspace_id="core",
                match_score=score,
                match_reason="test"
            )
            assert result.match_score == score
    
    def test_search_result_match_reasons(self):
        """Test common match reason values."""
        common_reasons = ["name_match", "fuzzy_match", "partial_match", "semantic_match"]
        
        for reason in common_reasons:
            result = StepSearchResult(
                step_name="test_step",
                workspace_id="core",
                match_score=0.8,
                match_reason=reason
            )
            assert result.match_reason == reason


class TestModelIntegration:
    """Test integration between different models."""
    
    def test_step_info_with_multiple_components(self):
        """Test StepInfo with multiple file components."""
        script_metadata = FileMetadata(
            path=Path("/test/script.py"),
            file_type="script",
            modified_time=datetime.now()
        )
        
        contract_metadata = FileMetadata(
            path=Path("/test/contract.py"),
            file_type="contract",
            modified_time=datetime.now()
        )
        
        step_info = StepInfo(
            step_name="complex_step",
            workspace_id="core",
            registry_data={"config_class": "ComplexConfig"},
            file_components={
                "script": script_metadata,
                "contract": contract_metadata
            }
        )
        
        assert len(step_info.file_components) == 2
        assert "script" in step_info.file_components
        assert "contract" in step_info.file_components
        assert step_info.file_components["script"].file_type == "script"
        assert step_info.file_components["contract"].file_type == "contract"
    
    def test_search_result_with_step_info_components(self):
        """Test StepSearchResult components match StepInfo components."""
        # Create StepInfo with components
        step_info = StepInfo(
            step_name="test_step",
            workspace_id="core",
            file_components={
                "script": FileMetadata(
                    path=Path("/test/script.py"),
                    file_type="script",
                    modified_time=datetime.now()
                ),
                "contract": FileMetadata(
                    path=Path("/test/contract.py"),
                    file_type="contract",
                    modified_time=datetime.now()
                )
            }
        )
        
        # Create search result with matching components
        search_result = StepSearchResult(
            step_name=step_info.step_name,
            workspace_id=step_info.workspace_id,
            match_score=1.0,
            match_reason="name_match",
            components_available=list(step_info.file_components.keys())
        )
        
        assert search_result.step_name == step_info.step_name
        assert search_result.workspace_id == step_info.workspace_id
        assert set(search_result.components_available) == set(step_info.file_components.keys())
    
    def test_model_serialization_compatibility(self):
        """Test that models can be serialized/deserialized."""
        # Create complex StepInfo
        step_info = StepInfo(
            step_name="serialization_test",
            workspace_id="test_workspace",
            registry_data={
                "config_class": "TestConfig",
                "description": "Test serialization"
            },
            file_components={
                "script": FileMetadata(
                    path=Path("/test/script.py"),
                    file_type="script",
                    modified_time=datetime(2023, 1, 1, 12, 0, 0)
                )
            }
        )
        
        # Test model dict conversion
        step_dict = step_info.model_dump()
        assert step_dict["step_name"] == "serialization_test"
        assert step_dict["workspace_id"] == "test_workspace"
        assert "registry_data" in step_dict
        assert "file_components" in step_dict
        
        # Test reconstruction from dict
        reconstructed = StepInfo.model_validate(step_dict)
        assert reconstructed.step_name == step_info.step_name
        assert reconstructed.workspace_id == step_info.workspace_id
        assert reconstructed.registry_data == step_info.registry_data
