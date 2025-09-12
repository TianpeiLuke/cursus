"""
Unit tests for workspace configuration models.

Tests the Pydantic V2 models for workspace step definitions and pipeline configurations.
"""

import pytest
import json
import yaml
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch

from cursus.workspace.core.config import (
    WorkspaceStepDefinition,
    WorkspacePipelineDefinition,
)


class TestWorkspaceStepDefinition:
    """Test cases for WorkspaceStepDefinition model."""

    def test_valid_step_definition(self):
        """Test creating a valid workspace step definition."""
        step_data = {
            "step_name": "test_step",
            "developer_id": "dev1",
            "step_type": "XGBoostTraining",
            "config_data": {"param1": "value1"},
            "workspace_root": "/path/to/workspace",
            "dependencies": ["dep1", "dep2"],
        }

        step = WorkspaceStepDefinition(**step_data)

        assert step.step_name == "test_step"
        assert step.developer_id == "dev1"
        assert step.step_type == "XGBoostTraining"
        assert step.config_data == {"param1": "value1"}
        assert step.workspace_root == "/path/to/workspace"
        assert step.dependencies == ["dep1", "dep2"]

    def test_step_definition_with_defaults(self):
        """Test creating step definition with default values."""
        step_data = {
            "step_name": "test_step",
            "developer_id": "dev1",
            "step_type": "XGBoostTraining",
            "config_data": {"param1": "value1"},
            "workspace_root": "/path/to/workspace",
        }

        step = WorkspaceStepDefinition(**step_data)

        assert step.dependencies == []  # Default empty list

    def test_invalid_step_name(self):
        """Test validation of step_name field."""
        step_data = {
            "step_name": "",  # Invalid empty string
            "developer_id": "dev1",
            "step_type": "XGBoostTraining",
            "config_data": {"param1": "value1"},
            "workspace_root": "/path/to/workspace",
        }

        with pytest.raises(ValueError) as exc_info:
            WorkspaceStepDefinition(**step_data)

        assert "step_name must be a non-empty string" in str(exc_info.value)

    def test_invalid_developer_id(self):
        """Test validation of developer_id field."""
        step_data = {
            "step_name": "test_step",
            "developer_id": "",  # Invalid empty string
            "step_type": "XGBoostTraining",
            "config_data": {"param1": "value1"},
            "workspace_root": "/path/to/workspace",
        }

        with pytest.raises(ValueError) as exc_info:
            WorkspaceStepDefinition(**step_data)

        assert "developer_id must be a non-empty string" in str(exc_info.value)

    def test_invalid_step_type(self):
        """Test validation of step_type field."""
        step_data = {
            "step_name": "test_step",
            "developer_id": "dev1",
            "step_type": "",  # Invalid empty string
            "config_data": {"param1": "value1"},
            "workspace_root": "/path/to/workspace",
        }

        with pytest.raises(ValueError) as exc_info:
            WorkspaceStepDefinition(**step_data)

        assert "step_type must be a non-empty string" in str(exc_info.value)

    def test_invalid_workspace_root(self):
        """Test validation of workspace_root field."""
        step_data = {
            "step_name": "test_step",
            "developer_id": "dev1",
            "step_type": "XGBoostTraining",
            "config_data": {"param1": "value1"},
            "workspace_root": "",  # Invalid empty string
        }

        with pytest.raises(ValueError) as exc_info:
            WorkspaceStepDefinition(**step_data)

        assert "workspace_root must be a non-empty string" in str(exc_info.value)

    def test_get_workspace_path(self):
        """Test get_workspace_path method."""
        step = WorkspaceStepDefinition(
            step_name="test_step",
            developer_id="dev1",
            step_type="XGBoostTraining",
            config_data={"param1": "value1"},
            workspace_root="/path/to/workspace",
        )

        # Test with relative path
        path = step.get_workspace_path("subdir/file.txt")
        assert path == "/path/to/workspace/subdir/file.txt"

        # Test without relative path
        path = step.get_workspace_path()
        assert path == "/path/to/workspace"

    def test_model_serialization(self):
        """Test model serialization and deserialization."""
        step_data = {
            "step_name": "test_step",
            "developer_id": "dev1",
            "step_type": "XGBoostTraining",
            "config_data": {"param1": "value1"},
            "workspace_root": "/path/to/workspace",
            "dependencies": ["dep1"],
        }

        step = WorkspaceStepDefinition(**step_data)

        # Test model_dump
        dumped = step.model_dump()
        assert dumped == step_data

        # Test recreation from dumped data
        step2 = WorkspaceStepDefinition(**dumped)
        assert step2.step_name == step.step_name
        assert step2.developer_id == step.developer_id


class TestWorkspacePipelineDefinition:
    """Test cases for WorkspacePipelineDefinition model."""

    @pytest.fixture
    def temp_dir(self):
        """Set up test fixtures."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up test fixtures
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_steps(self):
        """Create sample steps for testing."""
        return [
            WorkspaceStepDefinition(
                step_name="step1",
                developer_id="dev1",
                step_type="DataPreprocessing",
                config_data={"param1": "value1"},
                workspace_root="/workspace",
            ),
            WorkspaceStepDefinition(
                step_name="step2",
                developer_id="dev2",
                step_type="XGBoostTraining",
                config_data={"param2": "value2"},
                workspace_root="/workspace",
                dependencies=["step1"],
            ),
        ]

    def test_valid_pipeline_definition(self, sample_steps):
        """Test creating a valid workspace pipeline definition."""
        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline",
            workspace_root="/workspace",
            steps=sample_steps,
            global_config={"global_param": "global_value"},
        )

        assert pipeline.pipeline_name == "test_pipeline"
        assert pipeline.workspace_root == "/workspace"
        assert len(pipeline.steps) == 2
        assert pipeline.global_config == {"global_param": "global_value"}

    def test_pipeline_with_defaults(self, sample_steps):
        """Test creating pipeline with default values."""
        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline",
            workspace_root="/workspace",
            steps=sample_steps,
        )

        assert pipeline.global_config == {}  # Default empty dict

    def test_invalid_pipeline_name(self, sample_steps):
        """Test validation of pipeline_name field."""
        with pytest.raises(ValueError) as exc_info:
            WorkspacePipelineDefinition(
                pipeline_name="",  # Invalid empty string
                workspace_root="/workspace",
                steps=sample_steps,
            )

        assert "pipeline_name must be a non-empty string" in str(exc_info.value)

    def test_invalid_workspace_root(self, sample_steps):
        """Test validation of workspace_root field."""
        with pytest.raises(ValueError) as exc_info:
            WorkspacePipelineDefinition(
                pipeline_name="test_pipeline",
                workspace_root="",  # Invalid empty string
                steps=sample_steps,
            )

        assert "workspace_root must be a non-empty string" in str(exc_info.value)

    def test_empty_steps_list(self):
        """Test validation of empty steps list."""
        with pytest.raises(ValueError) as exc_info:
            WorkspacePipelineDefinition(
                pipeline_name="test_pipeline",
                workspace_root="/workspace",
                steps=[],  # Invalid empty list
            )

        assert "steps list cannot be empty" in str(exc_info.value)

    def test_duplicate_step_names(self):
        """Test validation of duplicate step names."""
        steps = [
            WorkspaceStepDefinition(
                step_name="duplicate_step",  # Same name
                developer_id="dev1",
                step_type="DataPreprocessing",
                config_data={"param1": "value1"},
                workspace_root="/workspace",
            ),
            WorkspaceStepDefinition(
                step_name="duplicate_step",  # Same name
                developer_id="dev2",
                step_type="XGBoostTraining",
                config_data={"param2": "value2"},
                workspace_root="/workspace",
            ),
        ]

        with pytest.raises(ValueError) as exc_info:
            WorkspacePipelineDefinition(
                pipeline_name="test_pipeline", workspace_root="/workspace", steps=steps
            )

        assert "Duplicate step names found in pipeline" in str(exc_info.value)

    def test_validate_workspace_dependencies_valid(self, sample_steps):
        """Test dependency validation with valid dependencies."""
        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline",
            workspace_root="/workspace",
            steps=sample_steps,
        )

        result = pipeline.validate_workspace_dependencies()

        assert result["valid"]
        assert result["errors"] == []
        assert "step1" in result["dependency_graph"]
        assert "step2" in result["dependency_graph"]
        assert result["dependency_graph"]["step2"] == ["step1"]

    def test_validate_workspace_dependencies_missing(self):
        """Test dependency validation with missing dependencies."""
        steps = [
            WorkspaceStepDefinition(
                step_name="step1",
                developer_id="dev1",
                step_type="DataPreprocessing",
                config_data={"param1": "value1"},
                workspace_root="/workspace",
                dependencies=["missing_step"],  # Missing dependency
            )
        ]

        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline", workspace_root="/workspace", steps=steps
        )

        result = pipeline.validate_workspace_dependencies()

        assert not result["valid"]
        assert len(result["errors"]) == 1
        assert "missing_step" in result["errors"][0]

    def test_validate_circular_dependencies(self):
        """Test detection of circular dependencies."""
        steps = [
            WorkspaceStepDefinition(
                step_name="step1",
                developer_id="dev1",
                step_type="DataPreprocessing",
                config_data={"param1": "value1"},
                workspace_root="/workspace",
                dependencies=["step2"],  # Circular dependency
            ),
            WorkspaceStepDefinition(
                step_name="step2",
                developer_id="dev2",
                step_type="XGBoostTraining",
                config_data={"param2": "value2"},
                workspace_root="/workspace",
                dependencies=["step1"],  # Circular dependency
            ),
        ]

        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline", workspace_root="/workspace", steps=steps
        )

        result = pipeline.validate_workspace_dependencies()

        assert not result["valid"]
        assert any("Circular dependencies" in error for error in result["errors"])

    def test_to_pipeline_config(self, sample_steps):
        """Test conversion to pipeline configuration format."""
        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline",
            workspace_root="/workspace",
            steps=sample_steps,
            global_config={"global_param": "global_value"},
        )

        config = pipeline.to_pipeline_config()

        assert config["pipeline_name"] == "test_pipeline"
        assert config["workspace_root"] == "/workspace"
        assert config["global_config"] == {"global_param": "global_value"}
        assert "steps" in config
        assert "step1" in config["steps"]
        assert "step2" in config["steps"]
        assert config["steps"]["step1"]["developer_id"] == "dev1"
        assert config["steps"]["step2"]["dependencies"] == ["step1"]

    def test_get_developers(self, sample_steps):
        """Test getting list of developers."""
        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline",
            workspace_root="/workspace",
            steps=sample_steps,
        )

        developers = pipeline.get_developers()

        assert set(developers) == {"dev1", "dev2"}

    def test_get_steps_by_developer(self, sample_steps):
        """Test getting steps by developer."""
        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline",
            workspace_root="/workspace",
            steps=sample_steps,
        )

        dev1_steps = pipeline.get_steps_by_developer("dev1")
        dev2_steps = pipeline.get_steps_by_developer("dev2")

        assert len(dev1_steps) == 1
        assert dev1_steps[0].step_name == "step1"
        assert len(dev2_steps) == 1
        assert dev2_steps[0].step_name == "step2"

    def test_get_step_by_name(self, sample_steps):
        """Test getting step by name."""
        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline",
            workspace_root="/workspace",
            steps=sample_steps,
        )

        step1 = pipeline.get_step_by_name("step1")
        step_missing = pipeline.get_step_by_name("missing_step")

        assert step1 is not None
        assert step1.step_name == "step1"
        assert step1.developer_id == "dev1"
        assert step_missing is None

    def test_validate_with_consolidated_managers(self, sample_steps):
        """Test validation with consolidated managers (Phase 2 optimization)."""
        mock_manager = Mock()
        mock_lifecycle_manager = Mock()
        mock_isolation_manager = Mock()
        mock_discovery_manager = Mock()
        mock_integration_manager = Mock()

        # Setup manager delegation
        mock_manager.lifecycle_manager = mock_lifecycle_manager
        mock_manager.isolation_manager = mock_isolation_manager
        mock_manager.discovery_manager = mock_discovery_manager
        mock_manager.integration_manager = mock_integration_manager

        # Mock validation responses
        mock_lifecycle_manager.validate_pipeline_lifecycle.return_value = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        mock_isolation_manager.validate_pipeline_isolation.return_value = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        mock_discovery_manager.validate_pipeline_dependencies.return_value = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        mock_integration_manager.validate_pipeline_integration.return_value = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline",
            workspace_root="/workspace",
            steps=sample_steps,
        )

        result = pipeline.validate_with_consolidated_managers(mock_manager)

        assert "validations" in result
        assert "lifecycle" in result["validations"]
        assert "isolation" in result["validations"]
        assert "discovery" in result["validations"]
        assert "integration" in result["validations"]
        assert "overall_valid" in result
        assert result["overall_valid"]

        # Verify manager methods were called
        mock_lifecycle_manager.validate_pipeline_lifecycle.assert_called_once()
        mock_isolation_manager.validate_pipeline_isolation.assert_called_once()
        mock_discovery_manager.validate_pipeline_dependencies.assert_called_once()
        mock_integration_manager.validate_pipeline_integration.assert_called_once()

    def test_json_file_operations(self, sample_steps, temp_dir):
        """Test JSON file save and load operations."""
        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline",
            workspace_root="/workspace",
            steps=sample_steps,
            global_config={"global_param": "global_value"},
        )

        temp_file = Path(temp_dir) / "test_config.json"

        # Test save to JSON
        pipeline.to_json_file(str(temp_file))

        # Test load from JSON
        loaded_pipeline = WorkspacePipelineDefinition.from_json_file(str(temp_file))

        assert loaded_pipeline.pipeline_name == pipeline.pipeline_name
        assert loaded_pipeline.workspace_root == pipeline.workspace_root
        assert len(loaded_pipeline.steps) == len(pipeline.steps)
        assert loaded_pipeline.global_config == pipeline.global_config

    def test_yaml_file_operations(self, sample_steps, temp_dir):
        """Test YAML file save and load operations."""
        pipeline = WorkspacePipelineDefinition(
            pipeline_name="test_pipeline",
            workspace_root="/workspace",
            steps=sample_steps,
            global_config={"global_param": "global_value"},
        )

        temp_file = Path(temp_dir) / "test_config.yaml"

        # Test save to YAML
        pipeline.to_yaml_file(str(temp_file))

        # Test load from YAML
        loaded_pipeline = WorkspacePipelineDefinition.from_yaml_file(str(temp_file))

        assert loaded_pipeline.pipeline_name == pipeline.pipeline_name
        assert loaded_pipeline.workspace_root == pipeline.workspace_root
        assert len(loaded_pipeline.steps) == len(pipeline.steps)
        assert loaded_pipeline.global_config == pipeline.global_config
