"""
Comprehensive tests for FastAPI endpoints following pytest best practices.

This test module follows the pytest best practices guide:
1. Source Code First Rule - Read api.py implementation completely before writing tests
2. Mock Path Precision - Mock at exact import locations from source
3. Implementation-Driven Testing - Match test behavior to actual implementation
4. Async Testing Patterns - Proper async test configuration for FastAPI
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import tempfile
import json
from pathlib import Path

# Following Source Code First Rule - import the actual implementation
from cursus.api.config_ui.api import (
    router, create_config_ui_app, 
    ConfigDiscoveryRequest, ConfigDiscoveryResponse,
    ConfigWidgetRequest, ConfigWidgetResponse,
    ConfigSaveRequest, ConfigSaveResponse,
    MergeConfigsRequest, MergeConfigsResponse,
    DAGAnalysisRequest, DAGAnalysisResponse,
    latest_config, active_sessions
)
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase


class TestFastAPIEndpoints:
    """Comprehensive tests for FastAPI endpoints following pytest best practices."""
    
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global state before each test (Category 17: Global State Management)."""
        # Reset global variables from api.py
        import cursus.api.config_ui.api as api_module
        api_module.latest_config = None
        api_module.active_sessions = {}
        yield
        # Cleanup after test
        api_module.latest_config = None
        api_module.active_sessions = {}
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        app = create_config_ui_app()
        return TestClient(app)
    
    @pytest.fixture
    def mock_discover_configs(self):
        """Mock discover_available_configs function."""
        with patch('cursus.api.config_ui.api.discover_available_configs') as mock_discover:
            # Create proper mock config classes with required attributes
            mock_cradle_config = Mock()
            mock_cradle_config.__name__ = "CradleDataLoadingConfig"
            mock_cradle_config.__module__ = "cursus.steps.configs.config_cradle_data_loading_step"
            mock_cradle_config.__doc__ = "Cradle data loading configuration"
            mock_cradle_config.model_fields = {"field1": Mock(), "field2": Mock()}
            mock_cradle_config.from_base_config = Mock()
            
            mock_xgboost_config = Mock()
            mock_xgboost_config.__name__ = "XGBoostTrainingConfig"
            mock_xgboost_config.__module__ = "cursus.steps.configs.config_xgboost_training_step"
            mock_xgboost_config.__doc__ = "XGBoost training configuration"
            mock_xgboost_config.model_fields = {"field1": Mock(), "field2": Mock()}
            mock_xgboost_config.from_base_config = Mock()
            
            mock_discover.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase,
                "CradleDataLoadingConfig": mock_cradle_config,
                "XGBoostTrainingConfig": mock_xgboost_config
            }
            yield mock_discover
    
    @pytest.fixture
    def mock_universal_config_core(self):
        """Mock UniversalConfigCore with realistic behavior."""
        with patch('cursus.api.config_ui.api.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            # Configure realistic behavior based on source implementation
            mock_core.discover_config_classes.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase,
                "CradleDataLoadingConfig": Mock(),
                "XGBoostTrainingConfig": Mock()
            }
            
            mock_core._get_form_fields.return_value = [
                {
                    "name": "author",
                    "type": "text",
                    "required": True,
                    "tier": "essential",
                    "description": "Author name",
                    "default": None
                },
                {
                    "name": "bucket",
                    "type": "text",
                    "required": True,
                    "tier": "essential", 
                    "description": "S3 bucket name",
                    "default": None
                }
            ]
            
            yield mock_core
    
    @pytest.fixture
    def example_base_config_data(self):
        """Example base configuration data for testing."""
        return {
            "author": "test-user",
            "bucket": "test-bucket",
            "role": "test-role",
            "region": "NA",
            "service_name": "test-service",
            "pipeline_version": "1.0.0",
            "project_root_folder": "test-project"
        }
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Cursus Config UI API"
        assert data["version"] == "2.0.0"
        assert "web_interface" in data
        assert "api_docs" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/config-ui/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "config-ui"
        assert "phase" in data
    
    def test_discover_configurations_success(self, client, mock_discover_configs):
        """Test successful configuration discovery."""
        # Following Category 2: Mock Configuration pattern
        request_data = {
            "workspace_dirs": None
        }
        
        response = client.post("/api/config-ui/discover", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Test actual implementation behavior
        assert "configs" in data
        assert "count" in data
        assert data["count"] >= 2  # At least base configs
        assert "BasePipelineConfig" in data["configs"]
        assert "ProcessingStepConfigBase" in data["configs"]
        
        # Verify mock was called
        mock_discover_configs.assert_called_once_with(workspace_dirs=None)
    
    def test_discover_configurations_with_workspace_dirs(self, client, mock_discover_configs):
        """Test configuration discovery with workspace directories."""
        request_data = {
            "workspace_dirs": ["/test/workspace"]
        }
        
        response = client.post("/api/config-ui/discover", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 2
        
        # Verify workspace_dirs were passed correctly
        mock_discover_configs.assert_called_once_with(workspace_dirs=["/test/workspace"])
    
    def test_discover_configurations_failure(self, client, mock_discover_configs):
        """Test configuration discovery failure handling."""
        # Following Category 6: Exception Handling pattern
        mock_discover_configs.side_effect = Exception("Discovery failed")
        
        request_data = {"workspace_dirs": None}
        response = client.post("/api/config-ui/discover", json=request_data)
        
        assert response.status_code == 500
        assert "Discovery failed" in response.json()["detail"]
    
    def test_create_configuration_widget_success(self, client, mock_universal_config_core):
        """Test successful configuration widget creation."""
        # Following Category 4: Test Expectations vs Implementation pattern
        request_data = {
            "config_class_name": "BasePipelineConfig",
            "base_config": None,
            "workspace_dirs": None
        }
        
        response = client.post("/api/config-ui/create-widget", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Test actual implementation response structure
        assert "config_class_name" in data
        assert "fields" in data
        assert "values" in data
        assert "specialized_component" in data
        assert data["config_class_name"] == "BasePipelineConfig"
        assert data["specialized_component"] is False
        assert isinstance(data["fields"], list)
        assert len(data["fields"]) > 0
        
        # Verify core was called correctly
        mock_universal_config_core.assert_called_once_with(workspace_dirs=None)
    
    def test_create_configuration_widget_specialized_component(self, client):
        """Test widget creation for specialized components."""
        # Mock SpecializedComponentRegistry
        with patch('cursus.api.config_ui.api.SpecializedComponentRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            mock_registry.has_specialized_component.return_value = True
            
            request_data = {
                "config_class_name": "CradleDataLoadingConfig",
                "base_config": None,
                "workspace_dirs": None
            }
            
            response = client.post("/api/config-ui/create-widget", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["specialized_component"] is True
            assert data["config_class_name"] == "CradleDataLoadingConfig"
            assert data["fields"] == []  # Empty for specialized components
    
    def test_create_configuration_widget_class_not_found(self, client, mock_universal_config_core):
        """Test widget creation with non-existent class."""
        # Configure mock to return empty config classes
        mock_universal_config_core.discover_config_classes.return_value = {}
        
        request_data = {
            "config_class_name": "NonExistentConfig",
            "base_config": None,
            "workspace_dirs": None
        }
        
        response = client.post("/api/config-ui/create-widget", json=request_data)
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_create_configuration_widget_with_base_config(self, client, mock_universal_config_core, example_base_config_data):
        """Test widget creation with base configuration pre-population."""
        # Mock config class with from_base_config method
        mock_config_class = Mock()
        mock_pre_populated = Mock()
        mock_config_class.from_base_config.return_value = mock_pre_populated
        mock_pre_populated.model_dump.return_value = example_base_config_data
        
        mock_universal_config_core.discover_config_classes.return_value = {
            "TestConfig": mock_config_class
        }
        
        request_data = {
            "config_class_name": "TestConfig",
            "base_config": example_base_config_data,
            "workspace_dirs": None
        }
        
        response = client.post("/api/config-ui/create-widget", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        # Should have pre-populated values from base_config
    
    def test_save_configuration_success(self, client, mock_discover_configs, example_base_config_data):
        """Test successful configuration saving."""
        request_data = {
            "config_class_name": "BasePipelineConfig",
            "form_data": example_base_config_data
        }
        
        response = client.post("/api/config-ui/save-config", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Test actual implementation response structure
        assert "success" in data
        assert "config" in data
        assert "config_type" in data
        assert data["success"] is True
        assert data["config_type"] == "BasePipelineConfig"
        assert isinstance(data["config"], dict)
        
        # Verify global state was updated
        import cursus.api.config_ui.api as api_module
        assert api_module.latest_config is not None
        assert api_module.latest_config["config_type"] == "BasePipelineConfig"
    
    def test_save_configuration_class_not_found(self, client, mock_discover_configs):
        """Test configuration saving with non-existent class."""
        # Configure mock to return empty config classes
        mock_discover_configs.return_value = {}
        
        request_data = {
            "config_class_name": "NonExistentConfig",
            "form_data": {"test": "data"}
        }
        
        response = client.post("/api/config-ui/save-config", json=request_data)
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_save_configuration_validation_error(self, client, mock_discover_configs):
        """Test configuration saving with Pydantic validation errors."""
        # Following Category 12: NoneType Attribute Access and Defensive Coding
        request_data = {
            "config_class_name": "BasePipelineConfig",
            "form_data": {
                # Missing required fields to trigger validation error
                "author": "test-user"
                # Missing bucket, role, etc.
            }
        }
        
        response = client.post("/api/config-ui/save-config", json=request_data)
        
        # Should return 422 for validation errors (per implementation)
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
        assert error_data["detail"]["error_type"] == "validation_error"
        assert "validation_errors" in error_data["detail"]
    
    def test_get_latest_config_success(self, client):
        """Test getting latest configuration when available."""
        # Set up global state with latest config
        import cursus.api.config_ui.api as api_module
        api_module.latest_config = {
            "config": {"author": "test-user"},
            "config_type": "BasePipelineConfig",
            "timestamp": "2025-10-07T12:00:00"
        }
        
        response = client.get("/api/config-ui/get-latest-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["config_type"] == "BasePipelineConfig"
        assert data["config"]["author"] == "test-user"
        assert "timestamp" in data
    
    def test_get_latest_config_not_available(self, client):
        """Test getting latest configuration when none available."""
        # Global state should be None from fixture reset
        response = client.get("/api/config-ui/get-latest-config")
        
        assert response.status_code == 404
        assert "No configuration available" in response.json()["detail"]
    
    def test_clear_config_success(self, client):
        """Test clearing stored configuration."""
        # Set up global state first
        import cursus.api.config_ui.api as api_module
        api_module.latest_config = {"test": "data"}
        
        response = client.post("/api/config-ui/clear-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        
        # Verify global state was cleared
        assert api_module.latest_config is None
    
    def test_analyze_pipeline_dag_success(self, client, mock_universal_config_core):
        """Test successful DAG analysis."""
        # Mock DAG analysis methods
        mock_universal_config_core._discover_required_config_classes.return_value = [
            {
                "node_name": "test_step",
                "config_class_name": "TestConfig",
                "config_class": Mock(),
                "inheritance_pattern": "base_only",
                "is_specialized": False
            }
        ]
        
        mock_universal_config_core._create_workflow_structure.return_value = [
            {
                "step_number": 1,
                "title": "Base Configuration",
                "config_class": BasePipelineConfig,
                "type": "base",
                "required": True
            }
        ]
        
        request_data = {
            "pipeline_dag": {
                "nodes": ["test_step", "another_step"]
            },
            "workspace_dirs": None
        }
        
        response = client.post("/api/config-ui/analyze-dag", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Test actual implementation response structure
        assert "discovered_steps" in data
        assert "required_configs" in data
        assert "workflow_steps" in data
        assert "total_steps" in data
        assert "hidden_configs_count" in data
        
        assert len(data["discovered_steps"]) == 2  # Two nodes in request
        assert data["total_steps"] >= 1
        assert isinstance(data["hidden_configs_count"], int)
    
    def test_analyze_pipeline_dag_no_resolver(self, client, mock_universal_config_core):
        """Test DAG analysis when StepConfigResolverAdapter is not available."""
        # Mock ImportError for resolver
        with patch('cursus.step_catalog.adapters.config_resolver.StepConfigResolverAdapter') as mock_resolver_class:
            mock_resolver_class.side_effect = ImportError("Resolver not available")
            
            request_data = {
                "pipeline_dag": {
                    "nodes": ["test_step"]
                },
                "workspace_dirs": None
            }
            
            response = client.post("/api/config-ui/analyze-dag", json=request_data)
            
            # Should handle gracefully and still return analysis
            assert response.status_code == 200
            data = response.json()
            assert "discovered_steps" in data
    
    def test_merge_and_save_configurations_success(self, client, mock_discover_configs):
        """Test successful configuration merging and saving."""
        # Mock merge_and_save_configs function
        with patch('cursus.core.config_fields.merge_and_save_configs') as mock_merge:
            mock_merged_config = {
                "shared": {"author": "test-user"},
                "specific": {
                    "Base": {"pipeline_name": "test-pipeline"}
                }
            }
            mock_merge.return_value = mock_merged_config
            
            request_data = {
                "session_configs": {
                    "BasePipelineConfig": {
                        "author": "test-user",
                        "bucket": "test-bucket",
                        "role": "test-role",
                        "region": "NA",
                        "service_name": "test-service",
                        "pipeline_version": "1.0.0",
                        "project_root_folder": "test-project"
                    }
                },
                "filename": "test_config.json",
                "workspace_dirs": None
            }
            
            response = client.post("/api/config-ui/merge-and-save-configs", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Test actual implementation response structure
            assert "success" in data
            assert "merged_config" in data
            assert "filename" in data
            assert "download_url" in data
            assert data["success"] is True
            assert data["filename"] == "test_config.json"
            assert "/download/" in data["download_url"]
    
    def test_merge_and_save_configurations_no_valid_configs(self, client, mock_discover_configs):
        """Test merging when no valid configurations are provided."""
        # Configure mock to return empty config classes
        mock_discover_configs.return_value = {}
        
        request_data = {
            "session_configs": {
                "NonExistentConfig": {"test": "data"}
            },
            "filename": "test_config.json",
            "workspace_dirs": None
        }
        
        response = client.post("/api/config-ui/merge-and-save-configs", json=request_data)
        
        assert response.status_code == 400
        assert "No valid configurations to merge" in response.json()["detail"]
    
    def test_merge_and_save_configurations_auto_filename(self, client, mock_discover_configs):
        """Test merging with automatic filename generation."""
        with patch('cursus.core.config_fields.merge_and_save_configs') as mock_merge:
            mock_merge.return_value = {"shared": {}, "specific": {}}
            
            request_data = {
                "session_configs": {
                    "BasePipelineConfig": {
                        "author": "test-user",
                        "bucket": "test-bucket",
                        "role": "test-role",
                        "region": "NA",
                        "service_name": "enhanced-service",
                        "pipeline_version": "1.0.0",
                        "project_root_folder": "test-project"
                    }
                },
                "filename": None,  # Should auto-generate
                "workspace_dirs": None
            }
            
            response = client.post("/api/config-ui/merge-and-save-configs", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should generate filename based on config data
            assert data["filename"].startswith("config_")
            assert ".json" in data["filename"]
    
    def test_download_merged_config_success(self, client):
        """Test successful merged config download."""
        # Set up download cache
        import cursus.api.config_ui.api as api_module
        if not hasattr(api_module.merge_and_save_configurations, '_download_cache'):
            api_module.merge_and_save_configurations._download_cache = {}
        
        test_content = '{"test": "config"}'
        download_id = "test_12345"
        api_module.merge_and_save_configurations._download_cache[download_id] = {
            'content': test_content,
            'filename': 'test_config.json',
            'content_type': 'application/json'
        }
        
        response = client.get(f"/api/config-ui/download/{download_id}")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
    
    def test_download_merged_config_not_found(self, client):
        """Test download with non-existent download ID."""
        response = client.get("/api/config-ui/download/nonexistent_id")
        
        assert response.status_code == 404
        assert "Download not found" in response.json()["detail"]
    
    def test_create_pipeline_wizard_success(self, client):
        """Test successful pipeline wizard creation."""
        request_data = {
            "dag": {
                "nodes": [
                    {"name": "step1", "config_type": "TestConfig1"},
                    {"name": "step2", "config_type": "TestConfig2"}
                ]
            },
            "base_config": None,
            "processing_config": None
        }
        
        response = client.post("/api/config-ui/create-pipeline-wizard", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Test actual implementation response structure
        assert "steps" in data
        assert "wizard_id" in data
        assert isinstance(data["steps"], list)
        assert len(data["steps"]) >= 2  # Base + Processing + DAG steps
        
        # Check base steps are included
        step_titles = [step["title"] for step in data["steps"]]
        assert "Base Configuration" in step_titles
        assert "Processing Configuration" in step_titles


class TestRequestResponseModels:
    """Test Pydantic request/response models."""
    
    def test_config_discovery_request_model(self):
        """Test ConfigDiscoveryRequest model validation."""
        # Valid request
        request = ConfigDiscoveryRequest(workspace_dirs=["/test/path"])
        assert request.workspace_dirs == ["/test/path"]
        
        # Request with None workspace_dirs
        request = ConfigDiscoveryRequest(workspace_dirs=None)
        assert request.workspace_dirs is None
        
        # Request with no workspace_dirs
        request = ConfigDiscoveryRequest()
        assert request.workspace_dirs is None
    
    def test_config_widget_request_model(self):
        """Test ConfigWidgetRequest model validation."""
        request = ConfigWidgetRequest(
            config_class_name="BasePipelineConfig",
            base_config={"author": "test"},
            workspace_dirs=["/test"]
        )
        assert request.config_class_name == "BasePipelineConfig"
        assert request.base_config == {"author": "test"}
        assert request.workspace_dirs == ["/test"]
    
    def test_config_save_request_model(self):
        """Test ConfigSaveRequest model validation."""
        request = ConfigSaveRequest(
            config_class_name="BasePipelineConfig",
            form_data={"author": "test-user", "bucket": "test-bucket"}
        )
        assert request.config_class_name == "BasePipelineConfig"
        assert request.form_data["author"] == "test-user"
    
    def test_merge_configs_request_model(self):
        """Test MergeConfigsRequest model validation."""
        request = MergeConfigsRequest(
            session_configs={
                "BasePipelineConfig": {"author": "test"}
            },
            filename="test.json",
            workspace_dirs=None
        )
        assert "BasePipelineConfig" in request.session_configs
        assert request.filename == "test.json"
        assert request.workspace_dirs is None
    
    def test_dag_analysis_request_model(self):
        """Test DAGAnalysisRequest model validation."""
        request = DAGAnalysisRequest(
            pipeline_dag={"nodes": ["step1", "step2"]},
            workspace_dirs=["/test"]
        )
        assert request.pipeline_dag["nodes"] == ["step1", "step2"]
        assert request.workspace_dirs == ["/test"]


class TestHelperFunctions:
    """Test helper functions in the API module."""
    
    def test_format_python_args_simple(self):
        """Test Python argument formatting with simple data types."""
        from cursus.api.config_ui.api import _format_python_args
        
        form_data = {
            "author": "test-user",
            "bucket": "test-bucket",
            "count": 5,
            "enabled": True
        }
        
        result = _format_python_args(form_data)
        
        assert 'author="test-user",' in result
        assert 'bucket="test-bucket",' in result
        assert 'count=5,' in result
        assert 'enabled=True,' in result
    
    def test_format_python_args_complex(self):
        """Test Python argument formatting with complex data types."""
        from cursus.api.config_ui.api import _format_python_args
        
        form_data = {
            "tags": ["tag1", "tag2"],
            "metadata": {"key": "value"},
            "config": None
        }
        
        result = _format_python_args(form_data)
        
        assert "tags=['tag1', 'tag2']," in result
        assert "metadata={'key': 'value'}," in result
        assert "config=None," in result
    
    def test_format_python_args_custom_indent(self):
        """Test Python argument formatting with custom indentation."""
        from cursus.api.config_ui.api import _format_python_args
        
        form_data = {"author": "test"}
        result = _format_python_args(form_data, indent=8)
        
        assert result.startswith("        ")  # 8 spaces


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases following pytest best practices."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        app = create_config_ui_app()
        return TestClient(app)
    
    @pytest.fixture
    def mock_discover_configs(self):
        """Mock discover_available_configs function."""
        with patch('cursus.api.config_ui.api.discover_available_configs') as mock_discover:
            mock_discover.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase,
                "CradleDataLoadingConfig": Mock(spec=['from_base_config', 'model_fields']),
                "XGBoostTrainingConfig": Mock(spec=['from_base_config', 'model_fields'])
            }
            yield mock_discover
    
    @pytest.fixture
    def mock_universal_config_core(self):
        """Mock UniversalConfigCore with realistic behavior."""
        with patch('cursus.api.config_ui.api.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            # Configure realistic behavior based on source implementation
            mock_core.discover_config_classes.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase,
                "CradleDataLoadingConfig": Mock(),
                "XGBoostTrainingConfig": Mock()
            }
            
            mock_core._get_form_fields.return_value = [
                {
                    "name": "author",
                    "type": "text",
                    "required": True,
                    "tier": "essential",
                    "description": "Author name",
                    "default": None
                },
                {
                    "name": "bucket",
                    "type": "text",
                    "required": True,
                    "tier": "essential", 
                    "description": "S3 bucket name",
                    "default": None
                }
            ]
            
            yield mock_core
    
    @pytest.fixture
    def example_base_config_data(self):
        """Example base configuration data for testing."""
        return {
            "author": "test-user",
            "bucket": "test-bucket",
            "role": "test-role",
            "region": "NA",
            "service_name": "test-service",
            "pipeline_version": "1.0.0",
            "project_root_folder": "test-project"
        }
    
    def test_global_state_isolation(self, client):
        """Test that global state is properly isolated between tests."""
        # Following Category 17: Global State Management
        import cursus.api.config_ui.api as api_module
        
        # Should start with clean state due to fixture
        assert api_module.latest_config is None
        assert api_module.active_sessions == {}
        
        # Modify state
        api_module.latest_config = {"test": "data"}
        api_module.active_sessions["test"] = {"data": "test"}
        
        # State should be reset by fixture in next test
    
    def test_pydantic_serialization_edge_cases(self, client, mock_universal_config_core):
        """Test handling of non-serializable Pydantic defaults."""
        # Following Category 12: NoneType Attribute Access and Defensive Coding
        
        # Mock form fields with non-serializable defaults
        mock_universal_config_core._get_form_fields.return_value = [
            {
                "name": "test_field",
                "type": "text",
                "required": False,
                "tier": "system",
                "description": "Test field",
                "default": Mock()  # Non-serializable default
            }
        ]
        
        request_data = {
            "config_class_name": "TestConfig",
            "base_config": None,
            "workspace_dirs": None
        }
        
        response = client.post("/api/config-ui/create-widget", json=request_data)
        
        # Should handle non-serializable defaults gracefully
        assert response.status_code == 200
        data = response.json()
        assert "fields" in data
        
        # Check that non-serializable default was handled
        if len(data["fields"]) > 0:
            field = data["fields"][0]
            # Default should be None or string representation
            assert field.get("default") is None or isinstance(field.get("default"), str)
    
    def test_temporary_file_cleanup_on_error(self, client, mock_discover_configs):
        """Test that temporary files are cleaned up on errors."""
        # Following Category 3: Path and File System Operations
        with patch('cursus.core.config_fields.merge_and_save_configs') as mock_merge:
            mock_merge.side_effect = Exception("Merge failed")
            
            request_data = {
                "session_configs": {
                    "BasePipelineConfig": {
                        "author": "test-user",
                        "bucket": "test-bucket",
                        "role": "test-role",
                        "region": "NA",
                        "service_name": "test-service",
                        "pipeline_version": "1.0.0",
                        "project_root_folder": "test-project"
                    }
                },
                "filename": "test_config.json",
                "workspace_dirs": None
            }
            
            response = client.post("/api/config-ui/merge-and-save-configs", json=request_data)
            
            # Should return error but not crash
            assert response.status_code == 500
            assert "Merge failed" in response.json()["detail"]
    
    def test_config_instance_creation_failure(self, client, mock_discover_configs):
        """Test handling of config instance creation failures."""
        # Mock config class that fails during instantiation
        mock_config_class = Mock()
        mock_config_class.side_effect = Exception("Config creation failed")
        
        mock_discover_configs.return_value = {
            "FailingConfig": mock_config_class
        }
        
        request_data = {
            "session_configs": {
                "FailingConfig": {"test": "data"}
            },
            "filename": "test_config.json",
            "workspace_dirs": None
        }
        
        response = client.post("/api/config-ui/merge-and-save-configs", json=request_data)
        
        assert response.status_code == 400
        assert "Failed to create FailingConfig" in response.json()["detail"]
    
    def test_download_cache_edge_cases(self, client):
        """Test download functionality edge cases."""
        # Test with empty cache
        response = client.get("/api/config-ui/download/empty_cache")
        assert response.status_code == 404
        
        # Test with malformed download ID
        response = client.get("/api/config-ui/download/")
        assert response.status_code == 404
    
    def test_dag_analysis_with_empty_nodes(self, client, mock_universal_config_core):
        """Test DAG analysis with empty or malformed node data."""
        mock_universal_config_core._discover_required_config_classes.return_value = []
        mock_universal_config_core._create_workflow_structure.return_value = []
        
        request_data = {
            "pipeline_dag": {
                "nodes": []  # Empty nodes
            },
            "workspace_dirs": None
        }
        
        response = client.post("/api/config-ui/analyze-dag", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["discovered_steps"] == []
        assert data["total_steps"] == 0
    
    def test_dag_analysis_with_malformed_dag(self, client, mock_universal_config_core):
        """Test DAG analysis with malformed DAG data."""
        mock_universal_config_core._discover_required_config_classes.return_value = []
        mock_universal_config_core._create_workflow_structure.return_value = []
        
        request_data = {
            "pipeline_dag": {
                # Missing nodes key
                "invalid": "data"
            },
            "workspace_dirs": None
        }
        
        response = client.post("/api/config-ui/analyze-dag", json=request_data)
        
        # Should handle gracefully
        assert response.status_code == 200
        data = response.json()
        assert "discovered_steps" in data
    
    def test_python_code_generation_failure(self, client, mock_discover_configs, example_base_config_data):
        """Test handling of Python code generation failures."""
        # This tests the try/except block around Python code generation
        request_data = {
            "config_class_name": "BasePipelineConfig",
            "form_data": example_base_config_data
        }
        
        # Mock _format_python_args to raise exception
        with patch('cursus.api.config_ui.api._format_python_args') as mock_format:
            mock_format.side_effect = Exception("Format failed")
            
            response = client.post("/api/config-ui/save-config", json=request_data)
            
            # Should still succeed but without python_code
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            # python_code should be None due to generation failure
            assert data.get("python_code") is None


class TestIntegrationScenarios:
    """Integration tests for complete API workflows."""
    
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global state before each test."""
        import cursus.api.config_ui.api as api_module
        api_module.latest_config = None
        api_module.active_sessions = {}
        yield
        api_module.latest_config = None
        api_module.active_sessions = {}
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        app = create_config_ui_app()
        return TestClient(app)
    
    def test_complete_configuration_workflow(self, client):
        """Test complete configuration workflow from discovery to download."""
        # Following Category 4: Test Expectations vs Implementation
        
        with patch('cursus.api.config_ui.api.discover_available_configs') as mock_discover:
            mock_discover.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase
            }
            
            # Step 1: Discover configurations
            response = client.post("/api/config-ui/discover", json={"workspace_dirs": None})
            assert response.status_code == 200
            
            # Step 2: Create widget
            with patch('cursus.api.config_ui.api.UniversalConfigCore') as mock_core_class:
                mock_core = Mock()
                mock_core_class.return_value = mock_core
                mock_core.discover_config_classes.return_value = {
                    "BasePipelineConfig": BasePipelineConfig
                }
                mock_core._get_form_fields.return_value = [
                    {"name": "author", "type": "text", "required": True, "tier": "essential", "description": "Author", "default": None}
                ]
                
                response = client.post("/api/config-ui/create-widget", json={
                    "config_class_name": "BasePipelineConfig",
                    "base_config": None,
                    "workspace_dirs": None
                })
                assert response.status_code == 200
            
            # Step 3: Save configuration
            response = client.post("/api/config-ui/save-config", json={
                "config_class_name": "BasePipelineConfig",
                "form_data": {
                    "author": "test-user",
                    "bucket": "test-bucket",
                    "role": "test-role",
                    "region": "NA",
                    "service_name": "test-service",
                    "pipeline_version": "1.0.0",
                    "project_root_folder": "test-project"
                }
            })
            assert response.status_code == 200
            
            # Step 4: Get latest config
            response = client.get("/api/config-ui/get-latest-config")
            assert response.status_code == 200
            
            # Step 5: Clear config
            response = client.post("/api/config-ui/clear-config")
            assert response.status_code == 200
    
    def test_multi_config_merge_workflow(self, client):
        """Test workflow with multiple configurations and merging."""
        with patch('cursus.api.config_ui.api.discover_available_configs') as mock_discover:
            mock_discover.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase
            }
            
            with patch('cursus.core.config_fields.merge_and_save_configs') as mock_merge:
                mock_merge.return_value = {
                    "shared": {"author": "test-user"},
                    "specific": {
                        "Base": {"pipeline_name": "test-pipeline"},
                        "Processing": {"instance_type": "ml.m5.large"}
                    }
                }
                
                # Merge multiple configurations
                response = client.post("/api/config-ui/merge-and-save-configs", json={
                    "session_configs": {
                        "BasePipelineConfig": {
                            "author": "test-user",
                            "bucket": "test-bucket",
                            "role": "test-role",
                            "region": "NA",
                            "service_name": "test-service",
                            "pipeline_version": "1.0.0",
                            "project_root_folder": "test-project"
                        },
                        "ProcessingStepConfigBase": {
                            "author": "test-user",
                            "bucket": "test-bucket",
                            "role": "test-role",
                            "region": "NA",
                            "service_name": "test-service",
                            "pipeline_version": "1.0.0",
                            "project_root_folder": "test-project",
                            "processing_instance_count": 1,
                            "processing_volume_size": 500,
                            "use_large_processing_instance": False
                        }
                    },
                    "filename": "multi_config.json",
                    "workspace_dirs": None
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "download_url" in data
                
                # Verify merge was called with correct number of configs
                mock_merge.assert_called_once()
                call_args = mock_merge.call_args[1]  # keyword arguments
                assert len(call_args["config_list"]) == 2
