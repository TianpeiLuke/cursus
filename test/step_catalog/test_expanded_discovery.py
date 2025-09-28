"""
Test expanded discovery methods for Phase 4.1 implementation.

This module tests the new discovery methods added in Phase 4.1:
- discover_contracts_with_scripts()
- detect_framework()
- discover_cross_workspace_components()
- get_builder_class_path()
- load_builder_class()
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from cursus.step_catalog.step_catalog import StepCatalog
from cursus.step_catalog.models import StepInfo, FileMetadata
from datetime import datetime


class TestExpandedDiscoveryMethods:
    """Test the expanded discovery methods added in Phase 4.1."""
    
    @pytest.fixture
    def mock_workspace_root(self, tmp_path):
        """Create a mock workspace root directory."""
        return tmp_path
    
    @pytest.fixture
    def catalog_with_test_data(self, mock_workspace_root):
        """Create a catalog with test data."""
        catalog = StepCatalog(workspace_dirs=[mock_workspace_root])
        
        # Mock some test data
        test_step_info = StepInfo(
            step_name="test_step",
            workspace_id="core",
            registry_data={
                'builder_step_name': 'XGBoostTraining',
                'config_class': 'XGBoostConfig',
                'sagemaker_step_type': 'Training'
            },
            file_components={
                'script': FileMetadata(
                    path=Path("test_script.py"),
                    file_type="script",
                    modified_time=datetime.now()
                ),
                'contract': FileMetadata(
                    path=Path("test_contract.py"),
                    file_type="contract",
                    modified_time=datetime.now()
                )
            }
        )
        
        catalog._step_index = {"test_step": test_step_info}
        catalog._workspace_steps = {"core": ["test_step"]}
        catalog._index_built = True
        
        return catalog
    
    def test_discover_contracts_with_scripts(self, catalog_with_test_data):
        """Test discovery of steps with both contracts and scripts."""
        catalog = catalog_with_test_data
        
        # Test discovery
        steps_with_both = catalog.discover_contracts_with_scripts()
        
        # Verify results
        assert isinstance(steps_with_both, list)
        assert "test_step" in steps_with_both
        assert len(steps_with_both) == 1
    
    def test_discover_contracts_with_scripts_empty(self, mock_workspace_root):
        """Test discovery when no steps have both contracts and scripts."""
        catalog = StepCatalog(workspace_dirs=[mock_workspace_root])
        catalog._step_index = {}
        catalog._index_built = True
        
        steps_with_both = catalog.discover_contracts_with_scripts()
        
        assert isinstance(steps_with_both, list)
        assert len(steps_with_both) == 0
    
    def test_detect_framework_from_registry(self, catalog_with_test_data):
        """Test framework detection from registry data."""
        catalog = catalog_with_test_data
        
        # Test framework detection
        framework = catalog.detect_framework("test_step")
        
        # Should detect xgboost from builder name
        assert framework == "xgboost"
        
        # Test caching
        framework_cached = catalog.detect_framework("test_step")
        assert framework_cached == "xgboost"
        assert "test_step" in catalog._framework_cache
    
    def test_detect_framework_from_step_name(self, mock_workspace_root):
        """Test framework detection from step name patterns."""
        catalog = StepCatalog(workspace_dirs=[mock_workspace_root])
        
        # Create test step with pytorch in name
        pytorch_step = StepInfo(
            step_name="pytorch_training",
            workspace_id="core",
            registry_data={},
            file_components={}
        )
        
        catalog._step_index = {"pytorch_training": pytorch_step}
        catalog._index_built = True
        
        framework = catalog.detect_framework("pytorch_training")
        assert framework == "pytorch"
    
    def test_detect_framework_not_found(self, mock_workspace_root):
        """Test framework detection when step not found."""
        catalog = StepCatalog(workspace_dirs=[mock_workspace_root])
        catalog._step_index = {}
        catalog._index_built = True
        
        framework = catalog.detect_framework("nonexistent_step")
        assert framework is None
    
    def test_discover_cross_workspace_components(self, catalog_with_test_data):
        """Test cross-workspace component discovery."""
        catalog = catalog_with_test_data
        
        # Test discovery
        components = catalog.discover_cross_workspace_components()
        
        # Verify results
        assert isinstance(components, dict)
        assert "core" in components
        assert isinstance(components["core"], list)
        assert "test_step:script" in components["core"]
        assert "test_step:contract" in components["core"]
    
    def test_discover_cross_workspace_components_filtered(self, catalog_with_test_data):
        """Test cross-workspace component discovery with workspace filter."""
        catalog = catalog_with_test_data
        
        # Test with specific workspace IDs
        components = catalog.discover_cross_workspace_components(["core"])
        
        assert isinstance(components, dict)
        assert "core" in components
        assert len(components) == 1
    
    def test_get_builder_class_path_from_registry(self, catalog_with_test_data):
        """Test getting builder class path from registry data."""
        catalog = catalog_with_test_data
        
        # Test path resolution
        builder_path = catalog.get_builder_class_path("test_step")
        
        # Should return registry-based path
        expected_path = "cursus.steps.builders.xgboosttraining.XGBoostTraining"
        assert builder_path == expected_path
    
    def test_get_builder_class_path_from_file(self, mock_workspace_root):
        """Test getting builder class path from file components."""
        catalog = StepCatalog(workspace_dirs=[mock_workspace_root])
        
        # Create test step with builder file
        test_step = StepInfo(
            step_name="file_step",
            workspace_id="core",
            registry_data={},
            file_components={
                'builder': FileMetadata(
                    path=Path("/path/to/builder.py"),
                    file_type="builder",
                    modified_time=datetime.now()
                )
            }
        )
        
        catalog._step_index = {"file_step": test_step}
        catalog._index_built = True
        
        builder_path = catalog.get_builder_class_path("file_step")
        assert builder_path == "/path/to/builder.py"
    
    def test_get_builder_class_path_not_found(self, mock_workspace_root):
        """Test getting builder class path when step not found."""
        catalog = StepCatalog(workspace_dirs=[mock_workspace_root])
        catalog._step_index = {}
        catalog._index_built = True
        
        builder_path = catalog.get_builder_class_path("nonexistent_step")
        assert builder_path is None
    
    def test_load_builder_class_registry_based(self, catalog_with_test_data):
        """Test loading builder class from registry-based path with BuilderAutoDiscovery."""
        catalog = catalog_with_test_data
        
        # Mock BuilderAutoDiscovery to return a builder class
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "XGBoostTraining"
        
        # Mock the builder_discovery component
        mock_builder_discovery = Mock()
        mock_builder_discovery.load_builder_class.return_value = mock_builder_class
        catalog.builder_discovery = mock_builder_discovery
        
        # Test loading
        builder_class = catalog.load_builder_class("test_step")
        
        # Verify results
        assert builder_class == mock_builder_class
        mock_builder_discovery.load_builder_class.assert_called_once_with("test_step")
    
    def test_load_builder_class_fallback_to_registry(self, catalog_with_test_data):
        """Test fallback to registry-based path when BuilderAutoDiscovery fails."""
        catalog = catalog_with_test_data
        
        # Mock BuilderAutoDiscovery to return None (not found)
        mock_builder_discovery = Mock()
        mock_builder_discovery.load_builder_class.return_value = None
        catalog.builder_discovery = mock_builder_discovery
        
        # Test loading - should fall back to registry-based path construction
        builder_class = catalog.load_builder_class("test_step")
        
        # Should return None since we can't actually import the module in tests
        # but the fallback path should be attempted
        assert builder_class is None
        
        # With job type variant fallback logic, it should be called twice:
        # 1. First with "test_step" (original name)
        # 2. Then with "test" (base name after removing "_step" suffix)
        assert mock_builder_discovery.load_builder_class.call_count == 2
        mock_builder_discovery.load_builder_class.assert_any_call("test_step")
        mock_builder_discovery.load_builder_class.assert_any_call("test")
    
    def test_load_builder_class_no_builder_discovery(self, catalog_with_test_data):
        """Test loading builder class when BuilderAutoDiscovery is not available."""
        catalog = catalog_with_test_data
        
        # Set builder_discovery to None (simulating import failure)
        catalog.builder_discovery = None
        
        # Test loading
        builder_class = catalog.load_builder_class("test_step")
        
        # Should return None and log warning
        assert builder_class is None
    
    def test_load_builder_class_not_found(self, mock_workspace_root):
        """Test loading builder class when path not found."""
        catalog = StepCatalog(workspace_dirs=[mock_workspace_root])
        catalog._step_index = {}
        catalog._index_built = True
        
        builder_class = catalog.load_builder_class("nonexistent_step")
        assert builder_class is None
    
    def test_load_builder_class_import_error(self, catalog_with_test_data):
        """Test loading builder class when import fails."""
        catalog = catalog_with_test_data
        
        # This will cause an import error since the module doesn't exist
        builder_class = catalog.load_builder_class("test_step")
        
        # Should return None and log warning
        assert builder_class is None
    
    def test_error_handling_in_discovery_methods(self, mock_workspace_root):
        """Test error handling in discovery methods."""
        catalog = StepCatalog(workspace_dirs=[mock_workspace_root])
        
        # Force an error by not building index and making _ensure_index_built fail
        with patch.object(catalog, '_ensure_index_built', side_effect=Exception("Test error")):
            # All methods should handle errors gracefully
            assert catalog.discover_contracts_with_scripts() == []
            assert catalog.detect_framework("test") is None
            assert catalog.discover_cross_workspace_components() == {}
            assert catalog.get_builder_class_path("test") is None
            assert catalog.load_builder_class("test") is None


class TestExpandedDiscoveryIntegration:
    """Integration tests for expanded discovery methods."""
    
    def test_framework_detection_integration(self, tmp_path):
        """Test framework detection with realistic data."""
        catalog = StepCatalog(tmp_path)
        
        # Create realistic test data
        xgboost_step = StepInfo(
            step_name="xgboost_training",
            workspace_id="core",
            registry_data={'builder_step_name': 'XGBoostTraining'},
            file_components={}
        )
        
        pytorch_step = StepInfo(
            step_name="pytorch_model",
            workspace_id="core",
            registry_data={},
            file_components={}
        )
        
        catalog._step_index = {
            "xgboost_training": xgboost_step,
            "pytorch_model": pytorch_step
        }
        catalog._index_built = True
        
        # Test detection
        assert catalog.detect_framework("xgboost_training") == "xgboost"
        assert catalog.detect_framework("pytorch_model") == "pytorch"
        assert catalog.detect_framework("unknown_step") is None
    
    def test_cross_workspace_discovery_integration(self, tmp_path):
        """Test cross-workspace discovery with multiple workspaces."""
        catalog = StepCatalog(tmp_path)
        
        # Create multi-workspace test data
        core_step = StepInfo(
            step_name="core_step",
            workspace_id="core",
            registry_data={},
            file_components={
                'script': FileMetadata(
                    path=Path("core_script.py"),
                    file_type="script",
                    modified_time=datetime.now()
                )
            }
        )
        
        dev_step = StepInfo(
            step_name="dev_step",
            workspace_id="developer_workspace",
            registry_data={},
            file_components={
                'contract': FileMetadata(
                    path=Path("dev_contract.py"),
                    file_type="contract",
                    modified_time=datetime.now()
                )
            }
        )
        
        catalog._step_index = {
            "core_step": core_step,
            "dev_step": dev_step
        }
        catalog._workspace_steps = {
            "core": ["core_step"],
            "developer_workspace": ["dev_step"]
        }
        catalog._index_built = True
        
        # Test discovery
        components = catalog.discover_cross_workspace_components()
        
        assert "core" in components
        assert "developer_workspace" in components
        assert "core_step:script" in components["core"]
        assert "dev_step:contract" in components["developer_workspace"]
