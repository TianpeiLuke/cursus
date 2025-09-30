"""
Integration tests for the step_catalog module.

Tests the complete integration of all components and the factory function.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from cursus.step_catalog import create_step_catalog, StepCatalog, StepInfo, FileMetadata, StepSearchResult, ConfigAutoDiscovery


class TestModuleIntegration:
    """Test complete module integration."""
    
    def test_module_imports(self):
        """Test that all public components can be imported."""
        from cursus.step_catalog import (
            StepCatalog,
            StepInfo,
            FileMetadata,
            StepSearchResult,
            ConfigAutoDiscovery,
            create_step_catalog
        )
        
        # All imports should succeed
        assert StepCatalog is not None
        assert StepInfo is not None
        assert FileMetadata is not None
        assert StepSearchResult is not None
        assert ConfigAutoDiscovery is not None
        assert create_step_catalog is not None
    
    def test_factory_function_basic(self):
        """Test basic factory function usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Test with explicit use_unified=True to get StepCatalog
            catalog = create_step_catalog(workspace_root, use_unified=True)
            
            assert isinstance(catalog, StepCatalog)
            # Updated for new dual search space API
            assert catalog.workspace_dirs == [workspace_root]
    
    def test_factory_function_with_feature_flags(self):
        """Test factory function with different feature flag configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Test explicit True - should return StepCatalog
            catalog = create_step_catalog(workspace_root, use_unified=True)
            assert isinstance(catalog, StepCatalog)
            
            # Test explicit False - should return LegacyDiscoveryWrapper
            catalog = create_step_catalog(workspace_root, use_unified=False)
            from cursus.step_catalog.adapters.legacy_wrappers import LegacyDiscoveryWrapper
            assert isinstance(catalog, LegacyDiscoveryWrapper)
            
            # Test None with environment variable false (default)
            with patch.dict(os.environ, {'USE_UNIFIED_CATALOG': 'false'}):
                catalog = create_step_catalog(workspace_root, use_unified=None)
                assert isinstance(catalog, LegacyDiscoveryWrapper)
    
    def test_factory_function_environment_variable(self):
        """Test factory function respects environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Test with environment variable set to true
            with patch.dict(os.environ, {'USE_UNIFIED_CATALOG': 'true'}):
                catalog = create_step_catalog(workspace_root)
                assert isinstance(catalog, StepCatalog)
            
            # Test with environment variable set to false
            with patch.dict(os.environ, {'USE_UNIFIED_CATALOG': 'false'}):
                catalog = create_step_catalog(workspace_root)
                from cursus.step_catalog.adapters.legacy_wrappers import LegacyDiscoveryWrapper
                assert isinstance(catalog, LegacyDiscoveryWrapper)
                catalog = create_step_catalog(workspace_root)
                from cursus.step_catalog.adapters.legacy_wrappers import LegacyDiscoveryWrapper
                assert isinstance(catalog, LegacyDiscoveryWrapper)
                catalog = create_step_catalog(workspace_root)
                from cursus.step_catalog.adapters.legacy_wrappers import LegacyDiscoveryWrapper
                assert isinstance(catalog, LegacyDiscoveryWrapper)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def realistic_workspace(self):
        """Create a realistic workspace structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Create core directory structure
            core_steps = workspace_root / "src" / "cursus" / "steps"
            
            # Create component directories
            scripts_dir = core_steps / "scripts"
            contracts_dir = core_steps / "contracts"
            configs_dir = core_steps / "configs"
            
            for dir_path in [scripts_dir, contracts_dir, configs_dir]:
                dir_path.mkdir(parents=True)
            
            # Create sample files
            (scripts_dir / "data_preprocessing.py").write_text("# Data preprocessing script")
            (scripts_dir / "model_training.py").write_text("# Model training script")
            (contracts_dir / "data_preprocessing_contract.py").write_text("# Data preprocessing contract")
            (configs_dir / "config_data_preprocessing_step.py").write_text("""
from pydantic import BaseModel

class DataPreprocessingConfig(BaseModel):
    input_path: str
    output_path: str
""")
            
            # Create workspace directory structure
            workspace_steps = (
                workspace_root / "development" / "projects" / "alpha" / 
                "src" / "cursus_dev" / "steps" / "scripts"
            )
            workspace_steps.mkdir(parents=True)
            (workspace_steps / "custom_preprocessing.py").write_text("# Custom preprocessing script")
            
            yield workspace_root
    
    def test_complete_discovery_workflow(self, realistic_workspace):
        """Test complete step discovery workflow."""
        # Mock package root to point to our test workspace
        with patch.object(StepCatalog, '_find_package_root', return_value=realistic_workspace / "src" / "cursus"):
            # Configure workspace directories to point directly to the steps directory
            workspace_dirs = [realistic_workspace / "development" / "projects" / "alpha" / "src" / "cursus_dev" / "steps"]
            catalog = create_step_catalog(workspace_dirs, use_unified=True)
            
            # Mock registry to avoid import issues
            mock_registry = {
                "data_preprocessing": {
                    "config_class": "DataPreprocessingConfig",
                    "description": "Preprocesses data for training"
                },
                "model_training": {
                    "config_class": "ModelTrainingConfig", 
                    "description": "Trains ML models"
                }
            }
            
            with patch('cursus.registry.step_names.get_step_names', return_value=mock_registry):
                # Test US1: Query by Step Name
                step_info = catalog.get_step_info("data_preprocessing")
                assert step_info is not None
                assert step_info.step_name == "data_preprocessing"
                assert step_info.workspace_id == "core"
                assert "script" in step_info.file_components
                assert "contract" in step_info.file_components
                
                # Test US2: Reverse Lookup
                script_path = realistic_workspace / "src" / "cursus" / "steps" / "scripts" / "data_preprocessing.py"
                found_step = catalog.find_step_by_component(str(script_path))
                assert found_step == "data_preprocessing"
                
                # Test US3: Multi-Workspace Discovery
                all_steps = catalog.list_available_steps()
                assert "data_preprocessing" in all_steps
                assert "model_training" in all_steps
                assert "custom_preprocessing" in all_steps
                
                core_steps = catalog.list_available_steps(workspace_id="core")
                workspace_steps = catalog.list_available_steps(workspace_id="steps")  # Workspace ID is "steps" (directory name)
                
                assert "data_preprocessing" in core_steps
                assert "model_training" in core_steps
                assert "custom_preprocessing" in workspace_steps
                
                # Test US4: Search
                search_results = catalog.search_steps("preprocessing")
                assert len(search_results) >= 2  # Should find both preprocessing steps
                
                step_names = [r.step_name for r in search_results]
                assert "data_preprocessing" in step_names
                assert "custom_preprocessing" in step_names
                
                # Test US5: Config Discovery
                with patch.object(catalog.config_discovery, 'discover_config_classes') as mock_discover:
                    mock_discover.return_value = {"DataPreprocessingConfig": Mock}
                    
                    config_classes = catalog.discover_config_classes()
                    assert "DataPreprocessingConfig" in config_classes
    
    def test_performance_and_metrics(self, realistic_workspace):
        """Test performance characteristics and metrics collection."""
        catalog = create_step_catalog(realistic_workspace, use_unified=True)
        
        with patch('cursus.registry.step_names.STEP_NAMES', {}):
            # Perform multiple operations
            catalog.get_step_info("data_preprocessing")
            catalog.get_step_info("nonexistent_step")
            catalog.search_steps("preprocessing")
            catalog.list_available_steps()
            
            # Check metrics
            metrics = catalog.get_metrics_report()
            
            assert metrics['total_queries'] > 0
            assert metrics['success_rate'] >= 0.0
            assert metrics['avg_response_time_ms'] >= 0.0
            assert metrics['index_build_time_s'] >= 0.0
            assert metrics['total_steps_indexed'] >= 0
            assert metrics['total_workspaces'] >= 0
    
    def test_error_resilience(self, realistic_workspace):
        """Test system resilience to various error conditions."""
        catalog = create_step_catalog(realistic_workspace, use_unified=True)
        
        # Test with registry import error
        with patch('cursus.registry.step_names.STEP_NAMES', side_effect=ImportError("Registry not found")):
            # Should not crash
            step_info = catalog.get_step_info("any_step")
            # May return None, but should not raise exception
            
            all_steps = catalog.list_available_steps()
            # Should return list (possibly empty), not crash
            assert isinstance(all_steps, list)
            
            search_results = catalog.search_steps("test")
            # Should return list (possibly empty), not crash
            assert isinstance(search_results, list)
        
        # Test with file system errors
        with patch('pathlib.Path.exists', return_value=False):
            # Should handle missing directories gracefully
            step_info = catalog.get_step_info("test_step")
            # Should not crash
        
        # Test with config discovery errors
        with patch.object(catalog.config_discovery, 'discover_config_classes', side_effect=Exception("Config error")):
            # Should handle config discovery errors gracefully
            try:
                config_classes = catalog.discover_config_classes()
                # If it returns, should be a dict
                assert isinstance(config_classes, dict)
            except Exception:
                # If it raises, that's also acceptable for this error condition
                pass


class TestBackwardCompatibility:
    """Test backward compatibility and migration scenarios."""
    
    def test_existing_api_compatibility(self):
        """Test that existing APIs remain compatible."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            catalog = create_step_catalog(workspace_root, use_unified=True)
            
            # Test that all expected methods exist and are callable
            assert hasattr(catalog, 'get_step_info')
            assert callable(catalog.get_step_info)
            
            assert hasattr(catalog, 'find_step_by_component')
            assert callable(catalog.find_step_by_component)
            
            assert hasattr(catalog, 'list_available_steps')
            assert callable(catalog.list_available_steps)
            
            assert hasattr(catalog, 'search_steps')
            assert callable(catalog.search_steps)
            
            assert hasattr(catalog, 'discover_config_classes')
            assert callable(catalog.discover_config_classes)
            
            assert hasattr(catalog, 'build_complete_config_classes')
            assert callable(catalog.build_complete_config_classes)
    
    def test_data_model_compatibility(self):
        """Test that data models are compatible with expected usage."""
        from datetime import datetime
        
        # Test StepInfo creation and usage
        step_info = StepInfo(
            step_name="test_step",
            workspace_id="core",
            registry_data={"config_class": "TestConfig"},
            file_components={
                "script": FileMetadata(
                    path=Path("/test/script.py"),
                    file_type="script",
                    modified_time=datetime.now()
                )
            }
        )
        
        # Test property access
        assert step_info.config_class == "TestConfig"
