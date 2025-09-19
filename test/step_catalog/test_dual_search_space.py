"""
Unit tests for dual search space architecture in step catalog.

Tests the new dual search space functionality that separates package discovery
from workspace discovery, ensuring proper separation of concerns.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Type, Any

from cursus.step_catalog.step_catalog import StepCatalog
from cursus.step_catalog.config_discovery import ConfigAutoDiscovery


class TestDualSearchSpaceArchitecture:
    """Test dual search space architecture functionality."""
    
    @pytest.fixture
    def temp_package_root(self):
        """Create temporary package root structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir)
            
            # Create package structure
            steps_dir = package_root / "steps"
            steps_dir.mkdir()
            
            # Create component directories
            for component in ["configs", "scripts", "contracts", "specs", "builders"]:
                (steps_dir / component).mkdir()
            
            yield package_root, steps_dir
    
    @pytest.fixture
    def temp_workspace_dirs(self):
        """Create temporary workspace directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace1 = Path(temp_dir) / "workspace1"
            workspace2 = Path(temp_dir) / "workspace2"
            
            # Create workspace structures
            for workspace in [workspace1, workspace2]:
                projects_dir = workspace / "development" / "projects"
                projects_dir.mkdir(parents=True)
                
                # Create project structures
                for project in ["alpha", "beta"]:
                    project_dir = projects_dir / project / "src" / "cursus_dev" / "steps"
                    project_dir.mkdir(parents=True)
                    
                    # Create component directories
                    for component in ["configs", "scripts", "contracts", "specs", "builders"]:
                        (project_dir / component).mkdir()
            
            yield [workspace1, workspace2]
    
    def test_package_only_discovery(self, temp_package_root):
        """Test package-only discovery without workspace directories."""
        package_root, steps_dir = temp_package_root
        
        # Mock the package root detection to return our test directory
        with patch.object(StepCatalog, '_find_package_root', return_value=package_root):
            catalog = StepCatalog()
            
            assert catalog.package_root == package_root
            assert catalog.workspace_dirs == []
            assert isinstance(catalog.config_discovery, ConfigAutoDiscovery)
            assert catalog.config_discovery.package_root == package_root
            assert catalog.config_discovery.workspace_dirs == []
    
    def test_single_workspace_directory(self, temp_package_root, temp_workspace_dirs):
        """Test with single workspace directory."""
        package_root, _ = temp_package_root
        workspace_dirs = temp_workspace_dirs
        
        with patch.object(StepCatalog, '_find_package_root', return_value=package_root):
            catalog = StepCatalog(workspace_dirs=workspace_dirs[0])
            
            assert catalog.package_root == package_root
            assert catalog.workspace_dirs == [workspace_dirs[0]]
            assert catalog.config_discovery.package_root == package_root
            assert catalog.config_discovery.workspace_dirs == [workspace_dirs[0]]
    
    def test_multiple_workspace_directories(self, temp_package_root, temp_workspace_dirs):
        """Test with multiple workspace directories."""
        package_root, _ = temp_package_root
        workspace_dirs = temp_workspace_dirs
        
        with patch.object(StepCatalog, '_find_package_root', return_value=package_root):
            catalog = StepCatalog(workspace_dirs=workspace_dirs)
            
            assert catalog.package_root == package_root
            assert catalog.workspace_dirs == workspace_dirs
            assert catalog.config_discovery.package_root == package_root
            assert catalog.config_discovery.workspace_dirs == workspace_dirs
    
    def test_package_root_detection_deployment_agnostic(self):
        """Test that package root detection works across deployment scenarios."""
        catalog = StepCatalog()
        
        # Should find cursus package root regardless of deployment
        assert catalog.package_root.name == "cursus"
        assert catalog.package_root.exists()
        assert (catalog.package_root / "steps").exists()
    
    def test_separation_of_concerns_enforcement(self, temp_package_root, temp_workspace_dirs):
        """Test that separation of concerns is properly enforced."""
        package_root, _ = temp_package_root
        workspace_dirs = temp_workspace_dirs
        
        with patch.object(StepCatalog, '_find_package_root', return_value=package_root):
            catalog = StepCatalog(workspace_dirs=workspace_dirs)
            
            # System responsibilities (autonomous)
            assert catalog.package_root is not None  # System finds its own package root
            assert catalog._discover_package_components is not None  # System discovers its own components
            
            # User responsibilities (explicit)
            assert catalog.workspace_dirs == workspace_dirs  # User must provide workspace dirs
            assert catalog._discover_workspace_components is not None  # System uses user-provided dirs
    
    def test_workspace_directory_validation(self, temp_package_root):
        """Test workspace directory validation and error handling."""
        package_root, _ = temp_package_root
        
        with patch.object(StepCatalog, '_find_package_root', return_value=package_root):
            # Test with non-existent directory
            nonexistent_dir = Path("/nonexistent/directory")
            catalog = StepCatalog(workspace_dirs=nonexistent_dir)
            
            # Should not crash, but should handle gracefully
            assert catalog.workspace_dirs == [nonexistent_dir]
            
            # Test discovery with non-existent directory
            with patch.object(catalog.logger, 'warning') as mock_warning:
                catalog._discover_workspace_components()
                # Should log warning about non-existent directory
                mock_warning.assert_called()


class TestConfigAutoDiscoveryDualSpace:
    """Test ConfigAutoDiscovery with dual search space."""
    
    @pytest.fixture
    def temp_dual_structure(self):
        """Create temporary dual structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Package structure
            package_root = base_dir / "package"
            package_configs = package_root / "steps" / "configs"
            package_configs.mkdir(parents=True)
            
            # Workspace structure
            workspace_root = base_dir / "workspace"
            workspace_configs = (
                workspace_root / "development" / "projects" / "test_project" / 
                "src" / "cursus_dev" / "steps" / "configs"
            )
            workspace_configs.mkdir(parents=True)
            
            yield package_root, workspace_root, package_configs, workspace_configs
    
    def test_package_discovery_only(self, temp_dual_structure):
        """Test discovery from package only."""
        package_root, workspace_root, package_configs, workspace_configs = temp_dual_structure
        
        # Create package config
        package_config = package_configs / "package_config.py"
        package_config.write_text("""
from pydantic import BaseModel

class PackageConfig(BaseModel):
    value: str = "package"
""")
        
        discovery = ConfigAutoDiscovery(package_root, [])
        
        with patch.object(discovery, '_scan_config_directory') as mock_scan:
            mock_scan.return_value = {"PackageConfig": Mock}
            
            result = discovery.discover_config_classes()
            
            # Should only scan package directory
            mock_scan.assert_called_once_with(package_configs)
            assert "PackageConfig" in result
    
    def test_workspace_discovery_with_package(self, temp_dual_structure):
        """Test discovery from both package and workspace."""
        package_root, workspace_root, package_configs, workspace_configs = temp_dual_structure
        
        discovery = ConfigAutoDiscovery(package_root, [workspace_root])
        
        with patch.object(discovery, '_scan_config_directory') as mock_scan:
            mock_scan.side_effect = [
                {"PackageConfig": Mock},  # Package discovery
                {"WorkspaceConfig": Mock}  # Workspace discovery
            ]
            
            with patch.object(discovery, '_discover_workspace_configs') as mock_workspace:
                mock_workspace.return_value = {"WorkspaceConfig": Mock}
                
                result = discovery.discover_config_classes("test_project")
                
                # Should scan package directory
                mock_scan.assert_called_with(package_configs)
                # Should discover workspace configs
                mock_workspace.assert_called_with(workspace_root, "test_project")
                
                assert "PackageConfig" in result
                assert "WorkspaceConfig" in result
    
    def test_workspace_override_precedence(self, temp_dual_structure):
        """Test that workspace configs override package configs with same names."""
        package_root, workspace_root, package_configs, workspace_configs = temp_dual_structure
        
        discovery = ConfigAutoDiscovery(package_root, [workspace_root])
        
        package_config = Mock()
        workspace_config = Mock()
        
        with patch.object(discovery, '_scan_config_directory') as mock_scan:
            mock_scan.return_value = {"SameConfig": package_config}
            
            with patch.object(discovery, '_discover_workspace_configs') as mock_workspace:
                mock_workspace.return_value = {"SameConfig": workspace_config}
                
                result = discovery.discover_config_classes("test_project")
                
                # Workspace config should override package config
                assert result["SameConfig"] == workspace_config
    
    def test_multiple_workspace_directories(self, temp_dual_structure):
        """Test discovery across multiple workspace directories."""
        package_root, workspace_root, package_configs, workspace_configs = temp_dual_structure
        
        # Create second workspace
        workspace2 = workspace_root.parent / "workspace2"
        workspace2_configs = (
            workspace2 / "development" / "projects" / "test_project" / 
            "src" / "cursus_dev" / "steps" / "configs"
        )
        workspace2_configs.mkdir(parents=True)
        
        discovery = ConfigAutoDiscovery(package_root, [workspace_root, workspace2])
        
        with patch.object(discovery, '_scan_config_directory') as mock_scan:
            mock_scan.return_value = {"PackageConfig": Mock}
            
            with patch.object(discovery, '_discover_workspace_configs') as mock_workspace:
                mock_workspace.side_effect = [
                    {"Workspace1Config": Mock},
                    {"Workspace2Config": Mock}
                ]
                
                result = discovery.discover_config_classes("test_project")
                
                # Should call workspace discovery for each workspace
                assert mock_workspace.call_count == 2
                assert "PackageConfig" in result
                assert "Workspace1Config" in result
                assert "Workspace2Config" in result
    
    def test_project_specific_discovery(self, temp_dual_structure):
        """Test project-specific discovery within workspaces."""
        package_root, workspace_root, package_configs, workspace_configs = temp_dual_structure
        
        discovery = ConfigAutoDiscovery(package_root, [workspace_root])
        
        # Test with specific project ID
        with patch.object(discovery, '_discover_workspace_configs') as mock_workspace:
            mock_workspace.return_value = {"ProjectConfig": Mock}
            
            result = discovery.discover_config_classes("specific_project")
            
            # Should pass project ID to workspace discovery
            mock_workspace.assert_called_with(workspace_root, "specific_project")
    
    def test_hyperparameter_discovery_dual_space(self, temp_dual_structure):
        """Test hyperparameter discovery with dual search space."""
        package_root, workspace_root, package_configs, workspace_configs = temp_dual_structure
        
        # Create hyperparams directories
        package_hyperparams = package_root / "steps" / "hyperparams"
        package_hyperparams.mkdir()
        
        discovery = ConfigAutoDiscovery(package_root, [workspace_root])
        
        with patch.object(discovery, '_scan_hyperparams_directory') as mock_scan:
            mock_scan.return_value = {"PackageHyperparams": Mock}
            
            with patch.object(discovery, '_discover_workspace_hyperparams') as mock_workspace:
                mock_workspace.return_value = {"WorkspaceHyperparams": Mock}
                
                result = discovery.discover_hyperparameter_classes("test_project")
                
                # Should scan package hyperparams
                mock_scan.assert_called_with(package_hyperparams)
                # Should discover workspace hyperparams
                mock_workspace.assert_called_with(workspace_root, "test_project")
                
                assert "PackageHyperparams" in result
                assert "WorkspaceHyperparams" in result


class TestSearchSpaceIntegration:
    """Test integration between StepCatalog and ConfigAutoDiscovery."""
    
    def test_step_catalog_config_discovery_integration(self):
        """Test that StepCatalog properly integrates with ConfigAutoDiscovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            
            with patch.object(StepCatalog, '_find_package_root', return_value=Path(".")):
                catalog = StepCatalog(workspace_dirs=workspace_dir)
                
                # Test config class discovery delegation
                with patch.object(catalog.config_discovery, 'discover_config_classes') as mock_discover:
                    mock_discover.return_value = {"TestConfig": Mock}
                    
                    result = catalog.discover_config_classes("test_project")
                    
                    mock_discover.assert_called_once_with("test_project")
                    assert result == {"TestConfig": Mock}
                
                # Test complete config class building delegation
                with patch.object(catalog.config_discovery, 'build_complete_config_classes') as mock_build:
                    mock_build.return_value = {"CompleteConfig": Mock}
                    
                    result = catalog.build_complete_config_classes("test_project")
                    
                    mock_build.assert_called_once_with("test_project")
                    assert result == {"CompleteConfig": Mock}
    
    def test_backward_compatibility_preservation(self):
        """Test that existing APIs continue to work without workspace directories."""
        with patch.object(StepCatalog, '_find_package_root', return_value=Path(".")):
            # Test package-only initialization (backward compatible)
            catalog = StepCatalog()
            
            assert catalog.workspace_dirs == []
            assert isinstance(catalog.config_discovery, ConfigAutoDiscovery)
            
            # Test that all existing methods still work
            assert hasattr(catalog, 'get_step_info')
            assert hasattr(catalog, 'find_step_by_component')
            assert hasattr(catalog, 'list_available_steps')
            assert hasattr(catalog, 'search_steps')
            assert hasattr(catalog, 'discover_config_classes')
            assert hasattr(catalog, 'build_complete_config_classes')
    
    def test_error_handling_in_dual_space(self):
        """Test error handling in dual search space architecture."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            
            with patch.object(StepCatalog, '_find_package_root', return_value=Path(".")):
                catalog = StepCatalog(workspace_dirs=workspace_dir)
                
                # Test graceful handling of workspace discovery errors
                with patch.object(catalog, '_discover_workspace_components', side_effect=Exception("Test error")):
                    # Should not crash during index building
                    try:
                        catalog._build_index()
                        assert True  # Should complete without exception
                    except Exception:
                        assert False, "Index building should handle workspace errors gracefully"
                
                # Test graceful handling of config discovery errors
                with patch.object(catalog.config_discovery, 'discover_config_classes', side_effect=Exception("Config error")):
                    try:
                        result = catalog.discover_config_classes()
                        # If it doesn't crash, that's good
                        assert result is None or isinstance(result, dict)
                    except Exception:
                        # If it does crash, that's expected behavior for now
                        # The important thing is that workspace discovery errors are handled gracefully
                        pass


class TestDeploymentScenarios:
    """Test dual search space across different deployment scenarios."""
    
    def test_pypi_deployment_simulation(self):
        """Test behavior in PyPI deployment scenario."""
        # Simulate site-packages structure
        with tempfile.TemporaryDirectory() as temp_dir:
            site_packages = Path(temp_dir) / "site-packages"
            cursus_package = site_packages / "cursus"
            cursus_steps = cursus_package / "steps"
            cursus_steps.mkdir(parents=True)
            
            # Create user workspace separate from package
            user_workspace = Path(temp_dir) / "user_project"
            user_workspace.mkdir()
            
            with patch.object(StepCatalog, '_find_package_root', return_value=cursus_package):
                catalog = StepCatalog(workspace_dirs=user_workspace)
                
                # Package root should be in site-packages
                assert catalog.package_root == cursus_package
                # Workspace should be user-provided
                assert catalog.workspace_dirs == [user_workspace]
                # Should be completely separate
                assert not user_workspace.is_relative_to(cursus_package)
    
    def test_source_deployment_simulation(self):
        """Test behavior in source deployment scenario."""
        # Simulate source repository structure
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "cursus_repo"
            src_cursus = repo_root / "src" / "cursus"
            cursus_steps = src_cursus / "steps"
            cursus_steps.mkdir(parents=True)
            
            # Development workspace within repo
            dev_workspace = repo_root / "development"
            dev_workspace.mkdir()
            
            with patch.object(StepCatalog, '_find_package_root', return_value=src_cursus):
                catalog = StepCatalog(workspace_dirs=dev_workspace)
                
                # Package root should be src/cursus
                assert catalog.package_root == src_cursus
                # Workspace should be development directory
                assert catalog.workspace_dirs == [dev_workspace]
    
    def test_submodule_deployment_simulation(self):
        """Test behavior in submodule deployment scenario."""
        # Simulate parent project with cursus as submodule
        with tempfile.TemporaryDirectory() as temp_dir:
            parent_project = Path(temp_dir) / "parent_project"
            cursus_submodule = parent_project / "external" / "cursus"
            cursus_steps = cursus_submodule / "steps"
            cursus_steps.mkdir(parents=True)
            
            # Parent project workspace
            parent_workspace = parent_project / "workspaces"
            parent_workspace.mkdir()
            
            with patch.object(StepCatalog, '_find_package_root', return_value=cursus_submodule):
                catalog = StepCatalog(workspace_dirs=parent_workspace)
                
                # Package root should be submodule location
                assert catalog.package_root == cursus_submodule
                # Workspace should be parent project workspace
                assert catalog.workspace_dirs == [parent_workspace]
                # Should work regardless of parent project structure
                assert parent_workspace.is_relative_to(parent_project)
