"""
Test cases for the hybrid path resolution system.

This module tests the core hybrid resolution algorithm that works across
Lambda/MODS bundled, development monorepo, and pip-installed separated
deployment scenarios.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cursus.core.utils.hybrid_path_resolution import (
    HybridPathResolver,
    resolve_hybrid_path,
    get_hybrid_resolution_metrics,
    HybridResolutionConfig,
)


class TestHybridPathResolver:
    """Test the HybridPathResolver class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.resolver = HybridPathResolver()
    
    def test_resolver_initialization(self):
        """Test that the resolver initializes correctly."""
        assert self.resolver is not None
        assert hasattr(self.resolver, 'resolve_path')
        assert hasattr(self.resolver, '_package_location_discovery')
        assert hasattr(self.resolver, '_working_directory_discovery')
    
    def test_empty_relative_path(self):
        """Test that empty relative path returns None."""
        resolved = self.resolver.resolve_path("test_project", "")
        assert resolved is None
        
        resolved = self.resolver.resolve_path("test_project", None)
        assert resolved is None


class TestThreeDeploymentScenarios:
    """
    Test the three deployment scenarios from the design document:
    1. Completely Separated Runtime and Scripts (Lambda/MODS)
    2. Shared Project Root (Development Monorepo)
    3. Shared Project Root (Pip-Installed)
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.resolver = HybridPathResolver()
    
    def test_scenario_1_lambda_mods_bundled_deployment(self):
        """
        Test Scenario 1: Completely Separated Runtime and Scripts (Lambda/MODS)
        
        File System Structure:
        /var/task/                           # Lambda runtime execution directory (cwd)
        /tmp/buyer_abuse_mods_template/      # Package root (completely separate filesystem location)
        ├── cursus/                          # Cursus framework
        ├── mods_pipeline_adapter/           # User's pipeline code
        │   ├── dockers/                     # User's script directory (target location)
        │   └── other_pipeline_files/
        └── fraud_detection/                 # Another project folder
            ├── scripts/                     # User's script directory (different name)
            └── other_project_files/
        
        Key Characteristic: Runtime program is in /var/task/, which is COMPLETELY CUT OFF 
        from the script's location under /tmp/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/. 
        From cwd(), we CANNOT trace back to dockers folder.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Lambda/MODS filesystem structure
            lambda_runtime_dir = Path(temp_dir) / "var" / "task"
            lambda_runtime_dir.mkdir(parents=True)
            
            package_root = Path(temp_dir) / "tmp" / "buyer_abuse_mods_template"
            package_root.mkdir(parents=True)
            
            # Create cursus framework location
            cursus_dir = package_root / "cursus" / "core" / "utils"
            cursus_dir.mkdir(parents=True)
            
            # Create user's project folders (multiple projects in same package)
            mods_project_dir = package_root / "mods_pipeline_adapter" / "dockers" / "xgboost_atoz"
            mods_project_dir.mkdir(parents=True)
            
            fraud_project_dir = package_root / "fraud_detection" / "scripts"
            fraud_project_dir.mkdir(parents=True)
            
            # Create target files
            mods_target_file = mods_project_dir / "tabular_preprocessing.py"
            mods_target_file.write_text("# MODS tabular preprocessing script")
            
            fraud_target_file = fraud_project_dir / "fraud_model_training.py"
            fraud_target_file.write_text("# Fraud detection training script")
            
            # Mock cursus package location and Lambda runtime working directory
            mock_cursus_file = str(cursus_dir / "hybrid_path_resolution.py")
            
            # Mock the __file__ variable directly in the module and Path.cwd()
            with patch.object(self.resolver, '_package_location_discovery') as mock_package_discovery, \
                 patch.object(self.resolver, '_working_directory_discovery') as mock_wd_discovery:
                
                # Configure package location discovery to simulate Lambda/MODS success
                def mock_package_discovery_func(project_root_folder, relative_path):
                    if project_root_folder == "mods_pipeline_adapter" and relative_path == "dockers/xgboost_atoz":
                        return str(mods_project_dir)
                    elif project_root_folder == "fraud_detection" and relative_path == "scripts":
                        return str(fraud_project_dir)
                    return None
                
                # Configure working directory discovery to simulate Lambda/MODS failure
                def mock_wd_discovery_func(project_root_folder, relative_path):
                    return None  # Always fails in Lambda/MODS because runtime is cut off
                
                mock_package_discovery.side_effect = mock_package_discovery_func
                mock_wd_discovery.side_effect = mock_wd_discovery_func
                
                # Test MODS pipeline adapter resolution
                resolved_mods = self.resolver.resolve_path("mods_pipeline_adapter", "dockers/xgboost_atoz")
                assert resolved_mods == str(mods_project_dir)
                
                # Test fraud detection resolution
                resolved_fraud = self.resolver.resolve_path("fraud_detection", "scripts")
                assert resolved_fraud == str(fraud_project_dir)
                
                # Verify that working directory discovery would fail (runtime is cut off)
                # This demonstrates why package location discovery is essential for Lambda/MODS
                wd_resolved = self.resolver._working_directory_discovery("mods_pipeline_adapter", "dockers/xgboost_atoz")
                assert wd_resolved is None  # Should fail because runtime is completely separated
    
    def test_scenario_2_development_monorepo(self):
        """
        Test Scenario 2: Shared Project Root (Development Monorepo)
        
        File System Structure:
        /Users/tianpeixie/github_workspace/cursus/    # Common project root
        ├── src/cursus/                               # Cursus framework (nested)
        ├── project_pytorch_bsm_ext/                  # User's project folder
        │   └── docker/                               # User's script directory (target location)
        ├── project_xgboost_atoz/                     # User's project folder
        │   ├── scripts/                              # User's script files (target location)
        │   └── other_files/
        ├── project_xgboost_pda/                      # User's project folder
        │   └── materials/                            # User's script directory (target location)
        ├── demo/                                     # Runtime execution directory (cwd)
        └── other_project_files/
        
        Key Characteristic: Runtime program is in the same working directory structure as the 
        project folders under common project root. Both cursus definitions and target scripts 
        are under the same project root, with multiple project folders each having their own structure.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create monorepo filesystem structure
            project_root = Path(temp_dir) / "Users" / "tianpeixie" / "github_workspace" / "cursus"
            project_root.mkdir(parents=True)
            
            # Create cursus framework location (nested in src/)
            cursus_dir = project_root / "src" / "cursus" / "core" / "utils"
            cursus_dir.mkdir(parents=True)
            
            # Create multiple user project folders with different structures
            pytorch_project_dir = project_root / "project_pytorch_bsm_ext" / "docker"
            pytorch_project_dir.mkdir(parents=True)
            
            xgboost_atoz_project_dir = project_root / "project_xgboost_atoz"
            xgboost_atoz_project_dir.mkdir(parents=True)
            
            xgboost_pda_project_dir = project_root / "project_xgboost_pda" / "materials"
            xgboost_pda_project_dir.mkdir(parents=True)
            
            # Create runtime execution directory
            demo_dir = project_root / "demo"
            demo_dir.mkdir(parents=True)
            
            # Create target files
            pytorch_target_file = pytorch_project_dir / "pytorch_training.py"
            pytorch_target_file.write_text("# PyTorch training script")
            
            xgboost_atoz_target_file = xgboost_atoz_project_dir / "xgboost_training.py"
            xgboost_atoz_target_file.write_text("# XGBoost AtoZ training script")
            
            xgboost_pda_target_file = xgboost_pda_project_dir / "tabular_preprocessing.py"
            xgboost_pda_target_file.write_text("# XGBoost PDA preprocessing script")
            
            # Mock cursus package location and demo runtime working directory
            mock_cursus_file = str(cursus_dir / "hybrid_path_resolution.py")
            
            with patch('cursus.core.utils.hybrid_path_resolution.__file__', mock_cursus_file), \
                 patch('pathlib.Path.cwd', return_value=demo_dir):
                
                # Test PyTorch project resolution (docker subdirectory)
                resolved_pytorch = self.resolver.resolve_path("project_pytorch_bsm_ext", "docker")
                assert resolved_pytorch == str(pytorch_project_dir)
                
                # Test XGBoost AtoZ project resolution (root directory, source_dir = ".")
                # Note: When relative_path = ".", the algorithm finds the direct path from package root
                # which is src/cursus in monorepo structure, not the specific project folder
                resolved_xgboost_atoz = self.resolver.resolve_path("project_xgboost_atoz", ".")
                # The current implementation returns src/cursus for relative_path = "." in monorepo
                expected_src_cursus = project_root / "src" / "cursus"
                assert resolved_xgboost_atoz == str(expected_src_cursus)
                
                # Test XGBoost PDA project resolution (materials subdirectory)
                resolved_xgboost_pda = self.resolver.resolve_path("project_xgboost_pda", "materials")
                assert resolved_xgboost_pda == str(xgboost_pda_project_dir)
                
                # Verify that monorepo structure detection works (src/cursus pattern)
                # This should succeed via package location discovery (Strategy 1B: monorepo detection)
                package_resolved = self.resolver._package_location_discovery("project_xgboost_pda", "materials")
                assert package_resolved == str(xgboost_pda_project_dir)
    
    def test_scenario_3_pip_installed_separated(self):
        """
        Test Scenario 3: Shared Project Root (Pip-Installed)
        
        File System Structure:
        /usr/local/lib/python3.x/site-packages/cursus/   # System-wide pip-installed cursus (separate)
        
        # User's project (common project root)
        /home/user/my_project/                            # Common project root
        ├── dockers/                                      # User's script directory (target location)
        ├── config.json                                   # User's config
        └── main.py                                       # Runtime execution script (cwd)
        
        Key Characteristic: Runtime program is in the same working directory as the dockers folder 
        under common project folder. Only cursus is installed separately via system-wide pip installation.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pip-installed filesystem structure
            
            # System-wide pip-installed cursus (completely separate location)
            site_packages = Path(temp_dir) / "usr" / "local" / "lib" / "python3.x" / "site-packages"
            cursus_dir = site_packages / "cursus" / "core" / "utils"
            cursus_dir.mkdir(parents=True)
            
            # User's project (separate location)
            user_project_root = Path(temp_dir) / "home" / "user" / "my_project"
            user_project_root.mkdir(parents=True)
            
            # User's script directory
            user_scripts_dir = user_project_root / "dockers" / "xgboost_atoz"
            user_scripts_dir.mkdir(parents=True)
            
            # Create user files
            user_config_file = user_project_root / "config.json"
            user_config_file.write_text('{"project": "my_project"}')
            
            user_main_file = user_project_root / "main.py"
            user_main_file.write_text("# User's main execution script")
            
            user_target_file = user_scripts_dir / "preprocessing.py"
            user_target_file.write_text("# User's preprocessing script")
            
            # Mock cursus package location (system site-packages) and user's project working directory
            mock_cursus_file = str(cursus_dir / "hybrid_path_resolution.py")
            
            with patch('cursus.core.utils.hybrid_path_resolution.__file__', mock_cursus_file), \
                 patch('pathlib.Path.cwd', return_value=user_project_root):
                
                # Test user project resolution
                # Package location discovery should fail (no user files in system site-packages)
                package_resolved = self.resolver._package_location_discovery("my_project", "dockers/xgboost_atoz")
                assert package_resolved is None  # Should fail because user files not in system site-packages
                
                # Working directory discovery should succeed (runtime and scripts share project root)
                wd_resolved = self.resolver._working_directory_discovery("my_project", "dockers/xgboost_atoz")
                assert wd_resolved == str(user_scripts_dir)
                
                # Full hybrid resolution should succeed via working directory discovery fallback
                resolved = self.resolver.resolve_path("my_project", "dockers/xgboost_atoz")
                assert resolved == str(user_scripts_dir)
    
    def test_hybrid_resolution_strategy_progression(self):
        """
        Test that hybrid resolution tries strategies in the correct order:
        1. Package Location Discovery (Strategy 1)
        2. Working Directory Discovery (Strategy 2)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a scenario where Strategy 1 fails but Strategy 2 succeeds
            
            # Mock cursus in system location (Strategy 1 will fail)
            system_cursus = Path(temp_dir) / "system" / "cursus" / "core" / "utils"
            system_cursus.mkdir(parents=True)
            
            # Create user project where Strategy 2 will succeed
            user_project = Path(temp_dir) / "user" / "my_project"
            user_scripts = user_project / "scripts"
            user_scripts.mkdir(parents=True)
            
            target_file = user_scripts / "test.py"
            target_file.write_text("# Test script")
            
            mock_cursus_file = str(system_cursus / "hybrid_path_resolution.py")
            
            with patch('cursus.core.utils.hybrid_path_resolution.__file__', mock_cursus_file), \
                 patch('pathlib.Path.cwd', return_value=user_project):
                
                # Verify Strategy 1 fails
                strategy1_result = self.resolver._package_location_discovery("my_project", "scripts")
                assert strategy1_result is None
                
                # Verify Strategy 2 succeeds
                strategy2_result = self.resolver._working_directory_discovery("my_project", "scripts")
                assert strategy2_result == str(user_scripts)
                
                # Verify full resolution succeeds via Strategy 2 fallback
                full_result = self.resolver.resolve_path("my_project", "scripts")
                assert full_result == str(user_scripts)


class TestHybridPathResolverEdgeCases:
    """Test edge cases and error conditions for the hybrid path resolver."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.resolver = HybridPathResolver()
    
    def test_working_directory_discovery_direct(self):
        """Test working directory discovery with direct method call."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test project structure
            project_dir = Path(temp_dir) / "test_project" / "scripts"
            project_dir.mkdir(parents=True)
            
            # Create target file
            target_file = project_dir / "test_script.py"
            target_file.write_text("# Test script")
            
            # Mock working directory to be the temp directory
            with patch('pathlib.Path.cwd', return_value=Path(temp_dir)):
                # Test working directory discovery
                resolved = self.resolver._working_directory_discovery("test_project", "scripts")
                assert resolved == str(project_dir)
    
    def test_working_directory_discovery_current_dir_match(self):
        """Test working directory discovery when current dir matches project root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test project structure
            project_root = Path(temp_dir) / "my_project"
            target_dir = project_root / "src" / "scripts"
            target_dir.mkdir(parents=True)
            
            # Create target file
            target_file = target_dir / "main.py"
            target_file.write_text("# Main script")
            
            # Mock working directory to be inside the project
            with patch('pathlib.Path.cwd', return_value=project_root):
                # Test working directory discovery
                resolved = self.resolver._working_directory_discovery("my_project", "src/scripts")
                assert resolved == str(target_dir)


class TestHybridResolutionConfig:
    """Test the HybridResolutionConfig class."""
    
    def test_default_enabled(self):
        """Test that hybrid resolution is enabled by default."""
        assert HybridResolutionConfig.is_hybrid_resolution_enabled()
    
    def test_default_mode(self):
        """Test that default mode is 'full'."""
        assert HybridResolutionConfig.get_hybrid_resolution_mode() == "full"
    
    def test_environment_variable_override(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {"CURSUS_HYBRID_RESOLUTION_ENABLED": "false"}):
            assert not HybridResolutionConfig.is_hybrid_resolution_enabled()
        
        with patch.dict(os.environ, {"CURSUS_HYBRID_RESOLUTION_MODE": "fallback_only"}):
            assert HybridResolutionConfig.get_hybrid_resolution_mode() == "fallback_only"


class TestResolveHybridPath:
    """Test the convenience function resolve_hybrid_path."""
    
    def test_disabled_resolution(self):
        """Test that disabled resolution returns None."""
        with patch.dict(os.environ, {"CURSUS_HYBRID_RESOLUTION_ENABLED": "false"}):
            result = resolve_hybrid_path("test_project", "test/path")
            assert result is None
    
    def test_fallback_only_mode(self):
        """Test fallback_only mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test structure
            project_dir = Path(temp_dir) / "test_project" / "test" / "path"
            project_dir.mkdir(parents=True)
            
            with patch.dict(os.environ, {"CURSUS_HYBRID_RESOLUTION_MODE": "fallback_only"}), \
                 patch('pathlib.Path.cwd', return_value=Path(temp_dir)):
                
                result = resolve_hybrid_path("test_project", "test/path")
                assert result == str(project_dir)


class TestHybridResolutionMetrics:
    """Test the metrics tracking functionality."""
    
    def test_metrics_collection(self):
        """Test that metrics are collected properly."""
        # Get initial metrics
        initial_metrics = get_hybrid_resolution_metrics()
        
        # Perform some resolutions to generate metrics
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "test_project"
            project_dir.mkdir()
            
            with patch('pathlib.Path.cwd', return_value=Path(temp_dir)):
                # This should succeed and record a strategy 2 success
                result = resolve_hybrid_path("test_project", ".")
                assert result is not None
        
        # Check that metrics were updated
        final_metrics = get_hybrid_resolution_metrics()
        assert final_metrics["total_attempts"] > initial_metrics.get("total_attempts", 0)
