"""
Tests for deployment-context-agnostic path resolution utilities.

This test suite is based on the deep dive analysis from the MODS pipeline path resolution error
that occurred during Lambda deployment. It mocks the exact scenario where configurations
created in development environments fail when executed in serverless deployment contexts.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Import the path resolution utilities
from cursus.core.utils.path_resolution import get_package_relative_path, resolve_package_relative_path


class TestDeploymentContextAgnosticPathResolution:
    """Test suite for package-aware path resolution across deployment contexts."""

    def test_get_package_relative_path_cursus_package(self):
        """Test package-relative path extraction for cursus package paths."""
        # Test case from MODS error analysis - cursus package path
        development_path = "/Users/lukexie/mods/src/BuyerAbuseModsTemplate/src/buyer_abuse_mods_template/cursus/core/base/config_base.py"
        expected_relative = "core/base/config_base.py"
        
        result = get_package_relative_path(development_path)
        assert result == expected_relative

    def test_get_package_relative_path_buyer_abuse_package(self):
        """Test package-relative path extraction for buyer_abuse_mods_template package paths."""
        # Test case from MODS error analysis - the exact problematic path
        development_path = "/home/ec2-user/SageMaker/BuyerAbuseModsTemplate/src/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
        expected_relative = "mods_pipeline_adapter/dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
        
        result = get_package_relative_path(development_path)
        assert result == expected_relative

    def test_get_package_relative_path_with_src_directory(self):
        """Test package-relative path extraction when src directory is present."""
        # Test case with src directory structure
        development_path = "/workspace/project/src/cursus/steps/builders/builder_tabular_preprocessing_step.py"
        expected_relative = "steps/builders/builder_tabular_preprocessing_step.py"
        
        result = get_package_relative_path(development_path)
        assert result == expected_relative

    def test_get_package_relative_path_already_relative(self):
        """Test that already relative paths are returned unchanged."""
        relative_path = "dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
        result = get_package_relative_path(relative_path)
        assert result == relative_path

    def test_get_package_relative_path_empty_or_none(self):
        """Test handling of empty or None paths."""
        assert get_package_relative_path("") == ""
        assert get_package_relative_path(None) == None

    def test_get_package_relative_path_no_package_indicator(self):
        """Test fallback behavior when no package indicators are found."""
        unknown_path = "/some/random/path/without/package/indicators/file.py"
        result = get_package_relative_path(unknown_path)
        # Should return original path as fallback
        assert result == unknown_path

    def test_resolve_package_relative_path_success(self):
        """Test successful resolution of package-relative paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock cursus package structure
            cursus_dir = Path(temp_dir) / "cursus"
            cursus_dir.mkdir()
            (cursus_dir / "__init__.py").touch()
            
            # Create the target file structure
            target_dir = cursus_dir / "dockers" / "xgboost_atoz" / "scripts"
            target_dir.mkdir(parents=True)
            target_file = target_dir / "tabular_preprocessing.py"
            target_file.touch()
            
            # Mock cursus module to point to our temp directory
            mock_cursus = MagicMock()
            mock_cursus.__file__ = str(cursus_dir / "__init__.py")
            
            with patch.dict('sys.modules', {'cursus': mock_cursus}):
                relative_path = "dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
                result = resolve_package_relative_path(relative_path)
                
                # Should resolve to the absolute path in temp directory
                expected = str(target_file.resolve())
                assert result == expected

    def test_resolve_package_relative_path_file_not_found(self):
        """Test behavior when resolved path doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock cursus package structure without the target file
            cursus_dir = Path(temp_dir) / "cursus"
            cursus_dir.mkdir()
            (cursus_dir / "__init__.py").touch()
            
            # Mock cursus module to point to our temp directory
            mock_cursus = MagicMock()
            mock_cursus.__file__ = str(cursus_dir / "__init__.py")
            
            with patch.dict('sys.modules', {'cursus': mock_cursus}):
                relative_path = "nonexistent/path/file.py"
                result = resolve_package_relative_path(relative_path)
                
                # Should return resolved absolute path even when file doesn't exist
                expected_path = str((cursus_dir / relative_path).resolve())
                assert result == expected_path

    def test_resolve_package_relative_path_already_absolute(self):
        """Test that already absolute paths are returned unchanged."""
        absolute_path = "/tmp/some/absolute/path/file.py"
        result = resolve_package_relative_path(absolute_path)
        assert result == absolute_path

    def test_resolve_package_relative_path_empty_or_none(self):
        """Test handling of empty or None paths."""
        assert resolve_package_relative_path("") == ""
        assert resolve_package_relative_path(None) == None

    def test_resolve_package_relative_path_cursus_import_error(self):
        """Test behavior when cursus module cannot be imported."""
        with patch.dict('sys.modules', {'cursus': None}):
            relative_path = "dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
            result = resolve_package_relative_path(relative_path)
            # Should return original path when cursus module is not available
            assert result == relative_path


class TestMODSLambdaDeploymentScenario:
    """
    Test suite that mocks the exact MODS Lambda deployment scenario from the error analysis.
    
    This recreates the context mismatch between development and Lambda environments
    that caused the "tabular_preprocessing.py is not a valid file" error.
    """

    def test_mods_lambda_context_mismatch_scenario(self):
        """
        Test the exact scenario from MODS error analysis with proper runtime/development separation.
        
        This test recreates the CRITICAL architecture mismatch:
        1. Development context: Configuration created with absolute paths
        2. Lambda runtime context: 
           - Execution from /var/task/ (MODSPythonLambda code)
           - Package installed to /tmp/buyer_abuse_mods_template/
           - cursus module at /tmp/buyer_abuse_mods_template/cursus/
           - Target files at /tmp/buyer_abuse_mods_template/mods_pipeline_adapter/
        3. Path resolution failure: Development paths don't work in Lambda
        4. Fix verification: Package-aware resolution works across separated contexts
        """
        # STEP 1: Mock Development Context (Configuration Creation Time)
        development_working_dir = "/Users/lukexie/mods/src/BuyerAbuseModsTemplate"
        development_absolute_path = "/home/ec2-user/SageMaker/BuyerAbuseModsTemplate/src/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
        
        # STEP 2: Mock Lambda Runtime Context (Execution Time)
        lambda_execution_dir = "/var/task"  # Where MODSPythonLambda code executes
        lambda_package_root = "/tmp/buyer_abuse_mods_template"  # Where package is installed
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create REALISTIC Lambda file system structure
            
            # 1. Lambda execution directory (where MODSPythonLambda runs)
            lambda_var_task = Path(temp_dir) / "var" / "task"
            lambda_var_task.mkdir(parents=True)
            
            # Create MODSPythonLambda files in execution directory
            (lambda_var_task / "resolve_mods_macros.py").write_text("# MODS Lambda execution script")
            (lambda_var_task / "pipeline_helper.py").write_text("# MODS pipeline helper")
            
            # 2. Lambda package installation directory
            lambda_tmp_package = Path(temp_dir) / "tmp" / "buyer_abuse_mods_template"
            lambda_tmp_package.mkdir(parents=True)
            
            # 3. Create cursus package structure (CRITICAL: cursus is a subdirectory)
            cursus_package_dir = lambda_tmp_package / "cursus"
            cursus_package_dir.mkdir()
            (cursus_package_dir / "__init__.py").touch()
            
            # 4. Create target file structure (CRITICAL: sibling to cursus, not child)
            target_dir = lambda_tmp_package / "mods_pipeline_adapter" / "dockers" / "xgboost_atoz" / "scripts"
            target_dir.mkdir(parents=True)
            target_file = target_dir / "tabular_preprocessing.py"
            target_file.write_text("# Mock tabular preprocessing script")
            
            # STEP 3: Test OLD (Problematic) Path Resolution
            # Simulate the old _convert_to_relative_path() behavior with REALISTIC context
            with patch('pathlib.Path.cwd', return_value=Path(lambda_execution_dir)):
                # This would generate a malformed path like the error showed
                # The old system would try to resolve from /var/task/ working directory
                old_problematic_path = "../../home/ec2-user/SageMaker/BuyerAbuseModsTemplate/src/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
                
                # Verify the old path doesn't work when resolved from execution directory
                resolved_old_path = Path(lambda_execution_dir) / old_problematic_path
                assert not resolved_old_path.exists(), "Old path resolution should fail in Lambda context"
                
                # Also verify it doesn't work when resolved from package directory
                resolved_old_from_package = lambda_tmp_package / old_problematic_path
                assert not resolved_old_from_package.exists(), "Old path should fail from package context too"
            
            # STEP 4: Test NEW (Fixed) Package-Aware Path Resolution
            # Use our new package-aware path resolution
            package_relative_path = get_package_relative_path(development_absolute_path)
            expected_package_relative = "mods_pipeline_adapter/dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
            assert package_relative_path == expected_package_relative
            
            # STEP 5: Test REALISTIC Lambda Runtime Resolution
            # Mock cursus module to point to the ACTUAL cursus package location in Lambda
            mock_cursus = MagicMock()
            mock_cursus.__file__ = str(cursus_package_dir / "__init__.py")  # Points to /tmp/buyer_abuse_mods_template/cursus/__init__.py
            
            with patch.dict('sys.modules', {'cursus': mock_cursus}):
                with patch('pathlib.Path.cwd', return_value=Path(lambda_execution_dir)):  # Execution from /var/task/
                    # CRITICAL TEST: Resolve package-relative path from cursus package location
                    # to sibling directory (mods_pipeline_adapter)
                    resolved_new_path = resolve_package_relative_path(package_relative_path)
                    
                    # CRITICAL TEST: Our enhanced path resolution should now handle sibling directories
                    # The resolution should work as follows:
                    # 1. cursus.__file__ = /tmp/buyer_abuse_mods_template/cursus/__init__.py
                    # 2. cursus_package_dir = /tmp/buyer_abuse_mods_template/cursus/
                    # 3. Try child path: /tmp/buyer_abuse_mods_template/cursus/mods_pipeline_adapter/... (doesn't exist)
                    # 4. Try sibling path: /tmp/buyer_abuse_mods_template/mods_pipeline_adapter/... (EXISTS!)
                    # 5. Return sibling path: /tmp/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz/scripts/tabular_preprocessing.py
                    
                    resolved_new_path = resolve_package_relative_path(package_relative_path)
                    
                    # Verify the resolution correctly finds the sibling directory file
                    expected_sibling_path = str(target_file.resolve())
                    actual_resolved_path = resolved_new_path
                    
                    print(f"Target file exists at: {target_file.resolve()}")
                    print(f"Cursus package at: {cursus_package_dir.resolve()}")
                    print(f"Resolved path: {actual_resolved_path}")
                    print(f"Expected path: {expected_sibling_path}")
                    
                    # The enhanced implementation should now correctly resolve to the sibling directory
                    assert actual_resolved_path == expected_sibling_path, f"Path resolution should find sibling directory: expected {expected_sibling_path}, got {actual_resolved_path}"
                    assert Path(actual_resolved_path).exists(), "Resolved path should exist in Lambda context"
                    
                    # Verify the file is accessible and readable
                    resolved_file = Path(actual_resolved_path)
                    assert resolved_file.is_file(), "Resolved path should be a file"
                    assert resolved_file.read_text() == "# Mock tabular preprocessing script", "File should be readable with correct content"

    def test_cross_deployment_context_portability(self):
        """
        Test that the same package-relative path works across different deployment contexts.
        
        This verifies the fix works in:
        1. Development environment (local file system)
        2. Lambda environment (/tmp/ package installation)
        3. Container environment (/usr/local/lib/python3.x/site-packages/)
        4. PyPI package installation (site-packages)
        """
        package_relative_path = "mods_pipeline_adapter/dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
        
        deployment_contexts = [
            # Development context
            {
                "name": "development",
                "package_root": "/Users/developer/project/src/cursus",
                "working_dir": "/Users/developer/project"
            },
            # Lambda context
            {
                "name": "lambda",
                "package_root": "/tmp/cursus",
                "working_dir": "/var/task"
            },
            # Container context
            {
                "name": "container",
                "package_root": "/usr/local/lib/python3.9/site-packages/cursus",
                "working_dir": "/app"
            },
            # PyPI package context
            {
                "name": "pypi",
                "package_root": "/home/user/.local/lib/python3.9/site-packages/cursus",
                "working_dir": "/home/user/project"
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for context in deployment_contexts:
                # Create package structure for this context
                package_root = Path(temp_dir) / context["name"] / Path(context["package_root"]).relative_to(Path(context["package_root"]).anchor)
                package_root.mkdir(parents=True)
                (package_root / "__init__.py").touch()
                
                # Create target file
                target_file = package_root / package_relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(f"# Mock script for {context['name']} context")
                
                # Mock cursus module for this context
                mock_cursus = MagicMock()
                mock_cursus.__file__ = str(package_root / "__init__.py")
                
                with patch.dict('sys.modules', {'cursus': mock_cursus}):
                    with patch('pathlib.Path.cwd', return_value=Path(context["working_dir"])):
                        # Test that package-relative path resolves correctly
                        resolved_path = resolve_package_relative_path(package_relative_path)
                        
                        assert resolved_path == str(target_file.resolve()), f"Path resolution failed in {context['name']} context"
                        assert Path(resolved_path).exists(), f"Resolved file doesn't exist in {context['name']} context"

    def test_mods_error_message_reproduction(self):
        """
        Test that reproduces the exact error message from the MODS analysis.
        
        This test verifies that:
        1. The old system would produce the exact error seen in Lambda
        2. The new system resolves the path correctly
        """
        # The exact problematic path from the error message
        problematic_relative_path = "../../home/ec2-user/SageMaker/BuyerAbuseModsTemplate/src/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
        
        # The original absolute path that generated the problematic relative path
        original_absolute_path = "/home/ec2-user/SageMaker/BuyerAbuseModsTemplate/src/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Lambda-like environment
            lambda_package_root = Path(temp_dir) / "tmp" / "buyer_abuse_mods_template"
            lambda_package_root.mkdir(parents=True)
            
            # Create the target file where it should be in Lambda
            correct_target = lambda_package_root / "mods_pipeline_adapter" / "dockers" / "xgboost_atoz" / "scripts" / "tabular_preprocessing.py"
            correct_target.parent.mkdir(parents=True)
            correct_target.write_text("# Correct target file")
            
            # Test 1: Verify the problematic path doesn't resolve correctly
            problematic_resolved = lambda_package_root / problematic_relative_path
            # This should not exist (reproducing the original error)
            assert not problematic_resolved.exists(), "Problematic path should not exist (reproducing original error)"
            
            # Test 2: Verify our fix generates the correct package-relative path
            correct_package_relative = get_package_relative_path(original_absolute_path)
            expected_correct = "mods_pipeline_adapter/dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
            assert correct_package_relative == expected_correct, "Package-relative path generation should be correct"
            
            # Test 3: Verify the correct path resolves successfully
            correct_resolved = lambda_package_root / correct_package_relative
            assert correct_resolved.exists(), "Correct package-relative path should resolve successfully"
            
            # Test 4: Verify the file contents are accessible
            assert correct_resolved.read_text() == "# Correct target file", "File should be readable with correct content"


class TestBasePipelineConfigIntegration:
    """Test integration with BasePipelineConfig using the new path resolution."""

    def test_base_config_uses_new_path_resolution(self):
        """Test that BasePipelineConfig uses the new package-aware path resolution."""
        from cursus.core.base.config_base import BasePipelineConfig
        
        # Create a minimal test config class
        class TestConfig(BasePipelineConfig):
            def __init__(self, **kwargs):
                # Provide minimal required fields for BasePipelineConfig
                defaults = {
                    'author': 'test_author',
                    'bucket': 'test-bucket',
                    'role': 'test-role',
                    'region': 'NA',
                    'service_name': 'test_service',
                    'pipeline_version': '1.0.0',
                    'source_dir': '/home/ec2-user/SageMaker/BuyerAbuseModsTemplate/src/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz'
                }
                defaults.update(kwargs)
                super().__init__(**defaults)
        
        # Create config instance
        config = TestConfig()
        
        # Test that portable_source_dir uses the new path resolution
        portable_path = config.portable_source_dir
        expected_portable = "mods_pipeline_adapter/dockers/xgboost_atoz"
        
        assert portable_path == expected_portable, f"Expected {expected_portable}, got {portable_path}"

    def test_config_get_resolved_path_method(self):
        """Test the new get_resolved_path helper method in BasePipelineConfig."""
        from cursus.core.base.config_base import BasePipelineConfig
        
        class TestConfig(BasePipelineConfig):
            def __init__(self, **kwargs):
                defaults = {
                    'author': 'test_author',
                    'bucket': 'test-bucket', 
                    'role': 'test-role',
                    'region': 'NA',
                    'service_name': 'test_service',
                    'pipeline_version': '1.0.0'
                }
                defaults.update(kwargs)
                super().__init__(**defaults)
        
        config = TestConfig()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock cursus package structure
            cursus_dir = Path(temp_dir) / "cursus"
            cursus_dir.mkdir()
            (cursus_dir / "__init__.py").touch()
            
            # Create target file
            target_dir = cursus_dir / "dockers" / "xgboost_atoz"
            target_dir.mkdir(parents=True)
            target_file = target_dir / "script.py"
            target_file.touch()
            
            # Mock cursus module
            mock_cursus = MagicMock()
            mock_cursus.__file__ = str(cursus_dir / "__init__.py")
            
            with patch.dict('sys.modules', {'cursus': mock_cursus}):
                # Test get_resolved_path method
                relative_path = "dockers/xgboost_atoz/script.py"
                resolved_path = config.get_resolved_path(relative_path)
                
                assert resolved_path == str(target_file.resolve())
