"""
Comprehensive tests for utility functions following pytest best practices.

This test module follows the pytest best practices guide:
1. Source Code First Rule - Read utils.py implementation completely before writing tests
2. Mock Path Precision - Mock at exact import locations from source
3. Implementation-Driven Testing - Match test behavior to actual implementation
4. Fixture Isolation - Design fixtures for complete test independence
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile

# Following Source Code First Rule - import the actual implementation
from cursus.api.config_ui.utils import discover_available_configs
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase


class TestDiscoverAvailableConfigs:
    """Comprehensive tests for discover_available_configs function following pytest best practices."""
    
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset any global state before each test (Category 17: Global State Management)."""
        yield
        # No global state in utils module currently
    
    @pytest.fixture
    def mock_universal_config_core(self):
        """
        Mock UniversalConfigCore with realistic behavior.
        
        Following Category 1: Mock Path Precision pattern - mock at exact import location.
        Source shows: from .core import UniversalConfigCore
        """
        with patch('cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            # Configure realistic discovery behavior based on source implementation
            mock_core.discover_config_classes.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase,
                "CradleDataLoadConfig": Mock(spec=['from_base_config', 'model_fields']),
                "XGBoostTrainingConfig": Mock(spec=['from_base_config', 'model_fields'])
            }
            
            yield mock_core
    
    @pytest.fixture
    def temp_workspace(self):
        """Create realistic temporary workspace structure (Category 9: Workspace and Path Resolution)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Create realistic directory structure based on source expectations
            dev_workspace = workspace_root / "dev1"
            dev_workspace.mkdir(parents=True)
            
            for component_type in ["scripts", "contracts", "specs", "configs"]:
                component_dir = dev_workspace / component_type
                component_dir.mkdir()
                sample_file = component_dir / f"sample_{component_type[:-1]}.py"
                sample_file.write_text(f"# Sample {component_type[:-1]} file")
            
            yield workspace_root
    
    def test_discover_available_configs_success(self, mock_universal_config_core):
        """Test successful configuration discovery without workspace directories."""
        # Following Category 2: Mock Configuration pattern
        result = discover_available_configs()
        
        # Test actual implementation behavior
        assert isinstance(result, dict)
        assert len(result) >= 2  # At least base configs
        assert "BasePipelineConfig" in result
        assert "ProcessingStepConfigBase" in result
        assert "CradleDataLoadConfig" in result  # From mock
        assert "XGBoostTrainingConfig" in result  # From mock
        
        # Verify UniversalConfigCore was initialized correctly
        mock_universal_config_core.discover_config_classes.assert_called_once()
    
    def test_discover_available_configs_with_workspace_dirs(self, mock_universal_config_core, temp_workspace):
        """Test configuration discovery with workspace directories."""
        workspace_dirs = [str(temp_workspace)]
        
        result = discover_available_configs(workspace_dirs=workspace_dirs)
        
        # Test actual implementation behavior
        assert isinstance(result, dict)
        assert len(result) >= 2
        
        # Verify UniversalConfigCore was initialized with workspace_dirs
        # Note: We need to check the call_args to verify workspace_dirs were passed
        # The mock_universal_config_core fixture mocks the instance, not the class
        # So we need to patch at the class level to verify initialization
    
    def test_discover_available_configs_with_workspace_dirs_class_init(self, temp_workspace):
        """Test configuration discovery with workspace directories - verify class initialization."""
        # Following Category 1: Mock Path Precision - mock at class level to verify init
        with patch('cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            mock_core.discover_config_classes.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase
            }
            
            workspace_dirs = [str(temp_workspace)]
            result = discover_available_configs(workspace_dirs=workspace_dirs)
            
            # Verify UniversalConfigCore was initialized with correct workspace_dirs
            mock_core_class.assert_called_once_with(workspace_dirs=workspace_dirs)
            assert isinstance(result, dict)
    
    def test_discover_available_configs_none_workspace_dirs(self, mock_universal_config_core):
        """Test configuration discovery with None workspace directories."""
        result = discover_available_configs(workspace_dirs=None)
        
        # Should handle None gracefully
        assert isinstance(result, dict)
        assert len(result) >= 2
    
    def test_discover_available_configs_empty_workspace_dirs(self, mock_universal_config_core):
        """Test configuration discovery with empty workspace directories list."""
        result = discover_available_configs(workspace_dirs=[])
        
        # Should handle empty list gracefully
        assert isinstance(result, dict)
        assert len(result) >= 2
    
    def test_discover_available_configs_core_failure(self):
        """Test configuration discovery when UniversalConfigCore fails."""
        # Following Category 6: Exception Handling pattern
        with patch('src.cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core_class.side_effect = Exception("Core initialization failed")
            
            # Should propagate the exception (based on implementation behavior)
            with pytest.raises(Exception, match="Core initialization failed"):
                discover_available_configs()
    
    def test_discover_available_configs_discovery_failure(self):
        """Test configuration discovery when discover_config_classes fails."""
        with patch('src.cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            mock_core.discover_config_classes.side_effect = Exception("Discovery failed")
            
            # Should propagate the exception (based on implementation behavior)
            with pytest.raises(Exception, match="Discovery failed"):
                discover_available_configs()
    
    def test_discover_available_configs_string_workspace_dirs(self, mock_universal_config_core):
        """Test configuration discovery with string workspace directories."""
        # Test that string paths are handled correctly
        workspace_dirs = ["/path/to/workspace", "/another/path"]
        
        result = discover_available_configs(workspace_dirs=workspace_dirs)
        
        # Should handle string paths gracefully
        assert isinstance(result, dict)
        assert len(result) >= 2
    
    def test_discover_available_configs_mixed_workspace_dirs(self, mock_universal_config_core, temp_workspace):
        """Test configuration discovery with mixed string and Path workspace directories."""
        workspace_dirs = [str(temp_workspace), Path("/another/path")]
        
        result = discover_available_configs(workspace_dirs=workspace_dirs)
        
        # Should handle mixed types gracefully
        assert isinstance(result, dict)
        assert len(result) >= 2
    
    def test_discover_available_configs_return_type_validation(self, mock_universal_config_core):
        """Test that discover_available_configs returns the expected data structure."""
        # Following Category 7: Data Structure Fidelity pattern
        
        # Configure mock to return real classes mixed with mock classes that have __name__
        mock_cradle_config = Mock()
        mock_cradle_config.__name__ = "CradleDataLoadConfig"
        mock_xgboost_config = Mock()
        mock_xgboost_config.__name__ = "XGBoostTrainingConfig"
        
        mock_universal_config_core.discover_config_classes.return_value = {
            "BasePipelineConfig": BasePipelineConfig,
            "ProcessingStepConfigBase": ProcessingStepConfigBase,
            "CradleDataLoadConfig": mock_cradle_config,
            "XGBoostTrainingConfig": mock_xgboost_config
        }
        
        result = discover_available_configs()
        
        # Verify return type and structure
        assert isinstance(result, dict)
        
        # Verify all values are classes (not instances)
        for config_name, config_class in result.items():
            assert isinstance(config_name, str)
            # Should be a class, not an instance
            assert hasattr(config_class, '__name__')  # Classes have __name__
            assert callable(config_class)  # Classes are callable
    
    def test_discover_available_configs_logging_behavior(self, mock_universal_config_core, caplog):
        """Test that appropriate logging messages are generated."""
        import logging
        
        with caplog.at_level(logging.INFO):
            result = discover_available_configs()
            
            # Check for expected logging messages
            assert "Discovering available configuration classes" in caplog.text
            assert f"Discovered {len(result)} configuration classes" in caplog.text
    
    def test_discover_available_configs_caching_behavior(self):
        """Test that configuration discovery doesn't cache results inappropriately."""
        # Following Category 2: Mock Configuration - test caching behavior
        with patch('src.cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core1 = Mock()
            mock_core2 = Mock()
            
            # Configure different return values for each call
            mock_core1.discover_config_classes.return_value = {"Config1": Mock()}
            mock_core2.discover_config_classes.return_value = {"Config2": Mock()}
            
            mock_core_class.side_effect = [mock_core1, mock_core2]
            
            # First call
            result1 = discover_available_configs()
            # Second call
            result2 = discover_available_configs()
            
            # Should create new UniversalConfigCore instances each time
            assert mock_core_class.call_count == 2
            assert result1 != result2  # Different results due to different mocks


class TestUtilsErrorHandlingAndEdgeCases:
    """Test error handling and edge cases following pytest best practices."""
    
    def test_workspace_dirs_type_validation(self):
        """Test handling of invalid workspace_dirs types."""
        # Following Category 12: NoneType Attribute Access and Defensive Coding
        with patch('src.cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            mock_core.discover_config_classes.return_value = {}
            
            # Test with invalid type (should be handled by UniversalConfigCore)
            result = discover_available_configs(workspace_dirs="invalid_string")
            
            # Should still work (UniversalConfigCore handles type conversion)
            assert isinstance(result, dict)
            mock_core_class.assert_called_once_with(workspace_dirs="invalid_string")
    
    def test_empty_discovery_result(self):
        """Test handling when no configurations are discovered."""
        with patch('src.cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            mock_core.discover_config_classes.return_value = {}  # Empty result
            
            result = discover_available_configs()
            
            # Should handle empty results gracefully
            assert isinstance(result, dict)
            assert len(result) == 0
    
    def test_malformed_discovery_result(self):
        """Test handling when discovery returns malformed data."""
        with patch('src.cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            # Return non-dict result
            mock_core.discover_config_classes.return_value = ["not", "a", "dict"]
            
            # Should propagate the issue (let UniversalConfigCore handle validation)
            result = discover_available_configs()
            assert result == ["not", "a", "dict"]  # Pass through whatever core returns
    
    def test_unicode_workspace_paths(self):
        """Test handling of Unicode characters in workspace paths."""
        with patch('src.cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            mock_core.discover_config_classes.return_value = {"TestConfig": Mock()}
            
            # Test with Unicode paths
            unicode_paths = ["/path/with/üñíçødé", "/另一个/路径"]
            
            result = discover_available_configs(workspace_dirs=unicode_paths)
            
            # Should handle Unicode paths gracefully
            assert isinstance(result, dict)
            mock_core_class.assert_called_once_with(workspace_dirs=unicode_paths)
    
    def test_very_long_workspace_paths(self):
        """Test handling of very long workspace paths."""
        with patch('src.cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            mock_core.discover_config_classes.return_value = {"TestConfig": Mock()}
            
            # Create very long path
            long_path = "/very/long/path/" + "a" * 1000 + "/workspace"
            
            result = discover_available_configs(workspace_dirs=[long_path])
            
            # Should handle long paths gracefully
            assert isinstance(result, dict)
            mock_core_class.assert_called_once_with(workspace_dirs=[long_path])
    
    def test_nonexistent_workspace_paths(self):
        """Test handling of non-existent workspace paths."""
        with patch('src.cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            mock_core.discover_config_classes.return_value = {"TestConfig": Mock()}
            
            # Test with non-existent paths
            nonexistent_paths = ["/does/not/exist", "/also/missing"]
            
            result = discover_available_configs(workspace_dirs=nonexistent_paths)
            
            # Should handle non-existent paths gracefully (UniversalConfigCore's responsibility)
            assert isinstance(result, dict)
            mock_core_class.assert_called_once_with(workspace_dirs=nonexistent_paths)


class TestUtilsIntegrationScenarios:
    """Integration tests for utils module following pytest best practices."""
    
    def test_utils_with_real_config_classes(self):
        """Test utils function with real configuration classes."""
        # Following Category 4: Test Expectations vs Implementation
        with patch('src.cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            # Use real config classes in mock return
            mock_core.discover_config_classes.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase
            }
            
            result = discover_available_configs()
            
            # Verify real classes are returned
            assert result["BasePipelineConfig"] == BasePipelineConfig
            assert result["ProcessingStepConfigBase"] == ProcessingStepConfigBase
            
            # Verify they are actual classes
            assert hasattr(result["BasePipelineConfig"], 'model_fields')
            assert hasattr(result["ProcessingStepConfigBase"], 'model_fields')
    
    def test_utils_function_signature_compatibility(self):
        """Test that utils function signature matches expected interface."""
        import inspect
        
        # Get function signature
        sig = inspect.signature(discover_available_configs)
        
        # Verify expected parameters
        params = list(sig.parameters.keys())
        assert 'workspace_dirs' in params
        
        # Verify parameter has correct default
        workspace_dirs_param = sig.parameters['workspace_dirs']
        assert workspace_dirs_param.default is None
    
    def test_utils_function_docstring_and_metadata(self):
        """Test that utils function has proper documentation."""
        # Verify function has docstring
        assert discover_available_configs.__doc__ is not None
        assert len(discover_available_configs.__doc__.strip()) > 0
        
        # Verify function name
        assert discover_available_configs.__name__ == "discover_available_configs"
    
    def test_utils_module_imports(self):
        """Test that utils module imports are working correctly."""
        # Following Category 1: Mock Path Issues - verify imports work
        try:
            from src.cursus.api.config_ui.core.utils import discover_available_configs
            from src.cursus.api.config_ui.core.core import UniversalConfigCore
            
            # Imports should work without errors
            assert callable(discover_available_configs)
            assert callable(UniversalConfigCore)
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_utils_with_multiple_workspace_types(self):
        """Test utils function with various workspace directory types."""
        with patch('src.cursus.api.config_ui.utils.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            mock_core.discover_config_classes.return_value = {"TestConfig": Mock()}
            
            # Test with different workspace directory formats
            test_cases = [
                None,
                [],
                ["/single/path"],
                ["/path1", "/path2", "/path3"],
                [Path("/path/object")],
                ["/string/path", Path("/path/object")]
            ]
            
            for workspace_dirs in test_cases:
                result = discover_available_configs(workspace_dirs=workspace_dirs)
                
                # Should handle all cases gracefully
                assert isinstance(result, dict)
                
                # Verify UniversalConfigCore was called with correct args
                expected_call = mock_core_class.call_args
                assert expected_call[1]['workspace_dirs'] == workspace_dirs
                
                # Reset mock for next iteration
                mock_core_class.reset_mock()
