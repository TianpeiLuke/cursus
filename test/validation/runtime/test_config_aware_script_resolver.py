"""
Tests for ConfigAwareScriptPathResolver.

This test suite validates the unified script path resolution system that replaces
unreliable discovery chains with config instances + hybrid path resolution.

Tests cover:
- Config instance-based entry point extraction
- Hybrid path resolution integration
- Phantom script elimination
- Deployment-agnostic resolution
- Validation and reporting capabilities
"""

import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

from cursus.validation.runtime.config_aware_script_resolver import ConfigAwareScriptPathResolver


class TestConfigAwareScriptPathResolver:
    """Test suite for the unified script path resolver."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.resolver = ConfigAwareScriptPathResolver()
    
    def test_resolver_initialization(self):
        """Test resolver initializes correctly."""
        assert self.resolver is not None
        assert hasattr(self.resolver, 'logger')
    
    def test_extract_entry_point_from_processing_config(self):
        """Test entry point extraction from ProcessingStepConfigBase."""
        # Mock config with processing_entry_point
        config = Mock()
        config.processing_entry_point = "tabular_preprocessing.py"
        
        entry_point = self.resolver._extract_entry_point_from_config(config)
        assert entry_point == "tabular_preprocessing.py"
    
    def test_extract_entry_point_from_training_config(self):
        """Test entry point extraction from TrainingStepConfigBase."""
        # Mock config with training_entry_point
        config = Mock()
        config.training_entry_point = "xgboost_training.py"
        # Remove processing_entry_point to test priority order
        del config.processing_entry_point
        
        entry_point = self.resolver._extract_entry_point_from_config(config)
        assert entry_point == "xgboost_training.py"
    
    def test_extract_entry_point_from_generic_config(self):
        """Test entry point extraction from generic config."""
        # Mock config with generic entry_point
        config = Mock()
        config.entry_point = "generic_script.py"
        # Remove other entry point fields
        del config.processing_entry_point
        del config.training_entry_point
        del config.inference_entry_point
        
        entry_point = self.resolver._extract_entry_point_from_config(config)
        assert entry_point == "generic_script.py"
    
    def test_extract_entry_point_none_found(self):
        """Test entry point extraction when no entry point exists."""
        # Mock config without any entry point fields
        config = Mock()
        del config.processing_entry_point
        del config.training_entry_point
        del config.inference_entry_point
        del config.entry_point
        
        entry_point = self.resolver._extract_entry_point_from_config(config)
        assert entry_point is None
    
    def test_get_effective_source_dir_resolved(self):
        """Test source directory extraction with resolved directory."""
        # Mock config with resolved_processing_source_dir (highest priority)
        config = Mock()
        config.resolved_processing_source_dir = "/resolved/scripts"
        
        source_dir = self.resolver._get_effective_source_dir(config)
        assert source_dir == "/resolved/scripts"
    
    def test_get_effective_source_dir_processing(self):
        """Test source directory extraction with processing_source_dir."""
        # Mock config with processing_source_dir
        config = Mock()
        config.processing_source_dir = "scripts"
        # Remove higher priority field
        del config.resolved_processing_source_dir
        del config.effective_source_dir
        
        source_dir = self.resolver._get_effective_source_dir(config)
        assert source_dir == "scripts"
    
    def test_get_effective_source_dir_generic(self):
        """Test source directory extraction with generic source_dir."""
        # Mock config with generic source_dir
        config = Mock()
        config.source_dir = "src"
        # Remove other source dir fields
        del config.resolved_processing_source_dir
        del config.effective_source_dir
        del config.processing_source_dir
        
        source_dir = self.resolver._get_effective_source_dir(config)
        assert source_dir == "src"
    
    def test_get_effective_source_dir_none_found(self):
        """Test source directory extraction when no source directory exists."""
        # Mock config without any source directory fields
        config = Mock()
        del config.resolved_processing_source_dir
        del config.effective_source_dir
        del config.processing_source_dir
        del config.source_dir
        
        source_dir = self.resolver._get_effective_source_dir(config)
        assert source_dir is None
    
    def test_resolve_script_path_via_config_method(self):
        """Test script resolution using config's built-in method."""
        # Mock config with get_resolved_script_path method
        config = Mock()
        config.processing_entry_point = "test_script.py"
        config.get_resolved_script_path.return_value = "/absolute/path/to/test_script.py"
        
        with patch('pathlib.Path.exists', return_value=True):
            script_path = self.resolver.resolve_script_path(config)
            assert script_path == "/absolute/path/to/test_script.py"
            config.get_resolved_script_path.assert_called_once()
    
    def test_resolve_script_path_via_hybrid_resolution(self):
        """Test script resolution using hybrid path resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create actual script file
            scripts_dir = Path(temp_dir) / "scripts"
            scripts_dir.mkdir()
            script_file = scripts_dir / "test_script.py"
            script_file.write_text("# Test script")
            
            # Mock config without get_resolved_script_path method
            config = Mock()
            config.processing_entry_point = "test_script.py"
            config.processing_source_dir = "scripts"
            # Ensure other source dir fields don't exist to test priority
            del config.resolved_processing_source_dir
            del config.effective_source_dir
            del config.source_dir
            # Remove get_resolved_script_path method
            del config.get_resolved_script_path
            
            # Mock hybrid resolution to return our temp file
            with patch('cursus.core.utils.hybrid_path_resolution.resolve_hybrid_path') as mock_resolve:
                mock_resolve.return_value = str(script_file)
                
                script_path = self.resolver.resolve_script_path(config)
                
                assert script_path == str(script_file)
                mock_resolve.assert_called_once_with(
                    project_root_folder=None,
                    relative_path="scripts/test_script.py"
                )
    
    def test_resolve_script_path_fallback_direct_path(self):
        """Test script resolution with direct path fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create actual script file
            scripts_dir = Path(temp_dir) / "scripts"
            scripts_dir.mkdir()
            script_file = scripts_dir / "test_script.py"
            script_file.write_text("# Test script")
            
            # Mock config without hybrid resolution available
            config = Mock()
            config.processing_entry_point = "test_script.py"
            config.processing_source_dir = str(scripts_dir)  # Use actual directory path
            # Ensure other source dir fields don't exist
            del config.resolved_processing_source_dir
            del config.effective_source_dir
            del config.source_dir
            del config.get_resolved_script_path
            
            # Mock hybrid resolution to fail, forcing fallback to direct path
            with patch('cursus.core.utils.hybrid_path_resolution.resolve_hybrid_path', side_effect=ImportError):
                script_path = self.resolver.resolve_script_path(config)
                assert script_path == str(script_file)
    
    def test_resolve_script_path_no_entry_point(self):
        """Test script resolution with no entry point (phantom script elimination)."""
        # Mock config without entry point - should return None (no phantom script)
        config = Mock()
        del config.processing_entry_point
        del config.training_entry_point
        del config.inference_entry_point
        del config.entry_point
        
        script_path = self.resolver.resolve_script_path(config)
        assert script_path is None  # No phantom script created
    
    def test_resolve_script_path_no_source_dir(self):
        """Test script resolution with no source directory."""
        # Mock config with entry point but no source directory
        config = Mock()
        config.processing_entry_point = "test_script.py"
        del config.get_resolved_script_path
        del config.resolved_processing_source_dir
        del config.effective_source_dir
        del config.processing_source_dir
        del config.source_dir
        
        script_path = self.resolver.resolve_script_path(config)
        assert script_path is None
    
    def test_resolve_script_path_file_not_exists(self):
        """Test script resolution when file doesn't exist."""
        # Mock config with valid entry point and source dir but file doesn't exist
        config = Mock()
        config.processing_entry_point = "nonexistent_script.py"
        config.processing_source_dir = "scripts"
        # Ensure other source dir fields don't exist
        del config.resolved_processing_source_dir
        del config.effective_source_dir
        del config.source_dir
        del config.get_resolved_script_path
        
        # FIXED: Mock at the correct import location
        with patch('cursus.core.utils.hybrid_path_resolution.resolve_hybrid_path') as mock_resolve:
            mock_resolve.return_value = "/resolved/path/to/scripts/nonexistent_script.py"
            with patch('pathlib.Path.exists', return_value=False):
                script_path = self.resolver.resolve_script_path(config)
                assert script_path is None
    
    def test_validate_config_for_script_resolution_success(self):
        """Test config validation for successful script resolution."""
        # Mock config that can resolve scripts
        config = Mock()
        config.processing_entry_point = "test_script.py"
        config.processing_source_dir = "scripts"
        # Ensure other source dir fields don't exist to test priority
        del config.resolved_processing_source_dir
        del config.effective_source_dir
        del config.source_dir
        config.get_resolved_script_path = Mock()
        
        validation = self.resolver.validate_config_for_script_resolution(config)
        
        assert validation['config_type'] == 'Mock'
        assert validation['has_entry_point'] == True
        assert validation['entry_point'] == "test_script.py"
        assert validation['has_source_dir'] == True
        assert validation['source_dir'] == "scripts"
        assert validation['can_resolve_script'] == True
        assert validation['resolution_method'] == 'config_method'
    
    def test_validate_config_for_script_resolution_no_script(self):
        """Test config validation for config without script (phantom elimination)."""
        # Mock config without entry point (data-only transformation)
        config = Mock()
        del config.processing_entry_point
        del config.training_entry_point
        del config.inference_entry_point
        del config.entry_point
        
        validation = self.resolver.validate_config_for_script_resolution(config)
        
        assert validation['has_entry_point'] == False
        assert validation['entry_point'] is None
        assert validation['can_resolve_script'] == False
    
    def test_validate_config_for_script_resolution_hybrid_method(self):
        """Test config validation for hybrid resolution method."""
        # Mock config without get_resolved_script_path method
        config = Mock()
        config.processing_entry_point = "test_script.py"
        config.processing_source_dir = "scripts"
        del config.get_resolved_script_path
        
        validation = self.resolver.validate_config_for_script_resolution(config)
        
        assert validation['resolution_method'] == 'hybrid_resolution'
    
    def test_get_resolution_report_comprehensive(self):
        """Test comprehensive resolution report generation."""
        # Mock multiple config instances with different characteristics
        config_with_script = Mock()
        config_with_script.processing_entry_point = "script1.py"
        config_with_script.processing_source_dir = "scripts"
        del config_with_script.get_resolved_script_path
        
        config_without_script = Mock()  # Phantom script - no entry point
        del config_without_script.processing_entry_point
        del config_without_script.training_entry_point
        del config_without_script.inference_entry_point
        del config_without_script.entry_point
        
        config_instances = {
            'script_step': config_with_script,
            'data_step': config_without_script
        }
        
        with patch.object(self.resolver, 'resolve_script_path') as mock_resolve:
            # First call returns script path, second call not called (no entry point)
            mock_resolve.return_value = "/path/to/script1.py"
            
            report = self.resolver.get_resolution_report(config_instances)
            
            assert report['total_configs'] == 2
            assert report['scripts_found'] == 1
            assert report['phantom_scripts_eliminated'] == 1
            assert report['resolution_success_rate'] == 0.5
            assert report['phantom_elimination_rate'] == 0.5
            
            # Check resolution details
            assert 'script_step' in report['resolution_details']
            assert 'data_step' in report['resolution_details']
            assert report['resolution_details']['script_step']['resolution_status'] == 'success'
            assert report['resolution_details']['data_step']['resolution_status'] == 'no_script'
            
            # Check summary flags
            summary = report['summary']
            assert summary['reliable_discovery'] == True
            assert summary['deployment_agnostic'] == True
            assert summary['no_fuzzy_matching'] == True
            assert summary['no_placeholder_scripts'] == True
            assert summary['config_based_validation'] == True
    
    def test_get_resolution_report_empty_configs(self):
        """Test resolution report with empty config instances."""
        config_instances = {}
        
        report = self.resolver.get_resolution_report(config_instances)
        
        assert report['total_configs'] == 0
        assert report['scripts_found'] == 0
        assert report['phantom_scripts_eliminated'] == 0
        assert report['resolution_success_rate'] == 0.0
        assert report['phantom_elimination_rate'] == 0.0
        assert report['resolution_details'] == {}
    
    def test_get_resolution_report_failed_resolution(self):
        """Test resolution report with failed script resolution."""
        # Mock config that should have script but resolution fails
        config = Mock()
        config.processing_entry_point = "missing_script.py"
        config.processing_source_dir = "scripts"
        del config.get_resolved_script_path
        
        config_instances = {'failing_step': config}
        
        with patch.object(self.resolver, 'resolve_script_path', return_value=None):
            report = self.resolver.get_resolution_report(config_instances)
            
            assert report['total_configs'] == 1
            assert report['scripts_found'] == 0
            assert report['phantom_scripts_eliminated'] == 0
            assert report['resolution_details']['failing_step']['resolution_status'] == 'failed'


class TestConfigAwareScriptPathResolverIntegration:
    """Integration tests for the unified script path resolver."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.resolver = ConfigAwareScriptPathResolver()
    
    def test_integration_with_real_config_structure(self):
        """Test resolver with realistic config structure."""
        # Create a more realistic mock config that mimics actual config classes
        class MockProcessingConfig:
            def __init__(self):
                self.processing_entry_point = "tabular_preprocessing.py"
                self.processing_source_dir = "scripts"
                self.resolved_processing_source_dir = None
            
            def get_resolved_script_path(self):
                if self.resolved_processing_source_dir:
                    return f"{self.resolved_processing_source_dir}/{self.processing_entry_point}"
                return None
        
        config = MockProcessingConfig()
        
        # Test entry point extraction
        entry_point = self.resolver._extract_entry_point_from_config(config)
        assert entry_point == "tabular_preprocessing.py"
        
        # Test source directory extraction
        source_dir = self.resolver._get_effective_source_dir(config)
        assert source_dir == "scripts"
        
        # Test validation
        validation = self.resolver.validate_config_for_script_resolution(config)
        assert validation['can_resolve_script'] == True
        assert validation['resolution_method'] == 'config_method'
    
    def test_integration_phantom_script_elimination(self):
        """Test phantom script elimination with realistic data config."""
        # Create mock data-only config (like CradleDataLoadingConfig)
        class MockDataConfig:
            def __init__(self):
                self.job_type = "data_loading"
                self.data_sources = ["s3://bucket/data"]
                # No entry point fields - this is data-only transformation
        
        config = MockDataConfig()
        
        # Should not find entry point
        entry_point = self.resolver._extract_entry_point_from_config(config)
        assert entry_point is None
        
        # Should not resolve script path
        script_path = self.resolver.resolve_script_path(config)
        assert script_path is None  # No phantom script created
        
        # Validation should indicate no script capability
        validation = self.resolver.validate_config_for_script_resolution(config)
        assert validation['can_resolve_script'] == False
        assert validation['has_entry_point'] == False
    
    def test_integration_multiple_config_types(self):
        """Test resolver with multiple different config types."""
        # Mock different config types
        class MockProcessingConfig:
            def __init__(self):
                self.processing_entry_point = "preprocessing.py"
                self.processing_source_dir = "scripts"
        
        class MockTrainingConfig:
            def __init__(self):
                self.training_entry_point = "training.py"
                self.source_dir = "ml_scripts"
        
        class MockDataConfig:
            def __init__(self):
                self.job_type = "data_loading"
                # No entry point
        
        configs = {
            'preprocessing': MockProcessingConfig(),
            'training': MockTrainingConfig(),
            'data_loading': MockDataConfig()
        }
        
        with patch.object(self.resolver, 'resolve_script_path') as mock_resolve:
            # Mock successful resolution for configs with entry points
            def mock_resolve_side_effect(config):
                if hasattr(config, 'processing_entry_point') or hasattr(config, 'training_entry_point'):
                    return f"/path/to/{type(config).__name__}_script.py"
                return None
            
            mock_resolve.side_effect = mock_resolve_side_effect
            
            report = self.resolver.get_resolution_report(configs)
            
            assert report['total_configs'] == 3
            assert report['scripts_found'] == 2  # preprocessing + training
            assert report['phantom_scripts_eliminated'] == 1  # data_loading
            assert report['resolution_success_rate'] == 2/3
            assert report['phantom_elimination_rate'] == 1/3


if __name__ == "__main__":
    pytest.main([__file__])
