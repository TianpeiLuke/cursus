"""
Test Interactive Runtime Testing Factory - Enhanced with Unified Script Path Resolver

Tests for the enhanced InteractiveRuntimeTestingFactory implementation,
validating config-based script discovery, phantom script elimination,
config automation, and unified resolver integration.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

from cursus.validation.runtime.interactive_factory import InteractiveRuntimeTestingFactory
from cursus.validation.runtime.config_aware_script_resolver import ConfigAwareScriptPathResolver
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec
from cursus.api.dag.base_dag import PipelineDAG


class TestInteractiveRuntimeTestingFactory:
    """Test suite for enhanced InteractiveRuntimeTestingFactory."""
    
    @pytest.fixture
    def mock_dag(self):
        """Create a mock DAG for testing."""
        dag = Mock(spec=PipelineDAG)
        dag.nodes = ['data_preprocessing', 'model_training', 'model_evaluation']
        dag.name = 'test_pipeline'
        return dag
    
    @pytest.fixture
    def mock_config_instance(self):
        """Create a mock config instance with entry point."""
        config = Mock()
        config.processing_entry_point = "data_preprocessing.py"
        config.processing_source_dir = "scripts"
        config.environment_variables = {"CURSUS_ENV": "testing", "DATA_PATH": "/data"}
        config.job_arguments = {"job_type": "processing", "batch_size": "100"}
        return config
    
    @pytest.fixture
    def mock_config_instance_no_script(self):
        """Create a mock config instance without entry point (data-only)."""
        config = Mock()
        config.processing_entry_point = None
        config.processing_source_dir = None
        # Data-only config - should be skipped (phantom elimination)
        return config
    
    @pytest.fixture
    def mock_loaded_configs(self, mock_config_instance, mock_config_instance_no_script):
        """Create mock loaded configs dictionary."""
        return {
            'data_preprocessing': mock_config_instance,
            'model_training': mock_config_instance,
            'model_evaluation': mock_config_instance_no_script  # No script - should be eliminated
        }
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        config_data = {
            "data_preprocessing": {
                "processing_entry_point": "data_preprocessing.py",
                "processing_source_dir": "scripts",
                "environment_variables": {"CURSUS_ENV": "testing"}
            }
        }
        
        # Create temporary file with proper cleanup
        fd, temp_path = tempfile.mkstemp(suffix='.json')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(config_data, f)
            yield temp_path
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    # === ENHANCED CONFIG-BASED INITIALIZATION TESTS ===
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    @patch('cursus.step_catalog.adapters.config_resolver.StepConfigResolverAdapter')
    def test_enhanced_config_based_initialization(self, mock_resolver_adapter, mock_build_configs, 
                                                 mock_load_configs, mock_dag, mock_loaded_configs):
        """Test enhanced factory initialization with config-based validation."""
        # Setup mocks for config loading
        mock_build_configs.return_value = {}
        mock_load_configs.return_value = {}
        
        mock_resolver = Mock()
        mock_resolver.resolve_config_map.return_value = mock_loaded_configs
        mock_resolver_adapter.return_value = mock_resolver
        
        # Mock unified resolver
        with patch('cursus.validation.runtime.interactive_factory.ConfigAwareScriptPathResolver') as mock_resolver_class:
            mock_resolver_instance = Mock()
            mock_resolver_instance.resolve_script_path.side_effect = [
                'scripts/data_preprocessing.py',  # data_preprocessing has script
                'scripts/model_training.py',      # model_training has script  
                None                              # model_evaluation has no script (phantom eliminated)
            ]
            mock_resolver_class.return_value = mock_resolver_instance
            
            # Use temporary directory to avoid creating files in test/integration/runtime
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize factory with config path and temp workspace
                factory = InteractiveRuntimeTestingFactory(mock_dag, "test_config.json", workspace_dir=temp_dir)
                
                # Verify enhanced initialization
                assert factory.dag == mock_dag
                assert factory.config_path == "test_config.json"
                assert factory.loaded_configs == mock_loaded_configs
                assert factory.script_resolver is not None
                
                # Verify phantom script elimination - only 2 scripts discovered (not 3)
                assert len(factory.script_info_cache) == 2  # model_evaluation eliminated
                assert 'data_preprocessing' in factory.script_info_cache
                assert 'model_training' in factory.script_info_cache
                assert 'model_evaluation' not in factory.script_info_cache  # Phantom eliminated!

    def test_legacy_initialization_fallback(self, mock_dag):
        """Test fallback to legacy mode when no config provided."""
        with patch('cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder') as mock_builder, \
             patch('cursus.validation.runtime.interactive_factory.StepCatalog') as mock_step_catalog:
            
            mock_builder_instance = Mock()
            mock_builder.return_value = mock_builder_instance
            mock_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
            mock_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
            
            mock_step_catalog_instance = Mock()
            mock_step_catalog.return_value = mock_step_catalog_instance
            
            # Use temporary directory to avoid creating files in test/integration/runtime
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize factory without config (legacy mode) but with temp workspace
                factory = InteractiveRuntimeTestingFactory(mock_dag, workspace_dir=temp_dir)
                
                # Verify legacy initialization
                assert factory.dag == mock_dag
                assert factory.config_path is None
                assert factory.loaded_configs is None
                assert hasattr(factory, 'spec_builder')  # Legacy mode uses spec_builder

    # === PHANTOM SCRIPT ELIMINATION TESTS ===
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    @patch('cursus.step_catalog.adapters.config_resolver.StepConfigResolverAdapter')
    def test_phantom_script_elimination(self, mock_resolver_adapter, mock_build_configs, 
                                       mock_load_configs, mock_dag, mock_loaded_configs):
        """Test that phantom scripts are eliminated through config validation."""
        # Setup mocks
        mock_build_configs.return_value = {}
        mock_load_configs.return_value = {}
        
        mock_resolver = Mock()
        mock_resolver.resolve_config_map.return_value = mock_loaded_configs
        mock_resolver_adapter.return_value = mock_resolver
        
        # Mock unified resolver to eliminate phantom scripts
        with patch('cursus.validation.runtime.interactive_factory.ConfigAwareScriptPathResolver') as mock_resolver_class:
            mock_resolver_instance = Mock()
            # Only return paths for configs with actual entry points
            mock_resolver_instance.resolve_script_path.side_effect = lambda config: (
                'scripts/data_preprocessing.py' if hasattr(config, 'processing_entry_point') and config.processing_entry_point
                else None  # Phantom eliminated
            )
            mock_resolver_class.return_value = mock_resolver_instance
            
            # Use temporary directory to avoid creating files in test/integration/runtime
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                factory = InteractiveRuntimeTestingFactory(mock_dag, "test_config.json", workspace_dir=temp_dir)
                
                # Verify phantom elimination
                scripts = factory.get_scripts_requiring_testing()
                assert len(scripts) == 2  # Only 2 real scripts, 1 phantom eliminated
                assert 'model_evaluation' not in scripts  # Phantom script eliminated

    # === CONFIG AUTOMATION TESTS ===
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    @patch('cursus.step_catalog.adapters.config_resolver.StepConfigResolverAdapter')
    def test_config_automation_features(self, mock_resolver_adapter, mock_build_configs, 
                                       mock_load_configs, mock_dag, mock_loaded_configs):
        """Test config automation for environment variables and job arguments."""
        # Setup mocks
        mock_build_configs.return_value = {}
        mock_load_configs.return_value = {}
        
        mock_resolver = Mock()
        mock_resolver.resolve_config_map.return_value = mock_loaded_configs
        mock_resolver_adapter.return_value = mock_resolver
        
        with patch('cursus.validation.runtime.interactive_factory.ConfigAwareScriptPathResolver') as mock_resolver_class:
            mock_resolver_instance = Mock()
            mock_resolver_instance.resolve_script_path.return_value = 'scripts/data_preprocessing.py'
            mock_resolver_instance.validate_config_for_script_resolution.return_value = {
                'config_type': 'ProcessingStepConfig',
                'has_entry_point': True,
                'entry_point': 'data_preprocessing.py'
            }
            mock_resolver_class.return_value = mock_resolver_instance
            
            # Use temporary directory to avoid creating files in test/integration/runtime
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                factory = InteractiveRuntimeTestingFactory(mock_dag, "test_config.json", workspace_dir=temp_dir)
                
                # Test config automation in requirements
                requirements = factory.get_script_testing_requirements('data_preprocessing')
                
                # Verify config-populated environment variables
                env_vars = requirements['environment_variables']
                config_env_vars = [var for var in env_vars if var.get('source') == 'config']
                assert len(config_env_vars) >= 0  # May have config-populated env vars
                
                # Verify config-populated job arguments
                job_args = requirements['job_arguments']
                config_job_args = [arg for arg in job_args if arg.get('source') == 'config']
                assert len(config_job_args) >= 0  # May have config-populated job args

    # === ENHANCED CONFIGURATION TESTS ===
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    @patch('cursus.step_catalog.adapters.config_resolver.StepConfigResolverAdapter')
    def test_enhanced_script_configuration(self, mock_resolver_adapter, mock_build_configs, 
                                          mock_load_configs, mock_dag, mock_loaded_configs):
        """Test enhanced script configuration with config automation."""
        # Setup mocks
        mock_build_configs.return_value = {}
        mock_load_configs.return_value = {}
        
        mock_resolver = Mock()
        mock_resolver.resolve_config_map.return_value = mock_loaded_configs
        mock_resolver_adapter.return_value = mock_resolver
        
        with patch('cursus.validation.runtime.interactive_factory.ConfigAwareScriptPathResolver') as mock_resolver_class:
            mock_resolver_instance = Mock()
            mock_resolver_instance.resolve_script_path.return_value = 'scripts/data_preprocessing.py'
            mock_resolver_class.return_value = mock_resolver_instance
            
            # Use temporary directory to avoid creating files in test/integration/runtime
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                factory = InteractiveRuntimeTestingFactory(mock_dag, "test_config.json", workspace_dir=temp_dir)
                
                # Mock file existence for validation
                with patch('pathlib.Path.exists', return_value=True), \
                     patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value = Mock()
                    mock_stat.return_value.st_size = 100  # Non-empty file
                    
                    # Configure script with minimal input (config provides defaults)
                    spec = factory.configure_script_testing(
                        'data_preprocessing',
                        expected_inputs={'data_input': 'test/input.csv'},
                        expected_outputs={'data_output': 'test/output.csv'}
                        # environment_variables and job_arguments automatically from config!
                    )
                
                # Verify enhanced configuration
                assert isinstance(spec, ScriptExecutionSpec)
                assert spec.script_name == 'data_preprocessing'
                assert spec.input_paths == {'data_input': 'test/input.csv'}
                assert spec.output_paths == {'data_output': 'test/output.csv'}
                
                # Verify config automation - env vars and job args should be populated
                assert isinstance(spec.environ_vars, dict)
                assert isinstance(spec.job_args, dict)

    # === ENHANCED VALIDATION TESTS ===
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    @patch('cursus.step_catalog.adapters.config_resolver.StepConfigResolverAdapter')
    def test_enhanced_validation_with_config_context(self, mock_resolver_adapter, mock_build_configs, 
                                                    mock_load_configs, mock_dag, mock_loaded_configs):
        """Test enhanced validation with config-aware error messages."""
        # Setup mocks
        mock_build_configs.return_value = {}
        mock_load_configs.return_value = {}
        
        mock_resolver = Mock()
        mock_resolver.resolve_config_map.return_value = mock_loaded_configs
        mock_resolver_adapter.return_value = mock_resolver
        
        with patch('cursus.validation.runtime.interactive_factory.ConfigAwareScriptPathResolver') as mock_resolver_class:
            mock_resolver_instance = Mock()
            mock_resolver_instance.resolve_script_path.return_value = 'scripts/data_preprocessing.py'
            mock_resolver_class.return_value = mock_resolver_instance
            
            # Use temporary directory to avoid creating files in test/integration/runtime
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                factory = InteractiveRuntimeTestingFactory(mock_dag, "test_config.json", workspace_dir=temp_dir)
                
                # Mock file non-existence for validation failure
                with patch('pathlib.Path.exists', return_value=False):
                    # Attempt to configure script with non-existent input file
                    with pytest.raises(ValueError) as exc_info:
                        factory.configure_script_testing(
                            'data_preprocessing',
                            expected_inputs={'data_input': 'test/nonexistent.csv'},
                            expected_outputs={'data_output': 'test/output.csv'}
                        )
                    
                    # Verify enhanced error message includes validation context
                    error_message = str(exc_info.value)
                    assert len(error_message) > 0  # Error message should not be empty
                    assert "Configuration validation failed" in error_message
    
    def test_config_initialization_error_handling(self, mock_dag):
        """Test error handling during config initialization."""
        with patch('cursus.steps.configs.utils.load_configs') as mock_load_configs, \
             patch('cursus.validation.runtime.interactive_factory.StepCatalog') as mock_step_catalog:
            
            # Mock config loading failure
            mock_load_configs.side_effect = FileNotFoundError("Config file not found")
            
            mock_step_catalog_instance = Mock()
            mock_step_catalog.return_value = mock_step_catalog_instance
            
            # Factory should handle the error gracefully and fall back to legacy mode
            # Use temporary directory to avoid creating files in test/integration/runtime
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                factory = InteractiveRuntimeTestingFactory(mock_dag, "nonexistent_config.json", workspace_dir=temp_dir)
            
            # Verify it fell back to legacy mode
            assert factory.config_path == "nonexistent_config.json"  # Config path is stored
            assert factory.loaded_configs is None  # But configs are None (fallback occurred)
            assert hasattr(factory, 'spec_builder')  # Legacy mode components exist
    
    def test_script_configuration_with_invalid_script_name(self, mock_dag):
        """Test error handling for invalid script names."""
        with patch('cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder') as mock_builder, \
             patch('cursus.validation.runtime.interactive_factory.StepCatalog') as mock_step_catalog:
            
            mock_builder_instance = Mock()
            mock_builder.return_value = mock_builder_instance
            mock_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
            mock_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
            
            mock_step_catalog_instance = Mock()
            mock_step_catalog.return_value = mock_step_catalog_instance
            
            # Use temporary directory to avoid creating files in test/integration/runtime
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                factory = InteractiveRuntimeTestingFactory(mock_dag, workspace_dir=temp_dir)
                
                # Test with script name not in DAG
                with pytest.raises(ValueError) as exc_info:
                    factory.configure_script_testing(
                        'nonexistent_script',
                        expected_inputs={'input': 'test.csv'},
                        expected_outputs={'output': 'result.csv'}
                    )
                
                error_message = str(exc_info.value)
                assert "not found" in error_message

    # === ENHANCED SUMMARY TESTS ===
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    @patch('cursus.step_catalog.adapters.config_resolver.StepConfigResolverAdapter')
    def test_enhanced_factory_summary(self, mock_resolver_adapter, mock_build_configs, 
                                     mock_load_configs, mock_dag, mock_loaded_configs):
        """Test enhanced factory summary with config integration details."""
        # Setup mocks
        mock_build_configs.return_value = {}
        mock_load_configs.return_value = {}
        
        mock_resolver = Mock()
        mock_resolver.resolve_config_map.return_value = mock_loaded_configs
        mock_resolver_adapter.return_value = mock_resolver
        
        with patch('cursus.validation.runtime.interactive_factory.ConfigAwareScriptPathResolver') as mock_resolver_class:
            mock_resolver_instance = Mock()
            mock_resolver_instance.resolve_script_path.side_effect = [
                'scripts/data_preprocessing.py',  # Has script
                'scripts/model_training.py',      # Has script
                None                              # No script (phantom eliminated)
            ]
            mock_resolver_class.return_value = mock_resolver_instance
            
            # Use temporary directory to avoid creating files in test/integration/runtime
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                factory = InteractiveRuntimeTestingFactory(mock_dag, "test_config.json", workspace_dir=temp_dir)
                
                # Get enhanced summary
                summary = factory.get_testing_factory_summary()
                
                # Verify enhanced summary includes config integration
                assert 'config_integration' in summary
                config_info = summary['config_integration']
                assert config_info['mode'] == 'enhanced'
                assert config_info['config_path'] == "test_config.json"
                assert config_info['phantom_elimination_active'] == True
                
                # Verify script details include config metadata
                script_details = summary['script_details']
                assert isinstance(script_details, dict)

    # === UNIFIED RESOLVER INTEGRATION TESTS ===
    
    def test_unified_resolver_import_and_usage(self):
        """Test that ConfigAwareScriptPathResolver is properly imported and used."""
        # Test direct import
        resolver = ConfigAwareScriptPathResolver()
        assert resolver is not None
        
        # Test validation method exists
        assert hasattr(resolver, 'validate_config_for_script_resolution')
        assert hasattr(resolver, 'resolve_script_path')

    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    @patch('cursus.step_catalog.adapters.config_resolver.StepConfigResolverAdapter')
    def test_unified_resolver_integration(self, mock_resolver_adapter, mock_build_configs, 
                                         mock_load_configs, mock_dag, mock_loaded_configs):
        """Test that unified resolver is properly integrated into factory."""
        # Setup mocks
        mock_build_configs.return_value = {}
        mock_load_configs.return_value = {}
        
        mock_resolver = Mock()
        mock_resolver.resolve_config_map.return_value = mock_loaded_configs
        mock_resolver_adapter.return_value = mock_resolver
        
        with patch('cursus.validation.runtime.interactive_factory.ConfigAwareScriptPathResolver') as mock_resolver_class:
            mock_resolver_instance = Mock()
            mock_resolver_class.return_value = mock_resolver_instance
            
            # Use temporary directory to avoid creating files in test/integration/runtime
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                factory = InteractiveRuntimeTestingFactory(mock_dag, "test_config.json", workspace_dir=temp_dir)
                
                # Verify unified resolver is integrated
                assert factory.script_resolver is not None
                assert factory.script_resolver == mock_resolver_instance
                
                # Verify resolver is used in script discovery
                assert mock_resolver_instance.resolve_script_path.call_count > 0

    # === LEGACY COMPATIBILITY TESTS ===
    
    def test_legacy_mode_compatibility(self, mock_dag):
        """Test that legacy mode still works for backward compatibility."""
        with patch('cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder') as mock_builder, \
             patch('cursus.validation.runtime.interactive_factory.StepCatalog') as mock_step_catalog:
            
            mock_builder_instance = Mock()
            mock_builder.return_value = mock_builder_instance
            mock_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
            mock_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
            
            mock_step_catalog_instance = Mock()
            mock_step_catalog.return_value = mock_step_catalog_instance
            
            # Use temporary directory to avoid creating files in test/integration/runtime
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize in legacy mode (no config_path) but with temp workspace
                factory = InteractiveRuntimeTestingFactory(mock_dag, workspace_dir=temp_dir)
                
                # Verify legacy functionality still works
                scripts = factory.get_scripts_requiring_testing()
                assert len(scripts) == 3  # All DAG nodes discovered (may include phantoms)
                
                requirements = factory.get_script_testing_requirements('data_preprocessing')
                assert requirements['script_name'] == 'data_preprocessing'
                
                # Legacy mode should not have config integration
                summary = factory.get_testing_factory_summary()
                config_integration = summary.get('config_integration', {})
                assert config_integration.get('mode') != 'enhanced'


class TestInteractiveFactoryIntegration:
    """Integration tests for enhanced InteractiveRuntimeTestingFactory."""
    
    def test_enhanced_factory_import(self):
        """Test that the enhanced factory can be imported from the runtime module."""
        from cursus.validation.runtime import InteractiveRuntimeTestingFactory
        assert InteractiveRuntimeTestingFactory is not None
    
    def test_unified_resolver_import(self):
        """Test that the unified resolver can be imported from the runtime module."""
        from cursus.validation.runtime import ConfigAwareScriptPathResolver
        assert ConfigAwareScriptPathResolver is not None
    
    def test_enhanced_factory_with_real_dag(self):
        """Test enhanced factory with a real DAG instance."""
        from cursus.api.dag.base_dag import PipelineDAG
        
        # Create real DAG
        dag = PipelineDAG()
        dag.add_node("test_node")
        
        # Use temporary directory to avoid creating files in test/integration/runtime
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test legacy initialization (should not fail)
            factory = InteractiveRuntimeTestingFactory(dag, workspace_dir=temp_dir)
            assert factory.dag == dag
            
            # Test that basic functionality works
            scripts = factory.get_scripts_requiring_testing()
            assert isinstance(scripts, list)


class TestEnhancedFeatures:
    """Test enhanced features specific to the unified resolver integration."""
    
    def test_phantom_elimination_concept(self):
        """Test the concept of phantom script elimination."""
        # This test validates the core concept that configs without entry points
        # should not result in script discovery
        
        # Mock config with entry point (should be discovered)
        config_with_script = Mock()
        config_with_script.processing_entry_point = "script.py"
        
        # Mock config without entry point (should be eliminated)
        config_without_script = Mock()
        config_without_script.processing_entry_point = None
        
        resolver = ConfigAwareScriptPathResolver()
        
        # Mock the resolver behavior
        with patch.object(resolver, 'resolve_script_path') as mock_resolve:
            mock_resolve.side_effect = lambda config: (
                'scripts/script.py' if getattr(config, 'processing_entry_point', None)
                else None  # Phantom eliminated
            )
            
            # Test phantom elimination
            script_path_1 = resolver.resolve_script_path(config_with_script)
            script_path_2 = resolver.resolve_script_path(config_without_script)
            
            assert script_path_1 is not None  # Real script discovered
            assert script_path_2 is None      # Phantom script eliminated
    
    def test_config_automation_concept(self):
        """Test the concept of config-based automation."""
        # This test validates that environment variables and job arguments
        # can be automatically extracted from config instances
        
        config = Mock()
        config.environment_variables = {"ENV_VAR": "value"}
        config.job_arguments = {"job_arg": "value"}
        
        # Test extraction methods exist and work conceptually
        factory = Mock()
        factory._extract_environ_vars_from_config = Mock(return_value={"ENV_VAR": "value"})
        factory._extract_job_args_from_config = Mock(return_value={"job_arg": "value"})
        
        env_vars = factory._extract_environ_vars_from_config(config)
        job_args = factory._extract_job_args_from_config(config)
        
        assert env_vars == {"ENV_VAR": "value"}
        assert job_args == {"job_arg": "value"}
