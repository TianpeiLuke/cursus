"""
Tests for execution document generator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from cursus.mods.exe_doc.generator import ExecutionDocumentGenerator
from cursus.mods.exe_doc.base import (
    ExecutionDocumentHelper,
    ExecutionDocumentGenerationError,
    ConfigurationNotFoundError,
    UnsupportedStepTypeError,
)


class MockHelper(ExecutionDocumentHelper):
    """Mock helper for testing."""
    
    def __init__(self, can_handle_types=None):
        self.can_handle_types = can_handle_types or []
    
    def can_handle_step(self, step_name: str, config) -> bool:
        return type(config).__name__ in self.can_handle_types
    
    def extract_step_config(self, step_name: str, config) -> dict:
        return {"mock_config": f"config_for_{step_name}"}


class TestExecutionDocumentGenerator:
    """Tests for ExecutionDocumentGenerator class."""
    
    @patch('cursus.steps.configs.utils.load_configs')
    def test_init_success(self, mock_load_configs):
        """Test successful initialization."""
        mock_load_configs.return_value = {"config1": Mock()}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        
        assert generator.config_path == "test_config.json"
        assert generator.sagemaker_session is None
        assert generator.role is None
        assert len(generator.configs) == 1
        assert len(generator.helpers) == 2  # cradle_helper and registration_helper
    
    @patch('cursus.steps.configs.utils.load_configs')
    def test_init_with_optional_params(self, mock_load_configs):
        """Test initialization with optional parameters."""
        mock_load_configs.return_value = {}
        mock_session = Mock()
        mock_resolver = Mock()
        
        generator = ExecutionDocumentGenerator(
            "test_config.json",
            sagemaker_session=mock_session,
            role="test-role",
            config_resolver=mock_resolver
        )
        
        assert generator.sagemaker_session == mock_session
        assert generator.role == "test-role"
        assert generator.config_resolver == mock_resolver
    
    @patch('cursus.steps.configs.utils.load_configs')
    def test_init_config_loading_failure(self, mock_load_configs):
        """Test initialization failure when config loading fails."""
        mock_load_configs.side_effect = Exception("Config loading failed")
        
        with pytest.raises(ExecutionDocumentGenerationError, match="Configuration loading failed"):
            ExecutionDocumentGenerator("test_config.json")
    
    @patch('cursus.steps.configs.utils.load_configs')
    def test_identify_relevant_steps(self, mock_load_configs):
        """Test _identify_relevant_steps method."""
        mock_config = Mock()
        mock_config.__class__.__name__ = "CradleDataLoadConfig"
        mock_load_configs.return_value = {"step1": mock_config}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        
        # Mock config resolver to return config for step1 only
        def mock_resolve_config(step_name, configs):
            if step_name == "step1":
                return mock_config
            return None
        
        generator.config_resolver.resolve_config_for_step = Mock(side_effect=mock_resolve_config)
        
        # Create mock DAG
        dag = Mock()
        dag.nodes = ["step1", "step2"]
        
        relevant_steps = generator._identify_relevant_steps(dag)
        
        # Only step1 should be relevant (has cradle config)
        assert "step1" in relevant_steps
        assert len(relevant_steps) == 1
    
    @patch('cursus.steps.configs.utils.load_configs')
    def test_filter_steps_by_helper(self, mock_load_configs):
        """Test _filter_steps_by_helper method."""
        mock_config = Mock()
        mock_config.__class__.__name__ = "CradleDataLoadConfig"
        mock_load_configs.return_value = {"step1": mock_config}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        generator.config_resolver.resolve_config_for_step = Mock(return_value=mock_config)
        
        # Mock the cradle helper to handle cradle configs
        generator.cradle_helper.can_handle_step = Mock(return_value=True)
        generator.registration_helper.can_handle_step = Mock(return_value=False)
        
        # Test with cradle helper
        filtered_steps = generator._filter_steps_by_helper(["step1"], generator.cradle_helper)
        assert "step1" in filtered_steps
        
        # Test with registration helper (should not match cradle config)
        filtered_steps = generator._filter_steps_by_helper(["step1"], generator.registration_helper)
        assert len(filtered_steps) == 0
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    def test_fill_execution_document_invalid_structure(self, mock_build_classes, mock_load_configs):
        """Test fill_execution_document with invalid document structure."""
        mock_build_classes.return_value = {}
        mock_load_configs.return_value = {}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        dag = Mock()
        dag.nodes = []
        
        invalid_doc = {"INVALID": "structure"}
        
        # New behavior: returns document with warning (matches DynamicPipelineTemplate)
        result = generator.fill_execution_document(dag, invalid_doc)
        assert result == invalid_doc  # Document returned unchanged
    
    @patch('cursus.steps.configs.utils.load_configs')
    def test_fill_execution_document_success(self, mock_load_configs):
        """Test successful execution document filling."""
        # Setup mocks
        mock_config = Mock()
        mock_config.__class__.__name__ = "CradleDataLoadConfig"
        mock_load_configs.return_value = {"step1": mock_config}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        
        # Mock the cradle helper to handle the config
        generator.cradle_helper.can_handle_step = Mock(return_value=True)
        generator.cradle_helper.extract_step_config = Mock(return_value={"mock_config": "config_for_step1"})
        
        # Setup DAG
        dag = Mock()
        dag.nodes = ["step1"]
        
        # Setup execution document
        execution_doc = {
            "PIPELINE_STEP_CONFIGS": {
                "step1": {"STEP_TYPE": ["PROCESSING_STEP"]}
            }
        }
        
        # Mock config resolver
        generator.config_resolver.resolve_config_for_step = Mock(return_value=mock_config)
        
        result = generator.fill_execution_document(dag, execution_doc)
        
        assert "PIPELINE_STEP_CONFIGS" in result
        assert "step1" in result["PIPELINE_STEP_CONFIGS"]
        assert "STEP_CONFIG" in result["PIPELINE_STEP_CONFIGS"]["step1"]
        assert result["PIPELINE_STEP_CONFIGS"]["step1"]["STEP_CONFIG"]["mock_config"] == "config_for_step1"
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    def test_get_config_for_step_resolver_success(self, mock_build_classes, mock_load_configs):
        """Test _get_config_for_step with successful resolver."""
        mock_build_classes.return_value = {}
        mock_config = Mock()
        mock_load_configs.return_value = {"config1": mock_config}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        generator.config_resolver.resolve_config_for_step = Mock(return_value=mock_config)
        
        result = generator._get_config_for_step("step1")
        
        assert result == mock_config
        generator.config_resolver.resolve_config_for_step.assert_called_once_with("step1", generator.configs)
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    def test_get_config_for_step_fallback_direct_match(self, mock_build_classes, mock_load_configs):
        """Test _get_config_for_step with fallback to direct match."""
        mock_build_classes.return_value = {}
        mock_config = Mock()
        mock_load_configs.return_value = {"step1": mock_config}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        generator.config_resolver.resolve_config_for_step = Mock(side_effect=Exception("Resolver failed"))
        
        result = generator._get_config_for_step("step1")
        
        assert result == mock_config
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    def test_get_config_for_step_fallback_pattern_match(self, mock_build_classes, mock_load_configs):
        """Test _get_config_for_step with fallback to pattern matching."""
        mock_build_classes.return_value = {}
        mock_config = Mock()
        mock_load_configs.return_value = {"data_loading_config": mock_config}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        generator.config_resolver.resolve_config_for_step = Mock(side_effect=Exception("Resolver failed"))
        
        result = generator._get_config_for_step("data_loading_step")
        
        assert result == mock_config
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    def test_names_match(self, mock_build_classes, mock_load_configs):
        """Test _names_match method."""
        mock_build_classes.return_value = {}
        mock_load_configs.return_value = {}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        
        assert generator._names_match("data_loading_step", "data_loading_config") is True
        assert generator._names_match("data-loading-step", "data_loading_config") is True
        assert generator._names_match("step1", "completely_different") is False
    
    @patch('cursus.steps.configs.utils.load_configs')
    def test_is_execution_doc_relevant_with_helper(self, mock_load_configs):
        """Test _is_execution_doc_relevant with helper."""
        mock_load_configs.return_value = {}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        
        # Mock the cradle helper to handle TestConfig
        generator.cradle_helper.can_handle_step = Mock(return_value=True)
        
        config = Mock()
        config.__class__.__name__ = "TestConfig"
        
        assert generator._is_execution_doc_relevant(config) is True
    
    @patch('cursus.steps.configs.utils.load_configs')
    @patch('cursus.steps.configs.utils.build_complete_config_classes')
    def test_is_execution_doc_relevant_fallback(self, mock_build_classes, mock_load_configs):
        """Test _is_execution_doc_relevant with fallback logic."""
        mock_build_classes.return_value = {}
        mock_load_configs.return_value = {}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        
        config = Mock()
        config.__class__.__name__ = "CradleDataLoadConfig"
        
        assert generator._is_execution_doc_relevant(config) is True
        
        config.__class__.__name__ = "UnknownConfig"
        assert generator._is_execution_doc_relevant(config) is False
    
    @patch('cursus.steps.configs.utils.load_configs')
    def test_optimized_architecture_flow(self, mock_load_configs):
        """Test the optimized architecture flow with early exit."""
        mock_load_configs.return_value = {}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        
        # Mock DAG with no relevant steps
        dag = Mock()
        dag.nodes = ["irrelevant_step"]
        
        # Mock _get_config_for_step to return None (no config found)
        generator._get_config_for_step = Mock(return_value=None)
        
        execution_doc = {
            "PIPELINE_STEP_CONFIGS": {}
        }
        
        result = generator.fill_execution_document(dag, execution_doc)
        
        # Should return unchanged document due to early exit
        assert result == execution_doc
        
    @patch('cursus.steps.configs.utils.load_configs')
    def test_conditional_helper_processing(self, mock_load_configs):
        """Test that helpers are only called when relevant steps exist."""
        mock_config = Mock()
        mock_config.__class__.__name__ = "CradleDataLoadConfig"
        mock_load_configs.return_value = {"step1": mock_config}
        
        generator = ExecutionDocumentGenerator("test_config.json")
        generator.config_resolver.resolve_config_for_step = Mock(return_value=mock_config)
        
        # Mock the helpers to handle the config properly
        generator.cradle_helper.can_handle_step = Mock(return_value=True)
        generator.registration_helper.can_handle_step = Mock(return_value=False)
        
        # Mock the helper-specific methods
        generator._fill_cradle_configurations = Mock()
        generator._fill_registration_configurations = Mock()
        
        dag = Mock()
        dag.nodes = ["step1"]  # Only cradle step
        
        execution_doc = {
            "PIPELINE_STEP_CONFIGS": {"step1": {}}
        }
        
        generator.fill_execution_document(dag, execution_doc)
        
        # Only cradle helper should be called
        generator._fill_cradle_configurations.assert_called_once()
        generator._fill_registration_configurations.assert_not_called()
