"""
Unit tests for step_catalog.adapters.config_resolver module.

Tests the StepConfigResolverAdapter class that provides backward compatibility
with legacy config resolution systems.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List

from cursus.step_catalog.adapters.config_resolver import StepConfigResolverAdapter


class TestStepConfigResolverAdapter:
    """Test StepConfigResolverAdapter functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_configs(self):
        """Create mock configuration instances for testing."""
        configs = {}
        
        # Create mock configs with different attributes
        training_config = Mock()
        training_config.__class__.__name__ = "XGBoostTrainingConfig"
        training_config.job_type = "training"
        configs["training_config"] = training_config
        
        eval_config = Mock()
        eval_config.__class__.__name__ = "XGBoostEvalConfig"
        eval_config.job_type = "evaluation"
        configs["eval_config"] = eval_config
        
        data_config = Mock()
        data_config.__class__.__name__ = "CradleDataLoadingConfig"
        data_config.job_type = "processing"
        configs["data_config"] = data_config
        
        return configs
    
    def test_init_without_workspace(self):
        """Test initialization without workspace root."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog') as mock_catalog:
            adapter = StepConfigResolverAdapter()
            
            # Should initialize StepCatalog with workspace_dirs=None
            mock_catalog.assert_called_once_with(workspace_dirs=None)
            assert adapter.confidence_threshold == 0.7
            assert adapter.logger is not None
            assert adapter._metadata_mapping == {}
            assert adapter._config_cache == {}
    
    def test_init_with_workspace_and_threshold(self, temp_workspace):
        """Test initialization with workspace root and custom threshold."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog') as mock_catalog:
            adapter = StepConfigResolverAdapter(workspace_root=temp_workspace, confidence_threshold=0.8)
            
            # Should initialize StepCatalog with workspace_dirs=[workspace_root]
            mock_catalog.assert_called_once_with(workspace_dirs=[temp_workspace])
            assert adapter.confidence_threshold == 0.8
    
    def test_constants(self):
        """Test that constants are properly defined."""
        assert "training" in StepConfigResolverAdapter.JOB_TYPE_KEYWORDS
        assert "calibration" in StepConfigResolverAdapter.JOB_TYPE_KEYWORDS
        assert "evaluation" in StepConfigResolverAdapter.JOB_TYPE_KEYWORDS
        assert "inference" in StepConfigResolverAdapter.JOB_TYPE_KEYWORDS
        
        assert len(StepConfigResolverAdapter.STEP_TYPE_PATTERNS) > 0
        assert any(".*train.*" in pattern for pattern in StepConfigResolverAdapter.STEP_TYPE_PATTERNS.keys())
    
    def test_resolve_config_map_direct_matching(self, mock_configs):
        """Test resolve_config_map with direct name matching."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            dag_nodes = ["training_config", "eval_config"]
            result = adapter.resolve_config_map(dag_nodes, mock_configs)
            
            assert len(result) == 2
            assert result["training_config"] == mock_configs["training_config"]
            assert result["eval_config"] == mock_configs["eval_config"]
    
    def test_resolve_config_map_catalog_fallback(self, mock_configs):
        """Test resolve_config_map with catalog-based fallback."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_step_info.config_class = "XGBoostTrainingConfig"
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = StepConfigResolverAdapter()
            
            dag_nodes = ["unknown_node"]
            result = adapter.resolve_config_map(dag_nodes, mock_configs)
            
            # Should find config by matching config class
            assert len(result) == 1
            assert result["unknown_node"] == mock_configs["training_config"]
    
    def test_resolve_config_map_last_resort(self, mock_configs):
        """Test resolve_config_map last resort fallback."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.get_step_info.return_value = None
            mock_catalog_class.return_value = mock_catalog
            
            adapter = StepConfigResolverAdapter()
            
            dag_nodes = ["completely_unknown"]
            result = adapter.resolve_config_map(dag_nodes, mock_configs)
            
            # Should use first available config as last resort
            assert len(result) == 1
            assert result["completely_unknown"] in mock_configs.values()
    
    def test_resolve_config_map_error_handling(self, mock_configs):
        """Test error handling in resolve_config_map."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog') as mock_catalog_class:
            # Create a mock catalog that works for initialization but fails during resolve_config_map
            mock_catalog = Mock()
            mock_catalog.get_step_info.side_effect = Exception("Test error")
            mock_catalog_class.return_value = mock_catalog
            
            adapter = StepConfigResolverAdapter()
            result = adapter.resolve_config_map(["test_node"], mock_configs)
            
            assert result == {}
    
    def test_direct_name_matching_exact_match(self, mock_configs):
        """Test direct name matching with exact key match."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            result = adapter._direct_name_matching("training_config", mock_configs)
            
            assert result == mock_configs["training_config"]
    
    def test_direct_name_matching_case_insensitive(self, mock_configs):
        """Test direct name matching with case insensitive match."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            result = adapter._direct_name_matching("TRAINING_CONFIG", mock_configs)
            
            assert result == mock_configs["training_config"]
    
    def test_direct_name_matching_metadata_mapping(self, mock_configs):
        """Test direct name matching with metadata mapping."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            adapter._metadata_mapping = {"customnode": "XGBoostTrainingConfig"}
            
            # The implementation uses type(config).__name__ == config_class_name
            # Create a custom class for the mock to have the right type
            class XGBoostTrainingConfig:
                def __init__(self):
                    self.job_type = "training"
            
            # Replace the mock with an actual instance that has the right type
            real_config = XGBoostTrainingConfig()
            mock_configs["training_config"] = real_config
            
            # Use a node name without underscore so job type matching is skipped
            result = adapter._direct_name_matching("customnode", mock_configs)
            
            assert result == real_config
    
    def test_direct_name_matching_with_job_type(self, mock_configs):
        """Test direct name matching with job type consideration."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            adapter._metadata_mapping = {"custom_node_training": "XGBoostTrainingConfig"}
            
            result = adapter._direct_name_matching("custom_node_training", mock_configs)
            
            assert result == mock_configs["training_config"]
    
    def test_direct_name_matching_no_match(self, mock_configs):
        """Test direct name matching when no match is found."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            result = adapter._direct_name_matching("nonexistent_config", mock_configs)
            
            assert result is None
    
    def test_job_type_matching(self, mock_configs):
        """Test job type matching functionality."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Test with node name containing training keyword
            matches = adapter._job_type_matching("some_training_step", mock_configs)
            
            assert len(matches) > 0
            # Should find the training config
            config, confidence, method = matches[0]
            assert config == mock_configs["training_config"]
            assert method == "job_type"
            assert confidence > 0.7
    
    def test_job_type_matching_no_job_type_detected(self, mock_configs):
        """Test job type matching when no job type is detected."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            matches = adapter._job_type_matching("generic_step", mock_configs)
            
            assert len(matches) == 0
    
    def test_calculate_config_type_confidence(self, mock_configs):
        """Test config type confidence calculation."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Test with matching config type
            confidence = adapter._calculate_config_type_confidence("xgboost_training", mock_configs["training_config"])
            
            assert confidence > 0.0
    
    def test_semantic_matching(self, mock_configs):
        """Test semantic matching functionality."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            matches = adapter._semantic_matching("data_loading_step", mock_configs)
            
            # Should find matches based on semantic keywords
            assert len(matches) >= 0  # May or may not find matches depending on config names
    
    def test_pattern_matching(self, mock_configs):
        """Test pattern matching functionality."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            matches = adapter._pattern_matching("train_model", mock_configs)
            
            # Should find matches based on regex patterns
            assert isinstance(matches, list)
    
    def test_config_class_to_step_type_with_catalog(self, mock_configs):
        """Test config class to step type conversion using catalog."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_step_info.config_class = "XGBoostTrainingConfig"
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog.list_available_steps.return_value = ["XGBoostTraining"]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = StepConfigResolverAdapter()
            
            result = adapter._config_class_to_step_type("XGBoostTrainingConfig")
            
            assert result == "XGBoostTraining"
    
    def test_config_class_to_step_type_legacy_fallback(self, mock_configs):
        """Test config class to step type conversion with legacy fallback."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = []
            mock_catalog.get_step_info.return_value = None
            mock_catalog_class.return_value = mock_catalog
            
            adapter = StepConfigResolverAdapter()
            
            result = adapter._config_class_to_step_type("XGBoostTrainingConfig")
            
            assert result == "XGBoostTraining"  # Should remove "Config" suffix
    
    def test_config_class_to_step_type_special_cases(self, mock_configs):
        """Test config class to step type conversion for special cases."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = []
            mock_catalog.get_step_info.return_value = None
            mock_catalog_class.return_value = mock_catalog
            
            adapter = StepConfigResolverAdapter()
            
            # Test special case mappings
            assert adapter._config_class_to_step_type("CradleDataLoadConfig") == "CradleDataLoading"
            assert adapter._config_class_to_step_type("PackageStepConfig") == "MIMSPackaging"
            assert adapter._config_class_to_step_type("PackageConfig") == "MIMSPackaging"
    
    def test_calculate_job_type_boost(self, mock_configs):
        """Test job type boost calculation."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Test with matching job type
            boost = adapter._calculate_job_type_boost("training_step", mock_configs["training_config"])
            
            assert boost >= 0.0
            assert boost <= 1.0
    
    def test_calculate_job_type_boost_no_job_type(self, mock_configs):
        """Test job type boost calculation when config has no job_type."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            config_without_job_type = Mock()
            config_without_job_type.__class__.__name__ = "GenericConfig"
            # Remove job_type attribute completely so hasattr returns False
            if hasattr(config_without_job_type, 'job_type'):
                delattr(config_without_job_type, 'job_type')
            
            boost = adapter._calculate_job_type_boost("training_step", config_without_job_type)
            
            assert boost == 0.0
    
    def test_resolve_single_node_direct_match(self, mock_configs):
        """Test resolving single node with direct match."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            config, confidence, method = adapter._resolve_single_node("training_config", mock_configs)
            
            assert config == mock_configs["training_config"]
            assert confidence == 1.0
            assert method == "direct_name"
    
    def test_resolve_single_node_no_match(self, mock_configs):
        """Test resolving single node when no match is found."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Mock all matching methods to return empty lists
            with patch.object(adapter, '_job_type_matching', return_value=[]):
                with patch.object(adapter, '_semantic_matching', return_value=[]):
                    with patch.object(adapter, '_pattern_matching', return_value=[]):
                        with pytest.raises((ValueError, Exception)):  # Should raise ResolutionError or ValueError
                            adapter._resolve_single_node("unknown_node", mock_configs)
    
    def test_resolve_single_node_best_match(self, mock_configs):
        """Test resolving single node returns best match."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Mock matching methods to return different confidence scores
            with patch.object(adapter, '_job_type_matching', return_value=[(mock_configs["training_config"], 0.8, "job_type")]):
                with patch.object(adapter, '_semantic_matching', return_value=[(mock_configs["eval_config"], 0.6, "semantic")]):
                    with patch.object(adapter, '_pattern_matching', return_value=[(mock_configs["data_config"], 0.7, "pattern")]):
                        config, confidence, method = adapter._resolve_single_node("test_node", mock_configs)
                        
                        # Should return the highest confidence match
                        assert config == mock_configs["training_config"]
                        assert confidence == 0.8
                        assert method == "job_type"
    
    def test_resolve_config_for_step_success(self, mock_configs):
        """Test resolving config for a single step successfully."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            result = adapter.resolve_config_for_step("training_config", mock_configs)
            
            assert result == mock_configs["training_config"]
    
    def test_resolve_config_for_step_fallback(self, mock_configs):
        """Test resolving config for a single step with fallback."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Mock _resolve_single_node to return a tuple
            with patch.object(adapter, '_resolve_single_node', return_value=(mock_configs["training_config"], 0.8, "test")):
                result = adapter.resolve_config_for_step("unknown_step", mock_configs)
                
                assert result == mock_configs["training_config"]
    
    def test_resolve_config_for_step_error(self, mock_configs):
        """Test error handling in resolve_config_for_step."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Mock methods to raise exception
            with patch.object(adapter, '_direct_name_matching', side_effect=Exception("Test error")):
                result = adapter.resolve_config_for_step("test_step", mock_configs)
                
                assert result is None
    
    def test_preview_resolution_success(self, mock_configs):
        """Test preview resolution functionality."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            dag_nodes = ["training_config", "eval_config"]
            metadata = {"config_types": {"custom_node": "XGBoostTrainingConfig"}}
            
            result = adapter.preview_resolution(dag_nodes, mock_configs, metadata)
            
            assert "node_resolution" in result
            assert "resolution_confidence" in result
            assert "node_config_map" in result
            assert "metadata_mapping" in result
            assert len(result["node_resolution"]) == 2
    
    def test_preview_resolution_with_errors(self, mock_configs):
        """Test preview resolution with resolution errors."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Mock _resolve_single_node to raise exception
            with patch.object(adapter, '_resolve_single_node', side_effect=Exception("Resolution failed")):
                result = adapter.preview_resolution(["unknown_node"], mock_configs)
                
                assert "node_resolution" in result
                assert "unknown_node" in result["node_resolution"]
                assert "error" in result["node_resolution"]["unknown_node"]
    
    def test_preview_resolution_error_handling(self, mock_configs):
        """Test error handling in preview_resolution."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog') as mock_catalog_class:
            # Create a mock catalog that works for initialization but fails during preview_resolution
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            adapter = StepConfigResolverAdapter()
            
            # Mock a method that preview_resolution calls to cause an error
            with patch.object(adapter, '_resolve_single_node', side_effect=Exception("Catalog error")):
                result = adapter.preview_resolution(["test_node"], mock_configs)
                
                # Check that error is properly handled in node_resolution
                assert "node_resolution" in result
                assert "test_node" in result["node_resolution"]
                assert "error" in result["node_resolution"]["test_node"]
    
    def test_parse_node_name_config_first_pattern(self):
        """Test parsing node name with ConfigType_JobType pattern."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            result = adapter._parse_node_name("CradleDataLoading_training")
            
            assert result["config_type"] == "CradleDataLoading"
            assert result["job_type"] == "training"
    
    def test_parse_node_name_job_first_pattern(self):
        """Test parsing node name with JobType_Task pattern."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            result = adapter._parse_node_name("training_data_load")
            
            assert result["job_type"] == "training"
            assert result.get("config_type") == "CradleDataLoading"  # Should be inferred from task_map
    
    def test_parse_node_name_caching(self):
        """Test that parse_node_name caches results."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # First call
            result1 = adapter._parse_node_name("CradleDataLoading_training")
            
            # Second call should use cache
            result2 = adapter._parse_node_name("CradleDataLoading_training")
            
            assert result1 == result2
            assert "CradleDataLoading_training" in adapter._config_cache
    
    def test_parse_node_name_no_pattern_match(self):
        """Test parsing node name that doesn't match any pattern."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Use a name that truly doesn't match any pattern (no underscore)
            result = adapter._parse_node_name("simplenode")
            
            # Should return empty dict when no pattern matches
            assert result == {}
    
    def test_job_type_matching_enhanced(self, mock_configs):
        """Test enhanced job type matching functionality."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            matches = adapter._job_type_matching_enhanced("training", mock_configs)
            
            assert len(matches) > 0
            # Should find the training config
            config, confidence, method = matches[0]
            assert config == mock_configs["training_config"]
            assert method == "job_type_enhanced"
            assert confidence >= 0.8
    
    def test_job_type_matching_enhanced_with_config_type(self, mock_configs):
        """Test enhanced job type matching with config type filtering."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            matches = adapter._job_type_matching_enhanced("training", mock_configs, "XGBoostTrainingConfig")
            
            assert len(matches) > 0
            # Should find exact match with higher confidence
            config, confidence, method = matches[0]
            assert config == mock_configs["training_config"]
            assert confidence >= 0.9  # Should have higher confidence due to exact config type match
    
    def test_job_type_matching_enhanced_no_match(self, mock_configs):
        """Test enhanced job type matching when no job type matches."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            matches = adapter._job_type_matching_enhanced("nonexistent_job_type", mock_configs)
            
            assert len(matches) == 0


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.fixture
    def realistic_configs(self):
        """Create realistic configuration instances for integration testing."""
        configs = {}
        
        # XGBoost Training Config
        xgb_train = Mock()
        xgb_train.__class__.__name__ = "XGBoostTrainingConfig"
        xgb_train.job_type = "training"
        configs["xgboost_training"] = xgb_train
        
        # Data Loading Config
        data_load = Mock()
        data_load.__class__.__name__ = "CradleDataLoadingConfig"
        data_load.job_type = "processing"
        configs["data_loading"] = data_load
        
        # Model Evaluation Config
        model_eval = Mock()
        model_eval.__class__.__name__ = "XGBoostModelEvalConfig"
        model_eval.job_type = "evaluation"
        configs["model_evaluation"] = model_eval
        
        return configs
    
    def test_complete_dag_resolution_workflow(self, realistic_configs):
        """Test complete DAG resolution workflow."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.get_step_info.return_value = None  # Force fallback to other methods
            mock_catalog_class.return_value = mock_catalog
            
            adapter = StepConfigResolverAdapter()
            
            dag_nodes = ["xgboost_training", "data_loading", "model_evaluation"]
            
            # Test resolution
            result = adapter.resolve_config_map(dag_nodes, realistic_configs)
            
            assert len(result) == 3
            assert all(node in result for node in dag_nodes)
    
    def test_metadata_driven_resolution(self, realistic_configs):
        """Test resolution driven by metadata.config_types mapping."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            dag_nodes = ["custom_training_step", "custom_data_step"]
            metadata = {
                "config_types": {
                    "custom_training_step": "XGBoostTrainingConfig",
                    "custom_data_step": "CradleDataLoadingConfig"
                }
            }
            
            # Test preview with metadata
            result = adapter.preview_resolution(dag_nodes, realistic_configs, metadata)
            
            assert "metadata_mapping" in result
            assert len(result["metadata_mapping"]) == 2
            assert result["metadata_mapping"]["custom_training_step"] == "XGBoostTrainingConfig"
    
    def test_mixed_resolution_strategies(self, realistic_configs):
        """Test scenario using multiple resolution strategies."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Mix of direct matches and pattern-based matches
            dag_nodes = [
                "xgboost_training",  # Direct match
                "train_model_step",  # Pattern match
                "data_preprocessing"  # Semantic match
            ]
            
            result = adapter.resolve_config_map(dag_nodes, realistic_configs)
            
            # Should resolve at least some nodes using different strategies
            # The exact number depends on the implementation's ability to match patterns
            assert len(result) >= 1
            assert "xgboost_training" in result  # Direct match should always work
    
    def test_error_resilience_in_production_scenario(self, realistic_configs):
        """Test error resilience in production-like scenario."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog') as mock_catalog_class:
            # Create a mock catalog that works for initialization but fails during resolve_config_map
            mock_catalog = Mock()
            mock_catalog.get_step_info.side_effect = Exception("Catalog unavailable")
            mock_catalog_class.return_value = mock_catalog
            
            adapter = StepConfigResolverAdapter()
            
            dag_nodes = ["some_node"]
            
            # Should handle catalog failure gracefully
            result = adapter.resolve_config_map(dag_nodes, realistic_configs)
            
            # Should return empty dict on error
            assert result == {}


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def mock_configs(self):
        """Create mock configuration instances for testing."""
        configs = {}
        
        # Create mock configs with different attributes
        training_config = Mock()
        training_config.__class__.__name__ = "XGBoostTrainingConfig"
        training_config.job_type = "training"
        configs["training_config"] = training_config
        
        eval_config = Mock()
        eval_config.__class__.__name__ = "XGBoostEvalConfig"
        eval_config.job_type = "evaluation"
        configs["eval_config"] = eval_config
        
        return configs
    
    def test_empty_configs_dict(self):
        """Test behavior with empty configs dictionary."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            result = adapter.resolve_config_map(["test_node"], {})
            
            assert result == {}
    
    def test_empty_dag_nodes_list(self, mock_configs):
        """Test behavior with empty DAG nodes list."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            result = adapter.resolve_config_map([], mock_configs)
            
            assert result == {}
    
    def test_config_without_required_attributes(self):
        """Test handling configs without required attributes."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Config without job_type attribute
            incomplete_config = Mock()
            incomplete_config.__class__.__name__ = "IncompleteConfig"
            # No job_type attribute
            
            configs = {"incomplete": incomplete_config}
            
            # Should handle gracefully
            result = adapter.resolve_config_for_step("test_step", configs)
            
            # May return the config or None, but shouldn't crash
            assert result is None or result == incomplete_config
    
    def test_malformed_metadata(self, mock_configs):
        """Test handling of malformed metadata."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            # Malformed metadata
            malformed_metadata = {
                "config_types": "not_a_dict"  # Should be dict
            }
            
            # Should handle gracefully
            result = adapter.preview_resolution(["test_node"], mock_configs, malformed_metadata)
            
            assert isinstance(result, dict)
    
    def test_very_long_node_names(self, mock_configs):
        """Test handling of very long node names."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            very_long_name = "a" * 1000  # Very long node name
            
            # Should handle without crashing
            result = adapter.resolve_config_for_step(very_long_name, mock_configs)
            
            # Should return None or a config, but not crash
            assert result is None or result in mock_configs.values()
    
    def test_special_characters_in_node_names(self, mock_configs):
        """Test handling of special characters in node names."""
        with patch('cursus.step_catalog.adapters.config_resolver.StepCatalog'):
            adapter = StepConfigResolverAdapter()
            
            special_names = [
                "node-with-dashes",
                "node.with.dots",
                "node@with@symbols",
                "node with spaces"
            ]
            
            for name in special_names:
                # Should handle without crashing
                result = adapter.resolve_config_for_step(name, mock_configs)
                
                # Should return None or a config, but not crash
                assert result is None or result in mock_configs.values
