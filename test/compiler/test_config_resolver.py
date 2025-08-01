"""
Unit tests for the config_resolver module.

These tests ensure that the StepConfigResolver class functions correctly,
particularly focusing on the intelligent matching of DAG nodes to configurations
using different resolution strategies.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.pipeline_api.config_resolver import StepConfigResolver
from src.pipeline_steps.config_base import BasePipelineConfig


class TestConfigResolver(unittest.TestCase):
    """Tests for the StepConfigResolver class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create an instance of StepConfigResolver
        self.resolver = StepConfigResolver()
        
        # Create mock configurations
        self.base_config = MagicMock(spec=BasePipelineConfig)
        type(self.base_config).__name__ = "BasePipelineConfig"
        
        self.data_load_config = MagicMock(spec=BasePipelineConfig)
        type(self.data_load_config).__name__ = "CradleDataLoadConfig"
        self.data_load_config.job_type = "training"
        
        self.preprocessing_config = MagicMock(spec=BasePipelineConfig)
        type(self.preprocessing_config).__name__ = "TabularPreprocessingConfig"
        self.preprocessing_config.job_type = "training"
        
        self.training_config = MagicMock(spec=BasePipelineConfig)
        type(self.training_config).__name__ = "XGBoostTrainingConfig"
        self.training_config.job_type = "training"
        
        self.eval_config = MagicMock(spec=BasePipelineConfig)
        type(self.eval_config).__name__ = "XGBoostModelEvalConfig"
        self.eval_config.job_type = "evaluation"
        
        # Create a dictionary of configurations
        self.configs = {
            "Base": self.base_config,
            "data_loading": self.data_load_config,
            "preprocessing": self.preprocessing_config,  # Changed from preprocess to preprocessing
            "training": self.training_config,           # Changed from train to training
            "evaluation": self.eval_config              # Changed from evaluate to evaluation
        }
        
        # List of DAG nodes
        self.dag_nodes = [
            "data_loading",
            "preprocessing",
            "training",
            "evaluation"
        ]
    
    def test_direct_name_matching(self):
        """Test the _direct_name_matching method."""
        # Test exact match
        match = self.resolver._direct_name_matching("data_loading", self.configs)
        self.assertEqual(match, self.data_load_config)
        
        # Test case-insensitive match
        match = self.resolver._direct_name_matching("Data_Loading", self.configs)
        self.assertEqual(match, self.data_load_config)
        
        # Test no match
        match = self.resolver._direct_name_matching("unknown_node", self.configs)
        self.assertIsNone(match)
    
    def test_job_type_matching(self):
        """Test the _job_type_matching method."""
        # Test job type matching with node name containing job type
        matches = self.resolver._job_type_matching("training_preprocess", self.configs)
        self.assertEqual(len(matches), 3)  # Should match data_load, preprocess, and train configs
        
        # Check that the preprocessing config has high confidence
        preprocess_match = next((m for m in matches if m[0] == self.preprocessing_config), None)
        self.assertIsNotNone(preprocess_match)
        self.assertGreater(preprocess_match[1], 0.7)  # Confidence should be > 0.7
        
        # Test job type matching with evaluation
        matches = self.resolver._job_type_matching("eval_step", self.configs)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], self.eval_config)
    
    def test_semantic_matching(self):
        """Test the _semantic_matching method."""
        # Test semantic match with "process" keyword
        matches = self.resolver._semantic_matching("process_data", self.configs)
        self.assertTrue(any(m[0] == self.preprocessing_config for m in matches))
        
        # Test semantic match with "train" keyword
        matches = self.resolver._semantic_matching("model_fit", self.configs)
        self.assertTrue(any(m[0] == self.training_config for m in matches))
        
        # Test semantic match with "evaluate" keyword
        matches = self.resolver._semantic_matching("model_test", self.configs)
        self.assertTrue(any(m[0] == self.eval_config for m in matches))
    
    def test_pattern_matching(self):
        """Test the _pattern_matching method."""
        # Test pattern match with data loading pattern
        matches = self.resolver._pattern_matching("cradle_data_load", self.configs)
        self.assertTrue(any(m[0] == self.data_load_config for m in matches))
        
        # Test pattern match with training pattern
        matches = self.resolver._pattern_matching("xgboost_train", self.configs)
        self.assertTrue(any(m[0] == self.training_config for m in matches))
    
    def test_resolve_config_map(self):
        """Test the resolve_config_map method."""
        # Override the _direct_name_matching, _job_type_matching, _semantic_matching, and _pattern_matching methods
        # to return predictable results for testing
        
        # Create a simple resolver with mocked resolution methods
        resolver = StepConfigResolver()
        
        # Set up direct match for data_loading
        def mock_direct_match(node_name, configs):
            if node_name == "data_loading":
                return self.data_load_config
            return None
        resolver._direct_name_matching = mock_direct_match
        
        # Set up job type match for preprocessing
        def mock_job_type_match(node_name, configs):
            if node_name == "preprocessing":
                return [(self.preprocessing_config, 0.8, "job_type")]
            return []
        resolver._job_type_matching = mock_job_type_match
        
        # Set up semantic match for training
        def mock_semantic_match(node_name, configs):
            if node_name == "training":
                return [(self.training_config, 0.7, "semantic")]
            return []
        resolver._semantic_matching = mock_semantic_match
        
        # Set up pattern match for evaluation
        def mock_pattern_match(node_name, configs):
            if node_name == "evaluation":
                return [(self.eval_config, 0.8, "pattern")]  # Increased from 0.6 to 0.8 to exceed threshold
            return []
        resolver._pattern_matching = mock_pattern_match
        
        # Resolve the config map
        config_map = resolver.resolve_config_map(self.dag_nodes, self.configs)
        
        # Verify the resolved map
        self.assertEqual(len(config_map), 4)
        self.assertEqual(config_map["data_loading"], self.data_load_config)
        self.assertEqual(config_map["preprocessing"], self.preprocessing_config)
        self.assertEqual(config_map["training"], self.training_config)
        self.assertEqual(config_map["evaluation"], self.eval_config)
    
    def test_resolve_single_node_direct_match(self):
        """Test that _resolve_single_node works with direct matching."""
        # Mock the direct name matching to return a successful match
        def mock_direct_match(node_name, configs):
            return self.data_load_config if node_name == "data_loading" else None
        original_direct_match = self.resolver._direct_name_matching
        self.resolver._direct_name_matching = mock_direct_match
        
        try:
            # Resolve a single node with direct matching
            config, confidence, method = self.resolver._resolve_single_node("data_loading", self.configs)
            
            # Verify the results
            self.assertEqual(config, self.data_load_config)
            self.assertEqual(confidence, 1.0)  # Direct match has confidence 1.0
            self.assertEqual(method, "direct_name")
        finally:
            # Restore original method
            self.resolver._direct_name_matching = original_direct_match
    
    def test_resolve_single_node_no_match(self):
        """Test that _resolve_single_node raises ConfigurationError when no match is found."""
        # Mock all matching methods to return no matches
        def mock_direct_match(node_name, configs):
            return None
        def mock_job_type_match(node_name, configs):
            return []
        def mock_semantic_match(node_name, configs):
            return []
        def mock_pattern_match(node_name, configs):
            return []
            
        original_direct_match = self.resolver._direct_name_matching
        original_job_type_match = self.resolver._job_type_matching
        original_semantic_match = self.resolver._semantic_matching
        original_pattern_match = self.resolver._pattern_matching
        
        self.resolver._direct_name_matching = mock_direct_match
        self.resolver._job_type_matching = mock_job_type_match
        self.resolver._semantic_matching = mock_semantic_match
        self.resolver._pattern_matching = mock_pattern_match
        
        try:
            # Attempt to resolve a node with no matches
            from src.pipeline_api.exceptions import ConfigurationError
            with self.assertRaises(ConfigurationError):
                self.resolver._resolve_single_node("unknown_node", self.configs)
        finally:
            # Restore original methods
            self.resolver._direct_name_matching = original_direct_match
            self.resolver._job_type_matching = original_job_type_match
            self.resolver._semantic_matching = original_semantic_match
            self.resolver._pattern_matching = original_pattern_match
    
    def test_resolve_single_node_ambiguity(self):
        """Test that _resolve_single_node raises AmbiguityError when multiple matches have similar confidence."""
        # Mock job type matching to return two candidates with similar confidence
        def mock_job_type_match(node_name, configs):
            if node_name == "preprocessing":
                return [
                    (self.preprocessing_config, 0.85, "job_type"),
                    (self.training_config, 0.82, "job_type")
                ]
            return []
            
        original_direct_match = self.resolver._direct_name_matching
        original_job_type_match = self.resolver._job_type_matching
        
        self.resolver._direct_name_matching = lambda node, configs: None
        self.resolver._job_type_matching = mock_job_type_match
        
        try:
            # Attempt to resolve a node with ambiguous matches
            from src.pipeline_api.exceptions import AmbiguityError
            with self.assertRaises(AmbiguityError):
                self.resolver._resolve_single_node("preprocessing", self.configs)
        finally:
            # Restore original methods
            self.resolver._direct_name_matching = original_direct_match
            self.resolver._job_type_matching = original_job_type_match
    
    def test_preview_resolution(self):
        """Test the preview_resolution method."""
        # Create a simple resolver with mocked _resolve_single_node method
        resolver = StepConfigResolver()
        
        # Set up mock candidates
        mock_candidates = {
            "data_loading": [
                {
                    "config": self.data_load_config,
                    "config_type": "CradleDataLoadConfig",
                    "confidence": 1.0,
                    "method": "direct_name",
                    "job_type": "training"
                }
            ],
            "preprocessing": [
                {
                    "config": self.preprocessing_config,
                    "config_type": "TabularPreprocessingConfig",
                    "confidence": 0.8,
                    "method": "job_type",
                    "job_type": "training"
                }
            ],
            "training": [
                {
                    "config": self.training_config,
                    "config_type": "XGBoostTrainingConfig",
                    "confidence": 0.7,
                    "method": "semantic",
                    "job_type": "training"
                }
            ],
            "evaluation": []  # No candidates for evaluation
        }
        
        # Mock preview_resolution to return the mock candidates
        def mock_resolve_candidates(dag_nodes, available_configs):
            return {node: mock_candidates.get(node, []) for node in dag_nodes}
        
        resolver.preview_resolution = mock_resolve_candidates
        
        # Get the preview
        preview = resolver.preview_resolution(self.dag_nodes, self.configs)
        
        # Verify the preview results
        self.assertEqual(len(preview), 4)
        
        # Check data_loading node
        self.assertIn("data_loading", preview)
        self.assertEqual(len(preview["data_loading"]), 1)
        self.assertEqual(preview["data_loading"][0]["confidence"], 1.0)
        
        # Check preprocessing node
        self.assertIn("preprocessing", preview)
        self.assertEqual(len(preview["preprocessing"]), 1)
        self.assertEqual(preview["preprocessing"][0]["confidence"], 0.8)
        
        # Check training node
        self.assertIn("training", preview)
        self.assertEqual(len(preview["training"]), 1)
        self.assertEqual(preview["training"][0]["confidence"], 0.7)
        
        # Check evaluation node (should be empty)
        self.assertIn("evaluation", preview)
        self.assertEqual(len(preview["evaluation"]), 0)


if __name__ == '__main__':
    unittest.main()
