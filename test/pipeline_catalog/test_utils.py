"""
Tests for the pipeline catalog utils module.
"""

import json
import unittest
from pathlib import Path
from unittest import mock

from cursus.pipeline_catalog import utils

class TestPipelineCatalogUtils(unittest.TestCase):
    """Test cases for the pipeline catalog utils module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock index for testing
        self.mock_index = {
            "pipelines": [
                {
                    "id": "test-xgboost-simple",
                    "name": "Test XGBoost Simple",
                    "path": "frameworks/xgboost/simple.py",
                    "framework": "xgboost",
                    "complexity": "simple",
                    "features": ["training"],
                    "tags": ["xgboost", "training", "beginner"]
                },
                {
                    "id": "test-pytorch-e2e",
                    "name": "Test PyTorch E2E",
                    "path": "frameworks/pytorch/end_to_end/standard_e2e.py",
                    "framework": "pytorch",
                    "complexity": "advanced",
                    "features": ["training", "evaluation", "registration"],
                    "tags": ["pytorch", "end-to-end", "evaluation"]
                }
            ]
        }
    
    @mock.patch("src.cursus.pipeline_catalog.utils.load_index")
    def test_get_pipeline_by_id(self, mock_load_index):
        """Test getting a pipeline by ID."""
        mock_load_index.return_value = self.mock_index
        
        # Test finding an existing pipeline
        pipeline = utils.get_pipeline_by_id("test-xgboost-simple")
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline["id"], "test-xgboost-simple")
        
        # Test with a non-existent pipeline
        pipeline = utils.get_pipeline_by_id("non-existent")
        self.assertIsNone(pipeline)
    
    @mock.patch("src.cursus.pipeline_catalog.utils.load_index")
    def test_filter_pipelines_by_framework(self, mock_load_index):
        """Test filtering pipelines by framework."""
        mock_load_index.return_value = self.mock_index
        
        # Filter by XGBoost framework
        pipelines = utils.filter_pipelines(framework="xgboost")
        self.assertEqual(len(pipelines), 1)
        self.assertEqual(pipelines[0]["id"], "test-xgboost-simple")
        
        # Filter by PyTorch framework
        pipelines = utils.filter_pipelines(framework="pytorch")
        self.assertEqual(len(pipelines), 1)
        self.assertEqual(pipelines[0]["id"], "test-pytorch-e2e")
        
        # Filter by non-existent framework
        pipelines = utils.filter_pipelines(framework="non-existent")
        self.assertEqual(len(pipelines), 0)
    
    @mock.patch("src.cursus.pipeline_catalog.utils.load_index")
    def test_filter_pipelines_by_complexity(self, mock_load_index):
        """Test filtering pipelines by complexity."""
        mock_load_index.return_value = self.mock_index
        
        # Filter by simple complexity
        pipelines = utils.filter_pipelines(complexity="simple")
        self.assertEqual(len(pipelines), 1)
        self.assertEqual(pipelines[0]["id"], "test-xgboost-simple")
        
        # Filter by advanced complexity
        pipelines = utils.filter_pipelines(complexity="advanced")
        self.assertEqual(len(pipelines), 1)
        self.assertEqual(pipelines[0]["id"], "test-pytorch-e2e")
    
    @mock.patch("src.cursus.pipeline_catalog.utils.load_index")
    def test_filter_pipelines_by_features(self, mock_load_index):
        """Test filtering pipelines by features."""
        mock_load_index.return_value = self.mock_index
        
        # Filter by training feature
        pipelines = utils.filter_pipelines(features=["training"])
        self.assertEqual(len(pipelines), 2)
        
        # Filter by registration feature
        pipelines = utils.filter_pipelines(features=["registration"])
        self.assertEqual(len(pipelines), 1)
        self.assertEqual(pipelines[0]["id"], "test-pytorch-e2e")
        
        # Filter by multiple features
        pipelines = utils.filter_pipelines(features=["training", "evaluation"])
        self.assertEqual(len(pipelines), 1)
        self.assertEqual(pipelines[0]["id"], "test-pytorch-e2e")
    
    @mock.patch("src.cursus.pipeline_catalog.utils.load_index")
    def test_filter_pipelines_by_tags(self, mock_load_index):
        """Test filtering pipelines by tags."""
        mock_load_index.return_value = self.mock_index
        
        # Filter by beginner tag
        pipelines = utils.filter_pipelines(tags=["beginner"])
        self.assertEqual(len(pipelines), 1)
        self.assertEqual(pipelines[0]["id"], "test-xgboost-simple")
        
        # Filter by end-to-end tag
        pipelines = utils.filter_pipelines(tags=["end-to-end"])
        self.assertEqual(len(pipelines), 1)
        self.assertEqual(pipelines[0]["id"], "test-pytorch-e2e")
    
    @mock.patch("src.cursus.pipeline_catalog.utils.load_index")
    def test_get_all_frameworks(self, mock_load_index):
        """Test getting all frameworks."""
        mock_load_index.return_value = self.mock_index
        
        frameworks = utils.get_all_frameworks()
        self.assertEqual(set(frameworks), {"xgboost", "pytorch"})
    
    @mock.patch("src.cursus.pipeline_catalog.utils.load_index")
    def test_get_all_features(self, mock_load_index):
        """Test getting all features."""
        mock_load_index.return_value = self.mock_index
        
        features = utils.get_all_features()
        self.assertEqual(set(features), {"training", "evaluation", "registration"})

if __name__ == "__main__":
    unittest.main()
