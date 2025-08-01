"""
Unit tests for the dag_compiler module.

This module tests the pipeline compilation process, particularly focusing on the 
conversion of PipelineDAG structures to SageMaker pipelines.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.pipeline_dag.base_dag import PipelineDAG
from src.pipeline_api.dag_compiler import PipelineDAGCompiler, compile_dag_to_pipeline
from src.pipeline_api.dynamic_template import DynamicPipelineTemplate
from src.pipeline_api.name_generator import validate_pipeline_name


class TestDagCompiler(unittest.TestCase):
    """Tests for the dag_compiler module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple DAG for testing
        self.dag = PipelineDAG()
        self.dag.add_node("data_loading")
        self.dag.add_node("preprocessing")
        self.dag.add_node("training")
        self.dag.add_edge("data_loading", "preprocessing")
        self.dag.add_edge("preprocessing", "training")
        
        # Mock config path (doesn't need to exist for these tests)
        self.config_path = "mock_config.json"

    @patch('src.pipeline_api.dag_compiler.Path')
    @patch('src.pipeline_api.dag_compiler.DynamicPipelineTemplate')
    @patch('src.pipeline_api.dag_compiler.StepBuilderRegistry')
    def test_compile_dag_pipeline_name_sanitization(self, mock_registry_class, mock_template_class, mock_path):
        """Test that compile_dag_to_pipeline sanitizes pipeline names."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        
        mock_template = MagicMock()
        mock_template.generate_pipeline.return_value = MagicMock()
        mock_template.base_config = MagicMock()
        mock_template.base_config.pipeline_name = "test.pipeline"
        mock_template.base_config.pipeline_version = "1.0.0"
        mock_template_class.return_value = mock_template
        
        # Create a compiler with the mocked template
        compiler = PipelineDAGCompiler(
            config_path=self.config_path,
            builder_registry=mock_registry
        )
        
        # Test with problematic pipeline name
        with patch('src.pipeline_api.name_generator.sanitize_pipeline_name') as mock_sanitize:
            mock_sanitize.return_value = "invalid-pipeline-name"
            pipeline = compiler.compile(self.dag, pipeline_name="invalid.pipeline.name")
            
            # Verify sanitize_pipeline_name was called with the problematic name
            mock_sanitize.assert_called_with("invalid.pipeline.name")
            
            # Verify sanitized name was used
            self.assertEqual(pipeline.name, "invalid-pipeline-name")
        
    @patch('src.pipeline_api.dag_compiler.Path')
    @patch('src.pipeline_api.dag_compiler.DynamicPipelineTemplate')
    @patch('src.pipeline_api.dag_compiler.StepBuilderRegistry')
    def test_default_pipeline_name_generation(self, mock_registry_class, mock_template_class, mock_path):
        """Test that default pipeline name generation uses name_generator."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        
        mock_template = MagicMock()
        mock_template.generate_pipeline.return_value = MagicMock()
        mock_template.base_config = MagicMock()
        mock_template.base_config.pipeline_name = "test.pipeline"
        mock_template.base_config.pipeline_version = "1.0.0"
        mock_template_class.return_value = mock_template
        
        # Create a compiler with the mocked template
        compiler = PipelineDAGCompiler(
            config_path=self.config_path,
            builder_registry=mock_registry
        )
        
        # Test with no pipeline name (should generate one)
        with patch('src.pipeline_api.dag_compiler.generate_pipeline_name') as mock_gen_name:
            mock_gen_name.return_value = "test-sanitized-1-0-pipeline"
            pipeline = compiler.compile(self.dag)
            
            # Verify generate_pipeline_name was called with correct args
            mock_gen_name.assert_called_with("test.pipeline", "1.0.0")
            self.assertEqual(pipeline.name, "test-sanitized-1-0-pipeline")


if __name__ == '__main__':
    unittest.main()
