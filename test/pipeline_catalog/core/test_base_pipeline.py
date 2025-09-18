"""
Unit tests for BasePipeline class.

Tests the abstract base class functionality including DAG compilation,
pipeline generation, and registry synchronization.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from cursus.pipeline_catalog.core.base_pipeline import BasePipeline
from cursus.api.dag.base_dag import PipelineDAG
from cursus.pipeline_catalog.shared_dags.enhanced_metadata import EnhancedDAGMetadata


class ConcretePipeline(BasePipeline):
    """Concrete implementation of BasePipeline for testing."""

    def create_dag(self) -> PipelineDAG:
        """Create a simple test DAG."""
        dag = PipelineDAG()
        dag.add_node("step1")
        dag.add_node("step2")
        dag.add_edge("step1", "step2")
        return dag

    def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
        """Return test metadata."""
        mock_metadata = Mock(spec=EnhancedDAGMetadata)
        mock_metadata.zettelkasten_metadata = Mock()
        mock_metadata.zettelkasten_metadata.atomic_id = "test_pipeline"
        return mock_metadata


class TestBasePipeline:
    """Test suite for BasePipeline class."""

    @pytest.fixture
    def mock_session(self):
        """Create mock PipelineSession."""
        session = Mock(spec=PipelineSession)
        session.get_caller_identity_arn.return_value = "arn:aws:iam::123456789012:role/test-role"
        return session

    @pytest.fixture
    def mock_dag_compiler(self):
        """Create mock DAG compiler."""
        compiler = Mock()
        compiler.validate_dag_compatibility.return_value = Mock(
            is_valid=True,
            missing_configs=[],
            unresolvable_builders=[],
            config_errors=[],
            dependency_issues=[],
            warnings=[]
        )
        compiler.compile_with_report.return_value = (Mock(spec=Pipeline), Mock(avg_confidence=0.95))
        compiler.get_last_template.return_value = Mock()
        return compiler

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create temporary config file."""
        config_file = tmp_path / "test_config.json"
        config_data = {
            "pipeline_name": "test_pipeline",
            "parameters": {"param1": "value1"}
        }
        config_file.write_text(json.dumps(config_data))
        return str(config_file)

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_init_with_defaults(self, mock_compiler_class, mock_session):
        """Test initialization with default parameters."""
        mock_compiler_class.return_value = Mock()
        
        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline()

            assert pipeline.sagemaker_session == mock_session
            assert pipeline.execution_role == "arn:aws:iam::123456789012:role/test-role"
            assert pipeline.enable_mods is True
            assert pipeline.validate is True
            assert pipeline.config == {}
            assert isinstance(pipeline.dag, PipelineDAG)

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_init_with_config_file(self, mock_compiler_class, mock_session, temp_config_file):
        """Test initialization with config file."""
        mock_compiler_class.return_value = Mock()
        
        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline(config_path=temp_config_file)

            assert pipeline.config_path == temp_config_file
            assert pipeline.config["pipeline_name"] == "test_pipeline"
            assert pipeline.config["parameters"]["param1"] == "value1"

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_init_with_nonexistent_config(self, mock_compiler_class, mock_session):
        """Test initialization with non-existent config file."""
        # Mock the compiler to not raise FileNotFoundError
        mock_compiler_class.return_value = Mock()
        
        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline(config_path="/nonexistent/config.json")

            assert pipeline.config_path == "/nonexistent/config.json"
            assert pipeline.config == {}

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_init_with_custom_parameters(self, mock_compiler_class, mock_session):
        """Test initialization with custom parameters."""
        mock_compiler_class.return_value = Mock()
        custom_role = "arn:aws:iam::123456789012:role/custom-role"
        
        pipeline = ConcretePipeline(
            sagemaker_session=mock_session,
            execution_role=custom_role,
            enable_mods=False,
            validate=False
        )

        assert pipeline.sagemaker_session == mock_session
        assert pipeline.execution_role == custom_role
        assert pipeline.enable_mods is False
        assert pipeline.validate is False

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_initialize_compiler(self, mock_compiler_class, mock_session):
        """Test DAG compiler initialization."""
        mock_compiler = Mock()
        mock_compiler_class.return_value = mock_compiler

        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline()

            assert pipeline.dag_compiler == mock_compiler
            mock_compiler_class.assert_called_once_with(
                config_path=None,
                sagemaker_session=mock_session,
                role="arn:aws:iam::123456789012:role/test-role"
            )

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_generate_pipeline_with_validation(self, mock_compiler_class, mock_session):
        """Test pipeline generation with validation enabled."""
        mock_compiler = Mock()
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.name = "test_pipeline"
        mock_report = Mock(avg_confidence=0.95)
        
        mock_compiler.validate_dag_compatibility.return_value = Mock(is_valid=True)
        mock_compiler.compile_with_report.return_value = (mock_pipeline, mock_report)
        mock_compiler.get_last_template.return_value = Mock()
        mock_compiler_class.return_value = mock_compiler

        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline(validate=True)
            result = pipeline.generate_pipeline()

            assert result == mock_pipeline
            mock_compiler.validate_dag_compatibility.assert_called_once_with(pipeline.dag)
            mock_compiler.compile_with_report.assert_called_once_with(dag=pipeline.dag)

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_generate_pipeline_validation_failed(self, mock_compiler_class, mock_session):
        """Test pipeline generation with validation failures."""
        mock_compiler = Mock()
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.name = "test_pipeline"
        mock_report = Mock(avg_confidence=0.95)
        
        # Mock validation failure
        validation_result = Mock(
            is_valid=False,
            missing_configs=["config1"],
            unresolvable_builders=["builder1"],
            config_errors=["error1"],
            dependency_issues=["issue1"]
        )
        mock_compiler.validate_dag_compatibility.return_value = validation_result
        mock_compiler.compile_with_report.return_value = (mock_pipeline, mock_report)
        mock_compiler_class.return_value = mock_compiler

        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline(validate=True)
            
            # Should still generate pipeline despite validation warnings
            result = pipeline.generate_pipeline()
            assert result == mock_pipeline

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_validate_dag_compatibility(self, mock_compiler_class, mock_session):
        """Test DAG compatibility validation."""
        mock_compiler = Mock()
        validation_result = Mock(
            is_valid=True,
            missing_configs=[],
            unresolvable_builders=[],
            config_errors=[],
            dependency_issues=[],
            warnings=[]
        )
        mock_compiler.validate_dag_compatibility.return_value = validation_result
        mock_compiler_class.return_value = mock_compiler

        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline()
            result = pipeline.validate_dag_compatibility()

            expected = {
                "is_valid": True,
                "missing_configs": [],
                "unresolvable_builders": [],
                "config_errors": [],
                "dependency_issues": [],
                "warnings": []
            }
            assert result == expected

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_preview_resolution(self, mock_compiler_class, mock_session):
        """Test resolution preview."""
        mock_compiler = Mock()
        preview_result = Mock(
            node_config_map={"node1": "config1"},
            config_builder_map={"config1": "builder1"},
            resolution_confidence={"node1": 0.9},
            ambiguous_resolutions=[],
            recommendations=[]
        )
        mock_compiler.preview_resolution.return_value = preview_result
        mock_compiler_class.return_value = mock_compiler

        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline()
            result = pipeline.preview_resolution()

            expected = {
                "node_config_map": {"node1": "config1"},
                "config_builder_map": {"config1": "builder1"},
                "resolution_confidence": {"node1": 0.9},
                "ambiguous_resolutions": [],
                "recommendations": []
            }
            assert result == expected

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_get_last_template(self, mock_compiler_class, mock_session):
        """Test getting last template."""
        mock_compiler = Mock()
        mock_template = Mock()
        mock_compiler.get_last_template.return_value = mock_template
        mock_compiler_class.return_value = mock_compiler

        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline()
            pipeline._last_template = mock_template
            
            result = pipeline.get_last_template()
            assert result == mock_template

    @patch('cursus.pipeline_catalog.core.base_pipeline.CatalogRegistry')
    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_sync_to_registry_success(self, mock_compiler_class, mock_registry_class, mock_session):
        """Test successful registry synchronization."""
        mock_registry = Mock()
        mock_registry.add_or_update_enhanced_node.return_value = True
        mock_registry_class.return_value = mock_registry

        mock_compiler_class.return_value = Mock()

        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline()
            result = pipeline.sync_to_registry()

            assert result is True
            mock_registry.add_or_update_enhanced_node.assert_called_once()

    @patch('cursus.pipeline_catalog.core.base_pipeline.CatalogRegistry')
    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_sync_to_registry_failure(self, mock_compiler_class, mock_registry_class, mock_session):
        """Test failed registry synchronization."""
        mock_registry = Mock()
        mock_registry.add_or_update_enhanced_node.return_value = False
        mock_registry_class.return_value = mock_registry

        mock_compiler_class.return_value = Mock()

        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline()
            result = pipeline.sync_to_registry()

            assert result is False

    @patch('cursus.pipeline_catalog.core.base_pipeline.CatalogRegistry')
    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_sync_to_registry_exception(self, mock_compiler_class, mock_registry_class, mock_session):
        """Test registry synchronization with exception."""
        mock_registry = Mock()
        mock_registry.add_or_update_enhanced_node.side_effect = Exception("Registry error")
        mock_registry_class.return_value = mock_registry

        mock_compiler_class.return_value = Mock()

        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline()
            result = pipeline.sync_to_registry()

            assert result is False

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_create_pipeline_compatibility_method(self, mock_compiler_class, mock_session):
        """Test create_pipeline compatibility method."""
        mock_compiler = Mock()
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.name = "test_pipeline"  # Add name attribute
        mock_report = Mock(avg_confidence=0.95)
        mock_template = Mock()
        
        mock_compiler.compile_with_report.return_value = (mock_pipeline, mock_report)
        mock_compiler.get_last_template.return_value = mock_template
        mock_compiler_class.return_value = mock_compiler

        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            with patch.object(ConcretePipeline, 'sync_to_registry', return_value=True):
                pipeline = ConcretePipeline()
                
                result = pipeline.create_pipeline()
                
                assert len(result) == 4
                pipeline_obj, report, compiler, template = result
                assert pipeline_obj == mock_pipeline
                assert report == mock_report
                assert compiler == mock_compiler
                assert template == mock_template

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_save_execution_document(self, mock_compiler_class, mock_session, tmp_path):
        """Test saving execution document."""
        mock_compiler_class.return_value = Mock()
        
        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline()
            
            document = {"test": "data"}
            output_path = tmp_path / "subdir" / "execution_doc.json"
            
            pipeline.save_execution_document(document, str(output_path))
            
            assert output_path.exists()
            saved_data = json.loads(output_path.read_text())
            assert saved_data == document

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_get_pipeline_config(self, mock_compiler_class, mock_session, temp_config_file):
        """Test getting pipeline configuration."""
        mock_compiler_class.return_value = Mock()
        
        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline(config_path=temp_config_file)
            
            config = pipeline.get_pipeline_config()
            
            assert config["pipeline_name"] == "test_pipeline"
            assert config["parameters"]["param1"] == "value1"
            
            # Ensure it's a copy
            config["new_key"] = "new_value"
            assert "new_key" not in pipeline.config

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_update_pipeline_config(self, mock_compiler_class, mock_session):
        """Test updating pipeline configuration."""
        mock_compiler_class.return_value = Mock()
        
        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline()
            
            updates = {"new_param": "new_value", "existing_param": "updated_value"}
            pipeline.update_pipeline_config(updates)
            
            assert pipeline.config["new_param"] == "new_value"
            assert pipeline.config["existing_param"] == "updated_value"

    @patch('cursus.pipeline_catalog.core.base_pipeline.PipelineDAGCompiler')
    def test_get_dag_info(self, mock_compiler_class, mock_session):
        """Test getting DAG information."""
        mock_compiler_class.return_value = Mock()
        
        with patch('cursus.pipeline_catalog.core.base_pipeline.PipelineSession', return_value=mock_session):
            pipeline = ConcretePipeline()
            
            dag_info = pipeline.get_dag_info()
            
            assert "nodes" in dag_info
            assert "edges" in dag_info
            assert "node_count" in dag_info
            assert "edge_count" in dag_info
            assert "is_valid" in dag_info
            
            assert dag_info["node_count"] == 2  # step1, step2
            assert dag_info["edge_count"] == 1  # step1 -> step2
            assert dag_info["is_valid"] is True
