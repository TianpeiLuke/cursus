"""
Tests for the pipeline catalog indexer.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest import mock
from typing import Dict, Any

import pytest

from src.cursus.pipeline_catalog.indexer import CatalogIndexer


class TestCatalogIndexer:
    """Tests for the CatalogIndexer class."""

    @pytest.fixture
    def temp_catalog_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create a basic directory structure
            frameworks_dir = temp_dir / "frameworks"
            components_dir = temp_dir / "components"
            frameworks_dir.mkdir()
            components_dir.mkdir()
            
            # Create subdirectories for frameworks
            (frameworks_dir / "xgboost").mkdir()
            (frameworks_dir / "xgboost" / "training").mkdir()
            (frameworks_dir / "pytorch").mkdir()
            
            yield temp_dir
        finally:
            # Clean up
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_pipeline_file(self, temp_catalog_dir):
        """Create a mock pipeline file for testing."""
        file_path = temp_catalog_dir / "frameworks" / "xgboost" / "test_pipeline.py"
        
        with open(file_path, "w") as f:
            f.write('''"""
XGBoost Test Pipeline

This is a test pipeline that includes training and evaluation.
"""

from typing import Dict, Any, Tuple
from src.cursus.api.dag.base_dag import PipelineDAG


def create_dag() -> PipelineDAG:
    """Create a test DAG."""
    dag = PipelineDAG()
    dag.add_node("TestNode")
    return dag


def create_pipeline(config_path, session, role) -> Tuple[Any, Dict[str, Any]]:
    """Create a test pipeline."""
    return {}, {}


def fill_execution_document(pipeline, document, compiler):
    """Fill execution document."""
    return document
''')
        
        return file_path

    @pytest.fixture
    def mock_index(self) -> Dict[str, Any]:
        """Create a mock index dictionary."""
        return {
            "pipelines": [
                {
                    "id": "xgboost-test",
                    "name": "XGBoost Test Pipeline",
                    "path": "frameworks/xgboost/test_pipeline.py",
                    "framework": "xgboost",
                    "complexity": "simple",
                    "features": ["training", "evaluation"],
                    "description": "This is a test pipeline that includes training and evaluation.",
                    "tags": ["xgboost", "training", "evaluation", "beginner"]
                }
            ]
        }

    def test_init(self, temp_catalog_dir):
        """Test initializing the indexer."""
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        assert indexer.catalog_root == temp_catalog_dir
        assert indexer.index_path == temp_catalog_dir / "index.json"

    def test_find_python_files(self, temp_catalog_dir, mock_pipeline_file):
        """Test finding Python files in a directory."""
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        files = indexer._find_python_files(temp_catalog_dir / "frameworks")
        
        assert len(files) == 1
        assert files[0] == mock_pipeline_file

    @mock.patch("importlib.util.spec_from_file_location")
    @mock.patch("inspect.getdoc")
    def test_process_pipeline_file(self, mock_getdoc, mock_spec_from_file_location, temp_catalog_dir, mock_pipeline_file):
        """Test processing a pipeline file."""
        # Setup mocks
        mock_getdoc.return_value = "XGBoost Test Pipeline\n\nThis is a test pipeline that includes training and evaluation."
        
        # Mock the module import process
        mock_module = mock.MagicMock()
        mock_module.__doc__ = "XGBoost Test Pipeline\n\nThis is a test pipeline that includes training and evaluation."
        mock_module.create_dag = mock.MagicMock()
        mock_module.create_pipeline = mock.MagicMock()
        
        mock_spec = mock.MagicMock()
        mock_spec.loader = mock.MagicMock()
        mock_spec_from_file_location.return_value = mock_spec
        
        # Create the indexer
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        # Test with mock module loading
        with mock.patch("importlib.util.module_from_spec", return_value=mock_module):
            entry = indexer._process_pipeline_file(mock_pipeline_file)
        
        # Verify the pipeline entry
        assert entry is not None
        assert entry["id"] == "xgboost-test-pipeline"
        assert entry["name"] == "XGBoost Test Pipeline"
        assert entry["framework"] == "xgboost"
        assert "training" in entry["features"]
        assert "evaluation" in entry["features"]
        assert "xgboost" in entry["tags"]

    def test_extract_id(self, temp_catalog_dir):
        """Test extracting a pipeline ID from a path."""
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        # Test with framework path
        rel_path = Path("frameworks/xgboost/training/with_calibration.py")
        assert indexer._extract_id(rel_path) == "xgboost-with-calibration"
        
        # Test with component path
        rel_path = Path("components/cradle_dataload.py")
        assert indexer._extract_id(rel_path) == "cradle-dataload"

    def test_extract_name(self, temp_catalog_dir):
        """Test extracting a pipeline name from a module."""
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        # Create a mock module
        mock_module = mock.MagicMock()
        mock_module.__doc__ = "XGBoost Test Pipeline\n\nThis is a test."
        
        assert indexer._extract_name(mock_module) == "XGBoost Test Pipeline"
        
        # Test with no docstring
        mock_module.__doc__ = None
        mock_module.__file__ = "/path/to/test_pipeline.py"
        
        assert indexer._extract_name(mock_module) == "Test Pipeline Pipeline"

    def test_extract_features(self, temp_catalog_dir):
        """Test extracting features from a docstring."""
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        # Test with training and evaluation
        docstring = "This pipeline includes training and evaluation steps."
        features = indexer._extract_features(docstring)
        assert "training" in features
        assert "evaluation" in features
        
        # Test with calibration
        docstring = "This pipeline calibrates the model probabilities."
        features = indexer._extract_features(docstring)
        assert "calibration" in features

    def test_determine_complexity(self, temp_catalog_dir):
        """Test determining the pipeline complexity."""
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        # Test with end-to-end path
        rel_path = Path("frameworks/xgboost/end_to_end/complete_e2e.py")
        assert indexer._determine_complexity(rel_path, "") == "advanced"
        
        # Test with simple in path
        rel_path = Path("frameworks/xgboost/simple.py")
        assert indexer._determine_complexity(rel_path, "") == "simple"
        
        # Test with complexity in docstring
        rel_path = Path("frameworks/xgboost/training/with_calibration.py")
        assert indexer._determine_complexity(rel_path, "This is an advanced pipeline.") == "advanced"

    def test_extract_tags(self, temp_catalog_dir):
        """Test extracting tags from a docstring and path."""
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        # Test with xgboost path and training docstring
        rel_path = Path("frameworks/xgboost/training/with_evaluation.py")
        docstring = "This pipeline includes training and evaluation steps."
        tags = indexer._extract_tags(docstring, rel_path)
        
        assert "xgboost" in tags
        assert "training" in tags
        assert "evaluation" in tags

    def test_extract_description(self, temp_catalog_dir):
        """Test extracting a description from a docstring."""
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        # Test with multi-paragraph docstring
        docstring = "XGBoost Test Pipeline\n\nThis is a test pipeline that includes training and evaluation."
        assert indexer._extract_description(docstring) == "This is a test pipeline that includes training and evaluation."
        
        # Test with single paragraph
        docstring = "XGBoost Test Pipeline"
        assert indexer._extract_description(docstring) == "XGBoost Test Pipeline"
        
        # Test with empty docstring
        docstring = ""
        assert indexer._extract_description(docstring) == "No description available"

    def test_merge_indices(self, temp_catalog_dir, mock_index):
        """Test merging two indices."""
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        # Create a new index with different pipeline
        new_index = {
            "pipelines": [
                {
                    "id": "pytorch-test",
                    "name": "PyTorch Test Pipeline",
                    "path": "frameworks/pytorch/test_pipeline.py",
                    "framework": "pytorch",
                    "complexity": "simple",
                    "features": ["training"],
                    "description": "This is a PyTorch test pipeline.",
                    "tags": ["pytorch", "training"]
                }
            ]
        }
        
        # Merge indices
        merged = indexer._merge_indices(mock_index, new_index)
        
        # Verify merged index
        assert len(merged["pipelines"]) == 2
        ids = [p["id"] for p in merged["pipelines"]]
        assert "xgboost-test" in ids
        assert "pytorch-test" in ids
        
        # Test with overlapping IDs (new should override existing)
        new_index = {
            "pipelines": [
                {
                    "id": "xgboost-test",
                    "name": "Updated XGBoost Pipeline",
                    "path": "frameworks/xgboost/test_pipeline.py",
                    "framework": "xgboost",
                    "complexity": "advanced",
                    "features": ["training", "evaluation", "registration"],
                    "description": "Updated description.",
                    "tags": ["xgboost", "advanced"]
                }
            ]
        }
        
        merged = indexer._merge_indices(mock_index, new_index)
        
        # Verify merged index
        assert len(merged["pipelines"]) == 1
        assert merged["pipelines"][0]["name"] == "Updated XGBoost Pipeline"
        assert merged["pipelines"][0]["complexity"] == "advanced"

    def test_validate_index(self, temp_catalog_dir, mock_index):
        """Test validating an index."""
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        # Test with valid index
        with mock.patch.object(indexer, 'catalog_root'):
            # Mock the catalog_root / pipeline["path"] check to return True
            indexer.catalog_root.__truediv__.return_value.exists.return_value = True
            is_valid, issues = indexer.validate_index(mock_index)
            
        assert is_valid
        assert len(issues) == 0
        
        # Test with missing required field
        invalid_index = {
            "pipelines": [
                {
                    "id": "xgboost-test",
                    "name": "XGBoost Test Pipeline",
                    # Missing path
                    "framework": "xgboost",
                    "complexity": "simple",
                    "features": ["training", "evaluation"],
                    "description": "This is a test pipeline.",
                    "tags": ["xgboost", "training", "evaluation"]
                }
            ]
        }
        
        is_valid, issues = indexer.validate_index(invalid_index)
        
        assert not is_valid
        assert len(issues) == 1
        assert "missing 'path' field" in issues[0]
        
        # Test with duplicate IDs
        duplicate_ids_index = {
            "pipelines": [
                {
                    "id": "xgboost-test",
                    "name": "XGBoost Test Pipeline 1",
                    "path": "frameworks/xgboost/test_pipeline1.py",
                    "framework": "xgboost",
                    "complexity": "simple",
                    "features": ["training"],
                    "description": "Test pipeline 1",
                    "tags": ["xgboost"]
                },
                {
                    "id": "xgboost-test",  # Duplicate ID
                    "name": "XGBoost Test Pipeline 2",
                    "path": "frameworks/xgboost/test_pipeline2.py",
                    "framework": "xgboost",
                    "complexity": "simple",
                    "features": ["training"],
                    "description": "Test pipeline 2",
                    "tags": ["xgboost"]
                }
            ]
        }
        
        with mock.patch.object(indexer, 'catalog_root'):
            # Mock the catalog_root / pipeline["path"] check to return True
            indexer.catalog_root.__truediv__.return_value.exists.return_value = True
            is_valid, issues = indexer.validate_index(duplicate_ids_index)
            
        assert not is_valid
        assert len(issues) == 1
        assert "Duplicate pipeline ID" in issues[0]

    @mock.patch.object(CatalogIndexer, 'generate_index')
    def test_update_index(self, mock_generate_index, temp_catalog_dir, mock_index):
        """Test updating the index."""
        # Setup
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        mock_generate_index.return_value = {"pipelines": [{"id": "new-pipeline", "name": "New Pipeline"}]}
        
        # Create an existing index file
        with open(indexer.index_path, "w") as f:
            json.dump(mock_index, f)
            
        # Mock _merge_indices and validate_index
        with mock.patch.object(indexer, '_merge_indices') as mock_merge:
            with mock.patch.object(indexer, 'validate_index', return_value=(True, [])):
                with mock.patch.object(indexer, 'save_index') as mock_save:
                    indexer.update_index()
                    
                    # Verify _merge_indices was called with correct arguments
                    mock_merge.assert_called_once()
                    args = mock_merge.call_args[0]
                    assert args[0] == mock_index
                    assert args[1] == {"pipelines": [{"id": "new-pipeline", "name": "New Pipeline"}]}
                    
                    # Verify save_index was called
                    mock_save.assert_called_once()

    def test_generate_index_integration(self, temp_catalog_dir, mock_pipeline_file):
        """Integration test for generating an index."""
        indexer = CatalogIndexer(catalog_root=temp_catalog_dir)
        
        # Mock the module loading and inspection
        with mock.patch('importlib.util.spec_from_file_location') as mock_spec:
            with mock.patch('importlib.util.module_from_spec') as mock_module_from_spec:
                # Setup the mocks
                mock_module = mock.MagicMock()
                mock_module.__doc__ = "XGBoost Test Pipeline\n\nThis is a test pipeline that includes training and evaluation."
                mock_module.create_dag = mock.MagicMock()
                mock_module.create_pipeline = mock.MagicMock()
                mock_module_from_spec.return_value = mock_module
                
                mock_spec_obj = mock.MagicMock()
                mock_spec_obj.loader = mock.MagicMock()
                mock_spec.return_value = mock_spec_obj
                
                # Call generate_index
                index = indexer.generate_index()
                
                # Verify the result
                assert "pipelines" in index
                assert len(index["pipelines"]) == 1
                assert index["pipelines"][0]["name"] == "XGBoost Test Pipeline"
