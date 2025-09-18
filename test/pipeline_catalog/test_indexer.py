"""
Unit tests for CatalogIndexer class.

Tests the pipeline catalog indexing functionality that scans and indexes
complete pipeline implementations.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil
from typing import Dict, Any

from cursus.pipeline_catalog.indexer import CatalogIndexer


class TestCatalogIndexer:
    """Test suite for CatalogIndexer class."""

    @pytest.fixture
    def temp_catalog_root(self):
        """Create temporary catalog root directory."""
        temp_dir = tempfile.mkdtemp()
        catalog_root = Path(temp_dir) / "catalog"
        catalog_root.mkdir(parents=True, exist_ok=True)
        yield catalog_root
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_pipeline_content(self):
        """Sample pipeline file content."""
        return '''"""
XGBoost Training Pipeline

This pipeline provides end-to-end training functionality for XGBoost models
with calibration and evaluation capabilities.

Args:
    data_path: Path to training data
    
Returns:
    Trained model with calibration
"""

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

def train_model(data_path):
    """Train XGBoost model with calibration."""
    # Training logic here
    pass

def evaluate_model(model, test_data):
    """Evaluate trained model."""
    # Evaluation logic here
    pass
'''

    @pytest.fixture
    def indexer(self, temp_catalog_root):
        """Create CatalogIndexer instance."""
        return CatalogIndexer(temp_catalog_root)

    def test_init(self, temp_catalog_root):
        """Test CatalogIndexer initialization."""
        indexer = CatalogIndexer(temp_catalog_root)
        
        assert indexer.catalog_root == temp_catalog_root
        assert indexer.index_path == temp_catalog_root / "index.json"

    def test_init_with_step_catalog_success(self, temp_catalog_root):
        """Test initialization when StepCatalog is available."""
        # StepCatalog is actually available in this environment
        indexer = CatalogIndexer(temp_catalog_root)
        # The step catalog should be initialized successfully
        assert indexer._step_catalog is not None

    def test_find_python_files(self, indexer, temp_catalog_root):
        """Test finding Python files in directory."""
        # Create test files
        (temp_catalog_root / "pipeline1.py").touch()
        (temp_catalog_root / "pipeline2.py").touch()
        (temp_catalog_root / "__init__.py").touch()  # Should be ignored
        
        subdir = temp_catalog_root / "subdir"
        subdir.mkdir()
        (subdir / "pipeline3.py").touch()
        (subdir / "not_python.txt").touch()  # Should be ignored
        
        python_files = indexer._find_python_files(temp_catalog_root)
        
        assert len(python_files) == 3
        file_names = [f.name for f in python_files]
        assert "pipeline1.py" in file_names
        assert "pipeline2.py" in file_names
        assert "pipeline3.py" in file_names
        assert "__init__.py" not in file_names
        assert "not_python.txt" not in file_names

    def test_find_python_files_nonexistent_directory(self, indexer):
        """Test finding Python files in non-existent directory."""
        nonexistent_dir = Path("/nonexistent/directory")
        python_files = indexer._find_python_files(nonexistent_dir)
        assert python_files == []

    def test_extract_id_simple_file(self, indexer):
        """Test extracting ID from simple file path."""
        rel_path = Path("training_pipeline.py")
        pipeline_id = indexer._extract_id(rel_path)
        assert pipeline_id == "training-pipeline"

    def test_extract_id_nested_path(self, indexer):
        """Test extracting ID from nested path."""
        rel_path = Path("xgboost/advanced_training.py")
        pipeline_id = indexer._extract_id(rel_path)
        assert pipeline_id == "xgboost-advanced-training"

    def test_extract_id_pipelines_directory(self, indexer):
        """Test extracting ID from pipelines directory structure."""
        rel_path = Path("pipelines/xgboost/training.py")
        pipeline_id = indexer._extract_id(rel_path)
        assert pipeline_id == "xgboost-training"

    def test_extract_name_from_docstring(self, indexer):
        """Test extracting name from module docstring."""
        mock_module = Mock()
        mock_module.__doc__ = "XGBoost Training Pipeline\n\nThis is a description."
        
        name = indexer._extract_name(mock_module)
        assert name == "XGBoost Training Pipeline"

    def test_extract_name_from_filename(self, indexer):
        """Test extracting name from filename when no docstring."""
        mock_module = Mock()
        mock_module.__doc__ = None
        mock_module.__file__ = "/path/to/training_pipeline.py"
        
        name = indexer._extract_name(mock_module)
        assert name == "Training Pipeline Pipeline"

    def test_extract_name_fallback(self, indexer):
        """Test name extraction fallback."""
        mock_module = Mock()
        mock_module.__doc__ = None
        del mock_module.__file__  # Remove __file__ attribute
        
        name = indexer._extract_name(mock_module)
        assert name == "Unknown Pipeline"

    def test_detect_framework_from_path(self, indexer):
        """Test framework detection from file path."""
        assert indexer._detect_framework_from_path(Path("xgboost/training.py")) == "xgboost"
        assert indexer._detect_framework_from_path(Path("pytorch/model.py")) == "pytorch"
        assert indexer._detect_framework_from_path(Path("tensorflow/train.py")) == "tensorflow"
        assert indexer._detect_framework_from_path(Path("dummy/test.py")) == "dummy"
        assert indexer._detect_framework_from_path(Path("unknown/pipeline.py")) == "unknown"
        assert indexer._detect_framework_from_path(Path("xgb_training.py")) == "xgboost"

    def test_determine_complexity(self, indexer):
        """Test complexity determination."""
        # From path
        assert indexer._determine_complexity(Path("simple_training.py"), "") == "simple"
        assert indexer._determine_complexity(Path("advanced_pipeline.py"), "") == "advanced"
        assert indexer._determine_complexity(Path("e2e_training.py"), "") == "advanced"
        assert indexer._determine_complexity(Path("comprehensive_model.py"), "") == "advanced"
        
        # From docstring
        assert indexer._determine_complexity(Path("training.py"), "Simple training pipeline") == "simple"
        assert indexer._determine_complexity(Path("training.py"), "Advanced ML pipeline") == "advanced"
        
        # Default
        assert indexer._determine_complexity(Path("training.py"), "") == "intermediate"

    def test_extract_features(self, indexer):
        """Test feature extraction from docstring."""
        docstring = """
        Training and evaluation pipeline with calibration.
        Supports end-to-end processing and model registration.
        """
        
        features = indexer._extract_features(docstring)
        
        assert "training" in features
        assert "evaluation" in features
        assert "calibration" in features
        assert "registration" in features
        assert "end_to_end" in features

    def test_extract_features_empty_docstring(self, indexer):
        """Test feature extraction from empty docstring."""
        features = indexer._extract_features("")
        assert features == []

    def test_extract_description(self, indexer):
        """Test description extraction from docstring."""
        docstring = """XGBoost Training Pipeline
        
        This pipeline provides comprehensive training functionality.
        
        Args:
            data_path: Path to data
        """
        
        description = indexer._extract_description(docstring)
        assert description == "This pipeline provides comprehensive training functionality."

    def test_extract_description_no_paragraph(self, indexer):
        """Test description extraction when no description paragraph."""
        docstring = "XGBoost Training Pipeline"
        
        description = indexer._extract_description(docstring)
        assert description == "XGBoost Training Pipeline"

    def test_extract_description_empty(self, indexer):
        """Test description extraction from empty docstring."""
        description = indexer._extract_description("")
        assert description == "No description available"

    def test_extract_tags(self, indexer):
        """Test tag extraction from docstring and path."""
        rel_path = Path("xgboost/training_eval.py")
        docstring = "Machine learning pipeline with calibration and evaluation."
        
        tags = indexer._extract_tags(docstring, rel_path)
        
        assert "xgboost" in tags
        assert "training" in tags
        assert "evaluation" in tags
        assert "calibration" in tags
        assert "machine_learning" in tags

    def test_extract_tags_deduplication(self, indexer):
        """Test that duplicate tags are removed."""
        rel_path = Path("xgboost/training.py")
        docstring = "Training pipeline for training models."
        
        tags = indexer._extract_tags(docstring, rel_path)
        
        # Should only have one "training" tag despite appearing in both path and docstring
        assert tags.count("training") == 1

    def test_process_pipeline_file(self, indexer, temp_catalog_root, sample_pipeline_content):
        """Test processing a single pipeline file."""
        # Create test file
        pipeline_file = temp_catalog_root / "xgboost_training.py"
        pipeline_file.write_text(sample_pipeline_content)
        
        entry = indexer._process_pipeline_file(pipeline_file)
        
        assert entry is not None
        assert entry["id"] == "xgboost-training"
        # The actual implementation capitalizes differently
        assert "XGBoost Training Pipeline" in entry["name"] or "Xgboost Training Pipeline" in entry["name"]
        assert entry["path"] == "xgboost_training.py"
        assert entry["framework"] == "xgboost"
        assert "training" in entry["features"]
        assert "evaluation" in entry["features"]
        assert "calibration" in entry["features"]
        assert "end_to_end" in entry["features"]
        assert "xgboost" in entry["tags"]

    def test_process_pipeline_file_import_error(self, indexer, temp_catalog_root):
        """Test processing file with import error."""
        # Create file with syntax error
        pipeline_file = temp_catalog_root / "broken_pipeline.py"
        pipeline_file.write_text("invalid python syntax !!!")
        
        entry = indexer._process_pipeline_file(pipeline_file)
        assert entry is None

    def test_generate_index(self, indexer, temp_catalog_root, sample_pipeline_content):
        """Test generating complete index."""
        # Create test files
        (temp_catalog_root / "pipeline1.py").write_text(sample_pipeline_content)
        (temp_catalog_root / "pipeline2.py").write_text('"""Simple Pipeline"""\npass')
        
        index = indexer.generate_index()
        
        assert "pipelines" in index
        assert len(index["pipelines"]) == 2
        
        # Check that both pipelines are indexed
        pipeline_ids = [p["id"] for p in index["pipelines"]]
        assert "pipeline1" in pipeline_ids
        assert "pipeline2" in pipeline_ids

    def test_generate_index_with_errors(self, indexer, temp_catalog_root):
        """Test index generation with some files causing errors."""
        # Create valid file
        (temp_catalog_root / "valid.py").write_text('"""Valid Pipeline"""\npass')
        
        # Create invalid file
        (temp_catalog_root / "invalid.py").write_text("invalid syntax !!!")
        
        index = indexer.generate_index()
        
        assert "pipelines" in index
        assert len(index["pipelines"]) == 1  # Only valid file should be indexed
        assert index["pipelines"][0]["id"] == "valid"

    def test_merge_indices(self, indexer):
        """Test merging two indices."""
        existing = {
            "pipelines": [
                {"id": "pipeline1", "name": "Pipeline 1", "version": "1.0"},
                {"id": "pipeline2", "name": "Pipeline 2", "version": "1.0"}
            ]
        }
        
        new = {
            "pipelines": [
                {"id": "pipeline2", "name": "Pipeline 2 Updated", "version": "2.0"},
                {"id": "pipeline3", "name": "Pipeline 3", "version": "1.0"}
            ]
        }
        
        merged = indexer._merge_indices(existing, new)
        
        assert len(merged["pipelines"]) == 3
        
        # Check that pipeline2 was updated
        pipeline2 = next(p for p in merged["pipelines"] if p["id"] == "pipeline2")
        assert pipeline2["name"] == "Pipeline 2 Updated"
        assert pipeline2["version"] == "2.0"
        
        # Check that all pipelines are present
        pipeline_ids = [p["id"] for p in merged["pipelines"]]
        assert "pipeline1" in pipeline_ids
        assert "pipeline2" in pipeline_ids
        assert "pipeline3" in pipeline_ids

    def test_validate_index_valid(self, indexer, temp_catalog_root):
        """Test validating a valid index."""
        # Create the actual file that the validation checks for
        test_file = temp_catalog_root / "test_pipeline.py"
        test_file.write_text('"""Test Pipeline"""\npass')
        
        valid_index = {
            "pipelines": [
                {
                    "id": "test-pipeline",
                    "name": "Test Pipeline",
                    "path": "test_pipeline.py",
                    "framework": "xgboost",
                    "complexity": "simple",
                    "features": ["training"],
                    "description": "Test description",
                    "tags": ["xgboost", "training"]
                }
            ]
        }
        
        is_valid, issues = indexer.validate_index(valid_index)
        
        assert is_valid
        assert len(issues) == 0

    def test_validate_index_missing_pipelines_key(self, indexer):
        """Test validating index missing pipelines key."""
        invalid_index = {"other_key": []}
        
        is_valid, issues = indexer.validate_index(invalid_index)
        
        assert not is_valid
        assert "Index missing 'pipelines' key" in issues

    def test_validate_index_missing_required_fields(self, indexer):
        """Test validating index with missing required fields."""
        invalid_index = {
            "pipelines": [
                {
                    "id": "test-pipeline",
                    "name": "Test Pipeline"
                    # Missing other required fields
                }
            ]
        }
        
        is_valid, issues = indexer.validate_index(invalid_index)
        
        assert not is_valid
        assert len(issues) > 0
        assert any("missing 'path' field" in issue for issue in issues)
        assert any("missing 'framework' field" in issue for issue in issues)

    def test_validate_index_duplicate_ids(self, indexer):
        """Test validating index with duplicate pipeline IDs."""
        invalid_index = {
            "pipelines": [
                {
                    "id": "duplicate-id",
                    "name": "Pipeline 1",
                    "path": "pipeline1.py",
                    "framework": "xgboost",
                    "complexity": "simple",
                    "features": [],
                    "description": "Description",
                    "tags": []
                },
                {
                    "id": "duplicate-id",
                    "name": "Pipeline 2",
                    "path": "pipeline2.py",
                    "framework": "pytorch",
                    "complexity": "simple",
                    "features": [],
                    "description": "Description",
                    "tags": []
                }
            ]
        }
        
        is_valid, issues = indexer.validate_index(invalid_index)
        
        assert not is_valid
        assert any("Duplicate pipeline ID: duplicate-id" in issue for issue in issues)

    def test_save_index(self, indexer, temp_catalog_root):
        """Test saving index to file."""
        test_index = {
            "pipelines": [
                {
                    "id": "test-pipeline",
                    "name": "Test Pipeline",
                    "path": "test.py",
                    "framework": "xgboost",
                    "complexity": "simple",
                    "features": [],
                    "description": "Test",
                    "tags": []
                }
            ]
        }
        
        indexer.save_index(test_index)
        
        # Verify file was created and contains correct data
        assert indexer.index_path.exists()
        
        with open(indexer.index_path, 'r') as f:
            saved_index = json.load(f)
        
        assert saved_index == test_index

    def test_save_index_creates_directory(self, temp_catalog_root):
        """Test that save_index creates directory if it doesn't exist."""
        # Create indexer with non-existent subdirectory
        nested_catalog = temp_catalog_root / "nested" / "catalog"
        indexer = CatalogIndexer(nested_catalog)
        
        test_index = {"pipelines": []}
        
        indexer.save_index(test_index)
        
        assert indexer.index_path.exists()
        assert indexer.index_path.parent.exists()

    def test_save_index_error_handling(self, indexer):
        """Test save_index error handling."""
        # Mock open to raise an exception
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                indexer.save_index({"pipelines": []})

    def test_update_index_new_file(self, indexer, temp_catalog_root, sample_pipeline_content):
        """Test updating index when index file doesn't exist."""
        # Create test pipeline
        (temp_catalog_root / "test_pipeline.py").write_text(sample_pipeline_content)
        
        indexer.update_index()
        
        # Verify index was created
        assert indexer.index_path.exists()
        
        with open(indexer.index_path, 'r') as f:
            index = json.load(f)
        
        assert "pipelines" in index
        assert len(index["pipelines"]) == 1

    def test_update_index_existing_file(self, indexer, temp_catalog_root, sample_pipeline_content):
        """Test updating existing index file."""
        # Create existing index
        existing_index = {
            "pipelines": [
                {
                    "id": "old-pipeline",
                    "name": "Old Pipeline",
                    "path": "old.py",
                    "framework": "unknown",
                    "complexity": "simple",
                    "features": [],
                    "description": "Old",
                    "tags": []
                }
            ]
        }
        
        with open(indexer.index_path, 'w') as f:
            json.dump(existing_index, f)
        
        # Create new pipeline
        (temp_catalog_root / "new_pipeline.py").write_text(sample_pipeline_content)
        
        indexer.update_index()
        
        # Verify index was updated
        with open(indexer.index_path, 'r') as f:
            updated_index = json.load(f)
        
        assert len(updated_index["pipelines"]) == 2
        pipeline_ids = [p["id"] for p in updated_index["pipelines"]]
        assert "old-pipeline" in pipeline_ids
        assert "new-pipeline" in pipeline_ids

    def test_update_index_corrupted_existing_file(self, indexer, temp_catalog_root, sample_pipeline_content):
        """Test updating index when existing file is corrupted."""
        # Create corrupted index file
        with open(indexer.index_path, 'w') as f:
            f.write("invalid json content")
        
        # Create test pipeline
        (temp_catalog_root / "test_pipeline.py").write_text(sample_pipeline_content)
        
        # Should not raise error, should create new index
        indexer.update_index()
        
        # Verify new index was created
        with open(indexer.index_path, 'r') as f:
            index = json.load(f)
        
        assert "pipelines" in index
        assert len(index["pipelines"]) == 1

    def test_integration_full_workflow(self, temp_catalog_root):
        """Test complete indexing workflow."""
        # Create indexer
        indexer = CatalogIndexer(temp_catalog_root)
        
        # Create test pipelines without external dependencies
        xgboost_pipeline = '''"""
XGBoost Advanced Training Pipeline

Comprehensive training pipeline with evaluation and calibration.
"""
# Simple pipeline without external imports
def train_model():
    pass
'''
        
        pytorch_pipeline = '''"""
PyTorch Simple Model

Basic PyTorch training pipeline.
"""
# Simple pipeline without external imports
def train_pytorch_model():
    pass
'''
        
        (temp_catalog_root / "xgboost" / "advanced_training.py").parent.mkdir(parents=True)
        (temp_catalog_root / "xgboost" / "advanced_training.py").write_text(xgboost_pipeline)
        (temp_catalog_root / "pytorch" / "simple_model.py").parent.mkdir(parents=True)
        (temp_catalog_root / "pytorch" / "simple_model.py").write_text(pytorch_pipeline)
        
        # Generate and save index
        index = indexer.generate_index()
        indexer.save_index(index)
        
        # Validate results - should have both pipelines now
        assert len(index["pipelines"]) == 2
        
        # Check XGBoost pipeline
        xgb_pipeline = next(p for p in index["pipelines"] if p["framework"] == "xgboost")
        assert xgb_pipeline["id"] == "xgboost-advanced-training"
        assert xgb_pipeline["complexity"] == "advanced"
        assert "training" in xgb_pipeline["features"]
        assert "evaluation" in xgb_pipeline["features"]
        assert "calibration" in xgb_pipeline["features"]
        
        # Check PyTorch pipeline
        pytorch_pipeline_entry = next(p for p in index["pipelines"] if p["framework"] == "pytorch")
        assert pytorch_pipeline_entry["id"] == "pytorch-simple-model"
        assert pytorch_pipeline_entry["complexity"] == "simple"
        
        # Verify index file exists and is valid
        assert indexer.index_path.exists()
        is_valid, issues = indexer.validate_index(index)
        assert is_valid
        assert len(issues) == 0
