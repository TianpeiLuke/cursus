"""Unit tests for EnhancedDataFlowManager."""

import unittest
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.cursus.validation.runtime.data.enhanced_data_flow_manager import (
    EnhancedDataFlowManager, DataCompatibilityReport
)


class TestEnhancedDataFlowManager(unittest.TestCase):
    """Test cases for EnhancedDataFlowManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = EnhancedDataFlowManager(self.temp_dir, testing_mode="pre_execution")
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_directories(self):
        """Test that initialization creates required directories."""
        workspace_path = Path(self.temp_dir)
        self.assertTrue(workspace_path.exists())
        self.assertTrue((workspace_path / "synthetic_data").exists())
        self.assertTrue((workspace_path / "s3_data").exists())
        self.assertTrue((workspace_path / "metadata").exists())
        self.assertEqual(self.manager.data_lineage, [])
        self.assertEqual(self.manager.testing_mode, "pre_execution")
    
    def test_init_with_post_execution_mode(self):
        """Test initialization with post-execution testing mode."""
        manager = EnhancedDataFlowManager(self.temp_dir, testing_mode="post_execution")
        self.assertEqual(manager.testing_mode, "post_execution")
    
    def test_setup_step_inputs_pre_execution_mode(self):
        """Test setup_step_inputs in pre-execution mode."""
        # Mock upstream reference with step_name and output_spec
        mock_output_spec = Mock()
        mock_output_spec.logical_name = "processed_data"
        
        mock_upstream_ref = Mock()
        mock_upstream_ref.step_name = "preprocessing"
        mock_upstream_ref.output_spec = mock_output_spec
        
        upstream_outputs = {"input_data": mock_upstream_ref}
        
        result = self.manager.setup_step_inputs("training", upstream_outputs)
        
        # Should return synthetic data path
        self.assertIn("input_data", result)
        expected_path = str(Path(self.temp_dir) / "synthetic_data" / "preprocessing" / "processed_data.csv")
        self.assertEqual(result["input_data"], expected_path)
        
        # Should track data lineage
        self.assertEqual(len(self.manager.data_lineage), 1)
        lineage_entry = self.manager.data_lineage[0]
        self.assertEqual(lineage_entry["to_step"], "training")
        self.assertEqual(lineage_entry["to_input"], "input_data")
        self.assertEqual(lineage_entry["from_step"], "preprocessing")
        self.assertEqual(lineage_entry["from_output"], "processed_data")
    
    def test_setup_step_inputs_post_execution_mode(self):
        """Test setup_step_inputs in post-execution mode."""
        manager = EnhancedDataFlowManager(self.temp_dir, testing_mode="post_execution")
        
        # Mock upstream reference
        mock_output_spec = Mock()
        mock_output_spec.logical_name = "model_output"
        
        mock_upstream_ref = Mock()
        mock_upstream_ref.step_name = "training"
        mock_upstream_ref.output_spec = mock_output_spec
        
        upstream_outputs = {"model": mock_upstream_ref}
        
        result = manager.setup_step_inputs("evaluation", upstream_outputs)
        
        # Should return S3 placeholder path
        self.assertIn("model", result)
        self.assertEqual(result["model"], "s3://placeholder/training/model_output")
    
    def test_setup_step_inputs_direct_path(self):
        """Test setup_step_inputs with direct path reference."""
        upstream_outputs = {"data": "/direct/path/to/data.csv"}
        
        result = self.manager.setup_step_inputs("test_step", upstream_outputs)
        
        self.assertEqual(result["data"], "/direct/path/to/data.csv")
        
        # Should still track lineage
        self.assertEqual(len(self.manager.data_lineage), 1)
    
    def test_resolve_synthetic_path_with_property_reference(self):
        """Test _resolve_synthetic_path with property reference object."""
        mock_output_spec = Mock()
        mock_output_spec.logical_name = "features"
        
        mock_upstream_ref = Mock()
        mock_upstream_ref.step_name = "feature_engineering"
        mock_upstream_ref.output_spec = mock_output_spec
        
        result = self.manager._resolve_synthetic_path("training", "input_features", mock_upstream_ref)
        
        expected_path = str(Path(self.temp_dir) / "synthetic_data" / "feature_engineering" / "features.csv")
        self.assertEqual(result, expected_path)
    
    def test_resolve_synthetic_path_with_direct_path(self):
        """Test _resolve_synthetic_path with direct path."""
        result = self.manager._resolve_synthetic_path("test_step", "input", "/path/to/file.csv")
        self.assertEqual(result, "/path/to/file.csv")
    
    def test_prepare_s3_path_resolution_with_property_reference(self):
        """Test _prepare_s3_path_resolution with property reference."""
        mock_output_spec = Mock()
        mock_output_spec.logical_name = "results"
        
        mock_upstream_ref = Mock()
        mock_upstream_ref.step_name = "analysis"
        mock_upstream_ref.output_spec = mock_output_spec
        
        result = self.manager._prepare_s3_path_resolution("reporting", "analysis_results", mock_upstream_ref)
        
        self.assertEqual(result, "s3://placeholder/analysis/results")
    
    def test_prepare_s3_path_resolution_with_direct_path(self):
        """Test _prepare_s3_path_resolution with direct path."""
        result = self.manager._prepare_s3_path_resolution("test_step", "input", "s3://bucket/path")
        self.assertEqual(result, "s3://bucket/path")
    
    def test_track_data_lineage_entry_with_property_reference(self):
        """Test _track_data_lineage_entry with property reference."""
        mock_output_spec = Mock()
        mock_output_spec.logical_name = "output_data"
        mock_output_spec.property_path = "outputs.data"
        mock_output_spec.data_type = "DataFrame"
        
        mock_upstream_ref = Mock()
        mock_upstream_ref.step_name = "source_step"
        mock_upstream_ref.output_spec = mock_output_spec
        
        self.manager._track_data_lineage_entry("target_step", "input_data", mock_upstream_ref)
        
        self.assertEqual(len(self.manager.data_lineage), 1)
        entry = self.manager.data_lineage[0]
        self.assertEqual(entry["to_step"], "target_step")
        self.assertEqual(entry["to_input"], "input_data")
        self.assertEqual(entry["from_step"], "source_step")
        self.assertEqual(entry["from_output"], "output_data")
        self.assertEqual(entry["property_path"], "outputs.data")
        self.assertEqual(entry["data_type"], "DataFrame")
        self.assertEqual(entry["testing_mode"], "pre_execution")
        self.assertIsInstance(entry["timestamp"], datetime)
    
    def test_track_data_lineage_entry_with_direct_reference(self):
        """Test _track_data_lineage_entry with direct reference."""
        self.manager._track_data_lineage_entry("step1", "input1", "/path/to/data")
        
        self.assertEqual(len(self.manager.data_lineage), 1)
        entry = self.manager.data_lineage[0]
        self.assertEqual(entry["to_step"], "step1")
        self.assertEqual(entry["to_input"], "input1")
        self.assertNotIn("from_step", entry)
        self.assertNotIn("from_output", entry)
    
    def test_create_data_lineage_report(self):
        """Test create_data_lineage_report."""
        # Add some lineage entries
        self.manager._track_data_lineage_entry("step1", "input1", "data1")
        self.manager._track_data_lineage_entry("step2", "input2", "data2")
        
        report = self.manager.create_data_lineage_report()
        
        self.assertEqual(report["total_transfers"], 2)
        self.assertEqual(report["testing_mode"], "pre_execution")
        # The unique_steps calculation includes empty strings from entries without from_step
        # So we have: step1, step2, and empty string = 3 unique values
        self.assertEqual(report["unique_steps"], 3)  # step1, step2, and empty string
        self.assertIn("generated_at", report)
        self.assertEqual(len(report["lineage_entries"]), 2)
    
    def test_validate_data_compatibility_success(self):
        """Test validate_data_compatibility with compatible data."""
        producer_output = {
            "files": {
                "data.csv": {
                    "format": "csv",
                    "size": 1024,
                    "schema": {
                        "columns": ["id", "name", "score"],
                        "column_types": {"id": "int", "name": "str", "score": "float"}
                    }
                }
            }
        }
        
        consumer_input_spec = {
            "required_files": ["data.csv"],
            "file_formats": {"data.csv": "csv"},
            "schemas": {
                "data.csv": {
                    "required_columns": ["id", "name"],
                    "column_types": {"id": "int", "name": "str"}
                }
            }
        }
        
        report = self.manager.validate_data_compatibility(producer_output, consumer_input_spec)
        
        self.assertIsInstance(report, DataCompatibilityReport)
        self.assertTrue(report.compatible)
        self.assertEqual(len(report.issues), 0)
        self.assertEqual(report.data_summary["total_files"], 1)
    
    def test_validate_data_compatibility_missing_file(self):
        """Test validate_data_compatibility with missing required file."""
        producer_output = {
            "files": {
                "data.csv": {"format": "csv"}
            }
        }
        
        consumer_input_spec = {
            "required_files": ["data.csv", "metadata.json"]
        }
        
        report = self.manager.validate_data_compatibility(producer_output, consumer_input_spec)
        
        self.assertFalse(report.compatible)
        self.assertIn("Missing required file: metadata.json", report.issues)
    
    def test_validate_data_compatibility_format_mismatch(self):
        """Test validate_data_compatibility with format mismatch."""
        producer_output = {
            "files": {
                "data.csv": {"format": "json"}
            }
        }
        
        consumer_input_spec = {
            "required_files": ["data.csv"],
            "file_formats": {"data.csv": "csv"}
        }
        
        report = self.manager.validate_data_compatibility(producer_output, consumer_input_spec)
        
        self.assertFalse(report.compatible)
        self.assertIn("Format mismatch for data.csv: expected csv, got json", report.issues)
    
    def test_validate_schemas_missing_columns(self):
        """Test _validate_schemas with missing columns."""
        output = {
            "files": {
                "data.csv": {
                    "schema": {
                        "columns": ["id", "name"],
                        "column_types": {"id": "int", "name": "str"}
                    }
                }
            }
        }
        
        input_spec = {
            "schemas": {
                "data.csv": {
                    "required_columns": ["id", "name", "score"],
                    "column_types": {"id": "int", "name": "str", "score": "float"}
                }
            }
        }
        
        issues = self.manager._validate_schemas(output, input_spec)
        
        self.assertEqual(len(issues), 1)
        self.assertIn("Missing columns in data.csv: {'score'}", issues[0])
    
    def test_validate_schemas_type_mismatch(self):
        """Test _validate_schemas with type mismatch."""
        output = {
            "files": {
                "data.csv": {
                    "schema": {
                        "columns": ["id", "score"],
                        "column_types": {"id": "str", "score": "float"}
                    }
                }
            }
        }
        
        input_spec = {
            "schemas": {
                "data.csv": {
                    "required_columns": ["id", "score"],
                    "column_types": {"id": "int", "score": "float"}
                }
            }
        }
        
        issues = self.manager._validate_schemas(output, input_spec)
        
        self.assertEqual(len(issues), 1)
        self.assertIn("Type mismatch in data.csv.id: expected int, got str", issues[0])
    
    def test_create_data_summary(self):
        """Test _create_data_summary."""
        output = {
            "files": {
                "data.csv": {"size": 1024, "format": "csv"},
                "metadata.json": {"size": 256, "format": "json"}
            }
        }
        
        summary = self.manager._create_data_summary(output)
        
        self.assertEqual(summary["total_files"], 2)
        self.assertEqual(summary["file_sizes"]["data.csv"], 1024)
        self.assertEqual(summary["file_sizes"]["metadata.json"], 256)
        self.assertIn("csv", summary["data_types"])
        self.assertIn("json", summary["data_types"])
    
    @patch('pandas.DataFrame.to_csv')
    def test_generate_synthetic_data_csv(self, mock_to_csv):
        """Test generate_synthetic_data for CSV format."""
        data_spec = {
            "training_data": {
                "format": "csv",
                "rows": 50,
                "columns": ["user_id", "score", "category"]
            }
        }
        
        result = self.manager.generate_synthetic_data("preprocessing", data_spec)
        
        self.assertIn("training_data", result)
        expected_path = str(Path(self.temp_dir) / "synthetic_data" / "preprocessing" / "training_data.csv")
        self.assertEqual(result["training_data"], expected_path)
        
        # Verify CSV generation was called
        mock_to_csv.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_generate_synthetic_data_json(self, mock_json_dump, mock_file_open):
        """Test generate_synthetic_data for JSON format."""
        data_spec = {
            "config": {
                "format": "json",
                "count": 20
            }
        }
        
        result = self.manager.generate_synthetic_data("setup", data_spec)
        
        self.assertIn("config", result)
        expected_path = str(Path(self.temp_dir) / "synthetic_data" / "setup" / "config.json")
        self.assertEqual(result["config"], expected_path)
        
        # Verify JSON generation was called
        mock_file_open.assert_called_once()
        mock_json_dump.assert_called_once()
    
    @patch('pandas.DataFrame')
    @patch('numpy.random.uniform')
    def test_generate_csv_data_with_scores(self, mock_uniform, mock_dataframe):
        """Test _generate_csv_data with score columns."""
        mock_uniform.return_value = [0.1, 0.2, 0.3]
        mock_df = Mock()
        mock_dataframe.return_value = mock_df
        
        file_path = Path(self.temp_dir) / "test.csv"
        spec = {
            "rows": 3,
            "columns": ["id", "accuracy_score", "precision_rate"]
        }
        
        self.manager._generate_csv_data(file_path, spec)
        
        # Verify DataFrame creation and save
        mock_dataframe.assert_called_once()
        mock_df.to_csv.assert_called_once_with(file_path, index=False)
        
        # Verify uniform distribution was used for score columns
        self.assertEqual(mock_uniform.call_count, 2)  # accuracy_score and precision_rate
    
    def test_generate_csv_data_with_ids(self):
        """Test _generate_csv_data with ID columns."""
        file_path = Path(self.temp_dir) / "test.csv"
        spec = {
            "rows": 5,
            "columns": ["user_id", "item_id", "name"]
        }
        
        self.manager._generate_csv_data(file_path, spec)
        
        # Verify file was created
        self.assertTrue(file_path.exists())
        
        # Read and verify content
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 5)
        self.assertIn("user_id", df.columns)
        self.assertIn("item_id", df.columns)
        self.assertIn("name", df.columns)
        
        # ID columns should have sequential values
        self.assertEqual(list(df["user_id"]), [1, 2, 3, 4, 5])
        self.assertEqual(list(df["item_id"]), [1, 2, 3, 4, 5])
    
    def test_generate_json_data(self):
        """Test _generate_json_data."""
        file_path = Path(self.temp_dir) / "test.json"
        spec = {"count": 15}
        
        self.manager._generate_json_data(file_path, spec)
        
        # Verify file was created
        self.assertTrue(file_path.exists())
        
        # Read and verify content
        with open(file_path) as f:
            data = json.load(f)
        
        self.assertIn("generated_at", data)
        self.assertEqual(data["spec"], spec)
        self.assertEqual(data["data"]["count"], 15)
        self.assertEqual(len(data["data"]["items"]), 15)
    
    def test_data_compatibility_report_model(self):
        """Test DataCompatibilityReport Pydantic model."""
        report = DataCompatibilityReport(
            compatible=False,
            issues=["Missing file"],
            warnings=["Performance warning"],
            data_summary={"files": 2}
        )
        
        self.assertFalse(report.compatible)
        self.assertEqual(report.issues, ["Missing file"])
        self.assertEqual(report.warnings, ["Performance warning"])
        self.assertEqual(report.data_summary, {"files": 2})
    
    def test_data_compatibility_report_defaults(self):
        """Test DataCompatibilityReport with default values."""
        report = DataCompatibilityReport(compatible=True)
        
        self.assertTrue(report.compatible)
        self.assertEqual(report.issues, [])
        self.assertEqual(report.warnings, [])
        self.assertEqual(report.data_summary, {})
    
    def test_multiple_step_lineage_tracking(self):
        """Test lineage tracking across multiple steps."""
        # Setup multiple upstream references
        mock_spec1 = Mock()
        mock_spec1.logical_name = "data1"
        mock_ref1 = Mock()
        mock_ref1.step_name = "step1"
        mock_ref1.output_spec = mock_spec1
        
        mock_spec2 = Mock()
        mock_spec2.logical_name = "data2"
        mock_ref2 = Mock()
        mock_ref2.step_name = "step2"
        mock_ref2.output_spec = mock_spec2
        
        upstream_outputs = {
            "input1": mock_ref1,
            "input2": mock_ref2
        }
        
        self.manager.setup_step_inputs("final_step", upstream_outputs)
        
        # Should have 2 lineage entries
        self.assertEqual(len(self.manager.data_lineage), 2)
        
        # Verify lineage report
        report = self.manager.create_data_lineage_report()
        self.assertEqual(report["total_transfers"], 2)
        self.assertEqual(report["unique_steps"], 3)  # step1, step2, final_step


if __name__ == '__main__':
    unittest.main()
