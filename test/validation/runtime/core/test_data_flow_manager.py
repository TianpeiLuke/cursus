"""Unit tests for DataFlowManager."""

import unittest
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil
import json
import os
from pathlib import Path
from datetime import datetime

from src.cursus.validation.runtime.core.data_flow_manager import DataFlowManager
from src.cursus.validation.runtime.utils.error_handling import DataFlowError


class TestDataFlowManager(unittest.TestCase):
    """Test cases for DataFlowManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DataFlowManager(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_directories(self):
        """Test that initialization creates required directories."""
        workspace_path = Path(self.temp_dir)
        self.assertTrue(workspace_path.exists())
        self.assertTrue((workspace_path / "inputs").exists())
        self.assertTrue((workspace_path / "outputs").exists())
        self.assertTrue((workspace_path / "metadata").exists())
        self.assertEqual(self.manager.data_lineage, [])
    
    def test_init_with_existing_directories(self):
        """Test initialization when directories already exist."""
        # Create directories first
        workspace_path = Path(self.temp_dir)
        (workspace_path / "inputs").mkdir(exist_ok=True)
        (workspace_path / "outputs").mkdir(exist_ok=True)
        (workspace_path / "metadata").mkdir(exist_ok=True)
        
        # Initialize manager - should not raise error
        manager = DataFlowManager(self.temp_dir)
        self.assertIsNotNone(manager)
    
    def test_setup_step_inputs_empty_step_name(self):
        """Test setup_step_inputs with empty step name."""
        with self.assertRaises(DataFlowError) as context:
            self.manager.setup_step_inputs("", {})
        
        self.assertIn("Step name cannot be empty", str(context.exception))
    
    def test_setup_step_inputs_no_upstream_outputs(self):
        """Test setup_step_inputs with no upstream outputs."""
        result = self.manager.setup_step_inputs("test_step", {})
        
        # Should create input directory and return path
        self.assertIn("input", result)
        input_path = Path(result["input"])
        self.assertTrue(input_path.exists())
        self.assertEqual(input_path.name, "test_step")
    
    @patch('os.path.exists')
    def test_setup_step_inputs_with_valid_upstream_outputs(self, mock_exists):
        """Test setup_step_inputs with valid upstream outputs."""
        mock_exists.return_value = True
        
        upstream_outputs = {
            "output1": "/path/to/output1",
            "output2": "/path/to/output2"
        }
        
        result = self.manager.setup_step_inputs("test_step", upstream_outputs)
        
        # Should return upstream outputs directly in Phase 1
        self.assertEqual(result, upstream_outputs)
        
        # Verify all paths were checked
        self.assertEqual(mock_exists.call_count, 2)
    
    @patch('os.path.exists')
    def test_setup_step_inputs_with_missing_upstream_output(self, mock_exists):
        """Test setup_step_inputs with missing upstream output."""
        mock_exists.return_value = False
        
        upstream_outputs = {"output1": "/nonexistent/path"}
        
        with self.assertRaises(DataFlowError) as context:
            self.manager.setup_step_inputs("test_step", upstream_outputs)
        
        self.assertIn("Upstream output 'output1' does not exist", str(context.exception))
    
    def test_setup_step_inputs_directory_creation_failure(self):
        """Test setup_step_inputs when directory creation fails."""
        # Create a manager with valid workspace, then mock mkdir to fail
        with patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied")):
            with self.assertRaises(DataFlowError) as context:
                self.manager.setup_step_inputs("test_step", {})
            
            self.assertIn("Failed to create input directory", str(context.exception))
    
    def test_capture_step_outputs_empty_step_name(self):
        """Test capture_step_outputs with empty step name."""
        with self.assertRaises(DataFlowError) as context:
            self.manager.capture_step_outputs("", {})
        
        self.assertIn("Step name cannot be empty", str(context.exception))
    
    def test_capture_step_outputs_no_output_paths(self):
        """Test capture_step_outputs with no output paths."""
        with self.assertRaises(DataFlowError) as context:
            self.manager.capture_step_outputs("test_step", {})
        
        self.assertIn("No output paths provided", str(context.exception))
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_capture_step_outputs_success(self, mock_json_dump, mock_file_open, mock_exists):
        """Test successful capture_step_outputs."""
        mock_exists.return_value = True
        
        output_paths = {
            "output1": "/path/to/output1",
            "output2": "/path/to/output2"
        }
        
        result = self.manager.capture_step_outputs("test_step", output_paths)
        
        # Should return output paths
        self.assertEqual(result, output_paths)
        
        # Verify metadata was saved
        mock_file_open.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # Verify data lineage was updated
        self.assertEqual(len(self.manager.data_lineage), 1)
        self.assertEqual(self.manager.data_lineage[0]["step"], "test_step")
        self.assertEqual(self.manager.data_lineage[0]["outputs"], output_paths)
    
    @patch('os.path.exists')
    def test_capture_step_outputs_missing_output_file(self, mock_exists):
        """Test capture_step_outputs with missing output file."""
        mock_exists.return_value = False
        
        output_paths = {"output1": "/nonexistent/output"}
        
        with self.assertRaises(DataFlowError) as context:
            self.manager.capture_step_outputs("test_step", output_paths)
        
        self.assertIn("Output 'output1' does not exist", str(context.exception))
    
    @patch('os.path.exists')
    @patch('builtins.open', side_effect=OSError("Permission denied"))
    def test_capture_step_outputs_metadata_save_failure(self, mock_file_open, mock_exists):
        """Test capture_step_outputs when metadata save fails."""
        mock_exists.return_value = True
        
        output_paths = {"output1": "/path/to/output1"}
        
        with self.assertRaises(DataFlowError) as context:
            self.manager.capture_step_outputs("test_step", output_paths)
        
        self.assertIn("Failed to save metadata", str(context.exception))
    
    def test_track_data_lineage_empty_step_name(self):
        """Test track_data_lineage with empty step name."""
        with self.assertRaises(DataFlowError) as context:
            self.manager.track_data_lineage("", {}, {})
        
        self.assertIn("Step name cannot be empty for lineage tracking", str(context.exception))
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    @patch('os.path.exists')
    def test_track_data_lineage_new_file(self, mock_exists, mock_json_load, mock_json_dump, mock_file_open):
        """Test track_data_lineage when lineage file doesn't exist."""
        mock_exists.return_value = False  # Lineage file doesn't exist
        
        inputs = {"input1": "/path/to/input1"}
        outputs = {"output1": "/path/to/output1"}
        
        self.manager.track_data_lineage("test_step", inputs, outputs)
        
        # Verify data lineage was updated
        self.assertEqual(len(self.manager.data_lineage), 1)
        lineage_entry = self.manager.data_lineage[0]
        self.assertEqual(lineage_entry["step_name"], "test_step")
        self.assertEqual(lineage_entry["inputs"], inputs)
        self.assertEqual(lineage_entry["outputs"], outputs)
        self.assertIn("timestamp", lineage_entry)
        
        # Verify file operations
        mock_json_dump.assert_called_once()
        # json.load should not be called since file doesn't exist
        mock_json_load.assert_not_called()
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('json.load')
    def test_track_data_lineage_existing_file(self, mock_json_load, mock_json_dump, mock_file_open, mock_path_exists):
        """Test track_data_lineage when lineage file exists."""
        mock_path_exists.return_value = True  # Lineage file exists
        existing_lineage = [{"step_name": "previous_step", "inputs": {}, "outputs": {}}]
        mock_json_load.return_value = existing_lineage
        
        inputs = {"input1": "/path/to/input1"}
        outputs = {"output1": "/path/to/output1"}
        
        self.manager.track_data_lineage("test_step", inputs, outputs)
        
        # Verify data lineage was updated
        self.assertEqual(len(self.manager.data_lineage), 1)
        
        # Verify file operations
        mock_json_load.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # Verify the saved data includes both existing and new entries
        saved_data = mock_json_dump.call_args[0][0]
        self.assertEqual(len(saved_data), 2)  # existing + new entry
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="invalid json")
    @patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    def test_track_data_lineage_json_decode_error(self, mock_json_load, mock_file_open, mock_path_exists):
        """Test track_data_lineage when existing file has invalid JSON."""
        mock_path_exists.return_value = True
        
        with self.assertRaises(DataFlowError) as context:
            self.manager.track_data_lineage("test_step", {}, {})
        
        self.assertIn("Failed to read existing lineage file", str(context.exception))
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', side_effect=IOError("Read error"))
    def test_track_data_lineage_read_error(self, mock_path_exists, mock_file_open):
        """Test track_data_lineage when file read fails."""
        mock_path_exists.return_value = True
        
        with self.assertRaises(DataFlowError) as context:
            self.manager.track_data_lineage("test_step", {}, {})
        
        self.assertIn("Failed to read existing lineage file", str(context.exception))
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump', side_effect=OSError("Write error"))
    @patch('json.load')
    @patch('os.path.exists')
    def test_track_data_lineage_write_error(self, mock_exists, mock_json_load, mock_json_dump, mock_file_open):
        """Test track_data_lineage when file write fails."""
        mock_exists.return_value = False
        
        with self.assertRaises(DataFlowError) as context:
            self.manager.track_data_lineage("test_step", {}, {})
        
        self.assertIn("Failed to save lineage data", str(context.exception))
    
    @patch('datetime.datetime')
    def test_track_data_lineage_timestamp_format(self, mock_datetime):
        """Test that track_data_lineage uses proper timestamp format."""
        mock_now = Mock()
        mock_now.__str__ = Mock(return_value="2023-01-01 12:00:00")
        mock_datetime.now.return_value = mock_now
        
        # Mock file operations to avoid actual file I/O
        with patch('os.path.exists', return_value=False), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'):
            
            self.manager.track_data_lineage("test_step", {}, {})
            
            # Verify timestamp was added
            self.assertEqual(len(self.manager.data_lineage), 1)
            self.assertIn("timestamp", self.manager.data_lineage[0])
    
    def test_data_lineage_integration(self):
        """Test integration between capture_step_outputs and data lineage tracking."""
        # Create a real output file for testing
        output_file = Path(self.temp_dir) / "test_output.txt"
        output_file.write_text("test content")
        
        output_paths = {"output": str(output_file)}
        
        # Capture outputs (which should update data lineage)
        result = self.manager.capture_step_outputs("test_step", output_paths)
        
        # Verify data lineage was updated
        self.assertEqual(len(self.manager.data_lineage), 1)
        self.assertEqual(self.manager.data_lineage[0]["step"], "test_step")
        self.assertEqual(self.manager.data_lineage[0]["outputs"], output_paths)
        
        # Verify metadata file was created
        metadata_file = Path(self.temp_dir) / "metadata" / "test_step_outputs.json"
        self.assertTrue(metadata_file.exists())
        
        # Verify metadata content
        with open(metadata_file) as f:
            metadata = json.load(f)
        self.assertEqual(metadata["step_name"], "test_step")
        self.assertEqual(metadata["output_paths"], output_paths)
        self.assertIn("captured_at", metadata)


if __name__ == '__main__':
    unittest.main()
