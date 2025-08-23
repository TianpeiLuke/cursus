"""Unit tests for LocalDataManager."""

import unittest
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil
import yaml
import os
from pathlib import Path

from src.cursus.validation.runtime.data.local_data_manager import LocalDataManager


class TestLocalDataManager(unittest.TestCase):
    """Test cases for LocalDataManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = LocalDataManager(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_directories(self):
        """Test that initialization creates required directories."""
        workspace_path = Path(self.temp_dir)
        local_data_path = workspace_path / "local_data"
        
        self.assertTrue(workspace_path.exists())
        self.assertTrue(local_data_path.exists())
        self.assertEqual(self.manager.workspace_dir, workspace_path)
        self.assertEqual(self.manager.local_data_dir, local_data_path)
    
    def test_init_creates_default_manifest(self):
        """Test that initialization creates default manifest file."""
        manifest_path = self.manager.manifest_path
        self.assertTrue(manifest_path.exists())
        
        # Verify manifest content
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        self.assertIn("version", manifest)
        self.assertIn("description", manifest)
        self.assertIn("scripts", manifest)
        self.assertIn("example_script", manifest["scripts"])
    
    def test_init_with_existing_manifest(self):
        """Test initialization when manifest already exists."""
        # Create a custom manifest
        custom_manifest = {
            "version": "2.0",
            "scripts": {
                "custom_script": {
                    "data": {"path": "custom_script/data.csv", "format": "csv"}
                }
            }
        }
        
        with open(self.manager.manifest_path, 'w') as f:
            yaml.dump(custom_manifest, f)
        
        # Create new manager with existing manifest
        manager = LocalDataManager(self.temp_dir)
        manifest = manager._load_manifest()
        
        self.assertEqual(manifest["version"], "2.0")
        self.assertIn("custom_script", manifest["scripts"])
    
    def test_get_data_for_script_existing_script(self):
        """Test get_data_for_script with existing script and files."""
        # Create test data file
        script_dir = self.manager.local_data_dir / "test_script"
        script_dir.mkdir(exist_ok=True)
        test_file = script_dir / "input.csv"
        test_file.write_text("test,data\n1,2\n")
        
        # Update manifest
        manifest = {
            "scripts": {
                "test_script": {
                    "input_data": {
                        "path": "test_script/input.csv",
                        "format": "csv",
                        "description": "Test input data"
                    }
                }
            }
        }
        self.manager._save_manifest(manifest)
        
        result = self.manager.get_data_for_script("test_script")
        
        self.assertIsNotNone(result)
        self.assertIn("input_data", result)
        self.assertEqual(result["input_data"], str(test_file))
    
    def test_get_data_for_script_missing_files(self):
        """Test get_data_for_script with missing data files."""
        # Update manifest with non-existent file
        manifest = {
            "scripts": {
                "test_script": {
                    "missing_data": {
                        "path": "test_script/missing.csv",
                        "format": "csv"
                    }
                }
            }
        }
        self.manager._save_manifest(manifest)
        
        with patch('src.cursus.validation.runtime.data.local_data_manager.logger') as mock_logger:
            result = self.manager.get_data_for_script("test_script")
            
            self.assertIsNone(result)
            mock_logger.warning.assert_called()
    
    def test_get_data_for_script_nonexistent_script(self):
        """Test get_data_for_script with non-existent script."""
        with patch('src.cursus.validation.runtime.data.local_data_manager.logger') as mock_logger:
            result = self.manager.get_data_for_script("nonexistent_script")
            
            self.assertIsNone(result)
            mock_logger.info.assert_called_with("No local data configured for script: nonexistent_script")
    
    def test_prepare_data_for_execution_with_data(self):
        """Test prepare_data_for_execution with existing data."""
        # Create test data file
        script_dir = self.manager.local_data_dir / "test_script"
        script_dir.mkdir(exist_ok=True)
        test_file = script_dir / "data.csv"
        test_content = "col1,col2\nvalue1,value2\n"
        test_file.write_text(test_content)
        
        # Update manifest
        manifest = {
            "scripts": {
                "test_script": {
                    "input": {
                        "path": "test_script/data.csv",
                        "format": "csv"
                    }
                }
            }
        }
        self.manager._save_manifest(manifest)
        
        # Prepare data for execution
        target_dir = Path(self.temp_dir) / "execution"
        self.manager.prepare_data_for_execution("test_script", str(target_dir))
        
        # Verify file was copied
        target_file = target_dir / "data.csv"
        self.assertTrue(target_file.exists())
        self.assertEqual(target_file.read_text(), test_content)
    
    def test_prepare_data_for_execution_no_data(self):
        """Test prepare_data_for_execution with no data available."""
        with patch('src.cursus.validation.runtime.data.local_data_manager.logger') as mock_logger:
            target_dir = Path(self.temp_dir) / "execution"
            self.manager.prepare_data_for_execution("nonexistent_script", str(target_dir))
            
            mock_logger.info.assert_called_with("No local data to prepare for script: nonexistent_script")
    
    def test_add_data_for_script_success(self):
        """Test successful addition of data file for script."""
        # Create source file
        source_file = Path(self.temp_dir) / "source_data.csv"
        source_content = "id,name\n1,test\n"
        source_file.write_text(source_content)
        
        result = self.manager.add_data_for_script(
            "new_script", 
            "input_data", 
            str(source_file), 
            "Test input data"
        )
        
        self.assertTrue(result)
        
        # Verify file was copied
        target_file = self.manager.local_data_dir / "new_script" / "source_data.csv"
        self.assertTrue(target_file.exists())
        self.assertEqual(target_file.read_text(), source_content)
        
        # Verify manifest was updated
        manifest = self.manager._load_manifest()
        self.assertIn("new_script", manifest["scripts"])
        self.assertIn("input_data", manifest["scripts"]["new_script"])
        
        file_info = manifest["scripts"]["new_script"]["input_data"]
        self.assertEqual(file_info["path"], "new_script/source_data.csv")
        self.assertEqual(file_info["format"], "csv")
        self.assertEqual(file_info["description"], "Test input data")
    
    def test_add_data_for_script_pickle_format(self):
        """Test adding pickle file with correct format detection."""
        # Create source pickle file
        source_file = Path(self.temp_dir) / "model.pkl"
        source_file.write_bytes(b"fake pickle data")
        
        result = self.manager.add_data_for_script("ml_script", "model", str(source_file))
        
        self.assertTrue(result)
        
        # Verify format was detected as pickle
        manifest = self.manager._load_manifest()
        file_info = manifest["scripts"]["ml_script"]["model"]
        self.assertEqual(file_info["format"], "pickle")
    
    def test_add_data_for_script_nonexistent_source(self):
        """Test adding data with non-existent source file."""
        with patch('src.cursus.validation.runtime.data.local_data_manager.logger') as mock_logger:
            result = self.manager.add_data_for_script(
                "test_script", 
                "data", 
                "/nonexistent/file.csv"
            )
            
            self.assertFalse(result)
            mock_logger.error.assert_called()
    
    def test_add_data_for_script_exception_handling(self):
        """Test add_data_for_script with exception during copy."""
        source_file = Path(self.temp_dir) / "source.csv"
        source_file.write_text("test data")
        
        with patch('shutil.copy2', side_effect=OSError("Permission denied")):
            with patch('src.cursus.validation.runtime.data.local_data_manager.logger') as mock_logger:
                result = self.manager.add_data_for_script("test_script", "data", str(source_file))
                
                self.assertFalse(result)
                mock_logger.error.assert_called()
    
    def test_list_data_for_script_existing(self):
        """Test list_data_for_script with existing script."""
        manifest = {
            "scripts": {
                "test_script": {
                    "input1": {"path": "test_script/input1.csv", "format": "csv"},
                    "input2": {"path": "test_script/input2.json", "format": "json"}
                }
            }
        }
        self.manager._save_manifest(manifest)
        
        result = self.manager.list_data_for_script("test_script")
        
        self.assertEqual(len(result), 2)
        self.assertIn("input1", result)
        self.assertIn("input2", result)
        self.assertEqual(result["input1"]["format"], "csv")
        self.assertEqual(result["input2"]["format"], "json")
    
    def test_list_data_for_script_nonexistent(self):
        """Test list_data_for_script with non-existent script."""
        result = self.manager.list_data_for_script("nonexistent_script")
        self.assertEqual(result, {})
    
    def test_list_all_scripts(self):
        """Test list_all_scripts."""
        manifest = {
            "scripts": {
                "script1": {"data": {"path": "script1/data.csv"}},
                "script2": {"data": {"path": "script2/data.json"}},
                "script3": {"data": {"path": "script3/data.pkl"}}
            }
        }
        self.manager._save_manifest(manifest)
        
        result = self.manager.list_all_scripts()
        
        self.assertEqual(len(result), 3)
        self.assertIn("script1", result)
        self.assertIn("script2", result)
        self.assertIn("script3", result)
    
    def test_list_all_scripts_empty(self):
        """Test list_all_scripts with no scripts."""
        manifest = {"scripts": {}}
        self.manager._save_manifest(manifest)
        
        result = self.manager.list_all_scripts()
        self.assertEqual(result, [])
    
    def test_remove_data_for_script_specific_key(self):
        """Test removing specific data key for script."""
        # Create test file and manifest
        script_dir = self.manager.local_data_dir / "test_script"
        script_dir.mkdir(exist_ok=True)
        test_file = script_dir / "data.csv"
        test_file.write_text("test data")
        
        manifest = {
            "scripts": {
                "test_script": {
                    "data1": {"path": "test_script/data.csv", "format": "csv"},
                    "data2": {"path": "test_script/other.json", "format": "json"}
                }
            }
        }
        self.manager._save_manifest(manifest)
        
        result = self.manager.remove_data_for_script("test_script", "data1")
        
        self.assertTrue(result)
        self.assertFalse(test_file.exists())
        
        # Verify manifest was updated
        updated_manifest = self.manager._load_manifest()
        self.assertNotIn("data1", updated_manifest["scripts"]["test_script"])
        self.assertIn("data2", updated_manifest["scripts"]["test_script"])
    
    def test_remove_data_for_script_all_data(self):
        """Test removing all data for script."""
        # Create test directory and files
        script_dir = self.manager.local_data_dir / "test_script"
        script_dir.mkdir(exist_ok=True)
        (script_dir / "file1.csv").write_text("data1")
        (script_dir / "file2.json").write_text("data2")
        
        manifest = {
            "scripts": {
                "test_script": {
                    "data1": {"path": "test_script/file1.csv"},
                    "data2": {"path": "test_script/file2.json"}
                },
                "other_script": {
                    "data": {"path": "other_script/data.csv"}
                }
            }
        }
        self.manager._save_manifest(manifest)
        
        result = self.manager.remove_data_for_script("test_script")
        
        self.assertTrue(result)
        self.assertFalse(script_dir.exists())
        
        # Verify manifest was updated
        updated_manifest = self.manager._load_manifest()
        self.assertNotIn("test_script", updated_manifest["scripts"])
        self.assertIn("other_script", updated_manifest["scripts"])
    
    def test_remove_data_for_script_nonexistent_script(self):
        """Test removing data for non-existent script."""
        with patch('src.cursus.validation.runtime.data.local_data_manager.logger') as mock_logger:
            result = self.manager.remove_data_for_script("nonexistent_script")
            
            self.assertFalse(result)
            mock_logger.warning.assert_called_with("No local data found for script: nonexistent_script")
    
    def test_remove_data_for_script_nonexistent_key(self):
        """Test removing non-existent data key."""
        manifest = {
            "scripts": {
                "test_script": {
                    "existing_data": {"path": "test_script/data.csv"}
                }
            }
        }
        self.manager._save_manifest(manifest)
        
        with patch('src.cursus.validation.runtime.data.local_data_manager.logger') as mock_logger:
            result = self.manager.remove_data_for_script("test_script", "nonexistent_key")
            
            self.assertFalse(result)
            mock_logger.warning.assert_called_with("Data key not found: test_script.nonexistent_key")
    
    def test_remove_data_for_script_exception_handling(self):
        """Test remove_data_for_script with exception during manifest save."""
        manifest = {
            "scripts": {
                "test_script": {
                    "data": {"path": "test_script/data.csv"}
                }
            }
        }
        self.manager._save_manifest(manifest)
        
        # Mock _save_manifest to raise an exception
        with patch.object(self.manager, '_save_manifest', side_effect=OSError("Permission denied")):
            with patch('src.cursus.validation.runtime.data.local_data_manager.logger') as mock_logger:
                result = self.manager.remove_data_for_script("test_script")
                
                self.assertFalse(result)
                mock_logger.error.assert_called()
    
    def test_load_manifest_success(self):
        """Test successful manifest loading."""
        test_manifest = {
            "version": "1.0",
            "scripts": {
                "test": {"data": {"path": "test/data.csv"}}
            }
        }
        
        with open(self.manager.manifest_path, 'w') as f:
            yaml.dump(test_manifest, f)
        
        result = self.manager._load_manifest()
        
        self.assertEqual(result["version"], "1.0")
        self.assertIn("test", result["scripts"])
    
    def test_load_manifest_file_not_found(self):
        """Test manifest loading when file doesn't exist."""
        # Remove manifest file
        if self.manager.manifest_path.exists():
            self.manager.manifest_path.unlink()
        
        with patch('src.cursus.validation.runtime.data.local_data_manager.logger') as mock_logger:
            result = self.manager._load_manifest()
            
            self.assertEqual(result, {})
            mock_logger.warning.assert_called()
    
    def test_load_manifest_invalid_yaml(self):
        """Test manifest loading with invalid YAML."""
        # Write invalid YAML
        with open(self.manager.manifest_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with patch('src.cursus.validation.runtime.data.local_data_manager.logger') as mock_logger:
            result = self.manager._load_manifest()
            
            self.assertEqual(result, {})
            mock_logger.warning.assert_called()
    
    def test_save_manifest_success(self):
        """Test successful manifest saving."""
        test_manifest = {
            "version": "2.0",
            "scripts": {
                "new_script": {"data": {"path": "new_script/data.csv"}}
            }
        }
        
        self.manager._save_manifest(test_manifest)
        
        # Verify file was saved correctly
        with open(self.manager.manifest_path) as f:
            saved_manifest = yaml.safe_load(f)
        
        self.assertEqual(saved_manifest["version"], "2.0")
        self.assertIn("new_script", saved_manifest["scripts"])
    
    def test_save_manifest_exception_handling(self):
        """Test manifest saving with exception."""
        test_manifest = {"test": "data"}
        
        with patch('builtins.open', side_effect=OSError("Permission denied")):
            with patch('src.cursus.validation.runtime.data.local_data_manager.logger') as mock_logger:
                self.manager._save_manifest(test_manifest)
                
                mock_logger.error.assert_called()
    
    def test_create_default_manifest(self):
        """Test creation of default manifest."""
        # Remove existing manifest
        if self.manager.manifest_path.exists():
            self.manager.manifest_path.unlink()
        
        self.manager._create_default_manifest()
        
        # Verify default manifest was created
        self.assertTrue(self.manager.manifest_path.exists())
        
        manifest = self.manager._load_manifest()
        self.assertEqual(manifest["version"], "1.0")
        self.assertIn("description", manifest)
        self.assertIn("example_script", manifest["scripts"])
    
    def test_integration_add_and_get_data(self):
        """Test integration between adding and getting data."""
        # Create source file
        source_file = Path(self.temp_dir) / "integration_test.csv"
        source_file.write_text("col1,col2\nval1,val2\n")
        
        # Add data
        success = self.manager.add_data_for_script(
            "integration_script", 
            "test_data", 
            str(source_file),
            "Integration test data"
        )
        self.assertTrue(success)
        
        # Get data
        data_paths = self.manager.get_data_for_script("integration_script")
        self.assertIsNotNone(data_paths)
        self.assertIn("test_data", data_paths)
        
        # Verify file content
        retrieved_file = Path(data_paths["test_data"])
        self.assertTrue(retrieved_file.exists())
        self.assertEqual(retrieved_file.read_text(), "col1,col2\nval1,val2\n")
    
    def test_integration_prepare_and_execute(self):
        """Test integration between data preparation and execution."""
        # Add test data
        source_file = Path(self.temp_dir) / "exec_test.json"
        source_file.write_text('{"key": "value"}')
        
        self.manager.add_data_for_script("exec_script", "config", str(source_file))
        
        # Prepare data for execution
        exec_dir = Path(self.temp_dir) / "execution"
        self.manager.prepare_data_for_execution("exec_script", str(exec_dir))
        
        # Verify data was prepared
        prepared_file = exec_dir / "exec_test.json"
        self.assertTrue(prepared_file.exists())
        self.assertEqual(prepared_file.read_text(), '{"key": "value"}')
    
    def test_edge_case_empty_manifest(self):
        """Test handling of completely empty manifest."""
        # Create empty manifest
        with open(self.manager.manifest_path, 'w') as f:
            f.write("")
        
        result = self.manager.list_all_scripts()
        self.assertEqual(result, [])
        
        # Should still be able to add data
        source_file = Path(self.temp_dir) / "test.csv"
        source_file.write_text("data")
        
        success = self.manager.add_data_for_script("new_script", "data", str(source_file))
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
