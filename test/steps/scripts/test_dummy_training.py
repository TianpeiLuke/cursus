import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import shutil
import tarfile
from pathlib import Path
import json

# Import the functions to be tested
from src.cursus.steps.scripts.dummy_training import (
    validate_model,
    ensure_directory,
    extract_tarfile,
    create_tarfile,
    copy_file,
    process_model_with_hyperparameters,
    main
)


class TestDummyTrainingHelpers(unittest.TestCase):
    """Unit tests for helper functions in the dummy training script."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_dummy_tar(self, tar_path: Path, files_content: dict):
        """Helper to create a dummy tar.gz file with specified content."""
        temp_content_dir = self.temp_dir / "temp_content"
        temp_content_dir.mkdir()
        
        # Create files in temp directory
        for filename, content in files_content.items():
            file_path = temp_content_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
        
        # Create tar file
        with tarfile.open(tar_path, "w:gz") as tar:
            for item in temp_content_dir.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(temp_content_dir)
                    tar.add(item, arcname=arcname)

    def test_validate_model_valid_tar(self):
        """Test model validation with valid tar.gz file."""
        # Create a valid tar.gz file
        tar_path = self.temp_dir / "model.tar.gz"
        self._create_dummy_tar(tar_path, {"model.pth": "dummy model content"})
        
        # Should return True without raising exception
        result = validate_model(tar_path)
        self.assertTrue(result)

    def test_validate_model_invalid_extension(self):
        """Test model validation with invalid file extension."""
        invalid_path = self.temp_dir / "model.txt"
        invalid_path.write_text("not a tar file")
        
        with self.assertRaises(ValueError) as context:
            validate_model(invalid_path)
        
        self.assertIn("Expected a .tar.gz file", str(context.exception))
        self.assertIn("INVALID_FORMAT", str(context.exception))

    def test_validate_model_invalid_tar(self):
        """Test model validation with invalid tar file."""
        invalid_tar = self.temp_dir / "model.tar.gz"
        invalid_tar.write_text("not a valid tar file")
        
        with self.assertRaises(ValueError) as context:
            validate_model(invalid_tar)
        
        self.assertIn("not a valid tar archive", str(context.exception))
        self.assertIn("INVALID_ARCHIVE", str(context.exception))

    def test_ensure_directory_creates_new_directory(self):
        """Test that ensure_directory creates a new directory."""
        new_dir = self.temp_dir / "new_directory"
        self.assertFalse(new_dir.exists())
        
        result = ensure_directory(new_dir)
        
        self.assertTrue(result)
        self.assertTrue(new_dir.exists())
        self.assertTrue(new_dir.is_dir())

    def test_ensure_directory_existing_directory(self):
        """Test that ensure_directory works with existing directory."""
        existing_dir = self.temp_dir / "existing"
        existing_dir.mkdir()
        
        result = ensure_directory(existing_dir)
        
        self.assertTrue(result)
        self.assertTrue(existing_dir.exists())

    def test_copy_file_success(self):
        """Test successful file copying."""
        src_file = self.temp_dir / "source.txt"
        dst_file = self.temp_dir / "dest" / "destination.txt"
        
        src_file.write_text("test content")
        
        copy_file(src_file, dst_file)
        
        self.assertTrue(dst_file.exists())
        self.assertEqual(dst_file.read_text(), "test content")

    def test_copy_file_nonexistent_source(self):
        """Test copying nonexistent source file raises FileNotFoundError."""
        src_file = self.temp_dir / "nonexistent.txt"
        dst_file = self.temp_dir / "destination.txt"
        
        with self.assertRaises(FileNotFoundError):
            copy_file(src_file, dst_file)

    def test_extract_tarfile_success(self):
        """Test successful tar file extraction."""
        # Create a tar file
        tar_path = self.temp_dir / "test.tar.gz"
        extract_path = self.temp_dir / "extracted"
        
        files_content = {
            "file1.txt": "content1",
            "subdir/file2.txt": "content2"
        }
        self._create_dummy_tar(tar_path, files_content)
        
        # Extract the tar file
        extract_tarfile(tar_path, extract_path)
        
        # Verify extraction
        self.assertTrue((extract_path / "file1.txt").exists())
        self.assertTrue((extract_path / "subdir" / "file2.txt").exists())
        self.assertEqual((extract_path / "file1.txt").read_text(), "content1")
        self.assertEqual((extract_path / "subdir" / "file2.txt").read_text(), "content2")

    def test_extract_tarfile_nonexistent_file(self):
        """Test extracting nonexistent tar file raises FileNotFoundError."""
        tar_path = self.temp_dir / "nonexistent.tar.gz"
        extract_path = self.temp_dir / "extracted"
        
        with self.assertRaises(FileNotFoundError):
            extract_tarfile(tar_path, extract_path)

    def test_create_tarfile_success(self):
        """Test successful tar file creation."""
        # Create source directory with files
        source_dir = self.temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("content1")
        (source_dir / "subdir").mkdir()
        (source_dir / "subdir" / "file2.txt").write_text("content2")
        
        # Create tar file
        tar_path = self.temp_dir / "output.tar.gz"
        create_tarfile(tar_path, source_dir)
        
        # Verify tar file was created and contains expected files
        self.assertTrue(tar_path.exists())
        
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getnames()
            self.assertIn("file1.txt", members)
            self.assertIn("subdir/file2.txt", members)

    def test_process_model_with_hyperparameters_success(self):
        """Test successful model processing with hyperparameters."""
        # Create input model tar
        model_path = self.temp_dir / "input_model.tar.gz"
        self._create_dummy_tar(model_path, {"model.pth": "model content"})
        
        # Create hyperparameters file
        hyperparams_path = self.temp_dir / "hyperparameters.json"
        hyperparams_data = {"learning_rate": 0.01, "epochs": 100}
        hyperparams_path.write_text(json.dumps(hyperparams_data))
        
        # Create output directory
        output_dir = self.temp_dir / "output"
        
        # Process model
        result_path = process_model_with_hyperparameters(
            model_path, hyperparams_path, output_dir
        )
        
        # Verify output
        expected_output = output_dir / "model.tar.gz"
        self.assertEqual(result_path, expected_output)
        self.assertTrue(expected_output.exists())
        
        # Verify contents of output tar
        extract_dir = self.temp_dir / "verify_extract"
        with tarfile.open(expected_output, "r:gz") as tar:
            tar.extractall(extract_dir)
        
        self.assertTrue((extract_dir / "model.pth").exists())
        self.assertTrue((extract_dir / "hyperparameters.json").exists())
        
        # Verify hyperparameters content
        extracted_hyperparams = json.loads((extract_dir / "hyperparameters.json").read_text())
        self.assertEqual(extracted_hyperparams, hyperparams_data)

    def test_process_model_with_hyperparameters_missing_model(self):
        """Test processing with missing model file."""
        model_path = self.temp_dir / "nonexistent_model.tar.gz"
        hyperparams_path = self.temp_dir / "hyperparameters.json"
        hyperparams_path.write_text('{"test": "value"}')
        output_dir = self.temp_dir / "output"
        
        with self.assertRaises(FileNotFoundError):
            process_model_with_hyperparameters(model_path, hyperparams_path, output_dir)

    def test_process_model_with_hyperparameters_missing_hyperparams(self):
        """Test processing with missing hyperparameters file."""
        model_path = self.temp_dir / "model.tar.gz"
        self._create_dummy_tar(model_path, {"model.pth": "content"})
        
        hyperparams_path = self.temp_dir / "nonexistent_hyperparams.json"
        output_dir = self.temp_dir / "output"
        
        with self.assertRaises(FileNotFoundError):
            process_model_with_hyperparameters(model_path, hyperparams_path, output_dir)


class TestDummyTrainingMain(unittest.TestCase):
    """Tests for the main function of the dummy training script."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Set up mock paths
        self.model_path = self.temp_dir / "model.tar.gz"
        self.hyperparams_path = self.temp_dir / "hyperparameters.json"
        self.output_dir = self.temp_dir / "output"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_dummy_tar(self, tar_path: Path, files_content: dict):
        """Helper to create a dummy tar.gz file with specified content."""
        temp_content_dir = self.temp_dir / "temp_content"
        temp_content_dir.mkdir(exist_ok=True)
        
        # Create files in temp directory
        for filename, content in files_content.items():
            file_path = temp_content_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
        
        # Create tar file
        with tarfile.open(tar_path, "w:gz") as tar:
            for item in temp_content_dir.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(temp_content_dir)
                    tar.add(item, arcname=arcname)

    @patch('src.cursus.steps.scripts.dummy_training.MODEL_INPUT_PATH')
    @patch('src.cursus.steps.scripts.dummy_training.HYPERPARAMS_INPUT_PATH')
    @patch('src.cursus.steps.scripts.dummy_training.MODEL_OUTPUT_DIR')
    def test_main_with_hyperparameters(self, mock_output_dir, mock_hyperparams_path, mock_model_path):
        """Test main function with hyperparameters file present."""
        # Set up mock paths
        mock_model_path.__str__ = lambda: str(self.model_path)
        mock_hyperparams_path.__str__ = lambda: str(self.hyperparams_path)
        mock_output_dir.__str__ = lambda: str(self.output_dir)
        
        # Create input files
        self._create_dummy_tar(self.model_path, {"model.pth": "model content"})
        hyperparams_data = {"learning_rate": 0.01, "epochs": 100}
        self.hyperparams_path.write_text(json.dumps(hyperparams_data))
        
        # Mock Path objects to return our temp paths
        with patch('src.cursus.steps.scripts.dummy_training.Path') as mock_path_class:
            def path_side_effect(path_str):
                if path_str == mock_model_path:
                    return self.model_path
                elif path_str == mock_hyperparams_path:
                    return self.hyperparams_path
                elif path_str == mock_output_dir:
                    return self.output_dir
                else:
                    return Path(path_str)
            
            mock_path_class.side_effect = path_side_effect
            
            # Run main function
            result = main()
            
            # Verify success
            self.assertEqual(result, 0)
            
            # Verify output file exists
            output_file = self.output_dir / "model.tar.gz"
            self.assertTrue(output_file.exists())

    @patch('src.cursus.steps.scripts.dummy_training.MODEL_INPUT_PATH')
    @patch('src.cursus.steps.scripts.dummy_training.HYPERPARAMS_INPUT_PATH')
    @patch('src.cursus.steps.scripts.dummy_training.MODEL_OUTPUT_DIR')
    def test_main_without_hyperparameters(self, mock_output_dir, mock_hyperparams_path, mock_model_path):
        """Test main function without hyperparameters file (fallback mode)."""
        # Set up mock paths
        mock_model_path.__str__ = lambda: str(self.model_path)
        mock_hyperparams_path.__str__ = lambda: str(self.hyperparams_path)
        mock_output_dir.__str__ = lambda: str(self.output_dir)
        
        # Create only model file (no hyperparameters)
        self._create_dummy_tar(self.model_path, {"model.pth": "model content"})
        # Don't create hyperparameters file
        
        # Mock Path objects
        with patch('src.cursus.steps.scripts.dummy_training.Path') as mock_path_class:
            def path_side_effect(path_str):
                if path_str == mock_model_path:
                    return self.model_path
                elif path_str == mock_hyperparams_path:
                    return self.hyperparams_path
                elif path_str == mock_output_dir:
                    return self.output_dir
                else:
                    return Path(path_str)
            
            mock_path_class.side_effect = path_side_effect
            
            # Run main function
            result = main()
            
            # Verify success
            self.assertEqual(result, 0)
            
            # Verify output file exists
            output_file = self.output_dir / "model.tar.gz"
            self.assertTrue(output_file.exists())

    @patch('src.cursus.steps.scripts.dummy_training.MODEL_INPUT_PATH')
    @patch('src.cursus.steps.scripts.dummy_training.HYPERPARAMS_INPUT_PATH')
    @patch('src.cursus.steps.scripts.dummy_training.MODEL_OUTPUT_DIR')
    def test_main_missing_model_file(self, mock_output_dir, mock_hyperparams_path, mock_model_path):
        """Test main function with missing model file."""
        # Set up mock paths
        mock_model_path.__str__ = lambda: str(self.model_path)
        mock_hyperparams_path.__str__ = lambda: str(self.hyperparams_path)
        mock_output_dir.__str__ = lambda: str(self.output_dir)
        
        # Don't create model file
        # Create hyperparameters file
        self.hyperparams_path.write_text('{"test": "value"}')
        
        # Mock Path objects
        with patch('src.cursus.steps.scripts.dummy_training.Path') as mock_path_class:
            def path_side_effect(path_str):
                if path_str == mock_model_path:
                    return self.model_path
                elif path_str == mock_hyperparams_path:
                    return self.hyperparams_path
                elif path_str == mock_output_dir:
                    return self.output_dir
                else:
                    return Path(path_str)
            
            mock_path_class.side_effect = path_side_effect
            
            # Run main function
            result = main()
            
            # Verify failure with appropriate exit code
            self.assertEqual(result, 1)  # FileNotFoundError

    @patch('src.cursus.steps.scripts.dummy_training.MODEL_INPUT_PATH')
    @patch('src.cursus.steps.scripts.dummy_training.HYPERPARAMS_INPUT_PATH')
    @patch('src.cursus.steps.scripts.dummy_training.MODEL_OUTPUT_DIR')
    def test_main_invalid_model_file(self, mock_output_dir, mock_hyperparams_path, mock_model_path):
        """Test main function with invalid model file."""
        # Set up mock paths
        mock_model_path.__str__ = lambda: str(self.model_path)
        mock_hyperparams_path.__str__ = lambda: str(self.hyperparams_path)
        mock_output_dir.__str__ = lambda: str(self.output_dir)
        
        # Create invalid model file (not a tar)
        self.model_path.write_text("not a tar file")
        
        # Mock Path objects
        with patch('src.cursus.steps.scripts.dummy_training.Path') as mock_path_class:
            def path_side_effect(path_str):
                if path_str == mock_model_path:
                    return self.model_path
                elif path_str == mock_hyperparams_path:
                    return self.hyperparams_path
                elif path_str == mock_output_dir:
                    return self.output_dir
                else:
                    return Path(path_str)
            
            mock_path_class.side_effect = path_side_effect
            
            # Run main function
            result = main()
            
            # Verify failure with appropriate exit code
            self.assertEqual(result, 2)  # ValueError

    @patch('src.cursus.steps.scripts.dummy_training.MODEL_INPUT_PATH')
    @patch('src.cursus.steps.scripts.dummy_training.HYPERPARAMS_INPUT_PATH')
    @patch('src.cursus.steps.scripts.dummy_training.MODEL_OUTPUT_DIR')
    @patch('src.cursus.steps.scripts.dummy_training.process_model_with_hyperparameters')
    def test_main_unexpected_error(self, mock_process, mock_output_dir, mock_hyperparams_path, mock_model_path):
        """Test main function with unexpected error."""
        # Set up mock paths
        mock_model_path.__str__ = lambda: str(self.model_path)
        mock_hyperparams_path.__str__ = lambda: str(self.hyperparams_path)
        mock_output_dir.__str__ = lambda: str(self.output_dir)
        
        # Create input files
        self._create_dummy_tar(self.model_path, {"model.pth": "content"})
        self.hyperparams_path.write_text('{"test": "value"}')
        
        # Mock process function to raise unexpected error
        mock_process.side_effect = RuntimeError("Unexpected error")
        
        # Mock Path objects
        with patch('src.cursus.steps.scripts.dummy_training.Path') as mock_path_class:
            def path_side_effect(path_str):
                if path_str == mock_model_path:
                    return self.model_path
                elif path_str == mock_hyperparams_path:
                    return self.hyperparams_path
                elif path_str == mock_output_dir:
                    return self.output_dir
                else:
                    return Path(path_str)
            
            mock_path_class.side_effect = path_side_effect
            
            # Run main function
            result = main()
            
            # Verify failure with appropriate exit code
            self.assertEqual(result, 3)  # RuntimeError


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
