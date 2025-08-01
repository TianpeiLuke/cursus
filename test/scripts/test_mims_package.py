# test/test_mims_package.py
import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import shutil
import tarfile
from pathlib import Path
import logging

# Add the project root to the Python path to allow for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the functions and main entrypoint from the script to be tested
from src.pipeline_scripts.mims_package import (
    ensure_directory,
    check_file_exists,
    list_directory_contents,
    copy_file_robust,
    copy_scripts,
    extract_tarfile,
    create_tarfile,
    main as package_main
)

# Disable logging for cleaner test output
logging.disable(logging.CRITICAL)

class TestMimsPackagingHelpers(unittest.TestCase):
    """Unit tests for the individual helper functions in the packaging script."""

    def setUp(self):
        """Set up a temporary directory for each test."""
        self.base_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.base_dir)

    def _create_dummy_file(self, path: Path, content: str = "dummy"):
        """Helper to create a dummy file within the temporary directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_ensure_directory(self):
        """Test that `ensure_directory` creates a directory if it doesn't exist."""
        new_dir = self.base_dir / "new_dir"
        self.assertFalse(new_dir.exists())
        self.assertTrue(ensure_directory(new_dir))
        self.assertTrue(new_dir.exists() and new_dir.is_dir())
        # Test that it returns True for an existing directory
        self.assertTrue(ensure_directory(new_dir))

    def test_copy_file_robust(self):
        """Test the robust file copying function."""
        src_file = self.base_dir / "source" / "file.txt"
        dst_file = self.base_dir / "dest" / "file.txt"
        self._create_dummy_file(src_file, "test content")

        # Test successful copy
        self.assertTrue(copy_file_robust(src_file, dst_file))
        self.assertTrue(dst_file.exists())
        self.assertEqual(dst_file.read_text(), "test content")

        # Test copying a non-existent file
        self.assertFalse(copy_file_robust(self.base_dir / "nonexistent.txt", dst_file))

    def test_create_and_extract_tarfile(self):
        """Test that tarball creation and extraction work as inverse operations."""
        source_dir = self.base_dir / "source_for_tar"
        output_tar_path = self.base_dir / "output.tar.gz"
        extract_dir = self.base_dir / "extracted"

        # Create some files to be tarred
        self._create_dummy_file(source_dir / "file1.txt")
        self._create_dummy_file(source_dir / "code" / "inference.py")

        # Create the tarball
        create_tarfile(output_tar_path, source_dir)
        self.assertTrue(output_tar_path.exists())

        # Extract the tarball
        extract_tarfile(output_tar_path, extract_dir)
        
        # Verify the extracted contents
        self.assertTrue((extract_dir / "file1.txt").exists())
        self.assertTrue((extract_dir / "code" / "inference.py").exists())


class TestMimsPackagingMainFlow(unittest.TestCase):
    """
    Integration-style tests for the main() function of the packaging script.
    This class uses patching to redirect the script's hardcoded paths to a
    temporary directory structure.
    """

    def setUp(self):
        """Set up a temporary directory structure mimicking the SageMaker environment."""
        self.base_dir = Path(tempfile.mkdtemp())
        
        # Define mock paths within the temporary directory
        self.model_path = self.base_dir / "input" / "model"
        self.script_path = self.base_dir / "input" / "script"
        self.output_path = self.base_dir / "output"
        self.working_dir = self.base_dir / "working"

        # Create the input directories
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.script_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.base_dir)

    def _create_dummy_file(self, path: Path, content: str = "dummy"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_main_flow_with_input_tar(self):
        """
        Test the main() function when the input model artifact is a tar.gz file.
        """
        # --- Arrange ---
        # Create dummy input files: a model and an inference script
        dummy_model_content_path = self.base_dir / "temp_model" / "model.pth"
        self._create_dummy_file(dummy_model_content_path, "pytorch-model-data")
        self._create_dummy_file(self.script_path / "inference.py", "import torch")
        
        # Create a tarball containing the model file
        input_tar_path = self.model_path / "model.tar.gz"
        with tarfile.open(input_tar_path, "w:gz") as tar:
            tar.add(dummy_model_content_path, arcname="model.pth")
        
        # --- Act ---
        # Use patch.object to replace the module-level constants with our temporary paths
        from src.pipeline_scripts import mims_package
        with patch.object(mims_package, 'MODEL_PATH', self.model_path), \
             patch.object(mims_package, 'SCRIPT_PATH', self.script_path), \
             patch.object(mims_package, 'OUTPUT_PATH', self.output_path), \
             patch.object(mims_package, 'WORKING_DIRECTORY', self.working_dir), \
             patch.object(mims_package, 'CODE_DIRECTORY', self.working_dir / 'code'):
            
            package_main()

        # --- Assert ---
        final_output_tar = self.output_path / "model.tar.gz"
        self.assertTrue(final_output_tar.exists(), "Final model.tar.gz was not created.")

        with tarfile.open(final_output_tar, "r:gz") as tar:
            members = tar.getnames()
            self.assertIn("model.pth", members)
            self.assertIn("code/inference.py", members)

    def test_main_flow_with_direct_files(self):
        """
        Test the main() function when model artifacts are provided as direct files
        instead of a tarball.
        """
        # --- Arrange ---
        # Create dummy input files directly in the mocked input directories
        self._create_dummy_file(self.model_path / "xgboost_model.bst", "xgboost-model-data")
        self._create_dummy_file(self.script_path / "requirements.txt", "pandas\nscikit-learn")

        # --- Act ---
        from src.pipeline_scripts import mims_package
        with patch.object(mims_package, 'MODEL_PATH', self.model_path), \
             patch.object(mims_package, 'SCRIPT_PATH', self.script_path), \
             patch.object(mims_package, 'OUTPUT_PATH', self.output_path), \
             patch.object(mims_package, 'WORKING_DIRECTORY', self.working_dir), \
             patch.object(mims_package, 'CODE_DIRECTORY', self.working_dir / 'code'):

            package_main()

        # --- Assert ---
        final_output_tar = self.output_path / "model.tar.gz"
        self.assertTrue(final_output_tar.exists())

        with tarfile.open(final_output_tar, "r:gz") as tar:
            members = tar.getnames()
            self.assertIn("xgboost_model.bst", members)
            self.assertIn("code/requirements.txt", members)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
