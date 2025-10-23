# test/test_mims_package.py
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import tarfile
from pathlib import Path
import logging

# Import the functions and main entrypoint from the script to be tested
from cursus.steps.scripts.package import (
    ensure_directory,
    check_file_exists,
    list_directory_contents,
    copy_file_robust,
    copy_scripts,
    extract_tarfile,
    create_tarfile,
    main as package_main,
)

# Disable logging for cleaner test output
logging.disable(logging.CRITICAL)


class TestMimsPackagingHelpers:
    """Unit tests for the individual helper functions in the packaging script."""

    @pytest.fixture
    def base_dir(self):
        """Set up a temporary directory for each test."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def _create_dummy_file(self, path: Path, content: str = "dummy"):
        """Helper to create a dummy file within the temporary directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_ensure_directory(self, base_dir):
        """Test that `ensure_directory` creates a directory if it doesn't exist."""
        new_dir = base_dir / "new_dir"
        assert not new_dir.exists()
        assert ensure_directory(new_dir) is True
        assert new_dir.exists() and new_dir.is_dir()
        # Test that it returns True for an existing directory
        assert ensure_directory(new_dir) is True

    def test_copy_file_robust(self, base_dir):
        """Test the robust file copying function."""
        src_file = base_dir / "source" / "file.txt"
        dst_file = base_dir / "dest" / "file.txt"
        self._create_dummy_file(src_file, "test content")

        # Test successful copy
        assert copy_file_robust(src_file, dst_file) is True
        assert dst_file.exists()
        assert dst_file.read_text() == "test content"

        # Test copying a non-existent file
        assert copy_file_robust(base_dir / "nonexistent.txt", dst_file) is False

    def test_create_and_extract_tarfile(self, base_dir):
        """Test that tarball creation and extraction work as inverse operations."""
        source_dir = base_dir / "source_for_tar"
        output_tar_path = base_dir / "output.tar.gz"
        extract_dir = base_dir / "extracted"

        # Create some files to be tarred
        self._create_dummy_file(source_dir / "file1.txt")
        self._create_dummy_file(source_dir / "code" / "inference.py")

        # Create the tarball
        create_tarfile(output_tar_path, source_dir)
        assert output_tar_path.exists()

        # Extract the tarball
        extract_tarfile(output_tar_path, extract_dir)

        # Verify the extracted contents
        assert (extract_dir / "file1.txt").exists()
        assert (extract_dir / "code" / "inference.py").exists()


class TestMimsPackagingMainFlow:
    """
    Integration-style tests for the main() function of the packaging script.
    This class uses patching to redirect the script's hardcoded paths to a
    temporary directory structure.
    """

    @pytest.fixture
    def setup_dirs(self):
        """Set up a temporary directory structure mimicking the SageMaker environment."""
        base_dir = Path(tempfile.mkdtemp())

        # Define mock paths within the temporary directory
        model_path = base_dir / "input" / "model"
        script_path = base_dir / "input" / "script"
        output_path = base_dir / "output"
        working_dir = base_dir / "working"

        # Create the input directories
        model_path.mkdir(parents=True, exist_ok=True)
        script_path.mkdir(parents=True, exist_ok=True)

        yield {
            'base_dir': base_dir,
            'model_path': model_path,
            'script_path': script_path,
            'output_path': output_path,
            'working_dir': working_dir
        }
        
        shutil.rmtree(base_dir)

    def _create_dummy_file(self, path: Path, content: str = "dummy"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_main_flow_with_input_tar(self, setup_dirs):
        """
        Test the main() function when the input model artifact is a tar.gz file.
        """
        dirs = setup_dirs
        base_dir = dirs['base_dir']
        model_path = dirs['model_path']
        script_path = dirs['script_path']
        output_path = dirs['output_path']
        working_dir = dirs['working_dir']

        # --- Arrange ---
        # Create dummy input files: a model and an inference script
        dummy_model_content_path = base_dir / "temp_model" / "model.pth"
        self._create_dummy_file(dummy_model_content_path, "pytorch-model-data")
        self._create_dummy_file(script_path / "inference.py", "import torch")

        # Create a tarball containing the model file
        input_tar_path = model_path / "model.tar.gz"
        with tarfile.open(input_tar_path, "w:gz") as tar:
            tar.add(dummy_model_content_path, arcname="model.pth")

        # --- Act ---
        # Set up input and output paths for the new main function signature
        input_paths = {
            "model_input": str(model_path),
            "inference_scripts_input": str(script_path),
        }
        output_paths = {"packaged_model": str(output_path)}
        environ_vars = {"WORKING_DIRECTORY": str(working_dir)}

        result = package_main(input_paths, output_paths, environ_vars)

        # --- Assert ---
        final_output_tar = output_path / "model.tar.gz"
        assert final_output_tar.exists(), "Final model.tar.gz was not created."
        assert result == final_output_tar

        with tarfile.open(final_output_tar, "r:gz") as tar:
            members = tar.getnames()
            assert "model.pth" in members
            assert "code/inference.py" in members

    def test_main_flow_with_direct_files(self, setup_dirs):
        """
        Test the main() function when model artifacts are provided as direct files
        instead of a tarball.
        """
        dirs = setup_dirs
        model_path = dirs['model_path']
        script_path = dirs['script_path']
        output_path = dirs['output_path']
        working_dir = dirs['working_dir']

        # --- Arrange ---
        # Create dummy input files directly in the mocked input directories
        self._create_dummy_file(
            model_path / "xgboost_model.bst", "xgboost-model-data"
        )
        self._create_dummy_file(
            script_path / "requirements.txt", "pandas\nscikit-learn"
        )

        # --- Act ---
        # Set up input and output paths for the new main function signature
        input_paths = {
            "model_input": str(model_path),
            "inference_scripts_input": str(script_path),
        }
        output_paths = {"packaged_model": str(output_path)}
        environ_vars = {"WORKING_DIRECTORY": str(working_dir)}

        result = package_main(input_paths, output_paths, environ_vars)

        # --- Assert ---
        final_output_tar = output_path / "model.tar.gz"
        assert final_output_tar.exists()
        assert result == final_output_tar

        with tarfile.open(final_output_tar, "r:gz") as tar:
            members = tar.getnames()
            assert "xgboost_model.bst" in members
            assert "code/requirements.txt" in members
