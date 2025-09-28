import pytest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil
import tarfile
from pathlib import Path
import json

# Import the functions to be tested
from cursus.steps.scripts.dummy_training import (
    validate_model,
    ensure_directory,
    extract_tarfile,
    create_tarfile,
    copy_file,
    process_model_with_hyperparameters,
    main,
)


class TestDummyTrainingHelpers:
    """Unit tests for helper functions in the dummy training script."""

    @pytest.fixture
    def temp_dir(self):
        """Set up test fixtures."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def _create_dummy_tar(self, temp_dir: Path, tar_path: Path, files_content: dict):
        """Helper to create a dummy tar.gz file with specified content."""
        temp_content_dir = temp_dir / "temp_content"
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

    def test_validate_model_valid_tar(self, temp_dir):
        """Test model validation with valid tar.gz file."""
        # Create a valid tar.gz file
        tar_path = temp_dir / "model.tar.gz"
        self._create_dummy_tar(temp_dir, tar_path, {"model.pth": "dummy model content"})

        # Should return True without raising exception
        result = validate_model(tar_path)
        assert result is True

    def test_validate_model_invalid_extension(self, temp_dir):
        """Test model validation with invalid file extension."""
        invalid_path = temp_dir / "model.txt"
        invalid_path.write_text("not a tar file")

        with pytest.raises(ValueError) as exc_info:
            validate_model(invalid_path)

        assert "Expected a .tar.gz file" in str(exc_info.value)
        assert "INVALID_FORMAT" in str(exc_info.value)

    def test_validate_model_invalid_tar(self, temp_dir):
        """Test model validation with invalid tar file."""
        invalid_tar = temp_dir / "model.tar.gz"
        invalid_tar.write_text("not a valid tar file")

        with pytest.raises(ValueError) as exc_info:
            validate_model(invalid_tar)

        assert "not a valid tar archive" in str(exc_info.value)
        assert "INVALID_ARCHIVE" in str(exc_info.value)

    def test_ensure_directory_creates_new_directory(self, temp_dir):
        """Test that ensure_directory creates a new directory."""
        new_dir = temp_dir / "new_directory"
        assert not new_dir.exists()

        result = ensure_directory(new_dir)

        assert result is True
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_directory_existing_directory(self, temp_dir):
        """Test that ensure_directory works with existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()

        result = ensure_directory(existing_dir)

        assert result is True
        assert existing_dir.exists()

    def test_copy_file_success(self, temp_dir):
        """Test successful file copying."""
        src_file = temp_dir / "source.txt"
        dst_file = temp_dir / "dest" / "destination.txt"

        src_file.write_text("test content")

        copy_file(src_file, dst_file)

        assert dst_file.exists()
        assert dst_file.read_text() == "test content"

    def test_copy_file_nonexistent_source(self, temp_dir):
        """Test copying nonexistent source file raises FileNotFoundError."""
        src_file = temp_dir / "nonexistent.txt"
        dst_file = temp_dir / "destination.txt"

        with pytest.raises(FileNotFoundError):
            copy_file(src_file, dst_file)

    def test_extract_tarfile_success(self, temp_dir):
        """Test successful tar file extraction."""
        # Create a tar file
        tar_path = temp_dir / "test.tar.gz"
        extract_path = temp_dir / "extracted"

        files_content = {"file1.txt": "content1", "subdir/file2.txt": "content2"}
        self._create_dummy_tar(temp_dir, tar_path, files_content)

        # Extract the tar file
        extract_tarfile(tar_path, extract_path)

        # Verify extraction
        assert (extract_path / "file1.txt").exists()
        assert (extract_path / "subdir" / "file2.txt").exists()
        assert (extract_path / "file1.txt").read_text() == "content1"
        assert (extract_path / "subdir" / "file2.txt").read_text() == "content2"

    def test_extract_tarfile_nonexistent_file(self, temp_dir):
        """Test extracting nonexistent tar file raises FileNotFoundError."""
        tar_path = temp_dir / "nonexistent.tar.gz"
        extract_path = temp_dir / "extracted"

        with pytest.raises(FileNotFoundError):
            extract_tarfile(tar_path, extract_path)

    def test_create_tarfile_success(self, temp_dir):
        """Test successful tar file creation."""
        # Create source directory with files
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("content1")
        (source_dir / "subdir").mkdir()
        (source_dir / "subdir" / "file2.txt").write_text("content2")

        # Create tar file
        tar_path = temp_dir / "output.tar.gz"
        create_tarfile(tar_path, source_dir)

        # Verify tar file was created and contains expected files
        assert tar_path.exists()

        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getnames()
            assert "file1.txt" in members
            assert "subdir/file2.txt" in members

    def test_process_model_with_hyperparameters_success(self, temp_dir):
        """Test successful model processing with hyperparameters."""
        # Create input model tar
        model_path = temp_dir / "input_model.tar.gz"
        self._create_dummy_tar(temp_dir, model_path, {"model.pth": "model content"})

        # Create hyperparameters file
        hyperparams_path = temp_dir / "hyperparameters.json"
        hyperparams_data = {"learning_rate": 0.01, "epochs": 100}
        hyperparams_path.write_text(json.dumps(hyperparams_data))

        # Create output directory
        output_dir = temp_dir / "output"

        # Process model
        result_path = process_model_with_hyperparameters(
            model_path, hyperparams_path, output_dir
        )

        # Verify output
        expected_output = output_dir / "model.tar.gz"
        assert result_path == expected_output
        assert expected_output.exists()

        # Verify contents of output tar
        extract_dir = temp_dir / "verify_extract"
        with tarfile.open(expected_output, "r:gz") as tar:
            tar.extractall(extract_dir)

        assert (extract_dir / "model.pth").exists()
        assert (extract_dir / "hyperparameters.json").exists()

        # Verify hyperparameters content
        extracted_hyperparams = json.loads(
            (extract_dir / "hyperparameters.json").read_text()
        )
        assert extracted_hyperparams == hyperparams_data

    def test_process_model_with_hyperparameters_missing_model(self, temp_dir):
        """Test processing with missing model file."""
        model_path = temp_dir / "nonexistent_model.tar.gz"
        hyperparams_path = temp_dir / "hyperparameters.json"
        hyperparams_path.write_text('{"test": "value"}')
        output_dir = temp_dir / "output"

        with pytest.raises(FileNotFoundError):
            process_model_with_hyperparameters(model_path, hyperparams_path, output_dir)

    def test_process_model_with_hyperparameters_missing_hyperparams(self, temp_dir):
        """Test processing with missing hyperparameters file."""
        model_path = temp_dir / "model.tar.gz"
        self._create_dummy_tar(temp_dir, model_path, {"model.pth": "content"})

        hyperparams_path = temp_dir / "nonexistent_hyperparams.json"
        output_dir = temp_dir / "output"

        with pytest.raises(FileNotFoundError):
            process_model_with_hyperparameters(model_path, hyperparams_path, output_dir)


class TestDummyTrainingMain:
    """Tests for the main function of the dummy training script."""

    @pytest.fixture
    def temp_dir(self):
        """Set up test fixtures."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def setup_paths(self, temp_dir):
        """Set up mock paths that match the script's expected structure."""
        # Create the directory structure the script expects
        models_dir = temp_dir / "opt" / "ml" / "code" / "models"
        hyperparams_dir = temp_dir / "opt" / "ml" / "code" / "hyperparams"
        output_dir = temp_dir / "output"
        
        models_dir.mkdir(parents=True, exist_ok=True)
        hyperparams_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "model.tar.gz"
        hyperparams_path = hyperparams_dir / "hyperparameters.json"
        
        return model_path, hyperparams_path, output_dir, temp_dir

    def _create_dummy_tar(self, temp_dir: Path, tar_path: Path, files_content: dict):
        """Helper to create a dummy tar.gz file with specified content."""
        temp_content_dir = temp_dir / "temp_content"
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

    @patch("cursus.steps.scripts.dummy_training.find_model_file")
    @patch("cursus.steps.scripts.dummy_training.find_hyperparams_file")
    def test_main_with_hyperparameters(self, mock_find_hyperparams, mock_find_model, temp_dir, setup_paths):
        """Test main function with hyperparameters file present."""
        model_path, hyperparams_path, output_dir, temp_root = setup_paths
        
        # Create input files
        self._create_dummy_tar(temp_dir, model_path, {"model.pth": "model content"})
        hyperparams_data = {"learning_rate": 0.01, "epochs": 100}
        hyperparams_path.write_text(json.dumps(hyperparams_data))

        # Mock the file finding functions to return our test paths
        mock_find_model.return_value = model_path
        mock_find_hyperparams.return_value = hyperparams_path

        # Set up input and output paths
        input_paths = {}  # Empty for SOURCE node
        output_paths = {"model_output": str(output_dir)}
        environ_vars = {}

        # Run main function
        result = main(input_paths, output_paths, environ_vars)

        # Verify success (should return Path object, not exit code)
        assert isinstance(result, Path)

        # Verify output file exists
        output_file = output_dir / "model.tar.gz"
        assert output_file.exists()

    @patch("cursus.steps.scripts.dummy_training.find_model_file")
    @patch("cursus.steps.scripts.dummy_training.find_hyperparams_file")
    def test_main_without_hyperparameters(self, mock_find_hyperparams, mock_find_model, temp_dir, setup_paths):
        """Test main function without hyperparameters file raises FileNotFoundError."""
        model_path, hyperparams_path, output_dir, temp_root = setup_paths
        
        # Create only model file (no hyperparameters)
        self._create_dummy_tar(temp_dir, model_path, {"model.pth": "model content"})
        # Don't create hyperparameters file

        # Mock the file finding functions
        mock_find_model.return_value = model_path
        mock_find_hyperparams.return_value = None  # No hyperparameters file

        # Set up input and output paths
        input_paths = {}  # Empty for SOURCE node
        output_paths = {"model_output": str(output_dir)}
        environ_vars = {}

        # Run main function and expect FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            main(input_paths, output_paths, environ_vars)
        
        assert "Hyperparameters file" in str(exc_info.value)

    @patch("cursus.steps.scripts.dummy_training.find_model_file")
    @patch("cursus.steps.scripts.dummy_training.find_hyperparams_file")
    def test_main_missing_model_file(self, mock_find_hyperparams, mock_find_model, setup_paths):
        """Test main function with missing model file raises FileNotFoundError."""
        model_path, hyperparams_path, output_dir, temp_root = setup_paths
        
        # Don't create model file
        # Create hyperparameters file
        hyperparams_path.write_text('{"test": "value"}')

        # Mock the file finding functions to simulate missing model
        mock_find_model.return_value = None  # No model file found
        mock_find_hyperparams.return_value = hyperparams_path

        # Set up input and output paths
        input_paths = {}  # Empty for SOURCE node
        output_paths = {"model_output": str(output_dir)}
        environ_vars = {}

        # Run main function and expect FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            main(input_paths, output_paths, environ_vars)
        
        assert "Model file" in str(exc_info.value)

    @patch("cursus.steps.scripts.dummy_training.find_model_file")
    @patch("cursus.steps.scripts.dummy_training.find_hyperparams_file")
    def test_main_invalid_model_file(self, mock_find_hyperparams, mock_find_model, setup_paths):
        """Test main function with invalid model file raises FileNotFoundError."""
        model_path, hyperparams_path, output_dir, temp_root = setup_paths
        
        # Create invalid model file (not a tar)
        model_path.write_text("not a tar file")
        # Create hyperparameters file
        hyperparams_path.write_text('{"test": "value"}')

        # Mock the file finding functions to return our test paths
        mock_find_model.return_value = model_path
        mock_find_hyperparams.return_value = hyperparams_path

        # Set up input and output paths
        input_paths = {}  # Empty for SOURCE node
        output_paths = {"model_output": str(output_dir)}
        environ_vars = {}

        # Run main function and expect tarfile.ReadError due to invalid tar file
        with pytest.raises(Exception) as exc_info:
            main(input_paths, output_paths, environ_vars)
        
        # The error could be tarfile.ReadError or any exception from the invalid tar processing
        assert "file could not be opened successfully" in str(exc_info.value) or "ReadError" in str(exc_info.value)

    @patch("cursus.steps.scripts.dummy_training.find_model_file")
    @patch("cursus.steps.scripts.dummy_training.find_hyperparams_file")
    @patch("cursus.steps.scripts.dummy_training.process_model_with_hyperparameters")
    def test_main_unexpected_error(self, mock_process, mock_find_hyperparams, mock_find_model, temp_dir, setup_paths):
        """Test main function with unexpected error raises RuntimeError."""
        model_path, hyperparams_path, output_dir, temp_root = setup_paths
        
        # Create input files
        self._create_dummy_tar(temp_dir, model_path, {"model.pth": "content"})
        hyperparams_path.write_text('{"test": "value"}')

        # Mock the file finding functions to return our test paths
        mock_find_model.return_value = model_path
        mock_find_hyperparams.return_value = hyperparams_path

        # Mock process function to raise unexpected error
        mock_process.side_effect = RuntimeError("Unexpected error")

        # Set up input and output paths
        input_paths = {}  # Empty for SOURCE node
        output_paths = {"model_output": str(output_dir)}
        environ_vars = {}

        # Run main function and expect RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            main(input_paths, output_paths, environ_vars)
        
        assert "Unexpected error" in str(exc_info.value)
