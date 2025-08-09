import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import shutil
from pathlib import Path
import json

# Import the functions to be tested
from src.cursus.steps.scripts.contract_utils import (
    validate_contract_environment,
    get_contract_paths,
    get_input_path,
    get_output_path,
    validate_required_files,
    log_contract_summary,
    find_files_in_input,
    create_output_file_path,
    validate_framework_requirements,
    ContractEnforcer
)


class MockContract:
    """Mock contract class for testing"""
    def __init__(self):
        self.entry_point = "test_script.py"
        self.description = "Test script for contract validation"
        self.required_env_vars = ["REQUIRED_VAR1", "REQUIRED_VAR2"]
        self.optional_env_vars = {"OPTIONAL_VAR1": "default1", "OPTIONAL_VAR2": "default2"}
        self.expected_input_paths = {
            "data_input": "/opt/ml/processing/input/data",
            "model_input": "/opt/ml/processing/input/model"
        }
        self.expected_output_paths = {
            "processed_output": "/opt/ml/processing/output/processed",
            "metrics_output": "/opt/ml/processing/output/metrics"
        }
        self.framework_requirements = {
            "pandas": ">=1.0.0",
            "numpy": ">=1.18.0"
        }


class TestContractUtilsHelpers(unittest.TestCase):
    """Unit tests for contract utility helper functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.contract = MockContract()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_get_contract_paths(self):
        """Test getting contract paths."""
        paths = get_contract_paths(self.contract)
        
        self.assertIn('inputs', paths)
        self.assertIn('outputs', paths)
        self.assertEqual(paths['inputs'], self.contract.expected_input_paths)
        self.assertEqual(paths['outputs'], self.contract.expected_output_paths)

    def test_get_input_path_valid(self):
        """Test getting valid input path."""
        path = get_input_path(self.contract, 'data_input')
        self.assertEqual(path, "/opt/ml/processing/input/data")

    def test_get_input_path_invalid(self):
        """Test getting invalid input path raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_input_path(self.contract, 'nonexistent_input')
        
        self.assertIn("Unknown input 'nonexistent_input'", str(context.exception))
        self.assertIn("Available inputs:", str(context.exception))

    @patch('src.cursus.steps.scripts.contract_utils.os.makedirs')
    def test_get_output_path_valid(self, mock_makedirs):
        """Test getting valid output path."""
        path = get_output_path(self.contract, 'processed_output')
        
        self.assertEqual(path, "/opt/ml/processing/output/processed")
        mock_makedirs.assert_called_once_with("/opt/ml/processing/output/processed", exist_ok=True)

    def test_get_output_path_invalid(self):
        """Test getting invalid output path raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_output_path(self.contract, 'nonexistent_output')
        
        self.assertIn("Unknown output 'nonexistent_output'", str(context.exception))
        self.assertIn("Available outputs:", str(context.exception))

    def test_find_files_in_input_existing_directory(self):
        """Test finding files in existing input directory."""
        # Create test files
        test_input_dir = self.temp_dir / "test_input"
        test_input_dir.mkdir()
        (test_input_dir / "file1.txt").write_text("content1")
        (test_input_dir / "file2.csv").write_text("content2")
        (test_input_dir / "subdir").mkdir()
        (test_input_dir / "subdir" / "file3.json").write_text("content3")

        # Mock the contract to use our temp directory
        contract = MockContract()
        contract.expected_input_paths["test_input"] = str(test_input_dir)

        # Test finding all files
        files = find_files_in_input(contract, "test_input", "*")
        self.assertEqual(len(files), 2)  # Only files, not directories
        
        # Test finding specific pattern
        csv_files = find_files_in_input(contract, "test_input", "*.csv")
        self.assertEqual(len(csv_files), 1)
        self.assertTrue(csv_files[0].endswith("file2.csv"))

    def test_find_files_in_input_nonexistent_directory(self):
        """Test finding files in nonexistent directory."""
        contract = MockContract()
        contract.expected_input_paths["nonexistent"] = "/nonexistent/path"
        
        files = find_files_in_input(contract, "nonexistent", "*")
        self.assertEqual(files, [])

    @patch('src.cursus.steps.scripts.contract_utils.os.makedirs')
    def test_create_output_file_path(self, mock_makedirs):
        """Test creating output file path."""
        file_path = create_output_file_path(self.contract, 'metrics_output', 'results.json')
        
        expected_path = "/opt/ml/processing/output/metrics/results.json"
        self.assertEqual(file_path, expected_path)
        mock_makedirs.assert_called_once_with("/opt/ml/processing/output/metrics", exist_ok=True)

    @patch('src.cursus.steps.scripts.contract_utils.logger')
    def test_log_contract_summary(self, mock_logger):
        """Test logging contract summary."""
        with patch.dict(os.environ, {
            'REQUIRED_VAR1': 'value1',
            'REQUIRED_VAR2': 'value2',
            'OPTIONAL_VAR1': 'custom_value'
        }):
            log_contract_summary(self.contract)
        
        # Verify that logger.info was called multiple times
        self.assertTrue(mock_logger.info.called)
        call_args = [call[0][0] for call in mock_logger.info.call_args_list]
        
        # Check that key information was logged
        self.assertTrue(any("Contract Summary" in arg for arg in call_args))
        self.assertTrue(any("test_script.py" in arg for arg in call_args))
        self.assertTrue(any("data_input" in arg for arg in call_args))


class TestContractEnvironmentValidation(unittest.TestCase):
    """Tests for contract environment validation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.contract = MockContract()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('src.cursus.steps.scripts.contract_utils.os.path.exists')
    @patch('src.cursus.steps.scripts.contract_utils.os.makedirs')
    def test_validate_contract_environment_success(self, mock_makedirs, mock_exists):
        """Test successful contract environment validation."""
        # Mock input paths exist
        mock_exists.return_value = True
        
        with patch.dict(os.environ, {
            'REQUIRED_VAR1': 'value1',
            'REQUIRED_VAR2': 'value2'
        }, clear=True):
            # Should not raise any exception
            validate_contract_environment(self.contract)
        
        # Verify output directories were created
        self.assertEqual(mock_makedirs.call_count, 2)

    def test_validate_contract_environment_missing_required_var(self):
        """Test validation failure with missing required environment variable."""
        with patch.dict(os.environ, {'REQUIRED_VAR1': 'value1'}, clear=True):
            with self.assertRaises(RuntimeError) as context:
                validate_contract_environment(self.contract)
            
            self.assertIn("Missing required environment variable: REQUIRED_VAR2", str(context.exception))

    @patch('src.cursus.steps.scripts.contract_utils.os.path.exists')
    def test_validate_contract_environment_missing_input_path(self, mock_exists):
        """Test validation failure with missing input path."""
        # Mock that input paths don't exist
        mock_exists.return_value = False
        
        with patch.dict(os.environ, {
            'REQUIRED_VAR1': 'value1',
            'REQUIRED_VAR2': 'value2'
        }, clear=True):
            with self.assertRaises(RuntimeError) as context:
                validate_contract_environment(self.contract)
            
            self.assertIn("Input path not found", str(context.exception))

    @patch('src.cursus.steps.scripts.contract_utils.os.path.exists')
    @patch('src.cursus.steps.scripts.contract_utils.os.makedirs')
    def test_validate_contract_environment_sets_defaults(self, mock_makedirs, mock_exists):
        """Test that optional environment variables get default values."""
        mock_exists.return_value = True
        
        with patch.dict(os.environ, {
            'REQUIRED_VAR1': 'value1',
            'REQUIRED_VAR2': 'value2'
        }, clear=True):
            validate_contract_environment(self.contract)
            
            # Check that defaults were set
            self.assertEqual(os.environ['OPTIONAL_VAR1'], 'default1')
            self.assertEqual(os.environ['OPTIONAL_VAR2'], 'default2')

    def test_validate_required_files_success(self):
        """Test successful required files validation."""
        # Create test files
        test_dir = self.temp_dir / "test_input"
        test_dir.mkdir()
        (test_dir / "required_file.txt").write_text("content")
        
        contract = MockContract()
        contract.expected_input_paths["test_input"] = str(test_dir)
        
        required_files = {"test_input": ["required_file.txt"]}
        
        # Should not raise any exception
        validate_required_files(contract, required_files)

    def test_validate_required_files_missing_file(self):
        """Test required files validation with missing file."""
        test_dir = self.temp_dir / "test_input"
        test_dir.mkdir()
        
        contract = MockContract()
        contract.expected_input_paths["test_input"] = str(test_dir)
        
        required_files = {"test_input": ["missing_file.txt"]}
        
        with self.assertRaises(RuntimeError) as context:
            validate_required_files(contract, required_files)
        
        self.assertIn("Required file not found", str(context.exception))

    def test_validate_required_files_unknown_input(self):
        """Test required files validation with unknown input."""
        required_files = {"unknown_input": ["some_file.txt"]}
        
        with self.assertRaises(RuntimeError) as context:
            validate_required_files(self.contract, required_files)
        
        self.assertIn("Unknown input 'unknown_input'", str(context.exception))

    def test_validate_required_files_none(self):
        """Test required files validation with None input."""
        # Should not raise any exception
        validate_required_files(self.contract, None)


class TestFrameworkValidation(unittest.TestCase):
    """Tests for framework requirements validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.contract = MockContract()

    def test_validate_framework_requirements_success(self):
        """Test successful framework validation."""
        # pandas and numpy should be available in test environment
        # Should not raise any exception
        validate_framework_requirements(self.contract)

    def test_validate_framework_requirements_missing_framework(self):
        """Test framework validation with missing framework."""
        contract = MockContract()
        contract.framework_requirements = {"nonexistent_framework": ">=1.0.0"}
        
        with self.assertRaises(RuntimeError) as context:
            validate_framework_requirements(contract)
        
        self.assertIn("Required framework not available: nonexistent_framework", str(context.exception))


class TestContractEnforcer(unittest.TestCase):
    """Tests for the ContractEnforcer context manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.contract = MockContract()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('src.cursus.steps.scripts.contract_utils.validate_contract_environment')
    @patch('src.cursus.steps.scripts.contract_utils.validate_framework_requirements')
    @patch('src.cursus.steps.scripts.contract_utils.validate_required_files')
    @patch('src.cursus.steps.scripts.contract_utils.log_contract_summary')
    def test_contract_enforcer_success(self, mock_log, mock_validate_files, mock_validate_frameworks, mock_validate_env):
        """Test successful contract enforcement."""
        with ContractEnforcer(self.contract) as enforcer:
            # Test that we can access enforcer methods
            self.assertIsInstance(enforcer, ContractEnforcer)
            
            # Test enforcer methods
            path = enforcer.get_input_path('data_input')
            self.assertEqual(path, "/opt/ml/processing/input/data")
        
        # Verify all validations were called
        mock_log.assert_called_once()
        mock_validate_env.assert_called_once()
        mock_validate_frameworks.assert_called_once()
        mock_validate_files.assert_called_once_with(self.contract, None)

    @patch('src.cursus.steps.scripts.contract_utils.validate_contract_environment')
    @patch('src.cursus.steps.scripts.contract_utils.validate_framework_requirements')
    @patch('src.cursus.steps.scripts.contract_utils.validate_required_files')
    @patch('src.cursus.steps.scripts.contract_utils.log_contract_summary')
    def test_contract_enforcer_with_required_files(self, mock_log, mock_validate_files, mock_validate_frameworks, mock_validate_env):
        """Test contract enforcement with required files."""
        required_files = {"data_input": ["data.csv"]}
        
        with ContractEnforcer(self.contract, required_files) as enforcer:
            pass
        
        # Verify required files validation was called with the right argument
        mock_validate_files.assert_called_once_with(self.contract, required_files)

    @patch('src.cursus.steps.scripts.contract_utils.validate_contract_environment')
    @patch('src.cursus.steps.scripts.contract_utils.validate_framework_requirements')
    @patch('src.cursus.steps.scripts.contract_utils.validate_required_files')
    @patch('src.cursus.steps.scripts.contract_utils.log_contract_summary')
    def test_contract_enforcer_exception_handling(self, mock_log, mock_validate_files, mock_validate_frameworks, mock_validate_env):
        """Test contract enforcer handles exceptions properly."""
        try:
            with ContractEnforcer(self.contract) as enforcer:
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
        
        # Verify validations were still called
        mock_log.assert_called_once()
        mock_validate_env.assert_called_once()

    @patch('src.cursus.steps.scripts.contract_utils.os.makedirs')
    def test_contract_enforcer_methods(self, mock_makedirs):
        """Test ContractEnforcer methods work correctly."""
        with patch('src.cursus.steps.scripts.contract_utils.validate_contract_environment'), \
             patch('src.cursus.steps.scripts.contract_utils.validate_framework_requirements'), \
             patch('src.cursus.steps.scripts.contract_utils.validate_required_files'), \
             patch('src.cursus.steps.scripts.contract_utils.log_contract_summary'):
            
            with ContractEnforcer(self.contract) as enforcer:
                # Test get_input_path
                input_path = enforcer.get_input_path('data_input')
                self.assertEqual(input_path, "/opt/ml/processing/input/data")
                
                # Test get_output_path
                output_path = enforcer.get_output_path('processed_output')
                self.assertEqual(output_path, "/opt/ml/processing/output/processed")
                
                # Test create_output_file_path
                file_path = enforcer.create_output_file_path('metrics_output', 'results.json')
                self.assertEqual(file_path, "/opt/ml/processing/output/metrics/results.json")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
