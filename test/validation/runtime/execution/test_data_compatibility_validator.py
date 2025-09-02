"""Unit tests for DataCompatibilityValidator with workspace awareness."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from src.cursus.validation.runtime.execution.data_compatibility_validator import (
    DataCompatibilityValidator, DataCompatibilityReport, DataSchemaInfo
)
from src.cursus.validation.runtime.utils.error_handling import ValidationError


class TestDataCompatibilityValidator(unittest.TestCase):
    """Test cases for DataCompatibilityValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_root = tempfile.mkdtemp()
        self.validator = DataCompatibilityValidator()
        self.workspace_validator = DataCompatibilityValidator(self.workspace_root)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.workspace_root, ignore_errors=True)
    
    def test_init_without_workspace(self):
        """Test initialization without workspace context."""
        self.assertIsNone(self.validator.workspace_root)
        self.assertEqual(self.validator.cross_workspace_validations, [])
        self.assertIsNotNone(self.validator.compatibility_rules)
    
    def test_init_with_workspace(self):
        """Test initialization with workspace context."""
        self.assertEqual(self.workspace_validator.workspace_root, self.workspace_root)
        self.assertEqual(self.workspace_validator.cross_workspace_validations, [])
        self.assertIsNotNone(self.workspace_validator.compatibility_rules)
    
    def test_validate_step_transition_basic(self):
        """Test basic step transition validation without workspace context."""
        producer_output = {
            "files": {
                "output.csv": {
                    "format": "csv",
                    "size": 1024
                }
            }
        }
        
        consumer_input_spec = {
            "required_files": ["output.csv"],
            "file_formats": {"output.csv": "csv"},
            "schemas": {}
        }
        
        result = self.validator.validate_step_transition(producer_output, consumer_input_spec)
        
        self.assertIsInstance(result, DataCompatibilityReport)
        self.assertTrue(result.compatible)
        self.assertEqual(len(result.issues), 0)
        self.assertIsNone(result.workspace_context)
    
    def test_validate_step_transition_with_workspace_context(self):
        """Test step transition validation with workspace context."""
        producer_output = {
            "files": {
                "output.csv": {
                    "format": "csv",
                    "size": 1024
                }
            }
        }
        
        consumer_input_spec = {
            "required_files": ["output.csv"],
            "file_formats": {"output.csv": "csv"},
            "schemas": {}
        }
        
        producer_workspace_info = {
            "developer_id": "dev1",
            "step_name": "producer_step",
            "step_type": "processing"
        }
        
        consumer_workspace_info = {
            "developer_id": "dev2",
            "step_name": "consumer_step",
            "step_type": "training"
        }
        
        result = self.workspace_validator.validate_step_transition(
            producer_output, 
            consumer_input_spec,
            producer_workspace_info,
            consumer_workspace_info
        )
        
        self.assertIsInstance(result, DataCompatibilityReport)
        self.assertIsNotNone(result.workspace_context)
        self.assertTrue(result.workspace_context['is_cross_workspace'])
        self.assertEqual(result.workspace_context['producer_developer'], "dev1")
        self.assertEqual(result.workspace_context['consumer_developer'], "dev2")
        self.assertIn("Cross-workspace dependency: dev1 -> dev2", result.warnings)
    
    def test_validate_step_transition_same_workspace(self):
        """Test step transition validation within same workspace."""
        producer_output = {
            "files": {
                "output.csv": {
                    "format": "csv",
                    "size": 1024
                }
            }
        }
        
        consumer_input_spec = {
            "required_files": ["output.csv"],
            "file_formats": {"output.csv": "csv"},
            "schemas": {}
        }
        
        producer_workspace_info = {
            "developer_id": "dev1",
            "step_name": "producer_step",
            "step_type": "processing"
        }
        
        consumer_workspace_info = {
            "developer_id": "dev1",
            "step_name": "consumer_step",
            "step_type": "training"
        }
        
        result = self.workspace_validator.validate_step_transition(
            producer_output, 
            consumer_input_spec,
            producer_workspace_info,
            consumer_workspace_info
        )
        
        self.assertIsInstance(result, DataCompatibilityReport)
        self.assertIsNotNone(result.workspace_context)
        self.assertFalse(result.workspace_context['is_cross_workspace'])
        self.assertEqual(len([w for w in result.warnings if "Cross-workspace" in w]), 0)
    
    def test_validate_step_transition_missing_files(self):
        """Test step transition validation with missing required files."""
        producer_output = {
            "files": {
                "wrong_output.csv": {
                    "format": "csv",
                    "size": 1024
                }
            }
        }
        
        consumer_input_spec = {
            "required_files": ["required_output.csv"],
            "file_formats": {"required_output.csv": "csv"},
            "schemas": {}
        }
        
        result = self.validator.validate_step_transition(producer_output, consumer_input_spec)
        
        self.assertFalse(result.compatible)
        self.assertIn("Missing required file: required_output.csv", result.issues)
    
    def test_validate_step_transition_format_mismatch(self):
        """Test step transition validation with format mismatch."""
        producer_output = {
            "files": {
                "output.csv": {
                    "format": "json",  # Wrong format
                    "size": 1024
                }
            }
        }
        
        consumer_input_spec = {
            "required_files": ["output.csv"],
            "file_formats": {"output.csv": "csv"},
            "schemas": {}
        }
        
        result = self.validator.validate_step_transition(producer_output, consumer_input_spec)
        
        self.assertFalse(result.compatible)
        self.assertIn("Format mismatch for output.csv: expected csv, got json", result.issues)
    
    def test_validate_step_transition_empty_producer_output(self):
        """Test step transition validation with empty producer output."""
        with self.assertRaises(ValidationError) as context:
            self.validator.validate_step_transition({}, {"required_files": []})
        
        self.assertIn("Producer output cannot be empty", str(context.exception))
    
    def test_validate_step_transition_empty_consumer_spec(self):
        """Test step transition validation with empty consumer spec."""
        producer_output = {"files": {"output.csv": {"format": "csv"}}}
        
        with self.assertRaises(ValidationError) as context:
            self.validator.validate_step_transition(producer_output, {})
        
        self.assertIn("Consumer input specification cannot be empty", str(context.exception))
    
    def test_perform_cross_workspace_checks(self):
        """Test cross-workspace specific validation checks."""
        producer_info = {
            "developer_id": "dev_with_underscores",
            "step_name": "producer_step",
            "step_type": "processing"
        }
        
        consumer_info = {
            "developer_id": "dev-with-hyphens",
            "step_name": "consumer_step",
            "step_type": "training"
        }
        
        issues = self.workspace_validator._perform_cross_workspace_checks(producer_info, consumer_info)
        
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("step type compatibility warning" in issue for issue in issues))
        self.assertTrue(any("Naming convention mismatch" in issue for issue in issues))
    
    def test_cross_workspace_validation_tracking(self):
        """Test that cross-workspace validations are tracked."""
        producer_output = {"files": {"output.csv": {"format": "csv"}}}
        consumer_input_spec = {"required_files": ["output.csv"], "file_formats": {"output.csv": "csv"}}
        
        producer_workspace_info = {"developer_id": "dev1", "step_name": "step1", "step_type": "processing"}
        consumer_workspace_info = {"developer_id": "dev2", "step_name": "step2", "step_type": "training"}
        
        # Perform validation
        result = self.workspace_validator.validate_step_transition(
            producer_output, consumer_input_spec, producer_workspace_info, consumer_workspace_info
        )
        
        # Check that validation was tracked
        self.assertEqual(len(self.workspace_validator.cross_workspace_validations), 1)
        validation_entry = self.workspace_validator.cross_workspace_validations[0]
        self.assertEqual(validation_entry['producer_step'], 'step1')
        self.assertEqual(validation_entry['producer_developer'], 'dev1')
        self.assertEqual(validation_entry['consumer_step'], 'step2')
        self.assertEqual(validation_entry['consumer_developer'], 'dev2')
    
    @patch('src.cursus.validation.runtime.execution.data_compatibility_validator.pd.read_csv')
    def test_analyze_csv_file(self, mock_read_csv):
        """Test CSV file analysis."""
        # Mock pandas DataFrame
        mock_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        mock_read_csv.return_value = mock_df
        
        test_file = Path(self.temp_dir) / "test.csv"
        test_file.write_text("col1,col2\n1,a\n2,b\n3,c\n")
        
        result = self.validator.analyze_file(test_file)
        
        self.assertIsInstance(result, DataSchemaInfo)
        self.assertEqual(result.data_format, "csv")
        self.assertEqual(len(result.columns), 2)
        self.assertIn("col1", result.columns)
        self.assertIn("col2", result.columns)
        self.assertEqual(result.num_rows, 3)
    
    def test_analyze_file_nonexistent(self):
        """Test file analysis with non-existent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.csv"
        
        with self.assertRaises(ValidationError) as context:
            self.validator.analyze_file(nonexistent_file)
        
        self.assertIn("File does not exist", str(context.exception))
    
    def test_check_compatibility_same_format(self):
        """Test compatibility check with same data formats."""
        producer_schema = DataSchemaInfo(
            columns=["id", "name"],
            column_types={"id": "int", "name": "string"},
            data_format="csv"
        )
        
        consumer_schema = DataSchemaInfo(
            columns=["id", "name"],
            column_types={"id": "int", "name": "string"},
            data_format="csv"
        )
        
        result = self.validator.check_compatibility(producer_schema, consumer_schema)
        
        self.assertTrue(result.compatible)
        self.assertEqual(len(result.issues), 0)
    
    def test_check_compatibility_missing_columns(self):
        """Test compatibility check with missing columns."""
        producer_schema = DataSchemaInfo(
            columns=["id"],
            column_types={"id": "int"},
            data_format="csv"
        )
        
        consumer_schema = DataSchemaInfo(
            columns=["id", "name"],
            column_types={"id": "int", "name": "string"},
            data_format="csv"
        )
        
        result = self.validator.check_compatibility(producer_schema, consumer_schema)
        
        self.assertFalse(result.compatible)
        self.assertIn("Missing required columns: name", result.issues)
    
    def test_check_compatibility_type_mismatch(self):
        """Test compatibility check with type mismatches."""
        producer_schema = DataSchemaInfo(
            columns=["id", "name"],
            column_types={"id": "string", "name": "string"},
            data_format="csv"
        )
        
        consumer_schema = DataSchemaInfo(
            columns=["id", "name"],
            column_types={"id": "int", "name": "string"},
            data_format="csv"
        )
        
        result = self.validator.check_compatibility(producer_schema, consumer_schema)
        
        self.assertFalse(result.compatible)
        self.assertIn("Incompatible types for column 'id': string -> int", result.issues)
    
    def test_check_compatibility_format_conversion(self):
        """Test compatibility check with format conversion."""
        producer_schema = DataSchemaInfo(
            columns=["id", "name"],
            column_types={"id": "int", "name": "string"},
            data_format="parquet"
        )
        
        consumer_schema = DataSchemaInfo(
            columns=["id", "name"],
            column_types={"id": "int", "name": "string"},
            data_format="csv"
        )
        
        result = self.validator.check_compatibility(producer_schema, consumer_schema)
        
        self.assertTrue(result.compatible)
        self.assertIn("Lossy conversion from parquet to csv", result.warnings)
    
    def test_are_types_compatible(self):
        """Test type compatibility checking."""
        # Same types
        self.assertTrue(self.validator._are_types_compatible("int", "int"))
        
        # Numeric compatibility
        self.assertTrue(self.validator._are_types_compatible("int", "float"))
        self.assertTrue(self.validator._are_types_compatible("int32", "float64"))
        
        # String compatibility
        self.assertTrue(self.validator._are_types_compatible("int", "string"))
        self.assertTrue(self.validator._are_types_compatible("float", "object"))
        
        # Incompatible types
        self.assertFalse(self.validator._are_types_compatible("string", "int"))
        self.assertFalse(self.validator._are_types_compatible("float", "bool"))
    
    def test_get_cross_workspace_validation_summary(self):
        """Test cross-workspace validation summary."""
        # Test without workspace context
        result_no_workspace = self.validator.get_cross_workspace_validation_summary()
        self.assertIn("error", result_no_workspace)
        
        # Add some mock validations
        self.workspace_validator.cross_workspace_validations = [
            {
                'producer_step': 'step1',
                'producer_developer': 'dev1',
                'consumer_step': 'step2',
                'consumer_developer': 'dev2',
                'validation_result': 'passed'
            },
            {
                'producer_step': 'step3',
                'producer_developer': 'dev1',
                'consumer_step': 'step4',
                'consumer_developer': 'dev2',
                'validation_result': 'failed',
                'issues': ['Format mismatch']
            }
        ]
        
        result = self.workspace_validator.get_cross_workspace_validation_summary()
        
        self.assertEqual(result['workspace_root'], self.workspace_root)
        self.assertEqual(result['total_validations'], 2)
        self.assertEqual(result['successful_validations'], 1)
        self.assertEqual(result['failed_validations'], 1)
        self.assertIn('dev1->dev2', result['developer_pairs'])
        self.assertEqual(result['developer_pairs']['dev1->dev2']['total'], 2)
        self.assertEqual(result['developer_pairs']['dev1->dev2']['passed'], 1)
        self.assertEqual(result['developer_pairs']['dev1->dev2']['failed'], 1)
        self.assertIn('Format mismatch', result['common_issues'])
    
    @patch('src.cursus.api.dag.workspace_dag.WorkspaceAwareDAG')
    def test_validate_workspace_data_flow(self, mock_dag_class):
        """Test workspace data flow validation."""
        # Test without workspace context
        result_no_workspace = self.validator.validate_workspace_data_flow(None, {})
        self.assertIn("error", result_no_workspace)
        
        # Mock WorkspaceAwareDAG
        mock_dag = Mock()
        mock_dag.validate_workspace_dependencies.return_value = {
            'cross_workspace_dependencies': [
                {
                    'dependency_step': 'producer_step',
                    'dependent_step': 'consumer_step',
                    'dependency_developer': 'dev1',
                    'dependent_developer': 'dev2'
                }
            ]
        }
        
        step_outputs = {
            'producer_step': {
                'output': {
                    'format': 'csv',
                    'path': '/path/to/output.csv'
                }
            }
        }
        
        result = self.workspace_validator.validate_workspace_data_flow(mock_dag, step_outputs)
        
        self.assertEqual(result['workspace_root'], self.workspace_root)
        self.assertEqual(result['total_cross_workspace_dependencies'], 1)
        self.assertEqual(len(result['validation_results']), 1)
        self.assertIn('dev1->dev2', result['developer_compatibility_matrix'])
        self.assertGreater(len(result['recommendations']), 0)
    
    def test_infer_type(self):
        """Test data type inference."""
        # Integer values
        self.assertEqual(self.validator._infer_type([1, 2, 3]), 'int')

        # Float values
        self.assertEqual(self.validator._infer_type([1.0, 2.5, 3.7]), 'float')

        # Mixed numeric values
        self.assertEqual(self.validator._infer_type([1, 2.5, 3]), 'float')

        # Boolean values - Note: In Python, bool is a subclass of int, so [True, False, True] infers as 'int'
        self.assertEqual(self.validator._infer_type([True, False, True]), 'int')

        # String values
        self.assertEqual(self.validator._infer_type(['a', 'b', 'c']), 'string')

        # Mixed types default to string
        self.assertEqual(self.validator._infer_type([1, 'a', True]), 'string')

        # With None values
        self.assertEqual(self.validator._infer_type([1, 2, None, 3]), 'int')
    
    def test_load_compatibility_rules(self):
        """Test loading of compatibility rules."""
        rules = self.validator._load_compatibility_rules()
        
        self.assertIn('csv', rules)
        self.assertIn('parquet', rules)
        self.assertIn('json', rules)
        
        # Check CSV rules
        csv_rules = rules['csv']
        self.assertIn('compatible_with', csv_rules)
        self.assertIn('conversion', csv_rules)
        self.assertIn('parquet', csv_rules['compatible_with'])
    
    def test_create_data_summary(self):
        """Test creation of data summary."""
        output = {
            "files": {
                "file1.csv": {
                    "format": "csv",
                    "size": 1024,
                    "schema": {
                        "columns": ["col1", "col2"],
                        "num_rows": 100
                    }
                },
                "file2.json": {
                    "format": "json",
                    "size": 512
                }
            }
        }
        
        summary = self.validator._create_data_summary(output)
        
        self.assertIn("file1.csv", summary)
        self.assertIn("file2.json", summary)
        
        file1_summary = summary["file1.csv"]
        self.assertEqual(file1_summary["format"], "csv")
        self.assertEqual(file1_summary["size"], 1024)
        self.assertEqual(file1_summary["columns"], 2)
        self.assertEqual(file1_summary["rows"], 100)
        
        file2_summary = summary["file2.json"]
        self.assertEqual(file2_summary["format"], "json")
        self.assertEqual(file2_summary["size"], 512)
        self.assertNotIn("columns", file2_summary)


if __name__ == '__main__':
    unittest.main()
