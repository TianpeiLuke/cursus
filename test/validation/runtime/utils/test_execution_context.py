"""Unit tests for ExecutionContext."""

import unittest
import argparse
from typing import Dict

from src.cursus.validation.runtime.utils.execution_context import ExecutionContext


class TestExecutionContext(unittest.TestCase):
    """Test cases for ExecutionContext model."""
    
    def test_execution_context_creation_minimal(self):
        """Test creating ExecutionContext with minimal required fields."""
        input_paths = {"input": "/path/to/input"}
        output_paths = {"output": "/path/to/output"}
        environ_vars = {"VAR1": "value1", "VAR2": "value2"}
        
        context = ExecutionContext(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars
        )
        
        self.assertEqual(context.input_paths, input_paths)
        self.assertEqual(context.output_paths, output_paths)
        self.assertEqual(context.environ_vars, environ_vars)
        self.assertIsNone(context.job_args)
    
    def test_execution_context_creation_with_job_args(self):
        """Test creating ExecutionContext with job_args."""
        input_paths = {"input": "/path/to/input"}
        output_paths = {"output": "/path/to/output"}
        environ_vars = {"VAR1": "value1"}
        
        # Create argparse.Namespace
        job_args = argparse.Namespace()
        job_args.verbose = True
        job_args.batch_size = 32
        job_args.learning_rate = 0.001
        
        context = ExecutionContext(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=job_args
        )
        
        self.assertEqual(context.input_paths, input_paths)
        self.assertEqual(context.output_paths, output_paths)
        self.assertEqual(context.environ_vars, environ_vars)
        self.assertEqual(context.job_args, job_args)
        self.assertTrue(context.job_args.verbose)
        self.assertEqual(context.job_args.batch_size, 32)
        self.assertEqual(context.job_args.learning_rate, 0.001)
    
    def test_execution_context_empty_paths(self):
        """Test ExecutionContext with empty path dictionaries."""
        context = ExecutionContext(
            input_paths={},
            output_paths={},
            environ_vars={}
        )
        
        self.assertEqual(context.input_paths, {})
        self.assertEqual(context.output_paths, {})
        self.assertEqual(context.environ_vars, {})
        self.assertIsNone(context.job_args)
    
    def test_execution_context_multiple_paths(self):
        """Test ExecutionContext with multiple input/output paths."""
        input_paths = {
            "train_data": "/path/to/train.csv",
            "test_data": "/path/to/test.csv",
            "config": "/path/to/config.json"
        }
        output_paths = {
            "model": "/path/to/model.pkl",
            "metrics": "/path/to/metrics.json",
            "predictions": "/path/to/predictions.csv"
        }
        environ_vars = {
            "AWS_REGION": "us-west-2",
            "S3_BUCKET": "my-bucket",
            "LOG_LEVEL": "INFO"
        }
        
        context = ExecutionContext(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars
        )
        
        self.assertEqual(len(context.input_paths), 3)
        self.assertEqual(len(context.output_paths), 3)
        self.assertEqual(len(context.environ_vars), 3)
        
        self.assertEqual(context.input_paths["train_data"], "/path/to/train.csv")
        self.assertEqual(context.output_paths["model"], "/path/to/model.pkl")
        self.assertEqual(context.environ_vars["AWS_REGION"], "us-west-2")
    
    def test_execution_context_complex_job_args(self):
        """Test ExecutionContext with complex job_args."""
        input_paths = {"input": "/path/to/input"}
        output_paths = {"output": "/path/to/output"}
        environ_vars = {"VAR": "value"}
        
        # Create complex argparse.Namespace
        job_args = argparse.Namespace()
        job_args.model_type = "xgboost"
        job_args.hyperparameters = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        }
        job_args.feature_columns = ["col1", "col2", "col3"]
        job_args.target_column = "target"
        job_args.cross_validation = True
        job_args.random_seed = 42
        
        context = ExecutionContext(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=job_args
        )
        
        self.assertEqual(context.job_args.model_type, "xgboost")
        self.assertEqual(context.job_args.hyperparameters["n_estimators"], 100)
        self.assertEqual(len(context.job_args.feature_columns), 3)
        self.assertEqual(context.job_args.target_column, "target")
        self.assertTrue(context.job_args.cross_validation)
        self.assertEqual(context.job_args.random_seed, 42)
    
    def test_execution_context_serialization(self):
        """Test ExecutionContext serialization to dict."""
        input_paths = {"input": "/path/to/input"}
        output_paths = {"output": "/path/to/output"}
        environ_vars = {"VAR": "value"}
        
        job_args = argparse.Namespace()
        job_args.verbose = True
        job_args.batch_size = 16
        
        context = ExecutionContext(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=job_args
        )
        
        # Test Pydantic v2 serialization
        context_dict = context.model_dump()
        
        self.assertIsInstance(context_dict, dict)
        self.assertEqual(context_dict["input_paths"], input_paths)
        self.assertEqual(context_dict["output_paths"], output_paths)
        self.assertEqual(context_dict["environ_vars"], environ_vars)
        self.assertEqual(context_dict["job_args"], job_args)
    
    def test_execution_context_serialization_none_job_args(self):
        """Test ExecutionContext serialization when job_args is None."""
        input_paths = {"input": "/path/to/input"}
        output_paths = {"output": "/path/to/output"}
        environ_vars = {"VAR": "value"}
        
        context = ExecutionContext(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars
        )
        
        context_dict = context.model_dump()
        
        self.assertIsInstance(context_dict, dict)
        self.assertEqual(context_dict["input_paths"], input_paths)
        self.assertEqual(context_dict["output_paths"], output_paths)
        self.assertEqual(context_dict["environ_vars"], environ_vars)
        self.assertIsNone(context_dict["job_args"])
    
    def test_execution_context_path_validation(self):
        """Test ExecutionContext with various path formats."""
        # Test with absolute paths
        context = ExecutionContext(
            input_paths={"input": "/absolute/path/to/input"},
            output_paths={"output": "/absolute/path/to/output"},
            environ_vars={}
        )
        self.assertTrue(context.input_paths["input"].startswith("/"))
        self.assertTrue(context.output_paths["output"].startswith("/"))
        
        # Test with relative paths
        context = ExecutionContext(
            input_paths={"input": "relative/path/to/input"},
            output_paths={"output": "relative/path/to/output"},
            environ_vars={}
        )
        self.assertFalse(context.input_paths["input"].startswith("/"))
        self.assertFalse(context.output_paths["output"].startswith("/"))
        
        # Test with mixed path formats
        context = ExecutionContext(
            input_paths={
                "abs_input": "/absolute/input",
                "rel_input": "relative/input"
            },
            output_paths={
                "abs_output": "/absolute/output",
                "rel_output": "relative/output"
            },
            environ_vars={}
        )
        self.assertTrue(context.input_paths["abs_input"].startswith("/"))
        self.assertFalse(context.input_paths["rel_input"].startswith("/"))
        self.assertTrue(context.output_paths["abs_output"].startswith("/"))
        self.assertFalse(context.output_paths["rel_output"].startswith("/"))
    
    def test_execution_context_environ_vars_types(self):
        """Test ExecutionContext with different environment variable types."""
        # All environment variables should be strings
        environ_vars = {
            "STRING_VAR": "string_value",
            "NUMBER_VAR": "123",  # Should be string, not int
            "BOOLEAN_VAR": "true",  # Should be string, not bool
            "EMPTY_VAR": "",
            "PATH_VAR": "/path/with/spaces and special chars!@#"
        }
        
        context = ExecutionContext(
            input_paths={"input": "/path"},
            output_paths={"output": "/path"},
            environ_vars=environ_vars
        )
        
        # All values should remain as strings
        for key, value in context.environ_vars.items():
            self.assertIsInstance(value, str)
        
        self.assertEqual(context.environ_vars["STRING_VAR"], "string_value")
        self.assertEqual(context.environ_vars["NUMBER_VAR"], "123")
        self.assertEqual(context.environ_vars["BOOLEAN_VAR"], "true")
        self.assertEqual(context.environ_vars["EMPTY_VAR"], "")
        self.assertEqual(context.environ_vars["PATH_VAR"], "/path/with/spaces and special chars!@#")
    
    def test_execution_context_job_args_edge_cases(self):
        """Test ExecutionContext with edge cases for job_args."""
        input_paths = {"input": "/path"}
        output_paths = {"output": "/path"}
        environ_vars = {}
        
        # Test with empty Namespace
        empty_args = argparse.Namespace()
        context = ExecutionContext(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=empty_args
        )
        self.assertEqual(context.job_args, empty_args)
        
        # Test with Namespace containing None values
        none_args = argparse.Namespace()
        none_args.param1 = None
        none_args.param2 = "value"
        none_args.param3 = None
        
        context = ExecutionContext(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=none_args
        )
        self.assertIsNone(context.job_args.param1)
        self.assertEqual(context.job_args.param2, "value")
        self.assertIsNone(context.job_args.param3)


if __name__ == '__main__':
    unittest.main()
