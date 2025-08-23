"""Unit tests for result models."""

import unittest
from datetime import datetime
from typing import Any

from src.cursus.validation.runtime.utils.result_models import ExecutionResult, TestResult


class TestExecutionResult(unittest.TestCase):
    """Test cases for ExecutionResult model."""
    
    def test_execution_result_creation_minimal(self):
        """Test creating ExecutionResult with minimal required fields."""
        result = ExecutionResult(
            success=True,
            execution_time=1.5,
            memory_usage=100
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.execution_time, 1.5)
        self.assertEqual(result.memory_usage, 100)
        self.assertIsNone(result.result_data)
        self.assertIsNone(result.error_message)
        self.assertIsNone(result.stack_trace)
    
    def test_execution_result_creation_full(self):
        """Test creating ExecutionResult with all fields."""
        result_data = {"status": "completed", "output_files": ["file1.txt", "file2.csv"]}
        error_message = "Warning: deprecated function used"
        stack_trace = "Traceback (most recent call last):\n  File..."
        
        result = ExecutionResult(
            success=True,
            execution_time=2.3,
            memory_usage=256,
            result_data=result_data,
            error_message=error_message,
            stack_trace=stack_trace
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.execution_time, 2.3)
        self.assertEqual(result.memory_usage, 256)
        self.assertEqual(result.result_data, result_data)
        self.assertEqual(result.error_message, error_message)
        self.assertEqual(result.stack_trace, stack_trace)
    
    def test_execution_result_failure(self):
        """Test creating ExecutionResult for failed execution."""
        result = ExecutionResult(
            success=False,
            execution_time=0.5,
            memory_usage=50,
            error_message="Script execution failed",
            stack_trace="Traceback..."
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Script execution failed")
        self.assertEqual(result.stack_trace, "Traceback...")
    
    def test_execution_result_validation(self):
        """Test ExecutionResult field validation."""
        # Test with negative execution time
        result = ExecutionResult(
            success=True,
            execution_time=-1.0,  # Negative time should be allowed (edge case)
            memory_usage=100
        )
        self.assertEqual(result.execution_time, -1.0)
        
        # Test with zero values
        result = ExecutionResult(
            success=False,
            execution_time=0.0,
            memory_usage=0
        )
        self.assertEqual(result.execution_time, 0.0)
        self.assertEqual(result.memory_usage, 0)
    
    def test_execution_result_serialization(self):
        """Test ExecutionResult serialization to dict."""
        result = ExecutionResult(
            success=True,
            execution_time=1.5,
            memory_usage=100,
            result_data={"key": "value"}
        )
        
        # Test Pydantic v2 serialization
        result_dict = result.model_dump()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["success"], True)
        self.assertEqual(result_dict["execution_time"], 1.5)
        self.assertEqual(result_dict["memory_usage"], 100)
        self.assertEqual(result_dict["result_data"], {"key": "value"})
        self.assertIsNone(result_dict["error_message"])
        self.assertIsNone(result_dict["stack_trace"])


class TestTestResult(unittest.TestCase):
    """Test cases for TestResult model."""
    
    def test_test_result_creation_minimal(self):
        """Test creating TestResult with minimal required fields."""
        result = TestResult(
            script_name="test_script",
            status="PASS",
            execution_time=1.5,
            memory_usage=100
        )
        
        self.assertEqual(result.script_name, "test_script")
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.execution_time, 1.5)
        self.assertEqual(result.memory_usage, 100)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.recommendations, [])
        self.assertIsInstance(result.timestamp, datetime)
    
    def test_test_result_creation_full(self):
        """Test creating TestResult with all fields."""
        recommendations = ["Optimize memory usage", "Add error handling"]
        error_message = "Script failed with ValueError"
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        
        result = TestResult(
            script_name="failing_script",
            status="FAIL",
            execution_time=0.8,
            memory_usage=50,
            error_message=error_message,
            recommendations=recommendations,
            timestamp=timestamp
        )
        
        self.assertEqual(result.script_name, "failing_script")
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.execution_time, 0.8)
        self.assertEqual(result.memory_usage, 50)
        self.assertEqual(result.error_message, error_message)
        self.assertEqual(result.recommendations, recommendations)
        self.assertEqual(result.timestamp, timestamp)
    
    def test_test_result_status_values(self):
        """Test TestResult with different status values."""
        statuses = ["PASS", "FAIL", "SKIP"]
        
        for status in statuses:
            result = TestResult(
                script_name="test_script",
                status=status,
                execution_time=1.0,
                memory_usage=100
            )
            self.assertEqual(result.status, status)
    
    def test_test_result_is_successful_pass(self):
        """Test is_successful method for PASS status."""
        result = TestResult(
            script_name="test_script",
            status="PASS",
            execution_time=1.0,
            memory_usage=100
        )
        
        self.assertTrue(result.is_successful())
    
    def test_test_result_is_successful_fail(self):
        """Test is_successful method for FAIL status."""
        result = TestResult(
            script_name="test_script",
            status="FAIL",
            execution_time=1.0,
            memory_usage=100
        )
        
        self.assertFalse(result.is_successful())
    
    def test_test_result_is_successful_skip(self):
        """Test is_successful method for SKIP status."""
        result = TestResult(
            script_name="test_script",
            status="SKIP",
            execution_time=0.0,
            memory_usage=0
        )
        
        self.assertFalse(result.is_successful())
    
    def test_test_result_default_timestamp(self):
        """Test that TestResult creates default timestamp."""
        before_creation = datetime.now()
        
        result = TestResult(
            script_name="test_script",
            status="PASS",
            execution_time=1.0,
            memory_usage=100
        )
        
        after_creation = datetime.now()
        
        # Timestamp should be between before and after creation
        self.assertGreaterEqual(result.timestamp, before_creation)
        self.assertLessEqual(result.timestamp, after_creation)
    
    def test_test_result_empty_recommendations(self):
        """Test TestResult with empty recommendations list."""
        result = TestResult(
            script_name="test_script",
            status="PASS",
            execution_time=1.0,
            memory_usage=100,
            recommendations=[]
        )
        
        self.assertEqual(result.recommendations, [])
    
    def test_test_result_multiple_recommendations(self):
        """Test TestResult with multiple recommendations."""
        recommendations = [
            "Optimize algorithm complexity",
            "Reduce memory footprint",
            "Add input validation",
            "Improve error messages"
        ]
        
        result = TestResult(
            script_name="test_script",
            status="FAIL",
            execution_time=5.0,
            memory_usage=1024,
            recommendations=recommendations
        )
        
        self.assertEqual(len(result.recommendations), 4)
        self.assertEqual(result.recommendations, recommendations)
    
    def test_test_result_serialization(self):
        """Test TestResult serialization to dict."""
        recommendations = ["Recommendation 1", "Recommendation 2"]
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        
        result = TestResult(
            script_name="test_script",
            status="PASS",
            execution_time=1.5,
            memory_usage=100,
            error_message="Warning message",
            recommendations=recommendations,
            timestamp=timestamp
        )
        
        # Test Pydantic v2 serialization
        result_dict = result.model_dump()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["script_name"], "test_script")
        self.assertEqual(result_dict["status"], "PASS")
        self.assertEqual(result_dict["execution_time"], 1.5)
        self.assertEqual(result_dict["memory_usage"], 100)
        self.assertEqual(result_dict["error_message"], "Warning message")
        self.assertEqual(result_dict["recommendations"], recommendations)
        self.assertEqual(result_dict["timestamp"], timestamp)
    
    def test_test_result_edge_cases(self):
        """Test TestResult with edge case values."""
        # Test with very long script name
        long_script_name = "a" * 1000
        result = TestResult(
            script_name=long_script_name,
            status="PASS",
            execution_time=0.001,  # Very short execution time
            memory_usage=1  # Very low memory usage
        )
        
        self.assertEqual(result.script_name, long_script_name)
        self.assertEqual(result.execution_time, 0.001)
        self.assertEqual(result.memory_usage, 1)
        
        # Test with very long error message
        long_error_message = "Error: " + "x" * 10000
        result = TestResult(
            script_name="test_script",
            status="FAIL",
            execution_time=100.0,  # Very long execution time
            memory_usage=10000,  # Very high memory usage
            error_message=long_error_message
        )
        
        self.assertEqual(result.error_message, long_error_message)
        self.assertEqual(result.execution_time, 100.0)
        self.assertEqual(result.memory_usage, 10000)


if __name__ == '__main__':
    unittest.main()
