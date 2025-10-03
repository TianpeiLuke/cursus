"""
Test module for validation alignment utilities.

Tests the utility functions used throughout the validation alignment system
including file operations, path resolution, and helper functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import os

from cursus.validation.alignment.utils.utils import (
    find_builder_class,
    resolve_workspace_path,
    get_step_name_from_path,
    extract_step_type_from_name,
    normalize_step_name,
    validate_workspace_structure,
    get_builder_file_path,
    load_builder_module,
    extract_class_methods,
    get_method_signature,
    analyze_method_implementation,
    get_file_modification_time,
    ensure_directory_exists,
    safe_file_read,
    safe_file_write,
    get_relative_path,
    is_valid_python_identifier,
    sanitize_step_name,
    get_workspace_root,
    find_files_by_pattern,
    get_python_files_in_directory,
    extract_imports_from_file,
    get_class_hierarchy,
    validate_python_syntax,
    format_validation_message,
    create_validation_issue,
    merge_validation_results,
    calculate_validation_score,
    get_validation_summary
)


class TestValidationAlignmentUtils:
    """Test cases for validation alignment utility functions."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create workspace structure
            workspace_path = Path(temp_dir) / "test_workspace"
            workspace_path.mkdir()
            
            # Create step builder files
            step_dir = workspace_path / "steps"
            step_dir.mkdir()
            
            # Create a sample step builder file
            builder_file = step_dir / "processing_step_builder.py"
            builder_content = '''
class ProcessingStepBuilder:
    def validate_configuration(self):
        pass
    
    def _get_inputs(self):
        return {}
    
    def create_step(self):
        pass
    
    def _create_processor(self):
        return {"processor_type": "ScriptProcessor"}
'''
            builder_file.write_text(builder_content)
            
            yield workspace_path

    def test_find_builder_class_with_valid_step(self, temp_workspace):
        """Test find_builder_class with valid step name."""
        step_name = "processing_step"
        workspace_dirs = [str(temp_workspace)]
        
        builder_class = find_builder_class(step_name, workspace_dirs)
        assert builder_class is not None
        assert hasattr(builder_class, 'validate_configuration')
        assert hasattr(builder_class, '_get_inputs')
        assert hasattr(builder_class, 'create_step')

    def test_find_builder_class_with_invalid_step(self, temp_workspace):
        """Test find_builder_class with invalid step name."""
        step_name = "nonexistent_step"
        workspace_dirs = [str(temp_workspace)]
        
        builder_class = find_builder_class(step_name, workspace_dirs)
        assert builder_class is None

    def test_find_builder_class_with_empty_workspace_dirs(self):
        """Test find_builder_class with empty workspace directories."""
        step_name = "processing_step"
        workspace_dirs = []
        
        builder_class = find_builder_class(step_name, workspace_dirs)
        assert builder_class is None

    def test_resolve_workspace_path_with_absolute_path(self):
        """Test resolve_workspace_path with absolute path."""
        absolute_path = "/absolute/path/to/workspace"
        workspace_dirs = ["/workspace1", "/workspace2"]
        
        resolved_path = resolve_workspace_path(absolute_path, workspace_dirs)
        assert resolved_path == Path(absolute_path)

    def test_resolve_workspace_path_with_relative_path(self):
        """Test resolve_workspace_path with relative path."""
        relative_path = "relative/path"
        workspace_dirs = ["/workspace1", "/workspace2"]
        
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.side_effect = lambda: str(self).endswith("/workspace1/relative/path")
            
            resolved_path = resolve_workspace_path(relative_path, workspace_dirs)
            assert resolved_path == Path("/workspace1/relative/path")

    def test_get_step_name_from_path(self):
        """Test get_step_name_from_path function."""
        # Test with builder file path
        builder_path = "/workspace/steps/processing_step_builder.py"
        step_name = get_step_name_from_path(builder_path)
        assert step_name == "processing_step"
        
        # Test with different naming patterns
        test_cases = [
            ("/path/training_step_builder.py", "training_step"),
            ("/path/createmodel_step_builder.py", "createmodel_step"),
            ("/path/transform_step_builder.py", "transform_step"),
            ("/path/custom_processing_step_builder.py", "custom_processing_step")
        ]
        
        for path, expected_name in test_cases:
            assert get_step_name_from_path(path) == expected_name

    def test_extract_step_type_from_name(self):
        """Test extract_step_type_from_name function."""
        test_cases = [
            ("processing_step", "Processing"),
            ("training_step", "Training"),
            ("createmodel_step", "CreateModel"),
            ("transform_step", "Transform"),
            ("data_processing_step", "Processing"),
            ("model_training_step", "Training"),
            ("batch_transform_step", "Transform"),
            ("unknown_step", None)
        ]
        
        for step_name, expected_type in test_cases:
            assert extract_step_type_from_name(step_name) == expected_type

    def test_normalize_step_name(self):
        """Test normalize_step_name function."""
        test_cases = [
            ("ProcessingStep", "processing_step"),
            ("DataProcessingStep", "data_processing_step"),
            ("XGBoostTrainingStep", "xgboost_training_step"),
            ("CreateModelStep", "create_model_step"),
            ("BatchTransformStep", "batch_transform_step"),
            ("processing-step", "processing_step"),
            ("processing.step", "processing_step"),
            ("processing step", "processing_step")
        ]
        
        for input_name, expected_name in test_cases:
            assert normalize_step_name(input_name) == expected_name

    def test_validate_workspace_structure(self, temp_workspace):
        """Test validate_workspace_structure function."""
        workspace_dirs = [str(temp_workspace)]
        
        issues = validate_workspace_structure(workspace_dirs)
        assert isinstance(issues, list)
        # Should have no issues for valid workspace structure
        assert len(issues) == 0

    def test_validate_workspace_structure_with_invalid_workspace(self):
        """Test validate_workspace_structure with invalid workspace."""
        workspace_dirs = ["/nonexistent/workspace"]
        
        issues = validate_workspace_structure(workspace_dirs)
        assert isinstance(issues, list)
        assert len(issues) > 0

    def test_get_builder_file_path(self, temp_workspace):
        """Test get_builder_file_path function."""
        step_name = "processing_step"
        workspace_dirs = [str(temp_workspace)]
        
        file_path = get_builder_file_path(step_name, workspace_dirs)
        assert file_path is not None
        assert file_path.exists()
        assert file_path.name == "processing_step_builder.py"

    def test_load_builder_module(self, temp_workspace):
        """Test load_builder_module function."""
        step_name = "processing_step"
        workspace_dirs = [str(temp_workspace)]
        
        module = load_builder_module(step_name, workspace_dirs)
        assert module is not None
        assert hasattr(module, 'ProcessingStepBuilder')

    def test_extract_class_methods(self, temp_workspace):
        """Test extract_class_methods function."""
        step_name = "processing_step"
        workspace_dirs = [str(temp_workspace)]
        
        builder_class = find_builder_class(step_name, workspace_dirs)
        methods = extract_class_methods(builder_class)
        
        assert isinstance(methods, dict)
        assert 'validate_configuration' in methods
        assert '_get_inputs' in methods
        assert 'create_step' in methods
        assert '_create_processor' in methods

    def test_get_method_signature(self, temp_workspace):
        """Test get_method_signature function."""
        step_name = "processing_step"
        workspace_dirs = [str(temp_workspace)]
        
        builder_class = find_builder_class(step_name, workspace_dirs)
        signature = get_method_signature(builder_class, 'validate_configuration')
        
        assert signature is not None
        assert 'self' in str(signature)

    def test_analyze_method_implementation(self, temp_workspace):
        """Test analyze_method_implementation function."""
        step_name = "processing_step"
        workspace_dirs = [str(temp_workspace)]
        
        builder_class = find_builder_class(step_name, workspace_dirs)
        analysis = analyze_method_implementation(builder_class, 'validate_configuration')
        
        assert isinstance(analysis, dict)
        assert 'method_name' in analysis
        assert 'signature' in analysis
        assert 'implementation_type' in analysis

    def test_get_file_modification_time(self, temp_workspace):
        """Test get_file_modification_time function."""
        builder_file = temp_workspace / "steps" / "processing_step_builder.py"
        
        mod_time = get_file_modification_time(str(builder_file))
        assert mod_time is not None
        assert isinstance(mod_time, float)

    def test_ensure_directory_exists(self):
        """Test ensure_directory_exists function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "new_directory"
            
            # Directory should not exist initially
            assert not test_dir.exists()
            
            # Create directory
            ensure_directory_exists(str(test_dir))
            
            # Directory should now exist
            assert test_dir.exists()
            assert test_dir.is_dir()

    def test_safe_file_read(self, temp_workspace):
        """Test safe_file_read function."""
        builder_file = temp_workspace / "steps" / "processing_step_builder.py"
        
        content = safe_file_read(str(builder_file))
        assert content is not None
        assert 'ProcessingStepBuilder' in content

    def test_safe_file_read_with_nonexistent_file(self):
        """Test safe_file_read with nonexistent file."""
        content = safe_file_read("/nonexistent/file.py")
        assert content is None

    def test_safe_file_write(self):
        """Test safe_file_write function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_file.txt"
            test_content = "Test content for file writing"
            
            success = safe_file_write(str(test_file), test_content)
            assert success is True
            assert test_file.exists()
            assert test_file.read_text() == test_content

    def test_get_relative_path(self):
        """Test get_relative_path function."""
        base_path = "/workspace/project"
        target_path = "/workspace/project/steps/processing_step_builder.py"
        
        relative_path = get_relative_path(target_path, base_path)
        assert relative_path == "steps/processing_step_builder.py"

    def test_is_valid_python_identifier(self):
        """Test is_valid_python_identifier function."""
        test_cases = [
            ("valid_identifier", True),
            ("ValidIdentifier", True),
            ("_private_var", True),
            ("var123", True),
            ("123invalid", False),
            ("invalid-name", False),
            ("invalid.name", False),
            ("class", False),  # Python keyword
            ("def", False),    # Python keyword
            ("", False)
        ]
        
        for identifier, expected_valid in test_cases:
            assert is_valid_python_identifier(identifier) == expected_valid

    def test_sanitize_step_name(self):
        """Test sanitize_step_name function."""
        test_cases = [
            ("valid_step_name", "valid_step_name"),
            ("Invalid-Step-Name", "invalid_step_name"),
            ("Invalid.Step.Name", "invalid_step_name"),
            ("Invalid Step Name", "invalid_step_name"),
            ("123invalid", "invalid"),
            ("_private_step", "private_step"),
            ("step@#$%name", "stepname")
        ]
        
        for input_name, expected_name in test_cases:
            assert sanitize_step_name(input_name) == expected_name

    def test_get_workspace_root(self, temp_workspace):
        """Test get_workspace_root function."""
        workspace_dirs = [str(temp_workspace)]
        
        root = get_workspace_root(workspace_dirs)
        assert root is not None
        assert Path(root).exists()

    def test_find_files_by_pattern(self, temp_workspace):
        """Test find_files_by_pattern function."""
        pattern = "*.py"
        
        files = find_files_by_pattern(str(temp_workspace), pattern)
        assert isinstance(files, list)
        assert len(files) > 0
        
        # Check that all found files match the pattern
        for file_path in files:
            assert file_path.endswith('.py')

    def test_get_python_files_in_directory(self, temp_workspace):
        """Test get_python_files_in_directory function."""
        python_files = get_python_files_in_directory(str(temp_workspace))
        
        assert isinstance(python_files, list)
        assert len(python_files) > 0
        
        # Check that all files are Python files
        for file_path in python_files:
            assert file_path.endswith('.py')

    def test_extract_imports_from_file(self, temp_workspace):
        """Test extract_imports_from_file function."""
        builder_file = temp_workspace / "steps" / "processing_step_builder.py"
        
        imports = extract_imports_from_file(str(builder_file))
        assert isinstance(imports, list)
        # The test file doesn't have imports, so should be empty
        assert len(imports) == 0

    def test_get_class_hierarchy(self, temp_workspace):
        """Test get_class_hierarchy function."""
        step_name = "processing_step"
        workspace_dirs = [str(temp_workspace)]
        
        builder_class = find_builder_class(step_name, workspace_dirs)
        hierarchy = get_class_hierarchy(builder_class)
        
        assert isinstance(hierarchy, list)
        assert len(hierarchy) > 0
        assert 'ProcessingStepBuilder' in hierarchy

    def test_validate_python_syntax(self, temp_workspace):
        """Test validate_python_syntax function."""
        builder_file = temp_workspace / "steps" / "processing_step_builder.py"
        
        is_valid, error_message = validate_python_syntax(str(builder_file))
        assert is_valid is True
        assert error_message is None

    def test_validate_python_syntax_with_invalid_file(self):
        """Test validate_python_syntax with invalid Python syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write("invalid python syntax $$$ !!!")
            temp_file.flush()
            
            is_valid, error_message = validate_python_syntax(temp_file.name)
            assert is_valid is False
            assert error_message is not None
            
            # Clean up
            os.unlink(temp_file.name)

    def test_format_validation_message(self):
        """Test format_validation_message function."""
        message = format_validation_message(
            level="ERROR",
            message="Test validation error",
            step_name="processing_step",
            method_name="validate_configuration"
        )
        
        assert isinstance(message, str)
        assert "ERROR" in message
        assert "Test validation error" in message
        assert "processing_step" in message
        assert "validate_configuration" in message

    def test_create_validation_issue(self):
        """Test create_validation_issue function."""
        issue = create_validation_issue(
            level="WARNING",
            message="Test warning message",
            step_name="training_step",
            method_name="_create_estimator",
            line_number=42
        )
        
        assert isinstance(issue, dict)
        assert issue["level"] == "WARNING"
        assert issue["message"] == "Test warning message"
        assert issue["step_name"] == "training_step"
        assert issue["method_name"] == "_create_estimator"
        assert issue["line_number"] == 42

    def test_merge_validation_results(self):
        """Test merge_validation_results function."""
        results1 = {
            "status": "PASSED",
            "issues": [{"level": "INFO", "message": "Info message"}],
            "rule_type": "universal"
        }
        
        results2 = {
            "status": "ISSUES_FOUND",
            "issues": [{"level": "WARNING", "message": "Warning message"}],
            "rule_type": "step_specific"
        }
        
        merged = merge_validation_results([results1, results2])
        
        assert isinstance(merged, dict)
        assert "status" in merged
        assert "issues" in merged
        assert "total_issues" in merged
        assert len(merged["issues"]) == 2

    def test_calculate_validation_score(self):
        """Test calculate_validation_score function."""
        validation_results = {
            "issues": [
                {"level": "ERROR", "message": "Error 1"},
                {"level": "WARNING", "message": "Warning 1"},
                {"level": "INFO", "message": "Info 1"}
            ]
        }
        
        score = calculate_validation_score(validation_results)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_get_validation_summary(self):
        """Test get_validation_summary function."""
        validation_results = {
            "status": "ISSUES_FOUND",
            "issues": [
                {"level": "ERROR", "message": "Error 1"},
                {"level": "WARNING", "message": "Warning 1"},
                {"level": "WARNING", "message": "Warning 2"},
                {"level": "INFO", "message": "Info 1"}
            ],
            "total_issues": 4
        }
        
        summary = get_validation_summary(validation_results)
        
        assert isinstance(summary, dict)
        assert "total_issues" in summary
        assert "error_count" in summary
        assert "warning_count" in summary
        assert "info_count" in summary
        assert "validation_score" in summary
        
        assert summary["total_issues"] == 4
        assert summary["error_count"] == 1
        assert summary["warning_count"] == 2
        assert summary["info_count"] == 1

    def test_error_handling_with_none_inputs(self):
        """Test that utility functions handle None inputs gracefully."""
        # Test functions that should handle None gracefully
        assert find_builder_class(None, []) is None
        assert get_step_name_from_path(None) is None
        assert extract_step_type_from_name(None) is None
        assert normalize_step_name(None) == ""
        assert get_file_modification_time(None) is None
        assert safe_file_read(None) is None
        assert is_valid_python_identifier(None) is False
        assert sanitize_step_name(None) == ""

    def test_error_handling_with_empty_inputs(self):
        """Test that utility functions handle empty inputs gracefully."""
        # Test functions that should handle empty strings gracefully
        assert find_builder_class("", []) is None
        assert get_step_name_from_path("") is None
        assert extract_step_type_from_name("") is None
        assert normalize_step_name("") == ""
        assert safe_file_read("") is None
        assert is_valid_python_identifier("") is False
        assert sanitize_step_name("") == ""

    def test_path_resolution_edge_cases(self):
        """Test path resolution with edge cases."""
        # Test with empty workspace directories
        resolved = resolve_workspace_path("test/path", [])
        assert resolved == Path("test/path")
        
        # Test with None workspace directories
        resolved = resolve_workspace_path("test/path", None)
        assert resolved == Path("test/path")

    def test_validation_result_edge_cases(self):
        """Test validation result handling with edge cases."""
        # Test merging empty results
        merged = merge_validation_results([])
        assert isinstance(merged, dict)
        assert merged["total_issues"] == 0
        
        # Test calculating score with no issues
        score = calculate_validation_score({"issues": []})
        assert score == 100.0
        
        # Test summary with empty results
        summary = get_validation_summary({"issues": [], "total_issues": 0})
        assert summary["total_issues"] == 0
        assert summary["validation_score"] == 100.0
