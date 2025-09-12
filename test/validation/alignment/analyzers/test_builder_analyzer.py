"""
Unit tests for BuilderCodeAnalyzer

Tests the enhanced builder code analysis capabilities including:
- Method call vs field access distinction
- Configuration field access detection
- Validation call detection
- Class definition analysis
"""

import pytest
import tempfile
import ast
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys


from cursus.validation.alignment.analyzers.builder_analyzer import BuilderCodeAnalyzer


class TestBuilderCodeAnalyzer:
    """Test cases for BuilderCodeAnalyzer"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = BuilderCodeAnalyzer()

    def test_analyze_builder_file_basic(self):
        """Test basic builder file analysis"""
        # Create a temporary builder file
        builder_content = """
class TestStepBuilder:
    def __init__(self, config):
        self.config = config
    
    def build_step(self):
        # Field access
        instance_type = self.config.processing_instance_type
        volume_size = self.config.processing_volume_size
        
        # Method call (should not be flagged as field access)
        script_path = self.config.get_script_path()
        
        return step
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(builder_content)
            f.flush()

            result = self.analyzer.analyze_builder_file(Path(f.name))

        # Clean up
        Path(f.name).unlink()

        # Check results
        assert "config_accesses" in result
        assert "class_definitions" in result

        # Should detect field accesses but not method calls
        config_accesses = result["config_accesses"]
        accessed_fields = {access["field_name"] for access in config_accesses}

        assert "processing_instance_type" in accessed_fields
        assert "processing_volume_size" in accessed_fields
        assert "get_script_path" not in accessed_fields  # Method call, not field access

    def test_distinguish_method_calls_from_field_access(self):
        """Test that method calls are not flagged as field accesses"""
        builder_content = """
class TestStepBuilder:
    def build_step(self):
        # These are method calls - should NOT be flagged
        path = self.config.get_script_path()
        args = self.config.get_processing_args()
        
        # These are field accesses - should be flagged
        instance_type = self.config.processing_instance_type
        count = self.config.processing_instance_count
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(builder_content)
            f.flush()

            result = self.analyzer.analyze_builder_file(Path(f.name))

        # Clean up
        Path(f.name).unlink()

        config_accesses = result["config_accesses"]
        accessed_fields = {access["field_name"] for access in config_accesses}

        # Should only contain field accesses, not method calls
        assert "processing_instance_type" in accessed_fields
        assert "processing_instance_count" in accessed_fields
        assert "get_script_path" not in accessed_fields
        assert "get_processing_args" not in accessed_fields

    def test_detect_validation_calls(self):
        """Test detection of validation method calls"""
        builder_content = """
class TestStepBuilder:
    def build_step(self):
        # Validation calls
        self.config.validate()
        validate_config(self.config)
        
        # Regular field access
        value = self.config.some_field
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(builder_content)
            f.flush()

            result = self.analyzer.analyze_builder_file(Path(f.name))

        # Clean up
        Path(f.name).unlink()

        # Should detect validation calls
        assert "validation_calls" in result
        validation_calls = result["validation_calls"]

        # Should have detected validation calls
        assert len(validation_calls) > 0

    def test_detect_class_definitions(self):
        """Test detection of class definitions"""
        builder_content = '''
class TestStepBuilder:
    """Main builder class"""
    pass

class HelperClass:
    """Helper class"""
    pass
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(builder_content)
            f.flush()

            result = self.analyzer.analyze_builder_file(Path(f.name))

        # Clean up
        Path(f.name).unlink()

        # Should detect both classes
        assert "class_definitions" in result
        class_defs = result["class_definitions"]

        class_names = {cls["class_name"] for cls in class_defs}
        assert "TestStepBuilder" in class_names
        assert "HelperClass" in class_names

    def test_complex_config_access_patterns(self):
        """Test detection of complex configuration access patterns"""
        builder_content = """
class TestStepBuilder:
    def build_step(self):
        # Direct field access
        instance_type = self.config.processing_instance_type
        
        # Nested attribute access
        s3_path = self.config.output_config.s3_uri
        
        # Conditional access
        if self.config.enable_feature:
            feature_config = self.config.feature_settings
        
        # Method call with config field as argument
        self.process_data(self.config.data_path)
        
        # Method call on config (should not be flagged)
        validated = self.config.validate_settings()
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(builder_content)
            f.flush()

            result = self.analyzer.analyze_builder_file(Path(f.name))

        # Clean up
        Path(f.name).unlink()

        config_accesses = result["config_accesses"]
        accessed_fields = {access["field_name"] for access in config_accesses}

        # Should detect field accesses
        assert "processing_instance_type" in accessed_fields
        assert "enable_feature" in accessed_fields
        assert "feature_settings" in accessed_fields
        assert "data_path" in accessed_fields

        # Should not detect method calls
        assert "validate_settings" not in accessed_fields

    def test_visit_attribute_method_vs_field(self):
        """Test the visit_Attribute method distinguishes methods from fields"""
        # Create a test file with both method calls and field accesses
        test_content = """
class TestBuilder:
    def build_step(self):
        # Field access
        field_value = self.config.field_name
        
        # Method call
        method_result = self.config.method_name()
        
        return None
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            f.flush()

            result = self.analyzer.analyze_builder_file(Path(f.name))

        # Clean up
        Path(f.name).unlink()

        # Check results
        accessed_fields = {access["field_name"] for access in result["config_accesses"]}

        assert "field_name" in accessed_fields
        assert "method_name" not in accessed_fields

    def test_error_handling_invalid_syntax(self):
        """Test error handling for files with invalid syntax"""
        # Create a file with invalid Python syntax
        invalid_content = """
class TestStepBuilder:
    def build_step(self
        # Missing closing parenthesis and colon
        invalid syntax here
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(invalid_content)
            f.flush()

            # Should handle gracefully and return error info
            result = self.analyzer.analyze_builder_file(Path(f.name))

        # Clean up
        Path(f.name).unlink()

        # Should return error information
        assert "error" in result
        assert "config_accesses" in result
        assert len(result["config_accesses"]) == 0

    def test_empty_file_handling(self):
        """Test handling of empty or minimal files"""
        empty_content = """# Empty builder file
pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(empty_content)
            f.flush()

            result = self.analyzer.analyze_builder_file(Path(f.name))

        # Clean up
        Path(f.name).unlink()

        # Should return valid structure with empty lists
        assert "config_accesses" in result
        assert "class_definitions" in result
        assert "validation_calls" in result

        assert len(result["config_accesses"]) == 0
        assert len(result["class_definitions"]) == 0

    def test_analyze_builder_code_direct(self):
        """Test direct analysis of builder AST"""
        # Create a simple AST for testing
        test_code = """
class TestBuilder:
    def build_step(self):
        value = self.config.test_field
        return value
"""

        test_ast = ast.parse(test_code)
        result = self.analyzer.analyze_builder_code(test_ast, test_code)

        # Check that analysis returns expected structure
        assert "config_accesses" in result
        assert "class_definitions" in result
        assert "validation_calls" in result

        # Should detect the config access
        accessed_fields = {access["field_name"] for access in result["config_accesses"]}
        assert "test_field" in accessed_fields


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
