"""
Tests for workspace documentation validation functionality.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from src.cursus.workspace.quality.documentation_validator import DocumentationQualityValidator


class TestDocumentationValidator(unittest.TestCase):
    """Test cases for DocumentationValidator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)
        self.doc_validator = DocumentationQualityValidator(str(self.workspace_path))

    def test_doc_validator_initialization(self):
        """Test that DocumentationValidator initializes correctly."""
        self.assertIsInstance(self.doc_validator, DocumentationQualityValidator)
        self.assertTrue(hasattr(self.doc_validator, 'validate_workspace_documentation'))

    def test_validate_workspace_documentation(self):
        """Test workspace documentation validation."""
        # Test workspace documentation validation
        doc_results = self.doc_validator.validate_workspace_documentation()
        self.assertIsInstance(doc_results, dict)

    def test_generate_documentation_quality_report(self):
        """Test documentation quality report generation."""
        # Test quality report generation
        quality_report = self.doc_validator.generate_documentation_quality_report()
        self.assertIsInstance(quality_report, dict)
        self.assertIn('overall_documentation_quality', quality_report)

    def test_docstring_coverage_validation(self):
        """Test docstring coverage validation."""
        # Create a temporary Python file for testing
        test_py_file = self.workspace_path / "test_module.py"
        test_py_file.write_text('''
def test_function():
    """Test function with docstring."""
    pass

def undocumented_function():
    pass
''')
        
        # Test docstring coverage
        coverage_result = self.doc_validator.validate_docstring_coverage(str(self.workspace_path))
        self.assertIsNotNone(coverage_result)
        self.assertTrue(hasattr(coverage_result, 'metrics'))


if __name__ == '__main__':
    unittest.main()
