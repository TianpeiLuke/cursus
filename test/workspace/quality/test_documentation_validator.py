"""
Tests for workspace documentation validation functionality.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from cursus.workspace.quality.documentation_validator import (
    DocumentationQualityValidator,
    DocumentationType,
    DocumentationValidationResult,
)


class TestDocumentationQualityValidator(unittest.TestCase):
    """Test cases for DocumentationQualityValidator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)
        self.doc_validator = DocumentationQualityValidator(str(self.workspace_path))

    def test_doc_validator_initialization(self):
        """Test that DocumentationQualityValidator initializes correctly."""
        self.assertIsInstance(self.doc_validator, DocumentationQualityValidator)
        self.assertTrue(hasattr(self.doc_validator, "validate_workspace_documentation"))
        self.assertTrue(
            hasattr(self.doc_validator, "generate_documentation_quality_report")
        )
        self.assertTrue(hasattr(self.doc_validator, "validate_docstring_coverage"))

    def test_validate_workspace_documentation(self):
        """Test workspace documentation validation."""
        # Test workspace documentation validation
        doc_results = self.doc_validator.validate_workspace_documentation()
        self.assertIsInstance(doc_results, dict)

        # Results should be DocumentationValidationResult objects
        for key, result in doc_results.items():
            if result:  # Some may be None if files don't exist
                self.assertIsInstance(result, DocumentationValidationResult)

    def test_generate_documentation_quality_report(self):
        """Test documentation quality report generation."""
        # Test quality report generation
        quality_report = self.doc_validator.generate_documentation_quality_report()
        self.assertIsInstance(quality_report, dict)
        self.assertIn("overall_documentation_quality", quality_report)
        self.assertIn("phase3_compliance_rate", quality_report)
        self.assertIn("meets_phase3_threshold", quality_report)
        self.assertIn("validation_results", quality_report)
        self.assertIn("summary", quality_report)

    def test_docstring_coverage_validation(self):
        """Test docstring coverage validation."""
        # Create a temporary Python file for testing
        test_py_file = self.workspace_path / "test_module.py"
        test_py_file.write_text(
            '''
def test_function():
    """Test function with docstring."""
    pass

def undocumented_function():
    pass

class TestClass:
    """Test class with docstring."""
    
    def documented_method(self):
        """Method with docstring."""
        pass
    
    def undocumented_method(self):
        pass
'''
        )

        # Test docstring coverage
        coverage_result = self.doc_validator.validate_docstring_coverage(
            str(self.workspace_path)
        )
        self.assertIsInstance(coverage_result, DocumentationValidationResult)
        self.assertTrue(hasattr(coverage_result, "metrics"))
        self.assertEqual(coverage_result.validation_type, "docstring_coverage")

        # Check that metrics are properly calculated
        self.assertIsNotNone(coverage_result.metrics.coverage_score)
        self.assertIsNotNone(coverage_result.metrics.completeness_score)

    def test_validate_documentation_file(self):
        """Test validation of individual documentation files."""
        # Create a test markdown file
        test_md_file = self.workspace_path / "test_api.md"
        test_md_file.write_text(
            """
# Test API Documentation

## Parameters

This section describes the parameters.

## Returns

This section describes the return values.

## Examples

```python
import test_module
result = test_module.test_function()
```

## Raises

This section describes exceptions.
"""
        )

        # Test API reference validation
        result = self.doc_validator.validate_documentation_file(
            str(test_md_file), DocumentationType.API_REFERENCE
        )

        self.assertIsInstance(result, DocumentationValidationResult)
        self.assertEqual(result.validation_type, "documentation_api_reference")
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.issues)
        self.assertIsNotNone(result.recommendations)

    def test_documentation_metrics_calculation(self):
        """Test that documentation metrics are calculated correctly."""
        # Create a comprehensive test file
        test_file = self.workspace_path / "comprehensive_doc.md"
        test_file.write_text(
            """
# Comprehensive Documentation

## Installation

Install using pip:

```bash
pip install test-package
```

## Basic Usage

Here's how to use the package:

```python
import test_package
result = test_package.main_function()
print(result)
```

## Examples

### Example 1: Basic Usage

```python
# This is a basic example
import test_package
test_package.hello_world()
```

### Example 2: Advanced Usage

```python
# This is an advanced example
import test_package
config = {"setting": "value"}
result = test_package.advanced_function(config)
```

Note: Make sure to configure your environment properly.

Important: This feature requires Python 3.8+.
"""
        )

        result = self.doc_validator.validate_documentation_file(
            str(test_file), DocumentationType.QUICK_START
        )

        # Check that all metric scores are between 0 and 1
        metrics = result.metrics
        self.assertGreaterEqual(metrics.completeness_score, 0.0)
        self.assertLessEqual(metrics.completeness_score, 1.0)
        self.assertGreaterEqual(metrics.clarity_score, 0.0)
        self.assertLessEqual(metrics.clarity_score, 1.0)
        self.assertGreaterEqual(metrics.overall_score, 0.0)
        self.assertLessEqual(metrics.overall_score, 1.0)

    def test_phase3_requirements_checking(self):
        """Test Phase 3 requirements validation."""
        # Create a high-quality documentation file
        high_quality_file = self.workspace_path / "high_quality.md"
        high_quality_file.write_text(
            """
# High Quality Documentation

## Installation

Complete installation instructions with examples.

```bash
pip install package
```

## Basic Usage

Comprehensive usage examples with explanations.

```python
import package
result = package.function()
```

## Examples

Multiple detailed examples with explanations.

```python
# Example 1
package.example1()

# Example 2  
package.example2()
```
"""
        )

        result = self.doc_validator.validate_documentation_file(
            str(high_quality_file), DocumentationType.QUICK_START
        )

        # Test Phase 3 requirements method
        phase3_compliant = result.meets_phase3_requirements()
        self.assertIsInstance(phase3_compliant, bool)


if __name__ == "__main__":
    unittest.main()
