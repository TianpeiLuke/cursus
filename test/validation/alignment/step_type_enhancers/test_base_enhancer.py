"""
Unit tests for base_enhancer.py module.

Tests the abstract base class for all step type enhancers.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from abc import ABC
from src.cursus.validation.alignment.step_type_enhancers.base_enhancer import BaseStepEnhancer
from src.cursus.validation.alignment.core_models import (
    ValidationResult,
    StepTypeAwareAlignmentIssue,
    SeverityLevel
)


class ConcreteStepEnhancer(BaseStepEnhancer):
    """Concrete implementation of BaseStepEnhancer for testing."""
    
    def __init__(self, step_type: str = "Test"):
        super().__init__(step_type)
    
    def enhance_validation(self, existing_results, script_name):
        """Concrete implementation for testing."""
        additional_issues = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.INFO,
                category="test_enhancement",
                message="Test enhancement applied",
                suggestion="Test suggestion",
                step_type=self.step_type
            )
        ]
        return self._merge_results(existing_results, additional_issues)


class TestBaseStepEnhancer(unittest.TestCase):
    """Test base step enhancer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.enhancer = ConcreteStepEnhancer("TestStep")
        
        # Mock validation result
        self.mock_validation_result = ValidationResult(
            is_valid=True,
            issues=[],
            summary={"message": "Test validation result"},
            metadata={"script_name": "test_script.py"}
        )
        
        # Mock existing issue
        self.existing_issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.WARNING,
            category="existing_issue",
            message="Existing issue message",
            suggestion="Existing suggestion"
        )

    def test_base_enhancer_initialization(self):
        """Test base enhancer initialization."""
        self.assertEqual(self.enhancer.step_type, "TestStep")
        self.assertEqual(self.enhancer.reference_examples, [])
        self.assertEqual(self.enhancer.framework_validators, {})

    def test_base_enhancer_is_abstract(self):
        """Test that BaseStepEnhancer is abstract."""
        # Should not be able to instantiate BaseStepEnhancer directly
        with self.assertRaises(TypeError):
            BaseStepEnhancer("Test")

    def test_enhance_validation_abstract_method(self):
        """Test that enhance_validation is abstract."""
        # The concrete implementation should work
        result = self.enhancer.enhance_validation(self.mock_validation_result, "test_script.py")
        self.assertIsNotNone(result)

    def test_merge_results_with_dict_existing_results(self):
        """Test merging results when existing_results is a dictionary."""
        existing_results = {
            'issues': [self.existing_issue],
            'success': False,
            'summary': 'Existing validation'
        }
        
        additional_issues = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.ERROR,
                category="additional_issue",
                message="Additional issue message",
                suggestion="Additional suggestion"
            )
        ]
        
        result = self.enhancer._merge_results(existing_results, additional_issues)
        
        # Verify that issues were merged
        self.assertEqual(len(result['issues']), 2)
        self.assertIn(self.existing_issue, result['issues'])
        self.assertIn(additional_issues[0], result['issues'])

    def test_merge_results_with_validation_result_object(self):
        """Test merging results when existing_results is a ValidationResult object."""
        existing_results = ValidationResult(
            is_valid=False,
            issues=[self.existing_issue],
            summary={"message": "Existing validation"},
            metadata={"script_name": "test_script.py"}
        )
        
        additional_issues = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.ERROR,
                category="additional_issue",
                message="Additional issue message",
                suggestion="Additional suggestion"
            )
        ]
        
        result = self.enhancer._merge_results(existing_results, additional_issues)
        
        # Verify that issues were merged
        self.assertEqual(len(result.issues), 2)
        self.assertIn(self.existing_issue, result.issues)
        self.assertIn(additional_issues[0], result.issues)

    def test_merge_results_with_empty_additional_issues(self):
        """Test merging results with empty additional issues."""
        existing_results = {
            'issues': [self.existing_issue],
            'success': True,
            'summary': 'Existing validation'
        }
        
        additional_issues = []
        
        result = self.enhancer._merge_results(existing_results, additional_issues)
        
        # Verify that only existing issues remain
        self.assertEqual(len(result['issues']), 1)
        self.assertIn(self.existing_issue, result['issues'])

    def test_merge_results_with_no_existing_issues(self):
        """Test merging results when existing results have no issues."""
        existing_results = {
            'issues': [],
            'success': True,
            'summary': 'Clean validation'
        }
        
        additional_issues = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.WARNING,
                category="new_issue",
                message="New issue message",
                suggestion="New suggestion"
            )
        ]
        
        result = self.enhancer._merge_results(existing_results, additional_issues)
        
        # Verify that only additional issues are present
        self.assertEqual(len(result['issues']), 1)
        self.assertIn(additional_issues[0], result['issues'])

    def test_merge_results_preserves_other_fields(self):
        """Test that merge_results preserves other fields in existing results."""
        existing_results = {
            'issues': [self.existing_issue],
            'success': False,
            'summary': 'Existing validation',
            'custom_field': 'custom_value',
            'metadata': {'key': 'value'}
        }
        
        additional_issues = []
        
        result = self.enhancer._merge_results(existing_results, additional_issues)
        
        # Verify that other fields are preserved
        self.assertEqual(result['success'], False)
        self.assertEqual(result['summary'], 'Existing validation')
        self.assertEqual(result['custom_field'], 'custom_value')
        self.assertEqual(result['metadata'], {'key': 'value'})

    def test_concrete_enhancer_implementation(self):
        """Test the concrete enhancer implementation."""
        result = self.enhancer.enhance_validation(self.mock_validation_result, "test_script.py")
        
        # Verify that enhancement was applied
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].category, "test_enhancement")
        self.assertEqual(result.issues[0].step_type, "TestStep")

    def test_enhancer_with_reference_examples(self):
        """Test enhancer with reference examples."""
        enhancer = ConcreteStepEnhancer("TestStep")
        enhancer.reference_examples = ["example1.py", "example2.py"]
        
        self.assertEqual(len(enhancer.reference_examples), 2)
        self.assertIn("example1.py", enhancer.reference_examples)
        self.assertIn("example2.py", enhancer.reference_examples)

    def test_enhancer_with_framework_validators(self):
        """Test enhancer with framework validators."""
        enhancer = ConcreteStepEnhancer("TestStep")
        mock_validator = Mock()
        enhancer.framework_validators = {"xgboost": mock_validator}
        
        self.assertEqual(len(enhancer.framework_validators), 1)
        self.assertIn("xgboost", enhancer.framework_validators)
        self.assertEqual(enhancer.framework_validators["xgboost"], mock_validator)

    def test_merge_results_handles_none_existing_results(self):
        """Test merge_results handles None existing results gracefully."""
        additional_issues = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.INFO,
                category="test_issue",
                message="Test message",
                suggestion="Test suggestion"
            )
        ]
        
        # This should handle None gracefully
        result = self.enhancer._merge_results(None, additional_issues)
        
        # Should return the additional issues in some form
        # The exact behavior depends on implementation
        self.assertIsNotNone(result)

    def test_merge_results_handles_invalid_existing_results(self):
        """Test merge_results handles invalid existing results gracefully."""
        additional_issues = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.INFO,
                category="test_issue",
                message="Test message",
                suggestion="Test suggestion"
            )
        ]
        
        # Test with string (invalid type)
        result = self.enhancer._merge_results("invalid", additional_issues)
        
        # Should handle gracefully
        self.assertIsNotNone(result)

    def test_step_type_property(self):
        """Test step_type property access."""
        enhancer = ConcreteStepEnhancer("CustomStepType")
        self.assertEqual(enhancer.step_type, "CustomStepType")

    def test_multiple_enhancements(self):
        """Test multiple enhancement calls."""
        # First enhancement
        result1 = self.enhancer.enhance_validation(self.mock_validation_result, "script1.py")
        
        # Second enhancement on the result of the first
        result2 = self.enhancer.enhance_validation(result1, "script2.py")
        
        # Should have accumulated issues
        self.assertEqual(len(result2.issues), 2)
        
        # Both issues should be test enhancements
        for issue in result2.issues:
            self.assertEqual(issue.category, "test_enhancement")
            self.assertEqual(issue.step_type, "TestStep")


class TestBaseStepEnhancerEdgeCases(unittest.TestCase):
    """Test edge cases for BaseStepEnhancer."""

    def test_enhancer_with_empty_step_type(self):
        """Test enhancer with empty step type."""
        enhancer = ConcreteStepEnhancer("")
        self.assertEqual(enhancer.step_type, "")

    def test_enhancer_with_none_step_type(self):
        """Test enhancer with None step type."""
        enhancer = ConcreteStepEnhancer(None)
        self.assertEqual(enhancer.step_type, None)

    def test_enhancer_inheritance_chain(self):
        """Test that enhancer properly inherits from ABC."""
        enhancer = ConcreteStepEnhancer("Test")
        self.assertIsInstance(enhancer, BaseStepEnhancer)
        self.assertIsInstance(enhancer, ABC)


if __name__ == '__main__':
    unittest.main()
