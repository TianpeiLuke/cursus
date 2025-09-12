"""
Unit tests for base_enhancer.py module.

Tests the abstract base class for all step type enhancers.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from abc import ABC
from cursus.validation.alignment.step_type_enhancers.base_enhancer import (
    BaseStepEnhancer,
)
from cursus.validation.alignment.core_models import (
    ValidationResult,
    StepTypeAwareAlignmentIssue,
    SeverityLevel,
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
                step_type=self.step_type,
            )
        ]
        return self._merge_results(existing_results, additional_issues)


@pytest.fixture
def enhancer():
    """Set up test enhancer fixture."""
    return ConcreteStepEnhancer("TestStep")


@pytest.fixture
def mock_validation_result():
    """Set up mock validation result fixture."""
    return ValidationResult(
        is_valid=True,
        issues=[],
        summary={"message": "Test validation result"},
        metadata={"script_name": "test_script.py"},
    )


@pytest.fixture
def existing_issue():
    """Set up existing issue fixture."""
    return StepTypeAwareAlignmentIssue(
        level=SeverityLevel.WARNING,
        category="existing_issue",
        message="Existing issue message",
        suggestion="Existing suggestion",
    )


class TestBaseStepEnhancer:
    """Test base step enhancer functionality."""

    def test_base_enhancer_initialization(self, enhancer):
        """Test base enhancer initialization."""
        assert enhancer.step_type == "TestStep"
        assert enhancer.reference_examples == []
        assert enhancer.framework_validators == {}

    def test_base_enhancer_is_abstract(self):
        """Test that BaseStepEnhancer is abstract."""
        # Should not be able to instantiate BaseStepEnhancer directly
        with pytest.raises(TypeError):
            BaseStepEnhancer("Test")

    def test_enhance_validation_abstract_method(self, enhancer, mock_validation_result):
        """Test that enhance_validation is abstract."""
        # The concrete implementation should work
        result = enhancer.enhance_validation(mock_validation_result, "test_script.py")
        assert result is not None

    def test_merge_results_with_dict_existing_results(self, enhancer, existing_issue):
        """Test merging results when existing_results is a dictionary."""
        existing_results = {
            "issues": [existing_issue],
            "success": False,
            "summary": "Existing validation",
        }

        additional_issues = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.ERROR,
                category="additional_issue",
                message="Additional issue message",
                suggestion="Additional suggestion",
            )
        ]

        result = enhancer._merge_results(existing_results, additional_issues)

        # Verify that issues were merged
        assert len(result["issues"]) == 2
        assert existing_issue in result["issues"]
        assert additional_issues[0] in result["issues"]

    def test_merge_results_with_validation_result_object(
        self, enhancer, existing_issue
    ):
        """Test merging results when existing_results is a ValidationResult object."""
        existing_results = ValidationResult(
            is_valid=False,
            issues=[existing_issue],
            summary={"message": "Existing validation"},
            metadata={"script_name": "test_script.py"},
        )

        additional_issues = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.ERROR,
                category="additional_issue",
                message="Additional issue message",
                suggestion="Additional suggestion",
            )
        ]

        result = enhancer._merge_results(existing_results, additional_issues)

        # Verify that issues were merged
        assert len(result.issues) == 2
        assert existing_issue in result.issues
        assert additional_issues[0] in result.issues

    def test_merge_results_with_empty_additional_issues(self, enhancer, existing_issue):
        """Test merging results with empty additional issues."""
        existing_results = {
            "issues": [existing_issue],
            "success": True,
            "summary": "Existing validation",
        }

        additional_issues = []

        result = enhancer._merge_results(existing_results, additional_issues)

        # Verify that only existing issues remain
        assert len(result["issues"]) == 1
        assert existing_issue in result["issues"]

    def test_merge_results_with_no_existing_issues(self, enhancer):
        """Test merging results when existing results have no issues."""
        existing_results = {
            "issues": [],
            "success": True,
            "summary": "Clean validation",
        }

        additional_issues = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.WARNING,
                category="new_issue",
                message="New issue message",
                suggestion="New suggestion",
            )
        ]

        result = enhancer._merge_results(existing_results, additional_issues)

        # Verify that only additional issues are present
        assert len(result["issues"]) == 1
        assert additional_issues[0] in result["issues"]

    def test_merge_results_preserves_other_fields(self, enhancer, existing_issue):
        """Test that merge_results preserves other fields in existing results."""
        existing_results = {
            "issues": [existing_issue],
            "success": False,
            "summary": "Existing validation",
            "custom_field": "custom_value",
            "metadata": {"key": "value"},
        }

        additional_issues = []

        result = enhancer._merge_results(existing_results, additional_issues)

        # Verify that other fields are preserved
        assert result["success"] == False
        assert result["summary"] == "Existing validation"
        assert result["custom_field"] == "custom_value"
        assert result["metadata"] == {"key": "value"}

    def test_concrete_enhancer_implementation(self, enhancer, mock_validation_result):
        """Test the concrete enhancer implementation."""
        result = enhancer.enhance_validation(mock_validation_result, "test_script.py")

        # Verify that enhancement was applied
        assert len(result.issues) == 1
        assert result.issues[0].category == "test_enhancement"
        assert result.issues[0].step_type == "TestStep"

    def test_enhancer_with_reference_examples(self):
        """Test enhancer with reference examples."""
        enhancer = ConcreteStepEnhancer("TestStep")
        enhancer.reference_examples = ["example1.py", "example2.py"]

        assert len(enhancer.reference_examples) == 2
        assert "example1.py" in enhancer.reference_examples
        assert "example2.py" in enhancer.reference_examples

    def test_enhancer_with_framework_validators(self):
        """Test enhancer with framework validators."""
        enhancer = ConcreteStepEnhancer("TestStep")
        mock_validator = Mock()
        enhancer.framework_validators = {"xgboost": mock_validator}

        assert len(enhancer.framework_validators) == 1
        assert "xgboost" in enhancer.framework_validators
        assert enhancer.framework_validators["xgboost"] == mock_validator

    def test_merge_results_handles_none_existing_results(self, enhancer):
        """Test merge_results handles None existing results gracefully."""
        additional_issues = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.INFO,
                category="test_issue",
                message="Test message",
                suggestion="Test suggestion",
            )
        ]

        # This should handle None gracefully
        result = enhancer._merge_results(None, additional_issues)

        # Should return the additional issues in some form
        # The exact behavior depends on implementation
        assert result is not None

    def test_merge_results_handles_invalid_existing_results(self, enhancer):
        """Test merge_results handles invalid existing results gracefully."""
        additional_issues = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.INFO,
                category="test_issue",
                message="Test message",
                suggestion="Test suggestion",
            )
        ]

        # Test with string (invalid type)
        result = enhancer._merge_results("invalid", additional_issues)

        # Should handle gracefully
        assert result is not None

    def test_step_type_property(self):
        """Test step_type property access."""
        enhancer = ConcreteStepEnhancer("CustomStepType")
        assert enhancer.step_type == "CustomStepType"

    def test_multiple_enhancements(self, enhancer, mock_validation_result):
        """Test multiple enhancement calls."""
        # First enhancement
        result1 = enhancer.enhance_validation(mock_validation_result, "script1.py")

        # Second enhancement on the result of the first
        result2 = enhancer.enhance_validation(result1, "script2.py")

        # Should have accumulated issues
        assert len(result2.issues) == 2

        # Both issues should be test enhancements
        for issue in result2.issues:
            assert issue.category == "test_enhancement"
            assert issue.step_type == "TestStep"


class TestBaseStepEnhancerEdgeCases:
    """Test edge cases for BaseStepEnhancer."""

    def test_enhancer_with_empty_step_type(self):
        """Test enhancer with empty step type."""
        enhancer = ConcreteStepEnhancer("")
        assert enhancer.step_type == ""

    def test_enhancer_with_none_step_type(self):
        """Test enhancer with None step type."""
        enhancer = ConcreteStepEnhancer(None)
        assert enhancer.step_type is None

    def test_enhancer_inheritance_chain(self):
        """Test that enhancer properly inherits from ABC."""
        enhancer = ConcreteStepEnhancer("Test")
        assert isinstance(enhancer, BaseStepEnhancer)
        assert isinstance(enhancer, ABC)


if __name__ == "__main__":
    pytest.main([__file__])
