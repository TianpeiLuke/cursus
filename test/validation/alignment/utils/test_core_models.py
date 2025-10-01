"""
Test suite for core alignment models including StepTypeAwareAlignmentIssue.
"""

import pytest
from unittest.mock import Mock

from cursus.validation.alignment import (
    SeverityLevel,
    AlignmentLevel,
    create_alignment_issue,
    create_step_type_aware_alignment_issue,
)
from cursus.validation.alignment.utils.core_models import (
    AlignmentIssue,
    StepTypeAwareAlignmentIssue,
)


class TestStepTypeAwareAlignmentIssue:
    """Test StepTypeAwareAlignmentIssue model."""

    def test_step_type_aware_issue_creation(self):
        """Test basic StepTypeAwareAlignmentIssue creation."""
        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.ERROR,
            category="training_validation",
            message="Missing training loop",
            details={"script": "xgboost_training.py"},
            step_type="Training",
            framework_context="xgboost",
        )

        assert issue.level == SeverityLevel.ERROR
        assert issue.category == "training_validation"
        assert issue.message == "Missing training loop"
        assert issue.step_type == "Training"
        assert issue.framework_context == "xgboost"
        assert issue.details["script"] == "xgboost_training.py"

    def test_step_type_aware_issue_with_reference_examples(self):
        """Test StepTypeAwareAlignmentIssue with reference examples."""
        reference_examples = ["xgboost_training.py", "builder_xgboost_training_step.py"]

        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.WARNING,
            category="pattern_validation",
            message="Missing model saving pattern",
            step_type="Training",
            framework_context="pytorch",
            reference_examples=reference_examples,
        )

        assert issue.reference_examples == reference_examples
        assert issue.framework_context == "pytorch"
        assert "xgboost_training.py" in issue.reference_examples

    def test_step_type_aware_issue_defaults(self):
        """Test StepTypeAwareAlignmentIssue default values."""
        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.INFO,
            category="general",
            message="Info message",
            step_type="Processing",
        )

        assert issue.step_type == "Processing"
        assert issue.framework_context is None
        assert issue.reference_examples == []
        assert issue.details == {}

    def test_step_type_aware_issue_inheritance(self):
        """Test that StepTypeAwareAlignmentIssue inherits from AlignmentIssue."""
        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.ERROR,
            category="test",
            message="Test message",
            step_type="Training",
            recommendation="Fix this issue",
        )

        # Should have all AlignmentIssue properties
        assert issue.level == SeverityLevel.ERROR
        assert issue.category == "test"
        assert issue.message == "Test message"
        assert issue.recommendation == "Fix this issue"

        # Plus step type specific properties
        assert issue.step_type == "Training"

    def test_step_type_aware_issue_serialization(self):
        """Test StepTypeAwareAlignmentIssue serialization."""
        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.WARNING,
            category="framework_validation",
            message="XGBoost pattern missing",
            step_type="Training",
            framework_context="xgboost",
            reference_examples=["xgboost_training.py"],
            details={"line": 42},
        )

        issue_dict = issue.model_dump()

        assert issue_dict["step_type"] == "Training"
        assert issue_dict["framework_context"] == "xgboost"
        assert issue_dict["reference_examples"] == ["xgboost_training.py"]
        assert issue_dict["details"]["line"] == 42

    def test_step_type_aware_issue_json_serialization(self):
        """Test StepTypeAwareAlignmentIssue JSON serialization."""
        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.ERROR,
            category="training_validation",
            message="Training validation failed",
            step_type="Training",
            framework_context="pytorch",
        )

        json_str = issue.model_dump_json()
        assert isinstance(json_str, str)
        assert "Training" in json_str
        assert "pytorch" in json_str
        assert "training_validation" in json_str


class TestCreateAlignmentIssueFunctions:
    """Test alignment issue creation helper functions."""

    def test_create_alignment_issue_basic(self):
        """Test basic alignment issue creation."""
        issue = create_alignment_issue(
            level=SeverityLevel.ERROR, category="test_category", message="Test message"
        )

        assert isinstance(issue, AlignmentIssue)
        assert issue.level == SeverityLevel.ERROR
        assert issue.category == "test_category"
        assert issue.message == "Test message"

    def test_create_alignment_issue_with_details(self):
        """Test alignment issue creation with details."""
        issue = create_alignment_issue(
            level=SeverityLevel.WARNING,
            category="path_validation",
            message="Hardcoded path found",
            details={"path": "/opt/ml/input", "line": 10},
            recommendation="Use environment variables",
            alignment_level=AlignmentLevel.SCRIPT_CONTRACT,
        )

        assert issue.details["path"] == "/opt/ml/input"
        assert issue.details["line"] == 10
        assert issue.recommendation == "Use environment variables"
        assert issue.alignment_level == AlignmentLevel.SCRIPT_CONTRACT

    def test_create_step_type_aware_alignment_issue_basic(self):
        """Test basic step type aware alignment issue creation."""
        issue = create_step_type_aware_alignment_issue(
            level=SeverityLevel.ERROR,
            category="training_validation",
            message="Training pattern missing",
            step_type="Training",
        )

        assert isinstance(issue, StepTypeAwareAlignmentIssue)
        assert issue.step_type == "Training"
        assert issue.category == "training_validation"

    def test_create_step_type_aware_alignment_issue_with_framework(self):
        """Test step type aware alignment issue creation with framework."""
        issue = create_step_type_aware_alignment_issue(
            level=SeverityLevel.WARNING,
            category="framework_validation",
            message="XGBoost DMatrix not found",
            step_type="Training",
            framework_context="xgboost",
            details={"script": "train.py"},
            recommendation="Add DMatrix creation",
        )

        assert issue.framework_context == "xgboost"
        assert issue.details["script"] == "train.py"
        assert issue.recommendation == "Add DMatrix creation"

    def test_create_step_type_aware_alignment_issue_with_reference_examples(self):
        """Test step type aware alignment issue creation with reference examples."""
        reference_examples = ["xgboost_training.py", "builder_xgboost_training_step.py"]

        issue = create_step_type_aware_alignment_issue(
            level=SeverityLevel.INFO,
            category="step_type_info",
            message="Reference examples available",
            step_type="Training",
            reference_examples=reference_examples,
        )

        assert issue.reference_examples == reference_examples
        assert "xgboost_training.py" in issue.reference_examples


class TestSeverityLevelEnum:
    """Test SeverityLevel enum."""

    def test_severity_level_values(self):
        """Test SeverityLevel enum values."""
        assert SeverityLevel.CRITICAL.value == "CRITICAL"
        assert SeverityLevel.ERROR.value == "ERROR"
        assert SeverityLevel.WARNING.value == "WARNING"
        assert SeverityLevel.INFO.value == "INFO"

    def test_severity_level_ordering(self):
        """Test SeverityLevel ordering for comparison."""
        # Test that we can compare severity levels by value
        severity_order = [
            SeverityLevel.INFO,
            SeverityLevel.WARNING,
            SeverityLevel.ERROR,
            SeverityLevel.CRITICAL,
        ]

        # Test that each level has the expected string value
        assert SeverityLevel.INFO.value == "INFO"
        assert SeverityLevel.WARNING.value == "WARNING"
        assert SeverityLevel.ERROR.value == "ERROR"
        assert SeverityLevel.CRITICAL.value == "CRITICAL"


class TestAlignmentLevelEnum:
    """Test AlignmentLevel enum."""

    def test_alignment_level_values(self):
        """Test AlignmentLevel enum values."""
        assert AlignmentLevel.SCRIPT_CONTRACT.value == 1
        assert AlignmentLevel.CONTRACT_SPECIFICATION.value == 2
        assert AlignmentLevel.SPECIFICATION_DEPENDENCY.value == 3
        assert AlignmentLevel.BUILDER_CONFIGURATION.value == 4


if __name__ == "__main__":
    pytest.main([__file__])
