"""
Test suite for core alignment models including StepTypeAwareAlignmentIssue.
"""

import unittest
from unittest.mock import Mock

from cursus.validation.alignment.alignment_utils import (
    SeverityLevel,
    AlignmentLevel,
    AlignmentIssue,
    StepTypeAwareAlignmentIssue,
    create_alignment_issue,
    create_step_type_aware_alignment_issue
)

class TestStepTypeAwareAlignmentIssue(unittest.TestCase):
    """Test StepTypeAwareAlignmentIssue model."""
    
    def test_step_type_aware_issue_creation(self):
        """Test basic StepTypeAwareAlignmentIssue creation."""
        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.ERROR,
            category="training_validation",
            message="Missing training loop",
            details={"script": "xgboost_training.py"},
            step_type="Training",
            framework_context="xgboost"
        )
        
        self.assertEqual(issue.level, SeverityLevel.ERROR)
        self.assertEqual(issue.category, "training_validation")
        self.assertEqual(issue.message, "Missing training loop")
        self.assertEqual(issue.step_type, "Training")
        self.assertEqual(issue.framework_context, "xgboost")
        self.assertEqual(issue.details["script"], "xgboost_training.py")
    
    def test_step_type_aware_issue_with_reference_examples(self):
        """Test StepTypeAwareAlignmentIssue with reference examples."""
        reference_examples = [
            "xgboost_training.py",
            "builder_xgboost_training_step.py"
        ]
        
        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.WARNING,
            category="pattern_validation",
            message="Missing model saving pattern",
            step_type="Training",
            framework_context="pytorch",
            reference_examples=reference_examples
        )
        
        self.assertEqual(issue.reference_examples, reference_examples)
        self.assertEqual(issue.framework_context, "pytorch")
        self.assertIn("xgboost_training.py", issue.reference_examples)
    
    def test_step_type_aware_issue_defaults(self):
        """Test StepTypeAwareAlignmentIssue default values."""
        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.INFO,
            category="general",
            message="Info message",
            step_type="Processing"
        )
        
        self.assertEqual(issue.step_type, "Processing")
        self.assertIsNone(issue.framework_context)
        self.assertEqual(issue.reference_examples, [])
        self.assertEqual(issue.details, {})
    
    def test_step_type_aware_issue_inheritance(self):
        """Test that StepTypeAwareAlignmentIssue inherits from AlignmentIssue."""
        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.ERROR,
            category="test",
            message="Test message",
            step_type="Training",
            recommendation="Fix this issue"
        )
        
        # Should have all AlignmentIssue properties
        self.assertEqual(issue.level, SeverityLevel.ERROR)
        self.assertEqual(issue.category, "test")
        self.assertEqual(issue.message, "Test message")
        self.assertEqual(issue.recommendation, "Fix this issue")
        
        # Plus step type specific properties
        self.assertEqual(issue.step_type, "Training")
    
    def test_step_type_aware_issue_serialization(self):
        """Test StepTypeAwareAlignmentIssue serialization."""
        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.WARNING,
            category="framework_validation",
            message="XGBoost pattern missing",
            step_type="Training",
            framework_context="xgboost",
            reference_examples=["xgboost_training.py"],
            details={"line": 42}
        )
        
        issue_dict = issue.model_dump()
        
        self.assertEqual(issue_dict["step_type"], "Training")
        self.assertEqual(issue_dict["framework_context"], "xgboost")
        self.assertEqual(issue_dict["reference_examples"], ["xgboost_training.py"])
        self.assertEqual(issue_dict["details"]["line"], 42)
    
    def test_step_type_aware_issue_json_serialization(self):
        """Test StepTypeAwareAlignmentIssue JSON serialization."""
        issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.ERROR,
            category="training_validation",
            message="Training validation failed",
            step_type="Training",
            framework_context="pytorch"
        )
        
        json_str = issue.model_dump_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("Training", json_str)
        self.assertIn("pytorch", json_str)
        self.assertIn("training_validation", json_str)

class TestCreateAlignmentIssueFunctions(unittest.TestCase):
    """Test alignment issue creation helper functions."""
    
    def test_create_alignment_issue_basic(self):
        """Test basic alignment issue creation."""
        issue = create_alignment_issue(
            level=SeverityLevel.ERROR,
            category="test_category",
            message="Test message"
        )
        
        self.assertIsInstance(issue, AlignmentIssue)
        self.assertEqual(issue.level, SeverityLevel.ERROR)
        self.assertEqual(issue.category, "test_category")
        self.assertEqual(issue.message, "Test message")
    
    def test_create_alignment_issue_with_details(self):
        """Test alignment issue creation with details."""
        issue = create_alignment_issue(
            level=SeverityLevel.WARNING,
            category="path_validation",
            message="Hardcoded path found",
            details={"path": "/opt/ml/input", "line": 10},
            recommendation="Use environment variables",
            alignment_level=AlignmentLevel.SCRIPT_CONTRACT
        )
        
        self.assertEqual(issue.details["path"], "/opt/ml/input")
        self.assertEqual(issue.details["line"], 10)
        self.assertEqual(issue.recommendation, "Use environment variables")
        self.assertEqual(issue.alignment_level, AlignmentLevel.SCRIPT_CONTRACT)
    
    def test_create_step_type_aware_alignment_issue_basic(self):
        """Test basic step type aware alignment issue creation."""
        issue = create_step_type_aware_alignment_issue(
            level=SeverityLevel.ERROR,
            category="training_validation",
            message="Training pattern missing",
            step_type="Training"
        )
        
        self.assertIsInstance(issue, StepTypeAwareAlignmentIssue)
        self.assertEqual(issue.step_type, "Training")
        self.assertEqual(issue.category, "training_validation")
    
    def test_create_step_type_aware_alignment_issue_with_framework(self):
        """Test step type aware alignment issue creation with framework."""
        issue = create_step_type_aware_alignment_issue(
            level=SeverityLevel.WARNING,
            category="framework_validation",
            message="XGBoost DMatrix not found",
            step_type="Training",
            framework_context="xgboost",
            details={"script": "train.py"},
            recommendation="Add DMatrix creation"
        )
        
        self.assertEqual(issue.framework_context, "xgboost")
        self.assertEqual(issue.details["script"], "train.py")
        self.assertEqual(issue.recommendation, "Add DMatrix creation")
    
    def test_create_step_type_aware_alignment_issue_with_reference_examples(self):
        """Test step type aware alignment issue creation with reference examples."""
        reference_examples = [
            "xgboost_training.py",
            "builder_xgboost_training_step.py"
        ]
        
        issue = create_step_type_aware_alignment_issue(
            level=SeverityLevel.INFO,
            category="step_type_info",
            message="Reference examples available",
            step_type="Training",
            reference_examples=reference_examples
        )
        
        self.assertEqual(issue.reference_examples, reference_examples)
        self.assertIn("xgboost_training.py", issue.reference_examples)

class TestSeverityLevelEnum(unittest.TestCase):
    """Test SeverityLevel enum."""
    
    def test_severity_level_values(self):
        """Test SeverityLevel enum values."""
        self.assertEqual(SeverityLevel.CRITICAL.value, "CRITICAL")
        self.assertEqual(SeverityLevel.ERROR.value, "ERROR")
        self.assertEqual(SeverityLevel.WARNING.value, "WARNING")
        self.assertEqual(SeverityLevel.INFO.value, "INFO")
    
    def test_severity_level_ordering(self):
        """Test SeverityLevel ordering for comparison."""
        # Test that we can compare severity levels by value
        severity_order = [
            SeverityLevel.INFO,
            SeverityLevel.WARNING,
            SeverityLevel.ERROR,
            SeverityLevel.CRITICAL
        ]
        
        # Test that each level has the expected string value
        self.assertEqual(SeverityLevel.INFO.value, "INFO")
        self.assertEqual(SeverityLevel.WARNING.value, "WARNING")
        self.assertEqual(SeverityLevel.ERROR.value, "ERROR")
        self.assertEqual(SeverityLevel.CRITICAL.value, "CRITICAL")

class TestAlignmentLevelEnum(unittest.TestCase):
    """Test AlignmentLevel enum."""
    
    def test_alignment_level_values(self):
        """Test AlignmentLevel enum values."""
        self.assertEqual(AlignmentLevel.SCRIPT_CONTRACT.value, 1)
        self.assertEqual(AlignmentLevel.CONTRACT_SPECIFICATION.value, 2)
        self.assertEqual(AlignmentLevel.SPECIFICATION_DEPENDENCY.value, 3)
        self.assertEqual(AlignmentLevel.BUILDER_CONFIGURATION.value, 4)

if __name__ == '__main__':
    unittest.main()
