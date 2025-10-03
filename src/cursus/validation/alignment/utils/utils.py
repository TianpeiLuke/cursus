"""
Utility functions for alignment validation.

Provides common utility functions used across alignment validation components.
"""

import os
from typing import List, Optional, Dict, Any
from .validation_models import ValidationIssue, IssueLevel


def normalize_path(path: str) -> str:
    """
    Normalize a path for comparison purposes.

    Args:
        path: Path to normalize

    Returns:
        Normalized path string
    """
    if path is None:
        return ""
    return os.path.normpath(path).replace("\\", "/")


def extract_logical_name_from_path(path: str) -> Optional[str]:
    """
    Extract logical name from a SageMaker path.

    For paths like '/opt/ml/processing/input/data', extracts 'data'.

    Args:
        path: SageMaker path

    Returns:
        Logical name or None if not extractable
    """
    # Common SageMaker path patterns
    patterns = [
        "/opt/ml/processing/input/",
        "/opt/ml/processing/output/",
        "/opt/ml/input/data/",
        "/opt/ml/model/",
        "/opt/ml/output/",
    ]

    normalized_path = normalize_path(path)

    for pattern in patterns:
        if normalized_path.startswith(pattern):
            remainder = normalized_path[len(pattern) :].strip("/")
            if remainder:
                # Return the first path component as logical name
                return remainder.split("/")[0]

    return None


def is_sagemaker_path(path: str) -> bool:
    """
    Check if a path is a SageMaker container path.

    Args:
        path: Path to check

    Returns:
        True if this is a SageMaker path
    """
    sagemaker_prefixes = [
        "/opt/ml/processing/",
        "/opt/ml/input/",
        "/opt/ml/model",
        "/opt/ml/output",
        "/opt/ml/code",
    ]

    normalized_path = normalize_path(path)
    return any(normalized_path.startswith(prefix) for prefix in sagemaker_prefixes)


def format_alignment_issue(issue: ValidationIssue) -> str:
    """
    Format a validation issue for display.

    Args:
        issue: The validation issue to format

    Returns:
        Formatted string representation
    """
    level_emoji = {
        IssueLevel.ERROR: "❌",
        IssueLevel.WARNING: "⚠️",
        IssueLevel.INFO: "ℹ️",
    }

    emoji = level_emoji.get(issue.level, "")
    level_name = issue.level.value

    result = f"{emoji} {level_name}: {issue.message}"

    if hasattr(issue, 'recommendation') and issue.recommendation:
        result += f"\n  💡 Recommendation: {issue.recommendation}"

    if issue.details:
        result += f"\n  📋 Details: {issue.details}"

    return result


def group_issues_by_severity(
    issues: List[ValidationIssue],
) -> Dict[IssueLevel, List[ValidationIssue]]:
    """
    Group validation issues by severity level.

    Args:
        issues: List of validation issues

    Returns:
        Dictionary mapping severity levels to lists of issues
    """
    grouped = {level: [] for level in IssueLevel}

    for issue in issues:
        grouped[issue.level].append(issue)

    return grouped


def get_highest_severity(issues: List[ValidationIssue]) -> Optional[IssueLevel]:
    """
    Get the highest severity level among a list of issues.

    Args:
        issues: List of validation issues

    Returns:
        Highest severity level or None if no issues
    """
    if not issues:
        return None

    severity_order = [
        IssueLevel.ERROR,
        IssueLevel.WARNING,
        IssueLevel.INFO,
    ]

    for severity in severity_order:
        if any(issue.level == severity for issue in issues):
            return severity

    return None


def validate_environment_setup() -> List[str]:
    """
    Validate that the environment is properly set up for alignment validation.

    Returns:
        List of validation issues found
    """
    issues = []

    # Check for required directories
    required_dirs = [
        "src/cursus/steps/scripts",
        "src/cursus/steps/contracts",
        "src/cursus/steps/specs",
        "src/cursus/steps/builders",
        "src/cursus/steps/configs",
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            issues.append(f"Required directory not found: {dir_path}")

    return issues


def get_validation_summary_stats(issues: List[ValidationIssue]) -> Dict[str, Any]:
    """
    Get summary statistics for a list of validation issues.

    Args:
        issues: List of validation issues

    Returns:
        Dictionary with summary statistics
    """
    if not issues:
        return {
            "total_issues": 0,
            "by_severity": {level.value: 0 for level in IssueLevel},
            "highest_severity": None,
            "has_errors": False,
        }

    grouped = group_issues_by_severity(issues)
    highest = get_highest_severity(issues)

    return {
        "total_issues": len(issues),
        "by_severity": {level.value: len(grouped[level]) for level in IssueLevel},
        "highest_severity": highest.value if highest else None,
        "has_errors": len(grouped[IssueLevel.ERROR]) > 0,
    }
