"""
Reporting and Visualization Module

This module contains all reporting, scoring, and visualization components
for the alignment validation system. It provides comprehensive reporting
capabilities with quality scoring and visual representations.

Components:
- validation_reporter.py: Consolidated reporting with comprehensive functionality

Reporting Features:
- Comprehensive validation result aggregation
- Quality scoring with weighted metrics
- Multiple export formats (JSON, HTML, text)
- Visual chart generation for score visualization
- Issue categorization and severity analysis
- Actionable recommendations generation
- Consolidated functionality from multiple previous reporters
"""

# Consolidated reporting - replaces alignment_reporter, alignment_scorer, enhanced_reporter
from .validation_reporter import (
    ValidationReporter,
    ReportingConfig,
    generate_quick_report,
    print_validation_summary,
    calculate_validation_scores,
)

__all__ = [
    # Consolidated reporting
    "ValidationReporter",
    "ReportingConfig",
    
    # Convenience functions
    "generate_quick_report",
    "print_validation_summary", 
    "calculate_validation_scores",
]
