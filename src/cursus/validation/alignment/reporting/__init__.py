"""
Reporting and Visualization Module

This module contains all reporting, scoring, and visualization components
for the alignment validation system. It provides comprehensive reporting
capabilities with quality scoring and visual representations.

Components:
- alignment_reporter.py: Core reporting classes and validation result management
- alignment_scorer.py: Quality scoring algorithms and rating systems
- enhanced_reporter.py: Enhanced reporting with advanced formatting and exports

Reporting Features:
- Comprehensive validation result aggregation
- Quality scoring with weighted metrics
- Multiple export formats (JSON, HTML, etc.)
- Visual chart generation for score visualization
- Issue categorization and severity analysis
- Actionable recommendations generation
"""

# Core reporting
from .alignment_reporter import (
    AlignmentReport,
    ValidationResult,
    AlignmentSummary,
)

# Scoring system
from .alignment_scorer import (
    AlignmentScorer,
    score_alignment_results,
)

# Enhanced reporting
from .enhanced_reporter import (
    EnhancedAlignmentReport,
)

__all__ = [
    # Core reporting
    "AlignmentReport",
    "ValidationResult", 
    "AlignmentSummary",
    
    # Scoring
    "AlignmentScorer",
    "score_alignment_results",
    
    # Enhanced reporting
    "EnhancedAlignmentReport",
]
