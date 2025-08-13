"""
Pattern Recognition for Alignment Validation

This package contains pattern recognition engines that identify acceptable
architectural patterns and filter false positives in validation results.
"""

from .pattern_recognizer import PatternRecognizer
from .file_resolver import HybridFileResolver

__all__ = [
    'PatternRecognizer',
    'HybridFileResolver'
]
