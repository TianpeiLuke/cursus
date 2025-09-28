"""
Pattern Recognition for Alignment Validation

This package contains pattern recognition engines that identify acceptable
architectural patterns and filter false positives in validation results.
"""

from .pattern_recognizer import PatternRecognizer
from ....step_catalog.adapters.file_resolver import HybridFileResolverAdapter as HybridFileResolver

__all__ = ["PatternRecognizer", "HybridFileResolver"]
