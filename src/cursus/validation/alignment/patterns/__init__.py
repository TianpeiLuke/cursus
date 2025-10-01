"""
Pattern Recognition Module

This module contains pattern recognition and framework detection components
for the alignment validation system. It identifies common patterns in code,
configurations, and project structures to enhance validation accuracy.

Components:
- framework_patterns.py: Framework detection patterns (XGBoost, PyTorch, etc.)
- pattern_recognizer.py: General pattern recognition utilities and algorithms

Pattern Recognition Features:
- Framework detection from imports and code patterns
- Project structure pattern analysis
- Configuration pattern matching
- Code style and convention detection
- Naming pattern recognition for component relationships
"""

# Framework detection
from .framework_patterns import (
    detect_framework_from_imports,
    detect_framework_from_script_content,
    get_framework_patterns,
    get_all_framework_patterns,
    detect_training_patterns,
    detect_xgboost_patterns,
    detect_pytorch_patterns,
    detect_sklearn_patterns,
    detect_pandas_patterns,
)

# General pattern recognition
from .pattern_recognizer import (
    PatternRecognizer,
    ValidationPatternFilter,
)

__all__ = [
    # Framework detection
    "detect_framework_from_imports",
    "detect_framework_from_script_content", 
    "get_framework_patterns",
    "get_all_framework_patterns",
    "detect_training_patterns",
    "detect_xgboost_patterns",
    "detect_pytorch_patterns",
    "detect_sklearn_patterns",
    "detect_pandas_patterns",
    
    # Pattern recognition
    "PatternRecognizer",
    "ValidationPatternFilter",
]
