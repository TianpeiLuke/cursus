"""
Analysis Components Module

This module contains all analysis components for the alignment validation system.
Consolidated from previous analyzers/ and static_analysis/ folders for better organization.

Components:
- builder_analyzer.py: High-level builder analysis and validation
- builder_argument_extractor.py: AST-based argument extraction from builders
- config_analyzer.py: Configuration file analysis
- import_analyzer.py: Import statement analysis and validation
- path_extractor.py: Path extraction and validation utilities
- script_analyzer.py: Script file analysis and parsing
"""

# High-level analysis components
from .builder_analyzer import BuilderCodeAnalyzer, BuilderPatternAnalyzer
from .config_analyzer import ConfigurationAnalyzer
from .script_analyzer import ScriptAnalyzer

# Specialized extractors and parsers
from .builder_argument_extractor import BuilderArgumentExtractor, BuilderRegistry, extract_builder_arguments
from .import_analyzer import ImportAnalyzer
from .path_extractor import PathExtractor

# Convenience functions
def analyze_imports(imports, script_content):
    """Convenience function to analyze imports."""
    analyzer = ImportAnalyzer(imports, script_content)
    return analyzer.get_import_summary()

def extract_paths(script_content, script_lines):
    """Convenience function to extract paths."""
    extractor = PathExtractor(script_content, script_lines)
    return extractor.get_path_summary()

__all__ = [
    # High-level analyzers
    "BuilderCodeAnalyzer",
    "BuilderPatternAnalyzer",
    "ConfigurationAnalyzer", 
    "ScriptAnalyzer",
    
    # Specialized extractors
    "BuilderArgumentExtractor",
    "BuilderRegistry",
    "extract_builder_arguments",
    "ImportAnalyzer",
    "analyze_imports",
    "PathExtractor",
    "extract_paths",
]
