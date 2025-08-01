"""
Core base classes for the autopipe framework.

This module provides the foundational base classes that are used throughout
the autopipe framework for configuration, contracts, specifications, and builders.
"""

from .enums import DependencyType, NodeType
from .contract_base import ScriptContract, ValidationResult, ScriptAnalyzer
from .hyperparameters_base import ModelHyperparameters
from .config_base import BasePipelineConfig
from .specification_base import DependencySpec, OutputSpec, StepSpecification
from .builder_base import StepBuilderBase

__all__ = [
    # Enums
    'DependencyType',
    'NodeType',
    
    # Contract classes
    'ScriptContract',
    'ValidationResult', 
    'ScriptAnalyzer',
    
    # Hyperparameters
    'ModelHyperparameters',
    
    # Configuration
    'BasePipelineConfig',
    
    # Specifications
    'DependencySpec',
    'OutputSpec',
    'StepSpecification',
    
    # Builders
    'StepBuilderBase',
]
