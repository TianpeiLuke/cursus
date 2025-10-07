"""
Generalized Config UI - Universal Configuration Interface System

This module provides a universal interface for creating, editing, and managing
any configuration that follows the BasePipelineConfig pattern with .from_base_config() 
method support.

Key Components:
- UniversalConfigCore: Core engine for configuration management
- MultiStepWizard: Multi-step pipeline configuration wizard
- Factory functions: Easy widget creation interface
"""

from .core import UniversalConfigCore
from .widget import MultiStepWizard, UniversalConfigWidget
from .utils import create_config_widget, create_pipeline_config_widget

__all__ = [
    'UniversalConfigCore',
    'MultiStepWizard', 
    'UniversalConfigWidget',
    'create_config_widget',
    'create_pipeline_config_widget'
]

__version__ = "1.0.0"
