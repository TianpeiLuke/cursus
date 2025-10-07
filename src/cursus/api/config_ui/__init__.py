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

from .core import UniversalConfigCore, create_config_widget
from .dag_manager import DAGConfigurationManager, create_pipeline_config_widget, analyze_pipeline_dag
from .widget import MultiStepWizard, UniversalConfigWidget
from .specialized_widgets import (
    HyperparametersConfigWidget, 
    SpecializedComponentRegistry,
    create_specialized_widget
)
from .jupyter_widget import (
    UniversalConfigWidget as JupyterUniversalConfigWidget,
    PipelineConfigWidget as JupyterPipelineConfigWidget,
    create_config_widget as create_jupyter_config_widget,
    create_pipeline_config_widget as create_jupyter_pipeline_config_widget,
    UniversalConfigWidgetWithServer
)

__all__ = [
    'UniversalConfigCore',
    'DAGConfigurationManager',
    'MultiStepWizard', 
    'UniversalConfigWidget',
    'HyperparametersConfigWidget',
    'SpecializedComponentRegistry',
    'create_config_widget',
    'create_pipeline_config_widget',
    'analyze_pipeline_dag',
    'create_specialized_widget',
    # Jupyter widget exports
    'JupyterUniversalConfigWidget',
    'JupyterPipelineConfigWidget', 
    'create_jupyter_config_widget',
    'create_jupyter_pipeline_config_widget',
    'UniversalConfigWidgetWithServer'
]

__version__ = "1.0.0"
