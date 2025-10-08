"""
Widget components for universal configuration management.

This module contains UI widgets for both web and Jupyter interfaces.
"""

from .widget import MultiStepWizard, UniversalConfigWidget
from .specialized_widgets import (
    HyperparametersConfigWidget,
    SpecializedComponentRegistry,
    create_specialized_widget
)
from .jupyter_widget import (
    UniversalConfigWidget as JupyterUniversalConfigWidget,
    CompleteConfigUIWidget as JupyterPipelineConfigWidget,
    create_config_widget as create_jupyter_config_widget,
    create_complete_config_ui_widget as create_jupyter_pipeline_config_widget,
    EnhancedSaveAllMergedWidget as UniversalConfigWidgetWithServer
)

__all__ = [
    'MultiStepWizard',
    'UniversalConfigWidget', 
    'HyperparametersConfigWidget',
    'SpecializedComponentRegistry',
    'create_specialized_widget',
    'JupyterUniversalConfigWidget',
    'JupyterPipelineConfigWidget',
    'create_jupyter_config_widget', 
    'create_jupyter_pipeline_config_widget',
    'UniversalConfigWidgetWithServer'
]
