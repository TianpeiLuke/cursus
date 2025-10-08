"""
Widget components for universal configuration management.

This module contains UI widgets for both web and Jupyter interfaces.
"""

# Handle both relative and absolute imports using centralized path setup
try:
    # Try relative imports first (when run as module)
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
    from .native import (
        NativeConfigWidget,
        NativePipelineWidget,
        NativeFieldRenderer,
        NativeFileManager,
        create_native_config_widget,
        create_native_pipeline_widget
    )
except ImportError:
    # Fallback: Set up cursus path and use absolute imports
    import sys
    from pathlib import Path
    
    # Add the core directory to path for import_utils
    current_dir = Path(__file__).parent
    core_dir = current_dir.parent / 'core'
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    
    from import_utils import ensure_cursus_path
    ensure_cursus_path()
    
    from cursus.api.config_ui.widgets.widget import MultiStepWizard, UniversalConfigWidget
    from cursus.api.config_ui.widgets.specialized_widgets import (
        HyperparametersConfigWidget,
        SpecializedComponentRegistry,
        create_specialized_widget
    )
    from cursus.api.config_ui.widgets.jupyter_widget import (
        UniversalConfigWidget as JupyterUniversalConfigWidget,
        CompleteConfigUIWidget as JupyterPipelineConfigWidget,
        create_config_widget as create_jupyter_config_widget,
        create_complete_config_ui_widget as create_jupyter_pipeline_config_widget,
        EnhancedSaveAllMergedWidget as UniversalConfigWidgetWithServer
    )
    from cursus.api.config_ui.widgets.native import (
        NativeConfigWidget,
        NativePipelineWidget,
        NativeFieldRenderer,
        NativeFileManager,
        create_native_config_widget,
        create_native_pipeline_widget
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
    'UniversalConfigWidgetWithServer',
    'NativeConfigWidget',
    'NativePipelineWidget',
    'NativeFieldRenderer',
    'NativeFileManager',
    'create_native_config_widget',
    'create_native_pipeline_widget'
]
