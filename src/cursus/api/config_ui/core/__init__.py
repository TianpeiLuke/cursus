"""
Core engine and utilities for universal configuration management.

This module contains the core business logic for configuration discovery,
management, and processing.
"""

# Handle both relative and absolute imports using centralized path setup
try:
    # Try relative imports first (when run as module)
    from .core import UniversalConfigCore, create_config_widget
    # DAGConfigurationManager functionality replaced by direct factory imports
    from ...factory import DAGConfigFactory, ConfigClassMapper
    from .utils import discover_available_configs
except ImportError:
    # Fallback: Set up cursus path and use absolute imports
    import sys
    from pathlib import Path
    
    # Add the core directory to path for import_utils
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from import_utils import ensure_cursus_path
    ensure_cursus_path()
    
    from cursus.api.config_ui.core.core import UniversalConfigCore, create_config_widget
    from cursus.api.config_ui.core.dag_manager import DAGConfigurationManager, create_pipeline_config_widget, analyze_pipeline_dag
    from cursus.api.config_ui.core.utils import discover_available_configs

__all__ = [
    'UniversalConfigCore',
    'DAGConfigurationManager', 
    'create_config_widget',
    'create_pipeline_config_widget',
    'analyze_pipeline_dag',
    'discover_available_configs'
]
