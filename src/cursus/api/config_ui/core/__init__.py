"""
Core engine and utilities for universal configuration management.

This module contains the core business logic for configuration discovery,
management, and processing.
"""

from .core import UniversalConfigCore, create_config_widget
from .dag_manager import DAGConfigurationManager, create_pipeline_config_widget, analyze_pipeline_dag
from .utils import discover_available_configs

__all__ = [
    'UniversalConfigCore',
    'DAGConfigurationManager', 
    'create_config_widget',
    'create_pipeline_config_widget',
    'analyze_pipeline_dag',
    'discover_available_configs'
]
