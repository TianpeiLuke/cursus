"""
Web interface components for universal configuration management.

This module contains FastAPI endpoints and web-specific functionality.
"""

from .api import router, create_config_ui_app

__all__ = [
    'router',
    'create_config_ui_app'
]
