"""
Import Resolution Utilities

Centralized path setup for both development (src/) and pip-installed scenarios.
This module sets up the Python path once, allowing normal imports to work everywhere.
"""

import sys
from pathlib import Path


def setup_cursus_path():
    """
    Set up cursus package path for both development and pip-installed scenarios.
    
    This function detects whether we're in:
    1. Development mode: src/cursus structure
    2. Pip-installed mode: direct cursus structure
    
    And adds the appropriate path to sys.path for absolute imports.
    
    Returns:
        Path: The cursus root directory that was added to sys.path
    """
    # Find the cursus package root by walking up the directory tree
    current_path = Path(__file__).parent
    cursus_root = None
    
    # Walk up to find cursus package root
    while current_path.parent != current_path:
        # Check if we're in development mode (src/cursus structure)
        if (current_path.parent / 'src' / 'cursus').exists():
            cursus_root = current_path.parent / 'src'
            break
        # Check if we're in pip-installed mode (direct cursus structure)
        elif current_path.name == 'cursus' and (current_path / 'core').exists():
            cursus_root = current_path.parent
            break
        current_path = current_path.parent
    
    # Add the appropriate path to sys.path
    if cursus_root and str(cursus_root) not in sys.path:
        sys.path.insert(0, str(cursus_root))
    
    return cursus_root


# Global flag to ensure we only set up the path once
_path_setup_done = False


def ensure_cursus_path():
    """
    Ensure cursus path is set up (idempotent - safe to call multiple times).
    
    Returns:
        Path: The cursus root directory
    """
    global _path_setup_done
    
    if not _path_setup_done:
        cursus_root = setup_cursus_path()
        _path_setup_done = True
        return cursus_root
    
    # Return the already configured path
    for path in sys.path:
        path_obj = Path(path)
        if (path_obj / 'cursus').exists():
            return path_obj
    
    # Fallback: set up again if not found
    return setup_cursus_path()
