"""
Test configuration and fixtures for the cursus test suite.

This conftest.py ensures that the src/ directory is properly added to the Python path
for all tests, eliminating the need for manual sys.path manipulation in individual test files.
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path for local development testing
# This needs to happen as early as possible, before any imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Convert to absolute path and add to sys.path
src_path_str = str(src_path.resolve())
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# Also set PYTHONPATH environment variable for subprocess calls
current_pythonpath = os.environ.get('PYTHONPATH', '')
if src_path_str not in current_pythonpath:
    if current_pythonpath:
        os.environ['PYTHONPATH'] = f"{src_path_str}{os.pathsep}{current_pythonpath}"
    else:
        os.environ['PYTHONPATH'] = src_path_str

# Verify the path was added correctly
print(f"conftest.py: Added {src_path_str} to Python path")
