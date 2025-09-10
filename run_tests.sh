#!/bin/bash
# Test runner script that sets PYTHONPATH before running pytest
# This ensures that src/ is in the Python path during test collection

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

# Run pytest with all arguments passed through
python -m pytest "$@"
