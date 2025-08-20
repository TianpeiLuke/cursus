"""
Cradle Data Loading Pipeline (Task-based View)

This is a reference to the Cradle data loading pipeline from the components directory.
"""

from src.cursus.pipeline_catalog.components.cradle_dataload import (
    create_dag,
    create_pipeline,
    fill_execution_document,
    save_execution_document
)

# These imports allow this module to be used as a drop-in replacement
# for the original module
__all__ = ["create_dag", "create_pipeline", "fill_execution_document", "save_execution_document"]
