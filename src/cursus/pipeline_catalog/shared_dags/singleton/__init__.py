"""
Singleton DAG Definitions

This module contains shared DAG definitions for single-step pipelines,
following the Zettelkasten principle of atomicity.
"""

__all__ = [
    "create_cradle_data_loading_singleton_dag",
]

# Import functions to make them available at package level
try:
    from .cradle_data_loading_dag import (
        create_cradle_data_loading_singleton_dag,
        get_dag_metadata as get_cradle_data_loading_metadata,
    )
except ImportError:
    pass
