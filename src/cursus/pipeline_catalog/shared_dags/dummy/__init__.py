"""
Dummy Shared DAG Definitions

This module contains shared DAG definitions for dummy/testing pipelines.
"""

__all__ = ["create_dummy_e2e_basic_dag"]

# Import functions to make them available at package level
try:
    from .e2e_basic_dag import (
        create_dummy_e2e_basic_dag,
        get_dag_metadata as get_e2e_basic_metadata,
    )
except ImportError:
    pass
