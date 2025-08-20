"""
XGBoost Shared DAG Definitions

This module contains shared DAG definitions for XGBoost-based pipelines.
"""

__all__ = [
    "create_xgboost_simple_dag",
    "create_xgboost_training_dag", 
    "create_xgboost_end_to_end_dag"
]

# Import functions to make them available at package level
try:
    from .simple_dag import create_xgboost_simple_dag, get_dag_metadata as get_simple_metadata
except ImportError:
    pass

try:
    from .training_dag import create_xgboost_training_dag, get_dag_metadata as get_training_metadata
except ImportError:
    pass

try:
    from .end_to_end_dag import create_xgboost_end_to_end_dag, get_dag_metadata as get_end_to_end_metadata
except ImportError:
    pass
