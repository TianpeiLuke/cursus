"""
XGBoost Shared DAG Definitions

This module contains shared DAG definitions for XGBoost-based pipelines.
"""

__all__ = [
    "create_xgboost_simple_dag",
    "create_xgboost_training_with_calibration_dag",
    "create_xgboost_training_with_evaluation_dag",
    "create_xgboost_complete_e2e_dag",
]

# Import functions to make them available at package level
try:
    from .simple_dag import (
        create_xgboost_simple_dag,
        get_dag_metadata as get_simple_metadata,
    )
except ImportError:
    pass

try:
    from .training_with_calibration_dag import (
        create_xgboost_training_with_calibration_dag,
        get_dag_metadata as get_training_calibration_metadata,
    )
except ImportError:
    pass

try:
    from .training_with_evaluation_dag import (
        create_xgboost_training_with_evaluation_dag,
        get_dag_metadata as get_training_evaluation_metadata,
    )
except ImportError:
    pass

try:
    from .complete_e2e_dag import (
        create_xgboost_complete_e2e_dag,
        get_dag_metadata as get_complete_e2e_metadata,
    )
except ImportError:
    pass
