"""
Multi-Task Learning (MTL) Shared DAG Definitions

This module provides shared DAG definitions for multi-task learning pipelines
using LightGBMMT (LightGBM Multi-Task) framework.
"""

from .complete_e2e_dag import (
    create_lightgbmmt_complete_e2e_dag,
    get_dag_metadata as get_complete_e2e_dag_metadata,
    validate_dag_structure as validate_complete_e2e_dag_structure,
)

from .lightgbmmt_with_label_ruleset_e2e_dag import (
    create_lightgbmmt_with_label_ruleset_e2e_dag,
    get_dag_metadata as get_label_ruleset_dag_metadata,
    validate_dag_structure as validate_label_ruleset_dag_structure,
)

from .ssl_training_dag import (
    create_lightgbmmt_ssl_training_dag,
    get_dag_metadata as get_ssl_training_dag_metadata,
    validate_dag_structure as validate_ssl_training_dag_structure,
)

__all__ = [
    # Complete E2E DAG
    "create_lightgbmmt_complete_e2e_dag",
    "get_complete_e2e_dag_metadata",
    "validate_complete_e2e_dag_structure",
    # Label Ruleset E2E DAG
    "create_lightgbmmt_with_label_ruleset_e2e_dag",
    "get_label_ruleset_dag_metadata",
    "validate_label_ruleset_dag_structure",
    # SSL Training DAG
    "create_lightgbmmt_ssl_training_dag",
    "get_ssl_training_dag_metadata",
    "validate_ssl_training_dag_structure",
]
