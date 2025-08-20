"""
XGBoost Training Pipeline (Task-based View)

This is a reference to the basic XGBoost training pipeline from the frameworks directory.
"""

from src.cursus.pipeline_catalog.frameworks.xgboost.training.with_evaluation import (
    create_dag,
    create_pipeline,
    fill_execution_document
)

# These imports allow this module to be used as a drop-in replacement
# for the original module
__all__ = ["create_dag", "create_pipeline", "fill_execution_document"]
