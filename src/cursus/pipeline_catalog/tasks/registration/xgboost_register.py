"""
XGBoost Registration Pipeline (Task-based View)

This is a reference to the XGBoost end-to-end pipeline which includes model registration.
"""

from src.cursus.pipeline_catalog.frameworks.xgboost.end_to_end.complete_e2e import (
    create_dag,
    create_pipeline,
    fill_execution_document
)

# These imports allow this module to be used as a drop-in replacement
# for the original module
__all__ = ["create_dag", "create_pipeline", "fill_execution_document"]
