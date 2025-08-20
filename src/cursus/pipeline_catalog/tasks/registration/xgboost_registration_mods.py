"""
MODS XGBoost Registration Pipeline (Task-based View)

This is a reference to the MODS-enhanced XGBoost end-to-end pipeline which includes model registration from the mods_frameworks directory.
Provides MODS integration with automatic template registration, enhanced metadata extraction,
and advanced pipeline tracking capabilities.
"""

from ...mods_frameworks.xgboost.end_to_end.complete_e2e_mods import (
    create_dag,
    create_pipeline,
    fill_execution_document
)

# These imports allow this module to be used as a drop-in replacement
# for the original module, but with MODS enhancements
__all__ = ["create_dag", "create_pipeline", "fill_execution_document"]
