"""
MODS PyTorch Training Pipeline (Task-based View)

This is a reference to the MODS-enhanced PyTorch training pipeline from the mods_frameworks directory.
Provides MODS integration with automatic template registration, enhanced metadata extraction,
and advanced pipeline tracking capabilities.
"""

from ...mods_frameworks.pytorch.training.basic_training_mods import (
    create_dag,
    create_pipeline,
    fill_execution_document
)

# These imports allow this module to be used as a drop-in replacement
# for the original module, but with MODS enhancements
__all__ = ["create_dag", "create_pipeline", "fill_execution_document"]
