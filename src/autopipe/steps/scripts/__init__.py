"""
Pipeline scripts module.

This module contains the actual processing scripts used by pipeline steps,
along with utilities for contract enforcement and validation.
"""

from .contract_utils import (
    validate_contract_environment,
    get_contract_paths,
    get_input_path,
    get_output_path,
    validate_required_files,
    log_contract_summary,
    find_files_in_input,
    create_output_file_path,
    validate_framework_requirements,
    ContractEnforcer
)

# Note: The actual script modules (currency_conversion, dummy_training, etc.)
# are typically executed as standalone scripts and don't need to be imported
# as modules. They are included here for completeness but may not be directly
# importable due to their script-oriented design.

__all__ = [
    # Contract utilities
    "validate_contract_environment",
    "get_contract_paths",
    "get_input_path",
    "get_output_path",
    "validate_required_files",
    "log_contract_summary",
    "find_files_in_input",
    "create_output_file_path",
    "validate_framework_requirements",
    "ContractEnforcer",
]
