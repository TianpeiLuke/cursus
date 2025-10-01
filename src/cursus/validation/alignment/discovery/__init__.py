"""
Discovery and I/O Operations Module

This module contains all data loading, file processing, and orchestration components.
Consolidated from previous loaders/, processors/, and orchestration/ folders for better organization.

Components:
- contract_loader.py: Contract file loading and parsing utilities
- specification_loader.py: Specification file loading and parsing utilities
- spec_file_processor.py: Specification file processing and transformation
- validation_orchestrator.py: Validation workflow orchestration and coordination

The discovery module handles:
- File discovery and loading across different component types
- Data parsing and preprocessing for validation
- Orchestration of validation workflows
- Integration with step catalog for workspace-aware discovery
"""

# Data loaders
from .contract_loader import ContractLoader
from .specification_loader import SpecificationLoader

# File processors
from .spec_file_processor import SpecificationFileProcessor


__all__ = [
    # Loaders
    "ContractLoader",
    "SpecificationLoader",
    
    # Processors
    "SpecificationFileProcessor",
]
