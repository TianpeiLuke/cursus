"""
Unified Alignment Tester Module

This module provides comprehensive validation of alignment rules between
scripts, contracts, specifications, and builders in the pipeline architecture.

The alignment validation covers four levels:
1. Script ↔ Contract Alignment
2. Contract ↔ Specification Alignment  
3. Specification ↔ Dependencies Alignment
4. Builder ↔ Configuration Alignment
"""

from .unified_alignment_tester import UnifiedAlignmentTester
from .reporting.alignment_reporter import AlignmentReport, ValidationResult, AlignmentIssue
from .core.script_contract_alignment import ScriptContractAlignmentTester
from .core.contract_spec_alignment import ContractSpecificationAlignmentTester
from .core.spec_dependency_alignment import SpecificationDependencyAlignmentTester
from .core.builder_config_alignment import BuilderConfigurationAlignmentTester
from .validators.testability_validator import TestabilityPatternValidator

__all__ = [
    "UnifiedAlignmentTester",
    "AlignmentReport",
    "ValidationResult",
    "AlignmentIssue",
    "ScriptContractAlignmentTester",
    "ContractSpecificationAlignmentTester",
    "SpecificationDependencyAlignmentTester",
    "BuilderConfigurationAlignmentTester",
    "TestabilityPatternValidator",
]
