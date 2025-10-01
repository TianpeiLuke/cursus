"""
Unified Alignment Tester Module

This module provides comprehensive validation of alignment rules between
scripts, contracts, specifications, and builders in the pipeline architecture.

The alignment validation covers four levels:
1. Script ↔ Contract Alignment
2. Contract ↔ Specification Alignment  
3. Specification ↔ Dependencies Alignment
4. Builder ↔ Configuration Alignment

Optimized Structure:
- analysis/: All analysis components (AST parsing, imports, paths, etc.)
- core/: Core alignment testers for each level
- discovery/: Data loading, file processing, and orchestration
- factories/: Factory classes for component selection
- patterns/: Pattern recognition and framework detection
- reporting/: Reporting, scoring, and visualization
- step_type_enhancers/: Step-type specific enhancement logic
- utils/: Utilities, models, and configuration
- validators/: Validation logic and rules
"""

# Main orchestrator
from .unified_alignment_tester import UnifiedAlignmentTester

# Core alignment testers
from .core.script_contract_alignment import ScriptContractAlignmentTester
from .core.contract_spec_alignment import ContractSpecificationAlignmentTester
from .core.spec_dependency_alignment import SpecificationDependencyAlignmentTester
from .core.builder_config_alignment import BuilderConfigurationAlignmentTester

# Reporting components
from .reporting.alignment_reporter import AlignmentReport, ValidationResult, AlignmentSummary
from .reporting.alignment_scorer import AlignmentScorer
from .reporting.enhanced_reporter import EnhancedAlignmentReport

# Key validators
from .validators.testability_validator import TestabilityPatternValidator
from .validators.contract_spec_validator import ContractSpecValidator
from .validators.dependency_validator import DependencyValidator

# Utilities
from .utils.core_models import (
    SeverityLevel,
    AlignmentLevel,
    create_alignment_issue,
    create_step_type_aware_alignment_issue,
)

__all__ = [
    # Main orchestrator
    "UnifiedAlignmentTester",
    
    # Core alignment testers
    "ScriptContractAlignmentTester",
    "ContractSpecificationAlignmentTester", 
    "SpecificationDependencyAlignmentTester",
    "BuilderConfigurationAlignmentTester",
    
    # Reporting
    "AlignmentReport",
    "ValidationResult",
    "AlignmentSummary",
    "AlignmentScorer",
    "EnhancedAlignmentReport",
    
    # Validators
    "TestabilityPatternValidator",
    "ContractSpecValidator",
    "DependencyValidator",
    
    # Utilities
    "SeverityLevel",
    "AlignmentLevel",
    "create_alignment_issue",
    "create_step_type_aware_alignment_issue",
]
