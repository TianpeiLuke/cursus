"""
Unified Alignment Tester Module

This module provides comprehensive validation of alignment rules between
scripts, contracts, specifications, and builders in the pipeline architecture.

The alignment validation covers four levels:
1. Script ↔ Contract Alignment
2. Contract ↔ Specification Alignment  
3. Specification ↔ Dependencies Alignment
4. Builder ↔ Configuration Alignment

Consolidated Structure (Post Phase 4):
- config/: Configuration and validation rulesets
- core/: Core alignment testers for each level
- reporting/: Consolidated reporting and scoring (validation_reporter.py)
- utils/: Consolidated utilities and models (validation_models.py)
- validators/: Remaining validation logic and rules
"""

# Main orchestrator
from .unified_alignment_tester import UnifiedAlignmentTester

# Core alignment testers
from .core.script_contract_alignment import ScriptContractAlignmentTester
from .core.contract_spec_alignment import ContractSpecificationAlignmentTester
from .core.spec_dependency_alignment import SpecificationDependencyAlignmentTester

# Consolidated reporting components
from .reporting import (
    ValidationReporter,
    ReportingConfig,
    generate_quick_report,
    print_validation_summary,
    calculate_validation_scores,
)

# Remaining validators
from .validators import (
    DependencyValidator,
    SageMakerPropertyPathValidator,
)

# Consolidated utilities and models
from .utils import (
    ValidationLevel,
    ValidationStatus,
    IssueLevel,
    RuleType,
    ValidationIssue,
    ValidationResult,
    ValidationSummary,
    MethodValidationInfo,
    StepValidationContext,
    create_validation_issue,
    create_validation_result,
    merge_validation_results,
    filter_issues_by_level,
    group_issues_by_method,
    format_validation_summary,
)

__all__ = [
    # Main orchestrator
    "UnifiedAlignmentTester",
    
    # Core alignment testers
    "ScriptContractAlignmentTester",
    "ContractSpecificationAlignmentTester", 
    "SpecificationDependencyAlignmentTester",
    
    # Consolidated reporting
    "ValidationReporter",
    "ReportingConfig",
    "generate_quick_report",
    "print_validation_summary",
    "calculate_validation_scores",
    
    # Remaining validators
    "DependencyValidator",
    "SageMakerPropertyPathValidator",
    
    # Consolidated utilities and models
    "ValidationLevel",
    "ValidationStatus",
    "IssueLevel",
    "RuleType",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSummary",
    "MethodValidationInfo",
    "StepValidationContext",
    "create_validation_issue",
    "create_validation_result",
    "merge_validation_results",
    "filter_issues_by_level",
    "group_issues_by_method",
    "format_validation_summary",
]
