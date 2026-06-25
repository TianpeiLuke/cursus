"""
Validators Module

This module contains all validation logic and rules for the alignment
validation system. Validators implement specific validation algorithms
and rule sets for different aspects of component alignment.

Components:
- contract_spec_validator.py: Contract and specification alignment validation
- dependency_classifier.py: Dependency classification and categorization
- dependency_validator.py: Dependency relationship validation
- property_path_validator.py: Property path validation and verification
- script_contract_validator.py: Script and contract alignment validation
- testability_validator.py: Testability pattern validation

Validation Features:
- Rule-based validation logic
- Configurable validation severity levels
- Detailed error reporting with recommendations
- Pattern-based validation algorithms
- Cross-component relationship validation
"""

# Core validators
from .dependency_validator import DependencyValidator

from .property_path_validator import SageMakerPropertyPathValidator

# Note: contract_spec_validator (ConsolidatedContractSpecValidator) was removed —
# Contract<->Spec logical-name/IO alignment is now enforced at StepInterface
# construction time (StepInterface._sync_and_align), so the Level-2 validator is
# redundant. SageMaker property-path validation remains via SageMakerPropertyPathValidator.

__all__ = [
    "DependencyValidator",
    "SageMakerPropertyPathValidator",
]
