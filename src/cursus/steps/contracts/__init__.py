"""
Script Contracts Module.

The per-step *_CONTRACT data files were removed: contract data now comes from the
unified .step.yaml interfaces (see cursus.steps.interfaces.load_step_interface and
cursus.core.base.step_interface.ContractSection). This package retains only the
contract validation tooling (ScriptAnalyzer-based) used to check script
implementations against their interfaces.
"""

# Contract validation helpers (no per-step data).
from ...core.base.contract_base import ValidationResult, ScriptAnalyzer
from .training_script_contract import TrainingScriptContract, TrainingScriptAnalyzer

__all__ = [
    "ValidationResult",
    "ScriptAnalyzer",
    "TrainingScriptContract",
    "TrainingScriptAnalyzer",
]
