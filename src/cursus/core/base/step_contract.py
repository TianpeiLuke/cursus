"""
StepContract — immutable data loaded from .step.yaml.

Replaces ScriptContract and TrainingScriptContract Pydantic models.
No path-convention validation; YAML is the single source of truth.
Path validation belongs in CI/lint tooling (JSON Schema), not in the runtime data holder.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class StepContract:
    """Immutable contract data loaded from .step.yaml."""

    entry_point: str
    expected_input_paths: Dict[str, str]
    expected_output_paths: Dict[str, str]
    expected_arguments: Dict[str, str] = field(default_factory=dict)
    required_env_vars: List[str] = field(default_factory=list)
    optional_env_vars: Dict[str, str] = field(default_factory=dict)
    framework_requirements: Dict[str, str] = field(default_factory=dict)
    description: str = ""
