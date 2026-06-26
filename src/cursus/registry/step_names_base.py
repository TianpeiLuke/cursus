"""
Base step-name registry — the single source of truth for step naming across config,
builders, and specifications.

This is a deliberately **dependency-free leaf module**: it loads the flat step table from
``step_names.yaml`` (next to this module) into ``STEP_NAMES`` and derives the
``CONFIG_STEP_REGISTRY`` / ``BUILDER_STEP_NAMES`` / ``SPEC_STEP_TYPES`` mappings. The
workspace-aware access layer (``step_names.py``) and the hybrid ``UnifiedRegistryManager``
(``hybrid/manager.py``) both read the raw data from here — keeping the data in this leaf
(rather than in ``step_names.py``) is what breaks the otherwise-circular import between the
access layer and the manager.

To add or edit a step, edit ``step_names.yaml`` — not this module.

(Formerly ``step_names_original.py``; renamed because it is the live source, not a backup.)
"""

from pathlib import Path
from typing import Dict

import yaml

# --- Load the step registry from YAML (the data source of truth) ---

_STEP_NAMES_YAML = Path(__file__).resolve().parent / "step_names.yaml"


def _load_step_names() -> Dict[str, Dict[str, str]]:
    """Load and validate the STEP_NAMES table from ``step_names.yaml``."""
    with open(_STEP_NAMES_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    step_names = data.get("step_names")
    if not isinstance(step_names, dict) or not step_names:
        raise ValueError(
            f"{_STEP_NAMES_YAML} must contain a non-empty 'step_names' mapping"
        )

    required_fields = {
        "config_class",
        "builder_step_name",
        "spec_type",
        "sagemaker_step_type",
        "description",
    }
    for name, info in step_names.items():
        if not isinstance(info, dict):
            raise ValueError(f"step '{name}' must map to an object, got {type(info)}")
        missing = required_fields - set(info)
        if missing:
            raise ValueError(
                f"step '{name}' is missing required field(s): {sorted(missing)}"
            )

    return step_names


# Core step name registry - canonical names used throughout the system.
STEP_NAMES: Dict[str, Dict[str, str]] = _load_step_names()

# Generate the mappings that existing code expects.
CONFIG_STEP_REGISTRY = {
    info["config_class"]: step_name for step_name, info in STEP_NAMES.items()
}

BUILDER_STEP_NAMES = {
    step_name: info["builder_step_name"] for step_name, info in STEP_NAMES.items()
}

# Generate step specification types.
SPEC_STEP_TYPES = {
    step_name: info["spec_type"] for step_name, info in STEP_NAMES.items()
}
