"""
Base step-name registry — the single source of truth for step naming across config,
builders, and specifications.

This is a deliberately **dependency-free leaf module**: it builds the flat step table
``STEP_NAMES`` from the per-step ``.step.yaml`` ``registry:`` blocks (via
``build_registry_from_interfaces``) and derives the ``CONFIG_STEP_REGISTRY`` /
``BUILDER_STEP_NAMES`` / ``SPEC_STEP_TYPES`` mappings. The workspace-aware access layer
(``step_names.py``) and the hybrid ``UnifiedRegistryManager`` (``hybrid/manager.py``) both
read the raw data from here — keeping the data in this leaf (rather than in ``step_names.py``)
is what breaks the otherwise-circular import between the access layer and the manager.

**Source of truth (FZ 31e1/31e1f Final Phase, 2026-06-28):** the standalone
``registry/step_names.yaml`` table was DELETED. The registry is now derived SOLELY from the
``.step.yaml`` ``registry:`` blocks + a 3-row ``_EXTRAS`` map (in ``interface_registry_loader``)
for the interface-less abstract steps (``Base`` / ``Processing`` / ``HyperparameterPrep``).
To add or edit a step, edit its ``.step.yaml`` ``registry:`` block — there is no separate table.
A golden snapshot (``tests/registry/step_names_registry_snapshot.json``) gates drift.

(Formerly ``step_names_original.py``; renamed because it is the live source, not a backup.)
"""

from typing import Dict

# --- Build the step registry from the per-step .step.yaml registry: blocks ---
#
# The lazy import keeps this module a dependency-free leaf — ``interface_registry_loader``
# imports only pathlib/typing/yaml, so no circular import is introduced.
def _build_step_names() -> Dict[str, Dict[str, str]]:
    from .interface_registry_loader import build_registry_from_interfaces

    return build_registry_from_interfaces()


STEP_NAMES: Dict[str, Dict[str, str]] = _build_step_names()

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
