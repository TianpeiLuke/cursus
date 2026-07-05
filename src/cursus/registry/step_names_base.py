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


def _rebuild_derived() -> None:
    """Rebuild the derived mappings from the current ``STEP_NAMES``, in place.

    ``STEP_NAMES`` is mutated in place (never reassigned) so that modules which did
    ``from ...step_names_base import STEP_NAMES`` keep pointing at the live dict; the three
    derived globals ARE reassigned here, but callers that need workspace-aware values read them
    through ``step_names.get_*`` accessors, not these snapshots.
    """
    global CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES, SPEC_STEP_TYPES
    CONFIG_STEP_REGISTRY = {
        info["config_class"]: step_name for step_name, info in STEP_NAMES.items()
    }
    BUILDER_STEP_NAMES = {
        step_name: info["builder_step_name"] for step_name, info in STEP_NAMES.items()
    }
    SPEC_STEP_TYPES = {
        step_name: info["spec_type"] for step_name, info in STEP_NAMES.items()
    }


# Generate the mappings that existing code expects.
CONFIG_STEP_REGISTRY: Dict[str, str] = {}
BUILDER_STEP_NAMES: Dict[str, str] = {}
SPEC_STEP_TYPES: Dict[str, str] = {}
_rebuild_derived()


def merge_pack_registry(pack_rows: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Merge an external step-pack's registry rows ON TOP of the package registry, in place.

    ADD-ONLY overlay: package steps are the floor and are never removed. ``STEP_NAMES`` is
    mutated in place (so import-time references stay live) with ``pack_rows`` layered on top,
    then the derived globals are rebuilt. This is the LOW-LEVEL merge — the access layer
    (:func:`cursus.registry.step_names.refresh_registry`) calls this AND then re-syncs the
    hybrid manager so the catalog sees the plugin steps.

    Args:
        pack_rows: ``{canonical_name: {config_class, builder_step_name, spec_type,
            sagemaker_step_type, description}}`` derived from the pack's ``.step.yaml`` files.

    Returns:
        ``{name: "collision"}`` for any pack name that shadowed an EXISTING package step —
        the caller decides whether to warn. Empty when every pack row is genuinely new.
    """
    collisions = {name: "collision" for name in pack_rows if name in STEP_NAMES}
    STEP_NAMES.update(
        pack_rows
    )  # in place — package rows preserved, pack rows added on top
    _rebuild_derived()
    return collisions
