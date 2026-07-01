"""Registry golden-snapshot gate (the Final-Phase successor to the step_names.yaml parity oracle).

`step_names.yaml` was deleted (FZ 31e1/31e1f Final Phase): the per-step `.step.yaml` `registry:`
blocks (+ a 3-row `_EXTRAS` map for the interface-less abstract steps) are now the SOLE source of the
STEP_NAMES table, derived by `build_registry_from_interfaces()`. The oracle used to compare the
derived table against the standalone yaml; with the yaml gone, the oracle becomes a GOLDEN SNAPSHOT —
`step_names_registry_snapshot.json` freezes the expected 48-row table so any accidental drift (a typo
in a `.step.yaml` registry block, a dropped step, a changed sagemaker_step_type) fails loudly.

Re-baseline intentionally after an authored registry change:
    CURSUS_UPDATE_REGISTRY_SNAPSHOT=1 python -m pytest tests/registry/test_registry_interface_parity.py
"""

import json
import os
from pathlib import Path

from cursus.registry.interface_registry_loader import build_registry_from_interfaces

_SNAPSHOT = Path(__file__).resolve().parent / "step_names_registry_snapshot.json"

_REQUIRED_KEYS = {
    "config_class",
    "builder_step_name",
    "spec_type",
    "sagemaker_step_type",
    "description",
}


def _load_snapshot():
    with open(_SNAPSHOT, encoding="utf-8") as f:
        return json.load(f)


def _maybe_rebaseline(derived):
    if os.environ.get("CURSUS_UPDATE_REGISTRY_SNAPSHOT") == "1":
        with open(_SNAPSHOT, "w", encoding="utf-8") as f:
            json.dump(derived, f, indent=2, sort_keys=True)
            f.write("\n")
        return True
    return False


def test_interface_registry_matches_golden_snapshot():
    """The interface-derived table (NO fallback — interfaces are the sole source) equals the frozen
    golden snapshot. This is the drift gate that replaced the step_names.yaml parity oracle."""
    derived = build_registry_from_interfaces()  # no fallback: .step.yaml + _EXTRAS only
    if _maybe_rebaseline(derived):
        return
    snapshot = _load_snapshot()
    assert set(derived) == set(snapshot), (
        "step-name key set diverged from snapshot: "
        f"only-in-derived={set(derived) - set(snapshot)}, "
        f"only-in-snapshot={set(snapshot) - set(derived)} "
        "(re-baseline with CURSUS_UPDATE_REGISTRY_SNAPSHOT=1 if intentional)"
    )
    for name, snap_row in snapshot.items():
        for key in _REQUIRED_KEYS:
            assert derived[name][key] == snap_row[key], (
                f"{name}.{key}: derived={derived[name][key]!r} != snapshot={snap_row[key]!r} "
                "(re-baseline with CURSUS_UPDATE_REGISTRY_SNAPSHOT=1 if intentional)"
            )


def test_interface_registry_is_self_sufficient_without_fallback():
    """Every step's data comes from its `.step.yaml` `registry:` block (or the 3-row `_EXTRAS`) —
    derivation needs NO external fallback table. This is what made dropping step_names.yaml safe."""
    derived = build_registry_from_interfaces()  # no fallback
    missing = [n for n, r in derived.items() if not r.get("sagemaker_step_type")]
    assert not missing, (
        f"rows missing sagemaker_step_type without a fallback: {missing}"
    )
    assert len(derived) >= 45, (
        f"expected the full registry (~48 rows), got {len(derived)}"
    )


def test_every_row_carries_sagemaker_step_type():
    """The load-bearing invariant: every row carries a non-empty sagemaker_step_type."""
    derived = build_registry_from_interfaces()
    missing = [n for n, r in derived.items() if not r.get("sagemaker_step_type")]
    assert not missing, f"rows missing sagemaker_step_type: {missing}"


def test_spec_type_equals_step_name_for_all_rows():
    """spec_type is always the canonical step name (the 31e1 redundancy finding)."""
    derived = build_registry_from_interfaces()
    mismatches = {n: r["spec_type"] for n, r in derived.items() if r["spec_type"] != n}
    assert not mismatches, f"spec_type != step_name for: {mismatches}"


def test_registry_section_sagemaker_step_type_pin_matches_live_valid_set():
    """The closed set `RegistrySection._SAGEMAKER_STEP_TYPES` (the author-time validator pin) MUST
    equal the live valid set `get_valid_sagemaker_step_types()`. The Pydantic validator rejects an
    out-of-set `registry.sagemaker_step_type` at author time (a typo can't silently mis-route); this
    conformance gate keeps the pinned allowlist from drifting from the registry's actual valid verbs.
    """
    from cursus.core.base.step_interface import RegistrySection
    from cursus.registry.step_names import get_valid_sagemaker_step_types

    assert set(RegistrySection._SAGEMAKER_STEP_TYPES) == set(
        get_valid_sagemaker_step_types()
    ), (
        "RegistrySection._SAGEMAKER_STEP_TYPES drifted from get_valid_sagemaker_step_types(): "
        f"pin-only={set(RegistrySection._SAGEMAKER_STEP_TYPES) - set(get_valid_sagemaker_step_types())}, "
        f"live-only={set(get_valid_sagemaker_step_types()) - set(RegistrySection._SAGEMAKER_STEP_TYPES)}"
    )


def test_config_class_convention_breakers_are_the_known_three():
    """Exactly three interface-backed steps break the <Name>Config convention — their real config
    class is `<Name>StepConfig`, declared via `registry.config_class` in the .step.yaml. These MUST
    be declared (not convention-derived) or CONFIG_STEP_REGISTRY maps a fictional config name and the
    real config fails to resolve to its builder (the registry-integrity bug the FZ 31e2 closure check
    caught for PyTorchModel/XGBoostModel — fixed 2026-06-29)."""
    derived = build_registry_from_interfaces()
    extras = {"Base", "Processing", "HyperparameterPrep"}  # the 3 interface-less rows
    breakers = {
        n: r["config_class"]
        for n, r in derived.items()
        if n not in extras and r["config_class"] != f"{n}Config"
    }
    assert breakers == {
        "BatchTransform": "BatchTransformStepConfig",
        "PyTorchModel": "PyTorchModelStepConfig",
        "XGBoostModel": "XGBoostModelStepConfig",
    }, f"unexpected config_class convention-breakers: {breakers}"
