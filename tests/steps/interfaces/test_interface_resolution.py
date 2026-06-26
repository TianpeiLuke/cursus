"""
Tests for step_name -> .step.yaml resolution in cursus.steps.interfaces.

Guards the hardened resolver: convention-first with a normalized-scan fallback, so a step
whose acronym/casing the hardcoded table doesn't know still resolves instead of raising
FileNotFoundError.
"""

import pytest

from cursus.steps import interfaces as ifaces


def test_all_listed_interfaces_resolve():
    names = ifaces.list_available_interfaces()
    assert len(names) >= 40
    for n in names:
        # Each discovered stem must resolve back to an existing file.
        p = ifaces._resolve_interface_path(n)
        assert p.exists()


def test_convention_path_for_known_acronym():
    # XGBoostTraining -> xgboost_training.step.yaml via the convention table.
    p = ifaces._resolve_interface_path("XGBoostTraining")
    assert p.name == "xgboost_training.step.yaml"


def test_normalized_fallback_resolves_casing_variant():
    # All-caps with no separators is NOT what the convention produces, but the canonical
    # fallback (alnum-lowercased) must still find the file.
    p = ifaces._resolve_interface_path("XGBOOSTTRAINING")
    assert p.name == "xgboost_training.step.yaml"


def test_canonical_key_collapses_case_and_separators():
    k = ifaces._canonical_key
    assert k("XGBoostTraining") == k("xgboost_training") == k("XGBoost_Training")


def test_missing_step_raises():
    with pytest.raises(FileNotFoundError):
        ifaces._resolve_interface_path("TotallyNotAStep_zzz")


def test_load_interface_uses_resolver():
    # End-to-end: load_interface returns a validated StepInterface for a real step.
    iface = ifaces.load_interface("XGBoostTraining")
    assert iface.step_type == "XGBoostTraining"
