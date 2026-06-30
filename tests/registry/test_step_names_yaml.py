"""
Tests for the interface-derived step-name registry (``step_names_base.STEP_NAMES``).

As of the FZ 31e1/31e1f Final Phase the standalone ``step_names.yaml`` was DELETED — the
registry is built from the per-step ``.step.yaml`` ``registry:`` blocks (+ a 3-row ``_EXTRAS``
map) by ``build_registry_from_interfaces()``. These tests guard that the built ``STEP_NAMES``
table stays well-formed and the derived mappings stay consistent. (The drift/golden-snapshot
gate lives in ``test_registry_interface_parity.py``.)
"""

import pytest

from cursus.registry import step_names_base as sno

_REQUIRED_FIELDS = {
    "config_class",
    "builder_step_name",
    "spec_type",
    "sagemaker_step_type",
    "description",
}


def test_step_names_loaded_non_empty():
    assert isinstance(sno.STEP_NAMES, dict)
    assert (
        len(sno.STEP_NAMES) >= 40
    )  # 48 at migration time; never expect it to shrink drastically


def test_every_entry_has_all_required_string_fields():
    for name, info in sno.STEP_NAMES.items():
        assert set(info) == _REQUIRED_FIELDS, (
            f"{name} has unexpected field set: {set(info)}"
        )
        for field, value in info.items():
            assert isinstance(value, str) and value, (
                f"{name}.{field} must be a non-empty string"
            )


def test_derived_mappings_consistent_with_step_names():
    # CONFIG_STEP_REGISTRY: config_class -> step_name
    assert sno.CONFIG_STEP_REGISTRY == {
        info["config_class"]: name for name, info in sno.STEP_NAMES.items()
    }
    # BUILDER_STEP_NAMES: step_name -> builder_step_name
    assert sno.BUILDER_STEP_NAMES == {
        name: info["builder_step_name"] for name, info in sno.STEP_NAMES.items()
    }
    # SPEC_STEP_TYPES: step_name -> spec_type
    assert sno.SPEC_STEP_TYPES == {
        name: info["spec_type"] for name, info in sno.STEP_NAMES.items()
    }


def test_config_classes_are_unique():
    # CONFIG_STEP_REGISTRY inverts config_class -> step_name, so duplicate config_class
    # names would silently collide. Assert there are none.
    config_classes = [info["config_class"] for info in sno.STEP_NAMES.values()]
    assert len(config_classes) == len(set(config_classes)), (
        "duplicate config_class values"
    )


def test_step_names_yaml_is_gone():
    """The standalone table file was deleted — the .step.yaml registry: blocks are the sole source."""
    from pathlib import Path

    yaml_path = Path(sno.__file__).resolve().parent / "step_names.yaml"
    assert not yaml_path.exists(), (
        "step_names.yaml was removed (FZ 31e1f Final Phase); the registry derives from the "
        ".step.yaml registry: blocks. If it reappeared, the source-of-truth split has regressed."
    )


def test_loader_raises_clearly_when_a_step_lacks_sagemaker_step_type():
    """With no fallback table, a .step.yaml missing its registry.sagemaker_step_type must fail
    LOUDLY at build (not silently drop the step)."""
    from cursus.registry.interface_registry_loader import build_registry_from_interfaces

    # Pass an explicit empty fallback against a temp dir containing one block-less interface.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "foo.step.yaml").write_text("step_type: Foo\ncontract:\n  entry_point: f.py\n")
        with pytest.raises(ValueError, match="sagemaker_step_type"):
            build_registry_from_interfaces(interfaces_dir=Path(d), fallback=None)


def test_known_steps_present():
    # Spot-check a few canonical steps survived with correct mappings.
    assert sno.STEP_NAMES["XGBoostTraining"]["config_class"] == "XGBoostTrainingConfig"
    assert (
        sno.STEP_NAMES["Registration"]["sagemaker_step_type"]
        == "MimsModelRegistrationProcessing"
    )
    assert sno.STEP_NAMES["Base"]["spec_type"] == "Base"
