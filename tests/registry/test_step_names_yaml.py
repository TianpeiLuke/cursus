"""
Tests for the YAML-backed step-name registry (step_names_base loads step_names.yaml).

Guards the migration that moved the STEP_NAMES table out of Python and into
step_names.yaml: the loaded data must stay well-formed, the derived mappings must stay
consistent, and the YAML must ship with the package.
"""

from pathlib import Path

import pytest
import yaml

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


def test_yaml_file_exists_and_parses():
    yaml_path = Path(sno.__file__).resolve().parent / "step_names.yaml"
    assert yaml_path.exists(), "step_names.yaml must sit next to step_names_base.py"
    data = yaml.safe_load(yaml_path.read_text())
    assert "step_names" in data
    # The module's loaded dict must equal what the YAML parses to.
    assert sno.STEP_NAMES == data["step_names"]


def test_loader_rejects_missing_field(tmp_path, monkeypatch):
    # A malformed YAML (entry missing a required field) must raise a clear ValueError.
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "step_names:\n  Foo:\n    config_class: FooConfig\n"
    )  # missing 4 fields
    monkeypatch.setattr(sno, "_STEP_NAMES_YAML", bad)
    with pytest.raises(ValueError, match="missing required field"):
        sno._load_step_names()


def test_known_steps_present():
    # Spot-check a few canonical steps survived the migration with correct mappings.
    assert sno.STEP_NAMES["XGBoostTraining"]["config_class"] == "XGBoostTrainingConfig"
    assert (
        sno.STEP_NAMES["Registration"]["sagemaker_step_type"]
        == "MimsModelRegistrationProcessing"
    )
    assert sno.STEP_NAMES["Base"]["spec_type"] == "Base"
