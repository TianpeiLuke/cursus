"""
Tests for StepInterface job-type variant merging (cursus.core.base.step_interface).

Covers the deep-merge fix: a ``.step.yaml`` variant that restates only a subset of
``spec.dependencies`` / ``outputs`` (or contract ``inputs`` / ``outputs``) must override
just those ports and preserve the rest of the base set — a shallow ``{**base, **variant}``
merge dropped the omitted base ports and broke construction (e.g. RiskTableMapping).

The regression sweep loads *every* shipped interface YAML for its base and *every* declared
job_type variant and asserts construction succeeds.
"""

import glob
import os

import pytest
import yaml

from cursus.core.base.step_interface import StepInterface, _deep_merge


# --- _deep_merge unit tests ---------------------------------------------------------


class TestDeepMerge:
    def test_subset_override_preserves_other_keys(self):
        base = {"a": {"x": 1}, "b": {"y": 2}, "c": 3}
        override = {"a": {"x": 9}}
        out = _deep_merge(base, override)
        assert out == {"a": {"x": 9}, "b": {"y": 2}, "c": 3}

    def test_nested_field_override_keeps_sibling_fields(self):
        base = {"dep": {"required": True, "type": "processing_output"}}
        override = {"dep": {"required": False}}
        out = _deep_merge(base, override)
        # only 'required' changes; 'type' survives
        assert out["dep"] == {"required": False, "type": "processing_output"}

    def test_non_dict_value_replaces(self):
        assert _deep_merge({"k": [1, 2]}, {"k": [3]}) == {"k": [3]}

    def test_inputs_not_mutated(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}  # base untouched
        assert override == {"a": {"y": 2}}

    def test_new_key_added(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


# --- Variant resolution behavior ----------------------------------------------------


def _risk_table_yaml():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    path = os.path.join(
        here, "src", "cursus", "steps", "interfaces", "risk_table_mapping.step.yaml"
    )
    with open(path) as f:
        return yaml.safe_load(f)


class TestVariantMerge:
    def test_variant_subset_preserves_base_dependency(self):
        """RiskTableMapping/training omits hyperparameters_s3_uri in its variant spec;
        it must still be present after the merge (the bug dropped it)."""
        data = _risk_table_yaml()
        si = StepInterface.from_yaml(data, job_type="training")
        assert "hyperparameters_s3_uri" in si.spec.dependencies
        assert "input_data" in si.spec.dependencies
        assert "model_artifacts_input" in si.spec.dependencies

    def test_variant_override_is_applied(self):
        """The variant's input_data override (extra semantic keyword 'train') is applied."""
        data = _risk_table_yaml()
        si = StepInterface.from_yaml(data, job_type="training")
        kw = si.spec.dependencies["input_data"].semantic_keywords or []
        assert "train" in kw

    def test_base_load_without_job_type(self):
        data = _risk_table_yaml()
        si = StepInterface.from_yaml(data)
        assert "hyperparameters_s3_uri" in si.spec.dependencies


# --- Regression sweep: every interface x every declared variant must construct ------


def _interface_files():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    pattern = os.path.join(here, "src", "cursus", "steps", "interfaces", "*.step.yaml")
    return sorted(glob.glob(pattern))


def _load_cases():
    """Yield (file_basename, job_type) for the base and each declared variant."""
    cases = []
    for f in _interface_files():
        with open(f) as fh:
            data = yaml.safe_load(fh) or {}
        name = os.path.basename(f)
        cases.append((name, None))  # base
        for jt in (data.get("variants") or {}).keys():
            cases.append((name, jt))
    return cases


_CASES = _load_cases()


def test_sweep_is_nonempty():
    # Guard: ensure the sweep actually found interfaces + variants (not silently empty).
    assert len(_interface_files()) >= 40
    assert sum(1 for _, jt in _CASES if jt is not None) >= 1


@pytest.mark.parametrize(
    "filename,job_type",
    _CASES,
    ids=[f"{n}:{jt or 'base'}" for n, jt in _CASES],
)
def test_every_interface_and_variant_constructs(filename, job_type):
    """Every shipped .step.yaml must construct for its base and each declared variant.

    This is the guard the brittleness review asked for: it would have caught the
    RiskTableMapping/BatchTransform shallow-merge ValidationError before release.
    """
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    path = os.path.join(here, "src", "cursus", "steps", "interfaces", filename)
    with open(path) as f:
        data = yaml.safe_load(f)
    si = StepInterface.from_yaml(data, job_type=job_type)
    assert si.step_type  # constructed and validated (alignment invariant held)


# --- registry.sagemaker_step_type closed-enum validation --------------------------


class TestSagemakerStepTypeValidation:
    """`registry.sagemaker_step_type` is the routing key (selects the PatternHandler). It is a
    CLOSED set, now enforced by RegistrySection's Pydantic validator at author time, so a typo or
    wrong value is caught by StepInterface.from_yaml / validate.step_interface / CI — not silently
    mis-routed at build."""

    from cursus.core.base.step_interface import RegistrySection

    def _make(self, sagemaker_step_type):
        return StepInterface.from_yaml(
            {
                "step_type": "X",
                "node_type": "internal",
                "registry": {"sagemaker_step_type": sagemaker_step_type},
                "contract": {
                    "entry_point": "x.py",
                    "inputs": {"d": {"path": "/opt/ml/processing/input"}},
                },
                "spec": {"dependencies": {"d": {"type": "processing_output"}}},
            }
        )

    @pytest.mark.parametrize(
        "bad", ["Procesing", "training", "Train", "Foo", "processing"]
    )
    def test_invalid_value_rejected(self, bad):
        with pytest.raises(Exception):
            self._make(bad)

    @pytest.mark.parametrize("good", list(RegistrySection._SAGEMAKER_STEP_TYPES))
    def test_every_valid_verb_accepted(self, good):
        si = self._make(good)
        assert si.registry.sagemaker_step_type == good

    def test_none_is_allowed(self):
        from cursus.core.base.step_interface import RegistrySection

        assert RegistrySection().sagemaker_step_type is None
        assert RegistrySection(sagemaker_step_type=None).sagemaker_step_type is None
