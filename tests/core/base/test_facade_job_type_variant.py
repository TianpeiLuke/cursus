"""
Regression: the TemplateStepBuilder facade must pass config.job_type into load_step_interface so a
variant-bearing step resolves its job-typed spec — NOT the base spec.

7 of the 45 steps declare job_type variants (risk_table_mapping, missing_value_imputation,
feature_selection, batch_transform, cradle_data_loading, temporal_*). A variant changes load-bearing
connection fields — e.g. RiskTableMapping's `model_artifacts_input.required` flips False (base) ->
True (validation/testing), and its `compatible_sources` narrows to [RiskTableMapping_Training]. The
legacy builder loads the variant (builder_risk_table_mapping_step.py:60-61, job_type=config.job_type);
the facade must reproduce that exactly. Before the fix the facade called load_step_interface(STEP_NAME)
with no job_type and silently got the base spec — turning a required cross-step dependency optional.
"""

import os
import tempfile

import pytest

from cursus.core.base.builder_templates import TemplateStepBuilder
from cursus.steps.interfaces import load_step_interface


class RiskTableMappingShell(TemplateStepBuilder):
    STEP_NAME = "RiskTableMapping"


@pytest.fixture
def cfg_factory():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "risk_table_mapping.py"), "w").write("# stub\n")
    from cursus.steps.configs.config_risk_table_mapping_step import RiskTableMappingConfig

    def make(job_type):
        return RiskTableMappingConfig(
            author="t", bucket="b", role="arn:aws:iam::123456789012:role/x", region="NA",
            service_name="s", pipeline_version="1.0.0", project_root_folder="p",
            processing_source_dir=d, processing_entry_point="risk_table_mapping.py",
            job_type=job_type,
        )

    return make


def _mai_required(spec):
    return spec.dependencies["model_artifacts_input"].required


def _mai_sources(spec):
    return list(spec.dependencies["model_artifacts_input"].compatible_sources)


def test_facade_resolves_variant_spec_matching_direct_loader(cfg_factory):
    """Facade spec == direct loader spec for the SAME job_type (the legacy builder's behavior)."""
    for jt in ("training", "validation", "testing"):
        facade_spec = RiskTableMappingShell(config=cfg_factory(jt)).spec
        _c, loader_spec = load_step_interface("RiskTableMapping", job_type=jt)
        assert _mai_required(facade_spec) == _mai_required(loader_spec), jt
        assert _mai_sources(facade_spec) == _mai_sources(loader_spec), jt


def test_variant_flips_required_flag_vs_base(cfg_factory):
    """The validation variant makes model_artifacts_input REQUIRED; base leaves it optional.

    This is the connection field the old (job_type-less) facade silently dropped.
    """
    _c, base = load_step_interface("RiskTableMapping")  # no job_type -> base spec
    assert _mai_required(base) is False

    validation_spec = RiskTableMappingShell(config=cfg_factory("validation")).spec
    assert _mai_required(validation_spec) is True
    # and the producer-identity list narrows to the training step only
    assert _mai_sources(validation_spec) == ["RiskTableMapping_Training"]


def test_facade_without_job_type_field_still_loads_base():
    """A non-variant shell whose config lacks job_type still loads (job_type=None -> base spec)."""
    import tempfile as _t

    d = _t.mkdtemp()
    open(os.path.join(d, "tabular_preprocess.py"), "w").write("# stub\n")
    from cursus.steps.configs.config_tabular_preprocessing_step import (
        TabularPreprocessingConfig,
    )

    class TabularShell(TemplateStepBuilder):
        STEP_NAME = "TabularPreprocessing"

    cfg = TabularPreprocessingConfig(
        author="t", bucket="b", role="arn:aws:iam::123456789012:role/x", region="NA",
        service_name="s", pipeline_version="1.0.0", project_root_folder="p",
        processing_source_dir=d, processing_entry_point="tabular_preprocess.py",
        label_name="label", job_type="training",
    )
    b = TabularShell(config=cfg)
    assert b.spec is not None
    assert b.spec.step_type == "TabularPreprocessing"
