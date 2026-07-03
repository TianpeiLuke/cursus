"""
Phase S1 handler gate: ModelCreationHandler parity with the hand-written XGBoostModelStepBuilder.

Proves the re-homed handler matches the builder for the session-independent constituents:
single-key model_data passthrough get_inputs, get_outputs->None, and build_step's caching-DROP
(warn, no cache_config), model= direct, make_compute-LAST orchestration. The model factory +
CreateModelStep are faked (a real Model/image retrieve needs a session).
"""

import pytest


@pytest.fixture
def cfg(tmp_path):
    from cursus.steps.configs.config_xgboost_model_step import XGBoostModelStepConfig

    (tmp_path / "inference.py").write_text("# stub\n")
    return XGBoostModelStepConfig(
        author="t", bucket="b", role="arn:aws:iam::123456789012:role/x", region="NA",
        service_name="s", pipeline_version="1.0.0", project_root_folder="p",
        source_dir=str(tmp_path), entry_point="inference.py",
    )


@pytest.fixture
def real(cfg):
    from cursus.step_catalog.step_catalog import StepCatalog

    XGBoostModelStepBuilder = StepCatalog().load_builder_class("XGBoostModel")

    return XGBoostModelStepBuilder(config=cfg)


@pytest.fixture
def handler():
    from cursus.core.base.builder_templates import ModelCreationHandler

    return ModelCreationHandler(knobs={})


def test_routes_to_model_creation_handler():
    from cursus.core.base.builder_templates import ModelCreationHandler, resolve_handler

    assert isinstance(resolve_handler("CreateModel"), ModelCreationHandler)


def test_get_inputs_passthrough_matches(real, handler):
    sample = {"model_data": "s3://artifacts/model.tar.gz"}
    assert real._get_inputs(dict(sample)) == handler.get_inputs(real, dict(sample)) == sample


def test_get_inputs_missing_model_data_both_raise(real, handler):
    with pytest.raises(ValueError, match="model_data"):
        real._get_inputs({})
    with pytest.raises(ValueError, match="model_data"):
        handler.get_inputs(real, {})


def test_get_outputs_is_none_both(real, handler):
    assert real._get_outputs({}) is None
    assert handler.get_outputs(real, {}) is None


def test_build_step_drops_caching_and_runs_make_compute_last(real, handler):
    from cursus.core.base.builder_templates import ModelCreationHandler

    captured = {}

    def fake_make_compute(b, model_data):
        captured["model_data"] = model_data
        return "FAKE_MODEL"

    recorded = {}
    warned = []

    class FakeCreateModelStep:
        def __init__(self, **kw):
            recorded.update(kw)
            self.name = kw.get("name")

    import sagemaker.workflow.steps as sws
    orig = sws.CreateModelStep
    sws.CreateModelStep = FakeCreateModelStep
    orig_warn = real.log_warning
    real.log_warning = lambda *a, **k: warned.append(a)
    try:
        h = ModelCreationHandler(knobs={"make_compute": fake_make_compute})
        h.build_step(
            real,
            inputs={"model_data": "s3://m"},
            dependencies=["dep1"],
            enable_caching=True,
        )
    finally:
        sws.CreateModelStep = orig
        real.log_warning = orig_warn

    # make_compute received the resolved model_data -> ran after get_inputs (last)
    assert captured["model_data"] == "s3://m"
    # model passed DIRECTLY (not step_args); NO cache_config kwarg
    assert recorded["model"] == "FAKE_MODEL"
    assert "cache_config" not in recorded
    assert recorded["depends_on"] == ["dep1"]
    # caching-drop warning fired
    assert any("does not support caching" in str(a) for a in warned)


def test_build_step_direct_model_data_override(real, handler):
    """A direct model_data= kwarg overrides the inputs dict (backward compat)."""
    from cursus.core.base.builder_templates import ModelCreationHandler

    captured = {}

    class FakeCreateModelStep:
        def __init__(self, **kw):
            self.name = kw.get("name")

    import sagemaker.workflow.steps as sws
    orig = sws.CreateModelStep
    sws.CreateModelStep = FakeCreateModelStep
    try:
        h = ModelCreationHandler(
            knobs={"make_compute": lambda b, md: captured.setdefault("md", md)}
        )
        h.build_step(
            real,
            inputs={"model_data": "s3://from-inputs"},
            model_data="s3://direct-override",
            dependencies=[],
            enable_caching=False,
        )
    finally:
        sws.CreateModelStep = orig

    assert captured["md"] == "s3://direct-override"
