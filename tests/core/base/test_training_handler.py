"""
Phase S1 handler gate: TrainingHandler — channel-parse get_inputs, single-str get_outputs, and
build_step's outputs-before-compute ordering + empty-inputs guard + standard caching.

Uses a lightweight fake builder (spec/contract/config) — the get_inputs/get_outputs logic depends
only on spec×contract, and the estimator factory + TrainingStep are faked, so no real config /
SageMaker session is needed.
"""

from types import SimpleNamespace

import pytest

from cursus.core.base.builder_templates import TrainingHandler, resolve_handler


def _stable(v):
    return repr(v.expr) if hasattr(v, "expr") else str(v)


def _channel_s3(ti):
    """Pull the S3Uri out of a TrainingInput (nested in its .config)."""
    return _stable(ti.config["DataSource"]["S3DataSource"]["S3Uri"])


def _dep(logical_name, required=True):
    return SimpleNamespace(logical_name=logical_name, required=required)


def _fake_builder(deps, input_paths, outputs_spec=("model_output",), skip_hp=True):
    spec = SimpleNamespace(
        step_type="XGBoostTraining",
        dependencies={d.logical_name: d for d in deps},
        outputs={o: SimpleNamespace(logical_name=o) for o in outputs_spec},
    )
    contract = SimpleNamespace(expected_input_paths=input_paths)
    return SimpleNamespace(
        spec=spec,
        contract=contract,
        config=SimpleNamespace(skip_hyperparameters_s3_uri=skip_hp),
        _get_base_output_path=lambda: "s3://bucket/base",
        _get_step_name=lambda: "XGBoostTraining",
        _get_cache_config=lambda enable: SimpleNamespace(enable_caching=enable),
        extract_inputs_from_dependencies=lambda d: {},
        log_info=lambda *a, **k: None,
        log_warning=lambda *a, **k: None,
    )


@pytest.fixture
def handler():
    return TrainingHandler(knobs={"output_path_token": "xgboost_training"})


def test_routes_to_training_handler():
    assert isinstance(resolve_handler("Training"), TrainingHandler)


def test_input_path_fans_out_to_train_val_test(handler):
    b = _fake_builder(
        deps=[_dep("input_path")],
        input_paths={"input_path": "/opt/ml/input/data"},
    )
    chans = handler.get_inputs(b, {"input_path": "s3://data/base"})
    assert set(chans) == {"train", "val", "test"}
    # each channel's s3_data is a Join ending in the split subdir
    assert "train" in _channel_s3(chans["train"])
    assert "val" in _channel_s3(chans["val"])


def test_channel_name_parsed_from_contract_path(handler):
    b = _fake_builder(
        deps=[_dep("hyperparameters_s3_uri", required=False)],
        input_paths={"hyperparameters_s3_uri": "/opt/ml/input/data/config"},
        skip_hp=False,  # don't skip, so it's processed
    )
    chans = handler.get_inputs(b, {"hyperparameters_s3_uri": "s3://hp"})
    # channel name comes from parts[5] == "config"
    assert "config" in chans


def test_skip_hyperparameters_when_configured(handler):
    b = _fake_builder(
        deps=[_dep("hyperparameters_s3_uri", required=False), _dep("input_path")],
        input_paths={"hyperparameters_s3_uri": "/opt/ml/input/data/config", "input_path": "/opt/ml/input/data"},
        skip_hp=True,
    )
    chans = handler.get_inputs(b, {"input_path": "s3://d", "hyperparameters_s3_uri": "s3://hp"})
    # hyperparameters skipped; only train/val/test from input_path
    assert set(chans) == {"train", "val", "test"}


def test_get_inputs_required_missing_raises(handler):
    b = _fake_builder(deps=[_dep("input_path")], input_paths={"input_path": "/opt/ml/input/data"})
    with pytest.raises(ValueError, match="input_path"):
        handler.get_inputs(b, {})


def test_get_outputs_single_join(handler):
    b = _fake_builder(deps=[], input_paths={})
    out = handler.get_outputs(b, {})
    # single value (not a list); a Join over [base, token]
    assert not isinstance(out, list)
    assert "xgboost_training" in _stable(out)


def test_get_outputs_explicit_wins(handler):
    b = _fake_builder(deps=[], input_paths={}, outputs_spec=("model_output",))
    assert handler.get_outputs(b, {"model_output": "s3://explicit"}) == "s3://explicit"


def test_build_step_ordering_outputs_before_compute():
    """make_compute (estimator factory) receives output_path -> runs AFTER get_outputs."""
    call_order = []

    def fake_make_compute(b, output_path):
        call_order.append(("make_compute", output_path))
        return "ESTIMATOR"

    recorded = {}

    class FakeTrainingStep:
        def __init__(self, **kw):
            call_order.append(("TrainingStep", None))
            recorded.update(kw)
            self.name = kw.get("name")

    b = _fake_builder(deps=[_dep("input_path")], input_paths={"input_path": "/opt/ml/input/data"})
    import sagemaker.workflow.steps as sws
    orig = sws.TrainingStep
    sws.TrainingStep = FakeTrainingStep
    try:
        h = TrainingHandler(knobs={"output_path_token": "xgboost_training", "make_compute": fake_make_compute})
        h.build_step(b, inputs={"input_path": "s3://d"}, dependencies=["dep"], enable_caching=True)
    finally:
        sws.TrainingStep = orig

    # make_compute got the resolved output_path and ran before TrainingStep
    assert call_order[0][0] == "make_compute"
    assert call_order[0][1] is not None
    assert call_order[1][0] == "TrainingStep"
    # standard caching: cache_config is present (NOT dropped)
    assert recorded["cache_config"].enable_caching is True
    assert recorded["estimator"] == "ESTIMATOR"
    assert recorded["depends_on"] == ["dep"]
    # channels passed as the inputs dict
    assert set(recorded["inputs"]) == {"train", "val", "test"}


def test_build_step_empty_inputs_guard():
    b = _fake_builder(deps=[], input_paths={})  # no deps -> empty channels
    h = TrainingHandler(knobs={"make_compute": lambda b, op: "E"})
    with pytest.raises(ValueError, match="No training inputs"):
        h.build_step(b, inputs={}, dependencies=[])


# --- true parity against the real XGBoostTrainingStepBuilder ---
# (the estimator factory / TrainingStep aren't exercised here, so the real builder constructs
# without a SageMaker session; we compare the session-independent get_inputs/get_outputs.)


@pytest.fixture
def real_builder(tmp_path):
    (tmp_path / "train_xgb.py").write_text("# stub\n")
    from cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingConfig
    from cursus.step_catalog.step_catalog import StepCatalog

    XGBoostTrainingStepBuilder = StepCatalog().load_builder_class("XGBoostTraining")

    cfg = XGBoostTrainingConfig(
        author="t", bucket="b", role="arn:aws:iam::123456789012:role/x", region="NA",
        service_name="s", pipeline_version="1.0.0", project_root_folder="p",
        training_entry_point="train_xgb.py", source_dir=str(tmp_path),
    )
    return XGBoostTrainingStepBuilder(config=cfg)


def _real_handler():
    return TrainingHandler(
        knobs={"output_path_token": "xgboost_training", "direct_input_keys": ["input_path"]}
    )


def test_parity_get_inputs_channels_match_real(real_builder):
    inp = {"input_path": "s3://data/base"}
    real_in = real_builder._get_inputs(dict(inp))
    hnd_in = _real_handler().get_inputs(real_builder, dict(inp))
    assert sorted(real_in) == sorted(hnd_in) == ["test", "train", "val"]
    for k in real_in:
        assert _channel_s3(real_in[k]) == _channel_s3(hnd_in[k])


def test_parity_get_outputs_matches_real(real_builder):
    real_out = real_builder._get_outputs({})
    hnd_out = _real_handler().get_outputs(real_builder, {})
    assert _stable(real_out) == _stable(hnd_out)
