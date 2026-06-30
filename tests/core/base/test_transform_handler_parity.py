"""
Phase S1 handler gate: TransformHandler parity with the hand-written BatchTransformStepBuilder.

Proves the re-homed TransformHandler.get_inputs/get_outputs produce identical results to the
builder's _get_inputs/_get_outputs, and that build_step's ordering (make_compute LAST, consuming
model_name + output_path) + singular-TransformInput + guarded-caching are faithful — all
session-independent (no real SageMaker session needed).
"""

import pytest

from cursus.core.base.builder_templates import TransformHandler, resolve_handler


def _stable(v):
    return repr(v.expr) if hasattr(v, "expr") else str(v)


@pytest.fixture
def cfg():
    from cursus.steps.configs.config_batch_transform_step import BatchTransformStepConfig

    return BatchTransformStepConfig(
        author="t", bucket="b", role="arn:aws:iam::123456789012:role/x", region="NA",
        service_name="s", pipeline_version="1.0.0", project_root_folder="p", job_type="testing",
    )


@pytest.fixture
def real(cfg):
    from cursus.steps.builders import BatchTransformStepBuilder

    return BatchTransformStepBuilder(config=cfg)


@pytest.fixture
def handler():
    return TransformHandler(knobs={})


def test_routes_to_transform_handler():
    assert isinstance(resolve_handler("Transform"), TransformHandler)


def test_get_inputs_tuple_matches(real, handler):
    sample = {"model_name": "my-model", "processed_data": "s3://src/data"}
    r_ti, r_mn = real._get_inputs(dict(sample))
    h_ti, h_mn = handler.get_inputs(real, dict(sample))
    assert r_mn == h_mn == "my-model"
    # TransformInput data + filters identical
    assert _stable(r_ti.data) == _stable(h_ti.data)
    assert r_ti.content_type == h_ti.content_type
    assert r_ti.split_type == h_ti.split_type
    assert r_ti.join_source == h_ti.join_source


def test_get_inputs_missing_model_name_both_raise(real, handler):
    only_data = {"processed_data": "s3://src/data"}
    with pytest.raises(ValueError, match="model_name"):
        real._get_inputs(dict(only_data))
    with pytest.raises(ValueError, match="model_name"):
        handler.get_inputs(real, dict(only_data))


def test_get_inputs_missing_processed_data_both_raise(real, handler):
    only_model = {"model_name": "my-model"}
    with pytest.raises(ValueError, match="processed_data"):
        real._get_inputs(dict(only_model))
    with pytest.raises(ValueError, match="processed_data"):
        handler.get_inputs(real, dict(only_model))


def test_get_outputs_generated_matches(real, handler):
    assert _stable(real._get_outputs({})) == _stable(handler.get_outputs(real, {}))


def test_get_outputs_explicit_matches(real, handler):
    # the explicit branch keys off the spec's output logical name; use the real builder's
    # spec to find a valid key, then assert both return the explicit value.
    out_logical = next(iter(real.spec.outputs.values())).logical_name
    explicit = {out_logical: "s3://explicit/out"}
    assert real._get_outputs(dict(explicit)) == handler.get_outputs(real, dict(explicit)) == "s3://explicit/out"


def test_build_step_make_compute_runs_last(real, handler):
    """make_compute receives the resolved (model_name, output_path) — proving it runs after
    get_inputs + get_outputs."""
    captured = {}

    def fake_make_compute(b, model_name, output_path):
        captured["model_name"] = model_name
        captured["output_path"] = output_path
        return object()  # a fake transformer

    recorded = {}

    class FakeTransformStep:
        def __init__(self, **kw):
            recorded.update(kw)

    # patch TransformStep where build_step imports it (sagemaker.workflow.steps)
    import sagemaker.workflow.steps as sws
    orig = sws.TransformStep
    sws.TransformStep = FakeTransformStep
    try:
        h = TransformHandler(knobs={"make_compute": fake_make_compute})
        h.build_step(
            real,
            inputs={"model_name": "M", "processed_data": "s3://d"},
            outputs={},
            dependencies=[],
            enable_caching=False,
        )
    finally:
        sws.TransformStep = orig

    assert captured["model_name"] == "M"
    assert captured["output_path"] is not None
    # singular TransformInput (not a list), depends_on defaulted, caching dropped (False)
    from sagemaker.inputs import TransformInput

    assert isinstance(recorded["inputs"], TransformInput)
    assert recorded["depends_on"] == []
    assert recorded["cache_config"] is None
