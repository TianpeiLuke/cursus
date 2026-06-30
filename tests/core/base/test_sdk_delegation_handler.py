"""
Phase S1 handler gate: SDKDelegationHandler — the 3 input modes + caching/depends_on behaviors.

The 4 real MODS-predefined builders (Cradle/Redshift/DataUploading/Registration) import the SAIS
SDK at module level and cannot load in this env, so this test injects the SDK step class as a
recording fake (the `sdk_step_class` knob) and a lightweight fake builder — exercising the exact
construction logic re-homed from those builders without a SAIS SDK or a SageMaker session.
"""

from types import SimpleNamespace

import pytest

from cursus.core.base.builder_templates import SDKDelegationHandler, resolve_handler


def _fake_builder(config=None, contract=None, spec="SPEC"):
    captured = {"depends_on_calls": []}

    def add_depends_on(deps):
        captured["depends_on_calls"].append(deps)

    b = SimpleNamespace(
        config=config or SimpleNamespace(region="eu-west-1", input_s3_location=None),
        contract=contract,
        spec=spec,
        role="arn:aws:iam::123456789012:role/x",
        session=object(),
        log_info=lambda *a, **k: None,
        log_warning=lambda *a, **k: None,
        _get_step_name=lambda: "MyStep",
        extract_inputs_from_dependencies=lambda deps: {},
        _captured=captured,
    )
    return b


def _recording_sdk_class(captured):
    class FakeSDKStep:
        def __init__(self, **kw):
            captured["ctor"] = kw
            self.name = kw.get("step_name")
            self.cache_config = SimpleNamespace(enable_caching=True)

        def add_depends_on(self, deps):
            captured["add_depends_on"] = deps

    return FakeSDKStep


def test_routes_to_sdk_delegation_handler():
    for t in ("CradleDataLoading", "RedshiftDataLoading", "MimsModelRegistrationProcessing"):
        assert isinstance(resolve_handler(t), SDKDelegationHandler)
    assert isinstance(resolve_handler("Processing", "delegation"), SDKDelegationHandler)


# --- input_mode = none (Cradle/Redshift) ---

def test_input_mode_none_returns_empty():
    h = SDKDelegationHandler(knobs={"input_mode": "none"})
    assert h.get_inputs(_fake_builder(), {}) == ([], None)


def test_cradle_force_off_caching_and_add_depends_on():
    cap = {}
    b = _fake_builder()
    h = SDKDelegationHandler(knobs={
        "sdk_step_class": _recording_sdk_class(cap),
        "input_mode": "none", "caching_mode": "force_off_attr",
    })
    step = h.build_step(b, inputs={}, dependencies=["d1"], enable_caching=False)
    # ctor got no input_s3_location / processing_input / depends_on
    assert set(cap["ctor"]) == {"step_name", "role", "sagemaker_session"}
    # deps wired via add_depends_on (not ctor)
    assert cap["add_depends_on"] == ["d1"]
    # caching force-off applied (enable_caching=False)
    assert step.cache_config.enable_caching is False
    # spec attached
    assert step._spec == "SPEC"


# --- input_mode = resolve_s3 (DataUploading) ---

def test_resolve_s3_from_inputs():
    h = SDKDelegationHandler(knobs={"input_mode": "resolve_s3"})
    pis, resolved = h.get_inputs(_fake_builder(), {"input_data": "s3://up/data"})
    assert pis == [] and resolved == "s3://up/data"


def test_resolve_s3_config_fallback():
    b = _fake_builder(config=SimpleNamespace(region="x", input_s3_location="s3://cfg/loc"))
    h = SDKDelegationHandler(knobs={"input_mode": "resolve_s3"})
    pis, resolved = h.get_inputs(b, {})
    assert resolved == "s3://cfg/loc"


def test_resolve_s3_missing_raises():
    b = _fake_builder(config=SimpleNamespace(region="x", input_s3_location=None))
    h = SDKDelegationHandler(knobs={"input_mode": "resolve_s3"})
    with pytest.raises(ValueError, match="input_data"):
        h.get_inputs(b, {})


def test_resolve_s3_passes_input_s3_location_to_ctor():
    cap = {}
    h = SDKDelegationHandler(knobs={
        "sdk_step_class": _recording_sdk_class(cap), "input_mode": "resolve_s3",
    })
    h.build_step(_fake_builder(), inputs={"input_data": "s3://up/x"}, dependencies=[])
    assert cap["ctor"]["input_s3_location"] == "s3://up/x"


# --- input_mode = mims_ordered (Registration) ---

def test_mims_ordered_packaged_model_first_required():
    contract = SimpleNamespace(expected_input_paths={
        "PackagedModel": "/opt/ml/processing/input/model",
        "GeneratedPayloadSamples": "/opt/ml/processing/mims_payload",
    })
    h = SDKDelegationHandler(knobs={"input_mode": "mims_ordered"})
    b = _fake_builder(contract=contract)
    # missing PackagedModel -> raise
    with pytest.raises(ValueError, match="PackagedModel"):
        h.get_inputs(b, {})
    # only PackagedModel -> 1 input
    pis, _ = h.get_inputs(b, {"PackagedModel": "s3://m.tar.gz"})
    assert len(pis) == 1 and pis[0].input_name == "PackagedModel"
    assert pis[0].s3_data_distribution_type == "FullyReplicated"
    assert pis[0].s3_input_mode == "File"
    # both -> ordered: PackagedModel then GeneratedPayloadSamples
    pis2, _ = h.get_inputs(b, {"PackagedModel": "s3://m", "GeneratedPayloadSamples": "s3://p"})
    assert [p.input_name for p in pis2] == ["PackagedModel", "GeneratedPayloadSamples"]


def test_registration_ctor_region_suffix_and_depends_on_ctor():
    cap = {}
    contract = SimpleNamespace(expected_input_paths={})
    b = _fake_builder(config=SimpleNamespace(region="eu-west-1"), contract=contract)
    h = SDKDelegationHandler(knobs={
        "sdk_step_class": _recording_sdk_class(cap), "input_mode": "mims_ordered",
        "depends_on_ctor": True, "append_region": True, "pass_performance_metadata": True,
        "outputs_return_none": True,
    })
    h.build_step(
        b, inputs={"PackagedModel": "s3://m"}, dependencies=["dep"],
        performance_metadata_location="s3://perf",
    )
    # region-suffixed step name (NOT hardcoded us-east-1)
    assert cap["ctor"]["step_name"] == "MyStep-eu-west-1"
    # ordered processing_input passed; depends_on via ctor; perf metadata passed
    assert cap["ctor"]["processing_input"][0].input_name == "PackagedModel"
    assert cap["ctor"]["depends_on"] == ["dep"]
    assert cap["ctor"]["performance_metadata_location"] == "s3://perf"
    # deps NOT also wired via add_depends_on (ctor mode)
    assert "add_depends_on" not in cap


def test_get_outputs_modes():
    h_none = SDKDelegationHandler(knobs={"outputs_return_none": True})
    assert h_none.get_outputs(_fake_builder(), {}) is None
    h_empty = SDKDelegationHandler(knobs={})
    assert h_empty.get_outputs(_fake_builder(), {}) == {}


def test_build_step_requires_sdk_step_class():
    h = SDKDelegationHandler(knobs={"input_mode": "none"})  # no sdk_step_class
    with pytest.raises(NotImplementedError, match="sdk_step_class"):
        h.build_step(_fake_builder(), inputs={}, dependencies=[])
