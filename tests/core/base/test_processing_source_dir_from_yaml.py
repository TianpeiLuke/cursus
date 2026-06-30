"""
Standardized Processing-step source_dir contract (the "two patterns, both step_args" decision):
source_dir is per-step DATA in the .step.yaml (contract.source_dir), read by ProcessingHandler as
the split_source_dir switch — NOT inferred from the processor class, NOT a per-shell knob.

  contract.source_dir: false (default) -> processor.run(code=<full_script_path>)        [no source_dir]
  contract.source_dir: true            -> processor.run(code=<entry>, source_dir=<dir>)  [split]

These run-against the real .step.yaml interfaces + a recording-fake processor (no SageMaker session).
"""

import warnings

warnings.simplefilter("ignore")

import pytest  # noqa: E402

from cursus.core.base.builder_templates import ProcessingHandler  # noqa: E402
from cursus.steps.interfaces import load_interface  # noqa: E402

# The 9 Processing steps whose script needs its directory uploaded (verified source audit).
_SOURCE_DIR_TRUE = {
    "RiskTableMapping", "TokenizerTraining", "DummyTraining", "LightGBMMTModelEval",
    "LightGBMMTModelInference", "PyTorchModelEval", "PyTorchModelInference",
    "BedrockProcessing", "PercentileModelCalibration",
}


class _Captured(Exception):
    def __init__(self, kw):
        self.kw = kw


class _FakeProc:
    def run(self, **kw):
        raise _Captured(kw)


def _fake_builder(script_path, contract_source_dir):
    from types import SimpleNamespace

    return SimpleNamespace(
        spec=SimpleNamespace(step_type="X", dependencies={}, outputs={}),
        contract=SimpleNamespace(
            expected_input_paths={}, expected_output_paths={}, input_channels={},
            source_dir=contract_source_dir,
        ),
        config=SimpleNamespace(get_script_path=lambda: script_path, job_type=None),
        _create_processor=lambda: _FakeProc(),
        _get_step_name=lambda: "X",
        _get_cache_config=lambda e: None,
        _get_job_arguments=lambda: None,
        extract_inputs_from_dependencies=lambda d: {},
        log_info=lambda *a, **k: None,
        log_warning=lambda *a, **k: None,
    )


def _run_kwargs(contract_source_dir, knobs=None):
    h = ProcessingHandler(knobs={"use_step_args": True, **(knobs or {})})
    b = _fake_builder("/proj/scripts/run.py", contract_source_dir)
    try:
        h.build_step(b, inputs={}, outputs={}, dependencies=[], enable_caching=False)
    except _Captured as c:
        return c.kw
    raise AssertionError("processor.run was not called")


def test_source_dir_false_runs_full_code_path():
    kw = _run_kwargs(contract_source_dir=False)
    assert kw["code"] == "/proj/scripts/run.py"
    assert kw.get("source_dir") is None


def test_source_dir_true_splits_into_source_dir_and_entry_point():
    kw = _run_kwargs(contract_source_dir=True)
    assert kw["code"] == "run.py"
    assert kw["source_dir"] == "/proj/scripts"


def test_explicit_knob_overrides_contract():
    # split_source_dir knob (when set) wins over contract.source_dir.
    kw = _run_kwargs(contract_source_dir=False, knobs={"split_source_dir": True})
    assert kw["code"] == "run.py" and kw["source_dir"] == "/proj/scripts"


@pytest.mark.parametrize("step_name", sorted(_SOURCE_DIR_TRUE))
def test_split_steps_declare_source_dir_true_in_yaml(step_name):
    iface = load_interface(step_name)
    assert iface.contract.source_dir is True, f"{step_name} must declare contract.source_dir: true"


def test_non_split_processing_steps_default_false():
    # A representative self-contained Processing step has no source_dir (default False).
    iface = load_interface("TabularPreprocessing")
    assert iface.contract.source_dir is False
