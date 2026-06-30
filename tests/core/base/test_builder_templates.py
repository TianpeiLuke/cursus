"""
Phase 0c tests: resolve_handler routing + TemplateStepBuilder facade + ProcessingHandler I/O.

These exercise the not-yet-wired template machinery in isolation (the 44 hand-written builders
remain the live path). The routing tests are the highest-value guard: they assert routing is by
sagemaker_step_type only (never by name), Processing sub-discriminates by step_assembly, and the
no-builder types raise.
"""

from types import SimpleNamespace

import pytest

from cursus.core.base.builder_templates import (
    NoBuilderError,
    ProcessingHandler,
    SDKDelegationHandler,
    TemplateStepBuilder,
    TrainingHandler,
    ModelCreationHandler,
    TransformHandler,
    resolve_handler,
)
from cursus.core.base.builder_base import StepBuilderBase


# --- routing ---


class TestResolveHandler:
    def test_one_to_one_types_route_by_type_alone(self):
        assert isinstance(resolve_handler("Training"), TrainingHandler)
        assert isinstance(resolve_handler("CreateModel"), ModelCreationHandler)
        assert isinstance(resolve_handler("Transform"), TransformHandler)
        assert isinstance(resolve_handler("CradleDataLoading"), SDKDelegationHandler)
        assert isinstance(resolve_handler("RedshiftDataLoading"), SDKDelegationHandler)
        assert isinstance(
            resolve_handler("MimsModelRegistrationProcessing"), SDKDelegationHandler
        )

    def test_processing_code_and_step_args_both_use_processing_handler(self):
        assert isinstance(resolve_handler("Processing", "code"), ProcessingHandler)
        assert isinstance(resolve_handler("Processing", "step_args"), ProcessingHandler)

    def test_processing_default_assembly_is_code(self):
        h = resolve_handler("Processing")  # no step_assembly
        assert isinstance(h, ProcessingHandler)
        assert h.knobs.get("use_step_args") is False

    def test_processing_step_args_sets_knob(self):
        h = resolve_handler("Processing", "step_args")
        assert h.knobs.get("use_step_args") is True

    def test_processing_delegation_routes_to_sdk_handler(self):
        assert isinstance(resolve_handler("Processing", "delegation"), SDKDelegationHandler)

    def test_no_builder_types_raise(self):
        with pytest.raises(NoBuilderError):
            resolve_handler("Base")
        with pytest.raises(NoBuilderError):
            resolve_handler("Lambda")

    def test_unknown_type_raises(self):
        with pytest.raises(NoBuilderError):
            resolve_handler("TotallyMadeUp")

    def test_unknown_processing_assembly_raises(self):
        with pytest.raises(ValueError):
            resolve_handler("Processing", "nonsense")

    def test_routing_never_uses_step_name(self):
        """DummyTraining is sagemaker_step_type=Processing -> must route as Processing, not Training."""
        h = resolve_handler("Processing", "step_args")  # DummyTraining's row
        assert isinstance(h, ProcessingHandler)
        assert not isinstance(h, TrainingHandler)


# --- facade contract ---


class TestTemplateStepBuilderContract:
    def test_is_step_builder_base_subclass(self):
        """The facade IS-A StepBuilderBase (preserves the assembler/discovery contract)."""
        assert issubclass(TemplateStepBuilder, StepBuilderBase)

    def test_shell_subclass_pattern(self):
        """A 2-line shell only declares STEP_NAME."""

        class MyStepBuilder(TemplateStepBuilder):
            STEP_NAME = "My"

        assert MyStepBuilder.STEP_NAME == "My"
        assert issubclass(MyStepBuilder, StepBuilderBase)

    def test_methods_require_bound_handler(self):
        """Delegating methods raise clearly if no handler is bound."""

        # build a facade without going through full __init__ machinery
        b = TemplateStepBuilder.__new__(TemplateStepBuilder)
        b._handler = None
        with pytest.raises(RuntimeError):
            b._get_inputs({})
        with pytest.raises(RuntimeError):
            b._get_outputs({})
        with pytest.raises(RuntimeError):
            b.create_step()

    def test_create_step_attaches_spec_even_if_handler_forgets(self):
        """FZ 31e1d3d2: _attach_spec is non-bypassable — the facade's create_step re-homes
        step._spec / step._contract regardless of whether the handler called _attach_spec.

        step._spec is the sole input to the builder-driven resolver-enrichment path
        (builder_base.py:929-930); a handler that returns a step without it would break that path
        silently (the assembler's primary path reads builder.spec, so there is no symptom on the
        common DAG route). The facade guarantees it.
        """
        sentinel_spec = SimpleNamespace(step_type="X")
        sentinel_contract = SimpleNamespace()

        class ForgetfulHandler:
            """A handler whose build_step deliberately does NOT call _attach_spec."""

            def build_step(self, b, **kwargs):
                return SimpleNamespace(name="raw-step")  # no _spec / _contract set

        b = TemplateStepBuilder.__new__(TemplateStepBuilder)
        b._handler = ForgetfulHandler()
        b.spec = sentinel_spec
        b.contract = sentinel_contract

        step = b.create_step()
        assert step._spec is sentinel_spec
        assert step._contract is sentinel_contract


# --- ProcessingHandler I/O against a mock builder ---


def _mock_builder(dependencies, outputs_spec, input_paths, output_paths, job_type="training"):
    """A minimal stand-in exposing the surface ProcessingHandler reads."""
    spec = SimpleNamespace(
        step_type="TabularPreprocessing",
        dependencies={
            d["logical_name"]: SimpleNamespace(
                logical_name=d["logical_name"], required=d["required"]
            )
            for d in dependencies
        },
        outputs={
            o: SimpleNamespace(logical_name=o) for o in outputs_spec
        },
    )
    contract = SimpleNamespace(
        expected_input_paths=input_paths, expected_output_paths=output_paths
    )
    logs = []
    return SimpleNamespace(
        spec=spec,
        contract=contract,
        config=SimpleNamespace(job_type=job_type),
        _get_base_output_path=lambda: "s3://bucket/base",
        log_info=lambda *a, **k: logs.append(a),
    )


class TestProcessingHandlerIO:
    def test_get_inputs_builds_processing_inputs(self):
        h = ProcessingHandler(knobs={"use_step_args": False})
        b = _mock_builder(
            dependencies=[{"logical_name": "DATA", "required": True}],
            outputs_spec=["processed_data"],
            input_paths={"DATA": "/opt/ml/processing/input/data"},
            output_paths={"processed_data": "/opt/ml/processing/output"},
        )
        pis = h.get_inputs(b, {"DATA": "s3://src/data"})
        assert len(pis) == 1
        assert pis[0].input_name == "DATA"
        assert pis[0].destination == "/opt/ml/processing/input/data"

    def test_get_inputs_skips_optional_absent_raises_required_absent(self):
        h = ProcessingHandler()
        b = _mock_builder(
            dependencies=[
                {"logical_name": "DATA", "required": True},
                {"logical_name": "SIG", "required": False},
            ],
            outputs_spec=["out"],
            input_paths={"DATA": "/in/data", "SIG": "/in/sig"},
            output_paths={"out": "/out"},
        )
        # optional absent -> skipped (only DATA returned)
        pis = h.get_inputs(b, {"DATA": "s3://d"})
        assert [p.input_name for p in pis] == ["DATA"]
        # required absent -> raises
        with pytest.raises(ValueError):
            h.get_inputs(b, {})

    def test_get_outputs_uses_explicit_then_generates(self):
        h = ProcessingHandler(knobs={"output_path_token": "tabular_preprocessing"})
        b = _mock_builder(
            dependencies=[],
            outputs_spec=["processed_data"],
            input_paths={},
            output_paths={"processed_data": "/opt/ml/processing/output"},
        )
        # explicit destination passes through
        pos = h.get_outputs(b, {"processed_data": "s3://explicit/dest"})
        assert pos[0].destination == "s3://explicit/dest"
        assert pos[0].source == "/opt/ml/processing/output"
        # generated destination is a Join (no explicit dest)
        pos2 = h.get_outputs(b, {})
        # Join object stringifies to include the token + job_type
        assert pos2[0].output_name == "processed_data"

    def test_build_step_requires_make_compute_knob(self):
        """Without a processor factory, build_step raises (per-step wiring pending)."""
        h = ProcessingHandler(knobs={"use_step_args": False})
        b = _mock_builder(
            dependencies=[], outputs_spec=[], input_paths={}, output_paths={}
        )
        with pytest.raises(NotImplementedError):
            h.build_step(b, inputs={}, outputs={}, dependencies=[])


class TestOutputTokenIsCanonicalSnakeByDefault:
    """FZ 31e1d3f: the generated output-path token DEFAULT is canonical_to_snake(step_type) — the
    package's own PascalCase->snake convention — in ALL three handlers that synthesize a destination
    (Processing/Training/Transform). A .step.yaml only declares output_path_token when it DEVIATES
    from this convention (shared namespaces, renames). This guards against a regression to the old
    step_type.lower() default, which never snake-cased multi-word types (TabularPreprocessing ->
    "tabularpreprocessing") and silently put 13 Processing + 5 Training/Transform steps on
    non-conventional S3 paths.
    """

    def _token_from_join(self, join, *, has_job_type):
        # Processing/Transform append job_type before the logical name; Training does not.
        # values = [base, token, (job_type?), (logical_name for Processing)]
        return join.values[1]

    def test_processing_default_token_is_canonical_snake(self):
        # _mock_builder uses step_type="TabularPreprocessing" -> "tabular_preprocessing"
        h = ProcessingHandler()  # NO knob -> must derive
        b = _mock_builder(
            dependencies=[],
            outputs_spec=["processed_data"],
            input_paths={},
            output_paths={"processed_data": "/opt/ml/processing/output"},
        )
        pos = h.get_outputs(b, {})
        assert pos[0].destination.values[1] == "tabular_preprocessing"

    def test_training_default_token_is_canonical_snake(self):
        h = TrainingHandler()
        b = _mock_builder(
            dependencies=[],
            outputs_spec=["model_output"],
            input_paths={},
            output_paths={"model_output": "/opt/ml/model"},
        )
        b.spec.step_type = "XGBoostTraining"
        join = h.get_outputs(b, {})
        assert join.values[1] == "xgboost_training"  # not "xgboosttraining"

    def test_transform_default_token_is_canonical_snake(self):
        h = TransformHandler()
        b = _mock_builder(
            dependencies=[],
            outputs_spec=["transform_output"],
            input_paths={},
            output_paths={"transform_output": "/opt/ml/output"},
        )
        b.spec.step_type = "BatchTransform"
        join = h.get_outputs(b, {})
        assert join.values[1] == "batch_transform"  # not "batchtransform"

    def test_output_token_is_NOT_overridable(self):
        # FZ 31e1d3f1b: output_path_token was removed — the S3 prefix is ALWAYS
        # canonical_to_snake(step_type), even if an output_path_token knob is (wrongly) injected.
        h = ProcessingHandler(knobs={"output_path_token": "model_evaluation"})  # ignored
        b = _mock_builder(
            dependencies=[],
            outputs_spec=["eval_output"],
            input_paths={},
            output_paths={"eval_output": "/opt/ml/processing/output"},
        )
        b.spec.step_type = "XGBoostModelEval"
        pos = h.get_outputs(b, {})
        assert pos[0].destination.values[1] == "xgboost_model_eval"  # derived, knob ignored


class TestAllHandlersImplemented:
    """S1 complete: all 4 construction handlers (Processing/Training/ModelCreation/Transform/
    SDKDelegation) are implemented — none remain a NotImplementedError stub. Their behavior is
    covered by the per-handler parity suites; this guards that the base verb set resolves and is
    constructible."""

    @pytest.mark.parametrize(
        "cls",
        [
            ProcessingHandler,
            TrainingHandler,
            ModelCreationHandler,
            TransformHandler,
            SDKDelegationHandler,
        ],
    )
    def test_handler_constructs_and_is_pattern_handler(self, cls):
        from cursus.core.base.builder_templates import PatternHandler

        h = cls()
        assert isinstance(h, PatternHandler)


def _build_step_mock_builder(use_step_args, script_path):
    """A builder stand-in that drives ProcessingHandler.build_step all the way to a REAL
    sagemaker ProcessingStep — including a real Processor returned by the make_compute knob.

    The pre-existing TestProcessingHandlerIO mocks stop at get_inputs/get_outputs and never
    construct a ProcessingStep, so they could not catch the step_args/processor XOR violation
    (FZ 31e1d3j2 — the SAIS end-to-end run did). This fixture closes that gap by exercising the
    actual ProcessingStep constructor, whose __init__ raises
    "either step_args or processor need to be given, but not both."

    `script_path` must point at a real file on disk: the 2B path calls processor.run(code=...),
    which validates the code file exists.
    """
    from sagemaker.sklearn.processing import SKLearnProcessor
    from sagemaker.workflow.pipeline_context import PipelineSession

    spec = SimpleNamespace(
        step_type="TabularPreprocessing",
        dependencies={
            "DATA": SimpleNamespace(logical_name="DATA", required=True),
        },
        outputs={"processed_data": SimpleNamespace(logical_name="processed_data")},
    )
    contract = SimpleNamespace(
        expected_input_paths={"DATA": "/opt/ml/processing/input/data"},
        expected_output_paths={"processed_data": "/opt/ml/processing/output"},
        circular_ref_check=False,
        skip_inputs=[],
        input_source_overrides={},
        sink=False,
    )

    def _make_processor(_b):
        # A real Processor on a PipelineSession so processor.run() DEFERS (no job started, no
        # AWS call) and returns step_args — exactly how the real pipeline compiles offline.
        return SKLearnProcessor(
            framework_version="1.2-1",
            role="arn:aws:iam::123456789012:role/dummy",
            instance_type="ml.m5.large",
            instance_count=1,
            sagemaker_session=PipelineSession(),
        )

    return SimpleNamespace(
        spec=spec,
        contract=contract,
        config=SimpleNamespace(
            job_type="training",
            get_script_path=lambda: script_path,
        ),
        knobs={"use_step_args": use_step_args, "make_compute": _make_processor},
        _get_base_output_path=lambda: "s3://bucket/base",
        _get_job_arguments=lambda: None,
        _get_step_name=lambda: "TabularPreprocessing_training",
        _get_cache_config=lambda enable: None,
        log_info=lambda *a, **k: None,
        _detect_circular_references=lambda v: False,
    )


@pytest.fixture
def _sm_region(monkeypatch):
    # SKLearnProcessor.run() resolves a region; set one so the test needs no AWS credentials.
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def _script_file(tmp_path):
    p = tmp_path / "preprocess.py"
    p.write_text("def main():\n    pass\n")
    return str(p)


class TestProcessingStepArgsProcessorXor:
    """Regression for FZ 31e1d3j2: ProcessingHandler.build_step must obey SageMaker's XOR — the
    2B (step_args) branch must NOT also pass `processor`, and the 2A branch must pass `processor`
    and NOT `step_args`. Both branches must produce a ProcessingStep without
    'either step_args or processor need to be given, but not both.'"""

    def test_2b_step_args_branch_does_not_also_pass_processor(
        self, _sm_region, _script_file
    ):
        h = ProcessingHandler(knobs={"use_step_args": True})
        b = _build_step_mock_builder(use_step_args=True, script_path=_script_file)
        h.knobs = b.knobs  # handler reads its own knobs; align with the builder's make_compute
        step = h.build_step(
            b, inputs={"DATA": "s3://src/data"}, outputs={}, dependencies=[]
        )
        # The step is built with step_args ONLY — processor is None on the step object.
        assert getattr(step, "step_args", None) is not None
        assert getattr(step, "processor", None) is None

    def test_2a_code_branch_passes_processor_not_step_args(
        self, _sm_region, _script_file
    ):
        h = ProcessingHandler(knobs={"use_step_args": False})
        b = _build_step_mock_builder(use_step_args=False, script_path=_script_file)
        h.knobs = b.knobs
        step = h.build_step(
            b, inputs={"DATA": "s3://src/data"}, outputs={}, dependencies=[]
        )
        assert getattr(step, "processor", None) is not None
        assert getattr(step, "step_args", None) is None
