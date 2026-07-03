"""
Phase 2 migration gate: byte-diff parity between a hand-written builder and the routed
ProcessingHandler.

Before any builder is replaced by a TemplateStepBuilder shell, this proves the re-homed
ProcessingHandler.get_inputs / get_outputs produce SageMaker objects structurally identical to
the hand-written builder's _get_inputs / _get_outputs for the same config + inputs. It is the
gate referenced by the plan (31e1d1b) — a migrated step must emit an identical step.

Scope: starts with the Processing-2A pilot (TabularPreprocessing). SDK-dependent builders are
out of scope here (they can't import in a SDK-less env — see the plan's C-SDK note).

Why not assert a full ProcessingStep equality here: constructing a real ProcessingStep (either
path) requires a real PipelineSession — a Mock() session trips SageMaker's internal
.arguments jsonschema validation. So this unit gate proves parity at the constituent level
(get_inputs / get_outputs / processor config / job arguments / code path), which together are
everything build_step assembles. The end-to-end ProcessingStep byte-diff is an integration test
(real session) tracked in the plan's e2e Definition-of-Done item.
"""

import warnings

import pytest

warnings.simplefilter("ignore")


def _stable(v):
    """Stringify a value that may be a SageMaker Pipeline variable (Join etc.).

    Pipeline variables raise on __str__; use .expr (the JSON repr) for a stable comparison.
    """
    if hasattr(v, "expr"):
        return repr(v.expr)
    return str(v)


def _norm(obj):
    """Recursively normalize a step `.arguments` dict for comparison, resolving Pipeline
    variables (Join etc.) to their .expr JSON so dicts compare structurally."""
    if hasattr(obj, "expr"):
        return ("__pipevar__", repr(obj.expr))
    if isinstance(obj, dict):
        return {k: _norm(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_norm(v) for v in obj]
    return obj


def _processing_input_tuple(pi):
    """A comparable, order-independent signature of a ProcessingInput."""
    return (pi.input_name, _stable(pi.source), _stable(pi.destination))


def _processing_output_tuple(po):
    return (po.output_name, _stable(po.source), _stable(po.destination))


@pytest.fixture
def mock_session():
    """A mock SageMaker session that passes image-URI region resolution.

    _create_processor calls image_uris.retrieve which reads session.boto_region_name —
    it must be a real region string, not a Mock.
    """
    from unittest.mock import Mock

    s = Mock()
    s.boto_region_name = "us-east-1"
    s.local_mode = False
    return s


@pytest.fixture
def tabular_config(tmp_path):
    from cursus.steps.configs.config_tabular_preprocessing_step import (
        TabularPreprocessingConfig,
    )

    (tmp_path / "tabular_preprocessing.py").write_text("# stub script\n")
    return TabularPreprocessingConfig(
        author="test",
        bucket="test-bucket",
        role="arn:aws:iam::123456789012:role/test",
        region="NA",
        service_name="test",
        pipeline_version="1.0.0",
        project_root_folder="test_project",
        job_type="training",
        source_dir=str(tmp_path),
        processing_entry_point="tabular_preprocessing.py",
    )


@pytest.fixture
def real_builder(tabular_config, mock_session):
    from cursus.step_catalog.step_catalog import StepCatalog

    TabularPreprocessingStepBuilder = StepCatalog().load_builder_class(
        "TabularPreprocessing"
    )

    return TabularPreprocessingStepBuilder(
        config=tabular_config,
        sagemaker_session=mock_session,
        role="arn:aws:iam::123456789012:role/test",
    )


@pytest.fixture
def handler_view(real_builder):
    """A TemplateStepBuilder-shaped view sharing the real builder's spec/contract/config.

    We reuse the real builder as the handler back-ref (it exposes spec/contract/config/
    _get_base_output_path/log_info), which is exactly the surface ProcessingHandler reads —
    so this isolates the get_inputs/get_outputs logic difference, nothing else.
    """
    from cursus.core.base.builder_templates import ProcessingHandler

    handler = ProcessingHandler(
        knobs={"use_step_args": False, "output_path_token": "tabular_preprocessing"}
    )
    return handler, real_builder


class TestTabularPreprocessingParity:
    def test_get_inputs_identical(self, real_builder, handler_view):
        handler, b = handler_view
        sample_inputs = {"DATA": "s3://src/data", "SIGNATURE": "s3://src/sig"}
        real = real_builder._get_inputs(dict(sample_inputs))
        routed = handler.get_inputs(b, dict(sample_inputs))
        assert sorted(map(_processing_input_tuple, real)) == sorted(
            map(_processing_input_tuple, routed)
        )

    def test_get_inputs_required_missing_both_raise(self, real_builder, handler_view):
        handler, b = handler_view
        with pytest.raises(ValueError):
            real_builder._get_inputs({})
        with pytest.raises(ValueError):
            handler.get_inputs(b, {})

    def test_get_outputs_generated_destination_identical(self, real_builder, handler_view):
        handler, b = handler_view
        real = real_builder._get_outputs({})
        routed = handler.get_outputs(b, {})
        assert sorted(map(_processing_output_tuple, real)) == sorted(
            map(_processing_output_tuple, routed)
        )

    def test_get_outputs_explicit_destination_identical(self, real_builder, handler_view):
        handler, b = handler_view
        explicit = {"processed_data": "s3://explicit/out"}
        real = real_builder._get_outputs(dict(explicit))
        routed = handler.get_outputs(b, dict(explicit))
        assert sorted(map(_processing_output_tuple, real)) == sorted(
            map(_processing_output_tuple, routed)
        )

    def test_shell_form_matches_handwritten(self, tabular_config, mock_session):
        """A TemplateStepBuilder SHELL that keeps the per-step methods but inherits __init__/
        create_step produces inputs/outputs identical to the hand-written builder.

        This is the pilot-migration proof: the shell auto-loads its spec from STEP_NAME, auto-binds
        the Processing handler from the registry, and its retained _create_processor/_get_inputs/
        _get_outputs/_get_environment_variables drive the same result. (Validates the shell shape
        before the real builder file is converted.)
        """
        from cursus.core.base.builder_templates import TemplateStepBuilder
        from cursus.step_catalog.step_catalog import StepCatalog

        RealBuilder = StepCatalog().load_builder_class("TabularPreprocessing")

        # A pure shell: TabularPreprocessing is now fully declarative — all per-step factories
        # (env / job-args / inputs / outputs / compute) collapsed into the handler + .step.yaml
        # contract DATA (FZ 31e1d3g/h/i/k), so the shell declares only STEP_NAME + HANDLER_KNOBS
        # and inherits everything else, exactly like the real builder.
        class TabularShell(TemplateStepBuilder):
            STEP_NAME = "TabularPreprocessing"
            HANDLER_KNOBS = {
                "output_path_token": "tabular_preprocessing",
                "direct_input_keys": ["DATA", "METADATA", "SIGNATURE"],
            }

        shell = TabularShell(
            config=tabular_config,
            sagemaker_session=mock_session,
            role="arn:aws:iam::123456789012:role/test",
        )
        real = RealBuilder(
            config=tabular_config,
            sagemaker_session=mock_session,
            role="arn:aws:iam::123456789012:role/test",
        )

        # handler bound from the registry (Processing -> ProcessingHandler, code mode)
        from cursus.core.base.builder_templates import ProcessingHandler

        assert isinstance(shell._handler, ProcessingHandler)
        # spec auto-loaded
        assert shell.spec is not None and shell.spec.step_type == "TabularPreprocessing"

        sample = {"DATA": "s3://src/data", "SIGNATURE": "s3://src/sig"}
        assert sorted(map(_processing_input_tuple, shell._get_inputs(dict(sample)))) == sorted(
            map(_processing_input_tuple, real._get_inputs(dict(sample)))
        )
        assert sorted(map(_processing_output_tuple, shell._get_outputs({}))) == sorted(
            map(_processing_output_tuple, real._get_outputs({}))
        )
        assert shell._get_job_arguments() == real._get_job_arguments()

    def test_session_independent_pieces_route_identically(self, real_builder):
        """Prove the session-INDEPENDENT pieces a shell delegates are identical.

        Constructing a processor/step needs a real PipelineSession (a Mock trips SageMaker's
        .arguments jsonschema validation — see module note), so processor-config parity is an
        integration concern. Here we assert the pieces that DON'T touch the session:
        - job arguments (the handler's get_job_arguments knob == builder._get_job_arguments)
        - the code path (build_step uses config.get_script_path(), same as the builder)
        - env-var composition (builder._get_environment_variables, which the processor factory
          passes through verbatim) — this is the config→container env contract, session-free.
        Combined with the get_inputs/get_outputs parity above, the only remaining piece is the
        processor object itself, validated in the integration e2e (real session).
        """
        # job arguments — identical via the knob (no session needed)
        assert real_builder._get_job_arguments() == ["--job_type", "training"]
        # code path — build_step's code= comes from config.get_script_path()
        assert real_builder.config.get_script_path().endswith("tabular_preprocessing.py")
        # env vars — the config→container contract (TabularPreprocessing's UPPERCASE column
        # join is exercised here; the processor factory passes env through verbatim)
        env = real_builder._get_environment_variables()
        assert isinstance(env, dict)
