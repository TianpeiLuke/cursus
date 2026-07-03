"""
Template-based step builder + construction-verb handlers (Phase 0c of the simplification).

This module introduces a single ``TemplateStepBuilder`` facade and a set of construction-verb
``PatternHandler`` strategies, selected at build time by ``resolve_handler(sagemaker_step_type,
step_assembly)``. The goal is to replace the 44 hand-written ``<Name>StepBuilder`` classes with
thin shells that declare only ``STEP_NAME`` and inherit everything else.

**Status: Phase S3 in progress — wired, 2/45 builders are live shells.** This facade IS now
routed: ``TabularPreprocessingStepBuilder`` and ``BatchTransformStepBuilder`` are live shells that
construct through it; the remaining 43 stay hand-written until their byte-diff-gated batch migrates
(see the plan's Phase S3). All 5 construction-verb handlers (ProcessingHandler, TrainingHandler,
ModelCreationHandler, TransformHandler, SDKDelegationHandler) are fully implemented + covered by
session-independent parity suites in ``tests/core/base/``.

Design notes (verified against source):
  * 6 construction verbs, but Processing-2A and Processing-2B collapse to ONE ``ProcessingHandler``
    with ``use_step_args`` / ``split_source_dir`` knobs (NI-1) — they share get_inputs/get_outputs
    verbatim and differ only in build_step.
  * The facade does NOT impose a fixed make_compute→inputs→outputs→build order — it hands the
    merged inputs/outputs to ``handler.build_step`` and lets the handler orchestrate, because
    Transform runs make_compute last (NI-3) and ModelCreation drops caching (NI-4).
  * ``sagemaker_step_type`` alone selects the handler for 5 verbs; only ``Processing`` needs the
    ``step_assembly`` sub-discriminator (code | step_args | delegation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from sagemaker.workflow.steps import ProcessingStep, Step

from .builder_base import StepBuilderBase
from ...registry.strategy_registry import (  # Edge A: one-directional runtime import
    KnobSpec,
    NoBuilderError,
    axis_name_for_step_type,
    register_no_builder,
    register_strategy,
    resolve_strategy,
)

# NoBuilderError is re-exported here for back-compat (it was historically defined in this
# module; routing vocabulary now lives in strategy_registry).
__all__ = [
    "TemplateStepBuilder",
    "PatternHandler",
    "ProcessingHandler",
    "TrainingHandler",
    "ModelCreationHandler",
    "TransformHandler",
    "SDKDelegationHandler",
    "resolve_handler",
    "NoBuilderError",
]


# ---------------------------------------------------------------------------
# Pattern handlers
# ---------------------------------------------------------------------------


def _overrides(builder, method_name: str) -> bool:
    """True if ``builder``'s class defines its own ``method_name`` (a per-step override),
    i.e. the bound method is NOT the one inherited from ``TemplateStepBuilder``.

    Lets the handler prefer a shell's per-step ``_get_inputs``/``_get_outputs`` (smooth
    transition: a builder keeps the deviating method, deletes only the boilerplate)."""
    own = getattr(type(builder), method_name, None)
    base = getattr(TemplateStepBuilder, method_name, None)
    return own is not None and own is not base


class PatternHandler(ABC):
    """Base strategy for one construction verb.

    A handler is stateless config: it holds per-step declarative ``knobs`` and receives the
    owning ``TemplateStepBuilder`` (``b``) on each call, reading ``b.config`` / ``b.spec`` /
    ``b.contract`` / ``b.role`` / ``b.session`` and calling base helpers
    (``b._get_step_name()``, ``b._get_base_output_path()``, ``b._get_cache_config()``,
    ``b.extract_inputs_from_dependencies()``, ``b.log_info`` …).
    """

    #: Config attributes this handler reads off ``b.config`` at BUILD time that are NOT expressible
    #: as a ``.step.yaml`` descriptor (FZ 31e1d3g3 Phase D1 / OQ 31e1d3h1). The B3 RegistryBinding
    #: validator unions this with the descriptor-derived attrs (compute ``*_field`` names,
    #: ``contract.input_source_overrides`` values) to check config-field COVERAGE — i.e. that the
    #: resolved config class actually supplies every field the bound handler will ``getattr`` at
    #: build time. Declared as DATA (not scraped from source) because some reads use a runtime attr
    #: name (``getattr(b.config, attr)`` with ``attr`` from the contract) that is statically
    #: undecidable. Empty for handlers that read only spec/contract, not config.
    requires_config_fields: tuple = ()

    def __init__(self, knobs: Optional[Dict[str, Any]] = None):
        self.knobs = knobs or {}

    # The per-verb construction. Owns ordering (make_compute / get_inputs / get_outputs are
    # invoked from here in the order the verb requires) and the final setattr(step, "_spec").
    @abstractmethod
    def build_step(self, b: "TemplateStepBuilder", **kwargs: Any) -> Step: ...

    # Spec×contract message passing, re-homed per verb. Default raises; verbs override.
    def get_inputs(self, b: "TemplateStepBuilder", inputs: Dict[str, Any]) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.get_inputs")

    def get_outputs(self, b: "TemplateStepBuilder", outputs: Dict[str, Any]) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.get_outputs")

    # --- shared helpers usable by every handler ---

    def _merge_inputs(
        self, b: "TemplateStepBuilder", kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The universal input-merge: extract-from-deps → inputs_raw override → direct keys.

        Mirrors the identical preamble every hand-written ``create_step`` runs. The
        ``direct_input_keys`` allowlist comes from the per-step handler knob (and may also be
        passed in kwargs); for each key, the value is taken from kwargs if the assembler/template
        passed it as a direct keyword.
        """
        inputs_raw = kwargs.get("inputs", {})
        dependencies = kwargs.get("dependencies", [])
        inputs: Dict[str, Any] = {}
        if dependencies:
            try:
                inputs.update(b.extract_inputs_from_dependencies(dependencies))
            except (
                Exception
            ) as e:  # pragma: no cover - matches builder log-and-continue
                b.log_warning("Failed to extract inputs from dependencies: %s", e)
        inputs.update(inputs_raw)
        # direct keyword inputs (e.g. DATA/METADATA from a template) declared per step via the
        # handler's `direct_input_keys` knob (or passed in kwargs).
        direct_keys = list(self.knobs.get("direct_input_keys", [])) + list(
            kwargs.get("direct_input_keys", [])
        )
        for key in direct_keys:
            if key in kwargs and key not in inputs:
                inputs[key] = kwargs[key]
        return inputs

    @staticmethod
    def _attach_spec(b: "TemplateStepBuilder", step: Step) -> Step:
        """The universal post-construction ``setattr(step, "_spec", ...)`` (+ contract)."""
        if getattr(b, "spec", None) is not None:
            setattr(step, "_spec", b.spec)
        if getattr(b, "contract", None) is not None:
            setattr(step, "_contract", b.contract)
        return step


_PROCESSING_KNOBS = (
    KnobSpec(
        "use_step_args",
        "bool",
        False,
        doc="2B (processor.run->step_args) vs 2A (code=)",
    ),
    KnobSpec(
        "split_source_dir",
        "bool",
        None,
        doc="2B: split get_script_path into source_dir+entry_point; None => read contract.source_dir",
    ),
    KnobSpec(
        "include_job_type_in_path",
        "bool",
        None,
        doc="config.job_type as a path segment; None => contract.include_job_type_in_path (default True)",
    ),
    KnobSpec(
        "make_compute",
        "callable",
        doc="processor factory escape-hatch; else compute.kind drives _create_compute",
    ),
    KnobSpec(
        "direct_input_keys", "list", [], doc="template-provided direct input keys"
    ),
)


@register_strategy(
    axis="step_assembly",
    name="code",
    verb="Processing",
    preset_knobs={"use_step_args": False},
    knobs=_PROCESSING_KNOBS,
)
@register_strategy(
    axis="step_assembly",
    name="step_args",
    verb="Processing",
    preset_knobs={"use_step_args": True},
    knobs=_PROCESSING_KNOBS,
)
class ProcessingHandler(PatternHandler):
    """Processing verb — covers both 2A (``code=``) and 2B (``processor.run()→step_args``).

    Knobs:
      * ``use_step_args`` (bool): 2B if True (``processor.run()`` then
        ``ProcessingStep(step_args=...)``), else 2A (``ProcessingStep(code=...)``).
      * ``split_source_dir`` (bool): for 2B, split ``get_script_path()`` into
        ``source_dir`` + ``code=entry_point``.
      * ``include_job_type_in_path`` (bool): whether ``config.job_type`` is a path segment.
        (The output-prefix segment itself is DERIVED from the step name — ``canonical_to_snake`` —
        not a knob; FZ 31e1d3f1b.)

    NOTE: ``make_compute`` (the processor factory) and ``get_environment_variables`` are
    per-step and currently still live on each builder; the migration re-homes them via a
    ``make_compute`` knob/hook. Until then a routed Processing step must supply its processor
    factory. This handler implements the *shared* get_inputs/get_outputs join and the build_step
    shape; the processor-factory wiring is completed per-step in the Phase-2 Batch-A/C work.
    """

    def get_inputs(
        self, b: "TemplateStepBuilder", inputs: Dict[str, Any]
    ) -> List["ProcessingInput"]:  # noqa: F821 (sagemaker type)
        from sagemaker.processing import ProcessingInput

        if not b.spec:
            raise ValueError("Step specification is required")
        if not b.contract:
            raise ValueError("Script contract is required for input mapping")

        # Three per-step DATA deviations from the standard spec×contract input loop (FZ 31e1d3i),
        # read from the .step.yaml contract (knob overrides allowed). These collapsed the last
        # _get_inputs overrides — the loop itself was already identical across them:
        #   - circular_ref_check: run the PipelineVariable circular-reference guard first.
        #   - skip_inputs: declared dependencies the script loads internally (not mounted as inputs).
        #   - input_source_overrides {logical: config_attr}: take the SOURCE from a config attr/method
        #     (config is the value source) instead of the resolved dependency dict.
        circular_ref_check = self.knobs.get("circular_ref_check")
        if circular_ref_check is None:
            circular_ref_check = getattr(b.contract, "circular_ref_check", False)
        if circular_ref_check:
            for input_name, input_value in inputs.items():
                if b._detect_circular_references(input_value):
                    raise ValueError(
                        f"Circular reference detected in input '{input_name}'"
                    )

        skip = set(
            self.knobs.get("skip_inputs")
            or getattr(b.contract, "skip_inputs", [])
            or []
        )
        source_overrides = (
            self.knobs.get("input_source_overrides")
            or getattr(b.contract, "input_source_overrides", {})
            or {}
        )

        processing_inputs = []
        extra = self.knobs.get("extra_processing_input_kwargs", {})
        for _, dependency_spec in b.spec.dependencies.items():
            logical_name = dependency_spec.logical_name
            if logical_name in skip:
                continue
            # Config-sourced input: the source comes from a config attr/method, not the dep dict.
            if logical_name in source_overrides:
                if logical_name not in b.contract.expected_input_paths:
                    raise ValueError(
                        f"No container path found for input: {logical_name}"
                    )
                attr = source_overrides[logical_name]
                resolved = getattr(b.config, attr)
                source = resolved() if callable(resolved) else resolved
                processing_inputs.append(
                    ProcessingInput(
                        input_name=logical_name,
                        source=source,
                        destination=b.contract.expected_input_paths[logical_name],
                        **extra,
                    )
                )
                continue
            if not dependency_spec.required and logical_name not in inputs:
                continue
            if dependency_spec.required and logical_name not in inputs:
                raise ValueError(f"Required input '{logical_name}' not provided")
            if logical_name not in b.contract.expected_input_paths:
                raise ValueError(f"No container path found for input: {logical_name}")
            container_path = b.contract.expected_input_paths[logical_name]
            processing_inputs.append(
                ProcessingInput(
                    input_name=logical_name,
                    source=inputs[logical_name],
                    destination=container_path,
                    **extra,
                )
            )
        return processing_inputs

    def get_outputs(
        self, b: "TemplateStepBuilder", outputs: Dict[str, Any]
    ) -> List["ProcessingOutput"]:  # noqa: F821
        from sagemaker.processing import ProcessingOutput
        from sagemaker.workflow.functions import Join

        from ...step_catalog.naming import canonical_to_snake

        if not b.spec:
            raise ValueError("Step specification is required")
        if not b.contract:
            raise ValueError("Script contract is required for output mapping")

        # SINK step (e.g. an uploader) produces no outputs (FZ 31e1d3i) — declared via contract.sink.
        if self.knobs.get("sink") or getattr(b.contract, "sink", False):
            return []

        # The output-destination S3 prefix segment is DERIVED from the step name —
        # canonical_to_snake(step_type) (the package's PascalCase->snake util, acronyms handled) — the
        # convention for ~all steps. OPT-IN override: contract.output_path_token, when set, is used
        # VERBATIM instead (FZ 31e1d3f1b re-introduced as an escape hatch, default-off) — needed when
        # an external consumer keys off a fixed S3 folder name that does not match the cursus step name
        # (e.g. PIPER scans <pipeline>/Model_Metric_Generation_Step/ for .metric files).
        # include_job_type_in_path STAYS a per-step knob (genuinely variable: some steps segment the
        # path by job_type, some don't), read knob->contract->default.
        token = getattr(b.contract, "output_path_token", None) or canonical_to_snake(
            b.spec.step_type
        )
        include_job_type = self.knobs.get("include_job_type_in_path")
        if include_job_type is None:
            include_job_type = getattr(b.contract, "include_job_type_in_path", True)

        processing_outputs = []
        for _, output_spec in b.spec.outputs.items():
            logical_name = output_spec.logical_name
            if logical_name not in b.contract.expected_output_paths:
                raise ValueError(f"No container path found for output: {logical_name}")
            container_path = b.contract.expected_output_paths[logical_name]
            if logical_name in outputs:
                destination = outputs[logical_name]
            else:
                base = b._get_base_output_path()
                values = [base, token]
                if include_job_type and getattr(b.config, "job_type", None):
                    values.append(b.config.job_type)
                values.append(logical_name)
                destination = Join(on="/", values=values)
                b.log_info(
                    "Using generated destination for '%s': %s",
                    logical_name,
                    destination,
                )
            processing_outputs.append(
                ProcessingOutput(
                    output_name=logical_name,
                    source=container_path,
                    destination=destination,
                )
            )
        return processing_outputs

    def build_step(self, b: "TemplateStepBuilder", **kwargs: Any) -> Step:
        inputs = self._merge_inputs(b, kwargs)
        outputs = kwargs.get("outputs", {})
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        make_compute = self.knobs.get("make_compute")
        if make_compute is None:
            # Resolution order (FZ 31e1d3k): a per-step _create_processor override wins (genuine
            # keep); else the declarative contract.compute descriptor drives the base
            # _create_compute(); else error. A migrated step declares `compute:` in its .step.yaml
            # and drops the factory.
            # A per-step _create_processor (class override OR a test/instance attr) wins; else the
            # declarative contract.compute descriptor drives the base _create_compute(); else error.
            if _overrides(b, "_create_processor") or "_create_processor" in vars(b):
                make_compute = lambda _b: _b._create_processor()  # noqa: E731
            elif getattr(getattr(b, "contract", None), "compute", None) and getattr(
                b.contract.compute, "kind", None
            ):
                make_compute = lambda _b: _b._create_compute()  # noqa: E731
            else:
                raise NotImplementedError(
                    "ProcessingHandler needs a `make_compute` knob, a builder _create_processor(), "
                    "or a contract.compute descriptor."
                )
        processor = make_compute(b)
        # Prefer the builder's own _get_inputs/_get_outputs (a shell override wins via MRO);
        # fall back to the handler's generic spec×contract join. Same for job arguments.
        proc_inputs = (
            b._get_inputs(inputs)
            if _overrides(b, "_get_inputs")
            else self.get_inputs(b, inputs)
        )
        proc_outputs = (
            b._get_outputs(outputs)
            if _overrides(b, "_get_outputs")
            else self.get_outputs(b, outputs)
        )
        # job-args come from the builder's _get_job_arguments (which delegates to
        # config.get_job_arguments() — the single source, FZ 31e1d3h). The former `get_job_arguments`
        # knob escape-hatch was never set by any builder once the config-collapse landed, so it was
        # removed (dead branch).
        job_args = b._get_job_arguments() if hasattr(b, "_get_job_arguments") else None
        step_name = b._get_step_name()
        script_path = b.config.get_script_path()
        cache_config = b._get_cache_config(enable_caching)

        # NOTE: `processor` is NOT in `common`. SageMaker's ProcessingStep enforces an XOR —
        # exactly one of `step_args` / `processor` may be given ("either step_args or processor
        # need to be given, but not both."). In 2B (`use_step_args`) the processor is already
        # embedded in `step_args` via processor.run(), so the step takes `step_args` ONLY; passing
        # `processor` too raises. 2A passes `processor` (no step_args). So `processor` is added
        # per-branch, never shared. (FZ 31e1d3j2 — caught by the SAIS end-to-end run.)
        common = dict(
            name=step_name,
            depends_on=dependencies,
            cache_config=cache_config,
        )
        if self.knobs.get("use_step_args"):
            entry_point = script_path
            source_dir = None
            # source_dir is per-step DATA in the .step.yaml (contract.source_dir); the knob is an
            # explicit override / back-compat. True => run(code=entry_point, source_dir=<dir>);
            # False => run(code=<full_script_path>). See ContractSection.source_dir.
            split = self.knobs.get("split_source_dir")
            if split is None:
                split = bool(getattr(b.contract, "source_dir", False))
            if split:
                from pathlib import Path as _P

                source_dir = str(_P(script_path).parent)
                entry_point = _P(script_path).name
            run_kwargs = dict(
                code=entry_point, inputs=proc_inputs, outputs=proc_outputs
            )
            if source_dir:
                run_kwargs["source_dir"] = source_dir
            if job_args:
                run_kwargs["arguments"] = job_args
            step_args = processor.run(**run_kwargs)
            # 2B: processor is embedded in step_args — do NOT also pass processor= (XOR).
            step = ProcessingStep(step_args=step_args, **common)
        else:
            # 2A: pass the processor directly (no step_args).
            step = ProcessingStep(
                processor=processor,
                inputs=proc_inputs,
                outputs=proc_outputs,
                code=script_path,
                job_arguments=job_args,
                **common,
            )
        return self._attach_spec(b, step)


@register_strategy(
    axis="sagemaker_step_type",
    name="Training",
    verb="Training",
    knobs=(
        KnobSpec(
            "make_compute",
            "callable",
            doc="estimator factory; defaults to builder._create_estimator",
        ),
        KnobSpec("direct_input_keys", "list", ["input_path"], doc="direct input keys"),
    ),
)
class TrainingHandler(PatternHandler):
    """Training verb — builds a ``TrainingStep(estimator, inputs=channels)``.

    Re-homed from ``builder_xgboost_training_step.py`` (3 of 4 builders use the path-parts channel
    parser; PyTorch keeps its own ``_get_inputs`` override). Distinctive:
      * ``get_inputs`` returns ``Dict[str, TrainingInput]`` keyed by **channel** name;
        ``input_path`` fans out to train/val/test; ``hyperparameters_s3_uri`` skipped when
        ``config.skip_hyperparameters_s3_uri``.
      * ``get_outputs`` returns a **single str/Join** ``output_path`` (not a list).
      * ``build_step`` ORDER: get_inputs → empty-guard → get_outputs → make_compute (the estimator
        is created WITH ``output_path`` threaded in, so outputs run BEFORE compute).
      * caching is STANDARD (``cache_config=_get_cache_config(enable_caching)``).
    """

    KNOBS = (
        KnobSpec(
            "make_compute",
            "callable",
            doc="estimator factory; defaults to builder._create_estimator",
        ),
        KnobSpec(
            "direct_input_keys",
            "list",
            ["input_path"],
            doc="direct input keys (input_path threads in)",
        ),
    )

    #: Fallback sub-channels for a fan-out input when the .step.yaml does not declare
    #: ``contract.inputs.<name>.channels`` (back-compat for not-yet-annotated interfaces). The
    #: source of truth is the YAML ``channels`` list — see ``channels_for``.
    DEFAULT_FANOUT_CHANNELS = ("train", "val", "test")

    def _create_data_channels_from_source(self, base_path, channels):
        from sagemaker.inputs import TrainingInput
        from sagemaker.workflow.functions import Join

        return {
            ch: TrainingInput(s3_data=Join(on="/", values=[base_path, f"{ch}/"]))
            for ch in channels
        }

    @classmethod
    def channels_for(cls, logical_name, container_path, declared_channels=None):
        """The SageMaker training channel name(s) a dependency maps to — the SINGLE SOURCE of the
        channel rule, shared by ``get_inputs`` (runtime) and the ``steps io`` tool (static).

        Priority: (1) the ``channels`` declared on the input in the ``.step.yaml`` (per-step DATA);
        (2) for the conventional ``input_path`` with no declaration, the back-compat
        ``DEFAULT_FANOUT_CHANNELS``; (3) the ``parts[5]`` of an ``/opt/ml/input/data/<channel>``
        container path; (4) the logical name itself. No resolved input value is needed.
        """
        if declared_channels:
            return list(declared_channels)
        if logical_name == "input_path":
            return list(cls.DEFAULT_FANOUT_CHANNELS)
        parts = (container_path or "").split("/")
        if len(parts) > 5 and parts[1:5] == ["opt", "ml", "input", "data"]:
            return [parts[5]]
        return [logical_name]

    def get_inputs(self, b, inputs):  # -> Dict[str, TrainingInput]
        from sagemaker.inputs import TrainingInput

        if not b.spec:
            raise ValueError("Step specification is required")
        if not b.contract:
            raise ValueError("Script contract is required for input mapping")
        training_inputs = {}
        for _, dep in b.spec.dependencies.items():
            logical_name = dep.logical_name
            if logical_name == "hyperparameters_s3_uri" and getattr(
                b.config, "skip_hyperparameters_s3_uri", False
            ):
                continue
            if not dep.required and logical_name not in inputs:
                continue
            if dep.required and logical_name not in inputs:
                raise ValueError(f"Required input '{logical_name}' not provided")
            if logical_name not in b.contract.expected_input_paths:
                raise ValueError(f"No container path found for input: {logical_name}")
            container_path = b.contract.expected_input_paths[logical_name]
            declared = getattr(b.contract, "input_channels", {}).get(logical_name)
            channels = self.channels_for(logical_name, container_path, declared)
            # A fan-out input (multiple channels under <base>/<channel>/) — declared in the
            # .step.yaml, or the conventional `input_path` for back-compat.
            if declared or logical_name == "input_path":
                training_inputs.update(
                    self._create_data_channels_from_source(
                        inputs[logical_name], channels
                    )
                )
            else:
                # Single channel: parts[5] of the container path, else the logical name.
                training_inputs[channels[0]] = TrainingInput(
                    s3_data=inputs[logical_name]
                )
        return training_inputs

    def get_outputs(self, b, outputs):  # -> str / Join (single)
        from sagemaker.workflow.functions import Join

        from ...step_catalog.naming import canonical_to_snake

        if not b.spec:
            raise ValueError("Step specification is required")
        if not b.contract:
            raise ValueError("Script contract is required for output mapping")
        for _, output_spec in b.spec.outputs.items():
            if output_spec.logical_name in outputs:
                return outputs[output_spec.logical_name]
        # The output S3 prefix is DERIVED from the step name — canonical_to_snake(step_type), the
        # CONVENTION (FZ 31e1d3f1b). OPT-IN override: contract.output_path_token used verbatim if set.
        token = getattr(b.contract, "output_path_token", None) or (
            canonical_to_snake(b.spec.step_type)
            if hasattr(b.spec, "step_type")
            else "training"
        )
        return Join(on="/", values=[b._get_base_output_path(), token])

    def build_step(self, b, **kwargs):
        from sagemaker.workflow.steps import TrainingStep

        inputs = self._merge_inputs(b, kwargs)
        # direct input_path kwarg threads in (builder :428-429)
        input_path = kwargs.get("input_path")
        if input_path is not None:
            inputs["input_path"] = input_path
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        step_name = b._get_step_name()
        # ORDER: inputs -> empty-guard -> outputs -> make_compute (estimator gets output_path)
        training_inputs = (
            b._get_inputs(inputs)
            if _overrides(b, "_get_inputs")
            else self.get_inputs(b, inputs)
        )
        if len(training_inputs) == 0:
            raise ValueError(
                "No training inputs available. Provide input_path or ensure dependencies "
                "supply necessary outputs."
            )
        output_path = (
            b._get_outputs({})
            if _overrides(b, "_get_outputs")
            else self.get_outputs(b, {})
        )
        make_compute = self.knobs.get("make_compute")
        if make_compute is None:
            if _overrides(b, "_create_estimator") or "_create_estimator" in vars(b):
                make_compute = lambda _b, op: _b._create_estimator(op)  # noqa: E731
            elif getattr(getattr(b, "contract", None), "compute", None) and getattr(
                b.contract.compute, "kind", None
            ):
                make_compute = lambda _b, op: _b._create_compute(op)  # noqa: E731
            else:
                raise NotImplementedError(
                    "TrainingHandler needs a `make_compute` knob, a builder _create_estimator(), "
                    "or a contract.compute descriptor."
                )
        estimator = make_compute(
            b, output_path
        )  # AFTER get_outputs (output_path threads in)
        try:
            step = TrainingStep(
                name=step_name,
                estimator=estimator,
                inputs=training_inputs,
                depends_on=dependencies,
                cache_config=b._get_cache_config(enable_caching),
            )
        except Exception as e:
            b.log_warning("Error creating TrainingStep: %s", str(e))
            raise ValueError(f"Failed to create TrainingStep: {str(e)}") from e
        return self._attach_spec(b, step)


_MODEL_CREATION_KNOBS = (
    KnobSpec(
        "make_compute",
        "callable",
        doc="model factory; defaults to builder._create_model",
    ),
    KnobSpec(
        "direct_input_keys", "list", [], doc="template-provided direct input keys"
    ),
)


@register_strategy(
    axis="sagemaker_step_type",
    name="CreateModel",
    verb="ModelCreation",
    knobs=_MODEL_CREATION_KNOBS,
)
class ModelCreationHandler(PatternHandler):
    """ModelCreation verb — builds a ``CreateModelStep(model=...)``.

    Distinctive (re-homed from ``builder_xgboost_model_step.py`` / ``builder_pytorch_model_step.py``):
      * ``get_inputs`` is a single-key ``{"model_data": ...}`` passthrough (NOT a spec×contract
        join, NOT the DummyTraining 3-tier resolution); raises if ``model_data`` absent.
      * ``get_outputs`` returns ``None`` (CreateModelStep auto-exposes ``properties.ModelName``).
      * ``make_compute`` (the model factory) runs **LAST**, consuming ``model_data``.
      * **caching is DROPPED** — CreateModelStep takes no ``cache_config``; warn on
        ``enable_caching=True`` and pass no cache config (the inverse of every other handler).
    """

    def get_inputs(self, b, inputs):  # -> {"model_data": ...}
        if "model_data" not in inputs:
            raise ValueError("Required input 'model_data' not found")
        return {"model_data": inputs["model_data"]}

    def get_outputs(self, b, outputs):  # -> None (CreateModelStep provides ModelName)
        return None

    def build_step(self, b, **kwargs):
        from sagemaker.workflow.steps import CreateModelStep

        inputs = self._merge_inputs(b, kwargs)
        # backward-compat: a direct model_data= kwarg overrides (builder :242-243)
        model_data = kwargs.get("model_data")
        if model_data is not None:
            inputs["model_data"] = model_data
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        step_name = b._get_step_name()
        model_inputs = (
            b._get_inputs(inputs)
            if _overrides(b, "_get_inputs")
            else self.get_inputs(b, inputs)
        )
        model_data_value = model_inputs["model_data"]

        make_compute = self.knobs.get("make_compute")
        if make_compute is None:
            # Resolution order (FZ 31e1d3k): a per-step _create_model override wins (genuine keep);
            # else the declarative contract.compute descriptor (kind: model) drives base
            # _create_compute(model_data=...); else error.
            if _overrides(b, "_create_model") or "_create_model" in vars(b):
                make_compute = lambda _b, md: _b._create_model(md)  # noqa: E731
            elif getattr(getattr(b, "contract", None), "compute", None) and getattr(
                b.contract.compute, "kind", None
            ):
                make_compute = lambda _b, md: _b._create_compute(model_data=md)  # noqa: E731
            else:
                raise NotImplementedError(
                    "ModelCreationHandler needs a `make_compute` knob, a builder _create_model(), "
                    "or a contract.compute descriptor."
                )
        model = make_compute(b, model_data_value)  # LAST, consumes model_data

        # CreateModelStep takes NO cache_config — drop caching, warn if requested.
        if enable_caching:
            b.log_warning(
                "CreateModelStep does not support caching - ignoring enable_caching=True"
            )
        try:
            step = CreateModelStep(
                name=step_name,
                model=model,  # model passed directly, NOT step_args
                depends_on=dependencies,
            )
        except Exception as e:
            b.log_warning("Error creating ModelStep: %s", str(e))
            raise ValueError(f"Failed to create ModelStep: {str(e)}") from e
        return self._attach_spec(b, step)


_TRANSFORM_KNOBS = (
    KnobSpec(
        "make_compute",
        "callable",
        doc="transformer factory; defaults to builder._create_transformer",
    ),
    KnobSpec(
        "direct_input_keys", "list", [], doc="template-provided direct input keys"
    ),
)


@register_strategy(
    axis="sagemaker_step_type",
    name="Transform",
    verb="Transform",
    knobs=_TRANSFORM_KNOBS,
)
class TransformHandler(PatternHandler):
    """Transform verb — builds a ``TransformStep(transformer, inputs=TransformInput)``.

    Distinctive (re-homed from ``builder_batch_transform_step.py``):
      * ``get_inputs`` returns a **2-tuple** ``(TransformInput, model_name)`` — spec-only (no
        contract), with hard-coded ``model_name``/``processed_data`` logical-name dispatch.
      * ``get_outputs`` returns a **single str/Join** (not a list).
      * ``make_compute`` (the Transformer factory) runs **LAST** — it consumes both ``model_name``
        (from inputs) and ``output_path`` (from outputs).
      * caching is guarded (``cache_config=... if enable_caching else None``).
    """

    #: Config fields the Transform build genuinely DEPENDS on. The Transform read-sites
    #: (builder_templates.py:691-695) also touch content_type/split_type/join_source/input_filter/
    #: output_filter, but those are all OPTIONAL on BatchTransformStepConfig (they carry defaults, so
    #: a missing value can't break the build), so they are NOT coverage requirements. Only ``job_type``
    #: is required (no default) and is read at :718 — its absence WOULD break the build.
    requires_config_fields = ("job_type",)

    def get_inputs(self, b, inputs):  # -> (TransformInput, model_name)
        from sagemaker.inputs import TransformInput

        if not b.spec:
            raise ValueError("Step specification is required")
        model_name = None
        input_data = None
        for _, dep in b.spec.dependencies.items():
            logical_name = dep.logical_name
            if not dep.required and logical_name not in inputs:
                continue
            if dep.required and logical_name not in inputs:
                raise ValueError(
                    f"Required input '{logical_name}' not provided. "
                    f"Expected from compatible sources: {dep.compatible_sources}"
                )
            if logical_name == "model_name":
                model_name = inputs[logical_name]
            elif logical_name == "processed_data":
                input_data = inputs[logical_name]
            else:
                b.log_warning(
                    "Unexpected logical name '%s' in specification dependencies",
                    logical_name,
                )
        if not model_name:
            raise ValueError(
                "model_name is required but not provided in inputs. "
                "Check that a model step (PytorchModel, XgboostModel) is properly connected."
            )
        if not input_data:
            raise ValueError(
                "processed_data is required but not provided in inputs. "
                "Check that a preprocessing step (TabularPreprocessing) is properly connected."
            )
        transform_input = TransformInput(
            data=input_data,
            content_type=b.config.content_type,
            split_type=b.config.split_type,
            join_source=b.config.join_source,
            input_filter=b.config.input_filter,
            output_filter=b.config.output_filter,
        )
        return transform_input, model_name

    def get_outputs(self, b, outputs):  # -> str / Join (single)
        from sagemaker.workflow.functions import Join

        from ...step_catalog.naming import canonical_to_snake

        if not b.spec:
            raise ValueError("Step specification is required")
        for _, output_spec in b.spec.outputs.items():
            logical_name = output_spec.logical_name
            if logical_name in outputs:
                return outputs[logical_name]
        base = b._get_base_output_path()
        # The output S3 prefix is DERIVED from the step name — canonical_to_snake(step_type), the
        # CONVENTION (FZ 31e1d3f1b). OPT-IN override: contract.output_path_token used verbatim if set.
        token = getattr(b.contract, "output_path_token", None) or (
            canonical_to_snake(b.spec.step_type)
            if hasattr(b.spec, "step_type")
            else "batch_transform"
        )
        return Join(on="/", values=[base, token, b.config.job_type])

    def build_step(self, b, **kwargs):
        from sagemaker.workflow.steps import TransformStep

        inputs = self._merge_inputs(b, kwargs)
        outputs = kwargs.get("outputs", {})
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        # ORDER (load-bearing): get_inputs -> get_outputs -> make_compute (LAST, consumes both).
        transform_input, model_name = (
            b._get_inputs(inputs)
            if _overrides(b, "_get_inputs")
            else self.get_inputs(b, inputs)
        )
        output_path = (
            b._get_outputs(outputs)
            if _overrides(b, "_get_outputs")
            else self.get_outputs(b, outputs)
        )
        make_compute = self.knobs.get("make_compute")
        if make_compute is None:
            # Resolution order (FZ 31e1d3k): a per-step _create_transformer override wins (genuine
            # keep); else the declarative contract.compute descriptor (kind: transformer) drives base
            # _create_compute(model_name=..., output_path=...); else error.
            if _overrides(b, "_create_transformer") or "_create_transformer" in vars(b):
                make_compute = lambda _b, mn, op: _b._create_transformer(mn, op)  # noqa: E731
            elif getattr(getattr(b, "contract", None), "compute", None) and getattr(
                b.contract.compute, "kind", None
            ):
                make_compute = lambda _b, mn, op: _b._create_compute(op, model_name=mn)  # noqa: E731
            else:
                raise NotImplementedError(
                    "TransformHandler needs a `make_compute` knob, a builder _create_transformer(), "
                    "or a contract.compute descriptor."
                )
        transformer = make_compute(b, model_name, output_path)
        step_name = b._get_step_name()
        step = TransformStep(
            name=step_name,
            transformer=transformer,
            inputs=transform_input,  # the singular TransformInput, NOT a list
            depends_on=dependencies or [],
            cache_config=(
                b._get_cache_config(enable_caching) if enable_caching else None
            ),
        )
        return self._attach_spec(b, step)


_SDK_DELEGATION_KNOBS = (
    KnobSpec(
        "sdk_step_class",
        "callable",
        required=True,
        doc="the SAIS SDK *Step class to instantiate",
    ),
    KnobSpec("input_mode", "str", "none", doc="none | resolve_s3 | mims_ordered"),
    KnobSpec(
        "input_logical_name",
        "str",
        "input_data",
        doc="resolve_s3: the dependency logical name",
    ),
    KnobSpec(
        "input_config_fallback_attr",
        "str",
        "input_s3_location",
        doc="resolve_s3: config fallback attr",
    ),
    KnobSpec(
        "depends_on_ctor",
        "bool",
        False,
        doc="True=depends_on= ctor kwarg; False=step.add_depends_on",
    ),
    KnobSpec("caching_mode", "str", "none", doc="none | force_off_attr"),
    KnobSpec(
        "outputs_return_none", "bool", False, doc="get_outputs returns None vs {}"
    ),
    KnobSpec(
        "log_output_paths",
        "bool",
        False,
        doc="log contract output paths (Cradle/Redshift)",
    ),
    KnobSpec(
        "append_region",
        "bool",
        False,
        doc="suffix step_name with '-<region>' (Registration)",
    ),
    KnobSpec(
        "region",
        "str",
        doc="region for the step-name suffix; defaults to config.region",
    ),
    KnobSpec(
        "pass_performance_metadata",
        "bool",
        False,
        doc="pass performance_metadata_location (Registration)",
    ),
)


@register_strategy(
    axis="sagemaker_step_type",
    name="CradleDataLoading",
    verb="SDKDelegation",
    knobs=_SDK_DELEGATION_KNOBS,
    preset_knobs={
        "input_mode": "none",
        "caching_mode": "force_off_attr",
        "log_output_paths": True,
    },
)
@register_strategy(
    axis="sagemaker_step_type",
    name="RedshiftDataLoading",
    verb="SDKDelegation",
    knobs=_SDK_DELEGATION_KNOBS,
    preset_knobs={
        "input_mode": "none",
        "caching_mode": "force_off_attr",
        "log_output_paths": True,
    },
)
@register_strategy(
    axis="sagemaker_step_type",
    name="MimsModelRegistrationProcessing",
    verb="SDKDelegation",
    knobs=_SDK_DELEGATION_KNOBS,
    preset_knobs={
        "input_mode": "mims_ordered",
        "depends_on_ctor": True,
        "outputs_return_none": True,
        "append_region": True,
        "pass_performance_metadata": True,
    },
)
@register_strategy(
    axis="step_assembly",
    name="delegation",
    verb="SDKDelegation",
    knobs=_SDK_DELEGATION_KNOBS,
    preset_knobs={"input_mode": "resolve_s3"},
)  # DataUploading
class SDKDelegationHandler(PatternHandler):
    """SDKDelegation verb — instantiate a SAIS SDK ``MODSPredefinedProcessingStep`` subclass directly.

    Covers Cradle / Redshift / DataUploading / Registration (re-homed from their builders). The
    SDK step builds its own processor/inputs internally, so there is no ``make_compute``. The SDK
    step *class* is injected via the ``sdk_step_class`` knob (the SAIS SDK can't be imported at
    registration time). Three ``input_mode``s: ``none`` ([] — Cradle/Redshift), ``resolve_s3``
    (DataUploading — resolve the input S3, pass as ``input_s3_location=``), ``mims_ordered``
    (Registration — an ordered ``ProcessingInput`` list, PackagedModel first). ``get_inputs``
    returns ``(processing_inputs, resolved_s3)``.
    """

    def get_inputs(self, b, inputs):  # -> (List[ProcessingInput], resolved_s3_or_None)
        mode = self.knobs.get("input_mode", "none")
        if mode == "none":
            return [], None
        if mode == "resolve_s3":
            dep = self.knobs.get("input_logical_name", "input_data")
            fallback_attr = self.knobs.get(
                "input_config_fallback_attr", "input_s3_location"
            )
            if dep in inputs:
                return [], inputs[dep]
            fallback = getattr(b.config, fallback_attr, None)
            if fallback:
                return [], fallback
            raise ValueError(
                f"Required input {dep!r} not provided and config.{fallback_attr} is not set."
            )
        if mode == "mims_ordered":
            from sagemaker.processing import ProcessingInput

            contract = getattr(b, "contract", None)
            paths = contract.expected_input_paths if contract else {}
            ordered = []
            model_logical = "PackagedModel"
            if model_logical not in inputs:
                raise ValueError(f"Required input '{model_logical}' not provided")
            ordered.append(
                ProcessingInput(
                    input_name=model_logical,
                    source=inputs[model_logical],
                    destination=paths.get(
                        model_logical, "/opt/ml/processing/input/model"
                    ),
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File",
                )
            )
            payload_logical = "GeneratedPayloadSamples"
            if payload_logical in inputs:
                ordered.append(
                    ProcessingInput(
                        input_name=payload_logical,
                        source=inputs[payload_logical],
                        destination=paths.get(
                            payload_logical, "/opt/ml/processing/mims_payload"
                        ),
                        s3_data_distribution_type="FullyReplicated",
                        s3_input_mode="File",
                    )
                )
            return ordered, None
        raise ValueError(f"unknown SDKDelegation input_mode {mode!r}")

    def get_outputs(self, b, outputs):  # -> None or {}
        if self.knobs.get("log_output_paths") and getattr(b, "contract", None):
            b.log_info(
                "Output paths (SDK-managed): %s", b.contract.expected_output_paths
            )
        return None if self.knobs.get("outputs_return_none") else {}

    def build_step(self, b, **kwargs):
        sdk_step_class = self.knobs.get("sdk_step_class")
        if sdk_step_class is None:
            raise NotImplementedError(
                "SDKDelegationHandler requires the `sdk_step_class` knob (the SAIS SDK *Step class)."
            )
        inputs = self._merge_inputs(b, kwargs)
        processing_inputs, resolved_s3 = (
            b._get_inputs(inputs)
            if _overrides(b, "_get_inputs")
            else self.get_inputs(b, inputs)
        )
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        step_name = b._get_step_name()
        if self.knobs.get("append_region"):
            region = self.knobs.get("region") or getattr(b.config, "region", None)
            step_name = f"{step_name}-{region}"

        ctor: Dict[str, Any] = {
            "step_name": step_name,
            "role": b.role,
            "sagemaker_session": b.session,
        }
        mode = self.knobs.get("input_mode", "none")
        if mode == "resolve_s3":
            ctor["input_s3_location"] = resolved_s3
        elif mode == "mims_ordered":
            ctor["processing_input"] = processing_inputs
            if self.knobs.get("pass_performance_metadata"):
                ctor["performance_metadata_location"] = kwargs.get(
                    "performance_metadata_location"
                )
        if self.knobs.get("depends_on_ctor"):
            ctor["depends_on"] = dependencies

        try:
            step = sdk_step_class(**ctor)
        except Exception as e:
            raise ValueError(f"Failed to create {step_name}: {e}") from e

        # caching: never via cache_config kwarg; force-off-attr conditionally disables.
        if self.knobs.get("caching_mode") == "force_off_attr":
            if not enable_caching and hasattr(step, "cache_config"):
                step.cache_config.enable_caching = False

        if not self.knobs.get("depends_on_ctor") and dependencies:
            step.add_depends_on(dependencies)

        return self._attach_spec(b, step)


# ---------------------------------------------------------------------------
# Routing — delegates to the strategy_registry (the single source of truth).
# The handler classes above self-register via @register_strategy decorations (below the class
# defs); Base/Lambda register as non-routable rows. resolve_handler maps the runtime
# (sagemaker_step_type, step_assembly) onto a registry (axis, name) lookup.
# ---------------------------------------------------------------------------

# Non-routable types (abstract base / builder-less) — registered so the registry is exhaustive.
register_no_builder(axis="sagemaker_step_type", name="Base", verb="Base")
register_no_builder(axis="sagemaker_step_type", name="Lambda", verb="Lambda")


def resolve_handler(
    sagemaker_step_type: str,
    step_assembly: Optional[str] = None,
    knobs: Optional[Dict[str, Any]] = None,
) -> PatternHandler:
    """Select and instantiate the construction-verb handler for a step.

    Routing is by ``sagemaker_step_type`` ONLY (never by step name — ``DummyTraining`` is
    ``Processing`` and must route as Processing). ``Processing`` is sub-discriminated by
    ``step_assembly`` (``code`` | ``step_args`` | ``delegation``, default ``code``). All routing
    data lives in the ``strategy_registry``; this function only maps the runtime args onto an
    (axis, name) lookup and merges the registry's preset knobs under the caller's knobs.
    """
    extra_knobs = knobs or {}
    axis, name = axis_name_for_step_type(sagemaker_step_type, step_assembly)
    info = resolve_strategy(axis, name)  # raises NoBuilderError for Base/Lambda/unknown
    merged = {**info.preset_knobs, **extra_knobs}  # preset UNDER caller, as before
    return info.handler(knobs=merged)


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class TemplateStepBuilder(StepBuilderBase):
    """Routed builder facade — the single concrete parent of the (future) 2-line shells.

    A shell declares only its registry key::

        class TabularPreprocessingStepBuilder(TemplateStepBuilder):
            STEP_NAME = "TabularPreprocessing"

    The facade keeps the exact 5-kwarg ``__init__`` contract the ``PipelineAssembler`` calls
    (``builder_base.__init__``), binds a handler via ``resolve_handler`` from the step's
    ``sagemaker_step_type`` (+ ``step_assembly``), and implements the abstract methods
    (``_get_inputs``/``_get_outputs``/``create_step``) by delegating to that handler. It is a
    ``StepBuilderBase`` subclass, so the assembler/catalog contract and the discovery hierarchy hold.

    Wired and live (Phase S3): ``TabularPreprocessingStepBuilder`` and ``BatchTransformStepBuilder``
    are real shells routing through this facade; the rest migrate per Phase S3 (byte-diff-gated).
    """

    #: Subclasses set this to their canonical registry step name (drives spec load + routing).
    #: The slot itself is declared on StepBuilderBase (the root that reads it in _get_step_name,
    #: FZ 31e1d3g3 Phase C1); re-stated here for local readability of the shell contract.
    STEP_NAME: Optional[str] = None
    #: Optional explicit step_assembly for Processing steps (code | step_args | delegation).
    #: If None, defaults to "code" for Processing (see resolve_handler).
    STEP_ASSEMBLY: Optional[str] = None
    #: Per-step handler knobs (direct_input_keys, split_source_dir, include_job_type_in_path, ...).
    HANDLER_KNOBS: Dict[str, Any] = {}

    def __init__(
        self,
        config: Any,
        sagemaker_session: Any = None,
        role: Optional[str] = None,
        registry_manager: Any = None,
        dependency_resolver: Any = None,
        spec: Any = None,
    ):
        # A shell that declares STEP_NAME loads its own spec from the unified YAML interface
        # (the same load_step_interface the hand-written builders use), unless a spec is passed.
        # job_type is a config field (present on the ~36 variant-bearing configs); pass it through
        # so a variant-bearing step resolves its job-typed spec — the variant carries a distinct
        # step_type (e.g. RiskTableMapping_Validation), required-flags, and compatible_sources that
        # the connection graph wires on. Mirrors the legacy variant builders
        # (builder_risk_table_mapping_step.py:60-61). job_type=None falls back to the base spec.
        if spec is None and self.STEP_NAME is not None:
            from ...steps.interfaces import load_step_interface

            _contract, spec = load_step_interface(
                self.STEP_NAME, job_type=getattr(config, "job_type", None)
            )
        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self._handler: Optional[PatternHandler] = None
        # Auto-bind the construction handler from the registry's sagemaker_step_type for this
        # step + the declared knobs. A shell needs no _bind_handler() call.
        if self.STEP_NAME is not None:
            self._auto_bind_handler()

    def _auto_bind_handler(self) -> None:
        """Bind the handler from the registry's sagemaker_step_type for STEP_NAME + knobs.

        The per-axis STRATEGY-SELECTION facts come from the ``.step.yaml`` ``patterns:`` section
        (FZ 31e1d3f1) — the interface is the blueprint that wires pattern injection, so editing the
        YAML steers the build with no Python change. The class attrs ``STEP_ASSEMBLY`` /
        ``HANDLER_KNOBS`` are a BACK-COMPAT FALLBACK (used only when the interface doesn't declare the
        field) — the 4 SDKDelegation builders still carry a code-only ``sdk_step_class`` knob, and any
        un-migrated step keeps working. Interface wins; class attr fills the gap.
        """
        from ...registry.step_names import get_sagemaker_step_type

        sm_type = get_sagemaker_step_type(self.STEP_NAME)
        patterns = getattr(self.spec, "patterns", None)

        # step_assembly: interface patterns.step_assembly, else the class-attr fallback.
        step_assembly = getattr(patterns, "step_assembly", None) or self.STEP_ASSEMBLY

        # knobs: the interface's declarative knobs (include_job_type_in_path /
        # direct_input_keys) UNDER the class-attr HANDLER_KNOBS — so a code-only knob (sdk_step_class
        # on the SDK builders, or any un-migrated leftover) still applies, but the YAML wins for the
        # migrated declarative axes (its values overwrite, since it's spread last).
        knobs = dict(self.HANDLER_KNOBS)
        if patterns is not None:
            knobs.update(patterns.as_knobs())

        self._bind_handler(
            sagemaker_step_type=sm_type,
            step_assembly=step_assembly,
            knobs=knobs,
        )

    def _bind_handler(
        self,
        sagemaker_step_type: str,
        step_assembly: Optional[str] = None,
        knobs: Optional[Dict[str, Any]] = None,
    ) -> PatternHandler:
        """Bind (and cache) the construction handler. Called by the wiring layer / tests."""
        self._handler = resolve_handler(sagemaker_step_type, step_assembly, knobs)
        return self._handler

    # --- abstract-method delegation ---

    def validate_configuration(
        self,
    ) -> None:  # pragma: no cover - per-step override expected
        return None

    def _get_inputs(self, inputs: Dict[str, Any]) -> Any:
        if self._handler is None:
            raise RuntimeError("handler not bound; call _bind_handler() first")
        return self._handler.get_inputs(self, inputs)

    def _get_outputs(self, outputs: Dict[str, Any]) -> Any:
        if self._handler is None:
            raise RuntimeError("handler not bound; call _bind_handler() first")
        return self._handler.get_outputs(self, outputs)

    def create_step(self, **kwargs: Any) -> Step:
        if self._handler is None:
            raise RuntimeError(
                "handler not bound; a routed shell must declare STEP_NAME (so __init__ auto-binds "
                f"its handler) or call _bind_handler() explicitly. STEP_NAME={self.STEP_NAME!r}"
            )
        step = self._handler.build_step(self, **kwargs)
        # Guarantee spec/contract are re-homed onto the Step regardless of whether the handler
        # remembered to call _attach_spec. step._spec is the sole input to the builder-driven
        # resolver-enrichment path (builder_base.py:929-930) and is read by downstream
        # introspection; the assembler's primary path reads builder.spec, so a handler that forgot
        # _attach_spec would break Path B silently with no Path-A symptom. Hoisting it here makes it
        # non-bypassable. _attach_spec is idempotent (plain setattr), so the handlers' own trailing
        # calls remain harmless (and keep the direct-handler unit tests valid). See FZ 31e1d3d2.
        return PatternHandler._attach_spec(self, step)
