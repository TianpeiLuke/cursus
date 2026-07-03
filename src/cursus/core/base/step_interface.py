"""
StepInterface — the single Pydantic-validated data structure for a .step.yaml.

Combines what was previously ScriptContract + StepSpecification into one validated
model. This is the single message passed between dep resolver, builder, and assembler.

It is intentionally a *superset* of the legacy data classes so it can stand in for
all of them during migration:

- ``StepInterface.contract`` (ContractSection) is a drop-in for ``ScriptContract`` /
  ``StepContract``: it exposes ``entry_point``, ``expected_input_paths``,
  ``expected_output_paths``, ``expected_arguments``, ``required_env_vars``,
  ``optional_env_vars``, ``framework_requirements`` and ``description``.
- ``StepInterface`` / ``StepInterface.spec`` (SpecSection) are a drop-in for
  ``StepSpecification``: ``step_type``, ``node_type``, ``dependencies``, ``outputs``,
  ``get_dependency()``, ``get_output()``, ``get_output_by_name_or_alias()``,
  ``list_required_dependencies()``, ``list_optional_dependencies()``,
  ``list_all_output_names()``, ``validate_specification()``,
  ``validate_contract_alignment()`` and ``script_contract``.
- Each ``DependencyDecl`` / ``OutputDecl`` is a drop-in for ``DependencySpec`` /
  ``OutputSpec``: it carries ``logical_name`` (auto-populated from the dict key),
  exposes ``dependency_type`` / ``output_type`` (aliases of ``type``), and supports
  ``matches_name_or_alias()``.

Validation rules:
- Contract inputs and spec dependencies must have matching keys.
- Contract outputs and spec outputs must have matching keys.
- Paths (when present) must be valid SageMaker paths (processing OR training).
- entry_point (when present) must be a .py file.

``entry_point`` and the port ``path`` fields are Optional: script-less SageMaker
steps (CreateModel / Transform — e.g. xgboost_model, pytorch_model, batch_transform)
declare them as ``null`` in YAML.
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, field_validator, model_validator

# Shared enums are the single source of truth (they include SINGULAR and the
# value-based __eq__/__hash__ the resolver/registry rely on).
from .enums import DependencyType, NodeType

if TYPE_CHECKING:
    from .contract_base import ValidationResult


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge ``override`` onto ``base``, returning a new dict.

    Used to apply a ``.step.yaml`` job-type variant over the base sections. Nested
    dicts are merged key-by-key (so a variant restating only some ``dependencies`` /
    ``outputs`` / ``inputs`` overrides just those entries' fields and preserves the
    rest of the base set); any non-dict value in ``override`` replaces the base value
    outright. Neither input is mutated.
    """
    result = dict(base)
    for key, ov in override.items():
        bv = result.get(key)
        if isinstance(bv, dict) and isinstance(ov, dict):
            result[key] = _deep_merge(bv, ov)
        else:
            result[key] = ov
    return result


# --- Valid SageMaker path prefixes (unified processing + training conventions) ---

VALID_INPUT_PREFIXES = (
    "/opt/ml/processing/",
    "/opt/ml/input/data",
    "/opt/ml/input/config",
    "/opt/ml/code",
)

VALID_OUTPUT_PREFIXES = (
    "/opt/ml/processing/",
    "/opt/ml/model",
    "/opt/ml/output/data",
    "/opt/ml/checkpoints",
)

_NODE_TYPE_BY_VALUE = {nt.value: nt for nt in NodeType}
_DEP_TYPE_BY_VALUE = {dt.value: dt for dt in DependencyType}


# --- Sub-models ---


class InputPort(BaseModel):
    """One contract input declaration."""

    path: Optional[str] = None
    required: bool = True
    #: Optional SageMaker training sub-channels this single input fans out into (e.g.
    #: ``[train, val, test]``). When set, the TrainingHandler creates one ``TrainingInput`` per
    #: sub-channel under ``<path>/<channel>/`` instead of a single channel. This is per-step DATA
    #: (the channel layout the script expects), so it lives in the ``.step.yaml`` — not hardcoded
    #: in the handler. Empty/None ⇒ the input maps to a single channel (see TrainingHandler).
    channels: List[str] = Field(default_factory=list)

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Optional[str]) -> Optional[str]:
        # Script-less steps (CreateModel/Transform) legitimately have null paths.
        if v is None:
            return v
        if not any(v.startswith(p) for p in VALID_INPUT_PREFIXES):
            raise ValueError(
                f"Input path must start with one of {VALID_INPUT_PREFIXES}, got: {v}"
            )
        return v


class OutputPort(BaseModel):
    """One contract output declaration."""

    path: Optional[str] = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not any(v.startswith(p) for p in VALID_OUTPUT_PREFIXES):
            raise ValueError(
                f"Output path must start with one of {VALID_OUTPUT_PREFIXES}, got: {v}"
            )
        return v


class EnvVars(BaseModel):
    """Environment variable declarations."""

    required: List[str] = Field(default_factory=list)
    optional: Dict[str, str] = Field(default_factory=dict)


class ComputeSpec(BaseModel):
    """Declarative spec for the step's COMPUTE object — the processor/estimator/model/transformer
    the builder constructs (FZ 31e1d3k).

    Every VALUE is a config field, so the compute object is fully constructable from config + this
    descriptor — which says WHICH SDK class and WHICH config fields. Lives in the ``.step.yaml`` so
    it is surfaced to users (the ``steps patterns`` view) and lets the builder template build the
    compute generically, replacing the near-identical per-step ``_create_processor`` /
    ``_create_estimator`` factories. Empty (``kind=None``) ⇒ the step keeps its own factory.
    """

    #: sklearn | xgboost | framework | script (processors) · estimator · model · transformer · None.
    kind: Optional[str] = None
    #: The config attr holding the framework version (e.g. ``processing_framework_version``).
    framework_version_field: Optional[str] = None
    #: Default framework version when the field is absent on the config — several steps used
    #: ``getattr(config, "processing_framework_version", "<default>")`` (defaults vary per step,
    #: e.g. ``1.0-1`` / ``1.2-1``). ``None`` ⇒ the field must be present.
    framework_version_default: Optional[str] = None
    #: The config attr holding the py version (framework processors / estimators).
    py_version_field: Optional[str] = None
    #: For ``framework`` processors / estimators / models: the SDK class NAME to use as the
    #: estimator_cls / model class (e.g. ``PyTorch``, ``SKLearn``, ``XGBoost``, ``PyTorchModel``).
    sdk_class: Optional[str] = None
    #: How the processing instance type is chosen: ``large_or_small`` (the
    #: ``use_large_processing_instance`` ternary) or ``fixed`` (a single field).
    instance_size_mode: str = "large_or_small"
    #: ``script`` kind only (EdxUploading): set the KMS volume key + shared network config + the
    #: ECR-from-role image. A genuinely special, declared-once deviation.
    kms_network: bool = False
    #: ``estimator`` kind only: explicitly retrieve the training image_uri (PyTorch-for-LightGBM).
    retrieve_image: bool = False
    #: ``model`` kind: the framework NAME passed to ``image_uris.retrieve`` for the INFERENCE image
    #: (``xgboost`` / ``pytorch``). Distinct from ``sdk_class`` (the model CLASS, e.g. ``XGBoostModel``):
    #: the class instantiates the model, this names the container image to retrieve.
    framework_name: Optional[str] = None
    #: ``estimator`` image-retrieval region locking. SAIS RESTRICTION: training images/jobs are
    #: forced to a fixed region (``us-east-1``) — an explicit platform constraint, NOT a bug. This is
    #: a TOGGLEABLE pattern: a step opts into locking (``lock_training_region: true``); to run in
    #: standard (unlocked) mode it sets ``lock_training_region: false`` here (or via config) — no
    #: code change. When False, the region comes from ``config.aws_region`` (the normal region).
    lock_training_region: bool = False
    #: The region used when ``lock_training_region`` is True (the SAIS-locked region).
    locked_region: str = "us-east-1"
    #: DEPENDENCY AXIS — the 3rd-party package this COMPUTE pattern needs at BUILD time (FZ 31e1d3l).
    #: ``none`` for the sagemaker-only kinds (sklearn/xgboost/framework/estimator/model/transformer);
    #: ``mods_workflow_core`` ONLY for the ``script`` kind with ``kms_network`` (the EdxUploading
    #: ScriptProcessor, which lazily imports ``KMS_ENCRYPTION_KEY_PARAM`` /
    #: ``PROCESSING_JOB_SHARED_NETWORK_CONFIG`` in ``builder_base._create_compute``). This is a
    #: CONSEQUENCE of ``kms_network`` — the validator keeps it consistent so it can't drift, and it is
    #: declared in the ``.step.yaml`` so the mods-vs-native split is visible (``steps patterns``).
    requires: str = "none"

    # Valid values, pinned to the SageMaker SDK surface (sagemaker 2.251.x). ``kms_network`` maps to
    # ScriptProcessor's volume_kms_key + network_config; ``framework``/``estimator``/``model`` need an
    # sdk_class; processors take a framework_version_field.
    _KINDS: ClassVar = (
        "sklearn",
        "xgboost",
        "framework",
        "script",
        "estimator",
        "model",
        "transformer",
    )
    _SDK_CLASSES: ClassVar = (
        "PyTorch",
        "SKLearn",
        "XGBoost",
        "PyTorchModel",
        "XGBoostModel",
    )
    #: The only 3rd-party package a compute pattern can require (the script/kms_network path).
    _REQUIRES: ClassVar = ("none", "mods_workflow_core")

    @model_validator(mode="after")
    def _validate_compute(self) -> "ComputeSpec":
        if self.kind is None:
            # empty descriptor — step keeps its own factory. A bare descriptor must not claim a dep.
            if self.requires != "none":
                raise ValueError("compute.requires must be 'none' when kind is unset")
            return self
        if self.kind not in self._KINDS:
            raise ValueError(f"compute.kind {self.kind!r} not in {self._KINDS}")
        if self.sdk_class is not None and self.sdk_class not in self._SDK_CLASSES:
            raise ValueError(
                f"compute.sdk_class {self.sdk_class!r} not in {self._SDK_CLASSES}"
            )
        # framework processors + estimators + models must name an sdk_class; the others must NOT.
        if self.kind in ("framework", "estimator", "model"):
            if not self.sdk_class:
                raise ValueError(f"compute.kind={self.kind!r} requires sdk_class")
        elif self.sdk_class:
            raise ValueError(f"compute.sdk_class invalid for kind={self.kind!r}")
        # processors + estimators + models take a framework version field
        if (
            self.kind in ("sklearn", "xgboost", "framework", "estimator", "model")
            and not self.framework_version_field
        ):
            raise ValueError(
                f"compute.kind={self.kind!r} requires framework_version_field"
            )
        # py_version only meaningful for framework processors + estimators + models
        if self.py_version_field and self.kind not in (
            "framework",
            "estimator",
            "model",
        ):
            raise ValueError(f"compute.py_version_field invalid for kind={self.kind!r}")
        # framework_name (the inference-image framework) is a model-only knob
        if self.framework_name and self.kind != "model":
            raise ValueError("compute.framework_name is only valid for kind='model'")
        if self.kind == "model" and not self.framework_name:
            raise ValueError(
                "compute.kind='model' requires framework_name (the inference-image framework)"
            )
        # kms_network is a ScriptProcessor-only knob
        if self.kms_network and self.kind != "script":
            raise ValueError("compute.kms_network is only valid for kind='script'")
        if self.instance_size_mode not in ("large_or_small", "fixed"):
            raise ValueError(
                f"compute.instance_size_mode {self.instance_size_mode!r} invalid"
            )
        # --- dependency axis: requires is a CONSEQUENCE of kms_network, kept consistent so it can't
        # drift from the actual lazy mods_workflow_core import in builder_base._create_compute. ---
        if self.requires not in self._REQUIRES:
            raise ValueError(
                f"compute.requires {self.requires!r} not in {self._REQUIRES}"
            )
        derived = "mods_workflow_core" if self.kms_network else "none"
        if self.requires != "none" and self.requires != derived:
            raise ValueError(
                f"compute.requires={self.requires!r} inconsistent with kms_network={self.kms_network} "
                f"(expected {derived!r})"
            )
        # auto-derive when omitted so the dep is always correct even if the .step.yaml leaves it blank
        self.requires = derived
        return self


class JobArgDecl(BaseModel):
    """DECLARATIVE record of one CLI argument the step's script accepts (FZ 31e1d3h).

    Documentation / alignment / introspection only — the TRUE argument list is built at runtime by
    ``config.get_job_arguments()`` (config is the single source). This just makes the script's
    argument surface visible in the ``.step.yaml`` (the analog of ``env_vars`` declaring names).
    """

    #: The ``--flag`` the script reads (e.g. ``--job_type``, ``--batch-size``).
    flag: str
    #: The config attribute the value comes from (e.g. ``job_type``). Empty for a bare boolean flag.
    source: str = ""


class RegistrySection(BaseModel):
    """The 'registry' section of a .step.yaml — the construction binding + its 3rd-party footprint.

    Previously this YAML block was silently dropped (StepInterface had no field for it), so the
    ``step_assembly`` and the create-step dependency had no declaration home. It is now a real
    section: ``sagemaker_step_type`` + ``step_assembly`` select the PatternHandler, and ``requires``
    declares the create_step axis's BUILD-time 3rd-party dependency.
    """

    #: The SageMaker verb that selects the PatternHandler (Processing / Training / CreateModel /
    #: Transform / the SAIS verbs). Mirrors the registry's ``sagemaker_step_type``.
    sagemaker_step_type: Optional[str] = None
    #: DEPRECATED — moved to ``patterns.step_assembly`` (FZ 31e1d3f1). Kept only as a back-compat
    #: read for any not-yet-migrated YAML; ``_auto_bind_handler`` + ``io_view`` prefer
    #: ``patterns.step_assembly``. No .step.yaml in this package declares it here anymore.
    step_assembly: Optional[str] = None
    #: DEPENDENCY AXIS — the 3rd-party package the CREATE_STEP pattern needs at BUILD time (FZ 31e1d3l).
    #: ``none`` for the native (sagemaker-only) handlers; ``secure_ai_sandbox_workflow_python_sdk`` for
    #: the SDKDelegation steps whose builder module imports a SAIS Step class at module level
    #: (Registration / CradleDataLoading / DataUploading / RedshiftDataLoading — fatal-on-load if the
    #: SDK is absent). Declared here so the mods/SAIS-vs-native split is authored data in the
    #: ``.step.yaml`` and visible in ``steps patterns``; a conformance gate keeps it equal to the
    #: builders' actual module-level SAIS imports.
    requires: str = "none"
    description: str = ""

    _REQUIRES: ClassVar = ("none", "secure_ai_sandbox_workflow_python_sdk")
    #: The closed set of SageMaker verbs ``sagemaker_step_type`` may take — the routing key that
    #: selects the PatternHandler at build time (``axis_name_for_step_type`` /
    #: ``resolve_handler``). Pinned here so a typo (e.g. ``"Procesing"``) or a wrong value is
    #: caught at AUTHOR time by ``StepInterface.from_yaml`` (hence by ``validate.step_interface`` /
    #: the CLI / CI) instead of silently mis-routing — or failing to synthesize a builder — later.
    #: Kept equal to the registry's ``get_valid_sagemaker_step_types()`` by a conformance test so
    #: this pin can never drift from the live valid set. The five buildable verbs
    #: (Processing / Training / Transform / CreateModel + the SAIS-delegation verbs
    #: CradleDataLoading / RedshiftDataLoading / MimsModelRegistrationProcessing) plus the
    #: no-builder rows (Base / Lambda / RegisterModel / Utility) that exist in the registry.
    _SAGEMAKER_STEP_TYPES: ClassVar = (
        "Base",
        "CradleDataLoading",
        "CreateModel",
        "Lambda",
        "MimsModelRegistrationProcessing",
        "Processing",
        "RedshiftDataLoading",
        "RegisterModel",
        "Training",
        "Transform",
        "Utility",
    )

    @model_validator(mode="after")
    def _validate_registry(self) -> "RegistrySection":
        if self.requires not in self._REQUIRES:
            raise ValueError(
                f"registry.requires {self.requires!r} not in {self._REQUIRES}"
            )
        if (
            self.sagemaker_step_type is not None
            and self.sagemaker_step_type not in self._SAGEMAKER_STEP_TYPES
        ):
            raise ValueError(
                f"registry.sagemaker_step_type {self.sagemaker_step_type!r} not in "
                f"{self._SAGEMAKER_STEP_TYPES}"
            )
        return self


class PatternsSection(BaseModel):
    """The 'patterns' section of a .step.yaml — the per-axis STRATEGY-SELECTION knobs (FZ 31e1d3f1).

    This is the BLUEPRINT that guides how the handlers combine/inject behavior per axis, so a step's
    implementation is NOT hard-wired in its builder shell. Distinct from ``contract`` (script-shaped
    I/O data), ``compute`` (the SDK compute object), ``registry`` (discovery + 3rd-party deps), and
    ``spec`` (DAG wiring). ``_auto_bind_handler`` reads these into the bound handler's knobs so
    editing the YAML steers the build with no Python change.

    ``use_step_args`` is intentionally NOT a field here: it is DERIVED from ``step_assembly`` (the
    ``step_args`` strategy preset sets ``use_step_args: True``, ``code`` sets False) — so it can never
    disagree with the routing verb.
    """

    #: Processing sub-verb that joins ``registry.sagemaker_step_type`` to pick the handler:
    #: ``code`` (2A, ``ProcessingStep(code=...)``) | ``step_args`` (2B, ``processor.run()``) |
    #: ``delegation`` (SDKDelegation). ``None`` ⇒ the handler's default (``code`` for Processing).
    step_assembly: Optional[str] = None
    #: NOTE: ``output_path_token`` was REMOVED (FZ 31e1d3f1b). The output-destination S3 prefix is
    #: DERIVED from the step name — ``canonical_to_snake(step_type)`` — not a declarable field: it
    #: corresponds to the step name by convention, and the historical deviations were non-standard.
    #: Whether ``config.job_type`` is a segment of the synthesized output destination.
    include_job_type_in_path: bool = True
    #: Logical input names passed straight through to the processor (not spec×contract joined) —
    #: the template-provided direct input allowlist.
    direct_input_keys: List[str] = Field(default_factory=list)

    _ASSEMBLIES: ClassVar = ("code", "step_args", "delegation")

    @model_validator(mode="after")
    def _validate_patterns(self) -> "PatternsSection":
        if (
            self.step_assembly is not None
            and self.step_assembly not in self._ASSEMBLIES
        ):
            raise ValueError(
                f"patterns.step_assembly {self.step_assembly!r} not in {self._ASSEMBLIES}"
            )
        return self

    def as_knobs(self) -> Dict[str, object]:
        """The HANDLER_KNOBS the bound handler reads — only NON-DEFAULT entries, so an unset field
        falls through to the strategy preset/contract default exactly as the old class-attr knobs did.
        ``step_assembly`` is routing (passed separately to ``resolve_handler``), not a knob."""
        knobs: Dict[str, object] = {}
        if not self.include_job_type_in_path:  # default True; only emit when deviating
            knobs["include_job_type_in_path"] = False
        if self.direct_input_keys:
            knobs["direct_input_keys"] = list(self.direct_input_keys)
        return knobs


class ContractSection(BaseModel):
    """
    The 'contract' section of a .step.yaml — script execution requirements.

    Drop-in for the legacy ScriptContract / StepContract: the ``expected_*`` /
    ``required_env_vars`` / ``optional_env_vars`` accessors flatten the structured
    ports back to the ``Dict[str, str]`` / ``List[str]`` shapes consumers expect.
    """

    entry_point: Optional[str] = None
    #: Whether this step's script needs its whole directory uploaded (sibling modules) — i.e.
    #: ``processor.run(code=entry_point, source_dir=<dir>)`` — vs a self-contained single script
    #: ``processor.run(code=<full_script_path>)``. This is per-step DATA (a script-packaging fact),
    #: so it lives in the ``.step.yaml`` rather than being inferred from the processor class. The
    #: ProcessingHandler reads it as the ``split_source_dir`` switch. Default False (self-contained).
    #: NOTE: a True value requires a FrameworkProcessor (``ScriptProcessor.run`` has no ``source_dir``).
    source_dir: bool = False
    #: OPT-IN override for the output-destination S3 prefix segment. Default ``None`` ⇒ the segment is
    #: DERIVED from the step name (``canonical_to_snake(step_type)``), the convention for ~all steps.
    #: When set to a non-empty string it is used VERBATIM as that segment instead of the derived token
    #: — needed when an EXTERNAL consumer keys off a fixed S3 folder name that does not match the
    #: cursus step name (e.g. PIPER scans ``<pipeline>/Model_Metric_Generation_Step/`` for ``.metric``
    #: files). This re-introduces the field removed in FZ 31e1d3f1b, but as an explicit escape hatch
    #: (default-off) rather than a routinely-set knob — the derived convention still holds by default.
    output_path_token: Optional[str] = None
    #: Whether ``config.job_type`` is a segment of the synthesized output destination. The other
    #: ``_get_outputs`` axis: some steps put job_type in the path, some don't. Default True. The
    #: ProcessingHandler reads this as the ``include_job_type_in_path`` knob.
    include_job_type_in_path: bool = True
    inputs: Dict[str, InputPort] = Field(default_factory=dict)
    outputs: Dict[str, OutputPort] = Field(default_factory=dict)
    #: Per-step input-resolution deviations from the standard spec×contract loop (FZ 31e1d3i),
    #: read by ProcessingHandler.get_inputs so the step needs no _get_inputs override:
    #:   circular_ref_check — run the PipelineVariable circular-reference guard before mapping.
    #:   skip_inputs — declared dependencies the script loads internally (not mounted as inputs).
    #:   input_source_overrides {logical_name: config_attr} — take the input SOURCE from a config
    #:     attr/method (config is the value source) instead of the resolved dependency value.
    circular_ref_check: bool = False
    skip_inputs: List[str] = Field(default_factory=list)
    input_source_overrides: Dict[str, str] = Field(default_factory=dict)
    #: A SINK step produces no outputs — ProcessingHandler.get_outputs returns ``[]`` (FZ 31e1d3i),
    #: so a sink step (e.g. an uploader) needs no _get_outputs override.
    sink: bool = False
    #: BACK-COMPAT MIRROR of the top-level ``StepInterface.compute`` (FZ 31e1d3k). The compute
    #: descriptor was promoted to a top-level ``.step.yaml`` section (peer of ``contract``/``spec``)
    #: because it describes the BUILDER's compute object, not the script contract — script-less steps
    #: (CreateModel/Transform) have a near-empty contract but a full compute. This field is kept and
    #: kept in sync by ``StepInterface._sync_and_align`` so existing ``b.contract.compute`` read sites
    #: still work; authors declare ``compute:`` at the top level now.
    compute: "ComputeSpec" = Field(default_factory=lambda: ComputeSpec())
    arguments: Dict[str, str] = Field(default_factory=dict)
    #: DECLARATIVE record of the CLI arguments the step's script accepts (FZ 31e1d3h) — each entry
    #: is ``{flag, source}`` (the ``--flag`` emitted and the config attribute it comes from). This is
    #: documentation / alignment / introspection ONLY: the TRUE values are produced at build time by
    #: ``config.get_job_arguments()`` (config is the single source). Mirrors how ``env_vars`` declares
    #: names while the config supplies values. Not used to drive ``_get_job_arguments``.
    job_arguments: List["JobArgDecl"] = Field(default_factory=list)
    env_vars: EnvVars = Field(default_factory=EnvVars)
    #: COMPUTED-S3-ENV pattern (FZ 31e1d3g3 Phase A3): env vars whose VALUE is an S3 sub-path under the
    #: pipeline's execution prefix (``base_output_path``), not a config field — e.g. a script that
    #: reads/writes an extra staging location. Maps ``ENV_VAR -> [segment, ...]``; the base
    #: ``_get_environment_variables`` sets ``ENV_VAR = Join(base_output_path, *segments)``. This is the
    #: declarative form of the formerly-hand-written ``_get_environment_variables`` overrides (e.g.
    #: BedrockBatchProcessing's BEDROCK_BATCH_INPUT/OUTPUT_S3_PATH) — the env analog of the
    #: output-destination token, so a step needs no Python to compute a runtime S3 env path.
    computed_env_paths: Dict[str, List[str]] = Field(default_factory=dict)
    framework_requirements: Dict[str, str] = Field(default_factory=dict)
    #: DEPENDENCY AXIS (runtime) — 3rd-party packages the step's SCRIPT imports at CONTAINER runtime
    #: (FZ 31e1d3l). This is ORTHOGONAL to build-time deps (``compute.requires`` / ``registry.requires``):
    #: these imports live in ``steps/scripts/<entry_point>`` and execute inside the SAIS Docker image,
    #: NOT during pipeline construction — they never affect offline import of cursus/builders. Kept on a
    #: separate descriptor so build-time vs runtime deps are never conflated. E.g. EdxUploading +
    #: RedshiftDataLoading scripts import ``secure_ai_sandbox_python_lib`` (a runtime, not build, dep).
    runtime_requires: List[str] = Field(default_factory=list)
    description: str = ""

    @field_validator("entry_point")
    @classmethod
    def validate_entry_point(cls, v: Optional[str]) -> Optional[str]:
        # Script-less SageMaker steps (CreateModel/Transform) have no entry_point.
        if v is None:
            return v
        if not v.endswith(".py"):
            raise ValueError(f"entry_point must be a .py file, got: {v}")
        return v

    # --- ScriptContract drop-in accessors ---

    @property
    def expected_input_paths(self) -> Dict[str, str]:
        return {
            name: port.path
            for name, port in self.inputs.items()
            if port.path is not None
        }

    @property
    def expected_output_paths(self) -> Dict[str, str]:
        return {
            name: port.path
            for name, port in self.outputs.items()
            if port.path is not None
        }

    @property
    def input_channels(self) -> Dict[str, List[str]]:
        """Per-input declared training sub-channels (``logical_name -> [channel, ...]``).

        Only inputs that declare a non-empty ``channels`` list appear. The TrainingHandler reads
        this to fan a single input into ``<path>/<channel>/`` sub-channels — the channel layout is
        per-step DATA in the ``.step.yaml``, not a handler constant.
        """
        return {
            name: list(port.channels)
            for name, port in self.inputs.items()
            if port.channels
        }

    @property
    def expected_arguments(self) -> Dict[str, str]:
        return self.arguments

    @property
    def required_env_vars(self) -> List[str]:
        return self.env_vars.required

    @property
    def optional_env_vars(self) -> Dict[str, str]:
        return self.env_vars.optional


class DependencyDecl(BaseModel):
    """
    One spec dependency declaration. Drop-in for the legacy DependencySpec.

    ``logical_name`` is auto-populated from the dict key by SpecSection's validator.
    ``dependency_type`` is exposed as an alias of ``type`` for the resolver/assembler.
    """

    logical_name: str = ""
    type: DependencyType = DependencyType.PROCESSING_OUTPUT
    required: bool = True
    compatible_sources: List[str] = Field(default_factory=list)
    semantic_keywords: List[str] = Field(default_factory=list)
    data_type: str = "S3Uri"
    description: str = ""

    @field_validator("type", mode="before")
    @classmethod
    def coerce_type(cls, v: object) -> object:
        # Accept the YAML string (e.g. "training_data") and map to the shared enum.
        if isinstance(v, str):
            return _DEP_TYPE_BY_VALUE.get(v, v)
        return v

    @property
    def dependency_type(self) -> DependencyType:
        """Legacy DependencySpec field name."""
        return self.type

    def matches_name_or_alias(self, name: str) -> bool:
        """Dependencies have no aliases; matches only the logical name."""
        return name == self.logical_name


class OutputDecl(BaseModel):
    """
    One spec output declaration. Drop-in for the legacy OutputSpec.

    ``logical_name`` is auto-populated from the dict key by SpecSection's validator.
    ``output_type`` is exposed as an alias of ``type``.
    """

    logical_name: str = ""
    type: DependencyType = DependencyType.PROCESSING_OUTPUT
    property_path: str = ""
    aliases: List[str] = Field(default_factory=list)
    semantic_keywords: List[str] = Field(default_factory=list)
    data_type: str = "S3Uri"
    description: str = ""

    @field_validator("type", mode="before")
    @classmethod
    def coerce_type(cls, v: object) -> object:
        if isinstance(v, str):
            return _DEP_TYPE_BY_VALUE.get(v, v)
        return v

    @property
    def output_type(self) -> DependencyType:
        """Legacy OutputSpec field name."""
        return self.type

    def matches_name_or_alias(self, name: str) -> bool:
        """Check if name matches the logical name or any alias (case-insensitive)."""
        if name == self.logical_name:
            return True
        name_lower = name.lower()
        return any(alias.lower() == name_lower for alias in self.aliases)


class SpecSection(BaseModel):
    """
    The 'spec' section of a .step.yaml — dependency resolution metadata.

    Drop-in for the legacy StepSpecification's dependency/output surface. The
    enclosing StepInterface mirrors ``step_type``/``node_type`` here so this object
    can be registered/consumed standalone where a StepSpecification was expected.
    """

    dependencies: Dict[str, DependencyDecl] = Field(default_factory=dict)
    outputs: Dict[str, OutputDecl] = Field(default_factory=dict)

    # Carried over from the parent StepInterface so SpecSection is a self-contained
    # StepSpecification stand-in (the registry/resolver read these off the spec).
    step_type: str = ""
    node_type: NodeType = NodeType.INTERNAL

    @model_validator(mode="after")
    def _populate_logical_names(self) -> "SpecSection":
        """Set each decl's logical_name from its dict key (single source of truth)."""
        for name, dep in self.dependencies.items():
            if not dep.logical_name:
                dep.logical_name = name
        for name, out in self.outputs.items():
            if not out.logical_name:
                out.logical_name = name
        return self

    # --- StepSpecification lookup API ---

    def get_dependency(self, logical_name: str) -> Optional[DependencyDecl]:
        return self.dependencies.get(logical_name)

    def get_output(self, logical_name: str) -> Optional[OutputDecl]:
        return self.outputs.get(logical_name)

    def get_output_by_name_or_alias(self, name: str) -> Optional[OutputDecl]:
        """Get output by logical name or alias (case-insensitive on aliases)."""
        if name in self.outputs:
            return self.outputs[name]
        name_lower = name.lower()
        for output in self.outputs.values():
            for alias in output.aliases:
                if alias.lower() == name_lower:
                    return output
        return None

    def list_all_output_names(self) -> List[str]:
        """All output logical names plus aliases."""
        names: List[str] = []
        for output in self.outputs.values():
            names.append(output.logical_name)
            names.extend(output.aliases)
        return names

    def list_required_dependencies(self) -> List[DependencyDecl]:
        return [d for d in self.dependencies.values() if d.required]

    def list_optional_dependencies(self) -> List[DependencyDecl]:
        return [d for d in self.dependencies.values() if not d.required]

    def validate_specification(self) -> List[str]:
        """Consistency check (legacy StepSpecification.validate_specification)."""
        errors: List[str] = []
        if (
            not self.dependencies
            and not self.outputs
            and self.node_type != NodeType.SINGULAR
        ):
            errors.append(f"Step '{self.step_type}' has no dependencies or outputs")
        return errors


# --- Job-type variants ---


class VariantDecl(BaseModel):
    """
    A job_type variant block from a .step.yaml (e.g. training / calibration).

    Holds the spec/contract overrides that are merged over the base when a
    builder requests a specific job_type. Stored as raw dicts because they are
    partial overrides, not standalone sections.
    """

    spec: Dict = Field(default_factory=dict)
    contract: Dict = Field(default_factory=dict)


# --- Main model ---


class StepInterface(BaseModel):
    """
    Validated representation of a .step.yaml file.

    This is the single message passed among dep resolver, builder, and assembler.
    Replaces the previous (ScriptContract|StepContract, StepSpecification) tuple.

    Build it from a parsed YAML dict via :meth:`from_yaml`, which applies any
    requested ``job_type`` variant before validation.
    """

    step_type: str
    node_type: NodeType = NodeType.INTERNAL
    registry: RegistrySection = Field(default_factory=RegistrySection)
    #: Declarative COMPUTE descriptor (FZ 31e1d3k) — a TOP-LEVEL section (peer of contract/spec)
    #: because it describes the BUILDER's compute object (processor/estimator/model/transformer), not
    #: the script contract: script-less steps (CreateModel/Transform) carry a near-empty contract but
    #: a full compute. ``_sync_and_align`` mirrors it onto ``contract.compute`` for back-compat. Empty
    #: (``kind=None``) ⇒ the step keeps its own factory.
    compute: ComputeSpec = Field(default_factory=lambda: ComputeSpec())
    #: Per-axis STRATEGY-SELECTION knobs (FZ 31e1d3f1) — the blueprint that wires pattern injection
    #: (step_assembly / include_job_type_in_path / direct_input_keys), read into
    #: the bound handler by ``_auto_bind_handler`` so the YAML steers the build, not a builder shell.
    patterns: PatternsSection = Field(default_factory=PatternsSection)
    contract: ContractSection
    spec: SpecSection = Field(default_factory=SpecSection)
    variants: Dict[str, VariantDecl] = Field(default_factory=dict)

    @field_validator("registry", "patterns", "spec", mode="before")
    @classmethod
    def _coerce_empty_section(cls, v: object) -> object:
        # A bare ``patterns:`` / ``registry:`` / ``spec:`` YAML key parses to None — treat an empty
        # section as the default (so dropping a section's last field doesn't break the load).
        if v is None:
            return {}
        return v

    @field_validator("node_type", mode="before")
    @classmethod
    def coerce_node_type(cls, v: object) -> object:
        if isinstance(v, str):
            return _NODE_TYPE_BY_VALUE.get(v, v)
        return v

    @classmethod
    def from_yaml(cls, data: Dict, job_type: Optional[str] = None) -> "StepInterface":
        """
        Build a StepInterface from a parsed ``.step.yaml`` dict, resolving variants.

        When ``job_type`` names a variant, that variant's ``spec``/``contract``
        overrides are **deep-merged** over the base sections before validation. The
        merge is recursive: a variant that lists only a subset of
        ``spec.dependencies`` (or ``outputs`` / contract ``inputs``) overrides just
        those ports' fields and leaves the rest of the base set intact — it does not
        replace the whole nested dict. Steps without a matching variant fall back to
        the base sections unchanged.

        A shallow merge here was a latent bug: because variants routinely restate
        only the ports they tweak, ``{**base, **variant}`` at the section level
        dropped every base port the variant happened to omit (e.g. it dropped
        ``hyperparameters_s3_uri`` from ``RiskTableMapping``'s variants, which then
        violated the contract↔spec alignment invariant and raised at construction).
        """
        data = dict(data)
        variants = data.get("variants") or {}
        if job_type and job_type in variants:
            variant = variants[job_type] or {}
            if variant.get("spec"):
                data["spec"] = _deep_merge(data.get("spec") or {}, variant["spec"])
            if variant.get("contract"):
                data["contract"] = _deep_merge(
                    data.get("contract") or {}, variant["contract"]
                )
            if variant.get("patterns"):
                data["patterns"] = _deep_merge(
                    data.get("patterns") or {}, variant["patterns"]
                )
        elif job_type and variants:
            # A job_type was requested but this step declares variants and none matches. Silently
            # using the base spec is a real correctness hazard: variants routinely tighten the base
            # (e.g. RiskTableMapping's variants flip model_artifacts_input from required=false to
            # required=true), so the base would drop a required dependency and the resolver would
            # never wire that edge — a structurally wrong step with no signal (deep dive 2026-07-03,
            # T6). Fail loud. (Gated on `variants` being non-empty so a variant-less step whose
            # config still carries a job_type is unaffected.)
            raise ValueError(
                f"Unknown job_type {job_type!r} for step "
                f"{data.get('step_type', '<unknown>')!r}; declared variants: {sorted(variants)}"
            )
        return cls(**data)

    @model_validator(mode="after")
    def _sync_and_align(self) -> "StepInterface":
        """Propagate step_type/node_type onto spec and check cross-section alignment."""
        # Keep spec's StepSpecification-stand-in fields in sync with the top level.
        if not self.spec.step_type:
            self.spec.step_type = self.step_type
        self.spec.node_type = self.node_type

        # Reconcile the promoted top-level `compute` with the back-compat `contract.compute` mirror
        # (FZ 31e1d3k). Authors declare `compute:` at the top level; the contract mirror keeps the
        # `b.contract.compute` read sites working. Exactly one side should be populated in a .step.yaml;
        # if both are (mid-migration), they must agree. Whichever is set becomes both.
        top = self.compute if self.compute.kind is not None else None
        contract_c = (
            self.contract.compute if self.contract.compute.kind is not None else None
        )
        if top is not None and contract_c is not None and top != contract_c:
            raise ValueError(
                "compute declared in BOTH the top-level section and contract.compute with different "
                "values; declare it once at the top level"
            )
        resolved = top or contract_c
        if resolved is not None:
            self.compute = resolved
            self.contract.compute = resolved

        # Contract inputs must each have a matching spec dependency.
        missing_deps = set(self.contract.inputs.keys()) - set(
            self.spec.dependencies.keys()
        )
        if missing_deps:
            raise ValueError(
                f"Contract inputs missing from spec dependencies: {missing_deps}"
            )

        # Contract outputs must each have a matching spec output.
        missing_outs = set(self.contract.outputs.keys()) - set(self.spec.outputs.keys())
        if missing_outs:
            raise ValueError(
                f"Contract outputs missing from spec outputs: {missing_outs}"
            )
        return self

    # --- ScriptContract drop-in accessors (delegate to contract) ---

    @property
    def script_contract(self) -> ContractSection:
        """Legacy StepSpecification.script_contract accessor."""
        return self.contract

    @property
    def entry_point(self) -> Optional[str]:
        return self.contract.entry_point

    @property
    def expected_input_paths(self) -> Dict[str, str]:
        return self.contract.expected_input_paths

    @property
    def expected_output_paths(self) -> Dict[str, str]:
        return self.contract.expected_output_paths

    @property
    def expected_arguments(self) -> Dict[str, str]:
        return self.contract.arguments

    @property
    def required_env_vars(self) -> List[str]:
        return self.contract.env_vars.required

    @property
    def optional_env_vars(self) -> Dict[str, str]:
        return self.contract.env_vars.optional

    @property
    def framework_requirements(self) -> Dict[str, str]:
        return self.contract.framework_requirements

    @property
    def description(self) -> str:
        return self.contract.description

    # --- StepSpecification drop-in accessors (delegate to spec) ---

    @property
    def dependencies(self) -> Dict[str, DependencyDecl]:
        return self.spec.dependencies

    @property
    def outputs(self) -> Dict[str, OutputDecl]:
        return self.spec.outputs

    def get_dependency(self, logical_name: str) -> Optional[DependencyDecl]:
        return self.spec.get_dependency(logical_name)

    def get_output(self, logical_name: str) -> Optional[OutputDecl]:
        return self.spec.get_output(logical_name)

    def get_output_by_name_or_alias(self, name: str) -> Optional[OutputDecl]:
        return self.spec.get_output_by_name_or_alias(name)

    def list_all_output_names(self) -> List[str]:
        return self.spec.list_all_output_names()

    def list_required_dependencies(self) -> List[DependencyDecl]:
        return self.spec.list_required_dependencies()

    def list_optional_dependencies(self) -> List[DependencyDecl]:
        return self.spec.list_optional_dependencies()

    def validate_specification(self) -> List[str]:
        return self.spec.validate_specification()

    def validate_contract_alignment(self) -> "ValidationResult":
        """
        Validate that the contract aligns with the spec.

        Mirrors legacy StepSpecification.validate_contract_alignment: every contract
        input must have a matching spec dependency, and every contract output a
        matching spec output (extra spec deps/outputs and output aliases allowed).
        Returns a ValidationResult (is_valid / errors).
        """
        from .contract_base import ValidationResult

        errors: List[str] = []

        contract_inputs = set(self.contract.expected_input_paths.keys())
        spec_dep_names = set(self.spec.dependencies.keys())
        missing_deps = contract_inputs - spec_dep_names
        if missing_deps:
            errors.append(
                f"Contract inputs missing from specification dependencies: {missing_deps}"
            )

        contract_outputs = set(self.contract.expected_output_paths.keys())
        # An output is satisfied by a matching logical name OR alias.
        for out_name in contract_outputs:
            if self.spec.get_output_by_name_or_alias(out_name) is None:
                errors.append(
                    f"Contract output '{out_name}' has no matching specification output"
                )

        if errors:
            return ValidationResult.error(errors)
        return ValidationResult.success()
