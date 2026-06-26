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

from typing import Dict, List, Optional, TYPE_CHECKING
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


class ContractSection(BaseModel):
    """
    The 'contract' section of a .step.yaml — script execution requirements.

    Drop-in for the legacy ScriptContract / StepContract: the ``expected_*`` /
    ``required_env_vars`` / ``optional_env_vars`` accessors flatten the structured
    ports back to the ``Dict[str, str]`` / ``List[str]`` shapes consumers expect.
    """

    entry_point: Optional[str] = None
    inputs: Dict[str, InputPort] = Field(default_factory=dict)
    outputs: Dict[str, OutputPort] = Field(default_factory=dict)
    arguments: Dict[str, str] = Field(default_factory=dict)
    env_vars: EnvVars = Field(default_factory=EnvVars)
    framework_requirements: Dict[str, str] = Field(default_factory=dict)
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
    contract: ContractSection
    spec: SpecSection = Field(default_factory=SpecSection)
    variants: Dict[str, VariantDecl] = Field(default_factory=dict)

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
        return cls(**data)

    @model_validator(mode="after")
    def _sync_and_align(self) -> "StepInterface":
        """Propagate step_type/node_type onto spec and check cross-section alignment."""
        # Keep spec's StepSpecification-stand-in fields in sync with the top level.
        if not self.spec.step_type:
            self.spec.step_type = self.step_type
        self.spec.node_type = self.node_type

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
