"""
``author.*`` — the agent step-authoring guidance namespace (FZ 31e1d3f5).

Under Design B a step is authored as ONE ``.step.yaml`` + ONE config class + ONE script; the
builder is synthesized and the registry is derived by construction (writing the ``.step.yaml`` IS
the registration). This small namespace gives an authoring agent the SOP + restrictions + a
constructibility proof, WITHOUT a generator (the agent writes the files with its own ``Write`` tool)
and WITHOUT duplicating any enforcement logic:

- ``author.checklist(sagemaker_step_type[, step_assembly])`` — the ordered author→validate→integrate
  SOP as DATA naming the existing tools to call at each step. The routing branch is derived from the
  live ``strategy_registry`` so the "which handler / which assembly" answer can't drift.
- ``author.rules(topic)`` — the restriction set, by INTROSPECTING the live enforcement objects
  (``PASCAL_CASE_PATTERN``, the two distinct ``_REQUIRES`` enums, ``_KINDS`` / ``_SDK_CLASSES`` /
  ``_ASSEMBLIES`` / ``_SAGEMAKER_STEP_TYPES``, the ``source_dir`` packaging fact). Guidance is
  definitionally equal to enforcement; a conformance/identity test keeps it from drifting.
- ``author.preflight_step(step_name | all)`` — the offline constructibility proof: a FLAT list of the
  same gates CI runs as its merge gate (interface validation + registry-snapshot parity +
  RegistryBindingValidator B3 + ``resolve_strategy`` routability). SDK-delegation rows report
  skip-not-error offline. This proves a step is CONSTRUCTIBLE (binds + synthesizes), not merely
  parseable, before a code review.

All three are read-only, stateless, and offline-safe. They compose existing engines/constants — they
add no new validation logic of their own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..envelope import ToolResult, ToolError
from ..registry import ToolDef

# The closed topic set for author.rules — keep in sync with the `_RULES_BUILDERS` dispatch below.
_RULE_TOPICS = ("naming", "packaging", "sdk_carveout", "reuse_class", "closure")


def _resolve_config_class(step_name: str):
    """Resolve a canonical step name to its config CLASS object (or raise ToolError).

    Shared by author.config_constraints + author.preflight_config; mirrors catalog.config_fields'
    resolution (registry name -> catalog config-class discovery)."""
    from ...registry.step_names import get_config_class_name
    from ...step_catalog import StepCatalog

    try:
        config_class_name = get_config_class_name(step_name)
    except (ValueError, KeyError) as exc:
        raise ToolError(str(exc), code="not_found")
    config_class = StepCatalog().discover_config_classes().get(config_class_name)
    if config_class is None:
        raise ToolError(
            f"config class '{config_class_name}' for step '{step_name}' could not be discovered",
            code="not_found",
        )
    return config_class_name, config_class


# ---------------------------------------------------------------------------
# author.rules
# ---------------------------------------------------------------------------


def _rules_naming() -> Dict[str, Any]:
    """Naming restrictions, read off the LIVE enforcement objects (validation_utils + RegistrySection)."""
    from ...registry.validation_utils import PASCAL_CASE_PATTERN
    from ...core.base.step_interface import RegistrySection

    return {
        "step_name": {
            "rule": "PascalCase; this string is `step_type` in the .step.yaml AND the canonical "
            "registry name (registry is derived by construction).",
            "pascal_case_regex": PASCAL_CASE_PATTERN.pattern,
        },
        "config_class": {
            "rule": "must be named `<StepName>Config` (the registry derives config_class by this "
            "convention; the 3 known breakers declare registry.config_class explicitly).",
            "suffix": "Config",
        },
        "builder_step_name": {
            "rule": "derived as `<StepName>StepBuilder` — there is NO builder file to write "
            "(synthesized onto TemplateStepBuilder).",
            "suffix": "StepBuilder",
        },
        "sagemaker_step_type": {
            "rule": "the routing key (selects the PatternHandler); must be one of the closed set.",
            "valid_values": list(RegistrySection._SAGEMAKER_STEP_TYPES),
        },
    }


def _rules_packaging() -> Dict[str, Any]:
    """The source_dir packaging constraint + the output-destination axes — Pydantic field facts on
    ContractSection, NOT standalone validators. Surfaced honestly as guidance the validator cannot
    enforce at author time (they are runtime/build consequences)."""
    return {
        "source_dir": {
            "field": "contract.source_dir",
            "default": False,
            "true": "bundles the whole script directory — the script MAY import sibling modules "
            "(shared libs). REQUIRES a FrameworkProcessor (step_assembly=step_args); "
            "ScriptProcessor.run has no source_dir.",
            "false": "ships a single `code=` file — the script CANNOT import a new sibling lib.",
        },
        "output_path_token": {
            "field": "contract.output_path_token",
            "default": None,
            "none": "DEFAULT — the output S3 prefix segment is DERIVED from the step name "
            "(canonical_to_snake(step_type)); the destination is Join(base, <derived_token>, "
            "[job_type], logical_name). Use this for ~all steps — the folder matches the step name.",
            "set": "OPT-IN override — the given string is used VERBATIM as that S3 prefix segment "
            "instead of the derived token. Set this ONLY when an EXTERNAL consumer keys off a fixed "
            "S3 folder name that does not match the cursus step name (e.g. PIPER scans "
            "<pipeline>/Model_Metric_Generation_Step/ for .metric files, so PiperMetricGeneration "
            "sets output_path_token: Model_Metric_Generation_Step).",
            "caveat": "This changes only the emitted S3 destination, NOT the SageMaker/DAG step name; "
            "prefer the derived default unless an external contract forces a fixed folder.",
        },
        "include_job_type_in_path": {
            "field": "contract.include_job_type_in_path",
            "default": True,
            "true": "config.job_type is a segment of the output destination "
            "(Join(base, token, job_type, logical_name)) — use for steps that fan out per split.",
            "false": "omit job_type from the destination (Join(base, token, logical_name)).",
        },
        "caveat": "These are script-packaging / output-destination facts derived from the Pydantic "
        "field semantics, not standalone validators — choose source_dir before writing imports, and "
        "leave output_path_token unset unless an external consumer requires a fixed folder name.",
        "sais_preamble": "Scripts that carry the SAIS secure-pypi install preamble (USE_SECURE_PYPI "
        "/ CA_REPOSITORY_ARN / CodeArtifact) MUST keep it — it is load-bearing, never remove it.",
    }


def _rules_sdk_carveout() -> Dict[str, Any]:
    """The SDK-delegation carve-out — read off the TWO DISTINCT _REQUIRES enums so the agent never
    conflates the build-time SAIS-step dep (registry) with the compute mods dep (compute)."""
    from ...core.base.step_interface import RegistrySection, ComputeSpec

    return {
        "registry_requires": {
            "field": "registry.requires",
            "valid_values": list(RegistrySection._REQUIRES),
            "rule": "set `secure_ai_sandbox_workflow_python_sdk` ONLY for an SDK-delegation step "
            "(CradleDataLoading / RedshiftDataLoading / DataUploading / Registration); its builder "
            "imports a SAIS *Step class. These steps are OFFLINE-UNDISCOVERABLE BY DESIGN — a lazy "
            "thunk imports the SDK only inside the SAIS environment.",
        },
        "compute_requires": {
            "field": "compute.requires",
            "valid_values": list(ComputeSpec._REQUIRES),
            "rule": "`mods_workflow_core` ONLY for the script/kms_network compute path (EdxUploading). "
            "Auto-derived from compute.kms_network — do not hand-set it inconsistently.",
        },
        "note": "registry.requires (the create_step/SDK axis) and compute.requires (the compute axis) "
        "are DISTINCT enums — do not conflate them.",
    }


def _rules_reuse_class() -> Dict[str, Any]:
    """The three script reuse classes — guidance for how much of the script to write vs leave open."""
    return {
        "shared": "packaging / payload / calibration / metrics / model-report — reusable as-is; "
        "the script is shipped functionality.",
        "model_dependent": "pytorch / lightgbmmt training + eval/inference families — import model "
        "packages cursus does NOT ship. Author the standardizable sub-steps (preprocess / dataset / "
        "dataloader / eval / compare / plot / save) and leave the MODEL section as a documented OPEN "
        "SECTION skeleton with a numbered checklist of what the project must implement.",
        "user_template": "everything else — a reference template the project adapts.",
        "decision": "If the script imports a model package cursus doesn't ship → model_dependent → "
        "open-section skeleton. If it's one of the shared utilities → shared. Else → user_template.",
    }


def _rules_closure() -> Dict[str, Any]:
    """The triangle-closure / registry-by-construction rules the author must respect."""
    return {
        "registry_by_construction": "STEP_NAMES is DERIVED from interfaces/*.step.yaml by "
        "build_registry_from_interfaces(). There is NO registry-edit step and NO builder export — "
        "writing the .step.yaml IS the registration. Never hand-edit a registry file.",
        "contract_spec_alignment": "contract.inputs keys must each have a matching spec.dependencies "
        "key, and contract.outputs a matching spec.outputs key — enforced at load by _sync_and_align "
        "(the old 'Level-2' check is now this Pydantic invariant).",
        "compatible_sources": "spec.dependencies.*.compatible_sources must use the EXACT upstream step "
        "name (case-sensitive) — a case typo silently loses the resolver's match bonus.",
        "gates": "validate.step_interface is the author-time gate; author.preflight_step proves "
        "constructibility (the same gates CI runs).",
    }


_RULES_BUILDERS = {
    "naming": _rules_naming,
    "packaging": _rules_packaging,
    "sdk_carveout": _rules_sdk_carveout,
    "reuse_class": _rules_reuse_class,
    "closure": _rules_closure,
}


def _rules(args: Dict[str, Any]) -> ToolResult:
    topic = args["topic"]
    if topic not in _RULES_BUILDERS:
        raise ToolError(
            f"unknown topic {topic!r}; choose one of {_RULE_TOPICS}",
            code="invalid_input",
        )
    return ToolResult.success({"topic": topic, "rules": _RULES_BUILDERS[topic]()})


# ---------------------------------------------------------------------------
# author.checklist
# ---------------------------------------------------------------------------

# Same-phase exemplar to copy section shapes from, by sagemaker_step_type (+ Processing flavor).
_EXEMPLAR_BY_TYPE = {
    "Training": "XGBoostTraining",
    "Transform": "BatchTransform",
    "CreateModel": "XGBoostModel",
    "Processing": "TabularPreprocessing",  # default; eval/postproc are also Processing
}


def _checklist(args: Dict[str, Any]) -> ToolResult:
    """The ordered author→validate→integrate SOP as DATA. The routing branch (handler + assembly) is
    DERIVED from the live strategy_registry so it can't drift from the build."""
    from ...registry import strategy_registry as sr

    step_type = args["sagemaker_step_type"]
    step_assembly = args.get("step_assembly")

    # Validate the routing key against the closed set (same pin the .step.yaml validator uses).
    from ...core.base.step_interface import RegistrySection

    if step_type not in RegistrySection._SAGEMAKER_STEP_TYPES:
        raise ToolError(
            f"sagemaker_step_type {step_type!r} not in {RegistrySection._SAGEMAKER_STEP_TYPES}",
            code="invalid_input",
        )

    # Resolve the bound handler from the registry (the analogue of "read the builder class").
    axis, name = sr.axis_name_for_step_type(step_type, step_assembly)
    routable = True
    handler_desc: Optional[Dict[str, Any]] = None
    try:
        info = sr.resolve_strategy(axis, name)
        handler_desc = sr.strategy_to_dict(info)
    except sr.NoBuilderError:
        routable = False  # Base/Lambda etc. — no construction handler

    exemplar = _EXEMPLAR_BY_TYPE.get(step_type, "TabularPreprocessing")

    steps: List[Dict[str, Any]] = [
        {
            "order": 1,
            "phase": "resolve",
            "action": "Confirm the routing decision (handler + knobs) for this step type.",
            "tool": "strategies.for_step_type",
            "args_hint": {
                "sagemaker_step_type": step_type,
                "step_assembly": step_assembly,
            },
        },
        {
            "order": 2,
            "phase": "resolve",
            "action": "Read the restriction set you must honor (naming, packaging, SDK, reuse class, closure).",
            "tool": "author.rules",
            "args_hint": {"topic": "naming"},
        },
        {
            "order": 3,
            "phase": "author",
            "action": f"Copy section shapes from the same-phase exemplar '{exemplar}' "
            "(its wired inputs/outputs/env/job-args and config fields).",
            "tool": "steps.io / steps.patterns / catalog.config_fields",
            "args_hint": {"step_name": exemplar},
        },
        {
            "order": 4,
            "phase": "author",
            "action": "Write the THREE artifacts with your Write tool: "
            "src/cursus/steps/interfaces/<snake>.step.yaml, "
            "src/cursus/steps/configs/config_<snake>_step.py (<StepName>Config), "
            "src/cursus/steps/scripts/<snake>.py (main(input_paths,output_paths,environ_vars,job_args)). "
            "Do NOT edit any registry file — writing the .step.yaml IS the registration.",
            "tool": "(your Write tool)",
            "args_hint": None,
        },
        {
            "order": 5,
            "phase": "validate",
            "action": "Run the author-time gate on the new step (Pydantic load + contract↔spec alignment "
            "+ compatible_sources check). Fix every blocking error and re-run.",
            "tool": "validate.step_interface",
            "args_hint": {"step_name": "<StepName>"},
        },
        {
            "order": 6,
            "phase": "preflight",
            "action": "Prove the step is CONSTRUCTIBLE (handler binds, builder synthesizes, config "
            "fields covered) — the same gates CI runs as its merge gate.",
            "tool": "author.preflight_step",
            "args_hint": {"step_name": "<StepName>"},
        },
        {
            "order": 7,
            "phase": "integrate",
            "action": "Add the step to a DAG, compile, then publish (brazil pb build → cursus-peru-dev "
            "→ cursus-peru-shared → CDK → CodeArtifact).",
            "tool": "compile.dag",
            "args_hint": None,
        },
    ]

    data = {
        "sagemaker_step_type": step_type,
        "step_assembly": step_assembly,
        "routable": routable,
        "bound_handler": handler_desc,
        "exemplar_step": exemplar,
        "steps": steps,
    }
    if not routable:
        return ToolResult.success(
            data,
            warnings=[
                f"{step_type!r} binds no construction handler (e.g. Base/Lambda are builder-less) — "
                "authoring a buildable step requires a routable type."
            ],
        )
    next_steps = [
        {
            "tool": "strategies.for_step_type",
            "when": "now, to start the loop",
            "why": "step 1 of the checklist — bind the concrete handler + knobs",
            "args_hint": {
                "sagemaker_step_type": step_type,
                "step_assembly": step_assembly,
            },
        }
    ]
    return ToolResult.success(data, next_steps=next_steps)


# ---------------------------------------------------------------------------
# author.preflight_step
# ---------------------------------------------------------------------------


def _gate_interface(targets: List[str], job_type: Optional[str]) -> Dict[str, Any]:
    """Gate 1 — the author-time interface gate (reuses validate_cli._validate_one_interface)."""
    from ...cli.validate_cli import _validate_one_interface

    results = [_validate_one_interface(t, job_type) for t in targets]
    n_err = sum(1 for r in results if not r["ok"])
    return {
        "name": "interface",
        "passed": n_err == 0,
        "detail": f"{len(results)} interface(s) validated, {n_err} with blocking errors"
        + (
            ": " + ", ".join(r["step"] for r in results if not r["ok"]) if n_err else ""
        ),
    }


def _gate_registry_snapshot() -> Dict[str, Any]:
    """Gate 2 — registry derives + is internally well-formed (every row carries sagemaker_step_type)."""
    try:
        from ...registry.interface_registry_loader import build_registry_from_interfaces

        derived = build_registry_from_interfaces()
        missing = [n for n, r in derived.items() if not r.get("sagemaker_step_type")]
        return {
            "name": "registry_parity",
            "passed": not missing and len(derived) >= 1,
            "detail": f"{len(derived)} rows derived"
            + (f"; rows missing sagemaker_step_type: {missing}" if missing else ""),
        }
    except Exception as e:
        return {
            "name": "registry_parity",
            "passed": False,
            "detail": f"registry derivation failed: {e}",
        }


def _gate_b3(step_name: str, workspace_dirs: Optional[List[str]]) -> Dict[str, Any]:
    """Gate 3 — RegistryBindingValidator B3: handler binds + builder loads + config-field coverage.
    SDK-delegation / no-builder rows report skip-not-error."""
    from ...validation.alignment.validators.registry_binding_validator import (
        RegistryBindingValidator,
    )

    validator = RegistryBindingValidator(workspace_dirs=workspace_dirs)
    result = validator.validate_builder_config_alignment(step_name)
    status = result.get("status")
    if status == "SKIPPED":
        return {
            "name": "registry_binding_b3",
            "passed": True,
            "detail": f"skipped (not an error): {result.get('reason', 'no-builder/SDK row')}",
        }
    errors = [
        i for i in result.get("issues", []) if i.get("level") in ("CRITICAL", "ERROR")
    ]
    return {
        "name": "registry_binding_b3",
        "passed": status == "COMPLETED" and not errors,
        "detail": f"status={status}"
        + (
            "; " + "; ".join(i.get("message", "") for i in errors[:3]) if errors else ""
        ),
    }


def _gate_routability(step_name: str) -> Dict[str, Any]:
    """Gate 4 — resolve_strategy routability for the step's (sagemaker_step_type, step_assembly).
    Probes the routing table STRUCTURALLY (no SAIS import); a no-builder/SDK row is a valid skip."""
    from ...registry import strategy_registry as sr

    try:
        from ...registry.step_names import get_sagemaker_step_type
        from ...steps.interfaces import load_interface

        sm_type = get_sagemaker_step_type(step_name)
        iface = load_interface(step_name)
        step_assembly = getattr(getattr(iface, "patterns", None), "step_assembly", None)
    except Exception as e:
        return {
            "name": "routability",
            "passed": False,
            "detail": f"could not load step routing facts: {e}",
        }

    if sm_type in ("Base", "Lambda"):
        return {
            "name": "routability",
            "passed": True,
            "detail": f"{sm_type} is a no-builder row (valid skip)",
        }
    axis, name = sr.axis_name_for_step_type(sm_type, step_assembly)
    try:
        sr.resolve_strategy(axis, name)
        return {
            "name": "routability",
            "passed": True,
            "detail": f"({axis}={name}) binds a handler",
        }
    except sr.NoBuilderError as e:
        return {"name": "routability", "passed": False, "detail": str(e)}


def _preflight_step(args: Dict[str, Any]) -> ToolResult:
    """The offline constructibility proof — a FLAT list of the four CI merge gates."""
    step_name = args.get("step_name")
    validate_all = bool(args.get("all", False))
    job_type = args.get("job_type")
    workspace_dirs = args.get("workspace_dirs")

    if not validate_all and (not isinstance(step_name, str) or not step_name.strip()):
        raise ToolError(
            "provide 'step_name' or set 'all' to true", code="invalid_input"
        )
    if workspace_dirs is not None and not isinstance(workspace_dirs, list):
        raise ToolError(
            "'workspace_dirs' must be a list of directory paths", code="invalid_input"
        )

    # The interface gate accepts a file-stem OR a canonical name; B3 / routability need the
    # CANONICAL step name (PascalCase). In --all mode the interface gate runs over the .step.yaml
    # file stems (mirroring the CLI), while B3/routability iterate the canonical registry names,
    # skipping the interface-less abstract rows (Base / Processing / HyperparameterPrep).
    if validate_all:
        from ...steps.interfaces import list_available_interfaces
        from ...registry.step_names import get_all_step_names

        interface_targets = sorted(list_available_interfaces())
        _ABSTRACT = {"Base", "Processing", "HyperparameterPrep"}
        binding_targets = sorted(n for n in get_all_step_names() if n not in _ABSTRACT)
    else:
        interface_targets = [step_name]
        binding_targets = [step_name]

    gates: List[Dict[str, Any]] = []
    # Gate 1: interface validation (the author-time gate, over the interface set).
    gates.append(_gate_interface(interface_targets, job_type))
    # Gate 2: registry derives + well-formed (global — a new YAML must not break derivation).
    gates.append(_gate_registry_snapshot())
    # Gates 3+4: per-step binding (B3) + routability over the canonical step names.
    for t in binding_targets:
        b3 = _gate_b3(t, workspace_dirs)
        route = _gate_routability(t)
        if validate_all:
            b3["step"] = t
            route["step"] = t
        gates.append(b3)
        gates.append(route)

    constructible = all(g["passed"] for g in gates)
    data = {
        "step_name": step_name if not validate_all else None,
        "scope": "all" if validate_all else step_name,
        "constructible": constructible,
        "gates": gates,
    }
    if not constructible:
        failed = [
            g["name"] + (f"[{g['step']}]" if g.get("step") else "")
            for g in gates
            if not g["passed"]
        ]
        return ToolResult.failure(
            f"step not constructible — failing gate(s): {', '.join(failed)}",
            code="not_constructible",
            details=data,
            remedy={
                "suggested_tools": [
                    "validate.step_interface",
                    "strategies.for_step_type",
                    "author.rules",
                ],
                "fix_action": "An 'interface' failure → fix the .step.yaml (run validate.step_interface "
                "for the detail). A 'registry_binding_b3' failure → the config class is missing a field "
                "the handler reads, or the builder can't synthesize. A 'routability' failure → the "
                "sagemaker_step_type/step_assembly binds no handler (run strategies.for_step_type).",
            },
        )
    next_steps = [
        {
            "tool": "compile.dag",
            "when": "the step is constructible and you have a DAG that uses it",
            "why": "preflight passed — the step will build; proceed to integrate + publish",
        }
    ]
    return ToolResult.success(data, next_steps=next_steps)


# ---------------------------------------------------------------------------
# author.config_constraints — surface a config class's legal VALUES (not just types)
# ---------------------------------------------------------------------------


def _config_constraints(args: Dict[str, Any]) -> ToolResult:
    """List a step's config-class fields WITH their closed-value constraints + required-no-default
    flags — the data the agent needs to write correct config VALUES (the Cat1/Cat3 bug class came
    from type-only listings that hid the @field_validator allowed sets / case-sensitivity)."""
    from ...api.factory.field_extractor import extract_field_requirements

    step_name = args.get("step_name")
    if not isinstance(step_name, str) or not step_name.strip():
        raise ToolError("'step_name' must be a non-empty string", code="invalid_input")

    config_class_name, config_class = _resolve_config_class(step_name)
    reqs = extract_field_requirements(config_class)

    required_no_default = [
        r["name"] for r in reqs if r.get("required") and r.get("default") is None
    ]
    constrained = [r for r in reqs if r.get("allowed_values")]

    data = {
        "step_name": step_name,
        "config_class": config_class_name,
        "field_count": len(reqs),
        "fields": reqs,  # each carries name/type/required/default + allowed_values/case_sensitive when constrained
        "required_no_default": required_no_default,  # the Tier-1 fields a value MUST be supplied for (Cat2)
        "constrained_fields": [
            {
                "name": r["name"],
                "allowed_values": r["allowed_values"],
                "case_sensitive": r.get("case_sensitive", True),
            }
            for r in constrained
        ],
    }
    return ToolResult.success(data, step_name=step_name)


# ---------------------------------------------------------------------------
# author.preflight_config — the VALUE gate: instantiate the config with user values
# ---------------------------------------------------------------------------


def _preflight_config(args: Dict[str, Any]) -> ToolResult:
    """Validate a config VALUE set against the live config class (model_validate) — the gate that
    catches wrong enum case, invalid enum, wrong type, and missing required fields (Cat1/2/3, 8 of
    the 21 empirical bugs) that the .step.yaml / B3 gates structurally cannot (they never
    instantiate a config with user values)."""
    step_name = args.get("step_name")
    values = args.get("values")
    if not isinstance(step_name, str) or not step_name.strip():
        raise ToolError("'step_name' must be a non-empty string", code="invalid_input")
    if not isinstance(values, dict):
        raise ToolError(
            "'values' must be an object of {field: value} config values",
            code="invalid_input",
        )

    config_class_name, config_class = _resolve_config_class(step_name)

    try:
        config_class.model_validate(values)
        return ToolResult.success(
            {
                "step_name": step_name,
                "config_class": config_class_name,
                "valid": True,
            },
            step_name=step_name,
            next_steps=[
                {
                    "tool": "author.preflight_step",
                    "when": "the config values are valid",
                    "why": "values validated; confirm the step is constructible end-to-end",
                }
            ],
        )
    except Exception as exc:  # pydantic ValidationError (+ any validator ValueError)
        # Render the per-field errors flatly so the agent can fix exactly the offending fields.
        errors: List[Dict[str, Any]] = []
        raw = getattr(exc, "errors", None)
        if callable(raw):
            try:
                for e in exc.errors():
                    errors.append(
                        {
                            "field": ".".join(str(p) for p in e.get("loc", ())),
                            "message": e.get("msg", ""),
                            "type": e.get("type", ""),
                        }
                    )
            except Exception:
                errors.append({"field": "", "message": str(exc), "type": "error"})
        else:
            errors.append({"field": "", "message": str(exc), "type": "error"})
        return ToolResult.failure(
            f"config values invalid for {config_class_name}: {len(errors)} error(s)",
            code="config_invalid",
            details={
                "step_name": step_name,
                "config_class": config_class_name,
                "valid": False,
                "errors": errors,
            },
            remedy={
                "suggested_tools": ["author.config_constraints"],
                "fix_action": "For a wrong enum/case error, call author.config_constraints to get the "
                "field's allowed_values + case_sensitive flag. For a missing-field error, supply the "
                "required_no_default field. Fix the named field(s) and re-run author.preflight_config.",
            },
        )


# ---------------------------------------------------------------------------
# author.check_script — script<->contract alignment, BOTH directions
# ---------------------------------------------------------------------------


def _check_script(args: Dict[str, Any]) -> ToolResult:
    """Check a step's SCRIPT against its contract in BOTH directions: the forward checks (main()
    signature + the keys the script uses are declared) AND the new reverse checks (every CLI arg the
    builder passes is parsed; every required env var is read). Catches the Cat4/Cat5 empirical bugs
    (custom script missing --job_type argparse; declared env var read-but-ignored) that the
    .step.yaml gate + B3 cannot. Offline, read-only; skips script-less steps (SDK/CreateModel/Transform)."""
    from ...step_catalog import StepCatalog
    from ...validation.alignment.analyzer.script_analyzer import ScriptAnalyzer

    step_name = args.get("step_name")
    if not isinstance(step_name, str) or not step_name.strip():
        raise ToolError("'step_name' must be a non-empty string", code="invalid_input")

    catalog = StepCatalog()
    step_info = catalog.get_step_info(step_name)
    script_comp = (
        step_info.file_components.get("script")
        if step_info and step_info.file_components
        else None
    )
    if step_info is None:
        return ToolResult.failure(
            f"step not found: '{step_name}'",
            code="not_found",
            details={"step_name": step_name},
        )
    if script_comp is None or getattr(script_comp, "path", None) is None:
        # Script-less steps (SDK-delegation / CreateModel / Transform) — nothing to check.
        return ToolResult.success(
            {
                "step_name": step_name,
                "status": "skipped",
                "reason": "no local script (SDK-delegation / script-less CreateModel/Transform)",
                "passed": True,
                "issues": [],
            },
            step_name=step_name,
        )

    # Build the reverse-check contract from the .step.yaml interface (shape (a): job_arguments[].flag
    # + env_vars.required — the authoritative source for Cat4/Cat5). Fall back gracefully.
    contract: Dict[str, Any] = {}
    sm_type: Optional[str] = None
    try:
        from ...steps.interfaces import load_interface

        iface = load_interface(step_name)
        c = iface.contract
        contract = {
            "expected_input_paths": c.expected_input_paths,
            "expected_output_paths": c.expected_output_paths,
            "required_env_vars": c.required_env_vars,
            "optional_env_vars": c.optional_env_vars,
            "expected_arguments": c.expected_arguments,
            "job_arguments": [
                {"flag": j.flag, "source": j.source} for j in c.job_arguments
            ],
            "env_vars": {
                "required": list(c.required_env_vars),
                "optional": dict(c.optional_env_vars),
            },
        }
        sm_type = getattr(getattr(iface, "registry", None), "sagemaker_step_type", None)
        # SDK-delegation steps (registry.requires == secure_ai_sandbox_workflow_python_sdk) ship a
        # SAIS-shaped script with no standard main(input_paths,...) — the builder delegates to the
        # SDK *Step class, not a cursus script entrypoint. Skip them like the script-less steps.
        requires = getattr(getattr(iface, "registry", None), "requires", "none")
        if requires == "secure_ai_sandbox_workflow_python_sdk":
            return ToolResult.success(
                {
                    "step_name": step_name,
                    "status": "skipped",
                    "reason": "SDK-delegation step (builder delegates to a SAIS *Step class; "
                    "the script has no standard main() entrypoint)",
                    "passed": True,
                    "issues": [],
                },
                step_name=step_name,
            )
    except Exception as exc:
        return ToolResult.failure(
            f"could not load interface/contract for '{step_name}': {exc}",
            code="not_found",
            details={"step_name": step_name},
        )

    try:
        analyzer = ScriptAnalyzer(str(script_comp.path))
    except Exception as exc:
        return ToolResult.failure(
            f"script could not be parsed: {exc}",
            code="script_analysis_error",
            details={"step_name": step_name, "script": str(script_comp.path)},
        )

    issues: List[Dict[str, Any]] = []
    # Forward: main() signature + the script-uses-but-undeclared checks.
    sig = analyzer.validate_main_function_signature()
    if not sig.get("has_main"):
        issues.append(
            {
                "severity": "CRITICAL",
                "category": "missing_main_function",
                "message": "Script must define main(input_paths, output_paths, environ_vars, job_args)",
                "recommendation": "Add the standard main() signature.",
            }
        )
    elif not sig.get("signature_valid"):
        issues.append(
            {
                "severity": "ERROR",
                "category": "invalid_main_signature",
                "message": f"main() signature mismatch: {sig.get('issues')}",
                "recommendation": "Fix to: def main(input_paths, output_paths, environ_vars, job_args)",
            }
        )
    issues.extend(analyzer.validate_contract_alignment(contract))
    # Reverse: declared args parsed + required env read (the Cat4/Cat5 gate).
    issues.extend(
        analyzer.validate_reverse_alignment(contract, sagemaker_step_type=sm_type)
    )

    passed = not any(i["severity"] in ("CRITICAL", "ERROR") for i in issues)
    data = {
        "step_name": step_name,
        "status": "checked",
        "sagemaker_step_type": sm_type,
        "script": str(script_comp.path),
        "passed": passed,
        "issues": issues,
    }
    if not passed:
        return ToolResult.failure(
            f"script check failed for {step_name}: "
            f"{sum(1 for i in issues if i['severity'] in ('CRITICAL', 'ERROR'))} blocking issue(s)",
            code="script_check_failed",
            details=data,
            remedy={
                "suggested_tools": ["steps.io", "author.config_constraints"],
                "fix_action": "unparsed_declared_arg → add parser.add_argument('--<flag>') in the "
                "script __main__ block OR remove the flag from contract.job_arguments. "
                "unread_required_env_var → read it via environ_vars.get('<VAR>') in main() OR demote "
                "it to env_vars.optional. invalid_main_signature → fix the main() signature.",
            },
        )
    return ToolResult.success(data, step_name=step_name)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[ToolDef] = [
    ToolDef(
        name="author.checklist",
        description=(
            "Return the ordered author→validate→integrate SOP for creating a new cursus step, as "
            "DATA naming the exact tool to call at each step (strategies.for_step_type → author.rules "
            "→ write the 3 artifacts → validate.step_interface → author.preflight_step → compile). The "
            "routing branch (bound handler + exemplar) is derived from the live strategy_registry. "
            "Start here when authoring a step."
        ),
        schema={
            "type": "object",
            "properties": {
                "sagemaker_step_type": {
                    "type": "string",
                    "description": "The step's SageMaker verb (Processing / Training / Transform / "
                    "CreateModel / a SAIS-delegation verb).",
                },
                "step_assembly": {
                    "type": "string",
                    "description": "Processing sub-verb (code | step_args | delegation); omit for "
                    "non-Processing types.",
                },
            },
            "required": ["sagemaker_step_type"],
            "additionalProperties": False,
        },
        handler=_checklist,
        tags=("planner",),
    ),
    ToolDef(
        name="author.rules",
        description=(
            "Return the authoring restriction set for one topic — naming (PascalCase / Config / "
            "StepBuilder / the valid sagemaker_step_type set), packaging (source_dir + SAIS preamble), "
            "sdk_carveout (the two distinct requires enums), reuse_class (shared / model_dependent / "
            "user_template), or closure (registry-by-construction + contract↔spec alignment). Values "
            "are read off the LIVE enforcement objects, so guidance can't drift from what the build "
            "enforces."
        ),
        schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": list(_RULE_TOPICS),
                    "description": "Which restriction set to return.",
                },
            },
            "required": ["topic"],
            "additionalProperties": False,
        },
        handler=_rules,
        tags=("validator",),
    ),
    ToolDef(
        name="author.preflight_step",
        description=(
            "Prove a step is CONSTRUCTIBLE (not merely parseable) before code review — a FLAT list of "
            "the four gates CI runs as its merge gate: interface validation, registry derivation/parity, "
            "RegistryBindingValidator B3 (handler binds + builder synthesizes + config-field coverage), "
            "and resolve_strategy routability. SDK-delegation / no-builder rows report skip-not-error "
            "offline. Pass 'step_name' for one step or 'all'=true for the whole suite (the CI gate)."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical step name to preflight. Omit when 'all' is true.",
                },
                "all": {
                    "type": "boolean",
                    "description": "Preflight every step (the CI merge-gate mode).",
                },
                "job_type": {
                    "type": "string",
                    "description": "Optional job_type variant for the interface gate.",
                },
                "workspace_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional workspace directories to widen discovery.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_preflight_step,
        tags=("validator",),
    ),
    ToolDef(
        name="author.config_constraints",
        description=(
            "List a step's config-class fields WITH their legal VALUES — each field's "
            "allowed_values (from a Literal/Enum type or a @field_validator allowed-set) + a "
            "case_sensitive flag, plus the required_no_default fields a value MUST be supplied for. "
            "Use this when authoring config VALUES: catalog.config_fields/config.requirements give "
            "only the field TYPE, which hid the enum/case constraints behind the largest empirical "
            "config-bug cluster. Sourced from the live config class so guidance == enforcement."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical step name whose config class to introspect "
                    "(e.g. 'TabularPreprocessing').",
                },
            },
            "required": ["step_name"],
            "additionalProperties": False,
        },
        handler=_config_constraints,
        tags=("validator",),
    ),
    ToolDef(
        name="author.preflight_config",
        description=(
            "Validate a set of config VALUES against the step's live config class "
            "(config_class.model_validate) — the gate that catches wrong enum case, invalid enum, "
            "wrong type, and missing required fields BEFORE a pipeline run. This is the value-level "
            "complement to author.preflight_step (which proves the step is constructible but never "
            "instantiates a config with user values). Returns per-field errors on failure."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical step name whose config class to validate against.",
                },
                "values": {
                    "type": "object",
                    "description": "The {field: value} config values to validate (the entry you "
                    "would put in config.json for this step).",
                    "additionalProperties": True,
                },
            },
            "required": ["step_name", "values"],
            "additionalProperties": False,
        },
        handler=_preflight_config,
        tags=("validator",),
    ),
    ToolDef(
        name="author.check_script",
        description=(
            "Check a step's SCRIPT against its contract in BOTH directions: forward (main() "
            "signature + the script's used keys are declared) AND reverse (every CLI arg the builder "
            "passes is parsed by the script's argparse; every required env var is actually read). "
            "Catches the two empirical script↔builder bugs — a custom script missing the --job_type "
            "argparse the builder passes (crash), and a required env var declared but read-and-"
            "ignored — that validate.step_interface + author.preflight_step cannot. The Cat-4 arg "
            "check runs only for Processing steps (Estimators take args via JSON). Skips script-less "
            "steps (SDK-delegation / CreateModel / Transform). Offline, read-only."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical step name whose script + contract to check "
                    "(e.g. 'TabularPreprocessing').",
                },
            },
            "required": ["step_name"],
            "additionalProperties": False,
        },
        handler=_check_script,
        tags=("validator",),
    ),
]
