"""
Per-step connection / I-O view (FZ 31e1d3d / 31e1d3b1 follow-up).

The Strategy+Facade collapse means a step is no longer a readable builder class — the
container source/destination paths, the runtime property-path references, and the (handler-derived)
nested training-channel expansion all live in the step's ``.step.yaml`` + its bound handler instead.
This module reads BOTH and renders one structured "what wires into / out of this step" view that the
``cursus steps io`` CLI and the ``steps.io`` MCP tool share.

It is the path/wiring analogue of ``catalog.step_spec`` (which gives the ports but NOT the container
paths or the channel fan-out): for each dependency it reports ``container_path`` (where the input
lands in the container) + the SageMaker training channel(s) it maps to; for each output it reports
``container_path`` (source) + ``property_path`` (the runtime ``properties.*`` reference a downstream
step resolves against). Pure introspection — no config, no SageMaker session.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _channels_for(
    sagemaker_step_type: str,
    logical_name: str,
    container_path: Optional[str],
    declared_channels: Optional[List[str]],
) -> Optional[List[str]]:
    """The SageMaker training channel name(s) a dependency maps to, for Training steps only.

    Delegates to ``TrainingHandler.channels_for`` — the SINGLE SOURCE of the channel rule that
    ``TrainingHandler.get_inputs`` uses at build time, which itself prefers the channels declared
    in the ``.step.yaml`` — so this view can never drift from what the builder emits. Returns
    ``None`` for non-Training steps (no fan-out concept).
    """
    if sagemaker_step_type != "Training" or not container_path:
        return None
    from ...core.base.builder_templates import TrainingHandler

    return TrainingHandler.channels_for(logical_name, container_path, declared_channels)


def describe_step_io(step_name: str, job_type: Optional[str] = None) -> Dict[str, Any]:
    """Return the connection/I-O view for one step, resolved from its ``.step.yaml`` (+ job_type).

    Raises whatever ``load_interface`` raises for an unknown step (FileNotFoundError) — callers
    convert it to their own not-found envelope.
    """
    from . import load_interface  # lazy: avoids import cycle at module load
    from ...registry.step_names import get_sagemaker_step_type

    iface = load_interface(step_name, job_type=job_type)
    try:
        sm_type = get_sagemaker_step_type(step_name)
    except Exception:
        sm_type = None

    in_paths = iface.contract.expected_input_paths
    out_paths = iface.contract.expected_output_paths
    declared_channels = iface.contract.input_channels

    inputs: List[Dict[str, Any]] = []
    for logical_name, dep in iface.spec.dependencies.items():
        container_path = in_paths.get(logical_name)
        entry: Dict[str, Any] = {
            "logical_name": logical_name,
            "container_path": container_path,  # where the input lands inside the container
            "required": dep.required,
            "type": getattr(getattr(dep, "dependency_type", None), "value", None)
            or str(getattr(dep, "dependency_type", "")),
            "compatible_sources": list(getattr(dep, "compatible_sources", []) or []),
            "semantic_keywords": list(getattr(dep, "semantic_keywords", []) or []),
        }
        channels = _channels_for(
            sm_type, logical_name, container_path, declared_channels.get(logical_name)
        )
        if channels is not None:
            entry["channels"] = (
                channels  # nested SageMaker training channels (train/val/test, ...)
            )
        inputs.append(entry)

    outputs: List[Dict[str, Any]] = []
    for logical_name, out in iface.spec.outputs.items():
        outputs.append(
            {
                "logical_name": logical_name,
                "container_path": out_paths.get(
                    logical_name
                ),  # source inside the container
                "property_path": getattr(out, "property_path", "")
                or "",  # runtime reference
                "type": getattr(getattr(out, "output_type", None), "value", None)
                or str(getattr(out, "output_type", "")),
                "aliases": list(getattr(out, "aliases", []) or []),
                "data_type": getattr(out, "data_type", None),
            }
        )

    return {
        "step_name": step_name,
        "step_type": iface.spec.step_type,
        "sagemaker_step_type": sm_type,
        "job_type": job_type,
        "inputs": inputs,
        "outputs": outputs,
    }


# Axes whose pattern the patterns-view reports + the builder method each maps to (for the
# custom-override marker). Order = construction flow.
_PATTERN_AXES = (
    ("create_step", "create_step"),
    ("env_vars", "_get_environment_variables"),
    ("job_arguments", "_get_job_arguments"),
    ("inputs", "_get_inputs"),
    ("outputs", "_get_outputs"),
)
# Compute-factory methods (any of these on a builder = a per-step compute override). Mapped to the
# 'compute' axis in the patterns view.
_COMPUTE_METHODS = (
    "_create_processor",
    "_get_processor",
    "_create_estimator",
    "_create_model",
    "_create_transformer",
)


def _scan_builder_overrides(step_name: str, method_names):
    """Import-free fallback: AST-scan the step's builder ``.py`` for which of ``method_names`` it
    defines in its class body. Used when the builder class can't be imported (SAIS SDK absent), so
    the patterns view stays faithful for SDK-bound steps. Returns ``(overridden_set, scanned_bool)``.
    """
    import ast
    import glob

    try:
        from ...step_catalog.naming import canonical_to_snake

        snake = canonical_to_snake(step_name)
    except Exception:
        snake = step_name.lower()
    matches = glob.glob(f"src/cursus/steps/builders/builder_{snake}_step.py")
    if not matches:
        # fall back to STEP_NAME grep across builder files
        for f in glob.glob("src/cursus/steps/builders/builder_*_step.py"):
            try:
                if f'STEP_NAME = "{step_name}"' in open(f).read():
                    matches = [f]
                    break
            except Exception:
                continue
    if not matches:
        return set(), False
    try:
        tree = ast.parse(open(matches[0]).read())
    except Exception:
        return set(), False
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.body:
                if isinstance(b, ast.FunctionDef) and b.name in method_names:
                    found.add(b.name)
    return found, True


def describe_step_patterns(
    step_name: str, job_type: Optional[str] = None
) -> Dict[str, Any]:
    """Return the per-axis PATTERN view for one step — the 'plugins' the TemplateStepBuilder uses
    (FZ 31e1d3j).

    Everything is DERIVED from the data that actually drives the build — the registry binding
    (``sagemaker_step_type`` + ``step_assembly`` → handler/verb) and the per-step ``.step.yaml``
    contract DATA the handlers read (``env_vars``, ``job_arguments``, ``circular_ref_check``,
    ``skip_inputs``, ``input_source_overrides``, ``sink``, ``include_job_type_in_path``,
    ``source_dir``) — plus the output S3 prefix DERIVED from the step name
    (``canonical_to_snake(step_type)``). So this view CANNOT drift from behavior: there is no separate
    ``patterns:`` field — the pattern is the consequence of the data.

    Where a builder still hand-overrides a method (a genuine per-step deviation), that axis is marked
    ``custom_override`` so the user sees exactly where the step departs from the declarative patterns.
    """
    from . import load_interface  # lazy: avoids import cycle at module load
    from ...registry.step_names import get_sagemaker_step_type

    iface = load_interface(step_name, job_type=job_type)
    c = iface.contract
    try:
        sm_type = get_sagemaker_step_type(step_name)
    except Exception:
        sm_type = None
    # Fall back to the .step.yaml registry block (now a real section) when the registry lookup is
    # unavailable (e.g. offline) — the YAML is the same source of truth and keeps the view populated.
    if not sm_type:
        sm_type = getattr(getattr(iface, "registry", None), "sagemaker_step_type", None)

    # --- which builder methods does this step still hand-override? (genuine keeps) ---
    # Tiered so the answer is faithful in every state (FZ 31e1d3g3 Phase E deleted the per-step
    # builder files — most steps are now SYNTHESIZED fileless shells with NO overrides by definition):
    #   1. load the class and inspect __dict__ (most accurate — incl. a synthesized shell, which has
    #      no own methods, so overridden = {} correctly);
    #   2. if the class won't load (SAIS SDK absent offline), STATICALLY scan a builder .py if one is
    #      still on disk (a not-yet-deleted hand-written builder) — an import-free fallback;
    #   3. if neither loads nor has a source file, it is a pure declarative shell → no overrides.
    # We never silently report "no override" when we couldn't actually check (case 2 vs 3 distinguished).
    overridden: set = set()
    builder_note = None
    method_names = [m for _, m in _PATTERN_AXES] + list(_COMPUTE_METHODS)
    builder_cls = None
    try:
        from ...step_catalog.step_catalog import StepCatalog

        builder_cls = StepCatalog().load_builder_class(step_name)
    except Exception:
        builder_cls = None
    if builder_cls is not None:
        overridden = {
            m for m in method_names if m in getattr(builder_cls, "__dict__", {})
        }
    else:
        overridden, scanned = _scan_builder_overrides(step_name, method_names)
        builder_note = (
            "builder class not importable here (e.g. SAIS SDK absent); custom_override determined "
            "by static source scan of the on-disk builder file"
            if scanned
            else "declarative shell — no per-step builder file (synthesized at runtime); no overrides"
        )

    # --- create_step: the bound construction handler (the registry binding) ---
    reg = getattr(iface, "registry", None)
    # step_assembly is now declared in patterns: (the blueprint axis read by _auto_bind_handler);
    # prefer it so this view binds from the SAME source as the build (closes the FZ 31e1d3f drift),
    # falling back to the legacy registry.step_assembly for any not-yet-migrated YAML.
    pat = getattr(iface, "patterns", None)
    step_assembly = getattr(pat, "step_assembly", None) or getattr(
        reg, "step_assembly", None
    )
    create_step_requires = getattr(reg, "requires", "none")
    handler_name = None
    try:
        from ...core.base.builder_templates import resolve_handler

        h = (
            resolve_handler(sm_type, step_assembly)
            if sm_type == "Processing"
            else resolve_handler(sm_type)
        )
        handler_name = type(h).__name__
    except Exception:
        handler_name = None

    patterns: Dict[str, Any] = {}

    patterns["create_step"] = {
        "handler": handler_name,
        "sagemaker_step_type": sm_type,
        "step_assembly": step_assembly,
        # dependency axis: the BUILD-time 3rd-party pkg the create_step pattern needs (FZ 31e1d3l).
        # "secure_ai_sandbox_workflow_python_sdk" = a hard module-level SAIS import (SDKDelegation
        # steps); "none" = native (sagemaker-only). Declared in .step.yaml registry.requires.
        "requires": create_step_requires,
        "custom_override": "create_step" in overridden,
    }

    # --- compute: the declared SDK processor/estimator the template builds (FZ 31e1d3k) ---
    # compute is a TOP-LEVEL .step.yaml section now; read it off the interface (falls back to the
    # contract mirror for any caller that passes only a contract).
    cspec = getattr(iface, "compute", None) or getattr(c, "compute", None)
    compute_kind = getattr(cspec, "kind", None)
    # map kind -> the SDK class surfaced to the user
    _kind_to_class = {
        "sklearn": "SKLearnProcessor",
        "xgboost": "XGBoostProcessor",
        "script": "ScriptProcessor",
        "framework": "FrameworkProcessor",
        "estimator": getattr(cspec, "sdk_class", None),
        "model": getattr(cspec, "sdk_class", None),
        "transformer": "Transformer",
    }
    compute_override = (
        "_create_processor" in overridden
        or "_get_processor" in overridden
        or "_create_estimator" in overridden
        or "_create_model" in overridden
        or "_create_transformer" in overridden
    )
    compute_requires = getattr(cspec, "requires", "none")
    patterns["compute"] = {
        "kind": compute_kind,
        "sdk_class": _kind_to_class.get(compute_kind) if compute_kind else None,
        "framework_version_field": getattr(cspec, "framework_version_field", None),
        "lock_training_region": getattr(cspec, "lock_training_region", False),
        # dependency axis: the BUILD-time 3rd-party pkg this compute pattern needs (FZ 31e1d3l).
        # "mods_workflow_core" = the script/kms_network ScriptProcessor path (lazy import, no
        # fallback); "none" = sagemaker-only. Declared in .step.yaml contract.compute.requires.
        "requires": compute_requires,
        "source": "config-built by builder_base._create_compute() from the compute descriptor"
        if compute_kind
        else "per-step factory (no compute descriptor)",
        "custom_override": compute_override,
    }

    # --- env vars: declared names (config supplies values) ---
    patterns["env_vars"] = {
        "source": "config.get_environment_variables() (config is the value source)",
        "declared_required": list(getattr(c, "required_env_vars", []) or []),
        "declared_optional": sorted((getattr(c, "optional_env_vars", {}) or {}).keys()),
        "custom_override": "_get_environment_variables" in overridden,
    }

    # --- job arguments: declarative flag→source ---
    patterns["job_arguments"] = {
        "source": "config.get_job_arguments() (config is the value source)",
        "declared": [
            {"flag": d.flag, "source": d.source}
            for d in getattr(c, "job_arguments", [])
        ],
        "custom_override": "_get_job_arguments" in overridden,
    }

    # --- inputs: the standard loop + which deviation flags are active ---
    in_devs = []
    if getattr(c, "circular_ref_check", False):
        in_devs.append("circular_ref_check")
    if getattr(c, "skip_inputs", None):
        in_devs.append(f"skip_inputs={list(c.skip_inputs)}")
    if getattr(c, "input_source_overrides", None):
        in_devs.append(f"input_source_overrides={dict(c.input_source_overrides)}")
    if getattr(c, "source_dir", False):
        in_devs.append("source_dir (split entry_point + source_dir)")
    patterns["inputs"] = {
        "pattern": "standard spec×contract loop"
        + (" + " + ", ".join(in_devs) if in_devs else ""),
        "deviations": in_devs,
        "custom_override": "_get_inputs" in overridden,
    }

    # --- outputs: sink vs generated-destination token/job_type ---
    # The S3 prefix token is DERIVED from the step name (canonical_to_snake) by default, unless the
    # contract sets an OPT-IN output_path_token override (FZ 31e1d3f1b re-introduced, default-off).
    # include_job_type_in_path remains a per-step knob.
    if getattr(c, "sink", False):
        out_pattern = "sink (no outputs)"
        _token = None
        _token_source = None
    else:
        from ...step_catalog.naming import canonical_to_snake

        _override = getattr(c, "output_path_token", None)
        _token = _override or canonical_to_snake(iface.spec.step_type)
        _token_source = (
            "override: contract.output_path_token"
            if _override
            else "derived: canonical_to_snake(step_type)"
        )
        token = _token
        jt = getattr(c, "include_job_type_in_path", True)
        out_pattern = f"generated destination Join(base, {token}{', job_type' if jt else ''}, logical_name)"
    patterns["outputs"] = {
        "pattern": out_pattern,
        "output_path_token": _token,
        "output_path_token_source": _token_source,
        "include_job_type_in_path": getattr(c, "include_job_type_in_path", True),
        "sink": getattr(c, "sink", False),
        "custom_override": "_get_outputs" in overridden,
    }

    # --- dependency-axis rollup: the step's whole 3rd-party footprint at a glance (FZ 31e1d3l),
    # BUILD-time (create_step / compute) kept separate from RUNTIME (the container script). This is
    # the mods-vs-native split the user asked to surface — sourced entirely from .step.yaml DATA. ---
    runtime_requires = list(getattr(c, "runtime_requires", []) or [])
    build_time = {}
    if create_step_requires and create_step_requires != "none":
        build_time["create_step"] = create_step_requires
    if compute_requires and compute_requires != "none":
        build_time["compute"] = compute_requires
    result: Dict[str, Any] = {
        "step_name": step_name,
        "step_type": iface.spec.step_type,
        "sagemaker_step_type": sm_type,
        "job_type": job_type,
        "patterns": patterns,
        "dependencies": {
            "build_time": build_time,  # {axis: pkg} — absent axes are native (sagemaker-only)
            "runtime": runtime_requires,  # pkgs the script needs in-container (NOT at build time)
            "native": not build_time and not runtime_requires,
        },
    }
    if builder_note:
        result["note"] = builder_note
    return result
