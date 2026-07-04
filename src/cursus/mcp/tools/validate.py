"""
``validate.*`` MCP tools — alignment, dependency, and script-execution checks.

This namespace wraps three cursus validation engines:
:class:`cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester`
(configuration-driven multi-level alignment validation),
:mod:`cursus.validation.script_testing.api` (``run_dag_scripts`` local script execution),
and the dependency resolver in :mod:`cursus.core.deps` (``create_pipeline_components`` +
``UnifiedDependencyResolver`` fed by specs loaded from the step catalog). It also exposes
the capability-metadata helpers ``get_validation_info`` / ``get_script_testing_info``.
All engine objects are converted to plain JSON-serializable dicts/lists before returning.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..envelope import ToolResult, ToolError
from ..registry import ToolDef

# One-line purpose of this namespace (collected by the registry for validate.help).
NAMESPACE = (
    "Alignment, dependency, and script-execution checks (validation, core.deps)."
)


def _alignment(args: Dict[str, Any]) -> ToolResult:
    """
    Run alignment validation across all configured levels and return summary metrics
    plus any critical issues.

    Optional ``workspace_dirs`` widens discovery beyond the package; ``target_scripts``
    restricts validation to a named subset of steps.
    """
    from ...validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

    workspace_dirs = args.get("workspace_dirs")
    target_scripts = args.get("target_scripts")

    if workspace_dirs is not None and not isinstance(workspace_dirs, list):
        raise ToolError(
            "'workspace_dirs' must be a list of directory paths",
            code="invalid_input",
        )
    if target_scripts is not None and not isinstance(target_scripts, list):
        raise ToolError(
            "'target_scripts' must be a list of step names",
            code="invalid_input",
        )

    tester = UnifiedAlignmentTester(workspace_dirs=workspace_dirs)

    if target_scripts:
        # run_full_validation returns {step_name: per-step result dict} for a subset.
        results = tester.run_full_validation(target_scripts=target_scripts)
        passed = sum(1 for r in results.values() if r.get("overall_status") == "PASSED")
        failed = sum(
            1
            for r in results.values()
            if (r.get("overall_status") or r.get("status")) == "FAILED"
        )
        excluded = sum(1 for r in results.values() if r.get("status") == "EXCLUDED")
        data: Dict[str, Any] = {
            "scope": "target_scripts",
            "target_scripts": list(target_scripts),
            "total_steps": len(results),
            "passed_steps": passed,
            "failed_steps": failed,
            "excluded_steps": excluded,
            "detailed_results": results,
        }
        return ToolResult.success(data, tool_scope="subset")

    # Full validation across all discovered steps: summary + critical issues.
    summary = tester.get_validation_summary()
    critical_issues = tester.get_critical_issues()

    data = {
        "scope": "all_steps",
        "total_steps": summary.get("total_steps", 0),
        "passed_steps": summary.get("passed_steps", 0),
        "failed_steps": summary.get("failed_steps", 0),
        "excluded_steps": summary.get("excluded_steps", 0),
        "pass_rate": summary.get("pass_rate", 0),
        "step_type_breakdown": summary.get("step_type_breakdown", {}),
        "critical_issues": critical_issues,
        "critical_issue_count": len(critical_issues),
    }
    return ToolResult.success(data)


def _step(args: Dict[str, Any]) -> ToolResult:
    """
    Run alignment validation for a single named step and return its per-level result.
    """
    from ...validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

    step_name = args["step_name"]
    workspace_dirs = args.get("workspace_dirs")

    if not isinstance(step_name, str) or not step_name.strip():
        raise ToolError("'step_name' must be a non-empty string", code="invalid_input")
    if workspace_dirs is not None and not isinstance(workspace_dirs, list):
        raise ToolError(
            "'workspace_dirs' must be a list of directory paths",
            code="invalid_input",
        )

    tester = UnifiedAlignmentTester(workspace_dirs=workspace_dirs)
    result = tester.run_validation_for_step(step_name)

    # The tester returns a plain dict (it never raises for an unknown step — it produces
    # a result with ERROR levels), so surface it directly.
    return ToolResult.success(result)


def _step_interface(args: Dict[str, Any]) -> ToolResult:
    """
    Validate a step's ``.step.yaml`` interface at AUTHOR time (FZ 31e1d3f2 / 31e1d3f5).

    This is the agent-callable promotion of ``cursus validate step-interface``: it wraps the
    CLI's ``_validate_one_interface`` / ``list_available_interfaces`` VERBATIM, so the agent
    gate, the CLI gate, and the CI ``validate step-interface --all`` gate are literally one
    code path. Each interface is loaded through the production ``StepInterface.from_yaml``
    path (surfacing Pydantic field errors + the contract↔spec ``_sync_and_align`` check) and
    run through the incompleteness checks (``compatible_sources`` case-typos).

    Pass ``step_name`` for one step (optionally a ``job_type`` variant), or ``all=true`` to
    validate every ``.step.yaml`` (the CI mode). Read-only and offline-safe.
    """
    # Reuse the CLI helpers verbatim — zero new validation logic (single source of truth).
    from ...cli.validate_cli import _validate_one_interface

    step_name = args.get("step_name")
    validate_all = bool(args.get("all", False))
    job_type = args.get("job_type")

    if not validate_all and (not isinstance(step_name, str) or not step_name.strip()):
        raise ToolError(
            "provide 'step_name' (a canonical step name) or set 'all' to true",
            code="invalid_input",
        )
    if job_type is not None and not isinstance(job_type, str):
        raise ToolError("'job_type' must be a string", code="invalid_input")

    if validate_all:
        from ...steps.interfaces import list_available_interfaces

        targets = sorted(list_available_interfaces())
    else:
        targets = [step_name]

    results = [_validate_one_interface(t, job_type) for t in targets]
    n_err = sum(1 for r in results if not r["ok"])
    n_warn = sum(len(r["warnings"]) for r in results)

    data = {
        "validated": len(results),
        "errors": n_err,
        "warnings": n_warn,
        "results": results,
    }

    if n_err:
        # Blocking error(s): point the agent at the read tools that fix the offending
        # section, mirroring the loop the dev-guide + FZ 31e1d3f5 describe.
        return ToolResult.failure(
            f"{n_err} interface(s) failed validation",
            code="validation_failed",
            details=data,
            remedy={
                "suggested_tools": [
                    "strategies.for_step_type",
                    "catalog.config_fields",
                    "validate.deps_explain",
                ],
                "fix_action": (
                    "Read each failing result's 'errors': a Pydantic/section error means "
                    "fix that field in the .step.yaml; a contract↔spec mismatch means align "
                    "the contract.inputs/outputs keys with spec.dependencies/outputs; a "
                    "compatible_sources warning means correct the case to the exact step "
                    "name. Re-run validate.step_interface, then author.preflight_step."
                ),
            },
        )

    # Clean: the natural next step in the author->validate->integrate loop.
    next_steps = [
        {
            "tool": "validate.alignment",
            "when": "the interface is valid and you want the full constructibility proof",
            "why": "interface validation is the author-time gate; alignment validate-all "
            "(B1/B2/B3) proves the step is constructible, not merely parseable",
        }
    ]
    warnings = (
        [f"{n_warn} non-blocking incompleteness warning(s) across interfaces"]
        if n_warn
        else None
    )
    return ToolResult.success(data, warnings=warnings, next_steps=next_steps)


def _deps_resolve(args: Dict[str, Any]) -> ToolResult:
    """
    Resolve declarative dependencies among a set of registered steps.

    Loads each step's specification from the step catalog, registers them in a fresh
    in-process :class:`UnifiedDependencyResolver`, then reports which dependencies could
    be matched to compatible outputs of the other steps. If ``step_names`` is omitted,
    every step in the catalog that has a specification is used.
    """
    from ...core.deps.factory import create_pipeline_components
    from ...step_catalog import StepCatalog

    step_names = args.get("step_names")
    workspace_dirs = args.get("workspace_dirs")

    if step_names is not None and not isinstance(step_names, list):
        raise ToolError(
            "'step_names' must be a list of step names", code="invalid_input"
        )
    if workspace_dirs is not None and not isinstance(workspace_dirs, list):
        raise ToolError(
            "'workspace_dirs' must be a list of directory paths",
            code="invalid_input",
        )

    catalog = StepCatalog(workspace_dirs=workspace_dirs)

    if not step_names:
        step_names = catalog.list_steps_with_specs()

    if not step_names:
        return ToolResult.failure(
            "no steps with specifications available to resolve",
            code="not_found",
        )

    components = create_pipeline_components()
    resolver = components["resolver"]

    registered: List[str] = []
    missing_specs: List[str] = []
    for name in step_names:
        spec = catalog.load_spec_class(name)
        if spec is None:
            missing_specs.append(name)
            continue
        resolver.register_specification(name, spec)
        registered.append(name)

    if not registered:
        return ToolResult.failure(
            "could not load a specification for any requested step",
            code="not_found",
            details={"missing_specs": missing_specs},
        )

    # get_resolution_report returns a fully JSON-safe report (property references are
    # stringified internally).
    report = resolver.get_resolution_report(registered)

    data = {
        "requested_steps": list(step_names),
        "registered_steps": registered,
        "missing_specs": missing_specs,
        "report": report,
    }
    warnings = (
        [f"no specification found for {len(missing_specs)} requested step(s)"]
        if missing_specs
        else None
    )
    return ToolResult.success(data, warnings=warnings)


def _run_scripts(args: Dict[str, Any]) -> ToolResult:
    """
    Execute the scripts of a DAG locally against a pipeline config and report per-script
    outcomes. Heavy: this actually imports and runs each step's script (and may pip-install
    declared imports). Non-destructive — runs locally, mutates no external/cloud state.
    """
    from ...api.dag.base_dag import PipelineDAG
    from ...validation.script_testing.api import run_dag_scripts, ScriptTestResult

    nodes = args["nodes"]
    config_path = args["config_path"]
    edges = args.get("edges") or []
    test_workspace_dir = args.get(
        "test_workspace_dir", "test/integration/script_testing"
    )
    use_dependency_resolution = args.get("use_dependency_resolution", True)

    if not isinstance(nodes, list) or not all(isinstance(n, str) for n in nodes):
        raise ToolError(
            "'nodes' must be a list of step-name strings", code="invalid_input"
        )
    if not nodes:
        raise ToolError("'nodes' must contain at least one node", code="invalid_input")
    if not isinstance(config_path, str) or not config_path.strip():
        raise ToolError(
            "'config_path' must be a non-empty string", code="invalid_input"
        )

    # Edges arrive as [from, to] pairs (JSON has no tuples); the DAG accepts tuples.
    edge_tuples: List[tuple] = []
    for edge in edges:
        if (
            not isinstance(edge, (list, tuple))
            or len(edge) != 2
            or not all(isinstance(e, str) for e in edge)
        ):
            raise ToolError(
                "each entry in 'edges' must be a [from, to] pair of step names",
                code="invalid_input",
            )
        edge_tuples.append((edge[0], edge[1]))

    dag = PipelineDAG(nodes=list(nodes), edges=edge_tuples)

    try:
        raw = run_dag_scripts(
            dag=dag,
            config_path=config_path,
            test_workspace_dir=test_workspace_dir,
            use_dependency_resolution=use_dependency_resolution,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise ToolError(str(exc), code="tool_error")

    # raw["script_results"] maps node -> ScriptTestResult, which is not JSON-serializable.
    script_results: Dict[str, Any] = {}
    for node_name, result in (raw.get("script_results") or {}).items():
        if isinstance(result, ScriptTestResult):
            script_results[node_name] = {
                "success": result.success,
                "output_files": result.output_files,
                "error_message": result.error_message,
                "execution_time": result.execution_time,
            }
        else:
            script_results[node_name] = str(result)

    data = {
        "pipeline_success": bool(raw.get("pipeline_success", False)),
        "execution_order": raw.get("execution_order", []),
        "total_scripts": raw.get("total_scripts", 0),
        "successful_scripts": raw.get("successful_scripts", 0),
        "script_results": script_results,
        "execution_summary": raw.get("execution_summary"),
    }
    return ToolResult.success(data)


def _builder(args: Dict[str, Any]) -> ToolResult:
    """
    Run the universal builder test suite for a single step's builder: builder↔config
    alignment, integration, step-creation, and (optionally) quality scoring. Returns the
    structured result dict from :class:`UniversalStepBuilderTest`.
    """
    from ...validation.builders.universal_test import UniversalStepBuilderTest

    step_name = args["step_name"]
    workspace_dirs = args.get("workspace_dirs")
    enable_scoring = args.get("enable_scoring", True)

    if not isinstance(step_name, str) or not step_name.strip():
        raise ToolError("'step_name' must be a non-empty string", code="invalid_input")
    if workspace_dirs is not None and not isinstance(workspace_dirs, list):
        raise ToolError(
            "'workspace_dirs' must be a list of directory paths",
            code="invalid_input",
        )

    tester = UniversalStepBuilderTest(
        workspace_dirs=workspace_dirs,
        enable_scoring=bool(enable_scoring),
    )
    # run_validation_for_step returns a plain dict (it does not raise for an unknown step —
    # it produces a result describing the failure), so surface it directly.
    result = tester.run_validation_for_step(step_name)
    return ToolResult.success(result, step_name=step_name)


def _deps_explain(args: Dict[str, Any]) -> ToolResult:
    """
    Explain the semantic similarity between two names: the overall score plus the
    per-component breakdown (string / token-overlap / synonym / substring) the dependency
    resolver uses for matching. Useful for debugging why two step/port names did or did
    not match.
    """
    from ...core.deps.semantic_matcher import SemanticMatcher

    name1 = args["name1"]
    name2 = args["name2"]
    if not isinstance(name1, str) or not name1.strip():
        raise ToolError("'name1' must be a non-empty string", code="invalid_input")
    if not isinstance(name2, str) or not name2.strip():
        raise ToolError("'name2' must be a non-empty string", code="invalid_input")

    matcher = SemanticMatcher()
    explanation = matcher.explain_similarity(name1, name2)
    return ToolResult.success(explanation, name1=name1, name2=name2)


def _info(args: Dict[str, Any]) -> ToolResult:
    """Return capability metadata for the validation and script-testing frameworks."""
    from ...validation import get_validation_info
    from ...validation.script_testing import get_script_testing_info

    data = {
        "validation": get_validation_info(),
        "script_testing": get_script_testing_info(),
    }
    return ToolResult.success(data)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[ToolDef] = [
    ToolDef(
        name="validate.alignment",
        description=(
            "Run configuration-driven alignment validation across all levels and return "
            "summary metrics (pass/fail/excluded counts, pass rate, step-type breakdown) "
            "plus critical issues. Pass 'target_scripts' to validate only specific steps."
        ),
        schema={
            "type": "object",
            "properties": {
                "target_scripts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of step names to validate; "
                    "omit to validate all discovered steps.",
                },
                "workspace_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional workspace directories to widen step "
                    "discovery beyond the installed package.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_alignment,
        tags=("validator",),
        when=(
            "Call this before compiling to prove steps are constructible — run all levels "
            "and get pass/fail counts and critical issues, optionally scoped to a subset."
        ),
        examples=(
            "validate.alignment {}  # validate every discovered step",
            'validate.alignment {"target_scripts": ["TabularPreprocessing", "XGBoostTraining"]}  # only these two steps',
            'validate.alignment {"target_scripts": ["XGBoostTraining"], "workspace_dirs": ["/path/to/project"]}  # widen discovery to a project dir',
        ),
    ),
    ToolDef(
        name="validate.step",
        description=(
            "Run alignment validation for a single named step and return its per-level "
            "result (status + any errors per validation level)."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical step name to validate (e.g. "
                    "'TabularPreprocessing').",
                },
                "workspace_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional workspace directories to widen step "
                    "discovery beyond the installed package.",
                },
            },
            "required": ["step_name"],
            "additionalProperties": False,
        },
        handler=_step,
        tags=("validator",),
        when=(
            "Call this to drill into one step's per-level result after a full "
            "validate.alignment run flagged it, or when iterating on a single step."
        ),
        examples=(
            'validate.step {"step_name": "TabularPreprocessing"}  # per-level result for one step',
            'validate.step {"step_name": "XGBoostTraining", "workspace_dirs": ["/path/to/project"]}  # widen discovery to a project dir',
        ),
    ),
    ToolDef(
        name="validate.step_interface",
        description=(
            "Validate a step's .step.yaml interface at AUTHOR time — the agent-callable form "
            "of `cursus validate step-interface`. Loads the interface through the production "
            "StepInterface.from_yaml path (Pydantic field errors + contract<->spec alignment) "
            "and flags compatible_sources case-typos. Pass 'step_name' (optionally 'job_type') "
            "for one step, or 'all'=true to validate every interface (the CI gate). This is the "
            "author-time gate before author.preflight_step / validate.alignment."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical step name to validate (e.g. "
                    "'XGBoostTraining'). Omit when 'all' is true.",
                },
                "all": {
                    "type": "boolean",
                    "description": "Validate every .step.yaml interface (CI mode). "
                    "When true, 'step_name' is ignored.",
                },
                "job_type": {
                    "type": "string",
                    "description": "Optional job_type variant to resolve (e.g. "
                    "'validation', 'calibration').",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_step_interface,
        tags=("validator",),
        when=(
            "Call this at AUTHOR time right after writing/editing a .step.yaml — the first "
            "gate before author.preflight_step and validate.alignment; or 'all'=true in CI."
        ),
        examples=(
            'validate.step_interface {"step_name": "XGBoostTraining"}  # validate one step interface',
            'validate.step_interface {"step_name": "TabularPreprocessing", "job_type": "calibration"}  # a specific job_type variant',
            'validate.step_interface {"all": true}  # CI mode: validate every .step.yaml',
        ),
    ),
    ToolDef(
        name="validate.deps_resolve",
        description=(
            "Resolve declarative dependencies among a set of steps using their catalog "
            "specifications: reports which required/optional dependencies match compatible "
            "outputs of the other steps, plus an overall resolution rate. Omit 'step_names' "
            "to use every catalog step that has a specification."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps to include in the resolution graph; omit to use "
                    "all steps with specifications.",
                },
                "workspace_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional workspace directories to widen step "
                    "discovery beyond the installed package.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_deps_resolve,
        tags=("validator",),
        when=(
            "Call this to check whether a set of steps wires up — which required/optional "
            "dependencies match compatible outputs of the others — before assembling a DAG."
        ),
        examples=(
            "validate.deps_resolve {}  # resolve across every catalog step with a spec",
            'validate.deps_resolve {"step_names": ["TabularPreprocessing", "XGBoostTraining", "XGBoostModelEval"]}  # just this pipeline',
            'validate.deps_resolve {"step_names": ["XGBoostTraining"], "workspace_dirs": ["/path/to/project"]}  # widen discovery to a project dir',
        ),
    ),
    ToolDef(
        name="validate.run_scripts",
        description=(
            "Execute a DAG's step scripts locally against a pipeline config file and return "
            "per-script success/error and execution order. Heavy: it imports and runs each "
            "script (and may pip-install their imports). Non-destructive — runs locally only."
        ),
        schema={
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Step names that make up the DAG (at least one).",
                },
                "config_path": {
                    "type": "string",
                    "description": "Path to the pipeline configuration JSON used to "
                    "resolve and validate each step's script.",
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "description": "Optional directed edges as [from_step, to_step] pairs.",
                },
                "test_workspace_dir": {
                    "type": "string",
                    "description": "Directory for the test workspace "
                    "(default 'test/integration/script_testing').",
                },
                "use_dependency_resolution": {
                    "type": "boolean",
                    "description": "Whether to use two-phase dependency resolution "
                    "(default true).",
                },
            },
            "required": ["nodes", "config_path"],
            "additionalProperties": False,
        },
        handler=_run_scripts,
        tags=("validator",),
        when=(
            "Call this to actually run a DAG's step scripts locally against a config file "
            "(a heavier end-to-end check) once the DAG and configs are ready."
        ),
        examples=(
            'validate.run_scripts {"nodes": ["TabularPreprocessing"], "config_path": "pipeline_config/config.json"}  # run one script',
            'validate.run_scripts {"nodes": ["TabularPreprocessing", "XGBoostTraining"], "edges": [["TabularPreprocessing", "XGBoostTraining"]], "config_path": "pipeline_config/config.json"}  # run a two-step chain',
            'validate.run_scripts {"nodes": ["XGBoostTraining"], "config_path": "pipeline_config/config.json", "use_dependency_resolution": false}  # skip two-phase dependency resolution',
        ),
    ),
    ToolDef(
        name="validate.builder",
        description=(
            "Run the universal builder test suite for one step's builder — builder↔config "
            "alignment, integration, step-creation, and (by default) quality scoring — and "
            "return the structured result. Use to validate a builder before registration."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical step name whose builder to test (e.g. "
                    "'XGBoostTraining').",
                },
                "enable_scoring": {
                    "type": "boolean",
                    "description": "Include quality scoring in the result (default true).",
                },
                "workspace_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional workspace directories to widen step "
                    "discovery beyond the installed package.",
                },
            },
            "required": ["step_name"],
            "additionalProperties": False,
        },
        handler=_builder,
        tags=("validator",),
        when=(
            "Call this to test one step's builder (builder<->config alignment, integration, "
            "step-creation, quality score) before registering or trusting that builder."
        ),
        examples=(
            'validate.builder {"step_name": "XGBoostTraining"}  # full builder suite with scoring',
            'validate.builder {"step_name": "TabularPreprocessing", "enable_scoring": false}  # skip the quality score',
            'validate.builder {"step_name": "XGBoostTraining", "workspace_dirs": ["/path/to/project"]}  # widen discovery to a project dir',
        ),
    ),
    ToolDef(
        name="validate.deps_explain",
        description=(
            "Explain the semantic similarity between two names: overall score plus the "
            "per-component breakdown (string / token / synonym / substring) the dependency "
            "resolver uses. Use to debug why two step or port names did or did not match."
        ),
        schema={
            "type": "object",
            "properties": {
                "name1": {
                    "type": "string",
                    "description": "First name to compare.",
                },
                "name2": {
                    "type": "string",
                    "description": "Second name to compare.",
                },
            },
            "required": ["name1", "name2"],
            "additionalProperties": False,
        },
        handler=_deps_explain,
        tags=("validator",),
        when=(
            "Call this to debug a dependency match: get the overall + per-component semantic "
            "similarity score for two names when a deps_resolve match looks wrong or missing."
        ),
        examples=(
            'validate.deps_explain {"name1": "processed_data", "name2": "input_data"}  # why two port names did/did not match',
            'validate.deps_explain {"name1": "TabularPreprocessing", "name2": "XGBoostTraining"}  # compare two step names',
        ),
    ),
    ToolDef(
        name="validate.info",
        description=(
            "Return capability metadata for the validation and script-testing frameworks "
            "(available components, supported user stories, key features)."
        ),
        schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=_info,
        tags=("validator",),
        when=(
            "Call this to discover what the validation and script-testing frameworks can do "
            "(available components, supported user stories, key features) before using them."
        ),
        examples=(
            "validate.info {}  # capability metadata for the validation frameworks",
        ),
    ),
]
