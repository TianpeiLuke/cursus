"""
CLI commands for local pipeline-script testing + step-interface validation.

`cursus validate run-scripts` executes a DAG's pipeline scripts locally, in dependency
order, with data-flow simulation between steps — a fast pre-deployment check that the
scripts actually run and hand data to each other. Wraps
``validation.script_testing.api.run_dag_scripts``. Engine imports are lazy.

`cursus validate step-interface` validates a `.step.yaml` at AUTHOR time (FZ 31e1d3f2): it
loads the interface through the production `StepInterface.from_yaml` path (surfacing Pydantic +
intra-step alignment errors) and runs incompleteness checks (compute descriptor coherence,
compatible_sources case-typos) — so an editor catches mistakes before a build/load trips them.

Examples:
    cursus validate run-scripts dag.json -c config.json
    cursus validate step-interface XGBoostTraining
    cursus validate step-interface RiskTableMapping --job-type validation
    cursus validate step-interface --all            # validate every .step.yaml (CI)
"""

import json
import logging

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group(name="validate")
def validate_cli():
    """Local pipeline-script testing and validation."""


@validate_cli.command(name="run-scripts")
@click.argument("dag_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--config-file",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the pipeline configuration JSON file.",
)
@click.option(
    "--workspace-dir",
    type=click.Path(),
    default="test/integration/script_testing",
    show_default=True,
    help="Working directory for script execution artifacts.",
)
@click.option(
    "--no-dependency-resolution",
    is_flag=True,
    help="Disable automatic data-flow dependency resolution between scripts.",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def run_scripts(dag_file, config_file, workspace_dir, no_dependency_resolution, format):
    """
    Execute a DAG's pipeline scripts locally in dependency order.

    DAG_FILE: path to a serialized DAG JSON file.

    Runs each script with simulated data flow so you can verify the scripts work together
    before deploying to SageMaker. Exits nonzero if any script fails.
    """
    fmt = format.lower()
    try:
        from ..api.dag import import_dag_from_json
        from ..validation.script_testing.api import run_dag_scripts

        dag = import_dag_from_json(dag_file)
        results = run_dag_scripts(
            dag=dag,
            config_path=config_file,
            test_workspace_dir=workspace_dir,
            use_dependency_resolution=not no_dependency_resolution,
        )
    except Exception as e:
        click.echo(f"❌ Failed to run DAG scripts: {e}", err=True)
        logger.error("run-scripts error", exc_info=True)
        raise SystemExit(1)

    if fmt == "json":
        click.echo(json.dumps(results, indent=2, default=str))
    else:
        click.echo(f"Script testing results for {dag_file}:")
        # results shape varies; surface a concise summary if recognizable.
        if isinstance(results, dict):
            for key in ("total", "passed", "failed", "errors", "success"):
                if key in results:
                    click.echo(f"  {key}: {results[key]}")
            if not any(
                k in results for k in ("total", "passed", "failed", "errors", "success")
            ):
                click.echo(json.dumps(results, indent=2, default=str))

    # Exit nonzero if the results indicate failures.
    if isinstance(results, dict):
        failed = results.get("failed") or results.get("errors")
        success = results.get("success")
        if (failed and failed > 0) or success is False:
            raise SystemExit(1)


def _validate_one_interface(step_name, job_type=None):
    """Validate a single .step.yaml. Returns {step, ok, errors:[...], warnings:[...]}.

    errors = blocking (FileNotFound / Pydantic ValidationError / cross-section misalignment, all
    raised by StepInterface.from_yaml). warnings = non-blocking incompleteness/quality findings
    (compatible_sources case-typos that silently weaken edges, FZ 31e1d3f gap 3).
    """
    from ..steps.interfaces import load_interface

    result = {"step": step_name, "job_type": job_type, "ok": True, "errors": [], "warnings": []}
    try:
        iface = load_interface(step_name, job_type=job_type)
    except FileNotFoundError as e:
        result["ok"] = False
        result["errors"].append(f"not found: {e}")
        return result
    except Exception as e:  # Pydantic ValidationError / _sync_and_align alignment errors
        result["ok"] = False
        result["errors"].append(f"{type(e).__name__}: {e}")
        return result

    # --- incompleteness / quality checks (non-blocking) ---
    # compatible_sources case-typos: an entry that ci-matches a real step but differs in case
    # silently loses the resolver's +10% bonus (FZ 31e1d3f gap 3).
    try:
        from ..registry.step_names import get_step_names

        names = set(get_step_names())
        lower = {n.lower(): n for n in names}
        for dep in iface.spec.dependencies.values():
            for src in dep.compatible_sources or []:
                if src not in names and lower.get(src.lower(), src) != src:
                    result["warnings"].append(
                        f"dep '{dep.logical_name}': compatible_sources {src!r} looks like a case "
                        f"typo of {lower[src.lower()]!r} (would silently lose the resolver bonus)"
                    )
    except Exception:
        pass  # registry unavailable — skip the optional check, don't fail validation

    return result


@validate_cli.command(name="step-interface")
@click.argument("step_name", required=False)
@click.option("--job-type", default=None, help="Resolve a job_type variant (e.g. validation).")
@click.option("--all", "validate_all", is_flag=True, help="Validate every .step.yaml interface (CI).")
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def step_interface(step_name, job_type, validate_all, format):
    """
    Validate a step's `.step.yaml` interface at author time (FZ 31e1d3f2).

    STEP_NAME: canonical step name (e.g. XGBoostTraining). Omit with --all to validate every step.

    Loads the interface through the production `StepInterface.from_yaml` path — surfacing Pydantic
    field errors + the contract↔spec cross-section alignment check — then runs incompleteness checks
    (compatible_sources case-typos). Exits nonzero if any interface has a blocking error.
    """
    fmt = format.lower()
    if not step_name and not validate_all:
        click.echo("❌ provide a STEP_NAME or --all", err=True)
        raise SystemExit(2)

    if validate_all:
        from ..steps.interfaces import list_available_interfaces

        targets = sorted(list_available_interfaces())
    else:
        targets = [step_name]

    results = [_validate_one_interface(t, job_type) for t in targets]
    n_err = sum(1 for r in results if not r["ok"])
    n_warn = sum(len(r["warnings"]) for r in results)

    if fmt == "json":
        click.echo(json.dumps(
            {"validated": len(results), "errors": n_err, "warnings": n_warn, "results": results},
            indent=2, default=str,
        ))
    else:
        for r in results:
            mark = "✅" if r["ok"] and not r["warnings"] else ("❌" if not r["ok"] else "⚠️ ")
            jt = f" [{r['job_type']}]" if r["job_type"] else ""
            click.echo(f"{mark} {r['step']}{jt}")
            for e in r["errors"]:
                click.echo(f"     ERROR: {e}")
            for w in r["warnings"]:
                click.echo(f"     warn:  {w}")
        click.echo(f"\nvalidated {len(results)} · {n_err} error(s) · {n_warn} warning(s)")

    if n_err:
        raise SystemExit(1)


def main():
    """Main entry point for the validate CLI."""
    return validate_cli()


if __name__ == "__main__":
    main()
