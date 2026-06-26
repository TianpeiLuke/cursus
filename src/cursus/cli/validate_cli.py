"""
CLI commands for local pipeline-script testing.

`cursus validate run-scripts` executes a DAG's pipeline scripts locally, in dependency
order, with data-flow simulation between steps — a fast pre-deployment check that the
scripts actually run and hand data to each other. Wraps
``validation.script_testing.api.run_dag_scripts``. Engine imports are lazy.

Examples:
    cursus validate run-scripts dag.json -c config.json
    cursus validate run-scripts dag.json -c config.json --workspace-dir ./test_ws
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


def main():
    """Main entry point for the validate CLI."""
    return validate_cli()


if __name__ == "__main__":
    main()
