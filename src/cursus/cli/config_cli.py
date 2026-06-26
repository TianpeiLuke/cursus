"""
CLI commands for inspecting the configuration a pipeline DAG requires.

`cursus config requirements` loads a DAG JSON and reports the configuration fields each
node needs (name, type, required, default, description) — the stateless introspection
side of ``api.factory.DAGConfigFactory``. Engine imports are lazy.

Note: actually *generating* a populated config set is an interactive, stateful workflow
(DAGConfigFactory.set_base_config / set_step_config across many calls, then
generate_all_configs), which does not map to a single one-shot command — use the
Python API or the config widget for that. This command covers the "what do I need to
provide?" question.

Examples:
    cursus config requirements dag.json
    cursus config requirements dag.json --step XGBoostTraining
    cursus config requirements dag.json --format json
"""

import logging

import click

from ._shared import echo_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group(name="config")
def config_cli():
    """Inspect the configuration a pipeline DAG requires."""


@config_cli.command(name="requirements")
@click.argument("dag_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--step",
    "step_name",
    help="Show requirements for one step only (default: base + all steps).",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def requirements(dag_file: str, step_name, format):
    """
    Show the configuration fields a DAG (or one of its steps) requires.

    DAG_FILE: path to a serialized DAG JSON file.
    """
    fmt = format.lower()
    try:
        from ..api.dag import import_dag_from_json
        from ..api.factory import DAGConfigFactory

        dag = import_dag_from_json(dag_file)
        factory = DAGConfigFactory(dag)

        if step_name:
            data = {
                "step": step_name,
                "fields": factory.get_step_requirements(step_name),
            }
        else:
            data = {
                "base": factory.get_base_config_requirements(),
                "steps": {
                    node: factory.get_step_requirements(node) for node in dag.nodes
                },
            }
    except Exception as e:
        click.echo(
            f"❌ Failed to read config requirements from '{dag_file}': {e}", err=True
        )
        logger.error("config requirements error", exc_info=True)
        raise SystemExit(1)

    if fmt == "json":
        echo_json(data)
        return

    def _print_fields(fields):
        for f in fields:
            req = "required" if f.get("required") else "optional"
            line = f"    - {f.get('name')} ({f.get('type')}) [{req}]"
            click.echo(line)
            if f.get("description"):
                click.echo(f"        {f['description']}")

    if step_name:
        click.echo(f"Config requirements for step '{step_name}':")
        _print_fields(data["fields"])
        return

    click.echo(f"Config requirements for DAG: {dag_file}\n")
    click.echo("Base pipeline config:")
    _print_fields(data["base"])
    click.echo("\nPer-step config:")
    for node, fields in data["steps"].items():
        click.echo(f"  {node}:")
        _print_fields(fields)


def main():
    """Main entry point for the config CLI."""
    return config_cli()


if __name__ == "__main__":
    main()
