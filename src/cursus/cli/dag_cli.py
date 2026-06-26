"""
CLI commands for inspecting and validating pipeline DAG JSON files.

`cursus dag validate` checks a serialized DAG (cycles, dangling edges, isolated nodes,
and — via the step catalog — that each node resolves to a known step) before you spend
time compiling it. Engine imports are lazy so `cursus --help` stays fast.

Examples:
    cursus dag validate dag.json
    cursus dag validate dag.json --format json
"""

import json
import logging

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group(name="dag")
def dag_cli():
    """Inspect and validate pipeline DAG JSON files."""


@dag_cli.command(name="validate")
@click.argument("dag_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def validate_dag(dag_file: str, format: str):
    """
    Validate the integrity of a serialized DAG JSON file.

    DAG_FILE: path to a DAG JSON file (as produced by the serializer / compiler).

    Reports cycles, dangling edges, isolated nodes, and nodes that do not resolve to a
    known step in the catalog. Exits nonzero if the DAG is invalid.
    """
    fmt = format.lower()
    try:
        from ..api.dag import import_dag_from_json, PipelineDAGResolver

        dag = import_dag_from_json(dag_file)
        resolver = PipelineDAGResolver(dag, validate_on_init=False)
        issues = resolver.validate_dag_integrity()
    except Exception as e:
        click.echo(f"❌ Failed to validate DAG '{dag_file}': {e}", err=True)
        logger.error("DAG validation error", exc_info=True)
        raise SystemExit(1)

    is_valid = not issues

    if fmt == "json":
        click.echo(
            json.dumps(
                {
                    "dag_file": dag_file,
                    "is_valid": is_valid,
                    "node_count": len(dag.nodes),
                    "edge_count": len(dag.edges),
                    "issues": issues,
                },
                indent=2,
                default=str,
            )
        )
        if not is_valid:
            raise SystemExit(1)
        return

    click.echo(f"DAG: {dag_file}")
    click.echo(f"  nodes: {len(dag.nodes)} | edges: {len(dag.edges)}")
    if is_valid:
        click.echo("✅ DAG is valid (no integrity issues found).")
        return

    click.echo("❌ DAG has integrity issues:")
    for category, items in issues.items():
        if items:
            click.echo(f"  {category}:")
            for item in items:
                click.echo(f"    - {item}")
    raise SystemExit(1)


def main():
    """Main entry point for the dag CLI."""
    return dag_cli()


if __name__ == "__main__":
    main()
