"""
CLI commands for the pre-built pipeline catalog.

`cursus pipeline-catalog recommend` ranks the catalog's pre-built shared DAGs against
your requirements (data type, labels, LLM need, framework, ...). It wraps the existing
``pipeline_catalog.core.agent_tool.pipeline_catalog_tool`` so the CLI and the
agent/MCP surfaces share one recommendation engine. Engine imports are lazy.

Examples:
    cursus pipeline-catalog recommend --data-type tabular --framework xgboost
    cursus pipeline-catalog list
    cursus pipeline-catalog get-dag <dag_id>
"""

import json
import logging

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group(name="pipeline-catalog")
def pipeline_catalog_cli():
    """Discover and recommend pre-built pipeline DAGs."""


@pipeline_catalog_cli.command(name="recommend")
@click.option(
    "--data-type",
    type=click.Choice(["text", "tabular", "mixed"], case_sensitive=False),
    help="Primary data type.",
)
@click.option(
    "--has-labels/--no-labels", default=True, help="Labeled data already exists."
)
@click.option("--needs-llm/--no-llm", default=False, help="LLM (Bedrock) needed.")
@click.option(
    "--multi-task/--single-task", default=False, help="Multiple output tasks."
)
@click.option(
    "--incremental/--first-time", default=False, help="Incremental retraining."
)
@click.option(
    "--framework",
    type=click.Choice(
        ["pytorch", "xgboost", "lightgbm", "lightgbmmt", "any"], case_sensitive=False
    ),
    help="Preferred ML framework.",
)
@click.option(
    "--gpu/--no-gpu", "gpu_available", default=True, help="GPU instances available."
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def recommend(
    data_type,
    has_labels,
    needs_llm,
    multi_task,
    incremental,
    framework,
    gpu_available,
    format,
):
    """Recommend pre-built pipeline DAGs for your requirements."""
    fmt = format.lower()
    try:
        from ..pipeline_catalog.core.agent_tool import pipeline_catalog_tool

        result = pipeline_catalog_tool(
            action="recommend",
            data_type=data_type.lower() if data_type else None,
            has_labels=has_labels,
            needs_llm=needs_llm,
            multi_task=multi_task,
            incremental=incremental,
            framework=framework.lower() if framework else None,
            gpu_available=gpu_available,
        )
    except Exception as e:
        click.echo(f"❌ Failed to get recommendations: {e}", err=True)
        logger.error("Pipeline-catalog recommend error", exc_info=True)
        raise SystemExit(1)

    if fmt == "json":
        click.echo(json.dumps(result, indent=2, default=str))
        return

    recs = result.get("recommendations", [])
    if not recs:
        click.echo("No matching pipeline DAGs found for those requirements.")
        return
    click.echo(f"Top {len(recs)} recommended pipeline DAG(s):\n")
    for i, rec in enumerate(recs, 1):
        click.echo(f"  {i}. {rec.get('dag_id')}  (score: {rec.get('score')})")
        if rec.get("framework"):
            click.echo(
                f"      framework: {rec['framework']} | nodes: {rec.get('node_count')}"
            )
        if rec.get("when_to_use"):
            click.echo(f"      when to use: {rec['when_to_use']}")


@pipeline_catalog_cli.command(name="list")
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def list_frameworks(format):
    """List the frameworks available across the pipeline catalog."""
    fmt = format.lower()
    try:
        from ..pipeline_catalog.core.agent_tool import pipeline_catalog_tool

        result = pipeline_catalog_tool(action="list_frameworks")
    except Exception as e:
        click.echo(f"❌ Failed to list catalog frameworks: {e}", err=True)
        logger.error("Pipeline-catalog list error", exc_info=True)
        raise SystemExit(1)

    if fmt == "json":
        click.echo(json.dumps(result, indent=2, default=str))
        return
    frameworks = result.get("frameworks", {})
    click.echo("Pipeline catalog frameworks (DAG counts):")
    for fw, count in sorted(frameworks.items()):
        click.echo(f"  {fw}: {count}")


@pipeline_catalog_cli.command(name="get-dag")
@click.argument("dag_id")
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="json",
    help="Output format.",
)
def get_dag(dag_id, format):
    """Show nodes/edges and requirements for a catalog DAG by DAG_ID."""
    try:
        from ..pipeline_catalog.core.agent_tool import pipeline_catalog_tool

        result = pipeline_catalog_tool(action="get_dag", dag_id=dag_id)
    except Exception as e:
        click.echo(f"❌ Failed to get DAG '{dag_id}': {e}", err=True)
        logger.error("Pipeline-catalog get-dag error", exc_info=True)
        raise SystemExit(1)

    if result.get("status") == "error":
        click.echo(f"❌ {result.get('message', 'DAG not found')}", err=True)
        raise SystemExit(1)
    click.echo(json.dumps(result, indent=2, default=str))


def main():
    """Main entry point for the pipeline-catalog CLI."""
    return pipeline_catalog_cli()


if __name__ == "__main__":
    main()
