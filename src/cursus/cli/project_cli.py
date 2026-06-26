"""
CLI command for discovering and inspecting Cursus pipeline projects.

The ``cursus projects`` command group answers "what pipeline projects exist under this
root, and what does each contain?" — for the ``projects/*`` folders under the
AmazonCursus root, or the per-pipeline folders under a consumer repo such as
BuyerAbuseModsTemplate.

A pipeline project is the standard deployable Cursus layout: a configuration directory
(``pipeline_config`` / ``pipeline_configs``) of config JSON files, alongside
``dockers``/``scripts`` and a pipeline-definition module. This command reads each
project's config JSON (no SageMaker/engine objects are constructed), so it is cheap and
safe to run anywhere.

Examples:
    cursus projects list --root ./projects
    cursus projects list --root ~/mods/src/BuyerAbuseModsTemplate/src/buyer_abuse_mods_template
    cursus projects show --root ./projects rnr_pytorch_bedrock
    cursus projects show munged_address_pytorch          # locate by name (generic search)
"""

import json
import logging
from typing import Optional

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group(name="projects")
def projects_cli():
    """Discover and inspect Cursus pipeline projects."""
    pass


@projects_cli.command(name="list")
@click.option(
    "--root",
    "-r",
    type=click.Path(exists=True, file_okay=False),
    help="Directory to scan for pipeline-project subdirectories.",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def list_projects(root: Optional[str], format: str):
    """
    List pipeline projects under a root directory.

    Scans each immediate subdirectory of --root and reports those that are recognizable
    Cursus pipeline projects (have a pipeline_config / pipeline_configs directory).
    """
    from ..core.utils import discover_pipeline_projects

    if not root:
        click.echo(
            "❌ --root is required for 'list' (the directory to scan).", err=True
        )
        raise SystemExit(1)

    fmt = format.lower()
    try:
        projects = discover_pipeline_projects(root=root)
    except Exception as e:
        click.echo(f"❌ Failed to discover projects under {root}: {e}", err=True)
        logger.error("Project discovery error", exc_info=True)
        raise SystemExit(1)

    if fmt == "json":
        click.echo(json.dumps([p.to_dict() for p in projects], indent=2))
        return 0

    if not projects:
        click.echo(f"No pipeline projects found under: {root}")
        return 0

    click.echo(f"Found {len(projects)} pipeline project(s) under {root}:\n")
    for p in projects:
        # Largest config (by node count) is the representative node total.
        max_nodes = max((c.node_count for c in p.config_files), default=0)
        click.echo(f"  {p.name}")
        click.echo(
            f"      configs: {p.config_file_count}"
            f" | nodes: {max_nodes}"
            f" | config types: {len(p.distinct_config_types)}"
            f" | dockers: {'yes' if p.has_dockers else 'no'}"
        )
    return 0


@projects_cli.command(name="show")
@click.argument("name")
@click.option(
    "--root",
    "-r",
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing the project (omit to locate it by name via generic search).",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def show_project(name: str, root: Optional[str], format: str):
    """
    Show details for one pipeline project NAME (nodes, config types, modules).

    With --root, looks for NAME under that directory; without --root, locates the
    uniquely-named project folder via cross-deployment generic discovery.
    """
    from ..core.utils import discover_pipeline_projects

    fmt = format.lower()
    try:
        matches = discover_pipeline_projects(root=root, names=[name])
    except Exception as e:
        click.echo(f"❌ Failed to discover project '{name}': {e}", err=True)
        logger.error("Project discovery error", exc_info=True)
        raise SystemExit(1)
    if not matches:
        click.echo(f"❌ Pipeline project '{name}' not found.", err=True)
        raise SystemExit(1)
    info = matches[0]

    if fmt == "json":
        click.echo(json.dumps(info.to_dict(), indent=2))
        return 0

    click.echo(f"Project: {info.name}")
    click.echo(f"  path: {info.path}")
    click.echo(f"  config dir: {info.config_dir}")
    click.echo(
        f"  dockers: {'yes' if info.has_dockers else 'no'} | scripts: {'yes' if info.has_scripts else 'no'}"
    )
    if info.pipeline_modules:
        click.echo(f"  pipeline modules: {', '.join(info.pipeline_modules)}")
    if info.distinct_config_types:
        click.echo(f"  config types ({len(info.distinct_config_types)}):")
        for t in info.distinct_config_types:
            click.echo(f"      - {t}")
    click.echo(f"  config files ({info.config_file_count}):")
    for c in info.config_files:
        if c.error:
            click.echo(f"      {c.file}: ⚠️  {c.error}")
        else:
            click.echo(f"      {c.file}: {c.node_count} node(s)")
    return 0


def main():
    """Main entry point for the projects CLI."""
    return projects_cli()


if __name__ == "__main__":
    main()
