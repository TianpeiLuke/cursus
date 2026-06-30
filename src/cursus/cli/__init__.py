"""Command-line interface for the Cursus package.

A single root ``click`` group composes the subcommand groups defined in the sibling
modules. Install the package to get a ``cursus`` console command (see ``[project.scripts]``
in pyproject.toml); ``python -m cursus.cli`` works too.

    cursus --help
    cursus catalog list --framework xgboost
    cursus compile -d dag.json -c config.json
    cursus projects list --root ./projects
"""

import click

from .alignment_cli import alignment
from .catalog_cli import catalog_cli
from .compile_cli import compile_pipeline
from .config_cli import config_cli
from .dag_cli import dag_cli
from .exec_doc_cli import exec_doc_cli
from .mcp_cli import mcp_cli
from .pipeline_catalog_cli import pipeline_catalog_cli
from .project_cli import projects_cli
from .registry_cli import registry_cli
from .steps_cli import steps_cli
from .strategies_cli import strategies_cli
from .validate_cli import validate_cli

__all__ = ["cli", "main"]


@click.group()
@click.version_option(package_name="cursus", message="cursus %(version)s")
def cli():
    """Cursus — specification-driven SageMaker pipeline development and validation."""


# Register each subcommand group/command under its canonical name. The objects already
# declare their own name= where it differs from the function name, so add_command()
# composes them natively (no argparse shim, no sys.argv mutation).
cli.add_command(alignment, name="alignment")
cli.add_command(catalog_cli, name="catalog")
cli.add_command(compile_pipeline, name="compile")
cli.add_command(config_cli, name="config")
cli.add_command(dag_cli, name="dag")
cli.add_command(exec_doc_cli, name="exec-doc")
cli.add_command(mcp_cli, name="mcp")
cli.add_command(pipeline_catalog_cli, name="pipeline-catalog")
cli.add_command(projects_cli, name="projects")
cli.add_command(registry_cli, name="registry")
cli.add_command(steps_cli, name="steps")
cli.add_command(strategies_cli, name="strategies")
cli.add_command(validate_cli, name="validate")

# Backward-compat shim: `from cursus.cli import main` and `python -m cursus.cli` keep
# working for one release. Kept thin so it can be removed later.
main = cli
