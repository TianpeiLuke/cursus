"""
CLI commands for execution document generation.

This module provides command-line tools for generating execution documents
from serialized DAG and configuration files, with full MODS integration.
"""

import json
import logging
import click
from pathlib import Path
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group(name="exec-doc")
def exec_doc_cli():
    """Execution document generation commands."""
    pass


@exec_doc_cli.command("generate")
@click.option(
    "--dag-file",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to serialized DAG JSON file",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration JSON file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="execution_doc.json",
    help="Output file path (default: execution_doc.json)",
)
@click.option(
    "--template",
    type=click.Path(exists=True),
    help="Base execution document template file (optional)",
)
@click.option(
    "--format",
    type=click.Choice(["json", "yaml"]),
    default="json",
    help="Output format (default: json)",
)
@click.option(
    "--role",
    help="IAM role ARN for AWS operations (optional)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output with detailed processing logs",
)
def generate_exec_doc(
    dag_file: str,
    config_file: str,
    output: str,
    template: Optional[str],
    format: str,
    role: Optional[str],
    verbose: bool,
):
    """
    Generate execution document from DAG and configuration files.

    This command takes a serialized DAG JSON file and a configuration JSON file,
    then generates an execution document with step-specific configurations filled
    by specialized helpers (Cradle, Registration, etc.).

    Examples:

        # Basic usage
        cursus exec-doc generate -d dag.json -c config.json

        # With custom output
        cursus exec-doc generate -d dag.json -c config.json -o my_exec_doc.json

        # With template
        cursus exec-doc generate -d dag.json -c config.json --template base_template.json

        # YAML output
        cursus exec-doc generate -d dag.json -c config.json --format yaml

        # With IAM role
        cursus exec-doc generate -d dag.json -c config.json --role arn:aws:iam::123:role/MyRole
    """
    try:
        # Set logging level based on verbose flag
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        click.echo("\n🔧 Execution Document Generation")
        click.echo("=" * 40)

        # Step 1: Load DAG from JSON
        click.echo(f"\n📂 Loading DAG from: {dag_file}")
        try:
            from ..api.dag.pipeline_dag_serializer import import_dag_from_json

            dag = import_dag_from_json(dag_file)
            click.echo(f"✓ DAG loaded: {len(dag.nodes)} nodes, {len(dag.edges)} edges")

            if verbose:
                click.echo(f"  Nodes: {list(dag.nodes.keys())}")

        except Exception as e:
            click.echo(f"❌ Failed to load DAG: {e}")
            logger.error(f"DAG loading error: {e}", exc_info=True)
            return

        # Step 2: Create or load base execution document
        click.echo(f"\n📋 Preparing execution document template")

        if template:
            click.echo(f"  Loading template from: {template}")
            try:
                with open(template, "r") as f:
                    execution_document = json.load(f)
                click.echo(f"✓ Template loaded")
            except Exception as e:
                click.echo(f"❌ Failed to load template: {e}")
                logger.error(f"Template loading error: {e}", exc_info=True)
                return
        else:
            click.echo(f"  Auto-generating base template from DAG")
            execution_document = {
                "PIPELINE_STEP_CONFIGS": {
                    node: {"STEP_CONFIG": {}, "STEP_TYPE": []} for node in dag.nodes
                }
            }
            click.echo(f"✓ Base template generated with {len(dag.nodes)} steps")

        # Step 3: Initialize ExecutionDocumentGenerator
        click.echo(f"\n⚙️  Initializing generator")
        click.echo(f"  Config file: {config_file}")
        if role:
            click.echo(f"  IAM role: {role}")

        try:
            from ..mods.exe_doc.generator import ExecutionDocumentGenerator

            generator = ExecutionDocumentGenerator(
                config_path=config_file,
                role=role,
            )

            config_count = len(generator.configs)
            click.echo(f"✓ Generator initialized with {config_count} configurations")

            if verbose:
                click.echo(f"  Loaded configs: {list(generator.configs.keys())}")

        except Exception as e:
            click.echo(f"❌ Failed to initialize generator: {e}")
            logger.error(f"Generator initialization error: {e}", exc_info=True)
            return

        # Step 4: Fill execution document
        click.echo(f"\n🔄 Filling execution document")

        try:
            filled_doc = generator.fill_execution_document(dag, execution_document)
            click.echo(f"✓ Execution document generated successfully")

            # Show processing summary
            if "PIPELINE_STEP_CONFIGS" in filled_doc:
                steps_with_config = sum(
                    1
                    for step_config in filled_doc["PIPELINE_STEP_CONFIGS"].values()
                    if step_config.get("STEP_CONFIG")
                )
                click.echo(f"\n📊 Processing Summary:")
                click.echo(f"  Total steps: {len(filled_doc['PIPELINE_STEP_CONFIGS'])}")
                click.echo(f"  Steps with configuration: {steps_with_config}")

                if verbose:
                    click.echo(f"\n  Configured steps:")
                    for step_name, step_config in filled_doc[
                        "PIPELINE_STEP_CONFIGS"
                    ].items():
                        if step_config.get("STEP_CONFIG"):
                            config_keys = len(step_config["STEP_CONFIG"])
                            click.echo(f"    - {step_name}: {config_keys} parameters")

        except Exception as e:
            click.echo(f"❌ Failed to fill execution document: {e}")
            logger.error(f"Execution document generation error: {e}", exc_info=True)
            return

        # Step 5: Save output
        click.echo(f"\n💾 Saving execution document")
        click.echo(f"  Output file: {output}")
        click.echo(f"  Format: {format}")

        try:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                with open(output_path, "w") as f:
                    json.dump(filled_doc, f, indent=2)
            elif format == "yaml":
                import yaml

                with open(output_path, "w") as f:
                    yaml.dump(filled_doc, f, default_flow_style=False)

            click.echo(f"✓ Execution document saved to: {output}")

            # Show file size
            file_size = output_path.stat().st_size
            click.echo(f"  File size: {file_size:,} bytes")

        except Exception as e:
            click.echo(f"❌ Failed to save execution document: {e}")
            logger.error(f"Output saving error: {e}", exc_info=True)
            return

        # Success summary
        click.echo(f"\n✅ Execution document generation complete!")
        click.echo(f"\nNext steps:")
        click.echo(f"  1. Review the generated execution document: {output}")
        click.echo(f"  2. Use with MODS for pipeline execution")

    except Exception as e:
        click.echo(f"\n❌ Unexpected error: {e}")
        logger.error(
            f"Unexpected error during execution document generation: {e}", exc_info=True
        )


def main():
    """Main entry point for exec-doc CLI."""
    return exec_doc_cli()


if __name__ == "__main__":
    main()
