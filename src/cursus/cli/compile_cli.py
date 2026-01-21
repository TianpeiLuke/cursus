"""
CLI command for compiling pipeline DAGs to SageMaker pipelines.

This module provides the `cursus compile` command for compiling serialized
DAG JSON files and configuration JSON files into SageMaker Pipeline objects
with optional deployment and execution capabilities.
"""

import click
import json
import logging
from pathlib import Path
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.command(name="compile")
@click.option(
    "--dag-file",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to serialized DAG JSON file",
)
@click.option(
    "--config-file",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration JSON file",
)
@click.option(
    "--pipeline-name",
    "-n",
    type=str,
    help="Override pipeline name (optional)",
)
@click.option(
    "--role",
    type=str,
    help="IAM role ARN for pipeline execution (optional)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Save pipeline definition to JSON file (optional)",
)
@click.option(
    "--upsert",
    is_flag=True,
    help="Create/update pipeline in SageMaker service",
)
@click.option(
    "--start",
    is_flag=True,
    help="Start pipeline execution after upserting (requires --upsert)",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format for console display",
)
@click.option(
    "--show-report",
    is_flag=True,
    help="Display detailed compilation report",
)
@click.option(
    "--validate-only",
    is_flag=True,
    help="Only validate compatibility, don't compile",
)
def compile_pipeline(
    dag_file: str,
    config_file: str,
    pipeline_name: Optional[str],
    role: Optional[str],
    output: Optional[str],
    upsert: bool,
    start: bool,
    format: str,
    show_report: bool,
    validate_only: bool,
):
    """
    Compile DAG and config JSON files to SageMaker Pipeline.

    This command takes a serialized DAG JSON file and a configuration JSON file,
    then compiles them into a SageMaker Pipeline object. The pipeline can be
    saved to a file, deployed to SageMaker, and/or executed.

    Examples:

        # Basic compilation (console output only)
        cursus compile -d dag.json -c config.json

        # Save pipeline definition to file
        cursus compile -d dag.json -c config.json -o pipeline_definition.json

        # Deploy to SageMaker (upsert)
        cursus compile -d dag.json -c config.json --upsert

        # Complete workflow (compile + upsert + start)
        cursus compile -d dag.json -c config.json --upsert --start

        # Validation only
        cursus compile -d dag.json -c config.json --validate-only

        # With detailed report
        cursus compile -d dag.json -c config.json --show-report
    """
    try:
        # Import required modules
        from ..api.dag import import_dag_from_json
        from ..core.compiler import compile_dag_to_pipeline, PipelineDAGCompiler

        # Validate that --start requires --upsert
        if start and not upsert:
            click.echo("❌ Error: --start flag requires --upsert flag", err=True)
            return 1

        # Step 1: Load DAG from JSON
        try:
            dag = import_dag_from_json(dag_file)
            dag_nodes = len(dag.nodes)
            dag_edges = len(dag.edges)

            if format == "text":
                click.echo(f"✓ DAG loaded: {dag_nodes} nodes, {dag_edges} edges")
        except Exception as e:
            click.echo(f"❌ Failed to load DAG from {dag_file}: {e}", err=True)
            logger.error(f"DAG loading error: {e}", exc_info=True)
            return 1

        # Step 2: Validate configuration file exists
        config_path = Path(config_file)
        if not config_path.exists():
            click.echo(f"❌ Configuration file not found: {config_file}", err=True)
            return 1

        # Load config to get step count
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
                # Count configs (assuming top-level keys are step configs)
                config_count = len(
                    [k for k in config_data.keys() if not k.startswith("_")]
                )
            if format == "text":
                click.echo(f"✓ Config loaded: {config_count} step configurations")
        except Exception as e:
            click.echo(f"❌ Failed to load config from {config_file}: {e}", err=True)
            return 1

        # Step 3: Validation only mode
        if validate_only:
            try:
                compiler = PipelineDAGCompiler(
                    config_path=config_file,
                    role=role,
                )
                validation_result = compiler.validate_dag_compatibility(dag)

                if format == "json":
                    result = {
                        "status": "validation_complete",
                        "is_valid": validation_result.is_valid,
                        "dag_nodes": dag_nodes,
                        "dag_edges": dag_edges,
                        "missing_configs": validation_result.missing_configs,
                        "unresolvable_builders": validation_result.unresolvable_builders,
                        "warnings": validation_result.warnings,
                    }
                    click.echo(json.dumps(result, indent=2))
                else:
                    click.echo(f"\nValidation Results:")
                    if validation_result.is_valid:
                        click.echo("✓ All DAG nodes have matching configurations")
                        click.echo("✓ All step builders resolved successfully")
                        click.echo("✓ No dependency issues found")
                        click.echo(f"\nValidation passed! Ready for compilation.")
                    else:
                        click.echo("❌ Validation failed!")
                        if validation_result.missing_configs:
                            click.echo(f"\nMissing configurations:")
                            for node in validation_result.missing_configs:
                                click.echo(f"  - {node}")
                        if validation_result.unresolvable_builders:
                            click.echo(f"\nUnresolvable builders:")
                            for node in validation_result.unresolvable_builders:
                                click.echo(f"  - {node}")
                        if validation_result.warnings:
                            click.echo(f"\nWarnings:")
                            for warning in validation_result.warnings:
                                click.echo(f"  - {warning}")

                return 0 if validation_result.is_valid else 1

            except Exception as e:
                click.echo(f"❌ Validation failed: {e}", err=True)
                logger.error(f"Validation error: {e}", exc_info=True)
                return 1

        # Step 4: Compile pipeline
        try:
            if show_report:
                # Use compiler with report
                compiler = PipelineDAGCompiler(
                    config_path=config_file,
                    role=role,
                )
                pipeline, report = compiler.compile_with_report(
                    dag=dag,
                    pipeline_name=pipeline_name,
                )

                if format == "text":
                    click.echo(f"✓ Pipeline compiled successfully")
                    click.echo(f"\n📋 Compilation Report:")
                    click.echo(f"   Pipeline: {report.pipeline_name}")
                    click.echo(f"   Steps: {len(report.steps)}")
                    click.echo(f"   Average confidence: {report.avg_confidence:.2f}")
                    click.echo(f"   Warnings: {len(report.warnings)}")

                    if report.warnings:
                        click.echo(f"\n   Warnings:")
                        for warning in report.warnings:
                            click.echo(f"     - {warning}")

                    click.echo(f"\n   Resolution Details:")
                    for node, details in report.resolution_details.items():
                        config_type = details.get("config_type", "Unknown")
                        builder_type = details.get("builder_type", "Unknown")
                        confidence = details.get("confidence", 0.0)
                        click.echo(
                            f"     {node} → {config_type} ({builder_type}, confidence: {confidence:.2f})"
                        )
            else:
                # Standard compilation
                pipeline = compile_dag_to_pipeline(
                    dag=dag,
                    config_path=config_file,
                    role=role,
                    pipeline_name=pipeline_name,
                )

                if format == "text":
                    click.echo(f"✓ Pipeline compiled successfully")

            # Get pipeline details
            pipeline_name_final = pipeline.name
            steps_count = len(pipeline.steps) if hasattr(pipeline, "steps") else "N/A"

            if format == "text":
                click.echo(f"\nPipeline: {pipeline_name_final}")
                click.echo(f"Steps: {steps_count} SageMaker steps created")

        except Exception as e:
            click.echo(f"❌ Failed to compile pipeline: {e}", err=True)
            logger.error(f"Compilation error: {e}", exc_info=True)
            return 1

        # Step 5: Save pipeline definition to file
        if output:
            try:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Get pipeline definition as JSON
                pipeline_definition = pipeline.definition()

                # Write to file
                with open(output_path, "w") as f:
                    f.write(pipeline_definition)

                if format == "text":
                    click.echo(f"✓ Pipeline definition saved to: {output}")

            except Exception as e:
                click.echo(f"❌ Failed to save pipeline definition: {e}", err=True)
                logger.error(f"Output save error: {e}", exc_info=True)
                return 1

        # Step 6: Upsert to SageMaker
        if upsert:
            try:
                if format == "text":
                    click.echo(f"\nUpserting to SageMaker...")

                response = pipeline.upsert(role_arn=role)

                pipeline_arn = response.get("PipelineArn", "N/A")

                if format == "text":
                    click.echo(f"✓ Pipeline created/updated")
                    click.echo(f"  Pipeline Name: {pipeline_name_final}")
                    click.echo(f"  Pipeline ARN: {pipeline_arn}")

            except Exception as e:
                click.echo(f"❌ Failed to upsert pipeline: {e}", err=True)
                logger.error(f"Upsert error: {e}", exc_info=True)
                return 1

        # Step 7: Start execution
        if start:
            try:
                if format == "text":
                    click.echo(f"\nStarting execution...")

                execution = pipeline.start()

                execution_arn = execution.arn if hasattr(execution, "arn") else "N/A"
                execution_id = (
                    execution_arn.split("/")[-1] if execution_arn != "N/A" else "N/A"
                )

                if format == "text":
                    click.echo(f"✓ Execution started")
                    click.echo(f"  Execution ARN: {execution_arn}")
                    click.echo(f"  Execution ID: {execution_id}")
                    click.echo(f"  Status: Executing")

                    # Try to extract region from ARN for console link
                    if execution_arn != "N/A" and ":" in execution_arn:
                        parts = execution_arn.split(":")
                        if len(parts) >= 4:
                            region = parts[3]
                            pipeline_name_url = pipeline_name_final.lower()
                            click.echo(f"\nMonitor execution at:")
                            click.echo(
                                f"  https://console.aws.amazon.com/sagemaker/home?region={region}#/pipelines/{pipeline_name_url}/executions/{execution_id}"
                            )

            except Exception as e:
                click.echo(f"❌ Failed to start execution: {e}", err=True)
                logger.error(f"Execution start error: {e}", exc_info=True)
                return 1

        # Step 8: JSON output format
        if format == "json":
            result = {
                "status": "success",
                "pipeline_name": pipeline_name_final,
                "dag_nodes": dag_nodes,
                "dag_edges": dag_edges,
                "steps_created": steps_count,
            }

            if upsert:
                result["pipeline_arn"] = pipeline_arn

            if start:
                result["execution_arn"] = execution_arn
                result["execution_id"] = execution_id

            click.echo(json.dumps(result, indent=2))

        return 0

    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        logger.error(f"Unexpected error in compile command: {e}", exc_info=True)
        return 1


def main():
    """Main entry point for compile CLI."""
    return compile_pipeline()


if __name__ == "__main__":
    main()
