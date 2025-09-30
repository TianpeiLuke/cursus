"""Enhanced command-line interface for pipeline runtime testing with step catalog integration."""

import click
import json
import sys
import os
from pathlib import Path

from ..validation.runtime.runtime_testing import RuntimeTester
from ..validation.runtime.runtime_models import (
    RuntimeTestingConfiguration,
    ScriptExecutionSpec,
    PipelineTestingSpec,
)
from ..validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder
from ..api.dag.base_dag import PipelineDAG

# Optional step catalog import
try:
    from ..step_catalog import StepCatalog
    STEP_CATALOG_AVAILABLE = True
except ImportError:
    StepCatalog = None
    STEP_CATALOG_AVAILABLE = False


@click.group()
@click.version_option(version="0.1.0")
def runtime():
    """Pipeline Runtime Testing CLI - Simplified

    Test individual scripts and complete pipelines for functionality
    and data flow compatibility.
    """
    pass


@runtime.command()
@click.argument("script_name")
@click.option(
    "--workspace-dir",
    default="./test_workspace",
    help="Workspace directory for test execution",
)
@click.option(
    "--step-catalog/--no-step-catalog",
    default=True,
    help="Enable/disable step catalog integration for enhanced testing",
)
@click.option(
    "--workspace-dirs",
    help="Comma-separated list of additional workspace directories for step catalog",
)
@click.option(
    "--output-format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format for results",
)
def test_script(script_name: str, workspace_dir: str, step_catalog: bool, workspace_dirs: str, output_format: str):
    """Test a single script functionality with optional step catalog enhancements

    SCRIPT_NAME: Name of the script to test
    """

    try:
        # Initialize step catalog if requested and available
        step_catalog_instance = None
        if step_catalog and STEP_CATALOG_AVAILABLE:
            workspace_list = [Path(workspace_dir)]
            if workspace_dirs:
                workspace_list.extend([Path(w.strip()) for w in workspace_dirs.split(',')])
            
            # Add environment workspaces
            env_workspaces = os.environ.get('CURSUS_DEV_WORKSPACES', '').split(':')
            for workspace in env_workspaces:
                if workspace and Path(workspace).exists():
                    workspace_list.append(Path(workspace))
            
            try:
                step_catalog_instance = StepCatalog(workspace_dirs=workspace_list)
            except Exception:
                # Silently ignore step catalog initialization errors
                pass

        # Initialize RuntimeTester and PipelineTestingSpecBuilder with step catalog
        tester = RuntimeTester(workspace_dir, step_catalog=step_catalog_instance)
        builder = PipelineTestingSpecBuilder(test_data_dir=workspace_dir, step_catalog=step_catalog_instance)
        
        # Build ScriptExecutionSpec for the script using node resolution
        script_spec = builder.resolve_script_execution_spec_from_node(script_name)
        
        # Get main function parameters
        main_params = builder.get_script_main_params(script_spec)
        
        # Test the script with step catalog enhancements if available
        if step_catalog_instance:
            result = tester.test_script_with_step_catalog_enhancements(script_spec, main_params)
        else:
            result = tester.test_script_with_spec(script_spec, main_params)

        if output_format == "json":
            click.echo(json.dumps(result.model_dump(), indent=2))
        else:
            status_color = "green" if result.success else "red"
            click.echo(f"Script: {script_name}")
            click.echo(f"Status: ", nl=False)
            click.secho(
                "PASS" if result.success else "FAIL", fg=status_color, bold=True
            )
            click.echo(f"Execution time: {result.execution_time:.3f}s")
            click.echo(
                f"Has main function: {'Yes' if result.has_main_function else 'No'}"
            )
            
            # Show step catalog enhancements if available
            if step_catalog_instance:
                click.echo(f"Step catalog: {'Enabled' if step_catalog_instance else 'Disabled'}")
                
                # Show framework detection if available
                framework = tester._detect_framework_if_needed(script_spec)
                if framework:
                    click.echo(f"Detected framework: {framework}")
                
                # Show builder consistency warnings if available
                warnings = tester._validate_builder_consistency_if_available(script_spec)
                if warnings:
                    click.echo("Builder consistency warnings:")
                    for warning in warnings:
                        click.echo(f"  - {warning}")

            if result.error_message:
                click.echo(f"Error: {result.error_message}")

        sys.exit(0 if result.success else 1)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@runtime.command()
@click.argument("pipeline_config")
@click.option(
    "--workspace-dir",
    default="./test_workspace",
    help="Workspace directory for test execution",
)
@click.option(
    "--step-catalog/--no-step-catalog",
    default=True,
    help="Enable/disable step catalog integration for enhanced pipeline testing",
)
@click.option(
    "--workspace-dirs",
    help="Comma-separated list of additional workspace directories for step catalog",
)
@click.option(
    "--output-format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format for results",
)
def test_pipeline(pipeline_config: str, workspace_dir: str, step_catalog: bool, workspace_dirs: str, output_format: str):
    """Test complete pipeline flow

    PIPELINE_CONFIG: Path to pipeline configuration file (JSON)
    """

    try:
        # Load pipeline configuration
        config_path = Path(pipeline_config)
        if not config_path.exists():
            click.echo(f"Pipeline config file not found: {pipeline_config}", err=True)
            sys.exit(1)

        with open(config_path) as f:
            config_data = json.load(f)

        # Initialize step catalog if requested and available
        step_catalog_instance = None
        if step_catalog and STEP_CATALOG_AVAILABLE:
            workspace_list = [Path(workspace_dir)]
            if workspace_dirs:
                workspace_list.extend([Path(w.strip()) for w in workspace_dirs.split(',')])
            
            # Add environment workspaces
            env_workspaces = os.environ.get('CURSUS_DEV_WORKSPACES', '').split(':')
            for workspace in env_workspaces:
                if workspace and Path(workspace).exists():
                    workspace_list.append(Path(workspace))
            
            try:
                step_catalog_instance = StepCatalog(workspace_dirs=workspace_list)
            except Exception:
                # Silently ignore step catalog initialization errors
                pass

        # Initialize RuntimeTester and PipelineTestingSpecBuilder with step catalog
        tester = RuntimeTester(workspace_dir, step_catalog=step_catalog_instance)
        builder = PipelineTestingSpecBuilder(test_data_dir=workspace_dir, step_catalog=step_catalog_instance)
        
        # Create PipelineDAG from config data
        # Assuming config_data has 'nodes' and 'edges' fields
        if 'nodes' not in config_data or 'edges' not in config_data:
            click.echo("Pipeline config must contain 'nodes' and 'edges' fields", err=True)
            sys.exit(1)
            
        dag = PipelineDAG()
        for node in config_data['nodes']:
            dag.add_node(node)
        for edge in config_data['edges']:
            if isinstance(edge, list) and len(edge) == 2:
                dag.add_edge(edge[0], edge[1])
            elif isinstance(edge, dict) and 'source' in edge and 'target' in edge:
                dag.add_edge(edge['source'], edge['target'])
        
        # Build PipelineTestingSpec from DAG
        pipeline_spec = builder.build_from_dag(dag, validate=False)
        
        # Test the pipeline with step catalog enhancements if available
        if step_catalog_instance:
            results = tester.test_pipeline_flow_with_step_catalog_enhancements(pipeline_spec)
        else:
            results = tester.test_pipeline_flow_with_spec(pipeline_spec)

        if output_format == "json":
            # Convert Pydantic models to dict for JSON serialization
            json_results = {
                "pipeline_success": results["pipeline_success"],
                "errors": results["errors"],
            }

            # Convert script results
            json_results["script_results"] = {}
            for script_name, result in results["script_results"].items():
                json_results["script_results"][script_name] = result.model_dump()

            # Convert data flow results
            json_results["data_flow_results"] = {}
            for flow_name, result in results["data_flow_results"].items():
                json_results["data_flow_results"][flow_name] = result.model_dump()

            # Add execution order if available
            if "execution_order" in results:
                json_results["execution_order"] = results["execution_order"]

            click.echo(json.dumps(json_results, indent=2))
        else:
            # Text output
            status_color = "green" if results["pipeline_success"] else "red"
            click.echo(f"Pipeline: {pipeline_config}")
            click.echo(f"Status: ", nl=False)
            click.secho(
                "PASS" if results["pipeline_success"] else "FAIL",
                fg=status_color,
                bold=True,
            )

            # Show execution order if available
            if "execution_order" in results:
                click.echo(f"Execution order: {' -> '.join(results['execution_order'])}")

            click.echo("\nScript Results:")
            for script_name, result in results["script_results"].items():
                script_color = "green" if result.success else "red"
                click.echo(f"  {script_name}: ", nl=False)
                click.secho("PASS" if result.success else "FAIL", fg=script_color)
                if not result.success:
                    click.echo(f"    Error: {result.error_message}")

            click.echo("\nData Flow Results:")
            for flow_name, result in results["data_flow_results"].items():
                flow_color = "green" if result.compatible else "red"
                click.echo(f"  {flow_name}: ", nl=False)
                click.secho("PASS" if result.compatible else "FAIL", fg=flow_color)
                if result.compatibility_issues:
                    for issue in result.compatibility_issues:
                        click.echo(f"    Issue: {issue}")

            if results["errors"]:
                click.echo("\nErrors:")
                for error in results["errors"]:
                    click.echo(f"  - {error}")

        sys.exit(0 if results["pipeline_success"] else 1)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@runtime.command()
@click.argument("script_a")
@click.argument("script_b")
@click.option(
    "--workspace-dir",
    default="./test_workspace",
    help="Workspace directory for test execution",
)
@click.option(
    "--step-catalog/--no-step-catalog",
    default=True,
    help="Enable/disable step catalog integration for enhanced compatibility testing",
)
@click.option(
    "--workspace-dirs",
    help="Comma-separated list of additional workspace directories for step catalog",
)
@click.option(
    "--output-format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format for results",
)
def test_compatibility(
    script_a: str, script_b: str, workspace_dir: str, step_catalog: bool, workspace_dirs: str, output_format: str
):
    """Test data compatibility between two scripts with optional step catalog enhancements

    SCRIPT_A: First script name
    SCRIPT_B: Second script name
    """

    try:
        # Initialize step catalog if requested and available
        step_catalog_instance = None
        if step_catalog and STEP_CATALOG_AVAILABLE:
            workspace_list = [Path(workspace_dir)]
            if workspace_dirs:
                workspace_list.extend([Path(w.strip()) for w in workspace_dirs.split(',')])
            
            # Add environment workspaces
            env_workspaces = os.environ.get('CURSUS_DEV_WORKSPACES', '').split(':')
            for workspace in env_workspaces:
                if workspace and Path(workspace).exists():
                    workspace_list.append(Path(workspace))
            
            try:
                step_catalog_instance = StepCatalog(workspace_dirs=workspace_list)
            except Exception:
                # Silently ignore step catalog initialization errors
                pass

        # Initialize RuntimeTester and PipelineTestingSpecBuilder with step catalog
        tester = RuntimeTester(workspace_dir, step_catalog=step_catalog_instance)
        builder = PipelineTestingSpecBuilder(test_data_dir=workspace_dir, step_catalog=step_catalog_instance)
        
        # Build ScriptExecutionSpecs for both scripts using node resolution
        spec_a = builder.resolve_script_execution_spec_from_node(script_a)
        spec_b = builder.resolve_script_execution_spec_from_node(script_b)
        
        # Test data compatibility with step catalog enhancements if available
        if step_catalog_instance:
            result = tester.test_data_compatibility_with_step_catalog_enhancements(spec_a, spec_b)
        else:
            result = tester.test_data_compatibility_with_specs(spec_a, spec_b)

        if output_format == "json":
            click.echo(json.dumps(result.model_dump(), indent=2))
        else:
            status_color = "green" if result.compatible else "red"
            click.echo(f"Data compatibility: {script_a} -> {script_b}")
            click.echo(f"Status: ", nl=False)
            click.secho(
                "PASS" if result.compatible else "FAIL", fg=status_color, bold=True
            )

            if result.data_format_a:
                click.echo(f"Script A output format: {result.data_format_a}")
            if result.data_format_b:
                click.echo(f"Script B input format: {result.data_format_b}")

            if result.compatibility_issues:
                click.echo("Issues:")
                for issue in result.compatibility_issues:
                    click.echo(f"  - {issue}")

        sys.exit(0 if result.compatible else 1)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


def main():
    """Main entry point for CLI"""
    runtime()


if __name__ == "__main__":
    main()
