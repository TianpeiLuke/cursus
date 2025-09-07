"""Simplified command-line interface for pipeline runtime testing."""

import click
import json
import sys
from pathlib import Path

from ..validation.runtime.runtime_testing import RuntimeTester
from ..validation.runtime.runtime_models import RuntimeTestingConfiguration
from ..validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder
from ..api.dag.base_dag import PipelineDAG


@click.group()
@click.version_option(version="0.1.0")
def runtime():
    """Pipeline Runtime Testing CLI - Simplified
    
    Test individual scripts and complete pipelines for functionality
    and data flow compatibility.
    """
    pass


@runtime.command()
@click.argument('script_name')
@click.option('--workspace-dir', default='./test_workspace',
              help='Workspace directory for test execution')
@click.option('--output-format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format for results')
def test_script(script_name: str, workspace_dir: str, output_format: str):
    """Test a single script functionality
    
    SCRIPT_NAME: Name of the script to test
    """
    
    try:
        tester = RuntimeTester(workspace_dir)
        result = tester.test_script(script_name)
        
        if output_format == 'json':
            click.echo(json.dumps(result.model_dump(), indent=2))
        else:
            status_color = 'green' if result.success else 'red'
            click.echo(f"Script: {script_name}")
            click.echo(f"Status: ", nl=False)
            click.secho("PASS" if result.success else "FAIL", fg=status_color, bold=True)
            click.echo(f"Execution time: {result.execution_time:.3f}s")
            click.echo(f"Has main function: {'Yes' if result.has_main_function else 'No'}")
            
            if result.error_message:
                click.echo(f"Error: {result.error_message}")
        
        sys.exit(0 if result.success else 1)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@runtime.command()
@click.argument('pipeline_config')
@click.option('--workspace-dir', default='./test_workspace',
              help='Workspace directory for test execution')
@click.option('--output-format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format for results')
def test_pipeline(pipeline_config: str, workspace_dir: str, output_format: str):
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
            config = json.load(f)
        
        tester = RuntimeTester(workspace_dir)
        results = tester.test_pipeline_flow(config)
        
        if output_format == 'json':
            # Convert Pydantic models to dict for JSON serialization
            json_results = {
                "pipeline_success": results["pipeline_success"],
                "errors": results["errors"]
            }
            
            # Convert script results
            json_results["script_results"] = {}
            for script_name, result in results["script_results"].items():
                json_results["script_results"][script_name] = result.model_dump()
            
            # Convert data flow results
            json_results["data_flow_results"] = {}
            for flow_name, result in results["data_flow_results"].items():
                json_results["data_flow_results"][flow_name] = result.model_dump()
            
            click.echo(json.dumps(json_results, indent=2))
        else:
            # Text output
            status_color = 'green' if results["pipeline_success"] else 'red'
            click.echo(f"Pipeline: {pipeline_config}")
            click.echo(f"Status: ", nl=False)
            click.secho("PASS" if results["pipeline_success"] else "FAIL", fg=status_color, bold=True)
            
            click.echo("\nScript Results:")
            for script_name, result in results["script_results"].items():
                script_color = 'green' if result.success else 'red'
                click.echo(f"  {script_name}: ", nl=False)
                click.secho("PASS" if result.success else "FAIL", fg=script_color)
                if not result.success:
                    click.echo(f"    Error: {result.error_message}")
            
            click.echo("\nData Flow Results:")
            for flow_name, result in results["data_flow_results"].items():
                flow_color = 'green' if result.compatible else 'red'
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
@click.argument('script_a')
@click.argument('script_b')
@click.option('--workspace-dir', default='./test_workspace',
              help='Workspace directory for test execution')
@click.option('--output-format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format for results')
def test_compatibility(script_a: str, script_b: str, workspace_dir: str, output_format: str):
    """Test data compatibility between two scripts
    
    SCRIPT_A: First script name
    SCRIPT_B: Second script name
    """
    
    try:
        tester = RuntimeTester(workspace_dir)
        sample_data = tester._generate_sample_data()
        result = tester.test_data_compatibility(script_a, script_b, sample_data)
        
        if output_format == 'json':
            click.echo(json.dumps(result.model_dump(), indent=2))
        else:
            status_color = 'green' if result.compatible else 'red'
            click.echo(f"Data compatibility: {script_a} -> {script_b}")
            click.echo(f"Status: ", nl=False)
            click.secho("PASS" if result.compatible else "FAIL", fg=status_color, bold=True)
            
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


if __name__ == '__main__':
    main()
