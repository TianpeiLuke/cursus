"""Command-line interface for pipeline runtime testing."""

import click
import sys
import json
import yaml
from pathlib import Path
import os

from ..validation.runtime.core.pipeline_script_executor import PipelineScriptExecutor
from ..validation.runtime.execution.pipeline_executor import PipelineExecutor
from ..validation.runtime.utils.result_models import TestResult
from ..validation.runtime.data.local_data_manager import LocalDataManager
from .runtime_s3_cli import s3
from .production_cli import production_cli

@click.group()
@click.version_option(version="0.1.0")
def runtime():
    """Pipeline Runtime Testing CLI
    
    Test individual scripts and complete pipelines for functionality,
    data flow compatibility, and performance.
    """
    pass

# Add S3 commands as a subgroup
runtime.add_command(s3)

# Add production commands as a subgroup
runtime.add_command(production_cli)

@runtime.command()
@click.argument('script_name')
@click.option('--data-source', default='synthetic', 
              type=click.Choice(['synthetic', 'local']),
              help='Data source for testing (synthetic, local)')
@click.option('--data-size', default='small',
              type=click.Choice(['small', 'medium', 'large']),
              help='Size of test data')
@click.option('--workspace-dir', default='./pipeline_testing',
              help='Workspace directory for test execution')
@click.option('--output-format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format for results')
def test_script(script_name: str, data_source: str, data_size: str, 
                workspace_dir: str, output_format: str):
    """Test a single script in isolation
    
    SCRIPT_NAME: Name of the script to test
    """
    
    click.echo(f"Testing script: {script_name}")
    click.echo(f"Data source: {data_source}")
    click.echo(f"Data size: {data_size}")
    click.echo("-" * 50)
    
    try:
        # Initialize executor
        executor = PipelineScriptExecutor(workspace_dir=workspace_dir)
        
        # Execute test
        result = executor.test_script_isolation(script_name, data_source)
        
        # Display results
        if output_format == 'json':
            _display_json_result(result)
        else:
            _display_text_result(result)
            
        # Exit with appropriate code
        sys.exit(0 if result.is_successful() else 1)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@runtime.command()
@click.option('--workspace-dir', default='./pipeline_testing',
              help='Workspace directory to list')
def list_results(workspace_dir: str):
    """List previous test results"""
    
    workspace_path = Path(workspace_dir)
    if not workspace_path.exists():
        click.echo(f"Workspace directory does not exist: {workspace_dir}")
        return
    
    click.echo(f"Test results in: {workspace_dir}")
    click.echo("-" * 50)
    
    # List output directories
    outputs_dir = workspace_path / "outputs"
    if outputs_dir.exists():
        for script_dir in outputs_dir.iterdir():
            if script_dir.is_dir():
                click.echo(f"Script: {script_dir.name}")
                
                # Check for metadata
                metadata_file = workspace_path / "metadata" / f"{script_dir.name}_outputs.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        click.echo(f"  - Last run: {metadata.get('captured_at', 'Unknown')}")
                    except:
                        pass
    else:
        click.echo("No test results found")

@runtime.command()
@click.option('--workspace-dir', default='./pipeline_testing',
              help='Workspace directory to clean')
@click.confirmation_option(prompt='Are you sure you want to clean the workspace?')
def clean_workspace(workspace_dir: str):
    """Clean workspace directory"""
    
    import shutil
    
    workspace_path = Path(workspace_dir)
    if workspace_path.exists():
        shutil.rmtree(workspace_path)
        click.echo(f"Cleaned workspace: {workspace_dir}")
    else:
        click.echo(f"Workspace directory does not exist: {workspace_dir}")

@runtime.command()
@click.argument('pipeline_path')
@click.option('--data-source', default='synthetic', 
              help='Data source for testing (synthetic)')
@click.option('--workspace-dir', default='./pipeline_testing',
              help='Workspace directory for test execution')
@click.option('--output-format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format for results')
def test_pipeline(pipeline_path: str, data_source: str, 
                workspace_dir: str, output_format: str):
    """Test a complete pipeline with data flow validation
    
    PIPELINE_PATH: Path to the pipeline definition YAML file
    """
    
    click.echo(f"Testing pipeline: {pipeline_path}")
    click.echo(f"Data source: {data_source}")
    click.echo("-" * 50)
    
    try:
        # Load pipeline definition
        pipeline_file = Path(pipeline_path)
        if not pipeline_file.exists():
            click.echo(f"Pipeline file not found: {pipeline_path}", err=True)
            sys.exit(1)
            
        with open(pipeline_file, 'r') as f:
            if pipeline_file.suffix.lower() == '.yaml' or pipeline_file.suffix.lower() == '.yml':
                pipeline_def = yaml.safe_load(f)
            else:
                pipeline_def = json.load(f)
        
        # Create pipeline executor
        executor = PipelineExecutor(workspace_dir=workspace_dir)
        
        # Execute pipeline test
        result = executor.execute_pipeline(pipeline_def, data_source)
        
        # Display results
        if output_format == 'json':
            click.echo(json.dumps(result.model_dump(), indent=2))
        else:
            _display_pipeline_result(result)
        
        # Exit with appropriate code
        sys.exit(0 if result.success else 1)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

def _display_pipeline_result(result):
    """Display pipeline execution result in text format."""
    
    status_color = 'green' if result.success else 'red'
    
    click.echo(f"Pipeline Status: ", nl=False)
    click.secho("SUCCESS" if result.success else "FAILURE", fg=status_color, bold=True)
    click.echo(f"Total Execution Time: {result.total_duration:.2f} seconds")
    click.echo(f"Peak Memory Usage: {result.memory_peak} MB")
    
    if result.error:
        click.echo(f"Error: {result.error}")
    
    click.echo("\nStep Results:")
    for step_result in result.completed_steps:
        step_status_color = 'green' if step_result.status == "SUCCESS" else 'red'
        click.echo(f"  - {step_result.step_name}: ", nl=False)
        click.secho(f"{step_result.status}", fg=step_status_color)
        click.echo(f"    Time: {step_result.execution_time:.2f}s, Memory: {step_result.memory_usage} MB")
        
        if step_result.error_message:
            click.echo(f"    Error: {step_result.error_message}")
        
        if step_result.data_validation_report and step_result.data_validation_report.issues:
            click.echo("    Data Flow Issues:")
            for issue in step_result.data_validation_report.issues:
                click.echo(f"      - {issue}")

@runtime.command()
@click.argument('script_path')
@click.option('--workspace-dir', default='./pipeline_testing',
              help='Workspace directory for test execution')
def discover_script(script_path: str, workspace_dir: str):
    """Check if a script can be discovered and analyzed
    
    SCRIPT_PATH: Name or path hint of the script to discover
    """
    
    executor = PipelineScriptExecutor(workspace_dir=workspace_dir)
    
    try:
        # Try to discover the script
        found_path = executor._discover_script_path(script_path)
        click.echo(f"Script discovered: {found_path}")
        click.echo(f"Script exists: {Path(found_path).exists()}")
        
        # Basic file info
        file_size = Path(found_path).stat().st_size
        click.echo(f"Script size: {file_size} bytes")
        
        # Try to import main function
        try:
            main_func = executor.script_manager.import_script_main(found_path)
            click.echo("Main function: Successfully imported")
        except Exception as e:
            click.echo(f"Main function: Import failed - {str(e)}")
            
    except Exception as e:
        click.echo(f"Script discovery failed: {str(e)}", err=True)
        sys.exit(1)

@runtime.command()
@click.argument('script_name')
@click.argument('data_files', nargs=-1, required=True)
@click.option('--workspace-dir', default='./pipeline_testing',
              help='Workspace directory for test execution')
@click.option('--description', default='',
              help='Description for the data files')
def add_local_data(script_name: str, data_files: tuple, workspace_dir: str, description: str):
    """Add local data files for a script
    
    SCRIPT_NAME: Name of the script
    DATA_FILES: One or more local data file paths
    """
    
    try:
        manager = LocalDataManager(workspace_dir)
        
        for i, data_file in enumerate(data_files):
            if not Path(data_file).exists():
                click.echo(f"Error: File not found: {data_file}", err=True)
                continue
                
            # Generate data key
            data_key = f"input_data_{i+1}" if len(data_files) > 1 else "input_data"
            
            # Add the file
            success = manager.add_data_for_script(
                script_name, 
                data_key, 
                data_file, 
                description or f"Local data file for {script_name}"
            )
            
            if success:
                click.echo(f"Added local data: {Path(data_file).name} for script {script_name}")
            else:
                click.echo(f"Failed to add local data: {data_file}", err=True)
        
        click.echo(f"Local data files added successfully for {script_name}")
        
    except Exception as e:
        click.echo(f"Error adding local data: {str(e)}", err=True)
        sys.exit(1)

@runtime.command()
@click.option('--workspace-dir', default='./pipeline_testing',
              help='Workspace directory for test execution')
@click.option('--script-name', default=None,
              help='Show data for specific script only')
def list_local_data(workspace_dir: str, script_name: str):
    """List available local data files"""
    
    try:
        manager = LocalDataManager(workspace_dir)
        
        if script_name:
            # Show data for specific script
            script_data = manager.list_data_for_script(script_name)
            if not script_data:
                click.echo(f"No local data files configured for script: {script_name}")
                return
            
            click.echo(f"Local data files for script: {script_name}")
            click.echo("-" * 40)
            
            for data_key, file_info in script_data.items():
                file_path = manager.local_data_dir / file_info["path"]
                status = "✓" if file_path.exists() else "✗"
                click.echo(f"  {status} {data_key}: {file_info['path']} ({file_info.get('format', 'unknown')})")
                if file_info.get('description'):
                    click.echo(f"    Description: {file_info['description']}")
        else:
            # Show all scripts with local data
            scripts = manager.list_all_scripts()
            if not scripts:
                click.echo("No local data files configured")
                return
            
            click.echo("Available local data files:")
            click.echo("-" * 40)
            
            for script in scripts:
                click.echo(f"\nScript: {script}")
                script_data = manager.list_data_for_script(script)
                for data_key, file_info in script_data.items():
                    file_path = manager.local_data_dir / file_info["path"]
                    status = "✓" if file_path.exists() else "✗"
                    click.echo(f"  {status} {data_key}: {file_info['path']} ({file_info.get('format', 'unknown')})")
        
    except Exception as e:
        click.echo(f"Error listing local data: {str(e)}", err=True)
        sys.exit(1)

@runtime.command()
@click.argument('script_name')
@click.option('--data-key', default=None,
              help='Remove specific data key (if not provided, removes all data for script)')
@click.option('--workspace-dir', default='./pipeline_testing',
              help='Workspace directory for test execution')
@click.confirmation_option(prompt='Are you sure you want to remove the local data?')
def remove_local_data(script_name: str, data_key: str, workspace_dir: str):
    """Remove local data for a script"""
    
    try:
        manager = LocalDataManager(workspace_dir)
        
        success = manager.remove_data_for_script(script_name, data_key)
        
        if success:
            if data_key:
                click.echo(f"Removed local data: {script_name}.{data_key}")
            else:
                click.echo(f"Removed all local data for script: {script_name}")
        else:
            click.echo(f"Failed to remove local data", err=True)
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"Error removing local data: {str(e)}", err=True)
        sys.exit(1)

def _display_text_result(result: TestResult):
    """Display test result in text format"""
    
    status_color = 'green' if result.is_successful() else 'red'
    
    click.echo(f"Status: ", nl=False)
    click.secho(result.status, fg=status_color, bold=True)
    click.echo(f"Execution Time: {result.execution_time:.2f} seconds")
    click.echo(f"Memory Usage: {result.memory_usage} MB")
    
    if result.error_message:
        click.echo(f"Error: {result.error_message}")
    
    if result.recommendations:
        click.echo("\nRecommendations:")
        for rec in result.recommendations:
            click.echo(f"  - {rec}")

def _display_json_result(result: TestResult):
    """Display test result in JSON format"""
    
    result_dict = {
        "script_name": result.script_name,
        "status": result.status,
        "execution_time": result.execution_time,
        "memory_usage": result.memory_usage,
        "error_message": result.error_message,
        "recommendations": result.recommendations,
        "timestamp": result.timestamp.isoformat()
    }
    
    click.echo(json.dumps(result_dict, indent=2))

# Entry point for CLI
def main():
    """Main entry point for CLI"""
    runtime()

if __name__ == '__main__':
    main()
