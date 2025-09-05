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
from .registry_cli import registry_cli

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

# Add registry commands as a subgroup
runtime.add_command(registry_cli)

@runtime.command()
@click.argument('script_name')
@click.option('--data-source', default='synthetic', 
              type=click.Choice(['synthetic', 'local', 's3']),
              help='Data source for testing (synthetic, local, s3)')
@click.option('--data-size', default='small',
              type=click.Choice(['small', 'medium', 'large']),
              help='Size of test data')
@click.option('--workspace-dir', default='./development/projects/project_alpha',
              help='Workspace directory for test execution')
@click.option('--output-format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format for results')
@click.option('--testing-mode', default='pre_execution',
              type=click.Choice(['pre_execution', 'post_execution']),
              help='Testing mode for script execution')
@click.option('--memory-threshold', default=1024, type=int,
              help='Memory usage threshold in MB for warnings')
@click.option('--time-threshold', default=60, type=float,
              help='Execution time threshold in seconds for warnings')
@click.option('--enable-contract-discovery', is_flag=True,
              help='Enable contract-based script discovery and execution')
def test_script(script_name: str, data_source: str, data_size: str, 
                workspace_dir: str, output_format: str, testing_mode: str,
                memory_threshold: int, time_threshold: float, enable_contract_discovery: bool):
    """Test a single script in isolation
    
    SCRIPT_NAME: Name of the script to test
    """
    
    click.echo(f"Testing script: {script_name}")
    click.echo(f"Data source: {data_source}")
    click.echo(f"Data size: {data_size}")
    click.echo("-" * 50)
    
    try:
        # Initialize executor with testing mode
        executor = PipelineScriptExecutor(workspace_dir=workspace_dir)
        
        # Execute test
        result = executor.test_script_isolation(script_name, data_source)
        
        # Check thresholds and add warnings
        warnings = []
        if result.execution_time > time_threshold:
            warnings.append(f"Execution time ({result.execution_time:.2f}s) exceeds threshold ({time_threshold}s)")
        if result.memory_usage > memory_threshold:
            warnings.append(f"Memory usage ({result.memory_usage}MB) exceeds threshold ({memory_threshold}MB)")
        
        # Display results with threshold warnings
        if output_format == 'json':
            result_dict = _get_json_result_dict(result)
            result_dict["threshold_warnings"] = warnings
            result_dict["thresholds"] = {
                "memory_mb": memory_threshold,
                "time_seconds": time_threshold
            }
            click.echo(json.dumps(result_dict, indent=2))
        else:
            _display_text_result(result, warnings)
        
        # Show configuration info if contract discovery is enabled
        if enable_contract_discovery:
            click.echo(f"\nContract Discovery: {'Enabled' if enable_contract_discovery else 'Disabled'}")
            click.echo(f"Testing Mode: {testing_mode}")
        
        # Exit with appropriate code
        sys.exit(0 if result.is_successful() else 1)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@runtime.command()
@click.option('--workspace-dir', default='./development/projects/project_alpha',
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
@click.option('--workspace-dir', default='./development/projects/project_alpha',
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
              type=click.Choice(['synthetic', 'local', 's3']),
              help='Data source for testing (synthetic, local, s3)')
@click.option('--workspace-dir', default='./development/projects/project_alpha',
              help='Workspace directory for test execution')
@click.option('--output-format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format for results')
@click.option('--testing-mode', default='pre_execution',
              type=click.Choice(['pre_execution', 'post_execution']),
              help='Testing mode for pipeline execution')
@click.option('--config-path', default=None,
              help='Path to configuration file for pipeline')
@click.option('--enable-data-flow-validation', is_flag=True,
              help='Enable enhanced data flow validation between steps')
@click.option('--memory-threshold', default=2048, type=int,
              help='Memory usage threshold in MB for warnings')
@click.option('--time-threshold', default=300, type=float,
              help='Total execution time threshold in seconds for warnings')
def test_pipeline(pipeline_path: str, data_source: str, 
                workspace_dir: str, output_format: str, testing_mode: str,
                config_path: str, enable_data_flow_validation: bool,
                memory_threshold: int, time_threshold: float):
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
@click.option('--workspace-dir', default='./development/projects/project_alpha',
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
@click.option('--workspace-dir', default='./development/projects/project_alpha',
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
@click.option('--workspace-dir', default='./development/projects/project_alpha',
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
@click.option('--workspace-dir', default='./development/projects/project_alpha',
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

@runtime.command()
@click.option('--workspace-dir', default='./development/projects/project_alpha',
              help='Workspace directory for test execution')
@click.option('--script-name', default=None,
              help='Show history for specific script only')
@click.option('--limit', default=10, type=int,
              help='Maximum number of history entries to show')
@click.option('--output-format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format for history')
def list_execution_history(workspace_dir: str, script_name: str, limit: int, output_format: str):
    """List execution history for scripts and pipelines"""
    
    try:
        executor = PipelineScriptExecutor(workspace_dir=workspace_dir)
        
        if not executor.execution_history:
            click.echo("No execution history found")
            return
        
        # Filter by script name if provided
        history = executor.execution_history
        if script_name:
            history = [entry for entry in history if entry.get('script_name') == script_name]
        
        # Limit results
        history = history[-limit:]
        
        if output_format == 'json':
            # Convert to JSON-serializable format
            json_history = []
            for entry in history:
                json_entry = {
                    "script_name": entry.get('script_name'),
                    "script_path": entry.get('script_path'),
                    "execution_result": {
                        "success": entry.get('execution_result', {}).get('success'),
                        "execution_time": entry.get('execution_result', {}).get('execution_time'),
                        "memory_usage": entry.get('execution_result', {}).get('memory_usage'),
                        "error_message": entry.get('execution_result', {}).get('error_message')
                    }
                }
                json_history.append(json_entry)
            
            click.echo(json.dumps(json_history, indent=2))
        else:
            click.echo(f"Execution History ({len(history)} entries):")
            click.echo("-" * 50)
            
            for i, entry in enumerate(reversed(history), 1):
                script_name = entry.get('script_name', 'Unknown')
                result = entry.get('execution_result', {})
                status = "SUCCESS" if result.get('success') else "FAILURE"
                status_color = 'green' if result.get('success') else 'red'
                
                click.echo(f"{i}. Script: {script_name}")
                click.echo(f"   Status: ", nl=False)
                click.secho(status, fg=status_color)
                click.echo(f"   Time: {result.get('execution_time', 0):.2f}s")
                click.echo(f"   Memory: {result.get('memory_usage', 0)} MB")
                if result.get('error_message'):
                    click.echo(f"   Error: {result.get('error_message')}")
                click.echo()
        
    except Exception as e:
        click.echo(f"Error listing execution history: {str(e)}", err=True)
        sys.exit(1)

@runtime.command()
@click.option('--workspace-dir', default='./development/projects/project_alpha',
              help='Workspace directory for test execution')
@click.confirmation_option(prompt='Are you sure you want to clear execution history?')
def clear_execution_history(workspace_dir: str):
    """Clear execution history"""
    
    try:
        executor = PipelineScriptExecutor(workspace_dir=workspace_dir)
        executor.execution_history.clear()
        click.echo("Execution history cleared")
        
    except Exception as e:
        click.echo(f"Error clearing execution history: {str(e)}", err=True)
        sys.exit(1)

@runtime.command()
@click.argument('script_name')
@click.option('--data-size', default='small',
              type=click.Choice(['small', 'medium', 'large']),
              help='Size of synthetic data to generate')
@click.option('--data-format', default='csv',
              type=click.Choice(['csv', 'json', 'parquet']),
              help='Format of synthetic data')
@click.option('--num-records', default=None, type=int,
              help='Number of records to generate (overrides data-size)')
@click.option('--workspace-dir', default='./development/projects/project_alpha',
              help='Workspace directory for test execution')
@click.option('--seed', default=42, type=int,
              help='Random seed for reproducible data generation')
def generate_synthetic_data(script_name: str, data_size: str, data_format: str, 
                          num_records: int, workspace_dir: str, seed: int):
    """Generate synthetic data for a script"""
    
    try:
        from ..validation.runtime.data.default_synthetic_data_generator import DefaultSyntheticDataGenerator
        
        # Determine number of records based on size
        if num_records is None:
            size_mapping = {
                'small': 100,
                'medium': 1000,
                'large': 10000
            }
            num_records = size_mapping[data_size]
        
        # Initialize data generator
        generator = DefaultSyntheticDataGenerator(workspace_dir, seed=seed)
        
        # Generate data
        output_path = Path(workspace_dir) / "inputs" / script_name / f"synthetic_data.{data_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Generating {num_records} records of synthetic data...")
        click.echo(f"Format: {data_format}")
        click.echo(f"Output: {output_path}")
        
        # Generate basic synthetic data (this would need to be implemented in the generator)
        data = generator.generate_basic_dataset(num_records, data_format)
        
        if data_format == 'csv':
            data.to_csv(output_path, index=False)
        elif data_format == 'json':
            data.to_json(output_path, orient='records', indent=2)
        elif data_format == 'parquet':
            data.to_parquet(output_path, index=False)
        
        click.echo(f"Synthetic data generated successfully: {output_path}")
        click.echo(f"Records: {len(data)}")
        click.echo(f"Columns: {list(data.columns)}")
        
    except ImportError:
        click.echo("Error: Synthetic data generator not available", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error generating synthetic data: {str(e)}", err=True)
        sys.exit(1)

@runtime.command()
@click.option('--workspace-dir', default='./development/projects/project_alpha',
              help='Workspace directory for test execution')
@click.option('--output-format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format for configuration')
def show_config(workspace_dir: str, output_format: str):
    """Show current runtime testing configuration"""
    
    try:
        config = {
            "workspace_dir": workspace_dir,
            "workspace_exists": Path(workspace_dir).exists(),
            "available_data_sources": ["synthetic", "local", "s3"],
            "available_testing_modes": ["pre_execution", "post_execution"],
            "available_output_formats": ["text", "json"],
            "default_thresholds": {
                "script_memory_mb": 1024,
                "script_time_seconds": 60,
                "pipeline_memory_mb": 2048,
                "pipeline_time_seconds": 300
            }
        }
        
        # Check for local data
        try:
            manager = LocalDataManager(workspace_dir)
            scripts_with_data = manager.list_all_scripts()
            config["local_data_scripts"] = len(scripts_with_data)
        except:
            config["local_data_scripts"] = 0
        
        # Check execution history
        try:
            executor = PipelineScriptExecutor(workspace_dir=workspace_dir)
            config["execution_history_entries"] = len(executor.execution_history)
        except:
            config["execution_history_entries"] = 0
        
        if output_format == 'json':
            click.echo(json.dumps(config, indent=2))
        else:
            click.echo("Runtime Testing Configuration:")
            click.echo("-" * 40)
            click.echo(f"Workspace Directory: {config['workspace_dir']}")
            click.echo(f"Workspace Exists: {'✓' if config['workspace_exists'] else '✗'}")
            click.echo(f"Available Data Sources: {', '.join(config['available_data_sources'])}")
            click.echo(f"Available Testing Modes: {', '.join(config['available_testing_modes'])}")
            click.echo(f"Local Data Scripts: {config['local_data_scripts']}")
            click.echo(f"Execution History Entries: {config['execution_history_entries']}")
            click.echo("\nDefault Thresholds:")
            for key, value in config['default_thresholds'].items():
                click.echo(f"  {key}: {value}")
        
    except Exception as e:
        click.echo(f"Error showing configuration: {str(e)}", err=True)
        sys.exit(1)

def _display_text_result(result: TestResult, warnings: list = None):
    """Display test result in text format with optional threshold warnings"""
    
    status_color = 'green' if result.is_successful() else 'red'
    
    click.echo(f"Status: ", nl=False)
    click.secho(result.status, fg=status_color, bold=True)
    click.echo(f"Execution Time: {result.execution_time:.2f} seconds")
    click.echo(f"Memory Usage: {result.memory_usage} MB")
    
    if result.error_message:
        click.echo(f"Error: {result.error_message}")
    
    # Display threshold warnings
    if warnings:
        click.echo("\nThreshold Warnings:")
        for warning in warnings:
            click.secho(f"  ⚠ {warning}", fg='yellow')
    
    if result.recommendations:
        click.echo("\nRecommendations:")
        for rec in result.recommendations:
            click.echo(f"  - {rec}")

def _get_json_result_dict(result: TestResult):
    """Get JSON-serializable dictionary from TestResult"""
    
    return {
        "script_name": result.script_name,
        "status": result.status,
        "execution_time": result.execution_time,
        "memory_usage": result.memory_usage,
        "error_message": result.error_message,
        "recommendations": result.recommendations,
        "timestamp": result.timestamp.isoformat()
    }

def _display_json_result(result: TestResult):
    """Display test result in JSON format"""
    
    result_dict = _get_json_result_dict(result)
    click.echo(json.dumps(result_dict, indent=2))

# Entry point for CLI
def main():
    """Main entry point for CLI"""
    runtime()

if __name__ == '__main__':
    main()
