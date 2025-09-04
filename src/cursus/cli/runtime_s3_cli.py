"""CLI commands for S3 integration with pipeline runtime testing."""

import click
import json
import yaml
from pathlib import Path
import sys
import logging

from ..validation.runtime.integration.s3_data_downloader import S3DataDownloader
from ..validation.runtime.integration.real_data_tester import RealDataTester
from ..validation.runtime.integration.workspace_manager import WorkspaceManager, WorkspaceConfig

logger = logging.getLogger(__name__)

@click.group()
def s3():
    """S3 integration commands for pipeline testing."""
    pass

@s3.command()
@click.option('--bucket', required=True, help='S3 bucket name')
@click.option('--pipeline', required=True, help='Pipeline name')
@click.option('--execution-id', help='Specific execution ID (optional)')
@click.option('--workspace-dir', default='./development/projects/project_alpha', help='Workspace directory')
@click.option('--output-format', type=click.Choice(['text', 'json']), default='text',
             help='Output format')
def discover(bucket, pipeline, execution_id, workspace_dir, output_format):
    """Discover available pipeline data in S3."""
    downloader = S3DataDownloader(workspace_dir)
    
    click.echo(f"Discovering data for pipeline '{pipeline}' in bucket '{bucket}'...")
    
    try:
        data_sources = downloader.discover_pipeline_data(bucket, pipeline, execution_id)
        
        if not data_sources:
            click.echo("No data sources found.")
            sys.exit(1)
        
        if output_format == 'json':
            # Convert to serializable format
            sources_data = []
            for source in data_sources:
                source_dict = source.model_dump()
                # Convert Path objects to strings
                for key in source_dict:
                    if isinstance(source_dict[key], Path):
                        source_dict[key] = str(source_dict[key])
                sources_data.append(source_dict)
                
            click.echo(json.dumps(sources_data, indent=2))
        else:
            for i, source in enumerate(data_sources, 1):
                click.echo(f"\n{i}. Execution: {source.execution_id}")
                click.echo(f"   Prefix: {source.prefix}")
                click.echo(f"   Steps: {list(source.step_outputs.keys())}")
                
                for step_name, files in source.step_outputs.items():
                    click.echo(f"     {step_name}: {len(files)} files")
                    
                    # Show some sample files (limited to 3)
                    if files:
                        sample_files = files[:3]
                        for file in sample_files:
                            click.echo(f"       - {file.split('/')[-1]}")
                        if len(files) > 3:
                            click.echo(f"       ... and {len(files) - 3} more")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@s3.command()
@click.option('--bucket', required=True, help='S3 bucket name')
@click.option('--pipeline', required=True, help='Pipeline name')
@click.option('--execution-id', help='Specific execution ID (optional)')
@click.option('--steps', help='Comma-separated list of steps to test')
@click.option('--workspace-dir', default='./development/projects/project_alpha', help='Workspace directory')
@click.option('--output-format', type=click.Choice(['text', 'json']), default='text',
             help='Output format')
def test_real_data(bucket, pipeline, execution_id, steps, workspace_dir, output_format):
    """Test pipeline with real S3 data."""
    tester = RealDataTester(workspace_dir)
    
    test_steps = steps.split(',') if steps else None
    
    click.echo(f"Creating test scenario for pipeline '{pipeline}'...")
    
    try:
        scenario = tester.create_test_scenario(
            pipeline, bucket, execution_id, test_steps
        )
        
        click.echo(f"Scenario: {scenario.scenario_name}")
        click.echo(f"Test steps: {scenario.test_steps}")
        
        click.echo("\nExecuting test scenario...")
        result = tester.execute_test_scenario(scenario)
        
        if output_format == 'json':
            # Convert to serializable format
            result_dict = result.model_dump()
            # Convert any Path objects to strings
            for key, value in result_dict.items():
                if isinstance(value, Path):
                    result_dict[key] = str(value)
            click.echo(json.dumps(result_dict, indent=2))
        else:
            if result.success:
                click.secho("✅ Test scenario completed successfully!", fg='green', bold=True)
                
                click.echo("\nPerformance Metrics:")
                for step_name, metrics in result.performance_metrics.items():
                    click.echo(f"  {step_name}:")
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, float):
                            click.echo(f"    {metric_name}: {metric_value:.2f}")
                        else:
                            click.echo(f"    {metric_name}: {metric_value}")
                
                click.echo("\nValidation Results:")
                for step_name, validation in result.data_validation_results.items():
                    click.echo(f"  {step_name}:")
                    passed = validation.get('passed', True)
                    click.echo(f"    Status: ", nl=False)
                    click.secho("PASSED" if passed else "FAILED", 
                               fg='green' if passed else 'red', bold=True)
                    
                    # Show warnings
                    warnings = validation.get('warnings', [])
                    if warnings:
                        click.echo(f"    Warnings:")
                        for warning in warnings:
                            click.echo(f"      - {warning}")
                    
                    # Show issues
                    issues = validation.get('issues', [])
                    if issues:
                        click.echo(f"    Issues:")
                        for issue in issues:
                            click.echo(f"      - {issue}")
            else:
                click.secho("❌ Test scenario failed!", fg='red', bold=True)
                click.echo(f"Error: {result.error_details}")
                
                # Show partial results if any
                if result.step_results:
                    click.echo("\nPartial Results:")
                    for step_name, step_result in result.step_results.items():
                        click.echo(f"  {step_name}: {step_result.status}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@s3.command()
@click.option('--workspace-dir', default='./development/projects/project_alpha', help='Workspace directory')
@click.option('--workspace-name', help='Specific workspace name to get info for (optional)')
@click.option('--output-format', type=click.Choice(['text', 'json']), default='text',
             help='Output format')
def workspace_info(workspace_dir, workspace_name, output_format):
    """Get information about test workspaces."""
    config = WorkspaceConfig(base_dir=Path(workspace_dir))
    manager = WorkspaceManager(config)
    
    try:
        info = manager.get_workspace_info(workspace_name)
        
        if output_format == 'json':
            # Convert to serializable format (need to handle Path objects)
            info_dict = json.dumps(info, default=lambda x: str(x) if isinstance(x, Path) else x, indent=2)
            click.echo(info_dict)
        else:
            if "error" in info:
                click.echo(f"Error: {info['error']}")
                sys.exit(1)
                
            if workspace_name:
                # Show info for specific workspace
                ws_info = info
                click.echo(f"Workspace: {ws_info['name']}")
                click.echo(f"Path: {ws_info['path']}")
                click.echo(f"Size: {ws_info['total_size_mb']:.2f} MB")
                click.echo(f"Last Modified: {ws_info['last_modified']}")
                click.echo("Files:")
                for dir_name, count in ws_info['files'].items():
                    click.echo(f"  {dir_name}: {count}")
            else:
                # Show summary of all workspaces
                click.echo(f"Total Cache Size: {info['cache_size_gb']:.2f} GB "
                          f"(max: {info['max_cache_size_gb']} GB)")
                click.echo(f"Cache Entries: {info['cache_entries']}")
                click.echo("\nWorkspaces:")
                for name, ws_info in info['workspaces'].items():
                    click.echo(f"  {name}:")
                    click.echo(f"    Size: {ws_info['total_size_mb']:.2f} MB")
                    click.echo(f"    Files: {sum(ws_info['files'].values())} "
                              f"(Last Modified: {ws_info['last_modified']})")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@s3.command()
@click.option('--workspace-dir', default='./development/projects/project_alpha', help='Workspace directory')
@click.option('--workspace-name', required=True, help='Workspace name to clean')
@click.confirmation_option(prompt='Are you sure you want to clean the workspace?')
def clean_workspace(workspace_dir, workspace_name):
    """Clean up a test workspace."""
    config = WorkspaceConfig(base_dir=Path(workspace_dir))
    manager = WorkspaceManager(config)
    
    try:
        manager.cleanup_workspace(workspace_name)
        click.echo(f"✅ Workspace '{workspace_name}' cleaned successfully!")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@s3.command()
@click.option('--workspace-dir', default='./development/projects/project_alpha', help='Workspace directory')
@click.confirmation_option(prompt='Are you sure you want to clean the cache?')
def clean_cache(workspace_dir):
    """Clean up cached S3 data."""
    config = WorkspaceConfig(base_dir=Path(workspace_dir))
    manager = WorkspaceManager(config)
    
    try:
        manager._cleanup_cache()
        click.echo("✅ Cache cleaned successfully!")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
