"""Command-line interface for simplified workspace management."""

import click
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

# Use simplified WorkspaceAPI
from ..workspace import WorkspaceAPI


@click.group(name="workspace")
def workspace_cli():
    """Simplified workspace management commands.

    Manage workspace-aware component discovery, validation, and pipeline creation
    using the unified WorkspaceAPI built on the step catalog architecture.
    """
    pass


@workspace_cli.command("init")
@click.option(
    "--workspace-dirs",
    multiple=True,
    type=click.Path(),
    help="Workspace directories to configure (can be specified multiple times)"
)
@click.option(
    "--output",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format",
)
def init_workspace(workspace_dirs: tuple, output: str):
    """Initialize workspace API with specified directories.
    
    This command demonstrates how to set up the WorkspaceAPI with
    user-specified workspace directories.
    """
    try:
        if workspace_dirs:
            workspace_paths = [Path(d) for d in workspace_dirs]
            click.echo(f"Initializing WorkspaceAPI with {len(workspace_paths)} directories:")
            for path in workspace_paths:
                click.echo(f"  - {path}")
        else:
            workspace_paths = None
            click.echo("Initializing WorkspaceAPI in package-only mode")
        
        click.echo("-" * 50)
        
        # Initialize WorkspaceAPI
        api = WorkspaceAPI(workspace_dirs=workspace_paths)
        
        # Get workspace summary
        summary = api.get_workspace_summary()
        
        if output == "json":
            click.echo(json.dumps(summary, indent=2, default=str))
        else:
            click.echo("✅ WorkspaceAPI initialized successfully")
            click.echo(f"📊 Configuration:")
            click.echo(f"   Total workspaces: {summary['total_workspaces']}")
            click.echo(f"   Total components: {summary['total_components']}")
            click.echo(f"   Workspace directories: {summary['workspace_directories']}")
            
            if summary.get('workspace_components'):
                click.echo(f"\n📁 Components by workspace:")
                for workspace_id, count in summary['workspace_components'].items():
                    click.echo(f"   {workspace_id}: {count} components")
        
    except Exception as e:
        click.echo(f"❌ Error initializing workspace: {str(e)}", err=True)
        sys.exit(1)


@workspace_cli.command("discover")
@click.option(
    "--workspace-dirs",
    multiple=True,
    type=click.Path(exists=True),
    help="Workspace directories to search (can be specified multiple times)"
)
@click.option(
    "--workspace-id",
    help="Filter components by specific workspace ID (directory name)"
)
@click.option(
    "--search",
    help="Search components with fuzzy matching"
)
@click.option(
    "--format",
    type=click.Choice(["table", "json", "list"]),
    default="table",
    help="Output format",
)
@click.option(
    "--show-details", 
    is_flag=True, 
    help="Show detailed component information"
)
def discover_components(
    workspace_dirs: tuple, 
    workspace_id: str, 
    search: str,
    format: str, 
    show_details: bool
):
    """Discover components across workspaces using step catalog."""
    
    try:
        # Initialize WorkspaceAPI
        workspace_paths = [Path(d) for d in workspace_dirs] if workspace_dirs else None
        api = WorkspaceAPI(workspace_dirs=workspace_paths)
        
        click.echo("🔍 Discovering components...")
        if workspace_id:
            click.echo(f"   Workspace filter: {workspace_id}")
        if search:
            click.echo(f"   Search query: {search}")
        click.echo("-" * 50)
        
        # Discover components
        if search:
            # Use search functionality
            components = api.search_components(search, workspace_id=workspace_id)
            component_names = [comp.step_name for comp in components]
        else:
            # Use discovery
            component_names = api.discover_components(workspace_id=workspace_id)
            components = []
        
        if not component_names:
            click.echo("No components found")
            return
        
        # Display results
        if format == "json":
            if search:
                # Include search result details
                result_data = []
                for comp in components:
                    result_data.append({
                        'step_name': comp.step_name,
                        'workspace_id': comp.workspace_id,
                        'file_components': {k: str(v.path) if v else None 
                                          for k, v in comp.file_components.items()}
                    })
                click.echo(json.dumps(result_data, indent=2))
            else:
                click.echo(json.dumps(component_names, indent=2))
                
        elif format == "list":
            for name in component_names:
                click.echo(name)
                
        else:  # table format
            click.echo(f"📋 Found {len(component_names)} components:")
            click.echo()
            
            if show_details:
                # Show detailed information
                for name in component_names[:10]:  # Limit to first 10 for readability
                    info = api.get_component_info(name)
                    if info:
                        click.echo(f"🔧 {name}")
                        click.echo(f"   Workspace: {info.workspace_id}")
                        click.echo(f"   Files:")
                        for comp_type, metadata in info.file_components.items():
                            if metadata:
                                click.echo(f"     {comp_type}: {metadata.path}")
                        click.echo()
                
                if len(component_names) > 10:
                    click.echo(f"... and {len(component_names) - 10} more components")
            else:
                # Simple list with workspace info
                cross_workspace = api.get_cross_workspace_components()
                for workspace_id, workspace_components in cross_workspace.items():
                    if workspace_components:
                        click.echo(f"📁 {workspace_id} ({len(workspace_components)} components):")
                        for comp in workspace_components[:5]:  # Show first 5
                            click.echo(f"   - {comp}")
                        if len(workspace_components) > 5:
                            click.echo(f"   ... and {len(workspace_components) - 5} more")
                        click.echo()
        
    except Exception as e:
        click.echo(f"❌ Error discovering components: {str(e)}", err=True)
        sys.exit(1)


@workspace_cli.command("validate")
@click.option(
    "--workspace-dirs",
    multiple=True,
    type=click.Path(exists=True),
    help="Workspace directories to validate"
)
@click.option(
    "--workspace-id",
    help="Validate specific workspace by ID"
)
@click.option(
    "--component",
    help="Validate specific component quality"
)
@click.option(
    "--compatibility",
    is_flag=True,
    help="Check cross-workspace compatibility"
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option(
    "--report",
    type=click.Path(),
    help="Save validation report to file"
)
def validate_workspace(
    workspace_dirs: tuple,
    workspace_id: str,
    component: str,
    compatibility: bool,
    format: str,
    report: str
):
    """Validate workspace structure, components, and compatibility."""
    
    try:
        # Initialize WorkspaceAPI
        workspace_paths = [Path(d) for d in workspace_dirs] if workspace_dirs else None
        api = WorkspaceAPI(workspace_dirs=workspace_paths)
        
        click.echo("🔍 Validating workspace...")
        if workspace_id:
            click.echo(f"   Target workspace: {workspace_id}")
        if component:
            click.echo(f"   Target component: {component}")
        click.echo("-" * 50)
        
        validation_results = {}
        overall_success = True
        
        # Validate workspace structure
        if workspace_dirs:
            click.echo("📁 Validating workspace structure...")
            for workspace_dir in workspace_paths:
                validation = api.validate_workspace_structure(workspace_dir)
                workspace_name = workspace_dir.name
                validation_results[f"structure_{workspace_name}"] = validation
                
                if format == "text":
                    status = "✅" if validation['valid'] else "❌"
                    click.echo(f"   {status} {workspace_name}: {'Valid' if validation['valid'] else 'Invalid'}")
                    if validation.get('warnings'):
                        for warning in validation['warnings']:
                            click.echo(f"      ⚠️  {warning}")
                
                overall_success = overall_success and validation['valid']
        
        # Validate workspace components
        if workspace_id:
            click.echo(f"\n🔧 Validating components in '{workspace_id}'...")
            result = api.validate_workspace_components(workspace_id)
            validation_results[f"components_{workspace_id}"] = {
                'is_valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'details': result.details
            }
            
            if format == "text":
                status = "✅" if result.is_valid else "❌"
                click.echo(f"   {status} Component validation: {'Passed' if result.is_valid else 'Failed'}")
                click.echo(f"   Components validated: {result.details.get('validated_components', 0)}")
                click.echo(f"   Total components: {result.details.get('total_components', 0)}")
                
                if result.errors:
                    click.echo("   Errors:")
                    for error in result.errors:
                        click.echo(f"      - {error}")
                
                if result.warnings:
                    click.echo("   Warnings:")
                    for warning in result.warnings:
                        click.echo(f"      - {warning}")
            
            overall_success = overall_success and result.is_valid
        
        # Validate specific component quality
        if component:
            click.echo(f"\n🎯 Validating component quality: {component}")
            quality_result = api.validate_component_quality(component)
            validation_results[f"quality_{component}"] = {
                'is_valid': quality_result.is_valid,
                'errors': quality_result.errors,
                'warnings': quality_result.warnings,
                'details': quality_result.details
            }
            
            if format == "text":
                status = "✅" if quality_result.is_valid else "❌"
                score = quality_result.details.get('quality_score', 0)
                click.echo(f"   {status} Quality validation: {'Passed' if quality_result.is_valid else 'Failed'}")
                click.echo(f"   Quality score: {score}/100")
                click.echo(f"   Component completeness: {quality_result.details.get('component_completeness', 0)}")
                
                missing = quality_result.details.get('missing_components', [])
                if missing:
                    click.echo(f"   Missing components: {', '.join(missing)}")
                
                if quality_result.errors:
                    click.echo("   Errors:")
                    for error in quality_result.errors:
                        click.echo(f"      - {error}")
            
            overall_success = overall_success and quality_result.is_valid
        
        # Check cross-workspace compatibility
        if compatibility:
            click.echo(f"\n🤝 Checking cross-workspace compatibility...")
            workspaces = api.list_all_workspaces()
            
            if len(workspaces) > 1:
                compat_result = api.validate_cross_workspace_compatibility(workspaces)
                validation_results['compatibility'] = {
                    'is_compatible': compat_result.is_compatible,
                    'issues': compat_result.issues,
                    'compatibility_matrix': compat_result.compatibility_matrix
                }
                
                if format == "text":
                    status = "✅" if compat_result.is_compatible else "❌"
                    click.echo(f"   {status} Cross-workspace compatibility: {'Compatible' if compat_result.is_compatible else 'Issues found'}")
                    
                    if compat_result.issues:
                        click.echo("   Issues:")
                        for issue in compat_result.issues:
                            click.echo(f"      - {issue}")
                    
                    click.echo("   Compatibility matrix:")
                    for ws_id, info in compat_result.compatibility_matrix.items():
                        conflicts = info.get('conflicts', 0)
                        total = info.get('total_components', 0)
                        click.echo(f"      {ws_id}: {total} components, {conflicts} conflicts")
                
                overall_success = overall_success and compat_result.is_compatible
            else:
                click.echo("   ⚠️  Need multiple workspaces for compatibility testing")
        
        # Display results
        if format == "json":
            click.echo(json.dumps(validation_results, indent=2, default=str))
        else:
            click.echo(f"\n📊 Validation Summary:")
            click.echo(f"   Overall result: {'✅ Success' if overall_success else '❌ Issues found'}")
        
        # Save report if requested
        if report:
            report_path = Path(report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                if report_path.suffix.lower() == '.json':
                    json.dump(validation_results, f, indent=2, default=str)
                else:
                    yaml.dump(validation_results, f, default_flow_style=False)
            
            click.echo(f"✓ Validation report saved: {report_path}")
        
        # Exit with appropriate code
        sys.exit(0 if overall_success else 1)
        
    except Exception as e:
        click.echo(f"❌ Error validating workspace: {str(e)}", err=True)
        sys.exit(1)


@workspace_cli.command("info")
@click.option(
    "--workspace-dirs",
    multiple=True,
    type=click.Path(exists=True),
    help="Workspace directories to analyze"
)
@click.option(
    "--component",
    help="Get detailed information about specific component"
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def workspace_info(workspace_dirs: tuple, component: str, format: str):
    """Get detailed workspace and component information."""
    
    try:
        # Initialize WorkspaceAPI
        workspace_paths = [Path(d) for d in workspace_dirs] if workspace_dirs else None
        api = WorkspaceAPI(workspace_dirs=workspace_paths)
        
        if component:
            # Get specific component information
            click.echo(f"🔧 Component Information: {component}")
            click.echo("-" * 50)
            
            info = api.get_component_info(component)
            if not info:
                click.echo(f"❌ Component not found: {component}")
                sys.exit(1)
            
            component_data = {
                'step_name': info.step_name,
                'workspace_id': info.workspace_id,
                'file_components': {}
            }
            
            for comp_type, metadata in info.file_components.items():
                if metadata:
                    component_data['file_components'][comp_type] = {
                        'path': str(metadata.path),
                        'exists': Path(metadata.path).exists()
                    }
            
            if format == "json":
                click.echo(json.dumps(component_data, indent=2))
            else:
                click.echo(f"Name: {info.step_name}")
                click.echo(f"Workspace: {info.workspace_id}")
                click.echo(f"Available files:")
                for comp_type, metadata in info.file_components.items():
                    if metadata:
                        exists = "✅" if Path(metadata.path).exists() else "❌"
                        click.echo(f"  {comp_type}: {metadata.path} {exists}")
        else:
            # Get workspace summary
            click.echo("📊 Workspace Information")
            click.echo("-" * 50)
            
            summary = api.get_workspace_summary()
            
            if format == "json":
                click.echo(json.dumps(summary, indent=2, default=str))
            else:
                click.echo(f"Total workspaces: {summary['total_workspaces']}")
                click.echo(f"Total components: {summary['total_components']}")
                click.echo(f"Workspace directories: {summary['workspace_directories']}")
                
                if summary.get('workspace_components'):
                    click.echo(f"\nComponents by workspace:")
                    for workspace_id, count in summary['workspace_components'].items():
                        click.echo(f"  {workspace_id}: {count} components")
                
                # Show system status
                status = api.get_system_status()
                success_rate = status['workspace_api']['success_rate']
                click.echo(f"\nSystem Status:")
                click.echo(f"  API success rate: {success_rate:.1%}")
                click.echo(f"  Total API calls: {status['workspace_api']['metrics']['api_calls']}")
        
    except Exception as e:
        click.echo(f"❌ Error getting workspace info: {str(e)}", err=True)
        sys.exit(1)


@workspace_cli.command("search")
@click.argument("query")
@click.option(
    "--workspace-dirs",
    multiple=True,
    type=click.Path(exists=True),
    help="Workspace directories to search"
)
@click.option(
    "--workspace-id",
    help="Filter search by specific workspace ID"
)
@click.option(
    "--format",
    type=click.Choice(["table", "json", "list"]),
    default="table",
    help="Output format",
)
def search_components(query: str, workspace_dirs: tuple, workspace_id: str, format: str):
    """Search components with fuzzy matching."""
    
    try:
        # Initialize WorkspaceAPI
        workspace_paths = [Path(d) for d in workspace_dirs] if workspace_dirs else None
        api = WorkspaceAPI(workspace_dirs=workspace_paths)
        
        click.echo(f"🔍 Searching for: '{query}'")
        if workspace_id:
            click.echo(f"   Workspace filter: {workspace_id}")
        click.echo("-" * 50)
        
        # Search components
        results = api.search_components(query, workspace_id=workspace_id)
        
        if not results:
            click.echo("No matching components found")
            return
        
        # Display results
        if format == "json":
            result_data = []
            for comp in results:
                result_data.append({
                    'step_name': comp.step_name,
                    'workspace_id': comp.workspace_id,
                    'file_components': {k: str(v.path) if v else None 
                                      for k, v in comp.file_components.items()}
                })
            click.echo(json.dumps(result_data, indent=2))
            
        elif format == "list":
            for comp in results:
                click.echo(comp.step_name)
                
        else:  # table format
            click.echo(f"📋 Found {len(results)} matching components:")
            click.echo()
            
            for comp in results:
                click.echo(f"🔧 {comp.step_name}")
                click.echo(f"   Workspace: {comp.workspace_id}")
                
                # Show available file types
                available_files = [k for k, v in comp.file_components.items() if v]
                if available_files:
                    click.echo(f"   Files: {', '.join(available_files)}")
                click.echo()
        
    except Exception as e:
        click.echo(f"❌ Error searching components: {str(e)}", err=True)
        sys.exit(1)


@workspace_cli.command("status")
@click.option(
    "--workspace-dirs",
    multiple=True,
    type=click.Path(exists=True),
    help="Workspace directories to check"
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def system_status(workspace_dirs: tuple, format: str):
    """Get comprehensive system status and metrics."""
    
    try:
        # Initialize WorkspaceAPI
        workspace_paths = [Path(d) for d in workspace_dirs] if workspace_dirs else None
        api = WorkspaceAPI(workspace_dirs=workspace_paths)
        
        click.echo("📊 System Status")
        click.echo("-" * 50)
        
        # Get system status
        status = api.get_system_status()
        
        if format == "json":
            click.echo(json.dumps(status, indent=2, default=str))
        else:
            # Display formatted status
            api_metrics = status['workspace_api']
            click.echo(f"WorkspaceAPI Status:")
            click.echo(f"  Success rate: {api_metrics['success_rate']:.1%}")
            click.echo(f"  Total API calls: {api_metrics['metrics']['api_calls']}")
            click.echo(f"  Successful operations: {api_metrics['metrics']['successful_operations']}")
            click.echo(f"  Failed operations: {api_metrics['metrics']['failed_operations']}")
            click.echo(f"  Workspace directories: {api_metrics['workspace_directories']}")
            
            # Manager status
            manager_status = status.get('manager', {})
            click.echo(f"\nWorkspace Manager:")
            click.echo(f"  Total components: {manager_status.get('total_components', 0)}")
            click.echo(f"  Total workspaces: {manager_status.get('total_workspaces', 0)}")
            
            # Validator status
            validator_status = status.get('validator', {})
            validator_metrics = validator_status.get('metrics', {})
            click.echo(f"\nValidator:")
            click.echo(f"  Validations performed: {validator_metrics.get('validations_performed', 0)}")
            click.echo(f"  Components validated: {validator_metrics.get('components_validated', 0)}")
            click.echo(f"  Compatibility checks: {validator_metrics.get('compatibility_checks', 0)}")
            
            # Integrator status
            integrator_status = status.get('integrator', {})
            integrator_metrics = integrator_status.get('metrics', {})
            click.echo(f"\nIntegrator:")
            click.echo(f"  Promotions: {integrator_metrics.get('promotions', 0)}")
            click.echo(f"  Integrations: {integrator_metrics.get('integrations', 0)}")
            click.echo(f"  Rollbacks: {integrator_metrics.get('rollbacks', 0)}")
        
    except Exception as e:
        click.echo(f"❌ Error getting system status: {str(e)}", err=True)
        sys.exit(1)


@workspace_cli.command("refresh")
@click.option(
    "--workspace-dirs",
    multiple=True,
    type=click.Path(exists=True),
    help="Workspace directories to refresh"
)
def refresh_catalog(workspace_dirs: tuple):
    """Refresh the step catalog to pick up new components."""
    
    try:
        # Initialize WorkspaceAPI
        workspace_paths = [Path(d) for d in workspace_dirs] if workspace_dirs else None
        api = WorkspaceAPI(workspace_dirs=workspace_paths)
        
        click.echo("🔄 Refreshing step catalog...")
        click.echo("-" * 50)
        
        # Get component count before refresh
        components_before = api.discover_components()
        before_count = len(components_before)
        
        # Refresh catalog
        success = api.refresh_catalog()
        
        if success:
            # Get component count after refresh
            components_after = api.discover_components()
            after_count = len(components_after)
            
            click.echo("✅ Catalog refresh successful")
            click.echo(f"   Components before: {before_count}")
            click.echo(f"   Components after: {after_count}")
            
            if after_count > before_count:
                click.echo(f"   ✨ Discovered {after_count - before_count} new components")
            elif after_count < before_count:
                click.echo(f"   🗑️  Removed {before_count - after_count} components")
            else:
                click.echo("   📊 No changes detected")
        else:
            click.echo("❌ Catalog refresh failed")
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"❌ Error refreshing catalog: {str(e)}", err=True)
        sys.exit(1)


def main():
    """Main entry point for workspace CLI."""
    workspace_cli()


if __name__ == "__main__":
    main()
