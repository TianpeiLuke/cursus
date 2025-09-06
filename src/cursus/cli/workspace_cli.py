"""Command-line interface for workspace lifecycle management."""

import click
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

# Phase 4: Use unified WorkspaceAPI
from ..workspace.api import WorkspaceAPI, WorkspaceStatus

# Updated imports for consolidated structure
from ..workspace.core import WorkspaceComponentRegistry
from ..workspace.validation import WorkspaceTestManager as WorkspaceManager
from ..workspace.validation import UnifiedValidationCore
from ..api.dag.workspace_dag import WorkspaceAwareDAG

@click.group(name='workspace')
def workspace_cli():
    """Workspace lifecycle management commands.
    
    Manage developer workspaces, validate isolation boundaries,
    and coordinate cross-workspace operations.
    """
    pass

@workspace_cli.command('create')
@click.argument('developer_name')
@click.option('--template', help='Workspace template to use')
@click.option('--from-existing', help='Clone from existing workspace')
@click.option('--interactive', is_flag=True, help='Interactive setup')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--config', type=click.File('r'), help='JSON configuration file')
@click.option('--output', type=click.Choice(['json', 'text']), default='text', help='Output format')
def create_workspace(developer_name: str, template: str, from_existing: str, 
                    interactive: bool, workspace_root: str, config, output: str):
    """Create a new developer workspace.
    
    DEVELOPER_NAME: Name/ID of the developer workspace to create
    """
    
    try:
        click.echo(f"Creating workspace for developer: {developer_name}")
        click.echo(f"Workspace root: {workspace_root}")
        click.echo("-" * 50)
        
        # Initialize Phase 4 WorkspaceAPI
        api = WorkspaceAPI(base_path=workspace_root)
        
        # Parse configuration overrides
        config_overrides = {}
        if config:
            try:
                config_overrides = json.load(config)
            except json.JSONDecodeError as e:
                click.echo(f"Error parsing config file: {e}", err=True)
                sys.exit(1)
        
        # Interactive setup
        if interactive:
            click.echo("Interactive workspace setup:")
            
            # Get workspace configuration
            if not template and not from_existing:
                template_choice = click.prompt(
                    'Choose template',
                    type=click.Choice(['basic', 'ml_pipeline', 'data_processing', 'custom']),
                    default='basic'
                )
                template = template_choice if template_choice != 'custom' else None
            
            # Confirm workspace root
            workspace_root = click.prompt('Workspace root directory', default=workspace_root)
            api = WorkspaceAPI(base_path=workspace_root)  # Reinitialize with new path
        
        # Create workspace using Phase 4 API
        result = api.setup_developer_workspace(
            developer_id=developer_name,
            template=template,
            config_overrides=config_overrides
        )
        
        # Display results
        if output == 'json':
            click.echo(result.model_dump_json(indent=2))
        else:
            if result.success:
                click.echo(f"✅ {result.message}")
                if result.warnings:
                    click.echo("⚠️  Warnings:")
                    for warning in result.warnings:
                        click.echo(f"   - {warning}")
                
                # Apply template if specified (legacy support)
                if template:
                    _apply_workspace_template(str(result.workspace_path), template)
                    click.echo(f"✓ Applied template: {template}")
                
                # Clone from existing workspace if specified (legacy support)
                if from_existing:
                    _clone_workspace(str(result.workspace_path), from_existing, workspace_root)
                    click.echo(f"✓ Cloned from: {from_existing}")
                
                # Show workspace structure
                click.echo("\nWorkspace structure:")
                _show_workspace_structure(str(result.workspace_path))
                
                # Show next steps
                click.echo("\nNext steps:")
                click.echo(f"  1. cd {result.workspace_path}")
                click.echo(f"  2. cursus workspace validate --workspace-path {result.workspace_path}")
                click.echo(f"  3. Start developing your components!")
            else:
                click.echo(f"❌ {result.message}", err=True)
                sys.exit(1)
        
    except Exception as e:
        click.echo(f"Error creating workspace: {str(e)}", err=True)
        sys.exit(1)

@workspace_cli.command('list')
@click.option('--active', is_flag=True, help='Show only active workspaces')
@click.option('--format', type=click.Choice(['table', 'json']), default='table',
              help='Output format')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--show-components', is_flag=True, help='Show component counts')
def list_workspaces(active: bool, format: str, workspace_root: str, show_components: bool):
    """List available developer workspaces."""
    
    try:
        # Initialize Phase 4 WorkspaceAPI
        api = WorkspaceAPI(base_path=workspace_root)
        
        # Get workspace list using unified API
        workspaces = api.list_workspaces()
        
        if not workspaces:
            click.echo("No workspaces found")
            return
        
        # Filter active workspaces if requested
        if active:
            # Consider workspace active if it has recent activity
            workspaces = [ws for ws in workspaces if _is_workspace_active_v4(ws)]
        
        if format == 'json':
            workspace_data = [ws.model_dump() for ws in workspaces]
            output = {
                "workspace_root": workspace_root,
                "total_workspaces": len(workspaces),
                "workspaces": workspace_data
            }
            click.echo(json.dumps(output, indent=2, default=str))
        else:
            # Table format
            click.echo(f"Developer Workspaces in: {workspace_root}")
            click.echo("-" * 80)
            
            if show_components:
                header = f"{'Developer':<15} {'Status':<10} {'Path':<30} {'Components':<15} {'Modified':<20}"
            else:
                header = f"{'Developer':<15} {'Status':<10} {'Path':<35} {'Modified':<20}"
            
            click.echo(header)
            click.echo("-" * len(header))
            
            for ws in workspaces:
                # Status with icon
                status_icons = {
                    WorkspaceStatus.HEALTHY: "✅ Healthy",
                    WorkspaceStatus.WARNING: "⚠️  Warning",
                    WorkspaceStatus.ERROR: "❌ Error",
                    WorkspaceStatus.UNKNOWN: "❓ Unknown"
                }
                status_display = status_icons.get(ws.status, "❓ Unknown")
                status_color = {
                    WorkspaceStatus.HEALTHY: 'green',
                    WorkspaceStatus.WARNING: 'yellow',
                    WorkspaceStatus.ERROR: 'red',
                    WorkspaceStatus.UNKNOWN: 'white'
                }.get(ws.status, 'white')
                
                path_display = str(ws.path)
                if len(path_display) > 30:
                    path_display = "..." + path_display[-27:]
                
                modified = ws.last_modified or "Unknown"
                
                if show_components:
                    # Get component counts (legacy support)
                    try:
                        registry = WorkspaceComponentRegistry(workspace_root)
                        components = registry.discover_components(ws.developer_id)
                        component_count = sum(len(components.get(t, {})) for t in ['builders', 'configs', 'contracts', 'specs', 'scripts'])
                        component_display = f"{component_count} total"
                    except:
                        component_display = "Unknown"
                    
                    click.echo(f"{ws.developer_id:<15} ", nl=False)
                    click.secho(f"{status_display:<10}", fg=status_color, nl=False)
                    click.echo(f" {path_display:<30} {component_display:<15} {modified:<20}")
                else:
                    click.echo(f"{ws.developer_id:<15} ", nl=False)
                    click.secho(f"{status_display:<10}", fg=status_color, nl=False)
                    click.echo(f" {path_display:<35} {modified:<20}")
        
    except Exception as e:
        click.echo(f"Error listing workspaces: {str(e)}", err=True)
        sys.exit(1)

@workspace_cli.command('validate')
@click.option('--workspace-path', type=click.Path(exists=True), help='Specific workspace path to validate')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--report', help='Output report path')
@click.option('--format', type=click.Choice(['text', 'json']), default='text',
              help='Output format')
@click.option('--strict', is_flag=True, help='Enable strict validation mode')
def validate_workspace(workspace_path: str, workspace_root: str, report: str, 
                      format: str, strict: bool):
    """Validate workspace isolation and compliance."""
    
    try:
        click.echo("Validating workspace...")
        click.echo(f"Workspace root: {workspace_root}")
        if workspace_path:
            click.echo(f"Target workspace: {workspace_path}")
        click.echo("-" * 50)
        
        # Initialize Phase 4 WorkspaceAPI
        api = WorkspaceAPI(base_path=workspace_root)
        
        if workspace_path:
            # Validate specific workspace
            result = api.validate_workspace(workspace_path)
            
            if format == 'json':
                click.echo(result.model_dump_json(indent=2))
            else:
                _display_validation_result_v4(result)
        else:
            # Get system health (validates all workspaces)
            health_result = api.get_system_health()
            
            if format == 'json':
                click.echo(health_result.model_dump_json(indent=2))
            else:
                _display_health_result_v4(health_result)
        
        # Save report if requested
        if report:
            report_path = Path(report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            result_to_save = result if workspace_path else health_result
            
            with open(report_path, 'w') as f:
                if report_path.suffix.lower() == '.json':
                    json.dump(result_to_save.model_dump(), f, indent=2, default=str)
                else:
                    yaml.dump(result_to_save.model_dump(), f, default_flow_style=False)
            
            click.echo(f"\n✓ Report saved: {report_path}")
        
        # Exit with appropriate code
        if workspace_path:
            sys.exit(0 if result.status in [WorkspaceStatus.HEALTHY, WorkspaceStatus.WARNING] else 1)
        else:
            sys.exit(0 if health_result.overall_status in [WorkspaceStatus.HEALTHY, WorkspaceStatus.WARNING] else 1)
        
    except Exception as e:
        click.echo(f"Error validating workspace: {str(e)}", err=True)
        sys.exit(1)

@workspace_cli.command('info')
@click.argument('developer_name')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--show-components', is_flag=True, help='Show detailed component information')
@click.option('--format', type=click.Choice(['text', 'json']), default='text',
              help='Output format')
def workspace_info(developer_name: str, workspace_root: str, show_components: bool, format: str):
    """Show detailed information about a workspace.
    
    DEVELOPER_NAME: Name/ID of the developer workspace
    """
    
    try:
        # Initialize workspace manager and registry
        manager = WorkspaceManager(workspace_root=workspace_root)
        registry = WorkspaceComponentRegistry(workspace_root)
        
        # Get workspace info
        workspace_info = manager.discover_workspaces()
        
        if developer_name not in workspace_info.workspaces:
            click.echo(f"Workspace not found: {developer_name}", err=True)
            sys.exit(1)
        
        workspace = workspace_info.workspaces[developer_name]
        
        # Get component information
        components = registry.discover_components(developer_name)
        
        # Prepare output
        info = {
            "developer_id": developer_name,
            "workspace_path": workspace.workspace_path,
            "valid": workspace.is_valid,
            "validation_errors": workspace.validation_errors,
            "last_modified": workspace.last_modified.isoformat() if workspace.last_modified else None,
            "components_summary": {
                "builders": len(components.get('builders', {})),
                "configs": len(components.get('configs', {})),
                "contracts": len(components.get('contracts', {})),
                "specs": len(components.get('specs', {})),
                "scripts": len(components.get('scripts', {}))
            }
        }
        
        if show_components:
            info["detailed_components"] = components
        
        if format == 'json':
            click.echo(json.dumps(info, indent=2))
        else:
            # Text format
            click.echo(f"Workspace Information: {developer_name}")
            click.echo("=" * 50)
            click.echo(f"Path: {info['workspace_path']}")
            
            status_color = 'green' if info['valid'] else 'red'
            click.echo(f"Status: ", nl=False)
            click.secho("Valid" if info['valid'] else "Invalid", fg=status_color)
            
            if info['validation_errors']:
                click.echo("Validation Errors:")
                for error in info['validation_errors']:
                    click.echo(f"  - {error}")
            
            click.echo(f"Last Modified: {info['last_modified'] or 'Unknown'}")
            
            click.echo("\nComponents Summary:")
            for comp_type, count in info['components_summary'].items():
                click.echo(f"  {comp_type.capitalize()}: {count}")
            
            if show_components and info.get('detailed_components'):
                click.echo("\nDetailed Components:")
                for comp_type, components_dict in info['detailed_components'].items():
                    if components_dict:
                        click.echo(f"\n  {comp_type.capitalize()}:")
                        for comp_name, comp_info in components_dict.items():
                            click.echo(f"    - {comp_name}: {comp_info.get('file_path', 'Unknown path')}")
        
    except Exception as e:
        click.echo(f"Error getting workspace info: {str(e)}", err=True)
        sys.exit(1)

@workspace_cli.command('health-check')
@click.option('--workspace', help='Specific workspace to check')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--fix-issues', is_flag=True, help='Attempt to fix detected issues')
@click.option('--format', type=click.Choice(['text', 'json']), default='text',
              help='Output format')
def health_check(workspace: str, workspace_root: str, fix_issues: bool, format: str):
    """Perform health check on workspace(s)."""
    
    try:
        click.echo("Performing workspace health check...")
        if workspace:
            click.echo(f"Target workspace: {workspace}")
        click.echo(f"Workspace root: {workspace_root}")
        click.echo("-" * 50)
        
        # Initialize workspace manager
        manager = WorkspaceManager(workspace_root=workspace_root)
        
        # Get workspace info
        workspace_info = manager.discover_workspaces()
        
        if workspace and workspace not in workspace_info.workspaces:
            click.echo(f"Workspace not found: {workspace}", err=True)
            sys.exit(1)
        
        # Select workspaces to check
        workspaces_to_check = {workspace: workspace_info.workspaces[workspace]} if workspace else workspace_info.workspaces
        
        health_results = {}
        overall_healthy = True
        
        for dev_id, ws_info in workspaces_to_check.items():
            click.echo(f"\nChecking workspace: {dev_id}")
            
            # Validate workspace structure
            is_valid, validation_errors = manager.validate_workspace_structure(ws_info.workspace_path, strict=True)
            
            # Check component integrity
            registry = WorkspaceComponentRegistry(workspace_root)
            components = registry.discover_components(dev_id)
            
            # Analyze component health
            component_issues = []
            for comp_type, comp_dict in components.items():
                for comp_name, comp_info in comp_dict.items():
                    file_path = comp_info.get('file_path')
                    if file_path and not Path(file_path).exists():
                        component_issues.append(f"Missing {comp_type} file: {file_path}")
            
            # Compile health result
            workspace_healthy = is_valid and not component_issues
            overall_healthy = overall_healthy and workspace_healthy
            
            health_results[dev_id] = {
                "healthy": workspace_healthy,
                "structure_valid": is_valid,
                "structure_errors": validation_errors,
                "component_issues": component_issues,
                "components_count": {k: len(v) for k, v in components.items()}
            }
            
            # Display immediate results
            if workspace_healthy:
                click.secho(f"  ✓ Healthy", fg='green')
            else:
                click.secho(f"  ✗ Issues detected", fg='red')
                
                if validation_errors:
                    click.echo("    Structure errors:")
                    for error in validation_errors:
                        click.echo(f"      - {error}")
                
                if component_issues:
                    click.echo("    Component issues:")
                    for issue in component_issues:
                        click.echo(f"      - {issue}")
                
                # Attempt fixes if requested
                if fix_issues:
                    click.echo("    Attempting fixes...")
                    fixed_count = _attempt_workspace_fixes(ws_info.workspace_path, validation_errors, component_issues)
                    if fixed_count > 0:
                        click.secho(f"    ✓ Fixed {fixed_count} issues", fg='green')
                    else:
                        click.echo("    No automatic fixes available")
        
        # Summary
        click.echo(f"\nHealth Check Summary:")
        click.echo(f"Total workspaces checked: {len(workspaces_to_check)}")
        healthy_count = sum(1 for result in health_results.values() if result['healthy'])
        click.echo(f"Healthy workspaces: {healthy_count}")
        click.echo(f"Workspaces with issues: {len(workspaces_to_check) - healthy_count}")
        
        if format == 'json':
            click.echo("\nDetailed Results:")
            click.echo(json.dumps(health_results, indent=2))
        
        # Exit with appropriate code
        sys.exit(0 if overall_healthy else 1)
        
    except Exception as e:
        click.echo(f"Error performing health check: {str(e)}", err=True)
        sys.exit(1)

@workspace_cli.command('remove')
@click.argument('developer_name')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--backup', is_flag=True, help='Create backup before removal')
@click.confirmation_option(prompt='Are you sure you want to remove this workspace?')
def remove_workspace(developer_name: str, workspace_root: str, backup: bool):
    """Remove a developer workspace.
    
    DEVELOPER_NAME: Name/ID of the developer workspace to remove
    """
    
    try:
        # Initialize workspace manager
        manager = WorkspaceManager(workspace_root=workspace_root)
        
        # Check if workspace exists
        workspace_info = manager.discover_workspaces()
        if developer_name not in workspace_info.workspaces:
            click.echo(f"Workspace not found: {developer_name}", err=True)
            sys.exit(1)
        
        workspace = workspace_info.workspaces[developer_name]
        workspace_path = Path(workspace.workspace_path)
        
        # Create backup if requested
        if backup:
            backup_path = Path(workspace_root) / "backups" / f"{developer_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copytree(workspace_path, backup_path)
            click.echo(f"✓ Backup created: {backup_path}")
        
        # Remove workspace
        import shutil
        shutil.rmtree(workspace_path)
        
        click.echo(f"✓ Workspace removed: {developer_name}")
        click.echo(f"  Path: {workspace_path}")
        
    except Exception as e:
        click.echo(f"Error removing workspace: {str(e)}", err=True)
        sys.exit(1)

@workspace_cli.command('promote')
@click.argument('workspace_path', type=click.Path(exists=True))
@click.option('--target', default='staging', help='Target environment')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--output', type=click.Choice(['json', 'text']), default='text', help='Output format')
def promote_artifacts(workspace_path: str, target: str, workspace_root: str, output: str):
    """Promote artifacts from workspace to target environment.
    
    WORKSPACE_PATH: Path to the workspace to promote from
    """
    
    try:
        click.echo(f"Promoting artifacts from workspace: {workspace_path}")
        click.echo(f"Target environment: {target}")
        click.echo("-" * 50)
        
        # Initialize Phase 4 WorkspaceAPI
        api = WorkspaceAPI(base_path=workspace_root)
        
        # Promote artifacts
        result = api.promote_workspace_artifacts(workspace_path, target)
        
        if output == 'json':
            click.echo(result.model_dump_json(indent=2))
        else:
            if result.success:
                click.echo(f"✅ {result.message}")
                if result.artifacts_promoted:
                    click.echo("Promoted artifacts:")
                    for artifact in result.artifacts_promoted:
                        click.echo(f"  - {artifact}")
            else:
                click.echo(f"❌ {result.message}", err=True)
                sys.exit(1)
        
    except Exception as e:
        click.echo(f"Error promoting artifacts: {str(e)}", err=True)
        sys.exit(1)

@workspace_cli.command('health')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--output', type=click.Choice(['json', 'text']), default='text', help='Output format')
def system_health(workspace_root: str, output: str):
    """Get overall system health report."""
    
    try:
        click.echo("Getting system health report...")
        click.echo(f"Workspace root: {workspace_root}")
        click.echo("-" * 50)
        
        # Initialize Phase 4 WorkspaceAPI
        api = WorkspaceAPI(base_path=workspace_root)
        
        # Get system health
        result = api.get_system_health()
        
        if output == 'json':
            click.echo(result.model_dump_json(indent=2))
        else:
            _display_health_result_v4(result)
        
    except Exception as e:
        click.echo(f"Error getting system health: {str(e)}", err=True)
        sys.exit(1)

@workspace_cli.command('cleanup')
@click.option('--inactive-days', type=int, default=30, help='Days of inactivity before cleanup')
@click.option('--dry-run/--no-dry-run', default=True, help='Show what would be cleaned without doing it')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--output', type=click.Choice(['json', 'text']), default='text', help='Output format')
def cleanup_workspaces(inactive_days: int, dry_run: bool, workspace_root: str, output: str):
    """Clean up inactive workspaces."""
    
    try:
        click.echo(f"Cleaning up inactive workspaces...")
        click.echo(f"Inactive threshold: {inactive_days} days")
        click.echo(f"Dry run: {dry_run}")
        click.echo(f"Workspace root: {workspace_root}")
        click.echo("-" * 50)
        
        # Initialize Phase 4 WorkspaceAPI
        api = WorkspaceAPI(base_path=workspace_root)
        
        # Cleanup workspaces
        result = api.cleanup_workspaces(inactive_days=inactive_days, dry_run=dry_run)
        
        if output == 'json':
            click.echo(result.model_dump_json(indent=2))
        else:
            if result.success:
                action = "Would clean" if dry_run else "Cleaned"
                click.echo(f"✅ {action} {len(result.cleaned_workspaces)} workspace(s)")
                
                if result.cleaned_workspaces:
                    click.echo(f"{action} workspaces:")
                    for ws_path in result.cleaned_workspaces:
                        click.echo(f"  - {ws_path}")
                
                if dry_run and result.cleaned_workspaces:
                    click.echo("\nRun with --no-dry-run to actually perform cleanup.")
            else:
                click.echo("❌ Cleanup failed", err=True)
                if result.errors:
                    for error in result.errors:
                        click.echo(f"  - {error}", err=True)
                sys.exit(1)
        
    except Exception as e:
        click.echo(f"Error cleaning up workspaces: {str(e)}", err=True)
        sys.exit(1)

def _apply_workspace_template(workspace_path: str, template: str):
    """Apply a workspace template to the created workspace."""
    
    workspace_dir = Path(workspace_path)
    
    if template == 'basic':
        # Create basic structure with example files
        (workspace_dir / "builders").mkdir(exist_ok=True)
        (workspace_dir / "configs").mkdir(exist_ok=True)
        (workspace_dir / "contracts").mkdir(exist_ok=True)
        (workspace_dir / "specs").mkdir(exist_ok=True)
        (workspace_dir / "scripts").mkdir(exist_ok=True)
        
        # Create README
        readme_content = f"""# Developer Workspace

This is a basic developer workspace for the Cursus pipeline system.

## Directory Structure

- `builders/` - Step builder implementations
- `configs/` - Configuration classes
- `contracts/` - Step contracts
- `specs/` - Step specifications
- `scripts/` - Pipeline scripts

## Getting Started

1. Implement your pipeline components in the appropriate directories
2. Use `cursus workspace validate-isolation` to check your workspace
3. Use `cursus workspace discover components` to see your components
4. Use `cursus runtime test-script` to test your scripts

## Workspace Isolation

Remember: Everything in this workspace stays in this workspace.
Only shared code in `src/cursus/` is available to all workspaces.
"""
        
        with open(workspace_dir / "README.md", 'w') as f:
            f.write(readme_content)
    
    elif template == 'ml_pipeline':
        # Create ML pipeline specific structure
        (workspace_dir / "builders").mkdir(exist_ok=True)
        (workspace_dir / "configs").mkdir(exist_ok=True)
        (workspace_dir / "contracts").mkdir(exist_ok=True)
        (workspace_dir / "specs").mkdir(exist_ok=True)
        (workspace_dir / "scripts").mkdir(exist_ok=True)
        (workspace_dir / "data").mkdir(exist_ok=True)
        (workspace_dir / "models").mkdir(exist_ok=True)
        (workspace_dir / "notebooks").mkdir(exist_ok=True)
        
        # Create ML-specific README
        readme_content = f"""# ML Pipeline Workspace

This workspace is configured for machine learning pipeline development.

## Directory Structure

- `builders/` - Step builder implementations
- `configs/` - Configuration classes
- `contracts/` - Step contracts
- `specs/` - Step specifications
- `scripts/` - Pipeline scripts
- `data/` - Local data files
- `models/` - Model artifacts
- `notebooks/` - Jupyter notebooks for experimentation

## ML Pipeline Components

Common ML pipeline steps you might implement:
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Model registration

## Getting Started

1. Start with data preprocessing components
2. Implement feature engineering steps
3. Create training and evaluation scripts
4. Test your pipeline with `cursus runtime test-pipeline`
"""
        
        with open(workspace_dir / "README.md", 'w') as f:
            f.write(readme_content)
    
    elif template == 'data_processing':
        # Create data processing specific structure
        (workspace_dir / "builders").mkdir(exist_ok=True)
        (workspace_dir / "configs").mkdir(exist_ok=True)
        (workspace_dir / "contracts").mkdir(exist_ok=True)
        (workspace_dir / "specs").mkdir(exist_ok=True)
        (workspace_dir / "scripts").mkdir(exist_ok=True)
        (workspace_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "data" / "processed").mkdir(exist_ok=True)
        (workspace_dir / "data" / "output").mkdir(exist_ok=True)
        
        # Create data processing README
        readme_content = f"""# Data Processing Workspace

This workspace is configured for data processing pipeline development.

## Directory Structure

- `builders/` - Step builder implementations
- `configs/` - Configuration classes
- `contracts/` - Step contracts
- `specs/` - Step specifications
- `scripts/` - Pipeline scripts
- `data/raw/` - Raw input data
- `data/processed/` - Intermediate processed data
- `data/output/` - Final output data

## Data Processing Components

Common data processing steps you might implement:
- Data ingestion
- Data cleaning and validation
- Data transformation
- Data aggregation
- Data export

## Getting Started

1. Place raw data in `data/raw/`
2. Implement data processing scripts
3. Create appropriate contracts and specs
4. Test your pipeline components
"""
        
        with open(workspace_dir / "README.md", 'w') as f:
            f.write(readme_content)

def _clone_workspace(target_path: str, source_workspace: str, workspace_root: str):
    """Clone components from an existing workspace."""
    
    source_path = Path(workspace_root) / "developers" / source_workspace
    target_dir = Path(target_path)
    
    if not source_path.exists():
        raise ValueError(f"Source workspace not found: {source_workspace}")
    
    # Copy component directories
    component_dirs = ['builders', 'configs', 'contracts', 'specs', 'scripts']
    
    for comp_dir in component_dirs:
        source_comp_dir = source_path / comp_dir
        target_comp_dir = target_dir / comp_dir
        
        if source_comp_dir.exists():
            import shutil
            if target_comp_dir.exists():
                shutil.rmtree(target_comp_dir)
            shutil.copytree(source_comp_dir, target_comp_dir)

def _show_workspace_structure(workspace_path: str):
    """Show the directory structure of a workspace."""
    
    workspace_dir = Path(workspace_path)
    
    for item in sorted(workspace_dir.iterdir()):
        if item.is_dir():
            click.echo(f"  📁 {item.name}/")
            # Show first few files in each directory
            files = list(item.iterdir())[:3]
            for file in files:
                if file.is_file():
                    click.echo(f"    📄 {file.name}")
            if len(list(item.iterdir())) > 3:
                click.echo(f"    ... and {len(list(item.iterdir())) - 3} more")
        else:
            click.echo(f"  📄 {item.name}")

def _is_workspace_active(workspace_info) -> bool:
    """Check if a workspace is considered active based on recent activity."""
    
    if not workspace_info.last_modified:
        return False
    
    # Consider active if modified within last 30 days
    from datetime import datetime, timedelta
    threshold = datetime.now() - timedelta(days=30)
    return workspace_info.last_modified > threshold

def _is_workspace_active_v4(workspace_info) -> bool:
    """Check if a Phase 4 WorkspaceInfo is considered active based on recent activity."""
    
    if not workspace_info.last_modified:
        return False
    
    # Consider active if modified within last 30 days
    from datetime import datetime, timedelta
    try:
        # Parse ISO format timestamp
        from datetime import datetime
        last_mod = datetime.fromisoformat(workspace_info.last_modified.replace('Z', '+00:00'))
        threshold = datetime.now() - timedelta(days=30)
        return last_mod > threshold
    except:
        # If parsing fails, consider it inactive
        return False

def _display_validation_result_v4(result):
    """Display Phase 4 ValidationReport in text format."""
    
    status_icons = {
        WorkspaceStatus.HEALTHY: "✅",
        WorkspaceStatus.WARNING: "⚠️",
        WorkspaceStatus.ERROR: "❌",
        WorkspaceStatus.UNKNOWN: "❓"
    }
    
    icon = status_icons.get(result.status, "❓")
    click.echo(f"{icon} Workspace: {result.workspace_path}")
    click.echo(f"Status: {result.status.value}")
    
    if result.issues:
        click.echo("\nIssues:")
        for issue in result.issues:
            click.echo(f"  - {issue}")
    
    if result.recommendations:
        click.echo("\nRecommendations:")
        for rec in result.recommendations:
            click.echo(f"  - {rec}")
    
    if result.isolation_violations:
        click.echo(f"\nIsolation violations: {len(result.isolation_violations)}")

def _display_health_result_v4(result):
    """Display Phase 4 HealthReport in text format."""
    
    status_icons = {
        WorkspaceStatus.HEALTHY: "✅",
        WorkspaceStatus.WARNING: "⚠️",
        WorkspaceStatus.ERROR: "❌",
        WorkspaceStatus.UNKNOWN: "❓"
    }
    
    icon = status_icons.get(result.overall_status, "❓")
    click.echo(f"{icon} Overall System Status: {result.overall_status.value}")
    
    if result.workspace_reports:
        click.echo(f"\nWorkspace Summary:")
        healthy = sum(1 for r in result.workspace_reports if r.status == WorkspaceStatus.HEALTHY)
        warning = sum(1 for r in result.workspace_reports if r.status == WorkspaceStatus.WARNING)
        error = sum(1 for r in result.workspace_reports if r.status == WorkspaceStatus.ERROR)
        
        click.echo(f"  ✅ Healthy: {healthy}")
        click.echo(f"  ⚠️  Warning: {warning}")
        click.echo(f"  ❌ Error: {error}")
    
    if result.system_issues:
        click.echo("\nSystem Issues:")
        for issue in result.system_issues:
            click.echo(f"  - {issue}")
    
    if result.recommendations:
        click.echo("\nRecommendations:")
        for rec in result.recommendations:
            click.echo(f"  - {rec}")

def _display_validation_result(result):
    """Display validation result in text format."""
    
    click.echo(f"Validation Results:")
    click.echo("=" * 50)
    
    # Overall summary
    success_rate = result.summary.success_rate
    status_color = 'green' if success_rate >= 0.8 else 'yellow' if success_rate >= 0.5 else 'red'
    
    click.echo(f"Overall Success Rate: ", nl=False)
    click.secho(f"{success_rate:.1%}", fg=status_color, bold=True)
    
    click.echo(f"Total Workspaces: {result.summary.total_workspaces}")
    click.echo(f"Successful: {result.summary.successful_workspaces}")
    click.echo(f"Failed: {result.summary.failed_workspaces}")
    
    # Individual workspace results
    if result.workspaces:
        click.echo("\nWorkspace Details:")
        for workspace_id, workspace_result in result.workspaces.items():
            status_color = 'green' if workspace_result.overall_success else 'red'
            click.echo(f"\n  {workspace_id}: ", nl=False)
            click.secho("PASS" if workspace_result.overall_success else "FAIL", fg=status_color)
            
            if workspace_result.validation_errors:
                click.echo("    Errors:")
                for error in workspace_result.validation_errors:
                    click.echo(f"      - {error}")
            
            if workspace_result.recommendations:
                click.echo("    Recommendations:")
                for rec in workspace_result.recommendations:
                    click.echo(f"      - {rec}")
    
    # Overall recommendations
    if result.recommendations:
        click.echo("\nOverall Recommendations:")
        for rec in result.recommendations:
            click.echo(f"  - {rec}")

@workspace_cli.command('discover')
@click.argument('component_type', type=click.Choice(['components', 'pipelines', 'scripts', 'builders', 'configs', 'contracts', 'specs']))
@click.option('--workspace', help='Target workspace (if not specified, searches all)')
@click.option('--type-filter', help='Component type filter')
@click.option('--format', type=click.Choice(['table', 'json']), default='table',
              help='Output format')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--show-details', is_flag=True, help='Show detailed component information')
def discover_components(component_type: str, workspace: str, type_filter: str, 
                       format: str, workspace_root: str, show_details: bool):
    """Discover components across workspaces.
    
    COMPONENT_TYPE: Type of components to discover
    """
    
    try:
        click.echo(f"Discovering {component_type}...")
        if workspace:
            click.echo(f"Target workspace: {workspace}")
        click.echo(f"Workspace root: {workspace_root}")
        click.echo("-" * 50)
        
        # Initialize registry
        registry = WorkspaceComponentRegistry(workspace_root)
        
        # Discover components
        if workspace:
            # Single workspace discovery
            components = registry.discover_components(workspace)
            if component_type == 'components':
                discovered = components
            else:
                discovered = {component_type: components.get(component_type, {})}
        else:
            # Cross-workspace discovery
            discovered = registry.discover_cross_workspace_components()
            if component_type != 'components':
                # Filter to specific component type
                filtered = {}
                for ws_id, ws_components in discovered.items():
                    if component_type in ws_components:
                        filtered[ws_id] = {component_type: ws_components[component_type]}
                discovered = filtered
        
        # Apply type filter if specified
        if type_filter:
            discovered = _apply_component_filter(discovered, type_filter)
        
        # Display results
        if format == 'json':
            click.echo(json.dumps(discovered, indent=2))
        else:
            _display_component_discovery(discovered, component_type, show_details)
        
    except Exception as e:
        click.echo(f"Error discovering components: {str(e)}", err=True)
        sys.exit(1)

@workspace_cli.command('build')
@click.argument('pipeline_name')
@click.option('--workspace', help='Primary workspace for the pipeline')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--cross-workspace', is_flag=True, help='Enable cross-workspace component usage')
@click.option('--output-path', help='Output path for generated pipeline')
@click.option('--dry-run', is_flag=True, help='Show what would be built without executing')
@click.option('--validate', is_flag=True, default=True, help='Validate pipeline before building')
def build_pipeline(pipeline_name: str, workspace: str, workspace_root: str, 
                  cross_workspace: bool, output_path: str, dry_run: bool, validate: bool):
    """Build pipeline using cross-workspace components.
    
    PIPELINE_NAME: Name of the pipeline to build
    """
    
    try:
        click.echo(f"Building pipeline: {pipeline_name}")
        if workspace:
            click.echo(f"Primary workspace: {workspace}")
        click.echo(f"Cross-workspace enabled: {cross_workspace}")
        click.echo("-" * 50)
        
        # Initialize workspace-aware DAG
        dag = WorkspaceAwareDAG(workspace_root=workspace_root)
        
        # Build pipeline
        if dry_run:
            click.echo("DRY RUN MODE - No actual building will occur")
            
            # Show what would be built
            build_plan = dag.create_build_plan(
                pipeline_name=pipeline_name,
                primary_workspace=workspace,
                enable_cross_workspace=cross_workspace
            )
            
            click.echo("\nBuild Plan:")
            click.echo(f"  Pipeline: {build_plan['pipeline_name']}")
            click.echo(f"  Primary Workspace: {build_plan['primary_workspace']}")
            click.echo(f"  Cross-workspace Components: {len(build_plan.get('cross_workspace_components', []))}")
            click.echo(f"  Total Steps: {len(build_plan.get('steps', []))}")
            
            if build_plan.get('cross_workspace_components'):
                click.echo("\n  Cross-workspace Components:")
                for comp in build_plan['cross_workspace_components']:
                    click.echo(f"    - {comp['name']} (from {comp['workspace']})")
        else:
            # Validate pipeline if requested
            if validate:
                click.echo("Validating pipeline...")
                validation_result = dag.validate_cross_workspace_pipeline(
                    pipeline_name=pipeline_name,
                    primary_workspace=workspace
                )
                
                if not validation_result.is_valid:
                    click.echo("Pipeline validation failed:", err=True)
                    for error in validation_result.errors:
                        click.echo(f"  - {error}", err=True)
                    sys.exit(1)
                
                click.echo("✓ Pipeline validation passed")
            
            # Build the pipeline
            result = dag.build_cross_workspace_pipeline(
                pipeline_name=pipeline_name,
                primary_workspace=workspace,
                enable_cross_workspace=cross_workspace,
                output_path=output_path
            )
            
            click.echo(f"✓ Pipeline built successfully")
            click.echo(f"  Output: {result.output_path}")
            click.echo(f"  Steps: {len(result.steps)}")
            click.echo(f"  Cross-workspace components: {len(result.cross_workspace_components)}")
        
    except Exception as e:
        click.echo(f"Error building pipeline: {str(e)}", err=True)
        sys.exit(1)

@workspace_cli.command('test-compatibility')
@click.option('--source-workspace', required=True, help='Source workspace')
@click.option('--target-workspace', required=True, help='Target workspace')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--component-type', help='Specific component type to test')
@click.option('--format', type=click.Choice(['text', 'json']), default='text',
              help='Output format')
def test_compatibility(source_workspace: str, target_workspace: str, workspace_root: str,
                      component_type: str, format: str):
    """Test compatibility between workspace components."""
    
    try:
        click.echo(f"Testing compatibility between workspaces:")
        click.echo(f"  Source: {source_workspace}")
        click.echo(f"  Target: {target_workspace}")
        if component_type:
            click.echo(f"  Component type: {component_type}")
        click.echo("-" * 50)
        
        # Initialize registry and validator
        registry = WorkspaceComponentRegistry(workspace_root)
        validator = UnifiedValidationCore()
        
        # Test compatibility
        compatibility_result = validator.test_cross_workspace_compatibility(
            source_workspace=source_workspace,
            target_workspace=target_workspace,
            component_type=component_type,
            workspace_root=workspace_root
        )
        
        # Display results
        if format == 'json':
            click.echo(json.dumps(compatibility_result.model_dump(), indent=2))
        else:
            _display_compatibility_result(compatibility_result, source_workspace, target_workspace)
        
        # Exit with appropriate code
        sys.exit(0 if compatibility_result.compatible else 1)
        
    except Exception as e:
        click.echo(f"Error testing compatibility: {str(e)}", err=True)
        sys.exit(1)

@workspace_cli.command('merge')
@click.argument('source_workspace')
@click.argument('target_workspace')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--component-type', help='Specific component type to merge')
@click.option('--component-name', help='Specific component to merge')
@click.option('--strategy', type=click.Choice(['copy', 'link', 'reference']), default='copy',
              help='Merge strategy')
@click.option('--dry-run', is_flag=True, help='Show what would be merged without executing')
@click.confirmation_option(prompt='Are you sure you want to merge components?')
def merge_components(source_workspace: str, target_workspace: str, workspace_root: str,
                    component_type: str, component_name: str, strategy: str, dry_run: bool):
    """Merge components between workspaces.
    
    SOURCE_WORKSPACE: Source workspace to merge from
    TARGET_WORKSPACE: Target workspace to merge into
    """
    
    try:
        click.echo(f"Merging components:")
        click.echo(f"  From: {source_workspace}")
        click.echo(f"  To: {target_workspace}")
        click.echo(f"  Strategy: {strategy}")
        if component_type:
            click.echo(f"  Component type: {component_type}")
        if component_name:
            click.echo(f"  Component name: {component_name}")
        click.echo("-" * 50)
        
        # Initialize registry
        registry = WorkspaceComponentRegistry(workspace_root)
        
        # Perform merge
        if dry_run:
            click.echo("DRY RUN MODE - No actual merging will occur")
            
            # Show what would be merged
            merge_plan = registry.create_merge_plan(
                source_workspace=source_workspace,
                target_workspace=target_workspace,
                component_type=component_type,
                component_name=component_name,
                strategy=strategy
            )
            
            click.echo("\nMerge Plan:")
            click.echo(f"  Components to merge: {len(merge_plan.components)}")
            
            for comp in merge_plan.components:
                click.echo(f"    - {comp.type}/{comp.name} ({comp.action})")
                if comp.conflicts:
                    click.echo(f"      Conflicts: {', '.join(comp.conflicts)}")
        else:
            # Execute merge
            merge_result = registry.merge_components(
                source_workspace=source_workspace,
                target_workspace=target_workspace,
                component_type=component_type,
                component_name=component_name,
                strategy=strategy
            )
            
            click.echo(f"✓ Merge completed")
            click.echo(f"  Components merged: {merge_result.merged_count}")
            click.echo(f"  Conflicts resolved: {merge_result.conflicts_resolved}")
            
            if merge_result.warnings:
                click.echo("  Warnings:")
                for warning in merge_result.warnings:
                    click.echo(f"    - {warning}")
        
    except Exception as e:
        click.echo(f"Error merging components: {str(e)}", err=True)
        sys.exit(1)

def _attempt_workspace_fixes(workspace_path: str, structure_errors: List[str], 
                           component_issues: List[str]) -> int:
    """Attempt to fix common workspace issues."""
    
    fixed_count = 0
    workspace_dir = Path(workspace_path)
    
    # Fix missing directories
    for error in structure_errors:
        if "missing directory" in error.lower():
            # Extract directory name from error message
            if "builders" in error:
                (workspace_dir / "builders").mkdir(exist_ok=True)
                fixed_count += 1
            elif "configs" in error:
                (workspace_dir / "configs").mkdir(exist_ok=True)
                fixed_count += 1
            elif "contracts" in error:
                (workspace_dir / "contracts").mkdir(exist_ok=True)
                fixed_count += 1
            elif "specs" in error:
                (workspace_dir / "specs").mkdir(exist_ok=True)
                fixed_count += 1
            elif "scripts" in error:
                (workspace_dir / "scripts").mkdir(exist_ok=True)
                fixed_count += 1
    
    # Fix missing __init__.py files
    component_dirs = ['builders', 'configs', 'contracts', 'specs', 'scripts']
    for comp_dir in component_dirs:
        init_file = workspace_dir / comp_dir / "__init__.py"
        if not init_file.exists() and (workspace_dir / comp_dir).exists():
            init_file.touch()
            fixed_count += 1
    
    return fixed_count

def _apply_component_filter(discovered: Dict, type_filter: str) -> Dict:
    """Apply component type filter to discovered components."""
    
    filtered = {}
    
    for workspace_id, components in discovered.items():
        filtered_components = {}
        
        for comp_type, comp_dict in components.items():
            if type_filter.lower() in comp_type.lower():
                filtered_components[comp_type] = comp_dict
        
        if filtered_components:
            filtered[workspace_id] = filtered_components
    
    return filtered

def _display_component_discovery(discovered: Dict, component_type: str, show_details: bool):
    """Display component discovery results in table format."""
    
    click.echo(f"Discovered {component_type}:")
    click.echo("=" * 60)
    
    total_components = 0
    
    for workspace_id, components in discovered.items():
        click.echo(f"\nWorkspace: {workspace_id}")
        click.echo("-" * 40)
        
        workspace_total = 0
        
        for comp_type, comp_dict in components.items():
            if comp_dict:
                click.echo(f"  {comp_type.capitalize()}: {len(comp_dict)}")
                workspace_total += len(comp_dict)
                
                if show_details:
                    for comp_name, comp_info in comp_dict.items():
                        click.echo(f"    - {comp_name}")
                        if comp_info.get('file_path'):
                            click.echo(f"      Path: {comp_info['file_path']}")
                        if comp_info.get('description'):
                            click.echo(f"      Description: {comp_info['description']}")
        
        click.echo(f"  Total: {workspace_total}")
        total_components += workspace_total
    
    click.echo(f"\nOverall Total: {total_components} components")

def _display_compatibility_result(result, source_workspace: str, target_workspace: str):
    """Display compatibility test results in text format."""
    
    click.echo(f"Compatibility Test Results:")
    click.echo("=" * 50)
    
    status_color = 'green' if result.compatible else 'red'
    click.echo(f"Overall Compatibility: ", nl=False)
    click.secho("COMPATIBLE" if result.compatible else "INCOMPATIBLE", fg=status_color, bold=True)
    
    click.echo(f"Compatibility Score: {result.compatibility_score:.1%}")
    
    if result.compatible_components:
        click.echo(f"\nCompatible Components: {len(result.compatible_components)}")
        for comp in result.compatible_components:
            click.secho(f"  ✓ {comp}", fg='green')
    
    if result.incompatible_components:
        click.echo(f"\nIncompatible Components: {len(result.incompatible_components)}")
        for comp in result.incompatible_components:
            click.secho(f"  ✗ {comp}", fg='red')
    
    if result.warnings:
        click.echo(f"\nWarnings:")
        for warning in result.warnings:
            click.echo(f"  ⚠ {warning}")
    
    if result.recommendations:
        click.echo(f"\nRecommendations:")
        for rec in result.recommendations:
            click.echo(f"  • {rec}")

# Phase 6.3: Enhanced Runtime Testing CLI Integration
@workspace_cli.command('test-runtime')
@click.argument('test_type', type=click.Choice(['script', 'pipeline', 'component', 'integration']))
@click.option('--workspace', help='Target workspace for testing')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--test-name', help='Specific test or component name')
@click.option('--cross-workspace', is_flag=True, help='Enable cross-workspace testing')
@click.option('--isolation-mode', type=click.Choice(['strict', 'permissive']), default='strict',
              help='Workspace isolation mode for testing')
@click.option('--output-path', help='Output path for test results')
@click.option('--format', type=click.Choice(['text', 'json', 'junit']), default='text',
              help='Test result format')
def test_runtime(test_type: str, workspace: str, workspace_root: str, test_name: str,
                cross_workspace: bool, isolation_mode: str, output_path: str, format: str):
    """Run workspace-aware runtime tests.
    
    TEST_TYPE: Type of runtime test to execute
    """
    
    try:
        click.echo(f"Running {test_type} runtime tests...")
        if workspace:
            click.echo(f"Target workspace: {workspace}")
        click.echo(f"Isolation mode: {isolation_mode}")
        click.echo(f"Cross-workspace: {cross_workspace}")
        click.echo("-" * 50)
        
        # Initialize runtime testing components
        from ..validation.runtime.pipeline_executor import PipelineExecutor
        from ..validation.runtime.pipeline_script_executor import PipelineScriptExecutor
        
        executor = PipelineExecutor(workspace_root=workspace_root)
        script_executor = PipelineScriptExecutor(workspace_root=workspace_root)
        
        # Configure testing parameters
        test_config = {
            'workspace_root': workspace_root,
            'target_workspace': workspace,
            'isolation_mode': isolation_mode,
            'cross_workspace_enabled': cross_workspace,
            'output_format': format
        }
        
        # Execute tests based on type
        if test_type == 'script':
            result = script_executor.test_workspace_script(
                script_name=test_name,
                workspace_id=workspace,
                **test_config
            )
        elif test_type == 'pipeline':
            result = executor.test_workspace_pipeline(
                pipeline_name=test_name,
                workspace_id=workspace,
                **test_config
            )
        elif test_type == 'component':
            result = _test_workspace_component(
                component_name=test_name,
                workspace_id=workspace,
                **test_config
            )
        elif test_type == 'integration':
            result = _test_workspace_integration(
                workspace_id=workspace,
                **test_config
            )
        
        # Display results
        if format == 'json':
            click.echo(json.dumps(result.model_dump(), indent=2))
        elif format == 'junit':
            _output_junit_results(result, output_path)
        else:
            _display_runtime_test_results(result, test_type)
        
        # Save results if output path specified
        if output_path and format != 'junit':
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                if format == 'json':
                    json.dump(result.model_dump(), f, indent=2)
                else:
                    yaml.dump(result.model_dump(), f, default_flow_style=False)
            
            click.echo(f"\n✓ Results saved: {output_file}")
        
        # Exit with appropriate code
        sys.exit(0 if result.success else 1)
        
    except Exception as e:
        click.echo(f"Error running runtime tests: {str(e)}", err=True)
        sys.exit(1)

# Phase 6.4: Validation and Alignment CLI Integration
@workspace_cli.command('validate-alignment')
@click.option('--workspace', help='Target workspace for validation')
@click.option('--workspace-root', default='./development',
              help='Root directory for workspaces')
@click.option('--validation-level', 
              type=click.Choice(['script_contract', 'contract_spec', 'spec_dependency', 'builder_config', 'all']),
              default='all', help='Validation level to execute')
@click.option('--cross-workspace', is_flag=True, help='Include cross-workspace validation')
@click.option('--strict-mode', is_flag=True, help='Enable strict validation mode')
@click.option('--output-path', help='Output path for validation report')
@click.option('--format', type=click.Choice(['text', 'json', 'html']), default='text',
              help='Validation report format')
@click.option('--fix-issues', is_flag=True, help='Attempt to fix detected alignment issues')
def validate_alignment(workspace: str, workspace_root: str, validation_level: str,
                      cross_workspace: bool, strict_mode: bool, output_path: str,
                      format: str, fix_issues: bool):
    """Validate workspace component alignment."""
    
    try:
        click.echo(f"Validating workspace alignment...")
        if workspace:
            click.echo(f"Target workspace: {workspace}")
        click.echo(f"Validation level: {validation_level}")
        click.echo(f"Cross-workspace: {cross_workspace}")
        click.echo(f"Strict mode: {strict_mode}")
        click.echo("-" * 50)
        
        # Initialize validation core
        validator = UnifiedValidationCore()
        
        # Configure validation parameters
        validation_config = {
            'workspace_root': workspace_root,
            'target_workspace': workspace,
            'validation_levels': [validation_level] if validation_level != 'all' else 
                               ['script_contract', 'contract_spec', 'spec_dependency', 'builder_config'],
            'cross_workspace_enabled': cross_workspace,
            'strict_mode': strict_mode,
            'auto_fix': fix_issues
        }
        
        # Execute validation
        result = validator.validate_workspace_alignment(**validation_config)
        
        # Display results
        if format == 'json':
            click.echo(json.dumps(result.model_dump(), indent=2))
        elif format == 'html':
            _generate_html_validation_report(result, output_path)
        else:
            _display_alignment_validation_results(result)
        
        # Save results if output path specified
        if output_path and format != 'html':
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                if format == 'json':
                    json.dump(result.model_dump(), f, indent=2)
                else:
                    yaml.dump(result.model_dump(), f, default_flow_style=False)
            
            click.echo(f"\n✓ Validation report saved: {output_file}")
        
        # Show fix summary if fixes were applied
        if fix_issues and hasattr(result, 'fixes_applied'):
            click.echo(f"\nFixes Applied: {result.fixes_applied}")
            if result.fixes_applied > 0:
                click.secho(f"✓ {result.fixes_applied} issues automatically fixed", fg='green')
        
        # Exit with appropriate code
        success_rate = result.summary.success_rate if hasattr(result, 'summary') else 0.0
        sys.exit(0 if success_rate >= 0.8 else 1)
        
    except Exception as e:
        click.echo(f"Error validating alignment: {str(e)}", err=True)
        sys.exit(1)

def _test_workspace_component(component_name: str, workspace_id: str, **config) -> Any:
    """Test individual workspace component."""
    # Implementation would test specific component functionality
    # This is a placeholder for the actual component testing logic
    pass

def _test_workspace_integration(workspace_id: str, **config) -> Any:
    """Test workspace integration scenarios."""
    # Implementation would test integration between workspace components
    # This is a placeholder for the actual integration testing logic
    pass

def _output_junit_results(result: Any, output_path: str):
    """Output test results in JUnit XML format."""
    # Implementation would generate JUnit XML format
    # This is a placeholder for the actual JUnit output logic
    pass

def _display_runtime_test_results(result: Any, test_type: str):
    """Display runtime test results in text format."""
    # Implementation would display formatted test results
    # This is a placeholder for the actual result display logic
    pass

def _generate_html_validation_report(result: Any, output_path: str):
    """Generate HTML validation report."""
    # Implementation would generate HTML report
    # This is a placeholder for the actual HTML generation logic
    pass

def _display_alignment_validation_results(result: Any):
    """Display alignment validation results in text format."""
    # Implementation would display formatted validation results
    # This is a placeholder for the actual result display logic
    pass

def main():
    """Main entry point for workspace CLI."""
    workspace_cli()


if __name__ == '__main__':
    main()
