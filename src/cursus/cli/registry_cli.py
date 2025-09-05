"""
Registry management CLI commands for hybrid registry system.
Simplified implementation following redundancy evaluation guide principles.
"""
import click
import json
from pathlib import Path
from typing import Optional, Dict, Any

from ..registry.hybrid.setup import (
    create_workspace_registry,
    create_workspace_structure,
    create_workspace_documentation,
    create_example_implementations,
    validate_workspace_setup,
    copy_registry_from_developer
)


@click.group(name='registry')
def registry_cli():
    """Registry management commands."""
    pass


@registry_cli.command('init-workspace')
@click.argument('developer_id')
@click.option('--workspace-path', help='Workspace path (default: developer_workspaces/developers/{developer_id})')
@click.option('--template', default='standard', type=click.Choice(['standard', 'minimal']), 
              help='Registry template to use')
@click.option('--copy-from', help='Copy registry configuration from existing developer')
@click.option('--force', is_flag=True, help='Overwrite existing workspace if it exists')
def init_workspace_registry(developer_id: str, workspace_path: Optional[str], template: str, 
                           copy_from: Optional[str], force: bool):
    """Initialize local registry for a developer workspace.
    
    Creates a complete developer workspace with:
    - Directory structure for custom step implementations
    - Local registry configuration
    - Documentation and usage examples
    - Integration with hybrid registry system
    
    Args:
        developer_id: Unique identifier for the developer
        workspace_path: Custom workspace path (optional)
        template: Registry template type (standard/minimal)
        copy_from: Copy registry from existing developer (optional)
        force: Overwrite existing workspace
    """
    # Determine workspace path
    if not workspace_path:
        workspace_path = f"developer_workspaces/developers/{developer_id}"
    
    workspace_dir = Path(workspace_path)
    
    # Check if workspace already exists
    if workspace_dir.exists() and not force:
        click.echo(f"❌ Workspace already exists: {workspace_path}")
        click.echo("   Use --force to overwrite or choose a different path")
        return
    
    try:
        click.echo(f"🚀 Setting up developer workspace for: {developer_id}")
        click.echo(f"📁 Workspace path: {workspace_path}")
        
        # Create workspace directory structure
        create_workspace_structure(workspace_path)
        click.echo("✅ Created workspace directory structure")
        
        # Create or copy registry
        if copy_from:
            registry_file = copy_registry_from_developer(workspace_path, developer_id, copy_from)
            click.echo(f"✅ Copied registry from developer: {copy_from}")
        else:
            registry_file = create_workspace_registry(workspace_path, developer_id, template)
            click.echo(f"✅ Created {template} registry template")
        
        # Create workspace documentation
        readme_file = create_workspace_documentation(Path(workspace_path), developer_id, registry_file)
        click.echo("✅ Created workspace documentation")
        
        # Create example implementations
        create_example_implementations(Path(workspace_path), developer_id)
        click.echo("✅ Created example step implementations")
        
        # Validate setup
        validate_workspace_setup(workspace_path, developer_id)
        click.echo("✅ Validated workspace setup")
        
        # Success summary
        click.echo(f"\n🎉 Developer workspace successfully created!")
        click.echo(f"📝 Registry file: {registry_file}")
        click.echo(f"📖 Documentation: {readme_file}")
        click.echo(f"\n🚀 Next steps:")
        click.echo(f"   1. Edit {registry_file} to add your custom steps")
        click.echo(f"   2. Implement your step components in src/cursus_dev/steps/")
        click.echo(f"   3. Test with: python -m cursus.cli.registry validate-registry --workspace {developer_id}")
        click.echo(f"   4. Set workspace context: export CURSUS_WORKSPACE_ID={developer_id}")
        
    except Exception as e:
        click.echo(f"❌ Failed to create developer workspace: {e}")
        # Cleanup on failure
        if workspace_dir.exists():
            import shutil
            shutil.rmtree(workspace_dir, ignore_errors=True)
            click.echo("🧹 Cleaned up partial workspace creation")


@registry_cli.command('list-steps')
@click.option('--workspace', help='Workspace ID to list steps from')
@click.option('--conflicts-only', is_flag=True, help='Show only conflicting steps')
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
def list_steps(workspace: Optional[str], conflicts_only: bool, format: str):
    """List steps in registry with optional workspace context."""
    try:
        # Import here to avoid circular imports
        from ..registry.hybrid.manager import UnifiedRegistryManager
        
        registry_manager = UnifiedRegistryManager()
        
        if conflicts_only:
            conflicts = registry_manager.get_step_conflicts()
            if format == 'json':
                # Convert to serializable format
                conflicts_data = {}
                for step_name, definitions in conflicts.items():
                    conflicts_data[step_name] = [
                        {
                            'workspace_id': d.workspace_id,
                            'registry_type': d.registry_type,
                            'description': d.description
                        } for d in definitions
                    ]
                click.echo(json.dumps(conflicts_data, indent=2))
            else:
                if conflicts:
                    click.echo("Step Name Conflicts:")
                    for step_name, definitions in conflicts.items():
                        workspaces = [d.workspace_id or 'core' for d in definitions]
                        click.echo(f"  {step_name}: {', '.join(workspaces)}")
                else:
                    click.echo("No step name conflicts found.")
        else:
            step_definitions = registry_manager.get_all_step_definitions(workspace)
            if format == 'json':
                legacy_dict = {name: defn.to_legacy_format() for name, defn in step_definitions.items()}
                click.echo(json.dumps(legacy_dict, indent=2))
            else:
                click.echo(f"Steps in {'workspace ' + workspace if workspace else 'core registry'}:")
                for step_name, definition in step_definitions.items():
                    source = f"({definition.workspace_id})" if definition.workspace_id else "(core)"
                    click.echo(f"  {step_name}: {definition.description} {source}")
                    
    except Exception as e:
        click.echo(f"❌ Error listing steps: {e}")


@registry_cli.command('resolve-step')
@click.argument('step_name')
@click.option('--workspace', help='Workspace context')
@click.option('--framework', help='Preferred framework')
@click.option('--environment', help='Environment tags (comma-separated)')
def resolve_step(step_name: str, workspace: Optional[str], framework: Optional[str], environment: Optional[str]):
    """Resolve step with conflict resolution."""
    try:
        # Import here to avoid circular imports
        from ..registry.hybrid.manager import UnifiedRegistryManager
        
        registry_manager = UnifiedRegistryManager()
        
        definition = registry_manager.get_step_definition(step_name, workspace)
        
        if definition:
            source = f"workspace '{definition.workspace_id}'" if definition.workspace_id else "core registry"
            click.echo(f"✅ Resolved '{step_name}' from {source}")
            click.echo(f"   Config: {definition.config_class}")
            click.echo(f"   Builder: {definition.builder_step_name}")
            click.echo(f"   Framework: {getattr(definition, 'framework', 'N/A') or 'N/A'}")
            click.echo(f"   Description: {definition.description}")
        else:
            click.echo(f"❌ Could not resolve step '{step_name}'")
            
    except Exception as e:
        click.echo(f"❌ Error resolving step: {e}")


@registry_cli.command('validate-registry')
@click.option('--workspace', help='Validate specific workspace registry')
@click.option('--check-conflicts', is_flag=True, help='Check for step name conflicts')
def validate_registry(workspace: Optional[str], check_conflicts: bool):
    """Validate registry consistency and conflicts."""
    try:
        # Import here to avoid circular imports
        from ..registry.hybrid.manager import UnifiedRegistryManager
        
        registry_manager = UnifiedRegistryManager()
        
        if workspace:
            # Validate specific workspace
            if workspace in registry_manager._workspace_steps:
                # Basic validation - check if registry can be loaded
                try:
                    local_definitions = registry_manager.get_local_only_definitions(workspace)
                    click.echo(f"✅ Workspace '{workspace}' registry is valid ({len(local_definitions)} local steps)")
                except Exception as e:
                    click.echo(f"❌ Workspace '{workspace}' registry error: {e}")
            else:
                click.echo(f"❌ Workspace '{workspace}' not found")
        else:
            # Validate all registries
            click.echo("Validating all registries...")
            
            # Check core registry
            try:
                core_definitions = registry_manager._core_steps
                click.echo(f"✅ Core registry: {len(core_definitions)} steps")
            except Exception as e:
                click.echo(f"❌ Core registry error: {e}")
            
            # Check workspace registries
            for workspace_id in registry_manager._workspace_steps:
                try:
                    local_definitions = registry_manager.get_local_only_definitions(workspace_id)
                    click.echo(f"✅ Workspace '{workspace_id}': {len(local_definitions)} local steps")
                except Exception as e:
                    click.echo(f"❌ Workspace '{workspace_id}' error: {e}")
        
        if check_conflicts:
            conflicts = registry_manager.get_step_conflicts()
            if conflicts:
                click.echo("\n⚠️  Step Name Conflicts Found:")
                for step_name, definitions in conflicts.items():
                    workspaces = [d.workspace_id or 'core' for d in definitions]
                    click.echo(f"  {step_name}: {', '.join(workspaces)}")
            else:
                click.echo("\n✅ No step name conflicts found")
                
    except Exception as e:
        click.echo(f"❌ Error validating registry: {e}")


@registry_cli.command('show-workspace')
@click.argument('workspace_id')
def show_workspace(workspace_id: str):
    """Show detailed information about a workspace registry."""
    try:
        # Import here to avoid circular imports
        from ..registry.hybrid.manager import UnifiedRegistryManager
        
        registry_manager = UnifiedRegistryManager()
        
        if workspace_id not in registry_manager._workspace_steps:
            click.echo(f"❌ Workspace '{workspace_id}' not found")
            return
        
        # Get workspace information
        local_definitions = registry_manager.get_local_only_definitions(workspace_id)
        all_definitions = registry_manager.get_all_step_definitions(workspace_id)
        
        click.echo(f"📋 Workspace: {workspace_id}")
        click.echo(f"📁 Path: {registry_manager.workspaces_root / workspace_id}")
        click.echo(f"📊 Local steps: {len(local_definitions)}")
        click.echo(f"📊 Total accessible steps: {len(all_definitions)}")
        
        if local_definitions:
            click.echo(f"\n🔧 Local Steps:")
            for step_name, definition in local_definitions.items():
                click.echo(f"  {step_name}: {definition.description}")
        
        # Check for overrides
        overrides = {name: defn for name, defn in local_definitions.items() 
                    if defn.registry_type == 'override'}
        if overrides:
            click.echo(f"\n🔄 Step Overrides:")
            for step_name, definition in overrides.items():
                click.echo(f"  {step_name}: {definition.description}")
                
    except Exception as e:
        click.echo(f"❌ Error showing workspace: {e}")


# Add the registry CLI to the main CLI group
def register_registry_cli(main_cli):
    """Register registry CLI commands with the main CLI."""
    main_cli.add_command(registry_cli)
