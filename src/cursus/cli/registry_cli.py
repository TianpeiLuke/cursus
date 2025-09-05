"""
CLI commands for hybrid registry management.

This module provides command-line tools for:
- Initializing developer workspaces
- Managing workspace registries
- Validating registry configurations
- Detecting and resolving conflicts
"""

import os
import sys
import click
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@click.group(name='registry')
def registry_cli():
    """Registry management commands for hybrid registry system."""
    pass

@registry_cli.command('init-workspace')
@click.argument('workspace_id')
@click.option('--workspace-path', help='Custom workspace path (default: developer_workspaces/developers/{workspace_id})')
@click.option('--template', default='standard', type=click.Choice(['minimal', 'standard', 'advanced']), 
              help='Registry template to use')
@click.option('--force', is_flag=True, help='Overwrite existing workspace if it exists')
def init_workspace(workspace_id: str, workspace_path: Optional[str], template: str, force: bool):
    """
    Initialize a new developer workspace with hybrid registry support.
    
    Creates a complete workspace structure including:
    - Directory structure for custom step implementations
    - Local registry configuration
    - Documentation and usage examples
    - Integration with hybrid registry system
    
    Args:
        workspace_id: Unique identifier for the developer workspace
        workspace_path: Custom workspace path (optional)
        template: Registry template type (minimal/standard/advanced)
        force: Overwrite existing workspace
    """
    try:
        # Validate workspace ID
        if not workspace_id or not workspace_id.replace('_', '').replace('-', '').isalnum():
            click.echo(f"❌ Invalid workspace ID: {workspace_id}")
            click.echo("   Workspace ID must contain only alphanumeric characters, hyphens, and underscores")
            return
        
        # Determine workspace path
        if not workspace_path:
            workspace_path = f"developer_workspaces/developers/{workspace_id}"
        
        workspace_dir = Path(workspace_path)
        
        # Check if workspace already exists
        if workspace_dir.exists() and not force:
            click.echo(f"❌ Workspace already exists: {workspace_path}")
            click.echo("   Use --force to overwrite or choose a different path")
            return
        
        click.echo(f"🚀 Initializing developer workspace: {workspace_id}")
        click.echo(f"📁 Workspace path: {workspace_path}")
        
        # Create workspace directory structure
        _create_workspace_structure(workspace_dir)
        click.echo("✅ Created workspace directory structure")
        
        # Create registry configuration
        registry_file = _create_workspace_registry(workspace_dir, workspace_id, template)
        click.echo(f"✅ Created {template} registry template")
        
        # Create workspace documentation
        readme_file = _create_workspace_documentation(workspace_dir, workspace_id, registry_file)
        click.echo("✅ Created workspace documentation")
        
        # Create example implementations
        _create_example_implementations(workspace_dir, workspace_id, template)
        click.echo("✅ Created example step implementations")
        
        # Success summary
        click.echo(f"\n🎉 Developer workspace successfully created!")
        click.echo(f"📝 Registry file: {registry_file}")
        click.echo(f"📖 Documentation: {readme_file}")
        click.echo(f"\n🚀 Next steps:")
        click.echo(f"   1. Edit {registry_file} to add your custom steps")
        click.echo(f"   2. Implement your step components in src/cursus_dev/steps/")
        click.echo(f"   3. Test with: python -m cursus.cli.registry validate-registry --workspace {workspace_id}")
        click.echo(f"   4. Set workspace context: export CURSUS_WORKSPACE_ID={workspace_id}")
        
    except Exception as e:
        click.echo(f"❌ Failed to create developer workspace: {e}")
        # Cleanup on failure
        if workspace_dir.exists():
            import shutil
            shutil.rmtree(workspace_dir, ignore_errors=True)
            click.echo("🧹 Cleaned up partial workspace creation")

@registry_cli.command('list-steps')
@click.option('--workspace', help='Workspace ID to list steps for')
@click.option('--conflicts-only', is_flag=True, help='Show only conflicting steps')
@click.option('--include-source', is_flag=True, help='Include source registry information')
def list_steps(workspace: Optional[str], conflicts_only: bool, include_source: bool):
    """List available steps in registry with optional workspace context."""
    try:
        from cursus.registry import (
            get_all_step_names, get_workspace_context, 
            has_workspace_conflicts, list_available_workspaces
        )
        
        effective_workspace = workspace or get_workspace_context()
        
        if conflicts_only:
            # Show only conflicting steps
            try:
                from cursus.registry.hybrid.manager import UnifiedRegistryManager
                manager = UnifiedRegistryManager()
                conflicts = manager.get_step_conflicts()
                
                if not conflicts:
                    click.echo("✅ No step name conflicts detected")
                    return
                
                click.echo(f"⚠️  Found {len(conflicts)} conflicting steps:")
                for step_name, definitions in conflicts.items():
                    click.echo(f"\n📍 Step: {step_name}")
                    for definition in definitions:
                        workspace_info = f" (workspace: {definition.workspace_id})" if definition.workspace_id else " (core)"
                        click.echo(f"   - {definition.registry_type}{workspace_info}")
                        
            except ImportError:
                click.echo("❌ Hybrid registry not available - cannot check conflicts")
                return
        else:
            # Show all steps
            steps = get_all_step_names(effective_workspace)
            
            if include_source:
                try:
                    from cursus.registry.hybrid.manager import UnifiedRegistryManager
                    manager = UnifiedRegistryManager()
                    all_steps = manager.list_all_steps(include_source=True)
                    
                    for source, step_list in all_steps.items():
                        click.echo(f"\n📂 {source.upper()} Registry:")
                        for step in sorted(step_list):
                            click.echo(f"   - {step}")
                            
                except ImportError:
                    click.echo(f"\n📂 Registry Steps ({len(steps)} total):")
                    for step in sorted(steps):
                        click.echo(f"   - {step}")
            else:
                workspace_info = f" (workspace: {effective_workspace})" if effective_workspace else " (core registry)"
                click.echo(f"\n📂 Available Steps{workspace_info} ({len(steps)} total):")
                for step in sorted(steps):
                    click.echo(f"   - {step}")
                    
    except Exception as e:
        click.echo(f"❌ Failed to list steps: {e}")

@registry_cli.command('validate-registry')
@click.option('--workspace', help='Workspace ID to validate')
@click.option('--check-conflicts', is_flag=True, help='Check for step name conflicts')
def validate_registry(workspace: Optional[str], check_conflicts: bool):
    """Validate registry configuration and check for issues."""
    try:
        from cursus.registry import get_workspace_context, get_all_step_names
        
        effective_workspace = workspace or get_workspace_context()
        
        click.echo(f"🔍 Validating registry...")
        if effective_workspace:
            click.echo(f"📁 Workspace: {effective_workspace}")
        else:
            click.echo(f"📁 Core registry")
        
        # Basic validation
        steps = get_all_step_names(effective_workspace)
        click.echo(f"✅ Found {len(steps)} steps")
        
        # Check for conflicts if requested
        if check_conflicts:
            try:
                from cursus.registry.hybrid.manager import UnifiedRegistryManager
                manager = UnifiedRegistryManager()
                conflicts = manager.get_step_conflicts()
                
                if conflicts:
                    click.echo(f"⚠️  Found {len(conflicts)} step name conflicts:")
                    for step_name, definitions in conflicts.items():
                        click.echo(f"   - {step_name}: {len(definitions)} definitions")
                else:
                    click.echo("✅ No step name conflicts detected")
                    
            except ImportError:
                click.echo("⚠️  Hybrid registry not available - skipping conflict check")
        
        # Registry status
        try:
            from cursus.registry.hybrid.manager import UnifiedRegistryManager
            manager = UnifiedRegistryManager()
            status = manager.get_registry_status()
            
            click.echo(f"\n📊 Registry Status:")
            for registry_id, info in status.items():
                if registry_id == 'core':
                    click.echo(f"   📂 Core: {info['step_count']} steps")
                else:
                    local_count = info.get('local_step_count', 0)
                    override_count = info.get('override_count', 0)
                    click.echo(f"   📂 {registry_id}: {local_count} local, {override_count} overrides")
                    
        except ImportError:
            click.echo("⚠️  Hybrid registry not available - limited status information")
        
        click.echo(f"\n✅ Registry validation completed")
        
    except Exception as e:
        click.echo(f"❌ Registry validation failed: {e}")

@registry_cli.command('resolve-step')
@click.argument('step_name')
@click.option('--workspace', help='Workspace context for resolution')
@click.option('--framework', help='Preferred framework for resolution')
def resolve_step(step_name: str, workspace: Optional[str], framework: Optional[str]):
    """Resolve a specific step name and show resolution details."""
    try:
        from cursus.registry import get_workspace_context
        
        effective_workspace = workspace or get_workspace_context()
        
        click.echo(f"🔍 Resolving step: {step_name}")
        if effective_workspace:
            click.echo(f"📁 Workspace context: {effective_workspace}")
        if framework:
            click.echo(f"🔧 Preferred framework: {framework}")
        
        try:
            from cursus.registry.hybrid.manager import UnifiedRegistryManager
            from cursus.registry.hybrid.models import ResolutionContext
            
            manager = UnifiedRegistryManager()
            context = ResolutionContext(
                workspace_id=effective_workspace,
                preferred_framework=framework
            )
            
            result = manager.get_step(step_name, context)
            
            if result.resolved:
                click.echo(f"✅ Step resolved successfully")
                click.echo(f"   📂 Source: {result.source_registry}")
                click.echo(f"   🔧 Strategy: {result.resolution_strategy}")
                if result.selected_definition:
                    click.echo(f"   📝 Config: {result.selected_definition.config_class}")
                    click.echo(f"   🏗️  Builder: {result.selected_definition.builder_step_name}")
                    if result.selected_definition.framework:
                        click.echo(f"   🔧 Framework: {result.selected_definition.framework}")
            else:
                click.echo(f"❌ Step resolution failed")
                for error in result.errors:
                    click.echo(f"   ❌ {error}")
                    
        except ImportError:
            # Fallback to basic resolution
            from cursus.registry import get_config_class_name, get_builder_step_name
            
            try:
                config_class = get_config_class_name(step_name, effective_workspace)
                builder_class = get_builder_step_name(step_name, effective_workspace)
                
                click.echo(f"✅ Step found (basic resolution)")
                click.echo(f"   📝 Config: {config_class}")
                click.echo(f"   🏗️  Builder: {builder_class}")
                
            except ValueError as e:
                click.echo(f"❌ Step not found: {e}")
        
    except Exception as e:
        click.echo(f"❌ Step resolution failed: {e}")

# Helper functions for workspace creation
def _create_workspace_structure(workspace_dir: Path) -> None:
    """Create complete workspace directory structure."""
    directories = [
        "src/cursus_dev/steps/builders",
        "src/cursus_dev/steps/configs", 
        "src/cursus_dev/steps/contracts",
        "src/cursus_dev/steps/scripts",
        "src/cursus_dev/steps/specs",
        "src/cursus_dev/registry",
        "test/unit",
        "test/integration", 
        "validation_reports",
        "examples",
        "docs"
    ]
    
    for dir_path in directories:
        full_path = workspace_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if "src/cursus_dev" in dir_path:
            init_file = full_path / "__init__.py"
            init_file.write_text('"""Package initialization."""\n')

def _create_workspace_registry(workspace_dir: Path, workspace_id: str, template: str) -> str:
    """Create workspace registry configuration file."""
    registry_file = workspace_dir / "src/cursus_dev/registry/workspace_registry.py"
    
    if template == 'minimal':
        content = f'''"""
Minimal workspace registry for {workspace_id}.
"""

# Workspace metadata
WORKSPACE_METADATA = {{
    "developer_id": "{workspace_id}",
    "template": "minimal",
    "description": "Minimal workspace registry for {workspace_id}"
}}

# Local step definitions (new steps specific to this workspace)
LOCAL_STEPS = {{
    # Add your custom steps here
    # Example:
    # "MyCustomStep": {{
    #     "config_class": "MyCustomStepConfig",
    #     "builder_step_name": "MyCustomStepBuilder",
    #     "spec_type": "MyCustomStep",
    #     "sagemaker_step_type": "Processing",
    #     "description": "My custom processing step"
    # }}
}}

# Step overrides (override core steps with custom implementations)
STEP_OVERRIDES = {{
    # Add step overrides here if needed
    # Example:
    # "XGBoostTraining": {{
    #     "config_class": "CustomXGBoostTrainingConfig",
    #     "builder_step_name": "CustomXGBoostTrainingStepBuilder",
    #     "spec_type": "CustomXGBoostTraining",
    #     "sagemaker_step_type": "Training",
    #     "description": "Custom XGBoost training with enhanced features"
    # }}
}}
'''
    elif template == 'advanced':
        content = f'''"""
Advanced workspace registry for {workspace_id}.
"""

# Workspace metadata
WORKSPACE_METADATA = {{
    "developer_id": "{workspace_id}",
    "template": "advanced",
    "description": "Advanced workspace registry for {workspace_id}",
    "version": "1.0.0",
    "frameworks": ["pytorch", "xgboost", "sklearn"],
    "environment_tags": ["development", "gpu"],
    "contact": "developer@company.com"
}}

# Local step definitions (new steps specific to this workspace)
LOCAL_STEPS = {{
    "CustomDataPreprocessing": {{
        "config_class": "CustomDataPreprocessingConfig",
        "builder_step_name": "CustomDataPreprocessingStepBuilder",
        "spec_type": "CustomDataPreprocessing",
        "sagemaker_step_type": "Processing",
        "description": "Custom data preprocessing with advanced transformations",
        "framework": "pandas",
        "environment_tags": ["development"],
        "priority": 90,
        "conflict_resolution_strategy": "workspace_priority"
    }},
    "AdvancedModelEvaluation": {{
        "config_class": "AdvancedModelEvaluationConfig",
        "builder_step_name": "AdvancedModelEvaluationStepBuilder",
        "spec_type": "AdvancedModelEvaluation",
        "sagemaker_step_type": "Processing",
        "description": "Advanced model evaluation with custom metrics",
        "framework": "sklearn",
        "environment_tags": ["development", "gpu"],
        "priority": 85,
        "conflict_resolution_strategy": "framework_match"
    }}
}}

# Step overrides (override core steps with custom implementations)
STEP_OVERRIDES = {{
    # Example: Override XGBoost training with custom implementation
    # "XGBoostTraining": {{
    #     "config_class": "EnhancedXGBoostTrainingConfig",
    #     "builder_step_name": "EnhancedXGBoostTrainingStepBuilder",
    #     "spec_type": "EnhancedXGBoostTraining",
    #     "sagemaker_step_type": "Training",
    #     "description": "Enhanced XGBoost training with hyperparameter optimization",
    #     "framework": "xgboost",
    #     "environment_tags": ["production", "gpu"],
    #     "priority": 75,
    #     "conflict_resolution_strategy": "workspace_priority"
    # }}
}}
'''
    else:  # standard template
        content = f'''"""
Standard workspace registry for {workspace_id}.
"""

# Workspace metadata
WORKSPACE_METADATA = {{
    "developer_id": "{workspace_id}",
    "template": "standard",
    "description": "Standard workspace registry for {workspace_id}",
    "version": "1.0.0"
}}

# Local step definitions (new steps specific to this workspace)
LOCAL_STEPS = {{
    "CustomProcessingStep": {{
        "config_class": "CustomProcessingStepConfig",
        "builder_step_name": "CustomProcessingStepBuilder",
        "spec_type": "CustomProcessingStep",
        "sagemaker_step_type": "Processing",
        "description": "Custom processing step for {workspace_id}",
        "framework": "pandas",
        "priority": 90
    }}
}}

# Step overrides (override core steps with custom implementations)
STEP_OVERRIDES = {{
    # Add step overrides here if needed
}}
'''
    
    registry_file.write_text(content)
    return str(registry_file)

def _create_workspace_documentation(workspace_dir: Path, workspace_id: str, registry_file: str) -> str:
    """Create comprehensive workspace documentation."""
    readme_file = workspace_dir / "README.md"
    readme_content = f"""# Developer Workspace: {workspace_id}

This workspace contains custom step implementations for developer {workspace_id}.

## Quick Start

### 1. Set Workspace Context
```bash
export CURSUS_WORKSPACE_ID={workspace_id}
```

### 2. Add Custom Steps
Edit `{registry_file}` to define your custom steps.

### 3. Implement Step Components
Create the corresponding implementation files in `src/cursus_dev/steps/`.

### 4. Test Your Implementation
```python
from cursus.registry import set_workspace_context, get_config_class_name

set_workspace_context("{workspace_id}")
config_class = get_config_class_name("MyCustomStep")  # Uses your local registry
```

## CLI Commands

```bash
# List steps in this workspace
python -m cursus.cli.registry list-steps --workspace {workspace_id}

# Validate registry
python -m cursus.cli.registry validate-registry --workspace {workspace_id}

# Check for conflicts
python -m cursus.cli.registry validate-registry --workspace {workspace_id} --check-conflicts
```

## Support

For questions or issues, validate your setup:
```bash
python -m cursus.cli.registry validate-registry --workspace {workspace_id}
```
"""
    
    readme_file.write_text(readme_content)
    return str(readme_file)

def _create_example_implementations(workspace_dir: Path, workspace_id: str, template: str) -> None:
    """Create example step implementations for reference."""
    examples_dir = workspace_dir / "examples"
    
    # Create example config
    example_config = examples_dir / "example_custom_step_config.py"
    example_config.write_text(f'''"""
Example custom step configuration for {workspace_id} workspace.
"""
from cursus.core.base.config_base import BasePipelineConfig
from pydantic import Field
from typing import Optional

class ExampleCustomStepConfig(BasePipelineConfig):
    """Example configuration for custom processing step."""
    
    # Custom parameters
    custom_parameter: str = Field(..., description="Custom processing parameter")
    optional_setting: Optional[bool] = Field(default=True, description="Optional setting")
    
    # Workspace identification
    workspace_id: str = Field(default="{workspace_id}", description="Workspace identifier")
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"
        validate_assignment = True
''')

if __name__ == '__main__':
    registry_cli()
