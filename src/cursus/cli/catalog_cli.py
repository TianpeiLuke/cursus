"""
CLI commands for step catalog management.

This module provides command-line tools for:
- Discovering and managing steps across workspaces
- Searching steps by name, framework, and components
- Viewing step information and components
- Managing workspace discovery
- Validating step catalog integrity
"""

import os
import sys
import click
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group(name="catalog")
def catalog_cli():
    """Step catalog management commands."""
    pass


@catalog_cli.command("list")
@click.option("--workspace", help="Filter by workspace ID")
@click.option("--job-type", help="Filter by job type (e.g., training, validation)")
@click.option("--framework", help="Filter by detected framework")
@click.option("--format", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--limit", type=int, help="Maximum number of results to show")
def list_steps(workspace: Optional[str], job_type: Optional[str], framework: Optional[str], format: str, limit: Optional[int]):
    """List available steps with optional filtering."""
    try:
        from ..step_catalog import StepCatalog
        
        catalog = StepCatalog()
        
        # Get all steps
        steps = catalog.list_available_steps(workspace_id=workspace, job_type=job_type)
        
        # Apply framework filter if specified
        if framework:
            filtered_steps = []
            for step_name in steps:
                detected_framework = catalog.detect_framework(step_name)
                if detected_framework and detected_framework.lower() == framework.lower():
                    filtered_steps.append(step_name)
            steps = filtered_steps
        
        # Apply limit if specified
        if limit:
            steps = steps[:limit]
        
        if format == "json":
            result = {
                "steps": steps,
                "total": len(steps),
                "filters": {
                    "workspace": workspace,
                    "job_type": job_type,
                    "framework": framework
                }
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n📂 Available Steps ({len(steps)} found):")
            click.echo("=" * 50)
            
            if not steps:
                click.echo("No steps found matching the criteria.")
                return
            
            for i, step_name in enumerate(steps, 1):
                # Get additional info for display
                step_info = catalog.get_step_info(step_name)
                workspace_info = f" [{step_info.workspace_id}]" if step_info and step_info.workspace_id != "core" else ""
                framework_info = catalog.detect_framework(step_name)
                framework_display = f" ({framework_info})" if framework_info else ""
                
                click.echo(f"{i:3d}. {step_name}{workspace_info}{framework_display}")
            
            # Show applied filters
            filters = []
            if workspace:
                filters.append(f"workspace: {workspace}")
            if job_type:
                filters.append(f"job_type: {job_type}")
            if framework:
                filters.append(f"framework: {framework}")
            
            if filters:
                click.echo(f"\nFilters applied: {', '.join(filters)}")
    
    except Exception as e:
        click.echo(f"❌ Failed to list steps: {e}")
        logger.error(f"Failed to list steps: {e}")


@catalog_cli.command("search")
@click.argument("query")
@click.option("--job-type", help="Filter by job type")
@click.option("--format", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--limit", type=int, default=10, help="Maximum number of results")
def search_steps(query: str, job_type: Optional[str], format: str, limit: int):
    """Search steps by name with fuzzy matching."""
    try:
        from ..step_catalog import StepCatalog
        
        catalog = StepCatalog()
        results = catalog.search_steps(query, job_type=job_type)
        
        # Apply limit
        if limit:
            results = results[:limit]
        
        if format == "json":
            json_results = []
            for result in results:
                json_results.append({
                    "step_name": result.step_name,
                    "workspace_id": result.workspace_id,
                    "match_score": result.match_score,
                    "match_reason": result.match_reason,
                    "components_available": result.components_available
                })
            
            click.echo(json.dumps({
                "query": query,
                "results": json_results,
                "total": len(results)
            }, indent=2))
        else:
            click.echo(f"\n🔍 Search Results for '{query}' ({len(results)} found):")
            click.echo("=" * 60)
            
            if not results:
                click.echo("No steps found matching the search query.")
                return
            
            for i, result in enumerate(results, 1):
                workspace_info = f" [{result.workspace_id}]" if result.workspace_id != "core" else ""
                components_info = f" ({len(result.components_available)} components)" if result.components_available else ""
                
                click.echo(f"{i:3d}. {result.step_name}{workspace_info} (score: {result.match_score:.2f}){components_info}")
                click.echo(f"     Reason: {result.match_reason}")
    
    except Exception as e:
        click.echo(f"❌ Failed to search steps: {e}")
        logger.error(f"Failed to search steps: {e}")


@catalog_cli.command("show")
@click.argument("step_name")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.option("--show-components", is_flag=True, help="Show detailed component information")
def show_step(step_name: str, format: str, show_components: bool):
    """Show detailed information about a specific step."""
    try:
        from ..step_catalog import StepCatalog
        
        catalog = StepCatalog()
        step_info = catalog.get_step_info(step_name)
        
        if not step_info:
            click.echo(f"❌ Step not found: {step_name}")
            return
        
        if format == "json":
            result = {
                "step_name": step_info.step_name,
                "workspace_id": step_info.workspace_id,
                "registry_data": step_info.registry_data,
                "file_components": {}
            }
            
            # Add file components info
            for comp_type, metadata in step_info.file_components.items():
                if metadata:
                    result["file_components"][comp_type] = {
                        "path": str(metadata.path),
                        "file_type": metadata.file_type,
                        "modified_time": metadata.modified_time.isoformat() if metadata.modified_time else None
                    }
            
            # Add framework detection
            framework = catalog.detect_framework(step_name)
            if framework:
                result["detected_framework"] = framework
            
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n📋 Step: {step_name}")
            click.echo("=" * (len(step_name) + 8))
            
            click.echo(f"Workspace: {step_info.workspace_id}")
            
            # Show framework if detected
            framework = catalog.detect_framework(step_name)
            if framework:
                click.echo(f"Framework: {framework}")
            
            # Show registry data
            if step_info.registry_data:
                click.echo(f"\n📝 Registry Information:")
                for key, value in step_info.registry_data.items():
                    if key not in ['__module__', '__qualname__']:
                        click.echo(f"  {key}: {value}")
            
            # Show file components
            if step_info.file_components:
                click.echo(f"\n🔧 Available Components:")
                for comp_type, metadata in step_info.file_components.items():
                    if metadata:
                        click.echo(f"  {comp_type}: {metadata.path}")
                        if show_components and metadata.modified_time:
                            click.echo(f"    Modified: {metadata.modified_time}")
            
            # Show job type variants
            if "_" not in step_name:  # Only for base step names
                variants = catalog.get_job_type_variants(step_name)
                if variants:
                    click.echo(f"\n🔄 Job Type Variants:")
                    for variant in variants:
                        click.echo(f"  {step_name}_{variant}")
    
    except Exception as e:
        click.echo(f"❌ Failed to show step: {e}")
        logger.error(f"Failed to show step: {e}")


@catalog_cli.command("components")
@click.argument("step_name")
@click.option("--type", "component_type", help="Filter by component type (script, contract, spec, builder, config)")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def show_components(step_name: str, component_type: Optional[str], format: str):
    """Show components available for a specific step."""
    try:
        from ..step_catalog import StepCatalog
        
        catalog = StepCatalog()
        step_info = catalog.get_step_info(step_name)
        
        if not step_info:
            click.echo(f"❌ Step not found: {step_name}")
            return
        
        components = step_info.file_components
        
        # Apply component type filter
        if component_type:
            components = {k: v for k, v in components.items() if k == component_type}
        
        if format == "json":
            result = {
                "step_name": step_name,
                "components": {}
            }
            
            for comp_type, metadata in components.items():
                if metadata:
                    result["components"][comp_type] = {
                        "path": str(metadata.path),
                        "file_type": metadata.file_type,
                        "modified_time": metadata.modified_time.isoformat() if metadata.modified_time else None
                    }
            
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🔧 Components for {step_name}:")
            click.echo("=" * (len(step_name) + 16))
            
            if not components:
                click.echo("No components found.")
                return
            
            for comp_type, metadata in components.items():
                if metadata:
                    click.echo(f"\n{comp_type.upper()}:")
                    click.echo(f"  Path: {metadata.path}")
                    click.echo(f"  Type: {metadata.file_type}")
                    if metadata.modified_time:
                        click.echo(f"  Modified: {metadata.modified_time}")
    
    except Exception as e:
        click.echo(f"❌ Failed to show components: {e}")
        logger.error(f"Failed to show components: {e}")


@catalog_cli.command("frameworks")
@click.option("--format", type=click.Choice(["table", "json"]), default="table", help="Output format")
def list_frameworks(format: str):
    """List detected frameworks across all steps."""
    try:
        from ..step_catalog import StepCatalog
        
        catalog = StepCatalog()
        steps = catalog.list_available_steps()
        
        framework_counts = {}
        step_frameworks = {}
        
        for step_name in steps:
            framework = catalog.detect_framework(step_name)
            if framework:
                framework_counts[framework] = framework_counts.get(framework, 0) + 1
                step_frameworks.setdefault(framework, []).append(step_name)
        
        if format == "json":
            result = {
                "frameworks": framework_counts,
                "steps_by_framework": step_frameworks
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🔧 Detected Frameworks ({len(framework_counts)} total):")
            click.echo("=" * 40)
            
            if not framework_counts:
                click.echo("No frameworks detected.")
                return
            
            for framework, count in sorted(framework_counts.items()):
                click.echo(f"{framework}: {count} steps")
                
                # Show first few steps as examples
                example_steps = step_frameworks[framework][:3]
                for step in example_steps:
                    click.echo(f"  - {step}")
                
                if len(step_frameworks[framework]) > 3:
                    remaining = len(step_frameworks[framework]) - 3
                    click.echo(f"  ... and {remaining} more")
    
    except Exception as e:
        click.echo(f"❌ Failed to list frameworks: {e}")
        logger.error(f"Failed to list frameworks: {e}")


@catalog_cli.command("workspaces")
@click.option("--format", type=click.Choice(["table", "json"]), default="table", help="Output format")
def list_workspaces(format: str):
    """List available workspaces and their step counts."""
    try:
        from ..step_catalog import StepCatalog
        
        catalog = StepCatalog()
        cross_workspace = catalog.discover_cross_workspace_components()
        
        if format == "json":
            result = {
                "workspaces": {}
            }
            
            for workspace_id, components in cross_workspace.items():
                steps = catalog.list_available_steps(workspace_id=workspace_id)
                result["workspaces"][workspace_id] = {
                    "step_count": len(steps),
                    "component_count": len(components),
                    "steps": steps
                }
            
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🏢 Available Workspaces ({len(cross_workspace)} total):")
            click.echo("=" * 40)
            
            if not cross_workspace:
                click.echo("No workspaces found.")
                return
            
            for workspace_id, components in cross_workspace.items():
                steps = catalog.list_available_steps(workspace_id=workspace_id)
                click.echo(f"\n{workspace_id}:")
                click.echo(f"  Steps: {len(steps)}")
                click.echo(f"  Components: {len(components)}")
                
                # Show first few steps as examples
                if steps:
                    example_steps = steps[:3]
                    click.echo(f"  Example steps:")
                    for step in example_steps:
                        click.echo(f"    - {step}")
                    
                    if len(steps) > 3:
                        remaining = len(steps) - 3
                        click.echo(f"    ... and {remaining} more")
    
    except Exception as e:
        click.echo(f"❌ Failed to list workspaces: {e}")
        logger.error(f"Failed to list workspaces: {e}")


@catalog_cli.command("metrics")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def show_metrics(format: str):
    """Show step catalog performance metrics."""
    try:
        from ..step_catalog import StepCatalog
        
        catalog = StepCatalog()
        metrics = catalog.get_metrics_report()
        
        if format == "json":
            click.echo(json.dumps(metrics, indent=2))
        else:
            click.echo(f"\n📊 Step Catalog Metrics:")
            click.echo("=" * 25)
            
            click.echo(f"Total Queries: {metrics['total_queries']}")
            click.echo(f"Success Rate: {metrics['success_rate']:.1%}")
            click.echo(f"Average Response Time: {metrics['avg_response_time_ms']:.2f}ms")
            click.echo(f"Index Build Time: {metrics['index_build_time_s']:.3f}s")
            click.echo(f"Total Steps Indexed: {metrics['total_steps_indexed']}")
            click.echo(f"Total Workspaces: {metrics['total_workspaces']}")
            
            if metrics['last_index_build']:
                click.echo(f"Last Index Build: {metrics['last_index_build']}")
    
    except Exception as e:
        click.echo(f"❌ Failed to show metrics: {e}")
        logger.error(f"Failed to show metrics: {e}")


@catalog_cli.command("discover")
@click.option("--workspace-dir", type=click.Path(exists=True), help="Workspace directory to discover")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def discover_workspace(workspace_dir: Optional[str], format: str):
    """Discover steps in a specific workspace directory."""
    try:
        from ..step_catalog import StepCatalog
        
        if workspace_dir:
            workspace_dirs = [Path(workspace_dir)]
        else:
            click.echo("❌ Please specify a workspace directory with --workspace-dir")
            return
        
        catalog = StepCatalog(workspace_dirs=workspace_dirs)
        workspace_id = Path(workspace_dir).name
        steps = catalog.list_available_steps(workspace_id=workspace_id)
        
        if format == "json":
            result = {
                "workspace_dir": workspace_dir,
                "workspace_id": workspace_id,
                "discovered_steps": steps,
                "total": len(steps)
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🔍 Discovery Results for {workspace_dir}:")
            click.echo("=" * 50)
            click.echo(f"Workspace ID: {workspace_id}")
            click.echo(f"Steps Found: {len(steps)}")
            
            if steps:
                click.echo(f"\nDiscovered Steps:")
                for i, step_name in enumerate(steps, 1):
                    step_info = catalog.get_step_info(step_name)
                    components = list(step_info.file_components.keys()) if step_info else []
                    components_info = f" ({', '.join(components)})" if components else ""
                    click.echo(f"{i:3d}. {step_name}{components_info}")
            else:
                click.echo("\nNo steps found in the specified workspace directory.")
    
    except Exception as e:
        click.echo(f"❌ Failed to discover workspace: {e}")
        logger.error(f"Failed to discover workspace: {e}")


def main():
    """Main entry point for catalog CLI."""
    return catalog_cli()


if __name__ == "__main__":
    main()
