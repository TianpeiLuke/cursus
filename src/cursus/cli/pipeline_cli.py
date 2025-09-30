"""
CLI commands for pipeline catalog management.

This module provides command-line tools for:
- Discovering and managing pipelines
- Searching pipelines by framework, tags, and use cases
- Viewing pipeline connections and alternatives
- Getting pipeline recommendations
- Validating pipeline registry
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


@click.group(name="pipeline")
def pipeline_cli():
    """Pipeline catalog management commands."""
    pass


@pipeline_cli.command("list")
@click.option("--framework", help="Filter by framework (e.g., xgboost, pytorch)")
@click.option("--complexity", help="Filter by complexity level")
@click.option("--tags", help="Comma-separated list of tags to filter by")
@click.option("--format", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--limit", type=int, help="Maximum number of results to show")
def list_pipelines(framework: Optional[str], complexity: Optional[str], tags: Optional[str], format: str, limit: Optional[int]):
    """List available pipelines with optional filtering."""
    try:
        from ..pipeline_catalog import get_catalog_info, discover_all_pipelines
        from ..pipeline_catalog.utils import create_catalog_manager
        
        manager = create_catalog_manager()
        
        # Build search criteria
        criteria = {}
        if framework:
            criteria["framework"] = framework
        if complexity:
            criteria["complexity"] = complexity
        if tags:
            criteria["tags"] = [tag.strip() for tag in tags.split(",")]
        
        # Discover pipelines
        if criteria:
            pipeline_ids = manager.discover_pipelines(**criteria)
        else:
            all_pipelines = discover_all_pipelines()
            pipeline_ids = all_pipelines.get("standard", []) + all_pipelines.get("mods", [])
        
        # Apply limit if specified
        if limit:
            pipeline_ids = pipeline_ids[:limit]
        
        if format == "json":
            result = {
                "pipelines": pipeline_ids,
                "total": len(pipeline_ids),
                "criteria": criteria
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n📂 Available Pipelines ({len(pipeline_ids)} found):")
            click.echo("=" * 50)
            
            if not pipeline_ids:
                click.echo("No pipelines found matching the criteria.")
                return
            
            for i, pipeline_id in enumerate(pipeline_ids, 1):
                click.echo(f"{i:3d}. {pipeline_id}")
            
            if criteria:
                criteria_str = []
                if framework:
                    criteria_str.append(f"framework: {framework}")
                if complexity:
                    criteria_str.append(f"complexity: {complexity}")
                if tags:
                    criteria_str.append(f"tags: {tags}")
                
                click.echo(f"\nFilters applied: {', '.join(criteria_str)}")
    
    except Exception as e:
        click.echo(f"❌ Failed to list pipelines: {e}")
        logger.error(f"Failed to list pipelines: {e}")


@pipeline_cli.command("discover")
@click.option("--framework", help="Discover pipelines by framework")
@click.option("--use-case", help="Discover pipelines by use case description")
@click.option("--tags", help="Comma-separated list of tags to search for")
@click.option("--format", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--limit", type=int, default=10, help="Maximum number of results")
def discover_pipelines(framework: Optional[str], use_case: Optional[str], tags: Optional[str], format: str, limit: int):
    """Discover pipelines using advanced search capabilities."""
    try:
        from ..pipeline_catalog.utils import create_catalog_manager
        
        manager = create_catalog_manager()
        
        # Build search criteria
        criteria = {}
        if framework:
            criteria["framework"] = framework
        if use_case:
            criteria["use_case"] = use_case
        if tags:
            criteria["tags"] = [tag.strip() for tag in tags.split(",")]
        
        if not criteria:
            click.echo("❌ Please specify at least one search criterion (--framework, --use-case, or --tags)")
            return
        
        # Perform discovery
        results = manager.discover_pipelines(**criteria)
        
        # Apply limit
        if limit:
            results = results[:limit]
        
        if format == "json":
            result = {
                "results": results,
                "total": len(results),
                "criteria": criteria,
                "limit": limit
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🔍 Discovery Results ({len(results)} found):")
            click.echo("=" * 40)
            
            if not results:
                click.echo("No pipelines found matching the criteria.")
                return
            
            for i, pipeline_id in enumerate(results, 1):
                click.echo(f"{i:3d}. {pipeline_id}")
            
            # Show search criteria
            criteria_str = []
            if framework:
                criteria_str.append(f"framework: {framework}")
            if use_case:
                criteria_str.append(f"use_case: {use_case}")
            if tags:
                criteria_str.append(f"tags: {tags}")
            
            click.echo(f"\nSearch criteria: {', '.join(criteria_str)}")
    
    except Exception as e:
        click.echo(f"❌ Failed to discover pipelines: {e}")
        logger.error(f"Failed to discover pipelines: {e}")


@pipeline_cli.command("show")
@click.argument("pipeline_id")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.option("--show-connections", is_flag=True, help="Show pipeline connections")
def show_pipeline(pipeline_id: str, format: str, show_connections: bool):
    """Show detailed information about a specific pipeline."""
    try:
        from ..pipeline_catalog import load_pipeline
        from ..pipeline_catalog.utils import create_catalog_manager
        
        # Try to load pipeline details
        try:
            pipeline_info = load_pipeline(pipeline_id)
        except Exception:
            pipeline_info = None
        
        if format == "json":
            result = {
                "pipeline_id": pipeline_id,
                "found": pipeline_info is not None,
                "info": pipeline_info if pipeline_info else None
            }
            
            if show_connections:
                manager = create_catalog_manager()
                connections = manager.get_pipeline_connections(pipeline_id)
                result["connections"] = connections
            
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n📋 Pipeline: {pipeline_id}")
            click.echo("=" * (len(pipeline_id) + 12))
            
            if pipeline_info:
                if hasattr(pipeline_info, '__dict__'):
                    for key, value in pipeline_info.__dict__.items():
                        if not key.startswith('_'):
                            click.echo(f"{key.replace('_', ' ').title()}: {value}")
                else:
                    click.echo(f"Pipeline found: {pipeline_info}")
            else:
                click.echo("Pipeline information not available or pipeline not found.")
            
            if show_connections:
                try:
                    manager = create_catalog_manager()
                    connections = manager.get_pipeline_connections(pipeline_id)
                    
                    if connections:
                        click.echo(f"\n🔗 Connections:")
                        for conn_type, targets in connections.items():
                            if targets:
                                click.echo(f"  {conn_type.title()}: {', '.join(targets)}")
                    else:
                        click.echo(f"\n🔗 No connections found")
                except Exception as e:
                    click.echo(f"\n⚠️  Could not load connections: {e}")
    
    except Exception as e:
        click.echo(f"❌ Failed to show pipeline: {e}")
        logger.error(f"Failed to show pipeline: {e}")


@pipeline_cli.command("connections")
@click.argument("pipeline_id")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def show_connections(pipeline_id: str, format: str):
    """Show connections for a specific pipeline."""
    try:
        from ..pipeline_catalog.utils import create_catalog_manager
        
        manager = create_catalog_manager()
        connections = manager.get_pipeline_connections(pipeline_id)
        
        if format == "json":
            result = {
                "pipeline_id": pipeline_id,
                "connections": connections
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🔗 Connections for {pipeline_id}:")
            click.echo("=" * (len(pipeline_id) + 17))
            
            if not connections:
                click.echo("No connections found.")
                return
            
            for conn_type, targets in connections.items():
                if targets:
                    click.echo(f"\n{conn_type.upper()}:")
                    for target in targets:
                        click.echo(f"  → {target}")
    
    except Exception as e:
        click.echo(f"❌ Failed to show connections: {e}")
        logger.error(f"Failed to show connections: {e}")


@pipeline_cli.command("alternatives")
@click.argument("pipeline_id")
@click.option("--format", type=click.Choice(["table", "json"]), default="table", help="Output format")
def show_alternatives(pipeline_id: str, format: str):
    """Show alternative pipelines for a given pipeline."""
    try:
        from ..pipeline_catalog.utils import get_pipeline_alternatives
        
        alternatives = get_pipeline_alternatives(pipeline_id)
        
        if format == "json":
            result = {
                "pipeline_id": pipeline_id,
                "alternatives": alternatives
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🔄 Alternatives to {pipeline_id}:")
            click.echo("=" * (len(pipeline_id) + 17))
            
            if not alternatives:
                click.echo("No alternatives found.")
                return
            
            for i, alt in enumerate(alternatives, 1):
                click.echo(f"{i:3d}. {alt}")
    
    except Exception as e:
        click.echo(f"❌ Failed to show alternatives: {e}")
        logger.error(f"Failed to show alternatives: {e}")


@pipeline_cli.command("path")
@click.argument("source")
@click.argument("target")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def find_path(source: str, target: str, format: str):
    """Find connection path between two pipelines."""
    try:
        from ..pipeline_catalog.utils import create_catalog_manager
        
        manager = create_catalog_manager()
        path = manager.find_path(source, target)
        
        if format == "json":
            result = {
                "source": source,
                "target": target,
                "path": path,
                "found": path is not None,
                "length": len(path) if path else 0
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🛤️  Path from {source} to {target}:")
            click.echo("=" * 50)
            
            if not path:
                click.echo("No connection path found.")
                return
            
            for i, step in enumerate(path):
                if i == 0:
                    click.echo(f"Start: {step}")
                elif i == len(path) - 1:
                    click.echo(f"End:   {step}")
                else:
                    click.echo(f"Step {i}: {step}")
            
            click.echo(f"\nPath length: {len(path)} steps")
    
    except Exception as e:
        click.echo(f"❌ Failed to find path: {e}")
        logger.error(f"Failed to find path: {e}")


@pipeline_cli.command("recommend")
@click.option("--use-case", required=True, help="Use case to get recommendations for")
@click.option("--framework", help="Preferred framework")
@click.option("--complexity", help="Preferred complexity level")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.option("--limit", type=int, default=5, help="Maximum number of recommendations")
def recommend_pipelines(use_case: str, framework: Optional[str], complexity: Optional[str], format: str, limit: int):
    """Get pipeline recommendations for a specific use case."""
    try:
        from ..pipeline_catalog.utils import create_catalog_manager
        
        manager = create_catalog_manager()
        
        # Build criteria
        criteria = {}
        if framework:
            criteria["framework"] = framework
        if complexity:
            criteria["complexity"] = complexity
        
        recommendations = manager.get_recommendations(use_case, **criteria)
        
        # Apply limit
        if limit:
            recommendations = recommendations[:limit]
        
        if format == "json":
            result = {
                "use_case": use_case,
                "criteria": criteria,
                "recommendations": recommendations,
                "total": len(recommendations)
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n💡 Recommendations for: {use_case}")
            click.echo("=" * (len(use_case) + 22))
            
            if not recommendations:
                click.echo("No recommendations found.")
                return
            
            for i, rec in enumerate(recommendations, 1):
                if isinstance(rec, dict):
                    pipeline_id = rec.get("pipeline_id", f"recommendation_{i}")
                    score = rec.get("score", 0.0)
                    reason = rec.get("reason", "No reason provided")
                    click.echo(f"{i:3d}. {pipeline_id} (score: {score:.2f})")
                    click.echo(f"     {reason}")
                else:
                    click.echo(f"{i:3d}. {rec}")
            
            if criteria:
                criteria_str = []
                if framework:
                    criteria_str.append(f"framework: {framework}")
                if complexity:
                    criteria_str.append(f"complexity: {complexity}")
                
                click.echo(f"\nCriteria: {', '.join(criteria_str)}")
    
    except Exception as e:
        click.echo(f"❌ Failed to get recommendations: {e}")
        logger.error(f"Failed to get recommendations: {e}")


@pipeline_cli.command("validate")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def validate_registry(format: str):
    """Validate pipeline registry integrity."""
    try:
        from ..pipeline_catalog.utils import create_catalog_manager
        
        manager = create_catalog_manager()
        validation_result = manager.validate_registry()
        
        if format == "json":
            click.echo(json.dumps(validation_result, indent=2))
        else:
            click.echo(f"\n🔍 Registry Validation Results:")
            click.echo("=" * 30)
            
            if validation_result.get("is_valid", False):
                click.echo("✅ Registry is valid")
                click.echo(f"Total issues: {validation_result.get('total_issues', 0)}")
            else:
                click.echo("❌ Registry validation failed")
                click.echo(f"Total issues: {validation_result.get('total_issues', 0)}")
                
                issues_by_severity = validation_result.get("issues_by_severity", {})
                for severity, count in issues_by_severity.items():
                    if count > 0:
                        click.echo(f"  {severity}: {count}")
                
                # Show some issues if available
                all_issues = validation_result.get("all_issues", [])
                if all_issues:
                    click.echo(f"\nSample issues:")
                    for issue in all_issues[:5]:  # Show first 5 issues
                        if isinstance(issue, dict):
                            message = issue.get("message", str(issue))
                            severity = issue.get("severity", "unknown")
                            click.echo(f"  [{severity}] {message}")
                        else:
                            click.echo(f"  {issue}")
    
    except Exception as e:
        click.echo(f"❌ Failed to validate registry: {e}")
        logger.error(f"Failed to validate registry: {e}")


@pipeline_cli.command("stats")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def show_stats(format: str):
    """Show pipeline catalog statistics."""
    try:
        from ..pipeline_catalog import get_catalog_info
        from ..pipeline_catalog.utils import create_catalog_manager
        
        # Get basic catalog info
        catalog_info = get_catalog_info()
        
        # Get detailed stats if available
        try:
            manager = create_catalog_manager()
            detailed_stats = manager.get_registry_stats()
            catalog_info.update(detailed_stats)
        except Exception:
            pass  # Use basic info only
        
        if format == "json":
            click.echo(json.dumps(catalog_info, indent=2))
        else:
            click.echo(f"\n📊 Pipeline Catalog Statistics:")
            click.echo("=" * 30)
            
            # Basic stats
            total_pipelines = catalog_info.get("total_pipelines", 0)
            standard_pipelines = catalog_info.get("standard_pipelines", 0)
            mods_pipelines = catalog_info.get("mods_pipelines", 0)
            shared_dags = catalog_info.get("shared_dags", 0)
            
            click.echo(f"Total Pipelines: {total_pipelines}")
            click.echo(f"  Standard: {standard_pipelines}")
            click.echo(f"  MODS: {mods_pipelines}")
            click.echo(f"Shared DAGs: {shared_dags}")
            
            # Frameworks
            frameworks = catalog_info.get("frameworks", [])
            if frameworks:
                click.echo(f"\nFrameworks ({len(frameworks)}):")
                for framework in frameworks:
                    click.echo(f"  - {framework}")
            
            # Complexity levels
            complexity_levels = catalog_info.get("complexity_levels", [])
            if complexity_levels:
                click.echo(f"\nComplexity Levels ({len(complexity_levels)}):")
                for level in complexity_levels:
                    click.echo(f"  - {level}")
            
            # Last updated
            last_updated = catalog_info.get("last_updated", "unknown")
            click.echo(f"\nLast Updated: {last_updated}")
            
            # Error info if present
            if "error" in catalog_info:
                click.echo(f"\n⚠️  {catalog_info['error']}")
    
    except Exception as e:
        click.echo(f"❌ Failed to get statistics: {e}")
        logger.error(f"Failed to get statistics: {e}")


def main():
    """Main entry point for pipeline CLI."""
    return pipeline_cli()


if __name__ == "__main__":
    main()
