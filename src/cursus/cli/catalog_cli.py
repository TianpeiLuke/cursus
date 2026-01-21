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
@click.option(
    "--format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.option("--limit", type=int, help="Maximum number of results to show")
def list_steps(
    workspace: Optional[str],
    job_type: Optional[str],
    framework: Optional[str],
    format: str,
    limit: Optional[int],
):
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
                if (
                    detected_framework
                    and detected_framework.lower() == framework.lower()
                ):
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
                    "framework": framework,
                },
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
                workspace_info = (
                    f" [{step_info.workspace_id}]"
                    if step_info and step_info.workspace_id != "core"
                    else ""
                )
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
@click.option(
    "--format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
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
                json_results.append(
                    {
                        "step_name": result.step_name,
                        "workspace_id": result.workspace_id,
                        "match_score": result.match_score,
                        "match_reason": result.match_reason,
                        "components_available": result.components_available,
                    }
                )

            click.echo(
                json.dumps(
                    {"query": query, "results": json_results, "total": len(results)},
                    indent=2,
                )
            )
        else:
            click.echo(f"\n🔍 Search Results for '{query}' ({len(results)} found):")
            click.echo("=" * 60)

            if not results:
                click.echo("No steps found matching the search query.")
                return

            for i, result in enumerate(results, 1):
                workspace_info = (
                    f" [{result.workspace_id}]" if result.workspace_id != "core" else ""
                )
                components_info = (
                    f" ({len(result.components_available)} components)"
                    if result.components_available
                    else ""
                )

                click.echo(
                    f"{i:3d}. {result.step_name}{workspace_info} (score: {result.match_score:.2f}){components_info}"
                )
                click.echo(f"     Reason: {result.match_reason}")

    except Exception as e:
        click.echo(f"❌ Failed to search steps: {e}")
        logger.error(f"Failed to search steps: {e}")


@catalog_cli.command("show")
@click.argument("step_name")
@click.option(
    "--format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option(
    "--show-components", is_flag=True, help="Show detailed component information"
)
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
                "file_components": {},
            }

            # Add file components info
            for comp_type, metadata in step_info.file_components.items():
                if metadata:
                    result["file_components"][comp_type] = {
                        "path": str(metadata.path),
                        "file_type": metadata.file_type,
                        "modified_time": metadata.modified_time.isoformat()
                        if metadata.modified_time
                        else None,
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
                    if key not in ["__module__", "__qualname__"]:
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
@click.option(
    "--type",
    "component_type",
    help="Filter by component type (script, contract, spec, builder, config)",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
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
            result = {"step_name": step_name, "components": {}}

            for comp_type, metadata in components.items():
                if metadata:
                    result["components"][comp_type] = {
                        "path": str(metadata.path),
                        "file_type": metadata.file_type,
                        "modified_time": metadata.modified_time.isoformat()
                        if metadata.modified_time
                        else None,
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
@click.option(
    "--format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
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
                "steps_by_framework": step_frameworks,
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
@click.option(
    "--format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list_workspaces(format: str):
    """List available workspaces and their step counts."""
    try:
        from ..step_catalog import StepCatalog

        catalog = StepCatalog()
        cross_workspace = catalog.discover_cross_workspace_components()

        if format == "json":
            result = {"workspaces": {}}

            for workspace_id, components in cross_workspace.items():
                steps = catalog.list_available_steps(workspace_id=workspace_id)
                result["workspaces"][workspace_id] = {
                    "step_count": len(steps),
                    "component_count": len(components),
                    "steps": steps,
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
@click.option(
    "--format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
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
            click.echo(
                f"Average Response Time: {metrics['avg_response_time_ms']:.2f}ms"
            )
            click.echo(f"Index Build Time: {metrics['index_build_time_s']:.3f}s")
            click.echo(f"Total Steps Indexed: {metrics['total_steps_indexed']}")
            click.echo(f"Total Workspaces: {metrics['total_workspaces']}")

            if metrics["last_index_build"]:
                click.echo(f"Last Index Build: {metrics['last_index_build']}")

    except Exception as e:
        click.echo(f"❌ Failed to show metrics: {e}")
        logger.error(f"Failed to show metrics: {e}")


@catalog_cli.command("discover")
@click.option(
    "--workspace-dir",
    type=click.Path(exists=True),
    help="Workspace directory to discover",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
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
                "total": len(steps),
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
                    components = (
                        list(step_info.file_components.keys()) if step_info else []
                    )
                    components_info = (
                        f" ({', '.join(components)})" if components else ""
                    )
                    click.echo(f"{i:3d}. {step_name}{components_info}")
            else:
                click.echo("\nNo steps found in the specified workspace directory.")

    except Exception as e:
        click.echo(f"❌ Failed to discover workspace: {e}")
        logger.error(f"Failed to discover workspace: {e}")


@catalog_cli.command("list-configs")
@click.option("--project-id", help="Filter by project/workspace")
@click.option("--framework", help="Filter by framework")
@click.option("--format", type=click.Choice(["table", "json"]), default="table")
@click.option("--show-fields", is_flag=True, help="Show field count for each config")
def list_configs(
    project_id: Optional[str],
    framework: Optional[str],
    format: str,
    show_fields: bool,
):
    """List all configuration classes."""
    try:
        from ..step_catalog import StepCatalog

        catalog = StepCatalog()
        config_classes = catalog.discover_config_classes(project_id)

        if format == "json":
            result = {
                "config_classes": list(config_classes.keys()),
                "total": len(config_classes),
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n📋 Configuration Classes ({len(config_classes)} found):")
            click.echo("=" * 50)

            for i, (name, cls) in enumerate(config_classes.items(), 1):
                field_info = ""
                if show_fields and hasattr(cls, "model_fields"):
                    field_count = len(cls.model_fields)
                    field_info = f" ({field_count} fields)"
                click.echo(f"{i:3d}. {name}{field_info}")

    except Exception as e:
        click.echo(f"❌ Failed to list configs: {e}")
        logger.error(f"Failed to list configs: {e}")


@catalog_cli.command("list-builders")
@click.option("--step-type", help="Filter by SageMaker step type")
@click.option("--framework", help="Filter by framework")
@click.option("--format", type=click.Choice(["table", "json"]), default="table")
@click.option("--show-path", is_flag=True, help="Show file path")
def list_builders(
    step_type: Optional[str],
    framework: Optional[str],
    format: str,
    show_path: bool,
):
    """List all builder classes."""
    try:
        from ..step_catalog import StepCatalog

        catalog = StepCatalog()

        if step_type:
            builders = catalog.get_builders_by_step_type(step_type)
        else:
            builders = catalog.get_all_builders()

        if framework:
            filtered = {}
            for name, builder in builders.items():
                detected_fw = catalog.detect_framework(name)
                if detected_fw and detected_fw.lower() == framework.lower():
                    filtered[name] = builder
            builders = filtered

        if format == "json":
            result = {"builders": list(builders.keys()), "total": len(builders)}
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🔧 Builder Classes ({len(builders)} found):")
            click.echo("=" * 50)

            for i, (name, builder) in enumerate(builders.items(), 1):
                step_info = catalog.get_step_info(name)
                type_info = ""
                if step_info and "sagemaker_step_type" in step_info.registry_data:
                    type_info = (
                        f" | Type: {step_info.registry_data['sagemaker_step_type']}"
                    )

                fw_info = ""
                detected_fw = catalog.detect_framework(name)
                if detected_fw:
                    fw_info = f" | Framework: {detected_fw}"

                click.echo(f"{i:3d}. {builder.__name__}{type_info}{fw_info}")

                if show_path:
                    path = catalog.get_builder_class_path(name)
                    if path:
                        click.echo(f"     Path: {path}")

    except Exception as e:
        click.echo(f"❌ Failed to list builders: {e}")
        logger.error(f"Failed to list builders: {e}")


@catalog_cli.command("list-contracts")
@click.option(
    "--with-scripts-only", is_flag=True, help="Only show contracts with scripts"
)
@click.option("--format", type=click.Choice(["table", "json"]), default="table")
@click.option("--show-entry-points", is_flag=True, help="Show script entry points")
def list_contracts(
    with_scripts_only: bool,
    format: str,
    show_entry_points: bool,
):
    """List all contract classes."""
    try:
        from ..step_catalog import StepCatalog

        catalog = StepCatalog()

        if with_scripts_only:
            contract_names = catalog.discover_contracts_with_scripts()
        else:
            # Get all steps and filter to those with contracts
            all_steps = catalog.list_available_steps()
            contract_names = []
            for step_name in all_steps:
                step_info = catalog.get_step_info(step_name)
                if step_info and step_info.file_components.get("contract"):
                    contract_names.append(step_name)

        entry_points = catalog.get_contract_entry_points() if show_entry_points else {}

        if format == "json":
            result = {
                "contracts": contract_names,
                "total": len(contract_names),
                "entry_points": entry_points if show_entry_points else {},
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n📜 Contract Classes ({len(contract_names)} found):")
            click.echo("=" * 50)

            for i, name in enumerate(contract_names, 1):
                entry_point = entry_points.get(name, "")
                ep_info = f"\n     Entry Point: {entry_point}" if entry_point else ""
                click.echo(f"{i:3d}. {name}Contract{ep_info}")

    except Exception as e:
        click.echo(f"❌ Failed to list contracts: {e}")
        logger.error(f"Failed to list contracts: {e}")


@catalog_cli.command("list-specs")
@click.option("--job-type", help="Filter by job type")
@click.option("--framework", help="Filter by framework")
@click.option("--format", type=click.Choice(["table", "json"]), default="table")
@click.option("--show-dependencies", is_flag=True, help="Show dependency count")
def list_specs(
    job_type: Optional[str],
    framework: Optional[str],
    format: str,
    show_dependencies: bool,
):
    """List all specification classes."""
    try:
        from ..step_catalog import StepCatalog

        catalog = StepCatalog()
        spec_names = catalog.list_steps_with_specs(job_type=job_type)

        if framework:
            filtered = []
            for name in spec_names:
                detected_fw = catalog.detect_framework(name)
                if detected_fw and detected_fw.lower() == framework.lower():
                    filtered.append(name)
            spec_names = filtered

        if format == "json":
            result = {"specifications": spec_names, "total": len(spec_names)}
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n📐 Specification Classes ({len(spec_names)} found):")
            click.echo("=" * 50)

            for i, name in enumerate(spec_names, 1):
                dep_info = ""
                if show_dependencies:
                    spec_instance = catalog.load_spec_class(name)
                    if spec_instance:
                        spec_dict = catalog.serialize_spec(spec_instance)
                        deps = spec_dict.get("dependencies", {})
                        dep_info = f" ({len(deps)} dependencies)"

                click.echo(f"{i:3d}. {name}Spec{dep_info}")

    except Exception as e:
        click.echo(f"❌ Failed to list specs: {e}")
        logger.error(f"Failed to list specs: {e}")


@catalog_cli.command("list-scripts")
@click.option("--project-id", help="Filter by project/workspace")
@click.option("--format", type=click.Choice(["table", "json"]), default="table")
@click.option("--show-path", is_flag=True, help="Show full file path")
def list_scripts(
    project_id: Optional[str],
    format: str,
    show_path: bool,
):
    """List all script files discovered."""
    try:
        from ..step_catalog import StepCatalog

        catalog = StepCatalog()
        script_names = catalog.list_available_scripts()

        if format == "json":
            result = {"scripts": script_names, "total": len(script_names)}
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n📜 Script Files ({len(script_names)} found):")
            click.echo("=" * 50)

            for i, name in enumerate(script_names, 1):
                path_info = ""
                if show_path:
                    script_info = catalog.get_script_info(name)
                    if script_info and "file_path" in script_info:
                        path_info = f"\n     Path: {script_info['file_path']}"

                click.echo(f"{i:3d}. {name}{path_info}")

    except Exception as e:
        click.echo(f"❌ Failed to list scripts: {e}")
        logger.error(f"Failed to list scripts: {e}")


@catalog_cli.command("search-field")
@click.argument("field_name")
@click.option("--field-type", help="Filter by field type (str, int, bool, dict, list)")
@click.option("--format", type=click.Choice(["table", "json"]), default="table")
@click.option("--show-default", is_flag=True, help="Show default values")
def search_field(
    field_name: str,
    field_type: Optional[str],
    format: str,
    show_default: bool,
):
    """Find steps with configs containing a specific field."""
    try:
        from ..step_catalog import StepCatalog

        catalog = StepCatalog()
        config_classes = catalog.discover_config_classes()

        results = []
        for config_name, config_class in config_classes.items():
            if hasattr(config_class, "model_fields"):
                fields = config_class.model_fields
                if field_name in fields:
                    field_info = fields[field_name]
                    field_type_str = str(field_info.annotation).replace("typing.", "")

                    # Apply type filter if specified
                    if field_type and field_type.lower() not in field_type_str.lower():
                        continue

                    default_value = (
                        field_info.default if hasattr(field_info, "default") else None
                    )

                    results.append(
                        {
                            "config_name": config_name,
                            "field_type": field_type_str,
                            "default": default_value,
                        }
                    )

        if format == "json":
            click.echo(
                json.dumps({"results": results, "total": len(results)}, indent=2)
            )
        else:
            click.echo(f"\n🔍 Steps with field '{field_name}':")
            click.echo("=" * 50)

            if not results:
                click.echo("No matching fields found.")
                return

            for i, result in enumerate(results, 1):
                click.echo(f"{i:3d}. Config: {result['config_name']}")
                click.echo(f"     Field Type: {result['field_type']}")
                if show_default and result["default"] is not None:
                    click.echo(f"     Default: {result['default']}")
                click.echo()

    except Exception as e:
        click.echo(f"❌ Failed to search field: {e}")
        logger.error(f"Failed to search field: {e}")


@catalog_cli.command("list-by-type")
@click.argument("step_type")
@click.option("--framework", help="Filter by framework")
@click.option("--format", type=click.Choice(["table", "json"]), default="table")
def list_by_type(
    step_type: str,
    framework: Optional[str],
    format: str,
):
    """Filter steps by SageMaker step type."""
    try:
        from ..step_catalog import StepCatalog

        catalog = StepCatalog()
        all_steps = catalog.list_available_steps()

        filtered_steps = []
        for step_name in all_steps:
            step_info = catalog.get_step_info(step_name)
            if (
                step_info
                and step_info.registry_data.get("sagemaker_step_type") == step_type
            ):
                if framework:
                    detected_fw = catalog.detect_framework(step_name)
                    if detected_fw and detected_fw.lower() == framework.lower():
                        filtered_steps.append(step_name)
                else:
                    filtered_steps.append(step_name)

        if format == "json":
            result = {
                "steps": filtered_steps,
                "type": step_type,
                "total": len(filtered_steps),
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(
                f"\n📦 Steps with type '{step_type}' ({len(filtered_steps)} found):"
            )
            click.echo("=" * 50)

            if not filtered_steps:
                click.echo("No steps found matching the criteria.")
                return

            for i, step_name in enumerate(filtered_steps, 1):
                fw_info = ""
                detected_fw = catalog.detect_framework(step_name)
                if detected_fw:
                    fw_info = f" (Framework: {detected_fw})"
                click.echo(f"{i:3d}. {step_name}{fw_info}")

    except Exception as e:
        click.echo(f"❌ Failed to list by type: {e}")
        logger.error(f"Failed to list by type: {e}")


@catalog_cli.command("fields")
@click.argument("step_name")
@click.option("--inherited", is_flag=True, help="Show inherited fields")
@click.option("--format", type=click.Choice(["table", "json"]), default="table")
@click.option("--show-types", is_flag=True, help="Show field types")
@click.option("--show-defaults", is_flag=True, help="Show default values")
def show_fields(
    step_name: str,
    inherited: bool,
    format: str,
    show_types: bool,
    show_defaults: bool,
):
    """Show all configuration fields for a step."""
    try:
        from ..step_catalog import StepCatalog

        catalog = StepCatalog()
        config_classes = catalog.discover_config_classes()

        # Find config for this step
        config_name = f"{step_name}Config"
        config_class = config_classes.get(config_name)

        if not config_class:
            click.echo(f"❌ No configuration found for step: {step_name}")
            return

        if format == "json":
            fields_data = {}
            if hasattr(config_class, "model_fields"):
                for fname, finfo in config_class.model_fields.items():
                    fields_data[fname] = {
                        "type": str(finfo.annotation),
                        "required": finfo.is_required()
                        if hasattr(finfo, "is_required")
                        else False,
                        "default": str(finfo.default)
                        if hasattr(finfo, "default")
                        else None,
                    }

            result = {
                "step_name": step_name,
                "config_class": config_name,
                "fields": fields_data,
                "total_fields": len(fields_data),
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🔧 Configuration Fields for {step_name}:")
            click.echo("=" * 50)
            click.echo(f"Config Class: {config_name}")

            if inherited:
                parent_name = catalog.get_immediate_parent_config_class(config_name)
                if parent_name:
                    click.echo(f"Parent Class: {parent_name}")

            click.echo()

            if hasattr(config_class, "model_fields"):
                fields = config_class.model_fields
                click.echo(f"Total Fields: {len(fields)}")
                click.echo()

                for fname, finfo in fields.items():
                    type_str = ""
                    if show_types:
                        type_str = f" ({str(finfo.annotation).replace('typing.', '')})"

                    default_str = ""
                    if (
                        show_defaults
                        and hasattr(finfo, "default")
                        and finfo.default is not None
                    ):
                        default_str = f"\n    Default: {finfo.default}"

                    required_str = (
                        " (required)"
                        if (hasattr(finfo, "is_required") and finfo.is_required())
                        else ""
                    )

                    click.echo(f"  - {fname}{type_str}{required_str}{default_str}")

    except Exception as e:
        click.echo(f"❌ Failed to show fields: {e}")
        logger.error(f"Failed to show fields: {e}")


@catalog_cli.command("component-info")
@click.argument("step_name")
@click.argument(
    "component_type",
    type=click.Choice(["config", "builder", "contract", "spec", "script"]),
)
@click.option("--format", type=click.Choice(["text", "json"]), default="text")
@click.option("--load", is_flag=True, help="Load and inspect the actual class")
def component_info(
    step_name: str,
    component_type: str,
    format: str,
    load: bool,
):
    """Get detailed information about a specific component."""
    try:
        from ..step_catalog import StepCatalog

        catalog = StepCatalog()
        step_info = catalog.get_step_info(step_name)

        if not step_info:
            click.echo(f"❌ Step not found: {step_name}")
            return

        component_metadata = step_info.file_components.get(component_type)

        if not component_metadata:
            click.echo(
                f"❌ Component type '{component_type}' not found for step: {step_name}"
            )
            return

        result = {
            "step_name": step_name,
            "component_type": component_type,
            "file_path": str(component_metadata.path),
            "modified": component_metadata.modified_time.isoformat()
            if component_metadata.modified_time
            else None,
        }

        if load:
            if component_type == "config":
                config_classes = catalog.discover_config_classes()
                config_name = f"{step_name}Config"
                config_class = config_classes.get(config_name)
                if config_class:
                    result["class_name"] = config_name
                    if hasattr(config_class, "model_fields"):
                        result["field_count"] = len(config_class.model_fields)

            elif component_type == "builder":
                builder_class = catalog.load_builder_class(step_name)
                if builder_class:
                    result["class_name"] = builder_class.__name__

            elif component_type == "contract":
                contract = catalog.load_contract_class(step_name)
                if contract:
                    contract_dict = catalog.serialize_contract(contract)
                    result["entry_point"] = contract_dict.get("entry_point")

            elif component_type == "spec":
                spec = catalog.load_spec_class(step_name)
                if spec:
                    spec_dict = catalog.serialize_spec(spec)
                    result["dependencies"] = list(
                        spec_dict.get("dependencies", {}).keys()
                    )

        if format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n📋 Component Info: {step_name} ({component_type})")
            click.echo("=" * 50)
            click.echo(f"Component Type: {component_type.title()}")
            click.echo(f"File Path: {result['file_path']}")
            if result.get("modified"):
                click.echo(f"Modified: {result['modified']}")

            if load:
                click.echo("\nClass Details:")
                if "class_name" in result:
                    click.echo(f"  Class Name: {result['class_name']}")
                if "field_count" in result:
                    click.echo(f"  Field Count: {result['field_count']}")
                if "entry_point" in result:
                    click.echo(f"  Entry Point: {result['entry_point']}")
                if "dependencies" in result:
                    click.echo(f"  Dependencies: {', '.join(result['dependencies'])}")

    except Exception as e:
        click.echo(f"❌ Failed to get component info: {e}")
        logger.error(f"Failed to get component info: {e}")


def main():
    """Main entry point for catalog CLI."""
    return catalog_cli()


if __name__ == "__main__":
    main()
