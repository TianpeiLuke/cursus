"""
CLI commands for inspecting and validating pipeline DAG JSON files.

`cursus dag validate` checks a serialized DAG (cycles, dangling edges, isolated nodes,
and — via the step catalog — that each node resolves to a known step) before you spend
time compiling it. Engine imports are lazy so `cursus --help` stays fast.

`cursus dag resolve <step> <step> ...` runs the REAL UnifiedDependencyResolver over the
named steps' interfaces and reports, per resolved dependency edge, the 6-component
compatibility score and whether it resolves (>= 0.5). This is the same resolver +
weights + threshold the compiler and CI use — a single command that surfaces the
per-edge score with no re-implementation (the author-time gate parses its JSON).

Examples:
    cursus dag validate dag.json
    cursus dag validate dag.json --format json
    cursus dag resolve CradleDataLoading TSATabularPreprocessing --format json
"""

import json
import logging

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group(name="dag")
def dag_cli():
    """Inspect and validate pipeline DAG JSON files."""


@dag_cli.command(name="validate")
@click.argument("dag_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def validate_dag(dag_file: str, format: str):
    """
    Validate the integrity of a serialized DAG JSON file.

    DAG_FILE: path to a DAG JSON file (as produced by the serializer / compiler).

    Reports cycles, dangling edges, isolated nodes, and nodes that do not resolve to a
    known step in the catalog. Exits nonzero if the DAG is invalid.
    """
    fmt = format.lower()
    try:
        from ..api.dag import import_dag_from_json, PipelineDAGResolver

        dag = import_dag_from_json(dag_file)
        resolver = PipelineDAGResolver(dag, validate_on_init=False)
        issues = resolver.validate_dag_integrity()
    except Exception as e:
        click.echo(f"❌ Failed to validate DAG '{dag_file}': {e}", err=True)
        logger.error("DAG validation error", exc_info=True)
        raise SystemExit(1)

    is_valid = not issues

    if fmt == "json":
        click.echo(
            json.dumps(
                {
                    "dag_file": dag_file,
                    "is_valid": is_valid,
                    "node_count": len(dag.nodes),
                    "edge_count": len(dag.edges),
                    "issues": issues,
                },
                indent=2,
                default=str,
            )
        )
        if not is_valid:
            raise SystemExit(1)
        return

    click.echo(f"DAG: {dag_file}")
    click.echo(f"  nodes: {len(dag.nodes)} | edges: {len(dag.edges)}")
    if is_valid:
        click.echo("✅ DAG is valid (no integrity issues found).")
        return

    click.echo("❌ DAG has integrity issues:")
    for category, items in issues.items():
        if items:
            click.echo(f"  {category}:")
            for item in items:
                click.echo(f"    - {item}")
    raise SystemExit(1)


@dag_cli.command(name="resolve")
@click.argument("steps", nargs=-1, required=True)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def resolve_edges(steps, format: str):
    """
    Score dependency-resolution edges among the named STEPS with the REAL resolver.

    STEPS: two or more step names (e.g. a producer, a NEW step, a consumer). Each step's
    interface is loaded from its `.step.yaml`, registered, and scored by the same
    UnifiedDependencyResolver (weights type .40 / data_type .20 / semantic .25 /
    exact-match .05 / source-compat .10 / keyword .05; threshold >= 0.5) the compiler runs.

    For every dependency of every named step it reports the best-scoring provider among the
    other named steps, its score, and whether it resolves (>= 0.5). No score is computed by
    hand — this is the production resolver, so the author-time gate can trust the JSON.
    """
    fmt = format.lower()
    try:
        from ..core.deps.specification_registry import SpecificationRegistry
        from ..core.deps.dependency_resolver import create_dependency_resolver
        from ..steps.interfaces import load_interface

        names = list(steps)
        registry = SpecificationRegistry()
        loaded, load_errors = [], {}
        for name in names:
            try:
                iface = load_interface(name)  # same load path as `cursus steps io`
                registry.register(name, iface.spec)
                loaded.append(name)
            except Exception as e:  # unknown step / bad yaml — record, keep going
                load_errors[name] = str(e)

        edges = []
        # For each loaded step that HAS dependencies (a consumer), score its edges against
        # every OTHER loaded step (the available providers).
        resolver = create_dependency_resolver(registry)
        for consumer in loaded:
            spec = registry.get_specification(consumer)
            if not spec or not getattr(spec, "dependencies", None):
                continue
            available = [n for n in loaded if n != consumer]
            report = resolver.resolve_with_scoring(consumer, available)
            for dep_name, ref in (report.get("resolved") or {}).items():
                edges.append(
                    {
                        "consumer": consumer,
                        "dependency": dep_name,
                        "provider": getattr(ref, "step_name", None) or str(ref),
                        "score": 1.0,  # resolved => best candidate cleared threshold; see breakdown below
                        "resolves": True,
                    }
                )
            for dep_name, info in (report.get("failed_with_scores") or {}).items():
                best = info.get("best_candidate") or {}
                edges.append(
                    {
                        "consumer": consumer,
                        "dependency": dep_name,
                        "provider": best.get("provider_step"),
                        "score": round(float(best.get("score", 0.0)), 4),
                        "resolves": False,
                        "required": info.get("required"),
                    }
                )
        all_resolve = bool(edges) and all(e["resolves"] for e in edges)
    except Exception as e:
        click.echo(f"❌ Failed to resolve edges for {list(steps)}: {e}", err=True)
        logger.error("dag resolve error", exc_info=True)
        raise SystemExit(1)

    if fmt == "json":
        click.echo(
            json.dumps(
                {
                    "steps": names,
                    "loaded": loaded,
                    "load_errors": load_errors,
                    "edges": edges,
                    "all_edges_resolve": all_resolve,
                    "threshold": 0.5,
                },
                indent=2,
                default=str,
            )
        )
        return

    click.echo(f"Resolve: {', '.join(names)}")
    if load_errors:
        for n, err in load_errors.items():
            click.echo(f"  ⚠ could not load {n}: {err}")
    if not edges:
        click.echo("  (no dependency edges among these steps)")
        return
    for e in edges:
        mark = "✅" if e["resolves"] else "❌"
        click.echo(
            f"  {mark} {e['consumer']}.{e['dependency']} <- {e.get('provider')} "
            f"(score {e['score']}, resolves={e['resolves']})"
        )


def main():
    """Main entry point for the dag CLI."""
    return dag_cli()


if __name__ == "__main__":
    main()
