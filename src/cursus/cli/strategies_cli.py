"""
CLI for inspecting the builder **strategy library** (FZ 31e1d3b1).

Under the Strategy + Facade design a step builder is no longer a class you open and read — it is a
*selection* of strategies + knobs bound at build time by ``resolve_handler``. This command group
makes that selection space discoverable, the same way ``cursus catalog`` makes the *step* registry
discoverable:

    cursus strategies axes
        # the routing axes (sagemaker_step_type, step_assembly) + strategy counts

    cursus strategies list [--axis sagemaker_step_type]
        # every registered strategy; filter by axis. Columns: axis | name | verb | #knobs

    cursus strategies show Training [--axis sagemaker_step_type]
        # full detail for one strategy: verb, handler, every knob, preset knobs

    cursus strategies for Training [--step-assembly code]
        # THE authoring shortcut: given a sagemaker_step_type (+ assembly for Processing), print the
        # strategy the facade would bind and the knobs in play — "what do I get if I declare this?"

    cursus strategies knobs --axis step_assembly --name code
        # just the declarative knobs a strategy accepts

Every subcommand reads ``cursus.registry.strategy_registry`` (the single source the runtime router
reads too), so the tool can never drift from what the builder actually does. Read-only.
"""

import json
import logging
from typing import Optional

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group(name="strategies")
def strategies_cli():
    """Inspect the builder strategy library (axes, strategies, knobs)."""
    pass


@strategies_cli.command(name="axes")
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def list_axes(format: str):
    """List the routing axes and how many strategies each carries."""
    from ..registry import strategy_registry as sr

    rows = sr.list_strategies()
    counts = {}
    for r in rows:
        counts.setdefault(r.axis, 0)
        counts[r.axis] += 1

    if format.lower() == "json":
        click.echo(
            json.dumps(
                [{"axis": a, "strategy_count": counts[a]} for a in sr.axes()],
                indent=2,
            )
        )
        return 0

    click.echo("Routing axes:\n")
    for a in sr.axes():
        click.echo(f"  {a}  ({counts[a]} strateg{'y' if counts[a] == 1 else 'ies'})")
    return 0


@strategies_cli.command(name="list")
@click.option("--axis", default=None, help="Filter to one routing axis.")
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def list_strategies_cmd(axis: Optional[str], format: str):
    """List registered strategies (optionally filtered by --axis)."""
    from ..registry import strategy_registry as sr

    rows = sorted(sr.list_strategies(axis=axis), key=lambda i: (i.axis, i.name))

    if format.lower() == "json":
        click.echo(json.dumps([sr.strategy_to_dict(i) for i in rows], indent=2))
        return 0

    if not rows:
        click.echo(f"No strategies found{f' on axis {axis!r}' if axis else ''}.")
        return 0

    click.echo(f"{len(rows)} strateg{'y' if len(rows) == 1 else 'ies'}:\n")
    click.echo(f"  {'AXIS':<22} {'NAME':<34} {'VERB':<14} KNOBS")
    for i in rows:
        flag = "" if i.routable else "  (no builder)"
        click.echo(f"  {i.axis:<22} {i.name:<34} {i.verb:<14} {len(i.knobs)}{flag}")
    return 0


@strategies_cli.command(name="show")
@click.argument("name")
@click.option(
    "--axis", default=None, help="Disambiguate when a name exists on >1 axis."
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def show_strategy(name: str, axis: Optional[str], format: str):
    """Show full detail for one strategy NAME (verb, handler, knobs, presets)."""
    from ..registry import strategy_registry as sr

    matches = sr.find_strategies(name, axis=axis)
    if not matches:
        scope = f" on axis {axis!r}" if axis else ""
        click.echo(f"❌ No strategy named {name!r}{scope}.", err=True)
        raise SystemExit(1)
    if len(matches) > 1:
        click.echo(
            f"❌ {name!r} is ambiguous across axes {[m.axis for m in matches]}; "
            f"pass --axis to disambiguate.",
            err=True,
        )
        raise SystemExit(1)
    info = matches[0]

    if format.lower() == "json":
        click.echo(json.dumps(sr.strategy_to_dict(info), indent=2))
        return 0

    d = sr.strategy_to_dict(info)
    click.echo(f"Strategy: {d['name']}")
    click.echo(f"  axis:        {d['axis']}")
    click.echo(f"  verb:        {d['verb'] or '(none)'}")
    click.echo(f"  handler:     {d['handler'] or '(no builder)'}")
    click.echo(f"  routable:    {d['routable']}")
    click.echo(f"  implemented: {d['implemented']}")
    if d["preset_knobs"]:
        click.echo("  preset knobs:")
        for k, v in d["preset_knobs"].items():
            click.echo(f"      {k} = {v}")
    if d["knobs"]:
        click.echo(f"  knobs ({len(d['knobs'])}):")
        for kn in d["knobs"]:
            req = " [required]" if kn["required"] else ""
            default = "" if kn["default"] is None else f" = {kn['default']}"
            click.echo(f"      {kn['name']}: {kn['type']}{default}{req}")
            if kn["doc"]:
                click.echo(f"          {kn['doc']}")
    else:
        click.echo("  knobs: (none)")
    return 0


@strategies_cli.command(name="for")
@click.argument("sagemaker_step_type")
@click.option(
    "--step-assembly",
    default=None,
    help="Processing sub-discriminator (code | step_args | delegation); default code.",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def for_step_type(sagemaker_step_type: str, step_assembly: Optional[str], format: str):
    """Show the strategy the facade binds for a SAGEMAKER_STEP_TYPE (the authoring shortcut)."""
    from ..registry import strategy_registry as sr

    axis, name = sr.axis_name_for_step_type(sagemaker_step_type, step_assembly)
    try:
        info = sr.resolve_strategy(axis, name)
    except sr.NoBuilderError as e:
        click.echo(
            f"❌ {sagemaker_step_type!r}"
            f"{f' / {step_assembly!r}' if step_assembly else ''} binds no builder: {e}",
            err=True,
        )
        raise SystemExit(1)

    if format.lower() == "json":
        click.echo(
            json.dumps(
                {
                    "sagemaker_step_type": sagemaker_step_type,
                    "step_assembly": step_assembly,
                    "resolved_axis": axis,
                    "resolved_name": name,
                    "strategy": sr.strategy_to_dict(info),
                },
                indent=2,
            )
        )
        return 0

    assembly_note = ""
    if axis == "step_assembly":
        assembly_note = f" (step_assembly={step_assembly or 'code'})"
    click.echo(f"{sagemaker_step_type}{assembly_note} binds:\n")
    d = sr.strategy_to_dict(info)
    click.echo(f"  handler: {d['handler']}  (verb={d['verb']})")
    click.echo(f"  routed via: {axis} = {name}")
    if d["preset_knobs"]:
        click.echo("  preset knobs:")
        for k, v in d["preset_knobs"].items():
            click.echo(f"      {k} = {v}")
    if d["knobs"]:
        click.echo(f"  available knobs ({len(d['knobs'])}):")
        for kn in d["knobs"]:
            click.echo(f"      {kn['name']}: {kn['type']}")
    return 0


@strategies_cli.command(name="knobs")
@click.option("--axis", required=True, help="Routing axis of the strategy.")
@click.option("--name", required=True, help="Strategy name on that axis.")
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def list_knobs(axis: str, name: str, format: str):
    """List the declarative knobs a strategy (--axis/--name) accepts."""
    from ..registry import strategy_registry as sr

    try:
        knobs = sr.knobs_for(axis, name)
    except sr.NoBuilderError as e:
        click.echo(f"❌ {e}", err=True)
        raise SystemExit(1)

    if format.lower() == "json":
        click.echo(json.dumps([sr.knob_to_dict(k) for k in knobs], indent=2))
        return 0

    if not knobs:
        click.echo(f"{axis}={name} declares no knobs.")
        return 0
    click.echo(f"{axis}={name} knobs ({len(knobs)}):\n")
    for kn in (sr.knob_to_dict(k) for k in knobs):
        req = " [required]" if kn["required"] else ""
        default = "" if kn["default"] is None else f" = {kn['default']}"
        click.echo(f"  {kn['name']}: {kn['type']}{default}{req}")
        if kn["doc"]:
            click.echo(f"      {kn['doc']}")
    return 0


def main():
    """Main entry point for the strategies CLI."""
    return strategies_cli()


if __name__ == "__main__":
    main()
