"""
CLI for inspecting a step's connection / I-O view (FZ 31e1d3d follow-up).

Under the Strategy + Facade design a step's builder is a near-empty shell — the container
source/destination paths, the runtime property-path references, and the nested training-channel
fan-out are no longer visible in a readable builder class; they live in the ``.step.yaml`` + the
bound handler. ``cursus steps io <name>`` renders that hidden wiring view:

    cursus steps io XGBoostTraining
        # per dependency: logical_name | container path | required | type, plus the SageMaker
        #   training channels it fans into (train/val/test); per output: container path +
        #   property_path runtime reference.

    cursus steps io RiskTableMapping --job-type validation
        # resolve the job_type variant (different required-flags / compatible_sources).

    cursus steps io XGBoostTraining --format json

Reads ``cursus.steps.interfaces.io_view.describe_step_io`` — the same interface + handler the
builder uses, so the view can never drift from what the step actually wires. Read-only.
"""

import json
import logging
from typing import Optional

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group(name="steps")
def steps_cli():
    """Inspect step interfaces (I/O paths, property references, channels)."""
    pass


@steps_cli.command(name="io")
@click.argument("step_name")
@click.option(
    "--job-type",
    default=None,
    help="Resolve a job_type variant (training | validation | testing | calibration | ...).",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def step_io(step_name: str, job_type: Optional[str], format: str):
    """Show the I/O connection view for STEP_NAME: input/output container paths, the runtime
    property-path references, and the nested training channels."""
    from ..steps.interfaces.io_view import describe_step_io

    try:
        view = describe_step_io(step_name, job_type=job_type)
    except FileNotFoundError:
        click.echo(f"❌ No interface found for step '{step_name}'.", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"❌ Failed to load I/O view for '{step_name}': {e}", err=True)
        logger.error("steps io error", exc_info=True)
        raise SystemExit(1)

    if format.lower() == "json":
        click.echo(json.dumps(view, indent=2, default=str))
        return 0

    jt = f"  (job_type={view['job_type']})" if view.get("job_type") else ""
    click.echo(
        f"{view['step_name']}{jt}  [step_type={view['step_type']}, "
        f"sagemaker_step_type={view['sagemaker_step_type']}]\n"
    )

    click.echo("INPUTS (consumer)")
    if not view["inputs"]:
        click.echo("  (none)")
    for i in view["inputs"]:
        req = "required" if i["required"] else "optional"
        path = (
            i["container_path"]
            if i["container_path"] is not None
            else "(no container path)"
        )
        click.echo(f"  {i['logical_name']:<26} {str(path):<42} {req}  [{i['type']}]")
        if i.get("channels"):
            click.echo(f"      └─ channels: {' '.join(i['channels'])}")
        if i["compatible_sources"]:
            srcs = ", ".join(i["compatible_sources"][:4])
            more = " …" if len(i["compatible_sources"]) > 4 else ""
            click.echo(f"         from: {srcs}{more}")

    click.echo("\nOUTPUTS (producer)")
    if not view["outputs"]:
        click.echo("  (none)")
    for o in view["outputs"]:
        path = (
            o["container_path"]
            if o["container_path"] is not None
            else "(no container path)"
        )
        click.echo(f"  {o['logical_name']:<26} {str(path):<42} [{o['type']}]")
        if o["property_path"]:
            click.echo(f"      ref: {o['property_path']}")
        if o["aliases"]:
            click.echo(f"      aliases: {', '.join(o['aliases'][:5])}")
    return 0


@steps_cli.command(name="patterns")
@click.argument("step_name")
@click.option(
    "--job-type",
    default=None,
    help="Resolve a job_type variant (training | validation | testing | calibration | ...).",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def step_patterns(step_name: str, job_type: Optional[str], format: str):
    """Show the construction PATTERNS (the 'plugins') the TemplateStepBuilder uses for STEP_NAME:
    the bound create_step handler + the env / job-arg / input / output patterns, all derived from
    the step's .step.yaml + registry binding. A ⚠ marks an axis the builder still hand-overrides."""
    from ..steps.interfaces.io_view import describe_step_patterns

    try:
        view = describe_step_patterns(step_name, job_type=job_type)
    except FileNotFoundError:
        click.echo(f"❌ No interface found for step '{step_name}'.", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"❌ Failed to load patterns view for '{step_name}': {e}", err=True)
        logger.error("steps patterns error", exc_info=True)
        raise SystemExit(1)

    if format.lower() == "json":
        click.echo(json.dumps(view, indent=2, default=str))
        return 0

    jt = f"  (job_type={view['job_type']})" if view.get("job_type") else ""
    click.echo(
        f"{view['step_name']}{jt}  [step_type={view['step_type']}, "
        f"sagemaker_step_type={view['sagemaker_step_type']}]\n"
    )
    p = view["patterns"]

    def _mark(axis):
        return "  ⚠ custom override" if p[axis].get("custom_override") else ""

    click.echo(
        f"create_step   {p['create_step']['handler']}"
        f"  (assembly={p['create_step']['step_assembly']}){_mark('create_step')}"
    )

    e = p["env_vars"]
    click.echo(f"env_vars      {e['source']}{_mark('env_vars')}")
    if e["declared_required"]:
        click.echo(f"                required: {', '.join(e['declared_required'])}")
    if e["declared_optional"]:
        click.echo(f"                optional: {', '.join(e['declared_optional'])}")

    j = p["job_arguments"]
    click.echo(f"job_arguments {j['source']}{_mark('job_arguments')}")
    for d in j["declared"]:
        src = f" ← config.{d['source']}" if d["source"] else ""
        click.echo(f"                {d['flag']}{src}")

    click.echo(f"inputs        {p['inputs']['pattern']}{_mark('inputs')}")
    click.echo(f"outputs       {p['outputs']['pattern']}{_mark('outputs')}")

    c = p.get("compute")
    if c:
        cls = f" → {c['sdk_class']}" if c.get("sdk_class") else ""
        kind = c.get("kind") or "per-step factory"
        click.echo(f"compute       {kind}{cls}{_mark('compute')}")
        if c.get("framework_version_field"):
            click.echo(
                f"                framework_version ← config.{c['framework_version_field']}"
            )
        if c.get("lock_training_region"):
            click.echo(
                "                training image region LOCKED (SAIS restriction)"
            )

    # --- dependency axis: the mods/SAIS-vs-native 3rd-party footprint (build-time vs runtime) ---
    dep = view.get("dependencies") or {}
    if dep.get("native"):
        click.echo("requires      (none — native sagemaker only)")
    else:
        click.echo("requires      (3rd-party)")
        bt = dep.get("build_time") or {}
        if bt:
            for axis, pkg in bt.items():
                tier = (
                    "HARD module-level"
                    if pkg == "secure_ai_sandbox_workflow_python_sdk"
                    else "lazy, no fallback on path"
                )
                click.echo(f"                build-time: {axis:<11} → {pkg} ({tier})")
        else:
            click.echo("                build-time: (none)")
        rt = dep.get("runtime") or []
        click.echo(f"                runtime:    {', '.join(rt) if rt else '(none)'}")

    if view.get("note"):
        click.echo(f"\nnote: {view['note']}")
    return 0


def main():
    """Main entry point for the steps CLI."""
    return steps_cli()


if __name__ == "__main__":
    main()
