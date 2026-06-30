"""
``steps.*`` — the agent-facing step connection / I-O view (FZ 31e1d3d follow-up).

Under the Strategy + Facade design the per-step builder is a near-empty shell, so the container
source/destination paths, the runtime ``property_path`` references, and the nested training-channel
fan-out are no longer readable from a builder class — they live in the ``.step.yaml`` + the bound
handler. ``steps.io`` exposes that wiring so an agent can see, for a step:
  - per dependency (consumer): container path, required, type, compatible_sources, and the
    SageMaker training channel(s) the input fans into (e.g. ``input_path -> [train, val, test]``);
  - per output (producer): container path + the runtime ``property_path`` a downstream step
    resolves against.

It is the path/wiring complement to ``catalog.step_spec`` (which gives the ports + property_path
but not the container paths or the channel fan-out). Reads the same interface + handler the builder
uses (``steps.interfaces.io_view.describe_step_io``), so it cannot drift. Read-only.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..envelope import ToolResult
from ..registry import ToolDef


def _io(args: Dict[str, Any]) -> ToolResult:
    step_name = args["step_name"]
    job_type = args.get("job_type")
    from ...steps.interfaces.io_view import describe_step_io

    try:
        view = describe_step_io(step_name, job_type=job_type)
    except FileNotFoundError:
        return ToolResult.failure(
            f"no interface found for step '{step_name}'",
            code="not_found",
            details={"step_name": step_name},
            remedy={
                "suggested_tools": ["catalog.list_steps", "catalog.search"],
                "fix_action": "Confirm the step name (some abstract steps have no interface); "
                "list or search the catalog for valid names.",
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ToolResult.failure(
            f"failed to load I/O view for '{step_name}': {exc}",
            code="internal_error",
            details={"step_name": step_name},
        )

    view["dependency_count"] = len(view["inputs"])
    view["output_count"] = len(view["outputs"])
    return ToolResult.success(view, step_name=step_name)


def _patterns(args: Dict[str, Any]) -> ToolResult:
    step_name = args["step_name"]
    job_type = args.get("job_type")
    from ...steps.interfaces.io_view import describe_step_patterns

    try:
        view = describe_step_patterns(step_name, job_type=job_type)
    except FileNotFoundError:
        return ToolResult.failure(
            f"no interface found for step '{step_name}'",
            code="not_found",
            details={"step_name": step_name},
            remedy={
                "suggested_tools": ["catalog.list_steps", "catalog.search"],
                "fix_action": "Confirm the step name; list or search the catalog for valid names.",
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ToolResult.failure(
            f"failed to load patterns view for '{step_name}': {exc}",
            code="internal_error",
            details={"step_name": step_name},
        )
    return ToolResult.success(view, step_name=step_name)


TOOLS: List[ToolDef] = [
    ToolDef(
        name="steps.patterns",
        description=(
            "Return a step's construction PATTERNS — the 'plugins' the TemplateStepBuilder composes "
            "for each axis: the bound create_step handler (from sagemaker_step_type + step_assembly), "
            "and the env-var / job-argument / input / output / compute patterns (all derived from the "
            "step's .step.yaml contract DATA + registry binding, so the view cannot drift from "
            "behavior). Each axis carries 'custom_override': true where the builder still hand-overrides "
            "that method (a genuine per-step deviation). A top-level 'dependencies' rollup reports the "
            "step's 3rd-party footprint (build_time {axis: pkg} for mods_workflow_core / SAIS-SDK vs "
            "runtime script deps vs native sagemaker-only). Use to see how a step is assembled — and "
            "what it costs to import — without reading a builder class."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical step name, e.g. 'XGBoostTraining', 'TabularPreprocessing'.",
                },
                "job_type": {
                    "type": "string",
                    "description": "Optional job_type variant (training | validation | testing | "
                    "calibration | ...) — resolves the variant's spec.",
                },
            },
            "required": ["step_name"],
            "additionalProperties": False,
        },
        handler=_patterns,
        tags=("planner",),
    ),
    ToolDef(
        name="steps.io",
        description=(
            "Return a step's connection / I-O view: for each dependency the container path, "
            "required flag, type, compatible_sources, and the SageMaker training channel(s) it "
            "fans into (e.g. input_path -> train/val/test); for each output the container path and "
            "the runtime property_path reference a downstream step resolves against. This is the "
            "path/wiring view the Facade hides from a readable builder class — the complement to "
            "catalog.step_spec. Use to wire a step or understand where its data lands."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_name": {
                    "type": "string",
                    "description": "Canonical step name, e.g. 'XGBoostTraining', 'BatchTransform'.",
                },
                "job_type": {
                    "type": "string",
                    "description": "Optional job_type variant (training | validation | testing | "
                    "calibration | ...) — resolves the variant's spec.",
                },
            },
            "required": ["step_name"],
            "additionalProperties": False,
        },
        handler=_io,
        tags=("planner",),
    ),
]
