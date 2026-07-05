"""
cursus.mcp tools for the ``project.*`` namespace.

Scaffold a NEW Cursus pipeline project — the phase-0 package skeleton every project shares,
regardless of framework, DAG, or features. ``project.init`` writes the fixed, knowable-at-t=0
files (a region-agnostic ``run_pipeline.py``, the ``@MODSTemplate`` deployment class that loads
``pipeline_config/dag.json``, a shared ``generate_config.py`` skeleton with ``project_root_folder``
filled and a TODO per-node value-init block, an empty ``dag.json`` stub, the folder tree with
per-folder READMEs) and a root ``README.md`` **action-item ledger** that hands every
context-dependent piece — authoring the DAG, copying scripts/handlers, filling the config values —
to its owning downstream workflow (``/cursus-author-step``, ``/cursus-configure-pipeline``).

This is a **deterministic** tool: it emits versioned package templates with the project name and
framework substituted in, so it works fully offline over the stateless JSON tool boundary (no
sub-agents, no engine import). The source-grounded generation variant (adapting the templates to a
live reference project as it drifts) lives in the ``cursus-init-project`` dynamic workflow;
``project.bring_up`` points a caller at the ``cursus-new-project`` orchestrator that chains
scaffold -> DAG -> config end-to-end. See FZ 31e1d3f5c / c1 / c2 / c3 in the Cursus Simplification
Trail for the design.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..envelope import ToolResult, ToolError
from ..registry import ToolDef

# One-line purpose of this namespace (collected by the registry for project.help).
NAMESPACE = (
    "Scaffold a new Cursus pipeline project (phase-0 skeleton + action-item ledger)."
)


# ---------------------------------------------------------------------------
# Framework matrix — the only per-framework variation at t=0.
# ---------------------------------------------------------------------------

_FRAMEWORKS: Dict[str, Dict[str, str]] = {
    "xgboost": {
        "training": "xgboost_training.py",
        "inference": "xgboost_inference_handler.py",
        "hp_class": "XGBoostModelHyperparameters",
        "reference": "atoz_xgboost",
    },
    "pytorch": {
        "training": "pytorch_training.py",
        "inference": "pytorch_inference_handler.py",
        "hp_class": "ModelHyperparameters",
        "reference": "munged_address_pytorch",
    },
    "lightgbmmt": {
        "training": "lightgbm_training.py",
        "inference": "lightgbm_inference_handler.py",
        "hp_class": "LightGBMModelHyperparameters",
        "reference": "cap_mtgbm",
    },
    "bedrock": {
        "training": "bedrock_train.py",
        "inference": "bedrock_inference_handler.py",
        "hp_class": "BedrockModelHyperparameters",
        "reference": "rnr_pytorch_bedrock",
    },
}


# ---------------------------------------------------------------------------
# Fixed file templates. Placeholders: {PROJECT} {CLASS} {PREFIX} {FRAMEWORK}
#   {TRAINING} {INFERENCE} {HP_CLASS}
# Grounded in the real BAMT projects (secure_delivery_template_by_cursus,
# munged_address_pytorch_na.py). The two Python entry files are a fixed skeleton; only the
# import prefix, the class name, and the TODO value/metadata literals vary per project.
# ---------------------------------------------------------------------------

_RUN_PIPELINE_PY = '''\
#!/usr/bin/env python3
"""Compile pipeline_config/dag.json against config.json and start a SageMaker execution.

Fixed, region-agnostic-DAG template (the DAG is loaded from pipeline_config/dag.json, the
config from pipeline_config/config_<REGION>.json). Scaffolded by cursus project.init.

Usage:
    python3 run_pipeline.py --preview                       # offline validate/preview only
    python3 run_pipeline.py --region NA                     # compile + execute
    python3 run_pipeline.py --config pipeline_config/config_NA.json --dag pipeline_config/dag.json
"""
import sys
import os
import argparse
import json
import logging
from pathlib import Path

# SAIS bootstrap (sets PYTHONNOUSERSITE + .pth loading). Present only inside the sandbox tree.
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    import buyer_abuse_mods_template.sais_environment.sais_env_setup  # noqa: F401
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dag(dag_path):
    from {PREFIX}.api.dag import import_dag_from_json

    dag = import_dag_from_json(str(dag_path))
    logger.info(f"DAG loaded from {{dag_path}}: {{len(dag.nodes)}} nodes, {{len(dag.edges)}} edges")
    return dag


def setup_session():
    from secure_ai_sandbox_python_lib.session import Session as SaisSession
    from mods_workflow_helper.utils.secure_session import create_secure_session_config
    from mods_workflow_helper.sagemaker_pipeline_helper import SecurityConfig
    from sagemaker.workflow.pipeline_context import PipelineSession

    sais = SaisSession(".")
    security_config = SecurityConfig(
        kms_key=sais.get_team_owned_bucket_kms_key(),
        security_group=sais.sandbox_vpc_security_group(),
        vpc_subnets=sais.sandbox_vpc_subnets(),
    )
    sm_config = create_secure_session_config(
        role_arn=PipelineSession().get_caller_identity_arn(),
        bucket_name=sais.team_owned_s3_bucket_name(),
        kms_key=sais.get_team_owned_bucket_kms_key(),
        vpc_subnet_ids=sais.sandbox_vpc_subnets(),
        vpc_security_groups=[sais.sandbox_vpc_security_group()],
    )
    ps = PipelineSession(default_bucket=sais.team_owned_s3_bucket_name(), sagemaker_config=sm_config)
    ps.config = sm_config
    role = PipelineSession().get_caller_identity_arn()
    logger.info(f"Session initialized. Bucket: {{sais.team_owned_s3_bucket_name()}}")
    return sais, security_config, ps, role


def compile_pipeline(dag, config_path, ps, role):
    from {PREFIX}.core.compiler.dag_compiler import PipelineDAGCompiler
    from mods_workflow_core.utils.constants import (
        PIPELINE_EXECUTION_TEMP_DIR,
        KMS_ENCRYPTION_KEY_PARAM,
        SECURITY_GROUP_ID,
        VPC_SUBNET,
    )

    compiler = PipelineDAGCompiler(
        config_path=str(config_path),
        sagemaker_session=ps,
        role=role,
        pipeline_parameters=[
            PIPELINE_EXECUTION_TEMP_DIR,
            KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID,
            VPC_SUBNET,
        ],
        anchor_file=__file__,  # caller hook: anchor dockers/ paths to this project folder
    )
    pipeline, report = compiler.compile_with_report(dag=dag)
    logger.info(f"Pipeline '{{pipeline.name}}' compiled. Avg confidence: {{report.avg_confidence:.2f}}")
    return pipeline, compiler


def main():
    parser = argparse.ArgumentParser(description="Run the {PROJECT} pipeline")
    parser.add_argument("--config", help="Path to config JSON (default: pipeline_config/config_<REGION>.json)")
    parser.add_argument("--dag", help="Path to DAG JSON (default: pipeline_config/dag.json)")
    parser.add_argument("--region", default="NA", help="Region alias for the default config filename")
    parser.add_argument("--preview", action="store_true", help="Preview only (no SAIS session, no execution)")
    parser.add_argument("--save-exe-doc", help="Save the execution document to this file")
    args = parser.parse_args()

    here = Path(__file__).parent
    config_path = Path(args.config) if args.config else here / "pipeline_config" / f"config_{{args.region}}.json"
    dag_path = Path(args.dag) if args.dag else here / "pipeline_config" / "dag.json"

    if not dag_path.exists():
        logger.error(f"DAG not found: {{dag_path}}")
        sys.exit(1)

    dag = load_dag(dag_path)

    if args.preview:
        from {PREFIX}.core.compiler.dag_compiler import PipelineDAGCompiler

        compiler = PipelineDAGCompiler(
            config_path=str(config_path),
            sagemaker_session=None,
            role=None,
            anchor_file=__file__,  # caller hook: anchor dockers/ paths to this project folder
        )
        preview = compiler.preview_resolution(dag)
        for node, config_type in preview.node_config_map.items():
            logger.info(f"  {{node}} -> {{config_type}}")
        validation = compiler.validate_dag_compatibility(dag)
        logger.info(f"DAG validation: {{'VALID' if validation.is_valid else 'INVALID'}}")
        logger.info("Preview mode - no execution. Done.")
        return

    if not config_path.exists():
        logger.error(f"Config not found: {{config_path}} (run generate_config.py --region {{args.region}} first)")
        sys.exit(1)

    sais, security_config, ps, role = setup_session()
    pipeline, _ = compile_pipeline(dag, config_path, ps, role)

    from {PREFIX}.mods.exe_doc.generator import ExecutionDocumentGenerator
    from mods_workflow_helper.sagemaker_pipeline_helper import SagemakerPipelineHelper

    default_doc = SagemakerPipelineHelper.get_pipeline_default_execution_document(pipeline)
    exe_doc = ExecutionDocumentGenerator(
        config_path=str(config_path), sagemaker_session=ps, role=role, anchor_file=__file__
    ).fill_execution_document(dag=dag, execution_document=default_doc)

    if args.save_exe_doc:
        with open(args.save_exe_doc, "w") as f:
            json.dump(exe_doc, f, indent=2)
        logger.info(f"Execution document saved to {{args.save_exe_doc}}")

    SagemakerPipelineHelper.start_pipeline_execution(
        pipeline=pipeline,
        secure_config=security_config,
        sagemaker_session=ps,
        preparation_space_local_root="/tmp",
        pipeline_execution_document=exe_doc,
    )
    logger.info("Pipeline execution started successfully")


if __name__ == "__main__":
    main()
'''


_PIPELINE_TEMPLATE_PY = '''\
"""MODS deployment entry for {PROJECT}: the @MODSTemplate class MODS discovers + compiles.

Loads the DAG from pipeline_config/dag.json (never inline) and compiles it via
PipelineDAGCompiler. Scaffolded by cursus project.init — fill the TODO metadata before deploy.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from sagemaker import Session
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString

from {PREFIX}.api.dag.base_dag import PipelineDAG
from {PREFIX}.core.compiler.dag_compiler import PipelineDAGCompiler
from mods.mods_template import MODSTemplate

DEFAULT_MODEL_CLASS = "{FRAMEWORK}"
DEFAULT_REGION = "NA"
DEFAULT_SERVICE_NAME = "TODO_ServiceName"

AUTHOR = "TODO_alias"
PIPELINE_VERSION = "0.0.1"
PIPELINE_DESCRIPTION = "TODO one-line pipeline description"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_pipeline_dag() -> PipelineDAG:
    """Load the DAG from pipeline_config/dag.json (authored later; NOT defined inline)."""
    from {PREFIX}.api.dag import import_dag_from_json

    dag_path = Path(__file__).parent / "pipeline_config" / "dag.json"
    dag = import_dag_from_json(str(dag_path))
    logger.info(f"Loaded DAG from dag.json: {{len(dag.nodes)}} nodes, {{len(dag.edges)}} edges")
    return dag


@MODSTemplate(author=AUTHOR, description=PIPELINE_DESCRIPTION, version=PIPELINE_VERSION)
class {CLASS}:
    """Bridges the Cursus DAG-compiler pipeline to the MODS Template interface."""

    def __init__(
        self,
        sagemaker_session: Optional[Session] = None,
        execution_role: Optional[str] = None,
        regional_alias: str = DEFAULT_REGION,
        pipeline_parameters: Optional[List[Union[str, ParameterString]]] = None,
    ) -> None:
        self.sagemaker_session = sagemaker_session or Session()
        self.execution_role = execution_role or self.sagemaker_session.get_caller_identity_arn()
        self.pipeline_parameters = pipeline_parameters

        config_dir = Path(__file__).resolve().parent / "pipeline_config"
        self.config_path = str(config_dir / f"config_{{regional_alias}}.json")
        logger.info(f"Using config path: {{self.config_path}}")

        self.dag = create_pipeline_dag()
        self.dag_compiler = PipelineDAGCompiler(
            config_path=self.config_path,
            sagemaker_session=self.sagemaker_session,
            role=self.execution_role,
            pipeline_parameters=self.pipeline_parameters,
            anchor_file=__file__,  # caller hook: anchor dockers/ paths to this project folder
        )
        logger.info("Initialized DAG compiler")

    def generate_pipeline(self) -> Pipeline:
        """MODS interface method — compile the loaded DAG against the config."""
        pipeline, report = self.dag_compiler.compile_with_report(dag=self.dag)
        logger.info(f"Pipeline '{{pipeline.name}}' created (avg confidence {{report.avg_confidence:.2f}})")
        return pipeline

    def validate_dag_compatibility(self) -> Dict[str, Any]:
        v = self.dag_compiler.validate_dag_compatibility(self.dag)
        return {{
            "is_valid": v.is_valid,
            "missing_configs": v.missing_configs,
            "unresolvable_builders": v.unresolvable_builders,
            "config_errors": v.config_errors,
            "dependency_issues": v.dependency_issues,
            "warnings": v.warnings,
        }}

    def preview_resolution(self) -> Dict[str, Any]:
        p = self.dag_compiler.preview_resolution(self.dag)
        return {{
            "node_config_map": p.node_config_map,
            "config_builder_map": p.config_builder_map,
            "resolution_confidence": p.resolution_confidence,
            "ambiguous_resolutions": p.ambiguous_resolutions,
            "recommendations": p.recommendations,
        }}
'''


_GENERATE_CONFIG_PY = '''\
#!/usr/bin/env python3
"""Generate pipeline_config/config_<REGION>.json via DAGConfigFactory.

Shared skeleton scaffolded by cursus project.init. project_root_folder is filled with the
project name; fill the TODO per-node value-init (or run /cursus-configure-pipeline to author it).
"""
import sys

sys.setrecursionlimit(5000)

import argparse
import json
from datetime import date
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate {PROJECT} pipeline config")
parser.add_argument("--region", default="NA", choices=["NA", "EU", "FE"])
args = parser.parse_args()
region = args.region
aws_region = {{"NA": "us-east-1", "EU": "eu-west-1", "FE": "us-west-2"}}[region]

# SAIS-derived base values, with an offline placeholder fallback.
try:
    from sagemaker.workflow.pipeline_context import PipelineSession
    from secure_ai_sandbox_python_lib.session import Session as SaisSession

    _sais = SaisSession(".")
    _bucket = _sais.team_owned_s3_bucket_name()
    _role = PipelineSession().get_caller_identity_arn()
    _author = _sais.owner_alias()
except Exception as _e:  # pragma: no cover - depends on the sandbox environment
    print(f"WARNING: could not initialize SAIS session ({{_e}}); using placeholder base values.")
    _bucket = "REPLACE_WITH_TEAM_BUCKET"
    _role = "REPLACE_WITH_EXECUTION_ROLE_ARN"
    _author = "REPLACE_WITH_AUTHOR_ALIAS"

from {PREFIX}.api.dag import import_dag_from_json
from {PREFIX}.api.factory.dag_config_factory import DAGConfigFactory

dag = import_dag_from_json(str(Path(__file__).parent / "pipeline_config" / "dag.json"))
# caller hook: anchor dockers/ paths to this project folder (Strategy 0)
factory = DAGConfigFactory(dag, anchor_file=__file__)
config_map = factory.get_config_class_map()
print(f"DAG Node -> Config Class map ({{len(config_map)}} steps):")
for node_name, config_class in config_map.items():
    print(f"  {{node_name:<35}} -> {{config_class.__name__}}")

factory.set_base_config(
    bucket=_bucket,
    role=_role,
    region=region,
    aws_region=aws_region,
    author=_author,
    service_name="TODO_ServiceName",
    pipeline_version="0.0.1",
    framework_version="TODO",
    py_version="py3",
    source_dir="dockers",
    project_root_folder="{PROJECT}",  # filled by cursus project.init (load-bearing)
    current_date=date.today().strftime("%Y-%m-%d"),
)
if factory.get_base_processing_config_requirements():
    factory.set_base_processing_config(
        processing_source_dir="dockers/scripts",
        processing_instance_type_large="ml.m5.12xlarge",
        processing_instance_type_small="ml.m5.4xlarge",
    )

pending = factory.get_pending_steps()
# ======================= TODO: value-init, one block per DAG node =======================
# For each node in `pending`, call factory.set_step_config(node, **values). The VALUES
# (field lists, transform SQL, EDX ARNs, hyperparameters, instance types) are project-specific.
# Run /cursus-configure-pipeline to author + gate this block (author.config_constraints +
# author.preflight_config). Example shape:
#   if "XGBoostTraining" in pending:
#       factory.set_step_config("XGBoostTraining", training_entry_point="{TRAINING}", ...)
# ========================================================================================

final_pending = factory.get_pending_steps()
if final_pending:
    print(f"Still pending (configure these first): {{final_pending}}")
    sys.exit(1)

from {PREFIX}.steps.configs.utils import merge_and_save_configs

out = Path(__file__).parent / "pipeline_config" / f"config_{{region}}.json"
out.parent.mkdir(parents=True, exist_ok=True)
merge_and_save_configs(factory.generate_all_configs(), str(out))
print(f"Saved: {{out}}")
'''


_DAG_STUB = {"dag": {"nodes": [], "edges": []}}


def _readme_pipeline_config() -> str:
    return (
        "# pipeline_config/\n\n"
        "Holds the per-region pipeline artifacts:\n\n"
        "- `dag.json` — the pipeline DAG (nodes + edges). Empty until you author it.\n"
        "- `config_<REGION>.json` — per-region step configs, produced by `generate_config.py`.\n"
        "- `exe_doc_<REGION>.json` — the MODS execution document (filled at runtime).\n"
    )


def _readme_dockers() -> str:
    return (
        "# dockers/\n\n"
        "The `source_dir` root. Put the framework model scripts the framework does NOT ship here at "
        "the ROOT of this folder — `<framework>_training.py`, `<framework>_inference_handler.py`, "
        "`<framework>_model_eval.py`. Subfolders: `scripts/` (Processing scripts), `processing/` "
        "(domain code), `hyperparams/` (per-region hyperparameters).\n"
    )


def _readme_scripts() -> str:
    return (
        "# dockers/scripts/\n\n"
        "The `processing_source_dir`: application-agnostic Processing step scripts, one per DAG node "
        "(e.g. `tabular_preprocessing.py`, `model_calibration.py`, `risk_table_mapping.py`). Copy the "
        "canonical version from `cursus/steps/scripts/` per DAG node, then adapt.\n"
    )


def _readme_processing() -> str:
    return (
        "# dockers/processing/\n\n"
        "Domain-specific processing code YOU write (feature engineering, tokenizers, datasets). "
        "Deployed projects vendor their own copy here (imported bare as `from processing.*`), rather "
        "than importing `cursus.processing`.\n"
    )


def _readme_hyperparams() -> str:
    return (
        "# dockers/hyperparams/\n\n"
        "Per-region training hyperparameters: `hyperparameters_<REGION>.json`, built from the "
        "full/cat/tab field lists + label + id + the framework defaults.\n"
    )


def _root_ledger_readme(project: str, framework: str, fw: Dict[str, str]) -> str:
    return (
        f"# {project} — Cursus pipeline project (scaffolded, NOT yet runnable)\n\n"
        "Created by `cursus project.init`. This is the phase-0 skeleton: the fixed structure + "
        "`run_pipeline.py` + the `@MODSTemplate` class + a `generate_config.py` skeleton with "
        "`project_root_folder` filled. Complete the steps below **in order** before the pipeline runs.\n\n"
        "## Action items (remaining)\n"
        "- [ ] **1. Author the DAG** -> write `pipeline_config/dag.json` (nodes + edges). "
        "For a NEW step type not in the registry, run `/cursus-author-step`.\n"
        "- [ ] **2. Copy per-node Processing scripts** into `dockers/scripts/` — one per DAG node, "
        "from `cursus/steps/scripts/`.\n"
        f"- [ ] **3. Copy the framework handlers** into `dockers/` — `{fw['training']}` + "
        f"`{fw['inference']}` — from a reference project, then adapt.\n"
        "- [ ] **4. Fill hyperparameters** -> `dockers/hyperparams/hyperparameters_<REGION>.json`.\n"
        f"- [ ] **5. Fill the @MODSTemplate metadata** — `AUTHOR` / `PIPELINE_DESCRIPTION` / "
        f"`DEFAULT_SERVICE_NAME` in `{project}_pipeline.py`.\n"
        "- [ ] **6. Fill the config value-init** -> the `TODO` block in `generate_config.py` "
        "(+ base `service_name`/`framework_version`). Run `/cursus-configure-pipeline` to author + gate it.\n"
        "- [ ] **7. Generate config** -> `python generate_config.py --region <R>` -> `config_<R>.json`.\n"
        "- [ ] **8. Preview then run** -> `python run_pipeline.py --preview`, then `--region <R>`.\n\n"
        "## What is already done\n"
        "- [x] Folder skeleton + per-folder READMEs\n"
        "- [x] `run_pipeline.py` (fixed, region-agnostic-DAG template)\n"
        f"- [x] `{project}_pipeline.py` — the `@MODSTemplate` class (loads `dag.json`; AUTHOR/desc are TODO)\n"
        f'- [x] `generate_config.py` skeleton with `project_root_folder="{project}"`\n'
        "- [x] empty `pipeline_config/dag.json` stub\n"
    )


def _pascal(name: str) -> str:
    """snake_case (or any) name -> PascalCase."""
    parts = [p for p in name.replace("-", "_").split("_") if p]
    return "".join(w[:1].upper() + w[1:] for w in parts)


def _init(args: Dict[str, Any]) -> ToolResult:
    """
    Scaffold a new phase-0 Cursus project package.

    Writes the fixed skeleton (run_pipeline.py, the @MODSTemplate class, the generate_config.py
    skeleton, an empty dag.json, per-folder READMEs) and the root README action-item ledger, with
    the project name + framework substituted in. Deterministic and offline — no engine import.
    """
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolError("'name' must be a non-empty project name", code="invalid_input")
    name = name.strip()

    framework = str(args.get("framework", "")).strip().lower()
    if framework not in _FRAMEWORKS:
        raise ToolError(
            f"'framework' must be one of {sorted(_FRAMEWORKS)}",
            code="invalid_input",
            details={"got": framework or None},
        )
    fw = _FRAMEWORKS[framework]

    target_dir = str(args.get("target_dir", "projects")).strip() or "projects"
    overwrite = bool(args.get("overwrite", False))

    project = f"{name}_{framework}"
    root = Path(target_dir) / project
    pascal_class = _pascal(name) + "Pipeline"
    # Import prefix: the vendored BAMT copy vs the AmazonCursus dev package.
    prefix = (
        "buyer_abuse_mods_template.cursus"
        if "buyer_abuse_mods_template" in target_dir
        else "cursus"
    )

    if root.exists() and not overwrite:
        return ToolResult.failure(
            f"target already exists: {root} (pass overwrite=true to allow writing into it)",
            code="already_exists",
            details={"target_path": str(root)},
            remedy={
                "suggested_tools": ["project.init"],
                "fix_action": "Choose a different name/target_dir, or re-call with overwrite=true.",
            },
        )

    subst = {
        "PROJECT": project,
        "CLASS": pascal_class,
        "PREFIX": prefix,
        "FRAMEWORK": framework,
        "TRAINING": fw["training"],
        "INFERENCE": fw["inference"],
        "HP_CLASS": fw["hp_class"],
    }

    def _fill(template: str) -> str:
        return template.format(**subst)

    # (relative path, contents) — everything knowable at t=0.
    files: List[tuple] = [
        ("__init__.py", ""),
        ("run_pipeline.py", _fill(_RUN_PIPELINE_PY)),
        (f"{project}_pipeline.py", _fill(_PIPELINE_TEMPLATE_PY)),
        ("generate_config.py", _fill(_GENERATE_CONFIG_PY)),
        ("pipeline_config/dag.json", json.dumps(_DAG_STUB, indent=2) + "\n"),
        ("pipeline_config/README.md", _readme_pipeline_config()),
        ("dockers/__init__.py", ""),
        ("dockers/README.md", _readme_dockers()),
        ("dockers/scripts/README.md", _readme_scripts()),
        ("dockers/processing/README.md", _readme_processing()),
        ("dockers/hyperparams/README.md", _readme_hyperparams()),
        ("README.md", _root_ledger_readme(project, framework, fw)),
    ]

    written: List[str] = []
    try:
        for rel, contents in files:
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(contents)
            written.append(rel)
    except OSError as exc:
        raise ToolError(
            f"failed to write scaffold under {root}: {exc}",
            code="internal_error",
            details={"exception": type(exc).__name__, "written_so_far": written},
        )

    data = {
        "project": project,
        "target_path": str(root),
        "framework": framework,
        "import_prefix": prefix,
        "pipeline_class": pascal_class,
        "files_written": written,
        "ready": True,
    }
    return ToolResult.success(
        data,
        next_steps=[
            {
                "tool": "/cursus-author-step",
                "when": "a DAG node is a NEW step type not in the registry",
                "why": "author the .step.yaml + config + script before building the DAG",
            },
            {
                "tool": "/cursus-configure-pipeline",
                "when": "after pipeline_config/dag.json is authored",
                "why": "fill the generate_config.py TODO value-init and write config_<REGION>.json",
                "args_hint": {"dag_nodes": ["<node>", "..."], "project": project},
            },
        ],
        file_count=len(written),
    )


def _bring_up(args: Dict[str, Any]) -> ToolResult:
    """
    Point the caller at the cursus-new-project auto-chain orchestrator.

    End-to-end bring-up (scaffold -> seed/author a DAG -> generate config) is a multi-phase
    dynamic workflow, not a single stateless tool call: it needs a workflow harness (the
    ``cursus-new-project`` script). This tool validates the inputs and returns the exact
    invocation, so an agent without a workflow runtime still gets an actionable next step
    rather than a half-run chain.
    """
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolError("'name' must be a non-empty project name", code="invalid_input")
    framework = str(args.get("framework", "")).strip().lower()
    if framework not in _FRAMEWORKS:
        raise ToolError(
            f"'framework' must be one of {sorted(_FRAMEWORKS)}",
            code="invalid_input",
            details={"got": framework or None},
        )
    dag_source = str(args.get("dag_source", "catalog")).strip().lower()
    if dag_source not in ("catalog", "manual"):
        raise ToolError(
            "'dag_source' must be 'catalog' or 'manual'", code="invalid_input"
        )

    wf_args = {
        "name": name.strip(),
        "framework": framework,
        "target_dir": str(args.get("target_dir", "projects")).strip() or "projects",
        "region": str(args.get("region", "NA")).strip() or "NA",
        "dag_source": dag_source,
    }
    return ToolResult.success(
        {
            "workflow": "cursus-new-project",
            "workflow_path": "src/cursus/mcp/workflows/cursus-new-project.js",
            "args": wf_args,
            "note": (
                "End-to-end bring-up composes cursus-init-project -> DAG seed/author -> "
                "cursus-configure-pipeline. Run the workflow with these args; dag_source='catalog' "
                "chains fully, dag_source='manual' stops after scaffold for a human to author the DAG."
            ),
        },
        next_steps=[
            {
                "tool": "project.init",
                "when": "you only need the phase-0 skeleton (no DAG/config chaining)",
                "why": "project.init is the deterministic scaffold-only step this orchestrator's first phase runs",
            }
        ],
    )


# ---------------------------------------------------------------------------
# Tool registry for this namespace
# ---------------------------------------------------------------------------

_FRAMEWORK_ENUM = sorted(_FRAMEWORKS)

TOOLS: List[ToolDef] = [
    ToolDef(
        name="project.init",
        description=(
            "Scaffold a NEW Cursus pipeline project (phase-0). Writes the fixed skeleton — a "
            "region-agnostic run_pipeline.py, the @MODSTemplate deployment class (loads "
            "pipeline_config/dag.json), a shared generate_config.py skeleton with project_root_folder "
            "filled + a TODO per-node value-init, an empty dag.json stub, the folder tree + per-folder "
            "READMEs — and a root README action-item ledger handing every context-dependent piece "
            "(author the DAG, copy scripts/handlers, fill config) to its owning downstream workflow. "
            "Deterministic and offline; generates only what is knowable before a DAG exists."
        ),
        schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Project base name (snake_case), e.g. 'secure_delivery'. The framework is appended as a suffix.",
                },
                "framework": {
                    "type": "string",
                    "enum": _FRAMEWORK_ENUM,
                    "description": "Model framework; becomes the project-name suffix and selects the handler/hyperparameter template.",
                },
                "target_dir": {
                    "type": "string",
                    "description": "Directory to create <name>_<framework>/ under. Default 'projects' (AmazonCursus dev); pass 'src/buyer_abuse_mods_template' for a BAMT deploy (also sets the import prefix).",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Write into an existing target directory instead of failing. Default false.",
                },
            },
            "required": ["name", "framework"],
            "additionalProperties": False,
        },
        handler=_init,
        destructive=True,  # creates a new folder tree on disk
        tags=("planner",),
        when=(
            "Call at the very start of a new pipeline project, before any DAG or config exists, "
            "to lay down the standard package skeleton + the checklist of what to do next."
        ),
        examples=(
            'project.init {"name": "secure_delivery", "framework": "xgboost"}  # -> projects/secure_delivery_xgboost/',
            'project.init {"name": "abuse_polygraph", "framework": "pytorch", "target_dir": "src/buyer_abuse_mods_template"}  # BAMT deploy target (sets import prefix)',
            'project.init {"name": "fraud_scan", "framework": "lightgbmmt", "overwrite": true}  # write into an existing folder',
        ),
    ),
    ToolDef(
        name="project.bring_up",
        description=(
            "Return the invocation for the cursus-new-project auto-chain orchestrator, which composes "
            "scaffold -> DAG (catalog-seeded or human-authored) -> config generation end-to-end. "
            "Bring-up is a multi-phase workflow, not a single stateless call, so this validates the "
            "inputs and hands back the exact workflow + args to run."
        ),
        schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Project base name (snake_case).",
                },
                "framework": {
                    "type": "string",
                    "enum": _FRAMEWORK_ENUM,
                    "description": "Model framework.",
                },
                "target_dir": {
                    "type": "string",
                    "description": "Where the project folder is created (default 'projects').",
                },
                "region": {
                    "type": "string",
                    "description": "Region alias for the generated config (default 'NA').",
                },
                "dag_source": {
                    "type": "string",
                    "enum": ["catalog", "manual"],
                    "description": "'catalog' seeds a shared DAG and chains fully; 'manual' stops after scaffold for a human to author the DAG.",
                },
            },
            "required": ["name", "framework"],
            "additionalProperties": False,
        },
        handler=_bring_up,
        tags=("planner",),
        when=(
            "Call when you want the whole project brought up end-to-end (skeleton + DAG + config), "
            "not just the phase-0 skeleton — it returns the orchestrator workflow to run."
        ),
        examples=(
            'project.bring_up {"name": "secure_delivery", "framework": "xgboost"}  # catalog DAG, full chain',
            'project.bring_up {"name": "new_model", "framework": "pytorch", "dag_source": "manual"}  # scaffold, then human authors the DAG',
            'project.bring_up {"name": "eu_risk", "framework": "bedrock", "region": "EU"}  # EU config target',
        ),
    ),
]
