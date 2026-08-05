#!/usr/bin/env python3
"""Generate the Step Catalog doc section from the live ``.step.yaml`` interfaces.

Writes real Markdown files to ``docs/steps-catalog/`` so BOTH doc surfaces pick them up:
  * MkDocs (code.amazon.com Releases-tab site) — reads ``docs/``.
  * The wiki — ``AmazonCursusWikiDocsCDK``'s ``transform_docs.py`` reads ``docs/*.md``.

(Unlike ``gen_api_ref.py``, which uses mkdocs-gen-files' *virtual* pages, this writes to
disk. Virtual pages never reach the wiki transform, and the step catalog must appear on
both surfaces — so it is generated to disk and committed.)

It writes:
  * ``steps-catalog/index.md`` — the main entry point: overview + a table of every step
    grouped by SageMaker step type, with node type, purpose, and consumed/produced types.
  * ``steps-catalog/<step>.md`` — one page per step: purpose, node type, SageMaker step
    type, inputs (dependency type / required / compatible producers), outputs (+ downstream
    consumers derived by a reverse index over ``compatible_sources``), framework
    requirements, and the container entry point.

Regenerate after adding or changing any ``.step.yaml``:

    python3 docs/gen_step_catalog.py

Run from the package root (``~/AmazonCursus/src/AmazonCursus``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent  # docs/
PKG_ROOT = HERE.parent  # AmazonCursus package root
INTERFACES = PKG_ROOT / "src" / "cursus" / "steps" / "interfaces"
OUT_DIR = HERE / "steps-catalog"

STEP_TYPE_BLURB = {
    "Processing": "Processing jobs — data prep, feature engineering, evaluation, packaging.",
    "Training": "Training jobs — fit a model from prepared data + hyperparameters.",
    "Tuning": "Hyperparameter tuning jobs — search a training step's hyperparameters for the best model.",
    "Transform": "Batch transform jobs — run inference over a dataset with a model.",
    "CreateModel": "Model creation — wrap trained artifacts into a deployable SageMaker model.",
    "CradleDataLoading": "Cradle data loading — pull source data via the SAIS/Cradle SDK.",
    "RedshiftDataLoading": "Redshift data loading — load data from Redshift via the SAIS SDK.",
    "MimsModelRegistrationProcessing": "Model registration — register a model with MIMS.",
}

NODE_TYPE_NOTE = {
    "source": "source (no inputs — originates data)",
    "internal": "internal (consumes upstream, produces downstream)",
    "sink": "sink (terminal — produces no pipeline outputs)",
}


def load_steps() -> list[dict]:
    steps = []
    for f in sorted(INTERFACES.glob("*.step.yaml")):
        y = yaml.safe_load(f.read_text()) or {}
        reg = y.get("registry") or {}
        contract = y.get("contract") or {}
        spec = y.get("spec") or {}
        # `compute:` is declared at the top level (the back-compat contract.compute mirror is
        # tolerated); read whichever carries a kind.
        compute = y.get("compute") or contract.get("compute") or {}
        steps.append(
            {
                "file": f.name,
                "slug": f.name[: -len(".step.yaml")],
                "step_type": y.get("step_type") or f.stem,
                "node_type": y.get("node_type", "internal"),
                "sagemaker_step_type": reg.get("sagemaker_step_type", "Processing"),
                "requires": reg.get("requires"),
                "description": (
                    contract.get("description") or reg.get("description") or ""
                ).strip(),
                "short_desc": (reg.get("description") or "").strip(),
                "entry_point": contract.get("entry_point"),
                "framework_requirements": contract.get("framework_requirements") or {},
                "dependencies": spec.get("dependencies") or {},
                "outputs": spec.get("outputs") or {},
                "compute": compute,
            }
        )
    return steps


def build_consumer_index(steps: list[dict]) -> dict[str, list[str]]:
    """Reverse index: step_type -> [step_types that list it as a compatible source]."""
    consumers: dict[str, set] = {s["step_type"]: set() for s in steps}
    for s in steps:
        for dep in s["dependencies"].values():
            for src in dep.get("compatible_sources") or []:
                consumers.setdefault(src, set()).add(s["step_type"])
    return {k: sorted(v) for k, v in consumers.items()}


def compute_kind_label(compute: dict) -> str:
    """A one-word compute-kind label for the index (e.g. 'byo_container', 'sklearn', '—')."""
    return compute.get("kind") or "—"


def render_compute_section(compute: dict) -> list:
    """Render the step's compute descriptor — how its container/estimator is built.

    Surfaces the additive extensions (FZ 31e1d3m/o): a `byo_container` kind runs a user-supplied
    ECR image verbatim (no image_uris.retrieve); `network_mode: config` attaches a per-step VPC.
    """
    kind = compute.get("kind")
    if not kind:
        return []
    L: list = ["## Compute\n", "| | |", "|---|---|", f"| **Compute kind** | `{kind}` |"]

    if kind == "byo_container":
        img_field = compute.get("image_uri_field", "image_uri")
        L.append(
            f"| **Image** | BYO container — `config.{img_field}` is passed VERBATIM to "
            "`AppSpecification.ImageUri` / `AlgorithmSpecification.TrainingImage` (no "
            "`image_uris.retrieve`; the framework deps live in the image's Dockerfile) |"
        )
        entry = compute.get("container_entrypoint")
        if entry:
            L.append(
                f"| **Container entrypoint** | `{entry}` (ContainerEntrypoint bypass — the image "
                "runs its own entrypoint instead of the SageMaker toolkit) |"
            )
    elif compute.get("sdk_class"):
        L.append(
            f"| **SDK class** | `{compute['sdk_class']}` (SageMaker DLC via `image_uris.retrieve`) |"
        )
    if compute.get("retrieve_framework"):
        L.append(
            f"| **Retrieve framework** | `{compute['retrieve_framework']}` (training DLC framework) |"
        )

    net = compute.get("network_mode")
    if net and net != "none":
        if net == "config":
            L.append(
                "| **Network** | `network_mode: config` — per-step VPC: attaches the step's own "
                "`subnets` / `security_group_ids` (overrides the session-wide default), for reaching "
                "a VPC-only data source |"
            )
        elif net == "shared":
            L.append(
                "| **Network** | `network_mode: shared` — the shared SAIS VpcConfig + volume KMS |"
            )
    elif compute.get("kms_network"):
        L.append(
            "| **Network** | `kms_network` — the shared SAIS VpcConfig + volume KMS (script kind) |"
        )
    L.append("")
    return L


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    for end in (". ", "! ", "? "):
        i = text.find(end)
        if i != -1:
            return text[: i + 1]
    return text if text.endswith((".", "!", "?")) else text + "."


def render_step_page(
    s: dict, consumers: dict[str, list[str]], slug_by_type: dict[str, str]
) -> str:
    L: list[str] = []
    L.append(f"# {s['step_type']}\n")
    if s["short_desc"]:
        L.append(f"**{s['short_desc']}**\n")

    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| **SageMaker step type** | `{s['sagemaker_step_type']}` |")
    L.append(
        f"| **Node type** | {NODE_TYPE_NOTE.get(s['node_type'], s['node_type'])} |"
    )
    if s["entry_point"]:
        L.append(f"| **Container entry point** | `{s['entry_point']}` |")
    if s["requires"]:
        L.append(
            f"| **Build-time requirement** | `{s['requires']}` (SAIS SDK — fatal on load if absent) |"
        )
    L.append(f"| **Interface file** | `steps/interfaces/{s['file']}` |")
    L.append("")

    L.extend(render_compute_section(s["compute"]))

    if s["description"] and s["description"] != s["short_desc"]:
        L.append("## Functionality\n")
        L.append(s["description"] + "\n")

    L.append("## Inputs (dependencies)\n")
    deps = s["dependencies"]
    if not deps:
        L.append(
            "_None — this is a source step; it originates data with no pipeline inputs._\n"
        )
    else:
        L.append("| Input | Type | Required | Compatible producers |")
        L.append("|-------|------|----------|----------------------|")
        for name, dep in deps.items():
            typ = dep.get("type", "—")
            req = "yes" if dep.get("required") else "no"
            srcs = dep.get("compatible_sources") or []
            links = [
                f"[{st}]({slug_by_type[st]}.md)" if st in slug_by_type else st
                for st in srcs
            ]
            L.append(
                f"| `{name}` | `{typ}` | {req} | {', '.join(links) if links else '—'} |"
            )
        L.append("")

    L.append("## Outputs\n")
    outs = s["outputs"]
    if not outs:
        L.append(
            "_None — this is a sink step; it produces no downstream pipeline outputs._\n"
        )
    else:
        L.append("| Output | Type |")
        L.append("|--------|------|")
        for name, out in outs.items():
            L.append(f"| `{name}` | `{out.get('type', '—')}` |")
        L.append("")

    downstream = consumers.get(s["step_type"], [])
    L.append("## Consumers (downstream steps)\n")
    if downstream:
        L.append("Steps that declare this step as a compatible input source:\n")
        for st in downstream:
            L.append(
                f"- [{st}]({slug_by_type[st]}.md)" if st in slug_by_type else f"- {st}"
            )
        L.append("")
    else:
        L.append(
            "_No cataloged step lists this step as a compatible source "
            "(it may be a terminal/sink step, or consumed via a generic source name)._\n"
        )

    fr = s["framework_requirements"]
    if fr:
        L.append("## Framework requirements\n")
        L.append("| Package | Version |")
        L.append("|---------|---------|")
        for pkg, ver in fr.items():
            L.append(f"| `{pkg}` | `{ver}` |")
        L.append("")

    L.append("---\n")
    L.append("← [Back to the Step Catalog](index.md)\n")
    return "\n".join(L)


def render_index(steps: list[dict], consumers: dict[str, list[str]]) -> str:
    L: list[str] = []
    L.append("# Step Catalog\n")
    L.append(
        f"Every pipeline step that cursus supports — **{len(steps)} steps** — generated "
        "directly from the `.step.yaml` interface files. Each row links to that step's page: "
        "its purpose, its inputs (with the upstream steps that can produce them), its outputs, "
        "and the downstream steps that consume it.\n"
    )
    L.append(
        "A cursus pipeline is a DAG of these steps. An edge is valid when a downstream step's "
        "input **type** matches an upstream step's output, and the upstream step is listed among "
        "the input's *compatible producers* — see [The DAG + Config → Pipeline model]"
        "(../concepts/dag_and_compilation.md) and [Registry and Step Catalog]"
        "(../concepts/registry_and_discovery.md).\n"
    )
    L.append(
        "The **Compute** column names how a step's container is built: an SDK-managed DLC "
        "(`sklearn` / `xgboost` / `framework` / `estimator` / `model`), the SAIS `script` image, "
        "or **`byo_container`** — a user-supplied ECR image run verbatim (no `image_uris.retrieve`), "
        "which is how a non-DLC framework (e.g. GraphStorm/DGL) enters cursus. A step may also "
        "declare a per-step VPC (`network_mode: config`) to reach a VPC-only data source — shown on "
        "its page's **Compute** section.\n"
    )

    by_type: dict[str, list[dict]] = {}
    for s in steps:
        by_type.setdefault(s["sagemaker_step_type"], []).append(s)

    order = [
        "CradleDataLoading",
        "RedshiftDataLoading",
        "Processing",
        "Training",
        "Transform",
        "CreateModel",
        "MimsModelRegistrationProcessing",
    ]
    ordered_types = [t for t in order if t in by_type] + [
        t for t in by_type if t not in order
    ]

    ordered_slugs: list[str] = []
    for t in ordered_types:
        group = sorted(by_type[t], key=lambda x: x["step_type"])
        L.append(f"## {t}\n")
        if t in STEP_TYPE_BLURB:
            L.append(f"_{STEP_TYPE_BLURB[t]}_\n")
        L.append("| Step | Node | Compute | Purpose | Consumes | Produces |")
        L.append("|------|------|---------|---------|----------|----------|")
        for s in group:
            ordered_slugs.append(s["slug"])
            purpose = first_sentence(s["short_desc"] or s["description"]) or "—"
            dep_types = sorted(
                {d.get("type", "") for d in s["dependencies"].values() if d.get("type")}
            )
            out_types = sorted(
                {o.get("type", "") for o in s["outputs"].values() if o.get("type")}
            )
            consumes = ", ".join(f"`{x}`" for x in dep_types) if dep_types else "—"
            produces = ", ".join(f"`{x}`" for x in out_types) if out_types else "—"
            compute_label = compute_kind_label(s["compute"])
            L.append(
                f"| [{s['step_type']}]({s['slug']}.md) | {s['node_type']} | `{compute_label}` "
                f"| {purpose} | {consumes} | {produces} |"
            )
        L.append("")

    L.append("---\n")
    L.append(
        "*This catalog is generated from `src/cursus/steps/interfaces/*.step.yaml` by "
        "`docs/gen_step_catalog.py`. To change a step's catalog entry, edit its `.step.yaml` "
        "and re-run the generator.*\n"
    )

    # Hidden Sphinx toctree so every per-step page is adopted into the site nav (and no
    # "document isn't included in any toctree" warnings). Ordered as the tables above.
    L.append("```{toctree}")
    L.append(":hidden:")
    L.append(":maxdepth: 1")
    L.append("")
    for slug in ordered_slugs:
        L.append(slug)
    L.append("```")
    return "\n".join(L)


def main() -> int:
    if not INTERFACES.is_dir():
        print(
            f"gen_step_catalog: interfaces dir {INTERFACES} not found", file=sys.stderr
        )
        return 1
    steps = load_steps()
    consumers = build_consumer_index(steps)
    slug_by_type = {s["step_type"]: s["slug"] for s in steps}

    # Rewrite the whole directory so removed steps don't leave stale pages behind.
    if OUT_DIR.exists():
        for old in OUT_DIR.glob("*.md"):
            old.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "index.md").write_text(render_index(steps, consumers), encoding="utf-8")
    for s in steps:
        (OUT_DIR / f"{s['slug']}.md").write_text(
            render_step_page(s, consumers, slug_by_type), encoding="utf-8"
        )
    print(f"gen_step_catalog: wrote {len(steps) + 1} pages into {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
