"""
Single source of truth for step-name <-> file-name conversion in the step catalog.

Historically each discovery module (script_discovery, builder_discovery, contract_discovery)
and the steps.interfaces loader carried its own PascalCase<->snake_case table of compound
acronyms (XGBoost, PyTorch, LightGBM, ...). Those tables drifted (e.g. LightGBM was in one
but not another; contract_discovery had none and mangled ``PyTorchTraining`` ->
``py_torch_training``), silently breaking name resolution on whichever path used the stale
table — the root cause of the TSA/SOPA filename races and the "8 of 40 discovered" class of
bug.

This module centralizes that logic:

- :data:`COMPOUND_ACRONYMS` — the one canonical list of multi-word tokens that must not be
  split on internal capitals.
- :func:`canonical_to_snake` — PascalCase canonical name -> snake_case file stem
  (``PyTorchTraining`` -> ``pytorch_training``).
- :func:`parts_to_pascal` — snake_case parts -> PascalCase canonical name (the inverse).
- :func:`canonical_key` — case/separator-insensitive key for robust fallback matching.

New frameworks are added in ONE place here. Better still, :func:`canonical_key` enables a
normalized directory scan so most new acronyms resolve with no table edit at all.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

# Compound acronyms / multi-word tokens whose internal capitalization must be preserved as
# a single unit when converting between PascalCase and snake_case. Order matters: longer
# tokens first so e.g. ``LightGBMMT`` is matched before ``LightGBM``.
COMPOUND_ACRONYMS: List[str] = [
    "LightGBMMT",
    "LightGBM",
    "XGBoost",
    "PyTorch",
    "TensorFlow",
    "SageMaker",
    "MLFlow",
    "AutoML",
]

# snake_case form -> canonical PascalCase form, derived from COMPOUND_ACRONYMS so the two
# directions can never disagree.
_SNAKE_TO_PASCAL = {a.lower(): a for a in COMPOUND_ACRONYMS}


# --- Job-type vocabulary (single source of truth) -----------------------------------
#
# These were previously duplicated (and had drifted) across step_catalog.py, spec_discovery.py
# and registry/step_names.py. Two genuinely-different concepts are kept distinct here:
#
# JOB_TYPE_SUFFIXES — trailing tokens that mark a *job-type variant* of a step (used to
#   detect/filter variants like ``xgboost_training`` or ``foo_inference``). It deliberately
#   does NOT contain "model": a step like ``XGBoostModel`` is a distinct step kind, not a
#   variant, and including "model" here would wrongly filter it out of list_available_steps.
JOB_TYPE_SUFFIXES = (
    "training",
    "validation",
    "testing",
    "calibration",
    "inference",
    "evaluation",
    "batch",
    "export",
    "scoring",
)

# JOB_TYPE_KEYWORDS — tokens used to *classify* which job type a spec/file name belongs to
#   (matched as a substring, first-hit-wins). This is the classification concept, distinct from
#   the variant-suffix concept above: it includes "model" (e.g. ``xgboost_model_eval`` ->
#   "model") and is intentionally ordered so the more specific structural words win first.
#   Order and membership preserve the historical spec_discovery classification behavior.
JOB_TYPE_KEYWORDS = (
    "training",
    "validation",
    "testing",
    "calibration",
    "model",
)

# Abstract/base config step names that are never concrete pipeline steps and must be excluded
# from discovery/listing. Previously duplicated as a literal set in step_catalog.py and
# validation/builders/universal_test.py.
BASE_CONFIGS = frozenset({"Base", "Processing"})


def is_job_type_variant(step_name: str) -> bool:
    """Return True if ``step_name`` ends in a known job-type suffix (``foo_training``).

    Matches on a trailing ``_<suffix>`` so it only fires on snake_case variant names and
    never on a base step whose name merely contains a job word.
    """
    lowered = step_name.lower()
    return any(lowered.endswith(f"_{suffix}") for suffix in JOB_TYPE_SUFFIXES)


def canonical_to_snake(canonical_name: str) -> str:
    """Convert a PascalCase canonical step name to its snake_case file stem.

    Compound acronyms are protected (``XGBoostTraining`` -> ``xgboost_training``, not
    ``x_g_boost_training``); the remaining PascalCase is split on capital boundaries,
    including runs of capitals followed by a word (``MyABCStep`` -> ``my_abc_step``).
    """
    # Protect compound acronyms by collapsing their internal capitals first
    # (e.g. "XGBoost" -> "Xgboost") so the generic regex treats them as one word.
    processed = canonical_name
    for acronym in COMPOUND_ACRONYMS:
        processed = processed.replace(acronym, acronym[0] + acronym[1:].lower())

    # Capital-run followed by a Capital+lowercase word: "ABCWord" -> "ABC_Word"
    result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", processed)
    # lowercase/digit followed by Capital: "wordWord" -> "word_Word"
    result = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", result)
    return result.lower()


def parts_to_pascal(parts: Iterable[str]) -> str:
    """Convert snake_case parts to a PascalCase canonical name (inverse of split).

    ``["xgboost", "training"]`` -> ``"XGBoostTraining"``. Parts matching a known compound
    acronym keep its canonical casing; others are simply capitalized.
    """
    out: List[str] = []
    for part in parts:
        out.append(_SNAKE_TO_PASCAL.get(part.lower(), part.capitalize()))
    return "".join(out)


def canonical_key(name: str) -> str:
    """Collapse a name/stem to a case- and separator-insensitive key.

    ``"XGBoostTraining"``, ``"xgboost_training"`` and ``"XGBoost_Training"`` all map to
    ``"xgboosttraining"``. Used as a robust fallback when the exact convention name misses,
    so a new acronym step still resolves without editing :data:`COMPOUND_ACRONYMS`.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


def resolve_base_step_name(node_name: str, known_step_names: Iterable[str]) -> Optional[str]:
    """Resolve a DAG node name to its base registry step name — ROBUSTLY, without a hardcoded
    suffix list.

    A node is ``<StepName>[_<suffix>...]`` where ``<suffix>`` is an arbitrary label (job_type,
    data-source tag, split name — e.g. ``_training``, ``_munged``, ``_sampling``, ``_tagging``,
    ``_baseline``, ``_embedding``). Since job_type is now open (any lowercase-alnum, deep dive
    Tranche 3), the suffix is NOT drawn from a fixed set, so matching a trailing token against
    ``JOB_TYPE_SUFFIXES`` misses real nodes. Instead: strip trailing ``_segment`` groups one at a
    time and return the first prefix that is an actual known step name (matched via
    :func:`canonical_key`, so compound-acronym casing like ``XgboostMt`` vs ``XGBoostMT`` also
    resolves). The step registry — not a suffix allowlist — is the authority for what a base name is.

    Returns the canonical known step name, or ``None`` if no prefix matches.
    """
    by_key = {}
    for s in known_step_names:
        by_key.setdefault(canonical_key(s), s)
    parts = node_name.split("_")
    for cut in range(len(parts), 0, -1):
        hit = by_key.get(canonical_key("_".join(parts[:cut])))
        if hit is not None:
            return hit
    return None


def split_job_type_suffix(node_name: str, known_step_names: Iterable[str]) -> tuple:
    """Split a node into ``(base_step_name, suffix_or_None)`` using the registry (robust).

    ``('TabularPreprocessing_training', [...]) -> ('TabularPreprocessing', 'training')``;
    ``('CradleDataLoading_munged', [...]) -> ('CradleDataLoading', 'munged')``;
    ``('CradleDataLoading', [...]) -> ('CradleDataLoading', None)``. Unlike a
    ``JOB_TYPE_SUFFIXES`` check, this handles ANY suffix because the base is validated against
    the actual step registry via :func:`resolve_base_step_name`. Returns ``(node_name, None)`` if
    the node doesn't resolve to a known base.
    """
    base = resolve_base_step_name(node_name, known_step_names)
    if base is None:
        return node_name, None
    if canonical_key(base) == canonical_key(node_name):
        return base, None
    # The suffix is whatever follows the base's snake prefix in the node name.
    suffix = node_name[len(base) + 1 :] if node_name.lower().startswith(base.lower()) else None
    if not suffix:
        # base matched via canonical_key (casing/separator differ); recover the trailing segment(s)
        n_base_parts = len(canonical_to_snake(base).split("_"))
        suffix = "_".join(node_name.split("_")[n_base_parts:]) or None
    return base, suffix
