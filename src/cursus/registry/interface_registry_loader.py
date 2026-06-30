"""
Interface-derived step registry loader — the SOLE source of the step-name registry.

This module derives the step-name registry table from the per-step ``.step.yaml`` interface
files (the "full Vector 3" end-state: the registry IS the interface files). As of the FZ
31e1/31e1f Final Phase (2026-06-28) the standalone ``registry/step_names.yaml`` table was
DELETED — ``step_names_base.STEP_NAMES`` is built by ``build_registry_from_interfaces()`` with
no external fallback, and a golden snapshot (``tests/registry/step_names_registry_snapshot.json``)
gates drift.

Derivation rules (per the 31e1a field spec):
  * ``spec_type``         = the canonical step name (it is ``== step_type`` for every row)
  * ``config_class``      = ``"<Name>Config"`` by convention, unless overridden
  * ``builder_step_name`` = ``"<Name>StepBuilder"`` by convention, unless overridden
  * ``sagemaker_step_type`` = irreducible; read from the ``.step.yaml`` ``registry:`` block
  * ``description``       = irreducible prose; read from the ``.step.yaml`` ``registry:`` block

Steps with no ``.step.yaml`` interface (abstract bases ``Base`` / ``Processing`` and the
builder-less ``HyperparameterPrep``) come from the small ``_EXTRAS`` map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

_INTERFACES_DIR = Path(__file__).resolve().parent.parent / "steps" / "interfaces"

# Steps that have no .step.yaml interface file — they cannot be derived from the interface
# walk and must be declared explicitly. (Verified: exactly these 3 rows in step_names.yaml
# have no matching interface file.)
_EXTRAS: Dict[str, Dict[str, str]] = {
    "Base": {
        "config_class": "BasePipelineConfig",
        "builder_step_name": "StepBuilderBase",
        "spec_type": "Base",
        "sagemaker_step_type": "Base",
        "description": "Base pipeline configuration",
    },
    "Processing": {
        "config_class": "ProcessingStepConfigBase",
        "builder_step_name": "ProcessingStepBuilder",
        "spec_type": "Processing",
        "sagemaker_step_type": "Processing",
        "description": "Base processing step",
    },
    "HyperparameterPrep": {
        "config_class": "HyperparameterPrepConfig",
        "builder_step_name": "HyperparameterPrepStepBuilder",
        "spec_type": "HyperparameterPrep",
        "sagemaker_step_type": "Lambda",
        "description": "Hyperparameter preparation step",
    },
}

# config_class values that break the "<Name>Config" convention. Now EMPTY (FZ 31e1d3g3 C3, #25):
# the 3 convention-breakers (BatchTransform / PyTorchModel / XGBoostModel) each declare their
# config_class in their own .step.yaml registry block, so the truth is authored data, not a Python
# override. The seam is retained (the loader still consults it) for any future genuine exception.
_CONFIG_CLASS_OVERRIDES: Dict[str, str] = {}


def _interface_step_type(data: dict) -> Optional[str]:
    """The canonical step name a .step.yaml declares (its ``step_type``)."""
    st = data.get("step_type")
    return st if isinstance(st, str) and st else None


def build_registry_from_interfaces(
    interfaces_dir: Optional[Path] = None,
    fallback: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Dict[str, str]]:
    """Derive the ``STEP_NAMES`` table from the ``.step.yaml`` interface files.

    Args:
        interfaces_dir: directory of ``*.step.yaml`` files (defaults to the package's
            ``steps/interfaces``).
        fallback: DEPRECATED transition-window arg (the legacy ``step_names.yaml`` table). It is
            no longer supplied by the package — every ``.step.yaml`` now carries a ``registry:``
            block, so derivation is self-sufficient. Retained only so an external caller can still
            pass a table; ``None`` (the default) uses the interface blocks + ``_EXTRAS`` alone.

    Returns:
        ``{canonical_name: {config_class, builder_step_name, spec_type, sagemaker_step_type,
        description}}`` — the same shape ``get_step_names()`` returns.
    """
    idir = interfaces_dir or _INTERFACES_DIR
    fallback = fallback or {}

    table: Dict[str, Dict[str, str]] = {
        name: dict(row) for name, row in _EXTRAS.items()
    }

    for path in sorted(idir.glob("*.step.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        name = _interface_step_type(data)
        if not name:
            continue

        registry_block = data.get("registry") or {}
        fb = fallback.get(name, {})

        sagemaker_step_type = registry_block.get("sagemaker_step_type") or fb.get(
            "sagemaker_step_type"
        )
        description = registry_block.get("description") or fb.get("description", "")
        config_class = (
            registry_block.get("config_class")
            or _CONFIG_CLASS_OVERRIDES.get(name)
            or f"{name}Config"
        )
        builder_step_name = (
            registry_block.get("builder_step_name") or f"{name}StepBuilder"
        )

        if not sagemaker_step_type:
            raise ValueError(
                f"{path.name}: cannot determine sagemaker_step_type — add a "
                f"`registry:` block with `sagemaker_step_type:` (no fallback row for {name!r})"
            )

        table[name] = {
            "config_class": config_class,
            "builder_step_name": builder_step_name,
            "spec_type": name,  # always == step_type
            "sagemaker_step_type": sagemaker_step_type,
            "description": description,
        }

    return table
