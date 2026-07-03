"""
Step Interface Loader

Loads unified .step.yaml files and emits a single validated StepInterface object.
This is the single entry point for all step interface data. The former per-step
steps/specs/ and steps/contracts/ folders are gone — the ``.step.yaml`` here is the
sole source, and its contract/spec sections are views onto one aligned object.

For backward compatibility, ``load_step_interface`` still returns a 2-tuple that
destructures as ``(contract, spec)``:

- ``contract`` is the StepInterface's ContractSection — a drop-in for the legacy
  ScriptContract/StepContract (exposes ``expected_input_paths`` etc.).
- ``spec`` is the StepInterface itself — a drop-in for the legacy StepSpecification
  (exposes ``dependencies``/``outputs``/``get_output_by_name_or_alias``/
  ``script_contract``/...).

Both elements are views onto the same validated StepInterface. Use
``load_interface(...)`` to get the StepInterface directly.

Usage:
    from cursus.steps.interfaces import load_step_interface, load_interface

    contract, spec = load_step_interface("TabularPreprocessing")
    iface = load_interface("CradleDataLoading", job_type="calibration")
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from ...core.base.step_interface import StepInterface, ContractSection

logger = logging.getLogger(__name__)

INTERFACES_DIR = Path(__file__).parent

# Cache loaded interfaces (keyed by step_name:job_type)
_cache: Dict[str, StepInterface] = {}


def _step_name_to_filename(step_name: str) -> str:
    """Convert StepName to filename: PyTorchTraining → pytorch_training"""
    import re

    # Handle known abbreviations that shouldn't be split
    replacements = {
        "PyTorch": "Pytorch",
        "XGBoost": "Xgboost",
        "LightGBMMT": "Lightgbmmt",
        "LightGBM": "Lightgbm",
    }
    s = step_name
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    return s


def _canonical_key(name: str) -> str:
    """Collapse a step name / filename stem to a separator- and case-insensitive key.

    'XGBoostTraining', 'xgboost_training', 'XGBoost_Training' all map to 'xgboosttraining'.
    Used as a robust fallback so new acronym steps resolve without editing the hardcoded
    abbreviation table in ``_step_name_to_filename``.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve_interface_path(step_name: str) -> Path:
    """Resolve a step name to its ``.step.yaml`` path.

    Tries the naming-convention filename first (fast, exact). If that file does not exist
    — e.g. a new step whose acronym the convention table doesn't know — falls back to a
    normalized scan of the interfaces directory matching on :func:`_canonical_key`. Raises
    ``FileNotFoundError`` only if neither resolves.
    """
    # 1) Convention-derived filename (the common, fast path).
    primary = INTERFACES_DIR / (_step_name_to_filename(step_name) + ".step.yaml")
    if primary.exists():
        return primary

    # 2) Normalized fallback: match the step name against every interface stem ignoring
    #    case and separators, so acronym/casing mismatches still resolve.
    want = _canonical_key(step_name)
    matches = [
        p
        for p in INTERFACES_DIR.glob("*.step.yaml")
        if _canonical_key(p.name[: -len(".step.yaml")]) == want
    ]
    if len(matches) == 1:
        logger.debug(
            "Resolved interface for '%s' via normalized fallback: %s",
            step_name,
            matches[0].name,
        )
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Ambiguous interface for '{step_name}': matched "
            f"{[m.name for m in matches]}"
        )

    raise FileNotFoundError(f"No interface file for '{step_name}': {primary}")


def load_interface(
    step_name: str,
    job_type: Optional[str] = None,
) -> StepInterface:
    """
    Load a step's unified interface from YAML as a single StepInterface.

    Args:
        step_name: PascalCase step name (e.g., "TabularPreprocessing", "CradleDataLoading")
        job_type: Optional job_type variant (e.g., "training", "calibration")

    Returns:
        Validated StepInterface (variant-resolved when job_type is given).
    """
    cache_key = f"{step_name}:{job_type or 'default'}"
    if cache_key in _cache:
        return _cache[cache_key]

    # Convention-first, with a normalized-scan fallback for acronym/casing mismatches.
    filepath = _resolve_interface_path(step_name)

    with open(filepath) as f:
        data = yaml.safe_load(f)

    # StepInterface.from_yaml resolves the job_type variant (if any) and validates.
    iface = StepInterface.from_yaml(data, job_type=job_type)

    _cache[cache_key] = iface
    return iface


def load_step_interface(
    step_name: str,
    job_type: Optional[str] = None,
) -> Tuple[ContractSection, StepInterface]:
    """
    Backward-compatible loader returning a ``(contract, spec)`` tuple.

    Both elements are views onto one validated StepInterface:
    - ``[0]`` = the ContractSection (ScriptContract/StepContract drop-in)
    - ``[1]`` = the StepInterface (StepSpecification drop-in)

    New code should prefer :func:`load_interface`.
    """
    iface = load_interface(step_name, job_type=job_type)
    return iface.contract, iface


def list_available_interfaces() -> List[str]:
    """List all available step interface names."""
    return [f.stem.replace(".step", "") for f in INTERFACES_DIR.glob("*.step.yaml")]


def clear_interface_cache() -> None:
    """Clear the in-process interface cache.

    Interfaces are cached by ``step_name:job_type`` for the life of the process. This
    helper lets tests and dev hot-reload pick up edited ``.step.yaml`` files without a
    restart (long-running processes otherwise serve stale interface data).
    """
    _cache.clear()
