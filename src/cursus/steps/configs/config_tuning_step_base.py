"""
Hyperparameter-Tuning Step Configuration Mixin (the search axis).

The ``Tuning`` construction verb (FZ 31e1d3p) wraps a training step's estimator in a SageMaker
``HyperparameterTuner`` and builds a ``TuningStep``. A tuning job *is* a training job with a search
wrapper, so a concrete tuning config REUSES the wrapped training config's fields (image, instance
type, channels — everything the estimator factory reads) and adds ONLY the search fields this mixin
supplies. The idiomatic composition is multiple inheritance::

    class GraphStormGNNTuningConfig(GraphStormGNNTrainingConfig, TuningStepConfigMixin):
        pass

so the config carries both the estimator fields (from the training config) and the search fields
(from this mixin). ``TuningHandler`` reads these fields off ``b.config`` at build time; the
``search_space`` shape is converted to SDK ParameterRange objects by the handler's
``_build_parameter_ranges`` (the SDK analog of Nexus's launch_hpo.py ``_build_param_ranges``).
"""

from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Literal, Optional
import logging

logger = logging.getLogger(__name__)


class TuningStepConfigMixin(BaseModel):
    """Search-axis fields for a hyperparameter-tuning step (the one behavior the Tuning verb adds).

    Composed with a wrapped training config (multiple inheritance) so a tuning step's single config
    supplies both the estimator inputs and the search configuration. Every field maps directly to a
    ``sagemaker.tuner.HyperparameterTuner`` constructor argument (SDK v2.x).
    """

    # ===== Tier 1: Essential search inputs =====
    #: The metric AMT optimizes, e.g. "val:is_abuse:roc_auc". Maps to HyperparameterTuner
    #: objective_metric_name.
    objective_metric_name: str = Field(
        description="The objective metric AMT optimizes (e.g. 'val:is_abuse:roc_auc').",
    )
    #: The search space. Shape (mirrors Nexus launch_hpo.py _build_param_ranges):
    #:   {"continuous":  [{"name": ..., "min": ..., "max": ..., "scaling": "Auto|Linear|Logarithmic|ReverseLogarithmic"}],
    #:    "integer":     [{"name": ..., "min": ..., "max": ..., "scaling": ...}],
    #:    "categorical": [{"name": ..., "values": [...]}]}
    #: The handler's _build_parameter_ranges converts each to a ContinuousParameter /
    #: IntegerParameter / CategoricalParameter.
    search_space: Dict[str, List[Dict[str, Any]]] = Field(
        description="Tunable ranges by kind (continuous / integer / categorical). "
        "Names are the hyperparameter keys the container reads (e.g. dot-paths like 'gsf.hyperparam.lr').",
    )

    # ===== Tier 2: System fields (overridable defaults) =====
    objective_type: Literal["Maximize", "Minimize"] = Field(
        default="Maximize",
        description="Whether to maximize or minimize the objective metric.",
    )
    tuning_strategy: Literal["Bayesian", "Random", "Hyperband", "Grid"] = Field(
        default="Bayesian", description="Search strategy (SDK v2.x supports all four)."
    )
    max_jobs: int = Field(
        default=20, ge=1, description="MaxNumberOfTrainingJobs — total trials to run."
    )
    max_parallel_jobs: int = Field(
        default=1, ge=1, description="MaxParallelTrainingJobs — concurrent trials."
    )
    early_stopping_type: Literal["Off", "Auto"] = Field(
        default="Off", description="Enable SageMaker's built-in early stopping."
    )
    #: Regex metric definitions [{"Name": ..., "Regex": ...}] scraped from the container's stdout.
    #: REQUIRED when the wrapped compute is a byo_container (a custom image has no SDK-inferred
    #: metrics — the objective must be regex-scraped, exactly as Nexus launch_hpo.py does). Optional
    #: for a managed-DLC estimator whose metrics the SDK already knows.
    metric_definitions: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Regex metric definitions scraped from container stdout; required for byo_container.",
    )

    @model_validator(mode="after")
    def _validate_search_space(self) -> "TuningStepConfigMixin":
        space = self.search_space or {}
        allowed = {"continuous", "integer", "categorical"}
        unknown = set(space) - allowed
        if unknown:
            raise ValueError(
                f"search_space has unknown range kinds {unknown}; allowed: {sorted(allowed)}"
            )
        total = sum(len(space.get(k, []) or []) for k in allowed)
        if total == 0:
            raise ValueError(
                "search_space must declare at least one tunable range (continuous / integer / categorical)"
            )
        for p in space.get("continuous", []) or []:
            _require_keys(p, ("name", "min", "max"), "continuous")
        for p in space.get("integer", []) or []:
            _require_keys(p, ("name", "min", "max"), "integer")
        for p in space.get("categorical", []) or []:
            _require_keys(p, ("name", "values"), "categorical")

        # byo_container objective must be regex-scraped: metric_definitions is required when the
        # (composed) config's compute is a byo_container. Detected via a `compute_kind` hint the
        # wrapped training config may expose, or the presence of an image_uri/*_image_uri field.
        if self._is_byo_container() and not self.metric_definitions:
            raise ValueError(
                "metric_definitions is required for a byo_container tuning step: a custom container's "
                "objective metric must be regex-scraped from stdout (SageMaker cannot infer it). "
                "Provide [{'Name': ..., 'Regex': ...}, ...] as Nexus launch_hpo.py does."
            )
        return self

    def _is_byo_container(self) -> bool:
        """Best-effort detection that the wrapped estimator is a bring-your-own container.

        A composed tuning config inherits the training config's fields; a byo_container training
        config carries a verbatim image URI field (e.g. ``training_image_uri`` / ``image_uri``).
        The interface's ``compute.kind: byo_container`` is the authoritative signal at build time;
        this config-level check is the author-time guard that catches a missing metric_definitions
        before compilation.
        """
        for attr in ("training_image_uri", "image_uri", "inference_image_uri"):
            if getattr(self, attr, None):
                return True
        return getattr(self, "compute_kind", None) == "byo_container"


def _require_keys(param: Dict[str, Any], keys: tuple, kind: str) -> None:
    missing = [k for k in keys if k not in param]
    if missing:
        raise ValueError(
            f"{kind} search_space entry {param!r} is missing required key(s) {missing}"
        )
