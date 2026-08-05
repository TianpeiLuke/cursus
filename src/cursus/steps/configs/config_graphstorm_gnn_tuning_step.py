"""
GraphStorm GNN Tuning Configuration (the Tuning verb's first interface, FZ 31e1d3p / plan 31e1d3p2).

A hyperparameter-tuning step over GraphStorm GNN training. A tuning job *is* a training job with a
search wrapper, so this config is the idiomatic composition of the wrapped training config
(``GraphStormGNNTrainingConfig`` — every estimator field the byo_container factory reads) and the
search-axis mixin (``TuningStepConfigMixin`` — objective / search_space / strategy / limits /
metric_definitions). ``TuningHandler`` reads both off ``b.config`` at build time: it builds the SAME
GraphStorm estimator ``GraphStormGNNTraining`` would, then wraps it in a ``HyperparameterTuner``.

Because the estimator is a ``byo_container`` (a custom GraphStorm image, no SDK-inferred metrics),
``metric_definitions`` is mandatory — the objective is regex-scraped from the container's stdout,
exactly as the Nexus ``launch_hpo.py`` launcher does.
"""

from .config_graphstorm_gnn_training_step import GraphStormGNNTrainingConfig
from .config_tuning_step_base import TuningStepConfigMixin


class GraphStormGNNTuningConfig(GraphStormGNNTrainingConfig, TuningStepConfigMixin):
    """GraphStorm/DGL R-GCN hyperparameter tuning in a bring-your-own GraphStorm container.

    Inherits every training field (``training_image_uri``, instance type/count, ``training_mode``,
    ``num_servers`` …) from ``GraphStormGNNTrainingConfig`` and adds the search-axis fields
    (``objective_metric_name``, ``search_space``, ``tuning_strategy``, ``max_jobs``,
    ``max_parallel_jobs``, ``early_stopping_type``, ``metric_definitions``) from
    ``TuningStepConfigMixin``. The mixin's validator enforces that ``metric_definitions`` is present
    (byo_container detected via the inherited ``training_image_uri``).
    """

    pass
