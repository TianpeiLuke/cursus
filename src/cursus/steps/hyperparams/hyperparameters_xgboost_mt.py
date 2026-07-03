"""
Hyperparameters for XGBoost Multi-Task (one_output_per_tree) model training.

Extends ModelHyperparameters with XGBoost-specific and multi-task parameters.
"""

from pydantic import Field, model_validator, PrivateAttr
from typing import Optional, Literal

from ...core.base.hyperparameters_base import ModelHyperparameters


class XgboostMtModelHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for XGBoost Multi-Task model (one_output_per_tree via XGBClassifier).

    Uses XGBoost native multi-output tree where each tree produces output
    for exactly one task in round-robin fashion.
    """

    # ========================================================================
    # TIER 1: ESSENTIAL USER INPUTS
    # ========================================================================

    task_label_names: list[str] = Field(
        description="List of task/label column names for multi-task learning (REQUIRED)."
    )

    main_task_index: int = Field(
        ge=0,
        description="Index of the main task in the task list.",
    )

    # ========================================================================
    # TIER 2: SYSTEM INPUTS WITH DEFAULTS
    # ========================================================================

    model_class: str = Field(
        default="xgboost_mt",
        description="Model class identifier for XGBoost multi-task",
    )

    # XGBoost parameters
    n_estimators: int = Field(
        default=600, ge=1, description="Number of boosting rounds"
    )
    max_depth: int = Field(default=8, ge=1, description="Maximum tree depth")
    learning_rate: float = Field(
        default=0.05, description="Boosting learning rate (eta)"
    )
    subsample: float = Field(
        default=0.9, gt=0.0, le=1.0, description="Row subsampling ratio"
    )
    colsample_bytree: float = Field(
        default=0.9, gt=0.0, le=1.0, description="Column subsampling ratio"
    )
    reg_alpha: float = Field(default=0.5, ge=0.0, description="L1 regularization")
    reg_lambda: float = Field(default=0.05, ge=0.0, description="L2 regularization")
    min_child_weight: float = Field(
        default=0.1, ge=0.0, description="Minimum child weight"
    )
    early_stopping_rounds: Optional[int] = Field(
        default=10, ge=1, description="Early stopping patience"
    )
    seed: int = Field(default=17, description="Random seed")

    # ========================================================================
    # DERIVED FIELDS
    # ========================================================================

    _num_tasks: Optional[int] = PrivateAttr(default=None)

    @property
    def num_tasks(self) -> int:
        """Get number of tasks derived from task_label_names."""
        if self._num_tasks is None:
            self._num_tasks = len(self.task_label_names)
        return self._num_tasks

    @model_validator(mode="after")
    def validate_xgboost_mt_hyperparameters(self) -> "XgboostMtModelHyperparameters":
        """Validate XGBoost MT specific hyperparameters."""
        super().validate_dimensions()
        self._num_tasks = len(self.task_label_names)

        if self._num_tasks < 2:
            raise ValueError(f"num_tasks must be >= 2, got {self._num_tasks}")
        if self.main_task_index >= self._num_tasks:
            raise ValueError(
                f"main_task_index ({self.main_task_index}) must be < num_tasks ({self._num_tasks})"
            )
        return self

    def get_public_init_fields(self) -> dict:
        """Override to include XGBoost MT-specific derived fields."""
        base_fields = super().get_public_init_fields()
        return {**base_fields, "num_tasks": self.num_tasks}
