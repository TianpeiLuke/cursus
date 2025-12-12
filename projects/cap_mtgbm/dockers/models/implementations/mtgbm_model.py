"""
Multi-Task Gradient Boosting Model (MT-GBM) implementation.

Implements shared tree structure for multi-task learning.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
import json
from pathlib import Path

from ..base.base_model import BaseMultiTaskModel


class MtgbmModel(BaseMultiTaskModel):
    """
    Multi-Task Gradient Boosting Model.

    Uses LightGBM with custom multi-task loss function and
    shared tree structures across related tasks.
    """

    def _prepare_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        feature_columns: Optional[list] = None,
        task_columns: Optional[list] = None,
    ) -> Tuple[lgb.Dataset, lgb.Dataset, Optional[lgb.Dataset]]:
        """
        Prepare data for LightGBM training.

        Converts DataFrames to LightGBM Dataset format.
        Training script controls schema (WHAT columns), model handles format conversion (HOW).

        Parameters
        ----------
        train_df : DataFrame
            Training data (should contain only feature_columns + task_columns)
        val_df : DataFrame
            Validation data
        test_df : DataFrame, optional
            Test data
        feature_columns : list, optional
            List of feature column names in exact order
            If not provided, falls back to tab_field_list + cat_field_list
        task_columns : list, optional
            List of task label column names in exact order
            If not provided, infers from data

        Returns
        -------
        train_data, val_data, test_data : tuple of lgb.Dataset
            Prepared LightGBM datasets
        """
        # Use provided columns or fallback to hyperparams
        if feature_columns is None:
            feature_columns = (
                self.hyperparams.tab_field_list + self.hyperparams.cat_field_list
            )
            self.logger.warning(
                "feature_columns not provided, falling back to hyperparams. "
                "Consider passing explicit columns from training script."
            )

        if task_columns is None:
            # Use explicit task_label_names from hyperparams
            if self.hyperparams.task_label_names:
                task_columns = self.hyperparams.task_label_names
                self.logger.warning(
                    f"task_columns not provided, using hyperparams.task_label_names: {task_columns}. "
                    "Consider passing explicit columns from training script."
                )
            else:
                raise ValueError(
                    "task_columns not provided and hyperparams.task_label_names is empty. "
                    "Must specify task columns either via method parameter or hyperparameters."
                )

        # Extract features and labels (simple slicing - no schema knowledge needed!)
        X_train = train_df[feature_columns].values
        y_train = train_df[task_columns].values

        # Get main task index from hyperparameters (default to 0 if not specified)
        main_task_idx = getattr(self.hyperparams, "main_task_index", 0)

        # Create LightGBM Dataset with multi-task labels stored properly
        # Pass main task label to LightGBM (for validation compatibility)
        # Store full multi-task labels as flattened array for custom loss retrieval
        train_data = lgb.Dataset(
            X_train,
            label=y_train[
                :, main_task_idx
            ],  # Main task (configurable) for LightGBM validation
            feature_name=feature_columns,
            categorical_feature=[
                c for c in feature_columns if c in self.hyperparams.cat_field_list
            ],
        )
        # Store full multi-task labels for custom loss function
        train_data.set_field("multi_task_labels", y_train.flatten())

        # Prepare validation data
        X_val = val_df[feature_columns].values
        y_val = val_df[task_columns].values
        val_data = lgb.Dataset(
            X_val, label=y_val[:, main_task_idx], reference=train_data
        )
        val_data.set_field("multi_task_labels", y_val.flatten())

        # Prepare test data if provided
        test_data = None
        if test_df is not None:
            X_test = test_df[feature_columns].values
            y_test = test_df[task_columns].values
            test_data = lgb.Dataset(
                X_test, label=y_test[:, main_task_idx], reference=train_data
            )
            test_data.set_field("multi_task_labels", y_test.flatten())

        self.logger.info(
            f"Prepared data: train={X_train.shape}, "
            f"val={X_val.shape}, "
            f"test={X_test.shape if test_df is not None else None}"
        )
        self.logger.info(
            f"Features: {len(feature_columns)} columns, Tasks: {len(task_columns)} columns"
        )

        return train_data, val_data, test_data

    def _initialize_model(self) -> None:
        """Initialize LightGBM model parameters."""
        # Build LightGBM parameters from hyperparameters
        self.lgb_params = {
            "boosting_type": self.hyperparams.boosting_type,
            "num_leaves": self.hyperparams.num_leaves,
            "learning_rate": self.hyperparams.learning_rate,
            "max_depth": self.hyperparams.max_depth,
            "min_data_in_leaf": self.hyperparams.min_data_in_leaf,
            "feature_fraction": self.hyperparams.feature_fraction,
            "bagging_fraction": self.hyperparams.bagging_fraction,
            "bagging_freq": self.hyperparams.bagging_freq,
            "lambda_l1": self.hyperparams.lambda_l1,
            "lambda_l2": self.hyperparams.lambda_l2,
            "min_gain_to_split": self.hyperparams.min_gain_to_split,
            "verbose": -1,
        }

        if self.hyperparams.seed is not None:
            self.lgb_params["seed"] = self.hyperparams.seed

        self.logger.info(f"Initialized model with params: {self.lgb_params}")

    def _train_model(
        self, train_data: lgb.Dataset, val_data: lgb.Dataset
    ) -> Dict[str, Any]:
        """
        Train MT-GBM model with custom loss function.

        Parameters
        ----------
        train_data : lgb.Dataset
            Training data
        val_data : lgb.Dataset
            Validation data

        Returns
        -------
        metrics : dict
            Training metrics
        """
        self.logger.info("Starting LightGBM training with custom loss...")

        # Create params copy for training (LightGBM 4.x compatibility)
        train_params = self.lgb_params.copy()

        # LightGBM 4.x: Set custom objective in params dict instead of fobj parameter
        train_params["objective"] = self.loss_function.objective

        # Prepare callbacks for LightGBM 4.x
        callbacks = []
        if self.hyperparams.early_stopping_rounds:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.hyperparams.early_stopping_rounds
                )
            )
        callbacks.append(lgb.log_evaluation(period=10))

        # Train with custom loss function (LightGBM 4.x API)
        self.model = lgb.train(
            train_params,
            train_data,
            num_boost_round=self.hyperparams.num_iterations,
            valid_sets=[val_data],
            valid_names=["valid"],
            feval=self._create_eval_function(),
            callbacks=callbacks,
        )

        # Extract training metrics
        metrics = {
            "num_iterations": self.model.num_trees(),
            "best_iteration": self.model.best_iteration,
            "feature_importance": self.model.feature_importance().tolist(),
        }

        self.logger.info(f"Training completed: {metrics['num_iterations']} trees")

        return metrics

    def _create_eval_function(self):
        """Create evaluation function for LightGBM."""

        def eval_func(preds, train_data):
            """Custom evaluation function wrapper."""
            task_scores, mean_score = self.loss_function.evaluate(preds, train_data)
            return "mean_auc", mean_score, True  # (name, value, is_higher_better)

        return eval_func

    def _predict(self, data: lgb.Dataset) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        data : lgb.Dataset
            Data to predict on

        Returns
        -------
        predictions : np.ndarray
            Raw predictions [N_samples * N_tasks]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get data from Dataset
        X = data.data
        predictions = self.model.predict(X)

        return predictions

    def _save_model(self, output_path: str) -> None:
        """
        Save model artifacts.

        Parameters
        ----------
        output_path : str
            Directory to save artifacts
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model
        model_file = output_dir / "lightgbmmt_model.txt"
        self.model.save_model(str(model_file))
        self.logger.info(f"Saved model to {model_file}")

        # Save hyperparameters
        hyperparams_file = output_dir / "hyperparameters.json"
        with open(hyperparams_file, "w") as f:
            json.dump(self.hyperparams.model_dump(), f, indent=2)
        self.logger.info(f"Saved hyperparameters to {hyperparams_file}")

        # Save training state
        state_file = output_dir / "training_state.json"
        with open(state_file, "w") as f:
            json.dump(self.training_state.to_checkpoint_dict(), f, indent=2)
        self.logger.info(f"Saved training state to {state_file}")

    def _load_model(self, model_path: str) -> None:
        """
        Load model artifacts.

        Parameters
        ----------
        model_path : str
            Path to model artifacts
        """
        model_dir = Path(model_path)

        # Load LightGBM model
        model_file = model_dir / "lightgbmmt_model.txt"
        self.model = lgb.Booster(model_file=str(model_file))
        self.logger.info(f"Loaded model from {model_file}")

        # Load training state if available
        state_file = model_dir / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state_dict = json.load(f)
            self.training_state = self.training_state.from_checkpoint_dict(state_dict)
            self.logger.info(f"Loaded training state from {state_file}")
