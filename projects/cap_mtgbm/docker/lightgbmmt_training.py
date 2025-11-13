#!/usr/bin/env python3
"""
LightGBMMT Multi-Task Training Script

Integrates refactored loss functions and model architecture for
multi-task gradient boosting training.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import numpy as np
import pandas as pd


from models.loss.loss_factory import LossFactory
from models.factory.model_factory import ModelFactory
from models.base.training_state import TrainingState
from hyperparams.hyperparameters_lightgbmmt import LightGBMMtModelHyperparameters


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_multi_label_data(
    input_path: str, hyperparams: LightGBMMtModelHyperparameters
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load multi-label training data.

    Expected structure:
    - input_path/train/*.csv
    - input_path/val/*.csv
    - input_path/test/*.csv (optional)

    Parameters
    ----------
    input_path : str
        Base path to input data
    hyperparams : LightGBMMtModelHyperparameters
        Model hyperparameters

    Returns
    -------
    train_df, val_df, test_df : tuple of DataFrames
        DataFrames with multi-label targets
    """
    logger.info(f"Loading data from {input_path}")

    # Load datasets
    train_path = os.path.join(input_path, "train", "data.csv")
    val_path = os.path.join(input_path, "val", "data.csv")
    test_path = os.path.join(input_path, "test", "data.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found at {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    test_df = None
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        logger.info(
            f"Loaded train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}"
        )
    else:
        logger.info(f"Loaded train: {train_df.shape}, val: {val_df.shape}, test: None")

    return train_df, val_df, test_df


def identify_task_columns(df: pd.DataFrame, num_tasks: Optional[int] = None) -> list:
    """
    Identify task label columns from dataframe.

    Parameters
    ----------
    df : DataFrame
        Data with task labels
    num_tasks : int, optional
        Expected number of tasks

    Returns
    -------
    task_columns : list
        List of task column names
    """
    # Strategy 1: Look for columns starting with 'task_'
    task_cols = [col for col in df.columns if col.startswith("task_")]

    if not task_cols:
        # Strategy 2: Look for columns with 'label' in name
        task_cols = [
            col for col in df.columns if "label" in col.lower() and col != "label"
        ]

    if not task_cols:
        # Strategy 3: Look for common fraud task patterns
        fraud_patterns = [
            "isFraud",
            "isCCfrd",
            "isDDfrd",
            "isGCfrd",
            "isLOCfrd",
            "isCimfrd",
        ]
        task_cols = [col for col in df.columns if col in fraud_patterns]

    if not task_cols:
        raise ValueError(
            "Could not identify task columns. Expected columns starting with 'task_' "
            "or common fraud patterns."
        )

    # Validate number of tasks if provided
    if num_tasks is not None and len(task_cols) != num_tasks:
        logger.warning(
            f"Found {len(task_cols)} task columns but num_tasks={num_tasks}. "
            f"Using found columns: {task_cols}"
        )

    logger.info(f"Identified {len(task_cols)} task columns: {task_cols}")
    return task_cols


def create_task_indices(
    train_df: pd.DataFrame, val_df: pd.DataFrame, task_columns: list
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Create task-specific indices for main task and subtasks.

    Parameters
    ----------
    train_df : DataFrame
        Training data with task labels
    val_df : DataFrame
        Validation data with task labels
    task_columns : list
        List of task column names

    Returns
    -------
    trn_sublabel_idx : dict
        Training indices for each task {task_id: np.ndarray}
    val_sublabel_idx : dict
        Validation indices for each task {task_id: np.ndarray}
    """
    num_tasks = len(task_columns)

    trn_sublabel_idx = {}
    val_sublabel_idx = {}

    for i, task_col in enumerate(task_columns):
        # Get indices where task label is positive (value == 1)
        trn_sublabel_idx[i] = np.where(train_df[task_col] == 1)[0]
        val_sublabel_idx[i] = np.where(val_df[task_col] == 1)[0]

    logger.info(f"Created indices for {num_tasks} tasks:")
    for i in range(num_tasks):
        logger.info(
            f"  Task {i} ({task_columns[i]}): "
            f"train_pos={len(trn_sublabel_idx[i])}, "
            f"val_pos={len(val_sublabel_idx[i])}"
        )

    return trn_sublabel_idx, val_sublabel_idx


def save_results(results: dict, model_output: str, evaluation_output: str) -> None:
    """
    Save training results, metrics, and visualizations.

    Parameters
    ----------
    results : dict
        Training results dictionary
    model_output : str
        Path to save model artifacts
    evaluation_output : str
        Path to save evaluation results
    """
    # Create output directories
    os.makedirs(evaluation_output, exist_ok=True)

    # Save evaluation metrics
    metrics_file = os.path.join(evaluation_output, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(results.get("evaluation_metrics", {}), f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")

    # Save training summary
    summary_file = os.path.join(evaluation_output, "training_summary.json")
    summary = {
        "model_type": results.get("model_type"),
        "training_metrics": results.get("training_metrics", {}),
        "best_epoch": results.get("training_state", {}).get("best_epoch"),
        "best_metric": results.get("training_state", {}).get("best_metric"),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved training summary to {summary_file}")


def train(
    hyperparams: LightGBMMtModelHyperparameters,
    input_path: str,
    model_output: str,
    evaluation_output: str,
) -> dict:
    """
    Main training function integrating refactored components.

    Parameters
    ----------
    hyperparams : LightGBMMtModelHyperparameters
        Complete hyperparameter configuration
    input_path : str
        S3 or local path to input data
    model_output : str
        Path to save model artifacts
    evaluation_output : str
        Path to save evaluation results

    Returns
    -------
    results : dict
        Training results with metrics and paths
    """
    logger.info("=" * 80)
    logger.info("Starting LightGBMMT Training")
    logger.info("=" * 80)

    # 1. Load multi-label data
    logger.info("\n[Step 1/7] Loading multi-label data...")
    train_df, val_df, test_df = load_multi_label_data(input_path, hyperparams)

    # 2. Identify task columns
    logger.info("\n[Step 2/7] Identifying task columns...")
    task_columns = identify_task_columns(train_df, hyperparams.num_tasks)

    # Update num_tasks in hyperparams if not set
    if hyperparams.num_tasks is None:
        hyperparams.num_tasks = len(task_columns)
        logger.info(f"Set num_tasks to {hyperparams.num_tasks}")

    # 3. Create task indices
    logger.info("\n[Step 3/7] Creating task-specific indices...")
    trn_sublabel_idx, val_sublabel_idx = create_task_indices(
        train_df, val_df, task_columns
    )

    # 4. Create loss function via LossFactory
    logger.info("\n[Step 4/7] Initializing loss function...")
    logger.info(f"Loss type: {hyperparams.loss_type}")
    logger.info(f"Main task weight: {hyperparams.loss_main_task_weight}")
    logger.info(f"Beta: {hyperparams.loss_beta}")

    loss_fn = LossFactory.create(
        loss_type=hyperparams.loss_type,
        num_label=len(task_columns),
        val_sublabel_idx=val_sublabel_idx,
        trn_sublabel_idx=trn_sublabel_idx,
        hyperparams=hyperparams,
    )

    # 5. Create training state for runtime tracking
    logger.info("\n[Step 5/7] Initializing training state...")
    training_state = TrainingState()

    # 6. Create model via ModelFactory
    logger.info("\n[Step 6/7] Creating model...")
    logger.info(f"Model type: mtgbm")
    logger.info(f"Num iterations: {hyperparams.num_iterations}")
    logger.info(f"Learning rate: {hyperparams.learning_rate}")
    logger.info(f"Num leaves: {hyperparams.num_leaves}")

    model = ModelFactory.create(
        model_type="mtgbm",
        loss_function=loss_fn,
        training_state=training_state,
        hyperparams=hyperparams,
    )

    # 7. Train model
    logger.info("\n[Step 7/7] Training model...")
    logger.info("=" * 80)
    results = model.train(train_df, val_df, test_df)
    logger.info("=" * 80)

    # 8. Save model and results
    logger.info("\nSaving artifacts...")
    model.save(model_output)
    save_results(results, model_output, evaluation_output)

    logger.info("\n" + "=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)

    return results


def main():
    """Entry point for SageMaker training."""
    logger.info("Starting SageMaker training job...")

    # SageMaker paths
    input_path = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data")
    model_output = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    evaluation_output = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    logger.info(f"Input path: {input_path}")
    logger.info(f"Model output: {model_output}")
    logger.info(f"Evaluation output: {evaluation_output}")

    # Load hyperparameters
    hyperparams_path = os.path.join(
        os.path.dirname(__file__), "hyperparams", "hyperparameters.json"
    )

    if os.path.exists(hyperparams_path):
        logger.info(f"Loading hyperparameters from {hyperparams_path}")
        with open(hyperparams_path) as f:
            hyperparams_dict = json.load(f)
        hyperparams = LightGBMMtModelHyperparameters(**hyperparams_dict)
    else:
        raise FileNotFoundError(f"Hyperparameters not found at {hyperparams_path}")

    # Train
    train(hyperparams, input_path, model_output, evaluation_output)


def test_mode():
    """Testability main for local development."""
    logger.info("=" * 80)
    logger.info("RUNNING IN TEST MODE")
    logger.info("=" * 80)

    # Create test hyperparameters
    test_hyperparams = LightGBMMtModelHyperparameters(
        # Essential fields from ModelHyperparameters
        full_field_list=["feature_1", "feature_2", "feature_3"],
        cat_field_list=["feature_1"],
        tab_field_list=["feature_2", "feature_3"],
        id_name="id",
        label_name="label",
        multiclass_categories=[0, 1],
        # LightGBM parameters
        num_leaves=31,
        learning_rate=0.1,
        num_iterations=10,  # Small for testing
        max_depth=5,
        # Multi-task parameters
        num_tasks=6,
        main_task_index=0,
        loss_type="adaptive",
        # Loss parameters
        loss_beta=0.2,
        loss_main_task_weight=1.0,
        loss_weight_lr=0.1,
        loss_patience=5,
    )

    logger.info("\nTest hyperparameters:")
    logger.info(f"  num_tasks: {test_hyperparams.num_tasks}")
    logger.info(f"  loss_type: {test_hyperparams.loss_type}")
    logger.info(f"  num_iterations: {test_hyperparams.num_iterations}")

    # Create test data directory structure
    test_data_dir = "./test_data"
    os.makedirs(f"{test_data_dir}/train", exist_ok=True)
    os.makedirs(f"{test_data_dir}/val", exist_ok=True)
    os.makedirs(f"{test_data_dir}/test", exist_ok=True)

    # Generate dummy multi-task data
    logger.info("\nGenerating dummy multi-task data...")
    n_samples = 100
    n_features = 3
    n_tasks = 6

    # Generate features
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    # Generate multi-task labels (binary)
    y_tasks = np.random.randint(0, 2, size=(n_samples, n_tasks))

    # Create DataFrames
    for split, size in [("train", 70), ("val", 20), ("test", 10)]:
        start_idx = {"train": 0, "val": 70, "test": 90}[split]
        end_idx = start_idx + size

        df_data = {
            "id": range(start_idx, end_idx),
            "feature_1": X[start_idx:end_idx, 0],
            "feature_2": X[start_idx:end_idx, 1],
            "feature_3": X[start_idx:end_idx, 2],
        }

        # Add task labels
        task_names = [
            "isFraud",
            "isCCfrd",
            "isDDfrd",
            "isGCfrd",
            "isLOCfrd",
            "isCimfrd",
        ]
        for i, task_name in enumerate(task_names):
            df_data[task_name] = y_tasks[start_idx:end_idx, i]

        df = pd.DataFrame(df_data)
        output_path = f"{test_data_dir}/{split}/data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"  Created {output_path}: {df.shape}")

    # Run training with test paths
    logger.info("\n" + "=" * 80)
    logger.info("Starting test training...")
    logger.info("=" * 80 + "\n")

    try:
        results = train(
            hyperparams=test_hyperparams,
            input_path=test_data_dir,
            model_output="./test_model",
            evaluation_output="./test_eval",
        )

        logger.info("\n" + "=" * 80)
        logger.info("TEST MODE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nResults:")
        logger.info(f"  Model type: {results.get('model_type')}")
        logger.info(
            f"  Best metric: {results.get('training_state', {}).get('best_metric')}"
        )

    except Exception as e:
        logger.error(f"\nTEST MODE FAILED: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LightGBMMT Multi-Task Training")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with dummy data for local development",
    )
    args = parser.parse_args()

    if args.test_mode:
        test_mode()
    else:
        main()
