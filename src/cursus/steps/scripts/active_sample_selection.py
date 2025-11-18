#!/usr/bin/env python3
"""
Active Sample Selection Script for Semi-Supervised and Active Learning.

This script implements intelligent sample selection from model predictions for:
1. Semi-Supervised Learning (SSL): High-confidence samples for pseudo-labeling
2. Active Learning (AL): Uncertain/diverse samples for human labeling

Supports multiple strategies:
- SSL: confidence_threshold, top_k_per_class
- AL: uncertainty (margin/entropy/least_confidence), diversity (k-center), BADGE

Author: Cursus Framework
Date: 2025-11-17
"""

import argparse
import glob
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# FILE I/O HELPER FUNCTIONS WITH FORMAT PRESERVATION
# ============================================================================


def _detect_file_format(file_path: "Path") -> str:
    """
    Detect the format of a data file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        Format string: 'csv', 'tsv', or 'parquet'
    """
    from pathlib import Path

    if isinstance(file_path, str):
        file_path = Path(file_path)

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return "csv"
    elif suffix == ".tsv":
        return "tsv"
    elif suffix == ".parquet":
        return "parquet"
    else:
        raise RuntimeError(f"Unsupported file format: {suffix}")


def load_dataframe_with_format(file_path: "Path") -> Tuple[pd.DataFrame, str]:
    """
    Load DataFrame and detect its format.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (DataFrame, format_string)
    """
    from pathlib import Path

    if isinstance(file_path, str):
        file_path = Path(file_path)

    detected_format = _detect_file_format(file_path)

    if detected_format == "csv":
        df = pd.read_csv(file_path)
    elif detected_format == "tsv":
        df = pd.read_csv(file_path, sep="\t")
    elif detected_format == "parquet":
        df = pd.read_parquet(file_path)
    else:
        raise RuntimeError(f"Unsupported format: {detected_format}")

    return df, detected_format


def save_dataframe_with_format(
    df: pd.DataFrame, output_path: "Path", format_str: str
) -> "Path":
    """
    Save DataFrame in specified format.

    Args:
        df: DataFrame to save
        output_path: Base output path (without extension)
        format_str: Format to save in ('csv', 'tsv', or 'parquet')

    Returns:
        Path to saved file
    """
    from pathlib import Path

    if isinstance(output_path, str):
        output_path = Path(output_path)

    if format_str == "csv":
        file_path = output_path.with_suffix(".csv")
        df.to_csv(file_path, index=False)
    elif format_str == "tsv":
        file_path = output_path.with_suffix(".tsv")
        df.to_csv(file_path, sep="\t", index=False)
    elif format_str == "parquet":
        file_path = output_path.with_suffix(".parquet")
        df.to_parquet(file_path, index=False)
    else:
        raise RuntimeError(f"Unsupported output format: {format_str}")

    return file_path


# ============================================================================
# Data Loading Component
# ============================================================================


def load_inference_data(
    inference_data_dir: str,
    id_field: str = "id",
) -> Tuple[pd.DataFrame, str]:
    """
    Load inference data from various upstream sources with format detection.

    Supports inference outputs from:
    - XGBoost/LightGBM/PyTorch model inference
    - Bedrock batch processing / Bedrock processing
    - Label ruleset execution

    Args:
        inference_data_dir: Path to inference output data
        id_field: Name of ID column

    Returns:
        Tuple of (DataFrame, format_string) where format is 'csv', 'tsv', or 'parquet'

    Raises:
        FileNotFoundError: If no data files found
        ValueError: If ID field not found
    """
    from pathlib import Path

    # Find all supported data files
    data_dir = Path(inference_data_dir)
    data_files = []
    for ext in [".csv", ".tsv", ".parquet"]:
        data_files.extend(list(data_dir.glob(f"**/*{ext}")))

    if not data_files:
        raise FileNotFoundError(
            f"No inference data files (.csv, .tsv, .parquet) found in {inference_data_dir}"
        )

    # Use first file found
    data_file = data_files[0]
    logger.info(f"Loading data file: {data_file}")

    # Load with format detection
    df, input_format = load_dataframe_with_format(data_file)
    logger.info(f"Detected input format: {input_format}")
    logger.info(f"Loaded inference data with shape {df.shape}")

    # Validate required columns
    if id_field not in df.columns:
        raise ValueError(f"ID field '{id_field}' not found in data")

    return df, input_format


def extract_score_columns(
    df: pd.DataFrame,
    score_field: Optional[str] = None,
    score_prefix: str = "prob_class_",
) -> List[str]:
    """
    Extract score columns from inference data.

    Priority:
    1. If SCORE_FIELD specified, use that single column
    2. Otherwise, use SCORE_FIELD_PREFIX to find all matching columns
    3. Fall back to auto-detection if prefix doesn't match

    Args:
        df: DataFrame with inference data
        score_field: Single score column name
        score_prefix: Prefix for finding multiple score columns

    Returns:
        List of score column names

    Raises:
        ValueError: If no valid score columns found
    """
    # Priority 1: Use explicit SCORE_FIELD
    if score_field and score_field in df.columns:
        logger.info(f"Using explicit score field: {score_field}")
        return [score_field]

    # Priority 2: Use SCORE_FIELD_PREFIX
    score_cols = [col for col in df.columns if col.startswith(score_prefix)]
    if score_cols:
        logger.info(
            f"Found {len(score_cols)} score columns with prefix '{score_prefix}'"
        )
        return score_cols

    # Priority 3: Auto-detection
    logger.info("Attempting auto-detection of score columns")

    # Check for LLM/Bedrock format
    llm_patterns = ["confidence_score", "prediction_score", "score"]
    for pattern in llm_patterns:
        matching = [col for col in df.columns if pattern in col.lower()]
        if matching:
            logger.info(f"Auto-detected score columns: {matching}")
            return matching

    # Check for ruleset format
    rule_patterns = ["rule_score", "label_confidence", "label_score"]
    for pattern in rule_patterns:
        matching = [col for col in df.columns if pattern in col.lower()]
        if matching:
            logger.info(f"Auto-detected score columns: {matching}")
            return matching

    raise ValueError(
        f"No valid score columns found. Tried SCORE_FIELD='{score_field}', "
        f"SCORE_FIELD_PREFIX='{score_prefix}', and auto-detection"
    )


def normalize_scores_to_probabilities(
    df: pd.DataFrame,
    score_cols: List[str],
) -> pd.DataFrame:
    """
    Normalize various score formats to probability distributions.

    Args:
        df: DataFrame with score columns
        score_cols: List of score column names

    Returns:
        DataFrame with normalized prob_class_* columns
    """
    df_norm = df.copy()

    # Check if already in probability format
    if all(col.startswith("prob_class_") for col in score_cols):
        return df_norm

    # Extract scores
    scores = df[score_cols].values

    # Check if already normalized
    row_sums = scores.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.01):
        # Apply softmax normalization
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        logger.info("Applied softmax normalization to scores")

    # Create standardized prob_class_* columns
    for i in range(scores.shape[1]):
        df_norm[f"prob_class_{i}"] = scores[:, i]

    return df_norm


# ============================================================================
# Sampling Strategy Component
# ============================================================================


class ConfidenceThresholdSampler:
    """Simple confidence-based selection for SSL pipelines."""

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        max_samples: int = 0,
        random_seed: int = 42,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed

    def select_batch(
        self,
        probabilities: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select samples where max probability exceeds threshold.

        Returns:
            Tuple of (selected_indices, confidence_scores)
        """
        if indices is None:
            indices = np.arange(len(probabilities))

        # Calculate max probability for each sample
        max_probs = np.max(probabilities, axis=1)

        # Select high-confidence samples
        high_conf_mask = max_probs >= self.confidence_threshold
        selected_indices = indices[high_conf_mask]
        selected_scores = max_probs[high_conf_mask]

        # Limit sample count if specified
        if self.max_samples > 0 and len(selected_indices) > self.max_samples:
            top_k_idx = np.argsort(selected_scores)[-self.max_samples :][::-1]
            selected_indices = selected_indices[top_k_idx]
            selected_scores = selected_scores[top_k_idx]

        return selected_indices, selected_scores


class TopKPerClassSampler:
    """Balanced selection ensuring representation across classes."""

    def __init__(self, k_per_class: int = 100, random_seed: int = 42):
        self.k_per_class = k_per_class
        self.random_seed = random_seed

    def select_batch(
        self,
        probabilities: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select top-k most confident samples per predicted class."""
        if indices is None:
            indices = np.arange(len(probabilities))

        max_probs = np.max(probabilities, axis=1)
        pred_labels = np.argmax(probabilities, axis=1)

        selected_idx_list = []
        selected_scores_list = []

        for class_idx in range(probabilities.shape[1]):
            class_mask = pred_labels == class_idx
            class_indices = indices[class_mask]
            class_probs = max_probs[class_mask]

            if len(class_indices) == 0:
                continue

            k = min(self.k_per_class, len(class_indices))
            top_k_idx = np.argsort(class_probs)[-k:][::-1]
            selected_idx_list.extend(class_indices[top_k_idx])
            selected_scores_list.extend(class_probs[top_k_idx])

        return np.array(selected_idx_list), np.array(selected_scores_list)


class UncertaintySampler:
    """Uncertainty-based sampling strategies."""

    def __init__(self, strategy: str = "margin", random_seed: int = 42):
        self.strategy = strategy
        self.random_seed = random_seed

    def compute_scores(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute uncertainty scores (higher = more uncertain)."""
        if self.strategy == "margin":
            sorted_probs = np.sort(probabilities, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            scores = -margin
        elif self.strategy == "entropy":
            eps = 1e-10
            scores = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
        elif self.strategy == "least_confidence":
            scores = 1 - np.max(probabilities, axis=1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return scores

    def select_batch(
        self,
        probabilities: np.ndarray,
        batch_size: int,
        indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select batch of most uncertain samples."""
        scores = self.compute_scores(probabilities)

        if indices is None:
            indices = np.arange(len(scores))

        batch_size = min(batch_size, len(indices))
        top_k_idx = np.argsort(scores)[-batch_size:][::-1]
        return indices[top_k_idx], scores[top_k_idx]


class DiversitySampler:
    """Core-set diversity sampling using k-center algorithm."""

    def __init__(self, metric: str = "euclidean", random_seed: int = 42):
        self.metric = metric
        self.random_seed = random_seed

    def select_batch(
        self,
        embeddings: np.ndarray,
        batch_size: int,
        indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select batch using farthest-first algorithm."""
        if indices is None:
            indices = np.arange(len(embeddings))

        batch_size = min(batch_size, len(embeddings))
        selected = []

        # Initialize with random point
        np.random.seed(self.random_seed)
        first_idx = np.random.randint(len(embeddings))
        selected.append(first_idx)

        # Compute initial distances
        min_distances = self._compute_distances(
            embeddings, embeddings[first_idx : first_idx + 1]
        ).flatten()

        # Iteratively select farthest points
        for _ in range(batch_size - 1):
            farthest_idx = np.argmax(min_distances)
            selected.append(farthest_idx)

            new_distances = self._compute_distances(
                embeddings, embeddings[farthest_idx : farthest_idx + 1]
            ).flatten()
            min_distances = np.minimum(min_distances, new_distances)

        selected_array = np.array(selected)
        scores = np.ones(len(selected))  # Diversity doesn't have per-sample scores

        return indices[selected_array], scores

    def _compute_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute pairwise distances."""
        if self.metric == "euclidean":
            return np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))
        elif self.metric == "cosine":
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
            Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
            return 1 - np.dot(X_norm, Y_norm.T)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")


class BADGESampler:
    """BADGE: Batch Active learning by Diverse Gradient Embeddings."""

    def __init__(self, metric: str = "euclidean", random_seed: int = 42):
        self.metric = metric
        self.random_seed = random_seed
        self.diversity_sampler = DiversitySampler(metric, random_seed)

    def compute_gradient_embeddings(
        self,
        features: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """Compute gradient embeddings for BADGE."""
        pseudo_labels = np.argmax(probabilities, axis=1)
        n_samples = len(probabilities)
        n_classes = probabilities.shape[1]

        one_hot = np.eye(n_classes)[pseudo_labels]
        delta = probabilities - one_hot

        gradient_embeddings = (delta[:, :, None] * features[:, None, :]).reshape(
            n_samples, -1
        )

        return gradient_embeddings

    def select_batch(
        self,
        features: np.ndarray,
        probabilities: np.ndarray,
        batch_size: int,
        indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select batch using BADGE algorithm."""
        gradient_embeddings = self.compute_gradient_embeddings(features, probabilities)
        return self.diversity_sampler.select_batch(
            gradient_embeddings, batch_size, indices
        )


# ============================================================================
# Selection Engine Component
# ============================================================================


def select_samples(
    df: pd.DataFrame,
    strategy: str,
    batch_size: int,
    strategy_config: Dict[str, Any],
    id_field: str = "id",
) -> pd.DataFrame:
    """
    Main selection function coordinating all sampling strategies.

    Args:
        df: DataFrame with processed data and predictions
        strategy: Sampling strategy name
        batch_size: Number of samples to select
        strategy_config: Strategy-specific configuration
        id_field: Name of ID column

    Returns:
        DataFrame with selected samples including selection metadata
    """
    # Extract probability columns
    prob_cols = [col for col in df.columns if col.startswith("prob_class_")]
    if not prob_cols:
        raise ValueError("No prob_class_* columns found in data")

    probabilities = df[prob_cols].values
    indices = np.arange(len(df))

    # Apply sampling strategy
    if strategy == "confidence_threshold":
        sampler = ConfidenceThresholdSampler(
            confidence_threshold=strategy_config.get("confidence_threshold", 0.9),
            max_samples=strategy_config.get("max_samples", 0),
            random_seed=strategy_config.get("random_seed", 42),
        )
        selected_idx, scores = sampler.select_batch(probabilities, indices)

    elif strategy == "top_k_per_class":
        sampler = TopKPerClassSampler(
            k_per_class=strategy_config.get("k_per_class", 100),
            random_seed=strategy_config.get("random_seed", 42),
        )
        selected_idx, scores = sampler.select_batch(probabilities, indices)

    elif strategy == "uncertainty":
        sampler = UncertaintySampler(
            strategy=strategy_config.get("uncertainty_mode", "margin"),
            random_seed=strategy_config.get("random_seed", 42),
        )
        selected_idx, scores = sampler.select_batch(probabilities, batch_size, indices)

    elif strategy == "diversity":
        # Extract embeddings or features
        emb_cols = [col for col in df.columns if col.startswith("emb_")]
        if emb_cols:
            embeddings = df[emb_cols].values
        else:
            feature_cols = strategy_config.get("feature_columns", [])
            if not feature_cols:
                raise ValueError(
                    "No embeddings or feature columns found for diversity sampling"
                )
            embeddings = df[feature_cols].values

        sampler = DiversitySampler(
            metric=strategy_config.get("metric", "euclidean"),
            random_seed=strategy_config.get("random_seed", 42),
        )
        selected_idx, scores = sampler.select_batch(embeddings, batch_size, indices)

    elif strategy == "badge":
        # Extract features
        emb_cols = [col for col in df.columns if col.startswith("emb_")]
        if emb_cols:
            features = df[emb_cols].values
        else:
            feature_cols = strategy_config.get("feature_columns", [])
            if not feature_cols:
                raise ValueError(
                    "No embeddings or feature columns found for BADGE sampling"
                )
            features = df[feature_cols].values

        sampler = BADGESampler(
            metric=strategy_config.get("metric", "euclidean"),
            random_seed=strategy_config.get("random_seed", 42),
        )
        selected_idx, scores = sampler.select_batch(
            features, probabilities, batch_size, indices
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create output dataframe
    selected_df = df.iloc[selected_idx].copy()
    selected_df["selection_score"] = scores
    selected_df["selection_rank"] = np.arange(1, len(selected_idx) + 1)

    logger.info(f"Selected {len(selected_df)} samples using {strategy} strategy")
    return selected_df


# ============================================================================
# Output Management Component
# ============================================================================


def save_selected_samples(
    selected_df: pd.DataFrame,
    output_dir: str,
    output_format: str = "csv",
) -> str:
    """
    Save selected samples using format preservation.

    Args:
        selected_df: DataFrame with selected samples
        output_dir: Output directory path
        output_format: Format to save in ('csv', 'tsv', or 'parquet')

    Returns:
        Path to saved file
    """
    from pathlib import Path

    os.makedirs(output_dir, exist_ok=True)

    # Use format-preserving save function
    output_base = Path(output_dir) / "selected_samples"
    output_path = save_dataframe_with_format(selected_df, output_base, output_format)

    logger.info(f"Saved selected samples (format={output_format}): {output_path}")
    return str(output_path)


def save_selection_metadata(
    metadata: Dict[str, Any],
    metadata_dir: str,
) -> str:
    """
    Save selection metadata to separate output channel.

    Args:
        metadata: Metadata dictionary
        metadata_dir: Metadata output directory path

    Returns:
        Path to saved metadata file
    """
    os.makedirs(metadata_dir, exist_ok=True)

    metadata_path = os.path.join(metadata_dir, "selection_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")
    return metadata_path


# ============================================================================
# Use Case Validation
# ============================================================================


def validate_strategy_for_use_case(strategy: str, use_case: str) -> None:
    """Validate strategy compatibility with use case."""
    SSL_STRATEGIES = {"confidence_threshold", "top_k_per_class"}
    ACTIVE_LEARNING_STRATEGIES = {"uncertainty", "diversity", "badge"}

    if use_case == "auto":
        return

    if use_case == "ssl":
        if strategy not in SSL_STRATEGIES:
            raise ValueError(
                f"Strategy '{strategy}' is NOT valid for SSL! "
                f"SSL requires confidence-based strategies: {SSL_STRATEGIES}. "
                f"Strategy '{strategy}' selects UNCERTAIN samples, which create "
                f"noisy pseudo-labels and degrade model performance."
            )

    elif use_case == "active_learning":
        if strategy not in ACTIVE_LEARNING_STRATEGIES:
            raise ValueError(
                f"Strategy '{strategy}' is NOT recommended for Active Learning! "
                f"Active Learning uses: {ACTIVE_LEARNING_STRATEGIES}. "
                f"Strategy '{strategy}' selects CONFIDENT samples, which wastes "
                f"human labeling effort on easy samples."
            )


# ============================================================================
# Main Function
# ============================================================================


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main function for active sample selection.

    Args:
        input_paths: Input data paths
        output_paths: Output data paths
        environ_vars: Environment variables
        job_args: Command-line arguments
    """
    logger.info("=" * 80)
    logger.info("Active Sample Selection - Starting")
    logger.info("=" * 80)

    # Extract configuration
    id_field = environ_vars.get("ID_FIELD", "id")
    strategy = environ_vars.get("SELECTION_STRATEGY", "confidence_threshold")
    use_case = environ_vars.get("USE_CASE", "auto")
    batch_size = int(environ_vars.get("BATCH_SIZE", "32"))
    output_format = environ_vars.get("OUTPUT_FORMAT", "csv")

    logger.info(f"Configuration:")
    logger.info(f"  Strategy: {strategy}")
    logger.info(f"  Use Case: {use_case}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  ID Field: {id_field}")
    logger.info(f"  Output Format: {output_format}")

    # Validate strategy for use case
    validate_strategy_for_use_case(strategy, use_case)

    # Build strategy configuration
    strategy_config = {
        "random_seed": int(environ_vars.get("RANDOM_SEED", "42")),
    }

    # Add SSL-specific parameters
    if strategy in ["confidence_threshold", "top_k_per_class"]:
        strategy_config["confidence_threshold"] = float(
            environ_vars.get("CONFIDENCE_THRESHOLD", "0.9")
        )
        strategy_config["k_per_class"] = int(environ_vars.get("K_PER_CLASS", "100"))
        strategy_config["max_samples"] = int(environ_vars.get("MAX_SAMPLES", "0"))

    # Add Active Learning-specific parameters
    if strategy in ["uncertainty", "diversity", "badge"]:
        strategy_config["uncertainty_mode"] = environ_vars.get(
            "UNCERTAINTY_MODE", "margin"
        )
        strategy_config["metric"] = environ_vars.get("METRIC", "euclidean")

    # Load inference data with format detection
    logger.info(f"Loading inference data from {input_paths['evaluation_data']}")
    df, input_format = load_inference_data(input_paths["evaluation_data"], id_field)
    logger.info(f"Loaded {len(df)} samples, detected format: {input_format}")

    # Extract score columns
    score_field = environ_vars.get("SCORE_FIELD")
    score_prefix = environ_vars.get("SCORE_FIELD_PREFIX", "prob_class_")

    score_cols = extract_score_columns(df, score_field, score_prefix)
    logger.info(f"Using {len(score_cols)} score columns: {score_cols}")

    # Normalize scores to probabilities
    if not all(col.startswith("prob_class_") for col in score_cols):
        logger.info("Normalizing scores to probabilities")
        df = normalize_scores_to_probabilities(df, score_cols)

    # Detect feature columns
    prob_cols = [col for col in df.columns if col.startswith("prob_class_")]
    emb_cols = [col for col in df.columns if col.startswith("emb_")]
    feature_cols = [
        col
        for col in df.columns
        if col not in [id_field] + prob_cols + emb_cols
        and df[col].dtype in [np.float32, np.float64, np.int32, np.int64]
    ]

    if feature_cols:
        strategy_config["feature_columns"] = feature_cols
        logger.info(f"Detected {len(feature_cols)} feature columns")

    # Select samples
    logger.info(f"Selecting samples using {strategy} strategy")
    selected_df = select_samples(
        df=df,
        strategy=strategy,
        batch_size=batch_size,
        strategy_config=strategy_config,
        id_field=id_field,
    )

    # Prepare metadata
    selection_metadata = {
        "strategy": strategy,
        "use_case": use_case,
        "batch_size": batch_size,
        "total_pool_size": len(df),
        "selected_count": len(selected_df),
        "strategy_config": {k: str(v) for k, v in strategy_config.items()},
        "timestamp": datetime.now().isoformat(),
        "job_type": job_args.job_type,
    }

    # Determine final output format - use "csv" as sentinel for format preservation
    # If OUTPUT_FORMAT is default "csv", use input format (format preservation)
    # If OUTPUT_FORMAT is explicitly set to something else, use that (override)
    final_format = output_format if output_format != "csv" else input_format
    logger.info(
        f"Output format: {final_format} (input: {input_format}, OUTPUT_FORMAT: {output_format})"
    )

    # Save results
    logger.info(f"Saving selected samples to {output_paths['selected_samples']}")
    output_path = save_selected_samples(
        selected_df=selected_df,
        output_dir=output_paths["selected_samples"],
        output_format=final_format,
    )

    # Save metadata to separate channel
    logger.info(f"Saving metadata to {output_paths['selection_metadata']}")
    metadata_path = save_selection_metadata(
        metadata=selection_metadata,
        metadata_dir=output_paths["selection_metadata"],
    )

    logger.info("=" * 80)
    logger.info(f"Active Sample Selection - Complete")
    logger.info(f"Selected {len(selected_df)} samples out of {len(df)}")
    logger.info(f"Samples saved to: {output_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("=" * 80)


# ============================================================================
# Entry Point
# ============================================================================

# Container path constants - aligned with script contract
CONTAINER_PATHS = {
    "EVALUATION_DATA_DIR": "/opt/ml/processing/input/evaluation_data",
    "SELECTED_SAMPLES_DIR": "/opt/ml/processing/output/selected_samples",
    "SELECTION_METADATA_DIR": "/opt/ml/processing/output/selection_metadata",
}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Active sample selection")
    parser.add_argument(
        "--job_type",
        type=str,
        required=True,
        help="Type of sampling job (e.g., ssl_selection, active_learning_selection)",
    )
    args = parser.parse_args()

    # Set up paths using contract-defined paths
    input_paths = {
        "evaluation_data": CONTAINER_PATHS["EVALUATION_DATA_DIR"],
    }

    output_paths = {
        "selected_samples": CONTAINER_PATHS["SELECTED_SAMPLES_DIR"],
        "selection_metadata": CONTAINER_PATHS["SELECTION_METADATA_DIR"],
    }

    # Collect environment variables
    environ_vars = {
        "ID_FIELD": os.environ.get("ID_FIELD", "id"),
        "SELECTION_STRATEGY": os.environ.get(
            "SELECTION_STRATEGY", "confidence_threshold"
        ),
        "USE_CASE": os.environ.get("USE_CASE", "auto"),
        "BATCH_SIZE": os.environ.get("BATCH_SIZE", "32"),
        "OUTPUT_FORMAT": os.environ.get("OUTPUT_FORMAT", "csv"),
        "CONFIDENCE_THRESHOLD": os.environ.get("CONFIDENCE_THRESHOLD", "0.9"),
        "K_PER_CLASS": os.environ.get("K_PER_CLASS", "100"),
        "MAX_SAMPLES": os.environ.get("MAX_SAMPLES", "0"),
        "UNCERTAINTY_MODE": os.environ.get("UNCERTAINTY_MODE", "margin"),
        "METRIC": os.environ.get("METRIC", "euclidean"),
        "RANDOM_SEED": os.environ.get("RANDOM_SEED", "42"),
        "SCORE_FIELD": os.environ.get("SCORE_FIELD"),
        "SCORE_FIELD_PREFIX": os.environ.get("SCORE_FIELD_PREFIX", "prob_class_"),
    }

    try:
        # Ensure output directory exists
        os.makedirs(output_paths["selected_samples"], exist_ok=True)

        # Call main function
        main(input_paths, output_paths, environ_vars, args)

        # Signal success
        success_path = os.path.join(output_paths["selected_samples"], "_SUCCESS")
        Path(success_path).touch()
        logger.info(f"Created success marker: {success_path}")

        sys.exit(0)
    except Exception as e:
        # Log error and create failure marker
        logger.exception(f"Script failed with error: {e}")
        failure_path = os.path.join(
            output_paths.get("selected_samples", "/tmp"), "_FAILURE"
        )
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(f"Error: {str(e)}")
        sys.exit(1)
