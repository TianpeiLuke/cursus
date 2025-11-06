#!/usr/bin/env python
"""
Temporal Feature Engineering Script

This script extracts comprehensive temporal features from normalized sequence data,
combining generic temporal features with time window aggregations. Designed to consume
the output from temporal_sequence_normalization and produce rich temporal features
for machine learning models.

Supports configurable feature types, time windows, and processing strategies.
"""

import os
import json
import argparse
import logging
import sys
import traceback
import tempfile
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List, Tuple, Union
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from collections import Counter
from scipy import stats

# Suppress pandas warnings for cleaner output
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# --- Default Configuration Values ---
# These will be overridden by environment variables passed via environ_vars

DEFAULT_SEQUENCE_GROUPING_FIELD = "customerId"
DEFAULT_TIMESTAMP_FIELD = "orderDate"
DEFAULT_VALUE_FIELDS = ["transactionAmount", "merchantRiskScore"]
DEFAULT_CATEGORICAL_FIELDS = ["merchantCategory", "paymentMethod"]
DEFAULT_FEATURE_TYPES = ["statistical", "temporal", "behavioral"]
DEFAULT_WINDOW_SIZES = [7, 14, 30, 90]
DEFAULT_AGGREGATION_FUNCTIONS = ["mean", "sum", "std", "min", "max", "count"]
DEFAULT_LAG_FEATURES = [1, 7, 14, 30]
DEFAULT_EXPONENTIAL_SMOOTHING_ALPHA = 0.3
DEFAULT_TIME_UNIT = "days"
DEFAULT_INPUT_FORMAT = "numpy"
DEFAULT_OUTPUT_FORMAT = "numpy"
DEFAULT_ENABLE_DISTRIBUTED_PROCESSING = False
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_MAX_WORKERS = "auto"
DEFAULT_FEATURE_PARALLELISM = True
DEFAULT_CACHE_INTERMEDIATE = True
DEFAULT_ENABLE_VALIDATION = True
DEFAULT_MISSING_VALUE_THRESHOLD = 0.95
DEFAULT_CORRELATION_THRESHOLD = 0.99
DEFAULT_VARIANCE_THRESHOLD = 0.01
DEFAULT_OUTLIER_DETECTION = True
DEFAULT_OUTPUT_PREFIX_GENERIC = "generic_"
DEFAULT_OUTPUT_PREFIX_WINDOW = "window_"

# --- Input Data Loading Functions ---


def load_normalized_sequences(
    input_dir: str, input_format: str = "numpy", logger: Optional[Callable] = None
) -> Dict[str, np.ndarray]:
    """
    Load normalized sequences from TemporalSequenceNormalization output.

    Args:
        input_dir: Path to normalized sequences directory
        input_format: Format of input data ("numpy", "parquet", "csv")
        logger: Optional logger function

    Returns:
        Dictionary containing:
        - "categorical": Categorical sequence arrays
        - "numerical": Numerical sequence arrays
        - "categorical_attention_mask": Attention masks for categorical data
        - "numerical_attention_mask": Attention masks for numerical data
        - "metadata": Loaded metadata dictionary
    """
    log = logger or print
    input_path = Path(input_dir)

    if not input_path.exists():
        raise RuntimeError(f"Normalized sequences directory not found: {input_dir}")

    # Load metadata first
    metadata_file = input_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        log(f"[INFO] Loaded metadata: {metadata}")
    else:
        log("[WARNING] No metadata file found")
        metadata = {}

    sequences = {"metadata": metadata}

    # Load sequence data based on format
    if input_format == "numpy":
        # Load .npy files
        for seq_type in ["categorical", "numerical"]:
            seq_file = input_path / f"{seq_type}.npy"
            if seq_file.exists():
                sequences[seq_type] = np.load(seq_file)
                log(f"[INFO] Loaded {seq_type} sequences: {sequences[seq_type].shape}")

            # Load attention masks
            mask_file = input_path / f"{seq_type}_attention_mask.npy"
            if mask_file.exists():
                sequences[f"{seq_type}_attention_mask"] = np.load(mask_file)
                log(
                    f"[INFO] Loaded {seq_type} attention mask: {sequences[f'{seq_type}_attention_mask'].shape}"
                )

    elif input_format == "parquet":
        # Load .parquet files and reshape if needed
        for seq_type in ["categorical", "numerical"]:
            seq_file = input_path / f"{seq_type}.parquet"
            if seq_file.exists():
                df = pd.read_parquet(seq_file)
                # Reshape based on metadata if available
                if "shapes" in metadata and seq_type in metadata["shapes"]:
                    target_shape = metadata["shapes"][seq_type]
                    sequences[seq_type] = df.values.reshape(target_shape)
                else:
                    sequences[seq_type] = df.values
                log(f"[INFO] Loaded {seq_type} sequences: {sequences[seq_type].shape}")

    elif input_format == "csv":
        # Load .csv files and reshape if needed
        for seq_type in ["categorical", "numerical"]:
            seq_file = input_path / f"{seq_type}.csv"
            if seq_file.exists():
                df = pd.read_csv(seq_file)
                # Reshape based on metadata if available
                if "shapes" in metadata and seq_type in metadata["shapes"]:
                    target_shape = metadata["shapes"][seq_type]
                    sequences[seq_type] = df.values.reshape(target_shape)
                else:
                    sequences[seq_type] = df.values
                log(f"[INFO] Loaded {seq_type} sequences: {sequences[seq_type].shape}")

    return sequences


def validate_input_data(
    normalized_data: Dict[str, np.ndarray], logger: Optional[Callable] = None
) -> None:
    """Validate the structure of normalized sequence data."""
    log = logger or print

    required_keys = ["categorical", "numerical"]
    missing_keys = [
        key
        for key in required_keys
        if key not in normalized_data or normalized_data[key] is None
    ]

    if missing_keys:
        raise RuntimeError(f"Missing required sequence data: {missing_keys}")

    # Validate shapes are consistent
    cat_shape = (
        normalized_data["categorical"].shape
        if "categorical" in normalized_data
        else None
    )
    num_shape = (
        normalized_data["numerical"].shape if "numerical" in normalized_data else None
    )

    if cat_shape and num_shape:
        if cat_shape[0] != num_shape[0]:
            raise RuntimeError(
                f"Inconsistent batch sizes: categorical {cat_shape[0]} vs numerical {num_shape[0]}"
            )
        if cat_shape[1] != num_shape[1]:
            raise RuntimeError(
                f"Inconsistent sequence lengths: categorical {cat_shape[1]} vs numerical {num_shape[1]}"
            )

    log("[INFO] Input data validation passed")


# --- Feature Engineering Operations ---


class GenericTemporalFeaturesOperation:
    """
    Extracts generic temporal features from normalized sequences.

    Extracted from TSA feature engineering requirements and general temporal modeling needs.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[Callable] = None):
        self.feature_types = config.get("feature_types", DEFAULT_FEATURE_TYPES)
        self.sequence_grouping_field = config.get(
            "sequence_grouping_field", DEFAULT_SEQUENCE_GROUPING_FIELD
        )
        self.timestamp_field = config.get("timestamp_field", DEFAULT_TIMESTAMP_FIELD)
        self.value_fields = config.get("value_fields", DEFAULT_VALUE_FIELDS)
        self.categorical_fields = config.get(
            "categorical_fields", DEFAULT_CATEGORICAL_FIELDS
        )
        self.output_prefix = config.get("output_prefix", DEFAULT_OUTPUT_PREFIX_GENERIC)
        self.log = logger or print

    def process(self, normalized_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract generic temporal features from normalized sequences.

        Args:
            normalized_data: Dictionary containing normalized sequence data

        Returns:
            Dictionary with extracted temporal features
        """
        self.log("[INFO] Extracting generic temporal features")

        # Extract sequences and attention masks
        categorical_seq = normalized_data.get("categorical")
        numerical_seq = normalized_data.get("numerical")
        cat_mask = normalized_data.get("categorical_attention_mask")
        num_mask = normalized_data.get("numerical_attention_mask")

        batch_size = (
            categorical_seq.shape[0]
            if categorical_seq is not None
            else numerical_seq.shape[0]
        )
        all_features = []
        feature_names = []

        # Process each entity in the batch
        for i in range(batch_size):
            entity_features = {}

            # Extract different types of features
            if "statistical" in self.feature_types:
                statistical_features = self._extract_statistical_features(
                    categorical_seq[i] if categorical_seq is not None else None,
                    numerical_seq[i] if numerical_seq is not None else None,
                    cat_mask[i] if cat_mask is not None else None,
                    num_mask[i] if num_mask is not None else None,
                )
                entity_features.update(statistical_features)

            if "temporal" in self.feature_types:
                temporal_features = self._extract_temporal_patterns(
                    numerical_seq[i] if numerical_seq is not None else None,
                    num_mask[i] if num_mask is not None else None,
                )
                entity_features.update(temporal_features)

            if "behavioral" in self.feature_types:
                behavioral_features = self._extract_behavioral_features(
                    categorical_seq[i] if categorical_seq is not None else None,
                    numerical_seq[i] if numerical_seq is not None else None,
                    cat_mask[i] if cat_mask is not None else None,
                    num_mask[i] if num_mask is not None else None,
                )
                entity_features.update(behavioral_features)

            # Convert to feature vector
            if i == 0:
                feature_names = sorted(entity_features.keys())

            feature_vector = [entity_features.get(name, 0.0) for name in feature_names]
            all_features.append(feature_vector)

        # Convert to numpy array
        features_array = np.array(all_features, dtype=np.float32)

        self.log(f"[INFO] Extracted generic temporal features: {features_array.shape}")

        return {"features": features_array, "feature_names": feature_names}

    def _extract_statistical_features(
        self,
        cat_seq: Optional[np.ndarray],
        num_seq: Optional[np.ndarray],
        cat_mask: Optional[np.ndarray],
        num_mask: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Extract statistical features from entity sequences."""
        features = {}

        # Process numerical sequences
        if num_seq is not None:
            # Use attention mask to identify valid timesteps
            valid_mask = (
                num_mask
                if num_mask is not None
                else np.ones(num_seq.shape[0], dtype=bool)
            )

            for field_idx in range(num_seq.shape[1]):
                field_name = f"num_field_{field_idx}"
                if field_idx < len(self.value_fields):
                    field_name = self.value_fields[field_idx]

                # Extract valid values for this field
                values = num_seq[valid_mask, field_idx]
                values = values[~np.isnan(values)]  # Remove NaN values

                if len(values) > 0:
                    # Basic statistics
                    features[f"{self.output_prefix}count_{field_name}"] = len(values)
                    features[f"{self.output_prefix}sum_{field_name}"] = np.sum(values)
                    features[f"{self.output_prefix}mean_{field_name}"] = np.mean(values)
                    features[f"{self.output_prefix}std_{field_name}"] = np.std(values)
                    features[f"{self.output_prefix}min_{field_name}"] = np.min(values)
                    features[f"{self.output_prefix}max_{field_name}"] = np.max(values)

                    # Percentiles
                    features[f"{self.output_prefix}p25_{field_name}"] = np.percentile(
                        values, 25
                    )
                    features[f"{self.output_prefix}p50_{field_name}"] = np.percentile(
                        values, 50
                    )
                    features[f"{self.output_prefix}p75_{field_name}"] = np.percentile(
                        values, 75
                    )

                    # Advanced statistics
                    if len(values) > 1:
                        features[f"{self.output_prefix}skew_{field_name}"] = stats.skew(
                            values
                        )
                        features[f"{self.output_prefix}kurtosis_{field_name}"] = (
                            stats.kurtosis(values)
                        )
                        features[f"{self.output_prefix}range_{field_name}"] = np.max(
                            values
                        ) - np.min(values)
                        cv = (
                            np.std(values) / np.mean(values)
                            if np.mean(values) != 0
                            else 0
                        )
                        features[f"{self.output_prefix}cv_{field_name}"] = cv

        # Process categorical sequences
        if cat_seq is not None:
            valid_mask = (
                cat_mask
                if cat_mask is not None
                else np.ones(cat_seq.shape[0], dtype=bool)
            )

            for field_idx in range(cat_seq.shape[1]):
                field_name = f"cat_field_{field_idx}"
                if field_idx < len(self.categorical_fields):
                    field_name = self.categorical_fields[field_idx]

                # Extract valid values for this field
                values = cat_seq[valid_mask, field_idx]
                values = values[values != 0]  # Remove padding (assuming 0 is padding)

                if len(values) > 0:
                    # Unique counts and diversity
                    unique_values = np.unique(values)
                    features[f"{self.output_prefix}unique_count_{field_name}"] = len(
                        unique_values
                    )
                    features[f"{self.output_prefix}diversity_{field_name}"] = len(
                        unique_values
                    ) / len(values)

                    # Most frequent category
                    counts = Counter(values)
                    most_common_count = counts.most_common(1)[0][1]
                    features[f"{self.output_prefix}mode_freq_{field_name}"] = (
                        most_common_count / len(values)
                    )

        return features

    def _extract_temporal_patterns(
        self, num_seq: Optional[np.ndarray], num_mask: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Extract temporal pattern features."""
        features = {}

        if num_seq is None:
            return features

        valid_mask = (
            num_mask if num_mask is not None else np.ones(num_seq.shape[0], dtype=bool)
        )

        # Assume last column before padding indicator is temporal (time delta)
        if num_seq.shape[1] > 1:
            temporal_col = -2  # Second to last column (last is padding indicator)
            time_deltas = num_seq[valid_mask, temporal_col]
            time_deltas = time_deltas[~np.isnan(time_deltas)]

            if len(time_deltas) > 1:
                # Time interval statistics
                features[f"{self.output_prefix}avg_time_delta"] = np.mean(time_deltas)
                features[f"{self.output_prefix}std_time_delta"] = np.std(time_deltas)
                features[f"{self.output_prefix}min_time_delta"] = np.min(time_deltas)
                features[f"{self.output_prefix}max_time_delta"] = np.max(time_deltas)

                # Temporal span and frequency
                total_span = np.max(time_deltas) - np.min(time_deltas)
                features[f"{self.output_prefix}temporal_span"] = total_span
                if total_span > 0:
                    features[f"{self.output_prefix}event_frequency"] = (
                        len(time_deltas) / total_span
                    )

                # Regularity measures
                if np.mean(time_deltas) > 0:
                    regularity = 1 / (1 + np.std(time_deltas) / np.mean(time_deltas))
                    features[f"{self.output_prefix}interval_regularity"] = regularity

        return features

    def _extract_behavioral_features(
        self,
        cat_seq: Optional[np.ndarray],
        num_seq: Optional[np.ndarray],
        cat_mask: Optional[np.ndarray],
        num_mask: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Extract behavioral pattern features."""
        features = {}

        # Activity concentration and consistency
        if num_seq is not None:
            valid_mask = (
                num_mask
                if num_mask is not None
                else np.ones(num_seq.shape[0], dtype=bool)
            )

            # Compute activity concentration (Gini coefficient)
            if np.sum(valid_mask) > 2:
                # Use time deltas for activity concentration
                if num_seq.shape[1] > 1:
                    time_deltas = num_seq[valid_mask, -2]  # Time delta column
                    time_deltas = time_deltas[~np.isnan(time_deltas)]
                    if len(time_deltas) > 1:
                        gini = self._compute_gini_coefficient(time_deltas)
                        features[f"{self.output_prefix}activity_concentration"] = gini

            # Consistency score based on coefficient of variation
            consistency_scores = []
            for field_idx in range(
                min(num_seq.shape[1] - 1, len(self.value_fields))
            ):  # Exclude padding column
                values = num_seq[valid_mask, field_idx]
                values = values[~np.isnan(values)]
                if len(values) > 1 and np.mean(values) != 0:
                    cv = np.std(values) / np.mean(values)
                    consistency_scores.append(1 / (1 + cv))

            if consistency_scores:
                features[f"{self.output_prefix}consistency_score"] = np.mean(
                    consistency_scores
                )

            # Trend analysis and volatility for value fields
            for field_idx in range(min(num_seq.shape[1] - 1, len(self.value_fields))):
                field_name = (
                    self.value_fields[field_idx]
                    if field_idx < len(self.value_fields)
                    else f"num_field_{field_idx}"
                )
                values = num_seq[valid_mask, field_idx]
                values = values[~np.isnan(values)]

                if len(values) > 1:
                    # Trend slope using linear regression
                    x = np.arange(len(values))
                    if len(values) > 1:
                        slope = np.polyfit(x, values, 1)[0]
                        features[f"{self.output_prefix}trend_slope_{field_name}"] = (
                            slope
                        )

                    # Volatility as standard deviation of returns
                    if len(values) > 1:
                        returns = np.diff(values) / (
                            values[:-1] + 1e-8
                        )  # Avoid division by zero
                        volatility = np.std(returns)
                        features[f"{self.output_prefix}volatility_{field_name}"] = (
                            volatility
                        )

        return features

    def _compute_gini_coefficient(self, values: np.ndarray) -> float:
        """Compute Gini coefficient for activity concentration."""
        if len(values) < 2:
            return 0.0

        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)

        # Gini coefficient formula
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_values)) / (
            n * np.sum(sorted_values)
        ) - (n + 1) / n
        return max(0, gini)  # Ensure non-negative


class TimeWindowAggregationsOperation:
    """
    Computes time window aggregations for multi-scale temporal analysis.

    Extracted from TSA time window feature requirements and temporal modeling needs.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[Callable] = None):
        self.window_sizes = config.get("window_sizes", DEFAULT_WINDOW_SIZES)
        self.aggregation_functions = config.get(
            "aggregation_functions", DEFAULT_AGGREGATION_FUNCTIONS
        )
        self.lag_features = config.get("lag_features", DEFAULT_LAG_FEATURES)
        self.exponential_smoothing_alpha = config.get(
            "exponential_smoothing_alpha", DEFAULT_EXPONENTIAL_SMOOTHING_ALPHA
        )
        self.time_unit = config.get("time_unit", DEFAULT_TIME_UNIT)
        self.output_prefix = config.get("output_prefix", DEFAULT_OUTPUT_PREFIX_WINDOW)
        self.value_fields = config.get("value_fields", DEFAULT_VALUE_FIELDS)
        self.log = logger or print

    def process(self, normalized_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute time window aggregations for sequences.

        Args:
            normalized_data: Dictionary containing normalized sequence data

        Returns:
            Dictionary with computed window aggregation features
        """
        self.log("[INFO] Computing time window aggregations")

        # Extract sequences and attention masks
        numerical_seq = normalized_data.get("numerical")
        num_mask = normalized_data.get("numerical_attention_mask")

        if numerical_seq is None:
            self.log("[WARNING] No numerical sequences found for window aggregations")
            return {"features": np.array([]), "feature_names": []}

        batch_size = numerical_seq.shape[0]
        all_features = []
        feature_names = []

        # Process each entity in the batch
        for i in range(batch_size):
            entity_features = {}

            # Compute rolling window features
            rolling_features = self._compute_rolling_features(
                numerical_seq[i], num_mask[i] if num_mask is not None else None
            )
            entity_features.update(rolling_features)

            # Compute lag features
            lag_features = self._compute_lag_features(
                numerical_seq[i], num_mask[i] if num_mask is not None else None
            )
            entity_features.update(lag_features)

            # Compute exponential smoothing features
            exp_smooth_features = self._compute_exponential_smoothing(
                numerical_seq[i], num_mask[i] if num_mask is not None else None
            )
            entity_features.update(exp_smooth_features)

            # Convert to feature vector
            if i == 0:
                feature_names = sorted(entity_features.keys())

            feature_vector = [entity_features.get(name, 0.0) for name in feature_names]
            all_features.append(feature_vector)

        # Convert to numpy array
        features_array = np.array(all_features, dtype=np.float32)

        self.log(f"[INFO] Computed window aggregation features: {features_array.shape}")

        return {"features": features_array, "feature_names": feature_names}

    def _compute_rolling_features(
        self, num_seq: np.ndarray, num_mask: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Compute rolling window aggregation features."""
        features = {}

        valid_mask = (
            num_mask if num_mask is not None else np.ones(num_seq.shape[0], dtype=bool)
        )
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return features

        # Process each value field (exclude padding indicator column)
        num_value_fields = min(num_seq.shape[1] - 1, len(self.value_fields))

        for field_idx in range(num_value_fields):
            field_name = (
                self.value_fields[field_idx]
                if field_idx < len(self.value_fields)
                else f"field_{field_idx}"
            )
            values = num_seq[valid_indices, field_idx]
            values = values[~np.isnan(values)]

            if len(values) == 0:
                continue

            for window_size in self.window_sizes:
                # Adjust window size to available data
                effective_window = min(window_size, len(values))

                if effective_window <= 0:
                    continue

                for agg_func in self.aggregation_functions:
                    try:
                        # Compute rolling aggregation for the most recent window
                        window_values = values[-effective_window:]

                        if agg_func == "mean":
                            result = np.mean(window_values)
                        elif agg_func == "sum":
                            result = np.sum(window_values)
                        elif agg_func == "std":
                            result = np.std(window_values)
                        elif agg_func == "min":
                            result = np.min(window_values)
                        elif agg_func == "max":
                            result = np.max(window_values)
                        elif agg_func == "count":
                            result = len(window_values)
                        else:
                            continue

                        feature_name = f"{self.output_prefix}rolling_{window_size}_{agg_func}_{field_name}"
                        features[feature_name] = (
                            float(result) if not np.isnan(result) else 0.0
                        )

                    except Exception:
                        # Handle edge cases gracefully
                        feature_name = f"{self.output_prefix}rolling_{window_size}_{agg_func}_{field_name}"
                        features[feature_name] = 0.0

        return features

    def _compute_lag_features(
        self, num_seq: np.ndarray, num_mask: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Compute lag features for historical values."""
        features = {}

        valid_mask = (
            num_mask if num_mask is not None else np.ones(num_seq.shape[0], dtype=bool)
        )
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return features

        # Process each value field (exclude padding indicator column)
        num_value_fields = min(num_seq.shape[1] - 1, len(self.value_fields))

        for field_idx in range(num_value_fields):
            field_name = (
                self.value_fields[field_idx]
                if field_idx < len(self.value_fields)
                else f"field_{field_idx}"
            )
            values = num_seq[valid_indices, field_idx]
            values = values[~np.isnan(values)]

            if len(values) == 0:
                continue

            for lag in self.lag_features:
                try:
                    # Get lagged value
                    if lag < len(values):
                        lag_value = values[-(lag + 1)]  # lag=1 means previous value
                    else:
                        lag_value = 0.0  # Default for insufficient history

                    feature_name = f"{self.output_prefix}lag_{lag}_{field_name}"
                    features[feature_name] = (
                        float(lag_value) if not np.isnan(lag_value) else 0.0
                    )

                except Exception:
                    # Handle edge cases gracefully
                    feature_name = f"{self.output_prefix}lag_{lag}_{field_name}"
                    features[feature_name] = 0.0

        return features

    def _compute_exponential_smoothing(
        self, num_seq: np.ndarray, num_mask: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Compute exponential smoothing features."""
        features = {}
        alpha = self.exponential_smoothing_alpha

        valid_mask = (
            num_mask if num_mask is not None else np.ones(num_seq.shape[0], dtype=bool)
        )
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return features

        # Process each value field (exclude padding indicator column)
        num_value_fields = min(num_seq.shape[1] - 1, len(self.value_fields))

        for field_idx in range(num_value_fields):
            field_name = (
                self.value_fields[field_idx]
                if field_idx < len(self.value_fields)
                else f"field_{field_idx}"
            )
            values = num_seq[valid_indices, field_idx]
            values = values[~np.isnan(values)]

            if len(values) == 0:
                continue

            try:
                # Compute exponential weighted moving average
                if len(values) == 1:
                    ewm_value = values[0]
                    ewm_std = 0.0
                else:
                    # Simple exponential smoothing
                    ewm_values = [values[0]]
                    for i in range(1, len(values)):
                        ewm_val = alpha * values[i] + (1 - alpha) * ewm_values[-1]
                        ewm_values.append(ewm_val)

                    ewm_value = ewm_values[-1]

                    # Compute exponential weighted standard deviation
                    squared_diffs = [
                        (values[i] - ewm_values[i]) ** 2 for i in range(len(values))
                    ]
                    ewm_var = squared_diffs[0]
                    for i in range(1, len(squared_diffs)):
                        ewm_var = alpha * squared_diffs[i] + (1 - alpha) * ewm_var
                    ewm_std = np.sqrt(ewm_var)

                feature_name = f"{self.output_prefix}exp_smooth_{field_name}"
                features[feature_name] = (
                    float(ewm_value) if not np.isnan(ewm_value) else 0.0
                )

                feature_name_std = f"{self.output_prefix}exp_smooth_std_{field_name}"
                features[feature_name_std] = (
                    float(ewm_std) if not np.isnan(ewm_std) else 0.0
                )

            except Exception:
                # Handle edge cases gracefully
                feature_name = f"{self.output_prefix}exp_smooth_{field_name}"
                features[feature_name] = 0.0
                feature_name_std = f"{self.output_prefix}exp_smooth_std_{field_name}"
                features[feature_name_std] = 0.0

        return features


# --- Feature Quality Control ---


class FeatureQualityController:
    """
    Comprehensive feature quality control and validation framework.

    Ensures engineered features meet quality standards for model consumption.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[Callable] = None):
        self.missing_threshold = config.get(
            "missing_value_threshold", DEFAULT_MISSING_VALUE_THRESHOLD
        )
        self.correlation_threshold = config.get(
            "correlation_threshold", DEFAULT_CORRELATION_THRESHOLD
        )
        self.variance_threshold = config.get(
            "variance_threshold", DEFAULT_VARIANCE_THRESHOLD
        )
        self.enable_outlier_detection = config.get(
            "outlier_detection", DEFAULT_OUTLIER_DETECTION
        )
        self.log = logger or print

    def validate_features(
        self, features: np.ndarray, feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Comprehensive feature validation and quality assessment.

        Args:
            features: Feature matrix (N_entities, N_features)
            feature_names: List of feature names

        Returns:
            Quality report with validation results and recommendations
        """
        self.log("[INFO] Validating feature quality")

        quality_report = {
            "validation_results": {},
            "quality_metrics": {},
            "recommendations": [],
            "feature_statistics": {},
        }

        # Convert to DataFrame for analysis
        df = pd.DataFrame(features, columns=feature_names)

        # Missing value analysis
        missing_analysis = self._analyze_missing_values(df)
        quality_report["validation_results"]["missing_values"] = missing_analysis

        # Correlation analysis
        correlation_analysis = self._analyze_correlations(df)
        quality_report["validation_results"]["correlations"] = correlation_analysis

        # Variance analysis
        variance_analysis = self._analyze_variance(df)
        quality_report["validation_results"]["variance"] = variance_analysis

        # Outlier detection
        if self.enable_outlier_detection:
            outlier_analysis = self._detect_outliers(df)
            quality_report["validation_results"]["outliers"] = outlier_analysis

        # Feature selection recommendations
        selection_recommendations = self._recommend_feature_selection(
            quality_report["validation_results"]
        )
        quality_report["recommendations"].extend(selection_recommendations)

        # Overall quality score
        quality_score = self._compute_quality_score(
            quality_report["validation_results"]
        )
        quality_report["quality_metrics"]["overall_score"] = quality_score

        self.log(
            f"[INFO] Feature quality validation completed. Overall score: {quality_score:.3f}"
        )

        return quality_report

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns in features."""
        missing_rates = df.isnull().mean()

        problematic_features = missing_rates[
            missing_rates > self.missing_threshold
        ].index.tolist()

        return {
            "missing_rates": missing_rates.to_dict(),
            "problematic_features": problematic_features,
            "max_missing_rate": missing_rates.max(),
            "avg_missing_rate": missing_rates.mean(),
        }

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature correlations to identify redundant features."""
        # Compute correlation matrix for numerical features
        numerical_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numerical_df.corr()

        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > self.correlation_threshold:
                    high_corr_pairs.append(
                        {
                            "feature1": correlation_matrix.columns[i],
                            "feature2": correlation_matrix.columns[j],
                            "correlation": corr_value,
                        }
                    )

        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlation_pairs": high_corr_pairs,
            "max_correlation": correlation_matrix.abs().max().max()
            if len(correlation_matrix) > 0
            else 0,
        }

    def _analyze_variance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature variance to identify low-variance features."""
        numerical_df = df.select_dtypes(include=[np.number])
        variances = numerical_df.var()

        low_variance_features = variances[
            variances < self.variance_threshold
        ].index.tolist()

        return {
            "variances": variances.to_dict(),
            "low_variance_features": low_variance_features,
            "min_variance": variances.min(),
            "avg_variance": variances.mean(),
        }

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in feature distributions."""
        numerical_df = df.select_dtypes(include=[np.number])
        outlier_info = {}

        for column in numerical_df.columns:
            series = numerical_df[column].dropna()
            if len(series) > 0:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_rate = len(outliers) / len(series)

                outlier_info[column] = {
                    "outlier_count": len(outliers),
                    "outlier_rate": outlier_rate,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                }

        return outlier_info

    def _recommend_feature_selection(
        self, validation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate feature selection recommendations."""
        recommendations = []

        # Recommend removing high missing value features
        if "missing_values" in validation_results:
            high_missing_features = validation_results["missing_values"][
                "problematic_features"
            ]
            if high_missing_features:
                recommendations.append(
                    f"Consider removing features with high missing rates: {high_missing_features}"
                )

        # Recommend removing low variance features
        if "variance" in validation_results:
            low_var_features = validation_results["variance"]["low_variance_features"]
            if low_var_features:
                recommendations.append(
                    f"Consider removing low variance features: {low_var_features}"
                )

        # Recommend removing highly correlated features
        if "correlations" in validation_results:
            high_corr_pairs = validation_results["correlations"][
                "high_correlation_pairs"
            ]
            if high_corr_pairs:
                recommendations.append(
                    f"Consider removing one feature from highly correlated pairs: {len(high_corr_pairs)} pairs found"
                )

        return recommendations

    def _compute_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Compute overall feature quality score."""
        score_components = []

        # Missing value score (lower missing rate = higher score)
        if "missing_values" in validation_results:
            missing_score = 1 - validation_results["missing_values"]["avg_missing_rate"]
            score_components.append(missing_score)

        # Variance score (higher average variance = higher score, up to a point)
        if "variance" in validation_results:
            avg_variance = validation_results["variance"]["avg_variance"]
            variance_score = min(1.0, avg_variance / 10.0)  # Normalize to 0-1 range
            score_components.append(variance_score)

        # Correlation score (fewer high correlations = higher score)
        if "correlations" in validation_results:
            high_corr_count = len(
                validation_results["correlations"]["high_correlation_pairs"]
            )
            correlation_score = max(
                0, 1 - high_corr_count / 10.0
            )  # Penalize many high correlations
            score_components.append(correlation_score)

        return np.mean(score_components) if score_components else 0.5


# --- Output Saving ---


def save_temporal_feature_tensors(
    feature_tensors: Dict[str, Any],
    output_dir: str,
    output_format: str = "numpy",
    logger: Optional[Callable] = None,
) -> None:
    """Save temporal feature tensors in the specified format."""
    log = logger or print
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Saving temporal feature tensors in {output_format} format")

    # Extract main components
    features = feature_tensors.get("features")
    feature_names = feature_tensors.get("feature_names", [])
    feature_metadata = feature_tensors.get("feature_metadata", {})
    quality_report = feature_tensors.get("quality_report", {})

    if output_format == "numpy":
        # Save main feature tensor
        if features is not None:
            features_file = output_path / "features.npy"
            np.save(features_file, features)
            log(f"[INFO] Saved {features_file} with shape {features.shape}")

        # Save feature names
        if feature_names:
            names_file = output_path / "feature_names.json"
            with open(names_file, "w") as f:
                json.dump(feature_names, f, indent=2)
            log(f"[INFO] Saved feature names to {names_file}")

    elif output_format == "parquet":
        # Save as parquet with feature names as columns
        if features is not None:
            if len(feature_names) == features.shape[1]:
                df = pd.DataFrame(features, columns=feature_names)
            else:
                df = pd.DataFrame(features)

            features_file = output_path / "features.parquet"
            df.to_parquet(features_file, index=False)
            log(f"[INFO] Saved {features_file} with shape {df.shape}")

    elif output_format == "csv":
        # Save as CSV with feature names as columns
        if features is not None:
            if len(feature_names) == features.shape[1]:
                df = pd.DataFrame(features, columns=feature_names)
            else:
                df = pd.DataFrame(features)

            features_file = output_path / "features.csv"
            df.to_csv(features_file, index=False)
            log(f"[INFO] Saved {features_file} with shape {df.shape}")

    # Save metadata
    metadata = {
        "feature_count": len(feature_names),
        "entity_count": features.shape[0] if features is not None else 0,
        "output_format": output_format,
        "feature_metadata": feature_metadata,
        "tensor_shapes": {
            "features": list(features.shape) if features is not None else []
        },
    }

    metadata_file = output_path / "feature_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"[INFO] Saved feature metadata to {metadata_file}")

    # Save quality report
    if quality_report:
        quality_file = output_path / "quality_report.json"
        with open(quality_file, "w") as f:
            json.dump(quality_report, f, indent=2)
        log(f"[INFO] Saved quality report to {quality_file}")


# --- Main Processing Logic ---


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, np.ndarray]:
    """
    Main logic for temporal feature engineering.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger object (defaults to print if None)

    Returns:
        Dictionary of temporal feature tensors
    """
    # Extract configuration from environ_vars with defaults
    sequence_grouping_field = environ_vars.get(
        "SEQUENCE_GROUPING_FIELD", DEFAULT_SEQUENCE_GROUPING_FIELD
    )
    timestamp_field = environ_vars.get("TIMESTAMP_FIELD", DEFAULT_TIMESTAMP_FIELD)

    # Parse JSON configuration
    value_fields = json.loads(
        environ_vars.get("VALUE_FIELDS", json.dumps(DEFAULT_VALUE_FIELDS))
    )
    categorical_fields = json.loads(
        environ_vars.get("CATEGORICAL_FIELDS", json.dumps(DEFAULT_CATEGORICAL_FIELDS))
    )
    feature_types = json.loads(
        environ_vars.get("FEATURE_TYPES", json.dumps(DEFAULT_FEATURE_TYPES))
    )
    window_sizes = json.loads(
        environ_vars.get("WINDOW_SIZES", json.dumps(DEFAULT_WINDOW_SIZES))
    )
    aggregation_functions = json.loads(
        environ_vars.get(
            "AGGREGATION_FUNCTIONS", json.dumps(DEFAULT_AGGREGATION_FUNCTIONS)
        )
    )
    lag_features = json.loads(
        environ_vars.get("LAG_FEATURES", json.dumps(DEFAULT_LAG_FEATURES))
    )

    # Processing configuration
    exponential_smoothing_alpha = float(
        environ_vars.get(
            "EXPONENTIAL_SMOOTHING_ALPHA", str(DEFAULT_EXPONENTIAL_SMOOTHING_ALPHA)
        )
    )
    time_unit = environ_vars.get("TIME_UNIT", DEFAULT_TIME_UNIT)
    input_format = environ_vars.get("INPUT_FORMAT", DEFAULT_INPUT_FORMAT)
    output_format = environ_vars.get("OUTPUT_FORMAT", DEFAULT_OUTPUT_FORMAT)

    # Quality control configuration
    enable_validation = (
        environ_vars.get("ENABLE_VALIDATION", str(DEFAULT_ENABLE_VALIDATION)).lower()
        == "true"
    )
    missing_value_threshold = float(
        environ_vars.get(
            "MISSING_VALUE_THRESHOLD", str(DEFAULT_MISSING_VALUE_THRESHOLD)
        )
    )
    correlation_threshold = float(
        environ_vars.get("CORRELATION_THRESHOLD", str(DEFAULT_CORRELATION_THRESHOLD))
    )
    variance_threshold = float(
        environ_vars.get("VARIANCE_THRESHOLD", str(DEFAULT_VARIANCE_THRESHOLD))
    )
    outlier_detection = (
        environ_vars.get("OUTLIER_DETECTION", str(DEFAULT_OUTLIER_DETECTION)).lower()
        == "true"
    )

    # Extract paths
    normalized_sequences_dir = input_paths["normalized_sequences"]
    output_dir = output_paths["temporal_feature_tensors"]

    # Use print function if no logger is provided
    log = logger or print

    # 1. Load normalized sequences
    log(f"[INFO] Loading normalized sequences from {normalized_sequences_dir}...")
    normalized_data = load_normalized_sequences(
        normalized_sequences_dir, input_format, logger=log
    )
    log(f"[INFO] Loaded normalized sequences")

    # 2. Validate input data
    validate_input_data(normalized_data, logger=log)

    # 3. Configure feature operations
    generic_config = {
        "feature_types": feature_types,
        "sequence_grouping_field": sequence_grouping_field,
        "timestamp_field": timestamp_field,
        "value_fields": value_fields,
        "categorical_fields": categorical_fields,
        "output_prefix": DEFAULT_OUTPUT_PREFIX_GENERIC,
    }

    window_config = {
        "window_sizes": window_sizes,
        "aggregation_functions": aggregation_functions,
        "lag_features": lag_features,
        "exponential_smoothing_alpha": exponential_smoothing_alpha,
        "time_unit": time_unit,
        "output_prefix": DEFAULT_OUTPUT_PREFIX_WINDOW,
        "value_fields": value_fields,
    }

    quality_config = {
        "missing_value_threshold": missing_value_threshold,
        "correlation_threshold": correlation_threshold,
        "variance_threshold": variance_threshold,
        "outlier_detection": outlier_detection,
    }

    # 4. Initialize feature operations
    generic_features_op = GenericTemporalFeaturesOperation(generic_config, logger=log)
    window_features_op = TimeWindowAggregationsOperation(window_config, logger=log)
    quality_controller = FeatureQualityController(quality_config, logger=log)

    # 5. Extract generic temporal features
    log("[INFO] Extracting generic temporal features...")
    generic_results = generic_features_op.process(normalized_data)

    # 6. Extract time window aggregation features
    log("[INFO] Extracting time window aggregation features...")
    window_results = window_features_op.process(normalized_data)

    # 7. Combine all features
    log("[INFO] Combining feature tensors...")
    all_feature_names = []
    all_features = []

    # Add generic features
    if generic_results["features"].size > 0:
        all_features.append(generic_results["features"])
        all_feature_names.extend(generic_results["feature_names"])

    # Add window features
    if window_results["features"].size > 0:
        all_features.append(window_results["features"])
        all_feature_names.extend(window_results["feature_names"])

    # Combine feature matrices
    if all_features:
        combined_features = np.concatenate(all_features, axis=1)
    else:
        combined_features = np.array([])
        all_feature_names = []

    log(f"[INFO] Combined feature tensor shape: {combined_features.shape}")

    # 8. Feature quality validation
    quality_report = {}
    if enable_validation and combined_features.size > 0:
        log("[INFO] Performing feature quality validation...")
        quality_report = quality_controller.validate_features(
            combined_features, all_feature_names
        )

    # 9. Prepare output tensors
    feature_tensors = {
        "features": combined_features,
        "feature_names": all_feature_names,
        "feature_metadata": {
            "generic_feature_count": len(generic_results["feature_names"])
            if generic_results["features"].size > 0
            else 0,
            "window_feature_count": len(window_results["feature_names"])
            if window_results["features"].size > 0
            else 0,
            "total_feature_count": len(all_feature_names),
            "entity_count": combined_features.shape[0]
            if combined_features.size > 0
            else 0,
            "configuration": {
                "feature_types": feature_types,
                "value_fields": value_fields,
                "categorical_fields": categorical_fields,
                "window_sizes": window_sizes,
                "aggregation_functions": aggregation_functions,
                "lag_features": lag_features,
            },
        },
        "quality_report": quality_report,
    }

    # 10. Save temporal feature tensors
    save_temporal_feature_tensors(
        feature_tensors, output_dir, output_format, logger=log
    )

    log("[INFO] Temporal feature engineering complete.")
    return feature_tensors


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--job_type",
            type=str,
            required=True,
            choices=["training", "validation", "testing", "calibration"],
            help="One of ['training','validation','testing','calibration']",
        )
        args = parser.parse_args()

        # Define standard SageMaker paths
        INPUT_NORMALIZED_SEQUENCES_DIR = "/opt/ml/processing/input/normalized_sequences"
        OUTPUT_DIR = "/opt/ml/processing/output"

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        # Read configuration from environment variables
        SEQUENCE_GROUPING_FIELD = os.environ.get(
            "SEQUENCE_GROUPING_FIELD", DEFAULT_SEQUENCE_GROUPING_FIELD
        )
        TIMESTAMP_FIELD = os.environ.get("TIMESTAMP_FIELD", DEFAULT_TIMESTAMP_FIELD)
        VALUE_FIELDS = os.environ.get("VALUE_FIELDS", json.dumps(DEFAULT_VALUE_FIELDS))
        CATEGORICAL_FIELDS = os.environ.get(
            "CATEGORICAL_FIELDS", json.dumps(DEFAULT_CATEGORICAL_FIELDS)
        )
        FEATURE_TYPES = os.environ.get(
            "FEATURE_TYPES", json.dumps(DEFAULT_FEATURE_TYPES)
        )
        WINDOW_SIZES = os.environ.get("WINDOW_SIZES", json.dumps(DEFAULT_WINDOW_SIZES))
        AGGREGATION_FUNCTIONS = os.environ.get(
            "AGGREGATION_FUNCTIONS", json.dumps(DEFAULT_AGGREGATION_FUNCTIONS)
        )
        LAG_FEATURES = os.environ.get("LAG_FEATURES", json.dumps(DEFAULT_LAG_FEATURES))
        EXPONENTIAL_SMOOTHING_ALPHA = os.environ.get(
            "EXPONENTIAL_SMOOTHING_ALPHA", str(DEFAULT_EXPONENTIAL_SMOOTHING_ALPHA)
        )
        TIME_UNIT = os.environ.get("TIME_UNIT", DEFAULT_TIME_UNIT)
        INPUT_FORMAT = os.environ.get("INPUT_FORMAT", DEFAULT_INPUT_FORMAT)
        OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", DEFAULT_OUTPUT_FORMAT)
        ENABLE_VALIDATION = os.environ.get(
            "ENABLE_VALIDATION", str(DEFAULT_ENABLE_VALIDATION)
        )
        MISSING_VALUE_THRESHOLD = os.environ.get(
            "MISSING_VALUE_THRESHOLD", str(DEFAULT_MISSING_VALUE_THRESHOLD)
        )
        CORRELATION_THRESHOLD = os.environ.get(
            "CORRELATION_THRESHOLD", str(DEFAULT_CORRELATION_THRESHOLD)
        )
        VARIANCE_THRESHOLD = os.environ.get(
            "VARIANCE_THRESHOLD", str(DEFAULT_VARIANCE_THRESHOLD)
        )
        OUTLIER_DETECTION = os.environ.get(
            "OUTLIER_DETECTION", str(DEFAULT_OUTLIER_DETECTION)
        )

        # Log key parameters
        logger.info(f"Starting temporal feature engineering with parameters:")
        logger.info(f"  Job Type: {args.job_type}")
        logger.info(f"  Sequence Grouping Field: {SEQUENCE_GROUPING_FIELD}")
        logger.info(f"  Timestamp Field: {TIMESTAMP_FIELD}")
        logger.info(f"  Value Fields: {VALUE_FIELDS}")
        logger.info(f"  Categorical Fields: {CATEGORICAL_FIELDS}")
        logger.info(f"  Feature Types: {FEATURE_TYPES}")
        logger.info(f"  Window Sizes: {WINDOW_SIZES}")
        logger.info(f"  Input Format: {INPUT_FORMAT}")
        logger.info(f"  Output Format: {OUTPUT_FORMAT}")
        logger.info(f"  Input Directory: {INPUT_NORMALIZED_SEQUENCES_DIR}")
        logger.info(f"  Output Directory: {OUTPUT_DIR}")

        # Set up path dictionaries
        input_paths = {"normalized_sequences": INPUT_NORMALIZED_SEQUENCES_DIR}

        output_paths = {"temporal_feature_tensors": OUTPUT_DIR}

        # Environment variables dictionary - pass all configuration to main
        environ_vars = {
            "SEQUENCE_GROUPING_FIELD": SEQUENCE_GROUPING_FIELD,
            "TIMESTAMP_FIELD": TIMESTAMP_FIELD,
            "VALUE_FIELDS": VALUE_FIELDS,
            "CATEGORICAL_FIELDS": CATEGORICAL_FIELDS,
            "FEATURE_TYPES": FEATURE_TYPES,
            "WINDOW_SIZES": WINDOW_SIZES,
            "AGGREGATION_FUNCTIONS": AGGREGATION_FUNCTIONS,
            "LAG_FEATURES": LAG_FEATURES,
            "EXPONENTIAL_SMOOTHING_ALPHA": EXPONENTIAL_SMOOTHING_ALPHA,
            "TIME_UNIT": TIME_UNIT,
            "INPUT_FORMAT": INPUT_FORMAT,
            "OUTPUT_FORMAT": OUTPUT_FORMAT,
            "ENABLE_VALIDATION": ENABLE_VALIDATION,
            "MISSING_VALUE_THRESHOLD": MISSING_VALUE_THRESHOLD,
            "CORRELATION_THRESHOLD": CORRELATION_THRESHOLD,
            "VARIANCE_THRESHOLD": VARIANCE_THRESHOLD,
            "OUTLIER_DETECTION": OUTLIER_DETECTION,
        }

        # Execute the main processing logic
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=logger.info,
        )

        # Log completion summary
        if result["features"].size > 0:
            shapes_summary = f"Features: {result['features'].shape}, Feature count: {len(result['feature_names'])}"
        else:
            shapes_summary = "No features generated"

        logger.info(
            f"Temporal feature engineering completed successfully. Output: {shapes_summary}"
        )
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error in temporal feature engineering script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
