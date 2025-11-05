#!/usr/bin/env python
"""
Temporal Sequence Normalization Script

This script normalizes temporal sequence data for machine learning models,
providing configurable operations for sequence ordering, validation, missing value handling,
time delta computation, and sequence padding/truncation.

Supports multiple data formats (CSV, TSV, JSON, Parquet) and provides extensive
configurability through environment variables.
"""

import os
import gzip
import tempfile
import shutil
import csv
import json
import argparse
import logging
import sys
import traceback
import re
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List, Tuple, Union
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# --- Default Configuration Values ---
# These will be overridden by environment variables passed via environ_vars

DEFAULT_SEQUENCE_LENGTH = 51
DEFAULT_SEQUENCE_SEPARATOR = "~"
DEFAULT_TEMPORAL_FIELD = "orderDate"
DEFAULT_SEQUENCE_GROUPING_FIELD = "customerId"
DEFAULT_RECORD_ID_FIELD = "objectId"
DEFAULT_MISSING_INDICATORS = ["", "My Text String", None]
DEFAULT_TIME_DELTA_MAX_SECONDS = 10000000
DEFAULT_PADDING_STRATEGY = "pre"
DEFAULT_TRUNCATION_STRATEGY = "post"
DEFAULT_ENABLE_MULTI_SEQUENCE = False
DEFAULT_SECONDARY_ENTITY_FIELD = "creditCardId"
DEFAULT_SEQUENCE_NAMING_PATTERN = "*_seq_by_{entity}.*"
DEFAULT_ENABLE_DISTRIBUTED_PROCESSING = False
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_MAX_WORKERS = "auto"
DEFAULT_VALIDATION_STRATEGY = "strict"
DEFAULT_OUTPUT_FORMAT = "numpy"
DEFAULT_INCLUDE_ATTENTION_MASKS = True

# --- Helper Functions (Reused from tabular_preprocessing.py) ---


def load_signature_columns(signature_path: str) -> Optional[list]:
    """Load column names from signature file."""
    signature_dir = Path(signature_path)
    if not signature_dir.exists():
        return None

    signature_files = list(signature_dir.glob("*"))
    if not signature_files:
        return None

    signature_file = signature_files[0]

    try:
        with open(signature_file, "r") as f:
            content = f.read().strip()
            if content:
                columns = [col.strip() for col in content.split(",")]
                return columns
    except Exception as e:
        raise RuntimeError(f"Error reading signature file {signature_file}: {e}")

    return None


def _is_gzipped(path: str) -> bool:
    return path.lower().endswith(".gz")


def _detect_separator_from_sample(sample_lines: str) -> str:
    """Use csv.Sniffer to detect a delimiter, defaulting to comma."""
    try:
        dialect = csv.Sniffer().sniff(sample_lines)
        return dialect.delimiter
    except Exception:
        return ","


def peek_json_format(file_path: Path, open_func: Callable = open) -> str:
    """Check if the JSON file is in JSON Lines or regular format."""
    try:
        with open_func(str(file_path), "rt") as f:
            first_char = f.read(1)
            if not first_char:
                raise ValueError("Empty file")
            f.seek(0)
            first_line = f.readline().strip()
            try:
                json.loads(first_line)
                return "lines" if first_char != "[" else "regular"
            except json.JSONDecodeError:
                f.seek(0)
                json.loads(f.read())
                return "regular"
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Error checking JSON format for {file_path}: {e}")


def _read_json_file(file_path: Path) -> pd.DataFrame:
    """Read a JSON or JSON Lines file into a DataFrame."""
    open_func = gzip.open if _is_gzipped(str(file_path)) else open
    fmt = peek_json_format(file_path, open_func)
    if fmt == "lines":
        return pd.read_json(str(file_path), lines=True, compression="infer")
    else:
        with open_func(str(file_path), "rt") as f:
            data = json.load(f)
        return pd.json_normalize(data if isinstance(data, list) else [data])


def _read_file_to_df(
    file_path: Path, column_names: Optional[list] = None
) -> pd.DataFrame:
    """Read a single file (CSV, TSV, JSON, Parquet) into a DataFrame."""
    suffix = file_path.suffix.lower()
    if suffix == ".gz":
        inner_ext = Path(file_path.stem).suffix.lower()
        if inner_ext in [".csv", ".tsv"]:
            with gzip.open(str(file_path), "rt") as f:
                sep = _detect_separator_from_sample(f.readline() + f.readline())
            if column_names:
                return pd.read_csv(
                    str(file_path),
                    sep=sep,
                    compression="gzip",
                    names=column_names,
                    header=0,
                    dtype=str,
                    keep_default_na=False,
                )
            else:
                return pd.read_csv(
                    str(file_path),
                    sep=sep,
                    compression="gzip",
                    dtype=str,
                    keep_default_na=False,
                )
        elif inner_ext == ".json":
            return _read_json_file(file_path)
        elif inner_ext.endswith(".parquet"):
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                with (
                    gzip.open(str(file_path), "rb") as f_in,
                    open(tmp.name, "wb") as f_out,
                ):
                    shutil.copyfileobj(f_in, f_out)
                df = pd.read_parquet(tmp.name)
            os.unlink(tmp.name)
            return df
        else:
            raise ValueError(f"Unsupported gzipped file type: {file_path}")
    elif suffix in [".csv", ".tsv"]:
        with open(str(file_path), "rt") as f:
            sep = _detect_separator_from_sample(f.readline() + f.readline())
        if column_names:
            return pd.read_csv(
                str(file_path),
                sep=sep,
                names=column_names,
                header=0,
                dtype=str,
                keep_default_na=False,
            )
        else:
            return pd.read_csv(
                str(file_path), sep=sep, dtype=str, keep_default_na=False
            )
    elif suffix == ".json":
        return _read_json_file(file_path)
    elif suffix.endswith(".parquet"):
        return pd.read_parquet(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def combine_shards(
    input_dir: str, signature_columns: Optional[list] = None
) -> pd.DataFrame:
    """Detect and combine all supported data shards in a directory."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    patterns = [
        "part-*.csv",
        "part-*.csv.gz",
        "part-*.tsv",
        "part-*.tsv.gz",
        "part-*.json",
        "part-*.json.gz",
        "part-*.parquet",
        "part-*.snappy.parquet",
        "part-*.parquet.gz",
        "*.csv",
        "*.csv.gz",
        "*.tsv",
        "*.tsv.gz",
        "*.json",
        "*.json.gz",
        "*.parquet",
        "*.snappy.parquet",
        "*.parquet.gz",
    ]
    all_shards = sorted([p for pat in patterns for p in input_path.glob(pat)])
    if not all_shards:
        raise RuntimeError(f"No CSV/TSV/JSON/Parquet shards found under {input_dir}")
    try:
        dfs = [_read_file_to_df(shard, signature_columns) for shard in all_shards]
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read or concatenate shards: {e}")


# --- Temporal Sequence Processing Operations ---


class SequenceOrderingOperation:
    """Handles temporal ordering of sequences."""

    def __init__(
        self, temporal_field: str, id_field: str, logger: Optional[Callable] = None
    ):
        self.temporal_field = temporal_field
        self.id_field = id_field
        self.log = logger or print

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort sequences by temporal field and handle duplicates."""
        self.log(f"[INFO] Ordering sequences by {self.temporal_field}")

        # Convert temporal field to numeric if needed
        if self.temporal_field in df.columns:
            df[self.temporal_field] = pd.to_numeric(
                df[self.temporal_field], errors="coerce"
            )

            # Sort by temporal field
            df = df.sort_values(by=self.temporal_field, ascending=True)

            # Handle duplicates by keeping the last occurrence
            df = df.drop_duplicates(subset=[self.id_field], keep="last")

            self.log(f"[INFO] Ordered {len(df)} sequences by {self.temporal_field}")
        else:
            self.log(
                f"[WARNING] Temporal field {self.temporal_field} not found, skipping ordering"
            )

        return df


class DataValidationOperation:
    """Validates sequence data integrity."""

    def __init__(
        self,
        validation_strategy: str,
        temporal_field: str,
        id_field: str,
        missing_indicators: List[str],
        logger: Optional[Callable] = None,
    ):
        self.validation_strategy = validation_strategy
        self.temporal_field = temporal_field
        self.id_field = id_field
        self.missing_indicators = missing_indicators
        self.log = logger or print

    def process(
        self, df: pd.DataFrame, sequence_fields: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Validate sequence data based on strategy."""
        self.log(
            f"[INFO] Validating sequence data with {self.validation_strategy} strategy"
        )

        initial_count = len(df)

        # Check for required fields
        required_fields = [self.temporal_field, self.id_field]
        missing_fields = [field for field in required_fields if field not in df.columns]

        if missing_fields:
            if self.validation_strategy == "strict":
                raise RuntimeError(f"Required fields missing: {missing_fields}")
            else:
                self.log(f"[WARNING] Missing fields in lenient mode: {missing_fields}")

        # Validate sequence field consistency
        for entity, fields in sequence_fields.items():
            if entity in ["categorical", "numerical"]:
                for field in fields:
                    if field in df.columns:
                        # Check for completely empty sequences
                        empty_mask = df[field].isin(self.missing_indicators)
                        if empty_mask.sum() > 0:
                            if self.validation_strategy == "strict":
                                df = df[~empty_mask]
                            else:
                                self.log(
                                    f"[WARNING] Found {empty_mask.sum()} empty sequences in {field}"
                                )

        final_count = len(df)
        if final_count < initial_count:
            self.log(
                f"[INFO] Validation removed {initial_count - final_count} invalid sequences"
            )

        return df


class MissingValueHandlingOperation:
    """Handles missing values in sequences."""

    def __init__(
        self,
        missing_indicators: List[str] = MISSING_INDICATORS,
        logger: Optional[Callable] = None,
    ):
        self.missing_indicators = missing_indicators
        self.log = logger or print

    def process(
        self, df: pd.DataFrame, sequence_fields: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Handle missing values in sequence fields."""
        self.log("[INFO] Handling missing values in sequences")

        for entity, fields in sequence_fields.items():
            for field in fields:
                if field in df.columns:
                    # Replace missing indicators with standardized missing value
                    for indicator in self.missing_indicators:
                        if indicator is None:
                            df[field] = df[field].fillna("")
                        else:
                            df[field] = df[field].replace(indicator, "")

        self.log("[INFO] Missing value handling completed")
        return df


class TimeDeltaComputationOperation:
    """Computes time deltas for temporal sequences."""

    def __init__(
        self,
        temporal_field: str = TEMPORAL_FIELD,
        max_seconds: int = TIME_DELTA_MAX_SECONDS,
        logger: Optional[Callable] = None,
    ):
        self.temporal_field = temporal_field
        self.max_seconds = max_seconds
        self.log = logger or print

    def process(self, sequence_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute time deltas for numerical sequences."""
        self.log("[INFO] Computing time deltas for sequences")

        for seq_type, seq_array in sequence_data.items():
            if "numerical" in seq_type and seq_array.shape[-1] > 1:
                # Assume last column before padding indicator is temporal
                temporal_col = -2
                if seq_array.shape[-1] > abs(temporal_col):
                    # Compute time deltas relative to the most recent timestamp
                    recent_time = seq_array[:, -1, temporal_col]
                    seq_array[:, :, temporal_col] = (
                        recent_time[:, np.newaxis] - seq_array[:, :, temporal_col]
                    )

                    # Cap time deltas
                    seq_array[:, :, temporal_col] = np.clip(
                        seq_array[:, :, temporal_col], 0, self.max_seconds
                    )

                    self.log(f"[INFO] Computed time deltas for {seq_type} sequences")

        return sequence_data


class SequencePaddingOperation:
    """Handles sequence padding and truncation."""

    def __init__(
        self,
        target_length: int = SEQUENCE_LENGTH,
        padding_strategy: str = PADDING_STRATEGY,
        truncation_strategy: str = TRUNCATION_STRATEGY,
        logger: Optional[Callable] = None,
    ):
        self.target_length = target_length
        self.padding_strategy = padding_strategy
        self.truncation_strategy = truncation_strategy
        self.log = logger or print

    def process(self, sequence_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Pad or truncate sequences to target length."""
        self.log(f"[INFO] Padding/truncating sequences to length {self.target_length}")

        padded_data = {}
        attention_masks = {}

        for seq_type, seq_array in sequence_data.items():
            batch_size, seq_len, feature_dim = seq_array.shape

            if seq_len == self.target_length:
                padded_data[seq_type] = seq_array
                if INCLUDE_ATTENTION_MASKS:
                    attention_masks[f"{seq_type}_attention_mask"] = np.ones(
                        (batch_size, seq_len), dtype=np.int8
                    )
            elif seq_len < self.target_length:
                # Padding needed
                pad_width = self.target_length - seq_len
                if self.padding_strategy == "pre":
                    pad_config = ((0, 0), (pad_width, 0), (0, 0))
                else:  # post
                    pad_config = ((0, 0), (0, pad_width), (0, 0))

                padded_array = np.pad(
                    seq_array, pad_config, mode="constant", constant_values=0
                )
                padded_data[seq_type] = padded_array

                if INCLUDE_ATTENTION_MASKS:
                    mask = np.zeros((batch_size, self.target_length), dtype=np.int8)
                    if self.padding_strategy == "pre":
                        mask[:, pad_width:] = 1
                    else:
                        mask[:, :seq_len] = 1
                    attention_masks[f"{seq_type}_attention_mask"] = mask
            else:
                # Truncation needed
                if self.truncation_strategy == "pre":
                    truncated_array = seq_array[:, -self.target_length :, :]
                else:  # post
                    truncated_array = seq_array[:, : self.target_length, :]

                padded_data[seq_type] = truncated_array

                if INCLUDE_ATTENTION_MASKS:
                    attention_masks[f"{seq_type}_attention_mask"] = np.ones(
                        (batch_size, self.target_length), dtype=np.int8
                    )

        # Add attention masks to the output
        if INCLUDE_ATTENTION_MASKS:
            padded_data.update(attention_masks)

        self.log(f"[INFO] Sequence padding/truncation completed")
        return padded_data


# --- Sequence Field Detection ---


def detect_sequence_fields(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Automatically detect sequence fields based on naming patterns."""
    sequence_fields = {"categorical": [], "numerical": [], "temporal": []}

    # Create pattern for sequence field detection
    entity_pattern = (
        f"({ENTITY_ID_FIELD}|{SECONDARY_ENTITY_FIELD})"
        if ENABLE_MULTI_SEQUENCE
        else ENTITY_ID_FIELD
    )
    pattern = SEQUENCE_NAMING_PATTERN.replace("{entity}", entity_pattern)

    for col in df.columns:
        # Check if column matches sequence pattern
        if re.search(pattern.replace("*", ".*"), col, re.IGNORECASE):
            # Determine field type based on naming conventions
            if any(
                cat_indicator in col.lower()
                for cat_indicator in ["cat_seq", "categorical"]
            ):
                sequence_fields["categorical"].append(col)
            elif any(
                num_indicator in col.lower()
                for num_indicator in ["num_seq", "numerical", "amount", "age", "count"]
            ):
                sequence_fields["numerical"].append(col)
        elif col == TEMPORAL_FIELD:
            sequence_fields["temporal"].append(col)

    # If no explicit sequence fields found, try to infer from column names
    if not sequence_fields["categorical"] and not sequence_fields["numerical"]:
        for col in df.columns:
            if SEQUENCE_SEPARATOR in str(df[col].iloc[0] if len(df) > 0 else ""):
                # Check if values look categorical or numerical
                sample_values = str(df[col].iloc[0]).split(SEQUENCE_SEPARATOR)[:5]
                try:
                    [float(v) for v in sample_values if v not in MISSING_INDICATORS]
                    sequence_fields["numerical"].append(col)
                except (ValueError, TypeError):
                    sequence_fields["categorical"].append(col)

    return sequence_fields


# --- Sequence Data Parsing ---


def parse_sequence_data(
    df: pd.DataFrame,
    sequence_fields: Dict[str, List[str]],
    logger: Optional[Callable] = None,
) -> Dict[str, np.ndarray]:
    """Parse sequence data from DataFrame into numpy arrays."""
    log = logger or print
    log("[INFO] Parsing sequence data into arrays")

    sequence_data = {}

    # Process categorical sequences
    if sequence_fields["categorical"]:
        cat_sequences = []
        label_encoders = {}

        for field in sequence_fields["categorical"]:
            if field in df.columns:
                # Parse sequences
                sequences = []
                for seq_str in df[field]:
                    if pd.isna(seq_str) or seq_str in MISSING_INDICATORS:
                        seq_values = [""]
                    else:
                        seq_values = str(seq_str).split(SEQUENCE_SEPARATOR)
                    sequences.append(seq_values)

                # Find max sequence length for this field
                max_len = max(len(seq) for seq in sequences) if sequences else 1

                # Pad sequences to same length
                padded_sequences = []
                for seq in sequences:
                    if len(seq) < max_len:
                        seq.extend([""] * (max_len - len(seq)))
                    padded_sequences.append(seq[:max_len])

                # Encode categorical values
                encoder = LabelEncoder()
                flat_values = [val for seq in padded_sequences for val in seq]
                encoder.fit(flat_values)
                label_encoders[field] = encoder

                # Transform sequences
                encoded_sequences = []
                for seq in padded_sequences:
                    encoded_seq = encoder.transform(seq)
                    encoded_sequences.append(encoded_seq)

                cat_sequences.append(np.array(encoded_sequences))

        if cat_sequences:
            # Stack all categorical sequences
            sequence_data["categorical"] = np.stack(cat_sequences, axis=-1)
            log(
                f"[INFO] Parsed categorical sequences: {sequence_data['categorical'].shape}"
            )

    # Process numerical sequences
    if sequence_fields["numerical"]:
        num_sequences = []

        for field in sequence_fields["numerical"]:
            if field in df.columns:
                # Parse sequences
                sequences = []
                for seq_str in df[field]:
                    if pd.isna(seq_str) or seq_str in MISSING_INDICATORS:
                        seq_values = [0.0]
                    else:
                        seq_values = []
                        for val in str(seq_str).split(SEQUENCE_SEPARATOR):
                            try:
                                seq_values.append(
                                    float(val) if val not in MISSING_INDICATORS else 0.0
                                )
                            except (ValueError, TypeError):
                                seq_values.append(0.0)
                    sequences.append(seq_values)

                # Find max sequence length for this field
                max_len = max(len(seq) for seq in sequences) if sequences else 1

                # Pad sequences to same length
                padded_sequences = []
                for seq in sequences:
                    if len(seq) < max_len:
                        seq.extend([0.0] * (max_len - len(seq)))
                    padded_sequences.append(seq[:max_len])

                num_sequences.append(np.array(padded_sequences))

        if num_sequences:
            # Stack all numerical sequences
            sequence_data["numerical"] = np.stack(num_sequences, axis=-1)

            # Add padding indicator column
            batch_size, seq_len, feature_dim = sequence_data["numerical"].shape
            padding_col = np.ones((batch_size, seq_len, 1))
            sequence_data["numerical"] = np.concatenate(
                [sequence_data["numerical"], padding_col], axis=-1
            )

            log(
                f"[INFO] Parsed numerical sequences: {sequence_data['numerical'].shape}"
            )

    return sequence_data


# --- Output Saving ---


def save_normalized_sequences(
    sequence_data: Dict[str, np.ndarray],
    output_dir: str,
    logger: Optional[Callable] = None,
) -> None:
    """Save normalized sequences in the specified format."""
    log = logger or print
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Saving normalized sequences in {OUTPUT_FORMAT} format")

    if OUTPUT_FORMAT == "numpy":
        for seq_type, seq_array in sequence_data.items():
            output_file = output_path / f"{seq_type}.npy"
            np.save(output_file, seq_array)
            log(f"[INFO] Saved {output_file} with shape {seq_array.shape}")

    elif OUTPUT_FORMAT == "parquet":
        # Convert arrays to DataFrames and save as parquet
        for seq_type, seq_array in sequence_data.items():
            if seq_array.ndim == 3:
                # Flatten 3D array to 2D for parquet storage
                batch_size, seq_len, feature_dim = seq_array.shape
                flattened = seq_array.reshape(batch_size, -1)
                df = pd.DataFrame(flattened)
                df.columns = [
                    f"{seq_type}_seq_{i}_feat_{j}"
                    for i in range(seq_len)
                    for j in range(feature_dim)
                ]
            else:
                df = pd.DataFrame(seq_array)

            output_file = output_path / f"{seq_type}.parquet"
            df.to_parquet(output_file, index=False)
            log(f"[INFO] Saved {output_file} with shape {df.shape}")

    elif OUTPUT_FORMAT == "csv":
        # Convert arrays to DataFrames and save as CSV
        for seq_type, seq_array in sequence_data.items():
            if seq_array.ndim == 3:
                # Flatten 3D array to 2D for CSV storage
                batch_size, seq_len, feature_dim = seq_array.shape
                flattened = seq_array.reshape(batch_size, -1)
                df = pd.DataFrame(flattened)
                df.columns = [
                    f"{seq_type}_seq_{i}_feat_{j}"
                    for i in range(seq_len)
                    for j in range(feature_dim)
                ]
            else:
                df = pd.DataFrame(seq_array)

            output_file = output_path / f"{seq_type}.csv"
            df.to_csv(output_file, index=False)
            log(f"[INFO] Saved {output_file} with shape {df.shape}")

    # Save metadata
    metadata = {
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_separator": SEQUENCE_SEPARATOR,
        "temporal_field": TEMPORAL_FIELD,
        "entity_id_field": ENTITY_ID_FIELD,
        "id_field": ID_FIELD,
        "output_format": OUTPUT_FORMAT,
        "include_attention_masks": INCLUDE_ATTENTION_MASKS,
        "shapes": {
            seq_type: list(seq_array.shape)
            for seq_type, seq_array in sequence_data.items()
        },
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"[INFO] Saved metadata to {metadata_file}")


# --- Main Processing Logic ---


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, np.ndarray]:
    """
    Main logic for temporal sequence normalization.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger object (defaults to print if None)

    Returns:
        Dictionary of normalized sequence arrays
    """
    # Extract configuration from environ_vars with defaults
    sequence_length = int(
        environ_vars.get("SEQUENCE_LENGTH", str(DEFAULT_SEQUENCE_LENGTH))
    )
    sequence_separator = environ_vars.get(
        "SEQUENCE_SEPARATOR", DEFAULT_SEQUENCE_SEPARATOR
    )
    temporal_field = environ_vars.get("TEMPORAL_FIELD", DEFAULT_TEMPORAL_FIELD)
    sequence_grouping_field = environ_vars.get(
        "SEQUENCE_GROUPING_FIELD", DEFAULT_SEQUENCE_GROUPING_FIELD
    )
    record_id_field = environ_vars.get("RECORD_ID_FIELD", DEFAULT_RECORD_ID_FIELD)

    # Parse JSON configuration
    missing_indicators = json.loads(
        environ_vars.get("MISSING_INDICATORS", json.dumps(DEFAULT_MISSING_INDICATORS))
    )
    time_delta_max_seconds = int(
        environ_vars.get("TIME_DELTA_MAX_SECONDS", str(DEFAULT_TIME_DELTA_MAX_SECONDS))
    )
    padding_strategy = environ_vars.get("PADDING_STRATEGY", DEFAULT_PADDING_STRATEGY)
    truncation_strategy = environ_vars.get(
        "TRUNCATION_STRATEGY", DEFAULT_TRUNCATION_STRATEGY
    )

    # Multi-sequence configuration
    enable_multi_sequence = (
        environ_vars.get(
            "ENABLE_MULTI_SEQUENCE", str(DEFAULT_ENABLE_MULTI_SEQUENCE)
        ).lower()
        == "true"
    )
    secondary_entity_field = environ_vars.get(
        "SECONDARY_ENTITY_FIELD", DEFAULT_SECONDARY_ENTITY_FIELD
    )
    sequence_naming_pattern = environ_vars.get(
        "SEQUENCE_NAMING_PATTERN", DEFAULT_SEQUENCE_NAMING_PATTERN
    )

    # Processing configuration
    validation_strategy = environ_vars.get(
        "VALIDATION_STRATEGY", DEFAULT_VALIDATION_STRATEGY
    )
    output_format = environ_vars.get("OUTPUT_FORMAT", DEFAULT_OUTPUT_FORMAT)
    include_attention_masks = (
        environ_vars.get(
            "INCLUDE_ATTENTION_MASKS", str(DEFAULT_INCLUDE_ATTENTION_MASKS)
        ).lower()
        == "true"
    )

    # Extract paths
    input_data_dir = input_paths["DATA"]
    input_signature_dir = input_paths.get("SIGNATURE", "")
    output_dir = output_paths["normalized_sequences"]

    # Use print function if no logger is provided
    log = logger or print

    # 1. Load signature columns if available
    signature_columns = (
        load_signature_columns(input_signature_dir) if input_signature_dir else None
    )
    if signature_columns:
        log(f"[INFO] Loaded signature with {len(signature_columns)} columns")
    else:
        log("[INFO] No signature file found, using default column handling")

    # 2. Combine data shards
    log(f"[INFO] Combining data shards from {input_data_dir}...")
    df = combine_shards(input_data_dir, signature_columns)
    log(f"[INFO] Combined data shape: {df.shape}")

    # 3. Process column names (handle __DOT__ replacement)
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]

    # 4. Detect sequence fields
    sequence_fields = detect_sequence_fields(
        df,
        sequence_separator,
        entity_id_field,
        secondary_entity_field,
        sequence_naming_pattern,
        enable_multi_sequence,
        temporal_field,
    )
    log(f"[INFO] Detected sequence fields: {sequence_fields}")

    # 5. Initialize processing operations with configuration
    ordering_op = SequenceOrderingOperation(temporal_field, record_id_field, logger=log)
    validation_op = DataValidationOperation(
        validation_strategy,
        temporal_field,
        record_id_field,
        missing_indicators,
        logger=log,
    )
    missing_value_op = MissingValueHandlingOperation(missing_indicators, logger=log)
    time_delta_op = TimeDeltaComputationOperation(
        temporal_field, time_delta_max_seconds, logger=log
    )
    padding_op = SequencePaddingOperation(
        sequence_length,
        padding_strategy,
        truncation_strategy,
        include_attention_masks,
        logger=log,
    )

    # 6. Apply sequence ordering
    df = ordering_op.process(df)

    # 7. Apply data validation
    df = validation_op.process(df, sequence_fields)

    # 8. Handle missing values
    df = missing_value_op.process(df, sequence_fields)

    # 9. Parse sequence data into arrays
    sequence_data = parse_sequence_data(
        df, sequence_fields, sequence_separator, missing_indicators, logger=log
    )

    # 10. Compute time deltas
    sequence_data = time_delta_op.process(sequence_data)

    # 11. Apply padding/truncation
    sequence_data = padding_op.process(sequence_data)

    # 12. Save normalized sequences
    save_normalized_sequences(
        sequence_data,
        output_dir,
        output_format,
        sequence_length,
        sequence_separator,
        temporal_field,
        entity_id_field,
        id_field,
        include_attention_masks,
        logger=log,
    )

    log("[INFO] Temporal sequence normalization complete.")
    return sequence_data


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
        INPUT_DATA_DIR = "/opt/ml/processing/input/data"
        INPUT_SIGNATURE_DIR = "/opt/ml/processing/input/signature"
        OUTPUT_DIR = "/opt/ml/processing/output"

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        # Read configuration from environment variables
        SEQUENCE_LENGTH = int(
            os.environ.get("SEQUENCE_LENGTH", str(DEFAULT_SEQUENCE_LENGTH))
        )
        SEQUENCE_SEPARATOR = os.environ.get(
            "SEQUENCE_SEPARATOR", DEFAULT_SEQUENCE_SEPARATOR
        )
        TEMPORAL_FIELD = os.environ.get("TEMPORAL_FIELD", DEFAULT_TEMPORAL_FIELD)
        SEQUENCE_GROUPING_FIELD = os.environ.get(
            "SEQUENCE_GROUPING_FIELD", DEFAULT_SEQUENCE_GROUPING_FIELD
        )
        RECORD_ID_FIELD = os.environ.get("RECORD_ID_FIELD", DEFAULT_RECORD_ID_FIELD)

        # Parse JSON configuration
        MISSING_INDICATORS = json.loads(
            os.environ.get("MISSING_INDICATORS", json.dumps(DEFAULT_MISSING_INDICATORS))
        )
        TIME_DELTA_MAX_SECONDS = int(
            os.environ.get(
                "TIME_DELTA_MAX_SECONDS", str(DEFAULT_TIME_DELTA_MAX_SECONDS)
            )
        )
        PADDING_STRATEGY = os.environ.get("PADDING_STRATEGY", DEFAULT_PADDING_STRATEGY)
        TRUNCATION_STRATEGY = os.environ.get(
            "TRUNCATION_STRATEGY", DEFAULT_TRUNCATION_STRATEGY
        )

        # Multi-sequence configuration
        ENABLE_MULTI_SEQUENCE = (
            os.environ.get(
                "ENABLE_MULTI_SEQUENCE", str(DEFAULT_ENABLE_MULTI_SEQUENCE)
            ).lower()
            == "true"
        )
        SECONDARY_ENTITY_FIELD = os.environ.get(
            "SECONDARY_ENTITY_FIELD", DEFAULT_SECONDARY_ENTITY_FIELD
        )
        SEQUENCE_NAMING_PATTERN = os.environ.get(
            "SEQUENCE_NAMING_PATTERN", DEFAULT_SEQUENCE_NAMING_PATTERN
        )

        # Processing configuration
        VALIDATION_STRATEGY = os.environ.get(
            "VALIDATION_STRATEGY", DEFAULT_VALIDATION_STRATEGY
        )
        OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", DEFAULT_OUTPUT_FORMAT)
        INCLUDE_ATTENTION_MASKS = (
            os.environ.get(
                "INCLUDE_ATTENTION_MASKS", str(DEFAULT_INCLUDE_ATTENTION_MASKS)
            ).lower()
            == "true"
        )

        # Log key parameters
        logger.info(f"Starting temporal sequence normalization with parameters:")
        logger.info(f"  Job Type: {args.job_type}")
        logger.info(f"  Sequence Length: {SEQUENCE_LENGTH}")
        logger.info(f"  Sequence Separator: '{SEQUENCE_SEPARATOR}'")
        logger.info(f"  Temporal Field: {TEMPORAL_FIELD}")
        logger.info(f"  Sequence Grouping Field: {SEQUENCE_GROUPING_FIELD}")
        logger.info(f"  Record ID Field: {RECORD_ID_FIELD}")
        logger.info(f"  Multi-Sequence Enabled: {ENABLE_MULTI_SEQUENCE}")
        logger.info(f"  Output Format: {OUTPUT_FORMAT}")
        logger.info(f"  Input Directory: {INPUT_DATA_DIR}")
        logger.info(f"  Input Signature Directory: {INPUT_SIGNATURE_DIR}")
        logger.info(f"  Output Directory: {OUTPUT_DIR}")

        # Set up path dictionaries
        input_paths = {"DATA": INPUT_DATA_DIR, "SIGNATURE": INPUT_SIGNATURE_DIR}

        output_paths = {"normalized_sequences": OUTPUT_DIR}

        # Environment variables dictionary - pass all configuration to main
        environ_vars = {
            "SEQUENCE_LENGTH": str(SEQUENCE_LENGTH),
            "SEQUENCE_SEPARATOR": SEQUENCE_SEPARATOR,
            "TEMPORAL_FIELD": TEMPORAL_FIELD,
            "SEQUENCE_GROUPING_FIELD": SEQUENCE_GROUPING_FIELD,
            "RECORD_ID_FIELD": RECORD_ID_FIELD,
            "MISSING_INDICATORS": json.dumps(MISSING_INDICATORS),
            "TIME_DELTA_MAX_SECONDS": str(TIME_DELTA_MAX_SECONDS),
            "PADDING_STRATEGY": PADDING_STRATEGY,
            "TRUNCATION_STRATEGY": TRUNCATION_STRATEGY,
            "ENABLE_MULTI_SEQUENCE": str(ENABLE_MULTI_SEQUENCE).lower(),
            "SECONDARY_ENTITY_FIELD": SECONDARY_ENTITY_FIELD,
            "SEQUENCE_NAMING_PATTERN": SEQUENCE_NAMING_PATTERN,
            "VALIDATION_STRATEGY": VALIDATION_STRATEGY,
            "OUTPUT_FORMAT": OUTPUT_FORMAT,
            "INCLUDE_ATTENTION_MASKS": str(INCLUDE_ATTENTION_MASKS).lower(),
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
        shapes_summary = ", ".join(
            [f"{name}: {arr.shape}" for name, arr in result.items()]
        )
        logger.info(
            f"Temporal sequence normalization completed successfully. Output shapes: {shapes_summary}"
        )
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error in temporal sequence normalization script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
