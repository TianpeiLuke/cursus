#!/usr/bin/env python3
"""
Pseudo Label Merge Script

Intelligently merges original labeled training data with pseudo-labeled or augmented samples
for Semi-Supervised Learning (SSL) and Active Learning workflows.

Key Features:
- Split-aware merge for training jobs (maintains train/test/val boundaries)
- Auto-inferred split ratios (adapts to base data proportions)
- Simple merge for validation/testing jobs
- Data format preservation (CSV/TSV/Parquet)
- Schema alignment and provenance tracking

Design: slipbox/1_design/pseudo_label_merge_script_design.md
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading Component
# ============================================================================


def _detect_file_format(file_path: Path) -> str:
    """
    Detect the format of a data file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        Format string: 'csv', 'tsv', or 'parquet'
    """
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return "csv"
    elif suffix == ".tsv":
        return "tsv"
    elif suffix == ".parquet":
        return "parquet"
    else:
        raise RuntimeError(f"Unsupported file format: {suffix}")


def load_dataframe_with_format(file_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Load DataFrame and detect its format.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (DataFrame, format_string)
    """
    detected_format = _detect_file_format(file_path)

    if detected_format == "csv":
        df = pd.read_csv(file_path)
    elif detected_format == "tsv":
        df = pd.read_csv(file_path, sep="\t")
    elif detected_format == "parquet":
        df = pd.read_parquet(file_path)
    else:
        raise RuntimeError(f"Unsupported format: {detected_format}")

    logger.info(f"Loaded {len(df)} rows from {file_path} (format: {detected_format})")
    return df, detected_format


def save_dataframe_with_format(
    df: pd.DataFrame, output_path: Path, format_str: str
) -> Path:
    """
    Save DataFrame in specified format.

    Args:
        df: DataFrame to save
        output_path: Base output path (without extension)
        format_str: Format to save in ('csv', 'tsv', or 'parquet')

    Returns:
        Path to saved file
    """
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

    logger.info(f"Saved {len(df)} rows to {file_path} (format: {format_str})")
    return file_path


def _read_file(file_path: Path) -> pd.DataFrame:
    """Read file based on extension (backward compatibility wrapper)."""
    df, _ = load_dataframe_with_format(file_path)
    return df


def _load_split_data(split_dir: Path, split_name: str) -> pd.DataFrame:
    """
    Load data from a specific split directory.

    Expected file patterns:
    - {split_name}_processed_data.csv
    - {split_name}_processed_data.tsv
    - {split_name}_processed_data.parquet
    - part-*.csv (for sharded data)
    - part-*.parquet (for sharded data)
    """
    # Try specific file names first
    for ext in [".csv", ".tsv", ".parquet"]:
        file_path = split_dir / f"{split_name}_processed_data{ext}"
        if file_path.exists():
            return _read_file(file_path)

    # Check for sharded data (same logic as _load_single_dataset)
    csv_shards = list(split_dir.glob("part-*.csv"))
    parquet_shards = list(split_dir.glob("part-*.parquet"))

    if csv_shards or parquet_shards:
        # Combine shards
        all_shards = sorted(csv_shards + parquet_shards)
        dfs = [_read_file(shard) for shard in all_shards]
        logger.info(f"Combining {len(all_shards)} shard files from {split_dir}")
        return pd.concat(dfs, ignore_index=True)

    # Fall back to pattern matching for single files
    csv_files = list(split_dir.glob("*.csv"))
    tsv_files = list(split_dir.glob("*.tsv"))
    parquet_files = list(split_dir.glob("*.parquet"))

    if parquet_files:
        return _read_file(parquet_files[0])
    elif csv_files:
        return _read_file(csv_files[0])
    elif tsv_files:
        return _read_file(tsv_files[0])
    else:
        raise FileNotFoundError(f"No data files found in {split_dir}")


def _load_single_dataset(data_dir: Path) -> pd.DataFrame:
    """Load data from directory (may contain shards)."""
    # Check for sharded data
    csv_shards = list(data_dir.glob("part-*.csv"))
    parquet_shards = list(data_dir.glob("part-*.parquet"))

    if csv_shards or parquet_shards:
        # Combine shards
        all_shards = sorted(csv_shards + parquet_shards)
        dfs = [_read_file(shard) for shard in all_shards]
        return pd.concat(dfs, ignore_index=True)

    # Single file
    for pattern in ["*.parquet", "*.csv", "*.tsv"]:
        files = list(data_dir.glob(pattern))
        if files:
            return _read_file(files[0])

    raise FileNotFoundError(f"No data files found in {data_dir}")


def load_base_data(
    base_data_dir: str,
    job_type: str,
) -> Dict[str, pd.DataFrame]:
    """
    Load base training data, detecting split structure automatically.

    Args:
        base_data_dir: Path to base data directory
        job_type: Job type (training, validation, testing, calibration)

    Returns:
        Dictionary mapping split names to DataFrames
        - Training job: {"train": df, "test": df, "val": df}
        - Other jobs: {job_type: df}
    """
    base_path = Path(base_data_dir)

    # Check for split structure (training job)
    if job_type == "training":
        # Look for train/test/val subdirectories
        train_dir = base_path / "train"
        test_dir = base_path / "test"
        val_dir = base_path / "val"

        if train_dir.exists() and test_dir.exists() and val_dir.exists():
            logger.info("Detected split structure (train/test/val)")
            return {
                "train": _load_split_data(train_dir, "train"),
                "test": _load_split_data(test_dir, "test"),
                "val": _load_split_data(val_dir, "val"),
            }
        else:
            # Fall back to single dataset
            logger.warning(
                f"Expected train/test/val splits for training job, "
                f"but found single dataset. Using simple merge."
            )
            return {job_type: _load_single_dataset(base_path)}
    else:
        # Non-training jobs: single dataset in job_type subdirectory
        job_dir = base_path / job_type
        if job_dir.exists():
            logger.info(f"Loading single dataset from {job_type} directory")
            return {job_type: _load_split_data(job_dir, job_type)}
        else:
            # Fall back to root directory
            logger.info(f"Loading single dataset from root directory")
            return {job_type: _load_single_dataset(base_path)}


def load_augmentation_data(aug_data_dir: str) -> pd.DataFrame:
    """
    Load augmentation data (always single dataset).

    Args:
        aug_data_dir: Path to augmentation data directory

    Returns:
        DataFrame with augmentation samples
    """
    aug_path = Path(aug_data_dir)

    # Try common file names
    for filename in ["selected_samples", "predictions", "labeled_data"]:
        for ext in [".parquet", ".csv", ".tsv"]:
            file_path = aug_path / f"{filename}{ext}"
            if file_path.exists():
                logger.info(f"Loading augmentation data from {file_path}")
                return _read_file(file_path)

    # Fall back to any data file
    return _load_single_dataset(aug_path)


# ============================================================================
# Split Detection Component
# ============================================================================


def detect_merge_strategy(
    base_splits: Dict[str, pd.DataFrame],
    job_type: str,
) -> str:
    """
    Determine merge strategy based on input structure.

    Args:
        base_splits: Dictionary of base data splits
        job_type: Job type

    Returns:
        "split_aware" or "simple"
    """
    # Training job with 3 splits → split-aware merge
    if job_type == "training" and set(base_splits.keys()) == {"train", "test", "val"}:
        logger.info("Using split-aware merge strategy")
        return "split_aware"

    # All other cases → simple merge
    logger.info("Using simple merge strategy")
    return "simple"


def extract_split_ratios(
    base_splits: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """
    Calculate split proportions from base data.

    Args:
        base_splits: Dictionary with train/test/val splits

    Returns:
        Dictionary with split proportions summing to 1.0
    """
    total = sum(len(df) for df in base_splits.values())

    ratios = {name: len(df) / total for name, df in base_splits.items()}

    logger.info(f"Extracted split ratios from base data: {ratios}")
    return ratios


# ============================================================================
# Schema Alignment Component
# ============================================================================


def _infer_common_dtype(dtype1, dtype2):
    """Infer common data type for two columns."""
    # Numeric types
    if pd.api.types.is_numeric_dtype(dtype1) and pd.api.types.is_numeric_dtype(dtype2):
        # Use float if either is float
        if pd.api.types.is_float_dtype(dtype1) or pd.api.types.is_float_dtype(dtype2):
            return "float64"
        else:
            return "int64"

    # String types
    if pd.api.types.is_string_dtype(dtype1) or pd.api.types.is_string_dtype(dtype2):
        return "object"

    # Default: use first type
    return dtype1


def align_schemas(
    base_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    label_field: str,
    pseudo_label_column: str = "pseudo_label",
    id_field: str = "id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align schemas between base and augmentation data.

    Handles:
    - Label column conversion (pseudo_label → label)
    - Common columns extraction
    - Data type compatibility

    Args:
        base_df: Base training data
        aug_df: Augmentation data
        label_field: Label column name
        pseudo_label_column: Pseudo-label column name in augmentation
        id_field: ID column name

    Returns:
        Tuple of (aligned_base_df, aligned_aug_df)
    """
    # Copy to avoid modifying originals
    base_aligned = base_df.copy()
    aug_aligned = aug_df.copy()

    # Handle label column conversion
    if pseudo_label_column in aug_aligned.columns:
        if label_field not in aug_aligned.columns:
            # Convert pseudo_label to label
            aug_aligned[label_field] = aug_aligned[pseudo_label_column]
            logger.info(
                f"Converted '{pseudo_label_column}' to '{label_field}' "
                f"in augmentation data"
            )
        # Drop pseudo_label column to avoid duplication
        aug_aligned = aug_aligned.drop(columns=[pseudo_label_column])

    # Ensure label field exists in augmentation data
    if label_field not in aug_aligned.columns:
        raise ValueError(
            f"Label field '{label_field}' not found in augmentation data. "
            f"Available columns: {aug_aligned.columns.tolist()}"
        )

    # Find common columns
    common_columns = sorted(set(base_aligned.columns) & set(aug_aligned.columns))

    # Ensure essential columns are present
    if id_field not in common_columns:
        logger.warning(f"ID field '{id_field}' not in common columns")

    if label_field not in common_columns:
        raise ValueError(f"Label field '{label_field}' not found in both datasets")

    logger.info(
        f"Schema alignment: {len(common_columns)} common columns "
        f"(base: {len(base_aligned.columns)}, aug: {len(aug_aligned.columns)})"
    )

    # Select only common columns
    base_aligned = base_aligned[common_columns]
    aug_aligned = aug_aligned[common_columns]

    # Align data types
    for col in common_columns:
        if base_aligned[col].dtype != aug_aligned[col].dtype:
            # Try to convert to common type
            try:
                common_type = _infer_common_dtype(
                    base_aligned[col].dtype, aug_aligned[col].dtype
                )
                base_aligned[col] = base_aligned[col].astype(common_type)
                aug_aligned[col] = aug_aligned[col].astype(common_type)
                logger.debug(f"Aligned dtype for column '{col}': {common_type}")
            except Exception as e:
                logger.warning(f"Could not align dtype for column '{col}': {e}")

    return base_aligned, aug_aligned


# ============================================================================
# Merge Strategy Component
# ============================================================================


def _split_by_ratios(
    df: pd.DataFrame,
    ratios: Dict[str, float],
    label_field: str,
    stratify: bool = True,
    random_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Split DataFrame into three parts using specified ratios.

    This is the core function for auto-inferred split distribution.

    Args:
        df: DataFrame to split
        ratios: Dictionary with split ratios (e.g., {"train": 0.7, "test": 0.15, "val": 0.15})
        label_field: Label column for stratification
        stratify: Use stratified splits if True
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping split names to DataFrames
    """
    # Ensure ratios sum to 1.0
    total_ratio = sum(ratios.values())
    if not abs(total_ratio - 1.0) < 1e-6:
        logger.warning(f"Ratios sum to {total_ratio}, normalizing to 1.0")
        ratios = {k: v / total_ratio for k, v in ratios.items()}

    # For three-way split, we need to do two sequential splits
    # First split: train vs (test+val)
    train_ratio = ratios.get("train", 0.7)
    test_ratio = ratios.get("test", 0.15)
    val_ratio = ratios.get("val", 0.15)

    if stratify and label_field in df.columns:
        # Stratified three-way split
        train_df, holdout_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=df[label_field],
        )

        # Second split: test vs val from holdout
        # Calculate test proportion of holdout
        test_proportion = test_ratio / (test_ratio + val_ratio)

        test_df, val_df = train_test_split(
            holdout_df,
            train_size=test_proportion,
            random_state=random_seed,
            stratify=holdout_df[label_field],
        )
    else:
        # Non-stratified three-way split
        train_df, holdout_df = train_test_split(
            df, train_size=train_ratio, random_state=random_seed
        )

        test_proportion = test_ratio / (test_ratio + val_ratio)

        test_df, val_df = train_test_split(
            holdout_df, train_size=test_proportion, random_state=random_seed
        )

    logger.info(
        f"Split by ratios: train={len(train_df)} ({train_ratio:.1%}), "
        f"test={len(test_df)} ({test_ratio:.1%}), "
        f"val={len(val_df)} ({val_ratio:.1%})"
    )

    return {
        "train": train_df,
        "test": test_df,
        "val": val_df,
    }


def merge_with_splits(
    base_splits: Dict[str, pd.DataFrame],
    augmentation_df: pd.DataFrame,
    label_field: str,
    use_auto_split_ratios: bool = True,
    train_ratio: Optional[float] = None,
    test_val_ratio: Optional[float] = None,
    stratify: bool = True,
    random_seed: int = 42,
    preserve_confidence: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Merge with proportional augmentation distribution across splits.

    Strategy:
    1. Auto-infer split ratios from base data (or use manual ratios if provided)
    2. Split augmentation data using calculated proportions
    3. Add provenance to all datasets
    4. Merge corresponding splits
    5. Return merged splits maintaining structure

    Args:
        base_splits: Dictionary with train/test/val DataFrames
        augmentation_df: Augmentation data to distribute
        label_field: Label column name for stratification
        use_auto_split_ratios: Auto-infer ratios from base data (recommended)
        train_ratio: Optional proportion for train split (None = auto-infer from base)
        test_val_ratio: Optional test vs val proportion of holdout (None = auto-infer)
        stratify: Use stratified splits if True
        random_seed: Random seed for reproducibility
        preserve_confidence: Keep confidence scores if present

    Returns:
        Dictionary with merged train/test/val DataFrames
    """
    logger.info(
        f"Starting split-aware merge: "
        f"base splits={list(base_splits.keys())}, "
        f"augmentation samples={len(augmentation_df)}"
    )

    # Determine split strategy: auto-infer or manual
    if use_auto_split_ratios or train_ratio is None:
        # Auto-infer ratios from base data (RECOMMENDED)
        logger.info("Auto-inferring split ratios from base data")
        ratios = extract_split_ratios(base_splits)
        logger.info(f"Auto-inferred ratios: {ratios}")

        # Split augmentation using actual base ratios
        aug_splits = _split_by_ratios(
            augmentation_df,
            ratios=ratios,
            label_field=label_field,
            stratify=stratify,
            random_seed=random_seed,
        )
    else:
        # Use manual ratios (backward compatibility)
        logger.info(
            f"Using manual ratios: train={train_ratio}, test_val={test_val_ratio}"
        )

        # Two-step split: train vs holdout, then test vs val
        if stratify and label_field in augmentation_df.columns:
            aug_train, aug_holdout = train_test_split(
                augmentation_df,
                train_size=train_ratio,
                random_state=random_seed,
                stratify=augmentation_df[label_field],
            )

            aug_test, aug_val = train_test_split(
                aug_holdout,
                test_size=test_val_ratio,
                random_state=random_seed,
                stratify=aug_holdout[label_field],
            )
        else:
            aug_train, aug_holdout = train_test_split(
                augmentation_df, train_size=train_ratio, random_state=random_seed
            )

            aug_test, aug_val = train_test_split(
                aug_holdout, test_size=test_val_ratio, random_state=random_seed
            )

        aug_splits = {
            "train": aug_train,
            "test": aug_test,
            "val": aug_val,
        }

    logger.info(
        f"Augmentation split sizes: "
        f"train={len(aug_splits['train'])}, "
        f"test={len(aug_splits['test'])}, "
        f"val={len(aug_splits['val'])}"
    )

    # Merge each split with provenance
    merged_splits = {}
    for split_name in ["train", "test", "val"]:
        base_df = base_splits[split_name].copy()
        aug_df = aug_splits[split_name].copy()

        # Add provenance
        base_df["data_source"] = "original"
        aug_df["data_source"] = "pseudo_labeled"

        # Remove confidence columns from base if not preserving
        if not preserve_confidence:
            confidence_cols = [
                col
                for col in aug_df.columns
                if "confidence" in col.lower() or "score" in col.lower()
            ]
            if confidence_cols:
                aug_df = aug_df.drop(columns=confidence_cols)

        # Combine
        merged_df = pd.concat([base_df, aug_df], ignore_index=True)

        logger.info(
            f"Merged {split_name}: "
            f"base={len(base_df)}, aug={len(aug_df)}, "
            f"total={len(merged_df)}"
        )

        merged_splits[split_name] = merged_df

    return merged_splits


def merge_simple(
    base_df: pd.DataFrame,
    augmentation_df: pd.DataFrame,
    preserve_confidence: bool = True,
) -> pd.DataFrame:
    """
    Simple merge for non-training job types.

    Args:
        base_df: Base dataset
        augmentation_df: Augmentation dataset
        preserve_confidence: Keep confidence scores if present

    Returns:
        Merged DataFrame with provenance
    """
    logger.info(
        f"Starting simple merge: "
        f"base samples={len(base_df)}, "
        f"augmentation samples={len(augmentation_df)}"
    )

    base_merged = base_df.copy()
    aug_merged = augmentation_df.copy()

    # Add provenance
    base_merged["data_source"] = "original"
    aug_merged["data_source"] = "pseudo_labeled"

    # Remove confidence columns from augmentation if not preserving
    if not preserve_confidence:
        confidence_cols = [
            col
            for col in aug_merged.columns
            if "confidence" in col.lower() or "score" in col.lower()
        ]
        if confidence_cols:
            aug_merged = aug_merged.drop(columns=confidence_cols)
            logger.info(f"Removed confidence columns: {confidence_cols}")

    # Combine
    merged_df = pd.concat([base_merged, aug_merged], ignore_index=True)

    logger.info(
        f"Merge complete: total={len(merged_df)} "
        f"(original={len(base_df)}, pseudo={len(augmentation_df)})"
    )

    return merged_df


# ============================================================================
# Provenance Tracking Component
# ============================================================================


def validate_provenance(
    merged_df: pd.DataFrame,
    expected_sources: set = {"original", "pseudo_labeled"},
) -> bool:
    """Validate provenance column in merged data."""
    if "data_source" not in merged_df.columns:
        logger.warning("Provenance column 'data_source' not found")
        return False

    actual_sources = set(merged_df["data_source"].unique())

    if not actual_sources.issubset(expected_sources):
        logger.warning(f"Unexpected data sources: {actual_sources - expected_sources}")
        return False

    logger.info(f"Provenance validation passed: {actual_sources}")
    return True


# ============================================================================
# Output Management Component
# ============================================================================


def save_merged_data(
    merged_splits: Dict[str, pd.DataFrame],
    output_dir: str,
    output_format: str = "csv",
    job_type: str = "training",
) -> Dict[str, str]:
    """
    Save merged data maintaining input structure.

    Args:
        merged_splits: Dictionary of merged DataFrames by split
        output_dir: Output directory path
        output_format: "csv", "tsv", or "parquet"
        job_type: Job type for file naming

    Returns:
        Dictionary mapping split names to output file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    for split_name, split_df in merged_splits.items():
        # Create split subdirectory
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)

        # Determine file name and extension
        if output_format.lower() == "parquet":
            filename = f"{split_name}_processed_data.parquet"
            file_path = split_dir / filename
            split_df.to_parquet(file_path, index=False)

        elif output_format.lower() == "tsv":
            filename = f"{split_name}_processed_data.tsv"
            file_path = split_dir / filename
            split_df.to_csv(file_path, sep="\t", index=False)

        else:  # default: csv
            filename = f"{split_name}_processed_data.csv"
            file_path = split_dir / filename
            split_df.to_csv(file_path, index=False)

        output_paths[split_name] = str(file_path)
        logger.info(
            f"Saved {split_name} split: {file_path} "
            f"(format={output_format}, shape={split_df.shape})"
        )

    return output_paths


def save_merge_metadata(
    output_dir: str,
    metadata: Dict[str, Any],
) -> str:
    """
    Save merge operation metadata.

    Args:
        output_dir: Output directory
        metadata: Metadata dictionary

    Returns:
        Path to metadata file
    """
    metadata_path = Path(output_dir) / "merge_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved merge metadata: {metadata_path}")
    return str(metadata_path)


# ============================================================================
# Main Function (Testability Interface)
# ============================================================================


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> Dict[str, pd.DataFrame]:
    """
    Main function for pseudo label merge.

    Args:
        input_paths: Dictionary with keys:
            - base_data: Path to base labeled data
            - augmentation_data: Path to augmentation data
        output_paths: Dictionary with keys:
            - merged_data: Path for merged output
        environ_vars: Dictionary with environment variables:
            - LABEL_FIELD: Label column name (REQUIRED)
            - ADD_PROVENANCE: Track data source (default: "true")
            - OUTPUT_FORMAT: Output format (default: "csv")
            - USE_AUTO_SPLIT_RATIOS: Auto-infer split ratios (default: "true")
            - TRAIN_RATIO: Train split proportion (default: None)
            - TEST_VAL_RATIO: Test vs val proportion (default: None)
            - PSEUDO_LABEL_COLUMN: Pseudo-label column name (default: "pseudo_label")
            - ID_FIELD: ID column name (default: "id")
            - PRESERVE_CONFIDENCE: Keep confidence scores (default: "true")
            - STRATIFY: Use stratified splits (default: "true")
            - RANDOM_SEED: Random seed (default: "42")
        job_args: Command-line arguments:
            - job_type: Type of merge job (training, validation, testing, calibration)

    Returns:
        Dictionary of merged DataFrames by split name
    """
    # Extract configuration
    job_type = job_args.job_type
    label_field = environ_vars.get("LABEL_FIELD")
    if not label_field:
        raise ValueError("LABEL_FIELD environment variable is required")

    add_provenance = environ_vars.get("ADD_PROVENANCE", "true").lower() == "true"
    output_format = environ_vars.get("OUTPUT_FORMAT", "csv")
    use_auto_split_ratios = (
        environ_vars.get("USE_AUTO_SPLIT_RATIOS", "true").lower() == "true"
    )

    # Parse split ratios (None if not provided or if auto-inference enabled)
    train_ratio_str = environ_vars.get("TRAIN_RATIO")
    train_ratio = (
        float(train_ratio_str)
        if train_ratio_str and not use_auto_split_ratios
        else None
    )

    test_val_ratio_str = environ_vars.get("TEST_VAL_RATIO")
    test_val_ratio = (
        float(test_val_ratio_str)
        if test_val_ratio_str and not use_auto_split_ratios
        else None
    )

    pseudo_label_column = environ_vars.get("PSEUDO_LABEL_COLUMN", "pseudo_label")
    id_field = environ_vars.get("ID_FIELD", "id")
    preserve_confidence = (
        environ_vars.get("PRESERVE_CONFIDENCE", "true").lower() == "true"
    )
    stratify = environ_vars.get("STRATIFY", "true").lower() == "true"
    random_seed = int(environ_vars.get("RANDOM_SEED", "42"))

    logger.info(f"Starting pseudo label merge with parameters:")
    logger.info(f"  Job Type: {job_type}")
    logger.info(f"  Label Field: {label_field}")
    logger.info(f"  Use Auto Split Ratios: {use_auto_split_ratios}")
    if not use_auto_split_ratios and train_ratio:
        logger.info(f"  Train Ratio: {train_ratio}")
        logger.info(f"  Test/Val Ratio: {test_val_ratio}")
    logger.info(f"  Output Format: {output_format}")

    # Load base training data
    logger.info(f"Loading base data from {input_paths['base_data']}")
    base_splits = load_base_data(
        base_data_dir=input_paths["base_data"],
        job_type=job_type,
    )
    logger.info(f"Loaded base data with splits: {list(base_splits.keys())}")

    # Load augmentation data
    logger.info(f"Loading augmentation data from {input_paths['augmentation_data']}")
    augmentation_df = load_augmentation_data(
        aug_data_dir=input_paths["augmentation_data"]
    )
    logger.info(f"Loaded {len(augmentation_df)} augmentation samples")

    # Detect merge strategy
    merge_strategy = detect_merge_strategy(base_splits, job_type)
    logger.info(f"Selected merge strategy: {merge_strategy}")

    # Perform merge based on strategy
    if merge_strategy == "split_aware":
        # Split-aware merge for training job
        logger.info("Performing split-aware merge")

        # Align schemas for each split (use first split as reference)
        first_split = list(base_splits.keys())[0]
        base_aligned, aug_aligned = align_schemas(
            base_df=base_splits[first_split],
            aug_df=augmentation_df,
            label_field=label_field,
            pseudo_label_column=pseudo_label_column,
            id_field=id_field,
        )
        # Use aligned augmentation for split distribution
        augmentation_df = aug_aligned

        # Perform split-aware merge
        merged_splits = merge_with_splits(
            base_splits=base_splits,
            augmentation_df=augmentation_df,
            label_field=label_field,
            use_auto_split_ratios=use_auto_split_ratios,
            train_ratio=train_ratio,
            test_val_ratio=test_val_ratio,
            stratify=stratify,
            random_seed=random_seed,
            preserve_confidence=preserve_confidence,
        )
    else:
        # Simple merge for non-training jobs
        logger.info("Performing simple merge")

        # Get single split
        split_name = list(base_splits.keys())[0]
        base_df = base_splits[split_name]

        # Align schemas
        base_aligned, aug_aligned = align_schemas(
            base_df=base_df,
            aug_df=augmentation_df,
            label_field=label_field,
            pseudo_label_column=pseudo_label_column,
            id_field=id_field,
        )

        # Perform simple merge
        merged_df = merge_simple(
            base_df=base_aligned,
            augmentation_df=aug_aligned,
            preserve_confidence=preserve_confidence,
        )

        merged_splits = {split_name: merged_df}

    # Validate provenance if added
    if add_provenance:
        for split_name, merged_df in merged_splits.items():
            validate_provenance(merged_df)

    # Save merged data
    logger.info(f"Saving merged data to {output_paths['merged_data']}")
    output_file_paths = save_merged_data(
        merged_splits=merged_splits,
        output_dir=output_paths["merged_data"],
        output_format=output_format,
        job_type=job_type,
    )

    # Generate and save metadata
    metadata = {
        "job_type": job_type,
        "merge_strategy": merge_strategy,
        "base_splits": {
            name: {"count": len(df), "shape": list(df.shape)}
            for name, df in base_splits.items()
        },
        "augmentation_count": len(augmentation_df),
        "merged_splits": {
            name: {"count": len(df), "shape": list(df.shape)}
            for name, df in merged_splits.items()
        },
        "configuration": {
            "label_field": label_field,
            "use_auto_split_ratios": use_auto_split_ratios,
            "train_ratio": train_ratio,
            "test_val_ratio": test_val_ratio,
            "stratify": stratify,
            "preserve_confidence": preserve_confidence,
            "random_seed": random_seed,
        },
        "output_paths": output_file_paths,
        "timestamp": datetime.now().isoformat(),
    }

    save_merge_metadata(
        output_dir=output_paths["merged_data"],
        metadata=metadata,
    )

    logger.info("Pseudo label merge complete")
    return merged_splits


# ============================================================================
# Container Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    import traceback

    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Pseudo label merge")
        parser.add_argument(
            "--job_type",
            type=str,
            required=True,
            choices=["training", "validation", "testing", "calibration"],
            help="Type of merge job",
        )
        args = parser.parse_args()

        # Read configuration from environment variables
        LABEL_FIELD = os.environ.get("LABEL_FIELD")
        if not LABEL_FIELD:
            raise ValueError("LABEL_FIELD environment variable is required")

        ADD_PROVENANCE = os.environ.get("ADD_PROVENANCE", "true")
        OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "csv")
        USE_AUTO_SPLIT_RATIOS = os.environ.get("USE_AUTO_SPLIT_RATIOS", "true")
        TRAIN_RATIO = os.environ.get("TRAIN_RATIO")
        TEST_VAL_RATIO = os.environ.get("TEST_VAL_RATIO")
        PSEUDO_LABEL_COLUMN = os.environ.get("PSEUDO_LABEL_COLUMN", "pseudo_label")
        ID_FIELD = os.environ.get("ID_FIELD", "id")
        PRESERVE_CONFIDENCE = os.environ.get("PRESERVE_CONFIDENCE", "true")
        STRATIFY = os.environ.get("STRATIFY", "true")
        RANDOM_SEED = os.environ.get("RANDOM_SEED", "42")

        # Define standard SageMaker paths as constants
        BASE_DATA_DIR = "/opt/ml/processing/input/base_data"
        AUGMENTATION_DATA_DIR = "/opt/ml/processing/input/augmentation_data"
        MERGED_DATA_DIR = "/opt/ml/processing/output/merged_data"

        # Set up logging (reconfigure with more detail for main execution)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,  # Override the earlier basic config
        )
        logger_main = logging.getLogger(__name__)

        # Log key parameters
        logger_main.info(f"Starting pseudo label merge with parameters:")
        logger_main.info(f"  Job Type: {args.job_type}")
        logger_main.info(f"  Label Field: {LABEL_FIELD}")
        logger_main.info(f"  Use Auto Split Ratios: {USE_AUTO_SPLIT_RATIOS}")
        if USE_AUTO_SPLIT_RATIOS.lower() != "true" and TRAIN_RATIO:
            logger_main.info(f"  Train Ratio: {TRAIN_RATIO}")
            logger_main.info(f"  Test/Val Ratio: {TEST_VAL_RATIO}")
        logger_main.info(f"  Output Format: {OUTPUT_FORMAT}")
        logger_main.info(f"  Base Data Directory: {BASE_DATA_DIR}")
        logger_main.info(f"  Augmentation Data Directory: {AUGMENTATION_DATA_DIR}")
        logger_main.info(f"  Merged Data Directory: {MERGED_DATA_DIR}")

        # Set up path dictionaries
        input_paths = {
            "base_data": BASE_DATA_DIR,
            "augmentation_data": AUGMENTATION_DATA_DIR,
        }

        output_paths = {
            "merged_data": MERGED_DATA_DIR,
        }

        # Environment variables dictionary
        environ_vars = {
            "LABEL_FIELD": LABEL_FIELD,
            "ADD_PROVENANCE": ADD_PROVENANCE,
            "OUTPUT_FORMAT": OUTPUT_FORMAT,
            "USE_AUTO_SPLIT_RATIOS": USE_AUTO_SPLIT_RATIOS,
            "TRAIN_RATIO": TRAIN_RATIO,
            "TEST_VAL_RATIO": TEST_VAL_RATIO,
            "PSEUDO_LABEL_COLUMN": PSEUDO_LABEL_COLUMN,
            "ID_FIELD": ID_FIELD,
            "PRESERVE_CONFIDENCE": PRESERVE_CONFIDENCE,
            "STRATIFY": STRATIFY,
            "RANDOM_SEED": RANDOM_SEED,
        }

        # Ensure output directory exists
        os.makedirs(output_paths["merged_data"], exist_ok=True)

        # Execute the main processing logic
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
        )

        # Log completion summary
        splits_summary = ", ".join(
            [f"{name}: {df.shape}" for name, df in result.items()]
        )
        logger_main.info(f"Merge completed successfully. Splits: {splits_summary}")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error in pseudo label merge script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
