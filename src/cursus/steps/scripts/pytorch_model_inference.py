#!/usr/bin/env python
"""
PyTorch Model Batch Inference Script

Performs BATCH INFERENCE ONLY (no evaluation, no metrics) for trained PyTorch Lightning models.
Optimized for large-scale batch predictions without labels.

Features:
- GPU/CPU automatic detection and explicit control
- Multi-modal model support (text, tabular, bimodal, trimodal)
- Format preservation (CSV/TSV/Parquet)
- Multi-GPU support with synchronization
- Predictions only (no metrics, no plots, no evaluation)

Key Differences from pytorch_model_eval.py:
- ❌ NO label column required in input data
- ❌ NO metrics computation (AUC, F1, etc.)
- ❌ NO plot generation (ROC, PR curves)
- ❌ NO comparison mode
- ✅ Only predictions saved
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import sys
import tarfile
import logging
from datetime import datetime
import time
import fcntl
import hashlib
import tempfile

# ============================================================================
# PACKAGE INSTALLATION CONFIGURATION
# ============================================================================

# Control which PyPI source to use via environment variable
# Set USE_SECURE_PYPI=true to use secure CodeArtifact PyPI
# Set USE_SECURE_PYPI=false or leave unset to use public PyPI
USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "false").lower() == "true"

# Logging setup for installation (uses logger configured below)
from subprocess import check_call
import boto3


def _get_secure_pypi_access_token() -> str:
    """
    Get CodeArtifact access token for secure PyPI.

    Returns:
        str: Authorization token for CodeArtifact

    Raises:
        Exception: If token retrieval fails
    """
    try:
        os.environ["AWS_STS_REGIONAL_ENDPOINTS"] = "regional"
        sts = boto3.client("sts", region_name="us-east-1")
        caller_identity = sts.get_caller_identity()
        assumed_role_object = sts.assume_role(
            RoleArn="arn:aws:iam::675292366480:role/SecurePyPIReadRole_"
            + caller_identity["Account"],
            RoleSessionName="SecurePypiReadRole",
        )
        credentials = assumed_role_object["Credentials"]
        code_artifact_client = boto3.client(
            "codeartifact",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            region_name="us-west-2",
        )
        token = code_artifact_client.get_authorization_token(
            domain="amazon", domainOwner="149122183214"
        )["authorizationToken"]

        print("✓ Successfully retrieved secure PyPI access token")
        return token

    except Exception as e:
        print(f"✗ Failed to retrieve secure PyPI access token: {e}")
        raise


def install_packages_from_public_pypi(packages: list) -> None:
    """
    Install packages from standard public PyPI.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
    """
    print(f"Installing {len(packages)} packages from public PyPI")
    print(f"Packages: {packages}")

    try:
        check_call([sys.executable, "-m", "pip", "install", *packages])
        print("✓ Successfully installed packages from public PyPI")
    except Exception as e:
        print(f"✗ Failed to install packages from public PyPI: {e}")
        raise


def install_packages_from_secure_pypi(packages: list) -> None:
    """
    Install packages from secure CodeArtifact PyPI.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
    """
    print(f"Installing {len(packages)} packages from secure PyPI")
    print(f"Packages: {packages}")

    try:
        token = _get_secure_pypi_access_token()
        index_url = f"https://aws:{token}@amazon-149122183214.d.codeartifact.us-west-2.amazonaws.com/pypi/secure-pypi/simple/"

        check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--index-url",
                index_url,
                *packages,
            ]
        )

        print("✓ Successfully installed packages from secure PyPI")
    except Exception as e:
        print(f"✗ Failed to install packages from secure PyPI: {e}")
        raise


def install_packages(packages: list, use_secure: bool = USE_SECURE_PYPI) -> None:
    """
    Install packages from PyPI source based on configuration.

    This is the main installation function that delegates to either public or
    secure PyPI based on the USE_SECURE_PYPI environment variable.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
        use_secure: If True, use secure CodeArtifact PyPI; if False, use public PyPI.
                   Defaults to USE_SECURE_PYPI environment variable.

    Environment Variables:
        USE_SECURE_PYPI: Set to "true" to use secure PyPI, "false" for public PyPI

    Example:
        # Install from public PyPI (default)
        install_packages(["pandas==1.5.0", "numpy"])

        # Install from secure PyPI
        os.environ["USE_SECURE_PYPI"] = "true"
        install_packages(["pandas==1.5.0", "numpy"])
    """
    print("=" * 70)
    print("PACKAGE INSTALLATION")
    print("=" * 70)
    print(f"PyPI Source: {'SECURE (CodeArtifact)' if use_secure else 'PUBLIC'}")
    print(
        f"Environment Variable USE_SECURE_PYPI: {os.environ.get('USE_SECURE_PYPI', 'not set')}"
    )
    print(f"Number of packages: {len(packages)}")
    print("=" * 70)

    try:
        if use_secure:
            install_packages_from_secure_pypi(packages)
        else:
            install_packages_from_public_pypi(packages)

        print("=" * 70)
        print("✓ PACKAGE INSTALLATION COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print("=" * 70)
        print("✗ PACKAGE INSTALLATION FAILED")
        print("=" * 70)
        raise


# ============================================================================
# INSTALL REQUIRED PACKAGES
# ============================================================================

# Load packages from requirements-secure.txt
requirements_file = os.path.join(os.path.dirname(__file__), "requirements-secure.txt")

try:
    with open(requirements_file, "r") as f:
        # Read lines, strip whitespace, and filter out comments and empty lines
        required_packages = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    print(f"Loaded {len(required_packages)} packages from {requirements_file}")

    # Install packages using unified installation function
    install_packages(required_packages)

    print("***********************Package Installation Complete*********************")

except FileNotFoundError:
    print(f"Warning: {requirements_file} not found. Skipping package installation.")
    print("Assuming packages are already installed in the environment.")
except Exception as e:
    print(f"Error loading or installing packages: {e}")
    raise

# ============================================================================
# IMPORT INSTALLED PACKAGES (AFTER INSTALLATION)
# ============================================================================

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch import nn

from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

# Import processing modules from bsm_pytorch
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../projects/bsm_pytorch/docker")
)

from ...processing.text.dialogue_processor import (
    HTMLNormalizerProcessor,
    EmojiRemoverProcessor,
    TextNormalizationProcessor,
    DialogueSplitterProcessor,
    DialogueChunkerProcessor,
)
from ...processing.text.bert_tokenize_processor import BertTokenizeProcessor
from ...processing.categorical.risk_table_processor import RiskTableMappingProcessor
from ...processing.numerical.numerical_imputation_processor import (
    NumericalVariableImputationProcessor,
)
from ...processing.validation import (
    validate_categorical_fields,
    validate_numerical_fields,
)
from ...processing.processor_registry import build_text_pipeline_from_steps
from ...processing.datasets.pipeline_datasets import PipelineDataset
from ...processing.dataloaders.pipeline_dataloader import build_collate_batch

from lightning_models.utils.pl_train import (
    model_inference,
    load_model,
    load_artifacts,
    is_main_process,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Container path constants - aligned with XGBoost inference contract
CONTAINER_PATHS = {
    "MODEL_DIR": "/opt/ml/processing/input/model",
    "EVAL_DATA_DIR": "/opt/ml/processing/input/eval_data",
    "OUTPUT_EVAL_DIR": "/opt/ml/processing/output/eval",
}


# ============================================================================
# FILE I/O HELPER FUNCTIONS WITH FORMAT PRESERVATION
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

    return file_path


# ============================================================================
# PREPROCESSING ARTIFACT LOADERS
# ============================================================================


def load_risk_tables(model_dir: str) -> Dict[str, Any]:
    """Load risk tables from pickle file."""
    import pickle as pkl

    risk_file = os.path.join(model_dir, "risk_table_map.pkl")
    if not os.path.exists(risk_file):
        logger.warning(f"Risk table file not found: {risk_file}")
        return {}

    try:
        with open(risk_file, "rb") as f:
            risk_tables = pkl.load(f)
        logger.info(f"Loaded risk tables for {len(risk_tables)} features")
        return risk_tables
    except Exception as e:
        logger.warning(f"Failed to load risk tables: {e}")
        return {}


def create_risk_processors(
    risk_tables: Dict[str, Any],
) -> Dict[str, RiskTableMappingProcessor]:
    """Create risk table processors for each categorical feature."""
    risk_processors = {}
    for feature, risk_table in risk_tables.items():
        processor = RiskTableMappingProcessor(
            column_name=feature,
            label_name="label",  # Not used during inference
            risk_tables=risk_table,
        )
        risk_processors[feature] = processor
    logger.info(f"Created {len(risk_processors)} risk table processors")
    return risk_processors


def load_imputation_dict(model_dir: str) -> Dict[str, Any]:
    """Load imputation dictionary from pickle file."""
    import pickle as pkl

    impute_file = os.path.join(model_dir, "impute_dict.pkl")
    if not os.path.exists(impute_file):
        logger.warning(f"Imputation file not found: {impute_file}")
        return {}

    try:
        with open(impute_file, "rb") as f:
            impute_dict = pkl.load(f)
        logger.info(f"Loaded imputation values for {len(impute_dict)} features")
        return impute_dict
    except Exception as e:
        logger.warning(f"Failed to load imputation dict: {e}")
        return {}


def create_numerical_processors(
    impute_dict: Dict[str, Any],
) -> Dict[str, NumericalVariableImputationProcessor]:
    """
    Create numerical imputation processors for each numerical feature.

    Uses single-column architecture - one processor per column.
    """
    numerical_processors = {}
    for feature, imputation_value in impute_dict.items():
        processor = NumericalVariableImputationProcessor(
            column_name=feature, imputation_value=imputation_value
        )
        numerical_processors[feature] = processor
    logger.info(f"Created {len(numerical_processors)} numerical imputation processors")
    return numerical_processors


# ============================================================================
# MODEL ARTIFACT LOADING
# ============================================================================


def decompress_model_artifacts(model_dir: str):
    """
    Securely checks for and extracts a model.tar.gz file with comprehensive safety measures.

    Security Features:
    - SHA-256 checksum verification (mandatory, no MD5 fallback)
    - Path traversal attack prevention
    - Symlink attack prevention
    - File locking to prevent race conditions
    - Thunder herd mitigation with exponential backoff
    - Safe temporary directory extraction with atomic move
    - Comprehensive error handling and logging

    Args:
        model_dir: Directory containing model.tar.gz and checksums

    Raises:
        RuntimeError: If security checks fail or extraction errors occur
    """
    model_tar_path = Path(model_dir) / "model.tar.gz"

    if not model_tar_path.exists():
        logger.info("No model.tar.gz found. Assuming artifacts are directly available.")
        return

    logger.info(f"Found model.tar.gz at {model_tar_path}")

    # ================================================================
    # SECURITY CHECK 1: SHA-256 Checksum Verification (MANDATORY)
    # ================================================================
    sha256_path = Path(model_dir) / "model.tar.gz.sha256"

    if not sha256_path.exists():
        raise RuntimeError(
            f"Security Error: SHA-256 checksum file not found at {sha256_path}. "
            "Cannot verify archive integrity. Extraction aborted."
        )

    logger.info("Verifying SHA-256 checksum...")

    # Read expected checksum
    try:
        with open(sha256_path, "r") as f:
            expected_sha256 = f.read().strip().split()[0]
    except Exception as e:
        raise RuntimeError(f"Failed to read SHA-256 checksum file: {e}")

    # Compute actual checksum
    sha256_hash = hashlib.sha256()
    try:
        with open(model_tar_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        actual_sha256 = sha256_hash.hexdigest()
    except Exception as e:
        raise RuntimeError(f"Failed to compute SHA-256 checksum: {e}")

    # Verify checksum match
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Security Error: SHA-256 checksum mismatch!\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual_sha256}\n"
            "Archive may be corrupted or tampered with. Extraction aborted."
        )

    logger.info("✓ SHA-256 checksum verification passed")

    # ================================================================
    # SECURITY CHECK 2: File Locking (Prevent Race Conditions)
    # ================================================================
    lock_file_path = Path(model_dir) / ".extraction.lock"

    try:
        lock_file = open(lock_file_path, "w")

        # Try to acquire exclusive lock with exponential backoff (thunder herd mitigation)
        max_attempts = 5
        base_wait = 1.0  # seconds

        for attempt in range(max_attempts):
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.info("✓ Acquired extraction lock")
                break
            except IOError:
                if attempt < max_attempts - 1:
                    wait_time = base_wait * (2**attempt) + (
                        0.1 * attempt
                    )  # Exponential + jitter
                    logger.info(
                        f"Extraction in progress by another process. "
                        f"Waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_attempts})..."
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        "Failed to acquire extraction lock after maximum attempts. "
                        "Another process may be stuck."
                    )

        # Check if already extracted (another process completed while we waited)
        expected_files = ["hyperparameters.json", "model.pth", "model_artifacts.pth"]
        if all((Path(model_dir) / f).exists() for f in expected_files):
            logger.info(
                "✓ Model already extracted by another process. Skipping extraction."
            )
            return

        # ================================================================
        # SECURE EXTRACTION: Use temporary directory + atomic move
        # ================================================================
        logger.info("Starting secure extraction...")

        # Create temporary extraction directory
        with tempfile.TemporaryDirectory(
            dir=model_dir, prefix=".extract_"
        ) as temp_extract_dir:
            temp_path = Path(temp_extract_dir)
            logger.info(f"Extracting to temporary directory: {temp_path}")

            # Extract with security checks
            with tarfile.open(model_tar_path, "r:gz") as tar:
                members = tar.getmembers()
                logger.info(f"Archive contains {len(members)} members")

                for member in members:
                    # ========================================================
                    # SECURITY CHECK 3: Path Traversal Prevention
                    # ========================================================
                    member_path = Path(temp_extract_dir) / member.name
                    try:
                        member_path.resolve().relative_to(temp_path.resolve())
                    except ValueError:
                        raise RuntimeError(
                            f"Security Error: Path traversal detected in archive member '{member.name}'. "
                            "Extraction aborted."
                        )

                    # ========================================================
                    # SECURITY CHECK 4: Symlink Attack Prevention
                    # ========================================================
                    if member.issym() or member.islnk():
                        raise RuntimeError(
                            f"Security Error: Symbolic/hard link detected in archive member '{member.name}'. "
                            "Extraction aborted."
                        )

                    # Safe to extract this member
                    tar.extract(member, path=temp_extract_dir)

                logger.info(
                    f"✓ Extracted {len(members)} members to temporary directory"
                )

            # ========================================================
            # ATOMIC MOVE: Move files from temp to target directory
            # ========================================================
            logger.info("Moving extracted files to model directory...")
            extracted_files = list(temp_path.rglob("*"))

            for src_file in extracted_files:
                if src_file.is_file():
                    rel_path = src_file.relative_to(temp_path)
                    dest_file = Path(model_dir) / rel_path

                    # Create parent directories if needed
                    dest_file.parent.mkdir(parents=True, exist_ok=True)

                    # Atomic move (rename)
                    src_file.replace(dest_file)

            logger.info("✓ Extraction complete")

        # Temporary directory automatically cleaned up here

    finally:
        # ================================================================
        # CLEANUP: Release lock and remove lock file
        # ================================================================
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            if lock_file_path.exists():
                lock_file_path.unlink()
            logger.info("✓ Released extraction lock")
        except Exception as e:
            logger.warning(f"Failed to clean up lock file: {e}")


def load_model_artifacts(
    model_dir: str,
) -> Tuple[nn.Module, Dict[str, Any], AutoTokenizer, Dict[str, Any]]:
    """
    Load trained PyTorch model and all preprocessing artifacts.

    Returns:
        - PyTorch Lightning model
        - Model configuration dictionary
        - Tokenizer for text processing
        - Preprocessing processors (categorical, imputation)
    """
    logger.info(f"Loading PyTorch model artifacts from {model_dir}")

    # Decompress the model tarball if it exists
    logger.info("Checking for model.tar.gz and decompressing if present")
    decompress_model_artifacts(model_dir)

    # Load hyperparameters
    hyperparams_path = os.path.join(model_dir, "hyperparameters.json")
    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)
    logger.info("Loaded hyperparameters.json")

    # Load model artifacts (config, embeddings, vocab, processors)
    artifact_path = os.path.join(model_dir, "model_artifacts.pth")
    artifacts = load_artifacts(
        artifact_path, model_class=hyperparams.get("model_class", "bimodal_bert")
    )
    logger.info("Loaded model_artifacts.pth")

    config = artifacts["config"]
    embedding_mat = artifacts.get("embedding_mat")
    vocab = artifacts.get("vocab")

    # Reconstruct tokenizer
    tokenizer_name = config.get("tokenizer", "bert-base-multilingual-cased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Reconstructed tokenizer: {tokenizer_name}")

    # Load trained model
    model_path = os.path.join(model_dir, "model.pth")
    model = load_model(model_path, model_class=config["model_class"], device_l="cpu")
    model.eval()  # Set to evaluation mode
    logger.info("Loaded model.pth and set to evaluation mode")

    # Load preprocessing artifacts (numerical imputation + risk tables)
    logger.info("Loading preprocessing artifacts...")
    risk_tables = load_risk_tables(model_dir)
    risk_processors = create_risk_processors(risk_tables)

    impute_dict = load_imputation_dict(model_dir)
    numerical_processors = create_numerical_processors(impute_dict)

    logger.info(
        f"Loaded {len(risk_processors)} risk processors and {len(numerical_processors)} numerical processors"
    )

    processors = {
        "risk_processors": risk_processors,
        "numerical_processors": numerical_processors,
        "embedding_mat": embedding_mat,
        "vocab": vocab,
    }

    logger.info(
        f"Model artifacts loaded successfully. Model class: {config['model_class']}"
    )
    return model, config, tokenizer, processors


# ============================================================================
# DATA PREPROCESSING
# ============================================================================


def create_pipeline_dataset(
    config: Dict[str, Any], inference_data_dir: str, filename: str
) -> PipelineDataset:
    """
    Create and initialize PipelineDataset with missing value handling.
    NO LABEL REQUIRED for inference.

    Args:
        config: Model configuration
        inference_data_dir: Directory containing inference data
        filename: Name of inference data file

    Returns:
        Initialized PipelineDataset
    """
    pipeline_dataset = PipelineDataset(
        config=config, file_dir=inference_data_dir, filename=filename
    )

    # Fill missing values (no label required)
    pipeline_dataset.fill_missing_value(
        label_name=None,  # No label for inference
        column_cat_name=config.get("cat_field_list", []),
    )
    logger.info("Created PipelineDataset and filled missing values")

    return pipeline_dataset


def data_preprocess_pipeline(
    config: Dict[str, Any], tokenizer: AutoTokenizer
) -> Tuple[AutoTokenizer, Dict[str, Any]]:
    """
    Build text preprocessing pipelines based on config.

    For bimodal: Uses text_name with default or configured steps
    For trimodal: Uses primary_text_name and secondary_text_name with separate step lists

    Args:
        config: Model configuration
        tokenizer: BERT tokenizer

    Returns:
        Tuple of (tokenizer, pipelines_dict)
    """
    pipelines = {}

    logger.info("=" * 70)
    logger.info("BUILDING TEXT PREPROCESSING PIPELINES")
    logger.info("=" * 70)

    # BIMODAL: Single text pipeline
    if not config.get("primary_text_name"):
        text_name = config.get("text_name")
        if not text_name:
            raise ValueError(
                "Config must have either 'text_name' or 'primary_text_name'"
            )

        # Use configured steps or fallback to default
        steps = config.get(
            "text_processing_steps",
            [
                "dialogue_splitter",
                "html_normalizer",
                "emoji_remover",
                "text_normalizer",
                "dialogue_chunker",
                "tokenizer",
            ],
        )

        pipelines[text_name] = build_text_pipeline_from_steps(
            processing_steps=steps,
            tokenizer=tokenizer,
            max_sen_len=config["max_sen_len"],
            chunk_trancate=config.get("chunk_trancate", False),
            max_total_chunks=config.get("max_total_chunks", 5),
            input_ids_key=config.get("text_input_ids_key", "input_ids"),
            attention_mask_key=config.get("text_attention_mask_key", "attention_mask"),
        )
        logger.info(f"✓ Built bimodal pipeline for '{text_name}' with steps: {steps}")

    # TRIMODAL: Dual text pipelines
    else:
        # Primary text pipeline (e.g., chat - full cleaning)
        primary_name = config["primary_text_name"]
        primary_steps = config.get(
            "primary_text_processing_steps",
            [
                "dialogue_splitter",
                "html_normalizer",
                "emoji_remover",
                "text_normalizer",
                "dialogue_chunker",
                "tokenizer",
            ],
        )

        pipelines[primary_name] = build_text_pipeline_from_steps(
            processing_steps=primary_steps,
            tokenizer=tokenizer,
            max_sen_len=config["max_sen_len"],
            chunk_trancate=config.get("chunk_trancate", False),
            max_total_chunks=config.get("max_total_chunks", 5),
            input_ids_key=config.get("text_input_ids_key", "input_ids"),
            attention_mask_key=config.get("text_attention_mask_key", "attention_mask"),
        )
        logger.info(
            f"✓ Built primary pipeline for '{primary_name}' with steps: {primary_steps}"
        )

        # Secondary text pipeline (e.g., events - minimal cleaning)
        secondary_name = config["secondary_text_name"]
        secondary_steps = config.get(
            "secondary_text_processing_steps",
            [
                "dialogue_splitter",
                "text_normalizer",
                "dialogue_chunker",
                "tokenizer",
            ],
        )

        pipelines[secondary_name] = build_text_pipeline_from_steps(
            processing_steps=secondary_steps,
            tokenizer=tokenizer,
            max_sen_len=config["max_sen_len"],
            chunk_trancate=config.get("chunk_trancate", False),
            max_total_chunks=config.get("max_total_chunks", 5),
            input_ids_key=config.get("text_input_ids_key", "input_ids"),
            attention_mask_key=config.get("text_attention_mask_key", "attention_mask"),
        )
        logger.info(
            f"✓ Built secondary pipeline for '{secondary_name}' with steps: {secondary_steps}"
        )

    logger.info(f"✅ Created {len(pipelines)} text preprocessing pipelines")
    logger.info("=" * 70)

    return tokenizer, pipelines


def apply_preprocessing_artifacts(
    pipeline_dataset: PipelineDataset,
    processors: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Apply numerical imputation and risk table mapping to dataset.
    Excludes text fields from risk table mapping to prevent overwriting tokenized text.

    Args:
        pipeline_dataset: Dataset to apply preprocessing to
        processors: Dictionary containing preprocessing processors
        config: Model configuration to identify text fields
    """
    logger.info("=" * 70)
    logger.info("APPLYING PREPROCESSING ARTIFACTS")
    logger.info("=" * 70)

    # === FIELD TYPE VALIDATION ===
    numerical_fields = config.get("tab_field_list", [])
    categorical_fields = config.get("cat_field_list", [])

    if numerical_fields:
        logger.info("Validating numerical field types...")
        try:
            validate_numerical_fields(
                pipeline_dataset.DataReader, numerical_fields, "inference"
            )
            logger.info("✓ Numerical field type validation passed")
        except Exception as e:
            logger.warning(f"Numerical field validation failed: {e}")

    if categorical_fields:
        logger.info("Validating categorical field types...")
        try:
            validate_categorical_fields(
                pipeline_dataset.DataReader, categorical_fields, "inference"
            )
            logger.info("✓ Categorical field type validation passed")
        except Exception as e:
            logger.warning(f"Categorical field validation failed: {e}")

    # === NUMERICAL IMPUTATION ===
    numerical_processors = processors.get("numerical_processors", {})
    if numerical_processors:
        logger.info(
            f"Applying {len(numerical_processors)} numerical imputation processors..."
        )
        for feature, processor in numerical_processors.items():
            if feature in pipeline_dataset.DataReader.columns:
                pipeline_dataset.add_pipeline(feature, processor)
        logger.info(f"✓ Applied {len(numerical_processors)} numerical processors")

    # === RISK TABLE MAPPING ===
    # Filter out text fields from risk table mapping
    text_fields = set()
    if config.get("text_name"):
        text_fields.add(config["text_name"])
    if config.get("primary_text_name"):
        text_fields.add(config["primary_text_name"])
    if config.get("secondary_text_name"):
        text_fields.add(config["secondary_text_name"])

    if text_fields:
        logger.info(f"ℹ️  Text fields to exclude from risk tables: {text_fields}")

    # Apply risk table mapping processors (excluding text fields)
    risk_processors = processors.get("risk_processors", {})
    if risk_processors:
        logger.info(f"Applying risk table mapping to categorical features...")
        excluded_count = 0
        applied_count = 0

        for feature, processor in risk_processors.items():
            if feature in text_fields:
                excluded_count += 1
                continue
            if feature in pipeline_dataset.DataReader.columns:
                pipeline_dataset.add_pipeline(feature, processor)
                applied_count += 1

        logger.info(f"✓ Applied {applied_count} risk table processors")
        if excluded_count > 0:
            logger.info(f"  Excluded {excluded_count} text fields from risk mapping")

    logger.info("=" * 70)


def create_dataloader(
    pipeline_dataset: PipelineDataset, config: Dict[str, Any]
) -> DataLoader:
    """
    Create DataLoader with appropriate collate function.

    Uses unified collate function for all model types.

    Args:
        pipeline_dataset: Dataset to create DataLoader for
        config: Model configuration

    Returns:
        Configured DataLoader
    """
    # Use unified collate function for all model types
    logger.info(
        f"Using collate batch for model: {config.get('model_class', 'bimodal')}"
    )

    # Use unified keys for all models (single tokenizer design)
    collate_batch = build_collate_batch(
        input_ids_key=config.get("text_input_ids_key", "input_ids"),
        attention_mask_key=config.get("text_attention_mask_key", "attention_mask"),
    )

    batch_size = config.get("batch_size", 32)
    dataloader = DataLoader(
        pipeline_dataset,
        collate_fn=collate_batch,
        batch_size=batch_size,
        shuffle=False,
    )
    logger.info(f"Created DataLoader with batch_size={batch_size}")

    return dataloader


def preprocess_inference_data(
    df: pd.DataFrame,
    config: Dict[str, Any],
    tokenizer: AutoTokenizer,
    processors: Dict[str, Any],
    inference_data_dir: str,
    filename: str,
) -> Tuple[PipelineDataset, DataLoader]:
    """
    Apply complete preprocessing pipeline to inference data.
    Orchestrates the creation of PipelineDataset and DataLoader.
    NO LABEL REQUIRED.

    Args:
        df: Input DataFrame
        config: Model configuration
        tokenizer: BERT tokenizer
        processors: Preprocessing processors
        inference_data_dir: Directory containing inference data
        filename: Name of inference data file

    Returns:
        Tuple of (PipelineDataset, DataLoader)
    """
    logger.info("=" * 70)
    logger.info(f"PREPROCESSING INFERENCE DATA: {filename}")
    logger.info("=" * 70)

    # Step 1: Create and initialize dataset (no label required)
    pipeline_dataset = create_pipeline_dataset(config, inference_data_dir, filename)

    # Step 2: Build and add text preprocessing pipelines (bimodal or trimodal)
    tokenizer, text_pipelines = data_preprocess_pipeline(config, tokenizer)

    logger.info("Registering text processing pipelines...")
    for field_name, pipeline in text_pipelines.items():
        logger.info(f"  Field: '{field_name}' -> Pipeline registered")
        pipeline_dataset.add_pipeline(field_name, pipeline)
    logger.info(f"✅ Registered {len(text_pipelines)} text processing pipelines")

    # Step 3: Apply preprocessing artifacts (numerical + categorical)
    apply_preprocessing_artifacts(pipeline_dataset, processors, config)

    # Step 4: Create DataLoader with appropriate collate function
    dataloader = create_dataloader(pipeline_dataset, config)

    logger.info("=" * 70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 70)

    return pipeline_dataset, dataloader


# ============================================================================
# DEVICE SETUP
# ============================================================================


def setup_device_environment(
    device: Union[str, int, List[int]] = "auto",
) -> Tuple[Union[str, int, List[int]], str]:
    """
    Set up device environment based on availability and config.
    Supports single GPU, multi-GPU, CPU, or automatic detection.

    Args:
        device: Device selection:
            - "auto": Use all available GPUs or CPU
            - "cpu": Force CPU usage
            - int: Use specific number of GPUs (e.g., 1, 2, 4)
            - List[int]: Use specific GPU IDs (e.g., [0, 1, 2, 3])
            - "cuda" or "gpu": Use single GPU (GPU 0)

    Returns:
        Tuple of (device_setting, accelerator_string)
        - device_setting can be: "cpu", int (GPU count), or List[int] (GPU IDs)
        - accelerator_string: "cpu" or "gpu"
    """
    if device == "auto":
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device_setting = gpu_count  # Use all available GPUs
            accelerator = "gpu"
            logger.info(f"Auto-detected {gpu_count} GPU(s) - using all for inference")
        else:
            device_setting = "cpu"
            accelerator = "cpu"
            logger.info("No GPU detected - using CPU for inference")
    elif device in ["cpu"]:
        device_setting = "cpu"
        accelerator = "cpu"
        logger.info("Forced CPU usage for inference")
    elif device in ["cuda", "gpu"]:
        device_setting = 1  # Single GPU
        accelerator = "gpu"
        logger.info("Using single GPU (GPU 0) for inference")
    elif isinstance(device, int):
        device_setting = device
        accelerator = "gpu"
        logger.info(f"Using {device} GPU(s) for inference")
    elif isinstance(device, list):
        device_setting = device
        accelerator = "gpu"
        logger.info(f"Using specific GPUs {device} for inference")
    else:
        # Fallback to auto
        logger.warning(f"Unknown device setting '{device}', falling back to 'auto'")
        return setup_device_environment("auto")

    # Log GPU information if using GPU
    if accelerator == "gpu":
        gpu_count = (
            len(device_setting) if isinstance(device_setting, list) else device_setting
        )
        logger.info(f"GPU Configuration:")

        if isinstance(device_setting, list):
            for gpu_id in device_setting:
                logger.info(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            for i in range(min(gpu_count, torch.cuda.device_count())):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        # Enable optimizations
        torch.backends.cudnn.benchmark = True

        # Log memory info for first GPU
        logger.info(
            f"GPU 0 Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        )
        logger.info(
            f"GPU 0 Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB"
        )

    return device_setting, accelerator


# ============================================================================
# PREDICTION GENERATION
# ============================================================================


def generate_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, int, List[int]] = "auto",
    accelerator: str = "auto",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate predictions using PyTorch Lightning inference.
    Supports single-GPU, multi-GPU, CPU, and automatic detection.
    NO LABELS REQUIRED.

    Args:
        model: PyTorch Lightning model
        dataloader: DataLoader for inference data
        device: Device setting (can be int, list, or string)
        accelerator: Accelerator type for Lightning

    Returns:
        Tuple of (predictions, dataframe_with_ids)
    """
    # Determine if multi-GPU inference
    is_multi_gpu = False
    if isinstance(device, int) and device > 1:
        is_multi_gpu = True
    elif isinstance(device, list) and len(device) > 1:
        is_multi_gpu = True

    logger.info("=" * 70)
    logger.info("RUNNING MODEL INFERENCE")
    logger.info("=" * 70)
    logger.info(f"Device setting: {device}")
    logger.info(f"Accelerator: {accelerator}")
    logger.info(f"Multi-GPU inference: {'Yes' if is_multi_gpu else 'No'}")
    logger.info("=" * 70)

    # Use Lightning's model_inference utility with dataframe return
    y_pred, _, df = model_inference(
        model,
        dataloader,
        accelerator=accelerator,
        device=device,
        model_log_path=None,  # No logging during inference
        return_dataframe=True,  # Get dataframe with IDs
    )

    logger.info("=" * 70)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Prediction shape: {y_pred.shape}")
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info("=" * 70)

    return y_pred, df


# ============================================================================
# OUTPUT MANAGEMENT
# ============================================================================


def save_predictions_with_dataframe(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    output_dir: str,
    input_format: str = "csv",
) -> None:
    """
    Save predictions by adding probability columns to existing dataframe.
    ONLY includes id and class probabilities (NO true labels).

    Args:
        df: Dataframe with IDs from inference (already aligned)
        y_prob: Predicted probabilities
        output_dir: Directory to save predictions
        input_format: Format to save in ('csv', 'tsv', or 'parquet')
    """
    logger.info(f"Saving predictions to {output_dir} in {input_format} format")

    # Make a copy to avoid modifying original
    out_df = df.copy()

    # Add probability columns
    num_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 1
    if num_classes == 1:
        # Binary with single probability
        out_df["prob_class_0"] = 1 - y_prob.squeeze()
        out_df["prob_class_1"] = y_prob.squeeze()
    else:
        for i in range(num_classes):
            out_df[f"prob_class_{i}"] = y_prob[:, i]

    output_base = Path(output_dir) / "inference_predictions"
    output_path = save_dataframe_with_format(out_df, output_base, input_format)
    logger.info(f"Saved predictions (format={input_format}): {output_path}")


def create_health_check_file(output_path: str) -> str:
    """Create a health check file to signal script completion."""
    health_path = output_path
    with open(health_path, "w") as f:
        f.write(f"healthy: {datetime.now().isoformat()}")
    return health_path


# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================


def load_inference_data(inference_data_dir: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Load the first data file found in the inference data directory.
    Returns a pandas DataFrame, the detected format, and the filename.
    """
    logger.info(f"Loading inference data from {inference_data_dir}")
    inference_files = sorted(
        [
            f
            for f in Path(inference_data_dir).glob("**/*")
            if f.suffix in [".csv", ".tsv", ".parquet"]
        ]
    )
    if not inference_files:
        logger.error("No inference data file found in inference_data input.")
        raise RuntimeError("No inference data file found in inference_data input.")

    inference_file = inference_files[0]
    logger.info(f"Using inference data file: {inference_file}")

    df, input_format = load_dataframe_with_format(inference_file)
    filename = inference_file.name
    logger.info(
        f"Loaded inference data shape: {df.shape}, format: {input_format}, filename: {filename}"
    )
    return df, input_format, filename


def get_id_column(df: pd.DataFrame, id_field: str) -> str:
    """
    Determine the ID column in the DataFrame.
    Falls back to the first column if not found.
    """
    id_col = id_field if id_field in df.columns else df.columns[0]
    logger.info(f"Using id_col: {id_col}")
    return id_col


def run_batch_inference(
    model: nn.Module,
    df: pd.DataFrame,
    config: Dict[str, Any],
    tokenizer: AutoTokenizer,
    processors: Dict[str, Any],
    inference_data_dir: str,
    filename: str,
    id_col: str,
    output_dir: str,
    input_format: str = "csv",
    device: Union[str, int, List[int]] = "auto",
) -> None:
    """
    Run model inference and save predictions.
    NO LABELS REQUIRED - predictions only.

    Args:
        model: PyTorch Lightning model
        df: Inference DataFrame
        config: Model configuration
        tokenizer: BERT tokenizer
        processors: Preprocessing processors
        inference_data_dir: Directory containing inference data
        filename: Name of inference data file
        id_col: Name of ID column (not used in refactored approach)
        output_dir: Directory to save predictions
        input_format: Input data format
        device: Device to use for inference
    """
    logger.info("Starting batch inference")

    # Preprocess data and create DataLoader
    pipeline_dataset, dataloader = preprocess_inference_data(
        df, config, tokenizer, processors, inference_data_dir, filename
    )

    # Setup device environment
    device_str, accelerator = setup_device_environment(device)

    # Generate predictions with dataframe (all ranks participate in DDP)
    y_prob, inf_df = generate_predictions(model, dataloader, device_str, accelerator)

    # ===================================================================
    # CRITICAL: Only main process performs post-processing
    # This prevents race conditions when multiple GPUs try to write
    # to the same files simultaneously
    # ===================================================================
    if is_main_process():
        logger.info("=" * 70)
        logger.info("POST-PROCESSING (MAIN PROCESS ONLY)")
        logger.info("=" * 70)

        # Save predictions with aligned dataframe
        save_predictions_with_dataframe(inf_df, y_prob, output_dir, input_format)

        logger.info("=" * 70)
        logger.info("POST-PROCESSING COMPLETE")
        logger.info("=" * 70)
    else:
        logger.info(
            f"Rank {dist.get_rank() if dist.is_initialized() else 'N/A'}: Skipping post-processing (not main process)"
        )

    # ===================================================================
    # CRITICAL: Synchronization barrier
    # Ensure all ranks wait until main process completes post-processing
    # before proceeding (e.g., before script exit)
    # ===================================================================
    if dist.is_initialized():
        logger.info("Waiting at synchronization barrier...")
        dist.barrier()
        logger.info("All ranks synchronized - inference complete")

    logger.info("Batch inference complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main entry point for PyTorch model batch inference script.
    Loads model and data, runs inference, and saves predictions ONLY.

    Args:
        input_paths: Dictionary of input paths
        output_paths: Dictionary of output paths
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
    """
    # Extract paths from parameters (aligned with XGBoost)
    model_dir = input_paths.get("model_input", input_paths.get("model_dir"))
    inference_data_dir = input_paths.get(
        "processed_data", input_paths.get("eval_data_dir")
    )
    output_dir = output_paths.get("eval_output", output_paths.get("output_eval_dir"))

    # Extract environment variables
    id_field = environ_vars.get("ID_FIELD", "id")

    # Parse device setting
    device_str = environ_vars.get("DEVICE", "auto")
    try:
        # Try to parse as JSON for list format: "[0,1,2,3]"
        if device_str.startswith("[") and device_str.endswith("]"):
            device = json.loads(device_str)
        # Try to parse as int for GPU count: "4"
        elif device_str.isdigit():
            device = int(device_str)
        # Use as string for: "auto", "cpu", "cuda", "gpu"
        else:
            device = device_str
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"Failed to parse DEVICE='{device_str}', using 'auto'")
        device = "auto"

    # Log job info
    job_type = job_args.job_type
    logger.info(f"Running PyTorch batch inference with job_type: {job_type}")
    logger.info(f"Device setting: {device}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting PyTorch batch inference script")

    # Load model artifacts
    model, config, tokenizer, processors = load_model_artifacts(model_dir)

    # Load inference data with format detection
    df, input_format, filename = load_inference_data(inference_data_dir)

    # Get ID column (NO LABEL REQUIRED)
    id_col = get_id_column(df, id_field)

    # Run batch inference
    run_batch_inference(
        model,
        df,
        config,
        tokenizer,
        processors,
        inference_data_dir,
        filename,
        id_col,
        output_dir,
        input_format,
        device,
    )

    logger.info("PyTorch batch inference script complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    # Set up paths using contract-defined paths (aligned with XGBoost)
    input_paths = {
        "model_input": CONTAINER_PATHS["MODEL_DIR"],
        "processed_data": CONTAINER_PATHS["EVAL_DATA_DIR"],
    }

    output_paths = {
        "eval_output": CONTAINER_PATHS["OUTPUT_EVAL_DIR"],
    }

    # Collect environment variables
    environ_vars = {
        "ID_FIELD": os.environ.get("ID_FIELD", "id"),
        "DEVICE": os.environ.get("DEVICE", "auto"),
        "BATCH_SIZE": os.environ.get("BATCH_SIZE", "32"),
    }

    try:
        # Call main function
        main(input_paths, output_paths, environ_vars, args)

        # Signal success
        success_path = os.path.join(output_paths["eval_output"], "_SUCCESS")
        Path(success_path).touch()
        logger.info(f"Created success marker: {success_path}")

        # Create health check file
        health_path = os.path.join(output_paths["eval_output"], "_HEALTH")
        create_health_check_file(health_path)
        logger.info(f"Created health check file: {health_path}")

        sys.exit(0)
    except Exception as e:
        # Log error and create failure marker
        logger.exception(f"Script failed with error: {e}")
        failure_path = os.path.join(output_paths.get("eval_output", "/tmp"), "_FAILURE")
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(f"Error: {str(e)}")
        sys.exit(1)
