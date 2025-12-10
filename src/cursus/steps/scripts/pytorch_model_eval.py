#!/usr/bin/env python
"""
PyTorch Model Evaluation Script

Evaluates trained PyTorch Lightning models with GPU/CPU support.
Follows the same contract and structure as xgboost_model_eval.py.

Features:
- GPU/CPU automatic detection and explicit control
- Multi-modal model support (text, tabular, bimodal, trimodal)
- Format preservation (CSV/TSV/Parquet)
- Comprehensive metrics computation
- ROC/PR curve generation
- Comparison mode support
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
from torch.utils.tensorboard import SummaryWriter

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
from ...processing.categorical.multiclass_label_processor import (
    MultiClassLabelProcessor,
)
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
from ...processing.dataloaders.pipeline_dataloader import (
    build_collate_batch,
    build_trimodal_collate_batch,
)

from lightning_models.utils.pl_train import (
    model_inference,
    load_model,
    load_artifacts,
)
from lightning_models.utils.dist_utils import is_main_process
from lightning_models.utils.pl_model_plots import (
    compute_metrics,
    roc_metric_plot,
    pr_metric_plot,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Container path constants - aligned with script contract
CONTAINER_PATHS = {
    "MODEL_DIR": "/opt/ml/processing/input/model",
    "EVAL_DATA_DIR": "/opt/ml/processing/input/eval_data",
    "OUTPUT_EVAL_DIR": "/opt/ml/processing/output/eval",
    "OUTPUT_METRICS_DIR": "/opt/ml/processing/output/metrics",
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
    Thread-safe extraction of model.tar.gz with atomic file locking.

    Uses fcntl-based locking to prevent race conditions when multiple workers
    attempt simultaneous extraction. Only one worker extracts while others wait.

    Security: Uses environment-driven directory resolution (SM_MODEL_DIR or tempfile)
    to comply with security scanner requirements.

    Args:
        model_dir: Directory containing model.tar.gz
    """
    model_tar_path = Path(model_dir) / "model.tar.gz"

    if not model_tar_path.exists():
        logger.info("No model.tar.gz found. Assuming artifacts are directly available.")
        return

    # Generate unique marker based on tar file path
    tar_hash = hashlib.sha256(str(model_tar_path).encode()).hexdigest()[:8]

    # Use environment-driven temp directory (security compliance)
    temp_base = os.environ.get("SM_MODEL_DIR", tempfile.gettempdir())
    lock_file = Path(temp_base) / f".model_extraction_{tar_hash}.lock"
    marker_file = Path(model_dir) / f".model_extracted_{tar_hash}.marker"
    failure_marker = Path(model_dir) / f".model_extraction_failed_{tar_hash}.marker"

    logger.info(f"Found model.tar.gz at {model_tar_path}")
    logger.info(f"Using lock file: {lock_file}")
    logger.info(f"Using marker file: {marker_file}")

    # Check for previous extraction failure (early exit to prevent thundering herd)
    if failure_marker.exists():
        error_msg = failure_marker.read_text()
        logger.error(f"✗ Model extraction previously failed (marker: {failure_marker})")
        logger.error(f"  Failure details: {error_msg}")
        raise RuntimeError(f"Model extraction previously failed: {error_msg}")

    # Check if already extracted
    if marker_file.exists():
        logger.info("✓ Model already extracted (marker exists). Skipping extraction.")
        return

    # Acquire lock for extraction
    logger.info("Acquiring lock for model extraction...")
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_file, "w") as lock_fp:
        try:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            logger.info("✓ Lock acquired")

            # Double-check marker after acquiring lock (another worker may have extracted)
            if marker_file.exists():
                logger.info("✓ Model extracted by another worker. Skipping extraction.")
                return

            # Double-check failure marker after acquiring lock
            if failure_marker.exists():
                error_msg = failure_marker.read_text()
                logger.error(f"✗ Model extraction failed by another worker")
                raise RuntimeError(f"Model extraction previously failed: {error_msg}")

            # Perform extraction with failure handling
            try:
                logger.info("Extracting model.tar.gz...")
                with tarfile.open(model_tar_path, "r:gz") as tar:
                    # Safe extraction: validate paths to prevent zip slip attacks
                    safe_extract_dir = Path(model_dir).resolve()
                    for member in tar.getmembers():
                        member_path = (safe_extract_dir / member.name).resolve()
                        if not str(member_path).startswith(str(safe_extract_dir)):
                            raise RuntimeError(
                                f"Attempted path traversal in tar file: {member.name}"
                            )
                    # Extract after validation
                    tar.extractall(path=model_dir)

                # Create marker to signal successful extraction
                marker_file.write_text(f"extracted: {datetime.now().isoformat()}")
                logger.info("✓ Extraction complete. Marker created.")

            except Exception as e:
                # Create failure marker to prevent thundering herd
                error_details = f"failed: {datetime.now().isoformat()}\nError: {str(e)}\nType: {type(e).__name__}"
                failure_marker.write_text(error_details)
                logger.error(f"✗ Extraction failed: {e}")
                logger.error(f"  Created failure marker: {failure_marker}")
                raise

        finally:
            # Release lock
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            logger.info("✓ Lock released")


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

    # Load model artifacts (config, embeddings, vocab, model_class)
    artifact_path = os.path.join(model_dir, "model_artifacts.pth")
    config, embedding_mat, vocab, model_class = load_artifacts(artifact_path)
    logger.info("Loaded model_artifacts.pth")

    # Reconstruct tokenizer
    tokenizer_name = config.get("tokenizer", "bert-base-multilingual-cased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Reconstructed tokenizer: {tokenizer_name}")

    # Load trained model
    model_path = os.path.join(model_dir, "model.pth")
    model = load_model(
        model_path,
        config=config,
        embedding_mat=embedding_mat,
        model_class=model_class,
        device_l="cpu",
    )
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

    # Extract label mappings for multiclass
    label_mappings = {
        "label_to_id": config.get("label_to_id"),
        "id_to_label": config.get("id_to_label"),
    }

    processors = {
        "label_mappings": label_mappings,
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
    config: Dict[str, Any], eval_data_dir: str, filename: str
) -> PipelineDataset:
    """
    Create and initialize PipelineDataset with missing value handling.

    Args:
        config: Model configuration
        eval_data_dir: Directory containing evaluation data
        filename: Name of evaluation data file

    Returns:
        Initialized PipelineDataset
    """
    pipeline_dataset = PipelineDataset(
        config=config, file_dir=eval_data_dir, filename=filename
    )

    # Fill missing values
    pipeline_dataset.fill_missing_value(
        label_name=config["label_name"],
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
                pipeline_dataset.DataReader, numerical_fields, "eval"
            )
            logger.info("✓ Numerical field type validation passed")
        except Exception as e:
            logger.warning(f"Numerical field validation failed: {e}")

    if categorical_fields:
        logger.info("Validating categorical field types...")
        try:
            validate_categorical_fields(
                pipeline_dataset.DataReader, categorical_fields, "eval"
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


def add_label_processor(
    pipeline_dataset: PipelineDataset,
    config: Dict[str, Any],
    processors: Dict[str, Any],
) -> None:
    """
    Add multiclass label processor if needed.

    Args:
        pipeline_dataset: Dataset to add label processor to
        config: Model configuration
        processors: Dictionary containing label mappings
    """
    if not config["is_binary"] and config["num_classes"] > 2:
        label_mappings = processors["label_mappings"]
        if label_mappings["label_to_id"]:
            label_processor = MultiClassLabelProcessor(
                label_to_id=label_mappings["label_to_id"],
                id_to_label=label_mappings["id_to_label"],
            )
            pipeline_dataset.add_pipeline(config["label_name"], label_processor)
            logger.info("Added multiclass label processor")


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


def preprocess_eval_data(
    df: pd.DataFrame,
    config: Dict[str, Any],
    tokenizer: AutoTokenizer,
    processors: Dict[str, Any],
    eval_data_dir: str,
    filename: str,
) -> Tuple[PipelineDataset, DataLoader]:
    """
    Apply complete preprocessing pipeline to evaluation data.
    Orchestrates the creation of PipelineDataset and DataLoader.

    Args:
        df: Input DataFrame
        config: Model configuration
        tokenizer: BERT tokenizer
        processors: Preprocessing processors
        eval_data_dir: Directory containing evaluation data
        filename: Name of evaluation data file

    Returns:
        Tuple of (PipelineDataset, DataLoader)
    """
    logger.info("=" * 70)
    logger.info(f"PREPROCESSING EVALUATION DATA: {filename}")
    logger.info("=" * 70)

    # Step 1: Create and initialize dataset
    pipeline_dataset = create_pipeline_dataset(config, eval_data_dir, filename)

    # Step 2: Build and add text preprocessing pipelines (bimodal or trimodal)
    tokenizer, text_pipelines = data_preprocess_pipeline(config, tokenizer)

    logger.info("Registering text processing pipelines...")
    for field_name, pipeline in text_pipelines.items():
        logger.info(f"  Field: '{field_name}' -> Pipeline registered")
        pipeline_dataset.add_pipeline(field_name, pipeline)
    logger.info(f"✅ Registered {len(text_pipelines)} text processing pipelines")

    # Step 3: Apply preprocessing artifacts (numerical + categorical)
    apply_preprocessing_artifacts(pipeline_dataset, processors, config)

    # Step 4: Add label processor for multiclass if needed
    add_label_processor(pipeline_dataset, config, processors)

    # Step 5: Create DataLoader with appropriate collate function
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
            logger.info(f"Auto-detected {gpu_count} GPU(s) - using all for evaluation")
        else:
            device_setting = "cpu"
            accelerator = "cpu"
            logger.info("No GPU detected - using CPU for evaluation")
    elif device in ["cpu"]:
        device_setting = "cpu"
        accelerator = "cpu"
        logger.info("Forced CPU usage for evaluation")
    elif device in ["cuda", "gpu"]:
        device_setting = 1  # Single GPU
        accelerator = "gpu"
        logger.info("Using single GPU (GPU 0) for evaluation")
    elif isinstance(device, int):
        device_setting = device
        accelerator = "gpu"
        logger.info(f"Using {device} GPU(s) for evaluation")
    elif isinstance(device, list):
        device_setting = device
        accelerator = "gpu"
        logger.info(f"Using specific GPUs {device} for evaluation")
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
    label_col: str = "label",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate predictions using PyTorch Lightning inference.
    Supports single-GPU, multi-GPU, CPU, and automatic detection.

    Args:
        model: PyTorch Lightning model
        dataloader: DataLoader for evaluation data
        device: Device setting (can be int, list, or string)
        accelerator: Accelerator type for Lightning
        label_col: Name of the label column in the dataframe

    Returns:
        Tuple of (y_pred probabilities, y_true labels, dataframe_with_ids)
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
    y_pred, y_true, df = model_inference(
        model,
        dataloader,
        accelerator=accelerator,
        device=device,
        model_log_path=None,  # No logging during evaluation
        return_dataframe=True,  # Get dataframe with IDs and labels
        label_col=label_col,  # Pass label column name
    )

    logger.info("=" * 70)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Prediction shape: {y_pred.shape}")
    logger.info(f"True labels shape: {y_true.shape}")
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info("=" * 70)

    return y_pred, y_true, df


# ============================================================================
# METRICS COMPUTATION
# ============================================================================


def log_metrics_summary(
    metrics: Dict[str, Union[int, float, str]], is_binary: bool = True
) -> None:
    """
    Log a nicely formatted summary of metrics for easy visibility in logs.

    Args:
        metrics: Dictionary of metrics to log
        is_binary: Whether these are binary classification metrics
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 80)
    logger.info(f"METRICS SUMMARY - {timestamp}")
    logger.info("=" * 80)

    # Log each metric with a consistent format
    for name, value in metrics.items():
        # Format numeric values to 4 decimal places
        if isinstance(value, (int, float)):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)

        # Add a special prefix for easy searching in logs
        logger.info(f"METRIC: {name.ljust(25)} = {formatted_value}")

    # Highlight key metrics based on task type
    logger.info("=" * 80)
    logger.info("KEY PERFORMANCE METRICS")
    logger.info("=" * 80)

    if is_binary:
        auc = metrics.get("auc_roc", "N/A")
        ap = metrics.get("average_precision", "N/A")
        f1 = metrics.get("f1_score", "N/A")
        if isinstance(auc, (int, float)):
            logger.info(f"METRIC_KEY: AUC-ROC               = {auc:.4f}")
        if isinstance(ap, (int, float)):
            logger.info(f"METRIC_KEY: Average Precision     = {ap:.4f}")
        if isinstance(f1, (int, float)):
            logger.info(f"METRIC_KEY: F1 Score              = {f1:.4f}")
    else:
        auc_macro = metrics.get("auc_roc_macro", "N/A")
        auc_micro = metrics.get("auc_roc_micro", "N/A")
        if isinstance(auc_macro, (int, float)):
            logger.info(f"METRIC_KEY: Macro AUC-ROC         = {auc_macro:.4f}")
        if isinstance(auc_micro, (int, float)):
            logger.info(f"METRIC_KEY: Micro AUC-ROC         = {auc_micro:.4f}")

    logger.info("=" * 80)


def compute_evaluation_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    Uses Lightning's compute_metrics utility for consistency.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        config: Model configuration

    Returns:
        Dictionary of metrics with XGBoost-compatible naming
    """
    logger.info("Computing evaluation metrics")

    task = "binary" if config["is_binary"] else "multiclass"
    num_classes = config["num_classes"]

    # Define metrics to compute
    output_metrics = ["auroc", "average_precision", "f1_score"]

    # Compute metrics using Lightning utility
    raw_metrics = compute_metrics(
        y_prob, y_true, output_metrics, task=task, num_classes=num_classes, stage="eval"
    )

    # Convert metric names to XGBoost format (remove 'eval/' prefix, rename 'auroc' to 'auc_roc')
    metrics = {}
    for key, value in raw_metrics.items():
        # Remove 'eval/' prefix
        new_key = key.replace("eval/", "")
        # Rename 'auroc' to 'auc_roc' for XGBoost compatibility
        new_key = new_key.replace("auroc", "auc_roc")
        metrics[new_key] = value

    logger.info(f"Computed {len(metrics)} metrics using Lightning utilities")

    # Log formatted metrics summary
    log_metrics_summary(metrics, is_binary=config["is_binary"])

    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================


def generate_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate ROC and PR curve plots using Lightning utilities.
    Returns dictionary of plot file paths.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        config: Model configuration
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to file paths
    """
    logger.info("Generating evaluation plots")

    plot_paths = {}
    task = "binary" if config["is_binary"] else "multiclass"
    num_classes = config["num_classes"]

    # Create TensorBoard writer for plot generation
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard_eval"))

    # Generate ROC curves
    roc_metric_plot(
        y_pred=y_prob,
        y_true=y_true,
        y_val_pred=y_prob,  # Use same data for validation in eval
        y_val_true=y_true,
        path=output_dir,
        task=task,
        num_classes=num_classes,
        writer=writer,
        global_step=0,
    )
    plot_paths["roc_curve"] = os.path.join(output_dir, "roc_curve.jpg")
    logger.info(f"Generated ROC curve: {plot_paths['roc_curve']}")

    # Generate PR curves
    pr_metric_plot(
        y_pred=y_prob,
        y_true=y_true,
        y_val_pred=y_prob,  # Use same data for validation in eval
        y_val_true=y_true,
        path=output_dir,
        task=task,
        num_classes=num_classes,
        writer=writer,
        global_step=0,
    )
    plot_paths["pr_curve"] = os.path.join(output_dir, "pr_curve.jpg")
    logger.info(f"Generated PR curve: {plot_paths['pr_curve']}")

    writer.close()

    return plot_paths


# ============================================================================
# OUTPUT MANAGEMENT
# ============================================================================


def save_predictions_with_dataframe(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    output_eval_dir: str,
    input_format: str = "csv",
) -> None:
    """
    Save predictions by adding probability columns to existing dataframe.
    Includes id, true label, and class probabilities.

    Args:
        df: Dataframe with IDs and labels from inference (already aligned)
        y_prob: Predicted probabilities
        output_eval_dir: Directory to save predictions
        input_format: Format to save in ('csv', 'tsv', or 'parquet')
    """
    logger.info(f"Saving predictions to {output_eval_dir} in {input_format} format")

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

    output_base = Path(output_eval_dir) / "eval_predictions"
    output_path = save_dataframe_with_format(out_df, output_base, input_format)
    logger.info(f"Saved predictions (format={input_format}): {output_path}")


def save_metrics(
    metrics: Dict[str, Union[int, float, str]], output_metrics_dir: str
) -> None:
    """
    Save computed metrics as JSON and text summary.

    Args:
        metrics: Dictionary of metrics
        output_metrics_dir: Directory to save metrics
    """
    # Convert tensors to Python scalars for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_serializable[key] = (
                value.item() if value.numel() == 1 else value.tolist()
            )
        else:
            metrics_serializable[key] = value

    # Save JSON
    json_path = os.path.join(output_metrics_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    logger.info(f"Saved metrics to {json_path}")

    # Save text summary
    summary_path = os.path.join(output_metrics_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PYTORCH MODEL EVALUATION METRICS\n")
        f.write("=" * 50 + "\n\n")

        # Key metrics (using XGBoost-compatible naming)
        if "auc_roc" in metrics:
            f.write(f"AUC-ROC:           {metrics['auc_roc']:.4f}\n")
        if "average_precision" in metrics:
            f.write(f"Average Precision: {metrics['average_precision']:.4f}\n")
        if "f1_score" in metrics:
            f.write(f"F1 Score:          {metrics['f1_score']:.4f}\n")

        f.write("\n" + "=" * 50 + "\n\n")
        f.write("ALL METRICS\n")
        f.write("=" * 50 + "\n")

        for name, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                f.write(f"{name}: {value:.6f}\n")
            else:
                f.write(f"{name}: {value}\n")

    logger.info(f"Saved metrics summary to {summary_path}")


def create_health_check_file(output_path: str) -> str:
    """Create a health check file to signal script completion."""
    health_path = output_path
    with open(health_path, "w") as f:
        f.write(f"healthy: {datetime.now().isoformat()}")
    return health_path


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================


def load_eval_data(eval_data_dir: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Load the first data file found in the evaluation data directory.
    Returns a pandas DataFrame, the detected format, and the filename.
    """
    logger.info(f"Loading eval data from {eval_data_dir}")
    eval_files = sorted(
        [
            f
            for f in Path(eval_data_dir).glob("**/*")
            if f.suffix in [".csv", ".tsv", ".parquet"]
        ]
    )
    if not eval_files:
        logger.error("No eval data file found in eval_data input.")
        raise RuntimeError("No eval data file found in eval_data input.")

    eval_file = eval_files[0]
    logger.info(f"Using eval data file: {eval_file}")

    df, input_format = load_dataframe_with_format(eval_file)
    filename = str(eval_file.relative_to(eval_data_dir))
    logger.info(
        f"Loaded eval data shape: {df.shape}, format: {input_format}, filename: {filename}"
    )
    return df, input_format, filename


def get_id_label_columns(
    df: pd.DataFrame, id_field: str, label_field: str
) -> Tuple[str, str]:
    """
    Determine the ID and label columns in the DataFrame.
    Falls back to the first and second columns if not found.
    """
    id_col = id_field if id_field in df.columns else df.columns[0]
    label_col = label_field if label_field in df.columns else df.columns[1]
    logger.info(f"Using id_col: {id_col}, label_col: {label_col}")
    return id_col, label_col


def evaluate_model(
    model: nn.Module,
    df: pd.DataFrame,
    config: Dict[str, Any],
    tokenizer: AutoTokenizer,
    processors: Dict[str, Any],
    eval_data_dir: str,
    filename: str,
    id_col: str,
    label_col: str,
    output_eval_dir: str,
    output_metrics_dir: str,
    input_format: str = "csv",
    device: Union[str, int, List[int]] = "auto",
) -> None:
    """
    Run model prediction and evaluation, then save predictions and metrics.

    For multi-GPU inference, only the main process performs post-processing
    (metrics computation, plotting, saving) to avoid race conditions and
    file corruption. All processes synchronize via barrier.

    Args:
        model: PyTorch Lightning model
        df: Evaluation DataFrame
        config: Model configuration
        tokenizer: BERT tokenizer
        processors: Preprocessing processors
        eval_data_dir: Directory containing evaluation data
        filename: Name of evaluation data file
        id_col: Name of ID column (not used in refactored approach)
        label_col: Name of label column (not used in refactored approach)
        output_eval_dir: Directory to save evaluation results
        output_metrics_dir: Directory to save metrics
        input_format: Input data format
        device: Device to use for inference
    """
    logger.info("Starting model evaluation")

    # Preprocess data and create DataLoader
    pipeline_dataset, dataloader = preprocess_eval_data(
        df, config, tokenizer, processors, eval_data_dir, filename
    )

    # Setup device environment
    device_str, accelerator = setup_device_environment(device)

    # Generate predictions with dataframe (all ranks participate in DDP)
    label_name = config.get("label_name", "label")
    y_prob, y_true, eval_df = generate_predictions(
        model, dataloader, device_str, accelerator, label_name
    )

    # ===================================================================
    # CRITICAL: Only main process performs post-processing
    # This prevents race conditions when multiple GPUs try to write
    # to the same files simultaneously
    # ===================================================================
    if is_main_process():
        logger.info("=" * 70)
        logger.info("POST-PROCESSING (MAIN PROCESS ONLY)")
        logger.info("=" * 70)

        # Compute metrics
        metrics = compute_evaluation_metrics(y_true, y_prob, config)

        # Generate plots
        plot_paths = generate_evaluation_plots(
            y_true, y_prob, config, output_metrics_dir
        )

        # Save predictions with aligned dataframe
        save_predictions_with_dataframe(eval_df, y_prob, output_eval_dir, input_format)

        # Save metrics
        save_metrics(metrics, output_metrics_dir)

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
        logger.info("All ranks synchronized - evaluation complete")

    logger.info("Model evaluation complete")


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
    Main entry point for PyTorch model evaluation script.
    Loads model and data, runs evaluation, and saves results.

    Args:
        input_paths: Dictionary of input paths
        output_paths: Dictionary of output paths
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
    """
    # Extract paths from parameters - using contract-defined logical names
    model_dir = input_paths.get("model_input", input_paths.get("model_dir"))
    eval_data_dir = input_paths.get("processed_data", input_paths.get("eval_data_dir"))
    output_eval_dir = output_paths.get(
        "eval_output", output_paths.get("output_eval_dir")
    )
    output_metrics_dir = output_paths.get(
        "metrics_output", output_paths.get("output_metrics_dir")
    )

    # Extract environment variables
    id_field = environ_vars.get("ID_FIELD", "order_id")
    label_field = environ_vars.get("LABEL_FIELD", "label")

    # Parse device setting - support multiple formats
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
    logger.info(f"Running PyTorch model evaluation with job_type: {job_type}")
    logger.info(f"Device setting: {device}")

    # Ensure output directories exist
    os.makedirs(output_eval_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)

    logger.info("Starting PyTorch model evaluation script")

    # Load model artifacts
    model, config, tokenizer, processors = load_model_artifacts(model_dir)

    # Load evaluation data with format detection
    df, input_format, filename = load_eval_data(eval_data_dir)

    # Get ID and label columns
    id_col, label_col = get_id_label_columns(df, id_field, label_field)

    # Evaluate model
    evaluate_model(
        model,
        df,
        config,
        tokenizer,
        processors,
        eval_data_dir,
        filename,
        id_col,
        label_col,
        output_eval_dir,
        output_metrics_dir,
        input_format,
        device,
    )

    logger.info("PyTorch model evaluation script complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    # Set up paths using contract-defined paths only
    input_paths = {
        "model_input": CONTAINER_PATHS["MODEL_DIR"],
        "processed_data": CONTAINER_PATHS["EVAL_DATA_DIR"],
    }

    output_paths = {
        "eval_output": CONTAINER_PATHS["OUTPUT_EVAL_DIR"],
        "metrics_output": CONTAINER_PATHS["OUTPUT_METRICS_DIR"],
    }

    # Collect environment variables - aligned with contract
    environ_vars = {
        # Required environment variables
        "ID_FIELD": os.environ.get("ID_FIELD", "order_id"),
        "LABEL_FIELD": os.environ.get("LABEL_FIELD", "label"),
        # Optional environment variables (from contract)
        "COMPARISON_MODE": os.environ.get("COMPARISON_MODE", "false"),
        "PREVIOUS_SCORE_FIELD": os.environ.get("PREVIOUS_SCORE_FIELD", ""),
        "COMPARISON_METRICS": os.environ.get("COMPARISON_METRICS", "all"),
        "STATISTICAL_TESTS": os.environ.get("STATISTICAL_TESTS", "true"),
        "COMPARISON_PLOTS": os.environ.get("COMPARISON_PLOTS", "true"),
        # Additional optional variables for device/performance tuning
        "DEVICE": os.environ.get("DEVICE", "auto"),
        "ACCELERATOR": os.environ.get("ACCELERATOR", "auto"),
        "BATCH_SIZE": os.environ.get("BATCH_SIZE", "32"),
        "NUM_WORKERS": os.environ.get("NUM_WORKERS", "0"),
    }

    try:
        # Call main function
        main(input_paths, output_paths, environ_vars, args)

        # Signal success
        success_path = os.path.join(output_paths["metrics_output"], "_SUCCESS")
        Path(success_path).touch()
        logger.info(f"Created success marker: {success_path}")

        # Create health check file
        health_path = os.path.join(output_paths["metrics_output"], "_HEALTH")
        create_health_check_file(health_path)
        logger.info(f"Created health check file: {health_path}")

        sys.exit(0)
    except Exception as e:
        # Log error and create failure marker
        logger.exception(f"Script failed with error: {e}")
        failure_path = os.path.join(
            output_paths.get("metrics_output", "/tmp"), "_FAILURE"
        )
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(f"Error: {str(e)}")
        sys.exit(1)
