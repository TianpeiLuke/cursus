#!/usr/bin/env python
"""
Train custom BPE tokenizer for Names3Risk fraud detection.

This script trains a Byte Pair Encoding (BPE) tokenizer optimized for
customer name data, matching the legacy OrderTextTokenizer implementation
with automatic vocabulary size tuning to achieve target compression ratio.

References:
    - Legacy: projects/names3risk_legacy/tokenizer.py
    - HuggingFace Tokenizers: https://github.com/huggingface/tokenizers
"""

import os
import json
import argparse
import logging
import sys
import traceback
import random
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Callable

# ============================================================================
# PACKAGE INSTALLATION CONFIGURATION
# ============================================================================

# Control which PyPI source to use via environment variable
# Set USE_SECURE_PYPI=true to use secure CodeArtifact PyPI
# Set USE_SECURE_PYPI=false or leave unset to use public PyPI
USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "false").lower() == "true"

# Logging setup for installation
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

import pandas as pd

# Import custom tokenizer from cursus processing module (relative import)
from ...processing.tokenizers import CompressionBPETokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Helper Functions ---


def load_train_texts(
    train_data_path: str,
    text_field: str = "text",
    log: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """
    Load training texts from parquet file.

    Args:
        train_data_path: Path to training data directory or parquet file
        text_field: Name of text field (default: "text")
        log: Optional logging function

    Returns:
        List of text strings for tokenizer training
    """
    log = log or logger.info
    log(f"Loading training data from {train_data_path}")

    train_path = Path(train_data_path)

    # Handle directory or file
    if train_path.is_dir():
        parquet_files = list(train_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {train_data_path}")
        train_file = parquet_files[0]
    else:
        train_file = train_path

    # Load data
    df = pd.read_parquet(train_file)

    if text_field not in df.columns:
        raise ValueError(
            f"Column '{text_field}' not found in data. Available: {df.columns.tolist()}"
        )

    # Extract texts and remove nulls
    texts = df[text_field].dropna().tolist()

    log(f"Loaded {len(texts):,} training texts")
    if texts:
        log(f"Sample text: {texts[0][:100]}...")

    return texts


def save_tokenizer_artifacts(
    tokenizer: CompressionBPETokenizer,
    output_dir: str,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Save tokenizer artifacts to output directory.

    Args:
        tokenizer: Trained CompressionBPETokenizer
        output_dir: Output directory path
        log: Optional logging function
    """
    log = log or logger.info
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer (main artifact)
    tokenizer_file = output_path / "tokenizer.json"
    tokenizer.tokenizer.save(str(tokenizer_file))
    log(f"Saved tokenizer to {tokenizer_file}")

    # Save vocabulary (for legacy compatibility)
    vocab = tokenizer.tokenizer.get_vocab()
    vocab_file = output_path / "vocab.json"
    with open(vocab_file, "w") as f:
        json.dump(vocab, f, indent=2)
    log(f"Saved vocabulary to {vocab_file}")

    # Save metadata
    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "model_type": "BPE",
        "special_tokens": [
            "[CLS]",
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]",
            "[MISSING]",
            "|",
        ],
        "normalizer": "NFKC",
        "pre_tokenizer": "Whitespace",
        "pad_token": tokenizer.pad_token,
        "cls_token": tokenizer.cls_token,
        "min_frequency": tokenizer.min_frequency,
    }
    metadata_file = output_path / "tokenizer_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"Saved metadata to {metadata_file}")


# --- Main Processing Logic ---


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger_func: Optional[Callable[[str], None]] = None,
) -> Dict[str, any]:
    """
    Main logic for tokenizer training, refactored for testability.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger_func: Optional logger function (defaults to print if None)

    Returns:
        Dictionary with training results and metadata
    """
    # Extract parameters
    train_data_path = input_paths["input_data"]
    output_dir = output_paths["model_artifacts_output"]

    text_field = environ_vars.get("TEXT_FIELD", "text")
    target_compression = float(environ_vars.get("TARGET_COMPRESSION", "2.5"))
    min_frequency = int(environ_vars.get("MIN_FREQUENCY", "25"))
    max_vocab_size = int(environ_vars.get("MAX_VOCAB_SIZE", "50000"))

    # Use print function if no logger is provided
    log = logger_func or print

    try:
        # 1. Load training texts
        log("[INFO] Step 1: Loading training texts...")
        texts = load_train_texts(train_data_path, text_field, log=log)

        if len(texts) == 0:
            raise ValueError("No training texts found")

        # 2. Train tokenizer using CompressionBPETokenizer
        log("[INFO] Step 2: Training BPE tokenizer with compression tuning...")
        tokenizer = CompressionBPETokenizer(min_frequency=min_frequency)
        tokenizer.train(
            texts=texts,
            target_compression=target_compression,
            max_vocab_size=max_vocab_size,
        )

        # 3. Save artifacts
        log("[INFO] Step 3: Saving tokenizer artifacts...")
        save_tokenizer_artifacts(tokenizer, output_dir, log=log)

        # 4. Return results
        results = {
            "vocab_size": tokenizer.vocab_size,
            "num_training_texts": len(texts),
            "target_compression": target_compression,
            "min_frequency": min_frequency,
            "pad_token": tokenizer.pad_token,
            "cls_token": tokenizer.cls_token,
        }

        log("[INFO] Tokenizer training completed successfully")
        return results

    except Exception as e:
        log(f"[ERROR] Tokenizer training failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Train BPE tokenizer for Names3Risk"
        )
        parser.add_argument(
            "--job_type",
            type=str,
            required=True,
            choices=["training", "validation", "testing", "calibration"],
            help="One of ['training','validation','testing','calibration']",
        )
        args = parser.parse_args()

        # Read configuration from environment variables
        TEXT_FIELD = os.environ.get("TEXT_FIELD", "text")
        TARGET_COMPRESSION = os.environ.get("TARGET_COMPRESSION", "2.5")
        MIN_FREQUENCY = os.environ.get("MIN_FREQUENCY", "25")
        MAX_VOCAB_SIZE = os.environ.get("MAX_VOCAB_SIZE", "50000")

        # Define standard SageMaker paths as constants
        INPUT_TRAIN_DATA_DIR = "/opt/ml/processing/input/train"
        OUTPUT_TOKENIZER_DIR = "/opt/ml/processing/output"

        # Log key parameters
        logger.info(f"Starting tokenizer training with parameters:")
        logger.info(f"  Job Type: {args.job_type}")
        logger.info(f"  Input Train Data Directory: {INPUT_TRAIN_DATA_DIR}")
        logger.info(f"  Output Tokenizer Directory: {OUTPUT_TOKENIZER_DIR}")
        logger.info(f"  Text Field: {TEXT_FIELD}")
        logger.info(f"  Target Compression: {TARGET_COMPRESSION}")
        logger.info(f"  Min Frequency: {MIN_FREQUENCY}")
        logger.info(f"  Max Vocab Size: {MAX_VOCAB_SIZE}")

        # Set up path dictionaries
        input_paths = {"input_data": INPUT_TRAIN_DATA_DIR}
        output_paths = {"model_artifacts_output": OUTPUT_TOKENIZER_DIR}

        # Environment variables dictionary
        environ_vars = {
            "TEXT_FIELD": TEXT_FIELD,
            "TARGET_COMPRESSION": TARGET_COMPRESSION,
            "MIN_FREQUENCY": MIN_FREQUENCY,
            "MAX_VOCAB_SIZE": MAX_VOCAB_SIZE,
        }

        # Execute the main processing logic
        results = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger_func=logger.info,
        )

        # Log completion summary
        logger.info(f"Training completed successfully:")
        logger.info(f"  Vocabulary size: {results['vocab_size']}")
        logger.info(f"  Training texts: {results['num_training_texts']}")
        logger.info(f"  PAD token ID: {results['pad_token']}")
        logger.info(f"  CLS token ID: {results['cls_token']}")

        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in tokenizer training script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
