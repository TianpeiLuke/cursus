import os
import json
import traceback
from io import StringIO, BytesIO
from pathlib import Path
import logging
from typing import List, Union, Dict, Tuple, Optional, Any
import pickle as pkl
import sys
import time
from contextlib import contextmanager

# ============================================================================
# PACKAGE INSTALLATION CONFIGURATION
# ============================================================================

# Control which PyPI source to use via environment variable
# Set USE_SECURE_PYPI=true to use secure CodeArtifact PyPI
# Set USE_SECURE_PYPI=false or leave unset to use public PyPI
USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "true").lower() == "true"

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
# INSTALL REQUIRED PACKAGES WITH MULTI-WORKER SAFETY
# ============================================================================

import fcntl
import hashlib
import torch  # Import torch early to check CUDA availability


def install_packages_once(requirements_file: str, use_secure: bool = USE_SECURE_PYPI):
    """
    Thread-safe package installation using file lock to prevent race conditions.

    When multiple TorchServe workers start simultaneously, only ONE worker
    installs packages while others wait. Uses file lock + installation marker.

    Args:
        requirements_file: Path to requirements file
        use_secure: Whether to use secure PyPI
    """
    # Create secure temp directory for lock files (CWE-379 mitigation)
    # Use environment-provided directory (SageMaker), otherwise Python's secure temp
    import tempfile

    secure_temp_dir = os.environ.get("SM_MODEL_DIR")
    if (
        not secure_temp_dir
        or not os.path.exists(secure_temp_dir)
        or not os.access(secure_temp_dir, os.W_OK)
    ):
        # Use Python's secure temp directory with restricted permissions
        secure_temp_dir = tempfile.gettempdir()

    # Create lock and marker files in secure directory
    lock_file = os.path.join(secure_temp_dir, ".pytorch_inference_packages.lock")

    # Create installation marker based on requirements file hash
    # Note: Using SHA256 for content hashing (not cryptographic security)
    with open(requirements_file, "rb") as f:
        req_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    marker_file = os.path.join(
        secure_temp_dir, f".packages_installed_{req_hash}.marker"
    )

    # Check if packages already installed (fast path)
    if os.path.exists(marker_file):
        print(f"✓ Packages already installed (marker found: {marker_file})")
        return

    # Check if previous installation failed (prevent thundering herd)
    failure_marker = f"{marker_file}.failed"
    if os.path.exists(failure_marker):
        with open(failure_marker, "r") as f:
            failure_info = f.read()
        print(f"✗ Previous installation failed (marker found: {failure_marker})")
        print(f"Failure details: {failure_info}")
        raise RuntimeError(
            f"Package installation previously failed. See {failure_marker} for details."
        )

    print(f"Acquiring installation lock: {lock_file}")

    # Acquire exclusive lock with timeout (prevents indefinite hang if worker crashes)
    with open(lock_file, "w") as lock:
        try:
            # Acquire lock with 300-second timeout
            timeout = 300
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    print("✓ Lock acquired")
                    break
                except BlockingIOError:
                    time.sleep(1)  # Retry every second
            else:
                raise TimeoutError(
                    f"Failed to acquire lock after {timeout} seconds - another worker may have crashed"
                )

            # Double-check marker (another worker may have installed while we waited)
            if os.path.exists(marker_file):
                print(f"✓ Packages installed by another worker (marker: {marker_file})")
                return

            # WE are the first worker - install packages
            print("📦 This worker will install packages")

            # Read requirements
            with open(requirements_file, "r") as f:
                required_packages = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]

            print(f"Loaded {len(required_packages)} packages from {requirements_file}")

            try:
                # Install packages
                install_packages(required_packages, use_secure)

                # Create success marker
                with open(marker_file, "w") as marker:
                    marker.write(f"Installed at {time.time()}\n")

                print(f"✓ Created installation marker: {marker_file}")
                print(
                    "***********************Package Installation Complete*********************"
                )

            except Exception as e:
                # Create failure marker to prevent thundering herd
                failure_marker = f"{marker_file}.failed"
                with open(failure_marker, "w") as marker:
                    marker.write(f"Failed at {time.time()}: {str(e)}\n")
                print(f"✗ Installation failed, created marker: {failure_marker}")
                raise  # Re-raise to prevent serving with missing dependencies

        finally:
            # Release lock automatically when 'with' block exits
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            print("✓ Lock released")


# Load packages from requirements file with multi-worker safety
# Dynamically select GPU or CPU requirements based on CUDA availability
if torch.cuda.is_available():
    requirements_file = os.path.join(
        os.path.dirname(__file__), "requirements-gpu-secure.txt"
    )
    print("=" * 70)
    print("✓ CUDA available - loading GPU-optimized dependencies")
    print("  Using: requirements-gpu-secure.txt (onnxruntime-gpu)")
    print("=" * 70)
else:
    requirements_file = os.path.join(
        os.path.dirname(__file__), "requirements-secure.txt"
    )
    print("=" * 70)
    print("✓ CUDA not available - loading CPU dependencies")
    print("  Using: requirements-secure.txt (onnxruntime)")
    print("=" * 70)

try:
    install_packages_once(requirements_file, USE_SECURE_PYPI)
except FileNotFoundError:
    print(f"Warning: {requirements_file} not found. Skipping package installation.")
    print("Assuming packages are already installed in the environment.")
except Exception as e:
    print(f"Error loading or installing packages: {e}")
    raise

# ============================================================================
# IMPORT INSTALLED PACKAGES (AFTER INSTALLATION)
# ============================================================================

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tokenizers import Tokenizer  # HuggingFace tokenizers for custom BPE

from processing.processors import (
    Processor,
)
from processing.text.dialogue_processor import (
    HTMLNormalizerProcessor,
    EmojiRemoverProcessor,
    TextNormalizationProcessor,
    DialogueSplitterProcessor,
    DialogueChunkerProcessor,
)
from processing.text.bert_tokenize_processor import BertTokenizeProcessor
from processing.categorical.categorical_label_processor import CategoricalLabelProcessor
from processing.categorical.multiclass_label_processor import MultiClassLabelProcessor
from processing.categorical.risk_table_processor import RiskTableMappingProcessor
from processing.numerical.numerical_imputation_processor import (
    NumericalVariableImputationProcessor,
)
from processing.processor_registry import build_text_pipeline_from_steps
from processing.datasets.pipeline_datasets import PipelineDataset
from processing.dataloaders.pipeline_dataloader import build_collate_batch

from lightning_models.utils.pl_train import (
    model_inference,
    model_online_inference,
    load_model,
    load_artifacts,
    load_onnx_model,
    load_bert_optimized_model,
)
from lightning_models.utils.dist_utils import get_rank, is_main_process
from pydantic import BaseModel, Field, ValidationError  # For Config Validation

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


# ============================================================================
# SHARED TOKENIZER CACHE CONFIGURATION
# ============================================================================

# Configure shared transformers cache for faster cold starts
# This allows multiple workers to share the same tokenizer cache
SHARED_CACHE_DIR = "/tmp/transformers_cache"
os.makedirs(SHARED_CACHE_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = SHARED_CACHE_DIR
os.environ["HF_HOME"] = SHARED_CACHE_DIR

logger.info(f"Configured shared tokenizer cache at: {SHARED_CACHE_DIR}")


# ============================================================================
# PERFORMANCE TIMING UTILITIES
# ============================================================================


@contextmanager
def log_timing(operation_name: str, logger_instance=None):
    """
    Context manager to measure and log execution time of operations.

    Usage:
        with log_timing("Model Inference"):
            model.predict(data)

    Args:
        operation_name: Name of the operation being timed
        logger_instance: Logger to use (defaults to module logger)
    """
    log = logger_instance or logger
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_ms = (time.time() - start_time) * 1000
        log.info(f"⏱️  {operation_name}: {elapsed_ms:.2f}ms")


# ================== Model, Data and Hyperparameter Folder =================
prefix = "/opt/ml/"
input_path = os.path.join(prefix, "input/data")
output_path = os.path.join(prefix, "output")
model_path = os.path.join(prefix, "model")
hparam_path = os.path.join(prefix, "input/config/hyperparameters.json")
checkpoint_path = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
train_channel = "train"
train_path = os.path.join(input_path, train_channel)
val_channel = "val"
val_path = os.path.join(input_path, val_channel)
test_channel = "test"
test_path = os.path.join(input_path, test_channel)
# ==========================================

# File names for preprocessing artifacts
RISK_TABLE_FILE = "risk_table_map.pkl"
IMPUTE_DICT_FILE = "impute_dict.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================================
class Config(BaseModel):
    id_name: str = "order_id"
    text_name: Optional[str] = None  # Changed to Optional to match training
    label_name: str = "label"
    batch_size: int = 32
    full_field_list: List[str] = Field(default_factory=list)
    cat_field_list: List[str] = Field(default_factory=list)
    tab_field_list: List[str] = Field(default_factory=list)
    max_sen_len: int = 512
    chunk_trancate: bool = False
    max_total_chunks: int = 5
    kernel_size: List[int] = Field(default_factory=lambda: [3, 5, 7])
    num_layers: int = 2
    num_channels: List[int] = Field(default_factory=lambda: [100, 100])
    hidden_common_dim: int = 100
    input_tab_dim: int = 11
    num_classes: int = 2
    is_binary: bool = True
    multiclass_categories: List[Union[int, str]] = Field(default_factory=lambda: [0, 1])
    max_epochs: int = 10
    lr: float = 0.02
    lr_decay: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0
    class_weights: List[float] = Field(default_factory=lambda: [1.0, 10.0])
    dropout_keep: float = 0.5
    optimizer: str = "SGD"
    fixed_tokenizer_length: bool = True
    is_embeddings_trainable: bool = True
    tokenizer: str = "bert-base-multilingual-cased"
    metric_choices: List[str] = Field(default_factory=lambda: ["auroc", "f1_score"])
    early_stop_metric: str = "val/f1_score"
    early_stop_patience: int = 3
    gradient_clip_val: float = 1.0
    model_class: str = "multimodal_bert"
    load_ckpt: bool = False
    val_check_interval: float = 0.25
    adam_epsilon: float = 1e-08
    fp16: bool = False
    use_gradient_checkpointing: bool = False  # Added to match training
    run_scheduler: bool = True
    reinit_pooler: bool = True
    reinit_layers: int = 2
    warmup_steps: int = 300
    text_input_ids_key: str = "input_ids"  # Configurable text input key
    text_attention_mask_key: str = "attention_mask"  # Configurable attention mask key
    # Added fields for trimodal model support
    primary_text_name: Optional[str] = None
    secondary_text_name: Optional[str] = None
    embed_size: Optional[int] = None  # Added for type consistency
    label_to_id: Optional[Dict[str, int]] = None  # Added: label to ID mapping
    id_to_label: Optional[List[str]] = None  # Added: ID to label mapping
    # Added fields for text processing steps configuration
    text_processing_steps: Optional[List[str]] = None
    primary_text_processing_steps: Optional[List[str]] = None
    secondary_text_processing_steps: Optional[List[str]] = None
    # Added fields for preprocessing artifact storage
    imputation_dict: Optional[Dict[str, float]] = None
    risk_tables: Optional[Dict[str, Dict]] = None
    # Added metadata fields
    _input_format: Optional[str] = None
    smooth_factor: float = 0.0
    count_threshold: int = 0

    def model_post_init(self, __context):
        # Validate consistency between multiclass_categories and num_classes
        if self.is_binary and self.num_classes != 2:
            raise ValueError("For binary classification, num_classes must be 2.")
        if not self.is_binary:
            if self.num_classes < 2:
                raise ValueError(
                    "For multiclass classification, num_classes must be >= 2."
                )
            if not self.multiclass_categories:
                raise ValueError(
                    "multiclass_categories must be provided for multiclass classification."
                )
            if len(self.multiclass_categories) != self.num_classes:
                raise ValueError(
                    f"num_classes={self.num_classes} does not match "
                    f"len(multiclass_categories)={len(self.multiclass_categories)}"
                )
            if len(set(self.multiclass_categories)) != len(self.multiclass_categories):
                raise ValueError("multiclass_categories must contain unique values.")
        else:
            # Optional: Warn if multiclass_categories is defined when binary
            if self.multiclass_categories and len(self.multiclass_categories) != 2:
                raise ValueError(
                    "For binary classification, multiclass_categories must contain exactly 2 items."
                )

        # New: validate class_weights length
        if self.class_weights and len(self.class_weights) != self.num_classes:
            raise ValueError(
                f"class_weights must have the same number of elements as num_classes "
                f"(expected {self.num_classes}, got {len(self.class_weights)})."
            )


# =================== Helper Functions ================
def read_feature_columns(model_dir: str) -> Optional[List[str]]:
    """
    Read feature columns in correct order from feature_columns.txt

    Args:
        model_dir: Directory containing model artifacts

    Returns:
        List[str]: Ordered list of feature column names, or None if file doesn't exist

    Raises:
        ValueError: If file format is invalid
    """
    feature_file = os.path.join(model_dir, "feature_columns.txt")

    if not os.path.exists(feature_file):
        logger.warning(f"feature_columns.txt not found in {model_dir}")
        return None

    ordered_features = []

    try:
        with open(feature_file, "r") as f:
            for line in f:
                # Skip comments
                if line.startswith("#"):
                    continue
                # Parse "<index>,<column_name>" format
                try:
                    idx, column = line.strip().split(",")
                    ordered_features.append(column)
                except ValueError:
                    continue

        if not ordered_features:
            raise ValueError(f"No valid feature columns found in {feature_file}")

        logger.info(
            f"Loaded {len(ordered_features)} ordered feature columns from feature_columns.txt"
        )
        return ordered_features
    except Exception as e:
        logger.error(f"Error reading feature columns file: {e}", exc_info=True)
        raise


def validate_input_data(input_data: pd.DataFrame, feature_columns: List[str]) -> None:
    """
    Validate input data meets requirements.

    Args:
        input_data: Input DataFrame
        feature_columns: Expected feature columns

    Raises:
        ValueError: If validation fails
    """
    if input_data.empty:
        raise ValueError("Input DataFrame is empty")

    # If input is headerless CSV, validate column count
    if all(isinstance(col, int) for col in input_data.columns):
        if len(input_data.columns) != len(feature_columns):
            raise ValueError(
                f"Input data has {len(input_data.columns)} columns but model expects {len(feature_columns)} features"
            )
    else:
        # Validate required features present
        missing_features = set(feature_columns) - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")


def assign_column_names(
    input_data: pd.DataFrame, feature_columns: List[str]
) -> pd.DataFrame:
    """
    Assign column names to headerless input data.

    Args:
        input_data: Input DataFrame
        feature_columns: Feature column names to assign

    Returns:
        DataFrame with assigned column names
    """
    df = input_data.copy()
    if all(isinstance(col, int) for col in df.columns):
        df.columns = feature_columns
    return df


def load_hyperparameters(model_dir: str) -> Dict[str, Any]:
    """
    Load hyperparameters from hyperparameters.json file.

    Args:
        model_dir: Directory containing model artifacts

    Returns:
        Dict[str, Any]: Hyperparameters dictionary, empty dict if file doesn't exist
    """
    hyperparams_file = os.path.join(model_dir, "hyperparameters.json")

    if not os.path.exists(hyperparams_file):
        logger.warning(f"hyperparameters.json not found in {model_dir}")
        return {}

    try:
        with open(hyperparams_file, "r") as f:
            hyperparams = json.load(f)
        logger.info(f"Loaded hyperparameters from hyperparameters.json")
        return hyperparams
    except Exception as e:
        logger.warning(f"Could not load hyperparameters.json: {e}")
        return {}


def get_text_field_names(config: Config) -> set:
    """Identify text field names to exclude from risk table processing."""
    text_fields = set()
    if hasattr(config, "text_name") and config.text_name:
        text_fields.add(config.text_name)
    if hasattr(config, "primary_text_name") and config.primary_text_name:
        text_fields.add(config.primary_text_name)
    if hasattr(config, "secondary_text_name") and config.secondary_text_name:
        text_fields.add(config.secondary_text_name)
    return text_fields


def create_text_field_for_names3risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'text' field by concatenating 4 name fields for Names3Risk inference.

    Replicates the exact preprocessing logic from tabular_preprocessing.py
    used during training data preparation.

    Args:
        df: Input DataFrame with raw fields

    Returns:
        DataFrame with 'text' field added (or original if text already exists)
    """
    text_fields = [
        "emailAddress",
        "billingAddressName",
        "customerName",
        "paymentAccountHolderName",
    ]

    # Only create if text field doesn't already exist
    if "text" not in df.columns:
        # Check if all required fields exist
        missing_fields = [f for f in text_fields if f not in df.columns]
        if missing_fields:
            logger.warning(
                f"Missing fields for text creation: {missing_fields}. "
                f"Creating empty text field as fallback."
            )
            # Create empty text field as fallback
            df["text"] = ""
        else:
            # Replicate exact preprocessing logic from tabular_preprocessing.py
            # Lines 349-353: df["text"] = df[text_fields].fillna("[MISSING]").agg("|".join, axis=1)
            df["text"] = df[text_fields].fillna("[MISSING]").agg("|".join, axis=1)
            logger.info(
                f"✓ Created 'text' field by concatenating {len(text_fields)} name fields"
            )
    else:
        logger.debug("'text' field already exists in input data, skipping creation")

    return df


def load_risk_tables(model_dir: str) -> Dict[str, Any]:
    """Load risk tables from pickle file."""
    risk_file = os.path.join(model_dir, RISK_TABLE_FILE)
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
    impute_file = os.path.join(model_dir, IMPUTE_DICT_FILE)
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
    """Create numerical imputation processors for each numerical feature."""
    numerical_processors = {}
    for feature, imputation_value in impute_dict.items():
        processor = NumericalVariableImputationProcessor(
            column_name=feature, imputation_value=imputation_value
        )
        numerical_processors[feature] = processor
    logger.info(f"Created {len(numerical_processors)} numerical imputation processors")
    return numerical_processors


def data_preprocess_pipeline(
    config: Config,
    tokenizer: Union[Tokenizer, AutoTokenizer],
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Tuple[Union[Tokenizer, AutoTokenizer], Dict[str, Processor]]:
    """
    Build text preprocessing pipelines using provided tokenizer.

    For Names3Risk models (lstm2risk, transformer2risk), uses custom BPE tokenizer.
    For other models (bimodal_bert, etc.), uses pretrained BERT tokenizer.

    For bimodal: Uses text_name with default or configured steps
    For trimodal: Uses primary_text_name and secondary_text_name with separate step lists

    Args:
        config: Configuration object
        tokenizer: Pre-loaded tokenizer (HuggingFace Tokenizer or AutoTokenizer)
        hyperparameters: Optional hyperparameters dict loaded from hyperparameters.json

    Returns:
        Tuple of (tokenizer, pipelines_dict)
    """
    # Determine if custom tokenizer is being used
    model_class = config.model_class
    needs_custom_tokenizer = model_class in ["lstm2risk", "transformer2risk"]

    # Log padding strategy
    padding_strategy = (
        "fixed (max_sen_len)" if config.fixed_tokenizer_length else "dynamic (longest)"
    )
    logger.info(f"Tokenizer padding strategy: {padding_strategy}")
    if not config.fixed_tokenizer_length:
        logger.info(
            f"Dynamic padding enabled - will pad to actual max length instead of {config.max_sen_len}"
        )

    pipelines = {}

    # Extract text field names from hyperparameters or config
    primary_text_name = None
    secondary_text_name = None

    if hyperparameters:
        primary_text_name = hyperparameters.get("primary_text_name")
        secondary_text_name = hyperparameters.get("secondary_text_name")
    else:
        # Fallback to config values
        primary_text_name = getattr(config, "primary_text_name", None)
        secondary_text_name = getattr(config, "secondary_text_name", None)

    # BIMODAL: Single text pipeline
    if not primary_text_name:
        # Use configured steps from hyperparameters or config, or fallback to default
        text_name = getattr(config, "text_name", None)
        if text_name is None and hyperparameters:
            text_name = hyperparameters.get("text_name")

        steps = None
        if hyperparameters:
            steps = hyperparameters.get("text_processing_steps")
        if steps is None:
            # Fallback to config attribute
            steps = getattr(config, "text_processing_steps", None)
        if steps is None:
            # Use smart defaults based on tokenizer type
            # For Names3Risk models with custom BPE tokenizer: use custom_bpe_tokenizer
            # For BERT models: use standard tokenizer
            if needs_custom_tokenizer:
                steps = [
                    "custom_bpe_tokenizer"
                ]  # Custom BPE for lstm2risk/transformer2risk
            else:
                # NOTE: html_normalizer disabled for performance (can add 50-100ms per request)
                steps = [
                    "dialogue_splitter",
                    # "html_normalizer",  # Disabled - very time consuming
                    "emoji_remover",
                    "text_normalizer",
                    "dialogue_chunker",
                    "tokenizer",
                ]

        if text_name:
            pipelines[text_name] = build_text_pipeline_from_steps(
                processing_steps=steps,
                tokenizer=tokenizer,
                max_sen_len=config.max_sen_len,
                chunk_trancate=config.chunk_trancate,
                max_total_chunks=config.max_total_chunks,
                input_ids_key=config.text_input_ids_key,
                attention_mask_key=config.text_attention_mask_key,
            )
            logger.info(f"Built bimodal pipeline for '{text_name}' with steps: {steps}")
            logger.info(
                f"  Output keys: input_ids={config.text_input_ids_key}, attention_mask={config.text_attention_mask_key}"
            )

    # TRIMODAL: Dual text pipelines
    else:
        # Primary text pipeline (e.g., chat - full cleaning)
        primary_steps = None
        if hyperparameters:
            primary_steps = hyperparameters.get("primary_text_processing_steps")
        if primary_steps is None:
            # Fallback to config attribute or default
            primary_steps = getattr(config, "primary_text_processing_steps", None)
        if primary_steps is None:
            # Default steps
            # NOTE: html_normalizer disabled for performance (can add 50-100ms per request)
            primary_steps = [
                "dialogue_splitter",
                # "html_normalizer",  # Disabled - very time consuming
                "emoji_remover",
                "text_normalizer",
                "dialogue_chunker",
                "tokenizer",
            ]

        pipelines[primary_text_name] = build_text_pipeline_from_steps(
            processing_steps=primary_steps,
            tokenizer=tokenizer,
            max_sen_len=config.max_sen_len,
            chunk_trancate=config.chunk_trancate,
            max_total_chunks=config.max_total_chunks,
            input_ids_key=config.text_input_ids_key,
            attention_mask_key=config.text_attention_mask_key,
        )
        logger.info(
            f"Built primary pipeline for '{primary_text_name}' with steps: {primary_steps}"
        )

        # Secondary text pipeline (e.g., events - minimal cleaning)
        secondary_steps = None
        if hyperparameters:
            secondary_steps = hyperparameters.get("secondary_text_processing_steps")
        if secondary_steps is None:
            # Fallback to config attribute or default
            secondary_steps = getattr(config, "secondary_text_processing_steps", None)
        if secondary_steps is None:
            # Default steps
            secondary_steps = [
                "dialogue_splitter",
                "text_normalizer",
                "dialogue_chunker",
                "tokenizer",
            ]

        pipelines[secondary_text_name] = build_text_pipeline_from_steps(
            processing_steps=secondary_steps,
            tokenizer=tokenizer,
            max_sen_len=config.max_sen_len,
            chunk_trancate=config.chunk_trancate,
            max_total_chunks=config.max_total_chunks,
            input_ids_key=config.text_input_ids_key,
            attention_mask_key=config.text_attention_mask_key,
        )
        logger.info(
            f"Built secondary pipeline for '{secondary_text_name}' with steps: {secondary_steps}"
        )

    return tokenizer, pipelines


def load_calibration_model(model_dir: str) -> Optional[Dict]:
    """
    Load calibration model if it exists. Supports both regular calibration models
    (calibration_model.pkl) and percentile calibration (percentile_score.pkl).

    Args:
        model_dir: Directory containing model artifacts

    Returns:
        Calibration model if found, None otherwise. Returns a dictionary with
        'type' and 'data' keys.
    """
    # Define calibration file constants
    CALIBRATION_DIR = "calibration"
    CALIBRATION_MODEL_FILE = "calibration_model.pkl"
    PERCENTILE_SCORE_FILE = "percentile_score.pkl"
    CALIBRATION_MODELS_DIR = "calibration_models"

    # Check for percentile calibration first
    percentile_path = os.path.join(model_dir, CALIBRATION_DIR, PERCENTILE_SCORE_FILE)
    if os.path.exists(percentile_path):
        logger.info(f"Loading percentile calibration from {percentile_path}")
        try:
            with open(percentile_path, "rb") as f:
                percentile_mapping = pkl.load(f)
                return {"type": "percentile", "data": percentile_mapping}
        except Exception as e:
            logger.warning(f"Failed to load percentile calibration: {e}")

    # Check for binary calibration model
    calibration_path = os.path.join(model_dir, CALIBRATION_DIR, CALIBRATION_MODEL_FILE)
    if os.path.exists(calibration_path):
        logger.info(f"Loading binary calibration model from {calibration_path}")
        try:
            with open(calibration_path, "rb") as f:
                return {"type": "regular", "data": pkl.load(f)}
        except Exception as e:
            logger.warning(f"Failed to load binary calibration model: {e}")

    # Check for multiclass calibration models
    multiclass_dir = os.path.join(model_dir, CALIBRATION_DIR, CALIBRATION_MODELS_DIR)
    if os.path.exists(multiclass_dir) and os.path.isdir(multiclass_dir):
        logger.info(f"Loading multiclass calibration models from {multiclass_dir}")
        try:
            calibrators = {}
            for file in os.listdir(multiclass_dir):
                if file.endswith(".pkl"):
                    class_name = file.replace("calibration_model_class_", "").replace(
                        ".pkl", ""
                    )
                    with open(os.path.join(multiclass_dir, file), "rb") as f:
                        calibrators[class_name] = pkl.load(f)
            if calibrators:
                return {"type": "regular_multiclass", "data": calibrators}
        except Exception as e:
            logger.warning(f"Failed to load multiclass calibration models: {e}")

    logger.info("No calibration model found")
    return None


def _interpolate_score(
    raw_score: float, lookup_table: List[Tuple[float, float]]
) -> float:
    """
    Interpolate calibrated score from lookup table.

    Args:
        raw_score: Raw model score (0-1)
        lookup_table: List of (raw_score, calibrated_score) tuples

    Returns:
        Interpolated calibrated score
    """
    # Boundary cases
    if raw_score <= lookup_table[0][0]:
        return lookup_table[0][1]
    if raw_score >= lookup_table[-1][0]:
        return lookup_table[-1][1]

    # Find bracketing points and perform linear interpolation
    for i in range(len(lookup_table) - 1):
        if lookup_table[i][0] <= raw_score <= lookup_table[i + 1][0]:
            x1, y1 = lookup_table[i]
            x2, y2 = lookup_table[i + 1]
            if x2 == x1:
                return y1
            return y1 + (y2 - y1) * (raw_score - x1) / (x2 - x1)

    return lookup_table[-1][1]


def apply_percentile_calibration(
    scores: np.ndarray, percentile_mapping: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Apply percentile score mapping to raw scores.

    Args:
        scores: Raw model prediction scores (N x 2 for binary classification)
        percentile_mapping: List of (raw_score, percentile) tuples

    Returns:
        Calibrated scores with same shape as input
    """
    # Apply percentile calibration to class-1 probabilities
    calibrated = np.zeros_like(scores)
    for i in range(scores.shape[0]):
        raw_class1_prob = scores[i, 1]
        calibrated_class1_prob = _interpolate_score(raw_class1_prob, percentile_mapping)
        calibrated[i, 1] = calibrated_class1_prob
        calibrated[i, 0] = 1 - calibrated_class1_prob

    return calibrated


def apply_regular_binary_calibration(
    scores: np.ndarray, calibrator: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Apply regular calibration to binary classification scores using lookup table.

    Args:
        scores: Raw model prediction scores (N x 2)
        calibrator: Lookup table List[Tuple[float, float]]

    Returns:
        Calibrated scores with same shape as input
    """
    calibrated = np.zeros_like(scores)

    # Apply lookup table calibration
    for i in range(scores.shape[0]):
        calibrated[i, 1] = _interpolate_score(scores[i, 1], calibrator)
        calibrated[i, 0] = 1 - calibrated[i, 1]

    return calibrated


def apply_regular_multiclass_calibration(
    scores: np.ndarray, calibrators: Dict[str, List[Tuple[float, float]]]
) -> np.ndarray:
    """
    Apply regular calibration to multiclass scores.

    Args:
        scores: Raw model prediction scores (N x num_classes)
        calibrators: Dictionary of calibration lookup tables, one per class

    Returns:
        Calibrated and normalized scores with same shape as input
    """
    calibrated = np.zeros_like(scores)

    # Apply calibration to each class
    for i in range(scores.shape[1]):
        class_name = str(i)
        if class_name in calibrators:
            for j in range(scores.shape[0]):
                calibrated[j, i] = _interpolate_score(
                    scores[j, i], calibrators[class_name]
                )
        else:
            calibrated[:, i] = scores[:, i]  # No calibrator for this class

    # Normalize probabilities to sum to 1
    row_sums = calibrated.sum(axis=1)
    calibrated = calibrated / row_sums[:, np.newaxis]

    return calibrated


def apply_calibration(
    scores: np.ndarray, calibrator: Dict[str, Any], is_multiclass: bool
) -> np.ndarray:
    """
    Apply calibration to raw model scores. Supports both regular calibration models
    and percentile calibration.

    Args:
        scores: Raw model prediction scores
        calibrator: Loaded calibration model(s) or percentile mapping
        is_multiclass: Whether this is a multiclass model

    Returns:
        Calibrated scores
    """
    if calibrator is None:
        return scores

    try:
        # Handle percentile calibration
        if calibrator.get("type") == "percentile":
            if is_multiclass:
                logger.warning(
                    "Percentile calibration not supported for multiclass, using raw scores"
                )
                return scores
            else:
                logger.info("Applying percentile calibration")
                return apply_percentile_calibration(scores, calibrator["data"])

        # Handle regular calibration models
        elif calibrator.get("type") in ["regular", "regular_multiclass"]:
            actual_calibrator = calibrator["data"]

            if calibrator.get("type") == "regular_multiclass" or is_multiclass:
                logger.info("Applying regular multiclass calibration")
                return apply_regular_multiclass_calibration(scores, actual_calibrator)
            else:
                logger.info("Applying regular binary calibration")
                return apply_regular_binary_calibration(scores, actual_calibrator)

        else:
            logger.warning(f"Unknown calibrator type: {calibrator.get('type')}")
            return scores

    except Exception as e:
        logger.error(f"Error applying calibration: {str(e)}", exc_info=True)
        return scores


def _safe_float_convert(val, default=0.0):
    """
    Convert any value to float, handling edge cases.

    Handles string representations of numbers, None values, NaN, etc.
    Critical for CSV input where numerical values come as strings.

    Args:
        val: Value to convert (can be str, int, float, None, np.float, etc.)
        default: Default value if conversion fails

    Returns:
        float: Converted value or default
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        logger.warning(
            f"Failed to convert value '{val}' to float, using default {default}"
        )
        return default


def preprocess_single_record_with_text(
    df: pd.DataFrame,
    config: Config,
    pipelines: Dict[str, Any],
    risk_processors: Dict[str, RiskTableMappingProcessor],
    numerical_processors: Dict[str, NumericalVariableImputationProcessor],
) -> Dict[str, torch.Tensor]:
    """
    Fast path for single-record preprocessing INCLUDING text features.

    Uses pre-built pipelines directly - NO DataLoader overhead!
    Processes text, tabular, and categorical features in a single pass.

    IMPORTANT: Replicates collate_batch behavior for trimodal models:
    - Handles List[Dict] return from tokenizer pipeline
    - Creates rank-3 tensors: (batch_size, num_chunks, seq_length)
    - Applies proper padding to match DataLoader behavior

    Args:
        df: Single-row DataFrame with feature values
        config: Configuration object
        pipelines: Pre-built text preprocessing pipelines
        risk_processors: Risk table processors for categorical features
        numerical_processors: Imputation processors for numerical features

    Returns:
        Dictionary with processed tensors ready for ONNX model
    """
    if len(df) != 1:
        raise ValueError(f"Expected single record, got {len(df)}")

    batch = {}
    record = df.iloc[0].to_dict()

    # Identify text fields from config (text_name, primary_text_name, secondary_text_name)
    text_field_names = get_text_field_names(config)

    # ===== 1. TABULAR FEATURES =====
    for feature in config.tab_field_list:
        # Skip text fields - they're processed separately through text pipelines
        if feature in text_field_names:
            continue

        val = record.get(feature)
        if feature in numerical_processors:
            val = numerical_processors[feature].process(val)
        # Convert to float explicitly before tensor creation (handles string values from CSV)
        val = _safe_float_convert(val, 0.0)
        batch[feature] = torch.tensor([[val]], dtype=torch.float32)

    # ===== 2. CATEGORICAL FEATURES =====
    for feature in config.cat_field_list:
        # Skip text fields - they're processed separately through text pipelines
        if feature in text_field_names:
            continue

        val = record.get(feature)
        if feature in risk_processors:
            val = risk_processors[feature].process(val)
        # Convert to float explicitly before tensor creation (handles string values from CSV)
        val = _safe_float_convert(val, 0.0)
        batch[feature] = torch.tensor([[val]], dtype=torch.float32)

    # ===== 3. TEXT FEATURES - TRIMODAL RANK-3 TENSOR CREATION =====
    for text_field_name, pipeline in pipelines.items():
        # Skip label field
        if text_field_name == config.label_name:
            continue

        # Get raw text
        raw_text = record.get(text_field_name, "")
        if raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text)):
            raw_text = ""
        raw_text = str(raw_text)

        try:
            # Pipeline returns List[Dict] where each dict has input_ids and attention_mask
            # Example: [{"input_ids": [101, 2023, ...], "attention_mask": [1, 1, ...]}, ...]
            tokenized_chunks = pipeline.process(raw_text)

            # Handle case where pipeline returns a single dict instead of list
            if isinstance(tokenized_chunks, dict):
                tokenized_chunks = [tokenized_chunks]

            # Extract input_ids and attention_masks from each chunk
            input_ids_list = []
            attention_mask_list = []

            for chunk_dict in tokenized_chunks:
                # Get the token IDs and attention mask for this chunk
                chunk_input_ids = chunk_dict.get(config.text_input_ids_key, [])
                chunk_attention_mask = chunk_dict.get(
                    config.text_attention_mask_key, []
                )

                # Convert to tensors
                input_ids_tensor = torch.tensor(chunk_input_ids, dtype=torch.long)
                attention_mask_tensor = torch.tensor(
                    chunk_attention_mask, dtype=torch.long
                )

                input_ids_list.append(input_ids_tensor)
                attention_mask_list.append(attention_mask_tensor)

            # Pad sequences to uniform length within chunks
            # This replicates pad_sequence behavior in collate_batch
            if input_ids_list:
                # Find max sequence length across all chunks
                max_seq_len = max(t.size(0) for t in input_ids_list)

                # Pad each chunk to max_seq_len
                padded_input_ids = []
                padded_attention_masks = []

                for ids_tensor, mask_tensor in zip(input_ids_list, attention_mask_list):
                    pad_len = max_seq_len - ids_tensor.size(0)
                    if pad_len > 0:
                        # Pad with zeros
                        ids_tensor = torch.nn.functional.pad(
                            ids_tensor, (0, pad_len), value=0
                        )
                        mask_tensor = torch.nn.functional.pad(
                            mask_tensor, (0, pad_len), value=0
                        )
                    padded_input_ids.append(ids_tensor)
                    padded_attention_masks.append(mask_tensor)

                # Stack chunks: (num_chunks, seq_length)
                stacked_input_ids = torch.stack(padded_input_ids)
                stacked_attention_masks = torch.stack(padded_attention_masks)

                # Add batch dimension: (1, num_chunks, seq_length) - rank 3!
                batch[f"{text_field_name}_{config.text_input_ids_key}"] = (
                    stacked_input_ids.unsqueeze(0)
                )
                batch[f"{text_field_name}_{config.text_attention_mask_key}"] = (
                    stacked_attention_masks.unsqueeze(0)
                )
            else:
                # No chunks - create empty rank-3 tensor
                max_len = config.max_sen_len
                batch[f"{text_field_name}_{config.text_input_ids_key}"] = torch.zeros(
                    (1, 1, max_len), dtype=torch.long
                )
                batch[f"{text_field_name}_{config.text_attention_mask_key}"] = (
                    torch.zeros((1, 1, max_len), dtype=torch.long)
                )

        except Exception as e:
            logger.error(
                f"Error processing text field '{text_field_name}': {e}", exc_info=True
            )
            # Fallback: create rank-3 tensor with single empty chunk
            max_len = config.max_sen_len
            batch[f"{text_field_name}_{config.text_input_ids_key}"] = torch.zeros(
                (1, 1, max_len), dtype=torch.long
            )
            batch[f"{text_field_name}_{config.text_attention_mask_key}"] = torch.zeros(
                (1, 1, max_len), dtype=torch.long
            )

    return batch


# =================== Model Function ======================
def model_fn(model_dir, context=None):
    model_filename = "model.pth"
    model_artifact_name = "model_artifacts.pth"
    onnx_model_path = os.path.join(model_dir, "model.onnx")

    load_config, embedding_mat, vocab, model_class = load_artifacts(
        os.path.join(model_dir, model_artifact_name), device_l=device
    )

    config = Config(**load_config)

    # ============================================================================
    # ONNX RUNTIME OPTIMIZATION CONFIGURATION
    # ============================================================================
    # Read optimization configuration from environment variables
    enable_bert_fusion = os.environ.get("ENABLE_BERT_FUSION", "false").lower() == "true"
    enable_profiling = (
        os.environ.get("ENABLE_ONNX_PROFILING", "false").lower() == "true"
    )
    inter_op_threads = int(os.environ.get("ONNX_INTER_OP_THREADS", "1"))
    intra_op_threads = int(os.environ.get("ONNX_INTRA_OP_THREADS", "4"))

    # ============================================================================
    # GPU/CUDA SUPPORT CONFIGURATION
    # ============================================================================
    import onnxruntime as ort

    # Detect available execution providers
    available_providers = ort.get_available_providers()

    # Configure execution providers (try GPU first, fallback to CPU)
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("✓ CUDA available, using GPU acceleration")

        # CUDA-specific optimization options
        provider_options = [
            {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
                "cudnn_conv_use_max_workspace": "1",
            },
            {},  # CPU options (empty)
        ]
    else:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
        logger.warning("CUDA not available, using CPU")

    logger.info(f"Active execution providers: {providers}")

    # Load model based on file type
    if os.path.exists(onnx_model_path):
        logger.info("Detected ONNX model.")

        if enable_bert_fusion:
            # ✅ Phase 1 + Phase 2: Full ONNX optimization with BERT fusion
            logger.info(
                "Loading with Phase 1 + Phase 2 optimizations (BERT fusion + SessionOptions)"
            )
            model = load_bert_optimized_model(
                model_dir=model_dir,
                enable_profiling=enable_profiling,
                inter_op_threads=inter_op_threads,
                intra_op_threads=intra_op_threads,
                providers=providers,
                provider_options=provider_options,
            )
        else:
            # Phase 1 only: SessionOptions optimization (no BERT fusion)
            logger.info("Loading with Phase 1 optimizations only (SessionOptions)")
            model = load_onnx_model(
                onnx_path=onnx_model_path,
                enable_profiling=enable_profiling,
                inter_op_threads=inter_op_threads,
                intra_op_threads=intra_op_threads,
                providers=providers,
                provider_options=provider_options,
            )
    else:
        logger.info("Detected PyTorch model.")
        model = load_model(
            os.path.join(model_dir, model_filename),
            config.model_dump(),
            embedding_mat,
            model_class,
            device_l=device,
        )
        model.eval()

    # Load feature columns if available (for alignment with XGBoost pattern)
    feature_columns = read_feature_columns(model_dir)

    # Load hyperparameters if available (for alignment with XGBoost pattern)
    hyperparameters = load_hyperparameters(model_dir)

    # Load preprocessing artifacts (NEW - similar to XGBoost handler)
    logger.info("Loading preprocessing artifacts...")
    risk_tables = load_risk_tables(model_dir)
    risk_processors = create_risk_processors(risk_tables) if risk_tables else {}

    impute_dict = load_imputation_dict(model_dir)
    numerical_processors = (
        create_numerical_processors(impute_dict) if impute_dict else {}
    )

    # ============================================================================
    # LOAD TOKENIZER FROM MODEL ARTIFACTS
    # ============================================================================
    # Priority order:
    # 1. Custom BPE tokenizer (lstm2risk/transformer2risk) from tokenizer.json
    # 2. BERT tokenizer from saved tokenizer/ directory
    # 3. Download from HuggingFace using config.tokenizer name
    # ============================================================================
    logger.info("Loading tokenizer...")

    custom_tokenizer_file = os.path.join(model_dir, "tokenizer.json")
    saved_tokenizer_dir = os.path.join(model_dir, "tokenizer")

    if os.path.exists(custom_tokenizer_file):
        # Load custom BPE tokenizer (lstm2risk/transformer2risk)
        logger.info(f"Loading custom BPE tokenizer from {custom_tokenizer_file}")
        tokenizer = Tokenizer.from_file(custom_tokenizer_file)
        config.pad_token_id = tokenizer.token_to_id("[PAD]")
        logger.info("✓ Loaded custom BPE tokenizer from model artifacts")

    elif os.path.exists(saved_tokenizer_dir):
        # Load saved BERT tokenizer
        logger.info(f"Loading saved BERT tokenizer from {saved_tokenizer_dir}")
        tokenizer = AutoTokenizer.from_pretrained(saved_tokenizer_dir, use_fast=True)
        logger.info("✓ Loaded BERT tokenizer from model artifacts")

    else:
        # Fall back to HuggingFace download
        if not config.tokenizer:
            config.tokenizer = "bert-base-multilingual-cased"
        logger.info(f"Loading tokenizer from HuggingFace: {config.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, use_fast=True)
        logger.info("✓ Loaded tokenizer from HuggingFace")

    # Reconstruct pipelines with loaded tokenizer
    tokenizer, pipelines = data_preprocess_pipeline(config, tokenizer, hyperparameters)

    # Add multiclass label processor if needed
    if not config.is_binary and config.num_classes > 2:
        # Check if we have label mappings from the config
        if config.label_to_id and config.id_to_label:
            # Use the saved mappings
            label_processor = MultiClassLabelProcessor(
                label_list=config.id_to_label, strict=True
            )
            pipelines[config.label_name] = label_processor
        elif config.multiclass_categories:
            # Fallback to multiclass_categories if available
            label_processor = MultiClassLabelProcessor(
                label_list=config.multiclass_categories, strict=True
            )
            pipelines[config.label_name] = label_processor

    # Load calibration model if available
    calibrator = load_calibration_model(model_dir)
    if calibrator:
        logger.info("Calibration model loaded successfully")

    return {
        "model": model,
        "config": config,
        "embedding_mat": embedding_mat,
        "vocab": vocab,
        "model_class": model_class,
        "pipelines": pipelines,
        "risk_processors": risk_processors,  # NEW
        "numerical_processors": numerical_processors,  # NEW
        "calibrator": calibrator,
        "feature_columns": feature_columns,
        "hyperparameters": hyperparameters,
    }


# =================== Input Function ================================
def input_fn(request_body, request_content_type, context=None):
    """
    Deserialize the Invoke request body into an object we can perform prediction on.
    """
    logger.info(
        f"Received request with Content-Type: {request_content_type}"
    )  # Log content type
    try:
        if request_content_type == "text/csv":
            logger.info("Processing content type: text/csv")
            decoded = (
                request_body.decode("utf-8")
                if isinstance(request_body, bytes)
                else request_body
            )
            logger.debug(
                f"Decoded CSV data:\n{decoded[:500]}..."
            )  # Optional: Log decoded data (be careful with large data)
            try:
                df = pd.read_csv(StringIO(decoded), header=None, index_col=None)
                logger.info(
                    f"Successfully parsed CSV into DataFrame. Shape: {df.shape}, Type: {type(df)}"
                )
                return df  # <--- Returns DataFrame here
            except Exception as parse_error:
                logger.error(f"Failed to parse CSV data: {parse_error}")
                # If parsing fails, it will fall through to the final except block
                raise  # Re-raise the parsing error to be caught below

        elif request_content_type == "application/json":
            logger.info("Processing content type: application/json")
            # ... your JSON handling ...
            # Ensure this branch also returns a DataFrame if called
            decoded = (
                request_body.decode("utf-8")
                if isinstance(request_body, bytes)
                else request_body
            )
            try:
                if "\n" in decoded:
                    # Multi-record JSON (NDJSON) handling
                    records = [
                        json.loads(line)
                        for line in decoded.strip().splitlines()
                        if line.strip()
                    ]
                    df = pd.DataFrame(records)
                else:
                    json_obj = json.loads(decoded)
                    if isinstance(json_obj, dict):
                        df = pd.DataFrame([json_obj])
                    elif isinstance(json_obj, list):
                        df = pd.DataFrame(json_obj)
                    else:
                        raise ValueError("Unsupported JSON structure")
                logger.info(
                    f"Successfully parsed JSON into DataFrame. Shape: {df.shape}"
                )
                return df
            except Exception as parse_error:
                logger.error(f"Failed to parse JSON data: {parse_error}")
                raise

        elif request_content_type == "application/x-parquet":
            logger.info("Processing content type: application/x-parquet")
            # ... your Parquet handling ...
            # Ensure this branch also returns a DataFrame if called
            df = pd.read_parquet(BytesIO(request_body))
            logger.info(
                f"Successfully parsed Parquet into DataFrame. Shape: {df.shape}, Type: {type(df)}"
            )
            return df  # <--- Returns DataFrame here

        else:
            logger.warning(f"Unsupported content type: {request_content_type}")
            raise ValueError(
                f"This predictor only supports CSV, JSON, or Parquet data. Received: {request_content_type}"
            )
    except Exception as e:
        logger.error(
            f"Failed to parse input ({request_content_type}). Error: {e}", exc_info=True
        )
        raise ValueError(
            f"Invalid input format or corrupted data. Error during parsing: {e}"
        ) from e


# ================== Prediction Function ============================
def predict_fn(input_object, model_data, context=None):
    if not isinstance(input_object, pd.DataFrame):
        raise TypeError("input data type must be pandas.DataFrame")

    model = model_data["model"]
    config = model_data["config"]
    pipelines = model_data["pipelines"]
    # Get preprocessing processors
    risk_processors = model_data.get("risk_processors", {})
    numerical_processors = model_data.get("numerical_processors", {})
    calibrator = model_data.get("calibrator")
    feature_columns = model_data.get("feature_columns")

    config_predict = config.model_dump()
    label_field = config_predict.get("label_name", None)

    if label_field:
        config_predict["full_field_list"] = [
            col for col in config_predict["full_field_list"] if col != label_field
        ]
        config_predict["cat_field_list"] = [
            col for col in config_predict["cat_field_list"] if col != label_field
        ]

    # Validate input data if feature columns are available
    if feature_columns:
        validate_input_data(input_object, feature_columns)

        # Assign column names if needed for headerless CSV input
        input_object = assign_column_names(input_object, feature_columns)

        # NOTE: Column reindexing removed for performance (was causing 600-700ms overhead)
        # PipelineDataset uses name-based column access, so column order doesn't matter

    # ============================================================================
    # NAMES3RISK PREPROCESSING: Create 'text' field from 4 name fields
    # ============================================================================
    # This replicates the preprocessing logic from tabular_preprocessing.py
    # that concatenates emailAddress, billingAddressName, customerName, and
    # paymentAccountHolderName into a single 'text' field for model input.
    #
    # During real-time inference, requests contain the raw 4 fields, but the
    # model expects a pre-concatenated 'text' field (as created during training).
    with log_timing("Names3Risk Text Field Creation"):
        input_object = create_text_field_for_names3risk(input_object)

    # FAST PATH: Single-record inference optimization
    if len(input_object) == 1:
        logger.info("Using fast path for single-record inference (bypasses DataLoader)")

        try:
            # ✨ NEW: Direct preprocessing without DataLoader overhead!
            # Processes text, tabular, and categorical features in one pass
            with log_timing("Fast Path - Preprocessing"):
                batch_dict = preprocess_single_record_with_text(
                    df=input_object,
                    config=config,
                    pipelines=pipelines,
                    risk_processors=risk_processors,
                    numerical_processors=numerical_processors,
                )

            # Run model inference directly with preprocessed batch
            logger.info("Model prediction...")
            with log_timing("Fast Path - Model Inference"):
                raw_probs = model_online_inference(model, [batch_dict])

            # Apply calibration if available
            if calibrator:
                try:
                    is_multiclass = not config.is_binary
                    with log_timing("Fast Path - Calibration"):
                        calibrated_probs = apply_calibration(
                            raw_probs, calibrator, is_multiclass
                        )
                    logger.info("Applied calibration to predictions")
                except Exception as e:
                    logger.warning(f"Failed to apply calibration: {e}")
                    calibrated_probs = raw_probs.copy()
            else:
                logger.info("No calibration model available, using raw predictions")
                calibrated_probs = raw_probs.copy()

            return {
                "raw_predictions": raw_probs,
                "calibrated_predictions": calibrated_probs,
            }

        except Exception as e:
            logger.warning(f"Fast path failed, falling back to DataLoader path: {e}")
            # Fall through to batch path below

    # BATCH PATH: Original DataFrame processing for multiple records (or fallback)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Using batch path for {len(input_object)} records")

    dataset = PipelineDataset(config_predict, dataframe=input_object)

    # Add text preprocessing pipelines
    for feature_name, pipeline in pipelines.items():
        dataset.add_pipeline(feature_name, pipeline)

    # Add numerical imputation processors
    for feature_name, processor in numerical_processors.items():
        if feature_name in dataset.DataReader.columns:
            dataset.add_pipeline(feature_name, processor)

    # Add risk table processors (excluding text fields)
    text_fields = get_text_field_names(config)
    for feature_name, processor in risk_processors.items():
        if (
            feature_name not in text_fields
            and feature_name in dataset.DataReader.columns
        ):
            dataset.add_pipeline(feature_name, processor)

    # Use unified collate function for all models (matches pytorch_training.py and pytorch_model_eval.py)
    model_class = config.model_class
    logger.info(f"Creating DataLoader for model: {model_class}")

    collate_batch = build_collate_batch(
        input_ids_key=config.text_input_ids_key,
        attention_mask_key=config.text_attention_mask_key,
    )
    logger.info(f"✓ Using unified collate function for {model_class}")
    logger.info(f"  - Input IDs key: {config.text_input_ids_key}")
    logger.info(f"  - Attention mask key: {config.text_attention_mask_key}")
    logger.info("  - Handles text fields automatically via pipeline_dataloader")

    batch_size = len(input_object)
    predict_dataloader = DataLoader(
        dataset, collate_fn=collate_batch, batch_size=batch_size
    )

    try:
        logger.info("Model prediction...")
        raw_probs = model_online_inference(model, predict_dataloader)

        # Apply calibration if available
        if calibrator:
            try:
                is_multiclass = not config.is_binary
                calibrated_probs = apply_calibration(
                    raw_probs, calibrator, is_multiclass
                )
                logger.info("Applied calibration to predictions")
            except Exception as e:
                logger.warning(f"Failed to apply calibration: {e}")
                calibrated_probs = raw_probs.copy()
        else:
            logger.info("No calibration model available, using raw predictions")
            calibrated_probs = raw_probs.copy()

        return {
            "raw_predictions": raw_probs,
            "calibrated_predictions": calibrated_probs,
        }
    except Exception:
        logger.error("Model scoring error:\n" + traceback.format_exc())
        return [-4]


# ================== Output Function ================================
def output_fn(prediction_output, accept="application/json"):
    """
    Serializes the multi-class prediction output with both raw and calibrated scores.

    Args:
        prediction_output: Dict with "raw_predictions" and "calibrated_predictions"
                          from predict_fn, or legacy numpy array/list format
        accept: The requested response MIME type (e.g., 'application/json').

    Returns:
        tuple: (response_body, content_type)
    """
    logger.info(
        f"Received prediction output of type: {type(prediction_output)} for accept type: {accept}"
    )

    # Step 1: Extract raw and calibrated predictions
    if isinstance(prediction_output, dict):
        raw_predictions = prediction_output.get("raw_predictions")
        calibrated_predictions = prediction_output.get("calibrated_predictions")
    else:
        # Backward compatibility: treat as raw predictions
        raw_predictions = prediction_output
        calibrated_predictions = prediction_output

    # Step 2: Convert to list format
    raw_scores_list = (
        raw_predictions.tolist()
        if isinstance(raw_predictions, np.ndarray)
        else raw_predictions
    )
    calibrated_scores_list = (
        calibrated_predictions.tolist()
        if isinstance(calibrated_predictions, np.ndarray)
        else calibrated_predictions
    )

    # Ensure list of lists format
    if not isinstance(raw_scores_list[0], list):
        raw_scores_list = [[score] for score in raw_scores_list]
        calibrated_scores_list = [[score] for score in calibrated_scores_list]

    try:
        is_multiclass = len(raw_scores_list[0]) > 2

        # Step 3: JSON output formatting
        if accept.lower() == "application/json":
            output_records = []
            for raw_probs, cal_probs in zip(raw_scores_list, calibrated_scores_list):
                max_idx = raw_probs.index(max(raw_probs)) if raw_probs else -1

                if not is_multiclass:
                    # Binary classification
                    record = {
                        "legacy-score": str(
                            raw_probs[1] if len(raw_probs) > 1 else raw_probs[0]
                        ),  # Raw class-1
                        "calibrated-score": str(
                            cal_probs[1] if len(cal_probs) > 1 else cal_probs[0]
                        ),  # Calibrated class-1
                        "score-percentile": str(
                            cal_probs[1] if len(cal_probs) > 1 else cal_probs[0]
                        ),  # Duplicate of calibrated-score (for compatibility)
                        # "custom-output-label": f"class-{max_idx}"
                        # if max_idx >= 0
                        # else "unknown",
                    }
                else:
                    # Multiclass
                    record = {}
                    for i in range(len(raw_probs)):
                        record[f"prob_{str(i + 1).zfill(2)}"] = str(raw_probs[i])
                        record[f"calibrated_prob_{str(i + 1).zfill(2)}"] = str(
                            cal_probs[i]
                        )
                    # record["custom-output-label"] = (
                    #     f"class-{max_idx}" if max_idx >= 0 else "unknown"
                    # )

                output_records.append(record)

            response = json.dumps({"predictions": output_records})
            return response, "application/json"

        # Step 4: CSV output formatting
        elif accept.lower() == "text/csv":
            csv_lines = []
            for raw_probs, cal_probs in zip(raw_scores_list, calibrated_scores_list):
                max_idx = raw_probs.index(max(raw_probs)) if raw_probs else -1

                if not is_multiclass:
                    # Binary classification: legacy-score, calibrated-score, score-percentile
                    raw_score = round(
                        float(raw_probs[1] if len(raw_probs) > 1 else raw_probs[0]), 4
                    )
                    cal_score = round(
                        float(cal_probs[1] if len(cal_probs) > 1 else cal_probs[0]), 4
                    )

                    line = [
                        f"{raw_score:.4f}",
                        f"{cal_score:.4f}",
                        f"{cal_score:.4f}",  # score-percentile (duplicate of calibrated-score)
                        # f"class-{max_idx}" if max_idx >= 0 else "unknown",
                    ]
                else:
                    # Multiclass: interleaved raw and calibrated probs
                    # Format raw probabilities
                    raw_formatted = [round(float(p), 4) for p in raw_probs]
                    raw_str = ",".join(f"{p:.4f}" for p in raw_formatted)

                    # Format calibrated probabilities
                    cal_formatted = [round(float(p), 4) for p in cal_probs]
                    cal_str = ",".join(f"{p:.4f}" for p in cal_formatted)

                    line = [
                        raw_str,
                        cal_str,
                        # f"class-{max_idx}" if max_idx >= 0 else "unknown",
                    ]

                csv_lines.append(",".join(map(str, line)))

            response_body = "\n".join(csv_lines) + "\n"
            return response_body, "text/csv"

        # Step 5: Unsupported content type
        else:
            logger.error(f"Unsupported accept type: {accept}")
            raise ValueError(f"Unsupported accept type: {accept}")

    # Step 6: Error handling
    except Exception as e:
        logger.error(
            f"Error during serialization in output_fn: {e}",
            exc_info=True,
        )
        error_response = json.dumps({"error": f"Failed to serialize output: {e}"})
        return error_response, "application/json"
