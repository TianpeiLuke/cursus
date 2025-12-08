#!/usr/bin/env python3
import os
import json
import sys
import traceback
import ast
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from typing import List, Tuple, Pattern, Union, Dict, Set, Optional
from collections.abc import Callable, Mapping

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
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import lightning.pytorch as pl
from lightning.pytorch.strategies import FSDPStrategy

from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings("ignore")

from ...processing.processors import (
    Processor,
)
from ...processing.text.dialogue_processor import (
    HTMLNormalizerProcessor,
    EmojiRemoverProcessor,
    TextNormalizationProcessor,
    DialogueSplitterProcessor,
    DialogueChunkerProcessor,
)
from ...processing.text.bert_tokenize_processor import BertTokenizeProcessor
from ...processing.categorical.categorical_label_processor import (
    CategoricalLabelProcessor,
)
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
from lightning_models.tabular.pl_tab_ae import TabAE
from lightning_models.text.pl_text_cnn import TextCNN
from lightning_models.bimodal.pl_bimodal_cnn import BimodalCNN
from lightning_models.bimodal.pl_bimodal_bert import BimodalBert
from lightning_models.bimodal.pl_bimodal_moe import BimodalBertMoE
from lightning_models.bimodal.pl_bimodal_gate_fusion import BimodalBertGateFusion
from lightning_models.bimodal.pl_bimodal_cross_attn import BimodalBertCrossAttn
from lightning_models.trimodal.pl_trimodal_bert import TrimodalBert
from lightning_models.trimodal.pl_trimodal_cross_attn import TrimodalCrossAttentionBert
from lightning_models.trimodal.pl_trimodal_gate_fusion import TrimodalGateFusionBert
from lightning_models.text.pl_bert_classification import TextBertClassification
from lightning_models.text.pl_lstm import TextLSTM
from lightning_models.utils.pl_train import (
    model_train,
    model_inference,
    predict_stack_transform,
    save_model,
    save_prediction,
    save_artifacts,
    load_model,
    load_artifacts,
    load_checkpoint,
)
from lightning_models.utils.pl_model_plots import (
    compute_metrics,
    roc_metric_plot,
    pr_metric_plot,
)
from lightning_models.utils.dist_utils import get_rank, is_main_process
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
)  # For Config Validation


# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # <-- THIS LINE IS MISSING

if is_main_process():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def log_once(logger, message, level=logging.INFO):
    if is_main_process():
        logger.log(level, message)


# -------------------------------------------------------------------------
# FILE I/O HELPER FUNCTIONS WITH FORMAT PRESERVATION
# -------------------------------------------------------------------------
def _detect_file_format(file_path: str) -> str:
    """
    Detect the format of a data file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        Format string: 'csv', 'tsv', or 'parquet'
    """
    from pathlib import Path

    suffix = Path(file_path).suffix.lower()

    if suffix == ".csv":
        return "csv"
    elif suffix == ".tsv":
        return "tsv"
    elif suffix == ".parquet":
        return "parquet"
    else:
        raise RuntimeError(f"Unsupported file format: {suffix}")


def load_dataframe_with_format(file_path: str) -> Tuple[pd.DataFrame, str]:
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
    df: pd.DataFrame, output_path: str, format_str: str
) -> str:
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

    return str(file_path)


# ================================================================================
class Config(BaseModel):
    id_name: str = "order_id"
    text_name: Optional[str] = (
        None  # Optional for trimodal (uses primary/secondary instead)
    )
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
    use_gradient_checkpointing: bool = False
    run_scheduler: bool = True
    reinit_pooler: bool = True
    reinit_layers: int = 2
    warmup_steps: int = 300
    text_input_ids_key: str = "input_ids"  # Configurable text input key
    text_attention_mask_key: str = "attention_mask"  # Configurable attention mask key
    primary_text_name: Optional[str] = (
        None  # Primary text field for trimodal (e.g., "chat")
    )
    secondary_text_name: Optional[str] = (
        None  # Secondary text field for trimodal (e.g., "shiptrack")
    )
    embed_size: Optional[int] = None  # Added for type consistency
    label_to_id: Optional[Dict[str, int]] = None  # Added: label to ID mapping
    id_to_label: Optional[List[str]] = None  # Added: ID to label mapping
    _input_format: Optional[str] = None  # Added: input data format for preservation
    smooth_factor: float = 0.0  # Risk table smoothing factor
    count_threshold: int = 0  # Risk table count threshold
    imputation_dict: Optional[Dict[str, float]] = None  # Imputation values
    risk_tables: Optional[Dict[str, Dict]] = None  # Risk table mappings
    # Text processing pipeline steps (optional, uses defaults if not provided)
    text_processing_steps: Optional[List[str]] = (
        None  # Processing steps for bimodal text
    )
    primary_text_processing_steps: Optional[List[str]] = (
        None  # Processing steps for primary text (trimodal)
    )
    secondary_text_processing_steps: Optional[List[str]] = (
        None  # Processing steps for secondary text (trimodal)
    )

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


# ------------------- Improved Hyperparameter Parser ----------------------
def safe_cast(val):
    if isinstance(val, str):
        val = val.strip()
        if val.lower() == "true":
            return True
        elif val.lower() == "false":
            return False
        if (val.startswith("[") and val.endswith("]")) or (
            val.startswith("{") and val.endswith("}")
        ):
            try:
                return json.loads(val)
            except Exception:
                pass
        try:
            return ast.literal_eval(val)
        except Exception:
            pass
    return val


def sanitize_config(config):
    for key, val in config.items():
        if isinstance(val, str) and val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        config[key] = safe_cast(val)
    return config


def load_parse_hyperparameters(hparam_path: str) -> Dict:
    converters = {
        "id_name": safe_cast,
        "text_name": safe_cast,
        "label_name": safe_cast,
        "tab_field_list": safe_cast,
        "full_field_list": safe_cast,
        "cat_field_list": safe_cast,
        "categorical_features_to_encode": safe_cast,
        "batch_size": safe_cast,
        "max_sen_len": safe_cast,
        "chunk_trancate": safe_cast,
        "max_total_chunks": safe_cast,
        "tokenizer": safe_cast,
        "hidden_common_dim": safe_cast,
        "input_tab_dim": safe_cast,
        "max_epochs": safe_cast,
        "num_classes": safe_cast,
        "is_binary": safe_cast,
        "multiclass_categories": safe_cast,
        "categorical_label_features": safe_cast,
        "kernel_size": safe_cast,
        "lr": safe_cast,
        "lr_decay": safe_cast,
        "momentum": safe_cast,
        "class_weights": safe_cast,
        "fixed_tokenizer_length": safe_cast,
        "is_embeddings_trainable": safe_cast,
        "optimizer": safe_cast,
        "num_layers": safe_cast,
        "num_channels": safe_cast,
        "weight_decay": safe_cast,
        "num_workers": safe_cast,
        "metric_choices": safe_cast,
        "early_stop_metric": safe_cast,
        "early_stop_patience": safe_cast,
        "model_class": safe_cast,
        "load_ckpt": safe_cast,
        "val_check_interval": safe_cast,
        "fp16": safe_cast,
        "gradient_clip_val": safe_cast,
        "run_scheduler": safe_cast,
        "reinit_pooler": safe_cast,
        "reinit_layers": safe_cast,
        "warmup_steps": safe_cast,
        "adam_epsilon": safe_cast,
        "text_input_ids_key": safe_cast,  # Added
        "text_attention_mask_key": safe_cast,  # Added
    }
    hyperparameters = {}
    with open(hparam_path, "r") as f:
        args = json.load(f)
        log_once(logger, "Hyperparameters for training job:")
        for key, value in args.items():
            if key in converters:
                try:
                    converted = converters[key](value)
                except Exception as e:
                    logger.warning(
                        f"Conversion error for key {key} with value {value}: {e}"
                    )
                    converted = value
                hyperparameters[key] = converted
                print(f"{key}: {converted} ({type(converted)})")
            else:
                hyperparameters[key] = value
                print(f"{key}: {value} ({type(value)})")
    return hyperparameters


# ----------------- Detect training, testing and validation file names --------
def find_first_data_file(
    data_dir: str, extensions: List[str] = [".tsv", ".csv", ".parquet"]
) -> Optional[str]:
    for fname in sorted(os.listdir(data_dir)):
        cleaned_fname = fname.strip().lower()
        if any(cleaned_fname.endswith(ext) for ext in extensions):
            return fname
    raise FileNotFoundError(
        f"No supported data file (.tsv, .csv, .parquet) found in {data_dir}"
    )


# ----------------- Artifact Loading/Saving Helpers -------------------------
def load_imputation_artifacts(artifacts_dir: str) -> Dict[str, float]:
    """Load pre-computed imputation dictionary from artifacts directory."""
    import pickle as pkl

    impute_file = os.path.join(artifacts_dir, "impute_dict.pkl")
    if not os.path.exists(impute_file):
        raise FileNotFoundError(f"Imputation artifacts not found: {impute_file}")

    with open(impute_file, "rb") as f:
        impute_dict = pkl.load(f)

    log_once(logger, f"Loaded imputation dict with {len(impute_dict)} fields")
    return impute_dict


def load_risk_table_artifacts(artifacts_dir: str) -> Dict[str, Dict]:
    """Load pre-computed risk tables from artifacts directory."""
    import pickle as pkl

    risk_file = os.path.join(artifacts_dir, "risk_table_map.pkl")
    if not os.path.exists(risk_file):
        raise FileNotFoundError(f"Risk table artifacts not found: {risk_file}")

    with open(risk_file, "rb") as f:
        risk_tables = pkl.load(f)

    log_once(logger, f"Loaded risk tables for {len(risk_tables)} fields")
    return risk_tables


def save_imputation_artifacts(
    imputation_dict: Dict[str, float], output_dir: str
) -> None:
    """Save imputation dictionary to model directory."""
    import pickle as pkl

    os.makedirs(output_dir, exist_ok=True)

    # Save pickle format
    impute_file = os.path.join(output_dir, "impute_dict.pkl")
    with open(impute_file, "wb") as f:
        pkl.dump(imputation_dict, f)

    # Save JSON format for readability
    impute_json = os.path.join(output_dir, "impute_dict.json")
    with open(impute_json, "w") as f:
        json.dump(imputation_dict, f, indent=2)

    log_once(logger, f"Saved imputation artifacts to {output_dir}")


def save_risk_table_artifacts(risk_tables: Dict[str, Dict], output_dir: str) -> None:
    """Save risk tables to model directory."""
    import pickle as pkl

    os.makedirs(output_dir, exist_ok=True)

    # Save pickle format
    risk_file = os.path.join(output_dir, "risk_table_map.pkl")
    with open(risk_file, "wb") as f:
        pkl.dump(risk_tables, f)

    # Save JSON format for readability
    risk_json = os.path.join(output_dir, "risk_table_map.json")
    json_tables = {}
    for field, tables in risk_tables.items():
        json_tables[field] = {
            "bins": {str(k): float(v) for k, v in tables.get("bins", {}).items()},
            "default_bin": float(tables.get("default_bin", 0.0)),
        }
    with open(risk_json, "w") as f:
        json.dump(json_tables, f, indent=2)

    log_once(logger, f"Saved risk table artifacts to {output_dir}")


# ----------------- Preprocessing Pipeline Builder (Imputation + Risk Tables) ------------------
def build_preprocessing_pipelines(
    config: Config,
    datasets: List[PipelineDataset],
    model_artifacts_dir: Optional[str] = None,
    use_precomputed_imputation: bool = False,
    use_precomputed_risk_tables: bool = False,
) -> Tuple[Dict[str, Processor], Dict[str, float], Dict[str, Dict]]:
    """
    Build preprocessing pipelines for numerical imputation and risk table mapping.

    Args:
        config: Configuration object
        datasets: List of [train, val, test] datasets
        model_artifacts_dir: Optional directory with pre-computed artifacts
        use_precomputed_imputation: Whether to use pre-computed imputation
        use_precomputed_risk_tables: Whether to use pre-computed risk tables

    Returns:
        Tuple of (pipelines_dict, imputation_dict, risk_tables_dict)
    """
    pipelines = {}
    imputation_dict = {}
    risk_tables = {}

    train_dataset = datasets[0]

    log_once(logger, "=" * 70)
    log_once(logger, "PREPROCESSING PIPELINE BUILDER")
    log_once(logger, "=" * 70)
    log_once(logger, f"USE_PRECOMPUTED_IMPUTATION: {use_precomputed_imputation}")
    log_once(logger, f"USE_PRECOMPUTED_RISK_TABLES: {use_precomputed_risk_tables}")
    log_once(logger, f"Model artifacts directory: {model_artifacts_dir}")
    log_once(logger, "=" * 70)

    # === FIELD TYPE VALIDATION ===
    if not use_precomputed_imputation and config.tab_field_list:
        log_once(logger, "Validating numerical field types before imputation...")
        validate_numerical_fields(
            train_dataset.DataReader, config.tab_field_list, "train"
        )
        log_once(logger, "✓ Numerical field type validation passed")

    if not use_precomputed_risk_tables and config.cat_field_list:
        log_once(
            logger, "Validating categorical field types before risk table mapping..."
        )
        validate_categorical_fields(
            train_dataset.DataReader, config.cat_field_list, "train"
        )
        log_once(logger, "✓ Categorical field type validation passed")

    # === 1. NUMERICAL IMPUTATION ===
    if config.tab_field_list:
        if use_precomputed_imputation and model_artifacts_dir:
            # Load pre-computed imputation
            log_once(logger, "Loading pre-computed imputation artifacts...")
            imputation_dict = load_imputation_artifacts(model_artifacts_dir)
            for field, value in imputation_dict.items():
                proc = NumericalVariableImputationProcessor(
                    column_name=field, imputation_value=value
                )
                pipelines[field] = proc
            log_once(logger, f"✓ Loaded imputation for {len(imputation_dict)} fields")
        else:
            # Fit inline
            log_once(logger, "Fitting numerical imputation inline...")
            for field in config.tab_field_list:
                proc = NumericalVariableImputationProcessor(
                    column_name=field, strategy="mean"
                )
                proc.fit(train_dataset.DataReader[field])
                pipelines[field] = proc
                imputation_dict[field] = proc.get_imputation_value()
            log_once(
                logger, f"✓ Fitted imputation for {len(config.tab_field_list)} fields"
            )

    # === 2. RISK TABLE MAPPING (replaces categorical label encoding) ===
    if config.cat_field_list:
        # Filter out text fields from categorical processing to prevent overwriting tokenized text
        text_fields = set()
        if config.text_name:
            text_fields.add(config.text_name)
        if config.primary_text_name:
            text_fields.add(config.primary_text_name)
        if config.secondary_text_name:
            text_fields.add(config.secondary_text_name)

        # Filter categorical fields to exclude text fields
        actual_cat_fields = [f for f in config.cat_field_list if f not in text_fields]

        if text_fields:
            log_once(
                logger,
                f"ℹ️  Excluding text fields from risk table mapping: {text_fields}",
            )
        log_once(
            logger, f"ℹ️  Actual categorical fields for risk table: {actual_cat_fields}"
        )

        if use_precomputed_risk_tables and model_artifacts_dir:
            # Load pre-computed risk tables
            log_once(logger, "Loading pre-computed risk table artifacts...")
            risk_tables = load_risk_table_artifacts(model_artifacts_dir)
            for field, tables in risk_tables.items():
                if field in actual_cat_fields:  # Only process filtered fields
                    proc = RiskTableMappingProcessor(
                        column_name=field,
                        label_name=config.label_name,
                        risk_tables=tables,
                    )
                    pipelines[field] = proc
            log_once(logger, f"✓ Loaded risk tables for {len(pipelines)} fields")
        else:
            # Fit inline
            log_once(logger, "Fitting risk tables inline...")
            for field in actual_cat_fields:  # Use filtered list
                proc = RiskTableMappingProcessor(
                    column_name=field,
                    label_name=config.label_name,
                    smooth_factor=config.smooth_factor,
                    count_threshold=config.count_threshold,
                )
                proc.fit(train_dataset.DataReader)
                pipelines[field] = proc
                risk_tables[field] = proc.get_risk_tables()
            log_once(
                logger, f"✓ Fitted risk tables for {len(actual_cat_fields)} fields"
            )

    log_once(logger, "=" * 70)
    log_once(logger, f"Total preprocessing pipelines created: {len(pipelines)}")
    log_once(logger, "=" * 70)

    return pipelines, imputation_dict, risk_tables


# ----------------- Dataset Loading -------------------------
def load_data_module(file_dir, filename, config: Config) -> PipelineDataset:
    log_once(logger, f"Loading pipeline dataset from {filename} in folder {file_dir}")
    pipeline_dataset = PipelineDataset(
        config=config.model_dump(), file_dir=file_dir, filename=filename
    )  # Pass as dict
    log_once(logger, f"Filling missing values in dataset {filename}")
    pipeline_dataset.fill_missing_value(
        label_name=config.label_name, column_cat_name=config.cat_field_list
    )
    return pipeline_dataset


# ----------------- Updated Data Preprocessing Pipeline ------------------
def data_preprocess_pipeline(
    config: Config,
) -> Tuple[AutoTokenizer, Dict[str, Processor]]:
    """
    Build text preprocessing pipelines based on config.

    For bimodal: Uses text_name with default or configured steps
    For trimodal: Uses primary_text_name and secondary_text_name with separate step lists
    """
    if not config.tokenizer:
        config.tokenizer = "bert-base-multilingual-cased"

    log_once(logger, f"Constructing tokenizer: {config.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    pipelines = {}

    # BIMODAL: Single text pipeline
    if not config.primary_text_name:
        # Use configured steps or fallback to default
        steps = getattr(
            config,
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

        pipelines[config.text_name] = build_text_pipeline_from_steps(
            processing_steps=steps,
            tokenizer=tokenizer,
            max_sen_len=config.max_sen_len,
            chunk_trancate=config.chunk_trancate,
            max_total_chunks=config.max_total_chunks,
            input_ids_key=config.text_input_ids_key,
            attention_mask_key=config.text_attention_mask_key,
        )
        log_once(
            logger,
            f"Built bimodal pipeline for '{config.text_name}' with steps: {steps}",
        )

    # TRIMODAL: Dual text pipelines
    else:
        # Primary text pipeline (e.g., chat - full cleaning)
        primary_steps = getattr(
            config,
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

        pipelines[config.primary_text_name] = build_text_pipeline_from_steps(
            processing_steps=primary_steps,
            tokenizer=tokenizer,
            max_sen_len=config.max_sen_len,
            chunk_trancate=config.chunk_trancate,
            max_total_chunks=config.max_total_chunks,
            input_ids_key=config.text_input_ids_key,
            attention_mask_key=config.text_attention_mask_key,
        )
        log_once(
            logger,
            f"Built primary pipeline for '{config.primary_text_name}' with steps: {primary_steps}",
        )

        # Secondary text pipeline (e.g., events - minimal cleaning)
        secondary_steps = getattr(
            config,
            "secondary_text_processing_steps",
            [
                "dialogue_splitter",
                "text_normalizer",
                "dialogue_chunker",
                "tokenizer",
            ],
        )

        pipelines[config.secondary_text_name] = build_text_pipeline_from_steps(
            processing_steps=secondary_steps,
            tokenizer=tokenizer,
            max_sen_len=config.max_sen_len,
            chunk_trancate=config.chunk_trancate,
            max_total_chunks=config.max_total_chunks,
            input_ids_key=config.text_input_ids_key,
            attention_mask_key=config.text_attention_mask_key,
        )
        log_once(
            logger,
            f"Built secondary pipeline for '{config.secondary_text_name}' with steps: {secondary_steps}",
        )

    return tokenizer, pipelines


# ----------------- Model Selection -----------------------
def model_select(
    model_class: str, config: Config, vocab_size: int, embedding_mat: torch.Tensor
) -> nn.Module:
    """
    Select and instantiate a model based on model_class string.

    Supports:
    - General categories: "bimodal", "trimodal"
    - Specific bimodal models: "bimodal_bert", "bimodal_cnn", etc.
    - Specific trimodal models: "trimodal_bert", etc.
    - Text-only models: "bert", "lstm"
    - Backward compatibility: "multimodal_*" maps to "bimodal_*"
    """
    model_map = {
        # General categories (default to bert variants)
        "bimodal": lambda: BimodalBert(config.model_dump()),
        "trimodal": lambda: TrimodalBert(config.model_dump()),
        # Specific bimodal models
        "bimodal_cnn": lambda: BimodalCNN(
            config.model_dump(), vocab_size, embedding_mat
        ),
        "bimodal_bert": lambda: BimodalBert(config.model_dump()),
        "bimodal_moe": lambda: BimodalBertMoE(config.model_dump()),
        "bimodal_gate_fusion": lambda: BimodalBertGateFusion(config.model_dump()),
        "bimodal_cross_attn": lambda: BimodalBertCrossAttn(config.model_dump()),
        # Specific trimodal models
        "trimodal_bert": lambda: TrimodalBert(config.model_dump()),
        "trimodal_cross_attn": lambda: TrimodalCrossAttentionBert(config.model_dump()),
        "trimodal_gate_fusion": lambda: TrimodalGateFusionBert(config.model_dump()),
        # Text-only models
        "bert": lambda: TextBertClassification(config.model_dump()),
        "lstm": lambda: TextLSTM(config.model_dump(), vocab_size, embedding_mat),
        # Backward compatibility (multimodal -> bimodal)
        "multimodal_cnn": lambda: BimodalCNN(
            config.model_dump(), vocab_size, embedding_mat
        ),
        "multimodal_bert": lambda: BimodalBert(config.model_dump()),
        "multimodal_moe": lambda: BimodalBertMoE(config.model_dump()),
        "multimodal_gate_fusion": lambda: BimodalBertGateFusion(config.model_dump()),
        "multimodal_cross_attn": lambda: BimodalBertCrossAttn(config.model_dump()),
    }

    return model_map.get(
        model_class, lambda: TextBertClassification(config.model_dump())
    )()


# ----------------- Training Setup -----------------------
def setup_training_environment(config: Config) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    return device


# ----------------- Data Loading and Preprocessing ------------------
def load_and_preprocess_data(
    config: Config,
    paths: Dict[str, str],
    model_artifacts_dir: Optional[str] = None,
    use_precomputed_imputation: bool = False,
    use_precomputed_risk_tables: bool = False,
) -> Tuple[List[PipelineDataset], AutoTokenizer, Dict]:
    """
    Loads and preprocesses the train/val/test datasets according to the provided config.

    Args:
        config: Configuration object
        paths: Dictionary of paths (train, val, test, checkpoint)
        model_artifacts_dir: Optional directory with pre-computed artifacts
        use_precomputed_imputation: Whether to use pre-computed imputation
        use_precomputed_risk_tables: Whether to use pre-computed risk tables

    Returns:
        Tuple of ([train_dataset, val_dataset, test_dataset], tokenizer, config)
    """
    train_filename = find_first_data_file(paths["train"])
    val_filename = find_first_data_file(paths["val"])
    test_filename = find_first_data_file(paths["test"])
    log_once(logger, "================================================")
    log_once(logger, f"Train folder: {paths['train']} | File: {train_filename}")
    log_once(logger, f"Validation folder: {paths['val']} | File: {val_filename}")
    log_once(logger, f"Test folder: {paths['test']} | File: {test_filename}")
    log_once(logger, "================================================")
    if not os.path.exists(paths["checkpoint"]):
        print(f"Creating checkpoint folder {paths['checkpoint']}")
        os.makedirs(paths["checkpoint"])

    # Detect input format from training data file
    train_file_path = os.path.join(paths["train"], train_filename)
    detected_format = _detect_file_format(train_file_path)
    log_once(logger, f"Detected input data format: {detected_format}")

    # Store format in config for output preservation
    config._input_format = detected_format

    # === Load raw datasets ===
    train_pipeline_dataset = load_data_module(paths["train"], train_filename, config)
    val_pipeline_dataset = load_data_module(paths["val"], val_filename, config)
    test_pipeline_dataset = load_data_module(paths["test"], test_filename, config)

    # === Build tokenizer and preprocessing pipelines ===
    tokenizer, pipelines = data_preprocess_pipeline(config)

    # Add pipelines for each text field
    log_once(logger, "=" * 70)
    log_once(logger, "REGISTERING TEXT PROCESSING PIPELINES:")
    for field_name, pipeline in pipelines.items():
        log_once(
            logger, f"  Field: '{field_name}' -> Pipeline: {type(pipeline).__name__}"
        )
        train_pipeline_dataset.add_pipeline(field_name, pipeline)
        val_pipeline_dataset.add_pipeline(field_name, pipeline)
        test_pipeline_dataset.add_pipeline(field_name, pipeline)
    log_once(logger, f"✅ Registered {len(pipelines)} text processing pipelines")
    log_once(logger, "=" * 70)

    # === Build preprocessing pipelines (numerical imputation + risk tables) ===
    preprocessing_pipelines, imputation_dict, risk_tables = (
        build_preprocessing_pipelines(
            config,
            [train_pipeline_dataset, val_pipeline_dataset, test_pipeline_dataset],
            model_artifacts_dir=model_artifacts_dir,
            use_precomputed_imputation=use_precomputed_imputation,
            use_precomputed_risk_tables=use_precomputed_risk_tables,
        )
    )

    # Add preprocessing pipelines to all datasets
    log_once(logger, "=" * 70)
    log_once(logger, "REGISTERING NUMERICAL/CATEGORICAL PREPROCESSING PIPELINES:")
    for field, processor in preprocessing_pipelines.items():
        log_once(logger, f"  Field: '{field}' -> Processor: {type(processor).__name__}")
        train_pipeline_dataset.add_pipeline(field, processor)
        val_pipeline_dataset.add_pipeline(field, processor)
        test_pipeline_dataset.add_pipeline(field, processor)
    log_once(
        logger, f"✅ Registered {len(preprocessing_pipelines)} preprocessing pipelines"
    )
    log_once(logger, "=" * 70)

    # Store artifacts in config for saving
    config.imputation_dict = imputation_dict
    config.risk_tables = risk_tables

    # === Add multiclass label processor if needed ===
    if not config.is_binary and config.num_classes > 2:
        if config.multiclass_categories:
            label_processor = MultiClassLabelProcessor(
                label_list=config.multiclass_categories, strict=True
            )
        else:
            label_processor = MultiClassLabelProcessor()
        train_pipeline_dataset.add_pipeline(config.label_name, label_processor)
        val_pipeline_dataset.add_pipeline(config.label_name, label_processor)
        test_pipeline_dataset.add_pipeline(config.label_name, label_processor)

        # Save mappings into config for use in inference/export
        config.label_to_id = label_processor.label_to_id
        config.id_to_label = label_processor.id_to_label
        print(config.label_to_id)
        print(config.id_to_label)
    else:
        config.label_to_id = None
        config.id_to_label = None

    return (
        [train_pipeline_dataset, val_pipeline_dataset, test_pipeline_dataset],
        tokenizer,
        config,
    )


# ----------------- Model Building -----------------------
def build_model_and_optimizer(
    config: Config, tokenizer: AutoTokenizer, datasets: List[PipelineDataset]
) -> Tuple[nn.Module, DataLoader, DataLoader, DataLoader, torch.Tensor]:
    # Use unified collate function for all model types
    logger.info(f"Using collate batch for model: {config.model_class}")

    # Use unified keys for all models (single tokenizer design)
    collate_batch = build_collate_batch(
        input_ids_key=config.text_input_ids_key,
        attention_mask_key=config.text_attention_mask_key,
    )

    train_pipeline_dataset, val_pipeline_dataset, test_pipeline_dataset = datasets

    train_dataloader = DataLoader(
        train_pipeline_dataset,
        collate_fn=collate_batch,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_pipeline_dataset, collate_fn=collate_batch, batch_size=config.batch_size
    )
    test_dataloader = DataLoader(
        test_pipeline_dataset,
        collate_fn=collate_batch,
        batch_size=config.batch_size,
    )

    log_once(logger, f"Extract pretrained embedding from model: {config.tokenizer}")
    embedding_model = AutoModel.from_pretrained(config.tokenizer)
    embedding_mat = embedding_model.embeddings.word_embeddings.weight
    log_once(
        logger, f"Embedding shape: [{embedding_mat.shape[0]}, {embedding_mat.shape[1]}]"
    )
    config.embed_size = embedding_mat.shape[1]
    vocab_size = tokenizer.vocab_size
    log_once(logger, f"Vocabulary Size: {vocab_size}")
    log_once(logger, f"Model choice: {config.model_class}")
    model = model_select(config.model_class, config, vocab_size, embedding_mat)
    return model, train_dataloader, val_dataloader, test_dataloader, embedding_mat


# ----------------- Save to ONNX -----------------------------
def export_model_to_onnx(
    model: torch.nn.Module,
    trainer,
    val_dataloader: DataLoader,
    onnx_path: Union[str, Path],
):
    """
    Export a (possibly FSDP-wrapped) MultimodalBert model to ONNX using a sample batch from the validation dataloader.

    Args:
        model (torch.nn.Module): The trained model or FSDP-wrapped model.
        trainer: The Lightning trainer used during training (for strategy check).
        val_dataloader (DataLoader): DataLoader to fetch a sample batch for tracing.
        onnx_path (Union[str, Path]): File path to save the ONNX model.

    Raises:
        RuntimeError: If export fails.
    """
    logger.info(f"Exporting model to ONNX: {onnx_path}")

    # 1. Sample and move batch to CPU
    try:
        sample_batch = next(iter(val_dataloader))
    except StopIteration:
        raise RuntimeError("Validation dataloader is empty. Cannot export ONNX.")

    sample_batch_cpu = {
        k: v.to("cpu") if isinstance(v, torch.Tensor) else v
        for k, v in sample_batch.items()
    }

    # 2. Handle FSDP unwrapping if needed
    model_to_export = model
    if isinstance(trainer.strategy, FSDPStrategy):
        if isinstance(model, FSDP):
            logger.info("Unwrapping FSDP model for ONNX export.")
            model_to_export = model.module
        else:
            logger.warning("Trainer uses FSDPStrategy, but model is not FSDP-wrapped.")

    # 3. Move model to CPU and export
    model_to_export = model_to_export.to("cpu").eval()

    try:
        model_to_export.export_to_onnx(onnx_path, sample_batch_cpu)
        logger.info(f"ONNX export completed: {onnx_path}")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise RuntimeError("Failed to export model to ONNX.") from e


# ----------------- Evaluation and Logging -----------------------
def save_predictions_with_dataframe(
    df: pd.DataFrame,
    predictions: np.ndarray,
    output_dir: str,
    split_name: str,
    output_format: str = "csv",
) -> None:
    """
    Save predictions by adding probability columns to existing dataframe.

    Args:
        df: Dataframe with IDs, labels, and other metadata from inference
        predictions: Prediction probabilities of shape (N, num_classes)
        output_dir: Directory to save predictions
        split_name: Name of split (e.g., 'val', 'test')
        output_format: Format to save in ('csv', 'tsv', or 'parquet')
    """
    os.makedirs(output_dir, exist_ok=True)

    # Make a copy to avoid modifying the original
    df_output = df.copy()

    # Add probability columns for each class
    num_classes = predictions.shape[1] if len(predictions.shape) > 1 else 1
    if num_classes == 1:
        # Binary with single probability
        df_output["prob_class_0"] = 1 - predictions.squeeze()
        df_output["prob_class_1"] = predictions.squeeze()
    else:
        for i in range(num_classes):
            df_output[f"prob_class_{i}"] = predictions[:, i]

    # Save with format preservation
    output_base = os.path.join(output_dir, f"{split_name}_predictions")
    saved_path = save_dataframe_with_format(df_output, output_base, output_format)
    log_once(
        logger, f"Saved {split_name} predictions (format={output_format}): {saved_path}"
    )


def evaluate_and_log_results(
    model: nn.Module,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    config: Config,
    trainer: pl.Trainer,
    val_dataset: PipelineDataset,
    test_dataset: PipelineDataset,
    paths: Dict[str, str],
) -> None:
    log_once(logger, "Inference Starts ...")
    # Request dataframe to extract IDs from inference results
    val_predict_labels, val_true_labels, val_df = model_inference(
        model,
        val_dataloader,
        accelerator="gpu",
        device="auto",
        model_log_path=paths["checkpoint"],
        return_dataframe=True,
        label_col=config.label_name,
    )
    test_predict_labels, test_true_labels, test_df = model_inference(
        model,
        test_dataloader,
        accelerator="gpu",
        device="auto",
        model_log_path=paths["checkpoint"],
        return_dataframe=True,
        label_col=config.label_name,
    )
    log_once(logger, "Inference Complete.")
    if is_main_process():
        task = "binary" if config.is_binary else "multiclass"
        num_classes = config.num_classes
        output_metrics = ["auroc", "average_precision", "f1_score"]
        metric_test = compute_metrics(
            test_predict_labels,
            test_true_labels,
            output_metrics,
            task=task,
            num_classes=num_classes,
            stage="test",
        )
        metric_val = compute_metrics(
            val_predict_labels,
            val_true_labels,
            output_metrics,
            task=task,
            num_classes=num_classes,
            stage="val",
        )
        log_once(logger, "Metric output for Hyperparameter optimization:")
        for key, value in metric_val.items():
            log_once(logger, f"{key} = {value:.4f}")
        for key, value in metric_test.items():
            log_once(logger, f"{key} = {value:.4f}")
        log_once(logger, "Saving metric plots...")
        writer = SummaryWriter(
            log_dir=os.path.join(paths["output"], "tensorboard_eval")
        )
        roc_metric_plot(
            y_pred=test_predict_labels,
            y_true=test_true_labels,
            y_val_pred=val_predict_labels,
            y_val_true=val_true_labels,
            path=paths["output"],
            task=task,
            num_classes=num_classes,
            writer=writer,
            global_step=trainer.global_step,
        )
        pr_metric_plot(
            y_pred=test_predict_labels,
            y_true=test_true_labels,
            y_val_pred=val_predict_labels,
            y_val_true=val_true_labels,
            path=paths["output"],
            task=task,
            num_classes=num_classes,
            writer=writer,
            global_step=trainer.global_step,
        )
        writer.close()

        # Save legacy tensor format for backward compatibility
        prediction_filename = os.path.join(paths["output"], "predict_results.pth")
        log_once(logger, f"Saving prediction result to {prediction_filename}")
        save_prediction(prediction_filename, test_true_labels, test_predict_labels)

        # NEW: Save predictions as DataFrames with format preservation
        log_once(logger, "Saving predictions as DataFrames with format preservation...")
        output_format = config._input_format or "csv"

        # Save validation predictions (df already has IDs and labels from inference)
        save_predictions_with_dataframe(
            df=val_df,
            predictions=val_predict_labels,
            output_dir=paths["output"],
            split_name="val",
            output_format=output_format,
        )

        # Save test predictions (df already has IDs and labels from inference)
        save_predictions_with_dataframe(
            df=test_df,
            predictions=test_predict_labels,
            output_dir=paths["output"],
            split_name="test",
            output_format=output_format,
        )

        log_once(logger, "Prediction DataFrames saved successfully")


# ----------------- Main Function ---------------------------
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main function to execute the PyTorch training logic.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
    """
    # Load hyperparameters with region-specific support
    # Get region from environ_vars parameter (for testability)
    region = environ_vars.get("REGION", "").upper()

    if region in ["NA", "EU", "FE"]:
        hparam_filename = f"hyperparameters_{region}.json"
        logger.info(f"Loading region-specific hyperparameters for region: {region}")
    else:
        hparam_filename = "hyperparameters.json"
        if region:
            logger.warning(
                f"Unknown REGION '{region}', falling back to default hyperparameters.json"
            )
        else:
            logger.info("No REGION specified, using default hyperparameters.json")

    hparam_dir = input_paths.get("hyperparameters_s3_uri")
    if not hparam_dir.endswith(hparam_filename):
        hparam_file = os.path.join(hparam_dir, hparam_filename)
    else:
        hparam_file = hparam_dir

    logger.info(f"Loading hyperparameters from: {hparam_file}")
    hyperparameters = load_parse_hyperparameters(hparam_file)
    hyperparameters = sanitize_config(hyperparameters)

    try:
        config = Config(**hyperparameters)  # Validate config
    except ValidationError as e:
        logger.error(f"Configuration Error: {e}")
        raise

    # Build paths dictionary from input/output paths
    paths = {
        "train": os.path.join(input_paths["input_path"], "train"),
        "val": os.path.join(input_paths["input_path"], "val"),
        "test": os.path.join(input_paths["input_path"], "test"),
        "model": output_paths.get("model_output", "/opt/ml/model"),
        "output": output_paths.get("evaluation_output", "/opt/ml/output/data"),
        "checkpoint": os.path.join(
            output_paths.get("evaluation_output", "/opt/ml/output/data"), "checkpoints"
        ),
    }

    log_once(logger, "Final Hyperparameters:")
    log_once(logger, json.dumps(config.model_dump(), indent=4))
    log_once(logger, "================================================")
    log_once(logger, "Starting the training process.")

    device = setup_training_environment(config)

    # Pass environment variables for preprocessing artifact control
    datasets, tokenizer, config = load_and_preprocess_data(
        config,
        paths,
        model_artifacts_dir=input_paths.get("model_artifacts_input"),
        use_precomputed_imputation=environ_vars.get(
            "USE_PRECOMPUTED_IMPUTATION", False
        ),
        use_precomputed_risk_tables=environ_vars.get(
            "USE_PRECOMPUTED_RISK_TABLES", False
        ),
    )

    model, train_dataloader, val_dataloader, test_dataloader, embedding_mat = (
        build_model_and_optimizer(config, tokenizer, datasets)
    )
    # update tab dimension
    config.input_tab_dim = len(config.tab_field_list)
    log_once(logger, "Training starts using pytorch.lightning ...")
    trainer = model_train(
        model,
        config.model_dump(),
        train_dataloader,
        val_dataloader,
        device="auto",
        model_log_path=paths["checkpoint"],
        early_stop_metric=config.early_stop_metric,
    )
    log_once(logger, "Training Complete.")
    log_once(logger, "Evaluating final model.")
    if config.load_ckpt:
        best_model_path = trainer.checkpoint_callback.best_model_path
        log_once(logger, f"Load best model from checkpoint {best_model_path}")
        model = load_checkpoint(
            best_model_path, model_class=config.model_class, device_l="cpu"
        )
    if is_main_process():
        model_filename = os.path.join(paths["model"], "model.pth")
        logger.info(f"Saving model to {model_filename}")
        save_model(model_filename, model)
        artifact_filename = os.path.join(paths["model"], "model_artifacts.pth")
        logger.info(f"Saving model artifacts to {artifact_filename}")
        save_artifacts(
            artifact_filename,
            config.model_dump(),
            embedding_mat,
            tokenizer.vocab,
            model_class=config.model_class,
        )

        # ------------------ ONNX Export ------------------
        onnx_path = os.path.join(paths["model"], "model.onnx")
        logger.info(f"Saving model as ONNX to {onnx_path}")
        export_model_to_onnx(model, trainer, val_dataloader, onnx_path)

        # ------------------ Save Hyperparameters Configuration ------------------
        hyperparameters_file = os.path.join(paths["model"], "hyperparameters.json")
        logger.info(f"Saving hyperparameters configuration to {hyperparameters_file}")
        with open(hyperparameters_file, "w") as f:
            json.dump(config.model_dump(), f, indent=2, sort_keys=True)
        logger.info(f"✓ Saved hyperparameters configuration to {hyperparameters_file}")

        # ------------------ Save Feature Columns ------------------
        feature_columns_file = os.path.join(paths["model"], "feature_columns.txt")
        logger.info(f"Saving feature column names to {feature_columns_file}")
        feature_columns = config.tab_field_list + config.cat_field_list
        with open(feature_columns_file, "w") as f:
            f.write("# Feature columns in exact order required for model inference\n")
            f.write("# DO NOT MODIFY THE ORDER OF THESE COLUMNS\n")
            f.write("# Each line contains: <column_index>,<column_name>\n")
            for idx, col_name in enumerate(feature_columns):
                f.write(f"{idx},{col_name}\n")
        logger.info(
            f"✓ Saved {len(feature_columns)} feature columns to {feature_columns_file}"
        )

        # ------------------ Save Preprocessing Artifacts ------------------
        if config.imputation_dict:
            logger.info("Saving numerical imputation artifacts...")
            save_imputation_artifacts(config.imputation_dict, paths["model"])

        if config.risk_tables:
            logger.info("Saving risk table artifacts...")
            save_risk_table_artifacts(config.risk_tables, paths["model"])

    # Extract datasets for evaluation
    train_dataset, val_dataset, test_dataset = datasets

    # CRITICAL FIX: Ensure all ranks finish training before evaluation
    # This prevents race conditions in distributed training where ranks might
    # try to read each other's test result files before they're fully written
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        log_once(
            logger, "All ranks synchronized after training - proceeding to evaluation"
        )

    # CRITICAL FIX: All ranks must participate in evaluation for distributed inference
    # PyTorch Lightning's test() method requires all ranks to participate in collective operations
    log_once(logger, f"Rank {get_rank()} starting evaluation...")
    evaluate_and_log_results(
        model,
        val_dataloader,
        test_dataloader,
        config,
        trainer,
        val_dataset,
        test_dataset,
        paths,
    )

    # CRITICAL FIX: Final barrier to ensure all ranks complete evaluation together
    # This prevents premature termination and ensures proper cleanup
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        log_once(logger, "All ranks synchronized after evaluation - ready to exit")


# ----------------- Entrypoint ---------------------------
if __name__ == "__main__":
    logger.info("Script starting...")

    # Container path constants
    CONTAINER_PATHS = {
        "INPUT_DATA": "/opt/ml/input/data",
        "MODEL_DIR": "/opt/ml/model",
        "OUTPUT_DATA": "/opt/ml/output/data",
        "CONFIG_DIR": "/opt/ml/code/hyperparams",  # Source directory path (matches XGBoost)
        "MODEL_ARTIFACTS_INPUT": "/opt/ml/input/data/model_artifacts_input",  # Optional pre-computed artifacts
    }

    # Define input and output paths using contract logical names
    # Use container defaults (no CLI arguments per contract)
    input_paths = {
        "input_path": CONTAINER_PATHS["INPUT_DATA"],
        "hyperparameters_s3_uri": CONTAINER_PATHS["CONFIG_DIR"],
    }

    # Add model_artifacts_input only if the directory exists (optional)
    if os.path.exists(CONTAINER_PATHS["MODEL_ARTIFACTS_INPUT"]):
        input_paths["model_artifacts_input"] = CONTAINER_PATHS["MODEL_ARTIFACTS_INPUT"]
        logger.info(
            f"Found model artifacts input directory: {CONTAINER_PATHS['MODEL_ARTIFACTS_INPUT']}"
        )

    output_paths = {
        "model_output": CONTAINER_PATHS["MODEL_DIR"],
        "evaluation_output": CONTAINER_PATHS["OUTPUT_DATA"],
    }

    # Collect environment variables for preprocessing artifact control
    environ_vars = {
        "USE_PRECOMPUTED_IMPUTATION": os.environ.get(
            "USE_PRECOMPUTED_IMPUTATION", "false"
        ).lower()
        == "true",
        "USE_PRECOMPUTED_RISK_TABLES": os.environ.get(
            "USE_PRECOMPUTED_RISK_TABLES", "false"
        ).lower()
        == "true",
        "REGION": os.environ.get("REGION", "NA"),
    }

    # Create empty args namespace to maintain function signature
    args = argparse.Namespace()

    try:
        logger.info(f"Starting main process with paths:")
        logger.info(f"  Data directory: {input_paths['input_path']}")
        logger.info(f"  Config directory: {input_paths['hyperparameters_s3_uri']}")
        logger.info(f"  Model directory: {output_paths['model_output']}")
        logger.info(f"  Output directory: {output_paths['evaluation_output']}")

        # Call the refactored main function
        main(input_paths, output_paths, environ_vars, args)

        logger.info("PyTorch training script completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Exception during training: {str(e)}")
        logger.error(traceback.format_exc())

        # Write failure file for compatibility
        failure_file = os.path.join(output_paths["evaluation_output"], "failure")
        try:
            with open(failure_file, "w") as f:
                f.write(
                    "Exception during training: "
                    + str(e)
                    + "\n"
                    + traceback.format_exc()
                )
        except Exception:
            pass  # Don't fail if we can't write the failure file

        sys.exit(1)
