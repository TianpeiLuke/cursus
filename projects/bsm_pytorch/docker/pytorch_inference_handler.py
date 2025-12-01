import os
import json
import traceback
from io import StringIO, BytesIO
from pathlib import Path
import logging
from typing import List, Union, Dict, Tuple, Optional, Any
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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
)
from lightning_models.utils.dist_utils import get_rank, is_main_process
from pydantic import BaseModel, Field, ValidationError  # For Config Validation

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # <-- THIS LINE IS MISSING

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================================
class Config(BaseModel):
    id_name: str = "order_id"
    text_name: str = "text"
    label_name: str = "label"
    batch_size: int = 32
    full_field_list: List[str] = Field(default_factory=list)
    cat_field_list: List[str] = Field(default_factory=list)
    tab_field_list: List[str] = Field(default_factory=list)
    categorical_features_to_encode: List[str] = Field(default_factory=list)
    header: int = 0
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
    run_scheduler: bool = True
    reinit_pooler: bool = True
    reinit_layers: int = 2
    warmup_steps: int = 300
    text_input_ids_key: str = "input_ids"  # Configurable text input key
    text_attention_mask_key: str = "attention_mask"  # Configurable attention mask key
    train_filename: Optional[str] = None
    val_filename: Optional[str] = None
    test_filename: Optional[str] = None
    embed_size: Optional[int] = None  # Added for type consistency
    model_path: str = "/opt/ml/model"  # Add model_path with a default value
    categorical_processor_mappings: Optional[Dict[str, Dict[str, int]]] = (
        None  # Add this line
    )
    label_to_id: Optional[Dict[str, int]] = None  # Added: label to ID mapping
    id_to_label: Optional[List[str]] = None  # Added: ID to label mapping

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


# =================== Preprocessing Artifact Loaders ================
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


# =================== Calibration Functions ================
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


# =================== Text Preprocessing Pipeline ================
def data_preprocess_pipeline(
    config: Config,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Tuple[AutoTokenizer, Dict[str, Processor]]:
    """
    Build text preprocessing pipelines based on config and hyperparameters.

    For bimodal: Uses text_name with configured or default steps
    For trimodal: Uses primary_text_name and secondary_text_name with separate steps

    Args:
        config: Configuration object
        hyperparameters: Optional hyperparameters dict loaded from hyperparameters.json

    Returns:
        Tuple of (tokenizer, pipelines_dict)
    """
    if not config.tokenizer:
        config.tokenizer = "bert-base-multilingual-cased"

    logger.info(f"Constructing tokenizer: {config.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    pipelines = {}

    # Extract processing steps from hyperparameters if available
    text_steps = None
    primary_steps = None
    secondary_steps = None
    primary_text_name = None
    secondary_text_name = None

    if hyperparameters:
        text_steps = hyperparameters.get("text_processing_steps")
        primary_steps = hyperparameters.get("primary_text_processing_steps")
        secondary_steps = hyperparameters.get("secondary_text_processing_steps")
        primary_text_name = hyperparameters.get("primary_text_name")
        secondary_text_name = hyperparameters.get("secondary_text_name")

    # BIMODAL: Single text pipeline
    if not primary_text_name:
        # Use configured steps from hyperparameters or fallback to default
        steps = text_steps or [
            "dialogue_splitter",
            "html_normalizer",
            "emoji_remover",
            "text_normalizer",
            "dialogue_chunker",
            "tokenizer",
        ]

        pipelines[config.text_name] = build_text_pipeline_from_steps(
            processing_steps=steps,
            tokenizer=tokenizer,
            max_sen_len=config.max_sen_len,
            chunk_trancate=config.chunk_trancate,
            max_total_chunks=config.max_total_chunks,
            input_ids_key=config.text_input_ids_key,
            attention_mask_key=config.text_attention_mask_key,
        )
        logger.info(
            f"Built bimodal pipeline for '{config.text_name}' with steps: {steps}"
        )

    # TRIMODAL: Dual text pipelines
    else:
        # Primary text pipeline (e.g., chat - full cleaning)
        primary_pipeline_steps = primary_steps or [
            "dialogue_splitter",
            "html_normalizer",
            "emoji_remover",
            "text_normalizer",
            "dialogue_chunker",
            "tokenizer",
        ]

        pipelines[primary_text_name] = build_text_pipeline_from_steps(
            processing_steps=primary_pipeline_steps,
            tokenizer=tokenizer,
            max_sen_len=config.max_sen_len,
            chunk_trancate=config.chunk_trancate,
            max_total_chunks=config.max_total_chunks,
            input_ids_key=config.text_input_ids_key,
            attention_mask_key=config.text_attention_mask_key,
        )
        logger.info(
            f"Built primary pipeline for '{primary_text_name}' with steps: {primary_pipeline_steps}"
        )

        # Secondary text pipeline (e.g., events - minimal cleaning)
        secondary_pipeline_steps = secondary_steps or [
            "dialogue_splitter",
            "text_normalizer",
            "dialogue_chunker",
            "tokenizer",
        ]

        pipelines[secondary_text_name] = build_text_pipeline_from_steps(
            processing_steps=secondary_pipeline_steps,
            tokenizer=tokenizer,
            max_sen_len=config.max_sen_len,
            chunk_trancate=config.chunk_trancate,
            max_total_chunks=config.max_total_chunks,
            input_ids_key=config.text_input_ids_key,
            attention_mask_key=config.text_attention_mask_key,
        )
        logger.info(
            f"Built secondary pipeline for '{secondary_text_name}' with steps: {secondary_pipeline_steps}"
        )

    return tokenizer, pipelines


# =================== Model Function ======================
def model_fn(model_dir, context=None):
    model_filename = "model.pth"
    model_artifact_name = "model_artifacts.pth"
    onnx_model_path = os.path.join(model_dir, "model.onnx")

    load_config, embedding_mat, vocab, model_class = load_artifacts(
        os.path.join(model_path, model_artifact_name), device_l=device
    )

    config = Config(**load_config)

    # Load model based on file type
    if os.path.exists(onnx_model_path):
        logger.info("Detected ONNX model.")
        model = load_onnx_model(onnx_model_path)
    else:
        logger.info("Detected PyTorch model.")
        model = load_model(
            os.path.join(model_path, model_filename),
            config.model_dump(),
            embedding_mat,
            model_class,
            device_l=device,
        )
        model.eval()

    # Load hyperparameters if available
    hyperparameters = load_hyperparameters(model_dir)

    ## reconstruct pipelines with hyperparameter-driven steps
    tokenizer, pipelines = data_preprocess_pipeline(config, hyperparameters)

    # === Add multiclass label processor if needed ===
    if not config.is_binary and config.num_classes > 2:
        if config.multiclass_categories:
            label_processor = MultiClassLabelProcessor(
                label_list=config.multiclass_categories, strict=True
            )
            pipelines[config.label_name] = label_processor

    # === Load preprocessing artifacts (numerical imputation + risk tables) ===
    logger.info("Loading preprocessing artifacts...")
    risk_tables = load_risk_tables(model_dir)
    risk_processors = create_risk_processors(risk_tables)

    impute_dict = load_imputation_dict(model_dir)
    numerical_processors = create_numerical_processors(impute_dict)

    logger.info(
        f"Loaded {len(risk_processors)} risk processors and {len(numerical_processors)} numerical processors"
    )

    # Load feature columns if available (for alignment with XGBoost pattern)
    feature_columns = read_feature_columns(model_dir)

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
        "calibrator": calibrator,
        "feature_columns": feature_columns,
        "hyperparameters": hyperparameters,
        "risk_processors": risk_processors,
        "numerical_processors": numerical_processors,
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
    calibrator = model_data.get("calibrator")
    risk_processors = model_data.get("risk_processors", {})
    numerical_processors = model_data.get("numerical_processors", {})

    config_predict = config.model_dump()
    label_field = config_predict.get("label_name", None)

    if label_field:
        config_predict["full_field_list"] = [
            col for col in config_predict["full_field_list"] if col != label_field
        ]
        config_predict["cat_field_list"] = [
            col for col in config_predict["cat_field_list"] if col != label_field
        ]

    dataset = PipelineDataset(config_predict, dataframe=input_object)

    # === Apply preprocessing artifacts (numerical imputation + risk tables) ===
    logger.info("Applying preprocessing to inference data...")

    # Apply numerical imputation processors
    for feature, processor in numerical_processors.items():
        if feature in dataset.DataReader.columns:
            logger.debug(f"Applying numerical imputation for feature: {feature}")
            dataset.add_pipeline(feature, processor)

    # Apply risk table mapping processors
    for feature, processor in risk_processors.items():
        if feature in dataset.DataReader.columns:
            logger.debug(f"Applying risk table mapping for feature: {feature}")
            dataset.add_pipeline(feature, processor)

    logger.info(
        f"Applied {len(numerical_processors)} numerical processors and {len(risk_processors)} risk processors"
    )

    # Apply text and other pipelines
    for feature_name, pipeline in pipelines.items():
        dataset.add_pipeline(feature_name, pipeline)

    collate_batch = build_collate_batch(
        input_ids_key=config.text_input_ids_key,
        attention_mask_key=config.text_attention_mask_key,
    )

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
                        "output-label": f"class-{max_idx}"
                        if max_idx >= 0
                        else "unknown",
                    }
                else:
                    # Multiclass
                    record = {}
                    for i in range(len(raw_probs)):
                        record[f"prob_{str(i + 1).zfill(2)}"] = str(raw_probs[i])
                        record[f"calibrated_prob_{str(i + 1).zfill(2)}"] = str(
                            cal_probs[i]
                        )
                    record["output-label"] = (
                        f"class-{max_idx}" if max_idx >= 0 else "unknown"
                    )

                output_records.append(record)

            response = json.dumps({"predictions": output_records})
            return response, "application/json"

        # Step 4: CSV output formatting
        elif accept.lower() == "text/csv":
            csv_lines = []
            for raw_probs, cal_probs in zip(raw_scores_list, calibrated_scores_list):
                max_idx = raw_probs.index(max(raw_probs)) if raw_probs else -1

                # Format raw probabilities
                raw_formatted = [round(float(p), 4) for p in raw_probs]
                raw_str = ",".join(f"{p:.4f}" for p in raw_formatted)

                # Format calibrated probabilities
                cal_formatted = [round(float(p), 4) for p in cal_probs]
                cal_str = ",".join(f"{p:.4f}" for p in cal_formatted)

                line = [
                    raw_str,
                    cal_str,
                    f"class-{max_idx}" if max_idx >= 0 else "unknown",
                ]
                csv_lines.append(",".join(map(str, line)))

            response_body = "\n".join(csv_lines) + "\n"
            return response_body, "text/csv"

        # Step 4: Unsupported content type
        else:
            logger.error(f"Unsupported accept type: {accept}")
            raise ValueError(f"Unsupported accept type: {accept}")

    # Step 5: Error handling
    except Exception as e:
        logger.error(
            f"Error during DataFrame creation or serialization in output_fn: {e}",
            exc_info=True,
        )
        error_response = json.dumps({"error": f"Failed to serialize output: {e}"})
        return error_response, "application/json"
