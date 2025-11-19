#!/usr/bin/env python
import os
import json
import argparse
import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
)
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import time
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import lightning.pytorch as pl

# Import PyTorch components
from processing.processors import Processor
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
from processing.datasets.bsm_datasets import BSMDataset
from processing.dataloaders.bsm_dataloader import (
    build_collate_batch,
    build_trimodal_collate_batch,
)

from lightning_models.utils.pl_train import (
    model_inference,
    model_online_inference,
    load_model,
    load_artifacts,
    load_onnx_model,
)
from lightning_models.utils.pl_model_plots import compute_metrics
from lightning_models.utils.dist_utils import get_rank, is_main_process
from pydantic import BaseModel, Field, ValidationError

import logging

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Import TriModalHyperparameters for inference config
try:
    from hyperparams.hyperparameters_trimodal import TriModalHyperparameters

    # Use TriModalHyperparameters as the Config class for full alignment
    Config = TriModalHyperparameters
except ImportError:
    logger.warning(
        "Could not import TriModalHyperparameters, falling back to basic Config"
    )

    # Fallback Config class if import fails
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
        multiclass_categories: List[Union[int, str]] = Field(
            default_factory=lambda: [0, 1]
        )
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
        text_input_ids_key: str = "input_ids"
        text_attention_mask_key: str = "attention_mask"
        train_filename: Optional[str] = None
        val_filename: Optional[str] = None
        test_filename: Optional[str] = None
        embed_size: Optional[int] = None
        model_path: str = "/opt/ml/model"
        categorical_processor_mappings: Optional[Dict[str, Dict[str, int]]] = None
        label_to_id: Optional[Dict[str, int]] = None
        id_to_label: Optional[List[str]] = None

        # === Trimodal Configuration Fields ===
        primary_text_name: Optional[str] = None
        secondary_text_name: Optional[str] = None
        primary_tokenizer: Optional[str] = None
        secondary_tokenizer: Optional[str] = None
        primary_hidden_common_dim: Optional[int] = None
        secondary_hidden_common_dim: Optional[int] = None
        primary_text_input_ids_key: str = "input_ids"
        primary_text_attention_mask_key: str = "attention_mask"
        secondary_text_input_ids_key: str = "input_ids"
        secondary_text_attention_mask_key: str = "attention_mask"
        primary_text_processing_steps: Optional[List[str]] = None
        secondary_text_processing_steps: Optional[List[str]] = None
        fusion_hidden_dim: Optional[int] = None
        fusion_dropout: float = 0.1
        cross_attention_heads: int = 8
        cross_attention_dropout: float = 0.1

        # Additional fields that might be saved during training
        primary_reinit_pooler: Optional[bool] = None
        primary_reinit_layers: Optional[int] = None
        secondary_reinit_pooler: Optional[bool] = None
        secondary_reinit_layers: Optional[int] = None
        is_embeddings_trainable: bool = True
        num_workers: int = 0
        categorical_label_features: Optional[List[str]] = None

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
                if len(set(self.multiclass_categories)) != len(
                    self.multiclass_categories
                ):
                    raise ValueError(
                        "multiclass_categories must contain unique values."
                    )
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


def load_pytorch_model_artifacts(
    model_dir: str,
) -> Tuple[torch.nn.Module, Dict[str, Any], AutoTokenizer, Dict[str, Processor]]:
    """
    Load the trained PyTorch model and all preprocessing artifacts from the specified directory.
    Returns model, config, tokenizer, and preprocessing pipelines.
    """
    logger.info(f"Loading PyTorch model artifacts from {model_dir}")

    model_filename = "model.pth"
    model_artifact_name = "model_artifacts.pth"
    hyperparams_filename = "hyperparameters.json"
    onnx_model_path = os.path.join(model_dir, "model.onnx")

    # Try to load hyperparameters from the saved hyperparameters.json first
    hyperparams_path = os.path.join(model_dir, hyperparams_filename)
    if os.path.exists(hyperparams_path):
        logger.info(f"Loading hyperparameters from {hyperparams_path}")
        with open(hyperparams_path, "r") as f:
            load_config = json.load(f)

        # Still need to load artifacts for embedding_mat, vocab, and model_class
        _, embedding_mat, vocab, model_class = load_artifacts(
            os.path.join(model_dir, model_artifact_name), device_l=device
        )
    else:
        # Fallback to loading config from artifacts (backward compatibility)
        logger.info(
            "Hyperparameters.json not found, loading config from model artifacts"
        )
        load_config, embedding_mat, vocab, model_class = load_artifacts(
            os.path.join(model_dir, model_artifact_name), device_l=device
        )

    config = Config(**load_config)

    # Load model based on file type
    if os.path.exists(onnx_model_path):
        logger.info("Detected ONNX model.")
        model = load_onnx_model(onnx_model_path)
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

    # Reconstruct preprocessing pipelines
    tokenizers, pipelines = data_preprocess_pipeline(config)

    # Add multiclass label processor if needed
    if not config.is_binary and config.num_classes > 2:
        if config.multiclass_categories:
            label_processor = MultiClassLabelProcessor(
                label_list=config.multiclass_categories, strict=True
            )
            pipelines[config.label_name] = label_processor

    return model, config.model_dump(), tokenizers, pipelines


def build_processing_pipeline(
    processing_steps: List[str],
    tokenizer: AutoTokenizer,
    config: Dict[str, Any],
    input_ids_key: str = "input_ids",
    attention_mask_key: str = "attention_mask",
) -> Processor:
    """
    Build a processing pipeline based on the specified steps.

    Args:
        processing_steps: List of processing step names
        tokenizer: Tokenizer to use for tokenization step
        config: Configuration dictionary
        input_ids_key: Key name for input_ids in tokenized output
        attention_mask_key: Key name for attention_mask in tokenized output

    Returns:
        Composed processor pipeline
    """
    # Map step names to processor classes
    step_map = {
        "dialogue_splitter": DialogueSplitterProcessor,
        "html_normalizer": HTMLNormalizerProcessor,
        "emoji_remover": EmojiRemoverProcessor,
        "text_normalizer": TextNormalizationProcessor,
        "dialogue_chunker": lambda: DialogueChunkerProcessor(
            tokenizer=tokenizer,
            max_tokens=config.get("max_sen_len", 512),
            truncate=config.get("chunk_trancate", False),
            max_total_chunks=config.get("max_total_chunks", 5),
        ),
        "tokenizer": lambda: BertTokenizeProcessor(
            tokenizer,
            add_special_tokens=True,
            max_length=config.get("max_sen_len", 512),
            input_ids_key=input_ids_key,
            attention_mask_key=attention_mask_key,
        ),
    }

    # Build pipeline by chaining processors
    pipeline = None
    for step_name in processing_steps:
        if step_name not in step_map:
            logger.warning(f"Unknown processing step '{step_name}', skipping")
            continue

        processor_class = step_map[step_name]
        processor = (
            processor_class() if not callable(processor_class) else processor_class()
        )

        if pipeline is None:
            pipeline = processor
        else:
            pipeline = pipeline >> processor

    if pipeline is None:
        raise ValueError(f"No valid processing steps found in: {processing_steps}")

    return pipeline


def data_preprocess_pipeline(
    config: Config,
) -> Tuple[Dict[str, AutoTokenizer], Dict[str, Processor]]:
    """
    Create preprocessing pipelines for text modalities.
    Supports both single text (bi-modal) and dual text (tri-modal) configurations
    with configurable processing steps.
    """
    if not config.tokenizer:
        config.tokenizer = "bert-base-multilingual-cased"

    tokenizers = {}
    pipelines = {}

    # Check if this is tri-modal configuration
    is_trimodal = config.primary_text_name and config.secondary_text_name

    if is_trimodal:
        logger.info("Setting up tri-modal text processing pipelines")

        # Primary text pipeline (e.g., chat)
        primary_tokenizer_name = config.primary_tokenizer or config.tokenizer
        logger.info(f"Constructing primary tokenizer: {primary_tokenizer_name}")
        primary_tokenizer = AutoTokenizer.from_pretrained(primary_tokenizer_name)

        # Get processing steps from config
        primary_steps = config.primary_text_processing_steps or [
            "dialogue_splitter",
            "html_normalizer",
            "emoji_remover",
            "text_normalizer",
            "dialogue_chunker",
            "tokenizer",
        ]
        logger.info(f"Primary text processing steps: {primary_steps}")

        primary_pipeline = build_processing_pipeline(
            primary_steps,
            primary_tokenizer,
            config.model_dump(),
            input_ids_key=config.primary_text_input_ids_key,
            attention_mask_key=config.primary_text_attention_mask_key,
        )

        # Secondary text pipeline (e.g., shiptrack)
        secondary_tokenizer_name = config.secondary_tokenizer or config.tokenizer
        logger.info(f"Constructing secondary tokenizer: {secondary_tokenizer_name}")
        secondary_tokenizer = AutoTokenizer.from_pretrained(secondary_tokenizer_name)

        # Get processing steps from config
        secondary_steps = config.secondary_text_processing_steps or [
            "dialogue_splitter",
            "text_normalizer",
            "dialogue_chunker",
            "tokenizer",
        ]
        logger.info(f"Secondary text processing steps: {secondary_steps}")

        secondary_pipeline = build_processing_pipeline(
            secondary_steps,
            secondary_tokenizer,
            config.model_dump(),
            input_ids_key=config.secondary_text_input_ids_key,
            attention_mask_key=config.secondary_text_attention_mask_key,
        )

        tokenizers = {"primary": primary_tokenizer, "secondary": secondary_tokenizer}

        pipelines = {
            config.primary_text_name: primary_pipeline,
            config.secondary_text_name: secondary_pipeline,
        }

        logger.info(f"Primary text field: {config.primary_text_name}")
        logger.info(f"Secondary text field: {config.secondary_text_name}")

    else:
        # Traditional bi-modal setup
        logger.info("Setting up bi-modal text processing pipeline")
        logger.info(f"Constructing tokenizer: {config.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

        # Use default processing steps for bi-modal
        default_steps = [
            "dialogue_splitter",
            "html_normalizer",
            "emoji_remover",
            "text_normalizer",
            "dialogue_chunker",
            "tokenizer",
        ]

        dialogue_pipeline = build_processing_pipeline(
            default_steps,
            tokenizer,
            config.model_dump(),
            input_ids_key=config.text_input_ids_key,
            attention_mask_key=config.text_attention_mask_key,
        )

        tokenizers = {"main": tokenizer}
        pipelines = {config.text_name: dialogue_pipeline}
        logger.info(f"Text field: {config.text_name}")

    return tokenizers, pipelines


def preprocess_pytorch_eval_data(
    df: pd.DataFrame, config: Dict[str, Any], pipelines: Dict[str, Processor]
) -> pd.DataFrame:
    """
    Apply PyTorch preprocessing pipelines to the evaluation DataFrame.
    Preserves any non-feature columns like id and label.
    """
    logger.info(f"Preprocessing evaluation data with shape: {df.shape}")

    # Create BSMDataset for processing
    dataset = BSMDataset(config, dataframe=df)

    # Apply all preprocessing pipelines
    for feature_name, pipeline in pipelines.items():
        if feature_name in df.columns:
            logger.info(f"Applying preprocessing pipeline for feature: {feature_name}")
            dataset.add_pipeline(feature_name, pipeline)
        else:
            logger.warning(f"Feature {feature_name} not found in evaluation data")

    logger.info(f"Preprocessing complete")
    return dataset


def load_eval_data(eval_data_dir: str) -> pd.DataFrame:
    """
    Load the first .csv or .parquet file found in the evaluation data directory.
    Returns a pandas DataFrame.
    """
    logger.info(f"Loading eval data from {eval_data_dir}")
    eval_files = sorted(
        [
            f
            for f in Path(eval_data_dir).glob("**/*")
            if f.suffix in [".csv", ".parquet"]
        ]
    )
    if not eval_files:
        logger.error("No eval data file found in eval_data input.")
        raise RuntimeError("No eval data file found in eval_data input.")
    eval_file = eval_files[0]
    logger.info(f"Using eval data file: {eval_file}")
    if eval_file.suffix == ".parquet":
        df = pd.read_parquet(eval_file)
    else:
        df = pd.read_csv(eval_file)
    logger.info(f"Loaded eval data shape: {df.shape}")
    return df


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


def save_predictions(
    ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    id_col: str,
    label_col: str,
    output_eval_dir: str,
) -> None:
    """
    Save predictions to a CSV file, including id, true label, and class probabilities.
    """
    logger.info(f"Saving predictions to {output_eval_dir}")
    prob_cols = [f"prob_class_{i}" for i in range(y_prob.shape[1])]
    out_df = pd.DataFrame({id_col: ids, label_col: y_true})
    for i, col in enumerate(prob_cols):
        out_df[col] = y_prob[:, i]
    out_path = os.path.join(output_eval_dir, "eval_predictions.csv")
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved predictions to {out_path}")


def save_metrics(
    metrics: Dict[str, Union[int, float, str]], output_metrics_dir: str
) -> None:
    """
    Save computed metrics as a JSON file.
    """
    out_path = os.path.join(output_metrics_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {out_path}")

    # Also create a plain text summary for easy viewing
    summary_path = os.path.join(output_metrics_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("METRICS SUMMARY\n")
        f.write("=" * 50 + "\n")

        # Write key metrics at the top
        if "auroc" in metrics:  # Binary classification
            f.write(f"AUC-ROC:           {metrics['auroc']:.4f}\n")
            if "average_precision" in metrics:
                f.write(f"Average Precision: {metrics['average_precision']:.4f}\n")
            if "f1_score" in metrics:
                f.write(f"F1 Score:          {metrics['f1_score']:.4f}\n")
        else:  # Multiclass classification
            f.write(f"AUC-ROC (Macro):   {metrics.get('auroc_macro', 'N/A'):.4f}\n")
            f.write(f"AUC-ROC (Micro):   {metrics.get('auroc_micro', 'N/A'):.4f}\n")
            f.write(f"F1 Score (Macro):  {metrics.get('f1_score_macro', 'N/A'):.4f}\n")

        f.write("=" * 50 + "\n\n")

        # Write all metrics
        f.write("ALL METRICS\n")
        f.write("=" * 50 + "\n")
        for name, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                f.write(f"{name}: {value:.6f}\n")
            else:
                f.write(f"{name}: {value}\n")

    logger.info(f"Saved metrics summary to {summary_path}")


def plot_and_save_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray, output_dir: str, prefix: str = ""
) -> None:
    """
    Plot ROC curve and save as JPG.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    out_path = os.path.join(output_dir, f"{prefix}roc_curve.jpg")
    plt.savefig(out_path, format="jpg")
    plt.close()
    logger.info(f"Saved ROC curve to {out_path}")


def plot_and_save_pr_curve(
    y_true: np.ndarray, y_score: np.ndarray, output_dir: str, prefix: str = ""
) -> None:
    """
    Plot Precision-Recall curve and save as JPG.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (AP = {ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    out_path = os.path.join(output_dir, f"{prefix}pr_curve.jpg")
    plt.savefig(out_path, format="jpg")
    plt.close()
    logger.info(f"Saved PR curve to {out_path}")


def evaluate_pytorch_model(
    model: torch.nn.Module,
    dataset: BSMDataset,
    config: Dict[str, Any],
    id_col: str,
    label_col: str,
    output_eval_dir: str,
    output_metrics_dir: str,
) -> None:
    """
    Run PyTorch model prediction and evaluation, then save predictions and metrics.
    Also generate and save ROC and PR curves as JPG.
    """
    logger.info("Evaluating PyTorch model")

    # Determine collate function based on model type and configuration
    is_trimodal_model = config.get("model_class", "") in [
        "trimodal_bert",
        "trimodal_cross_attn_bert",
        "trimodal_gate_fusion_bert",
    ]
    has_dual_text_config = config.get("primary_text_name") and config.get(
        "secondary_text_name"
    )

    if is_trimodal_model and has_dual_text_config:
        # For tri-modal models, use the enhanced collate function that handles multiple text fields
        logger.info(
            f"Using trimodal collate function for {config.get('model_class')} model"
        )
        bsm_collate_batch = build_trimodal_collate_batch(
            primary_input_ids_key=config.get("primary_text_input_ids_key", "input_ids"),
            primary_attention_mask_key=config.get(
                "primary_text_attention_mask_key", "attention_mask"
            ),
            secondary_input_ids_key=config.get(
                "secondary_text_input_ids_key", "input_ids"
            ),
            secondary_attention_mask_key=config.get(
                "secondary_text_attention_mask_key", "attention_mask"
            ),
        )
    else:
        # For bi-modal models (including those with dual text config but non-trimodal model)
        logger.info(
            f"Using bi-modal collate function for {config.get('model_class')} model"
        )
        # Use primary text keys if available, otherwise fall back to traditional text keys
        if has_dual_text_config:
            # Use primary text for bi-modal models with dual text config
            input_ids_key = config.get("primary_text_input_ids_key", "input_ids")
            attention_mask_key = config.get(
                "primary_text_attention_mask_key", "attention_mask"
            )
        else:
            # Traditional single text configuration
            input_ids_key = config.get("text_input_ids_key", "input_ids")
            attention_mask_key = config.get("text_attention_mask_key", "attention_mask")

        bsm_collate_batch = build_collate_batch(
            input_ids_key=input_ids_key,
            attention_mask_key=attention_mask_key,
        )

    # Create DataLoader
    batch_size = config.get("batch_size", 32)
    dataloader = DataLoader(
        dataset, collate_fn=bsm_collate_batch, batch_size=batch_size
    )

    # Run inference using existing PyTorch inference function
    logger.info("Running model inference...")
    y_prob = model_online_inference(model, dataloader)
    logger.info(f"Model prediction shape: {y_prob.shape}")

    # Convert to proper probability format
    if len(y_prob.shape) == 1:
        y_prob = np.column_stack([1 - y_prob, y_prob])
        logger.info("Converted binary prediction to two-column probabilities")

    # Get true labels and IDs from the original dataframe
    df = dataset.DataReader
    y_true = df[label_col].values
    ids = df[id_col].values

    # Determine the classification type from the model's saved hyperparameters
    is_binary_model = config.get("is_binary", True)

    if is_binary_model:
        logger.info(
            "Detected binary classification task based on model hyperparameters."
        )
        # Ensure y_true is also binary (0 or 1) for consistent metric calculation
        y_true = (y_true > 0).astype(int)

        # Compute metrics using existing PyTorch function
        task = "binary"
        num_classes = 2
        output_metrics = ["auroc", "average_precision", "f1_score"]

        # Convert to tensors for compute_metrics function
        y_prob_tensor = torch.tensor(y_prob)
        y_true_tensor = torch.tensor(y_true)

        metrics = compute_metrics(
            y_prob_tensor[:, 1],  # Use positive class probabilities for binary
            y_true_tensor,
            output_metrics,
            task=task,
            num_classes=num_classes,
            stage="test",
        )

        # Convert tensor values to float for JSON serialization
        metrics = {k: float(v) if torch.is_tensor(v) else v for k, v in metrics.items()}

        plot_and_save_roc_curve(y_true, y_prob[:, 1], output_metrics_dir)
        plot_and_save_pr_curve(y_true, y_prob[:, 1], output_metrics_dir)
    else:
        n_classes = y_prob.shape[1]
        logger.info(
            f"Detected multiclass classification task with {n_classes} classes."
        )

        # Compute metrics using existing PyTorch function
        task = "multiclass"
        num_classes = n_classes
        output_metrics = ["auroc", "average_precision", "f1_score"]

        # Convert to tensors for compute_metrics function
        y_prob_tensor = torch.tensor(y_prob)
        y_true_tensor = torch.tensor(y_true)

        metrics = compute_metrics(
            y_prob_tensor,
            y_true_tensor,
            output_metrics,
            task=task,
            num_classes=num_classes,
            stage="test",
        )

        # Convert tensor values to float for JSON serialization
        metrics = {k: float(v) if torch.is_tensor(v) else v for k, v in metrics.items()}

        for i in range(n_classes):
            y_true_bin = (y_true == i).astype(int)
            if len(np.unique(y_true_bin)) > 1:
                plot_and_save_roc_curve(
                    y_true_bin, y_prob[:, i], output_metrics_dir, prefix=f"class_{i}_"
                )
                plot_and_save_pr_curve(
                    y_true_bin, y_prob[:, i], output_metrics_dir, prefix=f"class_{i}_"
                )

    save_predictions(ids, y_true, y_prob, id_col, label_col, output_eval_dir)
    save_metrics(metrics, output_metrics_dir)
    logger.info("Evaluation complete")


def create_health_check_file(output_path: str) -> str:
    """Create a health check file to signal script completion."""
    health_path = output_path
    with open(health_path, "w") as f:
        f.write(f"healthy: {datetime.now().isoformat()}")
    return health_path


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
        input_paths (Dict[str, str]): Dictionary of input paths
        output_paths (Dict[str, str]): Dictionary of output paths
        environ_vars (Dict[str, str]): Dictionary of environment variables
        job_args (argparse.Namespace): Command line arguments
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
    id_field = environ_vars.get("ID_FIELD", "id")
    label_field = environ_vars.get("LABEL_FIELD", "label")

    # Log job info
    job_type = job_args.job_type
    logger.info(f"Running PyTorch model evaluation with job_type: {job_type}")

    # Ensure output directories exist
    os.makedirs(output_eval_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)

    logger.info("Starting PyTorch model evaluation script")

    # Load model artifacts
    model, config, tokenizers, pipelines = load_pytorch_model_artifacts(model_dir)

    # Load and preprocess data
    df = load_eval_data(eval_data_dir)

    # Get ID and label columns before preprocessing
    id_col, label_col = get_id_label_columns(df, id_field, label_field)

    # Process the data using PyTorch preprocessing pipelines
    dataset = preprocess_pytorch_eval_data(df, config, pipelines)

    logger.info(f"Final evaluation dataset ready for inference")

    # Evaluate model using the processed dataset
    evaluate_pytorch_model(
        model,
        dataset,
        config,
        id_col,
        label_col,
        output_eval_dir,
        output_metrics_dir,
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

    # Collect environment variables - ID_FIELD and LABEL_FIELD are required per contract
    environ_vars = {
        "ID_FIELD": os.environ.get("ID_FIELD", "id"),  # Fallback for testing
        "LABEL_FIELD": os.environ.get("LABEL_FIELD", "label"),  # Fallback for testing
    }

    try:
        # Call main function with testability parameters
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
