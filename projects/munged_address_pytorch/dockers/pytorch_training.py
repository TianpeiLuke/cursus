#!/usr/bin/env python3
"""
Simplified PyTorch Training Script for Munged Address Detection.

Text-only DistilBERT binary classifier. No multimodal, no risk tables,
no imputation, no streaming. ~400 lines vs generic 1631 lines.

Input:  /opt/ml/input/data/{train,val,test}/{split}_processed_data.csv
        Columns: shippingAddress, __tag__, orderDate, marketplaceId
Output: /opt/ml/model/ (model.pth, model.onnx, tokenizer/, hyperparameters.json)
        /opt/ml/output/data/ (predictions, metrics, tensorboard)
"""

import os
import sys
import json
import logging
import argparse
import traceback
from typing import Dict

from subprocess import check_call

# ============================================================================
# PACKAGE INSTALLATION
# ============================================================================

USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "false").lower() == "true"
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_secure_pypi_access_token() -> str:
    """Get CodeArtifact access token for secure PyPI."""
    import boto3

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
    return token


def install_packages():
    """Install packages from requirements-secure.txt via public or secure PyPI."""
    requirements_file = os.path.join(
        os.path.dirname(__file__), "requirements-secure.txt"
    )
    if not os.path.exists(requirements_file):
        logger.warning(f"{requirements_file} not found, skipping installation")
        return
    with open(requirements_file) as f:
        packages = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    logger.info(f"Installing {len(packages)} packages")
    logger.info(
        f"PyPI Source: {'SECURE (CodeArtifact)' if USE_SECURE_PYPI else 'PUBLIC'}"
    )

    if USE_SECURE_PYPI:
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
    else:
        check_call([sys.executable, "-m", "pip", "install", *packages])

    logger.info("Package installation complete")


# Only rank 0 installs packages (avoids pip lock conflicts in multi-GPU DDP)
if LOCAL_RANK == 0:
    install_packages()
# Other ranks wait for installation via a simple file-based sync
if LOCAL_RANK != 0:
    import time

    while not os.path.exists("/tmp/_install_done"):
        time.sleep(1)
else:
    open("/tmp/_install_done", "w").close()

# ============================================================================
# IMPORTS (after installation)
# ============================================================================

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


# ============================================================================
# DATASET
# ============================================================================


class TextDataset(torch.utils.data.Dataset):
    """Simple text dataset for BERT classification."""

    def __init__(
        self, df: pd.DataFrame, tokenizer, text_col: str, label_col: str, max_len: int
    ):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================================
# LIGHTNING MODULE
# ============================================================================


class MungedAddressBERT(pl.LightningModule):
    """DistilBERT binary classifier for munged address detection."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config["tokenizer"], num_labels=config["num_classes"]
        )
        weights = config.get("class_weights", [1.0, 1.0])
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float))
        self.loss_fn = None  # initialized in on_fit_start (needs correct device)
        self.val_preds = []
        self.val_labels = []
        self.save_hyperparameters()

    def on_fit_start(self):
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def _shared_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
        loss = self.loss_fn(logits, batch["labels"])
        preds = torch.softmax(logits, dim=1)[:, 1]
        return loss, preds, batch["labels"]

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.val_preds.extend(preds.detach().cpu().tolist())
        self.val_labels.extend(labels.detach().cpu().tolist())

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return
        # Gather predictions from all ranks (DDP splits val data across GPUs)
        # Use all_gather_object for variable-length lists (ranks may have different counts)
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_preds = [None] * world_size
            gathered_labels = [None] * world_size
            torch.distributed.all_gather_object(gathered_preds, self.val_preds)
            torch.distributed.all_gather_object(gathered_labels, self.val_labels)
            preds = np.array(sum(gathered_preds, []))
            labels = np.array(sum(gathered_labels, []))
        else:
            preds = np.array(self.val_preds)
            labels = np.array(self.val_labels)
        auc = roc_auc_score(labels, preds)
        f1 = f1_score(labels, (preds > 0.5).astype(int), average="weighted")
        self.log("val/auroc", auc, prog_bar=True, rank_zero_only=True)
        self.log("val/f1_score", f1, prog_bar=True, rank_zero_only=True)
        self.val_preds.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.get("weight_decay", 0.01),
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(params, lr=self.config["lr"])
        total_steps = self.trainer.estimated_stepping_batches
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.get("warmup_steps", 0),
            num_training_steps=total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


# ============================================================================
# DATA LOADING
# ============================================================================


def find_data_file(data_dir: str) -> str:
    """Find first CSV/Parquet file in directory."""
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith((".csv", ".tsv", ".parquet")):
            return os.path.join(data_dir, fname)
    raise FileNotFoundError(f"No data file found in {data_dir}")


def load_split(path: str) -> pd.DataFrame:
    """Load a single split file."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ============================================================================
# INFERENCE + METRICS
# ============================================================================


def predict(model, dataloader, device):
    """Run inference and return predictions + labels."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
    return np.array(all_preds), np.array(all_labels)


# ============================================================================
# MAIN
# ============================================================================


def log_rank0(msg):
    """Log only on rank 0 (avoids duplicate logs in multi-GPU DDP)."""
    if LOCAL_RANK == 0:
        logger.info(msg)


def main(input_paths: Dict, output_paths: Dict, environ_vars: Dict, job_args):
    """Main training logic."""
    log_rank0("=" * 70)
    log_rank0("PYTORCH TRAINING — Munged Address Detection (DistilBERT)")
    log_rank0("=" * 70)

    region = environ_vars.get("REGION", "NA")
    hparam_dir = input_paths.get("hyperparameters_s3_uri", "/opt/ml/code/hyperparams")
    hparam_file = os.path.join(hparam_dir, f"hyperparameters_{region}.json")
    if not os.path.exists(hparam_file):
        hparam_file = os.path.join(hparam_dir, "hyperparameters.json")

    with open(hparam_file) as f:
        config = json.load(f)
    log_rank0(f"Loaded hyperparameters from {hparam_file}")
    log_rank0(f"Region: {region}")
    log_rank0(f"Model: {config.get('tokenizer')}, classes: {config.get('num_classes')}")
    log_rank0(f"Training: epochs={config.get('max_epochs')}, lr={config.get('lr')}, "
              f"batch={config.get('batch_size')}, fp16={config.get('fp16')}")

    # Paths
    data_root = input_paths["input_path"]
    model_dir = output_paths.get("model_output", "/opt/ml/model")
    output_dir = output_paths.get("evaluation_output", "/opt/ml/output/data")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    log_rank0(f"Data root: {data_root}")
    log_rank0(f"Model dir: {model_dir}")
    log_rank0(f"Output dir: {output_dir}")

    # Load data
    text_col = config["text_name"]
    label_col = config["label_name"]
    max_len = config.get("max_sen_len", 128)

    train_df = load_split(find_data_file(os.path.join(data_root, "train")))
    val_df = load_split(find_data_file(os.path.join(data_root, "val")))
    test_df = load_split(find_data_file(os.path.join(data_root, "test")))

    log_rank0(f"Data loaded — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    log_rank0(f"Columns: {list(train_df.columns)}")
    log_rank0(f"Label distribution (train): {train_df[label_col].value_counts().to_dict()}")

    # Tokenizer (rank 0 downloads, others use cache)
    log_rank0(f"Loading tokenizer: {config['tokenizer']}")
    if LOCAL_RANK == 0:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if LOCAL_RANK != 0:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    log_rank0(f"Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # Datasets
    train_dataset = TextDataset(train_df, tokenizer, text_col, label_col, max_len)
    val_dataset = TextDataset(val_df, tokenizer, text_col, label_col, max_len)
    test_dataset = TextDataset(test_df, tokenizer, text_col, label_col, max_len)

    batch_size = config.get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, num_workers=0)
    log_rank0(f"DataLoaders: batch_size={batch_size}, "
              f"train={len(train_loader)} batches, val={len(val_loader)}, test={len(test_loader)}")

    # Model
    model = MungedAddressBERT(config)
    log_rank0(f"Model initialized: MungedAddressBERT (class_weights={config.get('class_weights', [1.0, 1.0])})")

    # Trainer
    callbacks = [
        EarlyStopping(
            monitor=config.get("early_stop_metric", "val/f1_score"),
            patience=config.get("early_stop_patience", 2),
            mode="max",
        ),
        ModelCheckpoint(monitor="val/f1_score", mode="max", save_top_k=1),
    ]

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log_rank0("=" * 70)
    log_rank0(f"Trainer setup: {gpu_count} GPU(s), accelerator=auto, devices=auto")
    log_rank0(f"Precision: {'16-mixed' if config.get('fp16', True) else '32'}, "
              f"gradient_clip={config.get('gradient_clip_val', 1.0)}")
    log_rank0(f"Early stopping: metric={config.get('early_stop_metric', 'val/f1_score')}, "
              f"patience={config.get('early_stop_patience', 2)}")
    if gpu_count > 1:
        log_rank0(f"Multi-GPU DDP: {gpu_count} GPUs (torchrun-managed)")
    elif gpu_count == 1:
        log_rank0("Single GPU training")
    else:
        log_rank0("CPU training")
    log_rank0("=" * 70)

    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 2),
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if config.get("fp16", True) else 32,
        gradient_clip_val=config.get("gradient_clip_val", 1.0),
        val_check_interval=config.get("val_check_interval", 1.0),
        callbacks=callbacks,
        default_root_dir=output_dir,
        enable_progress_bar=True,
    )

    # Train
    log_rank0("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    log_rank0(f"Training complete — best model: {trainer.checkpoint_callback.best_model_path}")
    log_rank0(f"Best val/f1_score: {trainer.checkpoint_callback.best_model_score}")

    # Synchronize all ranks after training (required for DDP multi-GPU)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        logger.info("All ranks synchronized after training")

    # Determine if this is the main process (rank 0)
    is_main = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    # Evaluate + save on main process only (avoids file corruption from multiple ranks)
    if is_main:
        logger.info("=" * 70)
        logger.info("EVALUATION & ARTIFACT SAVE (rank 0 only)")
        logger.info("=" * 70)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Running inference on val ({len(val_df)}) and test ({len(test_df)}) sets...")
        val_preds, val_labels = predict(model, val_loader, device)
        test_preds, test_labels = predict(model, test_loader, device)

        val_auc = roc_auc_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, (val_preds > 0.5).astype(int), average="weighted")
        test_auc = roc_auc_score(test_labels, test_preds)
        test_f1 = f1_score(
            test_labels, (test_preds > 0.5).astype(int), average="weighted"
        )
        test_acc = accuracy_score(test_labels, (test_preds > 0.5).astype(int))

        logger.info("-" * 40)
        logger.info(f"Val  — AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
        logger.info(f"Test — AUC: {test_auc:.4f}, F1: {test_f1:.4f}, Acc: {test_acc:.4f}")
        logger.info("-" * 40)

        # ================================================================
        # SAVE OUTPUT ARTIFACTS (must match downstream step expectations)
        # ================================================================

        # --- /opt/ml/output/data/ (evaluation_output) ---
        logger.info(f"Saving evaluation artifacts to {output_dir}")

        metrics = {
            "val_auroc": val_auc,
            "val_f1": val_f1,
            "test_auroc": test_auc,
            "test_f1": test_f1,
            "test_accuracy": test_acc,
        }
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        test_df_out = test_df.copy()
        test_df_out["prob_class_0"] = 1.0 - test_preds
        test_df_out["prob_class_1"] = test_preds
        test_df_out.to_csv(
            os.path.join(output_dir, "test_predictions.csv"), index=False
        )

        val_df_out = val_df.copy()
        val_df_out["prob_class_0"] = 1.0 - val_preds
        val_df_out["prob_class_1"] = val_preds
        val_df_out.to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)

        predict_results = {
            "test_true_labels": torch.tensor(test_labels),
            "test_predict_labels": torch.tensor(test_preds),
        }
        torch.save(predict_results, os.path.join(output_dir, "predict_results.pth"))

        tb_dir = os.path.join(output_dir, "tensorboard_eval")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_scalar("test/auroc", test_auc, 0)
        writer.add_scalar("test/f1_score", test_f1, 0)
        writer.add_scalar("val/auroc", val_auc, 0)
        writer.add_scalar("val/f1_score", val_f1, 0)
        writer.close()

        # --- /opt/ml/model/ (model_output) ---
        logger.info(f"Saving model artifacts to {model_dir}")

        model_path = os.path.join(model_dir, "model.pth")
        torch.save(model.model.state_dict(), model_path)
        logger.info(f"  model.pth ({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")

        artifacts = {
            "config": config,
            "vocab": tokenizer.get_vocab(),
            "model_class": "bert",
        }
        torch.save(artifacts, os.path.join(model_dir, "model_artifacts.pth"))
        logger.info("  model_artifacts.pth")

        with open(os.path.join(model_dir, "hyperparameters.json"), "w") as f:
            json.dump(config, f, indent=2)
        logger.info("  hyperparameters.json")

        with open(os.path.join(model_dir, "feature_columns.txt"), "w") as f:
            f.write("# Feature columns in exact order required for model inference\n")
            f.write("# DO NOT MODIFY THE ORDER OF THESE COLUMNS\n")
            f.write("# Each line contains: <column_index>,<column_name>\n")
        logger.info("  feature_columns.txt")

        # ONNX export (must unwrap DDP and move to CPU for clean export)
        try:
            # Unwrap: in DDP, model may be wrapped — get the raw HF model
            hf_model = model.model
            if hasattr(hf_model, "module"):
                hf_model = hf_model.module  # unwrap DDP/FSDP wrapper
            hf_model = hf_model.cpu().eval()

            dummy_input_ids = torch.randint(0, 1000, (1, max_len))
            dummy_attention = torch.ones(1, max_len, dtype=torch.long)
            onnx_path = os.path.join(model_dir, "model.onnx")
            torch.onnx.export(
                hf_model,
                (dummy_input_ids, dummy_attention),
                onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq_len"},
                    "attention_mask": {0: "batch", 1: "seq_len"},
                },
                opset_version=14,
            )
            logger.info(f"  model.onnx ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")
        except Exception as e:
            logger.warning(f"ONNX export failed (non-fatal): {e}")

        logger.info("=" * 70)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 70)

    # Final barrier to ensure all ranks exit together
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    input_paths = {
        "input_path": "/opt/ml/input/data",
        "hyperparameters_s3_uri": "/opt/ml/code/hyperparams",
    }
    output_paths = {
        "model_output": "/opt/ml/model",
        "evaluation_output": "/opt/ml/output/data",
    }
    environ_vars = {
        "REGION": os.environ.get("REGION", "NA"),
    }
    args = argparse.Namespace()

    try:
        main(input_paths, output_paths, environ_vars, args)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        failure_file = os.path.join("/opt/ml/output/data", "failure")
        with open(failure_file, "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())
        sys.exit(1)
