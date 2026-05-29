#!/usr/bin/env python3
"""
Simplified PyTorch Model Inference for Munged Address Calibration.

Uses Lightning Trainer.test() with ddp_spawn for proper multi-GPU inference.
Each rank saves predictions to file, parent process merges and saves final CSV.

Input:  model_input/ (model.pth + hyperparameters.json)
        processed_data/ (calibration data with shippingAddress column)
Output: eval_output/predictions.csv (with prob_class_0, prob_class_1)
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PACKAGE INSTALLATION (runs ONCE in parent process before DDP spawn)
# ============================================================================
from subprocess import check_call

USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "true").lower() == "true"


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
    return code_artifact_client.get_authorization_token(
        domain="amazon", domainOwner="149122183214"
    )["authorizationToken"]


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

    logger.info(
        f"Installing {len(packages)} packages from "
        f"{'SECURE' if USE_SECURE_PYPI else 'PUBLIC'} PyPI"
    )

    if USE_SECURE_PYPI:
        token = _get_secure_pypi_access_token()
        index_url = (
            f"https://aws:{token}@amazon-149122183214.d.codeartifact."
            f"us-west-2.amazonaws.com/pypi/secure-pypi/simple/"
        )
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


# Install ONCE in parent process (ddp_spawn children inherit installed packages)
install_packages()

# ============================================================================
# IMPORTS (after installation)
# ============================================================================

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lightning.pytorch as pl


# ============================================================================
# DATASET
# ============================================================================


class InferenceDataset(Dataset):
    """Simple text dataset for inference (no labels). Stores index for ordering."""

    def __init__(self, texts: List[str], tokenizer, max_len: int):
        self.texts = texts
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
            "index": idx,  # preserve original index for ordering after DDP gather
        }


# ============================================================================
# LIGHTNING MODULE (inference via test_step + file-based gather)
# ============================================================================


class InferenceModel(pl.LightningModule):
    """Wraps HuggingFace model for Lightning multi-GPU inference via test()."""

    def __init__(self, model, output_dir: str):
        super().__init__()
        self.model = model
        self.output_dir = output_dir
        self.predictions = []
        self.indices = []

    def test_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]
        self.predictions.append(probs.cpu())
        self.indices.append(batch["index"].cpu())

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.predictions).numpy()
        all_indices = torch.cat(self.indices).numpy()
        rank = self.global_rank
        world = self.trainer.world_size
        out_path = os.path.join(self.output_dir, f"_preds_rank{rank}.npz")
        np.savez(out_path, predictions=all_preds, indices=all_indices)
        logger.info(
            f"[Rank {rank}/{world}] Saved {len(all_preds)} predictions "
            f"(indices {all_indices.min()}-{all_indices.max()}) to {out_path}"
        )
        self.predictions.clear()
        self.indices.clear()


# ============================================================================
# HELPERS
# ============================================================================


def find_data_file(data_dir: str) -> str:
    """Find first CSV/Parquet file in directory (checks root then subdirectories)."""
    from pathlib import Path

    root = Path(data_dir)
    # Check root level first
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith((".csv", ".tsv", ".parquet")):
            return os.path.join(data_dir, fname)
    # Fallback: recursive search (handles {job_type}/{job_type}_processed_data.{ext} convention)
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix in (".csv", ".tsv", ".parquet"):
            return str(f)
    raise FileNotFoundError(f"No data file found in {data_dir}")


def load_data(data_dir: str) -> pd.DataFrame:
    """Load data from directory (finds first supported file)."""
    path = find_data_file(data_dir)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ============================================================================
# MAIN
# ============================================================================


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
):
    """Run batch inference on calibration data using Lightning Trainer."""
    logger.info("=" * 70)
    logger.info("PYTORCH MODEL INFERENCE — Munged Address Calibration")
    logger.info("=" * 70)

    model_dir = input_paths.get("model_input")
    data_dir = input_paths.get("processed_data")
    output_dir = output_paths.get("eval_output")
    os.makedirs(output_dir, exist_ok=True)

    batch_size = int(environ_vars.get("BATCH_SIZE", "64"))
    id_field = environ_vars.get("ID_FIELD", "")
    label_field = environ_vars.get("LABEL_FIELD", "")

    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"ID field: {id_field or '(none)'}")
    logger.info(f"Label field: {label_field or '(none)'}")

    # Load hyperparameters (parent process — before spawn)
    hparam_path = os.path.join(model_dir, "hyperparameters.json")
    with open(hparam_path) as f:
        config = json.load(f)
    logger.info(
        f"Config: tokenizer={config['tokenizer']}, "
        f"text_name={config['text_name']}, "
        f"num_classes={config['num_classes']}, "
        f"max_sen_len={config.get('max_sen_len', 128)}"
    )

    # Reconstruct model (parent process — children inherit via fork)
    logger.info(f"Loading model from {model_dir}/model.pth...")
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        config["tokenizer"], num_labels=config["num_classes"]
    )
    model_path = os.path.join(model_dir, "model.pth")
    hf_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    hf_model.eval()
    logger.info("Model loaded and set to eval mode")

    # Load tokenizer (parent process)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    logger.info(f"Tokenizer loaded: {config['tokenizer']}")

    # Load calibration data (parent process)
    logger.info(f"Loading calibration data from {data_dir}...")
    df = load_data(data_dir)
    text_col = config["text_name"]
    max_len = config.get("max_sen_len", 128)
    address_col = config.get("address_column", "saddr")
    address_delim = config.get("address_delimiter", "|||")

    logger.info(f"Data shape: {df.shape}, columns: {list(df.columns)}")

    # If text_col not present but address_col is, extract it
    if text_col not in df.columns and address_col in df.columns:
        logger.info(
            f"Column '{text_col}' not found — extracting from "
            f"'{address_col}' (split on '{address_delim}')"
        )
        df[text_col] = (
            df[address_col].astype(str).str.split(address_delim).str[0].str.strip()
        )

    texts = df[text_col].astype(str).tolist()
    logger.info(f"Prepared {len(texts)} text samples for inference")

    # Create dataset and dataloader (shuffle=False — critical for index ordering)
    # num_workers=0: avoids nested multiprocessing issues with ddp_spawn
    dataset = InferenceDataset(texts, tokenizer, max_len)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )
    logger.info(
        f"DataLoader: {len(dataloader)} batches, "
        f"batch_size={batch_size}, num_workers=0, shuffle=False"
    )

    # Wrap in Lightning module
    model = InferenceModel(hf_model, output_dir)

    # Lightning Trainer with strategy="auto":
    # - devices="auto" detects all available GPUs
    # - strategy="auto" selects ddp_spawn when no torchrun env exists
    #   (ProcessingStep = single process, no LOCAL_RANK → Lightning uses ddp_spawn)
    # - ddp_spawn forks child processes from parent (model + data inherited)
    # - DistributedSampler added automatically (shuffle=False preserved)
    # - Children write per-rank files, parent merges after
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
    )

    logger.info("=" * 70)
    logger.info(f"Starting inference: {gpu_count} GPU(s), strategy=auto")
    if gpu_count > 1:
        logger.info(
            f"Multi-GPU: Lightning will use ddp_spawn ({gpu_count} child processes)"
        )
        logger.info("Note: each child process logs independently with 'Rank N:' prefix")
        logger.info(
            "Parent process will resume after all children complete and merge results"
        )
    elif gpu_count == 1:
        logger.info("Single GPU inference (no spawn)")
    else:
        logger.info("CPU inference (no spawn)")
    logger.info("=" * 70)

    trainer.test(model, dataloaders=dataloader)

    # After ddp_spawn: children have exited, parent resumes here
    logger.info("=" * 70)
    logger.info("PARENT PROCESS: all spawn children returned — merging predictions")
    logger.info("=" * 70)

    # After ddp_spawn returns to parent — merge per-rank prediction files
    from pathlib import Path

    pred_files = sorted(Path(output_dir).glob("_preds_rank*.npz"))
    logger.info(f"Found {len(pred_files)} rank prediction file(s)")

    # Load all rank predictions with their original indices
    all_indices = []
    all_preds = []
    for f in pred_files:
        data = np.load(f)
        all_indices.append(data["indices"])
        all_preds.append(data["predictions"])
        logger.info(f"  {f.name}: {len(data['predictions'])} predictions merged")
        f.unlink()  # cleanup temp file

    # Reconstruct original order using saved indices
    all_indices = np.concatenate(all_indices)
    all_preds = np.concatenate(all_preds)
    all_probs = np.empty(len(df))
    all_probs[all_indices] = all_preds

    logger.info(f"Generated {len(all_probs)} predictions")
    logger.info(
        f"Score range: min={all_probs.min():.4f}, max={all_probs.max():.4f}, "
        f"mean={all_probs.mean():.4f}"
    )

    # Save predictions (format consumed by PercentileModelCalibration)
    df["prob_class_0"] = 1.0 - all_probs
    df["prob_class_1"] = all_probs
    output_path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} predictions to {output_path}")


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, default="calibration")
    args = parser.parse_args()

    input_paths = {
        "model_input": "/opt/ml/processing/input/model",
        "processed_data": "/opt/ml/processing/input/eval_data",
    }
    output_paths = {
        "eval_output": "/opt/ml/processing/output/eval",
    }
    environ_vars = {
        "BATCH_SIZE": os.environ.get("BATCH_SIZE", "64"),
        "ID_FIELD": os.environ.get("ID_FIELD", ""),
        "LABEL_FIELD": os.environ.get("LABEL_FIELD", ""),
    }

    try:
        main(input_paths, output_paths, environ_vars, args)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
