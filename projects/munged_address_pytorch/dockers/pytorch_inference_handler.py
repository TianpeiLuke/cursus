"""
SageMaker Inference Handler for Munged Address Detection (DistilBERT).

Loads model.pth + hyperparameters.json + calibration/percentile_score.pkl
from model.tar.gz, serves single-request real-time predictions via MIMS.

Input:  {"saddr": "123 Main St|||Apt 4|||City ST 12345"}
Output: {"predictions": [{"legacy-score": "0.87", "score-percentile": "0.65"}]}

Artifacts consumed (from PyTorchTraining + PercentileModelCalibration):
  /opt/ml/model/model.pth               — state_dict of AutoModelForSequenceClassification
  /opt/ml/model/hyperparameters.json     — tokenizer name, num_classes, max_sen_len, address_delimiter
  /opt/ml/model/calibration/percentile_score.pkl — List[Tuple[float, float]] from PercentileModelCalibration
"""

import os
import sys
import json
import time
import logging
import hashlib
import fcntl
import pickle as pkl
from typing import List, Tuple, Optional
from subprocess import check_call

import boto3

# ============================================================================
# PACKAGE INSTALLATION CONFIGURATION
# ============================================================================

USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "true").lower() == "true"


def _get_secure_pypi_access_token() -> str:
    """Get CodeArtifact access token for secure PyPI."""
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


def install_packages(packages: list, use_secure: bool = USE_SECURE_PYPI) -> None:
    """Install packages from public or secure PyPI."""
    print(f"Installing {len(packages)} packages from "
          f"{'SECURE (CodeArtifact)' if use_secure else 'PUBLIC'} PyPI")

    if use_secure:
        token = _get_secure_pypi_access_token()
        index_url = (
            f"https://aws:{token}@amazon-149122183214.d.codeartifact."
            f"us-west-2.amazonaws.com/pypi/secure-pypi/simple/"
        )
        check_call([sys.executable, "-m", "pip", "install", "--index-url", index_url, *packages])
    else:
        check_call([sys.executable, "-m", "pip", "install", *packages])

    print("Package installation complete")


# ============================================================================
# INSTALL REQUIRED PACKAGES WITH MULTI-WORKER SAFETY
# ============================================================================

import torch


def install_packages_once(requirements_file: str, use_secure: bool = USE_SECURE_PYPI):
    """
    Thread-safe package installation using file lock to prevent race conditions.
    When multiple TorchServe workers start simultaneously, only ONE worker
    installs packages while others wait.
    """
    import tempfile

    secure_temp_dir = os.environ.get("SM_MODEL_DIR")
    if (
        not secure_temp_dir
        or not os.path.exists(secure_temp_dir)
        or not os.access(secure_temp_dir, os.W_OK)
    ):
        secure_temp_dir = tempfile.gettempdir()

    lock_file = os.path.join(secure_temp_dir, ".pytorch_inference_packages.lock")

    with open(requirements_file, "rb") as f:
        req_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    marker_file = os.path.join(secure_temp_dir, f".packages_installed_{req_hash}.marker")

    if os.path.exists(marker_file):
        print(f"Packages already installed (marker: {marker_file})")
        return

    failure_marker = f"{marker_file}.failed"
    if os.path.exists(failure_marker):
        with open(failure_marker, "r") as f:
            failure_info = f.read()
        raise RuntimeError(f"Package installation previously failed: {failure_info}")

    with open(lock_file, "w") as lock:
        try:
            timeout = 300
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(1)
            else:
                raise TimeoutError(f"Failed to acquire lock after {timeout}s")

            if os.path.exists(marker_file):
                return

            with open(requirements_file, "r") as f:
                required_packages = [
                    line.strip() for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]

            try:
                install_packages(required_packages, use_secure)
                with open(marker_file, "w") as marker:
                    marker.write(f"Installed at {time.time()}\n")
            except Exception as e:
                with open(failure_marker, "w") as marker:
                    marker.write(f"Failed at {time.time()}: {str(e)}\n")
                raise

        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


# Select requirements file based on CUDA availability
if torch.cuda.is_available():
    _requirements_file = os.path.join(os.path.dirname(__file__), "requirements-gpu-secure.txt")
else:
    _requirements_file = os.path.join(os.path.dirname(__file__), "requirements-secure.txt")

try:
    install_packages_once(_requirements_file, USE_SECURE_PYPI)
except FileNotFoundError:
    print(f"Warning: {_requirements_file} not found. Skipping installation.")
except Exception as e:
    print(f"Error installing packages: {e}")
    raise

# ============================================================================
# IMPORTS (after installation)
# ============================================================================

import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cpu")

CALIBRATION_DIR = "calibration"
PERCENTILE_SCORE_FILE = "percentile_score.pkl"


# ============================================================================
# CALIBRATION UTILITIES
# ============================================================================


def _interpolate_score(raw_score: float, lookup_table: List[Tuple[float, float]]) -> float:
    """Linear interpolation on percentile lookup table."""
    if raw_score <= lookup_table[0][0]:
        return lookup_table[0][1]
    if raw_score >= lookup_table[-1][0]:
        return lookup_table[-1][1]
    for i in range(len(lookup_table) - 1):
        if lookup_table[i][0] <= raw_score <= lookup_table[i + 1][0]:
            x1, y1 = lookup_table[i]
            x2, y2 = lookup_table[i + 1]
            if x2 == x1:
                return y1
            return y1 + (y2 - y1) * (raw_score - x1) / (x2 - x1)
    return lookup_table[-1][1]


# ============================================================================
# SAGEMAKER HANDLER FUNCTIONS
# ============================================================================


def model_fn(model_dir):
    """Load model, tokenizer, config, and calibration from model.tar.gz."""
    logger.info(f"Loading model from {model_dir}")

    hparam_path = os.path.join(model_dir, "hyperparameters.json")
    with open(hparam_path) as f:
        config = json.load(f)

    tokenizer_name = config["tokenizer"]
    num_classes = config["num_classes"]
    max_sen_len = config.get("max_sen_len", 128)
    address_delimiter = config.get("address_delimiter", "|||")

    logger.info(f"Config: tokenizer={tokenizer_name}, classes={num_classes}, "
                f"max_len={max_sen_len}, delimiter='{address_delimiter}'")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_name, num_labels=num_classes
    )
    model_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded and set to eval mode")

    calibrator = None
    percentile_path = os.path.join(model_dir, CALIBRATION_DIR, PERCENTILE_SCORE_FILE)
    if os.path.exists(percentile_path):
        with open(percentile_path, "rb") as f:
            calibrator = pkl.load(f)
        logger.info(f"Loaded percentile calibration ({len(calibrator)} entries)")
    else:
        logger.warning("No calibration found — score-percentile will equal legacy-score")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "max_sen_len": max_sen_len,
        "address_delimiter": address_delimiter,
        "calibrator": calibrator,
    }


def input_fn(input_data, content_type):
    """Parse JSON request: {"saddr": "..."}."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")

    try:
        input_dict = json.loads(input_data)
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON: {input_data[:200]}")
        return {"inputs": ""}

    if not isinstance(input_dict, dict):
        logger.warning(f"Expected dict, got {type(input_dict)}")
        return {"inputs": ""}

    saddr = input_dict.get("saddr", "")
    if saddr is None:
        saddr = ""

    return {"inputs": str(saddr)}


def predict_fn(data, model_data):
    """Tokenize address, run model, apply calibration."""
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    max_sen_len = model_data["max_sen_len"]
    address_delimiter = model_data["address_delimiter"]
    calibrator = model_data["calibrator"]

    raw_text = data.get("inputs", "")
    text = str(raw_text).split(address_delimiter)[0].strip()

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_sen_len,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )

    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    raw_score = float(probs[1])

    if calibrator:
        percentile_score = _interpolate_score(raw_score, calibrator)
    else:
        percentile_score = raw_score

    return {"raw_score": raw_score, "percentile_score": percentile_score}


def output_fn(prediction, accept):
    """Format response as JSON matching MIMS registration contract."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")

    raw_score = prediction["raw_score"]
    percentile_score = prediction["percentile_score"]

    instances = [
        {
            "legacy-score": float(raw_score),
            "score-percentile": float(percentile_score),
        }
    ]

    return {"predictions": instances}
