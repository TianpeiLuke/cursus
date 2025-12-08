# Save this as: bsm/lightning_models/train_utils.py
import os
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import fcntl  # Unix file locking for multi-worker coordination
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor,
    DeviceStatsMonitor,
)

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy


from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType


import onnx
import onnxruntime as ort

# Note: Model imports moved inside functions to avoid circular dependencies


def setup_logger():
    import logging

    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


# ============================================================================
# FILE LOCKING FOR MULTI-WORKER COORDINATION
# ============================================================================


@contextmanager
def file_lock(lock_path: Path, timeout: int = 60):
    """
    Atomic file lock for multi-worker coordination in TorchServe environments.

    Prevents race conditions when multiple workers try to perform the same
    expensive operation (e.g., BERT model optimization). Only one worker
    acquires the lock and performs the operation; others wait for completion.

    Args:
        lock_path: Path to lock file (will be created/removed automatically)
        timeout: Maximum seconds to wait for lock acquisition (default: 60s)

    Yields:
        None when lock is successfully acquired

    Raises:
        TimeoutError: If lock cannot be acquired within timeout period

    Example:
        >>> lock_file = Path("/opt/ml/model/.optimization.lock")
        >>> with file_lock(lock_file, timeout=90):
        >>>     # Only one worker executes this block
        >>>     optimize_model()
        >>> # Lock automatically released, other workers proceed
    """
    lock_file = None
    try:
        # Create lock file
        lock_file = open(lock_path, "w")
        start_time = time.time()

        # Try to acquire exclusive lock with timeout
        while True:
            try:
                # LOCK_EX: Exclusive lock (only one process can hold it)
                # LOCK_NB: Non-blocking (return immediately if lock unavailable)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.info(f"✓ Acquired file lock: {lock_path}")
                break
            except IOError:
                # Lock is held by another process
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Could not acquire lock within {timeout}s. "
                        f"Another worker may be stuck or operation is taking too long."
                    )
                # Wait and retry
                time.sleep(1)
                if int(elapsed) % 10 == 0:  # Log every 10 seconds
                    logger.info(
                        f"Waiting for lock... ({int(elapsed)}s elapsed, max {timeout}s)"
                    )

        yield

    finally:
        # Always release lock and clean up
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                if lock_path.exists():
                    lock_path.unlink()
                logger.info(f"✓ Released file lock: {lock_path}")
            except Exception as e:
                logger.warning(f"Error releasing lock {lock_path}: {e}")


# ----------------- FDSP ---------------------
def my_auto_wrap_policy(
    module: nn.Module, recurse: bool, unwrapped_params: int, min_num_params: int = 1e5
) -> bool:
    """
    Custom FSDP auto wrap policy for multimodal models.

    This policy wraps:
    - TextBertBase (Transformer-based encoder)
    - TabAE (tabular encoder)
    - Any Linear / Conv2d / Embedding with large parameter counts

    Args:
        module (nn.Module): Module to inspect
        recurse (bool): Whether FSDP is recursing
        unwrapped_params (int): Number of unwrapped parameters
        min_num_params (int): Minimum number of params to wrap

    Returns:
        bool: Whether to wrap this module
    """
    # Lazy import to avoid circular dependency
    from ..text.pl_bert import TextBertBase
    from ..tabular.pl_tab_ae import TabAE

    return (
        isinstance(module, (TextBertBase, TabAE, nn.Linear, nn.Embedding, nn.Conv2d))
        and unwrapped_params >= min_num_params
    )


def is_fsdp_available():
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and dist.is_available()
        and dist.is_initialized()
    )


strategy = (
    FSDPStrategy(auto_wrap_policy=my_auto_wrap_policy, verbose=True)
    if is_fsdp_available()
    else "auto"
)
# -----------------------------


def model_train(
    model: pl.LightningModule,
    config: Dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: Union[int, str, List[int]] = "auto",
    model_log_path: str = "./model_logs",
    early_stop_metric: str = "val/f1_score",
) -> pl.Trainer:
    max_epochs = config.get("max_epochs", 10)
    early_stop_patience = config.get("early_stop_patience", 10)
    model_class = config.get("model_class", "multimodal_cnn")
    val_check_interval = config.get("val_check_interval", 1.0)
    use_fp16 = config.get("fp16", False)
    clip_val = config.get("gradient_clip_val", 0.0)

    logger_tb = TensorBoardLogger(save_dir=model_log_path, name="tensorboard_logs")
    monitor_mode = "min" if "loss" in early_stop_metric else "max"

    checkpoint_dir = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(checkpoint_dir),
        filename=f"{model_class}" + "-{epoch:02d}-{" + f"{early_stop_metric}" + ":.2f}",
        monitor=early_stop_metric,
        save_top_k=1,
        mode=monitor_mode,
        save_weights_only=False,
    )

    earlystopping_callback = EarlyStopping(
        monitor=early_stop_metric, patience=early_stop_patience, mode=monitor_mode
    )

    device_stats_callback = DeviceStatsMonitor(cpu_stats=False)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger_tb,
        default_root_dir=model_log_path,
        callbacks=[
            earlystopping_callback,
            checkpoint_callback,
            device_stats_callback,
            TQDMProgressBar(refresh_rate=10),
            LearningRateMonitor(logging_interval="step"),
        ],
        val_check_interval=config.get("val_check_interval", 1.0),
        sync_batchnorm=True if torch.cuda.is_available() else False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=device,
        strategy=strategy,  # You might need this
        # accumulate_grad_batches=1,
        precision=16 if use_fp16 else 32,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    return trainer


# ------------------ Utility Function -----------------
def extract_preds_and_labels(
    df: pd.DataFrame, is_binary: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    if is_binary:
        preds = torch.tensor(df["prob"].values.astype(float))
    else:
        preds = torch.tensor(
            [ast.literal_eval(p) if isinstance(p, str) else p for p in df["prob"]]
        )
    labels = torch.tensor(df["label"].values)
    return preds, labels


# ------------------ Inference ------------------------
def model_inference(
    model: pl.LightningModule,
    dataloader: DataLoader,
    accelerator: Union[str, int, List[int]] = "auto",
    device: Union[str, int, List[int]] = "auto",
    model_log_path: str = "./model_logs",
    return_dataframe: bool = False,
    label_col: str = "label",
) -> Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]
]:
    """
    Runs inference and returns predicted probabilities and true labels as tensors.
    Supports both binary and multiclass classification.

    CRITICAL: In distributed training, ALL ranks must call this function.
    Only rank 0 will process and return actual results; other ranks return dummy tensors.

    Args:
        model (pl.LightningModule): Trained Lightning model.
        dataloader (DataLoader): DataLoader for inference.
        accelerator (str/int/List[int]): Accelerator setting.
        device (str/int/List[int]): Device setting.
        model_log_path (str): Path to save logs.
        return_dataframe (bool): Whether to return the original dataframe.
        label_col (str): Name of the label column in the dataframe. Defaults to "label".

    Returns:
        Tuple of (y_pred, y_true) or (y_pred, y_true, df) depending on `return_dataframe`.
    """

    # Safe handling: force CPU if no GPU available
    resolved_accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    resolved_devices = 1 if resolved_accelerator == "cpu" else device

    tester = pl.Trainer(
        max_epochs=1,
        default_root_dir=model_log_path,
        enable_checkpointing=False,
        logger=False,
        callbacks=[TQDMProgressBar()],
        accelerator=resolved_accelerator,
        devices=resolved_devices,
        strategy="auto",  # Will use distributed strategy if torch.distributed is initialized
        inference_mode=True,
    )

    # All ranks participate in test (required for distributed collective operations)
    tester.test(model, dataloaders=dataloader)

    # CRITICAL FIX: Synchronize all ranks after test completes
    if dist.is_initialized():
        dist.barrier()

    # Only rank 0 processes results; other ranks return dummy tensors
    if dist.is_initialized() and dist.get_rank() != 0:
        # Non-main ranks return dummy tensors (won't be used by caller)
        dummy_preds = torch.zeros(1)
        dummy_labels = torch.zeros(1, dtype=torch.long)
        if return_dataframe:
            return dummy_preds, dummy_labels, pd.DataFrame()
        return dummy_preds, dummy_labels

    # Main rank (rank 0) processes results
    result_folder = model.test_output_folder
    if not result_folder or not os.path.exists(result_folder):
        raise RuntimeError(
            f"Expected test output folder '{result_folder}' does not exist."
        )

    # Match files like test_result_*.tsv from all ranks
    result_files = sorted(Path(result_folder).glob("test_result_*.tsv"))
    if not result_files:
        raise RuntimeError(f"No test result files found in {result_folder}.")

    dfs = []
    for f in result_files:
        try:
            dfs.append(pd.read_csv(f, sep="\t"))
        except Exception as e:
            print(f"[Warning] Skipping file {f} due to read error: {e}")

    if not dfs:
        raise RuntimeError("No valid result files could be loaded.")
    df = pd.concat(dfs, ignore_index=True)

    is_binary = model.task == "binary"
    if is_binary:
        y_pred = torch.tensor(df["prob"].values.astype(float))
    else:
        y_pred = torch.tensor(
            [ast.literal_eval(p) if isinstance(p, str) else p for p in df["prob"]]
        )

    y_true = torch.tensor(df[label_col].values).long()

    if return_dataframe:
        return y_pred, y_true, df
    else:
        return y_pred, y_true


def model_online_inference(
    model: Union[pl.LightningModule, ort.InferenceSession], dataloader: DataLoader
) -> np.ndarray:
    """
    Run online inference for either a PyTorch Lightning model or an ONNX Runtime session.
    """
    if isinstance(model, ort.InferenceSession):
        print("Running inference with ONNX Runtime.")
        predictions = []
        expected_input_names = [inp.name for inp in model.get_inputs()]

        for batch in dataloader:
            input_feed = {}
            for k in expected_input_names:
                if k not in batch:
                    raise KeyError(f"ONNX input '{k}' not found in batch")

                val = batch[k]

                # Convert to numpy with correct type
                if isinstance(val, torch.Tensor):
                    val_np = val.cpu().numpy()

                    # Ensure correct dtype
                    if "input_ids" in k or "attention_mask" in k:
                        val_np = val_np.astype("int64")  # Required for ONNX
                    else:
                        val_np = val_np.astype("float32")

                    input_feed[k] = val_np

                elif isinstance(val, list) and all(
                    isinstance(x, (int, float)) for x in val
                ):
                    # Fallback for list-based numeric features
                    val_np = np.array(val, dtype="float32").reshape(-1, 1)
                    input_feed[k] = val_np

                else:
                    # Skip fields like order_id (string/list[str]) or raise error
                    print(
                        f"[Warning] Skipping unsupported ONNX input field: '{k}' ({type(val)})"
                    )

            output = model.run(None, input_feed)[0]  # Run inference
            predictions.append(output)

        return np.concatenate(predictions, axis=0)

    else:
        print("Running inference with PyTorch model.")
        model.eval()
        predictions = []
        for batch in dataloader:
            _, preds, _ = model.run_epoch(batch, "pred")
            predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions, axis=0)


def predict_stack_transform(
    outputs: List[Union[torch.Tensor, Tuple[torch.Tensor]]],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(outputs[0], Tuple):
        pred_list, label_list = zip(*outputs)
        return torch.cat(pred_list), torch.cat(label_list)
    return torch.cat(outputs)


def unwrap_fsdp_model(model: nn.Module) -> nn.Module:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    return model.module if isinstance(model, FSDP) else model


def save_prediction(filename: str, y_true: List, y_pred: List):
    logger.info("Saving prediction.")
    torch.save({"y_true": y_true, "y_pred": y_pred}, filename)


def save_model(filename: str, model: nn.Module):
    logger.info("Saving model weights.")

    # Unwrap if wrapped in FSDP
    if isinstance(model, FSDP):
        # Use FSDP's full state dict context
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()
            if dist.get_rank() == 0:
                torch.save(state_dict, filename)
    else:
        torch.save(model.state_dict(), filename)


def save_artifacts(
    filename: str,
    config: Dict,
    embedding_mat: torch.Tensor,
    vocab: Dict[str, int],
    model_class: str,
):
    logger.info("Saving artifacts.")
    artifacts = {
        "config": config,
        "embedding_mat": embedding_mat,
        "vocab": vocab,
        "model_class": model_class,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "pytorch_lightning_version": __import__("lightning.pytorch").__version__,
    }
    torch.save(artifacts, filename)


def load_artifacts(
    filename: str, device_l: str = "cpu"
) -> Tuple[Dict, torch.Tensor, Dict, str]:
    logger.info("Loading artifacts.")
    artifacts = torch.load(filename, map_location=device_l)
    config = artifacts["config"]
    embedding_mat = artifacts["embedding_mat"]
    vocab = artifacts["vocab"]
    model_class = artifacts["model_class"]
    for k in ["torch_version", "transformers_version", "pytorch_lightning_version"]:
        logger.info(f"{k}: {artifacts.get(k, 'N/A')}")
    return config, embedding_mat, vocab, model_class


def load_model(
    filename: str,
    config: Dict,
    embedding_mat: torch.Tensor,
    model_class: str = "bimodal_bert",
    device_l: str = "cpu",
) -> nn.Module:
    """
    Load model weights into a fresh model instance.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    # Lazy imports to avoid circular dependencies
    from ..bimodal.pl_bimodal_cnn import BimodalCNN
    from ..text.pl_bert_classification import TextBertClassification
    from ..text.pl_lstm import TextLSTM
    from ..bimodal.pl_bimodal_bert import BimodalBert
    from ..bimodal.pl_bimodal_gate_fusion import BimodalBertGateFusion
    from ..bimodal.pl_bimodal_moe import BimodalBertMoE
    from ..bimodal.pl_bimodal_cross_attn import BimodalBertCrossAttn
    from ..trimodal.pl_trimodal_bert import TrimodalBert
    from ..trimodal.pl_trimodal_cross_attn import TrimodalCrossAttentionBert
    from ..trimodal.pl_trimodal_gate_fusion import TrimodalGateFusionBert

    logger.info("Instantiating model.")
    model = {
        # Bimodal models
        "bimodal": lambda: BimodalBert(config),  # Default bimodal
        "bimodal_cnn": lambda: BimodalCNN(
            config, embedding_mat.shape[0], embedding_mat
        ),
        "bimodal_bert": lambda: BimodalBert(config),
        "bimodal_gate_fusion": lambda: BimodalBertGateFusion(config),
        "bimodal_moe": lambda: BimodalBertMoE(config),
        "bimodal_cross_attn": lambda: BimodalBertCrossAttn(config),
        # Trimodal models
        "trimodal": lambda: TrimodalBert(config),  # Default trimodal
        "trimodal_bert": lambda: TrimodalBert(config),
        "trimodal_cross_attn_bert": lambda: TrimodalCrossAttentionBert(config),
        "trimodal_gate_fusion_bert": lambda: TrimodalGateFusionBert(config),
        # Text-only models
        "bert": lambda: TextBertClassification(config),
        "lstm": lambda: TextLSTM(config, embedding_mat.shape[0], embedding_mat),
        # Backward compatibility (multimodal -> bimodal)
        "multimodal_cnn": lambda: BimodalCNN(
            config, embedding_mat.shape[0], embedding_mat
        ),
        "multimodal_bert": lambda: BimodalBert(config),
        "multimodal_gate_fusion": lambda: BimodalBertGateFusion(config),
        "multimodal_moe": lambda: BimodalBertMoE(config),
        "multimodal_cross_attn": lambda: BimodalBertCrossAttn(config),
    }.get(model_class, lambda: BimodalBert(config))()

    try:
        logger.info(f"Loading model weights from: {filename}")
        model.load_state_dict(torch.load(filename, map_location=device_l))
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        raise RuntimeError("Model loading failed.") from e

    return model


def load_checkpoint(
    filename: str, model_class: str = "bimodal_bert", device_l: str = "cpu"
) -> nn.Module:
    # Lazy imports to avoid circular dependencies
    from ..bimodal.pl_bimodal_cnn import BimodalCNN
    from ..text.pl_bert_classification import TextBertClassification
    from ..text.pl_lstm import TextLSTM
    from ..bimodal.pl_bimodal_bert import BimodalBert
    from ..bimodal.pl_bimodal_gate_fusion import BimodalBertGateFusion
    from ..bimodal.pl_bimodal_moe import BimodalBertMoE
    from ..bimodal.pl_bimodal_cross_attn import BimodalBertCrossAttn

    logger.info("Loading checkpoint.")
    model_fn = {
        "bimodal_cnn": BimodalCNN,
        "bert": TextBertClassification,
        "lstm": TextLSTM,
        "bimodal_bert": BimodalBert,
        "bimodal_gate_fusion": BimodalBertGateFusion,
        "bimodal_moe": BimodalBertMoE,
        "bimodal_cross_attn": BimodalBertCrossAttn,
        # Backward compatibility
        "multimodal_cnn": BimodalCNN,
        "multimodal_bert": BimodalBert,
        "multimodal_gate_fusion": BimodalBertGateFusion,
        "multimodal_moe": BimodalBertMoE,
        "multimodal_cross_attn": BimodalBertCrossAttn,
    }.get(model_class, BimodalBert)
    return model_fn.load_from_checkpoint(filename, map_location=device_l)


def load_onnx_model(
    onnx_path: Union[str, Path],
    enable_profiling: bool = False,
    inter_op_threads: int = 1,
    intra_op_threads: int = 4,
) -> ort.InferenceSession:
    """
    Load ONNX model with production-grade optimization settings (Phase 1 Optimization).

    This implementation applies comprehensive SessionOptions configuration for:
    - 2-3x speedup through graph optimization
    - Optimized memory patterns and thread configuration
    - Hardware-specific execution providers (TensorRT, CUDA, CPU)

    Expected performance improvement: 335-545ms → 110-170ms per inference

    Args:
        onnx_path: Path to ONNX model file
        enable_profiling: Enable performance profiling for debugging (default: False)
        inter_op_threads: Number of threads for parallel operator execution (default: 1)
        intra_op_threads: Number of threads within operators for matrix operations (default: 4)

    Returns:
        ort.InferenceSession: Optimized ONNX Runtime InferenceSession

    Example:
        >>> session = load_onnx_model("model.onnx")
        >>> inputs = {
        >>>     "input_ids": np.array([[101, 1024, 102]]),
        >>>     "attention_mask": np.array([[1, 1, 1]]),
        >>>     "tab_field1": np.array([[0.3, 1.5]]),
        >>> }
        >>> outputs = session.run(None, inputs)
        >>> logits = outputs[0]
    """
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

    # ===== 1. Configure SessionOptions =====
    sess_options = ort.SessionOptions()

    # Graph optimization: Enable all optimizations (level 99)
    # - Constant folding: Pre-compute constant operations
    # - Redundant node elimination: Remove unused operations
    # - Operator fusion: Combine multiple ops into single kernel (BERT-specific)
    # - Shape inference: Optimize tensor shapes
    # - Common subexpression elimination: Reuse computed values
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Execution mode: Sequential for single-request serving (lower latency)
    # ORT_SEQUENTIAL: Operators run sequentially (lower latency, single request)
    # ORT_PARALLEL: Operators run in parallel (higher throughput, batch requests)
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Memory optimizations
    sess_options.enable_mem_pattern = True  # Reuse memory allocations across inferences
    sess_options.enable_cpu_mem_arena = True  # Use memory arena for faster allocation

    # Thread configuration (tune based on CPU cores)
    sess_options.intra_op_num_threads = (
        intra_op_threads  # Threads per operator (matrix ops)
    )
    sess_options.inter_op_num_threads = inter_op_threads  # Threads for parallel ops

    # Profiling (disable in production for performance)
    if enable_profiling:
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = "onnx_profile"
        logger.info(
            "ONNX profiling enabled - disable in production for best performance"
        )

    # ===== 2. Configure Execution Providers =====
    providers = []

    # Try TensorRT first (best GPU performance, 1.2-1.5x additional speedup)
    if ort.get_device() == "GPU":
        try:
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": True,  # Enable FP16 precision (2x faster)
                        "trt_engine_cache_enable": True,  # Cache compiled engines
                        "trt_engine_cache_path": "/tmp/trt_cache",
                        "trt_max_workspace_size": 2147483648,  # 2GB workspace
                    },
                )
            )
            logger.info("✓ TensorRT execution provider configured")
        except Exception as e:
            logger.warning(f"TensorRT provider not available: {e}")

    # Fallback to CUDA (standard GPU execution)
    if ort.get_device() == "GPU":
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
                    "cudnn_conv_algo_search": "EXHAUSTIVE",  # Find best convolution algorithm
                    "do_copy_in_default_stream": True,
                },
            )
        )
        logger.info("✓ CUDA execution provider configured")

    # CPU fallback (always available)
    providers.append("CPUExecutionProvider")

    # ===== 3. Create Optimized Session =====
    try:
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
        )

        logger.info(f"✓ Loaded ONNX model with Phase 1 optimizations from {onnx_path}")
        logger.info(f"  Graph optimization: ORT_ENABLE_ALL")
        logger.info(f"  Execution mode: ORT_SEQUENTIAL (low latency)")
        logger.info(f"  Memory optimization: ENABLED (mem_pattern + cpu_mem_arena)")
        logger.info(
            f"  Intra-op threads: {intra_op_threads} (parallel within operators)"
        )
        logger.info(f"  Inter-op threads: {inter_op_threads} (sequential operators)")
        logger.info(
            f"  Active providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}"
        )
        logger.info(f"  Expected inputs: {[inp.name for inp in session.get_inputs()]}")
        logger.info(f"  Expected performance: 2-3x speedup vs baseline")

        return session

    except Exception as e:
        raise RuntimeError(f"Failed to load optimized ONNX model: {e}")


def optimize_bert_model(
    input_model_path: Union[str, Path],
    output_model_path: Union[str, Path],
    model_type: str = "bert",
    num_heads: int = 12,
    hidden_size: int = 768,
) -> None:
    """
    Apply BERT-specific optimizations to ONNX model (Phase 2 Optimization).

    This function applies graph-level fusion optimizations for BERT models:
    - Multi-head attention fusion (Q/K/V projections + attention)
    - LayerNorm fusion (normalization + skip connections)
    - GELU activation fusion
    - Embedding layer fusion (token + position + segment)

    CRITICAL: This is graph optimization, NOT weight modification.
    Your fine-tuned BERT weights remain completely unchanged.

    Expected speedup: 1.5-2x additional on top of Phase 1 (110ms → 55-75ms)

    Args:
        input_model_path: Path to original ONNX model (model.onnx)
        output_model_path: Path to save optimized model (model_optimized.onnx)
        model_type: Model architecture type (default: "bert")
        num_heads: Number of attention heads in BERT model (default: 12 for BERT-base)
        hidden_size: Hidden dimension size in BERT model (default: 768 for BERT-base)

    Raises:
        RuntimeError: If optimization fails
    """
    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
    except ImportError as e:
        raise RuntimeError(
            "BERT fusion requires onnxruntime transformers package. "
            "Install with: pip install onnxruntime[transformers]"
        ) from e

    logger.info(f"Starting BERT fusion optimization: {input_model_path}")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Attention heads: {num_heads}")
    logger.info(f"  Hidden size: {hidden_size}")

    try:
        # Configure fusion options
        fusion_options = FusionOptions(model_type)
        fusion_options.enable_gelu = True  # Fuse GELU activation
        fusion_options.enable_layer_norm = True  # Fuse layer normalization
        fusion_options.enable_attention = True  # Fuse multi-head attention
        fusion_options.enable_skip_layer_norm = (
            True  # Fuse skip connections + layer norm
        )
        fusion_options.enable_embed_layer_norm = True  # Fuse embedding + layer norm
        fusion_options.enable_bias_skip_layer_norm = (
            True  # Fuse bias + skip + layer norm
        )
        fusion_options.enable_bias_gelu = True  # Fuse bias + GELU
        fusion_options.enable_gelu_approximation = (
            False  # Use exact GELU (better accuracy)
        )

        # Create optimizer instance
        optimizer_instance = optimizer.optimize_model(
            input=str(input_model_path),
            model_type=model_type,
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=fusion_options,
            opt_level=99,  # Maximum optimization level
            use_gpu=torch.cuda.is_available(),
        )

        # Save optimized model
        optimizer_instance.save_model_to_file(str(output_model_path))

        # Get fusion statistics
        fusion_stats = optimizer_instance.get_fused_operator_statistics()

        logger.info(f"✓ BERT fusion optimization completed successfully")
        logger.info(f"  Optimized model saved to: {output_model_path}")
        logger.info(f"  Fusion statistics: {fusion_stats}")
        logger.info(f"  Expected speedup: 1.5-2x additional (on top of Phase 1)")

    except Exception as e:
        logger.error(f"BERT fusion optimization failed: {e}")
        raise RuntimeError(f"Failed to optimize BERT model: {e}") from e


def load_bert_optimized_model(
    model_dir: Union[str, Path],
    enable_profiling: bool = False,
    inter_op_threads: int = 1,
    intra_op_threads: int = 4,
    force_reoptimize: bool = False,
    optimization_timeout: int = 90,
) -> ort.InferenceSession:
    """
    Load BERT-optimized ONNX model with atomic file locking (multi-worker safe).

    NEW: Added file locking to prevent race conditions when multiple workers
    initialize simultaneously (e.g., TorchServe with 8 workers on ml.m5.4xlarge).

    Only ONE worker performs the optimization, others wait for completion.

    This function implements the complete optimization workflow:
    1. Check for pre-optimized model (model_optimized.onnx)
    2. If not found, acquire file lock for safe optimization
    3. Apply BERT fusion to original model (one-time cost, single worker only)
    4. Load with Phase 1 SessionOptions configuration

    Expected performance: 2-3x (Phase 1) × 1.5-2x (Phase 2) = 3-6x total speedup

    Args:
        model_dir: Directory containing ONNX models
        enable_profiling: Enable performance profiling for debugging (default: False)
        inter_op_threads: Number of threads for parallel operator execution (default: 1)
        intra_op_threads: Number of threads within operators for matrix operations (default: 4)
        force_reoptimize: Force re-optimization even if optimized model exists (default: False)
        optimization_timeout: Maximum seconds to wait for optimization lock (default: 90s)

    Returns:
        Optimized ONNX Runtime InferenceSession with Phase 1 + Phase 2 optimizations

    Example:
        >>> # First load: creates model_optimized.onnx (one-time 10-30s cost)
        >>> session = load_bert_optimized_model("/opt/ml/model")
        >>> # Subsequent loads: uses cached model_optimized.onnx (fast)
        >>> session = load_bert_optimized_model("/opt/ml/model")
        >>>
        >>> # Run inference with optimized model
        >>> inputs = {
        >>>     "input_ids": np.array([[101, 1024, 102]]),
        >>>     "attention_mask": np.array([[1, 1, 1]]),
        >>>     "tab_field1": np.array([[0.3, 1.5]]),
        >>> }
        >>> outputs = session.run(None, inputs)
    """
    model_dir = Path(model_dir)
    original_model = model_dir / "model.onnx"
    optimized_model = model_dir / "model_optimized.onnx"
    lock_file = model_dir / ".model_optimization.lock"

    # Validate original model exists
    if not original_model.exists():
        raise FileNotFoundError(f"Original ONNX model not found: {original_model}")

    # FAST PATH: Optimized model already exists and no force reoptimize
    if not force_reoptimize and optimized_model.exists():
        logger.info("✓ Using cached BERT-optimized model")
        return load_onnx_model(
            onnx_path=optimized_model,
            enable_profiling=enable_profiling,
            inter_op_threads=inter_op_threads,
            intra_op_threads=intra_op_threads,
        )

    # OPTIMIZATION PATH: Need to create or recreate optimized model
    logger.info("BERT-optimized model not found, acquiring lock for optimization...")

    try:
        with file_lock(lock_file, timeout=optimization_timeout):
            # Double-check: another worker may have created it while we waited
            if not force_reoptimize and optimized_model.exists():
                logger.info(
                    "✓ Another worker completed optimization, using cached model"
                )
                return load_onnx_model(
                    onnx_path=optimized_model,
                    enable_profiling=enable_profiling,
                    inter_op_threads=inter_op_threads,
                    intra_op_threads=intra_op_threads,
                )

            # This worker won the race - perform optimization
            logger.info("✓ Lock acquired, starting BERT fusion optimization...")
            logger.info("  This is a one-time cost (10-30 seconds)")
            logger.info(
                "  Other workers will wait and use the optimized model once ready"
            )

            try:
                optimize_bert_model(
                    input_model_path=original_model,
                    output_model_path=optimized_model,
                    model_type="bert",
                    num_heads=12,  # Standard BERT-base configuration
                    hidden_size=768,  # Standard BERT-base configuration
                )
                logger.info("✓ BERT optimization completed successfully")

            except Exception as e:
                logger.error(f"BERT fusion optimization failed: {e}")
                logger.warning(
                    "Falling back to Phase 1 optimizations only (SessionOptions)"
                )
                # Fallback: load original model with Phase 1 optimizations
                return load_onnx_model(
                    onnx_path=original_model,
                    enable_profiling=enable_profiling,
                    inter_op_threads=inter_op_threads,
                    intra_op_threads=intra_op_threads,
                )

    except TimeoutError as e:
        logger.error(f"Lock acquisition timeout: {e}")
        logger.warning("Optimization is taking too long or another worker is stuck")
        logger.warning("Falling back to Phase 1 optimizations only (SessionOptions)")
        # Fallback: load original model with Phase 1 optimizations
        return load_onnx_model(
            onnx_path=original_model,
            enable_profiling=enable_profiling,
            inter_op_threads=inter_op_threads,
            intra_op_threads=intra_op_threads,
        )

    # Load the optimized model (either just created or found after waiting)
    logger.info("Loading BERT-optimized model with Phase 1 + Phase 2 optimizations")
    return load_onnx_model(
        onnx_path=optimized_model,
        enable_profiling=enable_profiling,
        inter_op_threads=inter_op_threads,
        intra_op_threads=intra_op_threads,
    )
