---
tags:
  - design
  - native_pytorch
  - ddp
  - fsdp
  - pipeline_parallelism
  - tensor_parallelism
  - distributed_training
keywords:
  - PyTorch 2.x
  - DistributedDataParallel
  - FullyShardedDataParallel
  - Pipeline Parallelism
  - Trimodal BERT
  - collective communication
  - ZeRO
topics:
  - model migration
  - distributed system design
  - training strategy
language: python
date of note: 2025-11-21
---

# Native PyTorch Migration Strategy: DDP, FSDP, and Pipeline Parallelism

## 1. Executive Summary

This document outlines the strategy for migrating the Trimodal BERT model from PyTorch Lightning (`pl.LightningModule`) to native PyTorch 2+. The primary objective is to gain fine-grained control over distributed training strategies, specifically enabling:

1.  **Distributed Data Parallel (DDP)** for standard scaling.
2.  **Fully Sharded Data Parallel (FSDP)** for large-model efficiency.
3.  **Pipeline Parallelism (PP)** for splitting the model across multiple GPUs (specifically 2-4 GPUs).
4.  **Tensor Parallelism (TP)** for splitting individual layers (future-proofing).

## 2. Model Architecture: Trimodal BERT

### 2.1 Current Lightning Implementation

The existing `TrimodalBert` model in `projects/rnr_pytorch_bedrock/docker/lightning_models/trimodal/pl_trimodal_bert.py` consists of:

1.  **Primary Text Branch** (`TextBertBase`):
    *   Pre-trained BERT encoder (12 layers, 768 hidden dim for `bert-base-cased`).
    *   Pooling layer extracting `[CLS]` token.
    *   Linear projection to `hidden_common_dim`.
    *   Processes dialogue/chat text.

2.  **Secondary Text Branch** (`TextBertBase`):
    *   Independent BERT encoder (same architecture).
    *   Processes shiptrack or other auxiliary text.

3.  **Tabular Branch** (`TabAE`):
    *   Simple MLP: `LayerNorm -> Linear(input_dim, hidden_common_dim) -> ReLU`.
    *   Processes numerical features.

4.  **Fusion Head**:
    *   Concatenates outputs: `[primary_dim + secondary_dim + tab_dim]`.
    *   Two-layer MLP with dropout.
    *   Final classification layer: `Linear(fusion_hidden_dim, num_classes)`.

### 2.2 Training Orchestration (Current)

*   Uses `pl.Trainer` with FSDP strategy configured via `strategy="fsdp"`.
*   Lightning handles:
    *   Distributed process group initialization.
    *   Automatic gradient synchronization.
    *   Optimizer state sharding.
    *   Checkpointing and logging.

### 2.3 Refactoring Goals

Remove Lightning dependencies to enable:
*   Direct control over distributed wrappers (DDP, FSDP, Pipeline).
*   Custom training loops with explicit forward/backward passes.
*   Fine-grained profiling and optimization.
*   Easier integration with non-Lightning tooling.

**See**: [Native PyTorch Implementation Plan](./native_pytorch_implementation_plan.md) for complete code examples of the refactored models.

## 3. Distributed Training Theory

### 3.1 Distributed Data Parallel (DDP)

#### 3.1.1 Mechanism

DDP replicates the entire model on each GPU and splits the input batch across processes:

1.  **Initialization**:
    *   `torch.distributed.init_process_group("nccl", ...)` creates a process group.
    *   Each process loads a full copy of the model.
    *   Model is wrapped: `model = DDP(model, device_ids=[local_rank])`.

2.  **Forward Pass**:
    *   Each process receives a different slice of the global batch.
    *   Forward pass is independent (no communication).

3.  **Backward Pass**:
    *   Gradients are computed locally.
    *   DDP registers hooks on parameters to trigger all-reduce.
    *   **All-Reduce**: Uses ring-based or tree-based algorithms to average gradients across all processes.
    *   Result: All processes have identical averaged gradients.

4.  **Optimizer Step**:
    *   Each process updates its local copy with the averaged gradients.
    *   Models remain synchronized.

#### 3.1.2 Components

*   **Process Group**: Logical grouping of processes (ranks).
*   **Backend**: `nccl` (GPU), `gloo` (CPU).
*   **Bucket Gradient Aggregation**: DDP groups gradients into buckets (~25MB) for efficient all-reduce.

#### 3.1.3 Use Case

*   Model fits comfortably on single GPU.
*   Goal: Increase effective batch size by distributing data.
*   Memory overhead: Full model + optimizer state per GPU.

### 3.2 Fully Sharded Data Parallel (FSDP)

#### 3.2.1 Mechanism

FSDP shards model parameters, gradients, and optimizer states across GPUs. It's inspired by ZeRO-3 (DeepSpeed).

**Key Idea**: Each GPU only holds a fraction (1/N) of the model parameters permanently. During forward/backward, layers are temporarily "unsharded" (all-gathered) as needed, then immediately discarded.

1.  **Initialization**:
    *   Model is wrapped recursively: `FSDP(model, auto_wrap_policy=...)`.
    *   Parameters are sharded across the process group.
    *   Each rank holds `params[rank::world_size]`.

2.  **Forward Pass (Layer-wise)**:
    *   **Pre-Forward All-Gather**: Before computing a layer, all-gather its full parameters from all ranks.
    *   **Compute**: Execute the layer with full parameters.
    *   **Discard**: Free the full parameters (keep only local shard).

3.  **Backward Pass (Layer-wise)**:
    *   **Pre-Backward All-Gather**: Re-gather full parameters for gradient computation.
    *   **Compute Gradients**: Compute gradients w.r.t. full parameters.
    *   **Reduce-Scatter**: Average gradients across ranks and scatter so each rank gets its shard.
    *   **Discard**: Free full gradients (keep only local gradient shard).

4.  **Optimizer Step**:
    *   Each rank updates its parameter shard.
    *   Optimizer states are also sharded (1/N memory per rank).

#### 3.2.2 Components

*   **Sharding Strategy**: `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OP` (ZeRO-2), `NO_SHARD` (DDP-like).
*   **Auto-Wrap Policy**: Determines which submodules to wrap. Common policies:
    *   `transformer_auto_wrap_policy`: Wraps transformer layers (e.g., `BertLayer`).
    *   `size_based_auto_wrap_policy`: Wraps modules exceeding a parameter count threshold.
*   **Mixed Precision**: Uses `MixedPrecision` object to specify compute/reduce dtypes (FP16/BF16).
*   **CPU Offload**: Optionally offload parameters/gradients to CPU.

#### 3.2.3 Use Case

*   Model is too large for single GPU (even with gradient checkpointing).
*   Goal: Maximize model capacity while maintaining memory efficiency.
*   Memory overhead: ~1/N of model + optimizer state per GPU.
*   Trade-off: Increased communication (all-gather/reduce-scatter) vs. memory savings.

### 3.3 Pipeline Parallelism (PP)

#### 3.3.1 Mechanism

Pipeline Parallelism splits the model into sequential stages, each assigned to a different GPU. Input batches are divided into microbatches that flow through stages in a pipelined fashion.

**Key Idea**: While GPU 0 processes microbatch 2, GPU 1 can simultaneously process microbatch 1 (which finished on GPU 0). This overlaps computation across stages.

1.  **Model Partitioning**:
    *   Model is divided into stages: `Stage0 -> Stage1 -> ... -> StageN`.
    *   Each stage is a sequential subset of layers.
    *   Stage `i` is assigned to GPU `i`.

2.  **Forward Pass (GPipe Schedule)**:
    *   Split batch into M microbatches.
    *   GPU 0 processes all M microbatches sequentially, sending outputs to GPU 1.
    *   GPU 1 waits for first microbatch from GPU 0, processes it, sends to GPU 2, etc.
    *   Results in a "fill-up" phase where pipeline gradually fills.

3.  **Backward Pass**:
    *   Once all microbatches complete forward on final stage, backward begins.
    *   Gradients flow in reverse: Stage N -> Stage N-1 -> ... -> Stage 0.
    *   Each stage accumulates gradients from all microbatches.

4.  **1F1B Schedule (Interleaved)**:
    *   More efficient than GPipe: interleaves forward and backward passes.
    *   After a stage completes forward for a microbatch, it immediately does backward for a previous microbatch.
    *   Reduces memory (fewer activations stored) and improves GPU utilization.

#### 3.3.2 Components

*   **PipelineStage**: Wrapper around a stage's submodule + device assignment.
*   **Schedule**: Defines the execution order (GPipe, 1F1B, Interleaved 1F1B).
*   **Microbatches**: Number of splits. More microbatches = better pipeline efficiency but higher memory for activations.
*   **Activation Checkpointing**: Reduce memory by recomputing activations during backward.

#### 3.3.3 Use Case for Trimodal BERT (2-4 GPUs)

**2-GPU Setup**:
*   **Stage 0 (GPU 0)**: Primary Text BERT (12 layers, ~110M params).
*   **Stage 1 (GPU 1)**: Secondary Text BERT + Tabular + Fusion Head (~110M + small).
*   Issue: Imbalanced stages. GPU 1 has more computation.

**4-GPU Setup (Balanced)**:
*   **Stage 0 (GPU 0)**: Primary Text BERT Layers 1-6.
*   **Stage 1 (GPU 1)**: Primary Text BERT Layers 7-12.
*   **Stage 2 (GPU 2)**: Secondary Text BERT Layers 1-12.
*   **Stage 3 (GPU 3)**: Tabular + Fusion Head + Loss.
*   Better balance, but Stage 2 is still heavy. Could split Secondary BERT similarly.

**Alternative: Branch Parallelism**:
*   Since Primary and Secondary branches are independent until fusion, we can use a "DAG-style" pipeline:
    *   GPU 0: Primary Text BERT (all layers).
    *   GPU 1: Secondary Text BERT (all layers).
    *   GPU 0 or 1: Tabular + Fusion (after gathering outputs).

#### 3.3.4 Trade-offs

*   **Pros**: Enables training models that don't fit on single GPU due to layer depth.
*   **Cons**:
    *   Pipeline bubbles (idle time during fill-up and drain phases).
    *   Increased latency per step.
    *   Complex to balance stages.
    *   Requires careful tuning of microbatch count.

### 3.4 Tensor Parallelism (TP)

#### 3.4.1 Mechanism

TP splits individual layers (e.g., matrix multiplications) across GPUs. For a linear layer `Y = XW`, the weight matrix `W` is partitioned column-wise or row-wise across devices.

**Column-Wise Parallelism**:
*   Split `W` into `[W0, W1, ..., Wn]` along columns.
*   Each GPU holds `Wi` and computes `Yi = X * Wi`.
*   Final output: `Y = [Y0, Y1, ..., Yn]` (concatenated).
*   All-gather required to reconstruct full output for next layer.

**Row-Wise Parallelism**:
*   Split `W` along rows: `W = [W0; W1; ...; Wn]`.
*   Each GPU computes partial result with local rows.
*   All-reduce required to sum partial outputs.

#### 3.4.2 Components

*   **DeviceMesh**: Defines the topology of devices (e.g., 2D mesh for TP + DP).
*   **DTensor (Distributed Tensor)**: Abstraction for tensors sharded across devices.
*   **Parallelization Plan**: Specifies which layers use colwise/rowwise/etc.

#### 3.4.3 Use Case

*   Extremely large layers (e.g., LLaMA with 8192 hidden dim).
*   Goal: Reduce per-GPU memory for individual layer weights.
*   Trade-off: High communication overhead (all-gather/all-reduce per layer).
*   Best combined with other strategies (e.g., TP + FSDP).

## 4. Hybrid Strategies for 2-4 GPUs

For the Trimodal BERT model on 2-4 GPUs, we recommend:

### 4.1 Small-Medium Models (<1B params)

*   **2 GPUs**: DDP (simple, efficient).
*   **4 GPUs**: DDP or FSDP (if memory-constrained).

### 4.2 Large Models (>1B params)

*   **2 GPUs**: FSDP with aggressive sharding.
*   **4 GPUs**: FSDP + Gradient Checkpointing.

### 4.3 Experimental (Deep Pipelines)

*   **2-4 GPUs**: Pipeline Parallelism with 1F1B schedule.
*   Suitable for models with sequential depth (e.g., 48-layer transformers).

## 5. Implementation Reference

For detailed code examples and step-by-step implementation, see [Native PyTorch Implementation Plan](./native_pytorch_implementation_plan.md).
