---
tags:
  - design
  - implementation
  - distributed_training
  - data_loading
  - streaming
keywords:
  - IterableDataset
  - FSDP
  - DDP
  - PyTorch Lightning
  - Distributed Data Sharding
  - Streaming Mode
  - PipelineIterableDataset
topics:
  - data pipeline
  - distributed training
  - memory efficiency
language: python
date of note: 2026-01-19
---

# IterableDataset Distributed Training Design

## 1. Executive Summary

This document addresses a critical gap in the current `PipelineIterableDataset` implementation: **missing rank-based data sharding for distributed training with FSDP/DDP in PyTorch Lightning**.

### 1.1 Problem Statement

Current implementation (`projects/names3risk_pytorch/dockers/processing/datasets/pipeline_iterable_datasets.py`) only implements **worker-level sharding** but lacks **rank-based sharding** for distributed GPU training. This causes:

- ❌ **Data duplication**: All GPU ranks process the same data shards
- ❌ **Training inefficiency**: Incorrect gradient aggregation across ranks
- ❌ **Wasted compute**: Multiple GPUs redundantly process identical batches

### 1.2 Root Cause

From PyTorch documentation and community forums:

> **"DistributedSampler does not affect the data fetching behavior of IterableDataset"**
> 
> Unlike regular `Dataset` where `DistributedSampler` automatically handles data distribution across ranks, `IterableDataset` requires **manual implementation of rank-based data sharding**.

### 1.3 Impact Assessment

**Current Status**:
- ✅ Batch mode (`PipelineDataset`) works correctly with FSDP/DDP (uses `DistributedSampler`)
- ⚠️ Streaming mode (`PipelineIterableDataset`) may have data duplication issues with FSDP/DDP

**Affected Components**:
- `projects/names3risk_pytorch/dockers/processing/datasets/pipeline_iterable_datasets.py`
- `projects/names3risk_pytorch/dockers/pytorch_training.py` (when `ENABLE_TRUE_STREAMING=true`)
- Any Lightning model using `FSDPStrategy` or `DDPStrategy` with streaming mode

### 1.4 Proposed Solution

Implement a **two-tier sharding strategy** in `PipelineIterableDataset.__iter__()`:

1. **Tier 1: Rank-based sharding** - Distribute shards across GPU ranks (FSDP/DDP dimension)
2. **Tier 2: Worker-based sharding** - Distribute shards across DataLoader workers (existing)

**Expected Benefits**:
- ✅ Correct distributed training behavior
- ✅ No data duplication across ranks
- ✅ Efficient memory usage with streaming
- ✅ Compatible with both DDP and FSDP strategies in PyTorch Lightning

---

## 2. Background and Analysis

### 2.1 Current Implementation Review

#### 2.1.1 Existing Code Structure

```python
# projects/names3risk_pytorch/dockers/processing/datasets/pipeline_iterable_datasets.py
class PipelineIterableDataset(IterableDataset):
    def __iter__(self) -> Iterator[Dict]:
        # Get worker info for multi-process data loading
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single process: process all shards
            shards_to_process = self.shard_files
        else:
            # Multi-process: split shards across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards_to_process = self.shard_files[worker_id::num_workers]  # ✅ Worker sharding
        
        # Shuffle shards if requested
        if self.shuffle_shards and shards_to_process:
            shards_list = list(shards_to_process)
            random.Random(42 + worker_id).shuffle(shards_list)
            shards_to_process = shards_list
        
        # Iterate through assigned shards...
```

**Analysis**:
- ✅ Correctly implements worker-level sharding using `torch.utils.data.get_worker_info()`
- ✅ Proper shuffling with deterministic seed per worker
- ❌ **Missing**: No rank-based sharding using `torch.distributed.get_rank()`
- ❌ **Missing**: No world_size consideration for distributed training

#### 2.1.2 Usage in Training Script

```python
# projects/names3risk_pytorch/dockers/pytorch_training.py
def load_data_module(file_dir, filename, config, use_streaming=False):
    if use_streaming and has_shards:
        pipeline_dataset = PipelineIterableDataset(
            config=config.model_dump(),
            file_dir=file_dir,
            shuffle_shards=True if "train" in file_dir else False,
        )
        return pipeline_dataset
    # ...

# Later in build_model_and_optimizer:
train_dataloader = DataLoader(
    train_pipeline_dataset,
    collate_fn=collate_batch,
    batch_size=batch_size,
    shuffle=False if use_streaming else True,  # No shuffle for streaming!
)
```

**Analysis**:
- ⚠️ No `DistributedSampler` used (not compatible with `IterableDataset`)
- ⚠️ Shuffle disabled for streaming mode (relying on dataset-level shuffle)
- ❌ No rank-based data distribution mechanism

#### 2.1.3 FSDP Strategy Configuration

```python
# projects/names3risk_pytorch/dockers/lightning_models/utils/pl_train.py
strategy = (
    FSDPStrategy(auto_wrap_policy=my_auto_wrap_policy, verbose=True)
    if is_fsdp_available()
    else "auto"
)

trainer = pl.Trainer(
    max_epochs=max_epochs,
    strategy=strategy,
    # ...
)
```

**Analysis**:
- ✅ FSDP strategy correctly configured for model parallelism
- ❌ But data loading doesn't account for multiple ranks
- ⚠️ Each rank may see identical data batches

### 2.2 Comparison with Batch Mode

#### 2.2.1 PipelineDataset (Batch Mode)

```python
# Regular Dataset inherits from torch.utils.data.Dataset
class PipelineDataset(Dataset):
    def __getitem__(self, idx):
        # Random access by index
        row = self.DataReader.iloc[idx].to_dict()
        # Apply pipelines...
        return row
    
    def __len__(self):
        return len(self.DataReader)
```

**Usage with DistributedSampler**:
```python
from torch.utils.data import DistributedSampler

train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,  # Automatically gets from torch.distributed
    rank=rank,                 # Automatically gets from torch.distributed
    shuffle=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,  # ✅ Handles rank-based distribution
)
```

**Why it works**:
- `DistributedSampler` generates rank-specific indices
- Each rank only sees its assigned subset of data
- PyTorch Lightning automatically sets up `DistributedSampler` for `Dataset`

#### 2.2.2 Why IterableDataset is Different

From PyTorch documentation:

> **"IterableDataset is designed for use cases where random access is expensive or infeasible. The __getitem__ method is not implemented; instead, iteration is performed via __iter__()."**

Key differences:

| Aspect | Dataset | IterableDataset |
|--------|---------|-----------------|
| Access Pattern | Random access (`__getitem__`) | Sequential iteration (`__iter__`) |
| Length | Required (`__len__`) | Optional (may be unknown) |
| Sampler | Compatible with `DistributedSampler` | ❌ **Not compatible** |
| Rank Distribution | Automatic (via sampler) | ❌ **Manual required** |
| Use Case | Fits in memory | Streaming from disk/network |

### 2.3 Official PyTorch Guidance

#### 2.3.1 PyTorch Forums Discussion

From [PyTorch Forums](https://discuss.pytorch.org/t/using-iterabledataset-with-distributeddataparallel/92589):

**Problem Identified**:
> "When IterableDataset is used with DDP, all GPUs process the same batch of data because DistributedSampler does not affect IterableDataset."

**Recommended Solution**:
> "Implement sharding logic directly within the IterableDataset class by taking the distributed training rank (shard ID) as an argument and having the dataset yield only the portion of data corresponding to that rank."

**Code Example from Forum**:
```python
class DistributedIterableDataset(IterableDataset):
    def __iter__(self):
        # Get distributed training info
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        
        # Shard data based on rank
        for i, item in enumerate(self.data_source):
            if i % world_size == rank:
                yield item
```

#### 2.3.2 PyTorch Lightning Documentation

From [PyTorch Lightning Docs on IterableDataset](https://lightning.ai/docs/pytorch/stable/data/iterables.html):

**Official Stance**:
- Lightning supports `IterableDataset` but does **not** automatically handle distributed sharding
- Users must implement rank-based sharding manually
- No warnings or errors if sharding is missing (silent data duplication)

**Best Practice**:
> "For distributed training with IterableDataset, implement rank-aware iteration in __iter__() using torch.distributed.get_rank() and torch.distributed.get_world_size()."

### 2.4 Compatibility Matrix

| Training Mode | Dataset Type | Sharding Mechanism | Current Status |
|---------------|--------------|-------------------|----------------|
| Single GPU | `PipelineDataset` | None needed | ✅ Works |
| Single GPU | `PipelineIterableDataset` | None needed | ✅ Works |
| DDP/FSDP | `PipelineDataset` | `DistributedSampler` (automatic) | ✅ Works |
| DDP/FSDP | `PipelineIterableDataset` | Manual rank sharding | ❌ **Missing** |

---

## 3. Proposed Design

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PipelineIterableDataset                      │
│                                                                 │
│  __iter__() Method:                                            │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ TIER 1: Rank-Based Sharding (NEW)                        │ │
│  │ ┌───────────────────────────────────────────────────────┐ │ │
│  │ │ if torch.distributed.is_initialized():               │ │ │
│  │ │     rank = get_rank()                                 │ │ │
│  │ │     world_size = get_world_size()                     │ │ │
│  │ │     shards_for_rank = shard_files[rank::world_size]  │ │ │
│  │ └───────────────────────────────────────────────────────┘ │ │
│  │                          ↓                                 │ │
│  │ TIER 2: Worker-Based Sharding (EXISTING)                 │ │
│  │ ┌───────────────────────────────────────────────────────┐ │ │
│  │ │ worker_info = get_worker_info()                       │ │ │
│  │ │ if worker_info:                                       │ │ │
│  │ │     worker_id = worker_info.id                        │ │ │
│  │ │     num_workers = worker_info.num_workers             │ │ │
│  │ │     shards = shards_for_rank[worker_id::num_workers] │ │ │
│  │ └───────────────────────────────────────────────────────┘ │ │
│  │                          ↓                                 │ │
│  │ TIER 3: Shard Iteration                                  │ │
│  │ ┌───────────────────────────────────────────────────────┐ │ │
│  │ │ for shard in shards:                                  │ │ │
│  │ │     df = load_shard(shard)                           │ │ │
│  │ │     for row in df:                                    │ │ │
│  │ │         yield process_row(row)                        │ │ │
│  │ └───────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow Example

**Scenario**: 8 GPUs (FSDP), 100 shards, 4 DataLoader workers per GPU

**Without Rank Sharding (CURRENT - BROKEN)**:
```
Rank 0: Workers [0,1,2,3] process shards [0,4,8,12,...,96] (25 shards each)
Rank 1: Workers [0,1,2,3] process shards [0,4,8,12,...,96] (25 shards each) ❌ DUPLICATE
Rank 2: Workers [0,1,2,3] process shards [0,4,8,12,...,96] (25 shards each) ❌ DUPLICATE
...
Rank 7: Workers [0,1,2,3] process shards [0,4,8,12,...,96] (25 shards each) ❌ DUPLICATE

Result: All 8 ranks process the same 100 shards → 8× data duplication!
```

**With Rank Sharding (PROPOSED - CORRECT)**:
```
Rank 0: Gets shards [0,8,16,24,...,96]  (13 shards)
  Worker 0: [0,32,64,96]      (4 shards)
  Worker 1: [8,40,72]         (3 shards)
  Worker 2: [16,48,80]        (3 shards)
  Worker 3: [24,56,88]        (3 shards)

Rank 1: Gets shards [1,9,17,25,...,97]  (13 shards)
  Worker 0: [1,33,65,97]      (4 shards)
  Worker 1: [9,41,73]         (3 shards)
  Worker 2: [17,49,81]        (3 shards)
  Worker 3: [25,57,89]        (3 shards)

...

Rank 7: Gets shards [7,15,23,31,...,99] (12 shards)
  Worker 0: [7,39,71]         (3 shards)
  Worker 1: [15,47,79]        (3 shards)
  Worker 2: [23,55,87]        (3 shards)
  Worker 3: [31,63,95]        (3 shards)

Result: Each of 100 shards processed exactly once → No duplication! ✅
```

### 3.3 Implementation Details

#### 3.3.1 Enhanced `__iter__()` Method

```python
def __iter__(self) -> Iterator[Dict]:
    """
    Iterate through dataset with proper distributed sharding.
    
    Implements two-tier sharding:
    1. Rank-based sharding: Distribute shards across GPU ranks (FSDP/DDP)
    2. Worker-based sharding: Distribute shards across DataLoader workers
    
    Example:
        8 GPUs, 4 workers/GPU, 100 shards:
        - Rank 0 gets shards [0, 8, 16, 24, ..., 96]
        - Rank 0's worker 0 gets [0, 32, 64, 96]
        - Rank 0's worker 1 gets [8, 40, 72]
        - ...
        - Rank 1 gets shards [1, 9, 17, 25, ..., 97]
        - etc.
    
    Yields:
        Dict: Processed row with applied pipelines
    """
    # ============================================================
    # TIER 1: Rank-Based Sharding (NEW)
    # ============================================================
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        
        # Distribute shards across ranks using round-robin
        shards_for_this_rank = self.shard_files[rank::world_size]
        
        if is_main_process():
            logger.info(
                f"[IterableDataset] Rank {rank}/{world_size}: "
                f"Assigned {len(shards_for_this_rank)}/{len(self.shard_files)} shards"
            )
    else:
        # Single GPU mode
        shards_for_this_rank = self.shard_files
        rank = 0
        world_size = 1
    
    # ============================================================
    # TIER 2: Worker-Based Sharding (EXISTING - ENHANCED)
    # ============================================================
    worker_info = torch.utils.data.get_worker_info()
    
    if worker_info is None:
        # Single worker: process all shards for this rank
        shards_to_process = shards_for_this_rank
        worker_id = 0
        num_workers = 1
    else:
        # Multiple workers: distribute rank's shards across workers
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        shards_to_process = shards_for_this_rank[worker_id::num_workers]
        
        logger.debug(
            f"[IterableDataset] Rank {rank}, Worker {worker_id}/{num_workers}: "
            f"Processing {len(shards_to_process)} shards"
        )
    
    # ============================================================
    # TIER 3: Shard Shuffling (EXISTING - ENHANCED)
    # ============================================================
    if self.shuffle_shards and shards_to_process:
        shards_list = list(shards_to_process)
        
        # Deterministic shuffle based on rank + worker + epoch
        # This ensures reproducibility while maintaining randomness
        shuffle_seed = (
            42 +                                    # Base seed
            rank * 10000 +                          # Rank offset
            worker_id * 100 +                       # Worker offset
            getattr(self, '_current_epoch', 0)      # Epoch offset (set externally)
        )
        
        random.Random(shuffle_seed).shuffle(shards_list)
        shards_to_process = shards_list
        
        logger.debug(
            f"[IterableDataset] Rank {rank}, Worker {worker_id}: "
            f"Shuffled with seed {shuffle_seed}"
        )
    
    # ============================================================
    # TIER 4: Shard Iteration (EXISTING)
    # ============================================================
    for shard_idx, shard_path in enumerate(shards_to_process):
        # Load shard
        df = self._load_shard(shard_path)
        
        # Apply type conversions
        df = self._postprocess_dataframe(df)
        
        # Yield rows from shard
        for idx in range(len(df)):
            row = df.iloc[idx].to_dict()
            
            # Apply processor pipelines
            for field_name, pipeline in self.processor_pipelines.items():
                if field_name in row:
                    row[field_name] = pipeline(row[field_name])
            
            yield row
        
        # Free memory after processing shard
        del df
        gc.collect()
```

#### 3.3.2 Epoch-Aware Shuffling

```python
class PipelineIterableDataset(IterableDataset):
    def __init__(self, ...):
        # ... existing init ...
        self._current_epoch = 0
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set current epoch for deterministic shuffling.
        
        Should be called at the start of each epoch, similar to
        DistributedSampler.set_epoch().
        
        Args:
            epoch: Current epoch number
        
        Example:
            >>> dataset.set_epoch(epoch)
            >>> for batch in dataloader:
            >>>     # Training step
        """
        self._current_epoch = epoch
        logger.debug(f"[IterableDataset] Epoch set to {epoch}")
```

#### 3.3.3 Diagnostic Utilities

```python
def get_shard_distribution_info(self) -> Dict[str, Any]:
    """
    Get diagnostic information about shard distribution.
    
    Returns:
        Dict containing:
            - total_shards: Total number of shards
            - shards_per_rank: Number of shards assigned to this rank
            - shards_per_worker: Number of shards per worker
            - rank: Current rank
            - world_size: Total number of ranks
            - worker_id: Current worker ID (if available)
            - num_workers: Total number of workers per rank
            - assigned_shards: List of shard files assigned to this rank/worker
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        shards_for_rank = self.shard_files[rank::world_size]
    else:
        rank = 0
        world_size = 1
        shards_for_rank = self.shard_files
    
    worker_info = torch.utils.data.get_worker_info()
    if worker_info:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        shards_for_worker = shards_for_rank[worker_id::num_workers]
    else:
        worker_id = 0
        num_workers = 1
        shards_for_worker = shards_for_rank
    
    return {
        "total_shards": len(self.shard_files),
        "shards_per_rank": len(shards_for_rank),
        "shards_per_worker": len(shards_for_worker),
        "rank": rank,
        "world_size": world_size,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "assigned_shards": [str(s) for s in shards_for_worker],
    }


def verify_no_shard_overlap(dataloader, num_ranks: int, num_workers: int) -> bool:
    """
    Verify that no shard is processed by multiple rank/worker combinations.
    
    WARNING: This is an expensive operation that requires coordination
    across all ranks. Use only for testing/validation.
    
    Args:
        dataloader: DataLoader using PipelineIterableDataset
        num_ranks: Number of distributed ranks
        num_workers: Number of workers per rank
    
    Returns:
        True if no overlap detected, False otherwise
    """
    if not torch.distributed.is_initialized():
        logger.warning("Not in distributed mode, skipping overlap check")
        return True
    
    rank = torch.distributed.get_rank()
    dataset = dataloader.dataset
    
    # Get this rank's shard assignment
    info = dataset.get_shard_distribution_info()
    local_shards = set(info["assigned_shards"])
    
    # Gather all shard assignments from all ranks
    all_shards = [None] * num_ranks
    torch.distributed.all_gather_object(all_shards, local_shards)
    
    if rank == 0:
        # Check for overlaps
        seen = set()
        overlap_found = False
        
        for rank_idx, shards in enumerate(all_shards):
            overlap = seen & shards
            if overlap:
                logger.error(
                    f"Rank {rank_idx} has overlapping shards: {overlap}"
                )
                overlap_found = True
            seen.update(shards)
        
        # Check for missing shards
        all_expected = set(str(s) for s in dataset.shard_files)
        missing = all_expected - seen
        if missing:
            logger.error(f"Missing shards not assigned to any rank: {missing}")
            overlap_found = True
        
        if not overlap_found:
            logger.info("✅ No shard overlap detected - distribution is correct!")
        
        return not overlap_found
    
    return True  # Non-main ranks return True
```

### 3.4 Integration with Training Script

#### 3.4.1 Updated `load_and_preprocess_data()`

```python
# projects/names3risk_pytorch/dockers/pytorch_training.py

def load_and_preprocess_data(
    config: Config,
    paths: Dict[str, str],
    model_artifacts_dir: Optional[str] = None,
    use_precomputed_imputation: bool = False,
    use_precomputed_risk_tables: bool = False,
    use_streaming: bool = False,
) -> Tuple[List[Union[PipelineDataset, PipelineIterableDataset]], AutoTokenizer, Dict]:
    """
    Loads and preprocesses the train/val/test datasets.
    
    NEW: Properly handles epoch setting for streaming datasets.
    """
    # ... existing code ...
    
    # Load datasets
    train_pipeline_dataset = load_data_module(
        paths["train"], train_filename, config, use_streaming
    )
    val_pipeline_dataset = load_data_module(
        paths["val"], val_filename, config, use_streaming
    )
    test_pipeline_dataset = load_data_module(
        paths["test"], test_filename, config, use_streaming
    )
    
    # ... existing preprocessing code ...
    
    return (
        [train_pipeline_dataset, val_pipeline_dataset, test_pipeline_dataset],
        tokenizer,
        config,
    )


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main training function.
    
    NEW: Sets epoch for streaming datasets to ensure proper shuffling.
    """
    # ... existing setup code ...
    
    # Training loop
    log_once(logger, "Training starts using pytorch.lightning ...")
    
    # NEW: Custom training loop for streaming mode
    if use_streaming:
        log_once(logger, "Using custom epoch callbacks for streaming mode")
        
        # Create custom callback for epoch setting
        class StreamingEpochCallback(pl.Callback):
            def on_train_epoch_start(self, trainer, pl_module):
                # Set epoch on all datasets
                epoch = trainer.current_epoch
                for dataset in datasets:
                    if hasattr(dataset, 'set_epoch'):
                        dataset.set_epoch(epoch)
                        logger.debug(f"Set epoch={epoch} on {type(dataset).__name__}")
        
        # Add callback
        callbacks = trainer.callbacks + [StreamingEpochCallback()]
        trainer.callbacks = callbacks
    
    trainer = model_train(
        model,
        config_dict,
        train_dataloader,
        val_dataloader,
        device="auto",
        model_log_path=paths["checkpoint"],
        early_stop_metric=config.early_stop_metric,
    )
    
    # ... rest of training ...
```

#### 3.4.2 Validation Script

```python
# projects/names3risk_pytorch/dockers/scripts/validate_streaming_sharding.py
"""
Validation script to verify correct shard distribution in streaming mode.

Usage:
    python -m scripts.validate_streaming_sharding --data_dir /path/to/shards
"""
import argparse
import torch
import torch.distributed as dist
from processing.datasets.pipeline_iterable_datasets import PipelineIterableDataset


def main(args):
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Create dataset
    config = {
        "label_name": "label",
        "text_name": "text",
        "full_field_list": [],
        "cat_field_list": [],
        "tab_field_list": [],
    }
    
    dataset = PipelineIterableDataset(
        config=config,
        file_dir=args.data_dir,
        shuffle_shards=False,  # Disable for deterministic testing
    )
    
    # Get distribution info
    info = dataset.get_shard_distribution_info()
    
    print(f"\n{'='*60}")
    print(f"Rank {rank}/{world_size} Shard Distribution:")
    print(f"{'='*60}")
    print(f"Total shards: {info['total_shards']}")
    print(f"Shards for this rank: {info['shards_per_rank']}")
    print(f"Workers per rank: {info['num_workers']}")
    print(f"Shards per worker: {info['shards_per_worker']}")
    print(f"\nAssigned shards:")
    for shard in info['assigned_shards']:
        print(f"  - {shard}")
    print(f"{'='*60}\n")
    
    # Verify no overlap
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    success = dataset.verify_no_shard_overlap(loader, world_size, 1)
    
    if rank == 0:
        if success:
            print("✅ VALIDATION PASSED: No shard overlap detected")
            exit(0)
        else:
            print("❌ VALIDATION FAILED: Shard overlap detected!")
            exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
```

---

## 4. Testing and Validation

### 4.1 Unit Tests

```python
# test/processing/test_iterable_dataset_distributed.py
"""Unit tests for distributed PipelineIterableDataset."""
import pytest
import torch
import torch.distributed as dist
from pathlib import Path
import tempfile
import pandas as pd

from processing.datasets.pipeline_iterable_datasets import PipelineIterableDataset


@pytest.fixture
def temp_sharded_data():
    """Create temporary directory with mock shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create 10 mock shards
        for i in range(10):
            df = pd.DataFrame({
                "id": [f"id_{i}_{j}" for j in range(100)],
                "label": [j % 2 for j in range(100)],
                "text": [f"text_{i}_{j}" for j in range(100)],
            })
            df.to_parquet(tmpdir / f"part-{i:05d}.parquet")
        
        yield tmpdir


class TestIterableDatasetDistributed:
    """Test suite for distributed sharding."""
    
    def test_single_gpu_mode(self, temp_sharded_data):
        """Test that single GPU mode processes all shards."""
        config = {
            "label_name": "label",
            "text_name": "text",
            "full_field_list": [],
            "cat_field_list": [],
            "tab_field_list": [],
        }
        
        dataset = PipelineIterableDataset(
            config=config,
            file_dir=str(temp_sharded_data),
        )
        
        info = dataset.get_shard_distribution_info()
        
        assert info["total_shards"] == 10
        assert info["shards_per_rank"] == 10  # All shards
        assert info["rank"] == 0
        assert info["world_size"] == 1
    
    @pytest.mark.skipif(
        not torch.distributed.is_available(),
        reason="Distributed not available"
    )
    def test_multi_gpu_rank_sharding(self, temp_sharded_data):
        """Test that ranks get non-overlapping shards."""
        # This test requires launching with torchrun
        # pytest -k test_multi_gpu_rank_sharding --distributed
        
        if not dist.is_initialized():
            pytest.skip("Not in distributed mode")
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        config = {
            "label_name": "label",
            "text_name": "text",
            "full_field_list": [],
            "cat_field_list": [],
            "tab_field_list": [],
        }
        
        dataset = PipelineIterableDataset(
            config=config,
            file_dir=str(temp_sharded_data),
        )
        
        info = dataset.get_shard_distribution_info()
        
        # Each rank should get ceil(10 / world_size) or floor(10 / world_size) shards
        expected_min = 10 // world_size
        expected_max = (10 + world_size - 1) // world_size
        
        assert expected_min <= info["shards_per_rank"] <= expected_max
        
        # Verify round-robin distribution
        expected_shards = [f"part-{i:05d}.parquet" 
                          for i in range(rank, 10, world_size)]
        assert info["assigned_shards"] == expected_shards
    
    def test_deterministic_shuffling(self, temp_sharded_data):
        """Test that shuffling is deterministic with same seed."""
        config = {
            "label_name": "label",
            "text_name": "text",
            "full_field_list": [],
            "cat_field_list": [],
            "tab_field_list": [],
        }
        
        dataset1 = PipelineIterableDataset(
            config=config,
            file_dir=str(temp_sharded_data),
            shuffle_shards=True,
        )
        dataset1.set_epoch(0)
        
        dataset2 = PipelineIterableDataset(
            config=config,
            file_dir=str(temp_sharded_data),
            shuffle_shards=True,
        )
        dataset2.set_epoch(0)
        
        # Same epoch should give same shuffle order
        items1 = [item for item in dataset1]
        items2 = [item for item in dataset2]
        
        assert items1 == items2
    
    def test_epoch_changes_shuffle(self, temp_sharded_data):
        """Test that different epochs give different shuffle orders."""
        config = {
            "label_name": "label",
            "text_name": "text",
            "full_field_list": [],
            "cat_field_list": [],
            "tab_field_list": [],
        }
        
        dataset = PipelineIterableDataset(
            config=config,
            file_dir=str(temp_sharded_data),
            shuffle_shards=True,
        )
        
        dataset.set_epoch(0)
        items_epoch0 = [item for item in dataset]
        
        dataset.set_epoch(1)
        items_epoch1 = [item for item in dataset]
        
        # Different epochs should give different orders
        # (with very high probability for 1000 items)
        assert items_epoch0 != items_epoch1
```

### 4.2 Integration Tests

```python
# test/integration/test_streaming_fsdp_training.py
"""Integration test for streaming mode with FSDP."""
import pytest
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies import FSDPStrategy

from processing.datasets.pipeline_iterable_datasets import PipelineIterableDataset
from lightning_models.bimodal.pl_bimodal_bert import BimodalBert


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs"
)
class TestStreamingFSDP:
    """Test streaming mode with FSDP strategy."""
    
    def test_fsdp_with_streaming_dataset(self, temp_sharded_data, config):
        """Test that FSDP training works with PipelineIterableDataset."""
        # Create dataset
        train_dataset = PipelineIterableDataset(
            config=config,
            file_dir=str(temp_sharded_data),
            shuffle_shards=True,
        )
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            num_workers=2,
        )
        
        # Create model
        model = BimodalBert(config)
        
        # Create trainer with FSDP
        trainer = pl.Trainer(
            max_epochs=2,
            strategy=FSDPStrategy(),
            devices=2,
            accelerator="gpu",
        )
        
        # Train
        trainer.fit(model, train_loader)
        
        # Verify training completed without errors
        assert trainer.current_epoch == 2
    
    def test_no_data_duplication(self, temp_sharded_data, config):
        """Verify that different ranks see different data."""
        if not torch.distributed.is_initialized():
            pytest.skip("Not in distributed mode")
        
        rank = torch.distributed.get_rank()
        
        dataset = PipelineIterableDataset(
            config=config,
            file_dir=str(temp_sharded_data),
            shuffle_shards=False,  # Disable for deterministic check
        )
        
        # Collect all IDs seen by this rank
        seen_ids = set()
        for item in dataset:
            seen_ids.add(item["id"])
        
        # Gather from all ranks
        all_ids = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(all_ids, seen_ids)
        
        if rank == 0:
            # Check for overlap
            all_seen = set()
            for rank_ids in all_ids:
                overlap = all_seen & rank_ids
                assert len(overlap) == 0, f"Data duplication detected: {overlap}"
                all_seen.update(rank_ids)
```

### 4.3 Performance Benchmarks

```python
# test/performance/benchmark_streaming_modes.py
"""Benchmark streaming mode performance."""
import time
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy


def benchmark_training_throughput(
    dataset_type: str,  # "batch" or "streaming"
    strategy: str,       # "ddp" or "fsdp"
    num_gpus: int,
    batch_size: int,
    num_epochs: int = 3,
):
    """Benchmark training throughput."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {dataset_type.upper()} mode with {strategy.upper()}")
    print(f"GPUs: {num_gpus}, Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Create dataset
    if dataset_type == "batch":
        dataset = PipelineDataset(...)
    else:
        dataset = PipelineIterableDataset(...)
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    # Create model
    model = BimodalBert(config)
    
    # Create strategy
    if strategy == "ddp":
        training_strategy = DDPStrategy()
    else:
        training_strategy = FSDPStrategy()
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        strategy=training_strategy,
        devices=num_gpus,
        accelerator="gpu",
        enable_progress_bar=False,
    )
    
    # Benchmark
    start_time = time.time()
    trainer.fit(model, loader)
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    samples_per_sec = (len(dataset) * num_epochs) / total_time
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {samples_per_sec:.2f} samples/sec")
    print(f"  Throughput per GPU: {samples_per_sec/num_gpus:.2f} samples/sec")
    
    return {
        "total_time": total_time,
        "throughput": samples_per_sec,
        "throughput_per_gpu": samples_per_sec / num_gpus,
    }


if __name__ == "__main__":
    results = {}
    
    # Benchmark different configurations
    for dataset_type in ["batch", "streaming"]:
        for strategy in ["ddp", "fsdp"]:
            for num_gpus in [1, 2, 4, 8]:
                key = f"{dataset_type}_{strategy}_{num_gpus}gpu"
                results[key] = benchmark_training_throughput(
                    dataset_type=dataset_type,
                    strategy=strategy,
                    num_gpus=num_gpus,
                    batch_size=32,
                )
    
    # Print comparison
    print(f"\n{'='*60}")
    print("Performance Comparison")
    print(f"{'='*60}\n")
    
    import pandas as pd
    df = pd.DataFrame(results).T
    print(df.to_string())
```

---

## 5. Migration Guide

### 5.1 Backward Compatibility

The proposed changes are **backward compatible**:

- ✅ Single GPU training: No changes needed (rank=0, world_size=1)
- ✅ Existing batch mode: Unaffected (uses `PipelineDataset`)
- ✅ Non-distributed streaming: Works as before
- ⚠️ Distributed streaming: Will now work correctly (was broken before)

### 5.2 Upgrade Checklist

For users currently using streaming mode with FSDP/DDP:

- [ ] **Step 1**: Update `PipelineIterableDataset` with new `__iter__()` implementation
- [ ] **Step 2**: Add `set_epoch()` calls in training loop (for proper shuffling)
- [ ] **Step 3**: Run validation script to verify no shard overlap
- [ ] **Step 4**: Run integration tests with FSDP/DDP
- [ ] **Step 5**: Benchmark throughput to ensure performance
- [ ] **Step 6**: Update documentation and examples

### 5.3 Configuration Changes

**No configuration changes required!**

The fix is transparent to users - existing code will automatically benefit:

```python
# Before (may have data duplication bug)
train_dataset = PipelineIterableDataset(
    config=config,
    file_dir="/data/train",
)

# After (automatically fixed)
train_dataset = PipelineIterableDataset(
    config=config,
    file_dir="/data/train",
)  # Same code, now works correctly with FSDP/DDP!
```

**Optional enhancement** (for epoch-aware shuffling):

```python
# In training loop
for epoch in range(num_epochs):
    train_dataset.set_epoch(epoch)  # NEW: Set epoch for deterministic shuffling
    
    for batch in train_loader:
        # Training step
        pass
```

---

## 6. Performance Analysis

### 6.1 Expected Performance Impact

| Aspect | Before Fix | After Fix | Change |
|--------|-----------|-----------|--------|
| **Single GPU** | Baseline | Baseline | No change |
| **Memory per GPU** | Same | Same | No change |
| **Throughput (DDP/FSDP)** | Incorrect (duplicated data) | Correct (unique data per rank) | ✅ **Fixed** |
| **Communication overhead** | Same | Same | No change |
| **Startup time** | Same | +negligible (rank check) | Negligible |

### 6.2 Throughput Improvements

**Scenario**: 8 GPUs with FSDP, 100 shards

**Before Fix (BROKEN)**:
```
Each rank processes: 100 shards (duplicated)
Total unique data: 100 shards
Effective throughput: 1× (all GPUs waste compute on duplicates)
Training time: T
```

**After Fix (CORRECT)**:
```
Each rank processes: 12-13 shards (distributed)
Total unique data: 100 shards (no duplication)
Effective throughput: 8× (each GPU processes unique data)
Training time: T/8 (approximately)
```

**Result**: ~8× speedup from fixing the bug!

### 6.3 Memory Efficiency

No change in memory usage - the fix only affects data distribution, not memory footprint.

---

## 7. Future Enhancements

### 7.1 Advanced Sharding Strategies

**Current**: Round-robin shard distribution
```python
shards_for_rank = shard_files[rank::world_size]
```

**Future Option 1**: Size-aware distribution (balance by shard size, not count)
```python
def distribute_shards_by_size(shard_files, rank, world_size):
    """Distribute shards to balance total data size per rank."""
    shard_sizes = [get_shard_size(f) for f in shard_files]
    # Use greedy bin packing algorithm
    assignments = balance_by_size(shard_sizes, world_size)
    return [shard_files[i] for i in assignments[rank]]
```

**Future Option 2**: Locality-aware distribution (for networked storage)
```python
def distribute_shards_by_locality(shard_files, rank, world_size):
    """Distribute shards based on storage locality."""
    # Prefer shards on local disk vs network storage
    local_shards, remote_shards = partition_by_locality(shard_files)
    # Assign local shards first
    ...
```

### 7.2 Dynamic Rebalancing

**Problem**: Uneven shard sizes can cause load imbalance

**Solution**: Dynamic work stealing
```python
class DynamicShardIterator:
    """Iterator with dynamic work stealing for load balancing."""
    
    def __init__(self, shard_queue: Queue, rank: int):
        self.shard_queue = shard_queue
        self.rank = rank
    
    def __iter__(self):
        while True:
            try:
                # Try to get next shard with timeout
                shard = self.shard_queue.get(timeout=1.0)
                yield from process_shard(shard)
            except queue.Empty:
                # Check if other ranks still have work
                if all_ranks_finished():
                    break
                # Try to steal work from busy ranks
                stolen_shard = steal_work_from_busy_rank()
                if stolen_shard:
                    yield from process_shard(stolen_shard)
```

### 7.3 Prefetching and Overlapping

**Current**: Sequential shard loading
```python
for shard in shards:
    df = load_shard(shard)  # Blocking I/O
    yield from process_rows(df)
```

**Future**: Asynchronous prefetching
```python
from concurrent.futures import ThreadPoolExecutor

class PrefetchingShardIterator:
    """Iterator with asynchronous shard prefetching."""
    
    def __init__(self, shards, prefetch_factor=2):
        self.shards = shards
        self.prefetch_factor = prefetch_factor
        self.executor = ThreadPoolExecutor(max_workers=prefetch_factor)
    
    def __iter__(self):
        # Start prefetching first N shards
        futures = []
        for shard in self.shards[:self.prefetch_factor]:
            futures.append(self.executor.submit(load_shard, shard))
        
        # Process shards as they complete
        for i, shard in enumerate(self.shards):
            # Get result from prefetch
            df = futures[i].result()
            
            # Start prefetching next shard
            next_idx = i + self.prefetch_factor
            if next_idx < len(self.shards):
                futures.append(
                    self.executor.submit(load_shard, self.shards[next_idx])
                )
            
            yield from process_rows(df)
```

### 7.4 Checkpointing and Resume

**Current**: No checkpoint support for streaming

**Future**: Stateful iteration with checkpointing
```python
class CheckpointableIterableDataset(PipelineIterableDataset):
    """IterableDataset with checkpoint/resume support."""
    
    def __init__(self, ...):
        super().__init__(...)
        self._processed_shards = set()
        self._current_shard_position = 0
    
    def state_dict(self) -> Dict:
        """Get current iteration state for checkpointing."""
        return {
            "processed_shards": list(self._processed_shards),
            "current_shard": self._current_shard,
            "current_position": self._current_shard_position,
            "epoch": self._current_epoch,
        }
    
    def load_state_dict(self, state: Dict):
        """Resume from checkpoint."""
        self._processed_shards = set(state["processed_shards"])
        self._current_shard = state["current_shard"]
        self._current_shard_position = state["current_position"]
        self._current_epoch = state["epoch"]
    
    def __iter__(self):
        # Skip already processed shards
        for shard in self.shards:
            if shard in self._processed_shards:
                continue
            
            df = load_shard(shard)
            
            # Resume from saved position
            start_idx = (self._current_shard_position 
                        if shard == self._current_shard else 0)
            
            for idx in range(start_idx, len(df)):
                yield process_row(df.iloc[idx])
                self._current_shard_position = idx + 1
            
            self._processed_shards.add(shard)
            self._current_shard_position = 0
```

---

## 8. Documentation Updates

### 8.1 Code Documentation

Update docstrings in `pipeline_iterable_datasets.py`:

```python
class PipelineIterableDataset(IterableDataset):
    """
    Streaming dataset for multimodal input with distributed training support.
    
    This dataset provides memory-efficient data loading by streaming from
    multiple shard files. It supports distributed training (DDP/FSDP) through
    automatic rank-based and worker-based shard distribution.
    
    **Distributed Training Support**:
    
    The dataset implements two-tier sharding for distributed training:
    
    1. **Rank-based sharding**: Shards are distributed across GPU ranks using
       round-robin assignment (rank 0 gets shards [0, world_size, 2*world_size, ...]).
       This ensures each GPU processes unique data in FSDP/DDP training.
    
    2. **Worker-based sharding**: Within each rank, shards are further distributed
       across DataLoader workers for parallel loading.
    
    **Example Usage**:
    
    Single GPU training:
        >>> dataset = PipelineIterableDataset(
        ...     config=config,
        ...     file_dir="/data/train",
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
    
    Distributed training (FSDP):
        >>> # Same code! Distribution happens automatically
        >>> dataset = PipelineIterableDataset(
        ...     config=config,
        ...     file_dir="/data/train",
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
        >>> 
        >>> trainer = pl.Trainer(strategy=FSDPStrategy(), devices=8)
        >>> trainer.fit(model, loader)
    
    Epoch-aware shuffling:
        >>> for epoch in range(num_epochs):
        ...     dataset.set_epoch(epoch)  # Important for deterministic shuffling
        ...     for batch in loader:
        ...         # Training step
    
    **Performance Characteristics**:
    
    - Memory: O(shard_size) per worker (not O(dataset_size))
    - I/O: Streamed from disk, one shard at a time
    - Throughput: Scales linearly with number of GPUs (with proper sharding)
    
    **Comparison with PipelineDataset**:
    
    | Aspect | PipelineDataset | PipelineIterableDataset |
    |--------|-----------------|-------------------------|
    | Memory | O(dataset_size) | O(shard_size) |
    | Access | Random (`__getitem__`) | Sequential (`__iter__`) |
    | Use case | Fits in memory | Large datasets |
    | Distributed | DistributedSampler | Manual rank sharding |
    
    Args:
        config: Configuration dictionary
        file_dir: Directory containing shard files
        filename: Optional single file (for backward compatibility)
        dataframe: Optional DataFrame (for testing)
        processor_pipelines: Pre-configured processors
        shard_pattern: Glob pattern for finding shards (default: "part-*.parquet")
        shuffle_shards: Whether to shuffle shard order (default: False)
    
    Attributes:
        shard_files: List of shard file paths
        processor_pipelines: Dict of field-specific processors
        shuffle_shards: Whether shuffling is enabled
        _current_
