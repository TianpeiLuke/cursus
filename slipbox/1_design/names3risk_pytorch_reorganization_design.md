---
tags:
  - design
  - refactoring
  - names3risk
  - code-organization
  - modularity
  - fraud-detection
keywords:
  - names3risk reorganization
  - pytorch modules
  - atomic components
  - lstm2risk
  - transformer2risk
  - bimodal architecture
  - code modularization
topics:
  - software architecture
  - code reorganization
  - fraud detection models
  - modular design
  - component extraction
language: python
date of note: 2026-01-04
---

# Names3Risk PyTorch Code Reorganization Design

## Overview

This document outlines the systematic reorganization of the Names3Risk legacy codebase into atomic, reusable PyTorch components following Zettelkasten knowledge management principles. The reorganization addresses code duplication between LSTM and Transformer implementations, improves component discoverability, and establishes a foundation for future fraud detection model development.

**Project**: Names3Risk - First-time buyer fraud detection using multi-modal (text + tabular) analysis

**Current State**: Monolithic model files (`lstm2risk.py`, `transformer2risk.py`) with duplicated components and tight coupling

**Target State**: Atomic component library in `projects/names3risk_pytorch/pytorch/` with application models consuming reusable building blocks

## Related Documents

### Foundational Design Principles
- **[PyTorch Module Reorganization Design](pytorch_module_reorganization_design.md)** - Core organizational principles and patterns guiding this refactoring
- **[Zettelkasten Knowledge Management Principles](../6_resources/zettelkasten_knowledge_management_principles.md)** - Atomicity, connectivity, and anti-category principles

### Names3Risk Model Documentation
- **[Names3Risk Model Design](names3risk_model_design.md)** - **PRIMARY** - Comprehensive architecture documentation for the fraud detection model
- **[Names3Risk Cursus Step Equivalency Analysis](../4_analysis/2025-12-31_names3risk_cursus_step_equivalency_analysis.md)** - Analysis of Names3Risk pipeline components and Cursus framework integration

### Related Refactoring Work
- **[TSA Lightning Refactoring Design](tsa_lightning_refactoring_design.md)** - Similar refactoring effort for Temporal Self-Attention models
- **[Algorithm Preserving Refactoring SOP](../6_resources/algorithm_preserving_refactoring_sop.md)** - Safe refactoring procedures

## Problem Statement

### Current Architecture Issues

The Names3Risk legacy codebase (`projects/names3risk_legacy/`) exhibits several architectural problems that impede development velocity and code quality:

#### 1. Code Duplication (Critical)

**AttentionPooling Duplication:**
- Appears in `lstm2risk.py` (99 lines)
- Appears in `transformer2risk.py` (103 lines)
- **Difference**: Input dimension parameterization (`2 * hidden_size` vs `embedding_size`)
- **Impact**: Bug fixes must be applied twice, behavior may diverge over time

**ResidualBlock Duplication:**
- Appears in `lstm2risk.py` (112-122)
- Appears in `transformer2risk.py` (146-158)
- **Difference**: LSTM version uses LayerNorm pre-activation, Transformer version uses dropout
- **Impact**: Inconsistent residual connection patterns across model types

#### 2. Monolithic File Structure

**lstm2risk.py (180 lines):**
```
- LSTMConfig (dataclass)
- AttentionPooling (class, 18 lines)
- ResidualBlock (class, 11 lines)
- TextProjection (class, 43 lines)
- LSTM2Risk (class, 81 lines)
- create_collate_fn (static method, 27 lines)
```

**transformer2risk.py (245 lines):**
```
- TransformerConfig (dataclass)
- FeedForward (class, 14 lines)
- Head (class, 27 lines)
- MultiHeadAttention (class, 16 lines)
- Block (class, 15 lines)
- ResidualBlock (class, 13 lines)
- AttentionPooling (class, 14 lines)
- TextProjection (class, 34 lines)
- Transformer2Risk (class, 71 lines)
- create_collate_fn (method, 22 lines)
```

**Issues:**
- Mixing atomic components (AttentionPooling) with composite blocks (TextProjection) with application models (LSTM2Risk)
- No clear separation between reusable components and application-specific code
- Difficult to discover what components exist without reading entire files
- Hard to import individual components without bringing in unrelated dependencies

#### 3. Limited Reusability

**Current Import Pattern:**
```python
# To use AttentionPooling from LSTM model
from projects.names3risk_legacy.lstm2risk import AttentionPooling

# Problem: Also imports LSTMConfig, ResidualBlock, TextProjection, LSTM2Risk
# Risk: Unintended dependencies on LSTM-specific implementations
```

**Desired Import Pattern:**
```python
# Clean, atomic import
from projects.names3risk_pytorch.dockers.pytorch.pooling import AttentionPooling

# Clear: Only imports the attention pooling component
# Reusable: Can be used in LSTM, Transformer, or any future model
```

#### 4. Unclear Component Dependencies

**Hidden Dependencies in lstm2risk.py:**
- `TextProjection` depends on `AttentionPooling` (implicit coupling via same file)
- `LSTM2Risk` depends on `TextProjection` and `ResidualBlock` (implicit coupling)
- No explicit import statements document these relationships

**Hidden Dependencies in transformer2risk.py:**
- `Block` depends on `MultiHeadAttention` and `FeedForward` (implicit coupling)
- `TextProjection` depends on `Block` and `AttentionPooling` (implicit coupling)
- `Transformer2Risk` depends on `TextProjection` and `ResidualBlock` (implicit coupling)

**Impact:**
- Developers must read entire files to understand component relationships
- Refactoring one component risks breaking dependent components
- Circular dependencies are difficult to detect

## Proposed Architecture

### Target Directory Structure

```
projects/names3risk_pytorch/
├── __init__.py                          # Project-level public API
├── README.md                            # Project overview and quick start
├── pipeline_configs/                    # Pipeline configurations
│   └── ...
│
└── dockers/                             # Docker container code (training/inference scripts)
    ├── __init__.py
    │
    ├── pytorch/                         # Atomic, reusable PyTorch components
    │   ├── __init__.py                 # Component library public API
    │   │
    │   ├── attention/                   # Attention mechanisms
    │   │   ├── __init__.py
    │   │   ├── attention_head.py       # Single attention head (from Head)
    │   │   └── multihead_attention.py  # Multi-head attention (from MultiHeadAttention)
    │   │
    │   ├── embeddings/                  # Embedding layers
    │   │   ├── __init__.py
    │   │   ├── token_embedding.py      # Token embeddings (extracted from TextProjection)
    │   │   └── positional_encoding.py  # Positional embeddings (extracted from TextProjection)
    │   │
    │   ├── feedforward/                 # Feedforward networks
    │   │   ├── __init__.py
    │   │   ├── mlp_block.py            # Standard MLP (from FeedForward)
    │   │   └── residual_block.py       # Unified residual block (merge both versions)
    │   │
    │   ├── pooling/                     # Pooling mechanisms
    │   │   ├── __init__.py
    │   │   └── attention_pooling.py    # Unified attention pooling (merge both versions)
    │   │
    │   ├── blocks/                      # Composite building blocks
    │   │   ├── __init__.py
    │   │   ├── transformer_block.py    # Complete transformer layer (from Block)
    │   │   ├── lstm_encoder.py         # LSTM-based text encoder (from LSTM TextProjection)
    │   │   └── transformer_encoder.py  # Transformer-based text encoder (from Transformer TextProjection)
    │   │
    │   └── fusion/                      # Multi-modal fusion mechanisms
    │       ├── __init__.py
    │       └── bimodal_concat.py       # Text + Tabular concatenation fusion
    │
    ├── hyperparams/                     # Hyperparameter classes (Cursus three-tier pattern)
    │   ├── __init__.py
    │   ├── hyperparameters_base.py     # Base hyperparameters (already exists)
    │   ├── hyperparameters_bimodal.py  # Bimodal base (already exists)
    │   ├── hyperparameters_lstm2risk.py     # LSTM2Risk-specific hyperparameters
    │   └── hyperparameters_transformer2risk.py  # Transformer2Risk-specific hyperparameters
    │
    ├── lightning_models/                # PyTorch Lightning model modules
    │   ├── __init__.py
    │   ├── bimodal/                     # Bimodal models (text + tabular fraud detection)
    │   │   ├── __init__.py
    │   │   ├── pl_bimodal_bert.py      # Existing BERT-based bimodal (already exists)
    │   │   ├── pl_lstm2risk.py         # NEW: LSTM2Risk Lightning module
    │   │   └── pl_transformer2risk.py  # NEW: Transformer2Risk Lightning module
    │   ├── tabular/                     # Tabular-only models
    │   ├── text/                        # Text-only models
    │   ├── trimodal/                    # Three-modality models
    │   └── utils/                       # Lightning utilities
    │
    ├── tokenizers/                      # Tokenization components (at dockers level)
    │   ├── __init__.py
    │   └── bpe_tokenizer.py            # BPE with compression tuning (from OrderTextTokenizer)
    │
    ├── processing/                      # Data processing components
    │   ├── __init__.py
    │   ├── processors.py               # Existing processor implementations
    │   │
    │   ├── dataloaders/                # DataLoader and collate functions
    │   │   ├── __init__.py
    │   │   ├── pipeline_dataloader.py  # Existing: build_collate_batch for pipelines
    │   │   └── names3risk_collate.py   # NEW: Names3Risk model collate functions
    │   │
    │   └── datasets/                    # Dataset implementations
    │       ├── __init__.py
    │       ├── pipeline_datasets.py    # Existing pipeline datasets
    │       ├── text_dataset.py         # Text dataset wrapper (from TextDataset)
    │       ├── tabular_dataset.py      # Tabular dataset wrapper (from TabularDataset)
    │       └── bimodal_dataset.py      # NEW: Combined text+tabular dataset
```

### Design Principles Applied

Following the **Five Core Zettelkasten Principles** from the PyTorch Module Reorganization Design:

#### 1. Principle of Atomicity
- **One Module = One Concept**: Each `.py` file contains exactly one conceptual component
- **Example**: `attention_pooling.py` contains ONLY attention pooling logic, not embedding or classification
- **Benefit**: Can import and test `AttentionPooling` without bringing in unrelated dependencies

#### 2. Principle of Connectivity
- **Explicit Dependencies**: All relationships declared via imports
- **Example**: `lstm_encoder.py` explicitly imports `from ..pooling import AttentionPooling`
- **Benefit**: Following imports reveals the dependency graph of the codebase

#### 3. Principle Against Categories
- **Flat Structure**: Maximum 2 levels deep (`pytorch/attention/` not `pytorch/layers/attention/mechanisms/`)
- **Semantic Groupings**: Folders describe function (`attention/`, `pooling/`) not hierarchy
- **Benefit**: Easy navigation without remembering arbitrary taxonomies

#### 4. Principle of Manual Linking
- **Documented Relationships**: Every module's docstring lists dependencies and consumers
- **Connection Registry**: Component index document maps concepts to implementations
- **Benefit**: Understanding "why this connection exists" without searching

#### 5. Principle of Dual-Form Structure
- **Code as Inner Form**: PyTorch implementation
- **Metadata as Outer Form**: Comprehensive docstrings with usage examples
- **Benefit**: Both humans and tools can understand the structure

## Component Migration Map

### Phase 1: Extract Duplicated Components (High Priority)

#### 1.1 AttentionPooling → `pytorch/pooling/attention_pooling.py`

**Status**: 🔴 **CRITICAL DUPLICATE**

**Current Locations:**
- `lstm2risk.py` lines 18-35: Uses `2 * config.hidden_size` as input dimension
- `transformer2risk.py` lines 133-145: Uses `config.embedding_size` as input dimension

**Key Differences:**
```python
# LSTM version
class AttentionPooling(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.attention = nn.Linear(2 * config.hidden_size, 1)  # Hardcoded dimension

# Transformer version
class AttentionPooling(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = nn.Linear(config.embedding_size, 1)  # Different hardcoded dimension
```

**Unified Implementation:**
```python
# pytorch/pooling/attention_pooling.py
"""
Attention-Weighted Sequence Pooling

Pools variable-length sequences into fixed-size representations using learned attention weights.

**Core Concept:**
Instead of simple max/mean pooling, learns to weight each sequence element by importance.
Essential for fraud detection where specific name patterns or email characteristics may be
critical fraud indicators.

**Architecture:**
1. Project each sequence element to attention score via linear layer
2. Apply sequence mask to ignore padding tokens
3. Normalize scores with softmax
4. Compute weighted sum of sequence elements

**Parameters:**
- input_dim (int): Dimension of input sequence elements
- dropout (float): Dropout probability for attention scores (default: 0.0)

**Forward Signature:**
Input:
  - sequence: (B, L, D) - Batch of sequences
  - lengths: (B,) - Actual lengths before padding (optional)
  
Output:
  - pooled: (B, D) - Pooled representations

**Dependencies:**
- torch.nn.Linear → Attention score projection
- torch.nn.functional.softmax → Score normalization

**Used By:**
- names3risk_pytorch.dockers.pytorch.blocks.lstm_encoder → LSTM sequence summarization
- names3risk_pytorch.dockers.pytorch.blocks.transformer_encoder → Transformer sequence summarization

**Alternative Approaches:**
- Mean pooling → Simpler but weights all tokens equally
- Max pooling → Takes most salient token but ignores others
- Last token → Simple but loses sequence context
- [CLS] token (transformers) → Requires special token, less flexible

**Usage Example:**
```python
from names3risk_pytorch.pytorch.pooling import AttentionPooling

# Create attention pooling layer
pooling = AttentionPooling(input_dim=256, dropout=0.1)

# Pool variable-length sequences (e.g., customer names)
sequences = torch.randn(32, 50, 256)  # (batch=32, max_len=50, dim=256)
lengths = torch.tensor([30, 45, 20, ...])  # Actual lengths before padding

pooled = pooling(sequences, lengths)  # (32, 256)
```

**References:**
- "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015)
- "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionPooling(nn.Module):
    """
    Attention-weighted sequence pooling.
    
    Learns to weight sequence elements by importance, then computes weighted sum.
    Handles variable-length sequences via masking.
    """
    
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.0
    ):
        """
        Initialize AttentionPooling.
        
        Args:
            input_dim: Dimension of input sequence elements (e.g., 256 for LSTM output,
                      128 for transformer embeddings)
            dropout: Dropout probability for attention scores (0.0 = no dropout)
        """
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence using learned attention weights.
        
        Args:
            sequence: (B, L, D) - Input sequences
            lengths: (B,) - Actual sequence lengths before padding (optional)
                    If None, assumes all sequences have same length
            
        Returns:
            pooled: (B, D) - Pooled representations
        """
        # Compute attention scores: (B, L, 1)
        scores = self.attention(sequence)
        
        # Apply mask to ignore padding if lengths provided
        if lengths is not None:
            # Create mask: (B, L) where True = valid token, False = padding
            mask = torch.arange(sequence.size(1), device=sequence.device).unsqueeze(0) < lengths.unsqueeze(1)
            # Mask out padding tokens by setting their scores to -inf
            scores = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        # Normalize scores with softmax: (B, L, 1)
        weights = F.softmax(scores, dim=1)
        weights = self.dropout(weights)
        
        # Weighted sum: (B, D)
        pooled = torch.sum(weights * sequence, dim=1)
        
        return pooled
```

**Migration Strategy:**
1. Create unified `attention_pooling.py` with parameterized `input_dim`
2. Update `lstm_encoder.py` to use: `AttentionPooling(input_dim=2 * config.hidden_size)`
3. Update `transformer_encoder.py` to use: `AttentionPooling(input_dim=config.embedding_size)`
4. Add deprecation warnings to legacy `lstm2risk.py` and `transformer2risk.py`
5. Run full test suite to verify no behavioral changes

#### 1.2 ResidualBlock → `pytorch/feedforward/residual_block.py`

**Status**: 🔴 **CRITICAL DUPLICATE**

**Current Locations:**
- `lstm2risk.py` lines 38-48: LayerNorm → Linear → ReLU → Linear (4x expansion)
- `transformer2risk.py` lines 146-158: Linear → ReLU → Linear → Dropout (1x expansion)

**Key Differences:**
```python
# LSTM version (pre-norm, 4x expansion)
class ResidualBlock(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.LayerNorm(4 * config.hidden_size),
            nn.Linear(4 * config.hidden_size, 16 * config.hidden_size),
            nn.ReLU(),
            nn.Linear(16 * config.hidden_size, 4 * config.hidden_size),
        )

# Transformer version (post-norm, 1x expansion)
class ResidualBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * config.hidden_size, 4 * config.hidden_size),
            nn.ReLU(),
            nn.Linear(4 * config.hidden_size, 4 * config.hidden_size),
            nn.Dropout(config.dropout_rate),
        )
```

**Unified Implementation:**
```python
# pytorch/feedforward/residual_block.py
"""
Residual Feedforward Block

Feedforward network with residual/skip connection for training stability.

**Core Concept:**
Applies non-linear transformation while preserving input via residual connection.
Enables deeper networks by mitigating vanishing gradient problem. Critical for
Names3Risk multi-layer classifier to learn complex fraud patterns.

**Architecture:**
- Pre-norm variant: LayerNorm → FFN → Residual connection
- Post-norm variant: FFN → Dropout → Residual connection

**Parameters:**
- dim (int): Input/output dimension
- expansion_factor (int): Hidden layer size multiplier (default: 4)
- dropout (float): Dropout probability (default: 0.0)
- activation (str): Activation function - 'relu', 'gelu', 'silu' (default: 'relu')
- norm_first (bool): Apply normalization before FFN (default: True)

**Forward Signature:**
Input:
  - x: (B, D) - Input features
  
Output:
  - output: (B, D) - x + FFN(x) or x + FFN(LayerNorm(x))

**Dependencies:**
- torch.nn.Linear → Feedforward layers
- torch.nn.LayerNorm → Optional normalization
- torch.nn.Dropout → Optional dropout

**Used By:**
- names3risk_pytorch.dockers.lightning_models.bimodal.pl_lstm2risk → LSTM2Risk Lightning module
- names3risk_pytorch.dockers.lightning_models.bimodal.pl_transformer2risk → Transformer2Risk Lightning module

**Alternative Approaches:**
- Plain FFN → No residual, harder to train deep networks
- Highway networks → Learnable gating, more parameters
- DenseNet connections → Concatenation instead of addition

**Usage Example:**
```python
from names3risk_pytorch.pytorch.feedforward import ResidualBlock

# Pre-norm residual block (LSTM style)
block = ResidualBlock(
    dim=512,
    expansion_factor=4,  # 512 -> 2048 -> 512
    norm_first=True
)

# Post-norm residual block (Transformer style)
block = ResidualBlock(
    dim=512,
    expansion_factor=1,  # 512 -> 512 -> 512
    dropout=0.2,
    norm_first=False
)

x = torch.randn(32, 512)
output = block(x)  # (32, 512)


**References:**
- "Deep Residual Learning for Image Recognition" (He et al., 2016)
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
"""

import torch
import torch.nn as nn
from typing import Literal


class ResidualBlock(nn.Module):
    """
    Feedforward block with residual connection.
    
    Supports both pre-norm and post-norm variants, configurable activation
    and expansion factor.
    """
    
    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
        activation: Literal['relu', 'gelu', 'silu'] = 'relu',
        norm_first: bool = True
    ):
        """
        Initialize ResidualBlock.
        
        Args:
            dim: Input/output dimension
            expansion_factor: Hidden layer size = dim * expansion_factor
            dropout: Dropout probability after second linear layer
            activation: Activation function name
            norm_first: If True, apply LayerNorm before FFN (pre-norm)
        """
        super().__init__()
        self.norm_first = norm_first
        hidden_dim = dim * expansion_factor
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Optional pre-norm
        if norm_first:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: (B, D) - Input features
            
        Returns:
            output: (B, D) - x + FFN(x) or x + FFN(LayerNorm(x))
        """
        if self.norm_first:
            return x + self.ffn(self.norm(x))
        else:
            return x + self.ffn(x)
```

**Migration Strategy:**
1. Create unified `residual_block.py` with configurable normalization and expansion
2. Update LSTM2Risk: `ResidualBlock(dim=4*hidden_size, expansion_factor=4, norm_first=True)`
3. Update Transformer2Risk: `ResidualBlock(dim=4*hidden_size, expansion_factor=1, dropout=0.2, norm_first=False)`
4. Verify identical behavior via numerical comparison tests
5. Add deprecation warnings to legacy implementations

### Phase 2: Extract Unique Atomic Components

#### 2.1 FeedForward (Transformer) → `pytorch/feedforward/mlp_block.py`

**Current Location:** `transformer2risk.py` lines 18-28

**Extraction Strategy:** Direct move with minor generalization for activation function

```python
# pytorch/feedforward/mlp_block.py
"""
Multi-Layer Perceptron Block

Standard feedforward network: Linear → Activation → Dropout → Linear → Dropout

**Core Concept:**
Position-wise feedforward network applied to each token independently.
Standard component in transformer architectures. For Names3Risk, processes
each character/subword embedding to capture non-linear patterns.

**Parameters:**
- input_dim (int): Input dimension
- hidden_dim (int): Hidden layer dimension (typically 4x input_dim)
- dropout (float): Dropout probability (default: 0.0)
- activation (str): Activation function (default: 'relu')

**Forward Signature:**
Input:
  - x: (B, L, D) or (B, D) - Input features
  
Output:
  - output: Same shape as input

**Dependencies:**
- torch.nn.Linear → Feedforward layers
- torch.nn.Dropout → Regularization

**Used By:**
- names3risk_pytorch.dockers.pytorch.blocks.transformer_block → Transformer FFN component

**Alternative Approaches:**
- names3risk_pytorch.pytorch.feedforward.residual_block → With skip connection
- Gated feedforward (GLU, SwiGLU) → Learnable gating

**Usage Example:**
```python
from names3risk_pytorch.pytorch.feedforward import MLPBlock

mlp = MLPBlock(input_dim=128, hidden_dim=512, dropout=0.2)
x = torch.randn(32, 50, 128)  # (batch, seq_len, dim)
output = mlp(x)  # (32, 50, 128)
"""

import torch
import torch.nn as nn
from typing import Literal


class MLPBlock(nn.Module):
    """Multi-layer perceptron block."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: Literal['relu', 'gelu', 'silu'] = 'relu'
    ):
        super().__init__()
        
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

#### 2.2 Head → `pytorch/attention/attention_head.py`

**Current Location:** `transformer2risk.py` lines 31-52

**Extraction Strategy:** Direct move, add comprehensive documentation

```python
# pytorch/attention/attention_head.py
"""
Single Attention Head

Computes scaled dot-product attention for one head in multi-head attention.

**Core Concept:**
Learns to focus on relevant parts of the input sequence via Query-Key-Value mechanism.
For Names3Risk, helps identify which characters/subwords in names/emails are most
indicative of fraud patterns.

**Architecture:**
1. Project input to Q, K, V via single linear layer
2. Compute attention scores: Q @ K^T / sqrt(d_k)
3. Apply optional mask (for padding or causal masking)
4. Softmax to get attention weights
5. Apply weights to V: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V

**Parameters:**
- embedding_dim (int): Input embedding dimension
- n_heads (int): Number of heads in parent MultiHeadAttention
- dropout (float): Dropout for attention weights (default: 0.0)

**Forward Signature:**
Input:
  - x: (B, L, D) - Input sequence
  - attn_mask: (B, L) - Attention mask (optional, True = attend, False = ignore)
  
Output:
  - output: (B, L, head_size) where head_size = embedding_dim // n_heads

**Dependencies:**
- torch.nn.Linear → Q, K, V projection
- torch.nn.functional.softmax → Attention weight normalization

**Used By:**
- names3risk_pytorch.dockers.pytorch.attention.multihead_attention → Combines multiple heads

**Alternative Approaches:**
- Additive attention (Bahdanau) → Concat Q,K then MLP instead of dot product
- Linear attention → Approximations for O(N) complexity

**Usage Example:**
```python
from names3risk_pytorch.pytorch.attention import AttentionHead

head = AttentionHead(embedding_dim=128, n_heads=8, dropout=0.1)
x = torch.randn(32, 50, 128)  # (batch, seq_len, embed_dim)
attn_mask = torch.ones(32, 50).bool()  # All tokens are valid

output = head(x, attn_mask)  # (32, 50, 16) where 16 = 128 // 8


**References:**
- "Attention Is All You Need" (Vaswani et al., 2017) - Original scaled dot-product attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionHead(nn.Module):
    """
    Single scaled dot-product attention head.
    
    Computes Q, K, V projections and attention for one head.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        dropout: float = 0.0
    ):
        """
        Initialize AttentionHead.
        
        Args:
            embedding_dim: Input embedding dimension (must be divisible by n_heads)
            n_heads: Number of heads in parent MultiHeadAttention (for dimension calculation)
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        
        assert embedding_dim % n_heads == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})"
        
        self.head_size = embedding_dim // n_heads
        
        # Single linear layer projects to Q, K, V (concatenated)
        self.qkv = nn.Linear(embedding_dim, 3 * self.head_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention for this head.
        
        Args:
            x: (B, L, D) - Input sequence
            attn_mask: (B, L) - Attention mask, True = attend, False = ignore
            
        Returns:
            output: (B, L, head_size)
        """
        B, L, D = x.shape
        
        # Project to Q, K, V and split: (B, L, 3 * head_size)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # Compute attention scores: (B, L, L)
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        
        # Apply mask if provided
        if attn_mask is not None:
            # attn_mask shape: (B, L) -> expand to (B, 1, L) for broadcasting
            scores = scores.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))
        
        # Normalize and apply dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Apply attention to values: (B, L, head_size)
        output = weights @ v
        
        return output
```

**Migration Strategy:**
1. Extract `Head` class to `attention_head.py`
2. Parameterize to accept `embedding_dim` and `n_heads`
3. Add comprehensive docstring with fraud detection context
4. Write unit tests for attention computation
5. Update `multihead_attention.py` to import and use

#### 2.3 MultiHeadAttention → `pytorch/attention/multihead_attention.py`

**Current Location:** `transformer2risk.py` lines 55-65

**Dependencies:** Requires `AttentionHead` from 2.2

**Extraction Strategy:** Direct move, import `AttentionHead`

```python
# pytorch/attention/multihead_attention.py
"""
Multi-Head Self-Attention

Combines multiple attention heads with output projection.

**Core Concept:**
Allows model to jointly attend to information from different representation subspaces.
For Names3Risk, different heads can focus on different aspects: character patterns,
email structure, name length, special characters, etc.

**Architecture:**
1. Split input into n_heads attention heads
2. Each head computes scaled dot-product attention
3. Concatenate head outputs
4. Project concatenated output back to embedding dimension

**Parameters:**
- embedding_dim (int): Input embedding dimension
- n_heads (int): Number of attention heads
- dropout (float): Dropout probability (default: 0.0)

**Forward Signature:**
Input:
  - x: (B, L, D) - Input sequence
  - attn_mask: (B, L) - Attention mask (optional)
  
Output:
  - output: (B, L, D)

**Dependencies:**
- names3risk_pytorch.dockers.pytorch.attention.attention_head → Individual attention heads

**Used By:**
- names3risk_pytorch.dockers.pytorch.blocks.transformer_block → Self-attention component

**Usage Example:**
```python
from names3risk_pytorch.pytorch.attention import MultiHeadAttention

mha = MultiHeadAttention(embedding_dim=128, n_heads=8, dropout=0.1)
x = torch.randn(32, 50, 128)  # (batch, seq_len, embed_dim)
attn_mask = torch.ones(32, 50).bool()

output = mha(x, attn_mask)  # (32, 50, 128)


**References:**
- "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention_head import AttentionHead


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        dropout: float = 0.0
    ):
        """
        Initialize MultiHeadAttention.
        
        Args:
            embedding_dim: Input embedding dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability for attention weights and output projection
        """
        super().__init__()
        
        # Create attention heads
        self.heads = nn.ModuleList([
            AttentionHead(embedding_dim, n_heads, dropout)
            for _ in range(n_heads)
        ])
        
        # Output projection
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: (B, L, D) - Input sequence
            attn_mask: (B, L) - Attention mask (optional)
            
        Returns:
            output: (B, L, D)
        """
        # Compute attention for each head and concatenate
        head_outputs = [head(x, attn_mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)  # (B, L, D)
        
        # Project back to embedding dimension
        output = self.proj(concatenated)
        output = self.dropout(output)
        
        return output
```

### Phase 3: Extract Composite Blocks

#### 3.1 Transformer Block → `pytorch/blocks/transformer_block.py`

**Current Location:** `transformer2risk.py` lines 68-78

**Dependencies:** Requires `MultiHeadAttention` and `MLPBlock`

```python
# pytorch/blocks/transformer_block.py
"""
Transformer Block

Complete transformer layer combining self-attention and feedforward network.

**Core Concept:**
Standard transformer encoder layer with pre-norm architecture. Processes
sequential data by allowing tokens to attend to each other, then applying
position-wise feedforward transformation.

**Architecture (Pre-Norm):**
1. x + Self-Attention(LayerNorm(x))
2. x + FFN(LayerNorm(x))

**Parameters:**
- embedding_dim (int): Model dimension
- n_heads (int): Number of attention heads
- ff_hidden_dim (int): Feedforward hidden dimension (typically 4x embedding_dim)
- dropout (float): Dropout probability

**Forward Signature:**
Input:
  - x: (B, L, D) - Input sequence
  - attn_mask: (B, L) - Attention mask (optional)
  
Output:
  - output: (B, L, D)

**Dependencies:**
- names3risk_pytorch.dockers.pytorch.attention.multihead_attention → Self-attention
- names3risk_pytorch.dockers.pytorch.feedforward.mlp_block → Feedforward network

**Used By:**
- names3risk_pytorch.dockers.pytorch.blocks.transformer_encoder → Stacked transformer layers

**Usage Example:**
```python
from names3risk_pytorch.pytorch.blocks import TransformerBlock

block = TransformerBlock(
    embedding_dim=128,
    n_heads=8,
    ff_hidden_dim=512,
    dropout=0.2
)

x = torch.randn(32, 50, 128)  # (batch, seq_len, embed_dim)
attn_mask = torch.ones(32, 50).bool()

output = block(x, attn_mask)  # (32, 50, 128)


**References:**
- "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from typing import Optional

from ..attention import MultiHeadAttention
from ..feedforward import MLPBlock


class TransformerBlock(nn.Module):
    """Complete transformer encoder layer with pre-norm architecture."""
    
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.0
    ):
        """
        Initialize TransformerBlock.
        
        Args:
            embedding_dim: Model dimension
            n_heads: Number of attention heads
            ff_hidden_dim: Feedforward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Self-attention with pre-norm
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.self_attn = MultiHeadAttention(embedding_dim, n_heads, dropout)
        
        # Feedforward with pre-norm
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = MLPBlock(embedding_dim, ff_hidden_dim, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: (B, L, D) - Input sequence
            attn_mask: (B, L) - Attention mask (optional)
            
        Returns:
            output: (B, L, D)
        """
        # Self-attention with residual
        x = x + self.self_attn(self.ln1(x), attn_mask)
        
        # Feedforward with residual
        x = x + self.ffn(self.ln2(x))
        
        return x
```

#### 3.2 LSTM Encoder → `pytorch/blocks/lstm_encoder.py`

**Current Location:** `lstm2risk.py` `TextProjection` class lines 51-83

**Dependencies:** Requires `AttentionPooling`

```python
# pytorch/blocks/lstm_encoder.py
"""
LSTM-Based Text Encoder

Bidirectional LSTM with attention pooling for text sequence encoding.

**Core Concept:**
Processes variable-length text sequences (customer names, emails) using LSTM
to capture sequential patterns, then pools to fixed-size representation using
learned attention weights.

**Architecture:**
1. Token Embedding: Vocabulary → Dense vectors
2. Bidirectional LSTM: Forward + Backward passes
3. Attention Pooling: Variable-length → Fixed-size
4. Layer Normalization: Output stabilization

**Parameters:**
- vocab_size (int): Vocabulary size for token embeddings
- embedding_dim (int): Token embedding dimension
- hidden_dim (int): LSTM hidden dimension
- num_layers (int): Number of LSTM layers
- dropout (float): Dropout probability
- bidirectional (bool): Use bidirectional LSTM (default: True)

**Forward Signature:**
Input:
  - tokens: (B, L) - Token IDs
  - lengths: (B,) - Actual sequence lengths (optional, for packing)
  
Output:
  - encoded: (B, 2*hidden_dim) - Encoded representations (2x for bidirectional)

**Dependencies:**
- names3risk_pytorch.dockers.pytorch.pooling.attention_pooling → Sequence pooling

**Used By:**
- names3risk_pytorch.dockers.lightning_models.bimodal.pl_lstm2risk → Text encoder for LSTM2Risk Lightning module

**Usage Example:**
```python
from names3risk_pytorch.pytorch.blocks import LSTMEncoder

encoder = LSTMEncoder(
    vocab_size=4000,
    embedding_dim=16,
    hidden_dim=128,
    num_layers=4,
    dropout=0.2
)

tokens = torch.randint(0, 4000, (32, 50))  # (batch, seq_len)
lengths = torch.randint(10, 50, (32,))  # Actual lengths

encoded = encoder(tokens, lengths)  # (32, 256) = (32, 2*128)


**References:**
- "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
"""

import torch
import torch.nn as nn
from typing import Optional

from ..pooling import AttentionPooling


class LSTMEncoder(nn.Module):
    """
    LSTM-based text sequence encoder with attention pooling.
    
    Encodes variable-length token sequences into fixed-size representations.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        dropout: float = 0.0,
        bidirectional: bool = True
    ):
        """
        Initialize LSTMEncoder.
        
        Args:
            vocab_size: Size of token vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability (applied between LSTM layers)
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention pooling (input_dim = 2*hidden_dim for bidirectional)
        lstm_output_dim = 2 * hidden_dim if bidirectional else hidden_dim
        self.pooling = AttentionPooling(input_dim=lstm_output_dim)
        
        # Output normalization
        self.norm = nn.LayerNorm(lstm_output_dim)
    
    def forward(
        self,
        tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode token sequences.
        
        Args:
            tokens: (B, L) - Token IDs
            lengths: (B,) - Actual sequence lengths before padding (optional)
            
        Returns:
            encoded: (B, 2*hidden_dim) - Encoded representations
        """
        # Token embeddings: (B, L, embedding_dim)
        embeddings = self.token_embedding(tokens)
        
        # LSTM encoding
        if lengths is not None:
            # Pack padded sequences for efficiency
            packed_emb = nn.utils.rnn.pack_padded_sequence(
                embeddings,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            packed_output, _ = self.lstm(packed_emb)
            # Unpack
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True
            )
        else:
            lstm_output, _ = self.lstm(embeddings)
        
        # Attention pooling: (B, 2*hidden_dim)
        pooled = self.pooling(lstm_output, lengths)
        
        # Layer normalization
        encoded = self.norm(pooled)
        
        return encoded
```

#### 3.3 Transformer Encoder → `pytorch/blocks/transformer_encoder.py`

**Current Location:** `transformer2risk.py` `TextProjection` class lines 148-179

**Dependencies:** Requires `TransformerBlock`, `AttentionPooling`

```python
# pytorch/blocks/transformer_encoder.py
"""
Transformer-Based Text Encoder

Transformer encoder with attention pooling for text sequence encoding.

**Core Concept:**
Processes text sequences using stacked transformer layers with self-attention,
allowing model to capture long-range dependencies and character patterns.
Uses learned positional encodings and attention pooling for final representation.

**Architecture:**
1. Token + Position Embeddings
2. N x Transformer Blocks
3. Attention Pooling: Variable-length → Fixed-size
4. Linear Projection: Transform to desired output dimension

**Parameters:**
- vocab_size (int): Vocabulary size
- embedding_dim (int): Token/position embedding dimension
- hidden_dim (int): Output projection dimension
- n_blocks (int): Number of transformer blocks
- n_heads (int): Number of attention heads per block
- ff_hidden_dim (int): Feedforward hidden dimension
- max_seq_len (int): Maximum sequence length for positional encoding
- dropout (float): Dropout probability

**Forward Signature:**
Input:
  - tokens: (B, L) - Token IDs
  - attn_mask: (B, L) - Attention mask (optional, True = attend, False = ignore)
  
Output:
  - encoded: (B, 2*hidden_dim) - Encoded representations

**Dependencies:**
- names3risk_pytorch.dockers.pytorch.blocks.transformer_block → Transformer layers
- names3risk_pytorch.dockers.pytorch.pooling.attention_pooling → Sequence pooling

**Used By:**
- names3risk_pytorch.dockers.lightning_models.bimodal.pl_transformer2risk → Text encoder for Transformer2Risk Lightning module

**Usage Example:**
```python
from names3risk_pytorch.pytorch.blocks import TransformerEncoder

encoder = TransformerEncoder(
    vocab_size=4000,
    embedding_dim=128,
    hidden_dim=256,
    n_blocks=8,
    n_heads=8,
    ff_hidden_dim=512,
    max_seq_len=100,
    dropout=0.2
)

tokens = torch.randint(0, 4000, (32, 50))  # (batch, seq_len)
attn_mask = torch.ones(32, 50).bool()  # All tokens valid

encoded = encoder(tokens, attn_mask)  # (32, 512) = (32, 2*256)


**References:**
- "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from typing import Optional

from .transformer_block import TransformerBlock
from ..pooling import AttentionPooling


class TransformerEncoder(nn.Module):
    """
    Transformer-based text encoder with attention pooling.
    
    Stacks transformer blocks and pools sequence to fixed representation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_blocks: int = 8,
        n_heads: int = 8,
        ff_hidden_dim: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.0
    ):
        """
        Initialize TransformerEncoder.
        
        Args:
            vocab_size: Size of token vocabulary
            embedding_dim: Dimension of token/position embeddings
            hidden_dim: Output projection dimension
            n_blocks: Number of transformer blocks
            n_heads: Number of attention heads per block
            ff_hidden_dim: Feedforward hidden dim (default: 4 * embedding_dim)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * embedding_dim
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                n_heads=n_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout
            )
            for _ in range(n_blocks)
        ])
        
        # Attention pooling
        self.pooling = AttentionPooling(input_dim=embedding_dim)
        
        # Output projection to match desired hidden_dim
        self.proj = nn.Linear(embedding_dim, 2 * hidden_dim)
    
    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode token sequences.
        
        Args:
            tokens: (B, L) - Token IDs
            attn_mask: (B, L) - Attention mask (optional)
            
        Returns:
            encoded: (B, 2*hidden_dim) - Encoded representations
        """
        B, L = tokens.shape
        
        # Token embeddings: (B, L, embedding_dim)
        token_emb = self.token_embedding(tokens)
        
        # Position embeddings: (L, embedding_dim) broadcast to (B, L, embedding_dim)
        positions = torch.arange(L, device=tokens.device)
        pos_emb = self.position_embedding(positions)
        
        # Combined embeddings
        x = token_emb + pos_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask)
        
        # Attention pooling: (B, embedding_dim)
        # Note: attn_mask not used in pooling (could add if needed)
        pooled = self.pooling(x)
        
        # Project to output dimension: (B, 2*hidden_dim)
        encoded = self.proj(pooled)
        
        return encoded
```

### Phase 4: Extract Preprocessing Components

#### 4.1 BPE Tokenizer → `tokenizers/bpe_tokenizer.py`

**Current Location:** `tokenizer.py` `OrderTextTokenizer` class

**Extraction Strategy:** Rename for clarity, maintain all functionality

```python
# tokenizers/bpe_tokenizer.py
"""
Compression-Tuned BPE Tokenizer

Byte-Pair Encoding tokenizer with automatic vocabulary size tuning to achieve
target compression rate.

**Core Concept:**
For fraud detection on name/email text, optimal vocabulary size balances:
- Too small → Many UNK tokens, loses information
- Too large → Overfitting to training data, poor generalization

Auto-tuning via binary search finds vocabulary size achieving target compression
rate on validation set (e.g., 2.5 chars per token).

**Parameters:**
- min_frequency (int): Minimum character frequency to include in vocabulary
- target_compression (float): Target compression rate (chars per token)
- max_vocab_size (int): Maximum allowed vocabulary size

**Methods:**
- train(): Train tokenizer on text corpus with compression tuning
- encode(): Tokenize single text to token IDs
- calculate_compression_rate(): Measure compression on text sample

**Dependencies:**
- tokenizers (HuggingFace) → BPE implementation
- unicodedata → Text normalization

**Used By:**
- names3risk_pytorch.dockers.processing.datasets → Tokenize names/emails during preprocessing

**Usage Example:**
```python
from names3risk_pytorch.dockers.tokenizers import CompressionBPETokenizer

# Train tokenizer
tokenizer = CompressionBPETokenizer(min_frequency=25)
texts = ["john.smith@email.com", "Jane Doe", ...]

tokenizer.train(
    texts,
    target_compression=2.5,  # Aim for 2.5 chars per token
    max_vocab_size=50000
)

# Encode text
tokens = tokenizer.encode("Alice Johnson|alice@email.com")
# Returns: [15, 234, 45, ...]

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"PAD token ID: {tokenizer.pad_token}")
print(f"CLS token ID: {tokenizer.cls_token}")


**References:**
- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016) - BPE for NLP
"""

import unicodedata
import random
from typing import List
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class CompressionBPETokenizer:
    """
    BPE tokenizer with automatic vocabulary size tuning.
    
    Trains tokenizer to achieve target compression rate via binary search
    on vocabulary size.
    """
    
    def __init__(self, min_frequency: int = 25):
        """
        Initialize tokenizer.
        
        Args:
            min_frequency: Minimum token frequency to include in vocabulary
        """
        self._tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self._tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.min_frequency = min_frequency
        self.pad_token = None
        self.cls_token = None
    
    def calculate_compression_rate(
        self,
        texts: List[str],
        sample_size: int = 10000
    ) -> float:
        """
        Calculate compression rate (chars per token) on text sample.
        
        Args:
            texts: List of texts to measure compression on
            sample_size: Maximum number of texts to sample
            
        Returns:
            compression_rate: Average characters per token
        """
        if len(texts) > sample_size:
            sample_texts = random.sample(texts, sample_size)
        else:
            sample_texts = texts
        
        total_chars = 0
        total_tokens = 0
        
        for text in sample_texts:
            normalized_text = unicodedata.normalize("NFKC", text)
            encoding = self._tokenizer.encode(normalized_text)
            total_chars += len(normalized_text)
            total_tokens += len(encoding.ids)
        
        return total_chars / total_tokens if total_tokens > 0 else 0.0
    
    def train(
        self,
        texts: List[str],
        target_compression: float = 2.5,
        max_vocab_size: int = 50000
    ) -> "CompressionBPETokenizer":
        """
        Train tokenizer with automatic vocab size tuning.
        
        Uses binary search on vocabulary size to achieve target compression rate.
        
        Args:
            texts: Training texts
            target_compression: Target chars per token (e.g., 2.5)
            max_vocab_size: Maximum vocabulary size
            
        Returns:
            self: Trained tokenizer
        """
        # Split data for training and validation
        random.shuffle(texts)
        split_idx = int(0.8 * len(texts))
        train_texts = texts[:split_idx]
        validation_texts = texts[split_idx:]
        
        print(f"Target compression: {target_compression:.1%}")
        print(f"Min frequency: {self.min_frequency}")
        print(f"Training on {len(train_texts)} texts, validating on {len(validation_texts)}")
        
        # Binary search on vocab_size
        vocab_low = 1000
        vocab_high = max_vocab_size
        best_compression = 0.0
        best_tokenizer = None
        best_vocab_size = None
        
        iteration = 0
        while vocab_low <= vocab_high and iteration < 15:
            iteration += 1
            current_vocab_size = (vocab_low + vocab_high) // 2
            
            print(f"\nIteration {iteration}: Testing vocab_size={current_vocab_size}")
            
            # Train with current vocab size
            trainer = BpeTrainer(
                vocab_size=current_vocab_size,
                special_tokens=["[CLS]", "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MISSING]", "|"],
                min_frequency=self.min_frequency
            )
            
            temp_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            temp_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            temp_tokenizer.train_from_iterator(
                (unicodedata.normalize("NFKC", text) for text in train_texts),
                trainer
            )
            
            # Measure compression on validation set
            self._tokenizer = temp_tokenizer
            compression = self.calculate_compression_rate(validation_texts)
            actual_vocab_size = temp_tokenizer.get_vocab_size()
            
            print(f"  Compression: {compression:.3f} ({compression:.1%})")
            print(f"  Actual vocab size: {actual_vocab_size}")
            
            # Track best result
            if abs(compression - target_compression) < abs(best_compression - target_compression):
                best_compression = compression
                best_tokenizer = temp_tokenizer
                best_vocab_size = actual_vocab_size
            
            # Adjust search range
            if compression < target_compression:
                vocab_low = current_vocab_size + 1
            else:
                vocab_high = current_vocab_size - 1
            
            # Early exit if close enough
            if abs(compression - target_compression) < 0.005:
                print(f"  ✓ Achieved target compression within tolerance!")
                break
        
        # Use best tokenizer
        if best_tokenizer is not None:
            self._tokenizer = best_tokenizer
            print(f"\nFinal tokenizer:")
            print(f"  Min frequency: {self.min_frequency}")
            print(f"  Vocab size: {best_vocab_size}")
            print(f"  Compression: {best_compression:.3f} ({best_compression:.1%})")
        else:
            print("\nWarning: No suitable tokenizer found, using last attempt")
        
        # Set up special tokens
        pad_tokens = self.encode("[PAD]")
        assert len(pad_tokens) == 1, "PAD token should be single token"
        self.pad_token = pad_tokens[0]
        
        cls_tokens = self.encode("[CLS]")
        assert len(cls_tokens) == 1, "CLS token should be single token"
        self.cls_token = cls_tokens[0]
        
        return self
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            token_ids: List of token IDs
        """
        normalized_text = unicodedata.normalize("NFKC", text)
        return self.tokenizer.encode(normalized_text).ids
    
    @property
    def tokenizer(self) -> Tokenizer:
        """Get underlying HuggingFace tokenizer."""
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, value: Tokenizer):
        """Set underlying tokenizer and update special tokens."""
        self._tokenizer = value
        
        pad_tokens = self.encode("[PAD]")
        assert len(pad_tokens) == 1
        self.pad_token = pad_tokens[0]
        
        cls_tokens = self.encode("[CLS]")
        assert len(cls_tokens) == 1
        self.cls_token = cls_tokens[0]
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()
```

## Implementation Timeline

### Week 1: Extract Duplicates & Atomic Components
**Goal**: Eliminate duplication and extract standalone components

**Tasks**:
- [ ] Create `projects/names3risk_pytorch/` directory structure
- [ ] Extract & unify `AttentionPooling` → `pytorch/pooling/attention_pooling.py`
- [ ] Extract & unify `ResidualBlock` → `pytorch/feedforward/residual_block.py`
- [ ] Extract `MLPBlock` → `pytorch/feedforward/mlp_block.py`
- [ ] Extract `AttentionHead` → `pytorch/attention/attention_head.py`
- [ ] Extract `MultiHeadAttention` → `pytorch/attention/multihead_attention.py`
- [ ] Write unit tests for all extracted components
- [ ] Add deprecation warnings to legacy code

**Success Metrics**:
- ✅ Zero duplication for `AttentionPooling` and `ResidualBlock`
- ✅ All tests passing with 100% coverage for atomic components
- ✅ Legacy models still functional with deprecation warnings

### Week 2: Extract Composite Blocks
**Goal**: Create composite building blocks from atomic components

**Tasks**:
- [ ] Extract `TransformerBlock` → `pytorch/blocks/transformer_block.py`
- [ ] Extract LSTM `TextProjection` → `pytorch/blocks/lstm_encoder.py`
- [ ] Extract Transformer `TextProjection` → `pytorch/blocks/transformer_encoder.py`
- [ ] Write integration tests for composite blocks
- [ ] Verify blocks match legacy behavior exactly

**Success Metrics**:
- ✅ All composite blocks use atomic components via imports
- ✅ Integration tests verify correct composition
- ✅ Numerical equivalence tests pass (legacy vs new)

### Week 3: Extract Preprocessing & Refactor Models
**Goal**: Complete preprocessing extraction and refactor application models

**Tasks**:
- [ ] Extract `OrderTextTokenizer` → `tokenizers/bpe_tokenizer.py`
- [ ] Extract `TextDataset`, `TabularDataset` → `processing/datasets/`
- [ ] Create new `BimodalDataset` combining text + tabular
- [ ] Create `processing/dataloaders/names3risk_collate.py` for Names3Risk model collate functions
- [ ] Refactor `LSTM2Risk` to use atomic components
- [ ] Refactor `Transformer2Risk` to use atomic components
- [ ] End-to-end tests for refactored models

**Success Metrics**:
- ✅ Models use only atomic/composite components, no internal classes
- ✅ Models match legacy behavior exactly
- ✅ All preprocessing components extracted and tested

### Week 4: Documentation & Validation
**Goal**: Complete documentation and verify migration success

**Tasks**:
- [ ] Write comprehensive docstrings for all components
- [ ] Create component index document
- [ ] Performance benchmarks (verify no regression)
- [ ] Migration guide for users
- [ ] Code review and feedback incorporation
- [ ] Archive legacy code with deprecation notice

**Success Metrics**:
- ✅ All components have standardized docstrings with examples
- ✅ Performance within 1% of legacy implementation
- ✅ Migration guide validated by team
- ✅ Zero production incidents post-migration

## Migration Strategy

### Backward Compatibility Approach

To ensure smooth migration without breaking existing code:

```python
# In legacy lstm2risk.py - add deprecation warnings
import warnings
from names3risk_pytorch.dockers.pytorch.pooling import AttentionPooling as _AttentionPooling
from names3risk_pytorch.dockers.pytorch.feedforward import ResidualBlock as _ResidualBlock

class AttentionPooling(_AttentionPooling):
    """
    Deprecated: Use names3risk_pytorch.dockers.pytorch.pooling.AttentionPooling instead.
    
    This wrapper is provided for backward compatibility and will be removed in v2.0.
    """
    def __init__(self, config):
        warnings.warn(
            "AttentionPooling from lstm2risk.py is deprecated. "
            "Use 'from names3risk_pytorch.dockers.pytorch.pooling import AttentionPooling' instead. "
            "This compatibility wrapper will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2
        )
        # Convert config to new interface
        super().__init__(input_dim=2 * config.hidden_size)

class ResidualBlock(_ResidualBlock):
    """
    Deprecated: Use names3risk_pytorch.dockers.pytorch.feedforward.ResidualBlock instead.
    """
    def __init__(self, config):
        warnings.warn(
            "ResidualBlock from lstm2risk.py is deprecated. "
            "Use 'from names3risk_pytorch.dockers.pytorch.feedforward import ResidualBlock' instead. "
            "This compatibility wrapper will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(
            dim=4 * config.hidden_size,
            expansion_factor=4,
            dropout=0.0,
            norm_first=True
        )
```

### Testing Strategy

#### Unit Tests (Per Atomic Component)
```python
# test/pytorch/pooling/test_attention_pooling.py
import torch
import pytest
from names3risk_pytorch.pytorch.pooling import AttentionPooling

class TestAttentionPooling:
    def test_forward_with_lengths(self):
        """Test attention pooling with variable-length sequences."""
        pooling = AttentionPooling(input_dim=64)
        sequences = torch.randn(4, 10, 64)
        lengths = torch.tensor([5, 8, 3, 10])
        
        output = pooling(sequences, lengths)
        
        assert output.shape == (4, 64)
        assert not torch.isnan(output).any()
    
    def test_forward_without_lengths(self):
        """Test attention pooling with fixed-length sequences."""
        pooling = AttentionPooling(input_dim=64)
        sequences = torch.randn(4, 10, 64)
        
        output = pooling(sequences)
        
        assert output.shape == (4, 64)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        pooling = AttentionPooling(input_dim=64)
        sequences = torch.randn(4, 10, 64, requires_grad=True)
        
        output = pooling(sequences)
        loss = output.sum()
        loss.backward()
        
        assert sequences.grad is not None
        assert not torch.isnan(sequences.grad).any()
```

#### Integration Tests (Composite Blocks)
```python
# test/pytorch/blocks/test_transformer_encoder.py
import torch
from names3risk_pytorch.pytorch.blocks import TransformerEncoder

class TestTransformerEncoder:
    def test_encoder_integration(self):
        """Test transformer encoder with all dependencies."""
        encoder = TransformerEncoder(
            vocab_size=4000,
            embedding_dim=128,
            hidden_dim=256,
            n_blocks=8,
            n_heads=8,
            ff_hidden_dim=512,
            max_seq_len=100,
            dropout=0.2
        )
        
        tokens = torch.randint(0, 4000, (32, 50))
        attn_mask = torch.ones(32, 50).bool()
        
        output = encoder(tokens, attn_mask)
        
        assert output.shape == (32, 512)  # 2 * hidden_dim
        assert not torch.isnan(output).any()
```

#### End-to-End Tests (Full Models)
```python
# test/models/test_lstm2risk.py
from names3risk_pytorch.models import LSTM2Risk, LSTMConfig

class TestLSTM2Risk:
    def test_model_with_refactored_components(self):
        """Verify LSTM2Risk works after refactoring."""
        config = LSTMConfig(
            embedding_size=16,
            dropout_rate=0.2,
            hidden_size=128,
            n_tab_features=100,
            n_embed=4000,
            n_lstm_layers=4
        )
        
        model = LSTM2Risk(config)
        
        batch = {
            "text": torch.randint(0, 4000, (32, 50)),
            "tabular": torch.randn(32, 100),
            "text_length": torch.randint(10, 50, (32,))
        }
        
        output = model(batch)
        
        assert output.shape == (32, 1)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output
```

#### Numerical Equivalence Tests
```python
# test/compatibility/test_legacy_equivalence.py
import torch
from projects.names3risk_legacy.lstm2risk import LSTM2Risk as LegacyLSTM2Risk
from names3risk_pytorch.models import LSTM2Risk as NewLSTM2Risk

class TestLegacyEquivalence:
    def test_lstm2risk_equivalence(self):
        """Verify new LSTM2Risk produces identical results to legacy."""
        config = LSTMConfig(...)
        
        # Initialize both models with same random seed
        torch.manual_seed(42)
        legacy_model = LegacyLSTM2Risk(config)
        
        torch.manual_seed(42)
        new_model = NewLSTM2Risk(config)
        
        # Same input
        batch = create_test_batch()
        
        # Compare outputs
        with torch.no_grad():
            legacy_output = legacy_model(batch)
            new_output = new_model(batch)
        
        # Should be numerically identical (within floating point precision)
        assert torch.allclose(legacy_output, new_output, atol=1e-6)
```

## Benefits Analysis

### Quantitative Improvements

**Code Duplication Reduction:**
- Before: 2 implementations of `AttentionPooling` (36 lines duplicated)
- After: 1 unified implementation (28 lines)
- **Savings**: 44% reduction in code

**Code Reusability:**
- Before: 13 components locked in 2 monolithic files
- After: 13+ atomic components usable independently
- **Improvement**: ∞% increase in reusability

**Import Clarity:**
- Before: Importing `AttentionPooling` brings 4 other classes
- After: Importing `AttentionPooling` brings only 1 class
- **Improvement**: 80% reduction in unintended dependencies

### Qualitative Improvements

**Developer Experience:**
- ✅ Easier to find components (semantic folder structure)
- ✅ Faster to understand components (focused modules)
- ✅ Simpler to test components (atomic isolation)
- ✅ Safer to refactor components (explicit dependencies)

**Code Quality:**
- ✅ Consistent interfaces across components
- ✅ Comprehensive documentation with examples
- ✅ Type hints enable IDE support
- ✅ Single source of truth eliminates divergence

**Maintainability:**
- ✅ Bug fixes applied once, propagate to all consumers
- ✅ Clear ownership per atomic module
- ✅ Easier onboarding for new developers
- ✅ Reduced technical debt

## Risks and Mitigation

### Risk 1: Breaking Changes During Migration
**Probability**: Medium | **Impact**: High

**Mitigation**:
- Maintain backward compatibility wrappers in legacy files
- Add deprecation warnings with clear migration path
- Run comprehensive test suite before each migration phase
- Keep legacy code as fallback for 2 major versions

**Contingency**: Immediate rollback to legacy implementation if critical issues discovered

### Risk 2: Performance Regression
**Probability**: Low | **Impact**: Medium

**Mitigation**:
- Benchmark performance before and after refactoring
- Profile hot paths to identify bottlenecks
- Use identical algorithms, only restructure code organization
- Validate numerical equivalence with legacy implementation

**Contingency**: Optimize specific components if performance degrades >5%

### Risk 3: Increased Complexity for Simple Use Cases
**Probability**: Low | **Impact**: Low

**Mitigation**:
- Provide high-level application models (LSTM2Risk, Transformer2Risk) that hide complexity
- Create usage examples for common patterns
- Maintain simple import paths for common components
- Document migration from legacy to new structure

**Contingency**: Create convenience wrappers if users struggle with new structure

## Success Criteria

### Technical Metrics
- [ ] **Zero Code Duplication**: All 13 components have single implementations
- [ ] **100% Test Coverage**: All atomic components have comprehensive unit tests
- [ ] **Performance Parity**: <1% difference in training/inference speed vs legacy
- [ ] **Numerical Equivalence**: Outputs match legacy implementation within 1e-6
- [ ] **Documentation Complete**: All modules have standardized docstrings with examples

### Process Metrics
- [ ] **Migration Timeline**: Complete within 4 weeks
- [ ] **Zero Breaking Changes**: All existing code continues to work with warnings
- [ ] **Team Adoption**: >80% of team comfortable with new structure after 1 week
- [ ] **Code Review Approval**: All phases reviewed and approved by 2+ reviewers

### Outcome Metrics
- [ ] **Reduced Maintenance**: 50% reduction in bug fix time for shared components
- [ ] **Increased Velocity**: 30% faster development for new fraud detection models
- [ ] **Better Onboarding**: New developers productive with Names3Risk in <3 days
- [ ] **Code Reuse**: At least 3 components reused in other projects within 6 months

## Future Enhancements

### Phase 5: Cross-Project Component Sharing (3 months post-migration)
**Goal**: Share Names3Risk components with other fraud detection projects

**Tasks**:
- Move `pytorch/` directory to shared library location
- Publish internal package for atomic components
- Create examples for other projects
- Track adoption across teams

### Phase 6: Additional Fraud Detection Components (6 months post-migration)
**Goal**: Expand component library based on learnings

**New Components**:
- Graph neural network components for relationship modeling
- Anomaly detection modules
- Ensemble methods for model combination
- Active learning components for label efficiency

### Phase 7: Automated Component Discovery (12 months post-migration)
**Goal**: Enable developers to discover components programmatically

**Features**:
- Component search tool by functionality
- Automatic dependency graph visualization
- Performance profiling per component
- Usage examples auto-generated from tests

## Appendices

### Appendix A: Complete Component Inventory

| Component | Legacy Location | New Location | Type | Status |
|-----------|----------------|--------------|------|--------|
| AttentionPooling | lstm2risk.py, transformer2risk.py | pytorch/pooling/attention_pooling.py | Atomic | 🔴 Duplicate |
| ResidualBlock | lstm2risk.py, transformer2risk.py | pytorch/feedforward/residual_block.py | Atomic | 🔴 Duplicate |
| FeedForward | transformer2risk.py | pytorch/feedforward/mlp_block.py | Atomic | ✅ Unique |
| Head | transformer2risk.py | pytorch/attention/attention_head.py | Atomic | ✅ Unique |
| MultiHeadAttention | transformer2risk.py | pytorch/attention/multihead_attention.py | Atomic | ✅ Unique |
| Block | transformer2risk.py | pytorch/blocks/transformer_block.py | Composite | ✅ Unique |
| TextProjection (LSTM) | lstm2risk.py | pytorch/blocks/lstm_encoder.py | Composite | ✅ Unique |
| TextProjection (Transformer) | transformer2risk.py | pytorch/blocks/transformer_encoder.py | Composite | ✅ Unique |
| OrderTextTokenizer | tokenizer.py | preprocessing/tokenizers/bpe_tokenizer.py | Preprocessing | ✅ Unique |
| TextDataset | dataset.py | preprocessing/datasets/text_dataset.py | Preprocessing | ✅ Unique |
| TabularDataset | dataset.py | preprocessing/datasets/tabular_dataset.py | Preprocessing | ✅ Unique |
| LSTM2Risk | lstm2risk.py | models/lstm2risk.py | Application | ✅ Unique |
| Transformer2Risk | transformer2risk.py | models/transformer2risk.py | Application | ✅ Unique |

### Appendix B: Dependency Graph

```
Application Models (models/)
├── LSTM2Risk
│   ├── LSTMEncoder (blocks/)
│   │   ├── AttentionPooling (pooling/)
│   │   └── nn.LSTM (PyTorch)
│   ├── ResidualBlock (feedforward/)
│   └── BimodalConcat (fusion/)
│
└── Transformer2Risk
    ├── TransformerEncoder (blocks/)
    │   ├── TransformerBlock (blocks/)
    │   │   ├── MultiHeadAttention (attention/)
    │   │   │   └── AttentionHead (attention/)
    │   │   └── MLPBlock (feedforward/)
    │   └── AttentionPooling (pooling/)
    ├── ResidualBlock (feedforward/)
    └── BimodalConcat (fusion/)
```

### Appendix C: Naming Conventions

**File Naming:**
- Format: `{concept_name}.py` (lowercase, underscores)
- Examples: `attention_pooling.py`, `lstm_encoder.py`, `bpe_tokenizer.py`
- Avoid: `utils.py`, `helpers.py`, `common.py`

**Class Naming:**
- Format: `{ConceptName}` (PascalCase)
- Examples: `AttentionPooling`, `LSTMEncoder`, `CompressionBPETokenizer`
- Match file name (snake_case file → PascalCase class)

**Directory Naming:**
- Format: `{category_name}/` (lowercase, underscores)
- Examples: `attention/`, `feedforward/`, `preprocessing/`
- Semantic names describing function, not hierarchy

---

**Document Status:** ✅ Ready for Implementation  
**Last Updated:** 2026-01-04  
**Authors:** AI Analysis Team  
**Reviewers:** [Pending]  
**Next Review:** After Week 1 completion

**Related Implementation Documents:**
- [PyTorch Module Reorganization Design](pytorch_module_reorganization_design.md) - Foundational principles
- [Names3Risk Model Design](names3risk_model_design.md) - Model architecture details
- [Algorithm Preserving Refactoring SOP](../6_resources/algorithm_preserving_refactoring_sop.md) - Safe refactoring procedures
