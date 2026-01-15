---
tags:
  - analysis
  - debugging
  - fraud-detection
  - pytorch
  - dimension-tracking
  - architecture-verification
keywords:
  - Names3Risk
  - Transformer2Risk
  - dimension analysis
  - shape mismatch
  - position embedding
  - sequence length
topics:
  - model debugging
  - architecture verification
  - dimension tracking
  - error diagnosis
language: python
date of note: 2026-01-14
---

# Transformer2Risk Dimension Analysis

## Executive Summary

This analysis provides a comprehensive dimension-by-dimension review of the pl_transformer2risk model architecture to diagnose runtime errors. Through systematic tracking of tensor shapes through the forward pass, we identified a **critical sequence length mismatch** between configured parameters (max_sen_len=100) and actual inputs (512 tokens).

**Key Findings:**
- ✅ **Architecture is correctly designed** - All layer dimensions align when given proper inputs
- ❌ **Critical bug: Sequence length mismatch** - Model configured for 100 tokens but receives 512
- ✅ **All fixes verified** - Attention mask dtype and double projection issues resolved
- 🔴 **Immediate action required** - Truncate inputs or increase position embedding size

**Root Cause:** The dataloader (`build_collate_batch`) does not truncate sequences to max_sen_len=100, while TransformerEncoder's position embedding only supports 100 positions. This causes IndexError before the model can execute.

**Verdict:** The model architecture is **functionally correct**. The issue is a **configuration mismatch** in the data pipeline that must be fixed before training can proceed.

## Related Documents
- **[Names3Risk PyTorch Component Correspondence Analysis](./2026-01-05_names3risk_pytorch_component_correspondence_analysis.md)** - Component mapping and architecture design
- **[Names3Risk PyTorch Training End-to-End Analysis](./2026-01-07_names3risk_pytorch_training_end_to_end_analysis.md)** - Training pipeline analysis
- **[Names3Risk PyTorch Reorganization Design](../1_design/names3risk_pytorch_reorganization_design.md)** - Architecture design principles
- **[Model Architecture Design Index](../00_entry_points/model_architecture_design_index.md)** - Architecture documentation index

## Methodology

### Analysis Approach

1. **Configuration Review**: Examined training logs to extract actual hyperparameters
2. **Forward Pass Tracing**: Tracked tensor dimensions layer-by-layer
3. **Error Diagnosis**: Identified position embedding IndexError as primary issue
4. **Legacy Comparison**: Verified differences between legacy and refactored implementations
5. **Solution Design**: Proposed three fix options with trade-offs

## Configuration Parameters (From Training Logs)

```python
n_embed = 3725          # Vocabulary size
embedding_size = 128    # Token/position embedding dimension
hidden_size = 256       # Hidden dimension
n_blocks = 8            # Number of transformer blocks
n_heads = 8             # Attention heads per block
max_sen_len = 100       # Maximum sequence length (CRITICAL!)
dropout_rate = 0.2      # Dropout probability
input_tab_dim = 11      # Number of tabular features
num_classes = 2         # Binary classification
batch_size = 2          # Example batch size
```

## Forward Pass Dimension Flow

### Input Stage

```
Batch Input:
├─ text_input_ids: (B, 1, L) = (2, 1, 512)  [from dataloader]
├─ attention_mask: (B, 1, L) = (2, 1, 512)  [from dataloader]
└─ tabular fields: 11 individual lists

After Preprocessing:
├─ text_tokens: (B, L) = (2, 512)  [squeezed chunk dimension]
├─ attn_mask: (B, L) = (2, 512)    [squeezed chunk dimension]
└─ tab_data: (B, F) = (2, 11)      [stacked from lists]
```

**⚠️ CRITICAL ISSUE DETECTED:**
- Input sequence length: L=512
- Configured max_sen_len: 100
- **Position embedding only supports 100 positions but receives 512!**

---

### Text Encoder Path: TransformerEncoder

#### 1. Token Embedding
```
Input:  text_tokens (B, L) = (2, 512)
Layer:  nn.Embedding(n_embed=3725, embedding_dim=128)
Output: token_emb (B, L, D) = (2, 512, 128) ✓
```

#### 2. Position Embedding
```
Input:  positions = torch.arange(L) = torch.arange(512)
Layer:  nn.Embedding(max_seq_len=100, embedding_dim=128)
ERROR:  ❌ IndexError: index out of range
        Trying to access positions[0:512] but only 100 embeddings exist!
```

**Root Cause:** 
- `position_embedding = nn.Embedding(max_seq_len=100, embedding_dim=128)`
- But `positions = torch.arange(512)` tries to index beyond [0, 99]

**Expected Flow (if L ≤ 100):**
```
Input:  positions (L,) = range(100)
Layer:  nn.Embedding(100, 128)
Output: pos_emb (L, D) = (100, 128)
        Broadcast to (B, L, D) = (2, 100, 128)
Combined: x = token_emb + pos_emb = (2, 100, 128) ✓
```

#### 3. Transformer Blocks (8x)
```
Input:  x (B, L, D) = (2, 100, 128)
Layer:  8x TransformerBlock(embedding_dim=128, n_heads=8, ff_hidden_dim=512)
        Each block:
          ├─ MultiHeadAttention: (B, L, D) → (B, L, D)
          │   └─ 8 heads × head_size(16) = 128
          └─ FeedForward: (B, L, D) → (B, L, 4D) → (B, L, D)
              └─ Linear(128 → 512) → ReLU → Linear(512 → 128)
Output: x (B, L, D) = (2, 100, 128) ✓
```

#### 4. Attention Pooling
```
Input:  x (B, L, D) = (2, 100, 128)
Layer:  AttentionPooling(input_dim=128)
        ├─ attention_scores = Linear(128 → 1) → (B, L, 1)
        ├─ weights = softmax(scores, dim=1) → (B, L, 1)
        └─ pooled = sum(weights * x, dim=1)
Output: pooled (B, D) = (2, 128) ✓
```

#### 5. Output Projection
```
Input:  pooled (B, D) = (2, 128)
Layer:  nn.Linear(embedding_dim=128, 2*hidden_size=512)
Output: text_hidden (B, 2H) = (2, 512) ✓
```

**Text Encoder Summary:**
```
text_encoder: (B, L) → (B, 2*H)
              (2, 100) → (2, 512)  ✓
```

---

### Tabular Encoder Path

#### Input
```
tab_data (B, F) = (2, 11)
```

#### Layer Breakdown
```
1. BatchNorm1d(11):         (2, 11) → (2, 11)
2. Linear(11 → 512):        (2, 11) → (2, 512)
3. ReLU + Dropout(0.2):     (2, 512) → (2, 512)
4. Linear(512 → 512):       (2, 512) → (2, 512)
5. LayerNorm(512):          (2, 512) → (2, 512)
6. ReLU + Dropout(0.2):     (2, 512) → (2, 512)
Output: tab_hidden (B, 2H) = (2, 512) ✓
```

**Tabular Encoder Summary:**
```
tab_encoder: (B, F) → (B, 2*H)
             (2, 11) → (2, 512) ✓
```

---

### Fusion & Classification Path

#### 1. Concatenation
```
Input:  text_hidden (B, 2H) = (2, 512)
        tab_hidden  (B, 2H) = (2, 512)
Output: combined    (B, 4H) = (2, 1024) ✓
```

#### 2. Classifier (4x ResidualBlock + Linear)

**ResidualBlock Structure (expansion_factor=1, post-norm):**
```python
class ResidualBlock:
    def forward(x):  # x: (B, 1024)
        residual = x
        x = Linear(1024 → 1024)(x)  # (B, 1024)
        x = ReLU(x)
        x = Linear(1024 → 1024)(x)  # (B, 1024)
        x = Dropout(x)
        x = x + residual              # (B, 1024) - residual connection
        x = LayerNorm(x)              # (B, 1024) - post-norm
        return x
```

**Full Classifier:**
```
Input: combined (B, 4H) = (2, 1024)

Block 1:
├─ ResidualBlock(1024) → (2, 1024)
├─ ReLU                → (2, 1024)
└─ Dropout(0.2)        → (2, 1024)

Block 2:
├─ ResidualBlock(1024) → (2, 1024)
├─ ReLU                → (2, 1024)
└─ Dropout(0.2)        → (2, 1024)

Block 3:
├─ ResidualBlock(1024) → (2, 1024)
├─ ReLU                → (2, 1024)
└─ Dropout(0.2)        → (2, 1024)

Block 4:
├─ ResidualBlock(1024) → (2, 1024)
├─ ReLU                → (2, 1024)
└─ Dropout(0.2)        → (2, 1024)

Final Projection:
└─ Linear(1024 → 2)    → (2, 2)

Output: logits (B, num_classes) = (2, 2) ✓
```

---

## Complete Forward Pass Summary

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                       │
├─────────────────────────────────────────────────────────────┤
│ text_tokens:  (2, 512)  ← ISSUE: L=512 but max_sen_len=100 │
│ attn_mask:    (2, 512)  ← ISSUE: L=512 but max_sen_len=100 │
│ tab_data:     (2, 11)   ✓                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ TEXT ENCODER (TransformerEncoder)                          │
├─────────────────────────────────────────────────────────────┤
│ token_emb:    (2, 512, 128)  ✓                             │
│ pos_emb:      FAILS - needs (512, 128) but only have (100, 128) │
│ [Should be]:  (2, 100, 128)                                │
│ → 8x TransformerBlock                                      │
│ → AttentionPooling                                         │
│ → Output projection                                        │
│ Output:       (2, 512)  [if input were (2, 100)]           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ TABULAR ENCODER (MLP)                                      │
├─────────────────────────────────────────────────────────────┤
│ BatchNorm + 2-layer MLP with LayerNorm                    │
│ Output:       (2, 512)  ✓                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ FUSION & CLASSIFICATION                                    │
├─────────────────────────────────────────────────────────────┤
│ Concatenate:  (2, 512) + (2, 512) = (2, 1024)  ✓          │
│ 4x ResidualBlock + Linear projection                       │
│ Output:       (2, 2)  ✓                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Critical Issues Identified

### 🔴 Issue 1: Sequence Length Mismatch

**Problem:**
- Input sequence length: **512 tokens**
- Configured max_sen_len: **100 tokens**
- Position embedding only has 100 entries

**Error:**
```python
positions = torch.arange(512)  # Creates [0, 1, 2, ..., 511]
pos_emb = self.position_embedding(positions)  # FAILS!
# IndexError: index 100 is out of bounds for dimension 0 with size 100
```

**Impact:** Model crashes before reaching the shape mismatch error

**Solution Options:**
1. **Truncate input:** Limit tokenized sequences to max_sen_len=100
2. **Increase max_sen_len:** Change to 512 (requires retraining)
3. **Use relative positional encoding:** Remove absolute position limit

---

## Comparison with Legacy Model

### Legacy transformer2risk.py
```python
config.block_size = 100  # Maximum sequence length

# In collate_fn:
texts = [item["text"][:self.block_size] for item in batch]  # Truncates to 100!

# Position embedding:
self.position_embedding_table = nn.Embedding(config.block_size, config.embedding_size)
# Creates Embedding(100, 128) ✓
```

### New pl_transformer2risk.py
```python
max_sen_len = 100  # Maximum sequence length

# TransformerEncoder:
self.position_embedding = nn.Embedding(max_seq_len=100, embedding_dim=128)
# Creates Embedding(100, 128) ✓

# BUT: Input from build_collate_batch is NOT truncated!
# Receives full 512-token sequences ❌
```

**Key Difference:** Legacy code truncates in collate_fn, new code doesn't!

---

## Recommended Fixes

### Fix 1: Truncate in Dataloader (Immediate)
```python
# In build_collate_batch or pipeline_dataloader:
max_seq_len = 100
text_tokens = text_tokens[:, :max_seq_len]  # Truncate to 100
attn_mask = attn_mask[:, :max_seq_len]      # Truncate to 100
```

### Fix 2: Increase Position Embedding Size (Requires Retraining)
```python
# In hyperparameters_transformer2risk.py:
max_sen_len = 512  # Match actual input length

# Note: This changes model architecture, requires full retraining
```

### Fix 3: Add Truncation to TransformerEncoder
```python
# In transformer_encoder.py forward():
def forward(self, tokens, attn_mask=None):
    B, L = tokens.shape
    
    # Truncate if needed
    if L > self.max_seq_len:
        tokens = tokens[:, :self.max_seq_len]
        if attn_mask is not None:
            attn_mask = attn_mask[:, :self.max_seq_len]
        L = self.max_seq_len
    
    # Rest of forward pass...
```

---

## Conclusion

The model architecture is **correctly designed** but has a **critical configuration mismatch**:

✅ **Correct:** All layer dimensions align properly  
✅ **Correct:** Text encoder outputs (B, 2*H)  
✅ **Correct:** Tab encoder outputs (B, 2*H)  
✅ **Correct:** Classifier expects (B, 4*H)  
❌ **BROKEN:** Input sequence length (512) exceeds max_sen_len (100)

**Priority:** Fix the sequence length issue before addressing the earlier shape mismatch error.
