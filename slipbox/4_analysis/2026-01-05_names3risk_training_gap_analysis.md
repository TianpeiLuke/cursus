# Names3Risk Training Gap Analysis: Legacy vs Current Implementation

**Date**: 2026-01-05  
**Author**: AI Assistant  
**Purpose**: Compare legacy train.py with current PyTorch Lightning implementation to identify gaps and migration requirements

## Executive Summary

This document compares the legacy Names3Risk training script (`projects/names3risk_legacy/train.py`) with the current PyTorch Lightning-based implementation (`projects/names3risk_pytorch/dockers/scripts/pytorch_training.py`) to identify missing features and guide the migration path.

### Key Findings

✅ **Completed**: OneCycleLR scheduler support added to `pl_lstm2risk.py` and `pl_transformer2risk.py`  
🔴 **Critical Gaps**: Custom tokenizer training, text field concatenation  
🟡 **Medium Priority**: Per-marketplace evaluation (considered ad-hoc, not needed)  
✅ **Already Handled**: Data loading, preprocessing, train/val/test splits

---

## 1. Detailed Task Comparison

### 1.1 Legacy train.py Tasks

| Task | Description | Implementation |
|------|-------------|----------------|
| **1. Feature File Loading** | Load and intersect tabular features from 3 region files (NA/EU/JP) | `REQUIRED_FEATURES` + region-specific files |
| **2. Multi-Region Data Loading** | Load parquet data from NA, EU, FE regions | `pl.concat()` with vertical stacking |
| **3. Text Field Concatenation** | Concatenate 4 fields: `email\|billing\|customer\|payment` | `pl.concat_str()` with separator |
| **4. Custom Tokenizer Training** | Train BPE tokenizer with compression tuning (~4K vocab) | `OrderTextTokenizer().train()` |
| **5. Data Filtering** | Filter amazon.com emails, deduplicate by customerId | `filter()` + `unique()` |
| **6. Train/Test Split** | 95/5 split, no validation set | `train_test_split(test_size=0.05)` |
| **7. Model Training** | Custom training loop with OneCycleLR | Manual `train_loop()` |
| **8. Per-Marketplace AUC** | Evaluate AUC separately for each marketplace | Group by `marketplaceCountryCode` |

### 1.2 Current pytorch_training.py Tasks

| Task | Description | Implementation |
|------|-------------|----------------|
| **1. Load Datasets** | Load pre-split train/val/test datasets | `load_datasets()` |
| **2. Load Tokenizer** | Load pretrained BERT tokenizer | `AutoTokenizer.from_pretrained()` |
| **3. Build Preprocessing** | Imputation, risk tables | `build_preprocessing_pipeline()` |
| **4. Create DataLoaders** | PyTorch DataLoaders | `create_dataloaders()` |
| **5. Model Selection** | Instantiate model (BimodalBert, LSTM2Risk, etc.) | `model_select()` |
| **6. Lightning Training** | PyTorch Lightning Trainer | `model_train()` |
| **7. Evaluation** | Overall test metrics | `model_inference()` |
| **8. ONNX Export** | Export for production | `model.export_to_onnx()` |

### 1.3 tabular_preprocessing.py Tasks (Upstream)

| Task | Description | Implementation |
|------|-------------|----------------|
| **1. Shard Loading** | Parallel shard reading | `combine_shards()` with multiprocessing |
| **2. Label Processing** | Numeric conversion, mapping | `pd.to_numeric()` |
| **3. Train/Val/Test Split** | Stratified 70/15/15 split | `train_test_split()` |
| **4. Format Conversion** | CSV/TSV/Parquet output | Configurable via `OUTPUT_FORMAT` |

---

## 2. Gap Analysis

### 2.1 🔴 Critical Gap: Custom Tokenizer Training

**Legacy Implementation**:
```python
tokenizer = OrderTextTokenizer().train(
    df_train.get_column("text").to_list()
)
config.n_embed = tokenizer.vocab_size  # ~4000 tokens
```

**Current Implementation**:
```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Uses pretrained BERT vocab (~30K tokens)
```

**Impact**:
- Names3Risk requires custom vocab optimized for name patterns
- BPE compression tuning creates compact, domain-specific vocabulary
- Pretrained BERT tokenizer not optimized for customer name data

**Required Changes**:
1. Create new SageMaker processing step: `tokenizer_training.py`
2. Load train data → concatenate text fields → train tokenizer
3. Save tokenizer artifacts to S3
4. Update `pytorch_training.py` to load custom tokenizer

### 2.2 🔴 Critical Gap: Text Field Concatenation

**Legacy Implementation**:
```python
text = pl.concat_str(
    [
        pl.col("emailAddress").fill_null("[MISSING]"),
        pl.col("billingAddressName").fill_null("[MISSING]"),
        pl.col("customerName").fill_null("[MISSING]"),
        pl.col("paymentAccountHolderName").fill_null("[MISSING]"),
    ],
    separator="|",
)
```

**Current Implementation**:
- Expects single pre-concatenated `text` field
- No multi-field fusion logic

**Impact**:
- Multi-field signal fusion missing
- Legacy separator logic (`|`) not preserved

**Required Changes**:
1. Add text concatenation to `tabular_preprocessing.py` OR
2. Add to `pytorch_training.py` data loading step
3. Maintain `[MISSING]` sentinel value convention

### 2.3 ✅ Resolved: OneCycleLR Scheduler

**Legacy Implementation**:
```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=EPOCHS * len(training_dataloader),
    pct_start=0.1,
)
scheduler.step()  # Called per batch
```

**Current Implementation** (✅ **NOW UPDATED**):
```python
# pl_lstm2risk.py & pl_transformer2risk.py
scheduler = OneCycleLR(
    optimizer,
    max_lr=self.lr,
    total_steps=self.trainer.estimated_stepping_batches,
    pct_start=0.1,  # 10% warmup (matches legacy)
    anneal_strategy="cos",
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
)
return {
    "optimizer": optimizer,
    "lr_scheduler": {
        "scheduler": scheduler,
        "interval": "step",  # Per batch (matches legacy)
    }
}
```

**Status**: ✅ **COMPLETE** - Both models now use OneCycleLR

### 2.4 🟡 Optional: Per-Marketplace AUC Evaluation

**Legacy Implementation**:
```python
for (mp,), df_mp in df_test.group_by("marketplaceCountryCode"):
    mp_auc[f"{mp}_auc"] = test_loop(model, dataloader).item()
logger.log_metrics(epoch, **mp_auc)
```

**Current Implementation**:
- Only overall test AUC computed
- No per-marketplace breakdown

**Decision**: 🟡 **NOT REQUIRED** - Per-marketplace evaluation is ad-hoc analysis, not core training requirement

---

## 3. Data Pipeline Comparison

### 3.1 Legacy Pipeline (Single Script)

```
Raw Parquet (NA/EU/FE)
  ↓
Filter amazon.com emails
  ↓
Concatenate text fields (email|billing|customer|payment)
  ↓
Train custom tokenizer (BPE)
  ↓
Deduplicate by customerId
  ↓
Train/Test split (95/5)
  ↓
Training with OneCycleLR
```

### 3.2 Current Pipeline (Multi-Step)

```
Step 1: tabular_preprocessing.py
  ↓
Raw Parquet → Train/Val/Test (70/15/15)
  ↓
Step 2: pytorch_training.py
  ↓
Load BERT tokenizer → Train with Lightning
  ↓
ONNX Export
```

### 3.3 Required Pipeline Updates

```
Step 1: tabular_preprocessing.py (UPDATED)
  ↓
Raw Parquet → Concatenate text fields → Train/Val/Test
  ↓
Step 2: tokenizer_training.py (NEW)
  ↓
Load train text → Train BPE tokenizer → Save artifacts
  ↓
Step 3: pytorch_training.py (UPDATED)
  ↓
Load custom tokenizer → Train with OneCycleLR → ONNX Export
```

---

## 4. Implementation Recommendations

### 4.1 Phase 1: Text Field Concatenation (High Priority)

**Option A**: Update `tabular_preprocessing.py`
```python
# In main() function, after loading data:
if "emailAddress" in df.columns:
    df = df.with_columns(
        text=pl.concat_str([
            pl.col("emailAddress").fill_null("[MISSING]"),
            pl.col("billingAddressName").fill_null("[MISSING]"),
            pl.col("customerName").fill_null("[MISSING]"),
            pl.col("paymentAccountHolderName").fill_null("[MISSING]"),
        ], separator="|")
    )
```

**Option B**: Update `pytorch_training.py`
```python
# In load_datasets() function:
def concatenate_text_fields(df):
    text_cols = ["emailAddress", "billingAddressName", 
                 "customerName", "paymentAccountHolderName"]
    df["text"] = df[text_cols].fillna("[MISSING]").agg("|".join, axis=1)
    return df

train_df = concatenate_text_fields(train_df)
```

**Recommendation**: Use Option A (preprocessing step) for consistency with data processing best practices.

### 4.2 Phase 2: Custom Tokenizer Training (High Priority)

Create new script: `projects/names3risk_pytorch/dockers/scripts/tokenizer_training.py`

```python
#!/usr/bin/env python3
"""
Train custom BPE tokenizer for Names3Risk.

Usage:
    python tokenizer_training.py \
        --train_data /opt/ml/processing/input/train \
        --output_dir /opt/ml/processing/output \
        --vocab_size 4000
"""

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

def train_tokenizer(texts, vocab_size=4000):
    """Train BPE tokenizer matching legacy OrderTextTokenizer."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MISSING]"],
        show_progress=True,
    )
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # Add post-processor for [CLS] and [SEP]
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    
    return tokenizer

def main():
    # Load train data
    train_df = pd.read_parquet("/opt/ml/processing/input/train")
    texts = train_df["text"].tolist()
    
    # Train tokenizer
    tokenizer = train_tokenizer(texts, vocab_size=4000)
    
    # Save artifacts
    output_dir = Path("/opt/ml/processing/output")
    tokenizer.save(str(output_dir / "tokenizer.json"))
    
    # Save vocab for legacy compatibility
    vocab = tokenizer.get_vocab()
    with open(output_dir / "vocab.json", "w") as f:
        json.dump(vocab, f)
```

### 4.3 Phase 3: Update pytorch_training.py (High Priority)

```python
# Replace BERT tokenizer loading:
# OLD:
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# NEW:
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("/opt/ml/input/data/tokenizer/tokenizer.json")

# Update hyperparameters:
hyperparams.n_embed = tokenizer.get_vocab_size()  # ~4000 instead of 30K
```

### 4.4 Phase 4: Update model_select() (Medium Priority)

```python
def model_select(config: Dict, model_class: str):
    """Select model based on model_class parameter."""
    
    if model_class == "lstm2risk":
        from lightning_models.bimodal import LSTM2RiskLightning
        return LSTM2RiskLightning(hyperparams)
    
    elif model_class == "transformer2risk":
        from lightning_models.bimodal import Transformer2RiskLightning
        return Transformer2RiskLightning(hyperparams)
    
    elif model_class == "bimodal_bert":
        from lightning_models.bimodal import BimodalBert
        return BimodalBert(config)
    
    else:
        raise ValueError(f"Unknown model_class: {model_class}")
```

---

## 5. Migration Checklist

### Phase 1: Scheduler Update ✅
- [x] Update `pl_lstm2risk.py` with OneCycleLR
- [x] Update `pl_transformer2risk.py` with OneCycleLR
- [x] Verify scheduler parameters match legacy (pct_start=0.1)

### Phase 2: Text Concatenation 🔴
- [ ] Add text field concatenation to `tabular_preprocessing.py`
- [ ] Test with 4 required fields (email, billing, customer, payment)
- [ ] Preserve `[MISSING]` sentinel convention
- [ ] Verify separator (`|`) matches legacy

### Phase 3: Custom Tokenizer 🔴
- [ ] Create `tokenizer_training.py` script
- [ ] Implement BPE training (vocab_size=4000)
- [ ] Add special tokens ([PAD], [UNK], [CLS], [SEP], [MISSING])
- [ ] Test compression ratio vs legacy
- [ ] Add SageMaker processing step to pipeline

### Phase 4: Integration 🟡
- [ ] Update `pytorch_training.py` to load custom tokenizer
- [ ] Update `model_select()` to support lstm2risk/transformer2risk
- [ ] Update hyperparameters (n_embed from tokenizer)
- [ ] Add integration tests

### Phase 5: Validation 🟡
- [ ] Compare vocab size (legacy vs new)
- [ ] Compare model architecture parameter counts
- [ ] Compare training dynamics (loss curves)
- [ ] Compare final AUC scores

---

## 6. Key Architectural Differences

### 6.1 Training Loop

| Aspect | Legacy | Current |
|--------|--------|---------|
| Framework | Manual PyTorch | PyTorch Lightning |
| Optimizer Step | Manual `optimizer.step()` | Automatic via Lightning |
| Gradient Accumulation | Manual batching | Lightning `accumulate_grad_batches` |
| Distributed Training | Not supported | FSDP/DDP via Lightning |
| Mixed Precision | Manual AMP | Lightning `precision=16` |
| Checkpointing | Manual `torch.save()` | Lightning callbacks |

### 6.2 Model Architecture

| Component | Legacy | Current |
|-----------|--------|---------|
| Text Encoder | LSTM or Transformer | ✅ Same (LSTMEncoder/TransformerEncoder) |
| Tabular Encoder | BatchNorm + MLP | ✅ Same |
| Fusion | ResidualBlocks | ✅ Same |
| Tokenizer | Custom BPE (~4K) | 🔴 BERT pretrained (~30K) |
| Scheduler | OneCycleLR | ✅ OneCycleLR (NOW UPDATED) |

---

## 7. Performance Considerations

### 7.1 Tokenizer Impact

**Legacy Custom BPE**:
- Vocab size: ~4000 tokens
- Embedding parameters: 4000 × 16 = 64K params
- Training time: ~30 seconds on train data

**Current BERT**:
- Vocab size: ~30,000 tokens
- Embedding parameters: 30000 × 128 = 3.84M params
- Pretrained (no training needed)

**Impact**:
- Custom tokenizer: 60× fewer embedding parameters
- Domain-specific compression (better for names)
- Faster training convergence on small vocab

### 7.2 Training Speed

**Improvements in Current**:
- ✅ Distributed training (FSDP/DDP)
- ✅ Mixed precision (FP16)
- ✅ Gradient accumulation
- ✅ DataLoader optimizations

**Regressions to Address**:
- 🔴 Larger vocab (30K vs 4K) → slower embedding lookup
- 🔴 No custom tokenizer compression

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
def test_text_concatenation():
    """Test text field concatenation matches legacy."""
    df = pd.DataFrame({
        "emailAddress": ["user@example.com"],
        "billingAddressName": ["John Doe"],
        "customerName": ["Jane Doe"],
        "paymentAccountHolderName": ["J. Doe"]
    })
    expected = "user@example.com|John Doe|Jane Doe|J. Doe"
    result = concatenate_text_fields(df)["text"][0]
    assert result == expected

def test_tokenizer_vocab_size():
    """Test custom tokenizer vocab size."""
    tokenizer = train_tokenizer(train_texts, vocab_size=4000)
    assert tokenizer.get_vocab_size() == 4000

def test_onecycle_scheduler():
    """Test OneCycleLR configuration."""
    model = LSTM2RiskLightning(hyperparams)
    config = model.configure_optimizers()
    scheduler = config["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, OneCycleLR)
    assert scheduler.pct_start == 0.1
```

### 8.2 Integration Tests

```python
def test_end_to_end_training():
    """Test full training pipeline with custom tokenizer."""
    # 1. Preprocess data with text concatenation
    # 2. Train custom tokenizer
    # 3. Train LSTM2Risk model
    # 4. Verify AUC matches legacy baseline
    pass
```

---

## 9. Timeline Estimate

| Phase | Tasks | Effort | Priority |
|-------|-------|--------|----------|
| **Phase 1** | OneCycleLR scheduler | ✅ 2 hours | 🔴 Critical |
| **Phase 2** | Text concatenation | 4 hours | 🔴 Critical |
| **Phase 3** | Custom tokenizer | 8 hours | 🔴 Critical |
| **Phase 4** | Integration | 8 hours | 🟡 High |
| **Phase 5** | Testing & validation | 8 hours | 🟡 High |
| **Total** | | **30 hours** | |

---

## 10. Conclusion

### Summary of Gaps

1. ✅ **OneCycleLR Scheduler**: RESOLVED - Updated both models
2. 🔴 **Custom Tokenizer Training**: CRITICAL - Requires new processing step
3. 🔴 **Text Field Concatenation**: CRITICAL - Missing in preprocessing
4. 🟡 **Per-Marketplace AUC**: OPTIONAL - Ad-hoc analysis only

### Next Steps

1. ✅ ~~Update `configure_optimizers()` in Lightning modules~~
2. Add text concatenation to `tabular_preprocessing.py`
3. Create `tokenizer_training.py` processing step
4. Update `pytorch_training.py` to load custom tokenizer
5. Add integration tests and validation

### Success Criteria

- [ ] Custom tokenizer vocab size matches legacy (~4K tokens)
- [ ] Text concatenation preserves legacy format (`email|billing|customer|payment`)
- [ ] OneCycleLR scheduler matches legacy (10% warmup, cosine decay)
- [ ] Model parameter count matches legacy (within 5%)
- [ ] Final test AUC matches legacy baseline (within 1%)

---

## References

- Legacy code: `projects/names3risk_legacy/train.py`
- Current implementation: `projects/names3risk_pytorch/dockers/scripts/pytorch_training.py`
- Preprocessing: `projects/names3risk_pytorch/dockers/scripts/tabular_preprocessing.py`
- Lightning models: `projects/names3risk_pytorch/dockers/lightning_models/bimodal/`
- Architecture analysis: `slipbox/4_analysis/2026-01-05_names3risk_pytorch_component_correspondence_analysis.md`
- PyTorch Lightning docs: https://lightning.ai/docs/pytorch/stable/common/optimization.html
