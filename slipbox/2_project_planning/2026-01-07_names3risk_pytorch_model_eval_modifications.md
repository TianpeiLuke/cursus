---
tags:
  - implementation_plan
  - model_evaluation
  - tokenizer_integration
  - names3risk
  - pytorch
keywords:
  - pytorch_model_eval
  - custom_tokenizer
  - BPE_tokenizer
  - lstm2risk
  - transformer2risk
  - model_evaluation
topics:
  - evaluation_infrastructure
  - tokenizer_loading
  - collate_functions
language: python
date of note: 2026-01-07
---

# Names3Risk PyTorch Model Evaluation Modifications

## Executive Summary

This document outlines the required modifications to `pytorch_model_eval.py` to support custom BPE tokenizers for Names3Risk models (lstm2risk, transformer2risk). The current implementation only supports BERT tokenizers, which prevents evaluation of Names3Risk models trained with custom tokenizers.

**Status:** Implementation Plan  
**Priority:** High - Required for Names3Risk model evaluation  
**Complexity:** Medium - Branching logic required in 2 functions

## Problem Statement

### Current Limitation

The `pytorch_model_eval.py` script currently hardcodes BERT tokenizer loading:

```python
# Line ~562 in load_model_artifacts()
tokenizer_name = config.get("tokenizer", "bert-base-multilingual-cased")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # ❌ BERT only
```

This prevents evaluation of Names3Risk models that use custom BPE tokenizers saved during training.

### Root Cause

- **Missing branching logic**: No model-class-based tokenizer selection
- **Wrong collate function**: Uses BERT collate for all models
- **No PAD token extraction**: Doesn't extract PAD token from custom tokenizers

## Solution Overview

Add branching logic similar to `pytorch_training.py` to:

1. **Detect model class** from config
2. **Load custom tokenizer** for Names3Risk models from saved `tokenizer.json`
3. **Load BERT tokenizer** for other models (existing behavior)
4. **Select appropriate collate function** based on model architecture

## Detailed Implementation Plan

### Phase 1: Update Tokenizer Loading (Function: `load_model_artifacts`)

**Location:** `projects/bsm_pytorch/dockers/pytorch_model_eval.py`, lines ~510-590

**Current Code:**
```python
def load_model_artifacts(model_dir: str):
    # ... existing code ...
    
    # Reconstruct tokenizer - ONLY BERT
    tokenizer_name = config.get("tokenizer", "bert-base-multilingual-cased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Reconstructed tokenizer: {tokenizer_name}")
```

**Modified Code:**
```python
def load_model_artifacts(model_dir: str):
    # ... existing code ...
    
    # NEW: Branch based on model class
    model_class = config.get("model_class", "bimodal_bert")
    logger.info(f"Loading tokenizer for model class: {model_class}")
    
    if model_class in ["lstm2risk", "transformer2risk"]:
        # BRANCH 1: Load custom BPE tokenizer from saved artifacts
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        
        if os.path.exists(tokenizer_path):
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.info(f"✓ Loaded custom BPE tokenizer from {tokenizer_path}")
            logger.info(f"  Vocabulary size: {tokenizer.get_vocab_size()}")
            
            # Extract PAD token ID for collate function
            pad_token_id = tokenizer.token_to_id("[PAD]")
            config["pad_token_id"] = pad_token_id if pad_token_id is not None else 0
            logger.info(f"  PAD token ID: {config['pad_token_id']}")
        else:
            raise FileNotFoundError(
                f"Custom tokenizer required for {model_class} but not found at {tokenizer_path}. "
                f"Please ensure the model was trained with pytorch_training.py which saves tokenizer.json."
            )
    else:
        # BRANCH 2: Load BERT tokenizer for other models (existing behavior)
        tokenizer_name = config.get("tokenizer", "bert-base-multilingual-cased")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        config["pad_token_id"] = tokenizer.pad_token_id
        logger.info(f"✓ Loaded BERT tokenizer: {tokenizer_name}")
        logger.info(f"  PAD token ID: {config['pad_token_id']}")
```

**Key Changes:**
- ✅ Detect model class from config
- ✅ Branch: Custom tokenizer for Names3Risk, BERT for others
- ✅ Load `tokenizer.json` from model directory
- ✅ Extract PAD token ID from custom tokenizer
- ✅ Store PAD token ID in config for collate function
- ✅ Add error handling with helpful message

### Phase 2: Update Collate Function Selection (Function: `create_dataloader`)

**Location:** `projects/bsm_pytorch/dockers/pytorch_model_eval.py`, lines ~895-925

**Current Code:**
```python
def create_dataloader(pipeline_dataset: PipelineDataset, config: Dict[str, Any]) -> DataLoader:
    # Use unified collate function for all model types
    logger.info(f"Using collate batch for model: {config.get('model_class', 'bimodal')}")
    
    collate_batch = build_collate_batch(
        input_ids_key=config.get("text_input_ids_key", "input_ids"),
        attention_mask_key=config.get("text_attention_mask_key", "attention_mask"),
    )
```

**Modified Code:**
```python
def create_dataloader(pipeline_dataset: PipelineDataset, config: Dict[str, Any]) -> DataLoader:
    """
    Create DataLoader with appropriate collate function.
    
    Supports:
    - LSTM models: Length-sorted batches for pack_padded_sequence
    - Transformer models: Attention masking and block_size truncation
    - BERT models: Standard BERT batching
    """
    model_class = config.get("model_class", "bimodal_bert")
    logger.info(f"Selecting collate function for model: {model_class}")
    
    if model_class in ["lstm2risk", "bimodal_lstm"]:
        # LSTM models: Need length sorting for pack_padded_sequence
        pad_token = config.get("pad_token_id", 0)
        collate_batch = build_lstm2risk_collate_fn(pad_token=pad_token)
        logger.info(f"✓ Using LSTM2Risk collate function (pad_token={pad_token})")
        logger.info("  - Sequences sorted by length (descending)")
        logger.info("  - Includes text_length for pack_padded_sequence")
        
    elif model_class in ["transformer2risk", "bimodal_transformer"]:
        # Transformer models: Need attention masking and block_size truncation
        pad_token = config.get("pad_token_id", 0)
        block_size = config.get("max_sen_len", 100)
        collate_batch = build_transformer2risk_collate_fn(
            pad_token=pad_token, 
            block_size=block_size
        )
        logger.info(f"✓ Using Transformer2Risk collate function")
        logger.info(f"  - Block size: {block_size}")
        logger.info(f"  - Includes attention mask (pad_token={pad_token})")
        
    else:
        # Default: BERT-based models
        collate_batch = build_collate_batch(
            input_ids_key=config.get("text_input_ids_key", "input_ids"),
            attention_mask_key=config.get("text_attention_mask_key", "attention_mask"),
        )
        logger.info(f"✓ Using default BERT collate function for {model_class}")
    
    batch_size = config.get("batch_size", 32)
    dataloader = DataLoader(
        pipeline_dataset,
        collate_fn=collate_batch,
        batch_size=batch_size,
        shuffle=False,
    )
    logger.info(f"Created DataLoader with batch_size={batch_size}")
    
    return dataloader
```

**Key Changes:**
- ✅ Branch by model class (lstm2risk, transformer2risk, others)
- ✅ Use `build_lstm2risk_collate_fn()` for LSTM models
- ✅ Use `build_transformer2risk_collate_fn()` for Transformer models
- ✅ Use `build_collate_batch()` for BERT models (existing)
- ✅ Pass PAD token ID from config to collate functions
- ✅ Add detailed logging for debugging

### Phase 3: Update Imports

**Location:** Top of file, after existing imports

**Add:**
```python
from processing.dataloaders.names3risk_collate import (
    build_lstm2risk_collate_fn,
    build_transformer2risk_collate_fn,
)
```

**Note:** These imports are already available in the project structure from the training script refactoring.

## Implementation Checklist

### Pre-Implementation

- [x] Analyze `pytorch_training.py` tokenizer handling
- [x] Analyze `pytorch_model_eval.py` current implementation
- [x] Identify required changes
- [x] Create implementation plan document

### Implementation

- [ ] Add import for custom collate functions
- [ ] Modify `load_model_artifacts()` function
  - [ ] Add model class detection
  - [ ] Add branching logic for tokenizer loading
  - [ ] Add custom tokenizer loading from `tokenizer.json`
  - [ ] Add PAD token extraction
  - [ ] Add error handling
- [ ] Modify `create_dataloader()` function
  - [ ] Add branching logic for collate function selection
  - [ ] Add LSTM2Risk collate function support
  - [ ] Add Transformer2Risk collate function support
  - [ ] Add detailed logging

### Testing

- [ ] Test with LSTM2Risk model
  - [ ] Verify custom tokenizer loads correctly
  - [ ] Verify PAD token ID extracted
  - [ ] Verify LSTM collate function used
  - [ ] Verify predictions generated
  - [ ] Verify metrics computed
- [ ] Test with Transformer2Risk model
  - [ ] Verify custom tokenizer loads correctly
  - [ ] Verify attention masking works
  - [ ] Verify block size truncation works
  - [ ] Verify predictions generated
  - [ ] Verify metrics computed
- [ ] Test with existing BERT models (regression test)
  - [ ] Verify BERT tokenizer still works
  - [ ] Verify BERT collate function still works
  - [ ] Verify backward compatibility maintained

## Technical Considerations

### 1. Tokenizer Type Handling

**Challenge:** Custom tokenizers use HuggingFace `Tokenizer` class, BERT uses `AutoTokenizer`

**Solution:** Type branching with proper imports:
```python
if model_class in ["lstm2risk", "transformer2risk"]:
    from tokenizers import Tokenizer  # HuggingFace tokenizers
    tokenizer = Tokenizer.from_file(...)
else:
    from transformers import AutoTokenizer  # BERT tokenizers
    tokenizer = AutoTokenizer.from_pretrained(...)
```

### 2. PAD Token ID Storage

**Challenge:** Collate functions need PAD token ID, but it's model-specific

**Solution:** Store in config dict during tokenizer loading:
```python
config["pad_token_id"] = pad_token_id  # Available to all downstream functions
```

### 3. Collate Function Compatibility

**Challenge:** Different models need different collate functions

**Solution:** Factory pattern with model class detection:
```python
collate_fn = select_collate_function(model_class, pad_token_id, block_size)
```

### 4. Error Handling

**Challenge:** Custom tokenizer might not exist if model wasn't trained correctly

**Solution:** Raise informative error with remediation steps:
```python
raise FileNotFoundError(
    f"Custom tokenizer required for {model_class} but not found. "
    f"Please ensure model was trained with pytorch_training.py."
)
```

## File Locations

### Source Files
- **pytorch_training.py**: `projects/names3risk_pytorch/dockers/pytorch_training.py`
  - Reference implementation (lines 1066-1094, 1195-1229)
  
- **pytorch_model_eval.py**: `projects/bsm_pytorch/dockers/pytorch_model_eval.py`
  - Target file for modifications

### Collate Functions
- **names3risk_collate.py**: `projects/names3risk_pytorch/dockers/processing/dataloaders/names3risk_collate.py`
  - `build_lstm2risk_collate_fn()` - LSTM collate factory
  - `build_transformer2risk_collate_fn()` - Transformer collate factory

### Model Artifacts (Runtime)
- **tokenizer.json**: `/opt/ml/model/tokenizer.json` (saved by pytorch_training.py)
- **vocab.json**: `/opt/ml/model/vocab.json` (saved by pytorch_training.py)
- **hyperparameters.json**: `/opt/ml/model/hyperparameters.json` (contains model_class)

## Verification Strategy

### Unit Test Scenarios

1. **LSTM2Risk Model:**
   - Input: Model trained with custom tokenizer
   - Expected: Tokenizer loaded, LSTM collate used, predictions generated

2. **Transformer2Risk Model:**
   - Input: Model trained with custom tokenizer
   - Expected: Tokenizer loaded, Transformer collate used, attention masks correct

3. **BERT Model (Regression):**
   - Input: Existing BERT model
   - Expected: BERT tokenizer loaded, BERT collate used, no breaking changes

### Integration Test Scenarios

1. **End-to-End Evaluation:**
   - Train Names3Risk model → Save artifacts → Load in eval script → Generate predictions
   - Expected: All steps succeed, metrics computed correctly

2. **Format Preservation:**
   - Evaluate with CSV/TSV/Parquet inputs
   - Expected: Output format matches input format

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking existing BERT models | High | Low | Add comprehensive tests for backward compatibility |
| Custom tokenizer not found | High | Medium | Add clear error messages with remediation steps |
| Wrong collate function selected | High | Low | Add model class validation and logging |
| PAD token ID incorrect | Medium | Low | Add validation after extraction |
| Import conflicts | Low | Low | Use conditional imports |

## Dependencies

### Python Packages
- `tokenizers` (HuggingFace) - For custom BPE tokenizer loading
- `transformers` (HuggingFace) - For BERT tokenizer loading (existing)
- `torch` - For collate function implementations (existing)

### Internal Modules
- `processing.dataloaders.names3risk_collate` - Custom collate functions
- `processing.datasets.pipeline_datasets` - Dataset handling (existing)
- `lightning_models.utils.pl_train` - Model loading utilities (existing)

## Success Criteria

- ✅ Custom tokenizers load successfully for Names3Risk models
- ✅ Appropriate collate functions selected based on model class
- ✅ PAD token IDs extracted and used correctly
- ✅ Predictions generated with correct shapes
- ✅ Metrics computed accurately
- ✅ Backward compatibility maintained for BERT models
- ✅ Clear error messages for missing artifacts
- ✅ Comprehensive logging for debugging

## Related Documents

- **[Names3Risk PyTorch Training Modifications](./2026-01-07_names3risk_pytorch_training_modifications.md)** - Training script modifications that save custom tokenizers
- **[Names3Risk PyTorch Training End-to-End Analysis](../4_analysis/2026-01-07_names3risk_pytorch_training_end_to_end_analysis.md)** - Comprehensive analysis of training script tokenizer flow
- **[Names3Risk PyTorch Reorganization Design](../1_design/names3risk_pytorch_reorganization_design.md)** - Overall architecture design

## Next Steps

1. **Immediate:** Implement modifications to `pytorch_model_eval.py`
2. **Testing:** Create test scenarios for all model types
3. **Documentation:** Update script documentation with tokenizer handling details
4. **Validation:** Run integration tests with trained Names3Risk models

---

**Document Status:** ✅ Ready for Implementation  
**Last Updated:** 2026-01-07  
**Implementer:** Ready for Act Mode execution
