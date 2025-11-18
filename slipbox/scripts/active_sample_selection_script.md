---
tags:
  - code
  - processing_script
  - active_learning
  - semi_supervised_learning
  - sample_selection
keywords:
  - active sample selection
  - semi-supervised learning
  - active learning
  - confidence-based sampling
  - uncertainty sampling
  - diversity sampling
  - BADGE algorithm
  - pseudo-labeling
  - sample selection strategies
topics:
  - active learning workflows
  - semi-supervised learning
  - intelligent sampling
  - model predictions
language: python
date of note: 2025-11-18
---

# Active Sample Selection Script Documentation

## Overview

The `active_sample_selection.py` script implements intelligent sample selection from model predictions for two primary machine learning workflows:

1. **Semi-Supervised Learning (SSL)**: Selects high-confidence predictions for automatic pseudo-labeling to augment training data
2. **Active Learning (AL)**: Selects uncertain or diverse samples for efficient human labeling

The script supports multiple selection strategies tailored to each use case and provides flexible integration with upstream prediction sources.

## Purpose and Major Tasks

### Primary Purpose
Intelligently select high-value samples from a pool of unlabeled or partially labeled data based on model predictions, enabling efficient use of computational or human labeling resources.

### Major Tasks
1. **Data Loading**: Load inference predictions from various upstream sources (XGBoost, LightGBM, PyTorch, Bedrock, Label Rulesets)
2. **Score Normalization**: Convert diverse score formats into standardized probability distributions
3. **Strategy-Based Selection**: Apply appropriate sampling strategy (confidence, uncertainty, diversity, or hybrid)
4. **Metadata Generation**: Track selection provenance, scores, and configuration
5. **Format Preservation**: Maintain input file format (CSV, TSV, Parquet) in outputs

## Script Contract

### Entry Point
```
active_sample_selection.py
```

### Input Paths
| Path | Location | Description |
|------|----------|-------------|
| `evaluation_data` | `/opt/ml/processing/input/evaluation_data` | Directory containing model predictions with probability scores |

### Output Paths
| Path | Location | Description |
|------|----------|-------------|
| `selected_samples` | `/opt/ml/processing/output/selected_samples` | Directory containing selected samples with selection metadata |
| `selection_metadata` | `/opt/ml/processing/output/selection_metadata` | Directory containing selection configuration and statistics |

### Required Environment Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `SELECTION_STRATEGY` | Sampling strategy name | `"confidence_threshold"`, `"uncertainty"` |
| `USE_CASE` | Use case validation mode | `"ssl"`, `"active_learning"`, `"auto"` |
| `ID_FIELD` | Column name for sample IDs | `"id"`, `"sample_id"` |
| `LABEL_FIELD` | Column name for labels (if present) | `"label"`, `"ground_truth"` |

### Optional Environment Variables

#### Core Parameters
| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_FORMAT` | `"csv"` | Output file format: `"csv"`, `"tsv"`, or `"parquet"` |
| `RANDOM_SEED` | `"42"` | Random seed for reproducibility |
| `SCORE_FIELD` | `""` | Single score column name for binary/custom scoring |
| `SCORE_FIELD_PREFIX` | `"prob_class_"` | Prefix for multiple probability columns |

#### SSL-Specific Parameters
| Variable | Default | Description | Range |
|----------|---------|-------------|-------|
| `CONFIDENCE_THRESHOLD` | `"0.9"` | Minimum confidence for selection | 0.5 - 1.0 |
| `MAX_SAMPLES` | `"0"` | Maximum samples to select (0 = no limit) | 0 - ∞ |
| `K_PER_CLASS` | `"100"` | Samples per class for balanced selection | 1 - ∞ |

#### Active Learning Parameters
| Variable | Default | Description | Options |
|----------|---------|-------------|---------|
| `UNCERTAINTY_MODE` | `"margin"` | Uncertainty calculation method | `"margin"`, `"entropy"`, `"least_confidence"` |
| `BATCH_SIZE` | `"32"` | Number of samples to select | 1 - ∞ |
| `METRIC` | `"euclidean"` | Distance metric for diversity | `"euclidean"`, `"cosine"` |

### Job Arguments
| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--job_type` | `str` | Yes | Job type identifier (e.g., `"ssl_selection"`, `"active_learning_selection"`) |

## Input Data Structure

### Expected Input Format
The script accepts predictions from various upstream sources:

```
evaluation_data/
├── predictions.csv (or .tsv, .parquet)
└── _SUCCESS (optional marker)
```

### Required Columns
- **ID Column**: Configurable via `ID_FIELD` (default: `"id"`)
- **Probability Columns**: One of:
  - `prob_class_0`, `prob_class_1`, ... (standard format)
  - `confidence_score`, `prediction_score` (LLM/Bedrock format)
  - `rule_score`, `label_confidence` (ruleset format)
  - Custom column specified via `SCORE_FIELD`

### Optional Columns
- **Label Column**: Ground truth labels (for validation)
- **Feature Columns**: Numeric features (preserved in output)
- **Embedding Columns**: `emb_0`, `emb_1`, ... (required for diversity/BADGE strategies)

### Supported Input Sources
1. **Model Inference**: XGBoost, LightGBM, PyTorch inference outputs
2. **LLM Inference**: Bedrock/LLM processing outputs
3. **Rule-Based**: Label ruleset execution outputs
4. **Model Evaluation**: Evaluation outputs with predictions

## Output Data Structure

### Selected Samples Output
```
selected_samples/
├── selected_samples.{csv|tsv|parquet}
└── _SUCCESS
```

**Columns in Output**:
- All original input columns (preserved)
- `selection_score`: Strategy-specific score (confidence/uncertainty/diversity)
- `selection_rank`: Rank within selected batch (1 = highest priority)

### Selection Metadata Output
```
selection_metadata/
└── selection_metadata.json
```

**Metadata Contents**:
```json
{
  "strategy": "confidence_threshold",
  "use_case": "ssl",
  "batch_size": 32,
  "total_pool_size": 10000,
  "selected_count": 1500,
  "strategy_config": {
    "confidence_threshold": "0.9",
    "random_seed": "42"
  },
  "timestamp": "2025-11-18T11:30:00",
  "job_type": "ssl_selection"
}
```

## Selection Strategies

### SSL Strategies (Confidence-Based)

#### 1. Confidence Threshold Strategy
**Use Case**: Simple high-confidence filtering for pseudo-labeling

**Algorithm**:
```python
1. For each sample, compute max_prob = max(P(class_i))
2. Select samples where max_prob >= threshold
3. Optionally limit to top-k by sorting on max_prob
```

**Parameters**:
- `CONFIDENCE_THRESHOLD`: Minimum confidence (default: 0.9)
- `MAX_SAMPLES`: Optional sample limit

**When to Use**:
- Simple SSL workflows
- Single threshold works well across classes
- Large unlabeled pools

**Complexity**: O(n)

#### 2. Top-K Per Class Strategy
**Use Case**: Balanced pseudo-labeling across classes

**Algorithm**:
```python
1. For each sample, compute predicted_class = argmax(P(class_i))
2. For each class c:
   a. Find all samples predicted as class c
   b. Sort by confidence
   c. Select top-k samples
3. Combine selections from all classes
```

**Parameters**:
- `K_PER_CLASS`: Samples per class (default: 100)

**When to Use**:
- Imbalanced class distributions
- Need uniform pseudo-label coverage
- Sufficient samples per class available

**Complexity**: O(n log n)

### Active Learning Strategies

#### 3. Uncertainty Strategy
**Use Case**: Find samples in decision boundary regions

**Uncertainty Modes**:

1. **Margin Sampling** (default):
   ```python
   uncertainty = 1 - (P(class_1st) - P(class_2nd))
   ```
   - Measures gap between top two predictions
   - High uncertainty when classes are close

2. **Entropy Sampling**:
   ```python
   uncertainty = -Σ P(class_i) * log(P(class_i))
   ```
   - Shannon entropy of probability distribution
   - High uncertainty when distribution is uniform

3. **Least Confidence**:
   ```python
   uncertainty = 1 - P(class_max)
   ```
   - Inverse of maximum probability
   - Simple uncertainty measure

**Parameters**:
- `UNCERTAINTY_MODE`: Uncertainty calculation method
- `BATCH_SIZE`: Number of samples to select

**When to Use**:
- Focus on decision boundaries
- Fast iterations needed
- No embedding/feature requirements

**Complexity**: O(n)

#### 4. Diversity Strategy
**Use Case**: Representative coverage of feature space

**Algorithm** (k-center/farthest-first):
```python
1. Initialize with random sample
2. Compute distances from selected to all unselected
3. Select farthest sample from selected set
4. Update minimum distances
5. Repeat until batch_size reached
```

**Requirements**:
- Embeddings (`emb_*` columns) OR
- Feature columns (specified in config)

**Parameters**:
- `BATCH_SIZE`: Number of samples
- `METRIC`: Distance metric (euclidean/cosine)

**When to Use**:
- Want representative sample coverage
- Have quality embeddings/features
- Batch size < 10K (due to O(n²) complexity)

**Complexity**: O(n² * d) where d = embedding dimension

#### 5. BADGE Strategy
**Use Case**: Hybrid uncertainty + diversity

**Algorithm**:
```python
1. Compute pseudo-labels: y_pseudo = argmax(P(class_i))
2. Create one-hot encoding: Y_onehot
3. Compute gradient embeddings:
   Δ = P - Y_onehot  # prediction error
   G = Δ ⊗ features  # gradient embeddings
4. Apply k-center on gradient embeddings
```

**Requirements**:
- Features or embeddings
- Probability distributions

**Parameters**:
- `BATCH_SIZE`: Number of samples
- `METRIC`: Distance metric

**When to Use**:
- Want both informative AND diverse samples
- Larger computational budget available
- Strong baseline model available

**Complexity**: O(n² * d * c) where c = number of classes

## Key Functions and Tasks

### Data Loading Component

#### `load_inference_data(inference_data_dir, id_field)`
**Purpose**: Load and validate inference predictions with format auto-detection

**Algorithm**:
1. Search for data files (.csv, .tsv, .parquet)
2. Detect file format from extension
3. Load using appropriate pandas reader
4. Validate ID field presence
5. Return DataFrame and detected format

**Returns**: `Tuple[pd.DataFrame, str]` - Data and format string

#### `extract_score_columns(df, score_field, score_prefix)`
**Purpose**: Identify probability/score columns using priority-based search

**Priority Order**:
1. Explicit `SCORE_FIELD` if specified
2. Columns matching `SCORE_FIELD_PREFIX`
3. Auto-detection of LLM patterns (confidence_score, prediction_score)
4. Auto-detection of ruleset patterns (rule_score, label_confidence)

**Returns**: `List[str]` - List of score column names

#### `normalize_scores_to_probabilities(df, score_cols)`
**Purpose**: Convert diverse score formats to standard probability distributions

**Algorithm**:
1. Check if already in prob_class_* format
2. Extract score matrix
3. Check if rows sum to 1.0
4. Apply softmax normalization if needed
5. Create standardized prob_class_* columns

**Returns**: `pd.DataFrame` - Normalized data

### Sampling Strategy Component

#### `ConfidenceThresholdSampler`
**Methods**:
- `select_batch(probabilities, indices)`: Select high-confidence samples

**Implementation**:
```python
max_probs = np.max(probabilities, axis=1)
high_conf_mask = max_probs >= threshold
selected = indices[high_conf_mask]
```

#### `TopKPerClassSampler`
**Methods**:
- `select_batch(probabilities, indices)`: Select top-k per predicted class

**Implementation**:
```python
for each class:
    class_samples = samples where argmax(prob) == class
    sort by max(prob) descending
    select top k
```

#### `UncertaintySampler`
**Methods**:
- `compute_scores(probabilities)`: Calculate uncertainty scores
- `select_batch(probabilities, batch_size, indices)`: Select uncertain samples

**Implementation Details**:
- Margin: `score = -(sorted_prob[-1] - sorted_prob[-2])`
- Entropy: `score = -Σ p * log(p)`
- Least Confidence: `score = 1 - max(prob)`

#### `DiversitySampler`
**Methods**:
- `select_batch(embeddings, batch_size, indices)`: K-center selection
- `_compute_distances(X, Y)`: Pairwise distance computation

**Implementation** (farthest-first):
```python
selected = [random_point]
min_distances = compute_distances(all_points, selected)
for _ in range(batch_size - 1):
    farthest = argmax(min_distances)
    selected.append(farthest)
    update min_distances with farthest
```

#### `BADGESampler`
**Methods**:
- `compute_gradient_embeddings(features, probabilities)`: Create gradient embeddings
- `select_batch(features, probabilities, batch_size, indices)`: BADGE selection

**Implementation**:
```python
pseudo_labels = argmax(probabilities)
one_hot = to_one_hot(pseudo_labels)
delta = probabilities - one_hot
gradient_emb = delta[:,:,None] * features[:,None,:]
gradient_emb = reshape(gradient_emb)
apply k-center on gradient_emb
```

### Selection Engine Component

#### `select_samples(df, strategy, batch_size, strategy_config, id_field)`
**Purpose**: Main selection coordinator that routes to appropriate strategy

**Algorithm**:
1. Extract probability columns from DataFrame
2. Create sample indices array
3. Route to strategy-specific sampler based on `strategy` parameter
4. Execute selection algorithm
5. Create output DataFrame with selection metadata
6. Return selected samples with scores and ranks

**Returns**: `pd.DataFrame` - Selected samples with metadata

### Output Management Component

#### `save_selected_samples(selected_df, output_dir, output_format)`
**Purpose**: Save selected samples with format preservation

**Algorithm**:
1. Create output directory
2. Use format-specific writer (CSV/TSV/Parquet)
3. Write DataFrame with proper extension
4. Log saved file path

**Returns**: `str` - Path to saved file

#### `save_selection_metadata(metadata, metadata_dir)`
**Purpose**: Save selection configuration and statistics

**Algorithm**:
1. Create metadata directory
2. Serialize metadata dictionary to JSON
3. Write to selection_metadata.json
4. Log metadata path

**Returns**: `str` - Path to metadata file

### Validation Component

#### `validate_strategy_for_use_case(strategy, use_case)`
**Purpose**: Enforce appropriate strategy-use case pairing

**Validation Logic**:
```python
SSL_STRATEGIES = {"confidence_threshold", "top_k_per_class"}
AL_STRATEGIES = {"uncertainty", "diversity", "badge"}

if use_case == "ssl" and strategy not in SSL_STRATEGIES:
    raise ValueError("Uncertainty strategies create noisy pseudo-labels")
    
if use_case == "active_learning" and strategy not in AL_STRATEGIES:
    raise ValueError("Confidence strategies waste human labeling effort")
```

## File I/O and Format Preservation

### Format Detection
```python
def _detect_file_format(file_path):
    suffix = file_path.suffix.lower()
    if suffix == ".csv": return "csv"
    elif suffix == ".tsv": return "tsv"
    elif suffix == ".parquet": return "parquet"
```

### Format-Preserving Save
The script implements intelligent format preservation:
1. Detects input file format during loading
2. Uses `OUTPUT_FORMAT` as override if specified
3. Falls back to input format if `OUTPUT_FORMAT` is default ("csv")
4. Applies appropriate writer for final format

## Use Case Validation

### SSL Validation Rules
**Allowed Strategies**: `confidence_threshold`, `top_k_per_class`
**Blocked Strategies**: `uncertainty`, `diversity`, `badge`
**Rationale**: Uncertainty-based selection creates noisy pseudo-labels that degrade model performance in SSL workflows

### Active Learning Validation Rules
**Allowed Strategies**: `uncertainty`, `diversity`, `badge`
**Blocked Strategies**: `confidence_threshold`, `top_k_per_class`
**Rationale**: Confidence-based selection wastes human labeling budget on easy samples that provide minimal learning value

### Auto Mode
**Validation**: None
**Use Case**: Advanced users who understand strategy implications

## Performance Characteristics

| Strategy | Time Complexity | Space Complexity | Best For |
|----------|----------------|------------------|----------|
| Confidence Threshold | O(n) | O(1) | Large datasets, SSL |
| Top-K Per Class | O(n log n) | O(n) | Balanced SSL, medium datasets |
| Uncertainty | O(n) | O(1) | Large datasets, AL |
| Diversity | O(n² * d) | O(n * d) | Small batches (<10K), AL |
| BADGE | O(n² * d * c) | O(n * d * c) | Small batches, hybrid AL |

Where:
- n = number of samples
- d = embedding/feature dimension
- c = number of classes

## Downstream Integration

### SSL Workflow
```
ModelInference → ActiveSampleSelection → PseudoLabelMerge → Training
```
1. Model generates predictions on unlabeled data
2. ActiveSampleSelection picks high-confidence samples
3. PseudoLabelMerge combines with labeled data
4. Training fine-tunes model on augmented dataset

### Active Learning Workflow
```
ModelInference → ActiveSampleSelection → HumanLabeling → Training
```
1. Model generates predictions on unlabeled pool
2. ActiveSampleSelection picks informative samples
3. Human annotators label selected samples
4. Training updates model with new labels
5. Iterate until performance goal or budget exhausted

## Error Handling

### Input Validation Errors
- **FileNotFoundError**: No data files found in input directory
- **ValueError**: ID field not found in data
- **ValueError**: No valid score columns detected

### Strategy Validation Errors
- **ValueError**: Strategy not valid for specified use case
- **ValueError**: Unknown strategy name

### Configuration Errors
- **ValueError**: No embeddings/features for diversity/BADGE
- **ValueError**: Unsupported file format

## Best Practices

### For SSL Workflows
1. Start with `confidence_threshold=0.9` for conservative pseudo-labeling
2. Use `top_k_per_class` if class imbalance present
3. Set `MAX_SAMPLES` to control pseudo-label volume
4. Monitor pseudo-label accuracy on validation set

### For Active Learning Workflows
1. Start with `uncertainty` (margin) for fast iterations
2. Use `diversity` when embeddings available and batch < 10K
3. Use `badge` for comprehensive sampling with computational budget
4. Typical `BATCH_SIZE`: 100-1000 for human labeling

### General Recommendations
1. Use `OUTPUT_FORMAT` to match downstream step requirements
2. Set `RANDOM_SEED` for reproducible experiments
3. Enable `USE_CASE` validation to prevent strategy mistakes
4. Monitor `selection_metadata.json` for selection statistics

## Example Configurations

### SSL Example: High-Confidence Filtering
```python
environ_vars = {
    "SELECTION_STRATEGY": "confidence_threshold",
    "USE_CASE": "ssl",
    "CONFIDENCE_THRESHOLD": "0.95",
    "MAX_SAMPLES": "5000",
    "ID_FIELD": "sample_id",
    "LABEL_FIELD": "",
}
```

### SSL Example: Balanced Class Selection
```python
environ_vars = {
    "SELECTION_STRATEGY": "top_k_per_class",
    "USE_CASE": "ssl",
    "K_PER_CLASS": "200",
    "ID_FIELD": "id",
}
```

### Active Learning Example: Uncertainty
```python
environ_vars = {
    "SELECTION_STRATEGY": "uncertainty",
    "USE_CASE": "active_learning",
    "UNCERTAINTY_MODE": "entropy",
    "BATCH_SIZE": "500",
    "ID_FIELD": "id",
}
```

### Active Learning Example: BADGE
```python
environ_vars = {
    "SELECTION_STRATEGY": "badge",
    "USE_CASE": "active_learning",
    "BATCH_SIZE": "1000",
    "METRIC": "cosine",
    "ID_FIELD": "sample_id",
}
```

## References

### Related Scripts
- [`pseudo_label_merge.py`](pseudo_label_merge_script.md): Combines selected samples with labeled data (TBD)
- Upstream prediction sources: XGBoost, PyTorch, and Bedrock processing scripts

### Related Documentation
- **Contract**: [`src/cursus/steps/contracts/active_sample_selection_contract.py`](../../src/cursus/steps/contracts/active_sample_selection_contract.py)

### Related Design Documents
- **[Active Sampling Script Design](../1_design/active_sampling_script_design.md)**: Overall architecture and design patterns for active sample selection
- **[Active Sampling Step Patterns](../1_design/active_sampling_step_patterns.md)**: Step builder patterns and integration strategies
- **[BADGE Algorithm Design](../1_design/active_sampling_badge.md)**: Detailed design for BADGE (Batch Active learning by Diverse Gradient Embeddings) implementation
- **[Core-Set and Leaf Core-Set Algorithms](../1_design/active_sampling_core_set_leaf_core_set.md)**: K-center diversity sampling algorithms and optimizations
- **[Uncertainty Sampling: Margin and Entropy](../1_design/active_sampling_uncertainty_margin_entropy.md)**: Uncertainty calculation methods and comparison

### Academic References
- [Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds](https://arxiv.org/abs/1906.03671): BADGE Algorithm (Ash et al., 2020)
- [Active Learning for Convolutional Neural Networks: A Core-Set Approach](https://arxiv.org/abs/1708.00489): K-Center Algorithm (Sener & Savarese, 2018)
- [Active Learning Literature Survey](https://minds.wisconsin.edu/handle/1793/60660): Comprehensive survey of active learning methods (Settles, 2009)
