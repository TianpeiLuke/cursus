# XGBoost Inference Latency Optimization Analysis

**Date:** 2025-11-20  
**Project:** projects/atoz_xgboost  
**Focus:** Real-time single-record inference latency reduction

## Executive Summary

Analyzed the XGBoost inference handler (`xgboost_inference.py`) for a Prebuilt SageMaker endpoint performing real-time single-record inference. Identified multiple latency bottlenecks primarily related to using pandas DataFrame operations for single-value processing.

**Key Finding:** The inference pipeline uses pandas DataFrame/Series operations throughout, which introduces significant overhead for single-record inference. For real-time endpoints processing one record at a time, this architecture is suboptimal.

---

## Architecture Overview

### Current Flow (Single Record)
```
input_fn() → predict_fn() → output_fn()
     ↓              ↓              ↓
 Parse CSV    Preprocess     Format JSON/CSV
  → DataFrame   → Series ops   → Response
```

### Key Components
1. **model_fn()**: One-time model loading (not a latency concern)
2. **input_fn()**: Parses input → pandas DataFrame
3. **predict_fn()**: Preprocessing + prediction
4. **output_fn()**: Format response

---

## Latency Bottlenecks Identified

### 1. **CRITICAL: DataFrame/Series Operations for Single Values**
**Location:** `predict_fn()` throughout  
**Impact:** HIGH

**Issues:**
```python
# Current approach for SINGLE record:
df = assign_column_names(input_data, feature_columns)  # DataFrame copy
df = apply_preprocessing(df, ...)                       # Multiple DataFrame copies
df = convert_to_numeric(df, feature_columns)           # More conversions
```

**Why This is Slow:**
- Creating DataFrame for 1 row has significant overhead
- Each processor operation creates DataFrame copies
- Pandas optimized for batch operations, not single values
- Index creation, alignment checking, type inference overhead

**Evidence from code:**
```python
# In apply_preprocessing() - line ~700
for feature, processor in risk_processors.items():
    if feature in df.columns:
        df[feature] = processor.transform(df[feature])  # Series operation

for feature, processor in numerical_processors.items():
    if feature in df.columns:
        df[feature] = processor.transform(df[feature])  # Series operation
```

### 2. **Column-by-Column Processing with Copies**
**Location:** `apply_preprocessing()` function  
**Impact:** MEDIUM-HIGH

**Issues:**
```python
def apply_preprocessing(df, feature_columns, risk_processors, numerical_processors):
    # Multiple iterations over columns
    for feature, processor in risk_processors.items():
        if feature in df.columns:
            df[feature] = processor.transform(df[feature])  # Copy overhead
    
    for feature, processor in numerical_processors.items():
        if feature in df.columns:
            df[feature] = processor.transform(df[feature])  # Copy overhead
```

**Problems:**
- Iterates through all features one by one
- Each transform may create DataFrame/Series copies
- Column existence checks for every feature
- For 50+ features, this is 100+ DataFrame operations

### 3. **Risk Table Mapping via Pandas .map()**
**Location:** `RiskTableMappingProcessor.transform()`  
**Impact:** MEDIUM

**Current implementation:**
```python
# In risk_table_processor.py
def transform(self, data: Union[pd.DataFrame, pd.Series, Any]):
    if isinstance(data, pd.Series):
        return (
            data.astype(str)                           # Type conversion
            .map(self.risk_tables["bins"])            # Dictionary lookup
            .fillna(self.risk_tables["default_bin"])  # Fill missing
        )
```

**Issues:**
- `.astype(str)` creates new Series
- `.map()` has overhead for single value
- `.fillna()` creates another copy

**Better approach for single value:**
```python
# Direct dictionary lookup (10-100x faster)
str_value = str(input_value)
return self.risk_tables["bins"].get(str_value, self.risk_tables["default_bin"])
```

### 4. **Numerical Imputation via Pandas .fillna()**
**Location:** `NumericalVariableImputationProcessor.transform()`  
**Impact:** MEDIUM

**Current implementation:**
```python
def transform(self, X: Union[pd.Series, pd.DataFrame, Any]):
    if isinstance(X, pd.Series):
        return X.fillna(self.imputation_value)  # Series operation
```

**Issues:**
- `.fillna()` overhead for single value
- Could be simple: `value if not pd.isna(value) else imputation_value`

### 5. **Type Conversions and Validations**
**Location:** Multiple places in `predict_fn()`  
**Impact:** MEDIUM

**Issues:**
```python
# In convert_to_numeric()
for col in feature_columns:
    df[col] = safe_numeric_conversion(df[col])  # Per-column conversion
    
# safe_numeric_conversion() does:
if pd.api.types.is_numeric_dtype(series):  # Type checking
    return series
numeric_series = pd.to_numeric(series, errors='coerce')
numeric_series = numeric_series.fillna(default_value)
```

**Problems:**
- Type checking for every column
- Series operations for conversions
- Multiple passes over data

### 6. **Excessive DataFrame Copying**
**Location:** Throughout `predict_fn()`  
**Impact:** MEDIUM

**Copy operations found:**
```python
df = input_data.copy()                  # Line ~650
df = input_data.copy()                  # assign_column_names
output_data = data.copy()               # In processor transforms
df[feature_columns] = df[...].astype(float)  # Type conversion copy
```

### 7. **Debug Logging in Hot Path**
**Location:** Throughout preprocessing  
**Impact:** LOW-MEDIUM

**Issues:**
```python
logger.debug(f"Converting {col} to numeric...")
logger.debug(f"After conversion {col}: unique values={df[col].unique()}")
```

- Even if not printed, string formatting happens
- Should use conditional logging: `if logger.isEnabledFor(logging.DEBUG):`

### 8. **Input Parsing Overhead**
**Location:** `input_fn()`  
**Impact:** LOW (acceptable for real-time)

**Current approach:**
```python
if request_content_type == CONTENT_TYPE_CSV:
    df = pd.read_csv(StringIO(decoded), header=None, index_col=None)
```

**Minor optimization opportunity:**
- For single record CSV, could parse directly without pandas
- But this is likely acceptable overhead

---

## Performance Impact Estimation

For a typical inference with 50 features:

| Operation | Current Approach | Estimated Overhead |
|-----------|-----------------|-------------------|
| DataFrame creation | `pd.DataFrame([row])` | ~100-500 μs |
| Column assignment | Series operations | ~10-50 μs per feature |
| Risk table mapping | `.map()` + `.fillna()` | ~20-100 μs per feature |
| Numerical imputation | `.fillna()` | ~10-50 μs per feature |
| Type conversions | `.astype()` per column | ~10-50 μs per feature |
| DataFrame copies | `.copy()` 3-5 times | ~100-300 μs each |

**Total Estimated Overhead:** ~5-15 ms for 50 features

**Expected after optimization:** ~0.1-1 ms (10-100x improvement)

---

## Recommended Optimizations

### Priority 1: Fast Path for Single Record Inference

**Create optimized single-value preprocessing:**

```python
def preprocess_single_record_fast(
    values: List[float],
    feature_columns: List[str],
    risk_processors: Dict,
    numerical_processors: Dict
) -> np.ndarray:
    """
    Fast path for single record - no DataFrame operations.
    
    Args:
        values: List of feature values (same order as feature_columns)
        feature_columns: Feature names
        risk_processors: Risk table processors
        numerical_processors: Imputation processors
    
    Returns:
        numpy array ready for XGBoost
    """
    processed = []
    
    for i, (col, val) in enumerate(zip(feature_columns, values)):
        # Apply risk table if categorical
        if col in risk_processors:
            val = risk_processors[col].risk_tables["bins"].get(
                str(val), 
                risk_processors[col].risk_tables["default_bin"]
            )
        
        # Apply imputation if numerical
        if col in numerical_processors:
            if pd.isna(val) or val is None:
                val = numerical_processors[col].imputation_value
        
        # Convert to float
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = 0.0
        
        processed.append(val)
    
    return np.array(processed, dtype=np.float32)
```

**Integration in predict_fn:**

```python
def predict_fn(input_data: pd.DataFrame, model_artifacts: Dict[str, Any]):
    # Detect single record
    if len(input_data) == 1:
        # Fast path
        values = input_data.iloc[0].tolist()
        processed_array = preprocess_single_record_fast(
            values, feature_columns, risk_processors, numerical_processors
        )
        processed_array = processed_array.reshape(1, -1)
    else:
        # Original DataFrame path for batch
        # ... existing code ...
```

**Expected Impact:** 10-50x speedup for preprocessing

### Priority 2: Cache Processor Lookups

**Problem:** Dictionary lookups in nested loops

**Solution:**
```python
# In model_fn(), create fast lookup structures
def model_fn(model_dir: str):
    # ... existing code ...
    
    # Pre-build lookup arrays for fast access
    feature_to_risk_table = {}
    feature_to_impute_value = {}
    
    for feature, processor in risk_processors.items():
        feature_to_risk_table[feature] = processor.risk_tables
    
    for feature, processor in numerical_processors.items():
        feature_to_impute_value[feature] = processor.imputation_value
    
    return {
        "model": model,
        "feature_to_risk_table": feature_to_risk_table,
        "feature_to_impute_value": feature_to_impute_value,
        # ... other artifacts ...
    }
```

### Priority 3: Vectorized Preprocessing for Batch

For batch requests, vectorize operations:

```python
def preprocess_batch_vectorized(df, risk_tables, impute_values):
    """Vectorized preprocessing for batch inference."""
    # Convert all categorical columns at once
    for col, risk_table in risk_tables.items():
        df[col] = df[col].astype(str).map(risk_table["bins"]).fillna(risk_table["default_bin"])
    
    # Fill all numerical columns at once
    for col, impute_val in impute_values.items():
        df[col].fillna(impute_val, inplace=True)
    
    return df
```

### Priority 4: Reduce Logging Overhead

```python
# Replace all debug logs in hot path
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Converting {col} to numeric...")
```

### Priority 5: Optimize Calibration Lookup

The code already has lookup table calibration (good!), but ensure it's used:

```python
# In apply_regular_binary_calibration()
# Already optimized with lookup table - verify it's being used
if isinstance(calibrator, list):
    # Uses fast interpolation - GOOD
```

### Priority 6: Pre-allocate Arrays

```python
# Instead of appending to list
processed = []
for val in values:
    processed.append(process(val))

# Pre-allocate numpy array
processed = np.zeros(len(values), dtype=np.float32)
for i, val in enumerate(values):
    processed[i] = process(val)
```

---

## Implementation Strategy

### Phase 1: Quick Wins (1-2 days)
1. Implement fast path for single record
2. Remove debug logging from hot path
3. Add conditional batch vs single-record routing

**Expected Improvement:** 5-10x latency reduction

### Phase 2: Deeper Optimizations (3-5 days)
1. Optimize processor lookup structures
2. Vectorize batch preprocessing
3. Pre-allocate arrays
4. Profile and tune

**Expected Improvement:** Additional 2-3x on top of Phase 1

### Phase 3: Advanced (Optional)
1. Consider Cython/numba for hot loops
2. Implement request batching if applicable
3. Add caching for repeated inputs

---

## Testing Strategy

### Performance Benchmarks

```python
import time
import numpy as np

def benchmark_inference(handler, test_data, n_iterations=1000):
    """Benchmark inference latency."""
    latencies = []
    
    for _ in range(n_iterations):
        start = time.perf_counter()
        handler.predict(test_data)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'min': np.min(latencies),
        'max': np.max(latencies),
    }
```

### Correctness Testing

```python
def test_optimization_correctness():
    """Ensure optimized path produces same results."""
    # Generate test cases
    test_records = generate_test_data(100)
    
    for record in test_records:
        # Original path
        result_original = predict_fn_original(record)
        
        # Optimized path
        result_optimized = predict_fn_optimized(record)
        
        # Compare
        np.testing.assert_array_almost_equal(
            result_original, 
            result_optimized,
            decimal=6
        )
```

---

## Monitoring Recommendations

### Key Metrics to Track

1. **Latency Percentiles**
   - P50, P95, P99 inference time
   - Track before/after optimization

2. **Throughput**
   - Requests per second
   - Should increase proportionally

3. **Error Rates**
   - Ensure optimizations don't break correctness

4. **Resource Utilization**
   - CPU usage should stay same or decrease
   - Memory usage should decrease

### CloudWatch Metrics

```python
import boto3
cloudwatch = boto3.client('cloudwatch')

def publish_metrics(inference_time_ms):
    cloudwatch.put_metric_data(
        Namespace='SageMaker/ModelLatency',
        MetricData=[
            {
                'MetricName': 'InferenceLatency',
                'Value': inference_time_ms,
                'Unit': 'Milliseconds'
            }
        ]
    )
```

---

## Conclusion

The current XGBoost inference handler has significant latency overhead due to using pandas DataFrame operations for single-record inference. By implementing a fast path that processes single records with direct dictionary lookups and numpy arrays, we can expect:

- **10-50x reduction in preprocessing latency** 
- **Overall 5-15x reduction in total inference time**
- **Same accuracy and correctness**

The optimizations are straightforward to implement and test, with minimal risk to the existing system.

### Next Steps

1. Implement Priority 1 (fast path) as proof of concept
2. Benchmark against current implementation  
3. Validate correctness with test suite
4. Deploy to staging endpoint
5. Monitor metrics and iterate

---

## Appendix: Code References

### Key Files
- `projects/atoz_xgboost/docker/xgboost_inference.py` - Main inference handler
- `projects/atoz_xgboost/docker/processing/categorical/risk_table_processor.py` - Risk table mapping
- `projects/atoz_xgboost/docker/processing/numerical/numerical_imputation_processor.py` - Numerical imputation

### Hot Path Functions
- `predict_fn()` - Lines ~650-850
- `apply_preprocessing()` - Lines ~700-730
- `convert_to_numeric()` - Lines ~750-780
- `RiskTableMappingProcessor.transform()` - External file
- `NumericalVariableImputationProcessor.transform()` - External file

---

## Related Documents

### Processor Optimization Implementation

📄 **[Processor Optimization Summary](./processor_optimization_summary.md)**

This document describes the **actual implementation** of the optimizations recommended in this analysis:

- **Bottlenecks Addressed:** Directly implements solutions for Priority 1 & Priority 3 bottlenecks identified above
- **RiskTableMappingProcessor:** Fast path implementation for single-value processing (10-100x speedup)
- **NumericalVariableImputationProcessor:** Optimized transform() method for single-record inference
- **Production-Ready Code:** Complete implementation in both `projects/atoz_xgboost/docker/processing/` and `src/cursus/processing/`
- **Backward Compatible:** Zero API changes, automatic performance improvement
- **Testing Guidelines:** Correctness validation and performance benchmarking examples

**Key Relationship:**
- This document (latency analysis) = **Problem identification & recommendations**
- Processor Optimization Summary = **Solution implementation & deployment guide**

Use both documents together:
1. Reference this analysis to understand **why** latency is high and **what** needs optimization
2. Reference the optimization summary to see **how** the optimizations were implemented and **how to test** them
