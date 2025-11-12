---
tags:
  - project_planning
  - preprocessing
  - processors
  - steps
  - alignment
  - implementation_plan
keywords:
  - processor step alignment
  - preprocessing artifacts
  - risk table mapping
  - missing value imputation
  - single-column pattern
  - artifact sharing
topics:
  - preprocessing alignment
  - processor refactoring
  - step script compatibility
  - real-time pipeline
language: python
date of note: 2025-11-11
---

# Processor-Step Preprocessing Alignment Implementation Plan

## Overview

This document provides a comprehensive implementation plan for aligning processor-based preprocessing operations (real-time inference pipelines) with step-based preprocessing operations (batch training pipelines) in the cursus framework. The alignment ensures that artifacts generated during training can be seamlessly used in real-time inference, and vice versa, enabling consistent preprocessing logic across both paradigms.

## Related Documentation

### Design Documents
- **[Atomic Processing Architecture Design](../1_design/atomic_processing_architecture_design.md)** - Foundation for single-column processor design and composition patterns
- **[Processor Design and Documentation Index](../00_entry_points/processing_steps_index.md)** - Entry point for processor documentation
- **[Step Design and Documentation Index](../00_entry_points/step_design_and_documentation_index.md)** - Entry point for step documentation

### Current Implementation
- **[Risk Table Mapping Script](../../src/cursus/steps/scripts/risk_table_mapping.py)** - Batch risk table mapping
- **[Missing Value Imputation Script](../../src/cursus/steps/scripts/missing_value_imputation.py)** - Batch imputation
- **[RiskTableMappingProcessor](../../src/cursus/processing/categorical/risk_table_processor.py)** - Real-time risk mapping
- **[NumericalImputationProcessor](../../src/cursus/processing/numerical/numerical_imputation_processor.py)** - Real-time imputation
- **[Base Processor](../../src/cursus/processing/processors.py)** - Base processor class

### Reference Implementation
- **[PyTorch Training Script](../../projects/bsm_pytorch/docker/pytorch_training.py)** - Real-world processor usage pattern

## Motivation

### Current State: Two Preprocessing Paradigms

The cursus framework currently supports two distinct preprocessing approaches:

1. **Step-Based Preprocessing (Batch Training)**
   - Scripts in `cursus/steps/scripts/`
   - Runs during SageMaker pipeline execution
   - Processes entire datasets (train/val/test splits)
   - Saves artifacts (e.g., `risk_table_map.pkl`, `impute_dict.pkl`)
   - Used in training pipelines

2. **Processor-Based Preprocessing (Real-Time Inference)**
   - Classes in `cursus/processing/`
   - Runs during real-time prediction
   - Processes single records
   - Loads artifacts from training
   - Used in inference pipelines

### The Alignment Problem

**Current Misalignments:**

1. **Architectural Inconsistency**
   - `RiskTableMappingProcessor` extends base `Processor` class ✓
   - `NumericalImputationProcessor` does NOT extend base `Processor` class ✗

2. **Interface Mismatch**
   - Base `Processor.process()` signature: single value → single value
   - `RiskTableMappingProcessor.process()`: single value → single value ✓
   - `NumericalImputationProcessor.process()`: dict → dict ✗

3. **Multi-Column vs Single-Column Design**
   - `RiskTableMappingProcessor`: Designed for single column ✓
   - `NumericalImputationProcessor`: Designed for multiple columns ✗
   - Real-world usage pattern (PyTorch training): One processor per column ✓

4. **Artifact Persistence Missing**
   - `RiskTableMappingProcessor`: Has `save_risk_tables()` / `load_risk_tables()` ✓
   - `NumericalImputationProcessor`: No save/load methods ✗

5. **Method Naming Inconsistency**
   - `RiskTableMappingProcessor`: `get_risk_tables()`, `set_risk_tables()`
   - `NumericalImputationProcessor`: `get_params()` (different pattern)

### Value Proposition of Alignment

**Unified Preprocessing Architecture:**
- ✅ **Artifact Sharing**: Training artifacts directly usable in inference
- ✅ **Consistent Interface**: Same patterns for all preprocessing processors
- ✅ **Real-Time Pipelines**: Chain processors with `>>` operator
- ✅ **Bidirectional Workflow**: Processors → Scripts and Scripts → Processors
- ✅ **Type Safety**: Proper inheritance from base `Processor` class
- ✅ **Maintainability**: Single design pattern for all preprocessing

## Implementation Roadmap

### Phase 1: Fix NumericalImputationProcessor Architecture (Critical Foundation) ✅ COMPLETED
**Duration**: 2-3 days  
**Risk**: Medium  
**Priority**: Critical  
**Dependencies**: None  
**Status**: ✅ Completed (2025-11-11)

- [x] Make `NumericalImputationProcessor` extend base `Processor` class
- [x] Change from multi-column to single-column design
- [x] Fix `process()` method signature to handle single values
- [x] Remove duplicate `__call__` and `__rshift__` methods
- [x] Update `transform()` method for consistency
- [x] Add column context tracking

**Additional Achievements:**
- [x] Added factory methods `from_imputation_dict()` and `from_script_artifacts()` (Phase 4 bonus)
- [x] Implemented `get_imputation_value()` and `set_imputation_value()` methods (Phase 3 bonus)
- [x] Added deprecation warning for `get_params()` method
- [x] Created backward compatibility layer with `NumericalVariableImputationProcessor`
- [x] Wrote comprehensive test suite (37 tests, all passing)

**Test Results:**
- 37 tests passed covering:
  - Base Processor inheritance ✓
  - Single-column architecture ✓
  - Pipeline chaining with `>>` operator ✓
  - All imputation strategies (mean, median, mode) ✓
  - Script artifact compatibility ✓
  - Factory methods ✓
  - Error handling and validation ✓
  - Real-world workflows ✓

### Phase 2: Add Artifact Persistence Methods (I/O Standardization) ✅ COMPLETED
**Duration**: 2-3 days  
**Risk**: Low  
**Priority**: Critical  
**Dependencies**: Phase 1  
**Status**: ✅ Completed (2025-11-11)

- [x] Add `save_imputation_value()` method
- [x] Add `load_imputation_value()` method
- [x] Ensure format matches script output (`impute_dict.pkl`)
- [x] Add JSON export for human readability
- [x] Implement artifact validation

**Implementation Details:**
- `save_imputation_value(output_dir)` - Saves both pickle and JSON files
- `load_imputation_value(filepath)` - Loads from directory or specific file
- Format: `{column_name}_impute_value.pkl` and `{column_name}_impute_value.json`
- Validation on load ensures numeric values only
- Full save/load roundtrip verified

**Test Results:**
- 46 tests passed (9 new tests for artifact persistence)
- Save/load functionality ✓
- Multiple processor handling ✓
- File validation ✓
- Roundtrip preservation ✓

### Phase 3: Standardize Method Naming (API Consistency) ✅ COMPLETED
**Duration**: 1-2 days  
**Risk**: Low  
**Priority**: High  
**Dependencies**: Phase 1, Phase 2  
**Status**: ✅ Completed (2025-11-11) - Implemented as bonus during Phase 1

- [x] Rename `get_params()` → `get_imputation_value()`
- [x] Add `set_imputation_value()` method
- [x] Consider domain-specific naming convention
- [x] Update all references in codebase

**Implementation Details:**
- `get_imputation_value()` - Returns fitted imputation value with proper error handling
- `set_imputation_value(value)` - Sets value with validation
- `get_params()` - Deprecated with DeprecationWarning, redirects users to new method
- Domain-specific naming: `get_imputation_value` / `set_imputation_value` matches `get_risk_tables` / `set_risk_tables` pattern

**Naming Convention Established:**
- `get_{artifact_type}()` - Retrieve fitted artifact
- `set_{artifact_type}()` - Set artifact (for pre-fitted processors)
- `save_{artifact_type}()` - Save artifact to disk
- `load_{artifact_type}()` - Load artifact from disk

**Test Coverage:**
- Already tested in Phase 1 test suite (44 tests passed after cleanup)
- Deprecation warning test included
- Getter/setter validation tests included

**Code Cleanup:**
- Removed redundant `NumericalVariableImputationProcessor` backward compatibility class
- Clean single-class implementation without deprecated aliases
- Updated all imports in `__init__.py` files

### Phase 4: Add Factory Methods (Usability Enhancement) ✅ COMPLETED
**Duration**: 1-2 days  
**Risk**: Low  
**Priority**: Medium  
**Dependencies**: Phase 1, Phase 2  
**Status**: ✅ Completed (2025-11-11) - Implemented as bonus during Phase 1

- [x] Add `from_imputation_dict()` class method
- [x] Add convenience constructors for common patterns
- [x] Add helper for creating processor collections
- [x] Document factory patterns

**Implementation Details:**
- `from_imputation_dict(imputation_dict)` - Creates multiple processors from dictionary
- `from_script_artifacts(artifacts_dir)` - Loads from script output directory
- Both methods return Dict[str, NumericalImputationProcessor]
- Full validation and error handling included

**Key Features:**
- Automatic processor creation for all columns in dictionary
- Type validation for column names and imputation values
- Seamless integration with script artifacts
- Supports custom filename parameter for flexibility

**Usage Examples:**
```python
# Load from script artifacts (one-liner)
processors = NumericalImputationProcessor.from_script_artifacts("model_artifacts/")

# Or from dictionary
processors = NumericalImputationProcessor.from_imputation_dict(impute_dict)

# Use in pipelines
for col, proc in processors.items():
    dataset.add_pipeline(col, proc)
```

**Test Coverage:**
- Already tested in Phase 1 test suite (46 tests passed)
- Factory method validation tests included
- Script compatibility tests included

### Phase 5: Verify Bidirectional Artifact Flow (Integration)
**Duration**: 2-3 days  
**Risk**: Medium  
**Priority**: Critical  
**Dependencies**: All previous phases

- [ ] Test Script → Processor workflow
- [ ] Test Processor → Script workflow
- [ ] Verify artifact format compatibility
- [ ] Add integration tests
- [ ] Document both workflows

### Phase 6: Documentation and Examples (Knowledge Transfer)
**Duration**: 1-2 days  
**Risk**: Low  
**Priority**: High  
**Dependencies**: All previous phases

- [ ] Update processor documentation
- [ ] Add usage examples
- [ ] Document artifact formats
- [ ] Create migration guide
- [ ] Add best practices guide

**Total Estimated Duration**: 9-15 days (2-3 weeks)

## Risk Table Mapping Alignment Analysis

### Current Implementation Comparison

**Script** (`risk_table_mapping.py`):
```python
# OfflineBinning class - handles MULTIPLE variables
class OfflineBinning:
    def fit(self, df, smooth_factor, count_threshold):
        # Creates risk tables for ALL variables
        for var in self.variables:
            self.risk_tables[var] = {
                "bins": {category: risk_score, ...},
                "default_bin": default_risk,
                "varName": var,
                "type": "categorical", 
                "mode": "categorical"
            }
    
    def transform(self, df):
        # Applies ALL risk tables
        for var, risk_table_info in self.risk_tables.items():
            df[var] = df[var].map(bins).fillna(default_bin)
```

**Processor** (`risk_table_processor.py`):
```python
# RiskTableMappingProcessor - handles SINGLE column
class RiskTableMappingProcessor(Processor):
    def __init__(self, column_name: str, ...):  # Single column
        self.column_name = column_name
        
    def fit(self, data):
        # Creates risk table for THIS column only
        self.risk_tables = {
            "bins": {category: risk_score, ...},
            "default_bin": default_risk
        }
    
    def process(self, input_value):
        # Processes SINGLE value
        return self.risk_tables["bins"].get(str(input_value), default_bin)
```

### Critical Alignment Issues Identified

**ISSUE 1: Different Artifact Formats** ⚠️ CRITICAL

**Script Output Format** (`risk_table_map.pkl`):
```python
{
    "payment_method": {
        "bins": {"CC": 0.15, "DC": 0.08},
        "default_bin": 0.10,
        "varName": "payment_method",
        "type": "categorical",
        "mode": "categorical"
    },
    "category": {
        "bins": {"A": 0.25, "B": 0.12},
        "default_bin": 0.18,
        "varName": "category",
        "type": "categorical",
        "mode": "categorical"
    }
}
```

**Processor Expected Format** (per column):
```python
{
    "bins": {"CC": 0.15, "DC": 0.08},
    "default_bin": 0.10
}
```

**Problem**: Processor's `__init__` and `load_risk_tables()` expect single-column format, but script produces multi-column format!

**ISSUE 2: File Naming Convention Mismatch**

- **Script**: Saves ALL variables in ONE file: `risk_table_map.pkl`
- **Processor**: Saves ONE variable per file: `risk_table_mapping_processor_{column}_risk_tables.pkl`

**ISSUE 3: Missing Format Conversion**

The processor lacks a factory method to:
1. Load multi-column script format
2. Extract single-column risk table
3. Create processor instance

### Alignment Status Summary

✅ **ALIGNED**:
- Both use same risk calculation logic (smooth_risk formula)
- Both use same `bins` and `default_bin` structure
- Both handle missing values with default_bin

✗ **MISALIGNED**:
- Artifact format (multi-column dict vs single-column dict)
- File naming convention (one file vs many files)
- No factory method for script→processor conversion

### Required Fixes

**Fix 1: Add Factory Method to RiskTableMappingProcessor**

```python
@classmethod
def from_script_artifacts(
    cls,
    artifacts_dir: Union[Path, str],
    column_name: str,
    label_name: str = "target"
) -> "RiskTableMappingProcessor":
    """
    Create processor from script output (risk_table_map.pkl).
    
    Args:
        artifacts_dir: Directory containing risk_table_map.pkl
        column_name: Name of column to extract risk table for
        label_name: Name of target variable
        
    Returns:
        Fitted processor for specified column
    """
    artifacts_path = Path(artifacts_dir)
    risk_table_file = artifacts_path / "risk_table_map.pkl"
    
    if not risk_table_file.exists():
        raise FileNotFoundError(f"risk_table_map.pkl not found in {artifacts_path}")
    
    # Load multi-column format
    with open(risk_table_file, "rb") as f:
        all_risk_tables = pkl.load(f)
    
    # Extract single column
    if column_name not in all_risk_tables:
        raise ValueError(f"Column '{column_name}' not found in risk tables")
    
    column_risk_table = all_risk_tables[column_name]
    
    # Extract core structure (ignore extra metadata)
    risk_tables = {
        "bins": column_risk_table["bins"],
        "default_bin": column_risk_table["default_bin"]
    }
    
    return cls(
        column_name=column_name,
        label_name=label_name,
        risk_tables=risk_tables
    )

@classmethod  
def from_script_artifacts_multi(
    cls,
    artifacts_dir: Union[Path, str],
    label_name: str = "target"
) -> Dict[str, "RiskTableMappingProcessor"]:
    """
    Create processors for ALL columns from script output.
    
    Args:
        artifacts_dir: Directory containing risk_table_map.pkl
        label_name: Name of target variable
        
    Returns:
        Dictionary mapping column names to processors
    """
    artifacts_path = Path(artifacts_dir)
    risk_table_file = artifacts_path / "risk_table_map.pkl"
    
    if not risk_table_file.exists():
        raise FileNotFoundError(f"risk_table_map.pkl not found in {artifacts_path}")
    
    # Load multi-column format
    with open(risk_table_file, "rb") as f:
        all_risk_tables = pkl.load(f)
    
    # Create processor for each column
    processors = {}
    for column_name, column_risk_table in all_risk_tables.items():
        risk_tables = {
            "bins": column_risk_table["bins"],
            "default_bin": column_risk_table["default_bin"]
        }
        
        proc = cls(
            column_name=column_name,
            label_name=label_name,
            risk_tables=risk_tables
        )
        processors[column_name] = proc
    
    return processors
```

**Fix 2: Document Format Conversion Pattern**

The key insight is that the script format is a **container** for multiple single-column formats:
```python
# Script format = Dict[column_name, single_column_format]
script_format = {
    "col1": {"bins": {...}, "default_bin": ..., ...},  # Extra metadata OK
    "col2": {"bins": {...}, "default_bin": ..., ...}
}

# Processor format = Just the inner dict (ignoring extra keys)
processor_format = {"bins": {...}, "default_bin": ...}
```

**Fix 3: Update Artifact Section in Phase 2**

Add section showing both formats are compatible once we add the factory method.

## Detailed Implementation Specifications

### Phase 1: Fix NumericalImputationProcessor Architecture

#### Problem Statement

**Current Implementation** (`src/cursus/processing/numerical/numerical_imputation_processor.py`):

```python
class NumericalVariableImputationProcessor:  # ✗ No base class
    def __init__(
        self,
        variables: Optional[List[str]] = None,  # ✗ Multi-column
        imputation_dict: Optional[Dict[str, Union[int, float]]] = None,  # ✗ Multi-column
        strategy: str = "mean",
    ):
        # ...

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:  # ✗ Wrong signature
        """Process dictionary, not single value"""
        # ...
```

**Issues:**
1. Does NOT extend base `Processor` class
2. Multi-column design (handles multiple columns in one instance)
3. `process()` method signature incompatible with base class
4. Cannot be used in pipeline chaining (`>>` operator)

**Correct Pattern** (from `RiskTableMappingProcessor`):

```python
class RiskTableMappingProcessor(Processor):  # ✓ Extends base
    def __init__(
        self,
        column_name: str,  # ✓ Single column
        label_name: str,
        smooth_factor: float = 0.0,
        count_threshold: int = 0,
        risk_tables: Optional[Dict] = None,
    ):
        super().__init__()  # ✓ Call parent
        self.column_name = column_name  # ✓ Single column context
        # ...

    def process(self, input_value: Any) -> float:  # ✓ Single value
        """Process a SINGLE value for this column."""
        if not self.is_fitted:
            raise RuntimeError("Must be fitted first")
        
        str_value = str(input_value)
        return self.risk_tables["bins"].get(str_value, self.risk_tables["default_bin"])
```

#### Solution: Single-Column Architecture

**1.1 Update Class Definition**

```python
from ..processors import Processor

class NumericalImputationProcessor(Processor):  # ← Add inheritance
    """
    A processor that performs imputation on a SINGLE numerical column.
    
    Designed for real-time inference pipelines where each processor
    handles one column and processors can be chained with >> operator.
    
    For batch processing of multiple columns, use one processor per column.
    """

    def __init__(
        self,
        column_name: str,  # ← SINGLE column
        imputation_value: Optional[Union[int, float]] = None,
        strategy: Optional[str] = None,
    ):
        """
        Initialize numerical imputation processor.
        
        Args:
            column_name: Name of the column to impute (single column)
            imputation_value: Pre-computed imputation value (for inference)
            strategy: Strategy for fitting ('mean', 'median', 'mode')
        """
        super().__init__()  # ← Call parent constructor
        self.processor_name = "numerical_imputation_processor"
        self.function_name_list = ["fit", "process", "transform"]
        
        self.column_name = column_name
        self.strategy = strategy
        self.is_fitted = imputation_value is not None
        
        if imputation_value is not None:
            self._validate_imputation_value(imputation_value)
            self.imputation_value = imputation_value
        else:
            self.imputation_value = None
```

**1.2 Fix process() Method**

```python
    def process(self, input_value: Union[int, float, Any]) -> Union[int, float]:
        """
        Process a SINGLE numerical value for this column.
        
        This method is called by __call__ (inherited from base Processor).
        It handles single-value processing for real-time inference.
        
        Args:
            input_value: Single value to impute if missing
            
        Returns:
            Imputed value (or original if not missing)
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Processor for column '{self.column_name}' must be fitted before processing"
            )
        
        # Handle missing values
        if pd.isna(input_value):
            return self.imputation_value
        
        return input_value
```

**1.3 Update fit() Method**

```python
    def fit(
        self, 
        X: Union[pd.Series, pd.DataFrame], 
        y: Optional[pd.Series] = None
    ) -> "NumericalImputationProcessor":
        """
        Fit imputation value on a Series (single column).
        
        Args:
            X: Series (preferred) or DataFrame with column_name
            y: Ignored (for sklearn compatibility)
            
        Returns:
            self (for method chaining)
        """
        # Extract Series if DataFrame provided
        if isinstance(X, pd.DataFrame):
            if self.column_name not in X.columns:
                raise ValueError(f"Column '{self.column_name}' not found in DataFrame")
            data = X[self.column_name]
        else:
            data = X
        
        # Calculate imputation value based on strategy
        if data.isna().all():
            self.imputation_value = 0.0  # or np.nan
            logger.warning(f"Column '{self.column_name}' has all NaN values")
        elif self.strategy == "mean":
            self.imputation_value = float(data.mean())
        elif self.strategy == "median":
            self.imputation_value = float(data.median())
        elif self.strategy == "mode":
            self.imputation_value = float(data.mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.is_fitted = True
        return self
```

**1.4 Update transform() Method**

```python
    def transform(
        self, 
        X: Union[pd.Series, pd.DataFrame, Any]
    ) -> Union[pd.Series, pd.DataFrame, float]:
        """
        Transform data using the fitted imputation value.
        
        Args:
            X: Series, DataFrame, or single value
            
        Returns:
            Imputed data in same format as input
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Processor for column '{self.column_name}' must be fitted before transforming"
            )
        
        # Handle Series
        if isinstance(X, pd.Series):
            return X.fillna(self.imputation_value)
        
        # Handle DataFrame
        elif isinstance(X, pd.DataFrame):
            if self.column_name not in X.columns:
                raise ValueError(f"Column '{self.column_name}' not found in DataFrame")
            df = X.copy()
            df[self.column_name] = df[self.column_name].fillna(self.imputation_value)
            return df
        
        # Handle single value (delegate to process)
        else:
            return self.process(X)
```

**1.5 Remove Duplicate Methods**

```python
    # ✗ REMOVE - inherited from base Processor
    # def __call__(self, input_data):
    #     return self.process(input_data)
    
    # ✗ REMOVE - inherited from base Processor
    # def __rshift__(self, other):
    #     if isinstance(self, ComposedProcessor):
    #         return ComposedProcessor(self.processors + [other])
    #     return ComposedProcessor([self, other])
```

#### Benefits of Single-Column Architecture

1. **✅ Consistent with Base Processor**: Follows base class contract
2. **✅ Enables Pipeline Chaining**: Can use `>>` operator
3. **✅ Matches Real-World Usage**: PyTorch training pattern
4. **✅ Aligns with RiskTableMappingProcessor**: Same design
5. **✅ Simpler Interface**: One processor, one column, one responsibility

### Phase 2: Add Artifact Persistence Methods

#### Problem Statement

**Current State:**
- `RiskTableMappingProcessor` has `save_risk_tables()` / `load_risk_tables()` ✓
- `NumericalImputationProcessor` has NO save/load methods ✗

**Required Workflow:**
```
Training Pipeline (Script):
  risk_table_mapping.py → risk_table_map.pkl
  missing_value_imputation.py → impute_dict.pkl

Inference Pipeline (Processor):
  load_risk_tables() ← risk_table_map.pkl
  load_imputation_dict() ← impute_dict.pkl  # ✗ MISSING
```

#### Solution: Artifact I/O Methods

**2.1 Add save_imputation_value() Method**

```python
    def save_imputation_value(self, output_dir: Union[Path, str]) -> None:
        """
        Save imputation value to disk.
        
        Creates two files:
        1. {column_name}_impute_value.pkl (for loading)
        2. {column_name}_impute_value.json (for human readability)
        
        Args:
            output_dir: Directory to save artifacts to
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Cannot save before fitting processor for column '{self.column_name}'"
            )
        
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save pickle (for loading)
        pkl_file = output_dir_path / f"{self.column_name}_impute_value.pkl"
        with open(pkl_file, "wb") as f:
            pkl.dump(self.imputation_value, f)
        
        # Save JSON (for readability)
        json_file = output_dir_path / f"{self.column_name}_impute_value.json"
        with open(json_file, "w") as f:
            json.dump({
                "column_name": self.column_name,
                "imputation_value": float(self.imputation_value),
                "strategy": self.strategy,
            }, f, indent=2)
        
        logger.info(f"Saved imputation value for '{self.column_name}' to {output_dir_path}")
```

**2.2 Add load_imputation_value() Method**

```python
    def load_imputation_value(self, filepath: Union[Path, str]) -> None:
        """
        Load imputation value from disk.
        
        Args:
            filepath: Path to pickle file or directory containing it
        """
        filepath_path = Path(filepath)
        
        # Handle directory path
        if filepath_path.is_dir():
            pkl_file = filepath_path / f"{self.column_name}_impute_value.pkl"
        else:
            pkl_file = filepath_path
        
        if not pkl_file.exists():
            raise FileNotFoundError(f"Imputation value file not found: {pkl_file}")
        
        # Load value
        with open(pkl_file, "rb") as f:
            loaded_value = pkl.load(f)
        
        # Validate
        self._validate_imputation_value(loaded_value)
        
        self.imputation_value = loaded_value
        self.is_fitted = True
        
        logger.info(f"Loaded imputation value for '{self.column_name}' from {pkl_file}")
```

**2.3 Add Validation Helper**

```python
    def _validate_imputation_value(self, value: Any) -> None:
        """Validate that imputation value is numeric."""
        if not isinstance(value, (int, float, np.number)):
            raise ValueError(
                f"Imputation value must be numeric, got {type(value)} for column '{self.column_name}'"
            )
        if pd.isna(value):
            logger.warning(f"Imputation value is NaN for column '{self.column_name}'")
```

#### Script-Compatible Format

**Key Requirement**: Match script output format for seamless integration.

**Script Output** (`missing_value_imputation.py`):
```python
# Format: impute_dict.pkl = {column_name: imputation_value}
impute_dict = {
    'age': 30.0,
    'income': 50000.0,
    'credit_score': 650.0
}
with open("impute_dict.pkl", "wb") as f:
    pkl.dump(impute_dict, f)
```

**Processor Save** (compatible):
```python
# Save each processor individually
for col, processor in processors.items():
    processor.save_imputation_value("model_artifacts/")

# Creates: age_impute_value.pkl, income_impute_value.pkl, ...
```

**Processor Load** (from script):
```python
# Load script output
with open("model_artifacts/impute_dict.pkl", "rb") as f:
    impute_dict = pkl.load(f)

# Create processors
processors = {
    col: NumericalImputationProcessor(
        column_name=col,
        imputation_value=val
    )
    for col, val in impute_dict.items()
}
```

### Phase 3: Standardize Method Naming

#### Problem Statement

**Inconsistent Naming:**
- `RiskTableMappingProcessor`: `get_risk_tables()`, `set_risk_tables()`
- `NumericalImputationProcessor`: `get_params()` ← different pattern

**Goal**: Consistent, domain-specific naming across all processors.

#### Solution: Domain-Specific Naming Convention

**3.1 Update NumericalImputationProcessor**

```python
    def get_imputation_value(self) -> Union[int, float]:
        """
        Get the fitted imputation value.
        
        Returns:
            Imputation value for this column
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Processor for column '{self.column_name}' has not been fitted"
            )
        return self.imputation_value
    
    def set_imputation_value(self, value: Union[int, float]) -> None:
        """
        Set imputation value (for pre-fitted processor).
        
        Args:
            value: Imputation value to use
        """
        self._validate_imputation_value(value)
        self.imputation_value = value
        self.is_fitted = True
```

**3.2 Deprecate get_params()**

```python
    def get_params(self) -> Dict[str, Any]:
        """
        Get processor parameters (DEPRECATED).
        
        Use get_imputation_value() instead for the fitted value.
        
        Returns:
            Dictionary with all parameters
        """
        import warnings
        warnings.warn(
            "get_params() is deprecated, use get_imputation_value() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return {
            "column_name": self.column_name,
            "imputation_value": self.imputation_value,
            "strategy": self.strategy,
        }
```

#### Naming Convention Summary

**Standardized Pattern:**
- `get_{artifact_type}()` - Retrieve fitted artifact
- `set_{artifact_type}()` - Set artifact (for pre-fitted processors)
- `save_{artifact_type}()` - Save artifact to disk
- `load_{artifact_type}()` - Load artifact from disk

**Examples:**
- `RiskTableMappingProcessor`: `get_risk_tables()`, `set_risk_tables()`, `save_risk_tables()`, `load_risk_tables()`
- `NumericalImputationProcessor`: `get_imputation_value()`, `set_imputation_value()`, `save_imputation_value()`, `load_imputation_value()`

### Phase 4: Add Factory Methods

#### Problem Statement

**Current Usage** (verbose):
```python
# Load script artifacts
with open("impute_dict.pkl", "rb") as f:
    impute_dict = pkl.load(f)

# Create processors manually
processors = {}
for col, val in impute_dict.items():
    proc = NumericalImputationProcessor(
        column_name=col,
        imputation_value=val
    )
    processors[col] = proc
```

**Goal**: Simpler, more ergonomic API.

#### Solution: Factory Class Methods

**4.1 Add from_imputation_dict() Factory**

```python
    @classmethod
    def from_imputation_dict(
        cls, 
        imputation_dict: Dict[str, Union[int, float]]
    ) -> Dict[str, "NumericalImputationProcessor"]:
        """
        Create processors from script output (impute_dict.pkl).
        
        This factory method simplifies creating multiple processors from
        the dictionary format used by missing_value_imputation.py script.
        
        Args:
            imputation_dict: Dictionary mapping column names to imputation values
                            Format: {column_name: imputation_value}
        
        Returns:
            Dictionary mapping column names to fitted processors
            
        Example:
            >>> with open("impute_dict.pkl", "rb") as f:
            ...     impute_dict = pkl.load(f)
            >>> processors = NumericalImputationProcessor.from_imputation_dict(impute_dict)
            >>> # Use processors in pipeline
            >>> for col, proc in processors.items():
            ...     dataset.add_pipeline(col, proc)
        """
        if not isinstance(imputation_dict, dict):
            raise TypeError("imputation_dict must be a dictionary")
        
        processors = {}
        for col, val in imputation_dict.items():
            if not isinstance(col, str):
                raise ValueError(f"Column name must be string, got {type(col)}")
            
            proc = cls(column_name=col, imputation_value=val)
            processors[col] = proc
        
        return processors
```

**4.2 Add from_script_artifacts() Factory**

```python
    @classmethod
    def from_script_artifacts(
        cls,
        artifacts_dir: Union[Path, str]
    ) -> Dict[str, "NumericalImputationProcessor"]:
        """
        Load processors from script output directory.
        
        Looks for impute_dict.pkl in the specified directory and
        creates processors from it.
        
        Args:
            artifacts_dir: Directory containing impute_dict.pkl
            
        Returns:
            Dictionary mapping column names to fitted processors
            
        Example:
            >>> processors = NumericalImputationProcessor.from_script_artifacts(
            ...     "model_artifacts/"
            ... )
            >>> for col, proc in processors.items():
            ...     dataset.add_pipeline(col, proc)
        """
        artifacts_path = Path(artifacts_dir)
        impute_dict_file = artifacts_path / "impute_dict.pkl"
        
        if not impute_dict_file.exists():
            raise FileNotFoundError(f"impute_dict.pkl not found in {artifacts_path}")
        
        with open(impute_dict_file, "rb") as f:
            impute_dict = pkl.load(f)
        
        return cls.from_imputation_dict(impute_dict)
```

**4.3 Usage Examples**

```python
# Example 1: Load from script artifacts
processors = NumericalImputationProcessor.from_script_artifacts("model_artifacts/")

# Example 2: Load from dict
with open("impute_dict.pkl", "rb") as f:
    impute_dict = pkl.load(f)
processors = NumericalImputationProcessor.from_imputation_dict(impute_dict)

# Example 3: Add to dataset pipeline (matches PyTorch training pattern)
for col, processor in processors.items():
    dataset.add_pipeline(col, processor)
```

### Phase 5: Verify Bidirectional Artifact Flow

#### Workflow 1: Script → Processor (Training → Inference)

**Use Case**: Train with scripts, deploy processors for real-time inference.

```python
# ========================================
# TRAINING PHASE (SageMaker Pipeline)
# ========================================

# 1. Run missing_value_imputation.py script
# Outputs: model_artifacts/impute_dict.pkl

# 2. Run risk_table_mapping.py script  
# Outputs: model_artifacts/risk_table_map.pkl

# ========================================
# INFERENCE PHASE (Real-Time Endpoint)
# ========================================

# 3. Load imputation processors
imputation_processors = NumericalImputationProcessor.from_script_artifacts(
    "model_artifacts/"
)

# 4. Load risk table processors
with open("model_artifacts/risk_table_map.pkl", "rb") as f:
    risk_tables = pkl.load(f)

risk_processors = {}
for var_name, risk_table in risk_tables.items():
    proc = RiskTableMappingProcessor(
        column_name=var_name,
        label_name="target",
        risk_tables=risk_table
    )
    risk_processors[var_name] = proc

# 5. Add all processors to dataset
for col, proc in imputation_processors.items():
    dataset.add_pipeline(col, proc)

for col, proc in risk_processors.items():
    dataset.add_pipeline(col, proc)

# 6. Process records in real-time
prediction = model.predict(dataset[0])
```

#### Workflow 2: Processor → Script (Experimentation → Production)

**Use Case**: Develop with processors in notebooks, save for batch processing.

```python
# ========================================
# EXPERIMENTATION PHASE (Jupyter Notebook)
# ========================================

# 1. Fit processors on sample data
imputation_processors = {}
for col in numerical_columns:
    proc = NumericalImputationProcessor(
        column_name=col,
        strategy="mean"
    )
    proc.fit(train_df[col])
    imputation_processors[col] = proc

# 2. Save in script-compatible format
impute_dict = {
    col: proc.get_imputation_value() 
    for col, proc in imputation_processors.items()
}

with open("model_artifacts/impute_dict.pkl", "wb") as f:
    pkl.dump(impute_dict, f)

# ========================================
# PRODUCTION PHASE (SageMaker Pipeline)
# ========================================

# 3. missing_value_imputation.py loads artifacts
# Uses: model_artifacts/impute_dict.pkl (created by processors!)

# 4. Batch processing on full dataset
# Script applies same imputation logic to train/val/test splits
```

#### Integration Tests

**5.1 Test Script → Processor**

```python
def test_script_to_processor_workflow():
    """Test loading script artifacts in processor."""
    # Run script
    result = run_missing_value_imputation_script(
        input_dir="test_data/",
        output_dir="test_output/",
        job_type="training"
    )
    
    # Load artifacts in processor
    processors = NumericalImputationProcessor.from_script_artifacts(
        "test_output/model_artifacts/"
    )
    
    # Verify processors work
    assert len(processors) > 0
    for col, proc in processors.items():
        assert proc.is_fitted
        assert proc.column_name == col
        
        # Test single-value processing
        test_value = None
        imputed = proc.process(test_value)
        assert imputed is not None
```

**5.2 Test Processor → Script**

```python
def test_processor_to_script_workflow():
    """Test using processor artifacts in script."""
    # Fit processors
    df = load_test_data()
    processors = {}
    for col in df.select_dtypes(include=np.number).columns:
        proc = NumericalImputationProcessor(
            column_name=col,
            strategy="mean"
        )
        proc.fit(df[col])
        processors[col] = proc
    
    # Save in script format
    impute_dict = {
        col: proc.get_imputation_value()
        for col, proc in processors.items()
    }
    
    with open("test_artifacts/impute_dict.pkl", "wb") as f:
        pkl.dump(impute_dict, f)
    
    # Run script with processor artifacts
    result = run_missing_value_imputation_script(
        input_dir="test_data/",
        output_dir="test_output/",
        model_artifacts_input_dir="test_artifacts/",
        job_type="validation"
    )
    
    # Verify script used artifacts correctly
    assert result["success"]
```

**5.3 Test Format Compatibility**

```python
def test_artifact_format_compatibility():
    """Test that script and processor artifacts have identical format."""
    # Create test data
    df = pd.DataFrame({
        'age': [25, 30, None, 40],
        'income': [50000, None, 60000, 70000]
    })
    
    # Fit processors
    processors = {}
    for col in ['age', 'income']:
        proc = NumericalImputationProcessor(column_name=col, strategy="mean")
        proc.fit(df[col])
        processors[col] = proc
    
    # Save as dict (script format)
    impute_dict_from_proc = {
        col: proc.get_imputation_value()
        for col, proc in processors.items()
    }
    
    # Run script
    # ... (script creates impute_dict)
    
    # Load script output
    with open("script_output/impute_dict.pkl", "rb") as f:
        impute_dict_from_script = pkl.load(f)
    
    # Verify formats match
    assert set(impute_dict_from_proc.keys()) == set(impute_dict_from_script.keys())
    for col in impute_dict_from_proc.keys():
        assert abs(impute_dict_from_proc[col] - impute_dict_from_script[col]) < 1e-6
```

### Phase 6: Documentation and Examples

#### 6.1 Update Processor Documentation

Create/update: `slipbox/0_developer_guide/processor_usage_guide.md`

```markdown
# Processor Usage Guide

## Single-Column Architecture

All preprocessing processors in cursus follow a single-column architecture:

- **One processor per column**: Each processor handles one column
- **Chainable with >>**: Processors can be chained for complex pipelines
- **Dataset integration**: Add processors via `dataset.add_pipeline(column, processor)`

## Real-Time Inference Pattern

```python
# 1. Load artifacts from training
processors = NumericalImputationProcessor.from_script_artifacts("model_artifacts/")

# 2. Add to dataset
for col, proc in processors.items():
    dataset.add_pipeline(col, proc)

# 3. Process records
for record in dataset:
    prediction = model.predict(record)
```

## Artifact Sharing

Processors and scripts share artifacts seamlessly:

**Training (Scripts)**:
- `risk_table_mapping.py` → `risk_table_map.pkl`
- `missing_value_imputation.py` → `impute_dict.pkl`

**Inference (Processors)**:
- Load artifacts with factory methods
- Process single records in real-time
```

#### 6.2 Add Usage Examples

Create: `examples/preprocessing_alignment_example.py`

```python
"""
Example: Processor-Step Preprocessing Alignment

Demonstrates bidirectional artifact flow between scripts and processors.
"""

import pandas as pd
import pickle as pkl
from pathlib import Path

from cursus.processing.numerical import NumericalImputationProcessor
from cursus.processing.categorical import RiskTableMappingProcessor


def training_phase_example():
    """Example: Training phase using scripts (batch processing)."""
    print("=== TRAINING PHASE ===")
    
    # Scripts run in SageMaker pipeline
    # risk_table_mapping.py → model_artifacts/risk_table_map.pkl
    # missing_value_imputation.py → model_artifacts/impute_dict.pkl
    
    print("Scripts completed. Artifacts saved to model_artifacts/")


def inference_phase_example():
    """Example: Inference phase using processors (real-time)."""
    print("\n=== INFERENCE PHASE ===")
    
    # Load imputation processors from script artifacts
    imputation_procs = NumericalImputationProcessor.from_script_artifacts(
        "model_artifacts/"
    )
    print(f"Loaded {len(imputation_procs)} imputation processors")
    
    # Load risk table processors
    with open("model_artifacts/risk_table_map.pkl", "rb") as f:
        risk_tables = pkl.load(f)
    
    risk_procs = {}
    for var_name, risk_table in risk_tables.items():
        proc = RiskTableMappingProcessor(
            column_name=var_name,
            label_name="target",
            risk_tables=risk_table
        )
        risk_procs[var_name] = proc
    
    print(f"Loaded {len(risk_procs)} risk table processors")
    
    # Process single record
    record = {
        'age': None,  # Will be imputed
        'category': 'A',  # Will be risk-mapped
        'income': 50000
    }
    
    # Apply processors
    if 'age' in imputation_procs:
        record['age'] = imputation_procs['age'].process(record['age'])
    
    if 'category' in risk_procs:
        record['category'] = risk_procs['category'].process(record['category'])
    
    print(f"Processed record: {record}")


def experimentation_phase_example():
    """Example: Experimentation in notebook, save for production."""
    print("\n=== EXPERIMENTATION PHASE ===")
    
    # Create sample data
    df = pd.DataFrame({
        'age': [25, 30, None, 40, 35],
        'income': [50000, None, 60000, 70000, 55000],
        'score': [0.8, 0.6, 0.9, None, 0.7]
    })
    
    # Fit processors
    processors = {}
    for col in ['age', 'income', 'score']:
        proc = NumericalImputationProcessor(column_name=col, strategy="mean")
        proc.fit(df[col])
        processors[col] = proc
        print(f"Fitted {col}: impute_value={proc.get_imputation_value():.2f}")
    
    # Save in script-compatible format
    Path("experiment_artifacts").mkdir(exist_ok=True)
    impute_dict = {
        col: proc.get_imputation_value()
        for col, proc in processors.items()
    }
    
    with open("experiment_artifacts/impute_dict.pkl", "wb") as f:
        pkl.dump(impute_dict, f)
    
    print("Saved artifacts to experiment_artifacts/")
    print("These can now be used by missing_value_imputation.py script!")


if __name__ == "__main__":
    # Run all examples
    training_phase_example()
    inference_phase_example()
    experimentation_phase_example()
```

#### 6.3 Document Artifact Formats

Create: `slipbox/0_developer_guide/preprocessing_artifact_formats.md`

```markdown
# Preprocessing Artifact Formats

## Overview

This document specifies the artifact formats used for preprocessing operations
in the cursus framework, ensuring compatibility between scripts and processors.

## Numerical Imputation

**File**: `impute_dict.pkl`

**Format**:
```python
{
    "column_name": imputation_value,
    ...
}
```

**Example**:
```python
{
    "age": 30.0,
    "income": 50000.0,
    "credit_score": 650.0
}
```

**Created By**:
- Script: `missing_value_imputation.py`
- Processor: `NumericalImputationProcessor` (via factory method)

**Used By**:
- Script: Load for validation/testing/calibration jobs
- Processor: Load via `from_script_artifacts()`

## Risk Table Mapping

**File**: `risk_table_map.pkl`

**Format**:
```python
{
    "variable_name": {
        "bins": {category_value: risk_score, ...},
        "default_bin": default_risk_score,
        "varName": "variable_name",
        "type": "categorical",
        "mode": "categorical"
    },
    ...
}
```

**Example**:
```python
{
    "payment_method": {
        "bins": {"CC": 0.15, "DC": 0.08, "ACH": 0.03},
        "default_bin": 0.10,
        "varName": "payment_method",
        "type": "categorical",
        "mode": "categorical"
    }
}
```

**Created By**:
- Script: `risk_table_mapping.py`
- Processor: `RiskTableMappingProcessor.save_risk_tables()`

**Used By**:
- Script: Load for validation/testing/calibration jobs
- Processor: Load via constructor or `load_risk_tables()`

## Compatibility Guidelines

1. **Pickle Format**: Use Python's `pickle` module for serialization
2. **JSON Sidecar**: Optionally save JSON version for human readability
3. **Key Consistency**: Column names must match exactly
4. **Type Consistency**: Values must be JSON-serializable when possible
5. **Version Compatibility**: Test across Python 3.8-3.11
```

#### 6.4 Create Migration Guide

Create: `slipbox/0_developer_guide/processor_migration_guide.md`

```markdown
# Processor Migration Guide

## Migrating from Multi-Column to Single-Column Pattern

### Old Pattern (Deprecated)

```python
# ✗ Old way - multi-column processor
processor = NumericalVariableImputationProcessor(
    variables=['age', 'income', 'score'],
    imputation_dict={'age': 30, 'income': 50000, 'score': 0.5}
)

# Process dictionary
result = processor.process({'age': None, 'income': 55000, 'score': None})
```

### New Pattern (Recommended)

```python
# ✓ New way - one processor per column
processors = {
    'age': NumericalImputationProcessor(column_name='age', imputation_value=30),
    'income': NumericalImputationProcessor(column_name='income', imputation_value=50000),
    'score': NumericalImputationProcessor(column_name='score', imputation_value=0.5)
}

# Add to dataset
for col, proc in processors.items():
    dataset.add_pipeline(col, proc)

# Process single value
age_imputed = processors['age'].process(None)  # Returns 30
```

### Benefits of New Pattern

1. **Type Safety**: Proper inheritance from base `Processor`
2. **Pipeline Chaining**: Use `>>` operator
3. **Clearer Semantics**: One processor, one responsibility
4. **Better Testing**: Test each column independently
5. **Dataset Integration**: Matches PyTorch training pattern

### Using Factory Methods

```python
# Load from script artifacts (easiest)
processors = NumericalImputationProcessor.from_script_artifacts("model_artifacts/")

# Or load from dict
with open("impute_dict.pkl", "rb") as f:
    impute_dict = pkl.load(f)
processors = NumericalImputationProcessor.from_imputation_dict(impute_dict)
```


## Implementation Timeline

### Week 1: Foundation (Phase 1-2)
**Days 1-5: Core Architecture**
- [ ] Day 1: Update `NumericalImputationProcessor` class definition and inheritance
- [ ] Day 2: Fix `process()`, `fit()`, and `transform()` methods
- [ ] Day 3: Add artifact persistence methods (`save_*`, `load_*`)
- [ ] Day 4: Unit tests for single-column architecture
- [ ] Day 5: Unit tests for artifact I/O

### Week 2: API & Usability (Phase 3-4)
**Days 6-10: Standardization**
- [ ] Day 6: Standardize method naming across processors
- [ ] Day 7: Add factory methods (`from_imputation_dict`, `from_script_artifacts`)
- [ ] Day 8: Update all references in codebase
- [ ] Day 9: Unit tests for factory methods
- [ ] Day 10: Code review and refinements

### Week 3: Integration & Documentation (Phase 5-6)
**Days 11-15: Testing and Docs**
- [ ] Day 11: Integration tests for Script → Processor workflow
- [ ] Day 12: Integration tests for Processor → Script workflow
- [ ] Day 13: Update processor documentation
- [ ] Day 14: Create usage examples and migration guide
- [ ] Day 15: Final review and validation

## Success Criteria

### Functional Requirements
- ✅ `NumericalImputationProcessor` extends base `Processor` class
- ✅ `process()` method handles single values
- ✅ Processors can be chained with `>>` operator
- ✅ Artifact I/O methods present and functional
- ✅ Factory methods simplify common usage patterns
- ✅ Script artifacts usable by processors
- ✅ Processor artifacts usable by scripts

### Technical Requirements
- ✅ All unit tests passing
- ✅ Integration tests for both workflows successful
- ✅ Artifact format compatibility verified
- ✅ Backward compatibility maintained where possible
- ✅ Code coverage >80% for modified code

### Quality Requirements
- ✅ Consistent naming conventions across processors
- ✅ Clear documentation with examples
- ✅ Migration guide for existing code
- ✅ Type hints throughout codebase

## Risk Mitigation

### Technical Risks

**Risk: Breaking Changes in Existing Code**
- **Mitigation**: Deprecation warnings, maintain backward compatibility where possible
- **Monitoring**: Run full test suite before and after changes

**Risk: Artifact Format Incompatibility**
- **Mitigation**: Extensive integration testing, format validation
- **Monitoring**: Test with real pipeline artifacts

**Risk: Performance Degradation**
- **Mitigation**: Benchmark single-column vs multi-column, optimize if needed
- **Monitoring**: Performance tests in CI/CD

### Implementation Risks

**Risk: Scope Creep**
- **Mitigation**: Stick to defined phases, defer enhancements
- **Monitoring**: Track against timeline

**Risk: Incomplete Migration**
- **Mitigation**: Search codebase for old patterns, update systematically
- **Monitoring**: Deprecation warning counts

## Conclusion

This implementation plan provides a comprehensive roadmap for aligning processor-based and step-based preprocessing operations in the cursus framework. The alignment enables seamless artifact sharing between training and inference, consistent interfaces across all preprocessing processors, and bidirectional workflows that support both experimentation and production use cases.

### Key Implementation Highlights

1. **Single-Column Architecture**: All processors handle one column, matching real-world usage
2. **Base Class Inheritance**: Proper type hierarchy enables pipeline chaining
3. **Artifact Persistence**: Save/load methods ensure training-inference compatibility
4. **Factory Methods**: Simplified APIs for common patterns
5. **Bidirectional Flow**: Scripts → Processors and Processors → Scripts both supported

### Implementation Benefits

- **Unified Design**: Consistent patterns across all preprocessing
- **Type Safety**: Proper inheritance hierarchy
- **Real-Time Pipelines**: Chain processors with `>>` operator
- **Artifact Sharing**: Training artifacts directly usable in inference
- **Developer Experience**: Simpler APIs, clearer semantics

### Next Steps

1. **Phase 1**: Begin with `NumericalImputationProcessor` refactoring (2-3 days)
2. **Phase 2**: Add artifact persistence methods (2-3 days)
3. **Phase 3**: Standardize method naming (1-2 days)
4. **Phase 4**: Add factory methods (1-2 days)
5. **Phase 5**: Integration testing (2-3 days)
6. **Phase 6**: Documentation (1-2 days)

**Total Timeline**: 9-15 days (2-3 weeks)

This plan ensures that preprocessing operations will be consistently designed, enabling sophisticated real-time inference pipelines while maintaining full compatibility with batch training workflows.

### References

#### Implementation References
- [Risk Table Mapping Script](../../src/cursus/steps/scripts/risk_table_mapping.py)
- [Missing Value Imputation Script](../../src/cursus/steps/scripts/missing_value_imputation.py)
- [RiskTableMappingProcessor](../../src/cursus/processing/categorical/risk_table_processor.py)
- [NumericalImputationProcessor](../../src/cursus/processing/numerical/numerical_imputation_processor.py)
- [Base Processor](../../src/cursus/processing/processors.py)
- [PyTorch Training Script](../../projects/bsm_pytorch/docker/pytorch_training.py)

#### Entry Points
- [Processor Design and Documentation Index](../00_entry_points/processing_steps_index.md)
- [Step Design and Documentation Index](../00_entry_points/step_design_and_documentation_index.md)

---

*This implementation plan provides comprehensive specification for aligning processor-based and step-based preprocessing operations, enabling seamless artifact sharing and consistent preprocessing logic across training and inference paradigms.*
