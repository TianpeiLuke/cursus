---
tags:
  - analysis
  - dependencies
  - simplification
  - package_management
  - code_reduction
keywords:
  - dependency cleanup
  - package simplification
  - jupyter removal
  - optional dependencies
  - requirements optimization
topics:
  - dependency management
  - package optimization
  - code simplification
language: python
date of note: 2025-09-06
---

# Dependency Cleanup Analysis - Post Runtime Testing Simplification

**Date**: September 6, 2025  
**Status**: ✅ **COMPLETED**  
**Impact**: Significant dependency reduction and better package organization

## 🎯 Executive Summary

Following the successful simplification of the pipeline runtime testing system (94% code reduction), this analysis identified and cleaned up unnecessary dependencies that were primarily used in the removed over-engineered modules. The cleanup resulted in better package organization, reduced installation footprint, and clearer dependency management.

## 📊 Dependency Analysis Results

### **Removed from Core Dependencies**

**Jupyter Ecosystem Dependencies (8 packages removed from requirements.txt):**
- `jupyter>=1.0.0` - Only used in deleted runtime/jupyter module
- `ipywidgets>=8.0.0` - Only used in deleted runtime/jupyter module  
- `plotly>=5.0.0` - Only used in deleted runtime/jupyter module
- `nbformat>=5.0.0` - Only used in deleted runtime/jupyter module
- `jinja2>=3.0.0` - Only used in deleted runtime/jupyter module
- `seaborn>=0.12.0` - Only used in workspace templates (moved to optional)
- `jupyterlab>=4.0.0` - Only used in deleted runtime/jupyter module
- `ipython>=8.0.0` - Only used in deleted runtime/jupyter module

**Impact**: These packages were primarily used in the deleted `src/cursus/validation/runtime/jupyter/` module (800 lines) that had 0% user value according to the redundancy analysis.

### **Retained Core Dependencies (Verified Usage)**

**Essential Dependencies Still Required:**
- ✅ **boto3/botocore/sagemaker** - Core AWS integration (used throughout)
- ✅ **pydantic** - Data validation models (used in simplified runtime testing)
- ✅ **PyYAML** - Configuration management (used throughout)
- ✅ **networkx** - DAG processing (used in `src/cursus/api/dag/pipeline_dag_resolver.py`)
- ✅ **click** - CLI framework (used in all CLI modules)
- ✅ **requests** - HTTP operations (used throughout)
- ✅ **pandas/numpy/scikit-learn** - Data processing (used in scripts and validation)
- ✅ **matplotlib** - Visualization (used in validation, scripts, workspace modules)
- ✅ **xgboost** - ML framework (used in specific scripts)

### **Improved Optional Dependencies Organization**

**Enhanced pyproject.toml Structure:**
```toml
[project.optional-dependencies]
# Jupyter notebook integration dependencies
jupyter = [
    "jupyter>=1.0.0",
    "ipywidgets>=8.0.0",
    "plotly>=5.0.0",
    "nbformat>=5.0.0",
    "jinja2>=3.0.0",
    "seaborn>=0.12.0",
    "jupyterlab>=4.0.0",
    "ipython>=8.0.0",
]
# Visualization dependencies (optional)
viz = [
    "seaborn>=0.12.0",
    "plotly>=5.0.0",
]
# ML Framework dependencies
pytorch = [...]
nlp = [...]
```

## 🔍 Usage Analysis by Module

### **Core Modules - Dependencies Verified**

| Module | Key Dependencies | Status | Justification |
|--------|------------------|--------|---------------|
| **cursus.core** | boto3, pydantic, PyYAML | ✅ **KEEP** | Core functionality |
| **cursus.validation** | pydantic, matplotlib | ✅ **KEEP** | Validation and scoring |
| **cursus.pipeline_catalog** | networkx, PyYAML | ✅ **KEEP** | DAG processing |
| **cursus.registry** | boto3, pydantic | ✅ **KEEP** | Registry operations |
| **cursus.workspace** | matplotlib, seaborn | ✅ **KEEP** | Template generation |
| **cursus.cli** | click | ✅ **KEEP** | CLI interface |

### **Removed Modules - Dependencies No Longer Needed**

| Removed Module | Dependencies Used | Lines Removed | Impact |
|----------------|-------------------|---------------|---------|
| **runtime/jupyter/** | jupyter, ipywidgets, plotly, nbformat | 800 lines | 0% user value |
| **runtime/production/** | Various production tools | 600 lines | 0% user value |
| **runtime/s3_integration/** | boto3 (still used elsewhere) | 500 lines | 0% user value |

## 📦 Installation Impact

### **Before Cleanup**
```bash
# Required 25+ packages including heavy Jupyter ecosystem
pip install cursus
# Installs: jupyter, jupyterlab, ipywidgets, plotly, nbformat, etc.
```

### **After Cleanup**
```bash
# Core installation - 17 essential packages only
pip install cursus

# Optional Jupyter support when needed
pip install cursus[jupyter]

# Optional visualization support
pip install cursus[viz]

# Full installation
pip install cursus[all]
```

### **Benefits**
- **Faster Installation**: Reduced core dependency count by ~30%
- **Smaller Footprint**: No unnecessary Jupyter ecosystem for basic usage
- **Better Organization**: Clear separation of core vs optional dependencies
- **User Choice**: Users can install only what they need

## ✅ Validation Results

### **Package Installation Test**
```bash
✅ pip install -e . - SUCCESS
✅ All core dependencies resolved correctly
✅ No missing imports in core functionality
```

### **Test Suite Validation**
```bash
✅ 14/14 tests pass (100% success rate)
✅ All
```