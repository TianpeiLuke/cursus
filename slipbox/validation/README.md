---
tags:
  - entry_point
  - test
  - validation
  - documentation
  - framework
keywords:
  - validation framework
  - step builder testing
  - alignment validation
  - standardization testing
  - universal test suite
  - unified alignment tester
topics:
  - validation framework
  - test suite documentation
  - alignment validation
  - standardization testing
language: python
date of note: 2025-08-18
---

# Cursus Validation Framework Documentation

This directory contains comprehensive documentation for the Cursus validation framework, which provides essential validation capabilities for step builders and component alignment through a simplified 3-function API.

## Framework Overview

The Cursus validation framework implements a **Simplified Integration** approach that achieves 67% integration complexity reduction while maintaining comprehensive validation coverage. The framework consists of two main validation systems:

1. **Standardization Tester** (Universal Step Builder Test) - Validates implementation quality and step builder pattern compliance
2. **Alignment Tester** (Unified Alignment Tester) - Validates component alignment across the four-tier architecture

## Core API Functions

The framework provides three primary validation functions:

- `validate_development()` - Development-time validation using Standardization Tester
- `validate_integration()` - Integration-time validation using Alignment Tester  
- `validate_production()` - Production readiness validation combining both testers

## Directory Structure

```
slipbox/validation/
├── README.md                    # This overview document
├── simple_integration.md        # Core 3-function API documentation
├── alignment/                   # Alignment validation system docs
│   ├── README.md               # Alignment system overview
│   ├── unified_alignment_tester.md
│   ├── script_contract_alignment.md
│   ├── contract_spec_alignment.md
│   ├── spec_dependency_alignment.md
│   ├── builder_config_alignment.md
│   └── step_type_enhancement.md
├── builders/                    # Step builder validation docs
│   ├── README.md               # Builder validation overview
│   ├── universal_test.md       # Universal test suite
│   ├── interface_tests.md
│   ├── specification_tests.md
│   ├── step_creation_tests.md
│   ├── integration_tests.md
│   └── variants/               # Step type-specific test variants
├── interface/                   # Interface validation docs
│   └── interface_standard_validator.md
├── naming/                      # Naming validation docs
│   └── naming_standard_validator.md
└── shared/                      # Shared utilities docs
    └── chart_utils.md
```

## Key Features

### Simplified Integration Strategy
- **3-Function API**: Clean interface with `validate_development`, `validate_integration`, and `validate_production`
- **Fail-Fast Approach**: Production validation stops at first failure for efficiency
- **Basic Correlation**: Simple pass/fail correlation between testers
- **Result Caching**: Performance optimization through validation result caching

### Comprehensive Coverage
- **4-Level Architecture Validation**: Complete alignment validation across all architectural tiers
- **Step Type Awareness**: Enhanced validation based on SageMaker step types
- **Framework Detection**: Automatic detection of ML frameworks (XGBoost, PyTorch, etc.)
- **Quality Scoring**: Integrated scoring system with quality ratings

### Enhanced Reporting
- **Structured Reports**: Detailed validation reports with actionable recommendations
- **Visual Charts**: Alignment score visualization (when matplotlib available)
- **Export Formats**: JSON and HTML export capabilities
- **Console Output**: Clear, formatted console reporting

## Usage Examples

### Basic Development Validation
```python
from cursus.validation import validate_development
from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder

results = validate_development(TabularPreprocessingStepBuilder)
print(f"Development validation {'passed' if results['passed'] else 'failed'}")
```

### Integration Validation
```python
from cursus.validation import validate_integration

results = validate_integration(['tabular_preprocessing'])
print(f"Integration validation {'passed' if results['passed'] else 'failed'}")
```

### Production Readiness Validation
```python
from cursus.validation import validate_production
from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder

results = validate_production(TabularPreprocessingStepBuilder, 'tabular_preprocessing')
print(f"Production validation: {results['status']}")
print(f"Both testers passed: {results['both_passed']}")
```

## Framework Architecture

### Standardization Tester (Universal Step Builder Test)
- **Level 1**: Interface compliance testing
- **Level 2**: Specification alignment testing
- **Level 3**: Step creation testing
- **Level 4**: Integration testing
- **Step Type Specific**: SageMaker step type validation

### Alignment Tester (Unified Alignment Tester)
- **Level 1**: Script ↔ Contract alignment
- **Level 2**: Contract ↔ Specification alignment
- **Level 3**: Specification ↔ Dependencies alignment
- **Level 4**: Builder ↔ Configuration alignment

## Quality Assurance

The validation framework includes comprehensive quality assurance features:

- **Scoring System**: Weighted scoring across all test levels
- **Rating Levels**: Quality ratings from "Excellent" to "Poor"
- **Issue Classification**: Severity levels (Critical, Error, Warning, Info)
- **Recommendations**: Actionable improvement suggestions

## Legacy Compatibility

The framework maintains backward compatibility with legacy functions:
- `validate_step_builder()` → `validate_development()` (deprecated)
- `validate_step_integration()` → `validate_integration()` (deprecated)

## Configuration

### Level 3 Validation Modes
- **Strict**: Rigorous dependency validation
- **Relaxed**: Balanced validation (default)
- **Permissive**: Lenient validation for development

### Feature Flags
- `ENABLE_STEP_TYPE_AWARENESS`: Enable step type-aware validation enhancements

## Performance Optimizations

- **Result Caching**: Validation results cached for performance
- **Fail-Fast**: Production validation stops at first critical failure
- **Lazy Loading**: Components loaded only when needed
- **Batch Processing**: Efficient validation of multiple components

## Getting Started

1. **Read the Core API Documentation**: Start with `simple_integration.md`
2. **Understand Builder Testing**: Review `builders/universal_test.md`
3. **Learn Alignment Validation**: Explore `alignment/unified_alignment_tester.md`
4. **Check Examples**: See usage examples in each component documentation

## Related Documentation

- **Design Documents**: See `slipbox/1_design/` for architectural designs
- **Developer Guide**: See `slipbox/0_developer_guide/validation_checklist.md`
- **Implementation**: See `src/cursus/validation/` for source code

## Framework Statistics

- **Complexity Reduction**: 67% integration complexity reduction achieved
- **API Functions**: 8 total functions (3 core + 5 utility/legacy)
- **Test Levels**: 4 standardization levels + 4 alignment levels
- **Step Types Supported**: All SageMaker step types (Training, Processing, Transform, etc.)

The Cursus validation framework provides a robust, efficient, and user-friendly approach to ensuring quality and alignment across the entire step builder ecosystem.
