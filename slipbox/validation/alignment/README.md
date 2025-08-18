---
tags:
  - entry_point
  - test
  - validation
  - alignment
  - documentation
keywords:
  - alignment validation
  - unified alignment tester
  - four-tier architecture
  - script contract alignment
  - specification dependency alignment
topics:
  - alignment validation
  - test framework
  - architectural validation
  - component integration
language: python
date of note: 2025-08-18
---

# Alignment Validation System

The Alignment Validation System provides comprehensive validation of component alignment across the four-tier architecture of the Cursus framework. It ensures that all components (scripts, contracts, specifications, builders, and configurations) are properly aligned and integrated.

## Overview

The Unified Alignment Tester orchestrates validation across four distinct alignment levels, each focusing on specific component relationships within the architectural stack.

## Four-Tier Architecture Validation

### Level 1: Script ↔ Contract Alignment
**Purpose**: Validates that processing scripts align with their corresponding script contracts

**Key Validations**:
- Script function signatures match contract specifications
- Required parameters are properly defined
- Return types align with contract expectations
- Environment variable usage consistency

**Tester**: `ScriptContractAlignmentTester`

### Level 2: Contract ↔ Specification Alignment
**Purpose**: Validates that script contracts align with step specifications

**Key Validations**:
- Contract parameters match specification requirements
- Property paths are correctly defined
- Configuration fields are properly mapped
- Type consistency across layers

**Tester**: `ContractSpecificationAlignmentTester`

### Level 3: Specification ↔ Dependencies Alignment
**Purpose**: Validates that step specifications align with their dependencies

**Key Validations**:
- Dependency declarations are complete
- Property path references are valid
- Configuration dependencies are satisfied
- Cross-specification consistency

**Tester**: `SpecificationDependencyAlignmentTester`

### Level 4: Builder ↔ Configuration Alignment
**Purpose**: Validates that step builders align with their configuration requirements

**Key Validations**:
- Builder configuration usage matches specifications
- Required configuration fields are accessed
- Configuration field types are correct
- Default value handling is consistent

**Tester**: `BuilderConfigurationAlignmentTester`

## Unified Alignment Tester

The `UnifiedAlignmentTester` class serves as the main orchestrator for comprehensive alignment validation.

### Key Features

- **Comprehensive Coverage**: Validates all four alignment levels
- **Step Type Awareness**: Enhanced validation based on SageMaker step types
- **Framework Detection**: Automatic detection of ML frameworks
- **Flexible Configuration**: Configurable validation modes and skip options
- **Rich Reporting**: Detailed reports with actionable recommendations

### Validation Modes

#### Level 3 Validation Modes
- **Strict**: Rigorous dependency validation with strict requirements
- **Relaxed**: Balanced validation (default) with reasonable flexibility
- **Permissive**: Lenient validation for development environments

### Usage Examples

#### Full Validation
```python
from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

tester = UnifiedAlignmentTester()
report = tester.run_full_validation()
print(f"Overall status: {report.get_validation_summary()['overall_status']}")
```

#### Specific Script Validation
```python
tester = UnifiedAlignmentTester()
results = tester.validate_specific_script('tabular_preprocessing')
print(f"Script validation: {results['overall_status']}")
```

#### Level-Specific Validation
```python
tester = UnifiedAlignmentTester()
report = tester.run_level_validation(level=1, target_scripts=['tabular_preprocessing'])
```

## Step Type Enhancement System

### Phase 1: Step Type Awareness
The framework includes step type awareness features that enhance validation with context-specific insights.

**Feature Flag**: `ENABLE_STEP_TYPE_AWARENESS=true`

### Phase 3: Step Type Enhancement Router
The `StepTypeEnhancementRouter` provides step type-specific validation enhancements for different SageMaker step types.

**Supported Step Types**:
- Training steps
- Processing steps
- Transform steps
- CreateModel steps
- RegisterModel steps
- Utility steps

## Reporting and Visualization

### Alignment Report
The `AlignmentReport` class provides comprehensive reporting capabilities:

- **Summary Generation**: High-level validation summaries
- **Issue Classification**: Severity-based issue categorization
- **Recommendations**: Actionable improvement suggestions
- **Export Formats**: JSON and HTML export options

### Quality Scoring
Integrated scoring system provides quality metrics:

- **Overall Score**: Weighted score across all levels (0-100)
- **Level Scores**: Individual scores for each alignment level
- **Quality Ratings**: Ratings from "Excellent" to "Poor"
- **Visual Charts**: Alignment score visualization (when matplotlib available)

### Issue Severity Levels
- **Critical**: Issues that prevent system operation
- **Error**: Issues that cause validation failures
- **Warning**: Issues that should be addressed but don't prevent operation
- **Info**: Informational messages and recommendations

## Configuration Options

### Directory Configuration
```python
tester = UnifiedAlignmentTester(
    scripts_dir="src/cursus/steps/scripts",
    contracts_dir="src/cursus/steps/contracts",
    specs_dir="src/cursus/steps/specs",
    builders_dir="src/cursus/steps/builders",
    configs_dir="src/cursus/steps/configs",
    level3_validation_mode="relaxed"
)
```

### Validation Customization
```python
# Skip specific levels
report = tester.run_full_validation(skip_levels=[3, 4])

# Target specific scripts
report = tester.run_full_validation(target_scripts=['tabular_preprocessing'])
```

## Performance Features

### Caching and Optimization
- **Result Caching**: Validation results cached for performance
- **Lazy Loading**: Components loaded only when needed
- **Batch Processing**: Efficient validation of multiple components
- **Error Recovery**: Graceful handling of validation errors

### Statistics and Monitoring
- **Validation Metrics**: Track validation frequency and performance
- **Cache Performance**: Monitor cache hit rates and effectiveness
- **Error Tracking**: Track and report validation errors

## Integration Points

### With Standardization Tester
The alignment system integrates with the Universal Step Builder Test for comprehensive validation coverage.

### With Step Type System
Enhanced validation based on detected step types and framework patterns.

### With Configuration System
Deep integration with the configuration management system for accurate validation.

## Best Practices

### Development Workflow
1. **Start with Level 1**: Ensure script-contract alignment first
2. **Progress Sequentially**: Validate each level before moving to the next
3. **Address Critical Issues**: Fix critical issues before proceeding
4. **Use Appropriate Mode**: Choose validation mode based on development phase

### Error Resolution
1. **Read Issue Details**: Examine detailed issue information
2. **Follow Recommendations**: Implement suggested fixes
3. **Re-validate**: Run validation again after fixes
4. **Monitor Progress**: Track improvement over time

### Performance Optimization
1. **Use Target Scripts**: Validate specific scripts when possible
2. **Skip Unnecessary Levels**: Skip levels that aren't relevant
3. **Monitor Statistics**: Track validation performance
4. **Clear Cache**: Clear cache when components change significantly

## Directory Structure

```
slipbox/validation/alignment/
├── README.md                           # This overview document
├── unified_alignment_tester.md         # Main orchestrator documentation
├── script_contract_alignment.md        # Level 1 validation
├── contract_spec_alignment.md          # Level 2 validation
├── spec_dependency_alignment.md        # Level 3 validation
├── builder_config_alignment.md         # Level 4 validation
├── step_type_enhancement.md            # Step type enhancement system
├── alignment_reporter.md               # Reporting system
├── alignment_utils.md                  # Utility functions
└── validators/                         # Specific validator documentation
    ├── contract_spec_validator.md
    ├── dependency_validator.md
    └── script_contract_validator.md
```

## Related Documentation

- **Core API**: See `../simple_integration.md` for the main API
- **Builder Testing**: See `../builders/` for standardization testing
- **Design Documents**: See `slipbox/1_design/` for architectural designs
- **Implementation**: See `src/cursus/validation/alignment/` for source code

The Alignment Validation System provides a comprehensive, flexible, and powerful framework for ensuring component alignment across the entire Cursus architecture.
