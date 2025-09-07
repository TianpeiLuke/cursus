---
tags:
  - code
  - cli
  - alignment
  - validation
  - testing
  - scripts
keywords:
  - alignment
  - validate
  - validate-all
  - validate-level
  - visualize
  - visualize-all
  - list-scripts
  - UnifiedAlignmentTester
  - AlignmentScorer
  - script validation
  - contract alignment
topics:
  - alignment validation
  - script testing
  - CLI tools
  - validation framework
language: python
date of note: 2024-12-07
---

# Alignment CLI

Command-line interface for the Unified Alignment Tester providing comprehensive alignment validation across all four levels of the cursus framework.

## Overview

The Alignment CLI provides comprehensive alignment validation tools for cursus scripts, supporting validation across four critical levels: Script ↔ Contract Alignment, Contract ↔ Specification Alignment, Specification ↔ Dependencies Alignment, and Builder ↔ Configuration Alignment. The CLI offers both individual script validation and batch validation capabilities with detailed reporting and visualization features.

The module supports multiple output formats including JSON and HTML reports, comprehensive scoring and quality rating systems, interactive visualization generation with charts and graphs, and flexible directory configuration for different project structures. All commands provide detailed help documentation and examples for effective usage.

## Classes and Methods

### Commands
- [`alignment`](#alignment) - Main command group for alignment validation tools
- [`validate`](#validate) - Validate alignment for a specific script
- [`validate-all`](#validate-all) - Validate alignment for all scripts in directory
- [`validate-level`](#validate-level) - Validate alignment for specific script at specific level
- [`visualize`](#visualize) - Generate visualization charts and scoring reports for specific script
- [`visualize-all`](#visualize-all) - Generate visualizations for all scripts
- [`list-scripts`](#list-scripts) - List all available scripts for validation

### Functions
- [`print_validation_summary`](#print_validation_summary) - Print validation results in formatted way
- [`save_report`](#save_report) - Save validation results to file
- [`generate_html_report`](#generate_html_report) - Generate HTML report for validation results
- [`main`](#main) - Main entry point for alignment CLI

## API Reference

### alignment

@click.group()

Main command group for alignment validation tools with context management.

```bash
# Access alignment commands
python -m cursus.cli alignment --help
```

### validate

validate(_script_name_, _--scripts-dir_, _--contracts-dir_, _--specs-dir_, _--builders-dir_, _--configs-dir_, _--output-dir_, _--format_, _--verbose_, _--show-scoring_)

Validates alignment for a specific script across all four validation levels.

**Parameters:**
- **script_name** (_str_) – Name of the script to validate (without .py extension).
- **--scripts-dir** (_Path_) – Directory containing scripts (default: src/cursus/steps/scripts).
- **--contracts-dir** (_Path_) – Directory containing contracts (default: src/cursus/steps/contracts).
- **--specs-dir** (_Path_) – Directory containing specifications (default: src/cursus/steps/specs).
- **--builders-dir** (_Path_) – Directory containing builders (default: src/cursus/steps/builders).
- **--configs-dir** (_Path_) – Directory containing configs (default: src/cursus/steps/configs).
- **--output-dir** (_Path_) – Output directory for reports (optional).
- **--format** (_Choice_) – Output format: json, html, or both (default: json).
- **--verbose** (_Flag_) – Show detailed output including all issues and recommendations.
- **--show-scoring** (_Flag_) – Show alignment scoring information and quality ratings.

```bash
# Basic validation
python -m cursus.cli alignment validate currency_conversion

# Detailed validation with scoring
python -m cursus.cli alignment validate currency_conversion --verbose --show-scoring

# Generate reports
python -m cursus.cli alignment validate dummy_training --output-dir ./reports --format html

# Custom directories
python -m cursus.cli alignment validate my_script --scripts-dir ./custom/scripts --verbose
```

### validate-all

validate-all(_--scripts-dir_, _--contracts-dir_, _--specs-dir_, _--builders-dir_, _--configs-dir_, _--output-dir_, _--format_, _--verbose_, _--continue-on-error_)

Validates alignment for all scripts in the scripts directory with comprehensive reporting.

**Parameters:**
- **--scripts-dir** (_Path_) – Directory containing scripts (default: src/cursus/steps/scripts).
- **--contracts-dir** (_Path_) – Directory containing contracts (default: src/cursus/steps/contracts).
- **--specs-dir** (_Path_) – Directory containing specifications (default: src/cursus/steps/specs).
- **--builders-dir** (_Path_) – Directory containing builders (default: src/cursus/steps/builders).
- **--configs-dir** (_Path_) – Directory containing configs (default: src/cursus/steps/configs).
- **--output-dir** (_Path_) – Output directory for reports (optional).
- **--format** (_Choice_) – Output format: json, html, or both (default: json).
- **--verbose** (_Flag_) – Show detailed output for all scripts.
- **--continue-on-error** (_Flag_) – Continue validation even if individual scripts fail.

```bash
# Validate all scripts
python -m cursus.cli alignment validate-all

# Comprehensive validation with reports
python -m cursus.cli alignment validate-all --output-dir ./reports --format both --verbose

# Continue on errors
python -m cursus.cli alignment validate-all --continue-on-error --verbose
```

### validate-level

validate-level(_script_name_, _level_, _--scripts-dir_, _--contracts-dir_, _--specs-dir_, _--builders-dir_, _--configs-dir_, _--verbose_)

Validates alignment for a specific script at a specific validation level.

**Parameters:**
- **script_name** (_str_) – Name of the script to validate (without .py extension).
- **level** (_int_) – Validation level (1-4): 1=Script↔Contract, 2=Contract↔Spec, 3=Spec↔Deps, 4=Builder↔Config.
- **--scripts-dir** (_Path_) – Directory containing scripts (default: src/cursus/steps/scripts).
- **--contracts-dir** (_Path_) – Directory containing contracts (default: src/cursus/steps/contracts).
- **--specs-dir** (_Path_) – Directory containing specifications (default: src/cursus/steps/specs).
- **--builders-dir** (_Path_) – Directory containing builders (default: src/cursus/steps/builders).
- **--configs-dir** (_Path_) – Directory containing configs (default: src/cursus/steps/configs).
- **--verbose** (_Flag_) – Show detailed output including recommendations.

```bash
# Validate specific level
python -m cursus.cli alignment validate-level currency_conversion 1 --verbose

# Test contract-specification alignment
python -m cursus.cli alignment validate-level dummy_training 2

# Check builder-configuration alignment
python -m cursus.cli alignment validate-level xgboost_training 4 --verbose
```

### visualize

visualize(_script_name_, _--scripts-dir_, _--contracts-dir_, _--specs-dir_, _--builders-dir_, _--configs-dir_, _--output-dir_, _--verbose_)

Generates visualization charts and scoring reports for a specific script.

**Parameters:**
- **script_name** (_str_) – Name of the script to validate (without .py extension).
- **--scripts-dir** (_Path_) – Directory containing scripts (default: src/cursus/steps/scripts).
- **--contracts-dir** (_Path_) – Directory containing contracts (default: src/cursus/steps/contracts).
- **--specs-dir** (_Path_) – Directory containing specifications (default: src/cursus/steps/specs).
- **--builders-dir** (_Path_) – Directory containing builders (default: src/cursus/steps/builders).
- **--configs-dir** (_Path_) – Directory containing configs (default: src/cursus/steps/configs).
- **--output-dir** (_Path_) – Output directory for visualization files (required).
- **--verbose** (_Flag_) – Show detailed output and level-by-level scores.

```bash
# Generate visualization
python -m cursus.cli alignment visualize currency_conversion --output-dir ./visualizations

# Detailed visualization with verbose output
python -m cursus.cli alignment visualize xgboost_model_evaluation --output-dir ./charts --verbose
```

### visualize-all

visualize-all(_--scripts-dir_, _--contracts-dir_, _--specs-dir_, _--builders-dir_, _--configs-dir_, _--output-dir_, _--verbose_, _--continue-on-error_)

Generates visualization charts and scoring reports for all scripts in the directory.

**Parameters:**
- **--scripts-dir** (_Path_) – Directory containing scripts (default: src/cursus/steps/scripts).
- **--contracts-dir** (_Path_) – Directory containing contracts (default: src/cursus/steps/contracts).
- **--specs-dir** (_Path_) – Directory containing specifications (default: src/cursus/steps/specs).
- **--builders-dir** (_Path_) – Directory containing builders (default: src/cursus/steps/builders).
- **--configs-dir** (_Path_) – Directory containing configs (default: src/cursus/steps/configs).
- **--output-dir** (_Path_) – Output directory for visualization files (required).
- **--verbose** (_Flag_) – Show detailed output for all scripts.
- **--continue-on-error** (_Flag_) – Continue visualization even if individual scripts fail.

```bash
# Generate all visualizations
python -m cursus.cli alignment visualize-all --output-dir ./visualizations

# Comprehensive visualization generation
python -m cursus.cli alignment visualize-all --output-dir ./charts --verbose --continue-on-error
```

### list-scripts

list-scripts(_--scripts-dir_)

Lists all available scripts that can be validated in the scripts directory.

**Parameters:**
- **--scripts-dir** (_Path_) – Directory containing scripts (default: src/cursus/steps/scripts).

```bash
# List available scripts
python -m cursus.cli alignment list-scripts

# List scripts in custom directory
python -m cursus.cli alignment list-scripts --scripts-dir ./custom/scripts
```

### print_validation_summary

print_validation_summary(_results_, _verbose=False_, _show_scoring=False_)

Prints validation results in a formatted way with color-coded output and detailed issue reporting.

**Parameters:**
- **results** (_Dict[str, Any]_) – Validation results dictionary from UnifiedAlignmentTester.
- **verbose** (_bool_) – Show detailed output including all issues and recommendations.
- **show_scoring** (_bool_) – Show alignment scoring information and quality ratings.

```python
from cursus.cli.alignment_cli import print_validation_summary

# Print basic summary
print_validation_summary(results)

# Print detailed summary with scoring
print_validation_summary(results, verbose=True, show_scoring=True)
```

### save_report

save_report(_script_name_, _results_, _output_dir_, _format_)

Saves validation results to file in specified format with JSON serialization handling.

**Parameters:**
- **script_name** (_str_) – Name of the script being reported.
- **results** (_Dict[str, Any]_) – Validation results dictionary.
- **output_dir** (_Path_) – Output directory for report files.
- **format** (_str_) – Output format ("json" or "html").

```python
from pathlib import Path
from cursus.cli.alignment_cli import save_report

# Save JSON report
save_report("my_script", results, Path("./reports"), "json")

# Save HTML report
save_report("my_script", results, Path("./reports"), "html")
```

### generate_html_report

generate_html_report(_script_name_, _results_)

Generates comprehensive HTML report for validation results with styling and interactive elements.

**Parameters:**
- **script_name** (_str_) – Name of the script being reported.
- **results** (_Dict[str, Any]_) – Validation results dictionary.

**Returns:**
- **str** – Complete HTML content for the report.

```python
from cursus.cli.alignment_cli import generate_html_report

# Generate HTML report content
html_content = generate_html_report("my_script", results)

# Save to file
with open("report.html", "w") as f:
    f.write(html_content)
```

### main

main()

Main entry point for alignment CLI with command group initialization.

```python
from cursus.cli.alignment_cli import main

# Run alignment CLI
main()
```

## Validation Levels

The alignment CLI validates four critical levels of alignment:

### Level 1: Script ↔ Contract Alignment
- Validates that script functions match contract specifications
- Checks parameter consistency and return value alignment
- Ensures proper error handling and exception management

### Level 2: Contract ↔ Specification Alignment  
- Validates that contracts align with step specifications
- Checks specification requirements against contract definitions
- Ensures proper interface consistency

### Level 3: Specification ↔ Dependencies Alignment
- Validates that specifications align with dependency requirements
- Checks dependency resolution and compatibility
- Ensures proper dependency injection patterns

### Level 4: Builder ↔ Configuration Alignment
- Validates that builders align with configuration classes
- Checks configuration parameter usage in builders
- Ensures proper configuration validation and handling

## Output Formats

### JSON Reports
- Machine-readable format for integration with other tools
- Complete validation results with detailed issue information
- Metadata including timestamps and validator versions
- JSON serialization with proper handling of complex objects

### HTML Reports
- Human-readable format with styling and interactive elements
- Color-coded status indicators and issue severity levels
- Summary statistics and detailed issue breakdowns
- Responsive design for different screen sizes

### Visualization Charts
- High-resolution PNG charts with scoring breakdowns
- Level-by-level score visualization with color coding
- Quality rating indicators and trend analysis
- Integration with scoring reports for comprehensive analysis

## Scoring System

The alignment CLI includes a comprehensive scoring system:

### Overall Score
- Calculated from all four validation levels
- Weighted scoring based on issue severity
- Range: 0-100 with quality rating categories

### Quality Ratings
- **Excellent** (90-100): Outstanding alignment with minimal issues
- **Good** (80-89): Strong alignment with minor issues
- **Satisfactory** (70-79): Acceptable alignment with some issues
- **Needs Work** (60-69): Significant issues requiring attention
- **Poor** (0-59): Major alignment problems requiring immediate attention

### Level Scores
- Individual scores for each validation level
- Detailed breakdown of issue types and severity
- Recommendations for improvement at each level

## Usage Patterns

### Development Workflow
```bash
# 1. List available scripts
python -m cursus.cli alignment list-scripts

# 2. Validate specific script during development
python -m cursus.cli alignment validate my_script --verbose --show-scoring

# 3. Focus on specific level if issues found
python -m cursus.cli alignment validate-level my_script 2 --verbose

# 4. Generate visualization for analysis
python -m cursus.cli alignment visualize my_script --output-dir ./analysis --verbose

# 5. Validate all scripts before release
python -m cursus.cli alignment validate-all --output-dir ./reports --format both
```

### CI/CD Integration
```bash
# Automated validation in CI pipeline
python -m cursus.cli alignment validate-all --format json --output-dir ./ci-reports

# Generate visualizations for documentation
python -m cursus.cli alignment visualize-all --output-dir ./docs/alignment-reports
```

### Quality Assurance
```bash
# Comprehensive quality assessment
python -m cursus.cli alignment validate-all --verbose --continue-on-error --output-dir ./qa-reports --format both

# Focus on failing scripts
python -m cursus.cli alignment validate failing_script --verbose --show-scoring --output-dir ./debug
```

## Error Handling

The alignment CLI provides comprehensive error handling:

- **Script Discovery**: Automatic discovery of Python scripts with proper filtering
- **Directory Validation**: Validates that required directories exist and are accessible
- **Validation Errors**: Graceful handling of validation failures with detailed error messages
- **Report Generation**: Robust report generation with fallback for serialization issues
- **Exit Codes**: Proper exit code handling for integration with automation tools

## Integration Points

- **UnifiedAlignmentTester**: Core validation engine for comprehensive alignment testing
- **AlignmentScorer**: Scoring system for quantitative alignment assessment
- **Validation Framework**: Integration with cursus validation infrastructure
- **Report Generation**: Multiple output formats for different use cases
- **CLI Framework**: Built on Click for robust command-line interface functionality

## Related Documentation

- [CLI Module](__init__.md) - Main CLI dispatcher and command routing
- [Builder Test CLI](builder_test_cli.md) - Step builder testing and validation
- [Validation CLI](validation_cli.md) - Naming and interface validation tools
- [Runtime Testing CLI](runtime_testing_cli.md) - Script runtime testing and benchmarking
- [Unified Alignment Tester](../../validation/alignment/unified_alignment_tester.md) - Core validation engine
- [Alignment Scorer](../../validation/alignment/alignment_scorer.md) - Scoring and quality assessment system
