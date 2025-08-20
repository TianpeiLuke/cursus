---
tags:
  - code
  - validation
  - alignment
  - static_analysis
  - builder_analysis
keywords:
  - builder analyzer
  - argument extraction
  - step builder analysis
  - AST parsing
  - job arguments
  - builder registry
  - config validation
topics:
  - validation framework
  - static code analysis
  - builder argument extraction
language: python
date of note: 2025-08-19
---

# Builder Analyzer

## Overview

The Builder Analyzer module provides specialized analysis of step builder classes to extract command-line arguments from `_get_job_arguments()` methods. This enables validation of config-driven arguments that are provided by builders but may not be declared in script contracts, supporting comprehensive alignment validation.

## Architecture

### Core Components

1. **BuilderArgumentExtractor**: AST-based extraction of job arguments from builder methods
2. **BuilderRegistry**: Mapping system for associating scripts with their corresponding builders
3. **Argument Analysis**: Comprehensive extraction of command-line arguments from builder code
4. **Name Resolution**: Intelligent mapping between script names and builder files

### Analysis Capabilities

The analyzer provides multi-faceted builder analysis:

- **AST Parsing**: Deep analysis of builder Python files using Abstract Syntax Trees
- **Argument Extraction**: Identification of command-line arguments from `_get_job_arguments()` methods
- **Script Mapping**: Association of scripts with their corresponding step builders
- **Name Variations**: Handling of common naming pattern variations

## Implementation Details

### BuilderArgumentExtractor Class

```python
class BuilderArgumentExtractor:
    """
    Extracts command-line arguments from step builder _get_job_arguments() methods.
    
    This class uses AST parsing to analyze builder Python files and extract
    the arguments that builders pass to scripts via the _get_job_arguments() method.
    """
```

#### Key Methods

##### `extract_job_arguments() -> Set[str]`

Extracts command-line arguments from the `_get_job_arguments()` method:

**Extraction Process:**
1. Parses builder file into AST
2. Locates `_get_job_arguments()` method
3. Identifies string literals starting with `--`
4. Extracts argument names (without `--` prefix)
5. Returns deduplicated set of argument names

**Supported Patterns:**
- Direct string literals: `"--learning-rate"`
- List elements: `["--epochs", "--batch-size"]`
- Both modern (`ast.Constant`) and legacy (`ast.Str`) AST nodes

##### `get_method_source() -> Optional[str]`

Retrieves the source code of the `_get_job_arguments()` method for debugging:

- Extracts method source from original file
- Provides context for argument extraction validation
- Useful for troubleshooting extraction issues

### BuilderRegistry Class

```python
class BuilderRegistry:
    """
    Registry for mapping script names to their corresponding step builders.
    
    This class helps find the appropriate builder file for a given script
    to enable builder argument extraction during validation.
    """
```

#### Key Methods

##### `get_builder_for_script(script_name: str) -> Optional[str]`

Finds the builder file path for a given script name:

**Resolution Strategy:**
1. Direct mapping from pre-built registry
2. Filename pattern matching
3. Content-based association analysis
4. Name variation handling

##### `_extract_script_names_from_builder(builder_file: Path) -> List[str]`

Extracts associated script names from a builder file using multiple heuristics:

**Extraction Heuristics:**
1. **Filename Pattern**: `builder_<script_name>_step.py` → `<script_name>`
2. **Config Imports**: `from ..configs.config_<name>_step import` → `<name>`
3. **Entry Point References**: `entry_point.*"<name>.py"` → `<name>`
4. **Name Variations**: Common naming pattern variations

##### `_generate_name_variations(name: str) -> List[str]`

Generates common naming variations for robust script-builder association:

**Supported Variations:**
- `preprocessing` ↔ `preprocess`
- `evaluation` ↔ `eval`
- `xgboost` ↔ `xgb`
- Additional domain-specific patterns

## Usage Examples

### Basic Argument Extraction

```python
from cursus.validation.alignment.static_analysis.builder_analyzer import BuilderArgumentExtractor

# Extract arguments from a specific builder file
builder_path = "/path/to/builder_xgboost_training_step.py"
extractor = BuilderArgumentExtractor(builder_path)

# Get job arguments
arguments = extractor.extract_job_arguments()
print(f"Builder provides arguments: {arguments}")
# Output: {'learning-rate', 'n-estimators', 'max-depth', 'subsample'}
```

### Builder Registry Usage

```python
from cursus.validation.alignment.static_analysis.builder_analyzer import BuilderRegistry

# Initialize registry with builders directory
builders_dir = "/path/to/builders"
registry = BuilderRegistry(builders_dir)

# Find builder for a script
script_name = "xgboost_training"
builder_path = registry.get_builder_for_script(script_name)

if builder_path:
    print(f"Builder for {script_name}: {builder_path}")
    
    # Extract arguments from the found builder
    extractor = BuilderArgumentExtractor(builder_path)
    arguments = extractor.extract_job_arguments()
    print(f"Arguments: {arguments}")
```

### Convenience Function

```python
from cursus.validation.alignment.static_analysis.builder_analyzer import extract_builder_arguments

# One-step argument extraction
script_name = "model_evaluation_xgb"
builders_dir = "/path/to/builders"

arguments = extract_builder_arguments(script_name, builders_dir)
print(f"Builder arguments for {script_name}: {arguments}")
```

### Registry Exploration

```python
# Get all script-to-builder mappings
registry = BuilderRegistry("/path/to/builders")
all_mappings = registry.get_all_mappings()

for script, builder in all_mappings.items():
    print(f"{script} -> {builder}")
```

### Method Source Debugging

```python
# Debug argument extraction by examining method source
extractor = BuilderArgumentExtractor("/path/to/builder_file.py")

# Get the actual method source code
method_source = extractor.get_method_source()
if method_source:
    print("_get_job_arguments method source:")
    print(method_source)

# Compare with extracted arguments
arguments = extractor.extract_job_arguments()
print(f"Extracted arguments: {arguments}")
```

## Integration Points

### Alignment Validation Framework

The Builder Analyzer integrates with the alignment validation system:

```python
class AlignmentValidator:
    def validate_script_builder_alignment(self, script_name, script_contract):
        # Extract builder arguments
        builder_args = extract_builder_arguments(script_name, self.builders_dir)
        
        # Get script contract arguments
        contract_args = set(script_contract.get('arguments', {}).keys())
        
        # Find arguments provided by builder but not in contract
        builder_only_args = builder_args - contract_args
        
        if builder_only_args:
            return {
                'status': 'warning',
                'message': f'Builder provides arguments not in contract: {builder_only_args}',
                'builder_args': builder_args,
                'contract_args': contract_args
            }
        
        return {'status': 'aligned'}
```

### Static Analysis Pipeline

Works as part of comprehensive static analysis:

- **AST Integration**: Uses Python AST parsing for accurate code analysis
- **Contract Validation**: Validates builder arguments against script contracts
- **Registry Management**: Maintains script-to-builder associations
- **Argument Reconciliation**: Identifies discrepancies between builders and contracts

### Configuration Validation

Supports configuration-driven validation:

- **Config-Builder Alignment**: Validates configuration classes against builder arguments
- **Argument Consistency**: Ensures consistent argument handling across components
- **Missing Argument Detection**: Identifies arguments missing from contracts
- **Redundant Argument Detection**: Finds unused arguments in builders

## Advanced Features

### AST-Based Analysis

Sophisticated code analysis using Python's AST module:

- **Method Location**: Precise identification of `_get_job_arguments()` methods
- **String Extraction**: Robust extraction of string literals from various AST node types
- **List Processing**: Handles argument lists and complex data structures
- **Version Compatibility**: Supports both modern and legacy AST node types

### Intelligent Name Mapping

Advanced script-to-builder association:

- **Pattern Recognition**: Multiple heuristics for name association
- **Content Analysis**: Code-based association discovery
- **Variation Handling**: Common naming pattern variations
- **Fallback Strategies**: Graceful handling of non-standard naming

### Error Resilience

Robust error handling throughout the analysis:

- **Parse Failures**: Graceful handling of malformed Python files
- **Missing Methods**: Continues analysis when methods are not found
- **File Access**: Handles file system access issues
- **AST Errors**: Recovers from AST parsing failures

## Performance Considerations

Optimized for large-scale builder analysis:

- **Lazy Loading**: On-demand file parsing and analysis
- **Registry Caching**: Pre-built mappings for fast lookup
- **AST Reuse**: Efficient AST parsing and reuse
- **Memory Management**: Efficient handling of large builder directories

## Testing and Validation

The analyzer supports comprehensive testing:

- **Mock Builders**: Can analyze synthetic builder files
- **AST Testing**: Validates AST parsing accuracy
- **Registry Testing**: Tests script-to-builder mapping
- **Argument Testing**: Verifies argument extraction accuracy

## Error Handling

Comprehensive error handling strategies:

1. **File Parsing**: Graceful handling of syntax errors and encoding issues
2. **Method Missing**: Continues analysis when `_get_job_arguments()` is not found
3. **AST Failures**: Recovers from AST parsing and traversal errors
4. **Registry Errors**: Handles missing or inaccessible builder directories

## Future Enhancements

Potential improvements for the analyzer:

1. **Enhanced Patterns**: Support for more complex argument patterns
2. **Type Analysis**: Argument type inference from builder code
3. **Documentation Extraction**: Extract argument documentation from builders
4. **Performance Metrics**: Track analysis performance and accuracy
5. **Integration APIs**: Enhanced integration with validation frameworks

## Conclusion

The Builder Analyzer provides essential static analysis capabilities for step builder argument extraction, enabling comprehensive validation of config-driven arguments in the alignment validation framework. Its AST-based approach ensures accurate and reliable extraction of builder-provided arguments, supporting robust validation of script-builder alignment.

The analyzer serves as a critical component in maintaining consistency between step builders and script contracts, enabling automated detection of argument mismatches and ensuring proper configuration-driven argument handling across the validation framework.
