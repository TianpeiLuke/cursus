---
tags:
  - code
  - cli
  - testing
  - step_builders
  - validation
keywords:
  - step builder testing
  - universal test system
  - CLI testing interface
  - builder validation
  - test automation
  - level testing
  - variant testing
topics:
  - step builder testing
  - test automation
  - validation framework
  - CLI design
language: python
date of note: 2025-08-07
---

# Builder Test CLI Documentation

## Overview

The Builder Test CLI (`src/cursus/cli/builder_test_cli.py`) provides a comprehensive command-line interface for running Universal Step Builder Tests. It implements the 4-level testing architecture defined by UniversalStepBuilderTestBase and supports variant-specific testing for different step builder types.

## Architecture

### Command Structure

The CLI supports multiple testing modes:

```
builder_test_cli
├── all <builder_class>              # Run all tests (universal suite)
├── level <1-4> <builder_class>      # Run specific level tests
├── variant <variant> <builder_class> # Run variant-specific tests
└── list-builders                    # List available builders
```

### Core Components

#### 1. Test Execution Engine
- **Universal Testing**: Complete test suite execution
- **Level-Specific Testing**: Targeted testing at specific levels
- **Variant Testing**: Specialized tests for step types
- **Result Aggregation**: Comprehensive result collection and reporting

#### 2. Builder Discovery System
- **Automatic Discovery**: Scans builders directory for available classes
- **AST Parsing**: Handles modules with missing dependencies
- **Import Validation**: Verifies builder class accessibility
- **Comprehensive Listing**: Shows all discoverable builders

#### 3. Result Reporting System
- **Formatted Output**: Structured test result presentation
- **Grouping Logic**: Organizes results by test categories
- **Progress Tracking**: Shows pass/fail statistics
- **Verbose Mode**: Detailed error information and suggestions

## Key Features

### 1. Four-Level Testing Architecture

#### Level 1: Interface Tests
```bash
python -m cursus.cli.builder_test_cli level 1 <builder_class>
```
- **Inheritance validation**: Ensures proper base class inheritance
- **Naming conventions**: Validates class and method naming
- **Required methods**: Checks for mandatory method implementations
- **Registry integration**: Verifies builder registry compliance
- **Documentation standards**: Validates docstring presence and format
- **Type hints**: Ensures proper type annotation usage
- **Error handling**: Validates exception handling patterns

#### Level 2: Specification Tests
```bash
python -m cursus.cli.builder_test_cli level 2 <builder_class>
```
- **Specification usage**: Validates spec integration
- **Contract alignment**: Ensures spec-contract consistency
- **Environment variables**: Tests environment variable handling
- **Job arguments**: Validates job argument processing

#### Level 3: Path Mapping Tests
```bash
python -m cursus.cli.builder_test_cli level 3 <builder_class>
```
- **Input path mapping**: Validates input path resolution
- **Output path mapping**: Tests output path generation
- **Property path validity**: Ensures property paths are accessible

#### Level 4: Integration Tests
```bash
python -m cursus.cli.builder_test_cli level 4 <builder_class>
```
- **Dependency resolution**: Tests dependency injection
- **Step creation**: Validates actual step instantiation
- **Step naming**: Ensures proper step name generation

### 2. Variant-Specific Testing

#### Processing Variant
```bash
python -m cursus.cli.builder_test_cli variant processing <builder_class>
```
- **Step type validation**: Ensures ProcessingStep creation
- **Processor creation**: Validates processor instantiation
- **Processing-specific methods**: Tests processing-related functionality

### 3. Automatic Builder Discovery

#### Discovery Algorithm
- **Filesystem scanning**: Searches `cursus/steps/builders` directory
- **Module importing**: Attempts dynamic module imports
- **Class extraction**: Identifies classes ending with "StepBuilder"
- **AST fallback**: Uses AST parsing for import failures
- **Dependency handling**: Gracefully handles missing dependencies

#### Discovery Features
- **Complete coverage**: Finds all 15+ available builders
- **Robust error handling**: Continues despite import failures
- **Sorted output**: Provides consistent, alphabetical listing
- **Path normalization**: Handles both development and installed environments

### 4. Comprehensive Result Reporting

#### Result Grouping
- **Level 1 (Interface)**: Interface compliance tests
- **Level 2 (Specification)**: Specification-driven tests
- **Level 3 (Path Mapping)**: Path resolution tests
- **Level 4 (Integration)**: Integration and creation tests
- **Step Type Specific**: Variant-specific functionality tests

#### Output Format
```
📊 Test Results Summary: 25/30 tests passed (83.3%)
============================================================

📁 Level 1 (Interface): 8/10 passed (80.0%)
  ✅ inheritance_validation
  ✅ naming_conventions
  ❌ required_methods
    💬 Missing required method: validate_configuration
  ✅ registry_integration
  ...

📁 Level 2 (Specification): 5/5 passed (100.0%)
  ✅ specification_usage
  ✅ contract_alignment
  ...
```

## Implementation Details

### Dynamic Class Import System
```python
def import_builder_class(class_path: str) -> Type:
    """Import a builder class from a module path."""
    # Handles src. prefix removal
    # Dynamic module importing
    # Class attribute access
    # Comprehensive error handling
```

### Test Execution Framework
```python
def run_level_tests(builder_class: Type, level: int, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """Run tests for a specific level."""
    # Level-to-class mapping
    # Test instance creation
    # Result collection
    # Error handling
```

### Builder Discovery Engine
```python
def list_available_builders() -> List[str]:
    """List available step builder classes by scanning the builders directory."""
    # Filesystem scanning
    # Dynamic importing with fallback
    # AST parsing for failed imports
    # Result aggregation and sorting
```

## Usage Patterns

### Development Testing Workflow
1. **Discovery**: `cursus test list-builders`
2. **Interface Validation**: `cursus test level 1 <builder>`
3. **Specification Testing**: `cursus test level 2 <builder>`
4. **Path Mapping**: `cursus test level 3 <builder>`
5. **Integration Testing**: `cursus test level 4 <builder>`
6. **Full Validation**: `cursus test all <builder>`

### Continuous Integration
```bash
# Test all builders at all levels
for builder in $(cursus test list-builders); do
    cursus test all "$builder" || exit 1
done
```

### Debugging Workflow
```bash
# Verbose mode for detailed error information
cursus test level 1 <builder> --verbose
```

## Error Handling

### Import Error Management
- **Graceful degradation**: Continues operation despite import failures
- **AST parsing fallback**: Extracts class names without importing
- **Clear error messages**: Provides actionable error information
- **Dependency guidance**: Suggests missing dependency installation

### Test Failure Reporting
- **Detailed messages**: Specific failure reasons
- **Suggestions**: Actionable improvement recommendations
- **Context information**: Relevant test context and expectations
- **Exit codes**: Proper exit codes for automation

## Integration Points

### 1. Universal Test Framework
- **Direct integration**: Uses UniversalStepBuilderTest classes
- **Result formatting**: Consistent with framework expectations
- **Test discovery**: Automatic test method discovery
- **Configuration passing**: Proper test configuration

### 2. Step Builder Registry
- **Builder enumeration**: Accesses registered builders
- **Class resolution**: Resolves builder classes from registry
- **Validation integration**: Uses registry for validation

### 3. CLI Framework Integration
- **Argument parsing**: Comprehensive argument validation
- **Help generation**: Automatic help text generation
- **Error handling**: Consistent error handling patterns
- **Output formatting**: Standardized output formatting

## Extension Points

### Adding New Test Levels
```python
# Add to test_classes mapping
test_classes = {
    1: InterfaceTests,
    2: SpecificationTests,
    3: PathMappingTests,
    4: IntegrationTests,
    5: NewTestLevel,  # New level
}
```

### Adding New Variants
```python
# Add to variant_classes mapping
variant_classes = {
    "processing": ProcessingStepBuilderTest,
    "training": TrainingStepBuilderTest,  # New variant
    "transform": TransformStepBuilderTest,  # New variant
}
```

### Custom Result Formatters
- **Pluggable formatters**: Support for custom output formats
- **Export options**: JSON, XML, or other structured formats
- **Integration hooks**: Hooks for CI/CD integration

## Performance Considerations

### Lazy Loading
- **Module imports**: Only import when needed
- **Test instantiation**: Create test instances on demand
- **Result caching**: Cache results for repeated operations

### Memory Management
- **Class cleanup**: Proper cleanup of imported classes
- **Result streaming**: Stream results for large test suites
- **Memory monitoring**: Track memory usage during testing

## Best Practices

### Test Organization
- **Logical grouping**: Group related tests together
- **Clear naming**: Use descriptive test names
- **Proper isolation**: Ensure test independence
- **Resource cleanup**: Clean up test resources

### Error Reporting
- **Actionable messages**: Provide clear, actionable error messages
- **Context preservation**: Maintain error context
- **Suggestion provision**: Offer improvement suggestions
- **Documentation links**: Link to relevant documentation

## Future Enhancements

### Planned Features
- **Parallel testing**: Run tests in parallel for speed
- **Test filtering**: Filter tests by tags or patterns
- **Custom test suites**: User-defined test combinations
- **Report generation**: Generate detailed test reports

### Integration Improvements
- **IDE integration**: Support for IDE test runners
- **CI/CD plugins**: Specialized CI/CD integrations
- **Monitoring integration**: Integration with monitoring systems
- **Notification systems**: Test result notifications

This CLI provides a comprehensive, user-friendly interface for validating step builders according to the Universal Step Builder Test architecture, ensuring consistent quality and compliance across all step builder implementations.
