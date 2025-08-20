---
tags:
  - code
  - validation
  - alignment
  - static_analysis
  - import_analysis
keywords:
  - import analyzer
  - static analysis
  - module imports
  - framework requirements
  - dependency analysis
  - import organization
  - usage patterns
topics:
  - validation framework
  - static code analysis
  - import management
language: python
date of note: 2025-08-19
---

# Import Analyzer

## Overview

The `ImportAnalyzer` class provides comprehensive analysis of import statements and framework usage in Python scripts. It supports alignment validation by analyzing module imports, framework requirements, dependency patterns, and import organization to ensure scripts meet project standards.

## Architecture

### Core Capabilities

1. **Import Categorization**: Classifies imports as standard library, third-party, or local modules
2. **Framework Detection**: Identifies framework requirements and potential versions
3. **Usage Analysis**: Analyzes how imported modules are actually used in the script
4. **Organization Validation**: Checks import organization against PEP 8 standards
5. **Dependency Management**: Finds unused imports and missing dependencies

### Analysis Dimensions

The analyzer provides multi-dimensional analysis of import patterns:

- **Structural**: Import types, organization, and categorization
- **Functional**: Usage patterns and dependency relationships
- **Compliance**: PEP 8 organization and style standards
- **Optimization**: Unused imports and missing dependencies

## Implementation Details

### Class Structure

```python
class ImportAnalyzer:
    """
    Analyzes import statements and framework usage in Python scripts.
    
    Provides insights into:
    - Module imports and their usage patterns
    - Framework requirements and versions
    - Standard library vs third-party dependencies
    - Import organization and conventions
    """
```

### Key Methods

#### `categorize_imports() -> Dict[str, List[ImportStatement]]`

Categorizes imports by type using comprehensive module databases:

- **Standard Library**: Built-in Python modules (os, sys, json, etc.)
- **Third-Party**: External packages (pandas, numpy, sklearn, etc.)
- **Local**: Project-specific modules (src, cursus, relative imports)
- **Unknown**: Unrecognized modules requiring investigation

**Features:**
- Extensive standard library module database
- Common data science and ML framework recognition
- Local module pattern detection
- Relative import handling

#### `extract_framework_requirements() -> Dict[str, Optional[str]]`

Identifies framework dependencies and attempts version extraction:

**Supported Frameworks:**
- Data Science: pandas, numpy, scipy, matplotlib, seaborn
- Machine Learning: scikit-learn, tensorflow, torch, xgboost, lightgbm
- Cloud/Infrastructure: boto3, sagemaker, requests
- Development: pydantic, joblib, pytest

**Version Detection:**
- Comment-based version hints
- Requirements file patterns
- Inline version specifications

#### `analyze_import_usage() -> Dict[str, Dict[str, Any]]`

Provides detailed usage analysis for each import:

```python
{
    'module_name': {
        'import_type': 'from_import' | 'direct_import',
        'alias': 'module_alias',
        'usage_count': int,
        'is_used': bool,
        'usage_patterns': ['function_call', 'attribute_access', ...],
        'line_number': int
    }
}
```

**Usage Pattern Detection:**
- Function calls: `module.function()`
- Class instantiation: `Module()`
- Attribute access: `module.attribute`
- Direct calls: `module()`

#### `find_unused_imports() -> List[ImportStatement]`

Identifies imports that are not referenced in the script:

- Analyzes actual usage patterns
- Excludes import statements from usage counts
- Handles both direct imports and from-imports
- Supports alias resolution

#### `find_missing_imports(required_modules: List[str]) -> List[str]`

Finds required modules that are not imported:

- Checks against required module list
- Handles module aliases and from-imports
- Supports partial module name matching
- Useful for framework compliance validation

#### `check_import_organization() -> Dict[str, Any]`

Validates import organization against PEP 8 standards:

**PEP 8 Import Order:**
1. Standard library imports
2. Third-party imports
3. Local application imports

**Validation Features:**
- Import order verification
- Spacing and grouping analysis
- Style compliance checking
- Detailed issue reporting with suggestions

#### `get_import_summary() -> Dict[str, Any]`

Provides comprehensive import analysis summary:

```python
{
    'total_imports': int,
    'categories': {'standard_library': int, 'third_party': int, ...},
    'framework_requirements': {'pandas': '1.3.0', ...},
    'unused_imports': int,
    'usage_summary': {'used_imports': int, 'unused_imports': int},
    'organization_issues': int,
    'organization_suggestions': int
}
```

## Usage Examples

### Basic Import Analysis

```python
from cursus.validation.alignment.static_analysis.import_analyzer import ImportAnalyzer
from cursus.validation.alignment.alignment_utils import ImportStatement

# Initialize with extracted imports and script content
imports = [
    ImportStatement(module_name='pandas', import_alias='pd', line_number=1),
    ImportStatement(module_name='numpy', import_alias='np', line_number=2),
    ImportStatement(module_name='sklearn.ensemble', is_from_import=True, 
                   imported_items=['RandomForestClassifier'], line_number=3)
]

script_content = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df = pd.DataFrame({'x': [1, 2, 3]})
arr = np.array([1, 2, 3])
model = RandomForestClassifier()
"""

analyzer = ImportAnalyzer(imports, script_content)
```

### Import Categorization

```python
# Categorize imports by type
categories = analyzer.categorize_imports()

print(f"Standard library: {len(categories['standard_library'])}")
print(f"Third-party: {len(categories['third_party'])}")
print(f"Local: {len(categories['local'])}")
```

### Framework Requirements Analysis

```python
# Extract framework requirements
requirements = analyzer.extract_framework_requirements()

for framework, version in requirements.items():
    print(f"{framework}: {version or 'version not specified'}")
```

### Usage Analysis

```python
# Analyze import usage
usage_analysis = analyzer.analyze_import_usage()

for module, analysis in usage_analysis.items():
    print(f"{module}: used {analysis['usage_count']} times")
    print(f"  Patterns: {analysis['usage_patterns']}")
```

### Import Organization Validation

```python
# Check import organization
organization = analyzer.check_import_organization()

if organization['issues']:
    print("Import organization issues:")
    for issue in organization['issues']:
        print(f"  Line {issue['line']}: {issue['message']}")
        print(f"    Suggestion: {issue['suggestion']}")
```

### Comprehensive Analysis

```python
# Get complete import summary
summary = analyzer.get_import_summary()

print(f"Total imports: {summary['total_imports']}")
print(f"Framework requirements: {summary['framework_requirements']}")
print(f"Unused imports: {summary['unused_imports']}")
print(f"Organization issues: {summary['organization_issues']}")
```

## Integration Points

### Alignment Validation Framework

The ImportAnalyzer integrates with the alignment validation system:

```python
class ScriptValidator:
    def validate_imports(self, script_path):
        # Extract imports from script
        imports = self.extract_imports(script_path)
        script_content = self.read_script(script_path)
        
        # Analyze imports
        analyzer = ImportAnalyzer(imports, script_content)
        
        # Validate against requirements
        required_frameworks = self.get_required_frameworks()
        missing = analyzer.find_missing_imports(required_frameworks)
        unused = analyzer.find_unused_imports()
        
        # Check organization
        organization = analyzer.check_import_organization()
        
        return {
            'missing_imports': missing,
            'unused_imports': unused,
            'organization_issues': organization['issues']
        }
```

### Static Analysis Pipeline

Works as part of comprehensive static analysis:

- **Import Extraction**: Receives ImportStatement objects from AST analysis
- **Usage Analysis**: Analyzes actual module usage in script content
- **Compliance Checking**: Validates against project standards
- **Report Generation**: Provides detailed analysis reports

### Framework Compliance

Supports framework-specific validation:

- **Data Science**: Validates pandas, numpy, scipy usage patterns
- **Machine Learning**: Checks sklearn, tensorflow, torch requirements
- **Cloud Integration**: Validates AWS/SageMaker dependencies
- **Development Tools**: Checks testing and utility framework usage

## Advanced Features

### Version Hint Extraction

The analyzer attempts to extract version information from:

- Inline comments: `# pandas==1.3.0`
- Requirements patterns: `pandas>=1.2.0`
- Version comments: `# pandas version 1.3.0`
- Shorthand notation: `# pandas v1.3`

### Usage Pattern Recognition

Sophisticated pattern matching for:

- **Function Calls**: `module.function()` patterns
- **Class Instantiation**: `Class()` patterns
- **Attribute Access**: `module.attribute` patterns
- **Method Chaining**: `obj.method().method()` patterns

### Import Organization Analysis

Comprehensive PEP 8 compliance checking:

- **Order Validation**: Standard → Third-party → Local
- **Spacing Analysis**: Appropriate gaps between import groups
- **Style Consistency**: Import statement formatting
- **Grouping Logic**: Related imports grouped together

## Error Handling

The analyzer implements robust error handling:

1. **Malformed Imports**: Gracefully handles parsing errors
2. **Missing Modules**: Continues analysis with unknown modules
3. **Content Analysis**: Handles regex pattern matching errors
4. **Version Extraction**: Falls back gracefully when version hints fail

## Performance Considerations

Optimized for large script analysis:

- **Regex Compilation**: Pre-compiles frequently used patterns
- **Content Scanning**: Efficient single-pass content analysis
- **Module Databases**: Fast lookup for module categorization
- **Pattern Caching**: Caches usage pattern results

## Testing and Validation

The analyzer supports comprehensive testing:

- **Mock Imports**: Can analyze synthetic import lists
- **Pattern Testing**: Validates usage pattern detection
- **Organization Testing**: Verifies PEP 8 compliance checking
- **Framework Testing**: Tests framework requirement extraction

## Future Enhancements

Potential improvements for the analyzer:

1. **Dynamic Analysis**: Runtime import usage tracking
2. **Dependency Graphs**: Visual dependency relationship mapping
3. **Version Compatibility**: Check version compatibility across imports
4. **Performance Metrics**: Track import analysis performance
5. **Custom Rules**: Configurable import validation rules

## Conclusion

The ImportAnalyzer provides comprehensive static analysis of Python import statements, supporting the alignment validation framework with detailed insights into module usage, framework requirements, and code organization. Its multi-dimensional analysis approach ensures scripts meet project standards while providing actionable feedback for improvement.

The analyzer serves as a critical component in maintaining code quality and consistency across the validation framework, enabling automated detection of import-related issues and compliance violations.
