---
tags:
  - test
  - validation
  - alignment
  - script
  - contract
keywords:
  - script contract alignment
  - level 1 validation
  - script analysis
  - contract validation
  - path alignment
  - environment variables
topics:
  - alignment validation
  - script validation
  - contract compliance
  - static analysis
language: python
date of note: 2025-08-18
---

# Script ↔ Contract Alignment Tester

The Script ↔ Contract Alignment Tester validates Level 1 alignment between processing scripts and their corresponding script contracts. It ensures that scripts use paths, environment variables, and arguments exactly as declared in their contracts.

## Overview

The `ScriptContractAlignmentTester` performs comprehensive validation of the alignment between processing scripts and their contracts. This is the foundation level of the four-tier alignment validation system, ensuring that the basic contract between scripts and their specifications is maintained.

## Purpose

Level 1 validation serves as the foundation for all higher-level validations by ensuring:

1. **Path Usage Alignment**: Scripts use file paths as declared in contracts
2. **Environment Variable Compliance**: Scripts access environment variables as specified
3. **Argument Alignment**: Script arguments match contract expectations
4. **File Operation Validation**: File operations align with declared inputs/outputs
5. **Testability Compliance**: Scripts follow testability patterns for validation

## Architecture

### Core Components

1. **Script Analyzer**: Static analysis of Python scripts using AST parsing
2. **Contract Loader**: Loading and parsing of script contracts
3. **File Resolver**: Flexible file discovery and path resolution
4. **Testability Validator**: Validation of script testability patterns
5. **Framework Detector**: Detection of ML frameworks and patterns

### Validation Flow

1. **File Discovery**: Locate script and corresponding contract files
2. **Contract Loading**: Load and parse contract specifications
3. **Script Analysis**: Perform static analysis of script code
4. **Alignment Validation**: Compare script usage with contract declarations
5. **Enhancement**: Apply step type-specific validation enhancements
6. **Result Compilation**: Compile validation results and issues

## Class Interface

### Constructor

```python
def __init__(self, scripts_dir: str, contracts_dir: str, builders_dir: Optional[str] = None):
    """
    Initialize the script-contract alignment tester.
    
    Args:
        scripts_dir: Directory containing processing scripts
        contracts_dir: Directory containing script contracts
        builders_dir: Optional directory containing step builders for enhanced validation
    """
```

### Key Methods

#### validate_all_scripts()

Validates alignment for all scripts or specified target scripts.

```python
def validate_all_scripts(self, target_scripts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Validate alignment for all scripts or specified target scripts.
    
    Args:
        target_scripts: Specific scripts to validate (None for all)
        
    Returns:
        Dictionary mapping script names to validation results
    """
```

**Usage Example**:
```python
tester = ScriptContractAlignmentTester(
    scripts_dir="src/cursus/steps/scripts",
    contracts_dir="src/cursus/steps/contracts"
)

# Validate all scripts
results = tester.validate_all_scripts()

# Validate specific scripts
results = tester.validate_all_scripts(target_scripts=['tabular_preprocessing'])
```

#### validate_script()

Validates alignment for a specific script.

```python
def validate_script(self, script_name: str) -> Dict[str, Any]:
    """
    Validate alignment for a specific script.
    
    Args:
        script_name: Name of the script to validate
        
    Returns:
        Validation result dictionary
    """
```

**Return Structure**:
```python
{
    'passed': bool,                    # Overall validation result
    'issues': [                        # List of validation issues
        {
            'severity': 'CRITICAL'|'ERROR'|'WARNING'|'INFO',
            'category': str,           # Issue category
            'message': str,            # Human-readable message
            'details': dict,           # Additional details
            'recommendation': str      # Suggested fix
        }
    ],
    'script_analysis': dict,           # Script analysis results
    'contract': dict                   # Contract specifications
}
```

## Validation Categories

### Path Usage Validation

Ensures scripts use file paths as declared in contracts:

```python
# Contract declaration
contract = {
    'inputs': {
        'training_data': {'path': '/opt/ml/input/data/training/'},
        'validation_data': {'path': '/opt/ml/input/data/validation/'}
    },
    'outputs': {
        'model_artifacts': {'path': '/opt/ml/model/'},
        'evaluation_results': {'path': '/opt/ml/output/'}
    }
}

# Script usage validation
# ✅ Correct usage
with open('/opt/ml/input/data/training/data.csv', 'r') as f:
    training_data = f.read()

# ❌ Incorrect usage (not in contract)
with open('/tmp/data.csv', 'r') as f:  # Path not declared in contract
    data = f.read()
```

**Validation Process**:
1. Extract all file paths from script using AST analysis
2. Normalize paths for comparison
3. Check each path against contract declarations
4. Report undeclared path usage as violations

### Environment Variable Validation

Validates that scripts access environment variables as specified in contracts:

```python
# Contract declaration
contract = {
    'environment_variables': {
        'required': ['SM_MODEL_DIR', 'SM_CHANNEL_TRAINING'],
        'optional': {'SM_CHANNEL_VALIDATION': '/opt/ml/input/data/validation'}
    }
}

# Script usage validation
# ✅ Correct usage
model_dir = os.environ['SM_MODEL_DIR']  # Required variable
training_dir = os.environ.get('SM_CHANNEL_TRAINING')  # Required variable

# ❌ Incorrect usage
undeclared_var = os.environ['CUSTOM_VAR']  # Not declared in contract
```

**Validation Process**:
1. Extract environment variable access from script
2. Check required variables are accessed
3. Validate optional variables are handled correctly
4. Report undeclared environment variable usage

### Argument Validation

Ensures script arguments align with contract expectations:

```python
# Contract declaration
contract = {
    'arguments': {
        'model_type': {'default': 'xgboost', 'required': False},
        'learning_rate': {'default': None, 'required': True},
        'max_depth': {'default': 6, 'required': False}
    }
}

# Script usage validation
# ✅ Correct usage
parser.add_argument('--model-type', default='xgboost')
parser.add_argument('--learning-rate', required=True)
parser.add_argument('--max-depth', type=int, default=6)

# ❌ Incorrect usage
parser.add_argument('--undeclared-arg')  # Not in contract
```

### File Operation Validation

Validates that file operations align with declared inputs and outputs:

```python
# Validation checks
1. Read operations use declared input paths
2. Write operations use declared output paths
3. File creation aligns with output specifications
4. Directory access matches contract declarations
```

## File Discovery System

### Hybrid File Resolution

The tester uses a hybrid approach for finding contract files:

1. **Entry Point Mapping**: Authoritative mapping from contract entry_point values
2. **FlexibleFileResolver**: Pattern-based file discovery as fallback
3. **Naming Convention**: Standard naming patterns as final fallback

```python
def _find_contract_file_hybrid(self, script_name: str) -> Optional[str]:
    # Method 1: Entry point mapping (authoritative)
    if script_filename in self._entry_point_to_contract:
        contract_path = self.contracts_dir / self._entry_point_to_contract[script_filename]
        if contract_path.exists():
            return str(contract_path)
    
    # Method 2: FlexibleFileResolver (pattern matching)
    flexible_path = self.file_resolver.find_contract_file(script_name)
    if flexible_path and Path(flexible_path).exists():
        return flexible_path
    
    # Method 3: Naming convention (final fallback)
    conventional_path = self.contracts_dir / f"{script_name}_contract.py"
    if conventional_path.exists():
        return str(conventional_path)
    
    return None
```

### Entry Point Mapping

The tester builds a mapping from contract entry_point values to contract files:

```python
def _build_entry_point_mapping(self) -> Dict[str, str]:
    """Build mapping from entry_point values to contract file names."""
    mapping = {}
    
    # Scan all contract files
    for contract_file in self.contracts_dir.glob("*_contract.py"):
        try:
            entry_point = self._extract_entry_point_from_contract(contract_file)
            if entry_point:
                mapping[entry_point] = contract_file.name
        except Exception:
            continue  # Skip contracts that can't be loaded
    
    return mapping
```

## Contract Loading

### Python Contract Loading

The tester loads contracts from Python modules with proper import handling:

```python
def _load_python_contract(self, contract_path: Path, script_name: str) -> Dict[str, Any]:
    """Load contract from Python module and convert to dictionary format."""
    
    # Handle relative imports by adding paths to sys.path
    project_root = str(contract_path.parent.parent.parent.parent)
    src_root = str(contract_path.parent.parent.parent)
    
    # Load module with proper package context
    spec = importlib.util.spec_from_file_location(f"{script_name}_contract", contract_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = 'cursus.steps.contracts'
    spec.loader.exec_module(module)
    
    # Find contract object using multiple naming patterns
    possible_names = [
        f"{script_name.upper()}_CONTRACT",
        f"{script_name}_CONTRACT",
        "CONTRACT"
    ]
    
    # Convert ScriptContract object to dictionary format
    contract_dict = {
        'entry_point': contract_obj.entry_point,
        'inputs': {...},      # From expected_input_paths
        'outputs': {...},     # From expected_output_paths
        'arguments': {...},   # From expected_arguments
        'environment_variables': {...}  # From required/optional env vars
    }
```

### Contract Format Conversion

Contracts are converted from object format to standardized dictionary format:

```python
# ScriptContract object attributes → Dictionary format
{
    'entry_point': contract_obj.entry_point,
    'inputs': {
        logical_name: {'path': path}
        for logical_name, path in contract_obj.expected_input_paths.items()
    },
    'outputs': {
        logical_name: {'path': path}
        for logical_name, path in contract_obj.expected_output_paths.items()
    },
    'arguments': {
        arg_name: {'default': default_value, 'required': default_value is None}
        for arg_name, default_value in contract_obj.expected_arguments.items()
    },
    'environment_variables': {
        'required': contract_obj.required_env_vars,
        'optional': contract_obj.optional_env_vars
    }
}
```

## Step Type Enhancement System

### Phase 2 Enhancement: Step Type-Specific Validation

The tester includes step type-specific validation enhancements:

```python
def _enhance_with_step_type_validation(self, script_name: str, analysis: Dict, contract: Dict) -> List[Dict]:
    """Add step type-specific validation to existing results."""
    
    # Detect step type from registry
    step_type = detect_step_type_from_registry(script_name)
    
    # Detect framework from imports
    framework = detect_framework_from_imports(analysis.get('imports', []))
    
    # Apply step type-specific validation
    if step_type == "Training":
        return self._validate_training_specific(script_name, analysis, contract, framework)
    elif step_type == "Processing":
        return self._validate_processing_framework_specific(script_name, analysis, contract, framework)
    
    return []
```

### Training-Specific Validation

For training scripts, additional validation includes:

```python
def _validate_training_specific(self, script_name: str, analysis: Dict, contract: Dict, framework: str) -> List[Dict]:
    """Add training-specific validation using existing patterns."""
    
    # Detect training patterns in script content
    training_patterns = detect_training_patterns(script_content)
    
    issues = []
    
    # Check for training loop patterns
    if not training_patterns.get('training_loop_patterns'):
        issues.append({
            'severity': 'WARNING',
            'category': 'training_pattern_missing',
            'message': 'Training script should contain model training logic',
            'recommendation': 'Add model training logic such as model.fit() or xgb.train()'
        })
    
    # Check for model saving patterns
    if not training_patterns.get('model_saving_patterns'):
        issues.append({
            'severity': 'WARNING',
            'category': 'training_model_saving_missing',
            'message': 'Training script should save model artifacts',
            'recommendation': 'Add model saving to /opt/ml/model/ directory'
        })
    
    return issues
```

### XGBoost-Specific Validation

For XGBoost training scripts:

```python
def _validate_xgboost_training_patterns(self, script_name: str, script_content: str) -> List[Dict]:
    """Validate XGBoost-specific training patterns."""
    
    xgb_patterns = detect_xgboost_patterns(script_content)
    issues = []
    
    # Check for XGBoost imports
    if not xgb_patterns.get('xgboost_imports'):
        issues.append({
            'severity': 'ERROR',
            'category': 'xgboost_import_missing',
            'message': 'XGBoost training script should import xgboost',
            'recommendation': 'Add XGBoost import: import xgboost as xgb'
        })
    
    # Check for DMatrix usage
    if not xgb_patterns.get('dmatrix_patterns'):
        issues.append({
            'severity': 'WARNING',
            'category': 'xgboost_dmatrix_missing',
            'message': 'XGBoost training should use DMatrix for data handling',
            'recommendation': 'Use xgb.DMatrix() for efficient data handling'
        })
    
    return issues
```

## Static Analysis Integration

### Script Analyzer

The tester uses the `ScriptAnalyzer` for comprehensive script analysis:

```python
analyzer = ScriptAnalyzer(str(script_path))
analysis = analyzer.get_all_analysis_results()

# Analysis results include:
{
    'imports': [...],           # Import statements
    'file_paths': [...],        # File path usage
    'env_vars': [...],          # Environment variable access
    'arguments': [...],         # Command line arguments
    'functions': [...],         # Function definitions
    'classes': [...]            # Class definitions
}
```

### AST-Based Analysis

Static analysis uses Python's AST module for accurate code parsing:

- **Import Analysis**: Extract all import statements and dependencies
- **Path Extraction**: Find all file path references in the code
- **Environment Variable Detection**: Identify os.environ and os.getenv usage
- **Argument Parsing**: Analyze argparse usage and argument definitions
- **Function Analysis**: Extract function signatures and calls

## Validation Rules

### Path Usage Rules

1. **Declared Paths Only**: Scripts should only use paths declared in contracts
2. **Logical Name Resolution**: Paths should resolve to logical names in contracts
3. **Input/Output Separation**: Input paths should only be read, output paths only written
4. **Path Normalization**: Paths are normalized for consistent comparison

### Environment Variable Rules

1. **Required Variables**: All required environment variables must be accessed
2. **Optional Variables**: Optional variables should be handled with defaults
3. **Undeclared Variables**: Scripts should not access undeclared environment variables
4. **Error Handling**: Environment variable access should include error handling

### Argument Rules

1. **Contract Alignment**: Script arguments must match contract declarations
2. **Default Values**: Default values should align between script and contract
3. **Required Arguments**: Required arguments must be properly validated
4. **Type Consistency**: Argument types should be consistent

### File Operation Rules

1. **Input Operations**: Read operations should use declared input paths
2. **Output Operations**: Write operations should use declared output paths
3. **Directory Creation**: Output directories should be created as needed
4. **Error Handling**: File operations should include proper error handling

## Error Categories and Severity

### Critical Errors
- **Missing Files**: Script or contract files not found
- **Parse Errors**: Syntax errors preventing analysis
- **Import Errors**: Critical import failures

### Errors
- **Undeclared Path Usage**: Using paths not declared in contract
- **Missing Required Variables**: Not accessing required environment variables
- **Contract Violations**: Direct violations of contract specifications

### Warnings
- **Missing Patterns**: Missing recommended patterns (training loops, model saving)
- **Framework Issues**: Framework-specific pattern violations
- **Testability Issues**: Violations of testability patterns

### Info
- **Framework Detection**: Information about detected frameworks
- **Pattern Recognition**: Information about recognized code patterns
- **Enhancement Context**: Step type and framework context information

## Integration Points

### With Unified Alignment Tester

```python
class UnifiedAlignmentTester:
    def _run_level1_validation(self, target_scripts):
        """Run Level 1: Script ↔ Contract alignment validation."""
        results = self.level1_tester.validate_all_scripts(target_scripts)
        
        for script_name, result in results.items():
            validation_result = ValidationResult(
                test_name=f"script_contract_{script_name}",
                passed=result.get('passed', False),
                details=result
            )
            
            # Convert issues to AlignmentIssue objects
            for issue in result.get('issues', []):
                alignment_issue = create_alignment_issue(...)
                validation_result.add_issue(alignment_issue)
```

### With Simple Integration API

```python
# Level 1 validation is part of integration validation
from cursus.validation import validate_integration

results = validate_integration(['tabular_preprocessing'])
# This internally uses ScriptContractAlignmentTester for Level 1 validation
```

## Usage Examples

### Basic Script Validation

```python
from cursus.validation.alignment.script_contract_alignment import ScriptContractAlignmentTester

# Initialize tester
tester = ScriptContractAlignmentTester(
    scripts_dir="src/cursus/steps/scripts",
    contracts_dir="src/cursus/steps/contracts"
)

# Validate specific script
result = tester.validate_script('tabular_preprocessing')

if result['passed']:
    print("✅ Script-contract alignment validated")
else:
    print("❌ Alignment issues found:")
    for issue in result['issues']:
        print(f"  {issue['severity']}: {issue['message']}")
```

### Batch Validation

```python
# Validate multiple scripts
scripts_to_validate = ['tabular_preprocessing', 'xgboost_training', 'model_eval']
results = tester.validate_all_scripts(target_scripts=scripts_to_validate)

# Check results
for script_name, result in results.items():
    status = "✅" if result['passed'] else "❌"
    print(f"{status} {script_name}: {len(result['issues'])} issues")
```

### Enhanced Validation with Builders

```python
# Include builder directory for enhanced validation
tester = ScriptContractAlignmentTester(
    scripts_dir="src/cursus/steps/scripts",
    contracts_dir="src/cursus/steps/contracts",
    builders_dir="src/cursus/steps/builders"  # Enhanced validation
)

result = tester.validate_script('tabular_preprocessing')
```

## Performance Features

### File Resolution Optimization

- **Entry Point Mapping**: Fast lookup using pre-built mapping
- **Flexible Fallback**: Pattern-based discovery when mapping fails
- **Path Caching**: Cache resolved paths for performance

### Analysis Caching

- **AST Caching**: Cache parsed AST trees for reuse
- **Analysis Results**: Cache analysis results for repeated validations
- **Contract Caching**: Cache loaded contracts to avoid re-parsing

### Error Recovery

- **Graceful Degradation**: Continue validation even with partial failures
- **Error Isolation**: Isolate errors to specific validation aspects
- **Detailed Reporting**: Provide detailed error information for debugging

## Best Practices

### For Script Developers

1. **Follow Contract Declarations**: Use only paths and variables declared in contracts
2. **Handle Errors Gracefully**: Include proper error handling for file operations
3. **Use Logical Names**: Reference paths using logical names from contracts
4. **Test Regularly**: Run Level 1 validation during development
5. **Update Contracts**: Keep contracts synchronized with script changes

### For Contract Authors

1. **Complete Declarations**: Declare all paths and variables used by scripts
2. **Accurate Specifications**: Ensure contract specifications match actual usage
3. **Clear Documentation**: Provide clear descriptions and recommendations
4. **Version Consistency**: Keep contracts synchronized with script versions
5. **Validation Testing**: Test contracts with actual script implementations

## Common Issues and Solutions

### Missing Contract Files

**Issue**: Contract file not found for script
**Solution**: 
1. Check naming conventions (`script_name_contract.py`)
2. Verify entry_point values in existing contracts
3. Create missing contract files

### Path Alignment Issues

**Issue**: Script uses undeclared paths
**Solution**:
1. Add path declarations to contract
2. Update script to use declared paths
3. Use logical name resolution

### Environment Variable Issues

**Issue**: Script accesses undeclared environment variables
**Solution**:
1. Add variable declarations to contract
2. Update script to use declared variables
3. Add proper error handling for missing variables

### Framework Pattern Issues

**Issue**: Missing framework-specific patterns
**Solution**:
1. Add required framework imports
2. Implement framework-specific patterns
3. Follow framework best practices

The Script ↔ Contract Alignment Tester provides the foundation for the entire alignment validation system, ensuring that the basic contract between scripts and their specifications is maintained and enforced.
