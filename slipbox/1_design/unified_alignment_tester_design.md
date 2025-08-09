---
tags:
  - design
  - validation
  - alignment
  - testing
  - architecture
keywords:
  - alignment validation
  - script contracts
  - step specifications
  - builder configuration
  - dependency resolution
  - unified tester
  - validation framework
topics:
  - alignment validation
  - testing architecture
  - validation framework
  - pipeline compliance
language: python
date of note: 2025-08-08
---

# Unified Alignment Tester Design

## Overview

The Unified Alignment Tester is a comprehensive validation system that systematically tests all four alignment principles defined in the alignment rules. It provides automated validation of the relationships between scripts, contracts, specifications, and builders to ensure proper architectural compliance.

## Motivation

The current validation system focuses primarily on step builder implementation correctness through the 4-level tester system. However, there's a need for systematic validation of the alignment rules that govern the relationships between all pipeline components:

1. **Script ↔ Contract Alignment**: Scripts must use exactly the paths defined in their contracts
2. **Contract ↔ Specification Alignment**: Logical names must match between contracts and specifications
3. **Specification ↔ Dependencies Alignment**: Dependencies must match upstream step outputs
4. **Builder ↔ Configuration Alignment**: Builders must handle configuration parameters correctly

The Unified Alignment Tester fills this gap by providing dedicated validation for each alignment level.

## Architecture

### Component Structure

```
src/cursus/validation/alignment/
├── __init__.py
├── unified_alignment_tester.py     # Main orchestrator
├── script_contract_alignment.py    # Level 1: Script ↔ Contract
├── contract_spec_alignment.py      # Level 2: Contract ↔ Specification  
├── spec_dependency_alignment.py    # Level 3: Specification ↔ Dependencies
├── builder_config_alignment.py     # Level 4: Builder ↔ Configuration
├── alignment_reporter.py           # Results reporting and analysis
├── alignment_utils.py              # Common utilities
└── static_analysis/                # Static code analysis tools
    ├── __init__.py
    ├── script_analyzer.py          # Script source code analysis
    ├── path_extractor.py           # Extract path usage from scripts
    └── import_analyzer.py          # Analyze script imports
```

### Core Classes

#### UnifiedAlignmentTester

```python
class UnifiedAlignmentTester:
    """
    Main orchestrator for alignment validation across all four levels.
    
    Coordinates validation between scripts, contracts, specifications, 
    and builders to ensure proper architectural alignment.
    """
    
    def __init__(self, 
                 script_path: str,
                 contract: ScriptContract,
                 specification: StepSpecification,
                 builder_class: Type[StepBuilderBase],
                 config: ConfigBase,
                 available_steps: List[StepSpecification] = None,
                 spec_registry: SpecificationRegistry = None):
        """
        Initialize with all components needed for alignment validation.
        
        Args:
            script_path: Path to the processing script
            contract: Script contract defining expected interface
            specification: Step specification defining pipeline integration
            builder_class: Step builder class for SageMaker integration
            config: Configuration instance for the builder
            available_steps: List of available pipeline steps for dependency resolution
            spec_registry: Registry for specification lookup
        """
        
    def run_full_alignment_validation(self) -> AlignmentReport:
        """
        Run comprehensive alignment validation across all four levels.
        
        Returns:
            AlignmentReport containing results from all alignment levels
        """
        
    def run_level_validation(self, level: int) -> Dict[str, Any]:
        """
        Run validation for a specific alignment level.
        
        Args:
            level: Alignment level (1-4)
            
        Returns:
            Dictionary containing validation results for the specified level
        """
        
    def validate_complete_pipeline_step(self) -> ComprehensiveReport:
        """
        Run both alignment validation and builder implementation validation.
        
        Combines alignment testing with the existing 4-level builder testing
        to provide complete step validation.
        
        Returns:
            ComprehensiveReport combining alignment and implementation results
        """
```

## Alignment Level Implementations

### Level 1: Script ↔ Contract Alignment

```python
class ScriptContractAlignmentTester:
    """
    Validates that scripts implement their contracts correctly.
    
    Based on Alignment Rule 1: Scripts must use exactly the paths defined 
    in their Script Contract. Environment variable names, input/output 
    directory structures, and file patterns must match the contract.
    """
    
    def __init__(self, script_path: str, contract: ScriptContract):
        self.script_path = script_path
        self.contract = contract
        self.script_analyzer = ScriptAnalyzer(script_path)
        
    def validate_path_usage(self) -> ValidationResult:
        """
        Validate script uses contract-defined paths exactly.
        
        Checks:
        - Script references all expected input paths
        - Script writes to all expected output paths
        - No hardcoded paths outside of contract
        - Path construction matches contract expectations
        """
        
    def validate_environment_variables(self) -> ValidationResult:
        """
        Validate script accesses declared environment variables.
        
        Checks:
        - Script accesses all required environment variables
        - Script only accesses declared environment variables
        - Optional environment variables have proper defaults
        - Environment variable usage patterns are correct
        """
        
    def validate_argument_usage(self) -> ValidationResult:
        """
        Validate script implements declared command-line arguments.
        
        Checks:
        - Script parser defines all expected arguments
        - Argument names match contract exactly (kebab-case)
        - Required vs optional arguments match contract
        - Argument types and choices match expectations
        """
        
    def validate_framework_imports(self) -> ValidationResult:
        """
        Validate script imports match framework requirements.
        
        Checks:
        - All required frameworks are imported
        - Import patterns are correct
        - Version compatibility (where detectable)
        - No conflicting framework usage
        """
        
    def validate_entry_point_consistency(self) -> ValidationResult:
        """
        Validate script filename matches contract entry point.
        
        Checks:
        - Script filename matches contract.entry_point
        - Script has proper main() function or __main__ block
        - Script is executable as specified in contract
        """
```

### Level 2: Contract ↔ Specification Alignment

```python
class ContractSpecificationAlignmentTester:
    """
    Validates alignment between script contracts and step specifications.
    
    Based on Alignment Rule 2: Logical names in the Script Contract 
    (expected_input_paths, expected_output_paths) must match dependency 
    names in the Step Specification.
    """
    
    def __init__(self, contract: ScriptContract, specification: StepSpecification):
        self.contract = contract
        self.specification = specification
        
    def validate_logical_name_alignment(self) -> ValidationResult:
        """
        Validate logical names match between contract and specification.
        
        Checks:
        - All contract input logical names have corresponding dependencies
        - All contract output logical names have corresponding outputs
        - No orphaned logical names in either contract or specification
        - Naming consistency and conventions
        """
        
    def validate_property_path_alignment(self) -> ValidationResult:
        """
        Validate OutputSpec property paths correspond to contract outputs.
        
        Checks:
        - Each OutputSpec has a corresponding contract output path
        - Property paths are valid for the step type
        - Property path structure matches SageMaker conventions
        - No missing or extra property paths
        """
        
    def validate_dependency_coverage(self) -> ValidationResult:
        """
        Validate all contract inputs have corresponding dependencies.
        
        Checks:
        - Every expected_input_path has a matching DependencySpec
        - DependencySpec logical names match contract keys
        - No missing dependencies for contract inputs
        - Dependency specifications are complete
        """
        
    def validate_contract_specification_consistency(self) -> ValidationResult:
        """
        Validate overall consistency between contract and specification.
        
        Checks:
        - Step type consistency
        - Framework requirements alignment
        - Description and metadata consistency
        - Version compatibility
        """
```

### Level 3: Specification ↔ Dependencies Alignment

```python
class SpecificationDependencyAlignmentTester:
    """
    Validates alignment between step specifications and pipeline dependencies.
    
    Based on Alignment Rule 3: Dependencies declared in the Step Specification 
    must match upstream step outputs by logical name or alias.
    """
    
    def __init__(self, 
                 specification: StepSpecification,
                 available_steps: List[StepSpecification],
                 spec_registry: SpecificationRegistry = None):
        self.specification = specification
        self.available_steps = available_steps
        self.spec_registry = spec_registry
        self.dependency_resolver = UnifiedDependencyResolver(spec_registry)
        
    def validate_dependency_resolution(self) -> ValidationResult:
        """
        Validate dependencies can be resolved from available steps.
        
        Checks:
        - All dependencies can find compatible sources
        - Dependency resolution produces valid matches
        - No circular dependencies
        - Resolution confidence scores are acceptable
        """
        
    def validate_compatible_sources(self) -> ValidationResult:
        """
        Validate compatible_sources lists are accurate and complete.
        
        Checks:
        - All listed compatible sources actually exist
        - All actual compatible sources are listed
        - Compatible source specifications are valid
        - No missing or extra compatible sources
        """
        
    def validate_output_property_paths(self) -> ValidationResult:
        """
        Validate property paths exist in upstream step outputs.
        
        Checks:
        - Property paths are valid for upstream step types
        - Property path structure matches SageMaker step definitions
        - Property paths resolve to actual step outputs
        - No invalid or unreachable property paths
        """
        
    def validate_dependency_graph_consistency(self) -> ValidationResult:
        """
        Validate dependency graph is consistent and acyclic.
        
        Checks:
        - No circular dependencies in the graph
        - All dependencies form a valid DAG
        - Dependency ordering is consistent
        - No orphaned or unreachable dependencies
        """
```

### Level 4: Builder ↔ Configuration Alignment

```python
class BuilderConfigurationAlignmentTester:
    """
    Validates alignment between step builders and their configurations.
    
    Based on Alignment Rule 4: Step Builders must pass configuration 
    parameters to SageMaker components according to the config class. 
    Environment variables set in the builder must cover all required_env_vars.
    """
    
    def __init__(self, 
                 builder_class: Type[StepBuilderBase],
                 config: ConfigBase,
                 contract: ScriptContract):
        self.builder_class = builder_class
        self.config = config
        self.contract = contract
        
    def validate_environment_variable_coverage(self) -> ValidationResult:
        """
        Validate builder sets all required environment variables from contract.
        
        Checks:
        - Builder._get_processor_env_vars() covers all contract.required_env_vars
        - Environment variable values are properly configured
        - Optional environment variables are handled correctly
        - No missing or extra environment variables
        """
        
    def validate_configuration_parameter_usage(self) -> ValidationResult:
        """
        Validate builder uses configuration parameters correctly.
        
        Checks:
        - All config parameters are used appropriately
        - Configuration values are passed to SageMaker components
        - Parameter types and formats are correct
        - No unused or missing configuration parameters
        """
        
    def validate_sagemaker_component_creation(self) -> ValidationResult:
        """
        Validate builder creates SageMaker components according to config.
        
        Checks:
        - Processor/Estimator/Transformer creation uses config values
        - Component parameters match configuration
        - Resource allocation follows configuration
        - Framework versions and images are correct
        """
        
    def validate_input_output_handling(self) -> ValidationResult:
        """
        Validate builder handles inputs/outputs according to contract.
        
        Checks:
        - _get_inputs() creates inputs matching contract expectations
        - _get_outputs() creates outputs matching contract expectations
        - Input/output paths align with contract definitions
        - Processing input/output configurations are correct
        """
```

## Static Analysis Components

### ScriptAnalyzer

```python
class ScriptAnalyzer:
    """
    Analyzes Python script source code to extract usage patterns.
    
    Uses AST parsing to identify:
    - Path references and construction
    - Environment variable access
    - Import statements
    - Function definitions and calls
    - Argument parsing patterns
    """
    
    def __init__(self, script_path: str):
        self.script_path = script_path
        self.ast_tree = self._parse_script()
        
    def extract_path_references(self) -> List[PathReference]:
        """Extract all path references from the script."""
        
    def extract_env_var_access(self) -> List[EnvVarAccess]:
        """Extract all environment variable access patterns."""
        
    def extract_imports(self) -> List[ImportStatement]:
        """Extract all import statements."""
        
    def extract_argument_definitions(self) -> List[ArgumentDefinition]:
        """Extract command-line argument definitions."""
```

### PathExtractor

```python
class PathExtractor:
    """
    Specialized extractor for path usage patterns in scripts.
    
    Identifies:
    - Hardcoded path strings
    - Path construction using os.path.join()
    - Path manipulation using pathlib
    - File operations (open, read, write)
    """
    
    def extract_hardcoded_paths(self) -> List[str]:
        """Find all hardcoded path strings."""
        
    def extract_path_constructions(self) -> List[PathConstruction]:
        """Find all dynamic path construction patterns."""
        
    def extract_file_operations(self) -> List[FileOperation]:
        """Find all file read/write operations."""
```

## Reporting System

### AlignmentReport

```python
class AlignmentReport:
    """
    Comprehensive report of alignment validation results.
    
    Contains results from all four alignment levels with detailed
    analysis and actionable recommendations.
    """
    
    def __init__(self):
        self.level1_results: Dict[str, ValidationResult] = {}
        self.level2_results: Dict[str, ValidationResult] = {}
        self.level3_results: Dict[str, ValidationResult] = {}
        self.level4_results: Dict[str, ValidationResult] = {}
        self.summary: AlignmentSummary = None
        
    def generate_summary(self) -> AlignmentSummary:
        """Generate executive summary of alignment status."""
        
    def get_critical_issues(self) -> List[AlignmentIssue]:
        """Get all critical alignment issues requiring immediate attention."""
        
    def get_recommendations(self) -> List[AlignmentRecommendation]:
        """Get actionable recommendations for fixing alignment issues."""
        
    def export_to_json(self) -> str:
        """Export report to JSON format."""
        
    def export_to_html(self) -> str:
        """Export report to HTML format with visualizations."""
```

### ValidationResult

```python
class ValidationResult:
    """
    Result of a single validation check.
    
    Contains detailed information about what was tested,
    whether it passed, and specific issues found.
    """
    
    def __init__(self, 
                 test_name: str,
                 passed: bool,
                 issues: List[AlignmentIssue] = None,
                 details: Dict[str, Any] = None):
        self.test_name = test_name
        self.passed = passed
        self.issues = issues or []
        self.details = details or {}
        self.timestamp = datetime.now()
        
    def add_issue(self, issue: AlignmentIssue):
        """Add an alignment issue to this result."""
        
    def get_severity_level(self) -> SeverityLevel:
        """Get the highest severity level among all issues."""
```

## Integration with Existing Systems

### Relationship to 4-Level Builder Tester

The Unified Alignment Tester complements the existing 4-level builder validation system:

```python
class ComprehensiveStepValidator:
    """
    Combines alignment validation with builder implementation validation
    to provide complete step validation coverage.
    """
    
    def __init__(self, 
                 script_path: str,
                 contract: ScriptContract,
                 specification: StepSpecification,
                 builder_class: Type[StepBuilderBase],
                 config: ConfigBase):
        
        # Initialize alignment tester
        self.alignment_tester = UnifiedAlignmentTester(
            script_path, contract, specification, builder_class, config
        )
        
        # Initialize builder implementation tester
        self.builder_tester = UniversalStepBuilderTestFactory.create_tester(
            builder_class, config=config, spec=specification, contract=contract
        )
        
    def run_complete_validation(self) -> ComprehensiveReport:
        """
        Run both alignment and implementation validation.
        
        Returns:
            ComprehensiveReport combining both validation types
        """
        alignment_results = self.alignment_tester.run_full_alignment_validation()
        builder_results = self.builder_tester.run_all_tests()
        
        return ComprehensiveReport(alignment_results, builder_results)
```

### Usage in CI/CD Pipeline

```python
# Example CI/CD integration
def validate_pipeline_step_changes():
    """Validate all changed pipeline steps in CI/CD."""
    
    changed_steps = detect_changed_steps()
    
    for step_info in changed_steps:
        validator = ComprehensiveStepValidator(
            script_path=step_info.script_path,
            contract=step_info.contract,
            specification=step_info.specification,
            builder_class=step_info.builder_class,
            config=create_test_config(step_info)
        )
        
        report = validator.run_complete_validation()
        
        if report.has_critical_issues():
            raise ValidationError(f"Critical alignment issues in {step_info.name}")
        
        # Generate and store validation report
        store_validation_report(step_info.name, report)
```

## Usage Examples

### Basic Alignment Validation

```python
from cursus.validation.alignment import UnifiedAlignmentTester
from cursus.steps.contracts.tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT
from cursus.steps.specs.tabular_preprocess_spec import TABULAR_PREPROCESS_SPEC
from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder

# Create alignment tester
tester = UnifiedAlignmentTester(
    script_path="src/cursus/steps/scripts/tabular_preprocess.py",
    contract=TABULAR_PREPROCESS_CONTRACT,
    specification=TABULAR_PREPROCESS_SPEC,
    builder_class=TabularPreprocessingStepBuilder,
    config=preprocessing_config
)

# Run full alignment validation
report = tester.run_full_alignment_validation()

# Check results
if report.has_critical_issues():
    print("Critical alignment issues found:")
    for issue in report.get_critical_issues():
        print(f"- {issue.level}: {issue.message}")
else:
    print("All alignment checks passed!")
```

### Level-Specific Validation

```python
# Test only script-contract alignment
script_contract_results = tester.run_level_validation(1)

# Test only contract-specification alignment  
contract_spec_results = tester.run_level_validation(2)

# Test only specification-dependency alignment
spec_dependency_results = tester.run_level_validation(3)

# Test only builder-configuration alignment
builder_config_results = tester.run_level_validation(4)
```

### Batch Validation

```python
from cursus.validation.alignment import BatchAlignmentValidator

# Validate multiple steps at once
validator = BatchAlignmentValidator()

# Add steps to validate
validator.add_step("TabularPreprocessing", script_path, contract, spec, builder_class, config)
validator.add_step("XGBoostTraining", script_path2, contract2, spec2, builder_class2, config2)
validator.add_step("ModelEvaluation", script_path3, contract3, spec3, builder_class3, config3)

# Run batch validation
batch_report = validator.run_batch_validation()

# Generate summary report
batch_report.generate_summary_report("alignment_validation_report.html")
```

## Benefits

### For Developers

1. **Early Detection**: Catch alignment issues during development, not deployment
2. **Clear Guidance**: Specific feedback on what needs to be fixed and how
3. **Comprehensive Coverage**: All alignment rules tested systematically
4. **Integration Ready**: Works with existing development workflows

### For System Quality

1. **Architectural Compliance**: Ensures all components follow alignment rules
2. **Consistency**: Standardized validation across all pipeline steps
3. **Reliability**: Reduces runtime failures due to misalignment
4. **Maintainability**: Makes refactoring safer by validating relationships

### for CI/CD

1. **Automated Validation**: No manual alignment checking required
2. **Detailed Reports**: Actionable feedback for fixing issues
3. **Regression Prevention**: Catches alignment regressions automatically
4. **Quality Gates**: Can block deployments with critical alignment issues

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create base alignment tester classes
2. Implement static analysis components
3. Create reporting system
4. Add basic validation results structure

### Phase 2: Level 1 & 2 Implementation
1. Implement Script ↔ Contract alignment validation
2. Implement Contract ↔ Specification alignment validation
3. Add comprehensive test coverage
4. Create usage examples and documentation

### Phase 3: Level 3 & 4 Implementation
1. Implement Specification ↔ Dependencies alignment validation
2. Implement Builder ↔ Configuration alignment validation
3. Integrate with existing dependency resolver
4. Add integration with 4-level builder tester

### Phase 4: Advanced Features
1. Add batch validation capabilities
2. Create HTML reporting with visualizations
3. Add CI/CD integration helpers
4. Create performance optimizations

### Phase 5: Integration & Documentation
1. Integrate with existing validation infrastructure
2. Create comprehensive documentation
3. Add migration guides from existing tools
4. Create training materials and examples

## Future Enhancements

### Advanced Static Analysis
- **AST-based Analysis**: Deeper code analysis using Python AST
- **Control Flow Analysis**: Understand conditional path usage
- **Data Flow Analysis**: Track variable usage patterns
- **Cross-file Analysis**: Analyze imports and dependencies

### Machine Learning Integration
- **Pattern Recognition**: Learn common alignment patterns
- **Anomaly Detection**: Identify unusual alignment patterns
- **Predictive Analysis**: Predict likely alignment issues
- **Automated Fixes**: Suggest or apply automatic fixes

### Visualization and Reporting
- **Interactive Reports**: Web-based interactive validation reports
- **Dependency Graphs**: Visual representation of dependency relationships
- **Alignment Dashboards**: Real-time alignment status monitoring
- **Trend Analysis**: Track alignment quality over time

### Integration Enhancements
- **IDE Integration**: Real-time alignment checking in development environments
- **Git Hooks**: Pre-commit alignment validation
- **Pipeline Integration**: Integration with SageMaker pipeline execution
- **Monitoring Integration**: Runtime alignment monitoring

## Conclusion

The Unified Alignment Tester provides a comprehensive solution for validating the alignment rules that are critical to the pipeline architecture. By systematically testing all four alignment levels, it ensures that scripts, contracts, specifications, and builders work together correctly.

The design leverages existing infrastructure while adding focused alignment validation capabilities. It provides clear, actionable feedback to developers and integrates seamlessly with existing development workflows.

This system will significantly improve the reliability and maintainability of the pipeline architecture by catching alignment issues early and providing clear guidance for resolution.
