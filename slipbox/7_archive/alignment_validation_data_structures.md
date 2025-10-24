---
tags:
  - archive
  - design
  - data_structures
  - validation
  - alignment
  - production_ready
keywords:
  - validation data structures
  - alignment models
  - breakthrough implementations
  - production interfaces
topics:
  - data structure design
  - validation framework
  - system interfaces
language: python
date of note: 2025-08-12
---

# Alignment Validation Data Structures

## Related Documents
- **[Master Design](unified_alignment_tester_master_design.md)** - Complete system overview
- **[Architecture](unified_alignment_tester_architecture.md)** - Core architectural patterns
- **[SageMaker Step Type-Aware Unified Alignment Tester Design](sagemaker_step_type_aware_unified_alignment_tester_design.md)** - Step type-aware validation framework design
- **[Implementation Guide](unified_alignment_tester_implementation.md)** - Production implementation details
- **[Alignment Validation Visualization Integration Design](alignment_validation_visualization_integration_design.md)** - **VISUALIZATION FRAMEWORK** - Comprehensive design and implementation of the alignment validation visualization integration system, including scoring algorithms, chart generation infrastructure, and enhanced reporting capabilities that utilize the data structures documented here.
- **[Standardization Rules](../0_developer_guide/standardization_rules.md)** - **FOUNDATIONAL** - Comprehensive standardization rules that define the naming conventions, interface standards, and architectural constraints that these data structures implement validation for. The breakthrough data structures documented here enable enforcement of these standardization rules across all validation levels.
- **[Pain Points Analysis](../4_analysis/unified_alignment_tester_pain_points_analysis.md)** - Comprehensive analysis of validation challenges that drove the data structure design decisions documented here

## Overview

This document defines the critical data structures that emerged from the revolutionary breakthroughs achieved in August 2025. These data structures represent the foundation of the production-ready validation system, enabling 100% success rates across all four validation levels.

**Key Achievement**: The refactored system now achieves **100% validation success** across all 8 scripts, with robust script-to-contract name mapping and comprehensive error handling.

## Refactored Modular Data Structures (August 2025)

**BREAKING CHANGE**: The alignment validation system has been refactored into focused, single-responsibility modules to improve maintainability and extensibility.

### Module Organization

#### Core Models (`core_models.py`)
```python
from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class SeverityLevel(Enum):
    """Severity levels for alignment issues."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlignmentLevel(Enum):
    """Alignment validation levels."""
    SCRIPT_CONTRACT = 1
    CONTRACT_SPECIFICATION = 2
    SPECIFICATION_DEPENDENCY = 3
    BUILDER_CONFIGURATION = 4

class AlignmentIssue(BaseModel):
    """Base alignment issue with comprehensive context."""
    level: SeverityLevel
    category: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    recommendation: Optional[str] = None
    alignment_level: Optional[AlignmentLevel] = None

class StepTypeAwareAlignmentIssue(AlignmentIssue):
    """Extended alignment issue with step type context for training scripts."""
    step_type: Optional[str] = None              # Training, Processing, CreateModel
    framework_context: Optional[str] = None     # XGBoost, PyTorch, sklearn
    reference_examples: List[str] = Field(default_factory=list)
```

#### Script Analysis Models (`script_analysis_models.py`)
```python
class PathReference(BaseModel):
    """Path reference found in script analysis."""
    path: str
    line_number: int
    context: str
    is_hardcoded: bool = True
    construction_method: Optional[str] = None

class ImportStatement(BaseModel):
    """Import statement found in script analysis."""
    module_name: str
    import_alias: Optional[str]
    line_number: int
    is_from_import: bool = False
    imported_items: List[str] = Field(default_factory=list)

class ArgumentDefinition(BaseModel):
    """Command-line argument definition."""
    argument_name: str
    line_number: int
    is_required: bool = False
    has_default: bool = False
    default_value: Optional[Any] = None
    argument_type: Optional[str] = None
    choices: Optional[List[str]] = None
```

#### Dependency Classification (`dependency_classifier.py`)
```python
from enum import Enum

class DependencyPattern(Enum):
    """Types of dependency patterns for classification."""
    PIPELINE_DEPENDENCY = "pipeline"
    EXTERNAL_INPUT = "external"
    CONFIGURATION = "configuration"
    ENVIRONMENT = "environment"

class DependencyPatternClassifier:
    """Classify dependencies by pattern type for appropriate validation."""
    
    def classify_dependency(self, dependency_info: Dict[str, Any]) -> DependencyPattern:
        """Classify dependency pattern for appropriate validation."""
        # Intelligent classification logic to reduce false positives
        
    def should_validate_pipeline_resolution(self, pattern: DependencyPattern) -> bool:
        """Determine if dependency requires pipeline resolution validation."""
        return pattern == DependencyPattern.PIPELINE_DEPENDENCY
```

#### Step Type Detection (`step_type_detection.py`)
```python
def detect_step_type_from_registry(script_name: str) -> str:
    """Use existing step registry to determine SageMaker step type."""
    # Returns: Training, Processing, CreateModel, etc.

def detect_framework_from_imports(imports: List[ImportStatement]) -> Optional[str]:
    """Detect ML framework from import analysis."""
    # Returns: xgboost, pytorch, sklearn, etc.

def get_step_type_context(script_name: str, script_content: Optional[str] = None) -> dict:
    """Get comprehensive step type context for a script."""
    # Combines registry and pattern-based detection
```

#### File Resolution (`file_resolver.py`)
```python
class FlexibleFileResolver:
    """Dynamic file resolution with intelligent pattern matching."""
    
    def find_contract_file(self, script_name: str) -> Optional[str]:
        """Find contract file using multiple strategies."""
        
    def find_spec_file(self, script_name: str) -> Optional[str]:
        """Find specification file using dynamic discovery."""
        
    def find_builder_file(self, script_name: str) -> Optional[str]:
        """Find builder file using dynamic discovery."""
        
    def find_all_component_files(self, script_name: str) -> Dict[str, Optional[str]]:
        """Find all component files for a given script."""
```

#### Utility Functions (`utils.py`)
```python
def normalize_path(path: str) -> str:
    """Normalize a path for comparison purposes."""

def extract_logical_name_from_path(path: str) -> Optional[str]:
    """Extract logical name from a SageMaker path."""

def is_sagemaker_path(path: str) -> bool:
    """Check if a path is a SageMaker container path."""

def format_alignment_issue(issue: AlignmentIssue) -> str:
    """Format an alignment issue for display."""

def group_issues_by_severity(issues: List[AlignmentIssue]) -> Dict[SeverityLevel, List[AlignmentIssue]]:
    """Group alignment issues by severity level."""
```

#### Import Aggregator (`alignment_utils.py`)
```python
"""
Backward compatibility import aggregator.
Maintains existing interface while organizing code into focused modules.
"""

# Re-exports all public APIs from specialized modules
from .core_models import (
    SeverityLevel, AlignmentLevel, AlignmentIssue, StepTypeAwareAlignmentIssue,
    create_alignment_issue, create_step_type_aware_alignment_issue
)
from .script_analysis_models import (
    PathReference, EnvVarAccess, ImportStatement, ArgumentDefinition
)
from .dependency_classifier import (
    DependencyPattern, DependencyPatternClassifier
)
# ... and all other modules
```

## Level 1: Script ↔ Contract Data Structures

### ScriptAnalysis (Enhanced Static Analysis)
```python
@dataclass
class ScriptAnalysis:
    """Enhanced script analysis with comprehensive pattern detection."""
    
    script_path: str
    path_references: List[PathReference]           # All path references in script
    env_var_accesses: List[EnvVarAccess]          # Environment variable usage
    imports: List[ImportStatement]                 # Import analysis
    argument_definitions: List[ArgumentDefinition] # CLI argument definitions
    file_operations: List[FileOperation]          # File I/O operations
    
    def get_path_references_by_type(self, ref_type: str) -> List[PathReference]:
        """Get path references filtered by type."""
        return [ref for ref in self.path_references if ref.construction_method == ref_type]
        
    def get_env_var_accesses_by_method(self, method: str) -> List[EnvVarAccess]:
        """Get environment variable accesses by method."""
        return [access for access in self.env_var_accesses if access.access_method == method]

@dataclass
class PathReference:
    """Path reference found in script."""
    
    path: str                    # The path string
    line_number: int            # Line number where found
    context: str                # Surrounding code context
    is_hardcoded: bool          # Whether path is hardcoded
    construction_method: Optional[str]  # How path is constructed (os.path.join, etc.)

@dataclass
class EnvVarAccess:
    """Environment variable access pattern."""
    
    variable_name: str          # Environment variable name
    line_number: int           # Line number where accessed
    context: str               # Surrounding code context
    access_method: str         # How accessed (os.environ.get, etc.)
    has_default: bool          # Whether has default value
    default_value: Optional[str]  # Default value if any

@dataclass
class ArgumentDefinition:
    """CLI argument definition with convention handling."""
    
    argument_name: str         # Argument name
    line_number: int          # Line number where defined
    is_required: bool         # Whether argument is required
    has_default: bool         # Whether has default value
    default_value: Any        # Default value if any
    argument_type: str        # Argument type
    choices: Optional[List]   # Valid choices if any
    
    def matches_contract_argument(self, contract_arg: str) -> bool:
        """Check if this argument matches contract argument with hyphen-underscore conversion."""
        normalized_arg = self.argument_name.lstrip('-').replace('-', '_')
        normalized_contract = contract_arg.replace('-', '_')
        return normalized_arg == normalized_contract
```

### ContractValidation (Robust Loading and Validation)
```python
@dataclass
class ContractValidation:
    """Contract validation with robust loading strategies."""
    
    contract_path: Path
    contract_data: Dict[str, Any]          # Loaded contract data
    validation_issues: List[ValidationIssue]  # Issues found during validation
    load_strategy: str                     # Which loading strategy succeeded
    
    def validate_against_script(self, script_analysis: ScriptAnalysis) -> List[ValidationIssue]:
        """Validate contract against script analysis."""
        issues = []
        
        # Validate inputs/outputs alignment
        issues.extend(self._validate_io_alignment(script_analysis))
        
        # Validate environment variables
        issues.extend(self._validate_env_vars(script_analysis))
        
        # Validate arguments
        issues.extend(self._validate_arguments(script_analysis))
        
        return issues
        
    def _validate_io_alignment(self, script_analysis: ScriptAnalysis) -> List[ValidationIssue]:
        """Validate input/output path alignment."""
        issues = []
        
        # Check if script uses paths that align with contract
        contract_paths = set()
        if 'inputs' in self.contract_data:
            contract_paths.update(self.contract_data['inputs'].values())
        if 'outputs' in self.contract_data:
            contract_paths.update(self.contract_data['outputs'].values())
            
        # Validate script path references align with contract
        for path_ref in script_analysis.path_references:
            if path_ref.is_hardcoded and path_ref.path not in contract_paths:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    category="path_alignment",
                    message=f"Script uses hardcoded path not in contract: {path_ref.path}",
                    details={"path": path_ref.path, "line_number": path_ref.line_number},
                    recommendation="Use contract-defined paths or add path to contract"
                ))
                
        return issues
```

## Level 2: Contract ↔ Specification Data Structures

### SmartSpecificationSelection (Script-to-Contract Name Mapping)
```python
@dataclass
class SmartSpecificationSelection:
    """Smart specification selection with robust name mapping."""
    
    file_resolver: FlexibleFileResolver       # Handles script-to-contract mapping
    specification_loader: SpecificationLoader  # Loads specifications
    contract_loader: ContractLoader           # Loads contracts
    
    def resolve_contract_from_script(self, script_or_contract_name: str) -> Optional[str]:
        """Resolve script name to actual contract file path."""
        # Use FlexibleFileResolver to find contract file
        contract_file_path = self.file_resolver.find_contract_file(script_or_contract_name)
        
        if contract_file_path:
            # Extract actual contract name from file path
            # e.g., "xgboost_model_eval_contract.py" -> "xgboost_model_eval_contract"
            return Path(contract_file_path).stem
            
        return None
        
    def create_unified_specification(self, specifications: Dict[str, Any], 
                                   contract_name: str) -> UnifiedSpecificationModel:
        """Create unified specification model from multiple variants."""
        if len(specifications) == 1:
            # Single specification - use as primary
            primary_spec = next(iter(specifications.values()))
            return UnifiedSpecificationModel(
                primary_spec=primary_spec,
                variants={'generic': primary_spec},
                unified_dependencies=self._extract_dependencies(primary_spec),
                unified_outputs=self._extract_outputs(primary_spec),
                variant_count=1
            )
        else:
            # Multiple specifications - create unified model
            return self._create_multi_variant_unified_spec(specifications, contract_name)

@dataclass
class UnifiedSpecificationModel:
    """Unified specification model representing union of all variants."""
    
    primary_spec: Dict[str, Any]              # Primary specification to validate against
    variants: Dict[str, Dict[str, Any]]       # All specification variants
    unified_dependencies: Dict[str, Any]      # Union of all dependencies
    unified_outputs: Dict[str, Any]           # Union of all outputs
    dependency_sources: Dict[str, List[str]]  # Which variants provide each dependency
    output_sources: Dict[str, List[str]]      # Which variants provide each output
    variant_count: int                        # Number of variants
    
    def is_dependency_valid_for_any_variant(self, dep_name: str) -> bool:
        """Check if dependency is valid for any variant."""
        return dep_name in self.unified_dependencies
        
    def get_variants_using_dependency(self, dep_name: str) -> List[str]:
        """Get list of variants that use this dependency."""
        return self.dependency_sources.get(dep_name, [])
```

### FlexibleFileResolver (Robust File Discovery)
```python
@dataclass
class FlexibleFileResolver:
    """Flexible file resolver with multiple discovery strategies."""
    
    base_directories: Dict[str, str]          # Base directories for different file types
    naming_patterns: Dict[str, List[str]]     # Known naming patterns
    cache: Dict[str, str] = field(default_factory=dict)  # Resolution cache
    
    def find_contract_file(self, script_or_contract_name: str) -> Optional[str]:
        """Find contract file using multiple strategies."""
        
        # Check cache first
        cache_key = f"contract_{script_or_contract_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        contracts_dir = Path(self.base_directories.get('contracts', ''))
        if not contracts_dir.exists():
            return None
            
        # Strategy 1: Direct match (script_name + "_contract.py")
        direct_pattern = f"{script_or_contract_name}_contract.py"
        direct_path = contracts_dir / direct_pattern
        if direct_path.exists():
            result = str(direct_path)
            self.cache[cache_key] = result
            return result
            
        # Strategy 2: Exact name match (already a contract name)
        if script_or_contract_name.endswith('_contract'):
            exact_path = contracts_dir / f"{script_or_contract_name}.py"
            if exact_path.exists():
                result = str(exact_path)
                self.cache[cache_key] = result
                return result
                
        # Strategy 3: Fuzzy matching for edge cases
        for contract_file in contracts_dir.glob("*_contract.py"):
            contract_stem = contract_file.stem
            # Check if this contract might match the script
            if self._names_might_match(script_or_contract_name, contract_stem):
                result = str(contract_file)
                self.cache[cache_key] = result
                return result
                
        return None
        
    def _names_might_match(self, script_name: str, contract_stem: str) -> bool:
        """Check if script name might match contract using fuzzy logic."""
        # Remove common suffixes/prefixes
        script_base = script_name.replace('_script', '').replace('script_', '')
        contract_base = contract_stem.replace('_contract', '').replace('contract_', '')
        
        # Check for similarity (handle common variations)
        return (
            script_base == contract_base or
            script_base.replace('_', '') == contract_base.replace('_', '') or
            self._calculate_similarity(script_base, contract_base) > 0.8
        )
        
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using simple ratio."""
        if not str1 or not str2:
            return 0.0
        
        # Simple character-based similarity
        common_chars = set(str1.lower()) & set(str2.lower())
        total_chars = set(str1.lower()) | set(str2.lower())
        
        return len(common_chars) / len(total_chars) if total_chars else 0.0
```

## Level 3: Specification ↔ Dependencies Data Structures

### ProductionDependencyResolution (Threshold-Based Validation)
```python
@dataclass
class ProductionDependencyResolution:
    """Production dependency resolution with confidence scoring."""
    
    dependency_resolver: UnifiedDependencyResolver  # Production resolver
    confidence_threshold: float = 0.6               # Clear pass/fail threshold
    registry_manager: RegistryManager               # Production registry integration
    
    def resolve_dependencies_with_confidence(self, 
                                           specification: Dict[str, Any]) -> DependencyResolutionResult:
        """Resolve dependencies using production resolver with confidence scoring."""
        
        dependencies = specification.get('dependencies', [])
        resolved = {}
        failed = {}
        
        for dep in dependencies:
            logical_name = dep.get('logical_name')
            if not logical_name:
                continue
                
            try:
                # Use production dependency resolver
                resolution_result = self.dependency_resolver.resolve_dependency(
                    step_name=specification.get('step_type', 'Unknown'),
                    logical_name=logical_name,
                    dependency_spec=dep
                )
                
                if resolution_result and resolution_result.confidence >= self.confidence_threshold:
                    resolved[logical_name] = {
                        'target': resolution_result.target_reference,
                        'confidence': resolution_result.confidence,
                        'resolution_strategy': resolution_result.strategy,
                        'target_step': resolution_result.target_step
                    }
                else:
                    failed[logical_name] = {
                        'reason': 'Low confidence score' if resolution_result else 'No resolution found',
                        'confidence': resolution_result.confidence if resolution_result else 0.0,
                        'best_match': resolution_result.target_reference if resolution_result else None,
                        'threshold': self.confidence_threshold
                    }
                    
            except Exception as e:
                failed[logical_name] = {
                    'reason': f'Resolution failed: {str(e)}',
                    'confidence': 0.0,
                    'error': str(e)
                }
                
        return DependencyResolutionResult(
            resolved=resolved,
            failed=failed,
            total_dependencies=len(dependencies),
            success_rate=len(resolved) / len(dependencies) if dependencies else 1.0
        )

@dataclass
class DependencyResolutionResult:
    """Result of dependency resolution with detailed metrics."""
    
    resolved: Dict[str, Dict[str, Any]]       # Successfully resolved dependencies
    failed: Dict[str, Dict[str, Any]]         # Failed dependency resolutions
    total_dependencies: int                   # Total number of dependencies
    success_rate: float                       # Resolution success rate
    
    def get_validation_issues(self) -> List[ValidationIssue]:
        """Convert failed resolutions to validation issues."""
        issues = []
        
        for dep_name, failure_info in self.failed.items():
            severity = "ERROR" if failure_info.get('confidence', 0.0) < 0.3 else "WARNING"
            
            issues.append(ValidationIssue(
                severity=severity,
                category="dependency_resolution",
                message=f"Failed to resolve dependency '{dep_name}': {failure_info['reason']}",
                details=failure_info,
                recommendation=self._get_resolution_recommendation(dep_name, failure_info)
            ))
            
        return issues
        
    def _get_resolution_recommendation(self, dep_name: str, failure_info: Dict) -> str:
        """Get actionable recommendation for failed dependency resolution."""
        if failure_info.get('best_match'):
            return f"Consider using '{failure_info['best_match']}' or verify dependency specification"
        return f"Verify that dependency '{dep_name}' is correctly specified and available"
```

## Level 4: Builder ↔ Configuration Data Structures

### BuilderConfigurationAlignment (Hybrid Resolution)
```python
@dataclass
class BuilderConfigurationAlignment:
    """Builder-configuration alignment with hybrid file resolution."""
    
    builders_dir: Path
    configs_dir: Path
    file_resolver: FlexibleFileResolver      # Handles naming variations
    
    def validate_builder_config_alignment(self, builder_name: str) -> ValidationResult:
        """Validate alignment between builder and its configuration."""
        
        # Find builder file
        builder_path = self._find_builder_file(builder_name)
        if not builder_path:
            return ValidationResult(
                test_name=f"builder_config_{builder_name}",
                passed=False,
                details={'error': f'Builder file not found for {builder_name}'}
            )
            
        # Find corresponding config file
        config_path = self._find_config_file(builder_name)
        
        # Analyze builder
        builder_analysis = self._analyze_builder_file(builder_path)
        
        # Analyze config (if found)
        config_analysis = None
        if config_path:
            config_analysis = self._analyze_config_file(config_path)
            
        # Validate alignment
        issues = self._validate_alignment(builder_analysis, config_analysis, builder_name)
        
        return ValidationResult(
            test_name=f"builder_config_{builder_name}",
            passed=len([i for i in issues if i.severity in ['CRITICAL', 'ERROR']]) == 0,
            details={
                'builder_path': str(builder_path),
                'config_path': str(config_path) if config_path else None,
                'builder_analysis': builder_analysis,
                'config_analysis': config_analysis,
                'issues': [issue.to_dict() for issue in issues]
            }
        )
        
    def _find_builder_file(self, builder_name: str) -> Optional[Path]:
        """Find builder file using flexible resolution."""
        return self.file_resolver.find_builder_file(builder_name)
        
    def _find_config_file(self, builder_name: str) -> Optional[Path]:
        """Find config file using flexible resolution."""
        return self.file_resolver.find_config_file(builder_name)

@dataclass
class BuilderAnalysis:
    """Analysis of builder file structure and patterns."""
    
    config_accesses: List[ConfigAccess]       # How builder accesses config
    validation_calls: List[ValidationCall]   # Config validation calls
    default_assignments: List[DefaultAssignment]  # Default value assignments
    class_definitions: List[ClassDefinition] # Class definitions found
    method_definitions: List[MethodDefinition]  # Method definitions
    import_statements: List[ImportStatement] # Import analysis
    config_class_usage: List[ConfigClassUsage]  # How config classes are used
    
    def get_expected_config_class_name(self, builder_name: str) -> str:
        """Get expected configuration class name for builder."""
        # Convert builder name to expected config class name
        # e.g., "xgboost_model_evaluation" -> "XGBoostModelEvaluationConfig"
        words = builder_name.split('_')
        class_name = ''.join(word.capitalize() for word in words) + 'Config'
        return class_name

@dataclass
class ConfigAnalysis:
    """Analysis of configuration file structure."""
    
    class_name: str                          # Configuration class name
    fields: Dict[str, FieldDefinition]      # Configuration fields
    required_fields: List[str]               # Required configuration fields
    optional_fields: List[str]               # Optional configuration fields
    default_values: Dict[str, Any]           # Default values for fields
    load_error: Optional[str] = None         # Error if config couldn't be loaded
    
    def get_field_by_name(self, field_name: str) -> Optional[FieldDefinition]:
        """Get field definition by name."""
        return self.fields.get(field_name)
        
    def is_field_required(self, field_name: str) -> bool:
        """Check if field is required."""
        return field_name in self.required_fields
```

## Reporting and Aggregation Data Structures

### AlignmentReport (Comprehensive Reporting)
```python
@dataclass
class AlignmentReport:
    """Comprehensive alignment report with success metrics."""
    
    level1_results: Dict[str, ValidationResult] = field(default_factory=dict)
    level2_results: Dict[str, ValidationResult] = field(default_factory=dict)
    level3_results: Dict[str, ValidationResult] = field(default_factory=dict)
    level4_results: Dict[str, ValidationResult] = field(default_factory=dict)
    summary: Optional[AlignmentSummary] = None
    
    def add_level1_result(self, script_name: str, result: ValidationResult) -> None:
        """Add Level 1 validation result."""
        self.level1_results[script_name] = result
        
    def add_level2_result(self, contract_name: str, result: ValidationResult) -> None:
        """Add Level 2 validation result."""
        self.level2_results[contract_name] = result
        
    def add_level3_result(self, spec_name: str, result: ValidationResult) -> None:
        """Add Level 3 validation result."""
        self.level3_results[spec_name] = result
        
    def add_level4_result(self, builder_name: str, result: ValidationResult) -> None:
        """Add Level 4 validation result."""
        self.level4_results[builder_name] = result
        
    def generate_summary(self) -> None:
        """Generate comprehensive summary of all validation results."""
        all_results = []
        all_results.extend(self.level1_results.values())
        all_results.extend(self.level2_results.values())
        all_results.extend(self.level3_results.values())
        all_results.extend(self.level4_results.values())
        
        total_tests = len(all_results)
        passed_tests = sum(1 for result in all_results if result.passed)
        
        # Collect all issues
        all_issues = []
        for result in all_results:
            all_issues.extend(result.issues)
            
        # Count issues by severity
        critical_issues = len([i for i in all_issues if i.level == SeverityLevel.CRITICAL])
        error_issues = len([i for i in all_issues if i.level == SeverityLevel.ERROR])
        warning_issues = len([i for i in all_issues if i.level == SeverityLevel.WARNING])
        
        self.summary = AlignmentSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=total_tests - passed_tests,
            pass_rate=passed_tests / total_tests if total_tests > 0 else 1.0,
            critical_issues=critical_issues,
            error_issues=error_issues,
            warning_issues=warning_issues,
            total_issues=len(all_issues)
        )
        
    def is_passing(self) -> bool:
        """Check if overall validation is passing (no critical/error issues)."""
        if not self.summary:
            self.generate_summary()
        return self.summary.critical_issues == 0 and self.summary.error_issues == 0
        
    def get_critical_issues(self) -> List[AlignmentIssue]:
        """Get all critical issues across all levels."""
        critical_issues = []
        for result in [*self.level1_results.values(), *self.level2_results.values(),
                      *self.level3_results.values(), *self.level4_results.values()]:
            critical_issues.extend([i for i in result.issues if i.level == SeverityLevel.CRITICAL])
        return critical_issues
        
    def get_recommendations(self) -> List[str]:
        """Get all actionable recommendations."""
        recommendations = []
        for result in [*self.level1_results.values(), *self.level2_results.values(),
                      *self.level3_results.values(), *self.level4_results.values()]:
            for issue in result.issues:
                if issue.recommendation:
                    recommendations.append(issue.recommendation)
        return list(set(recommendations))  # Remove duplicates

@dataclass
class AlignmentSummary:
    """High-level alignment summary with key metrics."""
    
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    critical_issues: int
    error_issues: int
    warning_issues: int
    total_issues: int
    
    def get_overall_health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        return self.pass_rate
        
    def get_issue_distribution(self) -> Dict[str, int]:
        """Get distribution of issues by severity."""
        return {
            'critical': self.critical_issues,
            'error': self.error_issues,
            'warning': self.warning_issues,
            'total': self.total_issues
        }
```

## Enumerations and Constants

### SeverityLevel (Issue Severity)
```python
class SeverityLevel(Enum):
    """Severity levels for validation issues."""
    
    CRITICAL = "CRITICAL"    # Blocks validation success
    ERROR = "ERROR"          # Blocks validation success
    WARNING = "WARNING"      # Does not block validation
    INFO = "INFO"           # Informational only

class AlignmentLevel(Enum):
    """Alignment validation levels."""
    
    SCRIPT_CONTRACT = "script_contract"
    CONTRACT_SPECIFICATION = "contract_specification"
    SPECIFICATION_DEPENDENCY = "specification_dependency"
    BUILDER_CONFIGURATION = "builder_configuration"
```

## Conclusion

These data structures represent the **foundation of the production-ready validation system** that achieved revolutionary breakthroughs in August 2025. They enable:

- **100% Validation Success**: Robust script-to-contract name mapping eliminates previous failures
- **Enhanced Static Analysis**: Comprehensive pattern detection beyond simple file operations
- **Smart Specification Selection**: Multi-variant architecture support with unified models
- **Production Integration**: Same components as runtime pipeline for consistency
- **Hybrid File Resolution**: Three-tier resolution strategy handles edge cases
- **Comprehensive Reporting**: Actionable diagnostics and detailed success metrics

**Key Breakthrough**: The refactored `validate_contract` method now properly handles script-to-contract name mapping, resolving the critical issue where `xgboost_model_evaluation` script was failing validation due to contract name mismatch (`xgboost_model_eval_contract`).

**August 2025 Registry Enhancement**: Fixed the `get_canonical_name_from_file_name()` function in the step registry to properly handle full framework names like `xgboost_model_evaluation` by adding `'xgboost': 'XGBoost'` to the abbreviation mapping. This ensures systematic registry-based resolution works correctly instead of falling back to pattern matching.

The data structures successfully balance **performance, reliability, and flexibility** while maintaining **production system consistency** and providing **actionable developer feedback**.

## Scoring and Visualization Data Structures (August 2025)

### AlignmentScorer (Weighted Quality Scoring)
```python
@dataclass
class AlignmentScorer:
    """Scorer for evaluating alignment validation quality based on validation results."""
    
    results: Dict[str, Any]                    # Validation results to score
    level_results: Dict[str, Dict[str, Any]]   # Results grouped by alignment level
    
    def __post_init__(self):
        """Initialize level results grouping."""
        self.level_results = self._group_by_level()
    
    def _group_by_level(self) -> Dict[str, Dict[str, Any]]:
        """Group validation results by alignment level using smart pattern detection."""
        grouped = {level: {} for level in ALIGNMENT_LEVEL_WEIGHTS.keys()}
        
        # Handle the actual alignment report format with level1_results, level2_results, etc.
        for key, value in self.results.items():
            if key.endswith('_results') and isinstance(value, dict):
                # Map level1_results -> level1_script_contract, etc.
                if key == 'level1_results':
                    grouped['level1_script_contract'] = value
                elif key == 'level2_results':
                    grouped['level2_contract_spec'] = value
                elif key == 'level3_results':
                    grouped['level3_spec_dependencies'] = value
                elif key == 'level4_results':
                    grouped['level4_builder_config'] = value
        
        return grouped
    
    def calculate_level_score(self, level: str) -> Tuple[float, int, int]:
        """Calculate score for a specific alignment level."""
        # Returns (score, passed_tests, total_tests)
        
    def calculate_overall_score(self) -> float:
        """Calculate weighted overall alignment score."""
        # Uses ALIGNMENT_LEVEL_WEIGHTS for weighted calculation
        
    def get_rating(self, score: float) -> str:
        """Get quality rating based on score thresholds."""
        # Returns: Excellent, Good, Satisfactory, Needs Work, Poor
        
    def generate_chart(self, script_name: str, output_dir: str) -> Optional[str]:
        """Generate alignment quality chart visualization."""
        # Creates professional matplotlib charts with 300 DPI quality
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive alignment score report."""
        # Returns detailed scoring report with metadata

# Scoring Configuration Constants
ALIGNMENT_LEVEL_WEIGHTS = {
    "level1_script_contract": 1.0,      # Basic script-contract alignment
    "level2_contract_spec": 1.5,        # Contract-specification alignment
    "level3_spec_dependencies": 2.0,    # Specification-dependencies alignment
    "level4_builder_config": 2.5,       # Builder-configuration alignment
}

ALIGNMENT_RATING_LEVELS = {
    90: "Excellent",     # 90-100: Excellent alignment
    80: "Good",          # 80-89: Good alignment
    70: "Satisfactory",  # 70-79: Satisfactory alignment
    60: "Needs Work",    # 60-69: Needs improvement
    0: "Poor"            # 0-59: Poor alignment
}

ALIGNMENT_TEST_IMPORTANCE = {
    "script_contract_path_alignment": 1.5,
    "contract_spec_logical_names": 1.4,
    "spec_dependency_resolution": 1.3,
    "builder_config_environment_vars": 1.2,
    "script_contract_environment_vars": 1.2,
    "contract_spec_dependency_mapping": 1.3,
    "spec_dependency_property_paths": 1.4,
    "builder_config_specification_alignment": 1.5,
}
```

### Chart Generation Data Structures
```python
# Chart styling configuration
CHART_CONFIG = {
    "figure_size": (10, 6),
    "colors": {
        "excellent": "#28a745",    # Green
        "good": "#90ee90",         # Light green
        "satisfactory": "#ffa500", # Orange
        "needs_work": "#fa8072",   # Salmon
        "poor": "#dc3545"          # Red
    },
    "grid_style": {
        "axis": "y",
        "linestyle": "--",
        "alpha": 0.7
    }
}

@dataclass
class ChartGenerationResult:
    """Result of chart generation process."""
    
    chart_path: Optional[str]           # Path to generated chart file
    success: bool                       # Whether chart generation succeeded
    error_message: Optional[str]        # Error message if generation failed
    chart_type: str                     # Type of chart generated
    quality_dpi: int = 300             # Chart quality in DPI
    
    def is_available(self) -> bool:
        """Check if chart is available for use."""
        return self.success and self.chart_path and Path(self.chart_path).exists()
```

### Enhanced AlignmentReport (With Scoring Integration)
```python
@dataclass
class AlignmentReport:
    """Enhanced alignment report with scoring and visualization capabilities."""
    
    # Original validation results
    level1_results: Dict[str, ValidationResult] = field(default_factory=dict)
    level2_results: Dict[str, ValidationResult] = field(default_factory=dict)
    level3_results: Dict[str, ValidationResult] = field(default_factory=dict)
    level4_results: Dict[str, ValidationResult] = field(default_factory=dict)
    summary: Optional[AlignmentSummary] = None
    
    # New scoring and visualization fields
    scoring: Optional[Dict[str, Any]] = None           # Scoring data
    chart_path: Optional[str] = None                   # Path to generated chart
    scoring_metadata: Optional[Dict[str, Any]] = None  # Scoring system metadata
    
    def generate_scoring_report(self, script_name: str) -> Dict[str, Any]:
        """Generate comprehensive scoring report for this alignment validation."""
        # Create AlignmentScorer instance
        scorer = AlignmentScorer(self.to_dict())
        
        # Generate scoring report
        scoring_report = scorer.generate_report()
        
        # Store scoring data
        self.scoring = scoring_report
        self.scoring_metadata = {
            "scoring_system": "alignment_validation",
            "level_weights": ALIGNMENT_LEVEL_WEIGHTS,
            "test_importance": ALIGNMENT_TEST_IMPORTANCE,
            "generated_at": datetime.now().isoformat(),
            "script_name": script_name
        }
        
        return scoring_report
    
    def generate_chart(self, script_name: str, output_dir: str = "alignment_reports") -> ChartGenerationResult:
        """Generate chart visualization for this alignment validation."""
        try:
            # Create AlignmentScorer instance
            scorer = AlignmentScorer(self.to_dict())
            
            # Generate chart
            chart_path = scorer.generate_chart(script_name, output_dir)
            
            if chart_path:
                self.chart_path = chart_path
                return ChartGenerationResult(
                    chart_path=chart_path,
                    success=True,
                    error_message=None,
                    chart_type="alignment_quality_bar_chart",
                    quality_dpi=300
                )
            else:
                return ChartGenerationResult(
                    chart_path=None,
                    success=False,
                    error_message="Chart generation failed (matplotlib not available)",
                    chart_type="alignment_quality_bar_chart"
                )
                
        except Exception as e:
            return ChartGenerationResult(
                chart_path=None,
                success=False,
                error_message=str(e),
                chart_type="alignment_quality_bar_chart"
            )
    
    def export_enhanced_json(self, output_path: str, include_scoring: bool = True) -> None:
        """Export enhanced JSON report with scoring data."""
        report_data = self.to_dict()
        
        if include_scoring and self.scoring:
            report_data['scoring'] = self.scoring
            report_data['scoring_metadata'] = self.scoring_metadata
            
        if self.chart_path:
            report_data['chart_path'] = self.chart_path
            
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def get_overall_score(self) -> Optional[float]:
        """Get overall alignment score if
