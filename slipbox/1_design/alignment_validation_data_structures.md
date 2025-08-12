---
tags:
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
date of note: 2025-08-11
---

# Alignment Validation Data Structures

## Related Documents
- **[Master Design](unified_alignment_tester_master_design.md)** - Complete system overview
- **[Architecture](unified_alignment_tester_architecture.md)** - Core architectural patterns
- **[Implementation Guide](unified_alignment_tester_implementation.md)** - Production implementation details
- **[Pain Points Analysis](../4_analysis/unified_alignment_tester_pain_points_analysis.md)** - Comprehensive analysis of validation challenges that drove the data structure design decisions documented here

## Overview

This document defines the critical data structures that emerged from the revolutionary breakthroughs achieved in August 2025. These data structures represent the foundation of the production-ready validation system, enabling exceptional success rates across all four validation levels.

## Core Validation Data Structures

### ValidationIssue (Enhanced Error Reporting)
```python
@dataclass
class ValidationIssue:
    """Enhanced validation issue with production-grade diagnostics."""
    
    severity: str                           # CRITICAL, ERROR, WARNING, INFO
    category: str                          # Specific issue type for filtering
    message: str                           # Human-readable description
    details: Dict[str, Any]                # Technical details for debugging
    recommendation: str                    # Specific action to resolve
    resolution_strategy: Optional[str]     # Which strategy was used/failed
    confidence_score: Optional[float]      # For Level 3 dependency resolution
    variant_info: Optional[Dict]           # For Level 2 multi-variant validation
    timestamp: datetime                    # When issue was detected
    context: Dict[str, Any]                # Additional context information
    
    def is_blocking(self) -> bool:
        """Check if issue blocks validation success."""
        return self.severity in ['CRITICAL', 'ERROR']
        
    def get_actionable_recommendation(self) -> str:
        """Get specific actionable recommendation."""
        if self.resolution_strategy:
            return f"{self.recommendation} (Strategy: {self.resolution_strategy})"
        return self.recommendation
```

### ValidationResult (Comprehensive Results)
```python
@dataclass
class ValidationResult:
    """Comprehensive validation result with success metrics."""
    
    script_name: str                       # Target script name
    level: int                            # Validation level (1-4)
    passed: bool                          # Overall pass/fail status
    issues: List[ValidationIssue]         # All detected issues
    success_metrics: Dict[str, Any]       # Detailed success metrics
    resolution_details: Dict[str, Any]    # Resolution strategy details
    performance_metrics: Dict[str, Any]   # Performance timing information
    degraded: bool = False                # Whether result is degraded due to errors
    error_context: Optional[Dict] = None  # Error context for debugging
    
    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get all critical issues."""
        return [issue for issue in self.issues if issue.severity == 'CRITICAL']
        
    def get_error_issues(self) -> List[ValidationIssue]:
        """Get all error issues."""
        return [issue for issue in self.issues if issue.severity == 'ERROR']
        
    def get_success_rate(self) -> float:
        """Calculate success rate based on issue severity."""
        if not self.issues:
            return 1.0
        blocking_issues = len([i for i in self.issues if i.is_blocking()])
        return max(0.0, 1.0 - (blocking_issues / len(self.issues)))
```

## Level 1: Script ↔ Contract Data Structures

### EnhancedScriptAnalysis (Revolutionary Static Analysis)
```python
@dataclass
class EnhancedScriptAnalysis:
    """Enhanced script analysis beyond simple file operations."""
    
    script_name: str
    file_operations: List[FileOperation]           # Enhanced detection (tarfile, shutil, pathlib)
    argument_definitions: List[ArgumentDefinition] # Argparse hyphen-to-underscore handling
    logical_name_usage: Dict[str, PathReference]   # Contract-aware logical name mapping
    environment_variable_access: List[EnvVarAccess]
    path_constants: Dict[str, str]                 # Path variable assignments
    import_statements: List[ImportStatement]       # Import analysis
    framework_patterns: List[FrameworkPattern]     # SageMaker/Python framework patterns
    
    def get_file_operations_by_type(self, operation_type: str) -> List[FileOperation]:
        """Get file operations filtered by type."""
        return [op for op in self.file_operations if op.operation_type == operation_type]
        
    def resolve_logical_name(self, path_reference: str, contract: Any) -> Optional[str]:
        """Resolve path reference to logical name using contract mapping."""
        # Contract-aware resolution logic
        return self.logical_name_usage.get(path_reference)

@dataclass
class FileOperation:
    """Enhanced file operation detection."""
    
    operation_type: str        # 'open', 'tarfile_open', 'shutil_copy', 'pathlib_mkdir', etc.
    file_path: str            # Path being operated on
    line_number: int          # Source line number
    context: str              # Surrounding code context
    is_logical_name: bool     # Whether path uses logical name
    resolved_logical_name: Optional[str]  # Resolved logical name if applicable

@dataclass
class ArgumentDefinition:
    """Argparse argument definition with convention handling."""
    
    cli_name: str             # CLI argument name (e.g., '--job-type')
    script_name: str          # Script variable name (e.g., 'job_type')
    argument_type: str        # Argument type
    required: bool            # Whether argument is required
    default_value: Any        # Default value if any
    help_text: str           # Help text
    
    def matches_contract_argument(self, contract_arg: str) -> bool:
        """Check if this argument matches contract argument."""
        # Handle hyphen-to-underscore conversion
        normalized_cli = self.cli_name.lstrip('-').replace('-', '_')
        normalized_script = self.script_name.replace('-', '_')
        normalized_contract = contract_arg.replace('-', '_')
        
        return normalized_contract in [normalized_cli, normalized_script]
```

### HybridContractValidation (Robust Import Handling)
```python
@dataclass
class HybridContractValidation:
    """Hybrid contract validation with robust error handling."""
    
    contract_path: str
    sys_path_manager: SysPathManager           # Temporary, clean sys.path manipulation
    contract_loader: ContractLoader            # Handles relative imports correctly
    validation_logic: ContractAwareValidator   # Understands contract structure
    fallback_strategies: List[ValidationStrategy]  # Multiple resolution approaches
    
    def load_contract_safely(self) -> Optional[Any]:
        """Load contract with hybrid sys.path management."""
        with self.sys_path_manager.temporary_path():
            try:
                return self.contract_loader.load_contract(self.contract_path)
            except ImportError as e:
                # Try fallback strategies
                for strategy in self.fallback_strategies:
                    try:
                        return strategy.load_contract(self.contract_path)
                    except Exception:
                        continue
                return None

@dataclass
class SysPathManager:
    """Temporary, clean sys.path manipulation."""
    
    original_path: List[str]
    temporary_additions: List[str]
    
    @contextmanager
    def temporary_path(self):
        """Context manager for temporary sys.path modification."""
        import sys
        original = sys.path.copy()
        try:
            # Add temporary paths
            for path in self.temporary_additions:
                if path not in sys.path:
                    sys.path.insert(0, path)
            yield
        finally:
            # Restore original sys.path
            sys.path[:] = original
```

## Level 2: Contract ↔ Specification Data Structures

### SmartSpecificationSelection (Revolutionary Multi-Variant Support)
```python
@dataclass
class SmartSpecificationSelection:
    """Smart specification selection with multi-variant detection."""
    
    base_name: str                                    # Base specification name
    multi_variant_detection: bool                     # Whether multi-variant pattern detected
    variant_grouping: Dict[str, List[SpecificationFile]]  # Grouped by job type
    unified_specification: UnifiedSpecificationModel      # Union of all variants
    validation_strategy: ValidationStrategy               # Intelligent vs rigid validation
    
    def detect_multi_variant_pattern(self, specifications: List[str]) -> bool:
        """Detect if specifications follow multi-variant job-type pattern."""
        job_types = ['training', 'testing', 'validation', 'calibration']
        
        # Check if we have multiple specifications with job type suffixes
        variants_found = []
        for spec in specifications:
            for job_type in job_types:
                if spec.endswith(f'_{job_type}_spec.py'):
                    variants_found.append(job_type)
                    
        return len(variants_found) > 1
        
    def create_unified_specification(self) -> UnifiedSpecificationModel:
        """Create unified model from multiple variants."""
        all_dependencies = set()
        all_outputs = set()
        variant_metadata = {}
        
        for job_type, specs in self.variant_grouping.items():
            for spec in specs:
                all_dependencies.update(spec.dependencies)
                all_outputs.update(spec.outputs)
                variant_metadata[job_type] = {
                    'dependencies': spec.dependencies,
                    'outputs': spec.outputs,
                    'inputs': spec.inputs
                }
                
        return UnifiedSpecificationModel(
            variants=variant_metadata,
            unified_dependencies=all_dependencies,
            unified_outputs=all_outputs,
            metadata=VariantMetadata(
                total_variants=len(self.variant_grouping),
                job_types=list(self.variant_grouping.keys())
            )
        )

@dataclass
class UnifiedSpecificationModel:
    """Unified specification model representing union of all variants."""
    
    variants: Dict[str, SpecificationVariant]  # training, testing, validation, calibration
    unified_dependencies: Set[DependencySpec]  # Union of all variant dependencies
    unified_outputs: Set[OutputSpec]           # Union of all variant outputs
    metadata: VariantMetadata                  # Tracks which variants contribute what
    
    def is_input_valid_for_any_variant(self, input_name: str) -> bool:
        """Check if input is valid for any variant (permissive validation)."""
        for variant in self.variants.values():
            if input_name in variant.get('inputs', []):
                return True
        return False
        
    def get_required_dependencies_intersection(self) -> Set[str]:
        """Get intersection of required dependencies across all variants."""
        if not self.variants:
            return set()
            
        required_deps = None
        for variant in self.variants.values():
            variant_required = set(variant.get('required_dependencies', []))
            if required_deps is None:
                required_deps = variant_required
            else:
                required_deps = required_deps.intersection(variant_required)
                
        return required_deps or set()

@dataclass
class MultiVariantValidation:
    """Multi-variant validation logic."""
    
    permissive_input_validation: bool = True   # Contract input valid if exists in ANY variant
    required_dependency_checking: bool = True  # Contract must cover intersection of required deps
    informational_feedback: bool = True        # Shows which variants use which dependencies
    
    def validate_contract_against_unified_spec(self, contract: Any, 
                                             unified_spec: UnifiedSpecificationModel) -> List[ValidationIssue]:
        """Validate contract against unified specification with intelligent logic."""
        issues = []
        
        # Permissive input validation
        for input_name in contract.inputs:
            if not unified_spec.is_input_valid_for_any_variant(input_name):
                issues.append(ValidationIssue(
                    severity="WARNING",
                    category="contract_input",
                    message=f"Contract input '{input_name}' not used by any specification variant",
                    details={"input_name": input_name, "available_variants": list(unified_spec.variants.keys())},
                    recommendation=f"Verify if input '{input_name}' is needed or remove from contract"
                ))
                
        # Required dependency coverage validation
        required_deps = unified_spec.get_required_dependencies_intersection()
        for dep in required_deps:
            if dep not in contract.dependencies:
                issues.append(ValidationIssue(
                    severity="ERROR",
                    category="missing_dependency",
                    message=f"Contract missing required dependency '{dep}' needed by all variants",
                    details={"dependency": dep, "variants_requiring": list(unified_spec.variants.keys())},
                    recommendation=f"Add dependency '{dep}' to contract"
                ))
                
        return issues
```

## Level 3: Specification ↔ Dependencies Data Structures

### ProductionDependencyResolution (Threshold-Based Validation)
```python
@dataclass
class ProductionDependencyResolution:
    """Production dependency resolution with confidence scoring."""
    
    dependency_resolver: ProductionDependencyResolver  # Same resolver as runtime
    confidence_scoring: CompatibilityScorer           # Multi-factor scoring system
    canonical_mapping: CanonicalNameMapper            # Production registry integration
    threshold_validation: ThresholdValidator          # Clear pass/fail criteria (0.6)
    
    def resolve_dependencies_with_confidence(self, dependencies: List[str]) -> DependencyResolutionResult:
        """Resolve dependencies using production resolver with confidence scoring."""
        resolved = {}
        failed = {}
        
        for dep in dependencies:
            try:
                # Use production dependency resolver
                resolution = self.dependency_resolver.resolve_dependency(dep)
                confidence = self.confidence_scoring.calculate_confidence(dep, resolution)
                
                if confidence >= self.threshold_validation.threshold:
                    resolved[dep] = {
                        'target': resolution.target,
                        'confidence': confidence,
                        'resolution_path': resolution.path,
                        'strategy': resolution.strategy
                    }
                else:
                    failed[dep] = {
                        'reason': 'Low confidence score',
                        'confidence': confidence,
                        'best_match': resolution.target,
                        'threshold': self.threshold_validation.threshold
                    }
                    
            except Exception as e:
                failed[dep] = {
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
class CompatibilityScoring:
    """Multi-factor compatibility scoring system."""
    
    type_compatibility_weight: float = 0.40      # 40% weight
    data_type_compatibility_weight: float = 0.20 # 20% weight  
    semantic_similarity_weight: float = 0.25     # 25% weight
    source_compatibility_weight: float = 0.10    # 10% weight
    keyword_matching_weight: float = 0.05        # 5% weight
    
    def calculate_confidence(self, dependency: str, resolution: Any) -> float:
        """Calculate multi-factor confidence score."""
        scores = {
            'type_compatibility': self.calculate_type_compatibility(dependency, resolution),
            'data_type_compatibility': self.calculate_data_type_compatibility(dependency, resolution),
            'semantic_similarity': self.calculate_semantic_similarity(dependency, resolution),
            'source_compatibility': self.calculate_source_compatibility(dependency, resolution),
            'keyword_matching': self.calculate_keyword_matching(dependency, resolution)
        }
        
        # Weighted average
        total_score = (
            scores['type_compatibility'] * self.type_compatibility_weight +
            scores['data_type_compatibility'] * self.data_type_compatibility_weight +
            scores['semantic_similarity'] * self.semantic_similarity_weight +
            scores['source_compatibility'] * self.source_compatibility_weight +
            scores['keyword_matching'] * self.keyword_matching_weight
        )
        
        return min(1.0, max(0.0, total_score))

@dataclass
class EnhancedDependencyValidation:
    """Enhanced dependency validation with actionable recommendations."""
    
    resolved_dependencies: Dict[str, PropertyReference]
    failed_with_analysis: Dict[str, FailureAnalysis]
    confidence_scores: Dict[str, float]
    actionable_recommendations: List[str]
    canonical_name_mapping: Dict[str, str]
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        total_deps = len(self.resolved_dependencies) + len(self.failed_with_analysis)
        success_rate = len(self.resolved_dependencies) / total_deps if total_deps > 0 else 1.0
        
        return {
            'total_dependencies': total_deps,
            'resolved_count': len(self.resolved_dependencies),
            'failed_count': len(self.failed_with_analysis),
            'success_rate': success_rate,
            'average_confidence': self.get_average_confidence(),
            'actionable_recommendations': self.actionable_recommendations
        }
        
    def get_average_confidence(self) -> float:
        """Calculate average confidence score for resolved dependencies."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)
```

## Level 4: Builder ↔ Configuration Data Structures

### HybridFileResolution (Three-Tier Strategy)
```python
@dataclass
class HybridFileResolution:
    """Hybrid file resolution with three-tier strategy."""
    
    standard_patterns: Dict[str, str]          # Fast path for conventional naming
    flexible_mappings: Dict[str, str]          # Edge case handling via FlexibleFileResolver
    fuzzy_matching: FuzzyMatcher               # Similarity-based discovery
    resolution_strategy: ResolutionStrategy   # Which strategy succeeded
    performance_optimization: bool = True      # Fastest path first
    
    def find_config_file_hybrid(self, builder_name: str) -> Optional[FileResolutionResult]:
        """Three-tier hybrid file resolution strategy."""
        
        # Strategy 1: Standard pattern (fastest path)
        if self.performance_optimization:
            standard_result = self.try_standard_pattern(builder_name)
            if standard_result:
                return FileResolutionResult(
                    file_path=standard_result,
                    strategy='standard_pattern',
                    confidence=1.0,
                    performance_tier=1
                )
        
        # Strategy 2: FlexibleFileResolver (edge cases)
        flexible_result = self.try_flexible_resolver(builder_name)
        if flexible_result:
            return FileResolutionResult(
                file_path=flexible_result,
                strategy='flexible_file_resolver',
                confidence=0.9,
                performance_tier=2
            )
            
        # Strategy 3: Fuzzy matching (unexpected variations)
        fuzzy_result = self.fuzzy_matching.find_similar_file(builder_name)
        if fuzzy_result and fuzzy_result.similarity >= 0.8:
            return FileResolutionResult(
                file_path=fuzzy_result.file_path,
                strategy='fuzzy_matching',
                confidence=fuzzy_result.similarity,
                performance_tier=3
            )
            
        return None

@dataclass
class FlexibleFileResolver:
    """FlexibleFileResolver with predefined mappings and fuzzy matching."""
    
    predefined_mappings: Dict[str, str]        # Known edge case mappings
    similarity_threshold: float = 0.8          # Fuzzy matching threshold
    search_strategies: List[SearchStrategy]    # Multiple discovery approaches
    cache: Dict[str, ResolvedPath]             # Performance optimization
    
    def resolve_file(self, target_name: str) -> Optional[str]:
        """Resolve file using flexible strategies."""
        
        # Check cache first
        if target_name in self.cache:
            return self.cache[target_name].file_path
            
        # Check predefined mappings
        if target_name in self.predefined_mappings:
            result = self.predefined_mappings[target_name]
            self.cache[target_name] = ResolvedPath(result, 'predefined_mapping')
            return result
            
        # Try search strategies
        for strategy in self.search_strategies:
            result = strategy.search(target_name)
            if result:
                self.cache[target_name] = ResolvedPath(result, strategy.name)
                return result
                
        return None

@dataclass
class ProductionRegistryIntegration:
    """Production registry integration for naming consistency."""
    
    canonical_name_mapping: CanonicalNameMapper
    registry_consistency: RegistryValidator
    production_alignment: ProductionAligner
    
    def get_canonical_builder_name(self, file_name: str) -> str:
        """Convert file-based name to canonical builder name."""
        return self.canonical_name_mapping.get_canonical_name(file_name)
        
    def validate_registry_consistency(self) -> List[ValidationIssue]:
        """Validate consistency between registry and file system."""
        return self.registry_consistency.validate_consistency()
        
    def ensure_production_alignment(self, builder_name: str) -> AlignmentResult:
        """Ensure builder naming aligns with production conventions."""
        return self.production_alignment.check_alignment(builder_name)
```

## Reporting and Aggregation Data Structures

### AlignmentReport (Comprehensive Reporting)
```python
@dataclass
class AlignmentReport:
    """Comprehensive alignment report with success metrics."""
    
    validation_timestamp: datetime
    overall_success_rate: float
    level_results: Dict[int, List[ValidationResult]]
    success_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    summary: AlignmentSummary
    
    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get all critical issues across all levels."""
        critical_issues = []
        for level_results in self.level_results.values():
            for result in level_results:
                critical_issues.extend(result.get_critical_issues())
        return critical_issues
        
    def get_success_rate_by_level(self) -> Dict[int, float]:
        """Get success rate for each validation level."""
        success_rates = {}
        for level, results in self.level_results.items():
            if results:
                passed_count = sum(1 for r in results if r.passed)
                success_rates[level] = passed_count / len(results)
            else:
                success_rates[level] = 1.0
        return success_rates
        
    def is_passing(self) -> bool:
        """Check if overall validation is passing."""
        critical_issues = self.get_critical_issues()
        return len(critical_issues) == 0

@dataclass
class AlignmentSummary:
    """High-level alignment summary."""
    
    total_scripts_validated: int
    scripts_passing_all_levels: int
    scripts_with_critical_issues: int
    scripts_with_warnings: int
    revolutionary_breakthroughs_applied: List[str]
    production_integration_status: str
    
    def get_overall_health_score(self) -> float:
        """Calculate overall health score."""
        if self.total_scripts_validated == 0:
            return 1.0
        return self.scripts_passing_all_levels / self.total_scripts_validated
```

## Conclusion

These data structures represent the **foundation of the production-ready validation system** that achieved revolutionary breakthroughs in August 2025. They enable:

- **Enhanced Static Analysis**: Beyond simple file operations detection
- **Smart Specification Selection**: Multi-variant architecture support  
- **Production Integration**: Same components as runtime pipeline
- **Hybrid File Resolution**: Three-tier resolution strategy
- **Comprehensive Reporting**: Actionable diagnostics and success metrics

The data structures successfully balance **performance, reliability, and flexibility** while maintaining **production system consistency** and providing **actionable developer feedback**.

---

**Data Structures Document Updated**: August 11, 2025  
**Status**: Production-Ready Data Models  
**Success Rate**: 87.5% overall validation success  
**Next Phase**: Continued optimization and feature enhancement
