"""
Specification ↔ Dependencies Alignment Tester

Validates alignment between step specifications and their dependency declarations.
Ensures dependency chains are consistent and resolvable.
"""

import os
import sys
import importlib.util
import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from .alignment_utils import (
    FlexibleFileResolver, DependencyPatternClassifier, DependencyPattern
)
from .level3_validation_config import Level3ValidationConfig, ValidationMode
from ...steps.registry.step_names import (
    get_step_name_from_spec_type, get_spec_step_type_with_job_type, validate_spec_type,
    get_canonical_name_from_file_name, STEP_NAMES, CONFIG_STEP_REGISTRY
)
from ...core.deps.factory import create_pipeline_components
from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

logger = logging.getLogger(__name__)


class SpecificationDependencyAlignmentTester:
    """
    Tests alignment between step specifications and their dependencies.
    
    Validates:
    - Dependency chains are consistent
    - All dependencies can be resolved
    - No circular dependencies exist
    - Data types match across dependency chains
    """
    
    def __init__(self, specs_dir: str, validation_config: Level3ValidationConfig = None):
        """
        Initialize the specification-dependency alignment tester.
        
        Args:
            specs_dir: Directory containing step specifications
            validation_config: Configuration for validation thresholds and behavior
        """
        self.specs_dir = Path(specs_dir)
        self.config = validation_config or Level3ValidationConfig.create_relaxed_config()
        
        # Initialize the dependency pattern classifier
        self.dependency_classifier = DependencyPatternClassifier()
        
        # Initialize the file resolver for finding specification files
        base_directories = {
            'specs': str(self.specs_dir)
        }
        self.file_resolver = FlexibleFileResolver(base_directories)
        
        # Initialize dependency resolver components
        self.pipeline_components = create_pipeline_components("level3_validation")
        self.dependency_resolver = self.pipeline_components["resolver"]
        self.spec_registry = self.pipeline_components["registry"]
        
        # Log configuration
        threshold_desc = self.config.get_threshold_description()
        logger.info(f"Level 3 validation initialized with {threshold_desc['mode']} mode")
        logger.debug(f"Thresholds: {threshold_desc['thresholds']}")
    
    def validate_all_specifications(self, target_scripts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Validate alignment for all specifications or specified target scripts.
        
        Args:
            target_scripts: Specific scripts to validate (None for all)
            
        Returns:
            Dictionary mapping specification names to validation results
        """
        results = {}
        
        # Discover specifications to validate
        if target_scripts:
            specs_to_validate = target_scripts
        else:
            specs_to_validate = self._discover_specifications()
        
        for spec_name in specs_to_validate:
            try:
                result = self.validate_specification(spec_name)
                results[spec_name] = result
            except Exception as e:
                results[spec_name] = {
                    'passed': False,
                    'error': str(e),
                    'issues': [{
                        'severity': 'CRITICAL',
                        'category': 'validation_error',
                        'message': f'Failed to validate specification {spec_name}: {str(e)}'
                    }]
                }
        
        return results
    
    def validate_specification(self, spec_name: str) -> Dict[str, Any]:
        """
        Validate alignment for a specific specification.
        
        Args:
            spec_name: Name of the specification to validate
            
        Returns:
            Validation result dictionary
        """
        # Find specification files (multiple files for different job types)
        spec_files = self._find_specification_files(spec_name)
        
        if not spec_files:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'missing_file',
                    'message': f'Specification file not found: {self.specs_dir / f"{spec_name}_spec.py"}',
                    'recommendation': f'Create the specification file {spec_name}_spec.py'
                }]
            }
        
        # Load specifications from Python files
        specifications = {}
        for spec_file in spec_files:
            try:
                job_type = self._extract_job_type_from_spec_file(spec_file)
                spec = self._load_specification_from_python(spec_file, spec_name, job_type)
                specifications[job_type] = spec
            except Exception as e:
                return {
                    'passed': False,
                    'issues': [{
                        'severity': 'CRITICAL',
                        'category': 'spec_parse_error',
                        'message': f'Failed to parse specification from {spec_file}: {str(e)}',
                        'recommendation': 'Fix Python syntax or specification structure'
                    }]
                }
        
        # Use the first specification for validation (they should be consistent)
        specification = next(iter(specifications.values()))
        
        # Load all specifications for dependency resolution
        all_specs = self._load_all_specifications()
        
        # Perform alignment validation
        issues = []
        
        # Validate dependency resolution
        resolution_issues = self._validate_dependency_resolution(specification, all_specs, spec_name)
        issues.extend(resolution_issues)
        
        # Validate circular dependencies
        circular_issues = self._validate_circular_dependencies(specification, all_specs, spec_name)
        issues.extend(circular_issues)
        
        # Validate data type consistency
        type_issues = self._validate_dependency_data_types(specification, all_specs, spec_name)
        issues.extend(type_issues)
        
        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue['severity'] in ['CRITICAL', 'ERROR'] for issue in issues
        )
        
        return {
            'passed': not has_critical_or_error,
            'issues': issues,
            'specification': specification
        }
    
    def _validate_dependency_resolution(self, specification: Dict[str, Any], all_specs: Dict[str, Dict[str, Any]], spec_name: str) -> List[Dict[str, Any]]:
        """Enhanced dependency validation with compatibility scoring."""
        issues = []
        
        # Handle case where step has no dependencies
        dependencies = specification.get('dependencies', [])
        if not dependencies:
            logger.info(f"✅ {spec_name} has no dependencies - validation passed")
            return issues
        
        # Populate the resolver registry with all specifications
        self._populate_resolver_registry(all_specs)
        
        # Get available step names using canonical names from the registry (single source of truth)
        available_steps = self._get_available_canonical_step_names(all_specs)
        
        try:
            # Convert spec_name to canonical name for dependency resolution
            canonical_spec_name = self._get_canonical_step_name(spec_name)
            
            # Use enhanced resolution with scoring
            resolution_result = self.dependency_resolver.resolve_with_scoring(
                canonical_spec_name, available_steps
            )
            
            resolved_deps = resolution_result['resolved']
            failed_deps = resolution_result['failed_with_scores']
            
            # Process resolved dependencies
            for dep_name, prop_ref in resolved_deps.items():
                if self.config.LOG_SUCCESSFUL_RESOLUTIONS:
                    logger.info(f"✅ Resolved {spec_name}.{dep_name} -> {prop_ref}")
            
            # Process failed dependencies with scoring
            for dep_name, failure_info in failed_deps.items():
                best_candidate = failure_info['best_candidate']
                is_required = failure_info['required']
                
                if best_candidate is None:
                    # No candidates found at all
                    if is_required:
                        issues.append({
                            'severity': 'CRITICAL',
                            'category': 'dependency_resolution',
                            'message': f'No compatible candidates found for required dependency: {dep_name}',
                            'details': {
                                'logical_name': dep_name,
                                'specification': spec_name,
                                'available_steps': available_steps,
                                'candidates_found': 0
                            },
                            'recommendation': f'Ensure a step exists that produces output compatible with {dep_name}'
                        })
                    else:
                        # Optional dependency with no candidates - just log
                        if self.config.LOG_FAILED_RESOLUTIONS:
                            logger.debug(f"Optional dependency {spec_name}.{dep_name} has no compatible candidates")
                else:
                    # Candidates found but below resolution threshold
                    score = best_candidate['score']
                    severity = self.config.determine_severity_from_score(score, is_required)
                    
                    # Only create issues for dependencies that don't pass validation
                    if not self.config.should_pass_validation(score):
                        issue = {
                            'severity': severity,
                            'category': 'dependency_compatibility',
                            'message': f'Dependency {dep_name} has low compatibility score: {score:.3f}',
                            'details': {
                                'logical_name': dep_name,
                                'specification': spec_name,
                                'best_match': {
                                    'provider': best_candidate['provider_step'],
                                    'output': best_candidate['output_name'],
                                    'score': score
                                },
                                'required': is_required,
                                'threshold_info': self.config.get_threshold_description()
                            },
                            'recommendation': self._generate_compatibility_recommendation(dep_name, best_candidate)
                        }
                        
                        # Add score breakdown if configured
                        if self.config.INCLUDE_SCORE_BREAKDOWN:
                            issue['details']['score_breakdown'] = best_candidate['score_breakdown']
                        
                        # Add alternative candidates if configured
                        if self.config.INCLUDE_ALTERNATIVE_CANDIDATES:
                            issue['details']['all_candidates'] = [
                                {
                                    'provider': c['provider_step'],
                                    'output': c['output_name'], 
                                    'score': c['score']
                                } for c in failure_info['all_candidates'][:self.config.MAX_ALTERNATIVE_CANDIDATES]
                            ]
                        
                        issues.append(issue)
                    
                    # Log the best attempt for transparency
                    if self.config.LOG_FAILED_RESOLUTIONS:
                        logger.info(f"🔍 Best match for {spec_name}.{dep_name}: "
                                  f"{best_candidate['provider_step']}.{best_candidate['output_name']} "
                                  f"(score: {score:.3f}, threshold: {self.config.PASS_THRESHOLD:.1f})")
                              
        except Exception as e:
            issues.append({
                'severity': 'ERROR',
                'category': 'resolver_error',
                'message': f'Dependency resolver failed: {str(e)}',
                'details': {'specification': spec_name, 'error': str(e)},
                'recommendation': 'Check specification format and dependency resolver configuration'
            })
        
        return issues
    
    def _generate_compatibility_recommendation(self, dep_name: str, best_candidate: Dict) -> str:
        """Generate specific recommendations based on compatibility analysis."""
        if 'score_breakdown' not in best_candidate:
            return f"Review dependency specification for {dep_name} and output specification for {best_candidate['output_name']}"
        
        score_breakdown = best_candidate['score_breakdown']
        recommendations = []
        
        if score_breakdown.get('type_compatibility', 0) < 0.2:
            recommendations.append(f"Consider changing dependency type or output type for better compatibility")
        
        if score_breakdown.get('semantic_similarity', 0) < 0.15:
            recommendations.append(f"Consider renaming '{dep_name}' or adding aliases to improve semantic matching")
        
        if score_breakdown.get('source_compatibility', 0) < 0.05:
            recommendations.append(f"Add '{best_candidate['provider_step']}' to compatible_sources for {dep_name}")
        
        if score_breakdown.get('data_type_compatibility', 0) < 0.1:
            recommendations.append(f"Align data types between dependency and output specifications")
        
        if not recommendations:
            recommendations.append(f"Review dependency specification for {dep_name} and output specification for {best_candidate['output_name']}")
        
        return "; ".join(recommendations)
    
    def _get_available_canonical_step_names(self, all_specs: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Get available canonical step names using the registry as single source of truth.
        
        This method queries the production registry to get the authoritative list of 
        canonical step names, ensuring alignment with production dependency resolution.
        
        Args:
            all_specs: Dictionary of all loaded specifications
            
        Returns:
            List of canonical step names from the production registry
        """
        from ...steps.registry.step_names import get_all_step_names
        
        # Get canonical step names from the production registry (single source of truth)
        canonical_names = get_all_step_names()
        
        logger.debug(f"Available canonical step names from registry: {canonical_names}")
        return canonical_names
    
    def _get_canonical_step_name(self, spec_file_name: str) -> str:
        """
        Convert specification file name to canonical step name using the registry.
        
        Uses the centralized FILE_NAME_TO_CANONICAL mapping as the single source of truth.
        
        Args:
            spec_file_name: File-based specification name (e.g., "dummy_training", "model_calibration", "model_evaluation_xgb")
            
        Returns:
            Canonical step name from the registry
        """
        try:
            # Use the centralized registry mapping (single source of truth)
            canonical_name = get_canonical_name_from_file_name(spec_file_name)
            logger.debug(f"Mapped spec file '{spec_file_name}' -> canonical '{canonical_name}' (registry)")
            return canonical_name
        except ValueError as e:
            logger.debug(f"Registry mapping failed for '{spec_file_name}': {e}")
        
        # Fallback: Try to load the specification file and get the step_type directly
        try:
            spec_files = self._find_specification_files(spec_file_name)
            if spec_files:
                spec_file = spec_files[0]  # Use the first file found
                job_type = self._extract_job_type_from_spec_file(spec_file)
                spec_dict = self._load_specification_from_python(spec_file, spec_file_name, job_type)
                
                # Get the step_type from the loaded specification
                step_type = spec_dict.get('step_type')
                if step_type:
                    # Use the production function to get canonical name from step_type
                    canonical_name = get_step_name_from_spec_type(step_type)
                    logger.debug(f"Mapped spec file '{spec_file_name}' -> step_type '{step_type}' -> canonical '{canonical_name}' (fallback)")
                    return canonical_name
        except Exception as e:
            logger.debug(f"Could not load specification file for '{spec_file_name}': {e}")
        
        # Final fallback: Convert file name to spec_type format and try registry lookup
        parts = spec_file_name.split('_')
        
        # Handle job type variants
        job_type_suffixes = ['training', 'validation', 'testing', 'calibration']
        job_type = None
        base_parts = parts
        
        if len(parts) > 1 and parts[-1] in job_type_suffixes:
            job_type = parts[-1]
            base_parts = parts[:-1]
        
        # Convert to PascalCase for spec_type
        spec_type_base = ''.join(word.capitalize() for word in base_parts)
        
        if job_type:
            spec_type = f"{spec_type_base}_{job_type.capitalize()}"
        else:
            spec_type = spec_type_base
        
        # Use production function to get canonical name (strips job type suffix)
        try:
            canonical_name = get_step_name_from_spec_type(spec_type)
            logger.debug(f"Mapped spec file '{spec_file_name}' -> spec_type '{spec_type}' -> canonical '{canonical_name}' (final fallback)")
            return canonical_name
        except Exception as e:
            # Ultimate fallback: return the base spec_type without job type suffix
            logger.warning(f"Failed to get canonical name for '{spec_file_name}' via all methods: {e}")
            return spec_type_base
    
    def _populate_resolver_registry(self, all_specs: Dict[str, Dict[str, Any]]):
        """Populate the dependency resolver registry with all specifications using canonical names."""
        for spec_name, spec_dict in all_specs.items():
            try:
                # Convert file-based spec name to canonical step name
                canonical_name = self._get_canonical_step_name(spec_name)
                
                # Convert dict back to StepSpecification object
                step_spec = self._dict_to_step_specification(spec_dict)
                
                # Register with canonical name
                self.dependency_resolver.register_specification(canonical_name, step_spec)
                logger.debug(f"Registered specification: '{spec_name}' as canonical '{canonical_name}'")
                
            except Exception as e:
                logger.warning(f"Failed to register {spec_name} with resolver: {e}")
    
    def _dict_to_step_specification(self, spec_dict: Dict[str, Any]) -> StepSpecification:
        """Convert specification dictionary back to StepSpecification object."""
        # Convert dependencies
        dependencies = {}
        for dep in spec_dict.get('dependencies', []):
            # Create DependencySpec using keyword arguments
            dep_data = {
                'logical_name': dep['logical_name'],
                'dependency_type': dep['dependency_type'],  # Keep as string, validator will convert
                'required': dep['required'],
                'compatible_sources': dep.get('compatible_sources', []),
                'data_type': dep['data_type'],
                'description': dep.get('description', ''),
                'semantic_keywords': dep.get('semantic_keywords', [])
            }
            dep_spec = DependencySpec(**dep_data)
            dependencies[dep['logical_name']] = dep_spec
        
        # Convert outputs
        outputs = {}
        for out in spec_dict.get('outputs', []):
            # Create OutputSpec using keyword arguments
            out_data = {
                'logical_name': out['logical_name'],
                'output_type': out['output_type'],  # Keep as string, validator will convert
                'property_path': out['property_path'],
                'data_type': out['data_type'],
                'description': out.get('description', ''),
                'aliases': out.get('aliases', [])
            }
            out_spec = OutputSpec(**out_data)
            outputs[out['logical_name']] = out_spec
        
        # Create StepSpecification using keyword arguments
        spec_data = {
            'step_type': spec_dict['step_type'],
            'node_type': spec_dict['node_type'],  # Keep as string, validator will convert
            'dependencies': dependencies,
            'outputs': outputs
        }
        return StepSpecification(**spec_data)
    
    def get_dependency_resolution_report(self, all_specs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed dependency resolution report using production resolver."""
        self._populate_resolver_registry(all_specs)
        
        # Convert file-based spec names to canonical names for the report
        available_steps = []
        for spec_name in all_specs.keys():
            try:
                canonical_name = self._get_canonical_step_name(spec_name)
                available_steps.append(canonical_name)
            except Exception as e:
                logger.warning(f"Could not get canonical name for {spec_name}: {e}")
                available_steps.append(spec_name)  # Fallback to file name
        
        return self.dependency_resolver.get_resolution_report(available_steps)
    
    def _is_compatible_output(self, required_logical_name: str, output_logical_name: str) -> bool:
        """Check if an output logical name is compatible with a required logical name using flexible matching."""
        if not required_logical_name or not output_logical_name:
            return False
        
        # Exact match
        if required_logical_name == output_logical_name:
            return True
        
        # Common data input/output patterns
        data_patterns = {
            'data_input': ['processed_data', 'training_data', 'input_data', 'data', 'model_input_data'],
            'input_data': ['processed_data', 'training_data', 'data_input', 'data', 'model_input_data'],
            'training_data': ['processed_data', 'data_input', 'input_data', 'data', 'model_input_data'],
            'processed_data': ['data_input', 'input_data', 'training_data', 'data', 'model_input_data'],
            'model_input_data': ['processed_data', 'data_input', 'input_data', 'training_data', 'data'],
            'data': ['processed_data', 'data_input', 'input_data', 'training_data', 'model_input_data']
        }
        
        # Check if required name has compatible patterns
        compatible_outputs = data_patterns.get(required_logical_name.lower(), [])
        if output_logical_name.lower() in compatible_outputs:
            return True
        
        # Check reverse mapping
        for pattern_key, pattern_values in data_patterns.items():
            if output_logical_name.lower() == pattern_key and required_logical_name.lower() in pattern_values:
                return True
        
        return False
    
    def _validate_circular_dependencies(self, specification: Dict[str, Any], all_specs: Dict[str, Dict[str, Any]], spec_name: str) -> List[Dict[str, Any]]:
        """Validate that no circular dependencies exist."""
        issues = []
        
        # Build dependency graph
        dependency_graph = {}
        for spec_name_key, spec in all_specs.items():
            dependencies = []
            for dep in spec.get('dependencies', []):
                logical_name = dep.get('logical_name')
                if logical_name:
                    # Find which spec produces this logical name
                    for producer_name, producer_spec in all_specs.items():
                        if producer_name == spec_name_key:
                            continue
                        for output in producer_spec.get('outputs', []):
                            if output.get('logical_name') == logical_name:
                                dependencies.append(producer_name)
                                break
            dependency_graph[spec_name_key] = dependencies
        
        # Check for circular dependencies using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        if spec_name in dependency_graph and has_cycle(spec_name):
            issues.append({
                'severity': 'ERROR',
                'category': 'circular_dependencies',
                'message': f'Circular dependency detected involving {spec_name}',
                'details': {'specification': spec_name},
                'recommendation': 'Remove circular dependencies by restructuring the dependency chain'
            })
        
        return issues
    
    def _validate_dependency_data_types(self, specification: Dict[str, Any], all_specs: Dict[str, Dict[str, Any]], spec_name: str) -> List[Dict[str, Any]]:
        """Validate data type consistency across dependency chains."""
        issues = []
        
        dependencies = specification.get('dependencies', [])
        
        for dep in dependencies:
            logical_name = dep.get('logical_name')
            expected_type = dep.get('data_type')
            
            if not logical_name or not expected_type:
                continue
            
            # Find the producer of this logical name
            producer_type = None
            producer_spec_name = None
            
            for other_spec_name, other_spec in all_specs.items():
                if other_spec_name == spec_name:
                    continue
                
                for output in other_spec.get('outputs', []):
                    if output.get('logical_name') == logical_name:
                        producer_type = output.get('data_type')
                        producer_spec_name = other_spec_name
                        break
                
                if producer_type:
                    break
            
            # Check type consistency
            if producer_type and producer_type != expected_type:
                issues.append({
                    'severity': 'WARNING',
                    'category': 'data_type_consistency',
                    'message': f'Data type mismatch for {logical_name}: expected={expected_type}, producer={producer_type}',
                    'details': {
                        'logical_name': logical_name,
                        'expected_type': expected_type,
                        'producer_type': producer_type,
                        'consumer': spec_name,
                        'producer': producer_spec_name
                    },
                    'recommendation': f'Align data types for {logical_name} between producer and consumer'
                })
        
        return issues
    
    def _find_specification_files(self, spec_name: str) -> List[Path]:
        """
        Find all specification files for a specification using hybrid approach.
        PRIMARY: Direct file matching, FALLBACK: FlexibleFileResolver for fuzzy name matching.
        """
        spec_files = []
        
        # PRIMARY METHOD: Direct file matching
        direct_spec_file = self.specs_dir / f"{spec_name}_spec.py"
        if direct_spec_file.exists():
            spec_files.append(direct_spec_file)
            
            # Look for job type variants in the same directory
            for job_type in ['training', 'validation', 'testing', 'calibration']:
                variant_file = self.specs_dir / f"{spec_name}_{job_type}_spec.py"
                if variant_file.exists() and variant_file not in spec_files:
                    spec_files.append(variant_file)
        
        # FALLBACK METHOD: Use FlexibleFileResolver for fuzzy name matching
        if not spec_files:
            primary_spec = self.file_resolver.find_spec_file(spec_name)
            if primary_spec:
                spec_files.append(Path(primary_spec))
                
                # Look for job type variants in the same directory
                spec_dir = Path(primary_spec).parent
                base_name = Path(primary_spec).stem.replace('_spec', '')
                
                # Find all variants: {base_name}_{job_type}_spec.py
                for job_type in ['training', 'validation', 'testing', 'calibration']:
                    variant_file = spec_dir / f"{base_name}_{job_type}_spec.py"
                    if variant_file.exists() and variant_file not in spec_files:
                        spec_files.append(variant_file)
        
        return spec_files
    
    def _extract_job_type_from_spec_file(self, spec_file: Path) -> str:
        """Extract job type from specification file name."""
        # Pattern: {spec_name}_{job_type}_spec.py or {spec_name}_spec.py
        stem = spec_file.stem
        parts = stem.split('_')
        if len(parts) >= 3 and parts[-1] == 'spec':
            return parts[-2]  # job_type is second to last part
        return 'default'
    
    def _load_specification_from_python(self, spec_path: Path, spec_name: str, job_type: str) -> Dict[str, Any]:
        """Load specification from Python file using robust sys.path management (same approach as Level 1 & 2)."""
        try:
            # Add the project root to sys.path temporarily to handle relative imports
            # Go up to the project root (where src/ is located)
            project_root = str(spec_path.parent.parent.parent.parent)  # Go up to project root
            src_root = str(spec_path.parent.parent.parent)  # Go up to src/ level
            specs_dir = str(spec_path.parent)
            
            paths_to_add = [project_root, src_root, specs_dir]
            added_paths = []
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    added_paths.append(path)
            
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(f"{spec_path.stem}", spec_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load specification module from {spec_path}")
                
                module = importlib.util.module_from_spec(spec)
                
                # Set the module's package to handle relative imports
                module.__package__ = 'cursus.steps.specs'
                
                spec.loader.exec_module(module)
            finally:
                # Remove added paths from sys.path
                for path in added_paths:
                    if path in sys.path:
                        sys.path.remove(path)
            
            # Use job type-aware constant name resolution
            expected_constant = self.file_resolver.find_spec_constant_name(spec_name, job_type)
            
            # Try the expected constant name first
            possible_names = []
            if expected_constant:
                possible_names.append(expected_constant)
            
            # Add fallback patterns
            possible_names.extend([
                f"{spec_name.upper()}_{job_type.upper()}_SPEC",
                f"{spec_name.upper()}_SPEC",
                f"{job_type.upper()}_SPEC"
            ])
            
            # Add dynamic discovery - scan for any constants ending with _SPEC
            spec_constants = [name for name in dir(module) 
                            if name.endswith('_SPEC') and not name.startswith('_')]
            possible_names.extend(spec_constants)
            
            spec_obj = None
            for spec_var_name in possible_names:
                if hasattr(module, spec_var_name):
                    spec_obj = getattr(module, spec_var_name)
                    break
            
            if spec_obj is None:
                raise ValueError(f"No specification constant found in {spec_path}. Tried: {possible_names}")
            
            # Convert StepSpecification object to dictionary
            dependencies = []
            for dep_name, dep_spec in spec_obj.dependencies.items():
                dependencies.append({
                    'logical_name': dep_spec.logical_name,
                    'dependency_type': dep_spec.dependency_type.value if hasattr(dep_spec.dependency_type, 'value') else str(dep_spec.dependency_type),
                    'required': dep_spec.required,
                    'compatible_sources': dep_spec.compatible_sources,
                    'data_type': dep_spec.data_type,
                    'description': dep_spec.description
                })
            
            outputs = []
            for out_name, out_spec in spec_obj.outputs.items():
                outputs.append({
                    'logical_name': out_spec.logical_name,
                    'output_type': out_spec.output_type.value if hasattr(out_spec.output_type, 'value') else str(out_spec.output_type),
                    'property_path': out_spec.property_path,
                    'data_type': out_spec.data_type,
                    'description': out_spec.description
                })
            
            return {
                'step_type': spec_obj.step_type,
                'node_type': spec_obj.node_type.value if hasattr(spec_obj.node_type, 'value') else str(spec_obj.node_type),
                'dependencies': dependencies,
                'outputs': outputs
            }
                
        except Exception as e:
            # If we still can't load it, provide a more detailed error
            raise ValueError(f"Failed to load specification from {spec_path}: {str(e)}")
    
    def _load_all_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Load all specification files."""
        all_specs = {}
        
        if self.specs_dir.exists():
            for spec_file in self.specs_dir.glob("*_spec.py"):
                spec_name = spec_file.stem.replace('_spec', '')
                # Remove job type suffix if present
                parts = spec_name.split('_')
                if len(parts) > 1:
                    # Try to identify if last part is a job type
                    potential_job_types = ['training', 'validation', 'testing', 'calibration']
                    if parts[-1] in potential_job_types:
                        spec_name = '_'.join(parts[:-1])
                
                if spec_name not in all_specs:
                    try:
                        job_type = self._extract_job_type_from_spec_file(spec_file)
                        spec = self._load_specification_from_python(spec_file, spec_name, job_type)
                        all_specs[spec_name] = spec
                    except Exception:
                        # Skip files that can't be parsed
                        continue
        
        return all_specs
    
    def _discover_specifications(self) -> List[str]:
        """
        Discover all specification files in the specifications directory.
        
        Only includes specifications that have actual files (not derived base names).
        This prevents validation errors for non-existent specification files.
        """
        specifications = set()
        
        if self.specs_dir.exists():
            for spec_file in self.specs_dir.glob("*_spec.py"):
                if spec_file.name.startswith('__'):
                    continue
                    
                # Use the actual file name (without .py extension) as the spec name
                # This ensures we only validate specifications that actually exist
                spec_name = spec_file.stem.replace('_spec', '')
                specifications.add(spec_name)
        
        return sorted(list(specifications))
