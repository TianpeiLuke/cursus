"""
Specification ↔ Dependencies Alignment Tester

Validates alignment between step specifications and their dependency declarations.
Ensures dependency chains are consistent and resolvable.
"""

import os
import sys
import importlib.util
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from .alignment_utils import (
    FlexibleFileResolver, DependencyPatternClassifier, DependencyPattern
)


class SpecificationDependencyAlignmentTester:
    """
    Tests alignment between step specifications and their dependencies.
    
    Validates:
    - Dependency chains are consistent
    - All dependencies can be resolved
    - No circular dependencies exist
    - Data types match across dependency chains
    """
    
    def __init__(self, specs_dir: str):
        """
        Initialize the specification-dependency alignment tester.
        
        Args:
            specs_dir: Directory containing step specifications
        """
        self.specs_dir = Path(specs_dir)
        
        # Initialize the dependency pattern classifier
        self.dependency_classifier = DependencyPatternClassifier()
        
        # Initialize the file resolver for finding specification files
        base_directories = {
            'specs': str(self.specs_dir)
        }
        self.file_resolver = FlexibleFileResolver(base_directories)
    
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
        """Validate that all dependencies can be resolved."""
        issues = []
        
        dependencies = specification.get('dependencies', [])
        
        for dep in dependencies:
            logical_name = dep.get('logical_name')
            if not logical_name:
                continue
            
            # Use the new dependency pattern classifier
            dependency_pattern = self.dependency_classifier.classify_dependency(dep)
            
            # Check if dependency can be resolved based on pattern
            if dependency_pattern == DependencyPattern.EXTERNAL_INPUT:
                # External dependencies don't need to be resolved within pipeline
                continue
            elif dependency_pattern == DependencyPattern.CONFIGURATION:
                # Configuration dependencies are handled by the config system
                continue
            elif dependency_pattern == DependencyPattern.ENVIRONMENT:
                # Environment dependencies are handled by environment variables
                continue
            elif dependency_pattern == DependencyPattern.PIPELINE_DEPENDENCY:
                # Pipeline dependencies must be resolved by other steps
                resolved = False
                for other_spec_name, other_spec in all_specs.items():
                    if other_spec_name == spec_name:
                        continue
                    
                    # Check if other spec produces this logical name
                    for output in other_spec.get('outputs', []):
                        if output.get('logical_name') == logical_name:
                            resolved = True
                            break
                    
                    if resolved:
                        break
                
                if not resolved:
                    issues.append({
                        'severity': 'ERROR',
                        'category': 'dependency_resolution',
                        'message': f'Cannot resolve pipeline dependency: {logical_name}',
                        'details': {
                            'logical_name': logical_name, 
                            'specification': spec_name,
                            'pattern': dependency_pattern.value
                        },
                        'recommendation': f'Create a step that produces output {logical_name} or change to external dependency'
                    })
            else:
                # This shouldn't happen with the new classifier, but handle gracefully
                issues.append({
                    'severity': 'WARNING',
                    'category': 'dependency_classification',
                    'message': f'Unknown dependency pattern for: {logical_name}',
                    'details': {
                        'logical_name': logical_name,
                        'specification': spec_name,
                        'dependency_type': dep.get('dependency_type'),
                        'compatible_sources': dep.get('compatible_sources'),
                        'pattern': str(dependency_pattern)
                    },
                    'recommendation': f'Review dependency configuration for {logical_name}'
                })
        
        return issues
    
    
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
        """Find all specification files for a specification."""
        spec_files = []
        if self.specs_dir.exists():
            # Look for files matching pattern: {spec_name}_{job_type}_spec.py or {spec_name}_spec.py
            for spec_file in self.specs_dir.glob(f"{spec_name}_*_spec.py"):
                spec_files.append(spec_file)
            
            # Also look for simple pattern: {spec_name}_spec.py
            simple_spec = self.specs_dir / f"{spec_name}_spec.py"
            if simple_spec.exists() and simple_spec not in spec_files:
                spec_files.append(simple_spec)
        
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
        """Load specification from Python file."""
        try:
            # Read the file content and modify imports to be absolute
            with open(spec_path, 'r') as f:
                content = f.read()
            
            # Replace common relative imports with absolute imports
            modified_content = content.replace(
                'from ...core.base.step_specification import StepSpecification',
                'from src.cursus.core.base.step_specification import StepSpecification'
            ).replace(
                'from ...core.base.dependency_specification import DependencySpecification',
                'from src.cursus.core.base.dependency_specification import DependencySpecification'
            ).replace(
                'from ...core.base.output_specification import OutputSpecification',
                'from src.cursus.core.base.output_specification import OutputSpecification'
            ).replace(
                'from ...core.base.enums import',
                'from src.cursus.core.base.enums import'
            ).replace(
                'from ...core.base.specification_base import',
                'from src.cursus.core.base.specification_base import'
            ).replace(
                'from ..registry.step_names import',
                'from src.cursus.steps.registry.step_names import'
            ).replace(
                'from ..contracts.',
                'from src.cursus.steps.contracts.'
            )
            
            # Add the project root to sys.path
            project_root = self.specs_dir.parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            try:
                # Create a temporary module from the modified content
                module_name = f"{spec_name}_{job_type}_spec_temp"
                spec = importlib.util.spec_from_loader(module_name, loader=None)
                module = importlib.util.module_from_spec(spec)
                
                # Execute the modified content in the module's namespace
                exec(modified_content, module.__dict__)
                
                # Look for the specification constant
                possible_names = [
                    f"{spec_name.upper()}_{job_type.upper()}_SPEC",
                    f"{spec_name.upper()}_SPEC",
                    f"{job_type.upper()}_SPEC"
                ]
                
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
                
            finally:
                # Clean up sys.path
                if str(project_root) in sys.path:
                    sys.path.remove(str(project_root))
                    
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
        """Discover all specification files in the specifications directory."""
        specifications = set()
        
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
                
                specifications.add(spec_name)
        
        return sorted(list(specifications))
