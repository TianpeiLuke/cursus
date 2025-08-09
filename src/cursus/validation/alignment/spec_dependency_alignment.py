"""
Specification ↔ Dependencies Alignment Tester

Validates alignment between step specifications and their dependency declarations.
Ensures dependency chains are consistent and resolvable.
"""

import os
import json
from typing import Dict, List, Any, Optional, Set
from pathlib import Path


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
        spec_path = self.specs_dir / f"{spec_name}_spec.json"
        
        # Check if file exists
        if not spec_path.exists():
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'missing_file',
                    'message': f'Specification file not found: {spec_path}',
                    'recommendation': f'Create the specification file {spec_name}_spec.json'
                }]
            }
        
        # Load specification
        try:
            with open(spec_path, 'r') as f:
                specification = json.load(f)
        except Exception as e:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'spec_parse_error',
                    'message': f'Failed to parse specification: {str(e)}',
                    'recommendation': 'Fix JSON syntax in specification file'
                }]
            }
        
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
            
            # Check if dependency can be resolved
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
                    'message': f'Cannot resolve dependency: {logical_name}',
                    'details': {'logical_name': logical_name, 'specification': spec_name},
                    'recommendation': f'Create a step that produces output {logical_name} or remove dependency'
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
    
    def _load_all_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Load all specification files."""
        all_specs = {}
        
        if self.specs_dir.exists():
            for spec_file in self.specs_dir.glob("*_spec.json"):
                spec_name = spec_file.stem.replace('_spec', '')
                try:
                    with open(spec_file, 'r') as f:
                        all_specs[spec_name] = json.load(f)
                except Exception:
                    # Skip files that can't be parsed
                    continue
        
        return all_specs
    
    def _discover_specifications(self) -> List[str]:
        """Discover all specification files in the specifications directory."""
        specifications = []
        
        if self.specs_dir.exists():
            for spec_file in self.specs_dir.glob("*_spec.json"):
                spec_name = spec_file.stem.replace('_spec', '')
                specifications.append(spec_name)
        
        return sorted(specifications)
