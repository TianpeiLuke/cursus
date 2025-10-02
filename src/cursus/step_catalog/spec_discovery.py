"""
Specification class auto-discovery for the unified step catalog system.

This module implements AST-based specification discovery from both core
and workspace directories, following the same pattern as other discovery components.
"""

import ast
import importlib
import logging
from pathlib import Path
from typing import Dict, Type, Optional, Any, List, Union

logger = logging.getLogger(__name__)


class SpecAutoDiscovery:
    """Specification class auto-discovery following the established discovery pattern."""
    
    def __init__(self, package_root: Path, workspace_dirs: List[Path]):
        """
        Initialize spec auto-discovery with dual search space support.
        
        Args:
            package_root: Root of the cursus package
            workspace_dirs: List of workspace directories to search
        """
        self.package_root = package_root
        self.workspace_dirs = workspace_dirs
        self.logger = logging.getLogger(__name__)
    
    def discover_spec_classes(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Auto-discover specification instances from package and workspace directories.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Dictionary mapping spec names to specification instances
        """
        discovered_specs = {}
        
        # Always scan package core specs
        core_spec_dir = self.package_root / "steps" / "specs"
        if core_spec_dir.exists():
            try:
                core_specs = self._scan_spec_directory(core_spec_dir)
                discovered_specs.update(core_specs)
                self.logger.info(f"Discovered {len(core_specs)} core specification instances")
            except Exception as e:
                self.logger.error(f"Error scanning core spec directory: {e}")
        
        # Scan workspace specs if workspace directories provided
        if self.workspace_dirs:
            for workspace_dir in self.workspace_dirs:
                try:
                    workspace_specs = self._discover_workspace_specs(workspace_dir, project_id)
                    # Workspace specs override core specs with same names
                    discovered_specs.update(workspace_specs)
                except Exception as e:
                    self.logger.error(f"Error scanning workspace spec directory {workspace_dir}: {e}")
        
        return discovered_specs
    
    def load_spec_class(self, step_name: str) -> Optional[Any]:
        """
        Load specification instance for a given step name.
        
        Args:
            step_name: Name of the step to load specification for
            
        Returns:
            Specification instance if found, None otherwise
        """
        try:
            # First try direct import using step name patterns
            spec_instance = self._try_direct_import(step_name)
            if spec_instance:
                return spec_instance
            
            # Try workspace discovery if available
            if self.workspace_dirs:
                for workspace_dir in self.workspace_dirs:
                    spec_instance = self._try_workspace_spec_import(step_name, workspace_dir)
                    if spec_instance:
                        return spec_instance
            
            self.logger.debug(f"No specification found for step: {step_name}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Error loading specification for step {step_name}: {e}")
            return None
    
    def _try_direct_import(self, step_name: str) -> Optional[Any]:
        """Try to import specification directly from package."""
        try:
            # Convert step name to spec module name patterns
            # Handle both CamelCase (XGBoostModel) and snake_case (xgboost_model) inputs
            step_name_lower = step_name.lower()
            
            # Convert CamelCase to snake_case if needed
            import re
            snake_case_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', step_name).lower()
            
            spec_module_patterns = [
                f"{step_name_lower}_spec",
                f"{snake_case_name}_spec",
                f"{step_name_lower}_model_spec", 
                f"{snake_case_name}_model_spec",
                f"{step_name_lower}_training_spec",
                f"{snake_case_name}_training_spec",
                f"{step_name_lower}_validation_spec",
                f"{snake_case_name}_validation_spec",
                f"{step_name_lower}_testing_spec",
                f"{snake_case_name}_testing_spec",
                f"{step_name_lower}_calibration_spec",
                f"{snake_case_name}_calibration_spec"
            ]
            
            # Remove duplicates while preserving order
            seen = set()
            spec_module_patterns = [x for x in spec_module_patterns if not (x in seen or seen.add(x))]
            
            for module_name in spec_module_patterns:
                try:
                    relative_module_path = f"..steps.specs.{module_name}"
                    module = importlib.import_module(relative_module_path, package=__package__)
                    
                    # Look for spec instances in the module
                    spec_instance = self._extract_spec_from_module(module, step_name)
                    if spec_instance:
                        self.logger.debug(f"Found specification for {step_name} in {module_name}")
                        return spec_instance
                        
                except ImportError:
                    continue
                except Exception as e:
                    self.logger.debug(f"Error importing {module_name}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error in direct import for {step_name}: {e}")
            return None
    
    def _extract_spec_from_module(self, module: Any, step_name: str) -> Optional[Any]:
        """Extract specification instance from a module."""
        # Common spec variable naming patterns
        spec_var_patterns = [
            f"{step_name.upper()}_SPEC",
            f"{step_name.upper()}_MODEL_SPEC",
            f"{step_name.upper()}_TRAINING_SPEC",
            "SPEC",
            "spec"
        ]
        
        for var_name in spec_var_patterns:
            if hasattr(module, var_name):
                spec_instance = getattr(module, var_name)
                # Verify it's a specification instance
                if self._is_spec_instance(spec_instance):
                    return spec_instance
        
        # Look for any StepSpecification instances in the module
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                attr = getattr(module, attr_name)
                if self._is_spec_instance(attr):
                    return attr
        
        return None
    
    def _is_spec_instance(self, obj: Any) -> bool:
        """Check if an object is a specification instance."""
        try:
            # Check if it has the expected attributes of a StepSpecification
            return (hasattr(obj, 'step_type') and 
                   hasattr(obj, 'dependencies') and 
                   hasattr(obj, 'outputs'))
        except Exception:
            return False
    
    def _scan_spec_directory(self, spec_dir: Path) -> Dict[str, Any]:
        """
        Scan directory for specification instances using AST parsing.
        
        Args:
            spec_dir: Directory to scan for spec files
            
        Returns:
            Dictionary mapping spec names to specification instances
        """
        spec_instances = {}
        
        try:
            for py_file in spec_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    # Parse file with AST to find spec variables
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    tree = ast.parse(source, filename=str(py_file))
                    
                    # Find spec assignments in the AST
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign) and self._is_spec_assignment(node):
                            try:
                                # Import the module using relative import pattern
                                relative_module_path = self._file_to_relative_module_path(py_file)
                                if relative_module_path:
                                    module = importlib.import_module(relative_module_path, package=__package__)
                                    
                                    # Extract spec instances from the module
                                    for target in node.targets:
                                        if isinstance(target, ast.Name):
                                            var_name = target.id
                                            if hasattr(module, var_name):
                                                spec_instance = getattr(module, var_name)
                                                if self._is_spec_instance(spec_instance):
                                                    # Use file name as key (without .py extension)
                                                    spec_key = py_file.stem
                                                    spec_instances[spec_key] = spec_instance
                                                    self.logger.debug(f"Found spec instance: {var_name} in {py_file}")
                                else:
                                    self.logger.warning(f"Could not determine relative module path for {py_file}")
                            except Exception as e:
                                self.logger.warning(f"Error importing spec from {py_file}: {e}")
                                continue
                
                except Exception as e:
                    self.logger.warning(f"Error processing spec file {py_file}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error scanning spec directory {spec_dir}: {e}")
        
        return spec_instances
    
    def _is_spec_assignment(self, assign_node: ast.Assign) -> bool:
        """
        Check if an assignment node is likely a spec assignment.
        
        Args:
            assign_node: AST assignment node
            
        Returns:
            True if the assignment appears to be a specification
        """
        # Check if the assignment target ends with _SPEC
        for target in assign_node.targets:
            if isinstance(target, ast.Name) and target.id.endswith('_SPEC'):
                return True
        
        # Check if the value is a StepSpecification constructor call
        if isinstance(assign_node.value, ast.Call):
            if isinstance(assign_node.value.func, ast.Name):
                if assign_node.value.func.id == 'StepSpecification':
                    return True
        
        return False
    
    def _discover_workspace_specs(self, workspace_dir: Path, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Discover specification instances in a workspace directory with simplified structure."""
        discovered = {}
        
        # Simplified structure: workspace_dir directly contains specs/
        spec_dir = workspace_dir / "specs"
        if spec_dir.exists():
            discovered.update(self._scan_spec_directory(spec_dir))
        
        return discovered
    
    def _try_workspace_spec_import(self, step_name: str, workspace_dir: Path) -> Optional[Any]:
        """Try to import specification from workspace directory with simplified structure."""
        try:
            # Simplified structure: workspace_dir directly contains specs/
            spec_dir = workspace_dir / "specs"
            if not spec_dir.exists():
                return None
            
            # Look for spec files matching the step name
            spec_patterns = [
                f"{step_name.lower()}_spec.py",
                f"{step_name.lower()}_model_spec.py",
                f"{step_name.lower()}_training_spec.py"
            ]
            
            for pattern in spec_patterns:
                spec_file = spec_dir / pattern
                if spec_file.exists():
                    # Use file-based loading for workspace specs
                    spec_instance = self._load_spec_from_file(spec_file, step_name)
                    if spec_instance:
                        return spec_instance
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error loading workspace spec for {step_name}: {e}")
            return None
    
    def _load_spec_from_file(self, spec_file: Path, step_name: str) -> Optional[Any]:
        """Load specification instance from a specific file."""
        try:
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("spec_module", spec_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Extract spec instance from the module
                return self._extract_spec_from_module(module, step_name)
                
        except Exception as e:
            self.logger.warning(f"Error loading spec from file {spec_file}: {e}")
            return None
    
    def find_specs_by_contract(self, contract_name: str) -> Dict[str, Any]:
        """
        Find all specifications that reference a specific contract.
        
        This method enables contract-specification alignment validation by finding
        specifications that are associated with a given contract name.
        
        Args:
            contract_name: Name of the contract to find specifications for
            
        Returns:
            Dictionary mapping spec names to specification instances
        """
        try:
            matching_specs = {}
            
            # Search core package specs
            core_spec_dir = self.package_root / "steps" / "specs"
            if core_spec_dir.exists():
                core_matches = self._find_specs_by_contract_in_dir(core_spec_dir, contract_name)
                matching_specs.update(core_matches)
            
            # Search workspace specs
            if self.workspace_dirs:
                for workspace_dir in self.workspace_dirs:
                    workspace_matches = self._find_specs_by_contract_in_workspace(workspace_dir, contract_name)
                    matching_specs.update(workspace_matches)
            
            self.logger.debug(f"Found {len(matching_specs)} specifications for contract '{contract_name}'")
            return matching_specs
            
        except Exception as e:
            self.logger.error(f"Error finding specs for contract {contract_name}: {e}")
            return {}
    
    def serialize_spec(self, spec_instance: Any) -> Dict[str, Any]:
        """
        Convert specification instance to dictionary format.
        
        This method provides standardized serialization of StepSpecification objects
        for use in validation and alignment testing.
        
        Args:
            spec_instance: StepSpecification instance to serialize
            
        Returns:
            Dictionary representation of the specification
        """
        try:
            if not self._is_spec_instance(spec_instance):
                raise ValueError("Object is not a valid specification instance")
            
            # Serialize dependencies
            dependencies = []
            if hasattr(spec_instance, 'dependencies') and spec_instance.dependencies:
                for dep_name, dep_spec in spec_instance.dependencies.items():
                    dependencies.append({
                        "logical_name": dep_spec.logical_name,
                        "dependency_type": (
                            dep_spec.dependency_type.value
                            if hasattr(dep_spec.dependency_type, "value")
                            else str(dep_spec.dependency_type)
                        ),
                        "required": dep_spec.required,
                        "compatible_sources": dep_spec.compatible_sources,
                        "data_type": dep_spec.data_type,
                        "description": dep_spec.description,
                    })
            
            # Serialize outputs
            outputs = []
            if hasattr(spec_instance, 'outputs') and spec_instance.outputs:
                for out_name, out_spec in spec_instance.outputs.items():
                    outputs.append({
                        "logical_name": out_spec.logical_name,
                        "output_type": (
                            out_spec.output_type.value
                            if hasattr(out_spec.output_type, "value")
                            else str(out_spec.output_type)
                        ),
                        "property_path": out_spec.property_path,
                        "data_type": out_spec.data_type,
                        "description": out_spec.description,
                    })
            
            return {
                "step_type": spec_instance.step_type,
                "node_type": (
                    spec_instance.node_type.value
                    if hasattr(spec_instance.node_type, "value")
                    else str(spec_instance.node_type)
                ),
                "dependencies": dependencies,
                "outputs": outputs,
            }
            
        except Exception as e:
            self.logger.error(f"Error serializing specification: {e}")
            return {}
    
    def load_all_specifications(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all specification instances from both package and workspace directories.
        
        This method provides comprehensive specification loading for validation frameworks
        and dependency analysis tools. It discovers and loads all available specifications,
        serializing them to dictionary format for easy consumption.
        
        Returns:
            Dictionary mapping step names to serialized specification dictionaries
        """
        try:
            all_specs = {}
            
            # Discover all specification instances
            discovered_specs = self.discover_spec_classes()
            
            # Serialize each specification to dictionary format
            for spec_name, spec_instance in discovered_specs.items():
                try:
                    if self._is_spec_instance(spec_instance):
                        serialized_spec = self.serialize_spec(spec_instance)
                        if serialized_spec:
                            all_specs[spec_name] = serialized_spec
                            self.logger.debug(f"Loaded and serialized specification: {spec_name}")
                    else:
                        self.logger.warning(f"Invalid specification instance for {spec_name}")
                        
                except Exception as e:
                    self.logger.warning(f"Error serializing specification {spec_name}: {e}")
                    continue
            
            self.logger.info(f"Successfully loaded {len(all_specs)} specifications")
            return all_specs
            
        except Exception as e:
            self.logger.error(f"Error loading all specifications: {e}")
            return {}
    
    def get_job_type_variants(self, base_step_name: str) -> List[str]:
        """
        Get all job type variants for a base step name.
        
        This method discovers different job type variants (training, validation, testing, etc.)
        for a given base step name by examining specification file naming patterns.
        
        Args:
            base_step_name: Base name of the step
            
        Returns:
            List of job type variants found
        """
        try:
            variants = []
            base_name_lower = base_step_name.lower()
            
            # Search core package specs
            core_spec_dir = self.package_root / "steps" / "specs"
            if core_spec_dir.exists():
                core_variants = self._find_job_type_variants_in_dir(core_spec_dir, base_name_lower)
                variants.extend(core_variants)
            
            # Search workspace specs
            if self.workspace_dirs:
                for workspace_dir in self.workspace_dirs:
                    workspace_variants = self._find_job_type_variants_in_workspace(workspace_dir, base_name_lower)
                    variants.extend(workspace_variants)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_variants = []
            for variant in variants:
                if variant not in seen:
                    seen.add(variant)
                    unique_variants.append(variant)
            
            self.logger.debug(f"Found job type variants for '{base_step_name}': {unique_variants}")
            return unique_variants
            
        except Exception as e:
            self.logger.error(f"Error finding job type variants for {base_step_name}: {e}")
            return []
    
    def _find_specs_by_contract_in_dir(self, spec_dir: Path, contract_name: str) -> Dict[str, Any]:
        """Find specifications that reference a contract in a specific directory."""
        matching_specs = {}
        
        try:
            for py_file in spec_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    if self._spec_file_references_contract(py_file, contract_name):
                        # Load the specification from this file
                        spec_instance = self._load_spec_from_file(py_file, contract_name)
                        if spec_instance:
                            spec_key = py_file.stem
                            matching_specs[spec_key] = spec_instance
                            self.logger.debug(f"Found matching spec: {spec_key} for contract {contract_name}")
                
                except Exception as e:
                    self.logger.warning(f"Error checking spec file {py_file} for contract {contract_name}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error scanning directory {spec_dir} for contract {contract_name}: {e}")
        
        return matching_specs
    
    def _find_specs_by_contract_in_workspace(self, workspace_dir: Path, contract_name: str) -> Dict[str, Any]:
        """Find specifications that reference a contract in workspace directories with simplified structure."""
        matching_specs = {}
        
        try:
            # Simplified structure: workspace_dir directly contains specs/
            spec_dir = workspace_dir / "specs"
            if spec_dir.exists():
                workspace_matches = self._find_specs_by_contract_in_dir(spec_dir, contract_name)
                matching_specs.update(workspace_matches)
        
        except Exception as e:
            self.logger.error(f"Error scanning workspace {workspace_dir} for contract {contract_name}: {e}")
        
        return matching_specs
    
    def _spec_file_references_contract(self, spec_file: Path, contract_name: str) -> bool:
        """Check if a specification file references a specific contract."""
        try:
            # Use naming convention approach as the primary method
            spec_name = spec_file.stem.replace("_spec", "")
            
            # Remove job type suffix if present
            parts = spec_name.split("_")
            if len(parts) > 1:
                potential_job_types = ["training", "validation", "testing", "calibration", "model"]
                if parts[-1] in potential_job_types:
                    spec_name = "_".join(parts[:-1])
            
            contract_base = contract_name.lower().replace("_contract", "")
            
            # Check if the step type matches the contract name
            if contract_base in spec_name.lower() or spec_name.lower() in contract_base:
                return True
            
            # Additional check: look for contract references in the file content
            try:
                with open(spec_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Simple string search for contract references
                    if contract_name.lower() in content.lower() or contract_base in content.lower():
                        return True
            except Exception:
                pass  # If file reading fails, rely on naming convention
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking if {spec_file} references contract {contract_name}: {e}")
            return False
    
    def _find_job_type_variants_in_dir(self, spec_dir: Path, base_name_lower: str) -> List[str]:
        """Find job type variants in a specific directory."""
        variants = []
        
        try:
            for py_file in spec_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                spec_name = py_file.stem.replace("_spec", "")
                
                # Check if this spec file matches the base step name
                if base_name_lower in spec_name.lower():
                    # Extract potential job type
                    parts = spec_name.split("_")
                    if len(parts) > 1:
                        potential_job_type = parts[-1].lower()
                        known_job_types = ["training", "validation", "testing", "calibration", "model"]
                        if potential_job_type in known_job_types:
                            variants.append(potential_job_type)
                        else:
                            variants.append("default")
                    else:
                        variants.append("default")
        
        except Exception as e:
            self.logger.error(f"Error finding job type variants in {spec_dir}: {e}")
        
        return variants
    
    def _find_job_type_variants_in_workspace(self, workspace_dir: Path, base_name_lower: str) -> List[str]:
        """Find job type variants in workspace directories with simplified structure."""
        variants = []
        
        try:
            # Simplified structure: workspace_dir directly contains specs/
            spec_dir = workspace_dir / "specs"
            if spec_dir.exists():
                workspace_variants = self._find_job_type_variants_in_dir(spec_dir, base_name_lower)
                variants.extend(workspace_variants)
        
        except Exception as e:
            self.logger.error(f"Error finding job type variants in workspace {workspace_dir}: {e}")
        
        return variants

    # PHASE 2 ENHANCEMENT: Smart Specification Integration
    def create_unified_specification(self, contract_name: str) -> Dict[str, Any]:
        """
        Create unified specification from multiple variants using smart selection.
        
        Integrates SmartSpecificationSelector logic:
        - Multi-variant specification discovery using existing find_specs_by_contract()
        - Union of dependencies and outputs from all variants
        - Smart validation logic with detailed feedback
        - Primary specification selection (training > generic > first available)
        
        Args:
            contract_name: Name of the contract to find specifications for
            
        Returns:
            Unified specification model with metadata
        """
        try:
            # Use existing find_specs_by_contract method
            specifications = self.find_specs_by_contract(contract_name)
            
            if not specifications:
                return {
                    "primary_spec": {},
                    "variants": {},
                    "unified_dependencies": {},
                    "unified_outputs": {},
                    "dependency_sources": {},
                    "output_sources": {},
                    "variant_count": 0,
                }
            
            # Apply smart specification logic
            return self._apply_smart_specification_logic(specifications, contract_name)
            
        except Exception as e:
            self.logger.error(f"Error creating unified specification for contract {contract_name}: {e}")
            return {
                "primary_spec": {},
                "variants": {},
                "unified_dependencies": {},
                "unified_outputs": {},
                "dependency_sources": {},
                "output_sources": {},
                "variant_count": 0,
            }
    
    def validate_logical_names_smart(self, contract: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
        """
        Smart validation using multi-variant specification logic.
        
        Implements the core Smart Specification Selection validation:
        - Contract input is valid if it exists in ANY variant
        - Contract must cover intersection of REQUIRED dependencies
        - Provides detailed feedback about which variants need what
        
        Args:
            contract: Contract dictionary
            contract_name: Name of the contract
            
        Returns:
            List of validation issues
        """
        try:
            # Create unified specification
            unified_spec = self.create_unified_specification(contract_name)
            
            # Apply smart validation logic
            return self._validate_smart_logical_names(contract, unified_spec, contract_name)
            
        except Exception as e:
            self.logger.error(f"Error in smart validation for contract {contract_name}: {e}")
            return [
                {
                    "severity": "ERROR",
                    "category": "smart_validation_error",
                    "message": f"Smart validation failed for contract {contract_name}: {str(e)}",
                    "details": {"contract": contract_name, "error": str(e)},
                    "recommendation": "Check contract and specification files for errors",
                }
            ]
    
    def _apply_smart_specification_logic(self, specifications: Dict[str, Any], contract_name: str) -> Dict[str, Any]:
        """
        Apply smart specification selection logic to create unified specification.
        
        Args:
            specifications: Dictionary of loaded specifications
            contract_name: Name of the contract being validated
            
        Returns:
            Unified specification model with metadata
        """
        try:
            # Group specifications by job type using registry patterns instead of hardcoded logic
            variants = {}
            
            # Categorize specifications by job type
            for spec_key, spec_data in specifications.items():
                job_type = self._extract_job_type_from_spec_name_registry(spec_key)
                variants[job_type] = spec_data
            
            # Create unified dependency and output sets
            unified_dependencies = {}
            unified_outputs = {}
            dependency_sources = {}  # Track which variants contribute each dependency
            output_sources = {}  # Track which variants contribute each output
            
            # Union all dependencies from all variants
            for variant_name, spec_data in variants.items():
                for dep in spec_data.get("dependencies", []):
                    logical_name = dep.get("logical_name")
                    if logical_name:
                        unified_dependencies[logical_name] = dep
                        if logical_name not in dependency_sources:
                            dependency_sources[logical_name] = []
                        dependency_sources[logical_name].append(variant_name)
                
                for output in spec_data.get("outputs", []):
                    logical_name = output.get("logical_name")
                    if logical_name:
                        unified_outputs[logical_name] = output
                        if logical_name not in output_sources:
                            output_sources[logical_name] = []
                        output_sources[logical_name].append(variant_name)
            
            # Select primary specification (prefer training, then generic, then first available)
            primary_spec = self._select_primary_specification(variants)
            
            return {
                "primary_spec": primary_spec,
                "variants": variants,
                "unified_dependencies": unified_dependencies,
                "unified_outputs": unified_outputs,
                "dependency_sources": dependency_sources,
                "output_sources": output_sources,
                "variant_count": len(variants),
            }
            
        except Exception as e:
            self.logger.error(f"Error applying smart specification logic: {e}")
            return {
                "primary_spec": {},
                "variants": {},
                "unified_dependencies": {},
                "unified_outputs": {},
                "dependency_sources": {},
                "output_sources": {},
                "variant_count": 0,
            }
    
    def _extract_job_type_from_spec_name_registry(self, spec_name: str) -> str:
        """
        Extract job type from specification name using registry patterns instead of hardcoded logic.
        
        Args:
            spec_name: Name of the specification
            
        Returns:
            Job type string
        """
        try:
            # Use registry-based job type detection instead of hardcoded patterns
            from ...registry.step_names import get_spec_step_type_with_job_type
            
            # Try to extract job type using registry patterns
            spec_name_lower = spec_name.lower()
            
            # Check for known job type patterns in the spec name
            known_job_types = ["training", "validation", "testing", "calibration", "model"]
            
            for job_type in known_job_types:
                if job_type in spec_name_lower:
                    return job_type
            
            # Default to generic if no specific job type found
            return "generic"
            
        except Exception as e:
            self.logger.warning(f"Error extracting job type from spec name {spec_name}: {e}")
            return "generic"
    
    def _select_primary_specification(self, variants: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the primary specification from available variants.
        
        Priority order:
        1. training (most common and comprehensive)
        2. generic (applies to all job types)
        3. first available variant
        
        Args:
            variants: Dictionary of specification variants
            
        Returns:
            Primary specification dictionary
        """
        if "training" in variants:
            return variants["training"]
        elif "generic" in variants:
            return variants["generic"]
        else:
            return next(iter(variants.values())) if variants else {}
    
    def _validate_smart_logical_names(self, contract: Dict[str, Any], unified_spec: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
        """
        Smart validation of logical names using multi-variant specification logic.
        
        This implements the core Smart Specification Selection validation:
        - Contract input is valid if it exists in ANY variant
        - Contract must cover intersection of REQUIRED dependencies
        - Provides detailed feedback about which variants need what
        
        Args:
            contract: Contract dictionary
            unified_spec: Unified specification model
            contract_name: Name of the contract
            
        Returns:
            List of validation issues
        """
        issues = []
        
        try:
            # Get logical names from contract
            contract_inputs = set(contract.get("inputs", {}).keys())
            contract_outputs = set(contract.get("outputs", {}).keys())
            
            # Get unified logical names from all specification variants
            unified_dependencies = unified_spec.get("unified_dependencies", {})
            unified_outputs = unified_spec.get("unified_outputs", {})
            dependency_sources = unified_spec.get("dependency_sources", {})
            output_sources = unified_spec.get("output_sources", {})
            variants = unified_spec.get("variants", {})
            
            # SMART VALIDATION LOGIC
            
            # 1. Check contract inputs against unified dependencies
            unified_dep_names = set(unified_dependencies.keys())
            
            # Contract inputs that are not in ANY variant are errors
            invalid_inputs = contract_inputs - unified_dep_names
            for logical_name in invalid_inputs:
                issues.append({
                    "severity": "ERROR",
                    "category": "logical_names",
                    "message": f"Contract input {logical_name} not declared in any specification variant",
                    "details": {
                        "logical_name": logical_name,
                        "contract": contract_name,
                        "available_variants": list(variants.keys()),
                        "available_dependencies": list(unified_dep_names),
                    },
                    "recommendation": f"Add {logical_name} to specification dependencies or remove from contract",
                })
            
            # 2. Check for required dependencies that contract doesn't provide
            required_deps = set()
            optional_deps = set()
            
            for dep_name, dep_spec in unified_dependencies.items():
                if dep_spec.get("required", False):
                    required_deps.add(dep_name)
                else:
                    optional_deps.add(dep_name)
            
            missing_required = required_deps - contract_inputs
            for logical_name in missing_required:
                # Find which variants require this dependency
                requiring_variants = dependency_sources.get(logical_name, [])
                issues.append({
                    "severity": "ERROR",
                    "category": "logical_names",
                    "message": f"Contract missing required dependency {logical_name}",
                    "details": {
                        "logical_name": logical_name,
                        "contract": contract_name,
                        "requiring_variants": requiring_variants,
                    },
                    "recommendation": f'Add {logical_name} to contract inputs (required by variants: {", ".join(requiring_variants)})',
                })
            
            # 3. Provide informational feedback for valid optional inputs
            valid_optional_inputs = contract_inputs & optional_deps
            for logical_name in valid_optional_inputs:
                supporting_variants = dependency_sources.get(logical_name, [])
                if len(supporting_variants) < len(variants):
                    # This input is only used by some variants - provide info
                    issues.append({
                        "severity": "INFO",
                        "category": "logical_names",
                        "message": f'Contract input {logical_name} used by variants: {", ".join(supporting_variants)}',
                        "details": {
                            "logical_name": logical_name,
                            "contract": contract_name,
                            "supporting_variants": supporting_variants,
                            "total_variants": len(variants),
                        },
                        "recommendation": f"Input {logical_name} is correctly declared for multi-variant support",
                    })
            
            # 4. Check contract outputs against unified outputs
            unified_output_names = set(unified_outputs.keys())
            
            # Contract outputs that are not in ANY variant are errors
            invalid_outputs = contract_outputs - unified_output_names
            for logical_name in invalid_outputs:
                issues.append({
                    "severity": "ERROR",
                    "category": "logical_names",
                    "message": f"Contract output {logical_name} not declared in any specification variant",
                    "details": {
                        "logical_name": logical_name,
                        "contract": contract_name,
                        "available_variants": list(variants.keys()),
                        "available_outputs": list(unified_output_names),
                    },
                    "recommendation": f"Add {logical_name} to specification outputs or remove from contract",
                })
            
            # 5. Check for missing outputs (less critical since outputs are usually consistent)
            missing_outputs = unified_output_names - contract_outputs
            for logical_name in missing_outputs:
                producing_variants = output_sources.get(logical_name, [])
                issues.append({
                    "severity": "WARNING",
                    "category": "logical_names",
                    "message": f"Contract missing output {logical_name}",
                    "details": {
                        "logical_name": logical_name,
                        "contract": contract_name,
                        "producing_variants": producing_variants,
                    },
                    "recommendation": f'Add {logical_name} to contract outputs (produced by variants: {", ".join(producing_variants)})',
                })
            
            # 6. Add summary information about multi-variant validation
            if len(variants) > 1:
                issues.append({
                    "severity": "INFO",
                    "category": "multi_variant_validation",
                    "message": f"Smart Specification Selection: validated against {len(variants)} variants",
                    "details": {
                        "contract": contract_name,
                        "variants": list(variants.keys()),
                        "total_dependencies": len(unified_dependencies),
                        "total_outputs": len(unified_outputs),
                        "contract_inputs": len(contract_inputs),
                        "contract_outputs": len(contract_outputs),
                    },
                    "recommendation": "Multi-variant validation completed successfully",
                })
            
        except Exception as e:
            self.logger.error(f"Error in smart logical names validation: {e}")
            issues.append({
                "severity": "ERROR",
                "category": "smart_validation_error",
                "message": f"Smart validation logic failed: {str(e)}",
                "details": {"contract": contract_name, "error": str(e)},
                "recommendation": "Check specification and contract files for structural issues",
            })
        
        return issues

    def _file_to_relative_module_path(self, file_path: Path) -> Optional[str]:
        """
        Convert file path to relative module path for use with importlib.import_module.
        
        This creates relative import paths like "..steps.specs.spec_name"
        that work with the package parameter in importlib.import_module.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Relative module path string or None if conversion fails
        """
        try:
            # Get the path relative to the package root
            try:
                relative_path = file_path.relative_to(self.package_root)
            except ValueError:
                # File is not under package root, might be in workspace
                self.logger.debug(f"File {file_path} not under package root {self.package_root}")
                return None
            
            # Convert path to module format
            parts = list(relative_path.parts)
            
            # Remove .py extension from the last part
            if parts[-1].endswith('.py'):
                parts[-1] = parts[-1][:-3]
            
            # Create relative module path with .. prefix for relative import
            # This works with importlib.import_module(relative_path, package=__package__)
            relative_module_path = '..' + '.'.join(parts)
            
            self.logger.debug(f"Converted {file_path} to relative module path: {relative_module_path}")
            return relative_module_path
            
        except Exception as e:
            self.logger.warning(f"Error converting file path {file_path} to relative module path: {e}")
            return None
