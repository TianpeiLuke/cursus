"""
Contract ↔ Specification Alignment Tester

Validates alignment between script contracts and step specifications.
Ensures logical names, data types, and dependencies are consistent.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from ....step_catalog.adapters.file_resolver import FlexibleFileResolverAdapter as FlexibleFileResolver
from ..validators.property_path_validator import SageMakerPropertyPathValidator
from ..factories.smart_spec_selector import SmartSpecificationSelector
from ..validators import ContractSpecValidator
from ....step_catalog.adapters.contract_adapter import ContractDiscoveryEngineAdapter as ContractDiscoveryEngine
from .validation_orchestrator import ValidationOrchestrator


class ContractSpecificationAlignmentTester:
    """
    Tests alignment between script contracts and step specifications.

    Validates:
    - Logical names match between contract and specification
    - Data types are consistent
    - Input/output specifications align
    - Dependencies are properly declared
    """

    def __init__(self, contracts_dir: str, specs_dir: str):
        """
        Initialize the contract-specification alignment tester.

        Args:
            contracts_dir: Directory containing script contracts
            specs_dir: Directory containing step specifications
        """
        self.contracts_dir = Path(contracts_dir)
        self.specs_dir = Path(specs_dir)

        # Initialize FlexibleFileResolver for robust file discovery
        base_directories = {
            "contracts": str(self.contracts_dir),
            "specs": str(self.specs_dir),
        }
        self.file_resolver = FlexibleFileResolver(base_directories)

        # Initialize property path validator
        self.property_path_validator = SageMakerPropertyPathValidator()

        # Initialize StepCatalog for loading
        from ....step_catalog import StepCatalog
        self.step_catalog = StepCatalog(workspace_dirs=None)

        # Initialize smart specification selector
        self.smart_spec_selector = SmartSpecificationSelector()

        # Initialize validator
        self.validator = ContractSpecValidator()

    def validate_all_contracts(
        self, target_scripts: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate alignment for all contracts or specified target scripts.

        Args:
            target_scripts: Specific scripts to validate (None for all)

        Returns:
            Dictionary mapping contract names to validation results
        """
        results = {}

        # Discover contracts to validate
        if target_scripts:
            contracts_to_validate = target_scripts
        else:
            # Only validate contracts that have corresponding scripts
            contracts_to_validate = self._discover_contracts_with_scripts()

        for contract_name in contracts_to_validate:
            try:
                result = self.validate_contract(contract_name)
                results[contract_name] = result
            except Exception as e:
                results[contract_name] = {
                    "passed": False,
                    "error": str(e),
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "validation_error",
                            "message": f"Failed to validate contract {contract_name}: {str(e)}",
                        }
                    ],
                }

        return results

    def validate_contract(self, script_or_contract_name: str) -> Dict[str, Any]:
        """
        Validate alignment for a specific contract using Smart Specification Selection.

        Args:
            script_or_contract_name: Name of the script or contract to validate

        Returns:
            Validation result dictionary
        """
        # Use FlexibleFileResolver to find the correct contract file
        contract_file_path = self.file_resolver.find_contract_file(
            script_or_contract_name
        )

        # Check if contract file exists
        if not contract_file_path:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "missing_file",
                        "message": f"Contract file not found for script: {script_or_contract_name}",
                        "details": {
                            "script": script_or_contract_name,
                            "searched_patterns": [
                                f"{script_or_contract_name}_contract.py",
                                "Known naming patterns from FlexibleFileResolver",
                            ],
                        },
                        "recommendation": f"Create contract file for {script_or_contract_name} or check naming patterns",
                    }
                ],
            }

        contract_path = Path(contract_file_path)
        if not contract_path.exists():
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "missing_file",
                        "message": f"Contract file not found: {contract_path}",
                        "recommendation": f"Create the contract file {contract_path.name}",
                    }
                ],
            }

        # Extract the actual contract name from the file path
        # e.g., "xgboost_model_eval_contract.py" -> "xgboost_model_eval_contract"
        actual_contract_name = contract_path.stem

        # Load contract using StepCatalog
        try:
            contract = self._load_contract_from_step_catalog(
                contract_path, actual_contract_name
            )
        except Exception as e:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "contract_load_error",
                        "message": f"Failed to load contract: {str(e)}",
                        "recommendation": "Fix Python syntax or contract structure in contract file",
                    }
                ],
            }

        # Find specification files using StepCatalog
        spec_files = self._find_specifications_by_contract(actual_contract_name)

        if not spec_files:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "ERROR",
                        "category": "missing_specification",
                        "message": f"No specification files found for {actual_contract_name}",
                        "recommendation": f"Create specification files that reference {actual_contract_name}",
                    }
                ],
            }

        # Load specifications using StepCatalog
        specifications = {}
        for spec_file in spec_files:
            try:
                spec = self._load_specification_from_step_catalog(spec_file, actual_contract_name)
                # Use the spec file name as the key since job type comes from config, not spec
                spec_key = spec_file.stem
                specifications[spec_key] = spec

            except Exception as e:
                return {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "spec_load_error",
                            "message": f"Failed to load specification from {spec_file}: {str(e)}",
                            "recommendation": "Fix Python syntax or specification structure",
                        }
                    ],
                }

        # SMART SPECIFICATION SELECTION: Create unified specification model
        unified_spec = self.smart_spec_selector.create_unified_specification(
            specifications, actual_contract_name
        )

        # Perform alignment validation against unified specification
        all_issues = []

        # Validate logical name alignment using smart multi-variant logic
        logical_issues = self.smart_spec_selector.validate_logical_names_smart(
            contract, unified_spec, actual_contract_name
        )
        all_issues.extend(logical_issues)

        # Validate data type consistency
        type_issues = self.validator.validate_data_types(
            contract, unified_spec["primary_spec"], actual_contract_name
        )
        all_issues.extend(type_issues)

        # Validate input/output alignment
        io_issues = self.validator.validate_input_output_alignment(
            contract, unified_spec["primary_spec"], actual_contract_name
        )
        all_issues.extend(io_issues)

        # NEW: Validate property path references (Level 2 enhancement)
        property_path_issues = self._validate_property_paths(
            unified_spec["primary_spec"], actual_contract_name
        )
        all_issues.extend(property_path_issues)

        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue["severity"] in ["CRITICAL", "ERROR"] for issue in all_issues
        )

        return {
            "passed": not has_critical_or_error,
            "issues": all_issues,
            "contract": contract,
            "specifications": specifications,
            "unified_specification": unified_spec,
        }

    # Methods moved to extracted components:
    # - _extract_contract_reference -> ContractDiscoveryEngine
    # - _extract_spec_name_from_file -> SpecificationFileProcessor
    # - _extract_job_type_from_spec_file -> SpecificationFileProcessor

    # Methods moved to extracted components:
    # - _load_specification_from_file -> SpecificationFileProcessor
    # - _load_specification_from_python -> SpecificationFileProcessor
    # - _validate_logical_names -> ContractSpecValidator (legacy version)
    # - _validate_data_types -> ContractSpecValidator
    # - _validate_input_output_alignment -> ContractSpecValidator

    # Methods moved to SmartSpecificationSelector:
    # - _extract_script_contract_from_spec -> ContractDiscoveryEngine
    # - _contracts_match -> ContractDiscoveryEngine
    # - _create_unified_specification -> SmartSpecificationSelector
    # - _extract_job_type_from_spec_name -> SpecificationFileProcessor
    # - _validate_logical_names_smart -> SmartSpecificationSelector

    def _discover_contracts(self) -> List[str]:
        """Discover all contract files in the contracts directory."""
        contracts = []

        if self.contracts_dir.exists():
            for contract_file in self.contracts_dir.glob("*_contract.py"):
                if not contract_file.name.startswith("__"):
                    contract_name = contract_file.stem.replace("_contract", "")
                    contracts.append(contract_name)

        return sorted(contracts)

    def _discover_contracts_with_scripts(self) -> List[str]:
        """
        Discover contracts that have corresponding scripts by checking their entry_point field.

        This method uses the ContractDiscoveryEngine to find contracts that have
        corresponding scripts, preventing validation errors for contracts without scripts.

        Returns:
            List of contract names that have corresponding scripts
        """
        # Use ContractDiscoveryEngine for robust contract discovery
        discovery_engine = ContractDiscoveryEngine(str(self.contracts_dir))
        return discovery_engine.discover_contracts_with_scripts()

    def _load_contract_from_step_catalog(self, contract_path: Path, contract_name: str) -> Dict[str, Any]:
        """Load contract from Python file using StepCatalog approach."""
        import sys
        import importlib.util
        
        try:
            # Add the project root to sys.path temporarily to handle relative imports
            project_root = str(contract_path.parent.parent.parent.parent)
            src_root = str(contract_path.parent.parent.parent)
            contract_dir = str(contract_path.parent)

            paths_to_add = [project_root, src_root, contract_dir]
            added_paths = []

            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    added_paths.append(path)

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    f"{contract_name}_contract", contract_path
                )
                if spec is None or spec.loader is None:
                    raise ImportError(
                        f"Could not load contract module from {contract_path}"
                    )

                module = importlib.util.module_from_spec(spec)
                module.__package__ = "cursus.steps.contracts"
                spec.loader.exec_module(module)
            finally:
                # Remove added paths from sys.path
                for path in added_paths:
                    if path in sys.path:
                        sys.path.remove(path)

            # Look for the contract object - try multiple naming patterns
            contract_obj = self._find_contract_object(module, contract_name)

            if contract_obj is None:
                raise AttributeError(f"No contract object found in {contract_path}")

            # Convert ScriptContract object to dictionary format
            return self._contract_to_dict(contract_obj, contract_name)

        except Exception as e:
            raise Exception(f"Failed to load Python contract from {contract_path}: {str(e)}")

    def _find_contract_object(self, module, contract_name: str):
        """Find the contract object in the loaded module using multiple naming patterns."""
        # Try various naming patterns
        possible_names = [
            f"{contract_name.upper()}_CONTRACT",
            f"{contract_name}_CONTRACT",
            f"{contract_name}_contract",
            "CONTRACT",
            "contract",
        ]

        # Also try to find any variable ending with _CONTRACT
        for attr_name in dir(module):
            if attr_name.endswith("_CONTRACT") and not attr_name.startswith("_"):
                possible_names.append(attr_name)

        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in possible_names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)

        for name in unique_names:
            if hasattr(module, name):
                contract_obj = getattr(module, name)
                # Verify it's actually a contract object
                if hasattr(contract_obj, "entry_point"):
                    return contract_obj

        return None

    def _contract_to_dict(self, contract_obj, contract_name: str) -> Dict[str, Any]:
        """Convert ScriptContract object to dictionary format."""
        contract_dict = {
            "entry_point": getattr(contract_obj, "entry_point", f"{contract_name}.py"),
            "inputs": {},
            "outputs": {},
            "arguments": {},
            "environment_variables": {
                "required": getattr(contract_obj, "required_env_vars", []),
                "optional": getattr(contract_obj, "optional_env_vars", {}),
            },
            "description": getattr(contract_obj, "description", ""),
            "framework_requirements": getattr(contract_obj, "framework_requirements", {}),
        }

        # Convert expected_input_paths to inputs format
        if hasattr(contract_obj, "expected_input_paths"):
            for logical_name, path in contract_obj.expected_input_paths.items():
                contract_dict["inputs"][logical_name] = {"path": path}

        # Convert expected_output_paths to outputs format
        if hasattr(contract_obj, "expected_output_paths"):
            for logical_name, path in contract_obj.expected_output_paths.items():
                contract_dict["outputs"][logical_name] = {"path": path}

        # Convert expected_arguments to arguments format
        if hasattr(contract_obj, "expected_arguments"):
            for arg_name, default_value in contract_obj.expected_arguments.items():
                contract_dict["arguments"][arg_name] = {
                    "default": default_value,
                    "required": default_value is None,
                }

        return contract_dict

    def _find_specifications_by_contract(self, contract_name: str) -> List[Path]:
        """Find specification files that reference a specific contract using StepCatalog."""
        matching_specs = []

        if not self.specs_dir.exists():
            return matching_specs

        # Search through all specification files
        for spec_file in self.specs_dir.glob("*_spec.py"):
            if spec_file.name.startswith("__"):
                continue

            try:
                # Check if this specification references the contract
                if self._specification_references_contract(spec_file, contract_name):
                    matching_specs.append(spec_file)

            except Exception:
                continue

        return matching_specs

    def _specification_references_contract(self, spec_file: Path, contract_name: str) -> bool:
        """Check if a specification references a specific contract."""
        # Use naming convention approach as the primary method
        spec_name = spec_file.stem.replace("_spec", "")
        
        # Remove job type suffix if present
        parts = spec_name.split("_")
        if len(parts) > 1:
            potential_job_types = ["training", "validation", "testing", "calibration"]
            if parts[-1] in potential_job_types:
                spec_name = "_".join(parts[:-1])

        contract_base = contract_name.lower().replace("_contract", "")

        # Check if the step type matches the contract name
        if contract_base in spec_name.lower() or spec_name.lower() in contract_base:
            return True

        return False

    def _load_specification_from_step_catalog(self, spec_file: Path, contract_name: str) -> Dict[str, Any]:
        """Load specification from Python file using StepCatalog approach."""
        import sys
        import importlib.util
        
        try:
            # Add the project root to sys.path temporarily to handle relative imports
            project_root = str(spec_file.parent.parent.parent.parent)
            src_root = str(spec_file.parent.parent.parent)
            specs_dir = str(spec_file.parent)

            paths_to_add = [project_root, src_root, specs_dir]
            added_paths = []

            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    added_paths.append(path)

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    f"{spec_file.stem}", spec_file
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load specification module from {spec_file}")

                module = importlib.util.module_from_spec(spec)
                module.__package__ = "cursus.steps.specs"
                spec.loader.exec_module(module)
            finally:
                # Remove added paths from sys.path
                for path in added_paths:
                    if path in sys.path:
                        sys.path.remove(path)

            # Find spec constant
            spec_name = spec_file.stem.replace("_spec", "")
            job_type = self._extract_job_type_from_spec_file(spec_file)
            
            possible_names = [
                f"{spec_name.upper()}_{job_type.upper()}_SPEC",
                f"{spec_name.upper()}_SPEC",
                f"{job_type.upper()}_SPEC",
            ]

            # Add dynamic discovery - scan for any constants ending with _SPEC
            spec_constants = [
                name for name in dir(module)
                if name.endswith("_SPEC") and not name.startswith("_")
            ]
            possible_names.extend(spec_constants)

            spec_obj = None
            for spec_var_name in possible_names:
                if hasattr(module, spec_var_name):
                    spec_obj = getattr(module, spec_var_name)
                    break

            if spec_obj is None:
                raise ValueError(f"No specification constant found in {spec_file}. Tried: {possible_names}")

            # Convert StepSpecification object to dictionary
            return self._step_specification_to_dict(spec_obj)

        except Exception as e:
            raise ValueError(f"Failed to load specification from {spec_file}: {str(e)}")

    def _extract_job_type_from_spec_file(self, spec_file: Path) -> str:
        """Extract job type from specification file name."""
        # Pattern: {spec_name}_{job_type}_spec.py or {spec_name}_spec.py
        stem = spec_file.stem
        parts = stem.split("_")
        if len(parts) >= 3 and parts[-1] == "spec":
            return parts[-2]  # job_type is second to last part
        return "default"

    def _step_specification_to_dict(self, spec_obj) -> Dict[str, Any]:
        """Convert StepSpecification object to dictionary representation."""
        dependencies = []
        for dep_name, dep_spec in spec_obj.dependencies.items():
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

        outputs = []
        for out_name, out_spec in spec_obj.outputs.items():
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
            "step_type": spec_obj.step_type,
            "node_type": (
                spec_obj.node_type.value
                if hasattr(spec_obj.node_type, "value")
                else str(spec_obj.node_type)
            ),
            "dependencies": dependencies,
            "outputs": outputs,
        }

    def _validate_property_paths(
        self, specification: Dict[str, Any], contract_name: str
    ) -> List[Dict[str, Any]]:
        """
        Validate SageMaker Step Property Path References (Level 2 Enhancement).

        Uses the dedicated SageMakerPropertyPathValidator to validate that property paths
        used in specification outputs are valid for the specified SageMaker step type.

        Args:
            specification: Specification dictionary
            contract_name: Name of the contract being validated

        Returns:
            List of validation issues
        """
        return self.property_path_validator.validate_specification_property_paths(
            specification, contract_name
        )
