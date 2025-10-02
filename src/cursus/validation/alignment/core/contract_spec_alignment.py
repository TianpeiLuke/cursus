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
        from ....step_catalog.contract_discovery import ContractAutoDiscovery
        self.step_catalog = StepCatalog(workspace_dirs=None)
        
        # Initialize ContractAutoDiscovery for advanced contract loading
        package_root = Path(__file__).parent.parent.parent.parent
        self.contract_discovery = ContractAutoDiscovery(
            package_root=package_root,
            workspace_dirs=[]  # Can be extended for workspace support
        )

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

        # Load contract using ContractAutoDiscovery
        try:
            # Extract step name from contract name (remove _contract suffix)
            step_name = actual_contract_name.replace("_contract", "")
            contract_obj = self.contract_discovery.load_contract_class(step_name)
            
            if contract_obj is None:
                # Fallback to file-based loading if auto-discovery fails
                contract_obj = self.contract_discovery._load_contract_from_file(contract_path, step_name)
            
            if contract_obj is None:
                raise ValueError(f"No contract object found for {step_name}")
            
            # Convert contract object to dictionary format
            contract = self._contract_to_dict(contract_obj, step_name)
            
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

        # Find and load specifications using enhanced StepCatalog
        specifications = self._find_specifications_by_contract(actual_contract_name)

        if not specifications:
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

        # Convert specification instances to dictionary format using StepCatalog
        spec_dicts = {}
        for spec_name, spec_instance in specifications.items():
            try:
                spec_dict = self.step_catalog.serialize_spec(spec_instance)
                spec_dicts[spec_name] = spec_dict
            except Exception as e:
                return {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "spec_serialization_error",
                            "message": f"Failed to serialize specification {spec_name}: {str(e)}",
                            "recommendation": "Check specification object structure",
                        }
                    ],
                }
        
        specifications = spec_dicts

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

    # Contract loading methods removed - now using ContractAutoDiscovery directly

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

    def _find_specifications_by_contract(self, contract_name: str) -> Dict[str, Any]:
        """Find specification files that reference a specific contract using StepCatalog."""
        # Use enhanced StepCatalog method for contract-specification discovery
        return self.step_catalog.find_specs_by_contract(contract_name)

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

    # REMOVED: Manual specification loading methods replaced by StepCatalog integration
    # - _load_specification_from_step_catalog() -> now uses step_catalog.find_specs_by_contract() + step_catalog.serialize_spec()
    # - _extract_job_type_from_spec_file() -> replaced by StepCatalog job type variant discovery
    # - _step_specification_to_dict() -> replaced by step_catalog.serialize_spec()
    # Total code reduction: ~90 lines eliminated through StepCatalog integration

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
