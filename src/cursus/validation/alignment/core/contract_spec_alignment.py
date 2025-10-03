"""
Contract ↔ Specification Alignment Tester

Validates alignment between script contracts and step specifications.
Ensures logical names, data types, and dependencies are consistent.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from ..validators.property_path_validator import SageMakerPropertyPathValidator
# PHASE 2 ENHANCEMENT: Use StepCatalog instead of SmartSpecificationSelector
# SmartSpecificationSelector functionality has been integrated into SpecAutoDiscovery
from ....step_catalog.adapters.contract_adapter import ContractDiscoveryEngineAdapter as ContractDiscoveryEngine


class ContractSpecificationAlignmentTester:
    """
    Tests alignment between script contracts and step specifications.

    Validates:
    - Logical names match between contract and specification
    - Data types are consistent
    - Input/output specifications align
    - Dependencies are properly declared
    """

    def __init__(self, workspace_dirs: Optional[List[Path]] = None):
        """
        Initialize the contract-specification alignment tester.

        Args:
            workspace_dirs: Optional list of workspace directories for workspace-aware discovery
        """
        # Store workspace directories
        self.workspace_dirs = workspace_dirs
        
        # Initialize property path validator
        self.property_path_validator = SageMakerPropertyPathValidator()

        # Initialize StepCatalog with workspace-aware discovery
        from ....step_catalog import StepCatalog
        self.step_catalog = StepCatalog(workspace_dirs=workspace_dirs)

        # PHASE 2 ENHANCEMENT: SmartSpecificationSelector functionality now integrated into StepCatalog
        # No separate smart_spec_selector needed - using StepCatalog methods directly

        # Note: ContractSpecValidator was removed during consolidation
        # TODO: Replace with consolidated validation logic
        self.validator = None

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
        Validate alignment for a specific contract using StepCatalog.

        Args:
            script_or_contract_name: Name of the script or contract to validate

        Returns:
            Validation result dictionary
        """
        # Load contract using StepCatalog
        try:
            contract_obj = self.step_catalog.load_contract_class(script_or_contract_name)
            
            if contract_obj is None:
                return {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "missing_file",
                            "message": f"Contract not found for script: {script_or_contract_name}",
                            "details": {
                                "script": script_or_contract_name,
                                "discovery_method": "StepCatalog.load_contract_class()",
                            },
                            "recommendation": f"Create contract for {script_or_contract_name} or check StepCatalog configuration",
                        }
                    ],
                }
            
            # Convert contract object to dictionary format
            contract = self._contract_to_dict(contract_obj, script_or_contract_name)
            
        except Exception as e:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "contract_load_error",
                        "message": f"Failed to load contract: {str(e)}",
                        "recommendation": "Fix contract structure or StepCatalog configuration",
                    }
                ],
            }

        # Find and load specifications using enhanced StepCatalog
        specifications = self._find_specifications_by_contract(script_or_contract_name)

        if not specifications:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "ERROR",
                        "category": "missing_specification",
                        "message": f"No specification files found for {script_or_contract_name}",
                        "recommendation": f"Create specification files that reference {script_or_contract_name}",
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

        # PHASE 2 ENHANCEMENT: Use StepCatalog smart specification methods
        # Create unified specification model using StepCatalog
        unified_spec = self.step_catalog.create_unified_specification(script_or_contract_name)

        # Perform alignment validation against unified specification
        all_issues = []

        # Validate logical name alignment using smart multi-variant logic from StepCatalog
        logical_issues = self.step_catalog.validate_logical_names_smart(
            contract, script_or_contract_name
        )
        all_issues.extend(logical_issues)

        # RESTORED: Use consolidated validation logic with restored ContractSpecValidator
        from ..validators.contract_spec_validator import ConsolidatedContractSpecValidator
        validator = ConsolidatedContractSpecValidator()
        
        # Restore logical name validation
        logical_name_issues = validator.validate_logical_names(
            contract, unified_spec["primary_spec"], script_or_contract_name
        )
        all_issues.extend(logical_name_issues)
        
        # Restore I/O alignment validation
        io_alignment_issues = validator.validate_input_output_alignment(
            contract, unified_spec["primary_spec"], script_or_contract_name
        )
        all_issues.extend(io_alignment_issues)

        # NEW: Validate property path references (Level 2 enhancement)
        property_path_issues = self._validate_property_paths(
            unified_spec["primary_spec"], script_or_contract_name
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

    def _discover_contracts_with_scripts(self) -> List[str]:
        """
        Discover contracts that have corresponding scripts using StepCatalog.

        Returns:
            List of contract names that have corresponding scripts
        """
        # Use StepCatalog for contract discovery
        return self.step_catalog.get_contract_entry_points()

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
