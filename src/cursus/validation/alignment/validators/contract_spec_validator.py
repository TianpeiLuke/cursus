"""
Restored Contract-Specification Validator Module

Contains the core validation logic for contract-specification alignment.
Restored from git history commit 1653f57^ with enhancements.

Handles data type validation, input/output alignment, and logical name validation.
"""

from typing import Dict, Any, List


class ConsolidatedContractSpecValidator:
    """
    Restored contract-specification validation logic.
    
    Provides methods for:
    - Logical name validation (restored from original)
    - Input/output alignment validation (restored from original)
    - Enhanced error reporting and malformed data handling
    """

    def validate_logical_names(
        self,
        contract: Dict[str, Any],
        specification: Dict[str, Any],
        contract_name: str,
        job_type: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate that logical names match between contract and specification.

        RESTORED from original ContractSpecValidator.validate_logical_names()
        This is the basic (non-smart) validation for single specifications.

        Args:
            contract: Contract dictionary
            specification: Specification dictionary
            contract_name: Name of the contract
            job_type: Job type (optional)

        Returns:
            List of validation issues
        """
        issues = []

        # Get logical names from contract (handle malformed data gracefully)
        contract_inputs_dict = contract.get("inputs", {})
        contract_inputs = (
            set(contract_inputs_dict.keys())
            if isinstance(contract_inputs_dict, dict)
            else set()
        )

        contract_outputs_dict = contract.get("outputs", {})
        contract_outputs = (
            set(contract_outputs_dict.keys())
            if isinstance(contract_outputs_dict, dict)
            else set()
        )

        # Get logical names from specification (handle malformed data gracefully)
        spec_dependencies = set()
        dependencies = specification.get("dependencies", [])
        if isinstance(dependencies, list):
            for dep in dependencies:
                if isinstance(dep, dict) and "logical_name" in dep:
                    spec_dependencies.add(dep["logical_name"])

        spec_outputs = set()
        outputs = specification.get("outputs", [])
        if isinstance(outputs, list):
            for output in outputs:
                if isinstance(output, dict) and "logical_name" in output:
                    spec_outputs.add(output["logical_name"])

        # Check for contract inputs not in spec dependencies
        missing_deps = contract_inputs - spec_dependencies
        for logical_name in missing_deps:
            issues.append(
                {
                    "severity": "ERROR",
                    "category": "logical_names",
                    "message": f"Contract input {logical_name} not declared as specification dependency",
                    "details": {
                        "logical_name": logical_name,
                        "contract": contract_name,
                    },
                    "recommendation": f"Add {logical_name} to specification dependencies",
                }
            )

        # Check for contract outputs not in spec outputs
        missing_outputs = contract_outputs - spec_outputs
        for logical_name in missing_outputs:
            issues.append(
                {
                    "severity": "ERROR",
                    "category": "logical_names",
                    "message": f"Contract output {logical_name} not declared as specification output",
                    "details": {
                        "logical_name": logical_name,
                        "contract": contract_name,
                    },
                    "recommendation": f"Add {logical_name} to specification outputs",
                }
            )

        return issues

    def validate_input_output_alignment(
        self,
        contract: Dict[str, Any],
        specification: Dict[str, Any],
        contract_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Validate input/output alignment between contract and specification.

        RESTORED from original ContractSpecValidator.validate_input_output_alignment()

        Args:
            contract: Contract dictionary
            specification: Specification dictionary
            contract_name: Name of the contract

        Returns:
            List of validation issues
        """
        issues = []

        # Check for specification dependencies without corresponding contract inputs (handle malformed data)
        dependencies = specification.get("dependencies", [])
        spec_deps = set()
        if isinstance(dependencies, list):
            for dep in dependencies:
                if isinstance(dep, dict):
                    logical_name = dep.get("logical_name")
                    if logical_name:
                        spec_deps.add(logical_name)

        contract_inputs_dict = contract.get("inputs", {})
        contract_inputs = (
            set(contract_inputs_dict.keys())
            if isinstance(contract_inputs_dict, dict)
            else set()
        )

        unmatched_deps = spec_deps - contract_inputs
        for logical_name in unmatched_deps:
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "input_output_alignment",
                    "message": f"Specification dependency {logical_name} has no corresponding contract input",
                    "details": {
                        "logical_name": logical_name,
                        "contract": contract_name,
                    },
                    "recommendation": f"Add {logical_name} to contract inputs or remove from specification dependencies",
                }
            )

        # Check for specification outputs without corresponding contract outputs (handle malformed data)
        outputs = specification.get("outputs", [])
        spec_outputs = set()
        if isinstance(outputs, list):
            for out in outputs:
                if isinstance(out, dict):
                    logical_name = out.get("logical_name")
                    if logical_name:
                        spec_outputs.add(logical_name)

        contract_outputs_dict = contract.get("outputs", {})
        contract_outputs = (
            set(contract_outputs_dict.keys())
            if isinstance(contract_outputs_dict, dict)
            else set()
        )

        unmatched_outputs = spec_outputs - contract_outputs
        for logical_name in unmatched_outputs:
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "input_output_alignment",
                    "message": f"Specification output {logical_name} has no corresponding contract output",
                    "details": {
                        "logical_name": logical_name,
                        "contract": contract_name,
                    },
                    "recommendation": f"Add {logical_name} to contract outputs or remove from specification outputs",
                }
            )

        return issues
