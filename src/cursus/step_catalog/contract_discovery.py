"""
Contract discovery for the unified step catalog system.

Interface-first: every contract is a *view* onto a validated ``StepInterface``
loaded from the step's ``.step.yaml`` (``iface.contract`` is a ``ContractSection``,
a drop-in for the legacy ScriptContract/StepContract). Discovery is driven by the
registry's canonical step names — no directory scan, no AST parse, no per-file
import. The former ``steps/contracts/`` folder scan is gone; the registry is the
single source of "which steps exist" and ``load_interface`` is the single source of
"what each step's contract is".
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)


class ContractAutoDiscovery:
    """Contract discovery sourced from step interfaces + the step registry."""

    def __init__(self, package_root: Path, workspace_dirs: List[Path]):
        """
        Initialize contract discovery.

        The ``package_root`` / ``workspace_dirs`` arguments are retained for a
        stable constructor signature across the discovery modules, but interface-
        first discovery reads from the registry + ``.step.yaml`` interfaces rather
        than scanning these directories.

        Args:
            package_root: Root of the cursus package
            workspace_dirs: List of workspace directories to search
        """
        self.package_root = package_root
        self.workspace_dirs = workspace_dirs
        self.logger = logging.getLogger(__name__)

    def discover_contract_classes(
        self, project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Discover all step contracts from the registry + interfaces.

        Args:
            project_id: Optional project ID (accepted for signature stability; the
                registry + interface loader are the source of truth)

        Returns:
            Dictionary mapping PascalCase canonical step name to its ContractSection
            (a view onto the step's validated StepInterface). Steps without an
            interface file are skipped.
        """
        from ..registry.step_names import get_all_step_names
        from ..steps.interfaces import load_interface

        discovered: Dict[str, Any] = {}

        for step_name in get_all_step_names():
            try:
                iface = load_interface(step_name)
            except FileNotFoundError:
                self.logger.debug(f"No interface file for step: {step_name}")
                continue
            except Exception as e:
                self.logger.warning(f"Error loading interface for {step_name}: {e}")
                continue

            contract = getattr(iface, "contract", None)
            if contract is not None:
                discovered[step_name] = contract

        self.logger.info(f"Discovered {len(discovered)} contracts from interfaces")
        return discovered

    def load_contract_class(self, step_name: str) -> Optional[Any]:
        """
        Load the contract for a specific step.

        Args:
            step_name: PascalCase canonical step name

        Returns:
            The step's ContractSection (view onto its StepInterface), or None if the
            step has no interface file.
        """
        from ..steps.interfaces import load_interface

        try:
            iface = load_interface(step_name)
        except FileNotFoundError:
            self.logger.warning(f"No contract found for step: {step_name}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading contract for {step_name}: {e}")
            return None

        return getattr(iface, "contract", None)

    def find_contracts_by_entry_point(self, entry_point: str) -> Dict[str, Any]:
        """
        Find contracts that reference a specific script entry point.

        Enables script-contract alignment validation by finding the contracts
        associated with a given script entry point.

        Args:
            entry_point: Script entry point (e.g., "model_evaluation_xgb.py").
                Matched with or without the ``.py`` extension.

        Returns:
            Dictionary mapping PascalCase canonical step name to its ContractSection.
        """
        try:
            matching_contracts: Dict[str, Any] = {}
            entry_point_base = entry_point.replace(".py", "")

            for step_name, contract in self.discover_contract_classes().items():
                contract_entry_point = getattr(contract, "entry_point", None)
                if not contract_entry_point:
                    continue
                if (
                    contract_entry_point == entry_point
                    or contract_entry_point.replace(".py", "") == entry_point_base
                ):
                    matching_contracts[step_name] = contract
                    self.logger.debug(
                        f"Found matching contract: {step_name} for entry point {entry_point}"
                    )

            self.logger.debug(
                f"Found {len(matching_contracts)} contracts for entry point '{entry_point}'"
            )
            return matching_contracts

        except Exception as e:
            self.logger.error(
                f"Error finding contracts for entry point {entry_point}: {e}"
            )
            return {}

    def get_contract_entry_points(self) -> Dict[str, str]:
        """
        Get all contract entry points for validation.

        Returns:
            Dictionary mapping PascalCase canonical step name to its entry point.
            Steps whose contract declares no entry_point (script-less
            CreateModel/Transform steps) are omitted.
        """
        try:
            entry_points: Dict[str, str] = {}

            for step_name, contract in self.discover_contract_classes().items():
                entry_point = getattr(contract, "entry_point", None)
                if entry_point:
                    entry_points[step_name] = entry_point
                    self.logger.debug(
                        f"Found entry point: {entry_point} for contract {step_name}"
                    )

            self.logger.debug(f"Found {len(entry_points)} contract entry points")
            return entry_points

        except Exception as e:
            self.logger.error(f"Error getting contract entry points: {e}")
            return {}

    def serialize_contract(self, contract_instance: Any) -> Dict[str, Any]:
        """
        Convert contract instance to dictionary format.

        This method provides standardized serialization of ScriptContract objects
        for use in script-contract alignment validation, following the same pattern
        as SpecAutoDiscovery.serialize_spec().

        Args:
            contract_instance: Contract instance to serialize

        Returns:
            Dictionary representation of the contract
        """
        try:
            if not self._is_contract_instance(contract_instance):
                raise ValueError("Object is not a valid contract instance")

            # Serialize contract fields using helper methods
            return {
                "entry_point": getattr(contract_instance, "entry_point", ""),
                "inputs": self._serialize_contract_inputs(contract_instance),
                "outputs": self._serialize_contract_outputs(contract_instance),
                "arguments": self._serialize_contract_arguments(contract_instance),
                "environment_variables": {
                    "required": getattr(contract_instance, "required_env_vars", []),
                    "optional": getattr(contract_instance, "optional_env_vars", {}),
                },
                "description": getattr(contract_instance, "description", ""),
                "framework_requirements": getattr(
                    contract_instance, "framework_requirements", {}
                ),
            }

        except Exception as e:
            self.logger.error(f"Error serializing contract: {e}")
            return {}

    def _is_contract_instance(self, obj: Any) -> bool:
        """Check if an object is a valid contract instance."""
        try:
            # Check if it has the expected attributes of a contract
            return (
                hasattr(obj, "entry_point")
                or hasattr(obj, "expected_input_paths")
                or hasattr(obj, "expected_output_paths")
            )
        except Exception:
            return False

    def _serialize_contract_inputs(self, contract_instance: Any) -> Dict[str, Any]:
        """Serialize contract input specifications."""
        inputs = {}
        try:
            if hasattr(contract_instance, "expected_input_paths"):
                for (
                    logical_name,
                    path,
                ) in contract_instance.expected_input_paths.items():
                    inputs[logical_name] = {"path": path}
        except Exception as e:
            self.logger.warning(f"Error serializing contract inputs: {e}")
        return inputs

    def _serialize_contract_outputs(self, contract_instance: Any) -> Dict[str, Any]:
        """Serialize contract output specifications."""
        outputs = {}
        try:
            if hasattr(contract_instance, "expected_output_paths"):
                for (
                    logical_name,
                    path,
                ) in contract_instance.expected_output_paths.items():
                    outputs[logical_name] = {"path": path}
        except Exception as e:
            self.logger.warning(f"Error serializing contract outputs: {e}")
        return outputs

    def _serialize_contract_arguments(self, contract_instance: Any) -> Dict[str, Any]:
        """Serialize contract argument specifications."""
        arguments = {}
        try:
            if hasattr(contract_instance, "expected_arguments"):
                for (
                    arg_name,
                    default_value,
                ) in contract_instance.expected_arguments.items():
                    arguments[arg_name] = {
                        "default": default_value,
                        "required": default_value is None,
                    }
        except Exception as e:
            self.logger.warning(f"Error serializing contract arguments: {e}")
        return arguments
