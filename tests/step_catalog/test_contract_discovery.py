"""
Unit tests for step_catalog.contract_discovery module.

Interface-first: every contract is a *view* onto a validated ``StepInterface``
loaded from the step's ``.step.yaml`` (``iface.contract`` is a ``ContractSection``).
Discovery is driven by the registry's canonical step names + ``load_interface`` —
there is no directory scan, no AST parse and no per-file import. These tests exercise
that interface-first model plus the pure ``serialize_contract`` serializer (kept
verbatim from the pre-migration module).
"""

import pytest
from pathlib import Path
from unittest.mock import Mock

import cursus
from cursus.step_catalog.contract_discovery import ContractAutoDiscovery


@pytest.fixture(scope="module")
def package_root():
    """Root of the installed cursus package (source of the .step.yaml interfaces)."""
    return Path(cursus.__file__).resolve().parent


class TestContractAutoDiscoveryInit:
    """Constructor keeps a stable signature even though discovery is registry-driven."""

    def test_init(self, package_root):
        """Test ContractAutoDiscovery initialization."""
        workspace_dirs = [package_root.parent]
        discovery = ContractAutoDiscovery(package_root, workspace_dirs)

        assert discovery.package_root == package_root
        assert discovery.workspace_dirs == workspace_dirs
        assert discovery.logger is not None

    def test_init_no_workspace(self, package_root):
        """Test ContractAutoDiscovery initialization without workspace directories."""
        discovery = ContractAutoDiscovery(package_root, None)

        assert discovery.package_root == package_root
        assert discovery.workspace_dirs is None


class TestInterfaceFirstDiscovery:
    """Discovery sources contracts from the registry + .step.yaml interfaces."""

    @pytest.fixture
    def contract_discovery(self, package_root):
        return ContractAutoDiscovery(package_root, [package_root.parent])

    def test_discover_contract_classes_from_interfaces(self, contract_discovery):
        """discover_contract_classes returns a dict keyed by PascalCase canonical name."""
        result = contract_discovery.discover_contract_classes()

        assert isinstance(result, dict)
        # Interface-first discovery is populated from the registry, not empty.
        assert len(result) > 0
        # Keyed by canonical PascalCase step name (not a file stem).
        assert "TabularPreprocessing" in result

    def test_load_contract_class_existing(self, contract_discovery):
        """Loading an existing step returns its ContractSection (view onto the interface)."""
        contract = contract_discovery.load_contract_class("TabularPreprocessing")

        assert contract is not None
        assert getattr(contract, "entry_point", None) == "tabular_preprocessing.py"

    def test_load_contract_class_not_found(self, contract_discovery):
        """Loading a step with no interface file returns None."""
        assert contract_discovery.load_contract_class("NonexistentStep123") is None

    def test_get_contract_entry_points_populated(self, contract_discovery):
        """get_contract_entry_points is non-empty and keyed by PascalCase step name.

        (Was 0 under the dead folder-scan path; now sourced from interfaces.)
        """
        entry_points = contract_discovery.get_contract_entry_points()

        assert isinstance(entry_points, dict)
        assert len(entry_points) > 0
        assert entry_points.get("TabularPreprocessing") == "tabular_preprocessing.py"

    def test_find_contracts_by_entry_point_match(self, contract_discovery):
        """find_contracts_by_entry_point matches on the script entry point (with/without .py)."""
        matches = contract_discovery.find_contracts_by_entry_point(
            "tabular_preprocessing.py"
        )
        assert "TabularPreprocessing" in matches

        # Also matches without the .py extension.
        matches_no_ext = contract_discovery.find_contracts_by_entry_point(
            "tabular_preprocessing"
        )
        assert "TabularPreprocessing" in matches_no_ext

    def test_find_contracts_by_entry_point_no_match(self, contract_discovery):
        """find_contracts_by_entry_point returns an empty dict when nothing matches."""
        assert (
            contract_discovery.find_contracts_by_entry_point("nonexistent_script_xyz.py")
            == {}
        )


class TestContractSerialization:
    """serialize_contract is a pure serializer kept verbatim across the migration."""

    @pytest.fixture
    def contract_discovery(self, package_root):
        return ContractAutoDiscovery(package_root, [package_root.parent])

    def test_serialize_contract_real_interface(self, contract_discovery):
        """serialize_contract on a real interface's ContractSection yields the expected keys."""
        contract = contract_discovery.load_contract_class("TabularPreprocessing")
        result = contract_discovery.serialize_contract(contract)

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "entry_point",
            "inputs",
            "outputs",
            "arguments",
            "environment_variables",
            "framework_requirements",
            "description",
        }
        assert result["entry_point"] == "tabular_preprocessing.py"
        assert isinstance(result["inputs"], dict)
        assert isinstance(result["outputs"], dict)
        assert set(result["environment_variables"].keys()) == {"required", "optional"}

    def test_serialize_contract_inputs_outputs_shape(self, contract_discovery):
        """Inputs/outputs serialize as {logical_name: {"path": ...}} mappings."""

        class FakeContract:
            entry_point = "script.py"
            expected_input_paths = {"input_data": "/opt/ml/input/data"}
            expected_output_paths = {"output_data": "/opt/ml/output/data"}
            expected_arguments = {"job_type": "training", "flag": None}
            required_env_vars = ["REQUIRED_VAR"]
            optional_env_vars = {"OPTIONAL_VAR": "default"}
            description = "fake"
            framework_requirements = {"sklearn": ">=1.0"}

        result = contract_discovery.serialize_contract(FakeContract())

        assert result["inputs"] == {"input_data": {"path": "/opt/ml/input/data"}}
        assert result["outputs"] == {"output_data": {"path": "/opt/ml/output/data"}}
        # arguments: required is True when the default is None.
        assert result["arguments"]["job_type"] == {
            "default": "training",
            "required": False,
        }
        assert result["arguments"]["flag"] == {"default": None, "required": True}
        assert result["environment_variables"]["required"] == ["REQUIRED_VAR"]
        assert result["environment_variables"]["optional"] == {"OPTIONAL_VAR": "default"}

    def test_serialize_contract_error_handling(self, contract_discovery):
        """serialize_contract returns {} for a non-contract object."""
        # A plain object without any of the contract attributes is not serializable.
        assert contract_discovery.serialize_contract(object()) == {}

    def test_is_contract_instance(self, contract_discovery):
        """_is_contract_instance recognizes objects carrying contract attributes."""
        mock_contract = Mock()
        mock_contract.entry_point = "script.py"
        assert contract_discovery._is_contract_instance(mock_contract) is True

        class SimpleObject:
            some_attr = "value"

        assert contract_discovery._is_contract_instance(SimpleObject()) is False
