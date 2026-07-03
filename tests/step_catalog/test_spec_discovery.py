"""
Unit tests for step_catalog.spec_discovery module.

Interface-first: every step specification is a *view* onto a validated
``StepInterface`` loaded from the step's ``.step.yaml``. The StepInterface is a
drop-in for the legacy StepSpecification (it exposes ``step_type``, ``node_type``,
``dependencies`` and ``outputs``). Discovery is driven by the registry's canonical
step names + the per-step ``variants`` block — there is no directory scan, no AST
parse and no per-file import. These tests exercise that interface-first model plus
the pure ``serialize_spec`` serializer and the smart-selection logic (kept verbatim).
"""

import pytest
from pathlib import Path
from unittest.mock import Mock

import cursus
from cursus.step_catalog.spec_discovery import SpecAutoDiscovery


@pytest.fixture(scope="module")
def package_root():
    """Root of the installed cursus package (source of the .step.yaml interfaces)."""
    return Path(cursus.__file__).resolve().parent


class TestSpecAutoDiscoveryInitialization:
    """Constructor keeps a stable signature even though discovery is registry-driven."""

    def test_init_package_only(self, package_root):
        """Test SpecAutoDiscovery initialization with package-only discovery."""
        discovery = SpecAutoDiscovery(package_root, [])

        assert discovery.package_root == package_root
        assert discovery.workspace_dirs == []
        assert discovery.logger is not None

    def test_init_with_workspace_dirs(self, package_root):
        """Test SpecAutoDiscovery initialization with workspace directories."""
        workspace_dirs = [package_root.parent]
        discovery = SpecAutoDiscovery(package_root, workspace_dirs)

        assert discovery.package_root == package_root
        assert discovery.workspace_dirs == workspace_dirs


class TestInterfaceFirstDiscovery:
    """Discovery sources specifications from the registry + .step.yaml interfaces."""

    @pytest.fixture
    def discovery(self, package_root):
        return SpecAutoDiscovery(package_root, [package_root.parent])

    def test_discover_spec_classes(self, discovery):
        """discover_spec_classes returns StepInterfaces keyed by PascalCase step name."""
        result = discovery.discover_spec_classes()

        assert isinstance(result, dict)
        assert len(result) > 0
        assert "TabularPreprocessing" in result

    def test_load_spec_class_existing(self, discovery):
        """Loading an existing step returns its StepInterface (a StepSpecification drop-in)."""
        spec = discovery.load_spec_class("TabularPreprocessing")

        assert spec is not None
        assert hasattr(spec, "step_type")
        assert hasattr(spec, "dependencies")
        assert hasattr(spec, "outputs")

    def test_load_spec_class_nonexistent(self, discovery):
        """Loading a step with no interface file returns None."""
        assert discovery.load_spec_class("NonexistentStep123") is None

    def test_is_spec_instance(self, discovery):
        """_is_spec_instance requires step_type + dependencies + outputs attributes."""
        valid_spec = Mock()
        valid_spec.step_type = "Processing"
        valid_spec.dependencies = {}
        valid_spec.outputs = {}
        assert discovery._is_spec_instance(valid_spec) is True

        invalid_spec = Mock()
        invalid_spec.step_type = "Processing"
        del invalid_spec.dependencies
        del invalid_spec.outputs
        assert discovery._is_spec_instance(invalid_spec) is False


class TestSpecSerialization:
    """serialize_spec is a pure serializer kept verbatim across the migration."""

    @pytest.fixture
    def discovery(self, package_root):
        return SpecAutoDiscovery(package_root, [package_root.parent])

    def test_serialize_spec_real_interface(self, discovery):
        """serialize_spec on a real interface yields the four required keys."""
        spec = discovery.load_spec_class("TabularPreprocessing")
        result = discovery.serialize_spec(spec)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"step_type", "node_type", "dependencies", "outputs"}
        assert isinstance(result["step_type"], str)
        assert isinstance(result["node_type"], str)
        assert isinstance(result["dependencies"], list)
        assert isinstance(result["outputs"], list)

    def test_serialize_spec_error_handling(self, discovery):
        """serialize_spec returns {} for an invalid spec object."""
        assert discovery.serialize_spec(None) == {}


class TestSpecContractMapping:
    """find_specs_by_contract resolves per-step (and per-variant) specifications."""

    @pytest.fixture
    def discovery(self, package_root):
        return SpecAutoDiscovery(package_root, [package_root.parent])

    def test_find_specs_by_contract_variant_step(self, discovery):
        """A step with variants yields one serialized spec per variant, keyed {Step}_{variant}."""
        result = discovery.find_specs_by_contract("CradleDataLoading")

        assert isinstance(result, dict)
        assert len(result) > 0
        # Keyed by {StepName}_{variant} so job type can be classified per entry.
        assert all(key.startswith("CradleDataLoading_") for key in result)
        # Each value is a serialized-spec dict.
        sample = next(iter(result.values()))
        assert set(sample.keys()) == {"step_type", "node_type", "dependencies", "outputs"}

    def test_find_specs_by_contract_accepts_file_stem(self, discovery):
        """A file-stem-ish contract name is bridged to the canonical step name."""
        result = discovery.find_specs_by_contract("tabular_preprocessing_contract")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_find_specs_by_contract_unknown(self, discovery):
        """An unresolvable contract name yields an empty dict."""
        assert discovery.find_specs_by_contract("nonexistent_step_123") == {}


class TestJobTypeVariants:
    """Job-type variant keys come from the .step.yaml `variants` block."""

    @pytest.fixture
    def discovery(self, package_root):
        return SpecAutoDiscovery(package_root, [package_root.parent])

    def test_get_job_type_variants(self, discovery):
        """A multi-variant step returns its variant KEYS (not VariantDecl values)."""
        variants = discovery.get_job_type_variants("CradleDataLoading")

        assert isinstance(variants, list)
        assert len(variants) >= 3
        assert "training" in variants
        assert "validation" in variants
        assert "testing" in variants

    def test_get_job_type_variants_no_variants(self, discovery):
        """A step with no interface file returns an empty variant list."""
        assert discovery.get_job_type_variants("NonexistentStep123") == []


class TestUnifiedSpecification:
    """create_unified_specification produces the 7-key smart-selection model."""

    @pytest.fixture
    def discovery(self, package_root):
        return SpecAutoDiscovery(package_root, [package_root.parent])

    def test_create_unified_specification(self, discovery):
        """A real multi-variant step yields the full 7-key unified model."""
        result = discovery.create_unified_specification("CradleDataLoading")

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "primary_spec",
            "variants",
            "unified_dependencies",
            "unified_outputs",
            "dependency_sources",
            "output_sources",
            "variant_count",
        }
        assert result["variant_count"] >= 1

    def test_create_unified_specification_no_specs(self, discovery):
        """An unresolvable contract yields the empty unified model (variant_count 0)."""
        result = discovery.create_unified_specification("nonexistent_contract_123")

        assert isinstance(result, dict)
        assert result["variant_count"] == 0

    def test_select_primary_specification_prefers_training(self, discovery):
        """_select_primary_specification prefers training > generic > first available."""
        assert discovery._select_primary_specification(
            {"validation": {"a": 1}, "training": {"b": 2}}
        ) == {"b": 2}
        assert discovery._select_primary_specification(
            {"validation": {"a": 1}, "generic": {"c": 3}}
        ) == {"c": 3}
        assert discovery._select_primary_specification({"validation": {"a": 1}}) == {
            "a": 1
        }
        assert discovery._select_primary_specification({}) == {}


class TestSmartValidation:
    """validate_logical_names_smart returns a list of issue dicts."""

    @pytest.fixture
    def discovery(self, package_root):
        return SpecAutoDiscovery(package_root, [package_root.parent])

    def test_validate_logical_names_smart(self, discovery):
        """Smart validation over a real step returns a list of issues."""
        contract = {
            "inputs": {"input_path": "s3://bucket/input"},
            "outputs": {"output_path": "s3://bucket/output"},
        }
        result = discovery.validate_logical_names_smart(contract, "CradleDataLoading")

        assert isinstance(result, list)

    def test_validate_logical_names_smart_unknown_contract(self, discovery):
        """Smart validation against an unresolvable contract still returns a list."""
        result = discovery.validate_logical_names_smart({}, "nonexistent_contract_123")

        assert isinstance(result, list)


class TestAllSpecificationsLoading:
    """load_all_specifications MUST stay populated (empty triggers a dead legacy fallback)."""

    @pytest.fixture
    def discovery(self, package_root):
        return SpecAutoDiscovery(package_root, [package_root.parent])

    def test_load_all_specifications_populated(self, discovery):
        """load_all_specifications returns a non-empty dict of serialized-spec dicts."""
        result = discovery.load_all_specifications()

        assert isinstance(result, dict)
        assert len(result) > 0
        assert "TabularPreprocessing" in result
        sample = result["TabularPreprocessing"]
        assert set(sample.keys()) == {"step_type", "node_type", "dependencies", "outputs"}
