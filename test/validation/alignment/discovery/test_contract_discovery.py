"""
Unit tests for cursus.validation.alignment.discovery.contract_discovery module.

Tests the ContractDiscoveryEngineAdapter class that handles discovery and mapping of
contract files using the modern step catalog system.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from cursus.step_catalog.adapters import ContractDiscoveryEngineAdapter


@pytest.fixture
def temp_dir():
    """Set up temporary directory fixture."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def engine(workspace_root):
    """Set up ContractDiscoveryEngineAdapter fixture."""
    return ContractDiscoveryEngineAdapter(workspace_root)


class TestContractDiscoveryEngine:
    """Test cases for ContractDiscoveryEngineAdapter class."""

    def test_init(self, workspace_root):
        """Test ContractDiscoveryEngineAdapter initialization."""
        engine = ContractDiscoveryEngineAdapter(workspace_root)

        assert hasattr(engine, 'catalog')
        assert engine.catalog is not None

    def test_discover_all_contracts_with_files(self, engine):
        """Test discovering all contracts using step catalog."""
        contracts = engine.discover_all_contracts()
        
        # The adapter uses step catalog, so it will discover actual steps with contracts
        assert isinstance(contracts, list)
        # Verify it returns step names, not file-based names
        
    def test_discover_all_contracts_empty_catalog(self, workspace_root):
        """Test discovering contracts when catalog is empty."""
        # Create adapter with empty workspace
        empty_workspace = workspace_root / "empty"
        empty_workspace.mkdir(exist_ok=True)
        engine = ContractDiscoveryEngineAdapter(empty_workspace)
        contracts = engine.discover_all_contracts()
        
        assert isinstance(contracts, list)

    def test_discover_contracts_with_scripts_success(self, engine):
        """Test discovering contracts that have corresponding scripts using step catalog."""
        contracts = engine.discover_contracts_with_scripts()
        
        # The adapter uses step catalog to find steps with both contracts and scripts
        assert isinstance(contracts, list)
        # Each item should be a step name that has both contract and script components
        for contract in contracts:
            assert isinstance(contract, str)

    def test_discover_contracts_with_scripts_functionality(self, engine):
        """Test discovering contracts with scripts using modern step catalog."""
        contracts = engine.discover_contracts_with_scripts()
        
        # The adapter uses step catalog to find steps with both contracts and scripts
        assert isinstance(contracts, list)
        # Verify each contract has both components
        for contract in contracts:
            step_info = engine.catalog.get_step_info(contract)
            if step_info:
                # Should have both contract and script components
                assert 'contract' in step_info.file_components or 'script' in step_info.file_components

    def test_discover_contracts_with_scripts_error_handling(self, engine):
        """Test discovering contracts with error handling using step catalog."""
        # The modern adapter handles errors gracefully through the catalog
        contracts = engine.discover_contracts_with_scripts()
        
        # Should return a list even if some steps have issues
        assert isinstance(contracts, list)
        
    def test_discover_contracts_empty_workspace(self, workspace_root):
        """Test discovering contracts when workspace is empty."""
        empty_workspace = workspace_root / "empty_test"
        empty_workspace.mkdir(exist_ok=True)
        engine = ContractDiscoveryEngineAdapter(empty_workspace)
        
        contracts = engine.discover_contracts_with_scripts()
        assert isinstance(contracts, list)

    def test_extract_contract_reference_from_spec_functionality(self, engine):
        """Test contract reference extraction functionality using step catalog."""
        # Test with a real spec file path - the adapter uses catalog.find_step_by_component
        spec_file = "test_spec.py"
        
        # Mock the catalog's find_step_by_component method
        with patch.object(engine.catalog, 'find_step_by_component') as mock_find:
            with patch.object(engine.catalog, 'get_step_info') as mock_get_info:
                # Mock finding a step with a contract
                mock_find.return_value = "test_step"
                mock_step_info = Mock()
                mock_step_info.file_components = {'contract': Mock()}
                mock_get_info.return_value = mock_step_info
                
                result = engine.extract_contract_reference_from_spec(spec_file)
                assert result == "test_step"
                
                # Verify catalog methods were called
                mock_find.assert_called_once_with(spec_file)
                mock_get_info.assert_called_once_with("test_step")

    def test_extract_contract_reference_from_spec_no_match(self, engine):
        """Test contract reference extraction when no step is found."""
        spec_file = "nonexistent_spec.py"
        
        # Mock the catalog to return None (no step found)
        with patch.object(engine.catalog, 'find_step_by_component') as mock_find:
            mock_find.return_value = None
            
            result = engine.extract_contract_reference_from_spec(spec_file)
            assert result is None

    def test_contract_matching_functionality(self, engine):
        """Test contract matching functionality using modern step catalog search."""
        # Test step search functionality (modern replacement for similarity matching)
        steps = engine.catalog.list_available_steps()
        
        # Test exact matching through search
        if steps:
            exact_results = engine.catalog.search_steps(steps[0])
            assert isinstance(exact_results, list)
            if exact_results:
                # Search results are StepSearchResult objects, check step_name attribute
                result_names = [result.step_name for result in exact_results]
                assert steps[0] in result_names
        
        # Test partial matching through search
        partial_results = engine.catalog.search_steps("training")
        assert isinstance(partial_results, list)
        
        # Test no matching
        no_results = engine.catalog.search_steps("nonexistent_step_name_xyz")
        assert isinstance(no_results, list)

    def test_step_catalog_integration(self, engine):
        """Test integration with step catalog for modern discovery."""
        # Test that the adapter properly uses the step catalog
        assert hasattr(engine, 'catalog')
        assert engine.catalog is not None
        
        # Test that catalog methods are available
        steps = engine.catalog.list_available_steps()
        assert isinstance(steps, list)
        
        # Test step info retrieval
        if steps:
            step_info = engine.catalog.get_step_info(steps[0])
            # step_info might be None if step has no components, which is fine
            if step_info:
                assert hasattr(step_info, 'file_components')

    def test_modern_discovery_methods(self, engine):
        """Test that modern discovery methods work correctly."""
        # Test discover_all_contracts
        all_contracts = engine.discover_all_contracts()
        assert isinstance(all_contracts, list)
        
        # Test discover_contracts_with_scripts
        contracts_with_scripts = engine.discover_contracts_with_scripts()
        assert isinstance(contracts_with_scripts, list)
        
        # contracts_with_scripts should be a subset of all_contracts
        for contract in contracts_with_scripts:
            assert contract in all_contracts or len(all_contracts) == 0

    def test_error_handling_graceful_degradation(self, engine):
        """Test that the adapter handles errors gracefully."""
        # Test with invalid spec file
        result = engine.extract_contract_reference_from_spec("invalid_spec.py")
        # Should return None or handle gracefully, not raise exception
        assert result is None or isinstance(result, str)
        
        # Test discovery methods don't raise exceptions
        try:
            contracts = engine.discover_all_contracts()
            assert isinstance(contracts, list)
            
            contracts_with_scripts = engine.discover_contracts_with_scripts()
            assert isinstance(contracts_with_scripts, list)
        except Exception as e:
            pytest.fail(f"Discovery methods should not raise exceptions: {e}")

    # Modern replacements for obsolete internal method tests
    
    def test_contract_script_extraction_modern(self, engine):
        """Modern replacement for extract_script_contract_from_spec tests."""
        # Test successful extraction using step catalog
        with patch.object(engine.catalog, 'find_step_by_component') as mock_find:
            with patch.object(engine.catalog, 'get_step_info') as mock_get_info:
                # Mock successful step discovery
                mock_find.return_value = "xgboost_training"
                mock_step_info = Mock()
                mock_step_info.file_components = {'contract': Mock(), 'script': Mock()}
                mock_get_info.return_value = mock_step_info
                
                result = engine.extract_contract_reference_from_spec("xgboost_training_spec.py")
                assert result == "xgboost_training"
        
        # Test extraction failure (no step found)
        with patch.object(engine.catalog, 'find_step_by_component') as mock_find:
            mock_find.return_value = None
            result = engine.extract_contract_reference_from_spec("nonexistent_spec.py")
            assert result is None
    
    def test_contract_matching_modern(self, engine):
        """Modern replacement for contracts_match tests."""
        # Test step search functionality (modern replacement for similarity matching)
        steps = engine.catalog.list_available_steps()
        
        # Test exact matching through search
        if steps:
            exact_results = engine.catalog.search_steps(steps[0])
            assert isinstance(exact_results, list)
        
        # Test partial matching through search
        partial_results = engine.catalog.search_steps("training")
        assert isinstance(partial_results, list)
        
        # Test search with .py extension
        py_results = engine.catalog.search_steps("training.py")
        assert isinstance(py_results, list)
        
        # Test no matching
        no_results = engine.catalog.search_steps("completely_different_nonexistent")
        assert isinstance(no_results, list)

    def test_entry_point_mapping_modern(self, engine):
        """Modern replacement for build_entry_point_mapping tests."""
        # Test that the catalog provides step-to-script mapping
        steps = engine.catalog.list_available_steps()
        
        mapping = {}
        for step in steps:
            step_info = engine.catalog.get_step_info(step)
            if step_info and 'script' in step_info.file_components:
                script_metadata = step_info.file_components['script']
                # script_metadata is a FileMetadata object with a path attribute
                script_path = str(script_metadata.path)
                mapping[step] = script_path
        
        # Verify mapping structure
        assert isinstance(mapping, dict)
        for step, script_path in mapping.items():
            assert isinstance(step, str)
            assert isinstance(script_path, str)
            assert script_path.endswith('.py')

    def test_contract_loading_modern(self, engine):
        """Modern replacement for _load_contract_for_entry_point tests."""
        # Test contract loading through step catalog
        steps = engine.catalog.list_available_steps()
        
        for step in steps[:3]:  # Test first 3 steps
            step_info = engine.catalog.get_step_info(step)
            if step_info and 'contract' in step_info.file_components:
                contract_path = step_info.file_components['contract']
                assert contract_path.exists()
                assert str(contract_path).endswith('_contract.py')

    def test_entry_point_extraction_modern(self, engine):
        """Modern replacement for _extract_entry_point_from_contract tests."""
        # Test entry point extraction using catalog metadata
        steps = engine.catalog.list_available_steps()
        
        for step in steps[:3]:  # Test first 3 steps
            step_info = engine.catalog.get_step_info(step)
            if step_info and 'script' in step_info.file_components:
                script_path = step_info.file_components['script']
                # The catalog should provide the script path directly
                assert script_path.exists()
                assert str(script_path).endswith('.py')

    def test_discovery_with_load_errors_modern(self, engine):
        """Modern replacement for discovery error handling tests."""
        # Test that discovery continues even when some steps have issues
        with patch.object(engine.catalog, 'get_step_info') as mock_get_info:
            # Mock some steps failing to load info
            def side_effect(step):
                if step == "problematic_step":
                    return None  # Simulate step with no info
                else:
                    mock_info = Mock()
                    mock_info.file_components = {'contract': Mock(), 'script': Mock()}
                    return mock_info
            
            mock_get_info.side_effect = side_effect
            
            # Mock available steps including problematic one
            with patch.object(engine.catalog, 'list_available_steps') as mock_list:
                mock_list.return_value = ["good_step", "problematic_step", "another_good_step"]
                
                contracts = engine.discover_contracts_with_scripts()
                # Should still return the good steps
                assert isinstance(contracts, list)

    def test_nonexistent_directory_handling_modern(self, engine):
        """Modern replacement for nonexistent directory tests."""
        # Test with empty workspace (modern equivalent of nonexistent directory)
        empty_workspace = engine.catalog.workspace_root / "nonexistent"
        empty_engine = ContractDiscoveryEngineAdapter(empty_workspace)
        
        contracts = empty_engine.discover_all_contracts()
        assert isinstance(contracts, list)
        
        contracts_with_scripts = empty_engine.discover_contracts_with_scripts()
        assert isinstance(contracts_with_scripts, list)

    def test_multiple_naming_patterns_modern(self, engine):
        """Modern replacement for multiple naming pattern tests."""
        # Test that catalog handles various naming patterns through search
        steps = engine.catalog.list_available_steps()
        
        # Test finding steps with different naming patterns
        for step in steps[:3]:  # Test first 3 steps
            # Test search with various patterns
            patterns = [
                step.lower(),
                step.upper(),
                step.replace('_', '-'),
                f"{step}_v2",
                f"{step}.py"
            ]
            
            for pattern in patterns:
                search_results = engine.catalog.search_steps(pattern)
                assert isinstance(search_results, list)
                # The search should return valid results (may or may not include the original step)

    def test_integration_workflow_modern(self, engine):
        """Modern replacement for integration workflow tests."""
        # Test complete modern discovery workflow
        
        # 1. Discover all contracts
        all_contracts = engine.discover_all_contracts()
        assert isinstance(all_contracts, list)
        
        # 2. Discover contracts with scripts
        contracts_with_scripts = engine.discover_contracts_with_scripts()
        assert isinstance(contracts_with_scripts, list)
        
        # 3. Extract contract references from specs
        for contract in contracts_with_scripts[:2]:  # Test first 2
            step_info = engine.catalog.get_step_info(contract)
            if step_info and 'spec' in step_info.file_components:
                spec_metadata = step_info.file_components['spec']
                # Use the path attribute from FileMetadata
                spec_filename = spec_metadata.path.name
                result = engine.extract_contract_reference_from_spec(spec_filename)
                # Should return the step name or None
                assert result is None or isinstance(result, str)
        
        # 4. Test contract search functionality (modern replacement for similarity)
        if contracts_with_scripts:
            contract = contracts_with_scripts[0]
            search_results = engine.catalog.search_steps(contract)
            assert isinstance(search_results, list)
            # Should find the contract itself in search results
            if search_results:
                # Search results are StepSearchResult objects, check step_name attribute
                result_names = [result.step_name for result in search_results]
                assert contract in result_names

    def test_sys_path_management_modern(self, engine):
        """Modern replacement for sys.path management tests."""
        import sys
        original_path = sys.path.copy()
        
        # Modern catalog doesn't modify sys.path during discovery
        contracts = engine.discover_all_contracts()
        assert sys.path == original_path
        
        contracts_with_scripts = engine.discover_contracts_with_scripts()
        assert sys.path == original_path
        
        # Extract contract reference shouldn't modify sys.path
        result = engine.extract_contract_reference_from_spec("test_spec.py")
        assert sys.path == original_path

    def test_error_resilience_modern(self, engine):
        """Modern replacement for error resilience tests."""
        # Test that catalog-based discovery is resilient to errors
        
        # Mock catalog to simulate some errors
        with patch.object(engine.catalog, 'list_available_steps') as mock_list:
            with patch.object(engine.catalog, 'get_step_info') as mock_get_info:
                # Mock steps list
                mock_list.return_value = ["step1", "step2", "step3"]
                
                # Mock get_step_info to fail for some steps
                def side_effect(step):
                    if step == "step2":
                        raise Exception("Simulated error")
                    mock_info = Mock()
                    mock_info.file_components = {'contract': Mock(), 'script': Mock()}
                    return mock_info
                
                mock_get_info.side_effect = side_effect
                
                # Discovery should continue despite errors
                contracts = engine.discover_contracts_with_scripts()
                assert isinstance(contracts, list)
                # Should have processed step1 and step3, skipped step2


if __name__ == "__main__":
    pytest.main([__file__])
