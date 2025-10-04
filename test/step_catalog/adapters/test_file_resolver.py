"""
Unit tests for step_catalog.adapters.file_resolver module.

Tests the FlexibleFileResolverAdapter, DeveloperWorkspaceFileResolverAdapter, and 
HybridFileResolverAdapter classes that provide backward compatibility with legacy 
file resolution systems.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List

from cursus.step_catalog.adapters.file_resolver import (
    FlexibleFileResolverAdapter,
    DeveloperWorkspaceFileResolverAdapter,
    HybridFileResolverAdapter
)


class TestFlexibleFileResolverAdapter:
    """Test FlexibleFileResolverAdapter functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_init(self, temp_workspace):
        """Test FlexibleFileResolverAdapter initialization."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog:
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # Should initialize StepCatalog with workspace_dirs=[workspace_root]
            mock_catalog.assert_called_once_with(workspace_dirs=[temp_workspace])
            assert adapter.logger is not None
            assert adapter.base_dirs is None  # Legacy compatibility
            assert isinstance(adapter.file_cache, dict)
    
    def test_refresh_cache_success(self, temp_workspace):
        """Test successful cache refresh."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["step1", "step2"]
            
            # Mock step info with file components
            mock_step_info1 = Mock()
            mock_script1 = Mock()
            mock_script1.path = Path("/path/to/step1.py")
            mock_contract1 = Mock()
            mock_contract1.path = Path("/path/to/step1_contract.py")
            mock_step_info1.file_components = {
                "script": mock_script1,
                "contract": mock_contract1
            }
            
            mock_step_info2 = Mock()
            mock_spec2 = Mock()
            mock_spec2.path = Path("/path/to/step2_spec.py")
            mock_step_info2.file_components = {
                "spec": mock_spec2
            }
            
            mock_catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # Check cache was populated
            assert "scripts" in adapter.file_cache
            assert "contracts" in adapter.file_cache
            assert "specs" in adapter.file_cache
            assert len(adapter.file_cache["scripts"]) > 0
            assert len(adapter.file_cache["contracts"]) > 0
            assert len(adapter.file_cache["specs"]) > 0
    
    def test_refresh_cache_error_handling(self, temp_workspace):
        """Test error handling in cache refresh."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.side_effect = Exception("Catalog error")
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # Should handle error gracefully
            assert isinstance(adapter.file_cache, dict)
    
    def test_extract_base_name(self, temp_workspace):
        """Test base name extraction from step names."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # Test PascalCase to snake_case conversion
            assert adapter._extract_base_name("XGBoostTraining", "script") == "x_g_boost_training"
            assert adapter._extract_base_name("DataLoading", "contract") == "data_loading"
            assert adapter._extract_base_name("SimpleStep", "spec") == "simple_step"
    
    def test_normalize_name(self, temp_workspace):
        """Test name normalization."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # Test various normalizations
            assert adapter._normalize_name("test-name") == "test_name"
            assert adapter._normalize_name("test.name") == "test_name"
            assert adapter._normalize_name("preprocess") == "preprocessing"
            assert adapter._normalize_name("eval") == "evaluation"
            assert adapter._normalize_name("xgb") == "xgboost"
            assert adapter._normalize_name("train") == "training"
    
    def test_calculate_similarity(self, temp_workspace):
        """Test similarity calculation."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # Test similarity scores
            assert adapter._calculate_similarity("test", "test") == 1.0
            assert adapter._calculate_similarity("test", "TEST") == 1.0
            assert adapter._calculate_similarity("test", "testing") > 0.5
            assert adapter._calculate_similarity("test", "completely_different") < 0.5
    
    def test_find_best_match_direct_catalog(self, temp_workspace):
        """Test finding best match via direct catalog lookup."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_script = Mock()
            mock_script.path = Path("/path/to/script.py")
            mock_step_info.file_components = {"script": mock_script}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            result = adapter._find_best_match("test_step", "script")
            
            assert result == "/path/to/script.py"
            mock_catalog.get_step_info.assert_called_with("test_step")
    
    def test_find_best_match_fuzzy_matching(self, temp_workspace):
        """Test finding best match via fuzzy matching."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.get_step_info.return_value = None  # No direct match
            mock_catalog.list_available_steps.return_value = ["XGBoostTraining", "DataLoading"]
            
            # Mock step info for fuzzy matching
            mock_step_info = Mock()
            mock_script = Mock()
            mock_script.path = Path("/path/to/xgboost_training.py")
            mock_step_info.file_components = {"script": mock_script}
            mock_catalog.get_step_info.side_effect = [None, mock_step_info, None]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            result = adapter._find_best_match("xgboost_training", "script")
            
            assert result == "/path/to/xgboost_training.py"
    
    def test_find_best_match_no_match(self, temp_workspace):
        """Test finding best match when no match exists."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.get_step_info.return_value = None
            mock_catalog.list_available_steps.return_value = []
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            result = adapter._find_best_match("nonexistent_step", "script")
            
            assert result is None
    
    def test_get_available_files_report(self, temp_workspace):
        """Test generating available files report."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["step1"]
            
            mock_step_info = Mock()
            mock_script = Mock()
            mock_script.path = Path("/path/to/step1.py")
            mock_step_info.file_components = {"script": mock_script}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            report = adapter.get_available_files_report()
            
            assert isinstance(report, dict)
            assert "scripts" in report
            assert "contracts" in report
            assert "specs" in report
            assert "builders" in report
            assert "configs" in report
            
            for component_type in report:
                assert "count" in report[component_type]
                assert "files" in report[component_type]
                assert "base_names" in report[component_type]
    
    def test_extract_base_name_from_spec(self, temp_workspace):
        """Test extracting base name from spec path."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # Test spec file name extraction
            assert adapter.extract_base_name_from_spec(Path("test_step_spec.py")) == "test_step"
            assert adapter.extract_base_name_from_spec(Path("xgboost_training_spec.py")) == "xgboost_training"
            assert adapter.extract_base_name_from_spec(Path("simple_spec.py")) == "simple"
            assert adapter.extract_base_name_from_spec(Path("no_spec_suffix.py")) == "no_spec_suffix"
    
    def test_find_spec_constant_name(self, temp_workspace):
        """Test finding spec constant name."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_spec = Mock()
            mock_spec.path = Path("/path/to/test_step_spec.py")
            mock_step_info.file_components = {"spec": mock_spec}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # Mock find_spec_file to return the path
            with patch.object(adapter, 'find_spec_file', return_value="/path/to/test_step_spec.py"):
                result = adapter.find_spec_constant_name("test_step")
                assert result == "TEST_STEP_TRAINING_SPEC"
                
                result = adapter.find_spec_constant_name("test_step", "validation")
                assert result == "TEST_STEP_VALIDATION_SPEC"
    
    def test_find_spec_constant_name_no_spec(self, temp_workspace):
        """Test finding spec constant name when no spec file exists."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            with patch.object(adapter, 'find_spec_file', return_value=None):
                result = adapter.find_spec_constant_name("test_step")
                assert result == "TEST_STEP_TRAINING_SPEC"
    
    def test_find_specification_file_legacy_alias(self, temp_workspace):
        """Test legacy alias for find_spec_file."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_spec = Mock()
            mock_spec.path = Path("/path/to/spec.py")
            mock_step_info.file_components = {"spec": mock_spec}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            result = adapter.find_specification_file("test_step")
            
            assert result == Path("/path/to/spec.py")
    
    def test_find_contract_file(self, temp_workspace):
        """Test finding contract file."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_contract = Mock()
            mock_contract.path = Path("/path/to/contract.py")
            mock_step_info.file_components = {"contract": mock_contract}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            result = adapter.find_contract_file("test_step")
            
            assert result == Path("/path/to/contract.py")
    
    def test_find_contract_file_not_found(self, temp_workspace):
        """Test finding contract file when not found."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.get_step_info.return_value = None
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            result = adapter.find_contract_file("nonexistent_step")
            
            assert result is None
    
    def test_find_spec_file(self, temp_workspace):
        """Test finding spec file."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_spec = Mock()
            mock_spec.path = Path("/path/to/spec.py")
            mock_step_info.file_components = {"spec": mock_spec}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            result = adapter.find_spec_file("test_step")
            
            assert result == Path("/path/to/spec.py")
    
    def test_find_builder_file(self, temp_workspace):
        """Test finding builder file."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_builder = Mock()
            mock_builder.path = Path("/path/to/builder.py")
            mock_step_info.file_components = {"builder": mock_builder}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            result = adapter.find_builder_file("test_step")
            
            assert result == Path("/path/to/builder.py")
    
    def test_find_config_file(self, temp_workspace):
        """Test finding config file."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_config = Mock()
            mock_config.path = Path("/path/to/config.py")
            mock_step_info.file_components = {"config": mock_config}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            result = adapter.find_config_file("test_step")
            
            assert result == Path("/path/to/config.py")
    
    def test_find_all_component_files(self, temp_workspace):
        """Test finding all component files for a step."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            
            mock_script = Mock()
            mock_script.path = Path("/path/to/script.py")
            mock_contract = Mock()
            mock_contract.path = Path("/path/to/contract.py")
            mock_spec = Mock()
            mock_spec.path = Path("/path/to/spec.py")
            
            mock_step_info.file_components = {
                "script": mock_script,
                "contract": mock_contract,
                "spec": mock_spec,
                "builder": None,
                "config": None
            }
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            result = adapter.find_all_component_files("test_step")
            
            assert result["script"] == Path("/path/to/script.py")
            assert result["contract"] == Path("/path/to/contract.py")
            assert result["spec"] == Path("/path/to/spec.py")
            assert result["builder"] is None
            assert result["config"] is None
    
    def test_find_all_component_files_error(self, temp_workspace):
        """Test error handling in find_all_component_files."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.get_step_info.side_effect = Exception("Catalog error")
            mock_catalog_class.return_value = mock_catalog
            
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            result = adapter.find_all_component_files("test_step")
            
            assert result == {}


class TestDeveloperWorkspaceFileResolverAdapter:
    """Test DeveloperWorkspaceFileResolverAdapter functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with developer structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Create developer workspace structure
            dev_workspace = workspace_root / "developers" / "test_dev"
            dev_workspace.mkdir(parents=True)
            
            # Create component directories
            (dev_workspace / "contracts").mkdir()
            (dev_workspace / "specs").mkdir()
            (dev_workspace / "builders").mkdir()
            (dev_workspace / "scripts").mkdir()
            (dev_workspace / "configs").mkdir()
            
            # Create shared workspace structure
            shared_workspace = workspace_root / "shared"
            shared_workspace.mkdir()
            (shared_workspace / "contracts").mkdir()
            (shared_workspace / "specs").mkdir()
            (shared_workspace / "builders").mkdir()
            (shared_workspace / "scripts").mkdir()
            (shared_workspace / "configs").mkdir()
            
            yield workspace_root
    
    def test_init_with_workspace_root(self, temp_workspace):
        """Test initialization with workspace root."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog:
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            mock_catalog.assert_called_once_with(workspace_dirs=[temp_workspace])
            assert adapter.workspace_root == temp_workspace
            assert adapter.workspace_mode is True
            assert adapter.developer_id == "test_dev"
            assert adapter.project_id == "test_dev"
    
    def test_init_without_workspace_root(self):
        """Test initialization without workspace root (single workspace mode)."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog:
            adapter = DeveloperWorkspaceFileResolverAdapter()
            
            mock_catalog.assert_called_once_with(workspace_dirs=[Path('.')])
            assert adapter.workspace_root is None
            assert adapter.workspace_mode is False
    
    def test_init_with_project_id_alias(self, temp_workspace):
        """Test initialization with project_id parameter (legacy alias)."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                project_id="test_project"
            )
            
            assert adapter.project_id == "test_project"
            assert adapter.developer_id == "test_project"
    
    def test_validate_workspace_structure_invalid_root(self):
        """Test workspace structure validation with invalid root."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            with pytest.raises(ValueError, match="Workspace root does not exist"):
                DeveloperWorkspaceFileResolverAdapter(
                    workspace_root=Path("/nonexistent/path"),
                    developer_id="test_dev"
                )
    
    def test_validate_workspace_structure_invalid_developer(self, temp_workspace):
        """Test workspace structure validation with invalid developer."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            with pytest.raises(ValueError, match="Developer workspace does not exist"):
                DeveloperWorkspaceFileResolverAdapter(
                    workspace_root=temp_workspace,
                    developer_id="nonexistent_dev"
                )
    
    def test_setup_workspace_paths(self, temp_workspace):
        """Test workspace path setup."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev",
                enable_shared_fallback=True
            )
            
            # Check developer workspace paths
            assert adapter.contracts_dir == temp_workspace / "developers" / "test_dev" / "contracts"
            assert adapter.specs_dir == temp_workspace / "developers" / "test_dev" / "specs"
            assert adapter.builders_dir == temp_workspace / "developers" / "test_dev" / "builders"
            assert adapter.scripts_dir == temp_workspace / "developers" / "test_dev" / "scripts"
            assert adapter.configs_dir == temp_workspace / "developers" / "test_dev" / "configs"
            
            # Check shared workspace paths
            assert adapter.shared_contracts_dir == temp_workspace / "shared" / "contracts"
            assert adapter.shared_specs_dir == temp_workspace / "shared" / "specs"
            assert adapter.shared_builders_dir == temp_workspace / "shared" / "builders"
            assert adapter.shared_scripts_dir == temp_workspace / "shared" / "scripts"
            assert adapter.shared_configs_dir == temp_workspace / "shared" / "configs"
    
    def test_find_workspace_file_catalog_lookup(self, temp_workspace):
        """Test workspace file finding via catalog lookup."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["test_step"]
            
            mock_step_info = Mock()
            mock_step_info.workspace_id = "test_dev"
            mock_contract = Mock()
            mock_contract.path = Path("/path/to/contract.py")
            mock_step_info.file_components = {"contract": mock_contract}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter._find_workspace_file("test_step", "contract", [".py"])
            
            assert result == "/path/to/contract.py"
    
    def test_find_workspace_file_directory_fallback(self, temp_workspace):
        """Test workspace file finding via directory fallback."""
        # Create a test contract file
        contract_file = temp_workspace / "developers" / "test_dev" / "contracts" / "test_step_contract.py"
        contract_file.write_text("# Test contract")
        
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = []
            mock_catalog.get_step_info.return_value = None
            mock_catalog_class.return_value = mock_catalog
            
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter._find_workspace_file("test_step", "contract", [".py"])
            
            assert result == str(contract_file)
    
    def test_find_workspace_file_shared_fallback(self, temp_workspace):
        """Test workspace file finding with shared fallback."""
        # Create a test contract file in shared workspace
        shared_contract = temp_workspace / "shared" / "contracts" / "test_step_contract.py"
        shared_contract.write_text("# Shared contract")
        
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = []
            mock_catalog.get_step_info.return_value = None
            mock_catalog_class.return_value = mock_catalog
            
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev",
                enable_shared_fallback=True
            )
            
            result = adapter._find_workspace_file("test_step", "contract", [".py"])
            
            assert result == str(shared_contract)
    
    def test_find_contract_file(self, temp_workspace):
        """Test finding contract file."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_contract = Mock()
            mock_contract.path = Path("/path/to/contract.py")
            mock_step_info.file_components = {"contract": mock_contract}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter.find_contract_file("test_step")
            
            assert result == "/path/to/contract.py"
    
    def test_find_spec_file(self, temp_workspace):
        """Test finding spec file."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_spec = Mock()
            mock_spec.path = Path("/path/to/spec.py")
            mock_step_info.file_components = {"spec": mock_spec}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter.find_spec_file("test_step")
            
            assert result == "/path/to/spec.py"
    
    def test_find_builder_file(self, temp_workspace):
        """Test finding builder file."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_builder = Mock()
            mock_builder.path = Path("/path/to/builder.py")
            mock_step_info.file_components = {"builder": mock_builder}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter.find_builder_file("test_step")
            
            assert result == "/path/to/builder.py"
    
    def test_find_config_file(self, temp_workspace):
        """Test finding config file."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_config = Mock()
            mock_config.path = Path("/path/to/config.py")
            mock_step_info.file_components = {"config": mock_config}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter.find_config_file("test_step")
            
            assert result == "/path/to/config.py"
    
    def test_find_script_file(self, temp_workspace):
        """Test finding script file."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_script = Mock()
            mock_script.path = Path("/path/to/script.py")
            mock_step_info.file_components = {"script": mock_script}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter.find_script_file("test_step")
            
            assert result == "/path/to/script.py"
    
    def test_find_file_in_directory_contract_patterns(self, temp_workspace):
        """Test finding file in directory with contract patterns."""
        # Create test contract files with different naming patterns
        contracts_dir = temp_workspace / "developers" / "test_dev" / "contracts"
        (contracts_dir / "test_step_contract.py").write_text("# Contract")
        
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter._find_file_in_directory(str(contracts_dir), "test_step", None, [".py"])
            
            assert result == str(contracts_dir / "test_step_contract.py")
    
    def test_find_file_in_directory_spec_patterns(self, temp_workspace):
        """Test finding file in directory with spec patterns."""
        # Create test spec files with different naming patterns
        specs_dir = temp_workspace / "developers" / "test_dev" / "specs"
        (specs_dir / "test_step_spec.py").write_text("# Spec")
        
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter._find_file_in_directory(str(specs_dir), "test_step", None, [".py"])
            
            assert result == str(specs_dir / "test_step_spec.py")
    
    def test_find_file_in_directory_builder_patterns(self, temp_workspace):
        """Test finding file in directory with builder patterns."""
        # Create test builder files with different naming patterns
        builders_dir = temp_workspace / "developers" / "test_dev" / "builders"
        (builders_dir / "builder_test_step.py").write_text("# Builder")
        
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter._find_file_in_directory(str(builders_dir), "test_step", None, [".py"])
            
            assert result == str(builders_dir / "builder_test_step.py")
    
    def test_find_file_in_directory_not_found(self, temp_workspace):
        """Test finding file in directory when not found."""
        contracts_dir = temp_workspace / "developers" / "test_dev" / "contracts"
        
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter._find_file_in_directory(str(contracts_dir), "nonexistent_step", None, [".py"])
            
            assert result is None
    
    def test_find_file_in_directory_nonexistent_dir(self, temp_workspace):
        """Test finding file in nonexistent directory."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            result = adapter._find_file_in_directory("/nonexistent/dir", "test_step", None, [".py"])
            
            assert result is None
    
    def test_get_workspace_info(self, temp_workspace):
        """Test getting workspace information."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev",
                enable_shared_fallback=True
            )
            
            info = adapter.get_workspace_info()
            
            assert info["workspace_mode"] is True
            assert info["workspace_root"] == str(temp_workspace)
            assert info["developer_id"] == "test_dev"
            assert info["enable_shared_fallback"] is True
            assert info["developer_workspace_exists"] is True
            assert info["shared_workspace_exists"] is True
    
    def test_get_workspace_info_single_mode(self):
        """Test getting workspace information in single workspace mode."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter()
            
            info = adapter.get_workspace_info()
            
            assert info["workspace_mode"] is False
            assert info["workspace_root"] is None
            assert info["developer_id"] is None
            assert info["enable_shared_fallback"] is False
    
    def test_get_workspace_info_error(self, temp_workspace):
        """Test error handling in get_workspace_info."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            # Mock workspace_root to cause an error
            adapter.workspace_root = Mock()
            adapter.workspace_root.__truediv__.side_effect = Exception("Path error")
            
            info = adapter.get_workspace_info()
            
            assert "error" in info
    
    def test_list_available_developers(self, temp_workspace):
        """Test listing available developers."""
        # Create additional developer workspaces
        (temp_workspace / "developers" / "dev1").mkdir(parents=True)
        (temp_workspace / "developers" / "dev2").mkdir(parents=True)
        
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            developers = adapter.list_available_developers()
            
            assert "test_dev" in developers
            assert "dev1" in developers
            assert "dev2" in developers
            assert len(developers) == 3
    
    def test_list_available_developers_single_mode(self):
        """Test listing developers in single workspace mode."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter()
            
            developers = adapter.list_available_developers()
            
            assert developers == []
    
    def test_list_available_developers_error(self, temp_workspace):
        """Test error handling in list_available_developers."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            # Mock workspace_root to cause an error
            adapter.workspace_root = Mock()
            adapter.workspace_root.__truediv__.side_effect = Exception("Path error")
            
            developers = adapter.list_available_developers()
            
            assert developers == []
    
    def test_switch_developer(self, temp_workspace):
        """Test switching to different developer workspace."""
        # Create additional developer workspace
        (temp_workspace / "developers" / "new_dev").mkdir(parents=True)
        
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            adapter.switch_developer("new_dev")
            
            assert adapter.developer_id == "new_dev"
            assert adapter.project_id == "new_dev"
    
    def test_switch_developer_single_mode(self):
        """Test switching developer in single workspace mode."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter()
            
            with pytest.raises(ValueError, match="Cannot switch developer in single workspace mode"):
                adapter.switch_developer("new_dev")
    
    def test_switch_developer_nonexistent(self, temp_workspace):
        """Test switching to nonexistent developer."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            with pytest.raises(ValueError, match="Developer workspace not found"):
                adapter.switch_developer("nonexistent_dev")


class TestHybridFileResolverAdapter:
    """Test HybridFileResolverAdapter functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_init(self, temp_workspace):
        """Test HybridFileResolverAdapter initialization."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog:
            adapter = HybridFileResolverAdapter(temp_workspace)
            
            # Should initialize StepCatalog with workspace_dirs=[workspace_root]
            mock_catalog.assert_called_once_with(workspace_dirs=[temp_workspace])
            assert adapter.logger is not None
    
    def test_resolve_file_pattern_success(self, temp_workspace):
        """Test successful file pattern resolution."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["xgboost_training", "data_loading", "model_evaluation"]
            
            # Mock step info for matching steps
            mock_step_info1 = Mock()
            mock_script1 = Mock()
            mock_script1.path = Path("/path/to/xgboost_training.py")
            mock_step_info1.file_components = {"script": mock_script1}
            
            mock_step_info2 = Mock()
            mock_step_info2.file_components = {}  # No script component
            
            mock_step_info3 = Mock()
            mock_script3 = Mock()
            mock_script3.path = Path("/path/to/model_evaluation.py")
            mock_step_info3.file_components = {"script": mock_script3}
            
            mock_catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2, mock_step_info3]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = HybridFileResolverAdapter(temp_workspace)
            result = adapter.resolve_file_pattern("model", "script")
            
            # Should find steps containing "model" in their name
            assert len(result) == 1
            assert Path("/path/to/model_evaluation.py") in result
    
    def test_resolve_file_pattern_no_matches(self, temp_workspace):
        """Test file pattern resolution with no matches."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["xgboost_training", "data_loading"]
            
            mock_step_info1 = Mock()
            mock_step_info1.file_components = {"script": Mock()}
            mock_step_info2 = Mock()
            mock_step_info2.file_components = {"script": Mock()}
            
            mock_catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = HybridFileResolverAdapter(temp_workspace)
            result = adapter.resolve_file_pattern("nonexistent", "script")
            
            assert result == []
    
    def test_resolve_file_pattern_error_handling(self, temp_workspace):
        """Test error handling in resolve_file_pattern."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.side_effect = Exception("Catalog error")
            mock_catalog_class.return_value = mock_catalog
            
            adapter = HybridFileResolverAdapter(temp_workspace)
            result = adapter.resolve_file_pattern("test", "script")
            
            assert result == []


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create comprehensive temporary workspace for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Create developer workspace structure
            dev_workspace = workspace_root / "developers" / "test_dev"
            dev_workspace.mkdir(parents=True)
            
            # Create component directories with test files
            contracts_dir = dev_workspace / "contracts"
            contracts_dir.mkdir()
            (contracts_dir / "xgboost_training_contract.py").write_text("# XGBoost training contract")
            
            specs_dir = dev_workspace / "specs"
            specs_dir.mkdir()
            (specs_dir / "xgboost_training_spec.py").write_text("# XGBoost training spec")
            
            builders_dir = dev_workspace / "builders"
            builders_dir.mkdir()
            (builders_dir / "builder_xgboost_training.py").write_text("# XGBoost training builder")
            
            scripts_dir = dev_workspace / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "xgboost_training.py").write_text("# XGBoost training script")
            
            # Create shared workspace with fallback files
            shared_workspace = workspace_root / "shared"
            shared_workspace.mkdir()
            shared_contracts = shared_workspace / "contracts"
            shared_contracts.mkdir()
            (shared_contracts / "data_loading_contract.py").write_text("# Shared data loading contract")
            
            yield workspace_root
    
    def test_complete_file_resolution_workflow(self, temp_workspace):
        """Test complete file resolution workflow across all adapters."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["XGBoostTraining", "DataLoading"]
            mock_catalog.get_step_info.return_value = None  # Force directory fallback
            mock_catalog_class.return_value = mock_catalog
            
            # Test FlexibleFileResolverAdapter
            flexible_adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # Test DeveloperWorkspaceFileResolverAdapter
            workspace_adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev",
                enable_shared_fallback=True
            )
            
            # Test HybridFileResolverAdapter
            hybrid_adapter = HybridFileResolverAdapter(temp_workspace)
            
            # Test file finding across adapters
            contract_result = workspace_adapter.find_contract_file("xgboost_training")
            spec_result = workspace_adapter.find_spec_file("xgboost_training")
            builder_result = workspace_adapter.find_builder_file("xgboost_training")
            script_result = workspace_adapter.find_script_file("xgboost_training")
            
            # Should find files via directory fallback
            assert contract_result is not None
            assert "xgboost_training_contract.py" in contract_result
            assert spec_result is not None
            assert "xgboost_training_spec.py" in spec_result
            assert builder_result is not None
            assert "builder_xgboost_training.py" in builder_result
            assert script_result is not None
            assert "xgboost_training.py" in script_result
    
    def test_workspace_fallback_mechanism(self, temp_workspace):
        """Test workspace fallback from developer to shared workspace."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = []
            mock_catalog.get_step_info.return_value = None
            mock_catalog_class.return_value = mock_catalog
            
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev",
                enable_shared_fallback=True
            )
            
            # Should find shared contract via fallback
            result = adapter.find_contract_file("data_loading")
            
            assert result is not None
            assert "data_loading_contract.py" in result
            assert "shared" in result


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_flexible_adapter_with_catalog_failure(self, temp_workspace):
        """Test FlexibleFileResolverAdapter with catalog failure."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = Exception("Catalog initialization failed")
            
            # Should still initialize but methods will fail gracefully
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # All methods should handle errors gracefully
            assert adapter.find_contract_file("test_step") is None
            assert adapter.find_spec_file("test_step") is None
            assert adapter.find_builder_file("test_step") is None
            assert adapter.find_config_file("test_step") is None
            assert adapter.find_all_component_files("test_step") == {}
    
    def test_workspace_adapter_with_invalid_paths(self, temp_workspace):
        """Test workspace adapter with invalid file paths."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = DeveloperWorkspaceFileResolverAdapter(
                workspace_root=temp_workspace,
                developer_id="test_dev"
            )
            
            # Test with None directory
            result = adapter._find_file_in_directory(None, "test_step", None, [".py"])
            assert result is None
            
            # Test with empty directory string
            result = adapter._find_file_in_directory("", "test_step", None, [".py"])
            assert result is None
    
    def test_name_normalization_edge_cases(self, temp_workspace):
        """Test name normalization with edge cases."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # Test with empty string
            assert adapter._normalize_name("") == ""
            
            # Test with special characters
            assert adapter._normalize_name("test@#$%") == "test@#$%"
            
            # Test with multiple replacements
            assert adapter._normalize_name("test-name.with_eval") == "test_name_with_evaluation"
    
    def test_similarity_calculation_edge_cases(self, temp_workspace):
        """Test similarity calculation with edge cases."""
        with patch('cursus.step_catalog.adapters.file_resolver.StepCatalog'):
            adapter = FlexibleFileResolverAdapter(temp_workspace)
            
            # Test with empty strings
            assert adapter._calculate_similarity("", "") == 1.0
            assert adapter._calculate_similarity("test", "") == 0.0
            assert adapter._calculate_similarity("", "test") == 0.0
            
            # Test with None (should handle gracefully)
            try:
                adapter._calculate_similarity(None, "test")
            except (TypeError, AttributeError):
                # It's acceptable for this to raise an exception with None input
                pass
