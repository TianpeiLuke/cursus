"""
Unit tests for step_catalog.adapters.legacy_wrappers module.

Tests the LegacyDiscoveryWrapper class and legacy functions that provide 
backward compatibility with legacy discovery systems.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List

from cursus.step_catalog.adapters.legacy_wrappers import (
    LegacyDiscoveryWrapper,
    build_complete_config_classes,
    detect_config_classes_from_json
)


class TestLegacyDiscoveryWrapper:
    """Test LegacyDiscoveryWrapper functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_init(self, temp_workspace):
        """Test LegacyDiscoveryWrapper initialization."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog:
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter') as mock_contract_engine:
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter') as mock_contract_manager:
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter') as mock_file_resolver:
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter') as mock_workspace_manager:
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter') as mock_hybrid_resolver:
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Should initialize StepCatalog with workspace_dirs=[workspace_root]
                                mock_catalog.assert_called_once_with(workspace_dirs=[temp_workspace])
                                
                                # Should initialize all adapters
                                mock_contract_engine.assert_called_once_with(temp_workspace)
                                mock_contract_manager.assert_called_once_with(temp_workspace)
                                mock_file_resolver.assert_called_once_with(temp_workspace)
                                mock_workspace_manager.assert_called_once_with(temp_workspace)
                                mock_hybrid_resolver.assert_called_once_with(temp_workspace)
                                
                                assert wrapper.workspace_root == temp_workspace
                                assert wrapper.logger is not None
    
    def test_refresh_cache_success(self, temp_workspace):
        """Test successful cache refresh."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
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
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Check cache was populated
                                assert hasattr(wrapper, 'file_cache')
                                assert "scripts" in wrapper.file_cache
                                assert "contracts" in wrapper.file_cache
                                assert "specs" in wrapper.file_cache
    
    def test_refresh_cache_error_handling(self, temp_workspace):
        """Test error handling in cache refresh."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.side_effect = Exception("Catalog error")
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Should handle error gracefully
                                assert hasattr(wrapper, 'file_cache')
    
    def test_extract_base_name(self, temp_workspace):
        """Test base name extraction from step names."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog'):
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Test PascalCase to snake_case conversion
                                assert wrapper._extract_base_name("XGBoostTraining", "script") == "x_g_boost_training"
                                assert wrapper._extract_base_name("DataLoading", "contract") == "data_loading"
                                assert wrapper._extract_base_name("SimpleStep", "spec") == "simple_step"
    
    def test_normalize_name(self, temp_workspace):
        """Test name normalization."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog'):
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Test various normalizations
                                assert wrapper._normalize_name("test-name") == "test_name"
                                assert wrapper._normalize_name("test.name") == "test_name"
                                assert wrapper._normalize_name("preprocess") == "preprocessing"
                                assert wrapper._normalize_name("eval") == "evaluation"
                                assert wrapper._normalize_name("xgb") == "xgboost"
                                assert wrapper._normalize_name("train") == "training"
    
    def test_calculate_similarity(self, temp_workspace):
        """Test similarity calculation."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog'):
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Test similarity scores
                                assert wrapper._calculate_similarity("test", "test") == 1.0
                                assert wrapper._calculate_similarity("test", "TEST") == 1.0
                                assert wrapper._calculate_similarity("test", "testing") > 0.5
                                assert wrapper._calculate_similarity("test", "completely_different") < 0.5
    
    def test_find_best_match_direct_catalog(self, temp_workspace):
        """Test finding best match via direct catalog lookup."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_script = Mock()
            mock_script.path = Path("/path/to/script.py")
            mock_step_info.file_components = {"script": mock_script}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                result = wrapper._find_best_match("test_step", "script")
                                
                                assert result == "/path/to/script.py"
                                mock_catalog.get_step_info.assert_called_with("test_step")
    
    def test_find_best_match_fuzzy_matching(self, temp_workspace):
        """Test finding best match via fuzzy matching."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
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
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                result = wrapper._find_best_match("xgboost_training", "script")
                                
                                assert result == "/path/to/xgboost_training.py"
    
    def test_find_best_match_no_match(self, temp_workspace):
        """Test finding best match when no match exists."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.get_step_info.return_value = None
            mock_catalog.list_available_steps.return_value = []
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                result = wrapper._find_best_match("nonexistent_step", "script")
                                
                                assert result is None
    
    def test_get_available_files_report(self, temp_workspace):
        """Test generating available files report."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["step1"]
            
            mock_step_info = Mock()
            mock_script = Mock()
            mock_script.path = Path("/path/to/step1.py")
            mock_step_info.file_components = {"script": mock_script}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                report = wrapper.get_available_files_report()
                                
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
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog'):
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Test spec file name extraction
                                assert wrapper.extract_base_name_from_spec(Path("test_step_spec.py")) == "test_step"
                                assert wrapper.extract_base_name_from_spec(Path("xgboost_training_spec.py")) == "xgboost_training"
                                assert wrapper.extract_base_name_from_spec(Path("simple_spec.py")) == "simple"
                                assert wrapper.extract_base_name_from_spec(Path("no_spec_suffix.py")) == "no_spec_suffix"
    
    def test_find_spec_constant_name(self, temp_workspace):
        """Test finding spec constant name."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog'):
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Mock find_spec_file to return the path
                                with patch.object(wrapper, 'find_spec_file', return_value="/path/to/test_step_spec.py"):
                                    result = wrapper.find_spec_constant_name("test_step")
                                    assert result == "TEST_STEP_TRAINING_SPEC"
                                    
                                    result = wrapper.find_spec_constant_name("test_step", "validation")
                                    assert result == "TEST_STEP_VALIDATION_SPEC"
    
    def test_find_spec_constant_name_no_spec(self, temp_workspace):
        """Test finding spec constant name when no spec file exists."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog'):
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                with patch.object(wrapper, 'find_spec_file', return_value=None):
                                    result = wrapper.find_spec_constant_name("test_step")
                                    assert result == "TEST_STEP_TRAINING_SPEC"
    
    def test_find_specification_file_legacy_alias(self, temp_workspace):
        """Test legacy alias for find_spec_file."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog'):
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                with patch.object(wrapper, 'find_spec_file', return_value="/path/to/spec.py"):
                                    result = wrapper.find_specification_file("test_step")
                                    assert result == Path("/path/to/spec.py")
    
    def test_find_contract_file(self, temp_workspace):
        """Test finding contract file."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_contract = Mock()
            mock_contract.path = Path("/path/to/contract.py")
            mock_step_info.file_components = {"contract": mock_contract}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                result = wrapper.find_contract_file("test_step")
                                
                                assert result == Path("/path/to/contract.py")
    
    def test_find_contract_file_not_found(self, temp_workspace):
        """Test finding contract file when not found."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.get_step_info.return_value = None
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                result = wrapper.find_contract_file("nonexistent_step")
                                
                                assert result is None
    
    def test_find_spec_file(self, temp_workspace):
        """Test finding spec file."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_spec = Mock()
            mock_spec.path = Path("/path/to/spec.py")
            mock_step_info.file_components = {"spec": mock_spec}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                result = wrapper.find_spec_file("test_step")
                                
                                assert result == Path("/path/to/spec.py")
    
    def test_find_builder_file(self, temp_workspace):
        """Test finding builder file."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_builder = Mock()
            mock_builder.path = Path("/path/to/builder.py")
            mock_step_info.file_components = {"builder": mock_builder}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                result = wrapper.find_builder_file("test_step")
                                
                                assert result == Path("/path/to/builder.py")
    
    def test_find_config_file(self, temp_workspace):
        """Test finding config file."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_step_info = Mock()
            mock_config = Mock()
            mock_config.path = Path("/path/to/config.py")
            mock_step_info.file_components = {"config": mock_config}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                result = wrapper.find_config_file("test_step")
                                
                                assert result == Path("/path/to/config.py")
    
    def test_find_all_component_files(self, temp_workspace):
        """Test finding all component files for a step."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
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
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                result = wrapper.find_all_component_files("test_step")
                                
                                assert result["script"] == Path("/path/to/script.py")
                                assert result["contract"] == Path("/path/to/contract.py")
                                assert result["spec"] == Path("/path/to/spec.py")
                                assert result["builder"] is None
                                assert result["config"] is None
    
    def test_find_all_component_files_error(self, temp_workspace):
        """Test error handling in find_all_component_files."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.get_step_info.side_effect = Exception("Catalog error")
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                result = wrapper.find_all_component_files("test_step")
                                
                                assert result == {}
    
    def test_delegation_methods(self, temp_workspace):
        """Test that wrapper properly delegates StepCatalog methods."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Test delegation methods
                                wrapper.get_step_info("test_step")
                                mock_catalog.get_step_info.assert_called_with("test_step", None)
                                
                                wrapper.find_step_by_component("test_component")
                                mock_catalog.find_step_by_component.assert_called_with("test_component")
                                
                                wrapper.list_available_steps()
                                mock_catalog.list_available_steps.assert_called_with(None, None)
                                
                                wrapper.search_steps("test_query")
                                mock_catalog.search_steps.assert_called_with("test_query", None)
                                
                                wrapper.discover_config_classes()
                                mock_catalog.discover_config_classes.assert_called_with(None)
                                
                                wrapper.build_complete_config_classes()
                                mock_catalog.build_complete_config_classes.assert_called_with(None)
                                
                                wrapper.get_job_type_variants("test_step")
                                mock_catalog.get_job_type_variants.assert_called_with("test_step")
                                
                                wrapper.get_metrics_report()
                                mock_catalog.get_metrics_report.assert_called_once()
                                
                                wrapper.discover_contracts_with_scripts()
                                mock_catalog.discover_contracts_with_scripts.assert_called_once()
                                
                                wrapper.detect_framework("test_step")
                                mock_catalog.detect_framework.assert_called_with("test_step")
                                
                                wrapper.discover_cross_workspace_components()
                                mock_catalog.discover_cross_workspace_components.assert_called_with(None)
                                
                                wrapper.get_builder_class_path("test_step")
                                mock_catalog.get_builder_class_path.assert_called_with("test_step")
                                
                                wrapper.load_builder_class("test_step")
                                mock_catalog.load_builder_class.assert_called_with("test_step")
    
    def test_get_adapter(self, temp_workspace):
        """Test getting specific legacy adapters."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog'):
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter') as mock_contract_engine:
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter') as mock_contract_manager:
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter') as mock_file_resolver:
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter') as mock_workspace_manager:
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter') as mock_hybrid_resolver:
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Test getting specific adapters
                                assert wrapper.get_adapter('contract_discovery_engine') == wrapper.contract_discovery_engine
                                assert wrapper.get_adapter('contract_discovery_manager') == wrapper.contract_discovery_manager
                                assert wrapper.get_adapter('flexible_file_resolver') == wrapper.flexible_file_resolver
                                assert wrapper.get_adapter('workspace_discovery_manager') == wrapper.workspace_discovery_manager
                                assert wrapper.get_adapter('hybrid_file_resolver') == wrapper.hybrid_file_resolver
                                
                                # Test getting non-existent adapter
                                assert wrapper.get_adapter('nonexistent_adapter') is None
    
    def test_get_unified_catalog(self, temp_workspace):
        """Test getting the underlying unified catalog."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                catalog = wrapper.get_unified_catalog()
                                
                                assert catalog == mock_catalog


class TestLegacyFunctions:
    """Test legacy functions for backward compatibility."""
    
    def test_build_complete_config_classes_success(self):
        """Test successful build_complete_config_classes function."""
        mock_config_classes = {
            "TestConfig": Mock(),
            "AnotherConfig": Mock()
        }
        
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.build_complete_config_classes.return_value = mock_config_classes
            mock_catalog_class.return_value = mock_catalog
            
            result = build_complete_config_classes("test_project")
            
            # Should create StepCatalog with workspace_dirs=None
            mock_catalog_class.assert_called_once_with(workspace_dirs=None)
            # Should call build_complete_config_classes with project_id
            mock_catalog.build_complete_config_classes.assert_called_once_with("test_project")
            # Should return the config classes
            assert result == mock_config_classes
    
    def test_build_complete_config_classes_error_fallback(self):
        """Test fallback to ConfigClassStoreAdapter when StepCatalog fails."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = Exception("Test error")
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ConfigClassStoreAdapter') as mock_store:
                mock_store.get_all_classes.return_value = {"FallbackConfig": Mock()}
                
                result = build_complete_config_classes()
                
                # Should fallback to ConfigClassStoreAdapter
                mock_store.get_all_classes.assert_called_once()
                assert "FallbackConfig" in result
    
    def test_detect_config_classes_from_json(self):
        """Test detect_config_classes_from_json function."""
        mock_config_classes = {"TestConfig": Mock()}
        
        with patch('cursus.step_catalog.adapters.legacy_wrappers.ConfigClassDetectorAdapter') as mock_detector:
            mock_detector.detect_from_json.return_value = mock_config_classes
            
            result = detect_config_classes_from_json("/test/config.json")
            
            mock_detector.detect_from_json.assert_called_once_with("/test/config.json")
            assert result == mock_config_classes


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_complete_legacy_wrapper_workflow(self, temp_workspace):
        """Test complete legacy wrapper workflow."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["XGBoostTraining", "DataLoading"]
            
            # Mock step info for complete workflow
            mock_step_info = Mock()
            mock_script = Mock()
            mock_script.path = Path("/path/to/xgboost_training.py")
            mock_contract = Mock()
            mock_contract.path = Path("/path/to/xgboost_training_contract.py")
            mock_spec = Mock()
            mock_spec.path = Path("/path/to/xgboost_training_spec.py")
            mock_builder = Mock()
            mock_builder.path = Path("/path/to/xgboost_training_builder.py")
            
            mock_step_info.file_components = {
                "script": mock_script,
                "contract": mock_contract,
                "spec": mock_spec,
                "builder": mock_builder
            }
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Test complete workflow
                                steps = wrapper.list_available_steps()
                                contract_file = wrapper.find_contract_file("XGBoostTraining")
                                spec_file = wrapper.find_spec_file("XGBoostTraining")
                                builder_file = wrapper.find_builder_file("XGBoostTraining")
                                all_files = wrapper.find_all_component_files("XGBoostTraining")
                                report = wrapper.get_available_files_report()
                                
                                # Verify results
                                assert steps == ["XGBoostTraining", "DataLoading"]
                                assert contract_file == Path("/path/to/xgboost_training_contract.py")
                                assert spec_file == Path("/path/to/xgboost_training_spec.py")
                                assert builder_file == Path("/path/to/xgboost_training_builder.py")
                                assert len(all_files) == 4
                                assert isinstance(report, dict)
    
    def test_adapter_integration(self, temp_workspace):
        """Test integration with all adapters."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog'):
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter') as mock_contract_engine:
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter') as mock_contract_manager:
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter') as mock_file_resolver:
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter') as mock_workspace_manager:
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter') as mock_hybrid_resolver:
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Test that all adapters are accessible
                                assert wrapper.contract_discovery_engine is not None
                                assert wrapper.contract_discovery_manager is not None
                                assert wrapper.flexible_file_resolver is not None
                                assert wrapper.workspace_discovery_manager is not None
                                assert wrapper.hybrid_file_resolver is not None
                                
                                # Test that adapters were initialized with correct parameters
                                mock_contract_engine.assert_called_once_with(temp_workspace)
                                mock_contract_manager.assert_called_once_with(temp_workspace)
                                mock_file_resolver.assert_called_once_with(temp_workspace)
                                mock_workspace_manager.assert_called_once_with(temp_workspace)
                                mock_hybrid_resolver.assert_called_once_with(temp_workspace)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_wrapper_with_catalog_failure(self, temp_workspace):
        """Test wrapper behavior when catalog fails."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = Exception("Catalog initialization failed")
            
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                # Should still initialize but catalog methods will fail
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Methods should handle catalog failures gracefully
                                assert wrapper.find_contract_file("test_step") is None
                                assert wrapper.find_spec_file("test_step") is None
                                assert wrapper.find_builder_file("test_step") is None
                                assert wrapper.find_config_file("test_step") is None
                                assert wrapper.find_all_component_files("test_step") == {}
    
    def test_similarity_calculation_edge_cases(self, temp_workspace):
        """Test similarity calculation with edge cases."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog'):
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Test with empty strings
                                assert wrapper._calculate_similarity("", "") == 1.0
                                assert wrapper._calculate_similarity("test", "") == 0.0
                                assert wrapper._calculate_similarity("", "test") == 0.0
                                
                                # Test with None (should handle gracefully)
                                try:
                                    wrapper._calculate_similarity(None, "test")
                                except (TypeError, AttributeError):
                                    # It's acceptable for this to raise an exception with None input
                                    pass
    
    def test_name_normalization_edge_cases(self, temp_workspace):
        """Test name normalization with edge cases."""
        with patch('cursus.step_catalog.adapters.legacy_wrappers.StepCatalog'):
            with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryEngineAdapter'):
                with patch('cursus.step_catalog.adapters.legacy_wrappers.ContractDiscoveryManagerAdapter'):
                    with patch('cursus.step_catalog.adapters.legacy_wrappers.FlexibleFileResolverAdapter'):
                        with patch('cursus.step_catalog.adapters.legacy_wrappers.WorkspaceDiscoveryManagerAdapter'):
                            with patch('cursus.step_catalog.adapters.legacy_wrappers.HybridFileResolverAdapter'):
                                
                                wrapper = LegacyDiscoveryWrapper(temp_workspace)
                                
                                # Test with empty string
                                assert wrapper._normalize_name("") == ""
                                
                                # Test with special characters
                                assert wrapper._normalize_name("test@#$%") == "test@#$%"
                                
                                # Test with multiple replacements
                                assert wrapper._normalize_name("test-name.with_eval") == "test_name_with_evaluation"
