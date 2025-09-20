"""
Unit tests for step_catalog.contract_discovery module.

Tests the ContractAutoDiscovery class that implements contract object discovery
from both core and workspace directories, with automatic object detection and
PascalCase to snake_case conversion.

Based on extensive debugging and trial-and-error process that identified
critical issues with relative import paths and contract object discovery.
"""

import pytest
import tempfile
import ast
import importlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Type, Any, List

from cursus.step_catalog.contract_discovery import ContractAutoDiscovery


class TestContractAutoDiscovery:
    """Test cases for ContractAutoDiscovery class."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Create core contract directory structure
            core_contract_dir = workspace_root / "cursus" / "steps" / "contracts"
            core_contract_dir.mkdir(parents=True, exist_ok=True)
            
            # Create workspace contract directory structure
            workspace_contract_dir = (
                workspace_root / "development" / "projects" / "test_project" / 
                "src" / "cursus_dev" / "steps" / "contracts"
            )
            workspace_contract_dir.mkdir(parents=True, exist_ok=True)
            
            yield workspace_root, core_contract_dir, workspace_contract_dir
    
    @pytest.fixture
    def contract_discovery(self, temp_workspace):
        """Create ContractAutoDiscovery instance with temporary workspace."""
        workspace_root, _, _ = temp_workspace
        package_root = workspace_root / "cursus"
        return ContractAutoDiscovery(package_root, [workspace_root])
    
    def test_init(self, temp_workspace):
        """Test ContractAutoDiscovery initialization."""
        workspace_root, _, _ = temp_workspace
        package_root = workspace_root / "cursus"
        discovery = ContractAutoDiscovery(package_root, [workspace_root])
        
        assert discovery.package_root == package_root
        assert discovery.workspace_dirs == [workspace_root]
        assert discovery.logger is not None
    
    def test_init_no_workspace(self, temp_workspace):
        """Test ContractAutoDiscovery initialization without workspace directories."""
        workspace_root, _, _ = temp_workspace
        package_root = workspace_root / "cursus"
        discovery = ContractAutoDiscovery(package_root, None)
        
        assert discovery.package_root == package_root
        assert discovery.workspace_dirs is None
    
    def test_pascal_to_snake_case_conversion(self, contract_discovery):
        """Test PascalCase to snake_case conversion - critical for contract discovery."""
        # Test cases based on our debugging experience
        test_cases = [
            ("XGBoostTraining", "xgboost_training"),
            ("XGBoostModel", "xgboost_model"),
            ("XGBoostModelEval", "xgboost_model_eval"),
            ("SimpleStep", "simple_step"),
            ("MLModelTraining", "mlmodel_training"),
            ("DataProcessing", "data_processing"),
            ("already_snake_case", "already_snake_case"),
            ("SingleWord", "single_word"),
            ("ABC", "abc"),
            ("XMLParser", "xmlparser"),
        ]
        
        for pascal_case, expected_snake_case in test_cases:
            result = contract_discovery._pascal_to_snake_case(pascal_case)
            assert result == expected_snake_case, f"Failed for {pascal_case}: got {result}, expected {expected_snake_case}"
    
    def test_is_contract_object_naming_patterns(self, contract_discovery):
        """Test contract object detection by naming patterns."""
        # Create a proper mock that won't be detected by attribute checking
        def create_non_contract_mock():
            mock = Mock()
            # Remove the dynamic attributes that make Mock look like a contract
            del mock.entry_point
            del mock.expected_input_paths
            del mock.expected_output_paths
            return mock
        
        # Test _CONTRACT suffix (most common pattern)
        assert contract_discovery._is_contract_object("XGBOOST_TRAIN_CONTRACT", Mock()) == True
        assert contract_discovery._is_contract_object("STEP_CONTRACT", Mock()) == True
        
        # Test Contract suffix
        assert contract_discovery._is_contract_object("MyContract", Mock()) == True
        assert contract_discovery._is_contract_object("ProcessingContract", Mock()) == True
        
        # Test non-contract names with objects that don't have contract attributes
        regular_obj = object()  # Plain object without contract attributes
        assert contract_discovery._is_contract_object("regular_variable", regular_obj) == False
        assert contract_discovery._is_contract_object("SomeClass", regular_obj) == False
        assert contract_discovery._is_contract_object("CONSTANT_VALUE", regular_obj) == False
    
    def test_is_contract_object_by_attributes(self, contract_discovery):
        """Test contract object detection by attributes - based on real contract structure."""
        # Mock contract object with expected attributes (like TrainingScriptContract)
        mock_contract = Mock()
        mock_contract.expected_input_paths = {"input_path": "/opt/ml/input/data"}
        mock_contract.expected_output_paths = {"model_output": "/opt/ml/model"}
        mock_contract.entry_point = "training_script.py"
        
        assert contract_discovery._is_contract_object("some_name", mock_contract) == True
        
        # Test with only entry_point
        mock_entry_point_only = Mock()
        mock_entry_point_only.entry_point = "script.py"
        assert contract_discovery._is_contract_object("some_name", mock_entry_point_only) == True
        
        # Test with Contract in type name
        mock_contract_type = Mock()
        type(mock_contract_type).__name__ = "TrainingScriptContract"
        assert contract_discovery._is_contract_object("some_name", mock_contract_type) == True
        
        # Test object without contract attributes - use a simple object
        class SimpleObject:
            def __init__(self):
                self.some_attr = "value"
        
        simple_obj = SimpleObject()
        assert contract_discovery._is_contract_object("some_name", simple_obj) == False
    
    def test_discover_contract_objects_in_module(self, contract_discovery):
        """Test automatic contract object discovery in module - core functionality."""
        # Create a simple mock module that won't cause recursion
        class MockModule:
            def __init__(self):
                # Contract objects (should be found)
                self.TRAINING_CONTRACT = type('ContractObj', (), {
                    'entry_point': 'training.py',
                    'expected_input_paths': {'input': '/path'},
                    'expected_output_paths': {'output': '/path'}
                })()
                
                self.EvalContract = type('ContractObj', (), {
                    'entry_point': 'eval.py'
                })()
                
                # Non-contract objects (should be ignored)
                self.SomeClass = type("MockClass", (), {})
                self.some_function = lambda: None
                self.regular_object = "just a string"
                self._private_attr = "private"
        
        mock_module = MockModule()
        
        result = contract_discovery._discover_contract_objects_in_module(mock_module)
        
        # Should find exactly 2 contract objects
        assert len(result) == 2
        contract_names = [name for name, obj in result]
        assert "TRAINING_CONTRACT" in contract_names
        assert "EvalContract" in contract_names
    
    def test_discover_contract_objects_error_handling(self, contract_discovery):
        """Test error handling in contract object discovery."""
        # Mock module that raises exception during attribute access
        mock_module = Mock()
        
        with patch('builtins.dir', side_effect=Exception("Module access error")):
            result = contract_discovery._discover_contract_objects_in_module(mock_module)
        
        # Should return empty list on error
        assert result == []
    
    def test_try_direct_import_success(self, contract_discovery):
        """Test successful direct import - the core bug we fixed."""
        step_name = "test_step"
        
        # Mock successful module import
        mock_module = Mock()
        mock_contract = Mock()
        mock_contract.entry_point = "test_step.py"
        
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = mock_module
            
            with patch.object(contract_discovery, '_discover_contract_objects_in_module') as mock_discover:
                mock_discover.return_value = [("TEST_STEP_CONTRACT", mock_contract)]
                
                result = contract_discovery._try_direct_import(step_name)
                
                # Should return the contract object
                assert result == mock_contract
                
                # Should use correct relative import path (..steps.contracts, not ...steps.contracts)
                mock_import.assert_called_once_with(
                    f"..steps.contracts.{step_name}_contract",
                    package="cursus.step_catalog"
                )
    
    def test_try_direct_import_import_error(self, contract_discovery):
        """Test direct import with ImportError - should be handled gracefully."""
        step_name = "nonexistent_step"
        
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No module named 'nonexistent_step_contract'")
            
            result = contract_discovery._try_direct_import(step_name)
            
            # Should return None on import error
            assert result is None
    
    def test_try_direct_import_no_contracts_found(self, contract_discovery):
        """Test direct import when module exists but no contracts found."""
        step_name = "empty_step"
        
        mock_module = Mock()
        
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = mock_module
            
            with patch.object(contract_discovery, '_discover_contract_objects_in_module') as mock_discover:
                mock_discover.return_value = []  # No contracts found
                
                result = contract_discovery._try_direct_import(step_name)
                
                # Should return None when no contracts found
                assert result is None
    
    def test_try_direct_import_beyond_top_level_package_error(self, contract_discovery):
        """Test the specific bug we fixed: 'attempted relative import beyond top-level package'."""
        step_name = "test_step"
        
        with patch('importlib.import_module') as mock_import:
            # This was the original error we encountered
            mock_import.side_effect = ImportError("attempted relative import beyond top-level package")
            
            result = contract_discovery._try_direct_import(step_name)
            
            # Should handle this specific error gracefully
            assert result is None
            
            # Should have tried with correct path (..steps.contracts, not ...steps.contracts)
            mock_import.assert_called_once_with(
                f"..steps.contracts.{step_name}_contract",
                package="cursus.step_catalog"
            )
    
    def test_load_contract_class_direct_success(self, contract_discovery):
        """Test successful contract loading via direct import."""
        step_name = "XGBoostTraining"
        mock_contract = Mock()
        mock_contract.entry_point = "xgboost_training.py"
        
        with patch.object(contract_discovery, '_try_direct_import') as mock_direct:
            mock_direct.return_value = mock_contract
            
            result = contract_discovery.load_contract_class(step_name)
            
            assert result == mock_contract
            mock_direct.assert_called_once_with(step_name)
    
    def test_load_contract_class_snake_case_conversion(self, contract_discovery):
        """Test contract loading with PascalCase to snake_case conversion."""
        step_name = "XGBoostTraining"
        mock_contract = Mock()
        mock_contract.entry_point = "xgboost_training.py"
        
        with patch.object(contract_discovery, '_try_direct_import') as mock_direct:
            # First call (original name) fails, second call (snake_case) succeeds
            mock_direct.side_effect = [None, mock_contract]
            
            result = contract_discovery.load_contract_class(step_name)
            
            assert result == mock_contract
            # Should be called twice: once with original name, once with snake_case
            assert mock_direct.call_count == 2
            mock_direct.assert_any_call("XGBoostTraining")
            mock_direct.assert_any_call("xgboost_training")
    
    def test_load_contract_class_workspace_fallback(self, contract_discovery):
        """Test contract loading fallback to workspace discovery."""
        step_name = "WorkspaceStep"
        mock_contract = Mock()
        mock_contract.entry_point = "workspace_step.py"
        
        with patch.object(contract_discovery, '_try_direct_import') as mock_direct:
            mock_direct.return_value = None  # Direct import fails
            
            with patch.object(contract_discovery, '_try_workspace_contract_import') as mock_workspace:
                mock_workspace.return_value = mock_contract
                
                result = contract_discovery.load_contract_class(step_name)
                
                assert result == mock_contract
                # Should try workspace import
                mock_workspace.assert_called()
    
    def test_load_contract_class_not_found(self, contract_discovery):
        """Test contract loading when contract is not found anywhere."""
        step_name = "NonexistentStep"
        
        with patch.object(contract_discovery, '_try_direct_import') as mock_direct:
            mock_direct.return_value = None
            
            with patch.object(contract_discovery, '_try_workspace_contract_import') as mock_workspace:
                mock_workspace.return_value = None
                
                result = contract_discovery.load_contract_class(step_name)
                
                assert result is None
    
    def test_load_contract_class_no_workspace_dirs(self):
        """Test contract loading when no workspace directories are configured."""
        package_root = Path("/test")
        discovery = ContractAutoDiscovery(package_root, None)
        
        step_name = "TestStep"
        
        with patch.object(discovery, '_try_direct_import') as mock_direct:
            mock_direct.return_value = None
            
            result = discovery.load_contract_class(step_name)
            
            # Should not attempt workspace discovery
            assert result is None
            mock_direct.assert_called()
    
    def test_try_workspace_contract_import_success(self, contract_discovery):
        """Test successful workspace contract import."""
        step_name = "workspace_step"
        workspace_dir = Path("/workspace")
        mock_contract = Mock()
        
        with patch.object(contract_discovery, '_load_contract_from_file') as mock_load:
            mock_load.return_value = mock_contract
            
            # Mock the file existence check and path operations
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.return_value = True
                
                with patch('pathlib.Path.iterdir') as mock_iterdir:
                    # Create a proper mock project directory
                    mock_project_dir = Mock(spec=Path)
                    mock_project_dir.is_dir.return_value = True
                    mock_project_dir.name = "test_project"
                    
                    # Mock the contract file path
                    mock_contract_file = Mock(spec=Path)
                    mock_contract_file.exists.return_value = True
                    
                    # Set up the path chain: project_dir / "src" / "cursus_dev" / "steps" / "contracts" / f"{step_name}_contract.py"
                    mock_project_dir.__truediv__ = Mock(return_value=Mock(
                        __truediv__=Mock(return_value=Mock(
                            __truediv__=Mock(return_value=Mock(
                                __truediv__=Mock(return_value=Mock(
                                    __truediv__=Mock(return_value=mock_contract_file)
                                ))
                            ))
                        ))
                    ))
                    
                    mock_iterdir.return_value = [mock_project_dir]
                    
                    result = contract_discovery._try_workspace_contract_import(step_name, workspace_dir)
                    
                    assert result == mock_contract
    
    def test_try_workspace_contract_import_no_projects_dir(self, contract_discovery):
        """Test workspace contract import when projects directory doesn't exist."""
        step_name = "test_step"
        workspace_dir = Path("/workspace")
        
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False  # projects_dir doesn't exist
            
            result = contract_discovery._try_workspace_contract_import(step_name, workspace_dir)
            
            assert result is None
    
    def test_load_contract_from_file_success(self, contract_discovery):
        """Test successful contract loading from file."""
        contract_path = Path("/path/to/contract.py")
        step_name = "test_step"
        mock_contract = Mock()
        
        with patch('importlib.util.spec_from_file_location') as mock_spec_from_file:
            mock_spec = Mock()
            mock_loader = Mock()
            mock_spec.loader = mock_loader
            mock_spec_from_file.return_value = mock_spec
            
            with patch('importlib.util.module_from_spec') as mock_module_from_spec:
                mock_module = Mock()
                mock_module.TEST_STEP_CONTRACT = mock_contract
                mock_module_from_spec.return_value = mock_module
                
                result = contract_discovery._load_contract_from_file(contract_path, step_name)
                
                assert result == mock_contract
                mock_loader.exec_module.assert_called_once_with(mock_module)
    
    def test_load_contract_from_file_fallback_search(self, contract_discovery):
        """Test contract loading from file with fallback attribute search."""
        contract_path = Path("/path/to/contract.py")
        step_name = "test_step"
        mock_contract = Mock()
        
        with patch('importlib.util.spec_from_file_location') as mock_spec_from_file:
            mock_spec = Mock()
            mock_loader = Mock()
            mock_spec.loader = mock_loader
            mock_spec_from_file.return_value = mock_spec
            
            with patch('importlib.util.module_from_spec') as mock_module_from_spec:
                # Create a simple class instead of Mock to avoid recursion issues
                class MockModule:
                    def __init__(self):
                        self.SOME_OTHER_CONTRACT = mock_contract
                        self.regular_attr = "value"
                
                mock_module = MockModule()
                mock_module_from_spec.return_value = mock_module
                
                result = contract_discovery._load_contract_from_file(contract_path, step_name)
                
                assert result == mock_contract
    
    def test_load_contract_from_file_no_spec(self, contract_discovery):
        """Test contract loading from file when spec creation fails."""
        contract_path = Path("/path/to/contract.py")
        step_name = "test_step"
        
        with patch('importlib.util.spec_from_file_location') as mock_spec_from_file:
            mock_spec_from_file.return_value = None  # Spec creation failed
            
            result = contract_discovery._load_contract_from_file(contract_path, step_name)
            
            assert result is None
    
    def test_load_contract_from_file_execution_error(self, contract_discovery):
        """Test contract loading from file with execution error."""
        contract_path = Path("/path/to/contract.py")
        step_name = "test_step"
        
        with patch('importlib.util.spec_from_file_location') as mock_spec_from_file:
            mock_spec = Mock()
            mock_loader = Mock()
            mock_loader.exec_module.side_effect = Exception("Execution error")
            mock_spec.loader = mock_loader
            mock_spec_from_file.return_value = mock_spec
            
            with patch('importlib.util.module_from_spec') as mock_module_from_spec:
                mock_module_from_spec.return_value = Mock()
                
                result = contract_discovery._load_contract_from_file(contract_path, step_name)
                
                assert result is None
    
    def test_discover_contract_classes_empty_directories(self, contract_discovery):
        """Test contract class discovery with empty directories."""
        result = contract_discovery.discover_contract_classes()
        
        # Should return empty dict for empty directories
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_discover_contract_classes_core_only(self, temp_workspace, contract_discovery):
        """Test contract class discovery from core directory only."""
        workspace_root, core_contract_dir, _ = temp_workspace
        
        with patch.object(contract_discovery, '_scan_contract_directory') as mock_scan:
            mock_scan.return_value = {"TestContract": Mock}
            
            result = contract_discovery.discover_contract_classes()
            
            assert mock_scan.called
            assert "TestContract" in result
    
    def test_discover_contract_classes_with_workspace(self, temp_workspace, contract_discovery):
        """Test contract class discovery from both core and workspace directories."""
        workspace_root, core_contract_dir, workspace_contract_dir = temp_workspace
        
        with patch.object(contract_discovery, '_scan_contract_directory') as mock_scan:
            mock_scan.return_value = {"WorkspaceContract": Mock}
            
            result = contract_discovery.discover_contract_classes("test_project")
            
            assert mock_scan.called
            assert "WorkspaceContract" in result
    
    def test_error_handling_in_load_contract_class(self, contract_discovery):
        """Test error handling in load_contract_class method."""
        step_name = "error_step"
        
        with patch.object(contract_discovery, '_try_direct_import') as mock_direct:
            mock_direct.side_effect = Exception("Unexpected error")
            
            result = contract_discovery.load_contract_class(step_name)
            
            # Should handle unexpected errors gracefully
            assert result is None
    
    def test_logging_behavior(self, contract_discovery):
        """Test that appropriate log messages are generated."""
        step_name = "test_step"
        mock_contract = Mock()
        
        with patch.object(contract_discovery.logger, 'debug') as mock_debug:
            with patch.object(contract_discovery.logger, 'warning') as mock_warning:
                with patch.object(contract_discovery.logger, 'error') as mock_error:
                    
                    # Test successful loading with debug logging
                    with patch.object(contract_discovery, '_try_direct_import') as mock_direct:
                        mock_direct.return_value = mock_contract
                        
                        result = contract_discovery.load_contract_class(step_name)
                        
                        assert result == mock_contract
                        mock_debug.assert_called()
                    
                    # Test not found with warning logging
                    with patch.object(contract_discovery, '_try_direct_import') as mock_direct:
                        mock_direct.return_value = None
                        
                        result = contract_discovery.load_contract_class(step_name)
                        
                        assert result is None
                        mock_warning.assert_called()
                    
                    # Test error with error logging
                    with patch.object(contract_discovery, '_try_direct_import') as mock_direct:
                        mock_direct.side_effect = Exception("Test error")
                        
                        result = contract_discovery.load_contract_class(step_name)
                        
                        assert result is None
                        mock_error.assert_called()


class TestContractAutoDiscoveryIntegration:
    """Integration tests for ContractAutoDiscovery based on real debugging experience."""
    
    def test_xgboost_training_contract_discovery(self):
        """Test discovery of actual XGBoost training contract - based on our debugging."""
        # This test simulates the exact scenario we debugged
        package_root = Path("cursus")
        discovery = ContractAutoDiscovery(package_root, None)
        
        # Mock the successful import that we achieved after fixing the bug
        mock_module = Mock()
        mock_contract = Mock()
        mock_contract.entry_point = "xgboost_training.py"
        mock_contract.expected_input_paths = {
            "input_path": "/opt/ml/input/data",
            "hyperparameters_s3_uri": "/opt/ml/input/data/config/hyperparameters.json"
        }
        mock_contract.expected_output_paths = {
            "model_output": "/opt/ml/model",
            "evaluation_output": "/opt/ml/output/data"
        }
        
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = mock_module
            
            with patch.object(discovery, '_discover_contract_objects_in_module') as mock_discover:
                mock_discover.return_value = [("XGBOOST_TRAIN_CONTRACT", mock_contract)]
                
                # Test the exact case that was failing before our fix
                result = discovery.load_contract_class("XGBoostTraining")
                
                assert result == mock_contract
                assert result.entry_point == "xgboost_training.py"
                assert "input_path" in result.expected_input_paths
                assert "model_output" in result.expected_output_paths
                
                # Verify correct import path (the bug we fixed)
                # The implementation first tries the original step name, then snake_case
                # Since our mock succeeds on the first call, only the first call is made
                # In reality, the first call would fail and it would try snake_case
                mock_import.assert_called_with(
                    "..steps.contracts.XGBoostTraining_contract",
                    package="cursus.step_catalog"
                )
    
    def test_xgboost_model_eval_contract_discovery(self):
        """Test discovery of XGBoost model evaluation contract."""
        package_root = Path("cursus")
        discovery = ContractAutoDiscovery(package_root, None)
        
        mock_module = Mock()
        mock_contract = Mock()
        mock_contract.entry_point = "xgboost_model_evaluation.py"
        mock_contract.expected_input_paths = {
            "model_input": "/opt/ml/input/model",
            "processed_data": "/opt/ml/input/data"
        }
        mock_contract.expected_output_paths = {
            "eval_output": "/opt/ml/output/evaluation",
            "metrics_output": "/opt/ml/output/metrics"
        }
        
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = mock_module
            
            with patch.object(discovery, '_discover_contract_objects_in_module') as mock_discover:
                mock_discover.return_value = [("XGBOOST_MODEL_EVAL_CONTRACT", mock_contract)]
                
                # Test both PascalCase and snake_case variants
                for step_name in ["XGBoostModelEval", "xgboost_model_eval"]:
                    result = discovery.load_contract_class(step_name)
                    
                    assert result == mock_contract
                    assert result.entry_point == "xgboost_model_evaluation.py"
    
    def test_contract_discovery_with_naming_variations(self):
        """Test contract discovery with various naming patterns we encountered."""
        package_root = Path("cursus")
        discovery = ContractAutoDiscovery(package_root, None)
        
        # Test cases based on our debugging experience
        naming_test_cases = [
            ("XGBoostTraining", "xgboost_training_contract", "XGBOOST_TRAIN_CONTRACT"),
            ("DataProcessing", "data_processing_contract", "DATA_PROCESSING_CONTRACT"),
            ("ModelEvaluation", "model_evaluation_contract", "MODEL_EVAL_CONTRACT"),
        ]
        
        for step_name, expected_module, contract_name in naming_test_cases:
            mock_module = Mock()
            mock_contract = Mock()
            mock_contract.entry_point = f"{expected_module.replace('_contract', '')}.py"
            
            with patch('importlib.import_module') as mock_import:
                mock_import.return_value = mock_module
                
                with patch.object(discovery, '_discover_contract_objects_in_module') as mock_discover:
                    mock_discover.return_value = [(contract_name, mock_contract)]
                    
                    result = discovery.load_contract_class(step_name)
                    
                    assert result == mock_contract
                    
                    # Verify the import path used (first attempt with original name)
                    # The implementation tries original name first, then snake_case
                    expected_import_path = f"..steps.contracts.{step_name}_contract"
                    mock_import.assert_called_with(
                        expected_import_path,
                        package="cursus.step_catalog"
                    )
    
    def test_relative_import_path_correctness(self):
        """Test that relative import paths are correct - the critical bug we fixed."""
        package_root = Path("cursus")
        discovery = ContractAutoDiscovery(package_root, None)
        
        step_name = "test_step"
        
        with patch('importlib.import_module') as mock_import:
            # Simulate the "beyond top-level package" error we encountered
            mock_import.side_effect = ImportError("attempted relative import beyond top-level package")
            
            result = discovery._try_direct_import(step_name)
            
            # Should handle the error gracefully
            assert result is None
            
            # Most importantly, should use correct relative path (2 dots, not 3)
            mock_import.assert_called_once_with(
                "..steps.contracts.test_step_contract",  # NOT "...steps.contracts.test_step_contract"
                package="cursus.step_catalog"
            )
    
    def test_automatic_contract_object_detection_realistic(self):
        """Test automatic contract object detection with realistic module content."""
        package_root = Path("cursus")
        discovery = ContractAutoDiscovery(package_root, None)
        
        # Create a realistic contract object (like XGBOOST_TRAIN_CONTRACT)
        class MockContract:
            def __init__(self):
                self.entry_point = "xgboost_training.py"
                self.expected_input_paths = {"input_path": "/opt/ml/input/data"}
                self.expected_output_paths = {"model_output": "/opt/ml/model"}
        
        mock_training_contract = MockContract()
        
        # Create a realistic module with various objects
        class MockModule:
            def __init__(self):
                self.__name__ = "cursus.steps.contracts.xgboost_training_contract"
                self.__file__ = "/path/to/xgboost_training_contract.py"
                self.TrainingScriptContract = type("TrainingScriptContract", (), {})  # Imported class (should be ignored)
                self.XGBOOST_TRAIN_CONTRACT = mock_training_contract  # Contract object (should be found)
                self.some_helper_function = lambda: None  # Function (should be ignored)
                self._private_var = "private"  # Private variable (should be ignored)
        
        mock_module = MockModule()
        
        result = discovery._discover_contract_objects_in_module(mock_module)
        
        # Should find exactly 1 contract object
        assert len(result) == 1
        contract_names = [name for name, obj in result]
        assert "XGBOOST_TRAIN_CONTRACT" in contract_names
    
    def test_complete_workflow_simulation(self):
        """Test complete contract discovery workflow simulation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            package_root = workspace_root / "cursus"
            discovery = ContractAutoDiscovery(package_root, [])
            
            # Create realistic directory structure
            core_contract_dir = package_root / "steps" / "contracts"
            core_contract_dir.mkdir(parents=True)
            
            # Create realistic contract file
            contract_file = core_contract_dir / "processing_step_contract.py"
            contract_file.write_text("""
from cursus.core.base.contract_base import ScriptContract

PROCESSING_STEP_CONTRACT = ScriptContract(
    entry_point="processing_step.py",
    expected_input_paths={
        "input_data": "/opt/ml/input/data",
        "config": "/opt/ml/input/config"
    },
    expected_output_paths={
        "processed_data": "/opt/ml/output/data",
        "metrics": "/opt/ml/output/metrics"
    }
)
""")
            
            # Mock the import to avoid actual module loading
            with patch('importlib.import_module') as mock_import:
                mock_module = Mock()
                mock_contract = Mock()
                mock_contract.entry_point = "processing_step.py"
                mock_contract.expected_input_paths = {
                    "input_data": "/opt/ml/input/data",
                    "config": "/opt/ml/input/config"
                }
                mock_contract.expected_output_paths = {
                    "processed_data": "/opt/ml/output/data",
                    "metrics": "/opt/ml/output/metrics"
                }
                mock_module.PROCESSING_STEP_CONTRACT = mock_contract
                mock_import.return_value = mock_module
                
                # Test the complete workflow
                result = discovery.load_contract_class("ProcessingStep")
                
                # Should successfully discover the contract
                assert result == mock_contract
                assert result.entry_point == "processing_step.py"
                assert "input_data" in result.expected_input_paths
                assert "processed_data" in result.expected_output_paths
    
    def test_edge_cases_and_error_scenarios(self):
        """Test various edge cases and error scenarios we encountered during debugging."""
        package_root = Path("cursus")
        discovery = ContractAutoDiscovery(package_root, None)
        
        # Test with empty step name
        result = discovery.load_contract_class("")
        assert result is None
        
        # Test with None step name (should handle gracefully)
        with patch.object(discovery, '_try_direct_import') as mock_direct:
            mock_direct.side_effect = TypeError("Expected string")
            result = discovery.load_contract_class(None)
            assert result is None
        
        # Test with very long step name
        long_name = "A" * 1000
        result = discovery.load_contract_class(long_name)
        assert result is None
        
        # Test with special characters in step name
        special_name = "Step@#$%^&*()"
        result = discovery.load_contract_class(special_name)
        assert result is None
    
    def test_performance_considerations(self):
        """Test performance-related aspects of contract discovery."""
        package_root = Path("cursus")
        discovery = ContractAutoDiscovery(package_root, None)
        
        # Test that multiple calls don't cause issues
        step_name = "TestStep"
        
        with patch.object(discovery, '_try_direct_import') as mock_direct:
            mock_direct.return_value = None
            
            # Multiple calls should all work
            for _ in range(5):
                result = discovery.load_contract_class(step_name)
                assert result is None
            
            # Should have been called 10 times (5 calls × 2 attempts each: original + snake_case)
            assert mock_direct.call_count == 10
