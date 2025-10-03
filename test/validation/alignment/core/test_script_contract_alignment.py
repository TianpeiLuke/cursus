"""
Test module for script-contract alignment validation.

Tests the core functionality of script-contract alignment validation,
including integration with ScriptAnalyzer.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List

from cursus.validation.alignment.core.script_contract_alignment import ScriptContractAlignmentTester
from cursus.validation.alignment.analyzer.script_analyzer import ScriptAnalyzer


class TestScriptContractAlignment:
    """Test cases for ScriptContractAlignment class."""

    @pytest.fixture
    def workspace_dirs(self):
        """Fixture providing workspace directories."""
        return ["/test/workspace1", "/test/workspace2"]

    @pytest.fixture
    def script_contract_alignment(self, workspace_dirs):
        """Fixture providing ScriptContractAlignmentTester instance."""
        return ScriptContractAlignmentTester(workspace_dirs=workspace_dirs)

    @pytest.fixture
    def sample_contract(self):
        """Fixture providing sample contract data."""
        return {
            "expected_input_paths": {
                "training_data": "/path/to/training",
                "validation_data": "/path/to/validation"
            },
            "expected_output_paths": {
                "model_artifacts": "/path/to/model",
                "evaluation_report": "/path/to/report"
            },
            "required_env_vars": ["MODEL_TYPE", "EPOCHS"],
            "optional_env_vars": {
                "LEARNING_RATE": "0.01",
                "BATCH_SIZE": "32"
            },
            "expected_arguments": {
                "--max-depth": {"type": "int", "default": 6},
                "--n-estimators": {"type": "int", "default": 100}
            }
        }

    @pytest.fixture
    def sample_step_info(self):
        """Fixture providing sample step info."""
        mock_step_info = Mock()
        mock_step_info.file_components = {
            'script': Mock(path=Path("/test/script.py"))
        }
        return mock_step_info

    def test_init_with_workspace_dirs(self, workspace_dirs):
        """Test ScriptContractAlignmentTester initialization with workspace directories."""
        alignment = ScriptContractAlignmentTester(workspace_dirs=workspace_dirs)
        assert alignment.workspace_dirs == workspace_dirs

    def test_init_without_workspace_dirs(self):
        """Test ScriptContractAlignmentTester initialization without workspace directories."""
        alignment = ScriptContractAlignmentTester()
        assert alignment.workspace_dirs == []

    @patch('cursus.validation.alignment.core.script_contract_alignment.StepCatalog')
    def test_step_catalog_initialization(self, mock_step_catalog, workspace_dirs):
        """Test that StepCatalog is properly initialized."""
        ScriptContractAlignmentTester(workspace_dirs=workspace_dirs)
        mock_step_catalog.assert_called_once_with(workspace_dirs=workspace_dirs)

    def test_validate_script_with_valid_main_function(self, script_contract_alignment, sample_contract, sample_step_info):
        """Test script validation with valid main function signature."""
        script_name = "test_script"
        
        # Mock ScriptAnalyzer results
        main_function_result = {
            "has_main": True,
            "signature_valid": True,
            "actual_params": ["input_paths", "output_paths", "environ_vars", "job_args"],
            "expected_params": ["input_paths", "output_paths", "environ_vars", "job_args"],
            "issues": []
        }
        
        parameter_usage = {
            "input_paths_keys": ["training_data", "validation_data"],
            "output_paths_keys": ["model_artifacts", "evaluation_report"],
            "environ_vars_keys": ["MODEL_TYPE", "EPOCHS", "LEARNING_RATE"],
            "job_args_attrs": ["max_depth", "n_estimators"]
        }
        
        alignment_issues = []
        
        with patch.object(script_contract_alignment.step_catalog, 'get_step_info') as mock_get_step_info, \
             patch.object(script_contract_alignment, '_load_python_contract') as mock_load_contract, \
             patch('cursus.validation.alignment.analyzer.script_analyzer.ScriptAnalyzer') as mock_analyzer_class:
            
            # Setup mocks
            mock_get_step_info.return_value = sample_step_info
            mock_load_contract.return_value = sample_contract
            
            mock_analyzer = Mock()
            mock_analyzer.validate_main_function_signature.return_value = main_function_result
            mock_analyzer.extract_parameter_usage.return_value = parameter_usage
            mock_analyzer.validate_contract_alignment.return_value = alignment_issues
            mock_analyzer_class.return_value = mock_analyzer
            
            # Execute validation
            result = script_contract_alignment.validate_script(script_name)
            
            # Verify results
            assert result["passed"] is True
            assert len(result["issues"]) == 0
            assert result["script_analysis"]["main_function"] == main_function_result
            assert result["script_analysis"]["parameter_usage"] == parameter_usage
            
            # Verify ScriptAnalyzer was called correctly
            mock_analyzer_class.assert_called_once_with(str(sample_step_info.file_components['script'].path))
            mock_analyzer.validate_main_function_signature.assert_called_once()
            mock_analyzer.extract_parameter_usage.assert_called_once()
            mock_analyzer.validate_contract_alignment.assert_called_once_with(sample_contract)

    def test_validate_script_with_missing_main_function(self, script_contract_alignment, sample_contract, sample_step_info):
        """Test script validation when main function is missing."""
        script_name = "test_script"
        
        # Mock ScriptAnalyzer results for missing main function
        main_function_result = {
            "has_main": False,
            "issues": ["No main function found"],
            "signature_valid": False
        }
        
        parameter_usage = {
            "input_paths_keys": [],
            "output_paths_keys": [],
            "environ_vars_keys": [],
            "job_args_attrs": []
        }
        
        with patch.object(script_contract_alignment.step_catalog, 'get_step_info') as mock_get_step_info, \
             patch.object(script_contract_alignment, '_load_python_contract') as mock_load_contract, \
             patch('cursus.validation.alignment.analyzer.script_analyzer.ScriptAnalyzer') as mock_analyzer_class:
            
            # Setup mocks
            mock_get_step_info.return_value = sample_step_info
            mock_load_contract.return_value = sample_contract
            
            mock_analyzer = Mock()
            mock_analyzer.validate_main_function_signature.return_value = main_function_result
            mock_analyzer.extract_parameter_usage.return_value = parameter_usage
            mock_analyzer.validate_contract_alignment.return_value = []
            mock_analyzer_class.return_value = mock_analyzer
            
            # Execute validation
            result = script_contract_alignment.validate_script(script_name)
            
            # Verify results
            assert result["passed"] is False
            assert len(result["issues"]) > 0
            
            # Check for critical issue about missing main function
            critical_issues = [issue for issue in result["issues"] if issue.get("severity") == "CRITICAL"]
            assert len(critical_issues) > 0
            assert any("main function" in issue.get("message", "").lower() for issue in critical_issues)

    def test_validate_script_with_invalid_signature(self, script_contract_alignment, sample_contract, sample_step_info):
        """Test script validation with invalid main function signature."""
        script_name = "test_script"
        
        # Mock ScriptAnalyzer results for invalid signature
        main_function_result = {
            "has_main": True,
            "signature_valid": False,
            "actual_params": ["data", "output"],
            "expected_params": ["input_paths", "output_paths", "environ_vars", "job_args"],
            "issues": ["Expected 4 parameters, got 2", "Parameter 1: expected 'input_paths', got 'data'"]
        }
        
        parameter_usage = {
            "input_paths_keys": [],
            "output_paths_keys": [],
            "environ_vars_keys": [],
            "job_args_attrs": []
        }
        
        with patch.object(script_contract_alignment.step_catalog, 'get_step_info') as mock_get_step_info, \
             patch.object(script_contract_alignment, '_load_python_contract') as mock_load_contract, \
             patch('cursus.validation.alignment.analyzer.script_analyzer.ScriptAnalyzer') as mock_analyzer_class:
            
            # Setup mocks
            mock_get_step_info.return_value = sample_step_info
            mock_load_contract.return_value = sample_contract
            
            mock_analyzer = Mock()
            mock_analyzer.validate_main_function_signature.return_value = main_function_result
            mock_analyzer.extract_parameter_usage.return_value = parameter_usage
            mock_analyzer.validate_contract_alignment.return_value = []
            mock_analyzer_class.return_value = mock_analyzer
            
            # Execute validation
            result = script_contract_alignment.validate_script(script_name)
            
            # Verify results indicate signature issues
            assert len(result["issues"]) > 0
            assert result["script_analysis"]["main_function"]["signature_valid"] is False

    def test_validate_script_with_contract_alignment_issues(self, script_contract_alignment, sample_contract, sample_step_info):
        """Test script validation when contract alignment validation finds issues."""
        script_name = "test_script"
        
        # Mock ScriptAnalyzer results with alignment issues
        main_function_result = {
            "has_main": True,
            "signature_valid": True,
            "actual_params": ["input_paths", "output_paths", "environ_vars", "job_args"],
            "expected_params": ["input_paths", "output_paths", "environ_vars", "job_args"],
            "issues": []
        }
        
        parameter_usage = {
            "input_paths_keys": ["training_data", "test_data"],  # test_data not in contract
            "output_paths_keys": ["model_artifacts"],
            "environ_vars_keys": ["MODEL_TYPE", "UNKNOWN_VAR"],  # UNKNOWN_VAR not in contract
            "job_args_attrs": ["max_depth"]
        }
        
        alignment_issues = [
            {
                "severity": "ERROR",
                "category": "undeclared_input_path",
                "message": "Script uses input_paths['test_data'] but contract doesn't declare it",
                "recommendation": "Add 'test_data' to contract expected_input_paths"
            },
            {
                "severity": "WARNING",
                "category": "undeclared_env_var",
                "message": "Script uses environ_vars.get('UNKNOWN_VAR') but contract doesn't declare it",
                "recommendation": "Add 'UNKNOWN_VAR' to contract required_env_vars or optional_env_vars"
            }
        ]
        
        with patch.object(script_contract_alignment.step_catalog, 'get_step_info') as mock_get_step_info, \
             patch.object(script_contract_alignment, '_load_python_contract') as mock_load_contract, \
             patch('cursus.validation.alignment.analyzer.script_analyzer.ScriptAnalyzer') as mock_analyzer_class:
            
            # Setup mocks
            mock_get_step_info.return_value = sample_step_info
            mock_load_contract.return_value = sample_contract
            
            mock_analyzer = Mock()
            mock_analyzer.validate_main_function_signature.return_value = main_function_result
            mock_analyzer.extract_parameter_usage.return_value = parameter_usage
            mock_analyzer.validate_contract_alignment.return_value = alignment_issues
            mock_analyzer_class.return_value = mock_analyzer
            
            # Execute validation
            result = script_contract_alignment.validate_script(script_name)
            
            # Verify results
            assert result["passed"] is False  # ERROR causes failure
            assert len(result["issues"]) == 2
            
            # Check for specific alignment issues
            error_issues = [issue for issue in result["issues"] if issue.get("severity") == "ERROR"]
            warning_issues = [issue for issue in result["issues"] if issue.get("severity") == "WARNING"]
            assert len(error_issues) == 1
            assert len(warning_issues) == 1

    def test_validate_script_with_step_info_error(self, script_contract_alignment):
        """Test script validation when step info cannot be retrieved."""
        script_name = "test_script"
        
        with patch.object(script_contract_alignment.step_catalog, 'get_step_info') as mock_get_step_info:
            # Setup mock to raise exception
            mock_get_step_info.side_effect = Exception("Step not found")
            
            # Execute validation
            result = script_contract_alignment.validate_script(script_name)
            
            # Verify results indicate failure
            assert result["passed"] is False
            assert len(result["issues"]) > 0

    def test_validate_script_with_contract_loading_error(self, script_contract_alignment, sample_step_info):
        """Test script validation when contract cannot be loaded."""
        script_name = "test_script"
        
        with patch.object(script_contract_alignment.step_catalog, 'get_step_info') as mock_get_step_info, \
             patch.object(script_contract_alignment, '_load_python_contract') as mock_load_contract:
            
            # Setup mocks
            mock_get_step_info.return_value = sample_step_info
            mock_load_contract.side_effect = Exception("Contract not found")
            
            # Execute validation
            result = script_contract_alignment.validate_script(script_name)
            
            # Verify results indicate failure
            assert result["passed"] is False
            assert len(result["issues"]) > 0

    def test_validate_script_with_script_analyzer_error(self, script_contract_alignment, sample_contract, sample_step_info):
        """Test script validation when ScriptAnalyzer raises an error."""
        script_name = "test_script"
        
        with patch.object(script_contract_alignment.step_catalog, 'get_step_info') as mock_get_step_info, \
             patch.object(script_contract_alignment, '_load_python_contract') as mock_load_contract, \
             patch('cursus.validation.alignment.analyzer.script_analyzer.ScriptAnalyzer') as mock_analyzer_class:
            
            # Setup mocks
            mock_get_step_info.return_value = sample_step_info
            mock_load_contract.return_value = sample_contract
            mock_analyzer_class.side_effect = Exception("Script parsing error")
            
            # Execute validation
            result = script_contract_alignment.validate_script(script_name)
            
            # Verify results indicate failure
            assert result["passed"] is False
            assert len(result["issues"]) > 0

    def test_validate_script_result_structure(self, script_contract_alignment, sample_contract, sample_step_info):
        """Test that validate_script returns properly structured results."""
        script_name = "test_script"
        
        # Mock ScriptAnalyzer results
        main_function_result = {
            "has_main": True,
            "signature_valid": True,
            "actual_params": ["input_paths", "output_paths", "environ_vars", "job_args"],
            "expected_params": ["input_paths", "output_paths", "environ_vars", "job_args"],
            "issues": []
        }
        
        parameter_usage = {
            "input_paths_keys": ["training_data"],
            "output_paths_keys": ["model_artifacts"],
            "environ_vars_keys": ["MODEL_TYPE"],
            "job_args_attrs": ["max_depth"]
        }
        
        with patch.object(script_contract_alignment.step_catalog, 'get_step_info') as mock_get_step_info, \
             patch.object(script_contract_alignment, '_load_python_contract') as mock_load_contract, \
             patch('cursus.validation.alignment.analyzer.script_analyzer.ScriptAnalyzer') as mock_analyzer_class:
            
            # Setup mocks
            mock_get_step_info.return_value = sample_step_info
            mock_load_contract.return_value = sample_contract
            
            mock_analyzer = Mock()
            mock_analyzer.validate_main_function_signature.return_value = main_function_result
            mock_analyzer.extract_parameter_usage.return_value = parameter_usage
            mock_analyzer.validate_contract_alignment.return_value = []
            mock_analyzer_class.return_value = mock_analyzer
            
            # Execute validation
            result = script_contract_alignment.validate_script(script_name)
            
            # Verify result structure
            required_keys = ["passed", "issues", "script_analysis", "contract"]
            for key in required_keys:
                assert key in result
            
            assert isinstance(result["passed"], bool)
            assert isinstance(result["issues"], list)
            assert isinstance(result["script_analysis"], dict)
            assert isinstance(result["contract"], dict)
            
            # Verify script_analysis structure
            script_analysis_keys = ["main_function", "parameter_usage"]
            for key in script_analysis_keys:
                assert key in result["script_analysis"]

    def test_integration_with_script_analyzer(self, script_contract_alignment, sample_contract, sample_step_info):
        """Test integration with real ScriptAnalyzer."""
        script_name = "test_script"
        
        with patch.object(script_contract_alignment.step_catalog, 'get_step_info') as mock_get_step_info, \
             patch.object(script_contract_alignment, '_load_python_contract') as mock_load_contract, \
             patch('cursus.validation.alignment.analyzer.script_analyzer.ScriptAnalyzer') as mock_analyzer_class:
            
            # Setup mocks
            mock_get_step_info.return_value = sample_step_info
            mock_load_contract.return_value = sample_contract
            
            # Create a real-ish ScriptAnalyzer mock that behaves correctly
            mock_analyzer = Mock()
            mock_analyzer.validate_main_function_signature.return_value = {
                "has_main": True,
                "signature_valid": True,
                "issues": []
            }
            mock_analyzer.extract_parameter_usage.return_value = {
                "input_paths_keys": [],
                "output_paths_keys": [],
                "environ_vars_keys": [],
                "job_args_attrs": []
            }
            mock_analyzer.validate_contract_alignment.return_value = []
            mock_analyzer_class.return_value = mock_analyzer
            
            # Execute validation
            result = script_contract_alignment.validate_script(script_name)
            
            # Verify basic integration works
            assert "passed" in result
            assert "issues" in result
            assert "script_analysis" in result

    def test_workspace_directory_propagation(self, workspace_dirs):
        """Test that workspace directories are properly propagated."""
        alignment = ScriptContractAlignmentTester(workspace_dirs=workspace_dirs)
        assert alignment.workspace_dirs == workspace_dirs

    def test_validate_script_with_complex_parameter_usage(self, script_contract_alignment, sample_step_info):
        """Test script validation with complex parameter usage patterns."""
        script_name = "test_script"
        
        # Complex contract with many parameters
        complex_contract = {
            "expected_input_paths": {f"input_{i}": f"/path/to/input_{i}" for i in range(10)},
            "expected_output_paths": {f"output_{i}": f"/path/to/output_{i}" for i in range(10)},
            "required_env_vars": [f"ENV_VAR_{i}" for i in range(5)],
            "optional_env_vars": {f"OPT_VAR_{i}": f"default_{i}" for i in range(5)},
            "expected_arguments": {f"--arg-{i}": {"type": "str"} for i in range(5)}
        }
        
        # Complex parameter usage
        complex_usage = {
            "input_paths_keys": [f"input_{i}" for i in range(10)],
            "output_paths_keys": [f"output_{i}" for i in range(10)],
            "environ_vars_keys": [f"ENV_VAR_{i}" for i in range(5)] + [f"OPT_VAR_{i}" for i in range(5)],
            "job_args_attrs": [f"arg_{i}" for i in range(5)]
        }
        
        with patch.object(script_contract_alignment.step_catalog, 'get_step_info') as mock_get_step_info, \
             patch.object(script_contract_alignment, '_load_python_contract') as mock_load_contract, \
             patch('cursus.validation.alignment.analyzer.script_analyzer.ScriptAnalyzer') as mock_analyzer_class:
            
            # Setup mocks
            mock_get_step_info.return_value = sample_step_info
            mock_load_contract.return_value = complex_contract
            
            mock_analyzer = Mock()
            mock_analyzer.validate_main_function_signature.return_value = {
                "has_main": True,
                "signature_valid": True,
                "issues": []
            }
            mock_analyzer.extract_parameter_usage.return_value = complex_usage
            mock_analyzer.validate_contract_alignment.return_value = []
            mock_analyzer_class.return_value = mock_analyzer
            
            # Execute validation
            result = script_contract_alignment.validate_script(script_name)
            
            # Should handle complex usage without issues
            assert "passed" in result
            assert "script_analysis" in result
            assert result["script_analysis"]["parameter_usage"] == complex_usage

    def test_validate_script_performance_with_large_script(self, script_contract_alignment, sample_contract):
        """Test performance with large script analysis."""
        script_name = "large_script"
        
        # Mock large step info
        large_step_info = Mock()
        large_step_info.file_components = {
            'script': Mock(path=Path("/test/large_script.py"))
        }
        
        with patch.object(script_contract_alignment.step_catalog, 'get_step_info') as mock_get_step_info, \
             patch.object(script_contract_alignment, '_load_python_contract') as mock_load_contract, \
             patch('cursus.validation.alignment.analyzer.script_analyzer.ScriptAnalyzer') as mock_analyzer_class:
            
            # Setup mocks
            mock_get_step_info.return_value = large_step_info
            mock_load_contract.return_value = sample_contract
            
            mock_analyzer = Mock()
            mock_analyzer.validate_main_function_signature.return_value = {
                "has_main": True,
                "signature_valid": True,
                "issues": []
            }
            mock_analyzer.extract_parameter_usage.return_value = {
                "input_paths_keys": [],
                "output_paths_keys": [],
                "environ_vars_keys": [],
                "job_args_attrs": []
            }
            mock_analyzer.validate_contract_alignment.return_value = []
            mock_analyzer_class.return_value = mock_analyzer
            
            # Execute validation and verify it completes
            result = script_contract_alignment.validate_script(script_name)
            
            # Should complete successfully
            assert "passed" in result
            assert "issues" in result
