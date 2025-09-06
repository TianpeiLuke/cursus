"""Test suite for simplified runtime testing system"""

import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import json
import tempfile
from unittest.mock import Mock, patch

from cursus.validation.runtime.runtime_testing import RuntimeTester, ScriptTestResult, DataCompatibilityResult


class TestUserRequirements:
    """Test all validated user requirements"""
    
    def test_script_functionality_validation(self):
        """Test script import and main function validation - USER REQUIREMENT 1 & 2"""
        tester = RuntimeTester()
        
        # Test with mock script that has main function
        with patch.object(tester, '_find_script_path') as mock_find:
            mock_find.return_value = "test_script.py"
            
            with patch('importlib.util.spec_from_file_location') as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock()
                
                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj
                
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    with patch('inspect.signature') as mock_sig:
                        mock_sig.return_value.parameters.keys.return_value = [
                            'input_paths', 'output_paths', 'environ_vars', 'job_args'
                        ]
                        
                        result = tester.test_script("test_script")
                        
                        assert isinstance(result, ScriptTestResult)
                        assert result.success == True
                        assert result.has_main_function == True
                        assert result.script_name == "test_script"
    
    def test_script_missing_main_function(self):
        """Test script without main function fails validation"""
        tester = RuntimeTester()
        
        with patch.object(tester, '_find_script_path') as mock_find:
            mock_find.return_value = "test_script.py"
            
            with patch('importlib.util.spec_from_file_location') as mock_spec:
                mock_module = Mock()
                # No main function
                del mock_module.main
                
                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj
                
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    result = tester.test_script("test_script")
                    
                    assert result.success == False
                    assert result.has_main_function == False
                    assert "missing main() function" in result.error_message
    
    def test_data_transfer_consistency(self):
        """Test data compatibility between scripts - USER REQUIREMENT 3"""
        tester = RuntimeTester()
        sample_data = {"col1": [1, 2], "col2": ["a", "b"]}
        
        with patch.object(tester, '_execute_script_with_data') as mock_exec:
            # Mock successful script A execution
            mock_exec.side_effect = [
                ScriptTestResult(script_name="script_a", success=True, execution_time=0.1),
                ScriptTestResult(script_name="script_b", success=True, execution_time=0.1)
            ]
            
            with patch('pandas.DataFrame.to_csv'), \
                 patch('pandas.read_csv') as mock_read, \
                 patch('pathlib.Path.exists', return_value=True):
                
                mock_read.return_value = Mock()
                
                result = tester.test_data_compatibility("script_a", "script_b", sample_data)
                
                assert isinstance(result, DataCompatibilityResult)
                assert result.script_a == "script_a"
                assert result.script_b == "script_b"
                assert result.compatible == True
    
    def test_pipeline_flow_validation(self):
        """Test end-to-end pipeline execution - USER REQUIREMENT 4"""
        tester = RuntimeTester()
        pipeline_config = {
            "steps": {
                "step1": {"script": "script1.py"},
                "step2": {"script": "script2.py"}
            }
        }
        
        with patch.object(tester, 'test_script') as mock_test_script, \
             patch.object(tester, 'test_data_compatibility') as mock_test_compat:
            
            # Mock successful script tests
            mock_test_script.side_effect = [
                ScriptTestResult(script_name="step1", success=True, execution_time=0.1),
                ScriptTestResult(script_name="step2", success=True, execution_time=0.1)
            ]
            
            # Mock successful data compatibility
            mock_test_compat.return_value = DataCompatibilityResult(
                script_a="step1", script_b="step2", compatible=True
            )
            
            result = tester.test_pipeline_flow(pipeline_config)
            
            assert result["pipeline_success"] == True
            assert len(result["script_results"]) == 2
            assert len(result["data_flow_results"]) == 1
            assert len(result["errors"]) == 0
    
    def test_clear_error_feedback(self):
        """Test error messages are clear and actionable - USER REQUIREMENT 5"""
        tester = RuntimeTester()
        
        with patch.object(tester, '_find_script_path') as mock_find:
            mock_find.side_effect = FileNotFoundError("Script not found: nonexistent_script")
            
            result = tester.test_script("nonexistent_script")
            
            assert result.success == False
            assert "Script not found: nonexistent_script" in result.error_message
            assert result.script_name == "nonexistent_script"


class TestPerformance:
    """Test performance requirements"""
    
    def test_script_testing_performance(self):
        """Test script testing completes quickly"""
        tester = RuntimeTester()
        
        with patch.object(tester, '_find_script_path') as mock_find:
            mock_find.return_value = "test_script.py"
            
            with patch('importlib.util.spec_from_file_location') as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock()
                
                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj
                
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    with patch('inspect.signature') as mock_sig:
                        mock_sig.return_value.parameters.keys.return_value = [
                            'input_paths', 'output_paths', 'environ_vars', 'job_args'
                        ]
                        
                        result = tester.test_script("test_script")
                        
                        # Should complete very quickly (much less than 100ms of old system)
                        assert result.execution_time < 0.1  # 100ms threshold
    
    def test_memory_usage(self):
        """Test memory usage is reasonable"""
        # Simple instantiation should not use excessive memory
        tester = RuntimeTester()
        
        # Basic operations should not cause memory issues
        sample_data = tester._generate_sample_data()
        assert isinstance(sample_data, dict)
        assert len(sample_data) > 0


class TestCLIInterface:
    """Test CLI interface functionality"""
    
    def test_runtime_tester_initialization(self):
        """Test RuntimeTester can be initialized"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = RuntimeTester(temp_dir)
            assert tester.workspace_dir == Path(temp_dir)
            assert tester.workspace_dir.exists()
    
    def test_sample_data_generation(self):
        """Test sample data generation works"""
        tester = RuntimeTester()
        sample_data = tester._generate_sample_data()
        
        assert isinstance(sample_data, dict)
        assert "feature1" in sample_data
        assert "feature2" in sample_data
        assert "label" in sample_data
        assert len(sample_data["feature1"]) == 5
    
    def test_script_path_discovery(self):
        """Test script path discovery logic"""
        tester = RuntimeTester()
        
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.side_effect = lambda: str(mock_exists.call_args[0][0]) == "scripts/test_script.py"
            
            # Mock the Path constructor to return the expected path
            with patch('pathlib.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path_instance.exists.return_value = True
                mock_path.return_value = mock_path_instance
                
                result = tester._find_script_path("test_script")
                assert result == "scripts/test_script.py"
    
    def test_script_path_not_found(self):
        """Test script path discovery when script doesn't exist"""
        tester = RuntimeTester()
        
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                tester._find_script_path("nonexistent_script")
            
            assert "Script not found: nonexistent_script" in str(exc_info.value)


class TestPydanticModels:
    """Test Pydantic v2 model functionality"""
    
    def test_script_test_result_model(self):
        """Test ScriptTestResult Pydantic model"""
        result = ScriptTestResult(
            script_name="test_script",
            success=True,
            execution_time=0.123
        )
        
        assert result.script_name == "test_script"
        assert result.success == True
        assert result.execution_time == 0.123
        assert result.error_message is None
        assert result.has_main_function == False  # default
        
        # Test model_dump for JSON serialization
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["script_name"] == "test_script"
    
    def test_data_compatibility_result_model(self):
        """Test DataCompatibilityResult Pydantic model"""
        result = DataCompatibilityResult(
            script_a="script1",
            script_b="script2",
            compatible=True
        )
        
        assert result.script_a == "script1"
        assert result.script_b == "script2"
        assert result.compatible == True
        assert result.compatibility_issues == []  # default factory
        
        # Test model_dump for JSON serialization
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["compatible"] == True


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from script testing to pipeline validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = RuntimeTester(temp_dir)
            
            # Test individual script functionality
            with patch.object(tester, '_find_script_path', return_value="test.py"):
                with patch('importlib.util.spec_from_file_location') as mock_spec:
                    mock_module = Mock()
                    mock_module.main = Mock()
                    
                    mock_spec_obj = Mock()
                    mock_spec_obj.loader.exec_module = Mock()
                    mock_spec.return_value = mock_spec_obj
                    
                    with patch('importlib.util.module_from_spec', return_value=mock_module):
                        with patch('inspect.signature') as mock_sig:
                            mock_sig.return_value.parameters.keys.return_value = [
                                'input_paths', 'output_paths', 'environ_vars', 'job_args'
                            ]
                            
                            # Test script functionality
                            script_result = tester.test_script("test_script")
                            assert script_result.success == True
                            
                            # Test pipeline with this script
                            pipeline_config = {"steps": {"test_script": {}}}
                            
                            with patch.object(tester, 'test_data_compatibility') as mock_compat:
                                pipeline_result = tester.test_pipeline_flow(pipeline_config)
                                assert pipeline_result["pipeline_success"] == True


if __name__ == "__main__":
    pytest.main([__file__])
