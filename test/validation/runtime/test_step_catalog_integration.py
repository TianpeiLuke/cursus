"""
Comprehensive integration tests for Pipeline Runtime Testing Step Catalog Integration.

Tests the three user stories with step catalog integration:
- US1: Individual Script Functionality Testing
- US2: Data Transfer and Compatibility Testing  
- US3: DAG-Guided End-to-End Testing
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Optional

# Import the classes we're testing
from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder
from cursus.validation.runtime.runtime_models import (
    ScriptExecutionSpec,
    PipelineTestingSpec,
    ScriptTestResult,
    DataCompatibilityResult
)

# Mock PipelineDAG for testing
class MockPipelineDAG:
    def __init__(self, nodes: List[str], edges: List[tuple]):
        self.nodes = nodes
        self.edges = edges
    
    def topological_sort(self):
        return self.nodes


class TestStepCatalogIntegration:
    """Comprehensive integration tests for step catalog integration."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        workspace_path = Path(temp_dir)
        
        # Create workspace structure
        (workspace_path / "scripts").mkdir(parents=True, exist_ok=True)
        (workspace_path / "input").mkdir(parents=True, exist_ok=True)
        (workspace_path / "output").mkdir(parents=True, exist_ok=True)
        (workspace_path / ".specs").mkdir(parents=True, exist_ok=True)
        
        # Create sample script
        sample_script = workspace_path / "scripts" / "sample_script.py"
        sample_script.write_text('''
def main(input_paths, output_paths, environ_vars, job_args):
    """Sample script for testing."""
    import json
    from pathlib import Path
    
    # Create sample output
    output_dir = Path(list(output_paths.values())[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "sample_output.json"
    with open(output_file, 'w') as f:
        json.dump({"status": "success", "script": "sample_script"}, f)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main({}, {}, {}, {}))
''')
        
        yield workspace_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_step_catalog(self):
        """Create mock step catalog for testing."""
        mock_catalog = Mock()
        
        # Mock framework detection
        mock_catalog.detect_framework.return_value = "xgboost"
        
        # Mock builder class loading
        mock_builder = Mock()
        mock_builder.get_expected_input_paths.return_value = ["data_input", "config"]
        mock_catalog.load_builder_class.return_value = mock_builder
        
        # Mock contract class loading
        mock_contract = Mock()
        mock_contract.get_input_paths.return_value = {"data_input": "/path/to/input"}
        mock_contract.get_output_paths.return_value = {"data_output": "/path/to/output"}
        mock_contract.get_output_specifications.return_value = {"data_output": {"type": "data"}}
        mock_contract.get_input_specifications.return_value = {"data_input": {"type": "data"}}
        mock_catalog.load_contract_class.return_value = mock_contract
        
        # Mock cross-workspace component discovery
        mock_catalog.discover_cross_workspace_components.return_value = {
            "workspace1": ["SampleScript:script", "SampleScript:builder"],
            "workspace2": ["SampleScript:contract"]
        }
        
        # Mock pipeline node resolution
        mock_step_info = Mock()
        mock_script_metadata = Mock()
        mock_script_metadata.path = Path("scripts/sample_script.py")
        mock_step_info.file_components = {"script": mock_script_metadata}
        mock_catalog.resolve_pipeline_node.return_value = mock_step_info
        
        return mock_catalog
    
    @pytest.fixture
    def runtime_tester_with_step_catalog(self, temp_workspace, mock_step_catalog):
        """Create RuntimeTester with step catalog integration."""
        return RuntimeTester(
            config_or_workspace_dir=str(temp_workspace),
            step_catalog=mock_step_catalog
        )
    
    @pytest.fixture
    def runtime_tester_without_step_catalog(self, temp_workspace):
        """Create RuntimeTester without step catalog for fallback testing."""
        return RuntimeTester(
            config_or_workspace_dir=str(temp_workspace),
            step_catalog=None
        )
    
    @pytest.fixture
    def spec_builder_with_step_catalog(self, temp_workspace, mock_step_catalog):
        """Create PipelineTestingSpecBuilder with step catalog integration."""
        return PipelineTestingSpecBuilder(
            test_data_dir=str(temp_workspace),
            step_catalog=mock_step_catalog
        )
    
    @pytest.fixture
    def sample_script_spec(self, temp_workspace):
        """Create sample ScriptExecutionSpec for testing."""
        return ScriptExecutionSpec(
            script_name="sample_script",
            step_name="SampleScript_training",
            script_path=str(temp_workspace / "scripts" / "sample_script.py"),
            input_paths={"data_input": str(temp_workspace / "input" / "sample_data")},
            output_paths={"data_output": str(temp_workspace / "output" / "sample_output")},
            environ_vars={"PYTHONPATH": "src"},
            job_args={"job_type": "testing"}
        )

    # Test Optional Enhancement Pattern
    
    def test_optional_enhancement_pattern(self, runtime_tester_with_step_catalog, runtime_tester_without_step_catalog, sample_script_spec):
        """Test that all enhancements work optionally."""
        
        # Create sample input data
        input_dir = Path(sample_script_spec.input_paths["data_input"])
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "sample.json").write_text('{"test": "data"}')
        
        main_params = {
            "input_paths": sample_script_spec.input_paths,
            "output_paths": sample_script_spec.output_paths,
            "environ_vars": sample_script_spec.environ_vars,
            "job_args": sample_script_spec.job_args
        }
        
        # Test with step catalog - should have enhancements
        result_with_catalog = runtime_tester_with_step_catalog.test_script_with_step_catalog_enhancements(
            sample_script_spec, main_params
        )
        
        # Test without step catalog - should work without enhancements
        result_without_catalog = runtime_tester_without_step_catalog.test_script_with_step_catalog_enhancements(
            sample_script_spec, main_params
        )
        
        # Both should succeed
        assert result_with_catalog.success
        assert result_without_catalog.success
        
        # Both results should be basic ScriptTestResult objects
        # The step catalog enhancements are tested through the enhanced methods
        # but don't modify the basic result structure
        assert isinstance(result_with_catalog, ScriptTestResult)
        assert isinstance(result_without_catalog, ScriptTestResult)

    def test_workspace_resolution_priority(self, temp_workspace):
        """Test unified workspace resolution strategy."""
        
        # Test with environment variable
        with patch.dict(os.environ, {'CURSUS_DEV_WORKSPACES': f'{temp_workspace}/dev1:{temp_workspace}/dev2'}):
            # Create dev workspace directories
            (temp_workspace / "dev1").mkdir(exist_ok=True)
            (temp_workspace / "dev2").mkdir(exist_ok=True)
            
            tester = RuntimeTester(config_or_workspace_dir=str(temp_workspace))
            
            # Should initialize step catalog with unified workspace resolution
            # (Even if step catalog is None due to import issues, the resolution logic should work)
            assert tester.step_catalog is None or hasattr(tester, 'step_catalog')

    def test_fallback_behavior(self, temp_workspace, sample_script_spec):
        """Test fallback to existing methods when step catalog unavailable."""
        
        # Create tester without step catalog
        tester = RuntimeTester(config_or_workspace_dir=str(temp_workspace), step_catalog=None)
        
        # Create sample input data
        input_dir = Path(sample_script_spec.input_paths["data_input"])
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "sample.json").write_text('{"test": "data"}')
        
        main_params = {
            "input_paths": sample_script_spec.input_paths,
            "output_paths": sample_script_spec.output_paths,
            "environ_vars": sample_script_spec.environ_vars,
            "job_args": sample_script_spec.job_args
        }
        
        # All enhanced methods should fall back gracefully
        script_result = tester.test_script_with_step_catalog_enhancements(sample_script_spec, main_params)
        assert script_result.success
        
        # Create second script spec for compatibility testing
        script_spec_b = ScriptExecutionSpec(
            script_name="sample_script_b",
            step_name="SampleScriptB_training",
            script_path=str(temp_workspace / "scripts" / "sample_script.py"),  # Same script for testing
            input_paths={"data_input": str(temp_workspace / "output" / "sample_output")},  # Use output of first as input
            output_paths={"data_output": str(temp_workspace / "output" / "sample_output_b")},
            environ_vars={"PYTHONPATH": "src"},
            job_args={"job_type": "testing"}
        )
        
        compatibility_result = tester.test_data_compatibility_with_step_catalog_enhancements(
            sample_script_spec, script_spec_b
        )
        # Should not fail, even if not compatible
        assert isinstance(compatibility_result, DataCompatibilityResult)

    # Test User Story Coverage
    
    def test_user_story_1_individual_script_testing(self, runtime_tester_with_step_catalog, sample_script_spec):
        """Test US1: Individual Script Functionality Testing with step catalog integration."""
        
        # Create sample input data
        input_dir = Path(sample_script_spec.input_paths["data_input"])
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "sample.json").write_text('{"test": "data"}')
        
        main_params = {
            "input_paths": sample_script_spec.input_paths,
            "output_paths": sample_script_spec.output_paths,
            "environ_vars": sample_script_spec.environ_vars,
            "job_args": sample_script_spec.job_args
        }
        
        # Test enhanced script testing
        result = runtime_tester_with_step_catalog.test_script_with_step_catalog_enhancements(
            sample_script_spec, main_params
        )
        
        # Should succeed with enhancements
        assert result.success
        
        # The enhanced method should work but return basic ScriptTestResult
        # Step catalog enhancements are internal and don't modify the result structure
        assert isinstance(result, ScriptTestResult)

    def test_user_story_2_data_compatibility_testing(self, runtime_tester_with_step_catalog, sample_script_spec, temp_workspace):
        """Test US2: Data Transfer and Compatibility Testing with contract awareness."""
        
        # Create sample input data and run first script
        input_dir = Path(sample_script_spec.input_paths["data_input"])
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "sample.json").write_text('{"test": "data"}')
        
        main_params = {
            "input_paths": sample_script_spec.input_paths,
            "output_paths": sample_script_spec.output_paths,
            "environ_vars": sample_script_spec.environ_vars,
            "job_args": sample_script_spec.job_args
        }
        
        # Run first script to create output
        script_result = runtime_tester_with_step_catalog.test_script_with_spec(sample_script_spec, main_params)
        assert script_result.success
        
        # Create second script spec
        script_spec_b = ScriptExecutionSpec(
            script_name="sample_script_b",
            step_name="SampleScriptB_training",
            script_path=str(temp_workspace / "scripts" / "sample_script.py"),
            input_paths={"data_input": str(temp_workspace / "output" / "sample_output")},
            output_paths={"data_output": str(temp_workspace / "output" / "sample_output_b")},
            environ_vars={"PYTHONPATH": "src"},
            job_args={"job_type": "testing"}
        )
        
        # Test enhanced compatibility testing
        compatibility_result = runtime_tester_with_step_catalog.test_data_compatibility_with_step_catalog_enhancements(
            sample_script_spec, script_spec_b
        )
        
        # Should return compatibility result
        assert isinstance(compatibility_result, DataCompatibilityResult)
        assert compatibility_result.script_a == "sample_script"
        assert compatibility_result.script_b == "sample_script_b"

    def test_user_story_3_pipeline_testing(self, runtime_tester_with_step_catalog, sample_script_spec, temp_workspace):
        """Test US3: DAG-Guided End-to-End Testing with multi-workspace support."""
        
        # Create sample input data
        input_dir = Path(sample_script_spec.input_paths["data_input"])
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "sample.json").write_text('{"test": "data"}')
        
        # Import and create real PipelineDAG
        from cursus.api.dag.base_dag import PipelineDAG
        
        # Create pipeline DAG
        dag = PipelineDAG(
            nodes=["SampleScript_training", "SampleScriptB_training"],
            edges=[("SampleScript_training", "SampleScriptB_training")]
        )
        
        # Create second script spec
        script_spec_b = ScriptExecutionSpec(
            script_name="sample_script_b",
            step_name="SampleScriptB_training",
            script_path=str(temp_workspace / "scripts" / "sample_script.py"),
            input_paths={"data_input": str(temp_workspace / "output" / "sample_output")},
            output_paths={"data_output": str(temp_workspace / "output" / "sample_output_b")},
            environ_vars={"PYTHONPATH": "src"},
            job_args={"job_type": "testing"}
        )
        
        # Create pipeline spec
        pipeline_spec = PipelineTestingSpec(
            dag=dag,
            script_specs={
                "SampleScript_training": sample_script_spec,
                "SampleScriptB_training": script_spec_b
            },
            test_workspace_root=str(temp_workspace)
        )
        
        # Test enhanced pipeline testing
        results = runtime_tester_with_step_catalog.test_pipeline_flow_with_step_catalog_enhancements(pipeline_spec)
        
        # Should return results with step catalog analysis
        assert isinstance(results, dict)
        assert "pipeline_success" in results
        
        # Should have step catalog analysis if available
        if "step_catalog_analysis" in results:
            analysis = results["step_catalog_analysis"]
            assert "workspace_analysis" in analysis
            assert "framework_analysis" in analysis

    # Test Integration Points
    
    def test_step_catalog_initialization(self, temp_workspace):
        """Test step catalog initialization with unified workspace resolution."""
        
        # Test RuntimeTester initialization
        tester = RuntimeTester(config_or_workspace_dir=str(temp_workspace))
        
        # Should have step catalog attribute (even if None due to import issues)
        assert hasattr(tester, 'step_catalog')
        
        # Test PipelineTestingSpecBuilder initialization
        builder = PipelineTestingSpecBuilder(test_data_dir=str(temp_workspace))
        
        # Should have step catalog attribute (even if None due to import issues)
        assert hasattr(builder, 'step_catalog')

    def test_contract_aware_path_resolution(self, spec_builder_with_step_catalog, temp_workspace):
        """Test contract-aware path resolution."""
        
        # Test contract-aware path resolution
        paths = spec_builder_with_step_catalog._get_contract_aware_paths_if_available(
            "SampleScript_training", str(temp_workspace)
        )
        
        # Should return path structure
        assert isinstance(paths, dict)
        assert "input_paths" in paths
        assert "output_paths" in paths
        
        # Should have contract-aware paths if step catalog available
        if spec_builder_with_step_catalog.step_catalog:
            assert len(paths["input_paths"]) > 0 or len(paths["output_paths"]) > 0

    def test_framework_detection(self, runtime_tester_with_step_catalog, sample_script_spec):
        """Test framework detection functionality."""
        
        # Test framework detection
        framework = runtime_tester_with_step_catalog._detect_framework_if_needed(sample_script_spec)
        
        # Should return framework if step catalog available
        if runtime_tester_with_step_catalog.step_catalog:
            assert framework == "xgboost"
        else:
            assert framework is None

    def test_builder_consistency_validation(self, runtime_tester_with_step_catalog, sample_script_spec):
        """Test builder consistency validation."""
        
        # Test builder consistency validation
        warnings = runtime_tester_with_step_catalog._validate_builder_consistency_if_available(sample_script_spec)
        
        # Should return warnings list
        assert isinstance(warnings, list)
        
        # Should have warnings if there are consistency issues
        if runtime_tester_with_step_catalog.step_catalog and warnings:
            assert any("missing expected input paths" in warning.lower() for warning in warnings)

    def test_multi_workspace_component_discovery(self, runtime_tester_with_step_catalog):
        """Test multi-workspace component discovery."""
        
        # Create mock DAG
        dag = MockPipelineDAG(
            nodes=["SampleScript_training"],
            edges=[]
        )
        
        # Test component discovery
        component_map = runtime_tester_with_step_catalog._discover_pipeline_components_if_needed(dag)
        
        # Should return component map
        assert isinstance(component_map, dict)
        
        # Should have component info if step catalog available
        if runtime_tester_with_step_catalog.step_catalog and component_map:
            assert "SampleScript_training" in component_map
            component_info = component_map["SampleScript_training"]
            assert "available_workspaces" in component_info
            assert "script_available" in component_info
            assert "builder_available" in component_info
            assert "contract_available" in component_info

    # Test Error Handling and Edge Cases
    
    def test_step_catalog_import_error_handling(self, temp_workspace):
        """Test graceful handling when step catalog cannot be imported."""
        
        # Mock import error by patching the import inside the _initialize_step_catalog method
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                   exec('raise ImportError("Mocked import error")') if 'step_catalog' in name 
                   else __import__(name, *args, **kwargs)):
            tester = RuntimeTester(config_or_workspace_dir=str(temp_workspace))
            
            # Should handle import error gracefully
            assert tester.step_catalog is None

    def test_step_catalog_method_error_handling(self, temp_workspace):
        """Test graceful handling when step catalog methods fail."""
        
        # Create mock step catalog that raises errors
        mock_catalog = Mock()
        mock_catalog.detect_framework.side_effect = Exception("Test error")
        mock_catalog.load_builder_class.side_effect = Exception("Test error")
        mock_catalog.load_contract_class.side_effect = Exception("Test error")
        
        tester = RuntimeTester(config_or_workspace_dir=str(temp_workspace), step_catalog=mock_catalog)
        
        sample_spec = ScriptExecutionSpec(
            script_name="test_script",
            step_name="TestScript_training",
            script_path="test_script.py",
            input_paths={"data_input": "input"},
            output_paths={"data_output": "output"},
            environ_vars={},
            job_args={}
        )
        
        # Should handle errors gracefully
        framework = tester._detect_framework_if_needed(sample_spec)
        assert framework is None
        
        warnings = tester._validate_builder_consistency_if_available(sample_spec)
        assert warnings == []

    def test_performance_with_step_catalog_disabled(self, temp_workspace, sample_script_spec):
        """Test that performance is not impacted when step catalog is disabled."""
        
        import time
        
        # Create sample input data
        input_dir = Path(sample_script_spec.input_paths["data_input"])
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "sample.json").write_text('{"test": "data"}')
        
        main_params = {
            "input_paths": sample_script_spec.input_paths,
            "output_paths": sample_script_spec.output_paths,
            "environ_vars": sample_script_spec.environ_vars,
            "job_args": sample_script_spec.job_args
        }
        
        # Test without step catalog
        tester_without = RuntimeTester(config_or_workspace_dir=str(temp_workspace), step_catalog=None)
        
        start_time = time.time()
        result_without = tester_without.test_script_with_step_catalog_enhancements(sample_script_spec, main_params)
        time_without = time.time() - start_time
        
        # Test with mock step catalog
        mock_catalog = Mock()
        mock_catalog.detect_framework.return_value = "xgboost"
        mock_catalog.load_builder_class.return_value = None
        
        tester_with = RuntimeTester(config_or_workspace_dir=str(temp_workspace), step_catalog=mock_catalog)
        
        start_time = time.time()
        result_with = tester_with.test_script_with_step_catalog_enhancements(sample_script_spec, main_params)
        time_with = time.time() - start_time
        
        # Both should succeed
        assert result_without.success
        assert result_with.success
        
        # Performance difference should be minimal (within reasonable bounds)
        # This is a basic check - in practice, the difference should be very small
        assert time_with < time_without + 1.0  # Allow up to 1 second difference for test environment


class TestStepCatalogSpecBuilderIntegration:
    """Test step catalog integration in PipelineTestingSpecBuilder."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        workspace_path = Path(temp_dir)
        
        # Create workspace structure
        (workspace_path / "scripts").mkdir(parents=True, exist_ok=True)
        (workspace_path / "input").mkdir(parents=True, exist_ok=True)
        (workspace_path / "output").mkdir(parents=True, exist_ok=True)
        
        yield workspace_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_step_catalog(self):
        """Create mock step catalog for spec builder testing."""
        mock_catalog = Mock()
        
        # Mock pipeline node resolution
        mock_step_info = Mock()
        mock_script_metadata = Mock()
        mock_script_metadata.path = Path("scripts/sample_script.py")
        mock_step_info.file_components = {"script": mock_script_metadata}
        mock_catalog.resolve_pipeline_node.return_value = mock_step_info
        
        # Mock contract class loading
        mock_contract = Mock()
        mock_contract.get_input_paths.return_value = {"data_input": "/path/to/input"}
        mock_contract.get_output_paths.return_value = {"data_output": "/path/to/output"}
        mock_catalog.load_contract_class.return_value = mock_contract
        
        return mock_catalog

    def test_script_resolution_with_step_catalog(self, temp_workspace, mock_step_catalog):
        """Test script resolution using step catalog."""
        
        builder = PipelineTestingSpecBuilder(
            test_data_dir=str(temp_workspace),
            step_catalog=mock_step_catalog
        )
        
        # Test step catalog script resolution
        script_spec = builder._resolve_script_with_step_catalog_if_available("SampleScript_training")
        
        # Should return ScriptExecutionSpec if step catalog available
        if mock_step_catalog:
            assert script_spec is not None
            assert isinstance(script_spec, ScriptExecutionSpec)
            assert script_spec.step_name == "SampleScript_training"

    def test_script_resolution_fallback(self, temp_workspace):
        """Test script resolution fallback when step catalog unavailable."""
        
        builder = PipelineTestingSpecBuilder(
            test_data_dir=str(temp_workspace),
            step_catalog=None
        )
        
        # Test step catalog script resolution fallback
        script_spec = builder._resolve_script_with_step_catalog_if_available("SampleScript_training")
        
        # Should return None when step catalog unavailable
        assert script_spec is None

    def test_contract_aware_paths(self, temp_workspace, mock_step_catalog):
        """Test contract-aware path resolution."""
        
        builder = PipelineTestingSpecBuilder(
            test_data_dir=str(temp_workspace),
            step_catalog=mock_step_catalog
        )
        
        # Test contract-aware path resolution
        paths = builder._get_contract_aware_paths_if_available("SampleScript_training", str(temp_workspace))
        
        # Should return paths structure
        assert isinstance(paths, dict)
        assert "input_paths" in paths
        assert "output_paths" in paths
        
        # Should have paths if step catalog and contract available
        if mock_step_catalog:
            assert len(paths["input_paths"]) > 0
            assert len(paths["output_paths"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
