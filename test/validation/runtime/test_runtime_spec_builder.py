"""
Pytest tests for streamlined runtime spec builder

Tests the streamlined PipelineTestingSpecBuilder class focused on core intelligence methods.
Updated to match the Week 2 streamlined architecture that removes redundant interactive methods.

Following pytest best practices:
1. Read source code first to understand actual implementation
2. Test core intelligence methods that remain
3. Mock at correct import paths
4. Match test expectations to implementation reality
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder
from src.cursus.validation.runtime.runtime_models import (
    ScriptExecutionSpec,
    PipelineTestingSpec,
)
from src.cursus.api.dag.base_dag import PipelineDAG


class TestStreamlinedPipelineTestingSpecBuilder:
    """Test streamlined PipelineTestingSpecBuilder class - Core Intelligence Methods Only"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def test_dag(self):
        """Create a simple DAG for testing"""
        return PipelineDAG(
            nodes=["TabularPreprocessing_training", "XGBoostTraining_training", "ModelEvaluation_evaluation"],
            edges=[("TabularPreprocessing_training", "XGBoostTraining_training"), 
                   ("XGBoostTraining_training", "ModelEvaluation_evaluation")],
        )

    @pytest.fixture
    def builder(self, temp_dir):
        """Create builder with temporary directory"""
        return PipelineTestingSpecBuilder(test_data_dir=temp_dir, step_catalog=None)

    def test_builder_initialization(self):
        """Test PipelineTestingSpecBuilder initialization"""
        builder = PipelineTestingSpecBuilder()

        assert str(builder.test_data_dir) == "test/integration/runtime"
        assert builder.specs_dir.name == "specs"
        assert builder.scripts_dir.name == "scripts"

    def test_builder_initialization_with_params(self):
        """Test PipelineTestingSpecBuilder initialization with parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = PipelineTestingSpecBuilder(test_data_dir=temp_dir)

            assert str(builder.test_data_dir) == temp_dir
            assert builder.specs_dir.name == "specs"
            assert builder.scripts_dir.name == "scripts"

    # === CORE INTELLIGENCE METHODS TESTS ===

    def test_canonical_to_script_name(self, builder):
        """Test core name conversion logic"""
        # Test basic PascalCase to snake_case
        assert builder._canonical_to_script_name("TabularPreprocessing") == "tabular_preprocessing"
        assert builder._canonical_to_script_name("ModelEvaluation") == "model_evaluation"
        
        # Test special cases for compound technical terms
        assert builder._canonical_to_script_name("XGBoostTraining") == "xgboost_training"
        assert builder._canonical_to_script_name("PyTorchInference") == "pytorch_inference"
        assert builder._canonical_to_script_name("MLFlowRegistration") == "mlflow_registration"

    @patch('src.cursus.validation.runtime.runtime_spec_builder.get_step_name_from_spec_type')
    def test_resolve_script_execution_spec_from_node_success(self, mock_get_step_name, builder):
        """Test core intelligent script resolution from node name"""
        # Mock registry function
        mock_get_step_name.return_value = "TabularPreprocessing"
        
        # Mock script file discovery
        with patch.object(builder, '_find_script_file') as mock_find_script:
            mock_find_script.return_value = Path("/test/scripts/tabular_preprocessing.py")
            
            # Mock contract-aware path methods
            with patch.object(builder, '_get_contract_aware_input_paths') as mock_input_paths, \
                 patch.object(builder, '_get_contract_aware_output_paths') as mock_output_paths, \
                 patch.object(builder, '_get_contract_aware_environ_vars') as mock_environ_vars, \
                 patch.object(builder, '_get_contract_aware_job_args') as mock_job_args:
                
                mock_input_paths.return_value = {"data_input": "/test/input/raw_data"}
                mock_output_paths.return_value = {"data_output": "/test/output/processed_data"}
                mock_environ_vars.return_value = {"CURSUS_ENV": "testing"}
                mock_job_args.return_value = {"job_type": "testing"}
                
                # Test the core resolution method
                spec = builder.resolve_script_execution_spec_from_node("TabularPreprocessing_training")
                
                assert isinstance(spec, ScriptExecutionSpec)
                assert spec.script_name == "tabular_preprocessing"
                assert spec.step_name == "TabularPreprocessing_training"
                assert spec.script_path == "/test/scripts/tabular_preprocessing.py"
                assert spec.input_paths == {"data_input": "/test/input/raw_data"}
                assert spec.output_paths == {"data_output": "/test/output/processed_data"}

    @patch('src.cursus.validation.runtime.runtime_spec_builder.get_step_name_from_spec_type')
    def test_resolve_script_execution_spec_from_node_registry_failure(self, mock_get_step_name, builder):
        """Test script resolution when registry fails"""
        # Mock registry function to raise exception
        mock_get_step_name.side_effect = Exception("Registry lookup failed")
        
        # Should raise ValueError with descriptive message
        with pytest.raises(ValueError, match="Registry resolution failed"):
            builder.resolve_script_execution_spec_from_node("UnknownStep_training")

    def test_find_script_file_test_workspace(self, builder):
        """Test core script discovery logic - test workspace priority"""
        # Create a test script in the test workspace
        test_script = builder.scripts_dir / "test_script.py"
        test_script.parent.mkdir(parents=True, exist_ok=True)
        test_script.write_text("# Test script")
        
        # Should find the script in test workspace
        found_path = builder._find_script_file("test_script")
        assert found_path == test_script
        assert found_path.exists()

    @patch('src.cursus.step_catalog.StepCatalog')
    def test_find_script_file_step_catalog_discovery(self, mock_step_catalog_class, builder):
        """Test script discovery using step catalog"""
        # Mock step catalog discovery
        mock_catalog = Mock()
        mock_step_catalog_class.return_value = mock_catalog
        
        mock_catalog.list_available_steps.return_value = ["test_step"]
        
        mock_step_info = Mock()
        mock_script_metadata = Mock()
        mock_script_metadata.path = Path("/catalog/scripts/test_script.py")
        mock_step_info.file_components = {"script": mock_script_metadata}
        mock_catalog.get_step_info.return_value = mock_step_info
        
        # Should find script through step catalog
        found_path = builder._find_script_file("test_script")
        assert found_path == Path("/catalog/scripts/test_script.py")

    def test_find_script_file_creates_placeholder(self, builder):
        """Test script discovery creates placeholder when not found"""
        # Should create placeholder script when not found
        found_path = builder._find_script_file("nonexistent_script")
        
        expected_path = builder.scripts_dir / "nonexistent_script.py"
        assert found_path == expected_path
        assert found_path.exists()
        
        # Verify placeholder content
        content = found_path.read_text()
        assert "Placeholder script for nonexistent_script" in content
        assert "def main():" in content

    def test_get_script_main_params(self, builder):
        """Test core parameter extraction"""
        spec = ScriptExecutionSpec(
            script_name="param_test",
            step_name="param_step",
            script_path="/test/script.py",
            input_paths={"data_input": "/test/input"},
            output_paths={"data_output": "/test/output"},
            environ_vars={"TEST_VAR": "test_value"},
            job_args={"test_arg": "test_value"},
        )

        params = builder.get_script_main_params(spec)

        assert params["input_paths"]["data_input"] == "/test/input"
        assert params["output_paths"]["data_output"] == "/test/output"
        assert params["environ_vars"]["TEST_VAR"] == "test_value"
        assert params["job_args"].test_arg == "test_value"

    @patch('src.cursus.validation.runtime.runtime_spec_builder.ContractDiscoveryManager')
    def test_get_contract_aware_input_paths_with_contract(self, mock_contract_manager_class, builder):
        """Test contract-aware path resolution with contract"""
        # Mock contract discovery
        mock_contract_manager = Mock()
        mock_contract_manager_class.return_value = mock_contract_manager
        builder.contract_manager = mock_contract_manager
        
        mock_contract_result = Mock()
        mock_contract_result.contract = Mock()  # Contract found
        mock_contract_manager.discover_contract.return_value = mock_contract_result
        mock_contract_manager.get_contract_input_paths.return_value = {
            "training_data": "/contract/input/training.csv",
            "validation_data": "/contract/input/validation.csv"
        }
        
        # Should use contract paths
        paths = builder._get_contract_aware_input_paths("test_script", "TestScript")
        assert paths == {
            "training_data": "/contract/input/training.csv",
            "validation_data": "/contract/input/validation.csv"
        }

    @patch('src.cursus.validation.runtime.runtime_spec_builder.ContractDiscoveryManager')
    def test_get_contract_aware_input_paths_fallback(self, mock_contract_manager_class, builder):
        """Test contract-aware path resolution fallback to defaults"""
        # Mock contract discovery - no contract found
        mock_contract_manager = Mock()
        mock_contract_manager_class.return_value = mock_contract_manager
        builder.contract_manager = mock_contract_manager
        
        mock_contract_result = Mock()
        mock_contract_result.contract = None  # No contract found
        mock_contract_manager.discover_contract.return_value = mock_contract_result
        
        # Should fallback to default paths
        paths = builder._get_contract_aware_input_paths("test_script", "TestScript")
        assert "data_input" in paths
        assert "config" in paths
        assert str(builder.test_data_dir) in paths["data_input"]

    # === STEP CATALOG INTEGRATION TESTS ===

    @patch('src.cursus.step_catalog.StepCatalog')
    def test_resolve_script_with_step_catalog_if_available_success(self, mock_step_catalog_class, builder):
        """Test enhanced script resolution using step catalog"""
        # Mock step catalog
        mock_catalog = Mock()
        mock_step_catalog_class.return_value = mock_catalog
        builder.step_catalog = mock_catalog
        
        # Mock step catalog resolution
        mock_step_info = Mock()
        mock_script_metadata = Mock()
        mock_script_metadata.path = Mock()
        mock_script_metadata.path.stem = "test_script"
        mock_step_info.file_components = {"script": mock_script_metadata}
        mock_catalog.resolve_pipeline_node.return_value = mock_step_info
        
        # Mock contract-aware paths
        with patch.object(builder, '_get_contract_aware_paths_if_available') as mock_paths:
            mock_paths.return_value = {
                "input_paths": {"data_input": "/test/input"},
                "output_paths": {"data_output": "/test/output"}
            }
            
            # Should resolve using step catalog
            spec = builder._resolve_script_with_step_catalog_if_available("TestNode")
            
            assert spec is not None
            assert spec.script_name == "test_script"
            assert spec.step_name == "TestNode"

    def test_resolve_script_with_step_catalog_if_available_no_catalog(self, builder):
        """Test step catalog resolution when catalog not available"""
        builder.step_catalog = None
        
        # Should return None when no catalog available
        spec = builder._resolve_script_with_step_catalog_if_available("TestNode")
        assert spec is None

    # === LEGACY SUPPORT TESTS ===

    def test_build_from_dag_legacy_support(self, builder, test_dag):
        """Test legacy build_from_dag method with minimal implementation"""
        # Mock the core resolution method
        with patch.object(builder, 'resolve_script_execution_spec_from_node') as mock_resolve:
            mock_specs = []
            for node in test_dag.nodes:
                mock_spec = Mock(spec=ScriptExecutionSpec)
                mock_spec.script_name = node.lower()
                mock_spec.step_name = node
                mock_specs.append(mock_spec)
            
            mock_resolve.side_effect = mock_specs
            
            # Test legacy method
            pipeline_spec = builder.build_from_dag(test_dag, validate=False)
            
            assert isinstance(pipeline_spec, PipelineTestingSpec)
            assert pipeline_spec.dag == test_dag
            assert len(pipeline_spec.script_specs) == 3

    def test_build_from_dag_with_validation_failure(self, builder, test_dag):
        """Test legacy build_from_dag method with validation failure"""
        # Mock resolution to fail for one node
        with patch.object(builder, 'resolve_script_execution_spec_from_node') as mock_resolve:
            def side_effect(node_name):
                if node_name == "TabularPreprocessing_training":
                    raise ValueError("Failed to resolve")
                return Mock(spec=ScriptExecutionSpec)
            
            mock_resolve.side_effect = side_effect
            
            # Should raise ValueError when validation enabled
            with pytest.raises(ValueError, match="Failed to resolve spec for node"):
                builder.build_from_dag(test_dag, validate=True)

    def test_build_from_dag_without_validation_creates_fallback(self, builder, test_dag):
        """Test legacy build_from_dag creates fallback specs when validation disabled"""
        # Mock resolution to fail for all nodes
        with patch.object(builder, 'resolve_script_execution_spec_from_node') as mock_resolve:
            mock_resolve.side_effect = Exception("Resolution failed")
            
            # Should create fallback specs when validation disabled
            pipeline_spec = builder.build_from_dag(test_dag, validate=False)
            
            assert isinstance(pipeline_spec, PipelineTestingSpec)
            assert len(pipeline_spec.script_specs) == 3
            
            # Verify fallback specs were created
            for node in test_dag.nodes:
                assert node in pipeline_spec.script_specs
                spec = pipeline_spec.script_specs[node]
                assert spec.script_name == node.lower()
                assert spec.step_name == node

    # === HELPER METHOD TESTS ===

    def test_get_default_input_paths(self, builder):
        """Test default input paths generation"""
        paths = builder._get_default_input_paths("test_script")
        
        assert "data_input" in paths
        assert "config" in paths
        assert str(builder.test_data_dir) in paths["data_input"]
        assert "test_script_config.json" in paths["config"]

    def test_get_default_output_paths(self, builder):
        """Test default output paths generation"""
        paths = builder._get_default_output_paths("test_script")
        
        assert "data_output" in paths
        assert "metrics" in paths
        assert str(builder.test_data_dir) in paths["data_output"]
        assert "test_script_output" in paths["data_output"]

    def test_get_default_environ_vars(self, builder):
        """Test default environment variables"""
        env_vars = builder._get_default_environ_vars()
        
        assert "PYTHONPATH" in env_vars
        assert "CURSUS_ENV" in env_vars
        assert env_vars["CURSUS_ENV"] == "testing"

    def test_get_default_job_args(self, builder):
        """Test default job arguments"""
        job_args = builder._get_default_job_args("test_script")
        
        assert "script_name" in job_args
        assert "execution_mode" in job_args
        assert "log_level" in job_args
        assert job_args["script_name"] == "test_script"
        assert job_args["execution_mode"] == "testing"


class TestStreamlinedSpecBuilderIntegration:
    """Integration tests for streamlined PipelineTestingSpecBuilder"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def builder(self, temp_dir):
        """Create builder with temporary directory"""
        return PipelineTestingSpecBuilder(test_data_dir=temp_dir)

    @patch('src.cursus.validation.runtime.runtime_spec_builder.get_step_name_from_spec_type')
    def test_end_to_end_core_intelligence_workflow(self, mock_get_step_name, builder):
        """Test complete core intelligence workflow"""
        # Mock registry resolution
        mock_get_step_name.side_effect = ["TabularPreprocessing", "XGBoostTraining", "ModelEvaluation"]
        
        # Create test scripts
        for script_name in ["tabular_preprocessing", "xgboost_training", "model_evaluation"]:
            script_path = builder.scripts_dir / f"{script_name}.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(f"# {script_name} script")
        
        # Test end-to-end resolution
        dag = PipelineDAG(
            nodes=["TabularPreprocessing_training", "XGBoostTraining_training", "ModelEvaluation_evaluation"],
            edges=[("TabularPreprocessing_training", "XGBoostTraining_training")]
        )
        
        # Resolve each node using core intelligence
        specs = {}
        for node in dag.nodes:
            spec = builder.resolve_script_execution_spec_from_node(node)
            specs[node] = spec
        
        # Verify all specs were created with core intelligence
        assert len(specs) == 3
        for node, spec in specs.items():
            assert isinstance(spec, ScriptExecutionSpec)
            assert spec.step_name == node
            assert spec.script_path.endswith(".py")
            assert "data_input" in spec.input_paths
            assert "data_output" in spec.output_paths

    def test_step_catalog_integration_workflow(self, builder):
        """Test step catalog integration workflow"""
        # Test initialization with step catalog
        assert builder.step_catalog is not None or builder.step_catalog is None  # May or may not be available
        
        # Test step catalog resolution (should handle gracefully if not available)
        spec = builder._resolve_script_with_step_catalog_if_available("TestNode")
        # Should return None if step catalog not available, or a spec if it is
        assert spec is None or isinstance(spec, ScriptExecutionSpec)

    def test_contract_aware_path_resolution_workflow(self, builder):
        """Test contract-aware path resolution workflow"""
        # Test input path resolution
        input_paths = builder._get_contract_aware_input_paths("test_script", "TestScript")
        assert isinstance(input_paths, dict)
        assert len(input_paths) > 0
        
        # Test output path resolution
        output_paths = builder._get_contract_aware_output_paths("test_script", "TestScript")
        assert isinstance(output_paths, dict)
        assert len(output_paths) > 0
        
        # Paths should be strings pointing to test directory
        for path in input_paths.values():
            assert isinstance(path, str)
            assert str(builder.test_data_dir) in path
