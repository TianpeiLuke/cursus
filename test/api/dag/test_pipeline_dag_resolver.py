"""Test enhanced pipeline DAG resolver with dynamic contract discovery."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from src.cursus.api.dag import PipelineDAG
from src.cursus.api.dag.pipeline_dag_resolver import PipelineDAGResolver
from src.cursus.core.base.contract_base import ScriptContract
from src.cursus.core.base.specification_base import StepSpecification


class TestPipelineDAGResolverEnhanced:
    """Test enhanced pipeline DAG resolver functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test DAG: preprocessing -> training -> evaluation
        self.dag = PipelineDAG(
            nodes=["preprocessing", "training", "evaluation"],
            edges=[("preprocessing", "training"), ("training", "evaluation")]
        )
        self.resolver = PipelineDAGResolver(self.dag)
    
    def test_basic_execution_plan_creation(self):
        """Test basic execution plan creation without contracts."""
        plan = self.resolver.create_execution_plan()
        
        assert plan.execution_order == ["preprocessing", "training", "evaluation"]
        assert "preprocessing" in plan.step_configs
        assert "training" in plan.step_configs
        assert "evaluation" in plan.step_configs
        
        # Check dependencies
        assert plan.dependencies["preprocessing"] == []
        assert plan.dependencies["training"] == ["preprocessing"]
        assert plan.dependencies["evaluation"] == ["training"]
    
    def test_data_flow_map_fallback(self):
        """Test data flow map creation with fallback approach."""
        plan = self.resolver.create_execution_plan()
        
        # Should use fallback generic approach
        assert plan.data_flow_map["preprocessing"] == {}
        assert plan.data_flow_map["training"] == {"input_0": "preprocessing:output"}
        assert plan.data_flow_map["evaluation"] == {"input_0": "training:output"}
    
    @patch('src.cursus.api.dag.pipeline_dag_resolver.get_canonical_name_from_file_name')
    @patch('src.cursus.api.dag.pipeline_dag_resolver.get_spec_step_type')
    def test_contract_discovery_success(self, mock_get_spec_type, mock_get_canonical):
        """Test successful contract discovery."""
        # Mock the registry helper functions
        mock_get_canonical.return_value = "xgboost_training"
        mock_get_spec_type.return_value = "XGBoostTrainingSpec"
        
        # Mock the specification with contract
        mock_contract = ScriptContract(
            entry_point="xgboost_training.py",
            expected_input_paths={"input_path": "/opt/ml/processing/input/data"},
            expected_output_paths={"model_path": "/opt/ml/processing/output/model"},
            required_env_vars=["SM_MODEL_DIR"]
        )
        mock_spec = Mock()
        mock_spec.script_contract = mock_contract
        
        # Mock the dynamic import and spec retrieval
        with patch.object(self.resolver, '_get_step_specification', return_value=mock_spec):
            contract = self.resolver._discover_step_contract("training")
            
            assert contract is not None
            assert contract.expected_input_paths == {"input_path": "/opt/ml/processing/input/data"}
            assert contract.expected_output_paths == {"model_path": "/opt/ml/processing/output/model"}
    
    @patch('src.cursus.api.dag.pipeline_dag_resolver.get_canonical_name_from_file_name')
    def test_contract_discovery_failure(self, mock_get_canonical):
        """Test contract discovery failure fallback."""
        # Mock failure in canonical name resolution
        mock_get_canonical.return_value = None
        
        contract = self.resolver._discover_step_contract("unknown_step")
        assert contract is None
    
    def test_spec_type_to_module_name_conversion(self):
        """Test spec type to module name conversion."""
        assert self.resolver._spec_type_to_module_name("XGBoostTrainingSpec") == "xgboost_training_spec"
        assert self.resolver._spec_type_to_module_name("TabularPreprocessingSpec") == "tabular_preprocessing_spec"
        assert self.resolver._spec_type_to_module_name("SimpleStep") == "simple_step_spec"
    
    def test_find_compatible_output_direct_match(self):
        """Test direct channel name matching."""
        output_channels = {
            "model_path": "/opt/ml/model",
            "output_path": "/opt/ml/output/data"
        }
        
        result = self.resolver._find_compatible_output(
            "model_path", "/opt/ml/input/data", output_channels
        )
        assert result == "model_path"
    
    def test_find_compatible_output_semantic_match(self):
        """Test semantic channel matching."""
        output_channels = {
            "output_path": "/opt/ml/output/data",
            "config_path": "/opt/ml/output/config"
        }
        
        result = self.resolver._find_compatible_output(
            "input_path", "/opt/ml/input/data", output_channels
        )
        assert result == "output_path"
    
    def test_find_compatible_output_fallback(self):
        """Test fallback to first available output."""
        output_channels = {
            "some_output": "/opt/ml/output/data"
        }
        
        result = self.resolver._find_compatible_output(
            "unknown_input", "/opt/ml/input/data", output_channels
        )
        assert result == "some_output"
    
    def test_path_compatibility_sagemaker_conventions(self):
        """Test SageMaker path compatibility rules."""
        # Model artifacts compatibility
        assert self.resolver._are_paths_compatible(
            "/opt/ml/model/model.tar.gz",
            "/opt/ml/model/output.tar.gz"
        )
        
        # Data flow compatibility
        assert self.resolver._are_paths_compatible(
            "/opt/ml/input/data/train.csv",
            "/opt/ml/output/data/processed.csv"
        )
        
        # Incompatible paths
        assert not self.resolver._are_paths_compatible(
            "/opt/ml/model/model.tar.gz",
            "/opt/ml/code/script.py"
        )
    
    @patch('src.cursus.api.dag.pipeline_dag_resolver.importlib.import_module')
    @patch('src.cursus.api.dag.pipeline_dag_resolver.get_spec_step_type')
    def test_get_step_specification_with_getter_function(self, mock_get_spec_type, mock_import):
        """Test specification retrieval with getter function."""
        # Mock the spec type lookup
        mock_get_spec_type.return_value = "XGBoostTrainingSpec"
        
        # Mock module with getter function
        mock_module = Mock()
        mock_spec = Mock()
        mock_module.get_xgboost_training_spec = Mock(return_value=mock_spec)
        mock_import.return_value = mock_module
        
        # Call with canonical name directly (this method expects canonical name, not step name)
        result = self.resolver._get_step_specification("xgboost_training")
        
        assert result == mock_spec
        mock_get_spec_type.assert_called_once_with("xgboost_training")
        mock_import.assert_called_once_with("cursus.steps.specs.xgboost_training_spec")
        mock_module.get_xgboost_training_spec.assert_called_once()
    
    @patch('src.cursus.api.dag.pipeline_dag_resolver.importlib.import_module')
    @patch('src.cursus.api.dag.pipeline_dag_resolver.get_spec_step_type')
    def test_get_step_specification_with_constant(self, mock_get_spec_type, mock_import):
        """Test specification retrieval with spec constant (actual pattern used in codebase)."""
        # Mock the spec type lookup
        mock_get_spec_type.return_value = "XGBoostTrainingSpec"
        
        # Create a simple object that only has the constant we want
        class MockModule:
            def __init__(self):
                self.XGBOOST_TRAINING_SPEC = Mock()
            
            def __getattr__(self, name):
                # This ensures hasattr returns False for non-existent attributes
                raise AttributeError(f"module has no attribute '{name}'")
        
        mock_module = MockModule()
        mock_import.return_value = mock_module
        
        # Call with canonical name directly (this method expects canonical name, not step name)
        result = self.resolver._get_step_specification("xgboost_training")
        
        assert result == mock_module.XGBOOST_TRAINING_SPEC
        mock_get_spec_type.assert_called_once_with("xgboost_training")
        mock_import.assert_called_once_with("cursus.steps.specs.xgboost_training_spec")
    
    def test_enhanced_data_flow_map_with_contracts(self):
        """Test enhanced data flow map creation with contracts."""
        # Mock contracts for each step
        preprocessing_contract = ScriptContract(
            entry_point="preprocessing.py",
            expected_input_paths={},
            expected_output_paths={"output_path": "/opt/ml/processing/output/data"},
            required_env_vars=[]
        )
        
        training_contract = ScriptContract(
            entry_point="training.py",
            expected_input_paths={"input_path": "/opt/ml/processing/input/data"},
            expected_output_paths={"model_path": "/opt/ml/processing/output/model"},
            required_env_vars=["SM_MODEL_DIR"]
        )
        
        evaluation_contract = ScriptContract(
            entry_point="evaluation.py",
            expected_input_paths={"model_path": "/opt/ml/processing/input/model"},
            expected_output_paths={"evaluation_path": "/opt/ml/processing/output/evaluation"},
            required_env_vars=["SM_MODEL_DIR"]
        )
        
        # Mock contract discovery
        def mock_discover_contract(step_name):
            contracts = {
                "preprocessing": preprocessing_contract,
                "training": training_contract,
                "evaluation": evaluation_contract
            }
            return contracts.get(step_name)
        
        with patch.object(self.resolver, '_discover_step_contract', side_effect=mock_discover_contract):
            plan = self.resolver.create_execution_plan()
            
            # Check enhanced data flow mapping
            assert plan.data_flow_map["preprocessing"] == {}
            assert plan.data_flow_map["training"] == {"input_path": "preprocessing:output_path"}
            assert plan.data_flow_map["evaluation"] == {"model_path": "training:model_path"}
    
    def test_dag_validation_with_cycles(self):
        """Test DAG validation with cycles."""
        # Create DAG with cycle
        cyclic_dag = PipelineDAG(
            nodes=["step1", "step2", "step3"],
            edges=[("step1", "step2"), ("step2", "step3"), ("step3", "step1")]
        )
        resolver = PipelineDAGResolver(cyclic_dag)
        
        issues = resolver.validate_dag_integrity()
        assert "cycles" in issues
        assert len(issues["cycles"]) > 0
    
    def test_dag_validation_with_dangling_dependencies(self):
        """Test DAG validation with dangling dependencies."""
        # Create DAG with dangling edge by manually setting edges after construction
        invalid_dag = PipelineDAG(
            nodes=["step1", "step2"],
            edges=[("step1", "step2")]
        )
        # Manually add a dangling edge to test validation
        invalid_dag.edges.append(("step2", "nonexistent_step"))
        
        resolver = PipelineDAGResolver(invalid_dag)
        
        issues = resolver.validate_dag_integrity()
        assert "dangling_dependencies" in issues
        assert any("nonexistent_step" in issue for issue in issues["dangling_dependencies"])
    
    def test_dag_validation_with_isolated_nodes(self):
        """Test DAG validation with isolated nodes."""
        # Create DAG with isolated node
        isolated_dag = PipelineDAG(
            nodes=["step1", "step2", "isolated_step"],
            edges=[("step1", "step2")]
        )
        resolver = PipelineDAGResolver(isolated_dag)
        
        issues = resolver.validate_dag_integrity()
        assert "isolated_nodes" in issues
        assert any("isolated_step" in issue for issue in issues["isolated_nodes"])


if __name__ == "__main__":
    pytest.main([__file__])
